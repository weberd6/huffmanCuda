#include <fstream>
#include <cstring>
#include <ctime>

#include "main.h"
#include "node.h"

void parallel_huffman_encode(unsigned char* data, unsigned int num_bytes, std::string filename) {

    const unsigned int NUM_VALS = 256;
    const unsigned int PADDING = ((num_bytes % 256) == 0) ? 0 : (256 - (num_bytes % 256));
    double duration;

    unsigned char* d_vals;
    unsigned int* d_frequencies;
    unsigned int* d_min_frequencies;
    unsigned int* d_min_indices;
    unsigned int* d_codes;
    unsigned int* d_lengths;
    unsigned int* d_data_lengths;
    unsigned int* d_lengths_partial_sums;
    unsigned char* d_compressed_data;

    unsigned int* h_frequencies = new unsigned int[NUM_VALS];
    unsigned int* h_min_frequencies = new unsigned int[2];
    unsigned int* h_min_indices = new unsigned int[2];

    std::clock_t malloc_start = std::clock();

    cudaMalloc(&d_vals, num_bytes*sizeof(unsigned char));
    cudaMalloc(&d_frequencies, NUM_VALS*sizeof(unsigned int));
    cudaMalloc(&d_min_frequencies, 2*sizeof(unsigned int));
    cudaMalloc(&d_min_indices, 2*sizeof(unsigned int));
    cudaMalloc(&d_codes, NUM_VALS*sizeof(unsigned int));
    cudaMalloc(&d_lengths, NUM_VALS*sizeof(unsigned int));

    cudaMalloc(&d_data_lengths, (num_bytes+PADDING)*sizeof(unsigned int));
    cudaMemset(d_data_lengths, 0, (num_bytes+PADDING)*sizeof(unsigned int));
    
    cudaMalloc(&d_lengths_partial_sums, (num_bytes+PADDING)*sizeof(unsigned int));
    cudaMemset(d_lengths_partial_sums, 0, (num_bytes+PADDING)*sizeof(unsigned int));

    duration = ( std::clock() - malloc_start ) / (double) CLOCKS_PER_SEC;
    std::cout << "Malloc time: " << duration*1000 << " ms" << std::endl;

    std::clock_t histogram_start = std::clock();

    cudaMemcpy(d_vals, data, num_bytes*sizeof(unsigned char), cudaMemcpyHostToDevice);
    checkCudaErrors(cudaGetLastError());
 
    cudaMemset(d_frequencies, 0, NUM_VALS*sizeof(unsigned int));
    checkCudaErrors(cudaGetLastError());

    computeHistogram(d_vals, d_frequencies, NUM_VALS, num_bytes);
    duration = ( std::clock() - histogram_start ) / (double) CLOCKS_PER_SEC;
    std::cout << "Histogram time: " << duration*1000 << " ms" << std::endl;

    std::clock_t build_tree_start = std::clock();

    unsigned int count = NUM_VALS;
    unsigned int* d_count;
    cudaMalloc(&d_count, sizeof(unsigned int));
    cudaMemcpy(d_count, &count, sizeof(count), cudaMemcpyHostToDevice);

    minimizeBins(d_frequencies, d_count, NUM_VALS);
    cudaMemcpy(&count, d_count, sizeof(count), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_frequencies, d_frequencies, NUM_VALS*sizeof(unsigned int), cudaMemcpyDeviceToHost);

    unsigned int sum = 0;
    Node leaf_nodes[NUM_VALS];
    Node* node_by_index[NUM_VALS];

    for (int i = 0; i < NUM_VALS; i++) {
     
        if (h_frequencies[i] != 0xFFFFFFFF) sum += h_frequencies[i];
        
        leaf_nodes[i].frequency = h_frequencies[i];
        leaf_nodes[i].symbol_index = i;
        node_by_index[i] = &leaf_nodes[i];
    }

    Node* root;
    Node* l;
    Node* r;
   
    while (count > 1) {
 
        get_minimum2(d_frequencies, NUM_VALS, d_min_frequencies);
        cudaMemcpy(h_min_frequencies, d_min_frequencies, 2*sizeof(unsigned int), cudaMemcpyDeviceToHost);

        update_histo_and_get_min_indices(d_frequencies, h_min_frequencies[0], h_min_frequencies[1], d_min_indices, NUM_VALS);
        cudaMemcpy(h_min_indices, d_min_indices, 2*sizeof(unsigned int), cudaMemcpyDeviceToHost);

        l = node_by_index[h_min_indices[0]];
        r = node_by_index[h_min_indices[1]];

        root = new Node(l, r, l->frequency + r->frequency);
        node_by_index[h_min_indices[1]] = root;

        count--;
    }


    unsigned int codes[NUM_VALS];
    unsigned int lengths[NUM_VALS];

    memset(codes, 0, NUM_VALS*sizeof(unsigned int));
    memset(lengths, 0, NUM_VALS*sizeof(unsigned int));

    generate_code(root, codes, lengths);

    duration = ( std::clock() - build_tree_start ) / (double) CLOCKS_PER_SEC;
    std::cout << "Build tree time: " << duration*1000 << " ms" << std::endl;

    std::clock_t compression_start = std::clock();
    
    cudaMemcpy(d_codes, codes, NUM_VALS*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, lengths, NUM_VALS*sizeof(unsigned int), cudaMemcpyHostToDevice);

    unsigned int compressed_num_bytes = get_compressed_length(d_lengths, d_vals, d_data_lengths, d_lengths_partial_sums, num_bytes);
    cudaMalloc(&d_compressed_data, compressed_num_bytes*sizeof(unsigned char*));

    unsigned int* d_block_offsets;
    unsigned int num_blocks = ceil((float)num_bytes/(DATA_BLOCK_SIZE/256.0));
    cudaMalloc(&d_block_offsets, num_blocks*sizeof(unsigned int));
    std::cout << "Num blocks: " << num_blocks << std::endl;

    compress_data(d_vals, d_codes, d_lengths, d_lengths_partial_sums, d_block_offsets, d_compressed_data, num_bytes);
    duration = ( std::clock() - compression_start ) / (double) CLOCKS_PER_SEC;
    std::cout << "Compression time: " << duration*1000 << " ms" << std::endl;

    unsigned int* h_block_offsets = new unsigned int[num_blocks];
    cudaMemcpy(h_block_offsets, d_block_offsets, num_blocks*sizeof(unsigned int), cudaMemcpyDeviceToHost);

    unsigned char* h_compressed_data = new unsigned char[compressed_num_bytes];
    cudaMemcpy(h_compressed_data, d_compressed_data, compressed_num_bytes*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    int lastindex = filename.find_last_of(".");
    std::string name = filename.substr(0, lastindex);
    std::string output_filename(name+".pc");
    std::ofstream ofs(output_filename.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);

    serialize_tree(root, ofs);
    ofs.write(reinterpret_cast<const char*>(&num_bytes), sizeof(num_bytes));
    ofs.write(reinterpret_cast<const char*>(&compressed_num_bytes), sizeof(compressed_num_bytes));
    ofs.write(reinterpret_cast<const char*>(h_block_offsets), num_blocks*sizeof(unsigned int));
    ofs.write(reinterpret_cast<const char*>(h_compressed_data), compressed_num_bytes*sizeof(unsigned char));
    ofs.close();

    delete[] h_frequencies;
    delete[] h_min_frequencies;
    delete[] h_min_indices;
    delete[] h_compressed_data;

    cudaFree(d_vals);
    cudaFree(d_frequencies);
    cudaFree(d_min_frequencies);
    cudaFree(d_min_indices);
    cudaFree(d_count);
    cudaFree(d_codes);
    cudaFree(d_lengths);
    cudaFree(d_data_lengths);
    cudaFree(d_lengths_partial_sums);
    cudaFree(d_compressed_data);
}

void parallel_huffman_decode(std::ifstream& ifs, std::string filename) {

    unsigned int* d_block_offsets;
    unsigned char* d_compressed_data;

    double duration;

    Node* root;
    deserialize_tree(root, ifs);

    unsigned int decompressed_length;
    ifs.read(reinterpret_cast<char*>(&decompressed_length), sizeof(decompressed_length));

    unsigned int compressed_length;
    ifs.read(reinterpret_cast<char*>(&compressed_length), sizeof(compressed_length));

    unsigned int num_blocks = ceil((float)decompressed_length/(DATA_BLOCK_SIZE/256.0));

    std::cout << "Num blocks: " << num_blocks << std::endl;

    cudaMalloc(&d_block_offsets, num_blocks*sizeof(unsigned int));
    unsigned int* h_block_offsets = new unsigned int[num_blocks];
    ifs.read(reinterpret_cast<char*>(h_block_offsets), num_blocks*sizeof(unsigned int));
    cudaMemcpy(d_block_offsets, h_block_offsets, num_blocks*sizeof(unsigned int), cudaMemcpyHostToDevice);
 
    cudaMalloc(&d_compressed_data, compressed_length*sizeof(unsigned char));
    unsigned char* h_compressed_data = new unsigned char[compressed_length];
    ifs.read(reinterpret_cast<char*>(h_compressed_data), compressed_length*sizeof(unsigned char));
    cudaMemcpy(d_compressed_data, h_compressed_data, compressed_length*sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Approximation on upper bound of maximum encoding length (16 bits)
    //  so that a entire tree can be linearized.
    // NOTE: Not all array elemements will be used unless the tree is perfect. (Which won't be the case)
    // NOTE: This is just a guess FIXME
    unsigned int array_size = 1 << 20;

    std::clock_t decompression_start = std::clock();

    NodeArray* h_huffman_tree = new NodeArray[array_size];
    tree_to_array(h_huffman_tree, 0, root);

    NodeArray* d_huffman_tree;
    cudaMalloc(&d_huffman_tree, array_size*sizeof(NodeArray));
    cudaMemcpy(d_huffman_tree, h_huffman_tree, array_size*sizeof(NodeArray), cudaMemcpyHostToDevice);

    unsigned char* d_decompressed_data;
    cudaMalloc(&d_decompressed_data, decompressed_length*sizeof(unsigned char));
    decompress_data(d_compressed_data, d_huffman_tree, d_block_offsets, d_decompressed_data,
                    decompressed_length);

    duration = ( std::clock() - decompression_start ) / (double) CLOCKS_PER_SEC;
    std::cout << "Decompression time: " << duration*1000 << " ms" << std::endl;

    int lastindex = filename.find_last_of(".");
    std::string name = filename.substr(0, lastindex);
    std::string output_filename(name+".pd");
    std::ofstream ofs(output_filename.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);

    unsigned char* h_decompressed_data = new unsigned char[decompressed_length];
  
    cudaMemcpy(h_decompressed_data, d_decompressed_data, decompressed_length*sizeof(unsigned char),  cudaMemcpyDeviceToHost);

//    h_decompressed_data[decompressed_length-1] = '\n';
    ofs.write(reinterpret_cast<const char*>(h_decompressed_data), decompressed_length*sizeof(unsigned char));
    ofs.close();

    delete[] h_compressed_data;
    delete[] h_huffman_tree;
    delete[] h_decompressed_data;

    cudaFree(d_block_offsets);
    cudaFree(d_compressed_data);
    cudaFree(d_huffman_tree);
    cudaFree(d_decompressed_data);
}


