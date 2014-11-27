#include "main.h"
#include "node.h"

#include <ctime>
#include <string>
#include <cstring>
#include <sys/stat.h>
#include <fstream>

void generate_code(Node *root, unsigned int code[], unsigned int length[]);

long getFileSize(std::string filename)
{
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

void print_frequencies(unsigned int* freq, const size_t size)
{
    for (unsigned int i = 0; i < size; i++)
        std::cout << freq[i] << std::endl;
}

void serialize_tree(Node* root, std::ofstream& ofs)
{
    if (root == NULL) {
        unsigned char all_ones = 255;
        ofs.write(reinterpret_cast<const char*>(&all_ones), sizeof(all_ones));
        ofs.write(reinterpret_cast<const char*>(&all_ones), sizeof(all_ones));
        ofs.write(reinterpret_cast<const char*>(&all_ones), sizeof(all_ones));
        return;
    }

    unsigned char i = root->symbol_index;
    ofs.write(reinterpret_cast<const char*>(&i), sizeof(i));

    serialize_tree(root->get_left_child(), ofs);
    serialize_tree(root->get_right_child(), ofs);
}

void parallel_huffman_encode(char* data, unsigned int num_bytes, std::string filename)
{
    const unsigned int NUM_VALS = 256;

    unsigned char* d_vals;
    unsigned int* d_frequencies;
    unsigned int* d_min_frequencies;
    unsigned int* d_min_indices;

    unsigned int* h_frequencies = (unsigned int*)malloc(NUM_VALS*sizeof(unsigned int));
    unsigned int* h_min_frequencies = (unsigned int*)malloc(2*sizeof(unsigned int));
    unsigned int* h_min_indices = (unsigned int*)malloc(2*sizeof(unsigned int));

    cudaMalloc(&d_vals, num_bytes*sizeof(unsigned char));
    cudaMalloc(&d_frequencies, NUM_VALS*sizeof(unsigned int));
    cudaMalloc(&d_min_frequencies, 2*sizeof(unsigned int));
    cudaMalloc(&d_min_indices, 2*sizeof(unsigned int));

    cudaMemcpy(d_vals, data, num_bytes*sizeof(unsigned char), cudaMemcpyHostToDevice);

    computeHistogram(d_vals, d_frequencies, NUM_VALS, num_bytes);

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

    unsigned int* d_codes;
    unsigned int* d_lengths;
    unsigned int* d_data_lengths;
    unsigned int* d_lengths_partial_sums;
    unsigned char* d_encoded_data;
    cudaMalloc(&d_codes, NUM_VALS*sizeof(unsigned int));
    cudaMalloc(&d_lengths, NUM_VALS*sizeof(unsigned int));
    cudaMalloc(&d_data_lengths, num_bytes*sizeof(unsigned int));
    cudaMalloc(&d_lengths_partial_sums, num_bytes*sizeof(unsigned int));
    
    cudaMemcpy(d_codes, codes, NUM_VALS*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, lengths, NUM_VALS*sizeof(unsigned int), cudaMemcpyHostToDevice);

    size_t compressed_num_bytes;
    compress_data(d_vals, d_codes, d_lengths, d_data_lengths, d_lengths_partial_sums, d_encoded_data, num_bytes, compressed_num_bytes);

    unsigned char* h_encoded_data = (unsigned char*)malloc(compressed_num_bytes*sizeof(unsigned char));
    cudaMemcpy(h_encoded_data, d_encoded_data, compressed_num_bytes*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    int lastindex = filename.find_last_of(".");
    std::string name = filename.substr(0, lastindex);
    std::string output_filename(name+".huff");
    std::ofstream ofs(output_filename.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);

    serialize_tree(root, ofs);

    //TODO save length partial sum for every 32 lengths for block offset used to decode

    ofs.write(reinterpret_cast<const char*>(h_encoded_data), compressed_num_bytes);

    free(h_frequencies);
    free(h_min_frequencies);
    free(h_min_indices);
    free(h_encoded_data);

    cudaFree(d_vals);
    cudaFree(d_frequencies);
    cudaFree(d_min_frequencies);
    cudaFree(d_min_indices);
    cudaFree(d_count);
    cudaFree(d_codes);
    cudaFree(d_lengths);
    cudaFree(d_data_lengths);
    cudaFree(d_lengths_partial_sums);
    cudaFree(d_encoded_data);
}

void deserialize_tree(Node* &d_root, std::ifstream& ifs)
{
    char ch;
    ifs.get(ch);

    if (255 == ch) {
        ifs.get(ch);
        if (255 == ch) {
            ifs.get(ch);
            if (255 == ch) {
                return;
            }
            else {
                ifs.unget();
                ifs.unget();
            }
            
        } else {
            ifs.unget();
        }
    }

    Node* h_root = new Node;
    h_root->symbol_index = ch;

    deserialize_tree(h_root->get_left_child(), ifs);
    deserialize_tree(h_root->get_right_child(), ifs);
}

void tree_to_array(NodeArray* nodes, unsigned int index, Node* root)
{
    if (root == NULL)
        return;

    nodes[index].symbol_index = root->symbol_index;
    nodes[index].left = 2*index+1;
    nodes[index].right = 2*index+2;

    tree_to_array(nodes, 2*index+1, root->get_left_child());
    tree_to_array(nodes, 2*index+2, root->get_right_child());
}

void paralell_huffman_decode(std::ifstream& ifs)
{
    Node* h_root;
    deserialize_tree(h_root, ifs);

    unsigned int max_length = 0; // TODO Max reduce to find maximum length which will give the depth of the tree
    unsigned int array_size = 1 << (max_length+1);

    NodeArray* nodes = new NodeArray[array_size];
    std::memset(nodes, 0, array_size*sizeof(NodeArray));

    tree_to_array(nodes, 0, h_root);

    //TODO copy tree to device

    //TODO call decode kernel
}

int main (int argc, char** argv)
{
    bool run_parallel = true;
    bool encode = true;
    std::string input_filename;

    if (1 == argc) {
        std::cout << "Missing argument: filename" << std::endl;
        exit(1);
    }

    for (int i = 1; i < argc; i++) {
        char* arg = argv[i];
        if (arg[0] == '-') {
            if (strcmp("-run_parallel", arg) == 0) {
                if (i == (argc-1)) {
                    std::cout << "Missing argument for run_parallel" << std::endl;
                    exit(1);
                }
                else {
                    run_parallel = atoi(argv[i]);
                    i++;
                }
            }
        } else {
            input_filename = argv[i];
        }
    }

    if (input_filename.empty()) {
        std::cout << "Invalid arguments" << std::endl;
        exit(1);
    }


    std::clock_t start = std::clock();
    double duration;

    if (encode) {

        long num_bytes = getFileSize(input_filename);
        char* data = new char[num_bytes];
        std::ifstream ifs(input_filename.c_str());
        if(!ifs) {
            std::cout << "Failed to open file: " << input_filename << std::endl;
        }

        ifs.read(data, num_bytes);

        if (run_parallel) {
            parallel_huffman_encode(data, num_bytes, input_filename);
        } else {

        }

    } else {

    }

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

    std::cout << "Elapsed time: " << duration*1000 << " ms" << std::endl;

    return 0;
}


