#include <queue>
#include <vector>
#include <stdlib.h>
#include <cmath>
#include <fstream>
#include <cstring>
#include <ctime>

#include "node.h"
#include "main.h"

// GenerateCode - generates a binary prefix code for a 2-tree
// Input:  root - the root of a 2-tree
// Output: Code[0:n-1] - array of binary strings, where Code[i] is the code for the symbol ai
void generate_code(Node *root, unsigned int code[], unsigned int length[]) {
    if (!root->get_left_child())
    {
        code[root->symbol_index] = root->get_value();
        length[root->symbol_index] = root->length;
    }
    else
    {
        Node *left = root->get_left_child();
        Node *right = root->get_right_child();
        left->set_value(root->get_value() & ~(1 << (root->length)));
        right->set_value(root->get_value() | (1 << (root->length)));
        left->length = root->length + 1;
        right->length = root->length + 1;
        generate_code(left, code, length);
        generate_code(right, code, length);
    }
}


// HuffmanCode
// Input:  a[], representing an alphabet, where a[i] == ai,
//         Freq[0:n-1] - an array of non-negative frequencies, where Freq[i] == fi
// Output: Code[0:n-1] - an array of binary strings for Huffman code, where Code[i] is the binary string encoding symbol ai, i=0,...,n-1
void huffman_code(Node* &root, unsigned char a[], unsigned int freq[], unsigned int code[], unsigned int lengths[], unsigned int numVals) {
    std::priority_queue<Node*, std::vector<Node*>, NodeGreater> q;

    // init leaf nodes
    for (int i = 0; i < numVals; i++) {
        if (freq[i] != 0) {
            Node *p = new Node();
            p->symbol_index = i;
            p->frequency = freq[i];
            q.push(p);
        }
    }

    while (q.size() > 1) {
        // remove smallest and second smallest frequencies from the queue
        Node *l = q.top();
        q.pop();

        Node *r = q.top();
        q.pop();

        // create a new subtree with the smallest nodes
        Node *subtree = new Node();
        subtree->set_left_child(l);
        subtree->set_right_child(r);

        // the new root's frequency is the sum of the children's frequencies
        subtree->frequency = (l->frequency) + (r->frequency);

        // insert the subtree into the heap
        q.push(subtree);
      }

    if (q.size() != 0) {
        root = q.top();
        generate_code(root, code, lengths);
    }
}

void compress_data(unsigned char* original_data, unsigned char* compressed_data, unsigned int* lengths,
                   unsigned int* codes, unsigned int num_bytes) {
    unsigned int byte_offset = 0;
    unsigned int bit_offset = 7;

    const unsigned int BITS_PER_BYTE = 8;

    for (int i = 0; i < num_bytes; i++)
    {
        unsigned int code = codes[original_data[i]];
        for (unsigned int j = 0; j < lengths[original_data[i]]; j++) {
            if ((code & (1 << j)) == (1 << j)) {
                compressed_data[byte_offset] |= (1 << bit_offset);
            } else {
                compressed_data[byte_offset] &= ~(1 << bit_offset);
            }
            bit_offset = (bit_offset - 1) % BITS_PER_BYTE;
            if (bit_offset == 7) byte_offset++;
        }
    }
}

void serial_huffman_encode(unsigned char* data, unsigned int num_bytes, std::string filename)
{
    const unsigned int NUM_VALS = 256;
    unsigned int frequencies[NUM_VALS];

//    data[num_bytes-1] = 254; //EOF char
//    unsigned char* bwt_data = new unsigned char[num_bytes];
//    burrow_wheelers_transform(data, num_bytes, bwt_data);

//    move_to_front_transform(bwt_data, num_bytes, data);

    std::clock_t histogram_start = std::clock();
    double duration;
    std::memset(frequencies, 0, NUM_VALS*sizeof(unsigned int));
    for (unsigned int i = 0; i < num_bytes; i++) {
        frequencies[data[i]]++;
    }
    duration = ( std::clock() - histogram_start ) / (double) CLOCKS_PER_SEC;
    std::cout << "Histogram time: " << duration*1000 << " ms" << std::endl;

    unsigned char* a = new unsigned char[NUM_VALS];
    for (unsigned int i = 0; i < NUM_VALS; i++) {
        a[i] = i;
    }

    unsigned int* lengths = new unsigned int[NUM_VALS];
    unsigned int* codes = new unsigned int[NUM_VALS];
    std::memset(lengths, 0, NUM_VALS*sizeof(unsigned int));
    std::memset(codes, 0, NUM_VALS*sizeof(unsigned int));

    Node* root;
    std::clock_t build_tree_start = std::clock();
    huffman_code(root, a, frequencies, codes, lengths, NUM_VALS);
    duration = ( std::clock() - build_tree_start ) / (double) CLOCKS_PER_SEC;
    std::cout << "Build tree time: " << duration*1000 << " ms" << std::endl;   

    unsigned int* data_lengths = new unsigned int[num_bytes];
    unsigned int compressed_length = 0;
    for (unsigned int i = 0; i < num_bytes; i++) {
        data_lengths[i] = lengths[data[i]];
        compressed_length += data_lengths[i];
    }

    unsigned int spare_bits = compressed_length % 8;
    compressed_length = std::ceil(compressed_length/8.0);

    unsigned char* compressed_data = new unsigned char[compressed_length];

    std::clock_t compress_start = std::clock();
    compress_data(data, compressed_data, lengths, codes, num_bytes);
    duration = ( std::clock() - compress_start ) / (double) CLOCKS_PER_SEC;
    std::cout << "Compress time: " << duration*1000 << " ms" << std::endl;

    int lastindex = filename.find_last_of(".");
    std::string name = filename.substr(0, lastindex);
    std::string output_filename(name+".sc");
    std::ofstream ofs(output_filename.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);

    serialize_tree(root, ofs);
    ofs.write(reinterpret_cast<const char*>(&num_bytes), sizeof(num_bytes));
    ofs.write(reinterpret_cast<const char*>(&compressed_length), sizeof(compressed_length));
    ofs.write(reinterpret_cast<const char*>(compressed_data), compressed_length);
    ofs.close();

//    delete[] bwt_data;
    delete[] a;
    delete[] lengths;
    delete[] codes;
    delete[] data_lengths;
    delete[] compressed_data;
}

void decode_data(unsigned char* compressed_data, unsigned int compressed_length,
                 unsigned char* decompressed_data, unsigned int decompressed_length, Node* root)
{
    const int BITS_PER_BYTE = 8;

    unsigned int byte_offset = 0;
    unsigned int bit_offset = 7;
    bool go_right;

    unsigned int decompressed_offset = 0;

    Node* current = root;

    while (decompressed_offset < decompressed_length) {
        go_right = ((compressed_data[byte_offset] & (1 << (bit_offset))) == (1 << (bit_offset)));

        if (go_right) {
            current = current->get_right_child();
        } else {
            current = current->get_left_child();
        }

        if (!current->get_left_child() && !current->get_right_child()) {
            decompressed_data[decompressed_offset++] = current->symbol_index;
            current = root;
        }

        bit_offset = (bit_offset - 1) % BITS_PER_BYTE;
        if (bit_offset == 7) byte_offset++;
    }
}

void serial_huffman_decode(std::ifstream& ifs, std::string filename)
{
    Node* root;
    deserialize_tree(root, ifs);

    unsigned int decompressed_length;
    ifs.read(reinterpret_cast<char*>(&decompressed_length), sizeof(decompressed_length));

    unsigned int compressed_length;
    ifs.read(reinterpret_cast<char*>(&compressed_length), sizeof(compressed_length));

    unsigned char* compressed_data = new unsigned char[compressed_length];
    ifs.read(reinterpret_cast<char*>(compressed_data), compressed_length);

    unsigned char* decompressed_data = new unsigned char[decompressed_length];

    std::clock_t decompress_start = std::clock();
    double duration;
    decode_data(compressed_data, compressed_length, decompressed_data, decompressed_length, root);
    duration = ( std::clock() - decompress_start ) / (double) CLOCKS_PER_SEC;
    std::cout << "Decompress time: " << duration*1000 << " ms" << std::endl;

//    unsigned char* bwt_data = new unsigned char[decompressed_length];
//    inverse_move_to_front_transform(decompressed_data, decompressed_length, bwt_data);

//    inverse_burrow_wheelers_transform(bwt_data, decompressed_length, decompressed_data, 254);

    int lastindex = filename.find_last_of(".");
    std::string name = filename.substr(0, lastindex);
    std::string output_filename(name+".sd");
    std::ofstream ofs(output_filename.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);

//    decompressed_data[decompressed_length-1] = '\n';
    ofs.write(reinterpret_cast<const char*>(decompressed_data), decompressed_length);
    ofs.close();

    delete[] compressed_data;
    delete[] decompressed_data;
//    delete[] bwt_data;
}

