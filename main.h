#ifndef __MAIN_H__
#define __MAIN_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <iostream>

#include "node.h"

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

const unsigned int DATA_BLOCK_SIZE = 4096;

struct NodeArray {
    NodeArray() {
        left = -1;
        right = -1;
    }
    unsigned int symbol_index;
    int left;
    int right;
};

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

void parallel_huffman_encode(unsigned char* data, unsigned int num_bytes, std::string filename);
void parallel_huffman_decode(std::ifstream& ifs, std::string filename);

void generate_code(Node *root, unsigned int code[], unsigned int length[]);
void serial_huffman_encode(unsigned char* data, unsigned int num_bytes, std::string filename);
void serial_huffman_decode(std::ifstream& ifs, std::string filename);

void burrow_wheelers_transform(unsigned char* data_in, unsigned int num_bytes,
                               unsigned char* data_out);
void inverse_burrow_wheelers_transform(unsigned char* data_in, unsigned int num_bytes,
                                       unsigned char* data_out, unsigned int EOF_char);

void move_to_front_transform(unsigned char* data_in, unsigned int num_bytes,
                             unsigned char* sequence);
void inverse_move_to_front_transform(unsigned char* sequence, unsigned int num_bytes,
                                     unsigned char* data_out);

long getFileSize(std::string filename);
void serialize_tree(Node* root, std::ofstream& ofs);
void deserialize_tree(Node* &d_root, std::ifstream& ifs);
void tree_to_array(NodeArray* nodes, unsigned int index, Node* root);
void print_tree(Node* root);

// These functions all make kernel calls
void computeHistogram(const unsigned char* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems);

void get_minimum2(const unsigned int* d_in,
                  const size_t numElems,
                  unsigned int* d_vals);

void update_histo_and_get_min_indices(unsigned int* d_in,
                                      unsigned int val1,
                                      unsigned int val2,
                                      unsigned int* d_indices,
                                      const size_t numElems);

void minimizeBins(unsigned int* d_in,
                  unsigned int* d_count,
                  const size_t numElems);

size_t get_compressed_length(unsigned int* d_lengths,
                                   unsigned char* d_original_data,
                                   unsigned int* d_data_lengths,
                                   unsigned int* d_lengths_partial_sums,
                                   const size_t num_bytes);

void compress_data(unsigned char* d_original_data,
                   unsigned int* d_codes,
                   unsigned int* d_lengths,
                   unsigned int* d_lengths_partial_sums,
                   unsigned char* d_encoded_data,
		   size_t compressed_num_bytes);

#endif

