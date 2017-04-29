#include "main.h"
#include "node.h"

#include <stdio.h>

__global__
void decompress(unsigned char* d_compressed_data,
                NodeArray* d_huffman_tree,
                unsigned int* d_block_offsets,
                unsigned char* d_decompressed_data,
                size_t num_bytes) {

    const unsigned int BITS_PER_BYTE = 8;

    unsigned int id = blockIdx.x * blockDim.x;

    if ((id + threadIdx.x) > (ceilf(num_bytes/(DATA_BLOCK_SIZE/blockDim.x)))) {
        return;
    }

    unsigned int block_offset = d_block_offsets[id + threadIdx.x];
    unsigned int byte_offset = block_offset / BITS_PER_BYTE;
    unsigned int bit_offset = BITS_PER_BYTE - 1 - (block_offset % BITS_PER_BYTE);  

    bool go_right;
    unsigned int current = 0; // rooot
    unsigned int i = 0;
    unsigned int index;

    while ((i < (DATA_BLOCK_SIZE/(blockDim.x))) && ((id + i) < num_bytes)) {
    
        go_right = ((d_compressed_data[byte_offset] & (1 << (bit_offset))) == (1 << (bit_offset)));

        if (go_right) {
            current = d_huffman_tree[current].right;
        } else {
            current = d_huffman_tree[current].left;
        }

        if ((d_huffman_tree[current].left == -1) && (d_huffman_tree[current].right == -1)) {
            index = blockIdx.x * DATA_BLOCK_SIZE + threadIdx.x * DATA_BLOCK_SIZE/blockDim.x + i;
            d_decompressed_data[index] = d_huffman_tree[current].symbol_index;
            current = 0;
            i++;
        }

        bit_offset = (bit_offset - 1) % BITS_PER_BYTE;
        if (bit_offset == 7) byte_offset++;
    }
}


void decompress_data(unsigned char* d_compressed_data,
                NodeArray* d_huffman_tree,
                unsigned int* d_block_offsets,
                unsigned char* d_decompressed_data,
                unsigned int num_bytes) {

    int numBlocks = ceil(((float)num_bytes)/DATA_BLOCK_SIZE);   
    decompress<<<numBlocks, 256>>>(d_compressed_data, d_huffman_tree, d_block_offsets, d_decompressed_data, num_bytes);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

