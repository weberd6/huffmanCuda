#include "main.h"
#include "node.h"

__global__
void reduce_max(const unsigned int* d_in,
                float* d_out,
                const size_t numRows,
                const size_t numCols)
{
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int threadId = threadIdx.x;

  extern __shared__ float shdata_max[];

  if (myId >= numRows*numCols)
    shdata_max[threadId] = 0;
  else
    shdata_max[threadId] = d_in[myId];
    
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (threadId < s)
    {
      shdata_max[threadId] = max(shdata_max[threadId], shdata_max[threadId + s]);
    }
    __syncthreads();
  }

  if (threadId == 0)
  {
    d_out[blockIdx.x] = shdata_max[0];
  }
}

__global__
void decompress(unsigned char* d_compressed_data,
                NodeArray* d_huffman_tree,
                unsigned int* d_block_offsets,
                unsigned char* d_decompressed_data,
                const size_t num_bytes)
{
    const unsigned int BITS_PER_BYTE = 8;

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int block_offset = d_block_offsets[blockIdx.x];
    unsigned int byte_offset = block_offset / BITS_PER_BYTE;
    unsigned int bit_offset = BITS_PER_BYTE - 1 - (block_offset % BITS_PER_BYTE);  
    bool go_right;

    unsigned int current = 0; // rooot

    for (int i = 0; i < (DATA_BLOCK_SIZE/blockDim.x); i++) {
        go_right = ((d_compressed_data[byte_offset] & (1 << (bit_offset))) == (1 << (bit_offset)));

        if (go_right) {
            current = d_huffman_tree[current].right;
        } else {
            current = d_huffman_tree[current].left;
        }

        if ((d_huffman_tree[d_huffman_tree[current].left].left != -1) 
            	&& (d_huffman_tree[d_huffman_tree[current].right].left != -1)) {

            d_decompressed_data[id + i] = d_huffman_tree[current].symbol_index;
            current = 0;
        }

        bit_offset = (bit_offset - 1) % BITS_PER_BYTE;
        if (bit_offset == 7) byte_offset++;
    }
}


