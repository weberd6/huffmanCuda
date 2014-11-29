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
                unsigned int* d_block_length,
                unsigned int* d_block_length_sums,
                unsigned char* d_decoded_data,
                const size_t num_bytes)
{
/*    int id = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int j = 0; // Root index
    unsigned int n = d_block_length[blockIdx.x]-1;
    unsigned int d_block_end_offset = d_block_length_sums[blockIdx.x] + n;
    bool go_left;

    for (int i = 0; i < (DATA_BLOCK_SIZE/blockDim.x); i++)
    {
        
    }*/
}


