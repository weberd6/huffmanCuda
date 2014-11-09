#include "main.h"

__global__
void histo(const char* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  extern __shared__ unsigned int s_histo[];
  int pos = blockIdx.x * blockDim.x + threadIdx.x;

  if (pos >= numVals)
      return;

  // Zero all shared
  s_histo[threadIdx.x] = 0;
  __syncthreads();

  atomicAdd(&s_histo[vals[pos]], 1);

  __syncthreads();

  atomicAdd(&histo[threadIdx.x], s_histo[threadIdx.x]);
}

void computeHistogram(const char* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  
  const int threadsPerBlock = 1024;
  int numBlocks = ceil(((float)numElems)/threadsPerBlock);
  histo<<<numBlocks, threadsPerBlock, threadsPerBlock*sizeof(unsigned int)>>>
      (d_vals, d_histo, numElems);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

