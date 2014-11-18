#include "main.h"

__global__
void updateHisto(unsigned int* d_in,
                unsigned int val1,
		unsigned int val2,
		unsigned int* d_indices,
                const size_t numElems)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  extern __shared__ int found[];

  found[0] = found[1] = 0;
  __syncthreads();

  if ((d_in[i] == val1) && (0 == atomicCAS(&found[0], 0, 1))) {
    d_in[i] = 0xFFFFFFFF;
    d_indices[0] = i;
    goto done;
  }

  if ((d_in[i] == val2) && (0 == atomicCAS(&found[1], 0, 1))) {
    d_in[i] = val1 + val2;
    d_indices[1] = i;
  }

  done:
    return;
}

__global__
void markZeroBins(unsigned int* d_in,
		unsigned int* d_count,
		const size_t numElems)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  
  if ((i < numElems) && (0 == d_in[i])) {
    d_in[i] = 0xFFFFFFFF;
    atomicSub(d_count, 1);
  }

}

__global__
void histo(const unsigned char* const vals, //INPUT
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

void update_histo_and_get_min_indices(unsigned int* d_in,
				unsigned int val1,
				unsigned int val2,
				unsigned int* d_indices,
				const size_t numElems)
{
  updateHisto<<<1, numElems, 2*sizeof(unsigned int)>>>(d_in, val1, val2, d_indices, numElems);
}

void minimizeBins(unsigned int* d_in,
		unsigned int* d_count,
		const size_t numElems)
{
  markZeroBins<<<1, numElems>>>(d_in, d_count, numElems);  
}

void computeHistogram(const unsigned char* const d_vals, //INPUT
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

