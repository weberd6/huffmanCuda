#include "main.h"

__global__
void update_histo(unsigned int* d_in,
                unsigned int val1,
		unsigned int val2,
		unsigned int* d_indices,
                const size_t numElems)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  extern __shared__ int found[];

  found[0] = found[1] = false;
  __syncthreads();

  if ((i < numElems) && (0 == atomicCAS(&found[0], 0, 1)) && (d_in[i] == val1)) {
    d_in[i] = 0xFFFFFFFF;
    d_indices[0] = i;
    goto done;
  }

  if ((i < numElems) && (0 == atomicCAS(&found[1], 0, 1)) && (d_in[i] == val2)) {
    d_in[i] = val1 + val2;
    d_indices[1] = i;
  }

  done:
    return;
}

__global__
void reduce_min2(const unsigned int* d_in,
                unsigned int* d_vals,
                const size_t numElems)
{ 
 int i = threadIdx.x + blockDim.x * blockIdx.x;
 int threadId = threadIdx.x;

  extern __shared__ unsigned int shdata_min[];

  if (i >= numElems)
    shdata_min[threadId] = 0xFFFFFFFF;  // Infinity
  else
    shdata_min[threadId] = d_in[i];

  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 1; s >>= 1)
  {
    if (threadId < s)
    {
      shdata_min[threadId] = min(shdata_min[threadId], shdata_min[threadId + s]);
      shdata_min[threadId + 1] = min(shdata_min[threadId + 1], shdata_min[threadId + s + 1]);
    }
    __syncthreads();
  }

  if (threadId == 0)
  {
    d_vals[0] = shdata_min[0];
    d_vals[1] = shdata_min[1];
  }
}

void update_histo_and_get_min_indices(unsigned int* d_in,
				unsigned int val1,
				unsigned int val2,
				unsigned int* d_indices,
				const size_t numElems)
{
  update_histo<<<1, numElems, 2*sizeof(unsigned int)>>>(d_in, val1, val2, d_indices, numElems);
}

// Assumes numElems is a power of 2 and is less than 1024
void get_minimum2(const unsigned int* d_in,
		  const size_t numElems,
		  unsigned int* d_vals)
{
  reduce_min2<<<1, numElems, numElems*sizeof(int)>>>(d_in, d_vals, numElems);
}



