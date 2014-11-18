#include "main.h"

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

  unsigned int second_min;
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (threadId < s)
    {
      second_min = max(shdata_min[threadId], shdata_min[threadId + s]);
      shdata_min[threadId] = min(shdata_min[threadId], shdata_min[threadId + s]);
    }
    __syncthreads();
  }

  if (threadId == 0)
  {
    d_vals[0] = shdata_min[0];
    d_vals[1] = second_min;
  }
}

// Assumes numElems is a power of 2 and is less than 1024
void get_minimum2(const unsigned int* d_in,
		  const size_t numElems,
		  unsigned int* d_vals)
{
  reduce_min2<<<1, numElems, numElems*sizeof(unsigned int)>>>(d_in, d_vals, numElems);
}



