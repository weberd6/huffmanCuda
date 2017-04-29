#include "main.h"

__global__
void reduce_min2(const unsigned int* d_in,
                unsigned int* d_vals,
                const size_t numElems) {

    int threadId = threadIdx.x;

    extern __shared__ unsigned int shdata_min[];

    if (threadId >= numElems)
        shdata_min[threadId] = 0xFFFFFFFF;  // Infinity
    else
        shdata_min[threadId] = d_in[threadId];

    __syncthreads();

    unsigned int first_min;
    unsigned int second_min;
    for (unsigned int s = blockDim.x / 2; s > 1; s >>= 1)
    {
        if ((threadId < s) && ((threadId % 2) == 0))
        {
            if ((shdata_min[threadId] < shdata_min[threadId+s])
                && (shdata_min[threadId+1] < shdata_min[threadId+s])
                && (shdata_min[threadId] < shdata_min[threadId+s+1])
                && (shdata_min[threadId+1] < shdata_min[threadId+s+1]))
            {
                first_min = min(shdata_min[threadId], shdata_min[threadId+1]);
                second_min = max(shdata_min[threadId], shdata_min[threadId+1]);
            }
            else if ((shdata_min[threadId+s] < shdata_min[threadId])
                    && (shdata_min[threadId+s] < shdata_min[threadId+1])
                    && (shdata_min[threadId+s+1] < shdata_min[threadId])
                    && (shdata_min[threadId+s+1] < shdata_min[threadId+1]))
            {
                first_min = min(shdata_min[threadId+s], shdata_min[threadId+s+1]);
                second_min = max(shdata_min[threadId+s], shdata_min[threadId+s+1]);
            }
            else {
                first_min = min(min(shdata_min[threadId], shdata_min[threadId+1]), min(shdata_min[threadId+s], shdata_min[threadId+s+1]));
                second_min = max(min(shdata_min[threadId], shdata_min[threadId+1]), min(shdata_min[threadId+s], shdata_min[threadId+s+1]));
            }

            shdata_min[threadId] = first_min;
            shdata_min[threadId+1] = second_min;
        }
        __syncthreads();
    }

    if (threadId == 0)
    {
        d_vals[0] = shdata_min[0];
        d_vals[1] = shdata_min[1];
    }
}

// Assumes numElems is a power of 2 and is less than 1024
void get_minimum2(const unsigned int* d_in,
                  const size_t numElems,
                  unsigned int* d_vals) {
    
    reduce_min2<<<1, numElems, numElems*sizeof(unsigned int)>>>(d_in, d_vals, numElems);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}



