#ifndef __MAIN_H__
#define __MAIN_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <iostream>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

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

#endif 

