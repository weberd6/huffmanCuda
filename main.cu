#include "main.h"

int main (int argc, char** argv) {
	// Ignore any cmd line args for now, later we can get a filename in/out

	unsigned int* d_vals;
	unsigned int* d_histo;
	unsigned int numElems = 1048576;
	unsigned int numBins = 128;

	cudaMalloc(&d_vals, numElems*sizeof(unsigned int));
	cudaMalloc(&d_histo, numBins*sizeof(unsigned int));

	cudaMemset(d_vals, 0, numElems*sizeof(unsigned int));
	cudaMemset(d_histo, 0, numBins*sizeof(unsigned int));

	computeHistogram(d_vals, d_histo, numBins, numElems);

	return 0;
}


