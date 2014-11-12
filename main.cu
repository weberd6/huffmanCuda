#include "main.h"
#include "node.h"

#include <ctime>
#include <string>
#include <cstring>
#include <sys/stat.h>
#include <fstream>

long getFileSize(std::string filename)
{
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

void parallel_huffman(char* data, unsigned int num_bytes) {
	char* d_vals;
	unsigned int* d_histo;
	unsigned int* d_min_vals;
	unsigned int* d_min_indices;

	unsigned int* h_histo = (unsigned int*)malloc(256*sizeof(unsigned int));
	unsigned int* h_min_vals = (unsigned int*)malloc(2*sizeof(unsigned int));
	unsigned int* h_min_indices = (unsigned int*)malloc(2*sizeof(unsigned int));	

	cudaMalloc(&d_vals, num_bytes*sizeof(unsigned char));
	cudaMalloc(&d_histo, 256*sizeof(unsigned int));
	cudaMalloc(&d_min_vals, 2*sizeof(unsigned int));
	cudaMalloc(&d_min_indices, 2*sizeof(unsigned int));

	cudaMemcpy(d_vals, data, num_bytes*sizeof(unsigned char), cudaMemcpyHostToDevice);

	computeHistogram(d_vals, d_histo, 256, num_bytes);
	cudaMemcpy(h_histo, d_histo, 256*sizeof(unsigned int), cudaMemcpyDeviceToHost);

	size_t numVals = 256;

	// TODO get rid of histo bins with 0 count

	Node* huffman_nodes = new Node[numVals];
	for (int i = 0; i < numVals; i++) {
		huffman_nodes[i].set_value(h_histo[i]);
	}

	unsigned int count = numVals;
	while (count > 1)
	{
		get_minimum2(d_histo, numVals, d_min_vals);
		cudaMemcpy(h_min_vals, d_min_vals, 2*sizeof(unsigned int), cudaMemcpyDeviceToHost);

		update_histo_and_get_min_indices(d_histo, h_min_vals[0], h_min_vals[1], d_min_indices, numVals);
		cudaMemcpy(h_min_indices, d_min_indices, 2*sizeof(unsigned int), cudaMemcpyDeviceToHost);

		Node* new_node = new Node(&huffman_nodes[h_min_indices[0]], &huffman_nodes[h_min_indices[1]], h_min_vals[0] + h_min_vals[1]);
		count--;
	}

}

int main (int argc, char** argv) {
	
	bool run_parallel = true;
	std::string input_filename;

	if (1 == argc) {
		std::cout << "Missing argument: filename" << std::endl;
		exit(1);
	}

	for (int i = 1; i < argc; i++) {
		char* arg = argv[i];
		if (arg[0] == '-') {
			if (strcmp("-run_parallel", arg) == 0) {
				if (i == (argc-1)) {
					std::cout << "Missing argument for run_parallel" << std::endl;
					exit(1);
				}
				else {
					run_parallel = argv[i];
					i++;
				}
			}
		} else {
			input_filename = argv[i];
		}
	}

	if (input_filename.empty()) {
		std::cout << "Invalid arguments" << std::endl;
		exit(1);
	}

	long num_bytes = getFileSize(input_filename);
	char* data = new char[num_bytes];
	std::ifstream ifs(input_filename.c_str());
	ifs.read(data, num_bytes);

	std::clock_t start = std::clock();
	double duration;

	if (run_parallel) {
		parallel_huffman(data, num_bytes);
	} else {

	}

	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

	std::cout << "Elapsed time: " << duration*1000 << " ms" << std::endl;

	return 0;
}


