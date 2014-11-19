#include "main.h"
#include "node.h"

#include <ctime>
#include <string>
#include <cstring>
#include <sys/stat.h>
#include <fstream>
#include <cassert>

long getFileSize(std::string filename)
{
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

void print_frequencies(unsigned int* freq, const size_t size)
{
	for (unsigned int i = 0; i < size; i++)
		std::cout << freq[i] << std::endl;
}

void parallel_huffman(char* data, unsigned int num_bytes)
{
	const unsigned int NUM_VALS = 256;

	unsigned char* d_vals;
	unsigned int* d_frequencies;
	unsigned int* d_min_frequencies;
	unsigned int* d_min_indices;

	unsigned int* h_frequencies = (unsigned int*)malloc(NUM_VALS*sizeof(unsigned int));
	unsigned int* h_min_frequencies = (unsigned int*)malloc(2*sizeof(unsigned int));
	unsigned int* h_min_indices = (unsigned int*)malloc(2*sizeof(unsigned int));	

	cudaMalloc(&d_vals, num_bytes*sizeof(unsigned char));
	cudaMalloc(&d_frequencies, NUM_VALS*sizeof(unsigned int));
	cudaMalloc(&d_min_frequencies, 2*sizeof(unsigned int));
	cudaMalloc(&d_min_indices, 2*sizeof(unsigned int));

	cudaMemcpy(d_vals, data, num_bytes*sizeof(unsigned char), cudaMemcpyHostToDevice);

	computeHistogram(d_vals, d_frequencies, NUM_VALS, num_bytes);

	unsigned int count = NUM_VALS;
	unsigned int* d_count;
	cudaMalloc(&d_count, sizeof(unsigned int));
	cudaMemcpy(d_count, &count, sizeof(count), cudaMemcpyHostToDevice);
	minimizeBins(d_frequencies, d_count, NUM_VALS);
	cudaMemcpy(&count, d_count, sizeof(count), cudaMemcpyDeviceToHost);

	cudaMemcpy(h_frequencies, d_frequencies, NUM_VALS*sizeof(unsigned int), cudaMemcpyDeviceToHost);

	unsigned int sum = 0;
	Node leaf_nodes[NUM_VALS];
	Node* node_by_index[NUM_VALS];

	for (int i = 0; i < NUM_VALS; i++) {
		if (h_frequencies[i] != 0xFFFFFFFF) sum += h_frequencies[i];
		leaf_nodes[i].frequency = h_frequencies[i];
		node_by_index[i] = &leaf_nodes[i];
	}

	Node* root;
	Node* l;
	Node* r;
	while (count > 1)
	{
		get_minimum2(d_frequencies, NUM_VALS, d_min_frequencies);
		cudaMemcpy(h_min_frequencies, d_min_frequencies, 2*sizeof(unsigned int), cudaMemcpyDeviceToHost);

//		std::cout << "Mins: " << h_min_frequencies[0] << " " << h_min_frequencies[1] << std::endl;

		update_histo_and_get_min_indices(d_frequencies, h_min_frequencies[0], h_min_frequencies[1], d_min_indices, NUM_VALS);
		cudaMemcpy(h_min_indices, d_min_indices, 2*sizeof(unsigned int), cudaMemcpyDeviceToHost);

		l = node_by_index[h_min_indices[0]];
		r = node_by_index[h_min_indices[1]];
//		std::cout << "Nodes: " << l->frequency << " " << r->frequency << std::endl;

		root = new Node(l, r, l->frequency + r->frequency);
		node_by_index[h_min_indices[1]] = root;

		count--;
	}

//	std::cout << "\nSize of file: " << num_bytes << " bytes" << std::endl;
//	std::cout << "Sum of frequencies: " << sum << std::endl;
//	std::cout << "Root huffman frequency: " <<  root->frequency << std::endl;
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
	if(!ifs) {
		std::cout << "Failed to open file: " << input_filename << std::endl;
	}

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


