#include "main.h"

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

	cudaMalloc(&d_vals, num_bytes*sizeof(unsigned char));
	cudaMalloc(&d_histo, 256*sizeof(unsigned int));

	cudaMemcpy(d_vals, data, num_bytes*sizeof(unsigned char), cudaMemcpyHostToDevice);

	computeHistogram(d_vals, d_histo, 256, num_bytes);
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


