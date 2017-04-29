#include <ctime>
#include <string>
#include <cstring>
#include <fstream>
#include <iostream> 

#include "main.h"

int main (int argc, char** argv) {

    bool parallel = false;
    bool encode = true;
    std::string input_filename;

    if (1 == argc) {
        std::cout << "Missing argument: filename" << std::endl;
        exit(1);
    }

    for (int i = 1; i < argc; i++) {
        char* arg = argv[i];
        if (arg[0] == '-') {
            if (strcmp("-p", arg) == 0) {
                parallel = true;
            } else if (strcmp("-s", arg) == 0) {
                parallel = false;
            } else if (strcmp("-c", arg) == 0) {
                encode = true;
            } else if (strcmp("-d", arg) == 0) {
                encode = false;
            } else {
                std::cout << "Invalid option: " << arg << std::endl;
            }
        } else {
            input_filename = argv[i];
        }
    }

    if (input_filename.empty()) {
        std::cout << "Invalid arguments" << std::endl;
        exit(1);
    }

    std::ifstream ifs(input_filename.c_str());
    if(!ifs) {
        std::cout << "Failed to open file: " << input_filename << std::endl;
        exit(1);
    }

    if (encode) {
        
        long num_bytes = getFileSize(input_filename);
        char* data = new char[num_bytes];
        ifs.read(data, num_bytes);

        std::clock_t start = std::clock();
        double duration;

        if (parallel) {
            parallel_huffman_encode(reinterpret_cast<unsigned char*>(data), num_bytes, input_filename);
        } else {
            serial_huffman_encode(reinterpret_cast<unsigned char*>(data), num_bytes, input_filename);
        }

        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
        std::cout << "Total time: " << duration*1000 << " ms" << std::endl;
    
    } else {
 
        std::clock_t start = std::clock();
        double duration;

        if (parallel) {
            parallel_huffman_decode(ifs, input_filename);
        } else {
            serial_huffman_decode(ifs, input_filename);
        }

        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
        std::cout << "Total time: " << duration*1000 << " ms" << std::endl;
    }

    ifs.close();

    return 0;
}

