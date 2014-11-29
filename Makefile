CC=g++
NVCC=/usr/local/cuda-5.0/bin/nvcc

CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include
#CUDA_INCLUDEPATH=/usr/local/cuda/lib64/include
#CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include
#CUDA_INCLUDEPATH=/Developer/NVIDIA/CUDA-5.0/include
#CUDA_INCLUDEPATH=/usr/local/cuda/include

CUDA_LIBPATH=/usr/local/cuda-5.0/lib64
#CUDA_LIBPATH=/usr/local/cuda/lib

NVCC_OPTS=-O3 -arch=sm_20 -m64 -g

GCC_OPTS=-O3 -m64

NAME=huffman

$(NAME): main.o histo.o min2.o compress.o decompress.o node.o parallel.o serial.o serialize.o Makefile
	$(NVCC) -o $(NAME) main.o histo.o min2.o compress.o decompress.o node.o parallel.o serial.o serialize.o -L $(NVCC_OPTS)

main.o: main.cu main.h node.h
	$(NVCC) -c main.cu  -l $(CUDA_LIBPATH) -I $(CUDA_INCLUDEPATH) $(NVCC_OPTS)

histo.o: histo.cu main.h
	$(NVCC) -c histo.cu $(NVCC_OPTS)

min2.o: min2.cu main.h
	$(NVCC) -c min2.cu $(NVCC_OPTS)

compress.o: compress.cu main.h
	$(NVCC) -c compress.cu $(NVCC_OPTS)

decompress.o: decompress.cu main.h
	$(NVCC) -c decompress.cu $(NVCC_OPTS)

node.o: node.cpp node.h
	$(CC) -c node.cpp $(GCC_OPTS)

parallel.o: parallel.cu main.h
	$(NVCC) -c parallel.cu $(NVCC_OPTS)

serial.o: serial.cpp main.h
	$(CC) -c serial.cpp $(GCC_OPTS)

serialize.o: serialize.cpp main.h
	$(CC) -c serialize.cpp $(GCC_OPTS)

clean:
	rm -f *.o $(NAME)



