NVCC=/usr/local/cuda/bin/nvcc

CUDA_INCLUDEPATH=/usr/local/cuda/include

CUDA_LIBPATH=/usr/local/cuda/lib

NVCC_OPTS=-Wno-deprecated-gpu-targets #-ccbin=gcc-4.8

NAME=compress

$(NAME): main.o histo.o min2.o compress.o decompress.o node.o parallel.o serial.o bwt.o mtf.o\
serialize.o Makefile
	$(NVCC) -o $(NAME) main.o histo.o min2.o compress.o decompress.o node.o parallel.o \
	serial.o bwt.o mtf.o serialize.o $(NVCC_OPTS)

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
	$(NVCC) -c node.cpp $(NVCC_OPTS)

parallel.o: parallel.cu main.h
	$(NVCC) -c parallel.cu $(NVCC_OPTS)

serial.o: serial.cpp main.h
	$(NVCC) -c serial.cpp $(NVCC_OPTS)

bwt.o: bwt.cpp main.h
	$(NVCC) -c bwt.cpp $(NVCC_OPTS)

mtf.o: mtf.cpp main.h
	$(NVCC) -c mtf.cpp $(NVCC_OPTS)

serialize.o: serialize.cpp main.h
	$(NVCC) -c serialize.cpp $(NVCC_OPTS)

clean:
	rm -f *.o $(NAME)



