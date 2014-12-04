#include "main.h"

__global__
void compress(unsigned char* d_original_data,
                 unsigned int* d_codes,
                 unsigned int* d_lengths,
                 unsigned int* d_block_offsets,
                 unsigned char* d_compressed_data)
{
    const unsigned int BITS_PER_BYTE = 8;

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int block_offset = d_block_offsets[blockIdx.x];
    unsigned int byte_offset = block_offset / BITS_PER_BYTE;
    unsigned int bit_offset = BITS_PER_BYTE - 1 - (block_offset % BITS_PER_BYTE);

    for (int i = 0; i < (DATA_BLOCK_SIZE/blockDim.x); i++)
    {
        unsigned int code = d_codes[d_original_data[id + i]];
        for (unsigned int j = 0; j < d_lengths[d_original_data[id + i]]; j++) {
            if ((code & (1 << j)) == (1 << j)) {
                d_compressed_data[byte_offset] |= (1 << bit_offset);
            } else {
                d_compressed_data[byte_offset] &= ~(1 << bit_offset);
            }
            bit_offset = (bit_offset - 1) % BITS_PER_BYTE;
            if (bit_offset == 7) byte_offset++;
        }
    }
}

__global__
void blelloch_scan_sum(unsigned int* d_in,
                       unsigned int* d_sums,
                       const size_t n)
{
    extern __shared__ unsigned int shdata[];
    int threadId = threadIdx.x;
    int offset = 1;

    shdata[2*threadId] = d_in[2*threadId];
    shdata[2*threadId+1] = d_in[2*threadId+1];

    //Reduce
    for (int d = n >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (threadId < d)
        {
            int ai = offset*(2*threadId+1)-1;
            int bi = offset*(2*threadId+2)-1;

            shdata[bi] += shdata[ai];
        }
        offset *= 2;
    }

    if (threadId == 0)
        shdata[n-1] = 0; // Set last element to identity element

    //Downsweep
    for (int d = 1; d < n; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (threadId < d)
        {
            int ai = offset*(2*threadId+1)-1;
            int bi = offset*(2*threadId+2)-1;

            float t = shdata[ai];
            shdata[ai] = shdata[bi];
            shdata[bi] += t;
        }
    }

    __syncthreads();

    // write results to device memory
    d_sums[2*threadId] = shdata[2*threadId];
    d_sums[2*threadId+1] = shdata[2*threadId+1];
}

__global__
void blelloch_scan_sum_large(unsigned int* const d_in,
                             unsigned int* d_sums_all,
                             const size_t n,
                             unsigned int* d_sums)
{
    int threadId = threadIdx.x;
    int offset = 1;

    unsigned int start_index = blockIdx.x*blockDim.x*2;

    d_sums_all[start_index + 2*threadId] = d_in[start_index + 2*threadId];
    d_sums_all[start_index + 2*threadId+1] = d_in[start_index + 2*threadId+1];

    //Reduce
    for (int d = n >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (threadId < d)
        {
            int ai = offset*(2*threadId+1)-1;
            int bi = offset*(2*threadId+2)-1;

        d_sums_all[start_index + bi] += d_sums_all[start_index + ai];
        }
        offset *= 2;
    }

    if (threadId == 0) {
        d_sums[blockIdx.x] = d_sums_all[start_index + n-1];
        d_sums_all[start_index + n-1] = 0; // Set last element to identity element
    }

    //Downsweep
    for (int d = 1; d < n; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (threadId < d)
        {
            int ai = offset*(2*threadId+1)-1;
            int bi = offset*(2*threadId+2)-1;

            float t = d_sums_all[start_index + ai];
            d_sums_all[start_index + ai] = d_sums_all[start_index + bi];
            d_sums_all[start_index + bi] += t;
        }
    }

    __syncthreads();
}

__global__
void add_constant(unsigned int* d_sums,
                  unsigned int* d_incr,
                  const size_t numElems)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= numElems)
        return;

    d_sums[pos] = d_sums[pos] + d_incr[blockIdx.x];
}

unsigned long upper_power_of_two(unsigned long v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

__global__
void set_data_lengths(unsigned int* d_lengths,
                      unsigned char* d_data_in,
                      unsigned int* d_data_lengths,
                      const size_t num_bytes)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= num_bytes)
        return;

    d_data_lengths[i] = d_lengths[d_data_in[i]];
}

__global__
void length_partial_sums_to_block_offsets(unsigned int* d_lengths_partial_sums, unsigned int* d_block_offsets)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if ((i % (DATA_BLOCK_SIZE/blockDim.x)) == 0) {
        d_block_offsets[i/(DATA_BLOCK_SIZE/blockDim.x)] = d_lengths_partial_sums[i];
    }
}

void large_scan_sum(unsigned int* const d_in,
                    unsigned int* d_all_sums,
                    const size_t numElems)
{
    const int B = 256;
    int numBlocks = ceil(((float)numElems)/B);
    int numBlocks2 = upper_power_of_two(numBlocks);

    unsigned int* sums;
    checkCudaErrors(cudaMalloc(&sums, numBlocks*sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(sums, 0, numBlocks*sizeof(unsigned int)));
    blelloch_scan_sum_large<<<numBlocks, B/2>>>(d_in, d_all_sums, B, sums);

    // Pad array by creating new array, zeroing all elements, and copying values
    unsigned int* padded_sums;
    checkCudaErrors(cudaMalloc(&padded_sums, numBlocks2*sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(padded_sums, 0, numBlocks2*sizeof(unsigned int)));
    checkCudaErrors(cudaMemcpy(padded_sums, sums, numBlocks*sizeof(unsigned int), cudaMemcpyDeviceToDevice));

    unsigned int* incr;
    checkCudaErrors(cudaMalloc(&incr, numBlocks2*sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(incr, 0, numBlocks2*sizeof(unsigned int)));
    blelloch_scan_sum<<<1, numBlocks2/2, numBlocks2*sizeof(unsigned int)>>>(padded_sums, incr, numBlocks2);

    add_constant<<<numBlocks, B>>>(d_all_sums, incr, numElems);

    checkCudaErrors(cudaFree(sums));
    checkCudaErrors(cudaFree(padded_sums));
    checkCudaErrors(cudaFree(incr));
}

size_t get_compressed_length(unsigned int* d_lengths,
                             unsigned char* d_original_data,
                             unsigned int* d_data_lengths,
                             unsigned int* d_lengths_partial_sums,
                             const size_t num_bytes)
{
    unsigned int compressed_num_bytes;

    int numBlocks = ceil(((float)num_bytes)/1024);
    set_data_lengths<<<numBlocks, 1024>>>(d_lengths, d_original_data, d_data_lengths, num_bytes);
    large_scan_sum(d_data_lengths, d_lengths_partial_sums, num_bytes);

    unsigned int* h_last_partial_sum = (unsigned int*)malloc(sizeof(unsigned int));
    unsigned int* h_last_length = (unsigned int*)malloc(sizeof(unsigned int));
    cudaMemcpy(h_last_partial_sum, &d_lengths_partial_sums[num_bytes-1], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_last_length, &d_data_lengths[num_bytes-1], sizeof(unsigned int), cudaMemcpyDeviceToHost);

    compressed_num_bytes = ceil((*h_last_partial_sum + *h_last_length)/8.0);

    return compressed_num_bytes;
}

void compress_data(unsigned char* d_original_data,
                   unsigned int* d_codes,
                   unsigned int* d_lengths,
                   unsigned int* d_lengths_partial_sums,
                   unsigned int* d_block_offsets,
                   unsigned char* d_compressed_data,
		   size_t num_bytes)
{
    int numBlocks = ceil(((float)num_bytes)/256);

    length_partial_sums_to_block_offsets<<<numBlocks, 256>>>(d_lengths_partial_sums, d_block_offsets);

    compress<<<numBlocks, 256>>>(d_original_data, d_codes, d_lengths, d_block_offsets, d_compressed_data);

}

