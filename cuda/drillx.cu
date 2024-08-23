#include <stdint.h>
#include <stdio.h>
#include <vector>
#include "drillx.h"
#include "equix/include/equix.h"
#include "hashx/include/hashx.h"
#include "equix/src/context.h"
#include "equix/src/solver.h"
#include "equix/src/solver_heap.h"
#include "hashx/src/context.h"

const int BATCH_SIZE = 16384;
__device__ __constant__ int NUM_HASHING_ROUNDS;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err); \
        } \
    } while (0)

extern "C" void set_num_hashing_rounds(int rounds) {
    CUDA_CHECK(cudaMemcpyToSymbol(NUM_HASHING_ROUNDS, &rounds, sizeof(int)));
}

extern "C" void hash(uint8_t *challenge, uint8_t *nonce, uint64_t *out) {
    MemoryPool memPool(BATCH_SIZE);

    std::vector<uint8_t> seed(40);
    memcpy(seed.data(), challenge, 32);

    for (int i = 0; i < BATCH_SIZE; i++) {
        uint64_t nonce_offset = *((uint64_t*)nonce) + i;
        memcpy(seed.data() + 32, &nonce_offset, 8);
        memPool.ctxs[i] = hashx_alloc(HASHX_INTERPRETED);
        if (!memPool.ctxs[i] || !hashx_make(memPool.ctxs[i], seed.data(), 40)) {
            return;
        }
    }

    int threadsPerBlock = 256;  // Adjusted for optimal occupancy
    int blocksPerGrid = (BATCH_SIZE * INDEX_SPACE + threadsPerBlock - 1) / threadsPerBlock;

    cudaStream_t streams[2];
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // Allocate pinned host memory for async data transfer
    uint64_t *host_out[2];
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaHostAlloc(&host_out[i], BATCH_SIZE * INDEX_SPACE * sizeof(uint64_t), cudaHostAllocDefault));
    }

    int current_stream = 0;

    // Launch computation on streams and overlap with memory transfers
    for (int batch_start = 0; batch_start < BATCH_SIZE; batch_start += BATCH_SIZE / 2) {
        int batch_size = std::min(BATCH_SIZE / 2, BATCH_SIZE - batch_start);

        do_hash_stage0i<<<blocksPerGrid, threadsPerBlock, 0, streams[current_stream]>>>(
            memPool.ctxs + batch_start, memPool.hash_space + batch_start, NUM_HASHING_ROUNDS);
        CUDA_CHECK(cudaGetLastError());

        for (int i = 0; i < batch_size; i++) {
            CUDA_CHECK(cudaMemcpyAsync(host_out[current_stream] + i * INDEX_SPACE, 
                memPool.hash_space[batch_start + i], 
                INDEX_SPACE * sizeof(uint64_t), 
                cudaMemcpyDeviceToHost, 
                streams[current_stream]));
        }

        current_stream = 1 - current_stream; // Switch between streams
    }

    // Synchronize streams and copy results to output array
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        memcpy(out + i * (BATCH_SIZE / 2) * INDEX_SPACE, host_out[i], (BATCH_SIZE / 2) * INDEX_SPACE * sizeof(uint64_t));
        CUDA_CHECK(cudaFreeHost(host_out[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
}

__global__ void do_hash_stage0i(hashx_ctx** ctxs, uint64_t** hash_space, int num_hashing_rounds) {
    uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item < BATCH_SIZE * INDEX_SPACE) {
        uint32_t batch_idx = item / INDEX_SPACE;
        uint32_t i = item % INDEX_SPACE;

        for (int round = 0; round < num_hashing_rounds; ++round) {
            hash_stage0i(ctxs[batch_idx], hash_space[batch_idx], i);
        }
    }
}

extern "C" void solve_all_stages(uint64_t *hashes, uint8_t *out, uint32_t *sols, int num_sets) {
    uint64_t *d_hashes;
    solver_heap *d_heaps;
    equix_solution *d_solutions;
    uint32_t *d_num_sols;

    CUDA_CHECK(cudaMalloc(&d_hashes, num_sets * INDEX_SPACE * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_heaps, num_sets * sizeof(solver_heap)));
    CUDA_CHECK(cudaMalloc(&d_solutions, num_sets * EQUIX_MAX_SOLS * sizeof(equix_solution)));
    CUDA_CHECK(cudaMalloc(&d_num_sols, num_sets * sizeof(uint32_t)));

    equix_solution *h_solutions;
    uint32_t *h_num_sols;
    CUDA_CHECK(cudaHostAlloc(&h_solutions, num_sets * EQUIX_MAX_SOLS * sizeof(equix_solution), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_num_sols, num_sets * sizeof(uint32_t), cudaHostAllocDefault));

    CUDA_CHECK(cudaMemcpy(d_hashes, hashes, num_sets * INDEX_SPACE * sizeof(uint64_t), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256; // Adjusted for optimal occupancy
    int blocksPerGrid = (num_sets + threadsPerBlock - 1) / threadsPerBlock;

    solve_all_stages_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_hashes, d_heaps, d_solutions, d_num_sols);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_solutions, d_solutions, num_sets * EQUIX_MAX_SOLS * sizeof(equix_solution), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_num_sols, d_num_sols, num_sets * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_sets; i++) {
        sols[i] = h_num_sols[i];
        if (h_num_sols[i] > 0) {
            memcpy(out + i * sizeof(equix_solution), &h_solutions[i * EQUIX_MAX_SOLS], sizeof(equix_solution));
        }
    }

    CUDA_CHECK(cudaFree(d_hashes));
    CUDA_CHECK(cudaFree(d_heaps));
    CUDA_CHECK(cudaFree(d_solutions));
    CUDA_CHECK(cudaFree(d_num_sols));

    CUDA_CHECK(cudaFreeHost(h_solutions));
    CUDA_CHECK(cudaFreeHost(h_num_sols));
}
