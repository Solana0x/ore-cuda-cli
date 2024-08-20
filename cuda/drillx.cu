#include <stdint.h>
#include <stdio.h>
#include "drillx.h"
#include "equix/include/equix.h"
#include "hashx/include/hashx.h"
#include "equix/src/context.h"
#include "equix/src/solver.h"
#include "equix/src/solver_heap.h"
#include "hashx/src/context.h"

const int BATCH_SIZE = 8192; 
const int NUM_HASHING_ROUNDS = 1; 
const int THREADS_PER_BLOCK = 1024;
const int INDEX_SPACE = 64; // Assuming a defined value for INDEX_SPACE

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err); \
        } \
    } while (0)

__constant__ int d_num_hashing_rounds;

extern "C" void set_num_hashing_rounds(int rounds) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_hashing_rounds, &rounds, sizeof(int)));
}

__global__ void do_hash_stage0i(hashx_ctx** ctxs, uint64_t** hash_space, int num_batches) {
    uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item < num_batches * INDEX_SPACE) {
        uint32_t batch_idx = item / INDEX_SPACE;
        uint32_t i = item % INDEX_SPACE;

        for (int round = 0; round < d_num_hashing_rounds; ++round) {
            hash_stage0i(ctxs[batch_idx], hash_space[batch_idx], i);
        }
    }
}

extern "C" void hash(uint8_t *challenge, uint8_t *nonce, uint64_t *out) {
    // Use managed memory for all allocations
    MemoryPool memPool(BATCH_SIZE);
    uint8_t seed[40];
    memcpy(seed, challenge, 32);

    // Managed memory eliminates the need for cudaMemcpy
    for (int i = 0; i < BATCH_SIZE; i++) {
        uint64_t nonce_offset = *((uint64_t*)nonce) + i;
        memcpy(seed + 32, &nonce_offset, 8);
        memPool.ctxs[i] = hashx_alloc(HASHX_INTERPRETED);
        if (!memPool.ctxs[i] || !hashx_make(memPool.ctxs[i], seed, 40)) {
            return;
        }
    }

    int blocksPerGrid = (BATCH_SIZE * INDEX_SPACE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Launch kernel asynchronously
    do_hash_stage0i<<<blocksPerGrid, THREADS_PER_BLOCK, 0, stream>>>(memPool.ctxs, memPool.hash_space, BATCH_SIZE);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // Directly copy the computed hashes from the managed memory to the output
    for (int i = 0; i < BATCH_SIZE; i++) {
        memcpy(out + i * INDEX_SPACE, memPool.hash_space[i], INDEX_SPACE * sizeof(uint64_t));
    }
}

extern "C" void solve_all_stages(uint64_t *hashes, uint8_t *out, uint32_t *sols, int num_sets) {
    uint64_t *d_hashes;
    solver_heap *d_heaps;
    equix_solution *d_solutions;
    uint32_t *d_num_sols;

    // Use managed memory for the solver's memory requirements
    CUDA_CHECK(cudaMallocManaged(&d_hashes, num_sets * INDEX_SPACE * sizeof(uint64_t)));
    CUDA_CHECK(cudaMallocManaged(&d_heaps, num_sets * sizeof(solver_heap)));
    CUDA_CHECK(cudaMallocManaged(&d_solutions, num_sets * EQUIX_MAX_SOLS * sizeof(equix_solution)));
    CUDA_CHECK(cudaMallocManaged(&d_num_sols, num_sets * sizeof(uint32_t)));

    // Directly assign hashes since we're using unified memory
    memcpy(d_hashes, hashes, num_sets * INDEX_SPACE * sizeof(uint64_t));

    int blocksPerGrid = (num_sets + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch the kernel to solve all stages
    solve_all_stages_kernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_hashes, d_heaps, d_solutions, d_num_sols);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results from managed memory to the output
    for (int i = 0; i < num_sets; i++) {
        sols[i] = d_num_sols[i];
        if (d_num_sols[i] > 0) {
            memcpy(out + i * sizeof(equix_solution), &d_solutions[i * EQUIX_MAX_SOLS], sizeof(equix_solution));
        }
    }

    // Free managed memory
    CUDA_CHECK(cudaFree(d_hashes));
    CUDA_CHECK(cudaFree(d_heaps));
    CUDA_CHECK(cudaFree(d_solutions));
    CUDA_CHECK(cudaFree(d_num_sols));
}
