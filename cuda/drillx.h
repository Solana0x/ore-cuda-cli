#ifndef DRILLX_H
#define DRILLX_H

#include "equix/include/equix.h"
#include "hashx/include/hashx.h"
#include "equix/src/context.h"
#include "equix/src/solver_heap.h"
#include "hashx/src/context.h"

extern const int BATCH_SIZE;
extern const int NUM_HASHING_ROUNDS;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err); \
        } \
    } while (0)

struct MemoryPool {
    uint64_t** hash_space;
    hashx_ctx** ctxs;

    MemoryPool(int batchSize) {
        CUDA_CHECK(cudaHostAlloc(&ctxs, batchSize * sizeof(hashx_ctx*), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostAlloc(&hash_space, batchSize * sizeof(uint64_t*), cudaHostAllocMapped));
        for (int i = 0; i < batchSize; i++) {
            CUDA_CHECK(cudaMalloc(&hash_space[i], INDEX_SPACE * sizeof(uint64_t)));
        }
    }

    ~MemoryPool() {
        for (int i = 0; i < BATCH_SIZE; i++) {
            CUDA_CHECK(cudaFree(hash_space[i]));
        }
        CUDA_CHECK(cudaFreeHost(hash_space));
        CUDA_CHECK(cudaFreeHost(ctxs));
    }
};

extern "C" void drillx_hash(uint8_t *challenge, uint8_t *nonce, uint64_t *out);
extern "C" void solve_all_stages(uint64_t *hashes, uint8_t *out, uint32_t *sols, int num_sets);
__global__ void do_hash_stage0i(hashx_ctx** ctxs, uint64_t** hash_space, int num_hashing_rounds);

#endif
