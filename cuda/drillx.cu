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

const int BATCH_SIZE = 8192;
__constant__ int d_num_hashing_rounds;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err); \
        } \
    } while (0)

extern "C" void set_num_hashing_rounds(int rounds) {
    int adjustedRounds = (rounds > 0) ? rounds : 1;
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_hashing_rounds, &adjustedRounds, sizeof(int)));
}

extern "C" void hash(uint8_t *challenge, uint8_t *nonce, uint64_t *out) {
    MemoryPool memPool(BATCH_SIZE);

    std::vector<uint8_t> seed(40);
    memcpy(seed.data(), challenge, 32);

    uint64_t *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, BATCH_SIZE * INDEX_SPACE * sizeof(uint64_t)));

    hashx_ctx **d_ctxs;
    CUDA_CHECK(cudaMalloc(&d_ctxs, BATCH_SIZE * sizeof(hashx_ctx*)));
    hashx_ctx *h_ctxs[BATCH_SIZE];
    for (int i = 0; i < BATCH_SIZE; i++) {
        uint64_t nonce_offset = *((uint64_t*)nonce) + i;
        memcpy(seed.data() + 32, &nonce_offset, 8);
        h_ctxs[i] = hashx_alloc(HASHX_INTERPRETED);
        if (!h_ctxs[i] || !hashx_make(h_ctxs[i], seed.data(), 40)) {
            return;
        }
    }

    CUDA_CHECK(cudaMemcpy(d_ctxs, h_ctxs, BATCH_SIZE * sizeof(hashx_ctx*), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (BATCH_SIZE * INDEX_SPACE + threadsPerBlock - 1) / threadsPerBlock;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Correct the kernel launch to match the kernel definition
    do_hash_stage0i<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_ctxs, memPool.hash_space, d_out, d_num_hashing_rounds);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(out, d_out, BATCH_SIZE * INDEX_SPACE * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_ctxs));
}

__global__ void do_hash_stage0i(hashx_ctx** ctxs, uint64_t** hash_space, uint64_t* out, int num_hashing_rounds) {
    __shared__ uint64_t shared_hash_space[256];

    uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item < BATCH_SIZE * INDEX_SPACE) {
        uint32_t batch_idx = item / INDEX_SPACE;
        uint32_t i = item % INDEX_SPACE;

        for (int round = 0; round < num_hashing_rounds; ++round) {
            hash_stage0i(ctxs[batch_idx], hash_space[batch_idx], i);
        }

        shared_hash_space[threadIdx.x] = hash_space[batch_idx][i];
        __syncthreads();

        if (threadIdx.x < 256) {
            out[item] = shared_hash_space[threadIdx.x];
        }
    }
}
