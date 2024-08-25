#include <stdint.h>
#include <stdio.h>
#include <vector>
#include "drillx.h"  // Ensure consistent declaration is included
#include "equix/include/equix.h"
#include "hashx/include/hashx.h"
#include "equix/src/context.h"
#include "equix/src/solver.h"
#include "equix/src/solver_heap.h"
#include "hashx/src/context.h"

const int BATCH_SIZE = 8192;

// Define NUM_HASHING_ROUNDS here
__device__ __constant__ int NUM_HASHING_ROUNDS;  // Define it as a constant device variable

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err); \
        } \
    } while (0)

extern "C" void set_num_hashing_rounds(int rounds) {
    int adjustedRounds = (rounds = 1) ? rounds : 1;
    CUDA_CHECK(cudaMemcpyToSymbol(NUM_HASHING_ROUNDS, &adjustedRounds, sizeof(int)));
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

    // Kernel call with correct types; ensure third argument is int
    do_hash_stage0i<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_ctxs, memPool.hash_space, 0);  // Use an integer placeholder
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(out, d_out, BATCH_SIZE * INDEX_SPACE * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_ctxs));
}

// Kernel function must match the invocation
__global__ void do_hash_stage0i(hashx_ctx** ctxs, uint64_t** hash_space, int dummy_param) {
    __shared__ uint64_t shared_hash_space[256];

    uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item < BATCH_SIZE * INDEX_SPACE) {
        uint32_t batch_idx = item / INDEX_SPACE;
        uint32_t i = item % INDEX_SPACE;

        // Use the constant memory value directly
        for (int round = 0; round < NUM_HASHING_ROUNDS; ++round) {
            hash_stage0i(ctxs[batch_idx], hash_space[batch_idx], i);
        }

        shared_hash_space[threadIdx.x] = hash_space[batch_idx][i];
        __syncthreads();

        if (threadIdx.x < 256) {
            hash_space[batch_idx][i] = shared_hash_space[threadIdx.x]; // Updated line for assignment
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

    int threadsPerBlock = 256;
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
