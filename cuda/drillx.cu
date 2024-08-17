#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "drillx.h"
#include "equix/include/equix.h"
#include "hashx/include/hashx.h"
#include "equix/src/context.h"
#include "equix/src/solver.h"
#include "equix/src/solver_heap.h"
#include "hashx/src/context.h"

const int BATCH_SIZE = 16384; // Large batch size
const int NUM_HASHING_ROUNDS = 1;

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
    // Increase the CUDA heap size to handle larger allocations
    size_t heapSize = 24L * 1024L * 1024L * 1024L; // 24 GB
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize));

    // Allocate the memory pool on the host
    MemoryPool* memPool = new MemoryPool(BATCH_SIZE);

    uint8_t seed[40];
    memcpy(seed, challenge, 32);

    for (int i = 0; i < BATCH_SIZE; i++) {
        uint64_t nonce_offset = *((uint64_t*)nonce) + i;
        memcpy(seed + 32, &nonce_offset, 8);
        memPool->ctxs[i] = hashx_alloc(HASHX_INTERPRETED);
        if (!memPool->ctxs[i] || !hashx_make(memPool->ctxs[i], seed, 40)) {
            delete memPool; // Clean up before returning
            return;
        }
    }

    int threadsPerBlock = 1024;
    int blocksPerGrid = (BATCH_SIZE * INDEX_SPACE + threadsPerBlock - 1) / threadsPerBlock;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Perform the hashing on the GPU
    do_hash_stage0i<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(memPool->ctxs, memPool->hash_space, NUM_HASHING_ROUNDS);
    CUDA_CHECK(cudaGetLastError());

    // Synchronize the stream to ensure the kernel execution completes
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Ensure hash results are transferred from GPU to CPU
    for (int i = 0; i < BATCH_SIZE; i++) {
        // Copy data from the GPU to the CPU
        CUDA_CHECK(cudaMemcpy(out + i * INDEX_SPACE, memPool->hash_space[i], INDEX_SPACE * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    }

    // Synchronize after copying the data
    CUDA_CHECK(cudaDeviceSynchronize());

    // Destroy the stream
    CUDA_CHECK(cudaStreamDestroy(stream));

    // Clean up memory pool
    delete memPool;
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
    // Increase the CUDA heap size to handle larger allocations
    size_t heapSize = 24L * 1024L * 1024L * 1024L; // 24 GB
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize));

    // Use unified memory for automatic paging between CPU and GPU
    uint64_t *d_hashes;
    solver_heap *d_heaps;
    equix_solution *d_solutions;
    uint32_t *d_num_sols;

    CUDA_CHECK(cudaMallocManaged(&d_hashes, num_sets * INDEX_SPACE * sizeof(uint64_t)));
    CUDA_CHECK(cudaMallocManaged(&d_heaps, num_sets * sizeof(solver_heap)));
    CUDA_CHECK(cudaMallocManaged(&d_solutions, num_sets * EQUIX_MAX_SOLS * sizeof(equix_solution)));
    CUDA_CHECK(cudaMallocManaged(&d_num_sols, num_sets * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_hashes, hashes, num_sets * INDEX_SPACE * sizeof(uint64_t), cudaMemcpyHostToDevice));

    int threadsPerBlock = 1024;
    int blocksPerGrid = (num_sets + threadsPerBlock - 1) / threadsPerBlock;

    solve_all_stages_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_hashes, d_heaps, d_solutions, d_num_sols);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out, d_solutions, num_sets * EQUIX_MAX_SOLS * sizeof(equix_solution), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sols, d_num_sols, num_sets * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_hashes));
    CUDA_CHECK(cudaFree(d_heaps));
    CUDA_CHECK(cudaFree(d_solutions));
    CUDA_CHECK(cudaFree(d_num_sols));
}
