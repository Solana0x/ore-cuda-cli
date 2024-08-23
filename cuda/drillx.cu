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
__constant__ int NUM_HASHING_ROUNDS = 1;  // Use __constant__ memory for NUM_HASHING_ROUNDS

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err); \
        } \
    } while (0)

extern "C" void set_num_hashing_rounds(int rounds) {
    // Updated to set the constant variable in device code
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

    int threadsPerBlock = 256;  // Reduced number of threads per block to reduce memory footprint
    int blocksPerGrid = (BATCH_SIZE * INDEX_SPACE + threadsPerBlock - 1) / threadsPerBlock;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    do_hash_stage0i<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(memPool.ctxs, memPool.hash_space, NUM_HASHING_ROUNDS);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int i = 0; i < BATCH_SIZE; i++) {
        CUDA_CHECK(cudaMemcpyAsync(out + i * INDEX_SPACE, memPool.hash_space[i], INDEX_SPACE * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
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

    // Use cudaMallocManaged for memory that can be shared between host and device
    CUDA_CHECK(cudaMallocManaged(&d_hashes, num_sets * INDEX_SPACE * sizeof(uint64_t)));
    CUDA_CHECK(cudaMallocManaged(&d_heaps, num_sets * sizeof(solver_heap)));
    CUDA_CHECK(cudaMallocManaged(&d_solutions, num_sets * EQUIX_MAX_SOLS * sizeof(equix_solution)));
    CUDA_CHECK(cudaMallocManaged(&d_num_sols, num_sets * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_hashes, hashes, num_sets * INDEX_SPACE * sizeof(uint64_t), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;  // Adjusted to a more optimal value
    int blocksPerGrid = (num_sets + threadsPerBlock - 1) / threadsPerBlock;

    solve_all_stages_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_hashes, d_heaps, d_solutions, d_num_sols);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    // Use unified memory to avoid multiple memcpy calls
    for (int i = 0; i < num_sets; i++) {
        sols[i] = d_num_sols[i];
        if (d_num_sols[i] > 0) {
            memcpy(out + i * sizeof(equix_solution), &d_solutions[i * EQUIX_MAX_SOLS], sizeof(equix_solution));
        }
    }

    CUDA_CHECK(cudaFree(d_hashes));
    CUDA_CHECK(cudaFree(d_heaps));
    CUDA_CHECK(cudaFree(d_solutions));
    CUDA_CHECK(cudaFree(d_num_sols));
}
