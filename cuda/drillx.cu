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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err); \
        } \
    } while (0)

// Memory pool to persistently allocate and reuse GPU memory across multiple hash invocations
struct PersistentMemoryPool {
    uint64_t** d_hash_space;
    hashx_ctx** d_ctxs;
    cudaStream_t stream;

    PersistentMemoryPool(size_t batch_size) {
        CUDA_CHECK(cudaMalloc(&d_hash_space, batch_size * sizeof(uint64_t*)));
        CUDA_CHECK(cudaMalloc(&d_ctxs, batch_size * sizeof(hashx_ctx*)));
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    ~PersistentMemoryPool() {
        if (d_hash_space) {
            CUDA_CHECK(cudaFree(d_hash_space));
        }
        if (d_ctxs) {
            CUDA_CHECK(cudaFree(d_ctxs));
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
};

// Global persistent memory pool
PersistentMemoryPool *memPool = nullptr;

extern "C" void initialize_memory_pool() {
    if (memPool == nullptr) {
        memPool = new PersistentMemoryPool(BATCH_SIZE);
        if (!memPool) {
            fprintf(stderr, "Failed to initialize memory pool\n");
            exit(EXIT_FAILURE);
        }
    }
}

extern "C" void destroy_memory_pool() {
    if (memPool != nullptr) {
        delete memPool;
        memPool = nullptr;
    }
}

extern "C" void set_num_hashing_rounds(int rounds) {
    CUDA_CHECK(cudaMemcpyToSymbol(NUM_HASHING_ROUNDS, &rounds, sizeof(int)));
}

extern "C" void hash(uint8_t *challenge, uint8_t *nonce, uint64_t *out) {
    initialize_memory_pool();  // Initialize memory pool once if not done already

    uint8_t seed[40];
    memcpy(seed, challenge, 32);

    for (int i = 0; i < BATCH_SIZE; i++) {
        uint64_t nonce_offset = *((uint64_t*)nonce) + i;
        memcpy(seed + 32, &nonce_offset, 8);
        memPool->d_ctxs[i] = hashx_alloc(HASHX_INTERPRETED);
        if (!memPool->d_ctxs[i] || !hashx_make(memPool->d_ctxs[i], seed, 40)) {
            fprintf(stderr, "Failed to allocate or make hashx context\n");
            return;
        }
    }

    int threadsPerBlock = 1024;  // Maximum threads per block for better parallelism
    int blocksPerGrid = (BATCH_SIZE * INDEX_SPACE + threadsPerBlock - 1) / threadsPerBlock;

    // Use shared memory for faster data access during hashing operations
    do_hash_stage0i<<<blocksPerGrid, threadsPerBlock, 0, memPool->stream>>>(memPool->d_ctxs, memPool->d_hash_space, NUM_HASHING_ROUNDS);
    CUDA_CHECK(cudaGetLastError());

    // Overlap memory transfers with computation
    for (int i = 0; i < BATCH_SIZE; i++) {
        CUDA_CHECK(cudaMemcpyAsync(out + i * INDEX_SPACE, memPool->d_hash_space[i], INDEX_SPACE * sizeof(uint64_t), cudaMemcpyDeviceToHost, memPool->stream));
    }

    // Synchronize the stream to ensure all transfers are complete
    CUDA_CHECK(cudaStreamSynchronize(memPool->stream));
}

__global__ void do_hash_stage0i(hashx_ctx** ctxs, uint64_t** hash_space, int num_hashing_rounds) {
    __shared__ uint64_t shared_data[1024];  // Use shared memory for faster access
    uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item < BATCH_SIZE * INDEX_SPACE) {
        uint32_t batch_idx = item / INDEX_SPACE;
        uint32_t i = item % INDEX_SPACE;

        for (int round = 0; round < num_hashing_rounds; ++round) {
            // Copy data into shared memory for faster access
            shared_data[threadIdx.x] = hash_space[batch_idx][i];
            __syncthreads();
            hash_stage0i(ctxs[batch_idx], shared_data, i);
            __syncthreads();
            hash_space[batch_idx][i] = shared_data[threadIdx.x];
        }
    }
}

extern "C" void solve_all_stages(uint64_t *hashes, uint8_t *out, uint32_t *sols, int num_sets) {
    static uint64_t *d_hashes = nullptr;
    static solver_heap *d_heaps = nullptr;
    static equix_solution *d_solutions = nullptr;
    static uint32_t *d_num_sols = nullptr;

    // Allocate memory on the GPU only once
    if (!d_hashes) {
        CUDA_CHECK(cudaMalloc(&d_hashes, num_sets * INDEX_SPACE * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_heaps, num_sets * sizeof(solver_heap)));
        CUDA_CHECK(cudaMalloc(&d_solutions, num_sets * EQUIX_MAX_SOLS * sizeof(equix_solution)));
        CUDA_CHECK(cudaMalloc(&d_num_sols, num_sets * sizeof(uint32_t)));
    }

    equix_solution *h_solutions;
    uint32_t *h_num_sols;
    CUDA_CHECK(cudaHostAlloc(&h_solutions, num_sets * EQUIX_MAX_SOLS * sizeof(equix_solution), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_num_sols, num_sets * sizeof(uint32_t), cudaHostAllocDefault));

    // Use asynchronous transfers for hashes
    CUDA_CHECK(cudaMemcpyAsync(d_hashes, hashes, num_sets * INDEX_SPACE * sizeof(uint64_t), cudaMemcpyHostToDevice, memPool->stream));

    int threadsPerBlock = 1024;
    int blocksPerGrid = (num_sets + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel for solving
    solve_all_stages_kernel<<<blocksPerGrid, threadsPerBlock, 0, memPool->stream>>>(d_hashes, d_heaps, d_solutions, d_num_sols);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(memPool->stream));

    // Copy results asynchronously
    CUDA_CHECK(cudaMemcpyAsync(h_solutions, d_solutions, num_sets * EQUIX_MAX_SOLS * sizeof(equix_solution), cudaMemcpyDeviceToHost, memPool->stream));
    CUDA_CHECK(cudaMemcpyAsync(h_num_sols, d_num_sols, num_sets * sizeof(uint32_t), cudaMemcpyDeviceToHost, memPool->stream));

    CUDA_CHECK(cudaStreamSynchronize(memPool->stream));

    for (int i = 0; i < num_sets; i++) {
        sols[i] = h_num_sols[i];
        if (h_num_sols[i] > 0) {
            memcpy(out + i * sizeof(equix_solution), &h_solutions[i * EQUIX_MAX_SOLS], sizeof(equix_solution));
        }
    }

    CUDA_CHECK(cudaFreeHost(h_solutions));
    CUDA_CHECK(cudaFreeHost(h_num_sols));
}
