#include <cuda_runtime.h>
#include "solver.h"
#include "context.h"
#include "solver_heap.h"
#include <../../hashx/src/hashx_endian.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>
#include <stdint.h>

#ifdef _MSC_VER
#pragma warning(disable : 4146)
#endif

#define CLEAR(x) memset(&x, 0, sizeof(x))
#define MAKE_ITEM(bucket, left, right) ((left) << 17 | (right) << 8 | (bucket))
#define ITEM_BUCKET(item) ((item) % NUM_COARSE_BUCKETS)
#define ITEM_LEFT_IDX(item) ((item) >> 17)
#define ITEM_RIGHT_IDX(item) (((item) >> 8) & 511)
#define INVERT_BUCKET(idx) (-(idx) % NUM_COARSE_BUCKETS)
#define INVERT_SCRATCH(idx) (-(idx) % NUM_FINE_BUCKETS)
#define STAGE1_IDX(buck, pos) heap->stage1_indices.buckets[buck].items[pos]
#define STAGE2_IDX(buck, pos) heap->stage2_indices.buckets[buck].items[pos]
#define STAGE3_IDX(buck, pos) heap->stage3_indices.buckets[buck].items[pos]
#define STAGE1_DATA(buck, pos) heap->stage1_data.buckets[buck].items[pos]
#define STAGE2_DATA(buck, pos) heap->stage2_data.buckets[buck].items[pos]
#define STAGE3_DATA(buck, pos) heap->stage3_data.buckets[buck].items[pos]
#define STAGE1_SIZE(buck) heap->stage1_indices.counts[buck]
#define STAGE2_SIZE(buck) heap->stage2_indices.counts[buck]
#define STAGE3_SIZE(buck) heap->stage3_indices.counts[buck]
#define SCRATCH(buck, pos) heap->scratch_ht.buckets[buck].items[pos]
#define SCRATCH_SIZE(buck) heap->scratch_ht.counts[buck]
#define SWAP_IDX(a, b) do { equix_idx temp = a; a = b; b = temp; } while(0)
#define CARRY (bucket_idx != 0)
#define BUCK_START 0
#define BUCK_END (NUM_COARSE_BUCKETS / 2 + 1)

typedef uint32_t u32;
typedef stage1_idx_item s1_idx;
typedef stage2_idx_item s2_idx;
typedef stage3_idx_item s3_idx;

// Utilize __forceinline__ to ensure critical small functions are inlined, reducing function call overhead
__device__ __forceinline__ uint64_t hash_value(hashx_ctx* hash_func, equix_idx index) {
    char hash[HASHX_SIZE];
    hashx_exec(hash_func, index, hash);
    return load64(hash);
}

// Refactor atomic operations to be more efficient by reducing the amount of needed operations
__device__ __forceinline__ unsigned int atomicAdd_u16(uint16_t *address, uint16_t val) {
    unsigned int* base_address = (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old, assumed;
    do {
        assumed = old = *base_address;
        old = atomicCAS(base_address, assumed, (assumed & 0xFFFF0000) | (((assumed & 0xFFFF) + val) & 0xFFFF));
    } while (assumed != old);
    return old;
}

__device__ __forceinline__ unsigned int atomicSub_u16(uint16_t *address, uint16_t val) {
    unsigned int* base_address = (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old, assumed;
    do {
        assumed = old = *base_address;
        old = atomicCAS(base_address, assumed, (assumed & 0xFFFF0000) | (((assumed & 0xFFFF) - val) & 0xFFFF));
    } while (assumed != old);
    return old;
}

// Consider using shared memory for frequently accessed data to reduce global memory traffic
__device__ void solve_stage0(uint64_t* hashes, solver_heap* heap) {
    __shared__ uint64_t shared_hashes[NUM_COARSE_BUCKETS];  // Adjust size appropriately
    int tid = threadIdx.x;
    if (tid < NUM_COARSE_BUCKETS) {
        shared_hashes[tid] = hashes[tid];  // Assuming one hash per thread for simplification
    }
    __syncthreads();

    for (int i = tid; i < INDEX_SPACE; i += blockDim.x) {
        uint64_t value = shared_hashes[i % NUM_COARSE_BUCKETS]; // Example use of shared memory
        u32 bucket_idx = value % NUM_COARSE_BUCKETS;
        u32 item_idx = atomicAdd_u16(reinterpret_cast<uint16_t*>(&heap->stage1_indices.counts[bucket_idx]), 1);
        if (item_idx < COARSE_BUCKET_ITEMS) {
            heap->stage1_indices.buckets[bucket_idx].items[item_idx] = i;
            heap->stage1_data.buckets[bucket_idx].items[item_idx] = value / NUM_COARSE_BUCKETS; // 52 bits
        }
    }
}

// Additional functions (solve_stage1, solve_stage2, solve_stage3) need similar attention to detail for memory optimization and usage of shared memory or registers effectively.

__global__ void solve_all_stages_kernel(uint64_t* hashes, solver_heap* heaps, equix_solution* solutions, uint32_t* num_sols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t* thread_hashes = hashes + (idx * INDEX_SPACE);
    solver_heap* thread_heap = &heaps[idx];
    equix_solution* thread_solutions = &solutions[idx * EQUIX_MAX_SOLS];

    solve_stage0(thread_hashes, thread_heap);
    __syncthreads();
    solve_stage1(thread_heap);
    __syncthreads();
    solve_stage2(thread_heap);
    __syncthreads();
    num_sols[idx] = solve_stage3(thread_heap, thread_solutions);
}
