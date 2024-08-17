#include <cuda_runtime.h>
#include "solver.h"
#include "context.h"
#include "solver_heap.h"
#include <hashx_endian.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>

#define CUDA_CHECK_ERROR(call) \
    { cudaError_t err = call; if(err != cudaSuccess) { printf("CUDA Error: %s:%d, ", __FILE__, __LINE__); printf("code: %d, reason: %s\n", err, cudaGetErrorString(err)); exit(1); } }

#define CLEAR(x) memset(&x, 0, sizeof(x))
#define MAKE_ITEM(bucket, left, right) ((left) << 17 | (right) << 8 | (bucket))
#define ITEM_BUCKET(item) (item) % NUM_COARSE_BUCKETS
#define ITEM_LEFT_IDX(item) (item) >> 17
#define ITEM_RIGHT_IDX(item) ((item) >> 8) & 511
#define INVERT_BUCKET(idx) -(idx) % NUM_COARSE_BUCKETS
#define INVERT_SCRATCH(idx) -(idx) % NUM_FINE_BUCKETS
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
#define SWAP_IDX(a, b)      \
    do {                    \
        equix_idx temp = a; \
        a = b;              \
        b = temp;           \
    } while(0)
#define CARRY (bucket_idx != 0)
#define BUCK_START 0
#define BUCK_END (NUM_COARSE_BUCKETS / 2 + 1)

typedef uint32_t u32;
typedef stage1_idx_item s1_idx;
typedef stage2_idx_item s2_idx;
typedef stage3_idx_item s3_idx;

__device__ uint64_t hash_value_device(hashx_ctx* hash_func, equix_idx index) {
    char hash[HASHX_SIZE];
    hashx_exec(hash_func, index, hash);
    return load64(hash);
}

__device__ void build_solution_stage1_device(equix_idx* output, solver_heap* heap, s2_idx root) {
    u32 bucket = ITEM_BUCKET(root);
    u32 bucket_inv = INVERT_BUCKET(bucket);
    u32 left_parent_idx = ITEM_LEFT_IDX(root);
    u32 right_parent_idx = ITEM_RIGHT_IDX(root);
    s1_idx left_parent = STAGE1_IDX(bucket, left_parent_idx);
    s1_idx right_parent = STAGE1_IDX(bucket_inv, right_parent_idx);
    output[0] = left_parent;
    output[1] = right_parent;
    if (!tree_cmp1(&output[0], &output[1])) {
        SWAP_IDX(output[0], output[1]);
    }
}

__device__ void build_solution_stage2_device(equix_idx* output, solver_heap* heap, s3_idx root) {
    u32 bucket = ITEM_BUCKET(root);
    u32 bucket_inv = INVERT_BUCKET(bucket);
    u32 left_parent_idx = ITEM_LEFT_IDX(root);
    u32 right_parent_idx = ITEM_RIGHT_IDX(root);
    s2_idx left_parent = STAGE2_IDX(bucket, left_parent_idx);
    s2_idx right_parent = STAGE2_IDX(bucket_inv, right_parent_idx);
    build_solution_stage1_device(&output[0], heap, left_parent);
    build_solution_stage1_device(&output[2], heap, right_parent);
    if (!tree_cmp2(&output[0], &output[2])) {
        SWAP_IDX(output[0], output[2]);
        SWAP_IDX(output[1], output[3]);
    }
}

__device__ void build_solution_device(equix_solution* solution, solver_heap* heap, s3_idx left, s3_idx right) {
    build_solution_stage2_device(&solution->idx[0], heap, left);
    build_solution_stage2_device(&solution->idx[4], heap, right);
    if (!tree_cmp4(&solution->idx[0], &solution->idx[4])) {
        SWAP_IDX(solution->idx[0], solution->idx[4]);
        SWAP_IDX(solution->idx[1], solution->idx[5]);
        SWAP_IDX(solution->idx[2], solution->idx[6]);
        SWAP_IDX(solution->idx[3], solution->idx[7]);
    }
}

__global__ void solve_stage0_kernel(hashx_ctx* hash_func, solver_heap* heap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= INDEX_SPACE) return;

    uint64_t value = hash_value_device(hash_func, i);
    u32 bucket_idx = value % NUM_COARSE_BUCKETS;
    u32 item_idx = atomicAdd(&STAGE1_SIZE(bucket_idx), 1);
    if (item_idx >= COARSE_BUCKET_ITEMS)
        return;
    STAGE1_IDX(bucket_idx, item_idx) = i;
    STAGE1_DATA(bucket_idx, item_idx) = value / NUM_COARSE_BUCKETS; /* 52 bits */
}

__global__ void solve_stage1_kernel(solver_heap* heap) {
    int bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx >= BUCK_END) return;

    u32 cpl_bucket = INVERT_BUCKET(bucket_idx);
    u32 cpl_buck_size = STAGE1_SIZE(cpl_bucket);

    for (u32 item_idx = 0; item_idx < cpl_buck_size; ++item_idx) {
        stage1_data_item value = STAGE1_DATA(cpl_bucket, item_idx);
        u32 fine_buck_idx = value % NUM_FINE_BUCKETS;
        u32 fine_item_idx = atomicAdd(&SCRATCH_SIZE(fine_buck_idx), 1);
        if (fine_item_idx >= FINE_BUCKET_ITEMS)
            continue;
        SCRATCH(fine_buck_idx, fine_item_idx) = item_idx;
        if (cpl_bucket == bucket_idx) {
            MAKE_PAIRS1
        }
    }

    if (cpl_bucket != bucket_idx) {
        u32 buck_size = STAGE1_SIZE(bucket_idx);
        for (u32 item_idx = 0; item_idx < buck_size; ++item_idx) {
            MAKE_PAIRS1
        }
    }
}

__global__ void solve_stage2_kernel(solver_heap* heap) {
    int bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx >= BUCK_END) return;

    u32 cpl_bucket = INVERT_BUCKET(bucket_idx);
    u32 cpl_buck_size = STAGE2_SIZE(cpl_bucket);

    for (u32 item_idx = 0; item_idx < cpl_buck_size; ++item_idx) {
        stage2_data_item value = STAGE2_DATA(cpl_bucket, item_idx);
        u32 fine_buck_idx = value % NUM_FINE_BUCKETS;
        u32 fine_item_idx = atomicAdd(&SCRATCH_SIZE(fine_buck_idx), 1);
        if (fine_item_idx >= FINE_BUCKET_ITEMS)
            continue;
        SCRATCH(fine_buck_idx, fine_item_idx) = item_idx;
        if (cpl_bucket == bucket_idx) {
            MAKE_PAIRS2
        }
    }

    if (cpl_bucket != bucket_idx) {
        u32 buck_size = STAGE2_SIZE(bucket_idx);
        for (u32 item_idx = 0; item_idx < buck_size; ++item_idx) {
            MAKE_PAIRS2
        }
    }
}

__global__ void solve_stage3_kernel(solver_heap* heap, equix_solution output[EQUIX_MAX_SOLS], int* sols_found) {
    int bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx >= BUCK_END) return;

    u32 cpl_bucket = INVERT_BUCKET(bucket_idx);
    u32 cpl_buck_size = STAGE3_SIZE(cpl_bucket);

    for (u32 item_idx = 0; item_idx < cpl_buck_size; ++item_idx) {
        stage3_data_item value = STAGE3_DATA(cpl_bucket, item_idx);
        u32 fine_buck_idx = value % NUM_FINE_BUCKETS;
        u32 fine_item_idx = atomicAdd(&SCRATCH_SIZE(fine_buck_idx), 1);
        if (fine_item_idx >= FINE_BUCKET_ITEMS)
            continue;
        SCRATCH(fine_buck_idx, fine_item_idx) = item_idx;
        if (cpl_bucket == bucket_idx) {
            MAKE_PAIRS3
        }
    }

    if (cpl_bucket != bucket_idx) {
        u32 buck_size = STAGE3_SIZE(bucket_idx);
        for (u32 item_idx = 0; item_idx < buck_size; ++item_idx) {
            MAKE_PAIRS3
        }
    }
}

int equix_solver_solve_cuda(hashx_ctx* hash_func, solver_heap* heap, equix_solution output[EQUIX_MAX_SOLS]) {
    // Allocate device memory and copy data
    hashx_ctx* d_hash_func;
    solver_heap* d_heap;
    equix_solution* d_output;
    int* d_sols_found;

    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_hash_func, sizeof(hashx_ctx)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_heap, sizeof(solver_heap)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_output, sizeof(equix_solution) * EQUIX_MAX_SOLS));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_sols_found, sizeof(int)));

    CUDA_CHECK_ERROR(cudaMemcpy(d_hash_func, hash_func, sizeof(hashx_ctx), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_heap, heap, sizeof(solver_heap), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemset(d_sols_found, 0, sizeof(int)));

    // Launch kernels
    solve_stage0_kernel<<<(INDEX_SPACE + 255) / 256, 256>>>(d_hash_func, d_heap);
    CUDA_CHECK_ERROR(cudaGetLastError());

    solve_stage1_kernel<<<(BUCK_END + 255) / 256, 256>>>(d_heap);
    CUDA_CHECK_ERROR(cudaGetLastError());

    solve_stage2_kernel<<<(BUCK_END + 255) / 256, 256>>>(d_heap);
    CUDA_CHECK_ERROR(cudaGetLastError());

    solve_stage3_kernel<<<(BUCK_END + 255) / 256, 256>>>(d_heap, d_output, d_sols_found);
    CUDA_CHECK_ERROR(cudaGetLastError());

    // Copy results back to host
    CUDA_CHECK_ERROR(cudaMemcpy(output, d_output, sizeof(equix_solution) * EQUIX_MAX_SOLS, cudaMemcpyDeviceToHost));
    int sols_found;
    CUDA_CHECK_ERROR(cudaMemcpy(&sols_found, d_sols_found, sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK_ERROR(cudaFree(d_hash_func));
    CUDA_CHECK_ERROR(cudaFree(d_heap));
    CUDA_CHECK_ERROR(cudaFree(d_output));
    CUDA_CHECK_ERROR(cudaFree(d_sols_found));

    return sols_found;
}
