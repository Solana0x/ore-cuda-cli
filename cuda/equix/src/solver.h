/* Copyright (c) 2020 tevador <tevador@gmail.com> */
/* See LICENSE for licensing information */

#ifndef SOLVER_H
#define SOLVER_H

#include <../include/equix.h>
#include <../../hashx/src/hashx_endian.h>
#include <stdbool.h>
#include "context.h"

#define EQUIX_STAGE1_MASK ((1ull << 15) - 1)
#define EQUIX_STAGE2_MASK ((1ull << 30) - 1)
#define EQUIX_FULL_MASK ((1ull << 60) - 1)

__device__ inline bool tree_cmp1(const equix_idx* left, const equix_idx* right) {
    // Compare 16-bit values using CUDA vector types
    return *left <= *right;
}

__device__ inline bool tree_cmp2(const equix_idx* left, const equix_idx* right) {
    // Use CUDA intrinsics to load and compare 32-bit values
    uint32_t left_val = load32(left);
    uint32_t right_val = load32(right);
    return left_val <= right_val;
}

__device__ inline bool tree_cmp4(const equix_idx* left, const equix_idx* right) {
    // Use CUDA intrinsics to load and compare 64-bit values
    uint64_t left_val = load64(left);
    uint64_t right_val = load64(right);
    return left_val <= right_val;
}

__device__ void hash_stage0i(hashx_ctx* hash_func, uint64_t* out, uint32_t i);

uint32_t equix_solver_solve(uint64_t* hashes, solver_heap* heap, equix_solution output[EQUIX_MAX_SOLS]);

__global__ void solve_all_stages_kernel(uint64_t* hashes, solver_heap* heaps, equix_solution* solutions, uint32_t* num_sols);

#endif
