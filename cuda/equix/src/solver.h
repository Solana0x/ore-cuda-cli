/* Copyright (c) 2020 tevador <tevador@gmail.com> */
/* See LICENSE for licensing information */

#ifndef SOLVER_H
#define SOLVER_H

#include <../include/equix.h>
#include <../../hashx/src/hashx_endian.h>
#include <stdbool.h>
#include <immintrin.h>  // Header for SIMD intrinsics
#include "context.h"

#define EQUIX_STAGE1_MASK ((1ull << 15) - 1)
#define EQUIX_STAGE2_MASK ((1ull << 30) - 1)
#define EQUIX_FULL_MASK ((1ull << 60) - 1)

__device__ inline bool tree_cmp1(const equix_idx* left, const equix_idx* right) {
    // Load 16-bit values from both pointers
    __m128i left_vec = _mm_loadu_si128((__m128i*)left);
    __m128i right_vec = _mm_loadu_si128((__m128i*)right);
    // Compare using SIMD
    __m128i cmp_result = _mm_cmpeq_epi16(left_vec, right_vec);
    // Return if left <= right for all elements
    return _mm_movemask_epi8(cmp_result) != 0;
}

__device__ inline bool tree_cmp2(const equix_idx* left, const equix_idx* right) {
    // Load 32-bit values from both pointers
    __m128i left_vec = _mm_loadu_si128((__m128i*)left);
    __m128i right_vec = _mm_loadu_si128((__m128i*)right);
    // Compare using SIMD
    __m128i cmp_result = _mm_cmpeq_epi32(left_vec, right_vec);
    // Return if left <= right for all elements
    return _mm_movemask_epi8(cmp_result) != 0;
}

__device__ inline bool tree_cmp4(const equix_idx* left, const equix_idx* right) {
    // Load 64-bit values from both pointers
    __m128i left_vec = _mm_loadu_si128((__m128i*)left);
    __m128i right_vec = _mm_loadu_si128((__m128i*)right);
    // Compare using SIMD
    __m128i cmp_result = _mm_cmpeq_epi64(left_vec, right_vec);
    // Return if left <= right for all elements
    return _mm_movemask_epi8(cmp_result) != 0;
}

__device__ void hash_stage0i(hashx_ctx* hash_func, uint64_t* out, uint32_t i);

uint32_t equix_solver_solve(uint64_t* hashes, solver_heap* heap, equix_solution output[EQUIX_MAX_SOLS]);

__global__ void solve_all_stages_kernel(uint64_t* hashes, solver_heap* heaps, equix_solution* solutions, uint32_t* num_sols);

#endif
