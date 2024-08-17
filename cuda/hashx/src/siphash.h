/* Copyright (c) 2020 tevador <tevador@gmail.com> */
/* See LICENSE for licensing information */

#ifndef SIPHASH_H
#define SIPHASH_H

#include <stdint.h>
#include <cuda_runtime.h>

#define ROTL(x, b) (((x) << (b)) | ((x) >> (64 - (b))))
#define SIPROUND(v0, v1, v2, v3) \
  do { \
    v0 += v1; v2 += v3; v1 = ROTL(v1, 13);   \
    v3 = ROTL(v3, 16); v1 ^= v0; v3 ^= v2;   \
    v0 = ROTL(v0, 32); v2 += v1; v0 += v3;   \
    v1 = ROTL(v1, 17);  v3 = ROTL(v3, 21);   \
    v1 ^= v2; v3 ^= v0; v2 = ROTL(v2, 32);   \
  } while (0)

typedef struct siphash_state {
    uint64_t v0, v1, v2, v3;
} siphash_state;

#ifdef __cplusplus
extern "C" {
#endif

// Optimized hashx_siphash13_ctr for GPU execution
__device__ uint64_t hashx_siphash13_ctr(uint64_t input, const siphash_state* keys) {
    uint64_t v0 = keys->v0 ^ 0x736f6d6570736575ULL;
    uint64_t v1 = keys->v1 ^ 0x646f72616e646f6dULL;
    uint64_t v2 = keys->v2 ^ 0x6c7967656e657261ULL;
    uint64_t v3 = keys->v3 ^ 0x7465646279746573ULL ^ input;

    SIPROUND(v0, v1, v2, v3);
    SIPROUND(v0, v1, v2, v3);
    SIPROUND(v0, v1, v2, v3);

    v0 ^= input;
    v2 ^= 0xff;
    
    SIPROUND(v0, v1, v2, v3);
    SIPROUND(v0, v1, v2, v3);
    SIPROUND(v0, v1, v2, v3);
    SIPROUND(v0, v1, v2, v3);

    return v0 ^ v1 ^ v2 ^ v3;
}

// Optimized hashx_siphash24_ctr_state512 for GPU execution
__global__ void hashx_siphash24_ctr_state512(const siphash_state* keys, uint64_t* input, uint64_t* state_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint64_t v0 = keys->v0 ^ 0x736f6d6570736575ULL;
        uint64_t v1 = keys->v1 ^ 0x646f72616e646f6dULL;
        uint64_t v2 = keys->v2 ^ 0x6c7967656e657261ULL;
        uint64_t v3 = keys->v3 ^ 0x7465646279746573ULL ^ input[idx];

        SIPROUND(v0, v1, v2, v3);
        SIPROUND(v0, v1, v2, v3);

        v0 ^= input[idx];
        v2 ^= 0xff;

        SIPROUND(v0, v1, v2, v3);
        SIPROUND(v0, v1, v2, v3);
        SIPROUND(v0, v1, v2, v3);
        SIPROUND(v0, v1, v2, v3);

        state_out[8 * idx] = v0 ^ v1 ^ v2 ^ v3;
    }
}

#ifdef __cplusplus
}
#endif

#endif
