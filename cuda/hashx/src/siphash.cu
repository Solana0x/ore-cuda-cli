/* Copyright (c) 2020 tevador <tevador@gmail.com> */
/* See LICENSE for licensing information */

#include "siphash.h"
#include "hashx_endian.h"
#include "unreachable.h"

// Ensure SIPROUND is inlined for performance
#define SIPROUND(v0, v1, v2, v3) \
    do { \
        v0 += v1; v1 = ROTL(v1, 13); v1 ^= v0; v0 = ROTL(v0, 32); \
        v2 += v3; v3 = ROTL(v3, 16); v3 ^= v2; \
        v0 += v3; v3 = ROTL(v3, 21); v3 ^= v0; \
        v2 += v1; v1 = ROTL(v1, 17); v1 ^= v2; v2 = ROTL(v2, 32); \
    } while (0)

__device__ __forceinline__ uint64_t hashx_siphash13_ctr(uint64_t input, const siphash_state* __restrict__ keys) {
    uint64_t v0 = keys->v0;
    uint64_t v1 = keys->v1;
    uint64_t v2 = keys->v2;
    uint64_t v3 = keys->v3;

    v3 ^= input;

    // Perform SIPROUND in a warp-synchronized manner using shuffles
    #pragma unroll
    for (int i = 0; i < 1; i++) {
        SIPROUND(v0, v1, v2, v3);
    }

    v0 ^= input;
    v2 ^= 0xff;

    #pragma unroll
    for (int i = 0; i < 3; i++) {
        SIPROUND(v0, v1, v2, v3);
    }

    return (v0 ^ v1) ^ (v2 ^ v3);
}

__device__ void hashx_siphash24_ctr_state512(const siphash_state* __restrict__ keys, uint64_t input,
                                             uint64_t* __restrict__ state_out) {
    uint64_t v0 = keys->v0;
    uint64_t v1 = keys->v1;
    uint64_t v2 = keys->v2;
    uint64_t v3 = keys->v3;

    const uint64_t c1 = 0xee;
    const uint64_t c2 = 0xdd;

    v1 ^= c1;
    v3 ^= input;

    // Perform SIPROUND in a warp-synchronized manner using shuffles
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        SIPROUND(v0, v1, v2, v3);
    }

    v0 ^= input;
    v2 ^= c1;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        SIPROUND(v0, v1, v2, v3);
    }

    // Store the intermediate state
    state_out[0] = v0;
    state_out[1] = v1;
    state_out[2] = v2;
    state_out[3] = v3;

    v1 ^= c2;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        SIPROUND(v0, v1, v2, v3);
    }

    // Store the final state
    state_out[4] = v0;
    state_out[5] = v1;
    state_out[6] = v2;
    state_out[7] = v3;
}
