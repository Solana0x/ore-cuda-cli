/* Copyright (c) 2020 tevador <tevador@gmail.com> */
/* See LICENSE for licensing information */

/* Original code from Argon2 reference source code package used under CC0
 * https://github.com/P-H-C/phc-winner-argon2
 * Copyright 2015
 * Daniel Dinu, Dmitry Khovratovich, Jean-Philippe Aumasson, and Samuel Neves
*/

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "blake2.h"
#include "hashx_endian.h"

__constant__ uint64_t blake2b_IV[8] = {
    UINT64_C(0x6a09e667f3bcc908), UINT64_C(0xbb67ae8584caa73b),
    UINT64_C(0x3c6ef372fe94f82b), UINT64_C(0xa54ff53a5f1d36f1),
    UINT64_C(0x510e527fade682d1), UINT64_C(0x9b05688c2b3e6c1f),
    UINT64_C(0x1f83d9abfb41bd6b), UINT64_C(0x5be0cd19137e2179) };

__device__ FORCE_INLINE uint64_t rotr64(const uint64_t w, const unsigned c) {
    return (w >> c) | (w << (64 - c));
}

__device__ FORCE_INLINE void blake2b_set_lastblock(blake2b_state* S) {
    S->f[0] = (uint64_t)-1;
}

__device__ FORCE_INLINE void blake2b_increment_counter(blake2b_state* S,
                                                      uint64_t inc) {
    S->t[0] += inc;
    S->t[1] += (S->t[0] < inc);
}

__device__ FORCE_INLINE void blake2b_init0(blake2b_state* S) {
    memset(S, 0, sizeof(*S));
    memcpy(S->h, blake2b_IV, sizeof(S->h));
}

__global__ void blake2b_compress_kernel(blake2b_state* S, const uint8_t* block, int rounds) {
    __shared__ uint64_t m[16];
    __shared__ uint64_t v[16];

    unsigned int i = threadIdx.x;
    if (i < 16) {
        m[i] = load64(block + i * sizeof(m[i]));
    }

    if (i < 8) {
        v[i] = S->h[i];
    }
    if (i < 16) {
        v[8 + i] = (i < 8) ? blake2b_IV[i] : (blake2b_IV[i - 8] ^ S->t[i - 8]);
    }
    if (i == 15) {
        v[14] ^= S->f[0];
        v[15] ^= S->f[1];
    }

    __syncthreads();

    for (int r = 0; r < rounds; r++) {
        // Apply the G function in a thread-parallel manner
        if (i < 16) {
            if (i % 4 == 0) v[i % 4] += v[(i + 1) % 4] + m[BLAKE2_SIGMA_ ## r ## _ ## (i % 16)];
            if (i % 4 == 3) v[(i + 3) % 4] = rotr64(v[(i + 3) % 4] ^ v[i % 4], 32);
            if (i % 4 == 2) v[(i + 2) % 4] += v[(i + 3) % 4];
            if (i % 4 == 1) v[(i + 1) % 4] = rotr64(v[(i + 1) % 4] ^ v[(i + 2) % 4], 24);
        }
        __syncthreads();
    }

    if (i < 8) {
        S->h[i] = S->h[i] ^ v[i] ^ v[i + 8];
    }
}

__global__ void blake2b_compress_4r_kernel(blake2b_state* S, const uint8_t* block) {
    blake2b_compress_kernel(S, block, 4);
}

extern "C" int hashx_blake2b_update(blake2b_state* S, const void* in, size_t inlen) {
    const uint8_t* pin = (const uint8_t*)in;

    if (inlen == 0) {
        return 0;
    }

    /* Sanity check */
    if (S == NULL || in == NULL) {
        return -1;
    }

    /* Is this a reused state? */
    if (S->f[0] != 0) {
        return -1;
    }

    if (S->buflen + inlen > BLAKE2B_BLOCKBYTES) {
        /* Complete current block */
        size_t left = S->buflen;
        size_t fill = BLAKE2B_BLOCKBYTES - left;
        memcpy(&S->buf[left], pin, fill);
        blake2b_increment_counter(S, BLAKE2B_BLOCKBYTES);
        blake2b_compress_kernel<<<1, 16>>>(S, S->buf, 12);
        S->buflen = 0;
        inlen -= fill;
        pin += fill;
        /* Avoid buffer copies when possible */
        while (inlen > BLAKE2B_BLOCKBYTES) {
            blake2b_increment_counter(S, BLAKE2B_BLOCKBYTES);
            blake2b_compress_kernel<<<1, 16>>>(S, pin, 12);
            inlen -= BLAKE2B_BLOCKBYTES;
            pin += BLAKE2B_BLOCKBYTES;
        }
    }
    memcpy(&S->buf[S->buflen], pin, inlen);
    S->buflen += (unsigned int)inlen;
    return 0;
}

extern "C" int hashx_blake2b_final(blake2b_state* S, void* out, size_t outlen) {
    uint8_t buffer[BLAKE2B_OUTBYTES] = { 0 };

    /* Sanity checks */
    if (S == NULL || out == NULL || outlen < S->outlen) {
        return -1;
    }

    /* Is this a reused state? */
    if (S->f[0] != 0) {
        return -1;
    }

    blake2b_increment_counter(S, S->buflen);
    blake2b_set_lastblock(S);
    memset(&S->buf[S->buflen], 0, BLAKE2B_BLOCKBYTES - S->buflen); /* Padding */
    blake2b_compress_kernel<<<1, 16>>>(S, S->buf, 12);

    cudaMemcpy(out, S->h, outlen, cudaMemcpyDeviceToHost);

    return 0;
}

/* 4-round version of Blake2b */
extern "C" void hashx_blake2b_4r(const blake2b_param* params, const void* in, 
	size_t inlen, void* out) {

	blake2b_state state;
	const uint8_t* p = (const uint8_t*)params;

	blake2b_init0(&state);
	/* IV XOR Parameter Block */
	for (unsigned i = 0; i < 8; ++i) {
		state.h[i] ^= load64(&p[i * sizeof(state.h[i])]);
	}

	const uint8_t* pin = (const uint8_t*)in;

	while (inlen > BLAKE2B_BLOCKBYTES) {
		blake2b_increment_counter(&state, BLAKE2B_BLOCKBYTES);
		blake2b_compress_4r_kernel<<<1, 16>>>(&state, pin);
		inlen -= BLAKE2B_BLOCKBYTES;
		pin += BLAKE2B_BLOCKBYTES;
	}

	memcpy(state.buf, pin, inlen);
	blake2b_increment_counter(&state, inlen);
	blake2b_set_lastblock(&state);
	blake2b_compress_4r_kernel<<<1, 16>>>(&state, state.buf);

	cudaMemcpy(out, state.h, sizeof(state.h), cudaMemcpyDeviceToHost);
}
