#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "blake2.h"
#include "hashx_endian.h"

#define THREADS_PER_BLOCK 256

static const uint64_t blake2b_IV[8] = {
    UINT64_C(0x6a09e667f3bcc908), UINT64_C(0xbb67ae8584caa73b),
    UINT64_C(0x3c6ef372fe94f82b), UINT64_C(0xa54ff53a5f1d36f1),
    UINT64_C(0x510e527fade682d1), UINT64_C(0x9b05688c2b3e6c1f),
    UINT64_C(0x1f83d9abfb41bd6b), UINT64_C(0x5be0cd19137e2179)
};

__constant__ uint8_t blake2b_sigma[12][16];

static __device__ __forceinline__ uint64_t rotr64(const uint64_t w, const unsigned c) {
    return (w >> c) | (w << (64 - c));
}

static __device__ __forceinline__ void G(uint64_t* v, const uint64_t* m, const int r, const int i, const int j, int a, int b, int c, int d) {
    a = a + b + m[blake2b_sigma[r][i]];
    d = rotr64(d ^ a, 32);
    c = c + d;
    b = rotr64(b ^ c, 24);
    a = a + b + m[blake2b_sigma[r][j]];
    d = rotr64(d ^ a, 16);
    c = c + d;
    b = rotr64(b ^ c, 63);
}

__global__ void blake2b_compress_kernel(uint64_t* d_h, uint64_t* d_m, uint64_t* d_v, int rounds) {
    int tid = threadIdx.x;
    int round = blockIdx.x;

    if (round < rounds) {
        __shared__ uint64_t v[16];
        __shared__ uint64_t m[16];

        for (int i = 0; i < 16; ++i) {
            v[i] = d_v[i];
            m[i] = d_m[i];
        }

        G(v, m, round, 0, 1, 0, 4, 8, 12);
        G(v, m, round, 2, 3, 1, 5, 9, 13);
        G(v, m, round, 4, 5, 2, 6, 10, 14);
        G(v, m, round, 6, 7, 3, 7, 11, 15);
        G(v, m, round, 8, 9, 0, 5, 10, 15);
        G(v, m, round, 10, 11, 1, 6, 11, 12);
        G(v, m, round, 12, 13, 2, 7, 8, 13);
        G(v, m, round, 14, 15, 3, 4, 9, 14);

        __syncthreads();

        for (int i = 0; i < 8; ++i) {
            d_h[i] ^= v[i] ^ v[i + 8];
        }
    }
}

void blake2b_compress(blake2b_state* S, const uint8_t* block, int rounds) {
    uint64_t m[16];
    uint64_t v[16];
    uint64_t h[8];

    for (int i = 0; i < 16; ++i) {
        m[i] = load64(block + i * sizeof(m[i]));
    }

    for (int i = 0; i < 8; ++i) {
        v[i] = S->h[i];
        h[i] = S->h[i];
    }

    v[8] = blake2b_IV[0];
    v[9] = blake2b_IV[1];
    v[10] = blake2b_IV[2];
    v[11] = blake2b_IV[3];
    v[12] = blake2b_IV[4] ^ S->t[0];
    v[13] = blake2b_IV[5] ^ S->t[1];
    v[14] = blake2b_IV[6] ^ S->f[0];
    v[15] = blake2b_IV[7] ^ S->f[1];

    uint64_t* d_h;
    uint64_t* d_m;
    uint64_t* d_v;

    cudaMalloc(&d_h, sizeof(h));
    cudaMalloc(&d_m, sizeof(m));
    cudaMalloc(&d_v, sizeof(v));

    cudaMemcpy(d_h, h, sizeof(h), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, m, sizeof(m), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, sizeof(v), cudaMemcpyHostToDevice);

    int num_blocks = rounds;
    int threads_per_block = THREADS_PER_BLOCK;

    blake2b_compress_kernel<<<num_blocks, threads_per_block>>>(d_h, d_m, d_v, rounds);

    cudaMemcpy(h, d_h, sizeof(h), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 8; ++i) {
        S->h[i] ^= h[i];
    }

    cudaFree(d_h);
    cudaFree(d_m);
    cudaFree(d_v);
}

void hashx_blake2b_4r(const blake2b_param* params, const void* in, size_t inlen, void* out) {
    blake2b_state state;
    const uint8_t* p = (const uint8_t*)params;

    blake2b_init0(&state);
    for (unsigned i = 0; i < 8; ++i) {
        state.h[i] ^= load64(&p[i * sizeof(state.h[i])]);
    }

    const uint8_t* pin = (const uint8_t*)in;

    while (inlen > BLAKE2B_BLOCKBYTES) {
        blake2b_increment_counter(&state, BLAKE2B_BLOCKBYTES);
        blake2b_compress(&state, pin, 4);
        inlen -= BLAKE2B_BLOCKBYTES;
        pin += BLAKE2B_BLOCKBYTES;
    }

    memcpy(state.buf, pin, inlen);
    blake2b_increment_counter(&state, inlen);
    blake2b_set_lastblock(&state);
    blake2b_compress(&state, state.buf, 4);

    /* Output hash */
    memcpy(out, state.h, sizeof(state.h));
}
