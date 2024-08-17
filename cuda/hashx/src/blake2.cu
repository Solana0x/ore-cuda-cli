#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define BLAKE2B_BLOCKBYTES 128

__constant__ uint64_t blake2b_IV[8] = {
    UINT64_C(0x6a09e667f3bcc908), UINT64_C(0xbb67ae8584caa73b),
    UINT64_C(0x3c6ef372fe94f82b), UINT64_C(0xa54ff53a5f1d36f1),
    UINT64_C(0x510e527fade682d1), UINT64_C(0x9b05688c2b3e6c1f),
    UINT64_C(0x1f83d9abfb41bd6b), UINT64_C(0x5be0cd19137e2179)
};

__device__ __forceinline__ uint64_t rotr64(const uint64_t w, const unsigned c) {
    return (w >> c) | (w << (64 - c));
}

#define G(r, i, j, a, b, c, d, m)           \
    do {                                    \
        a = a + b + m[i];                   \
        d = rotr64(d ^ a, 32);              \
        c = c + d;                          \
        b = rotr64(b ^ c, 24);              \
        a = a + b + m[j];                   \
        d = rotr64(d ^ a, 16);              \
        c = c + d;                          \
        b = rotr64(b ^ c, 63);              \
    } while (0)

#define ROUND(r, v, m)                      \
    do {                                    \
        G(r, 0, 1, v[0], v[4], v[8], v[12], m);   \
        G(r, 2, 3, v[1], v[5], v[9], v[13], m);   \
        G(r, 4, 5, v[2], v[6], v[10], v[14], m);  \
        G(r, 6, 7, v[3], v[7], v[11], v[15], m);  \
        G(r, 8, 9, v[0], v[5], v[10], v[15], m);  \
        G(r, 10, 11, v[1], v[6], v[11], v[12], m);\
        G(r, 12, 13, v[2], v[7], v[8], v[13], m); \
        G(r, 14, 15, v[3], v[4], v[9], v[14], m); \
    } while (0)

__global__ void blake2b_compress(uint64_t* h, const uint8_t* in, uint64_t* t, uint64_t* f, size_t num_blocks) {
    uint64_t v[16];
    uint64_t m[16];

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_blocks; i += blockDim.x * gridDim.x) {
        const uint8_t* block = in + i * BLAKE2B_BLOCKBYTES;

        for (int j = 0; j < 16; ++j) {
            m[j] = ((uint64_t*)block)[j];
        }

        for (int j = 0; j < 8; ++j) {
            v[j] = h[j];
        }

        v[8] = blake2b_IV[0];
        v[9] = blake2b_IV[1];
        v[10] = blake2b_IV[2];
        v[11] = blake2b_IV[3];
        v[12] = blake2b_IV[4] ^ t[0];
        v[13] = blake2b_IV[5] ^ t[1];
        v[14] = blake2b_IV[6] ^ f[0];
        v[15] = blake2b_IV[7] ^ f[1];

        ROUND(0, v, m);
        ROUND(1, v, m);
        ROUND(2, v, m);
        ROUND(3, v, m);
        ROUND(4, v, m);
        ROUND(5, v, m);
        ROUND(6, v, m);
        ROUND(7, v, m);
        ROUND(8, v, m);
        ROUND(9, v, m);
        ROUND(10, v, m);
        ROUND(11, v, m);

        for (int j = 0; j < 8; ++j) {
            h[j] ^= v[j] ^ v[j + 8];
        }
    }
}

extern "C" void blake2b_hash_gpu(uint64_t* h, const uint8_t* in, size_t inlen, uint64_t* t, uint64_t* f, size_t num_blocks) {
    uint8_t* d_in;
    uint64_t* d_h;
    uint64_t* d_t;
    uint64_t* d_f;

    cudaMalloc(&d_in, inlen);
    cudaMalloc(&d_h, 8 * sizeof(uint64_t));
    cudaMalloc(&d_t, 2 * sizeof(uint64_t));
    cudaMalloc(&d_f, 2 * sizeof(uint64_t));

    cudaMemcpy(d_in, in, inlen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, h, 8 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t, t, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f, f, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (num_blocks + threads_per_block - 1) / threads_per_block;

    blake2b_compress<<<blocks_per_grid, threads_per_block>>>(d_h, d_in, d_t, d_f, num_blocks);

    cudaMemcpy(h, d_h, 8 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_h);
    cudaFree(d_t);
    cudaFree(d_f);
}
