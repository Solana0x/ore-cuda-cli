#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <cuda.h>

#define BLAKE2B_BLOCKBYTES 128
#define BLAKE2B_OUTBYTES 64

// Blake2b constants
__constant__ uint64_t blake2b_IV[8] = {
    UINT64_C(0x6a09e667f3bcc908), UINT64_C(0xbb67ae8584caa73b),
    UINT64_C(0x3c6ef372fe94f82b), UINT64_C(0xa54ff53a5f1d36f1),
    UINT64_C(0x510e527fade682d1), UINT64_C(0x9b05688c2b3e6c1f),
    UINT64_C(0x1f83d9abfb41bd6b), UINT64_C(0x5be0cd19137e2179)
};

// Predefined Sigma Constants
__constant__ uint8_t blake2b_sigma[12][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
    { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
    { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
    { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
    { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
    { 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
    { 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
    { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
    { 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 }
};

// Rotate right function
__device__ uint64_t rotr64(const uint64_t w, const unsigned c) {
    return (w >> c) | (w << (64 - c));
}

// G function for Blake2b's F function
__device__ void G(uint64_t &a, uint64_t &b, uint64_t &c, uint64_t &d, uint64_t m1, uint64_t m2, unsigned r, unsigned i, unsigned j) {
    a = a + b + m1;
    d = rotr64(d ^ a, 32);
    c = c + d;
    b = rotr64(b ^ c, 24);
    a = a + b + m2;
    d = rotr64(d ^ a, 16);
    c = c + d;
    b = rotr64(b ^ c, 63);
}

// Compress function for Blake2b
__global__ void blake2b_compress(uint64_t* h, const uint8_t* block, uint64_t* t, uint64_t* f, size_t num_blocks) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_blocks) {
        uint64_t m[16];
        uint64_t v[16];
        unsigned int i;

        for (i = 0; i < 16; ++i) {
            m[i] = ((uint64_t *)block)[i + idx * 16];
        }

        for (i = 0; i < 8; ++i) {
            v[i] = h[i];
        }

        v[8] = blake2b_IV[0];
        v[9] = blake2b_IV[1];
        v[10] = blake2b_IV[2];
        v[11] = blake2b_IV[3];
        v[12] = blake2b_IV[4] ^ t[0];
        v[13] = blake2b_IV[5] ^ t[1];
        v[14] = blake2b_IV[6] ^ f[0];
        v[15] = blake2b_IV[7] ^ f[1];

        for (int round = 0; round < 12; ++round) {
            G(v[0], v[4], v[8], v[12], m[blake2b_sigma[round][0]], m[blake2b_sigma[round][1]], round, 0, 1);
            G(v[1], v[5], v[9], v[13], m[blake2b_sigma[round][2]], m[blake2b_sigma[round][3]], round, 2, 3);
            G(v[2], v[6], v[10], v[14], m[blake2b_sigma[round][4]], m[blake2b_sigma[round][5]], round, 4, 5);
            G(v[3], v[7], v[11], v[15], m[blake2b_sigma[round][6]], m[blake2b_sigma[round][7]], round, 6, 7);
            G(v[0], v[5], v[10], v[15], m[blake2b_sigma[round][8]], m[blake2b_sigma[round][9]], round, 8, 9);
            G(v[1], v[6], v[11], v[12], m[blake2b_sigma[round][10]], m[blake2b_sigma[round][11]], round, 10, 11);
            G(v[2], v[7], v[8], v[13], m[blake2b_sigma[round][12]], m[blake2b_sigma[round][13]], round, 12, 13);
            G(v[3], v[4], v[9], v[14], m[blake2b_sigma[round][14]], m[blake2b_sigma[round][15]], round, 14, 15);
        }

        for (i = 0; i < 8; ++i) {
            h[i] ^= v[i] ^ v[i + 8];
        }
    }
}

// Entry point for Blake2b hashing on the GPU
void blake2b_hash_gpu(uint64_t* h, const uint8_t* in, size_t inlen, uint64_t* t, uint64_t* f, size_t num_blocks) {
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

int main() {
    uint64_t h[8];
    uint64_t t[2] = { 0, 0 };
    uint64_t f[2] = { 0, 0 };
    uint8_t in[BLAKE2B_BLOCKBYTES] = {0}; // Replace with actual input data
    size_t inlen = sizeof(in);
    size_t num_blocks = 1; // Adjust as per input size

    // Initialize h with IV
    memcpy(h, blake2b_IV, sizeof(blake2b_IV));

    blake2b_hash_gpu(h, in, inlen, t, f, num_blocks);

    // Output the hash
    for (int i = 0; i < 8; ++i) {
        printf("%016lx", h[i]);
    }
    printf("\n");

    return 0;
}
