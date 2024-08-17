#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <../include/hashx.h>
#include "blake2.h"
#include "hashx_endian.h"
#include "program.h"
#include "context.h"
#include "compiler.h"

#if HASHX_SIZE > 32
#error HASHX_SIZE cannot be more than 32
#endif

#ifndef HASHX_BLOCK_MODE
#define HASHX_INPUT_ARGS input
#else
#define HASHX_INPUT_ARGS input, size
#endif

// Optimize the initialization process using asynchronous operations
__global__ void initialize_program_kernel(hashx_program* program, siphash_state keys) {
    if (!hashx_program_generate(&keys, program)) {
        return;
    }
}

static int initialize_program(hashx_ctx* ctx, hashx_program* program, siphash_state keys[2]) {
    // Use CUDA stream for asynchronous initialization
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    initialize_program_kernel<<<1, 1, 0, stream>>>(program, keys[0]);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

#ifndef HASHX_BLOCK_MODE
    cudaMemcpyAsync(&ctx->keys, &keys[1], 32, cudaMemcpyHostToDevice, stream);
#else
    cudaMemcpyAsync(&ctx->params.salt, &keys[1], 32, cudaMemcpyHostToDevice, stream);
#endif

#ifndef NDEBUG
    ctx->has_program = true;
#endif
    return 1;
}

int hashx_make(hashx_ctx* ctx, const void* seed, size_t size) {
    assert(ctx != NULL && ctx != HASHX_NOTSUPP);
    assert(seed != NULL || size == 0);    

    siphash_state keys[2];
    blake2b_state hash_state;

    // Initialize Blake2b state and hash the seed
    hashx_blake2b_init_param(&hash_state, &hashx_blake2_params);
    hashx_blake2b_update(&hash_state, seed, size);
    hashx_blake2b_final(&hash_state, &keys, sizeof(keys));

    if (ctx->type & HASHX_COMPILED) {
        hashx_program program;
        if (!initialize_program(ctx, &program, keys)) {
            return 0;
        }
        hashx_compile(&program, ctx->code);
        return 1;
    }

    return initialize_program(ctx, ctx->program, keys);
}

__device__ __forceinline__ void sipround_device(uint64_t& v0, uint64_t& v1, uint64_t& v2, uint64_t& v3) {
    v0 += v1; v2 += v3; v1 = __funnelshift_l(v1, v1, 13);
    v3 = __funnelshift_l(v3, v3, 16); v1 ^= v0; v3 ^= v2;
    v0 = __funnelshift_l(v0, v0, 32); v2 += v1; v0 += v3;
    v1 = __funnelshift_l(v1, v1, 17); v3 = __funnelshift_l(v3, v3, 21);
    v1 ^= v2; v3 ^= v0; v2 = __funnelshift_l(v2, v2, 32);
}

__global__ void hashx_exec_kernel(const hashx_ctx* ctx, const uint8_t* input, size_t size, uint8_t* output) {
    __shared__ uint64_t r[8]; // Shared memory for intermediate values

    if (threadIdx.x < 8) {
        r[threadIdx.x] = 0; // Initialize shared memory
    }
    __syncthreads();

#ifndef HASHX_BLOCK_MODE
    hashx_siphash24_ctr_state512(&ctx->keys, input, r);
#else
    hashx_blake2b_4r(&ctx->params, input, size, r);
#endif

    __syncthreads();

    if (ctx->type & HASHX_COMPILED) {
        ctx->func(r);
    } else {
        hashx_program_execute(ctx->program, r);
    }

    // Hash finalization
#ifndef HASHX_BLOCK_MODE
    r[0] += ctx->keys.v0;
    r[1] += ctx->keys.v1;
    r[6] += ctx->keys.v2;
    r[7] += ctx->keys.v3;
#else
    const uint8_t* p = (const uint8_t*)&ctx->params;
    r[0] ^= load64(&p[8 * 0]);
    r[1] ^= load64(&p[8 * 1]);
    r[2] ^= load64(&p[8 * 2]);
    r[3] ^= load64(&p[8 * 3]);
    r[4] ^= load64(&p[8 * 4]);
    r[5] ^= load64(&p[8 * 5]);
    r[6] ^= load64(&p[8 * 6]);
    r[7] ^= load64(&p[8 * 7]);
#endif

    sipround_device(r[0], r[1], r[2], r[3]);
    sipround_device(r[4], r[5], r[6], r[7]);

    __syncthreads();

    // Optimized output
#if HASHX_SIZE > 0
#if HASHX_SIZE % 8 == 0
    if (threadIdx.x < 4) {
        ((uint64_t*)output)[threadIdx.x * 2] = r[threadIdx.x] ^ r[threadIdx.x + 4];
    }
#else /* any output size */
    if (threadIdx.x < 4) {
        uint8_t temp_out[32];
        store64(temp_out + threadIdx.x * 8, r[threadIdx.x] ^ r[threadIdx.x + 4]);
        memcpy(output + threadIdx.x * 8, temp_out, HASHX_SIZE);
    }
#endif
#endif
}

__host__ void hashx_exec(const hashx_ctx* ctx, const void* input, size_t size, void* output) {
    // Launch the kernel with one block and multiple threads
    hashx_exec_kernel<<<1, 256>>>(ctx, (const uint8_t*)input, size, (uint8_t*)output);
    cudaDeviceSynchronize();
}
