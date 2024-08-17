#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <hashx.h>
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

// Precompute constants
__constant__ uint8_t precomputed_salt[32];

static __device__ void sip_round(uint64_t &v0, uint64_t &v1, uint64_t &v2, uint64_t &v3) {
    v0 += v1; v2 += v3;
    v1 = __funnelshift_r(v1, v1, 13); // Use __funnelshift intrinsic for rotation
    v3 = __funnelshift_r(v3, v3, 16);
    v1 ^= v0; v3 ^= v2;
    v0 = __funnelshift_l(v0, v0, 32);
    v2 += v1; v0 += v3;
    v1 = __funnelshift_r(v1, v1, 17);
    v3 = __funnelshift_r(v3, v3, 21);
    v1 ^= v2; v3 ^= v0;
    v2 = __funnelshift_l(v2, v2, 32);
}

__global__ void hashx_kernel(const hashx_ctx* ctx, const void* input, void* output) {
    assert(ctx != NULL && ctx != HASHX_NOTSUPP);
    assert(output != NULL);

    uint64_t r[8];
    const uint8_t* salt = precomputed_salt;

    // Use shared memory for storing intermediate results
    __shared__ uint64_t shared_r[8];

    // Load input into shared memory for coalesced memory access
    shared_r[threadIdx.x] = ((uint64_t*)input)[threadIdx.x];
    __syncthreads();

#ifndef HASHX_BLOCK_MODE
    hashx_siphash24_ctr_state512(&ctx->keys, shared_r, r);  // Updated function call
#else
    hashx_blake2b_4r(&ctx->params, shared_r, size, r);
#endif

    if (ctx->type & HASHX_COMPILED) {
        ctx->func(r);
    } else {
        hashx_program_execute(ctx->program, r);
    }

    // Finalization
    r[0] ^= load64(&salt[0]);
    r[1] ^= load64(&salt[8]);
    r[2] ^= load64(&salt[16]);
    r[3] ^= load64(&salt[24]);

    // Optimize SIPROUND calls
    sip_round(r[0], r[1], r[2], r[3]);
    sip_round(r[4], r[5], r[6], r[7]);

    // Coalesced write back to global memory
    if (threadIdx.x < 4) {
        ((uint64_t*)output)[threadIdx.x] = r[threadIdx.x] ^ r[threadIdx.x + 4];
    }
}

static int initialize_program(hashx_ctx* ctx, hashx_program* program, siphash_state keys[2]) {
    if (!hashx_program_generate(&keys[0], program)) {
        return 0;
    }
#ifndef HASHX_BLOCK_MODE
    memcpy(&ctx->keys, &keys[1], 32);
#else
    memcpy(precomputed_salt, &keys[1], 32);
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
    hashx_blake2b_init_param(&hash_state, &hashx_blake2_params);
    hashx_blake2b_update(&hash_state, seed, size);
    hashx_blake2b_final(&hash_state, &keys, sizeof(keys));

    if (ctx->type & HASHX_COMPILED) {
        hashx_program program;
        if (!initialize_program(ctx, &program, keys)) {  // Corrected function call
            return 0;
        }
        hashx_compile(&program, ctx->code);
        return 1;
    }
    return initialize_program(ctx, ctx->program, keys);  // Corrected function call
}
