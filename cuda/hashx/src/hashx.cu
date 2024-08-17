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

// Initialization of the program (No changes, keeping it simple)
static int initialize_program(hashx_ctx* ctx, hashx_program* program, siphash_state keys[2]) {
	if (!hashx_program_generate(&keys[0], program)) {
		return 0;
	}
#ifndef HASHX_BLOCK_MODE
	memcpy(&ctx->keys, &keys[1], 32);
#else
	memcpy(&ctx->params.salt, &keys[1], 32);
#endif
#ifndef NDEBUG
	ctx->has_program = true;
#endif
	return 1;
}

// Function to handle the creation of the hash
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
		if (!initialize_program(ctx, &program, keys)) {
			return 0;
		}
		hashx_compile(&program, ctx->code);
		return 1;
	}

	return initialize_program(ctx, ctx->program, keys);
}

// Device function for hash execution (Optimized with shared memory usage and reduced divergence)
__device__ void hashx_exec(const hashx_ctx* ctx, HASHX_INPUT_ARGS, void* output) {
	assert(ctx != NULL && ctx != HASHX_NOTSUPP);
	assert(output != NULL);
	assert(ctx->has_program);

	// Use shared memory for intermediate calculations to reduce global memory access
	__shared__ uint64_t r[8];

#ifndef HASHX_BLOCK_MODE
	hashx_siphash24_ctr_state512(&ctx->keys, input, r);
#else
	hashx_blake2b_4r(&ctx->params, input, size, r);
#endif

	if (ctx->type & HASHX_COMPILED) {
		ctx->func(r);
	} else {
		hashx_program_execute(ctx->program, r);
	}

	// Finalize the hash to remove bias
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

	// Perform final rounds of SIP hash
	SIPROUND(r[0], r[1], r[2], r[3]);
	SIPROUND(r[4], r[5], r[6], r[7]);

	// Output the result
#if HASHX_SIZE > 0
	// Handle output in 8-byte chunks
#if HASHX_SIZE % 8 == 0
	uint8_t* temp_out = (uint8_t*)output;
#if HASHX_SIZE >= 8
	store64(temp_out + 0, r[0] ^ r[4]);
#endif
#if HASHX_SIZE >= 16
	store64(temp_out + 8, r[1] ^ r[5]);
#endif
#if HASHX_SIZE >= 24
	store64(temp_out + 16, r[2] ^ r[6]);
#endif
#if HASHX_SIZE >= 32
	store64(temp_out + 24, r[3] ^ r[7]);
#endif
#else // Handle any other output size
	uint8_t temp_out[32];
#if HASHX_SIZE > 0
	store64(temp_out + 0, r[0] ^ r[4]);
#endif
#if HASHX_SIZE > 8
	store64(temp_out + 8, r[1] ^ r[5]);
#endif
#if HASHX_SIZE > 16
	store64(temp_out + 16, r[2] ^ r[6]);
#endif
#if HASHX_SIZE > 24
	store64(temp_out + 24, r[3] ^ r[7]);
#endif
	memcpy(output, temp_out, HASHX_SIZE);
#endif
#endif
}
