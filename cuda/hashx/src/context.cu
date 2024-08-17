#include <stdlib.h>
#include <string.h>
#include <hashx.h>
#include "context.h"
#include "compiler.h"
#include "program.h"

#define STRINGIZE_INNER(x) #x
#define STRINGIZE(x) STRINGIZE_INNER(x)

#ifndef HASHX_SALT
#define HASHX_SALT "HashX v1"
#endif

// Device constant for Blake2b parameters
__device__ const blake2b_param hashx_blake2_params = {
    64, 0, 1, 1, 0, 0, 0, 0, { 0 }, STRINGIZE(HASHX_SALT), { 0 }
};

hashx_ctx* hashx_alloc(hashx_type type) {
    hashx_ctx* ctx = NULL;

    // Allocate unified memory for context and check for errors
    cudaError_t err = cudaMallocManaged(&ctx, sizeof(hashx_ctx));
    if (err != cudaSuccess || ctx == NULL) {
        fprintf(stderr, "Failed to allocate memory for hashx_ctx: %s\n", cudaGetErrorString(err));
        return NULL;
    }

    // Initialize pointers to NULL to avoid undefined behavior
    ctx->code = NULL;
    ctx->program = NULL;

    // Allocate memory based on the type of context
    if (type & HASHX_COMPILED) {
        if (!hashx_compiler_init(ctx)) {
            cudaFree(ctx);
            return NULL;
        }
        ctx->type = HASHX_COMPILED;
    } else {
        err = cudaMallocManaged(&ctx->program, sizeof(hashx_program));
        if (err != cudaSuccess || ctx->program == NULL) {
            fprintf(stderr, "Failed to allocate memory for hashx_program: %s\n", cudaGetErrorString(err));
            cudaFree(ctx);
            return NULL;
        }
        ctx->type = HASHX_INTERPRETED;
    }

#ifdef HASHX_BLOCK_MODE
    // Direct initialization of blake2b_param structure (faster than memcpy)
    ctx->params = hashx_blake2_params;
#endif

    return ctx;
}

void hashx_free(hashx_ctx* ctx) {
    if (ctx != NULL && ctx != HASHX_NOTSUPP) {
        if (ctx->type & HASHX_COMPILED) {
            hashx_compiler_destroy(ctx);
        } else {
            cudaFree(ctx->program);
        }
        cudaFree(ctx);
    }
}
