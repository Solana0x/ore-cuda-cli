#include <stdlib.h>
#include <string.h>
#include <../include/hashx.h>
#include "context.h"
#include "compiler.h"
#include "program.h"

#define STRINGIZE_INNER(x) #x
#define STRINGIZE(x) STRINGIZE_INNER(x)

#ifndef HASHX_SALT
#define HASHX_SALT HashX v1
#endif

// The definition remains here
__device__ const blake2b_param hashx_blake2_params = {
    64, 0, 1, 1, 0, 0, 0, 0, { 0 }, STRINGIZE(HASHX_SALT), { 0 }
};

hashx_ctx* hashx_alloc(hashx_type type) {
    hashx_ctx* ctx = NULL;

    // Allocate unified memory for context
    cudaError_t err = cudaMallocManaged(&ctx, sizeof(hashx_ctx));
    if (err != cudaSuccess) {
        return NULL;
    }

    // Initialize pointers to NULL
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
        if (err != cudaSuccess) {
            cudaFree(ctx);
            return NULL;
        }
        ctx->type = HASHX_INTERPRETED;
    }

#ifdef HASHX_BLOCK_MODE
    // Directly initialize the blake2b_param structure
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
