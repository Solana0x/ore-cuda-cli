#include <stdlib.h>
#include <string.h>
#include <../include/hashx.h>
#include "context.h"
#include "compiler.h"
#include "program.h"

#define STRINGIZE_INNER(x) #x
#define STRINGIZE(x) STRINGIZE_INNER(x)

#ifndef HASHX_SALT
#define HASHX_SALT "HashX v1"
#endif

__device__ const blake2b_param hashx_blake2_params = {
    64, 0, 1, 1, 0, 0, 0, 0, { 0 }, STRINGIZE(HASHX_SALT), { 0 }
};

hashx_ctx* hashx_alloc(hashx_type type) {
    hashx_ctx* ctx;

    // Allocate memory for context
    if (cudaMallocManaged(&ctx, sizeof(hashx_ctx)) != cudaSuccess) {
        return NULL;
    }

    // Initialize the context
    ctx->code = NULL;
    ctx->program = NULL;
    
    // Set the type
    ctx->type = HASHX_UNDEFINED;  // Replace 'HASHX_UNDEFINED' with the appropriate value

    // Choose the appropriate type
    if (type & HASHX_COMPILED) {
        if (!hashx_compiler_init(ctx)) {
            cudaFree(ctx);
            return NULL;
        }
        ctx->type = HASHX_COMPILED;
    } else {
        if (cudaMallocManaged(&ctx->program, sizeof(hashx_program)) != cudaSuccess) {
            cudaFree(ctx);
            return NULL;
        }
        ctx->type = HASHX_INTERPRETED;
    }

#ifdef HASHX_BLOCK_MODE
    cudaMemcpy(&ctx->params, &hashx_blake2_params, sizeof(blake2b_param), cudaMemcpyDefault);
#endif

    return ctx;
}

void hashx_free(hashx_ctx* ctx) {
    if (ctx != NULL && ctx != HASHX_NOTSUPP) {
        if (ctx->type & HASHX_COMPILED) {
            hashx_compiler_destroy(ctx);
        } else if (ctx->program != NULL) {
            cudaFree(ctx->program);
        }
        cudaFree(ctx);
    }
}
