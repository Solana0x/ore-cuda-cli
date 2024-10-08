#ifndef CONTEXT_H
#define CONTEXT_H

#include <stdbool.h>
#include "../include/hashx.h"
#include "blake2.h"
#include "siphash.h"

typedef void program_func(uint64_t r[8]);

#ifdef __cplusplus
extern "C" {
#endif

// Only declare the variable here
extern __device__ const blake2b_param hashx_blake2_params;

#ifdef __cplusplus
}
#endif

typedef struct hashx_program hashx_program;

/* HashX context. */
typedef struct hashx_ctx {
    union {
        uint8_t* code;
        program_func* func;
        hashx_program* program;
    };
    hashx_type type;
#ifndef HASHX_BLOCK_MODE
    siphash_state keys;
#else
    blake2b_param params;
#endif
#ifndef NDEBUG
    bool has_program;
#endif
} hashx_ctx;

#endif
