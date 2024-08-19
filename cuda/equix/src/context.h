#ifndef CONTEXT_H
#define CONTEXT_H

#include <../include/equix.h>
#include <../../hashx/include/hashx.h>

typedef struct solver_heap solver_heap;
typedef struct __attribute__((aligned(64))) equix_ctx {
    hashx_ctx* __restrict__ hash_func;  // Use __restrict__ to help with potential pointer aliasing
    solver_heap* __restrict__ heap;
    equix_ctx_flags flags;
} equix_ctx;

#endif
