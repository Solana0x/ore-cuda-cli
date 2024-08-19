#ifndef CONTEXT_H
#define CONTEXT_H

#include <../include/equix.h>
#include <../../hashx/include/hashx.h>

typedef struct solver_heap solver_heap;
typedef struct __attribute__((aligned(64))) equix_ctx {
    hashx_ctx* hash_func;  // Standard pointer without restrict
    solver_heap* heap;     // Standard pointer without restrict
    equix_ctx_flags flags;
} equix_ctx;

#endif
