/* Copyright (c) 2020 tevador <tevador@gmail.com> */
/* See LICENSE for licensing information */

#ifndef CONTEXT_H
#define CONTEXT_H

#include <../include/equix.h>

typedef struct solver_heap solver_heap;

typedef struct equix_ctx {
	hashx_ctx* hash_func;
	solver_heap* heap;
	equix_ctx_flags flags;
} equix_ctx;

#endif
