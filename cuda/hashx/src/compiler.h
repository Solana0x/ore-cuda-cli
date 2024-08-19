/* Copyright (c) 2020 tevador <tevador@gmail.com> */
/* See LICENSE for licensing information */

#ifndef COMPILER_H
#define COMPILER_H

#include <stdint.h>
#include <stdbool.h>
#include <../include/hashx.h>
#include "program.h"

#define HASHX_COMPILER 0
#define hashx_compile

HASHX_PRIVATE bool hashx_compiler_init(hashx_ctx* compiler);
HASHX_PRIVATE void hashx_compiler_destroy(hashx_ctx* compiler);

// Alignment macro, optimized for the target architecture (typically 64-byte alignment for modern CPUs)
#define ALIGN_SIZE(pos, align) ((((pos) - 1) / (align) + 1) * (align))

// Optimized page size for potentially faster memory operations (considering larger page size)
#define COMP_PAGE_SIZE 65536  // Increased from 4096 to 65536 bytes

// Adjusted reserve size to provide more buffer space, potentially reducing memory reallocation overhead
#define COMP_RESERVE_SIZE 2048  // Increased from 1024 to 2048 bytes

// Average instruction size (remain the same, assuming 5 bytes per instruction is accurate)
#define COMP_AVG_INSTR_SIZE 5

// Code size calculation, aligned to the new page size
#define COMP_CODE_SIZE                                                        \
	ALIGN_SIZE(                                                               \
		HASHX_PROGRAM_MAX_SIZE * COMP_AVG_INSTR_SIZE + COMP_RESERVE_SIZE,     \
	COMP_PAGE_SIZE)

#endif
