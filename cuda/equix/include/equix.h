/* 
 * Copyright (c) 2020 tevador <tevador@gmail.com>
 * See LICENSE for licensing information 
 */

#ifndef EQUIX_H
#define EQUIX_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>  // For dynamic memory allocation

/* Maximum number of solutions returned by the solver */
#define EQUIX_MAX_SOLS 16

/* Number of indices */
#define EQUIX_NUM_IDX 32

/* 16-bit index type */
typedef uint16_t equix_idx;

/* Solution structure */
typedef struct equix_solution {
    equix_idx* idx;  // Dynamically allocated array
} equix_solution;

/* Opaque structure for Equi-X context */
typedef struct equix_ctx equix_ctx;

/* Context creation flags */
typedef enum equix_ctx_flags {
    EQUIX_CTX_VERIFY = 0,       /* Context for verification */
    EQUIX_CTX_SOLVE = 1,        /* Context for solving */
    EQUIX_CTX_COMPILE = 2,      /* Compile internal hash function */
} equix_ctx_flags;

/* Sentinel value indicating unsupported type */
#define EQUIX_NOTSUPP ((equix_ctx*)-1)

/* Shared/static library definitions */
#if defined(EQUIX_SHARED)
    #define EQUIX_API __attribute__ ((visibility ("default")))
#else
    #define EQUIX_API
#endif

#define EQUIX_PRIVATE __attribute__ ((visibility ("hidden")))

#ifdef __cplusplus
extern "C" {
#endif

/* Allocate an Equi-X context */
EQUIX_API equix_ctx* equix_alloc(equix_ctx_flags flags);

/* Free an Equi-X context */
EQUIX_API void equix_free(equix_ctx* ctx);

/*
 * Initialize the solution structure.
 * This function allocates memory for the indices in the solution.
 *
 * @param solution is a pointer to an equix_solution structure.
 */
EQUIX_API void equix_solution_init(equix_solution* solution);

/*
 * Free the memory allocated for the solution structure.
 *
 * @param solution is a pointer to an equix_solution structure.
 */
EQUIX_API void equix_solution_free(equix_solution* solution);

/*
 * Solve the problem using all available CPU cores.
 *
 * @param ctx is a pointer to the context.
 * @param solutions is a pointer to an array of equix_solution structures.
 * @param max_sols is the maximum number of solutions to find (should not exceed EQUIX_MAX_SOLS).
 *
 * @return the number of solutions found.
 */
EQUIX_API int equix_solve_parallel_cpu(equix_ctx* ctx, equix_solution* solutions, int max_sols);

/*
 * Solve the problem using GPU acceleration.
 *
 * @param ctx is a pointer to the context.
 * @param solutions is a pointer to an array of equix_solution structures.
 * @param max_sols is the maximum number of solutions to find (should not exceed EQUIX_MAX_SOLS).
 *
 * @return the number of solutions found.
 */
EQUIX_API int equix_solve_parallel_gpu(equix_ctx* ctx, equix_solution* solutions, int max_sols);

#ifdef __cplusplus
}
#endif

#endif /* EQUIX_H */
