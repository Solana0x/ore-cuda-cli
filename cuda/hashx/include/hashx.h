#ifndef HASHX_H
#define HASHX_H

#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>  // Ensure CUDA is included

/*
 * Input of the hash function.
 *
 * Counter mode (default): a 64-bit unsigned integer
 * Block mode: pointer to a buffer and the number of bytes to be hashed
*/
#ifndef HASHX_BLOCK_MODE
#define HASHX_INPUT uint64_t input
#else
#define HASHX_INPUT const void* input, size_t size
#endif

/* The default (and maximum) hash size is 32 bytes */
#ifndef HASHX_SIZE
#define HASHX_SIZE 32
#endif

/* Opaque struct representing a HashX instance */
typedef struct __attribute__((aligned(64))) hashx_ctx hashx_ctx;

/* Type of hash function */
typedef enum hashx_type {
    HASHX_INTERPRETED,
    HASHX_COMPILED
} hashx_type;

/* Sentinel value used to indicate unsupported type */
#define HASHX_NOTSUPP ((hashx_ctx*)-1)

/* Shared/static library definitions */
#ifdef HASHX_SHARED
    #define HASHX_API __attribute__ ((visibility ("default")))
#else
    #define HASHX_API __attribute__ ((visibility ("hidden")))
#endif
#define HASHX_PRIVATE __attribute__ ((visibility ("hidden")))

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Allocate a HashX instance.
 *
 * @param type is the type of instance to be created.
 *
 * @return pointer to a new HashX instance. Returns NULL on memory allocation 
 *         failure and HASHX_NOTSUPP if the requested type is not supported.
*/
__host__ HASHX_API hashx_ctx* hashx_alloc(hashx_type type);

/*
 * Create a new HashX function from seed.
 *
 * @param ctx is pointer to a HashX instance.
 * @param seed is a pointer to the seed value.
 * @param size is the size of the seed.
 *
 * @return 1 on success, 0 on failure.                                         
*/
__host__ HASHX_API int hashx_make(hashx_ctx* __restrict__ ctx, const void* __restrict__ seed, size_t size);

/*
 * Execute the HashX function.
 *
 * @param ctx is pointer to a HashX instance. A HashX function must have
 *        been previously created by calling hashx_make.
 * @param HASHX_INPUT is the input to be hashed (see definition above).
 * @param output is a pointer to the result buffer. HASHX_SIZE bytes will be
 *        written.
 */
__device__ HASHX_API void hashx_exec(const hashx_ctx* __restrict__ ctx, HASHX_INPUT, void* __restrict__ output);

/*
 * Free a HashX instance.
 *
 * @param ctx is pointer to a HashX instance.
*/
__host__ HASHX_API void hashx_free(hashx_ctx* ctx);

#ifdef __cplusplus
}
#endif

#endif
