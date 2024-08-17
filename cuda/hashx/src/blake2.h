#ifndef PORTABLE_BLAKE2_H
#define PORTABLE_BLAKE2_H

#include <stdint.h>
#include <stddef.h>
#include "../include/hashx.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* Blake2b constants */
enum blake2b_constant {
    BLAKE2B_BLOCKBYTES = 128,
    BLAKE2B_OUTBYTES = 64,
    BLAKE2B_KEYBYTES = 64,
    BLAKE2B_SALTBYTES = 16,
    BLAKE2B_PERSONALBYTES = 16
};

/* Blake2b parameter block structure */
#pragma pack(push, 1)
typedef struct blake2b_param {
    uint8_t digest_length;                   /* 1 */
    uint8_t key_length;                      /* 2 */
    uint8_t fanout;                          /* 3 */
    uint8_t depth;                           /* 4 */
    uint32_t leaf_length;                    /* 8 */
    uint64_t node_offset;                    /* 16 */
    uint8_t node_depth;                      /* 17 */
    uint8_t inner_length;                    /* 18 */
    uint8_t reserved[14];                    /* 32 */
    uint8_t salt[BLAKE2B_SALTBYTES];         /* 48 */
    uint8_t personal[BLAKE2B_PERSONALBYTES]; /* 64 */
} blake2b_param;
#pragma pack(pop)

/* Blake2b state structure */
typedef struct blake2b_state {
    uint64_t h[8];                           /* Chained state */
    uint64_t t[2];                           /* Total number of bytes */
    uint64_t f[2];                           /* Finalization flags */
    uint8_t buf[BLAKE2B_BLOCKBYTES];         /* Buffer for data block */
    unsigned buflen;                         /* Current size of the buffer */
    unsigned outlen;                         /* Desired digest size */
    uint8_t last_node;                       /* Indicates the last node */
} blake2b_state;

/* Compile-time size checks */
enum {
    blake2_size_check_0 = 1 / !!(sizeof(uint8_t) == 1),
    blake2_size_check_1 = 1 / !!(sizeof(blake2b_param) == 64)
};

/* Function declarations */
int hashx_blake2b_init_param(blake2b_state* S, const blake2b_param* P);
int hashx_blake2b_update(blake2b_state* S, const void* in, size_t inlen);
int hashx_blake2b_final(blake2b_state* S, void* out, size_t outlen);
HASHX_PRIVATE void hashx_blake2b_4r(const blake2b_param* P, const void* in, size_t inlen, void* out);

#if defined(__cplusplus)
}
#endif

#endif /* PORTABLE_BLAKE2_H */
