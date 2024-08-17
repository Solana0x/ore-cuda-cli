#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "blake2.h"
#include "hashx_endian.h"

// Constants for Blake2b
static const uint64_t blake2b_IV[8] = {
    UINT64_C(0x6a09e667f3bcc908), UINT64_C(0xbb67ae8584caa73b),
    UINT64_C(0x3c6ef372fe94f82b), UINT64_C(0xa54ff53a5f1d36f1),
    UINT64_C(0x510e527fade682d1), UINT64_C(0x9b05688c2b3e6c1f),
    UINT64_C(0x1f83d9abfb41bd6b), UINT64_C(0x5be0cd19137e2179)
};

// Predefined Sigma Constants
static const uint8_t blake2b_sigma[12][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
    { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
    { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
    { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
    { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
    { 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
    { 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
    { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
    { 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 }
};

// Rotate right function
static FORCE_INLINE uint64_t rotr64(const uint64_t w, const unsigned c) {
    return (w >> c) | (w << (64 - c));
}

// Set the last block indicator flag
static FORCE_INLINE void blake2b_set_lastblock(blake2b_state* S) {
    S->f[0] = (uint64_t)-1;
}

// Increment the counter
static FORCE_INLINE void blake2b_increment_counter(blake2b_state* S, uint64_t inc) {
    S->t[0] += inc;
    if (S->t[0] < inc) {
        S->t[1]++;
    }
}

// Initialize the Blake2b state with IVs
static FORCE_INLINE void blake2b_init0(blake2b_state* S) {
    memset(S, 0, sizeof(*S));
    memcpy(S->h, blake2b_IV, sizeof(S->h));
}

// Initialize the Blake2b state with parameters
int hashx_blake2b_init_param(blake2b_state* S, const blake2b_param* P) {
    if (!P || !S) {
        return -1;
    }

    blake2b_init0(S);
    const unsigned char* p = (const unsigned char*)P;
    for (unsigned int i = 0; i < 8; ++i) {
        S->h[i] ^= load64(&p[i * sizeof(S->h[i])]);
    }
    S->outlen = P->digest_length;
    return 0;
}

// G function for Blake2b's F function
#define G(r, i, j, a, b, c, d)                                               \
    do {                                                                     \
        a += b + m[blake2b_sigma[r][i]];                                     \
        d = rotr64(d ^ a, 32);                                               \
        c += d;                                                              \
        b = rotr64(b ^ c, 24);                                               \
        a += b + m[blake2b_sigma[r][j]];                                     \
        d = rotr64(d ^ a, 16);                                               \
        c += d;                                                              \
        b = rotr64(b ^ c, 63);                                               \
    } while (0)

// Round function with the inner rounds of Blake2b
#define ROUND(r)                                                             \
    do {                                                                     \
        G(r,  0,  1, v[0], v[4], v[8], v[12]);                               \
        G(r,  2,  3, v[1], v[5], v[9], v[13]);                               \
        G(r,  4,  5, v[2], v[6], v[10], v[14]);                              \
        G(r,  6,  7, v[3], v[7], v[11], v[15]);                              \
        G(r,  8,  9, v[0], v[5], v[10], v[15]);                              \
        G(r, 10, 11, v[1], v[6], v[11], v[12]);                              \
        G(r, 12, 13, v[2], v[7], v[8], v[13]);                               \
        G(r, 14, 15, v[3], v[4], v[9], v[14]);                               \
    } while (0)

// Compress function for Blake2b
static void blake2b_compress(blake2b_state* S, const uint8_t* block) {
    uint64_t m[16];
    uint64_t v[16];

    for (unsigned int i = 0; i < 16; ++i) {
        m[i] = load64(block + i * sizeof(m[i]));
    }

    for (unsigned int i = 0; i < 8; ++i) {
        v[i] = S->h[i];
    }

    v[8] = blake2b_IV[0];
    v[9] = blake2b_IV[1];
    v[10] = blake2b_IV[2];
    v[11] = blake2b_IV[3];
    v[12] = blake2b_IV[4] ^ S->t[0];
    v[13] = blake2b_IV[5] ^ S->t[1];
    v[14] = blake2b_IV[6] ^ S->f[0];
    v[15] = blake2b_IV[7] ^ S->f[1];

    // 12 rounds
    for (unsigned int i = 0; i < 12; ++i) {
        ROUND(i);
    }

    for (unsigned int i = 0; i < 8; ++i) {
        S->h[i] ^= v[i] ^ v[i + 8];
    }
}

// 4-round version of the Blake2b compress function
static void blake2b_compress_4r(blake2b_state* S, const uint8_t* block) {
    uint64_t m[16];
    uint64_t v[16];

    for (unsigned int i = 0; i < 16; ++i) {
        m[i] = load64(block + i * sizeof(m[i]));
    }

    for (unsigned int i = 0; i < 8; ++i) {
        v[i] = S->h[i];
    }

    v[8] = blake2b_IV[0];
    v[9] = blake2b_IV[1];
    v[10] = blake2b_IV[2];
    v[11] = blake2b_IV[3];
    v[12] = blake2b_IV[4] ^ S->t[0];
    v[13] = blake2b_IV[5] ^ S->t[1];
    v[14] = blake2b_IV[6] ^ S->f[0];
    v[15] = blake2b_IV[7] ^ S->f[1];

    // 4 rounds
    for (unsigned int i = 0; i < 4; ++i) {
        ROUND(i);
    }

    for (unsigned int i = 0; i < 8; ++i) {
        S->h[i] ^= v[i] ^ v[i + 8];
    }
}

// Update function for Blake2b
int hashx_blake2b_update(blake2b_state* S, const void* in, size_t inlen) {
    if (!S || !in || inlen == 0) {
        return 0;
    }

    const uint8_t* pin = (const uint8_t*)in;

    while (inlen > 0) {
        size_t fill = BLAKE2B_BLOCKBYTES - S->buflen;

        if (inlen > fill) {
            memcpy(S->buf + S->buflen, pin, fill);
            blake2b_increment_counter(S, BLAKE2B_BLOCKBYTES);
            blake2b_compress(S, S->buf);
            S->buflen = 0;
            inlen -= fill;
            pin += fill;
        } else {
            memcpy(S->buf + S->buflen, pin, inlen);
            S->buflen += (unsigned int)inlen;
            break;
        }
    }

    return 0;
}

// Finalization function for Blake2b
int hashx_blake2b_final(blake2b_state* S, void* out, size_t outlen) {
    if (!S || !out || outlen < S->outlen || S->f[0] != 0) {
        return -1;
    }

    blake2b_increment_counter(S, S->buflen);
    blake2b_set_lastblock(S);
    memset(S->buf + S->buflen, 0, BLAKE2B_BLOCKBYTES - S->buflen); /* Padding */
    blake2b_compress(S, S->buf);

    for (unsigned int i = 0; i < 8; ++i) {
        store64((uint8_t*)out + sizeof(S->h[i]) * i, S->h[i]);
    }

    return 0;
}

// 4-round version of Blake2b
void hashx_blake2b_4r(const blake2b_param* params, const void* in, size_t inlen, void* out) {
    blake2b_state state;

    if (!params || !in || !out) return;

    blake2b_init0(&state);

    for (unsigned int i = 0; i < 8; ++i) {
        state.h[i] ^= load64(((const uint8_t*)params) + i * sizeof(state.h[i]));
    }

    const uint8_t* pin = (const uint8_t*)in;

    while (inlen > BLAKE2B_BLOCKBYTES) {
        blake2b_increment_counter(&state, BLAKE2B_BLOCKBYTES);
        blake2b_compress_4r(&state, pin);
        inlen -= BLAKE2B_BLOCKBYTES;
        pin += BLAKE2B_BLOCKBYTES;
    }

    memcpy(state.buf, pin, inlen);
    blake2b_increment_counter(&state, inlen);
    blake2b_set_lastblock(&state);
    blake2b_compress_4r(&state, state.buf);

    /* Output hash */
    memcpy(out, state.h, sizeof(state.h));
}
