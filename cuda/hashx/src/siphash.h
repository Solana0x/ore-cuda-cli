#ifndef SIPHASH_H
#define SIPHASH_H

#include <stdint.h>

// Rotate left macro
#define ROTL(x, b) (((x) << (b)) | ((x) >> (64 - (b))))

// SipHash round function
#define SIPROUND(v0, v1, v2, v3) \
  do { \
    v0 += v1; v2 += v3; v1 = ROTL(v1, 13);   \
    v3 = ROTL(v3, 16); v1 ^= v0; v3 ^= v2;   \
    v0 = ROTL(v0, 32); v2 += v1; v0 += v3;   \
    v1 = ROTL(v1, 17);  v3 = ROTL(v3, 21);   \
    v1 ^= v2; v3 ^= v0; v2 = ROTL(v2, 32);   \
  } while (0)

// SipHash state structure
typedef struct siphash_state {
    uint64_t v0, v1, v2, v3;
} siphash_state;

// Function to compute SipHash-1-3
uint64_t hashx_siphash13_ctr(uint64_t input, const siphash_state* keys) {
    uint64_t v0 = keys->v0;
    uint64_t v1 = keys->v1;
    uint64_t v2 = keys->v2;
    uint64_t v3 = keys->v3;

    v3 ^= input;

    SIPROUND(v0, v1, v2, v3);
    SIPROUND(v0, v1, v2, v3);

    v0 ^= input;
    v2 ^= 0xff;

    SIPROUND(v0, v1, v2, v3);
    SIPROUND(v0, v1, v2, v3);
    SIPROUND(v0, v1, v2, v3);
    SIPROUND(v0, v1, v2, v3);

    return (v0 ^ v1) ^ (v2 ^ v3);
}

// Function to compute SipHash-2-4 and store state in an array
void hashx_siphash24_ctr_state512(const siphash_state* keys, uint64_t input, uint64_t state_out[8]) {
    uint64_t v0 = keys->v0;
    uint64_t v1 = keys->v1;
    uint64_t v2 = keys->v2;
    uint64_t v3 = keys->v3;

    v3 ^= input;

    SIPROUND(v0, v1, v2, v3);
    SIPROUND(v0, v1, v2, v3);

    v0 ^= input;
    v2 ^= 0xff;

    SIPROUND(v0, v1, v2, v3);
    SIPROUND(v0, v1, v2, v3);
    SIPROUND(v0, v1, v2, v3);
    SIPROUND(v0, v1, v2, v3);

    // Store the result in the output array
    state_out[0] = v0;
    state_out[1] = v1;
    state_out[2] = v2;
    state_out[3] = v3;
    state_out[4] = v0 ^ v1;
    state_out[5] = v2 ^ v3;
    state_out[6] = v0 ^ v2;
    state_out[7] = v1 ^ v3;
}

#endif // SIPHASH_H
