#ifndef SIPHASH_H
#define SIPHASH_H

#include <stdint.h>
#include <immintrin.h> // for AVX2 intrinsics on CPU

#define ROTL(x, b) (((x) << (b)) | ((x) >> (64 - (b))))

#ifdef __CUDACC__
    #define INLINE __device__ __forceinline__
#else
    #define INLINE inline
#endif

#define SIPROUND(v0, v1, v2, v3) \
  do { \
    v0 += v1; v2 += v3; v1 = ROTL(v1, 13);   \
    v3 = ROTL(v3, 16); v1 ^= v0; v3 ^= v2;   \
    v0 = ROTL(v0, 32); v2 += v1; v0 += v3;   \
    v1 = ROTL(v1, 17);  v3 = ROTL(v3, 21);   \
    v1 ^= v2; v3 ^= v0; v2 = ROTL(v2, 32);   \
  } while (0)

typedef struct siphash_state {
    uint64_t v0, v1, v2, v3;
} siphash_state;

#ifdef __cplusplus
extern "C" {
#endif

INLINE uint64_t hashx_siphash13_ctr(uint64_t input, const siphash_state* keys) {
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

__device__ INLINE void hashx_siphash24_ctr_state512(const siphash_state* keys, uint64_t input, uint64_t state_out[8]) {
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

    state_out[0] = v0;
    state_out[1] = v1;
    state_out[2] = v2;
    state_out[3] = v3;
    state_out[4] = v0 ^ v1;
    state_out[5] = v2 ^ v3;
    state_out[6] = v0 ^ v2;
    state_out[7] = v1 ^ v3;
}

#ifdef __cplusplus
}
#endif

#endif // SIPHASH_H
