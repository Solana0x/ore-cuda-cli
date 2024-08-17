#ifndef SIPHASH_H
#define SIPHASH_H

#include <stdint.h>
#include <immintrin.h> // for AVX2 intrinsics

#define ROTL(x, b) (((x) << (b)) | ((x) >> (64 - (b))))

#define ADD(a, b) _mm256_add_epi64(a, b)
#define XOR(a, b) _mm256_xor_si256(a, b)
#define ROT13(x) _mm256_or_si256(_mm256_slli_epi64(x, 13), _mm256_srli_epi64(x, 51))
#define ROT16(x) _mm256_shuffle_epi8(x, _mm256_set_epi64x(0x0D0C0B0A09080F0EULL, 0x0504030201000706ULL, 0x0D0C0B0A09080F0EULL, 0x0504030201000706ULL))
#define ROT17(x) _mm256_or_si256(_mm256_slli_epi64(x, 17), _mm256_srli_epi64(x, 47))
#define ROT21(x) _mm256_or_si256(_mm256_slli_epi64(x, 21), _mm256_srli_epi64(x, 43))
#define ROT32(x) _mm256_shuffle_epi32(x, _MM_SHUFFLE(2, 3, 0, 1))

#define SIPROUNDXN(v0, v1, v2, v3) \
  do { \
    v0 = ADD(v0, v1); v2 = ADD(v2, v3); \
    v1 = ROT13(v1); v3 = ROT16(v3); \
    v1 = XOR(v1, v0); v3 = XOR(v3, v2); \
    v0 = ROT32(v0); v2 = ADD(v2, v1); \
    v0 = ADD(v0, v3); v1 = ROT17(v1); \
    v3 = ROT21(v3); \
    v1 = XOR(v1, v2); v3 = XOR(v3, v0); \
    v2 = ROT32(v2); \
  } while (0)

typedef struct siphash_state {
    __m256i v0, v1, v2, v3;
} siphash_state;

#ifdef __cplusplus
extern "C" {
#endif

HASHX_PRIVATE uint64_t hashx_siphash13_ctr(uint64_t input, const siphash_state* keys) {
    __m256i v0 = _mm256_set1_epi64x(keys->v0);
    __m256i v1 = _mm256_set1_epi64x(keys->v1);
    __m256i v2 = _mm256_set1_epi64x(keys->v2);
    __m256i v3 = _mm256_set1_epi64x(keys->v3);
    __m256i packet = _mm256_set1_epi64x(input);

    v3 = XOR(v3, packet);
    SIPROUNDXN(v0, v1, v2, v3);
    SIPROUNDXN(v0, v1, v2, v3);
    v0 = XOR(v0, packet);
    v2 = XOR(v2, _mm256_set1_epi64x(0xffLL));
    SIPROUNDXN(v0, v1, v2, v3);
    SIPROUNDXN(v0, v1, v2, v3);
    SIPROUNDXN(v0, v1, v2, v3);
    SIPROUNDXN(v0, v1, v2, v3);

    __m256i result = XOR(XOR(v0, v1), XOR(v2, v3));
    return _mm256_extract_epi64(result, 0); // Returning the first 64-bit value as the hash result
}

__device__ HASHX_PRIVATE void hashx_siphash24_ctr_state512(const siphash_state* keys, uint64_t input, uint64_t state_out[8]) {
    __m256i v0 = _mm256_set1_epi64x(keys->v0);
    __m256i v1 = _mm256_set1_epi64x(keys->v1);
    __m256i v2 = _mm256_set1_epi64x(keys->v2);
    __m256i v3 = _mm256_set1_epi64x(keys->v3);
    __m256i packet = _mm256_set1_epi64x(input);

    v3 = XOR(v3, packet);
    SIPROUNDXN(v0, v1, v2, v3);
    SIPROUNDXN(v0, v1, v2, v3);
    v0 = XOR(v0, packet);
    v2 = XOR(v2, _mm256_set1_epi64x(0xffLL));
    SIPROUNDXN(v0, v1, v2, v3);
    SIPROUNDXN(v0, v1, v2, v3);
    SIPROUNDXN(v0, v1, v2, v3);
    SIPROUNDXN(v0, v1, v2, v3);

    __m256i result = XOR(XOR(v0, v1), XOR(v2, v3));
    _mm256_storeu_si256((__m256i*)state_out, result); // Store the result in state_out
}

#ifdef __cplusplus
}
#endif

#endif // SIPHASH_H
