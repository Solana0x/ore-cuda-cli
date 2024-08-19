#include "siphash.h"
#include "hashx_endian.h"
#include "unreachable.h"

__device__ __forceinline__ uint64_t hashx_siphash13_ctr(uint64_t input, const siphash_state* keys) {
  uint64_t v0 = keys->v0;
  uint64_t v1 = keys->v1;
  uint64_t v2 = keys->v2;
  uint64_t v3 = keys->v3;

  v3 ^= input;

  SIPROUND(v0, v1, v2, v3);

  v0 ^= input;
  v2 ^= 0xff;

  SIPROUND(v0, v1, v2, v3);
  SIPROUND(v0, v1, v2, v3);
  SIPROUND(v0, v1, v2, v3);

  return (v0 ^ v1) ^ (v2 ^ v3);
}

__device__ void hashx_siphash24_ctr_state512(const siphash_state* keys, uint64_t input,
                                             uint64_t state_out[8]) {
  uint64_t v0 = keys->v0;
  uint64_t v1 = keys->v1;
  uint64_t v2 = keys->v2;
  uint64_t v3 = keys->v3;

  const uint64_t c1 = 0xee;
  const uint64_t c2 = 0xdd;

  v1 ^= c1;
  v3 ^= input;

  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    SIPROUND(v0, v1, v2, v3);
  }

  v0 ^= input;
  v2 ^= c1;

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    SIPROUND(v0, v1, v2, v3);
  }

  state_out[0] = v0;
  state_out[1] = v1;
  state_out[2] = v2;
  state_out[3] = v3;

  v1 ^= c2;

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    SIPROUND(v0, v1, v2, v3);
  }

  state_out[4] = v0;
  state_out[5] = v1;
  state_out[6] = v2;
  state_out[7] = v3;
}
