#ifndef SOLVER_HEAP_H
#define SOLVER_HEAP_H

#include <stdint.h>
#include <immintrin.h>  // Header for SIMD intrinsics
#include "../include/equix.h"

#define INDEX_SPACE (UINT32_C(1) << 16)
#define NUM_COARSE_BUCKETS 256
#define NUM_FINE_BUCKETS 128
#define COARSE_BUCKET_ITEMS 336
#define FINE_BUCKET_ITEMS 12

/* Use aligned memory for better cache performance */
#define CACHE_LINE_SIZE 64

typedef uint16_t fine_item;

typedef struct __attribute__((aligned(CACHE_LINE_SIZE))) fine_bucket {
    fine_item items[FINE_BUCKET_ITEMS];
} fine_bucket;

typedef struct __attribute__((aligned(CACHE_LINE_SIZE))) fine_hashtab {
    uint8_t counts[NUM_FINE_BUCKETS];
    fine_bucket buckets[NUM_FINE_BUCKETS];
} fine_hashtab;

typedef equix_idx stage1_idx_item;  /* 16 bits */
typedef uint64_t stage1_data_item;  /* 52 bits */

typedef struct __attribute__((aligned(CACHE_LINE_SIZE))) stage1_idx_bucket {
    stage1_idx_item items[COARSE_BUCKET_ITEMS];
} stage1_idx_bucket;

typedef struct __attribute__((aligned(CACHE_LINE_SIZE))) stage1_data_bucket {
    stage1_data_item items[COARSE_BUCKET_ITEMS];
} stage1_data_bucket;

typedef struct __attribute__((aligned(CACHE_LINE_SIZE))) stage1_idx_hashtab {
    uint16_t counts[NUM_COARSE_BUCKETS];
    stage1_idx_bucket buckets[NUM_COARSE_BUCKETS];
} stage1_idx_hashtab;

typedef struct __attribute__((aligned(CACHE_LINE_SIZE))) stage1_data_hashtab {
    stage1_data_bucket buckets[NUM_COARSE_BUCKETS];
} stage1_data_hashtab;

typedef uint32_t stage2_idx_item; /* 26 bits: 8 bits = left bucket index
                                              9 bits = left item index
                                              9 bits = right item index */

typedef struct __attribute__((aligned(CACHE_LINE_SIZE))) stage2_idx_bucket {
    stage2_idx_item items[COARSE_BUCKET_ITEMS];
} stage2_idx_bucket;

typedef struct __attribute__((aligned(CACHE_LINE_SIZE))) stage2_idx_hashtab {
    uint16_t counts[NUM_COARSE_BUCKETS];
    stage2_idx_bucket buckets[NUM_COARSE_BUCKETS];
} stage2_idx_hashtab;

#ifdef SOLVER_PACKED_STAGE2
#pragma pack(push, 1)
typedef struct stage2_data_item {
    uint32_t upper : 22;
    uint8_t middle : 8;
    uint8_t lower : 7;
} stage2_data_item;
#pragma pack(pop)
#else
typedef uint64_t stage2_data_item;  /* 37 bits */
#endif

typedef struct __attribute__((aligned(CACHE_LINE_SIZE))) stage2_data_bucket {
    stage2_data_item items[COARSE_BUCKET_ITEMS];
} stage2_data_bucket;

typedef struct __attribute__((aligned(CACHE_LINE_SIZE))) stage2_data_hashtab {
    stage2_data_bucket buckets[NUM_COARSE_BUCKETS];
} stage2_data_hashtab;

typedef uint32_t stage3_data_item;  /* 22 bits */

typedef struct __attribute__((aligned(CACHE_LINE_SIZE))) stage3_data_bucket {
    stage3_data_item items[COARSE_BUCKET_ITEMS];
} stage3_data_bucket;

typedef struct __attribute__((aligned(CACHE_LINE_SIZE))) stage3_data_hashtab {
    stage3_data_bucket buckets[NUM_COARSE_BUCKETS];
} stage3_data_hashtab;

typedef stage2_idx_hashtab stage3_idx_hashtab;
typedef stage2_idx_item stage3_idx_item;

typedef struct __attribute__((aligned(CACHE_LINE_SIZE))) solver_heap {
    stage1_idx_hashtab stage1_indices;           /* 172,544 bytes */
    stage2_idx_hashtab stage2_indices;           /* 344,576 bytes */
    stage2_data_hashtab stage2_data;             /* 688,128 bytes */
    union {
        stage1_data_hashtab stage1_data;         /* 688,128 bytes */
        struct {
            stage3_idx_hashtab stage3_indices;   /* 344,576 bytes */
            stage3_data_hashtab stage3_data;     /* 344,064 bytes */
        };
    };
    fine_hashtab scratch_ht;                     /* 3,200 bytes */
} solver_heap;                                   /* TOTAL: 1,897,088 bytes */

/* Optimized functions with SIMD and loop unrolling */

/* Function to process stage1 index bucket using SIMD */
static inline void process_stage1_bucket_simd(stage1_idx_bucket* bucket) {
    __m256i* data = (__m256i*)bucket->items;
    
    for (int i = 0; i < COARSE_BUCKET_ITEMS / (sizeof(__m256i) / sizeof(stage1_idx_item)); ++i) {
        __m256i vec = _mm256_load_si256(&data[i]);
        vec = _mm256_add_epi16(vec, _mm256_set1_epi16(1));  // Example operation: add 1 to each element
        _mm256_store_si256(&data[i], vec);
    }
}

/* Function to process stage2 index bucket with loop unrolling */
static inline void process_stage2_bucket_unrolled(stage2_idx_bucket* bucket) {
    stage2_idx_item* items = bucket->items;
    
    for (int i = 0; i < COARSE_BUCKET_ITEMS; i += 4) {
        items[i] += 1;
        items[i+1] += 1;
        items[i+2] += 1;
        items[i+3] += 1;
    }
}

/* Optimized processing for stage1 data bucket */
static inline void process_stage1_data_bucket(stage1_data_bucket* bucket) {
    __m256i* data = (__m256i*)bucket->items;
    
    for (int i = 0; i < COARSE_BUCKET_ITEMS / (sizeof(__m256i) / sizeof(stage1_data_item)); ++i) {
        __m256i vec = _mm256_load_si256(&data[i]);
        vec = _mm256_add_epi64(vec, _mm256_set1_epi64x(1));  // Example operation: add 1 to each element
        _mm256_store_si256(&data[i], vec);
    }
}

/* Optimized processing for stage3 data bucket */
static inline void process_stage3_bucket(stage3_data_bucket* bucket) {
    stage3_data_item* items = bucket->items;
    
    for (int i = 0; i < COARSE_BUCKET_ITEMS; ++i) {
        items[i] += 1;  // Simple example operation
    }
}

#endif /* SOLVER_HEAP_H */
