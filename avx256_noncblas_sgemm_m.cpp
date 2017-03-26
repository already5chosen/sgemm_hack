#include <string.h>
#include <stdio.h>
#include <intrin.h>
#include <algorithm>

#define func_name avx256_noncblas_sgemm_m
#define MM_FMADD(a, b, c) _mm256_add_ps(_mm256_mul_ps((a),(b)), (c))

typedef float   scalar_t;
typedef __m256  fp_vector_t;
typedef __m256i int_vector_t;

#define MM_BROADCAST_Sx(a)           _mm256_broadcast_ss((a))
#define MM_MUL_Px(a, b)              _mm256_mul_ps((a),(b))
#define MM_STOREU_Px(a, b)           _mm256_storeu_ps((a),(b))
#define MM_LOADU_Px(a)               _mm256_loadu_ps((a))
#define MM_MASKSTOREU_Px(a, mask, b) _mm256_maskstore_ps((a),(mask),(b))
#define MM_MASKLOADU_Px(a, mask)     _mm256_maskload_ps((a),(mask))


enum {
 bb_nRows = 53,
};

#include "avxnnn_noncblas_sgemm_m.cpp"
