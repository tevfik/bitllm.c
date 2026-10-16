/**
 * @file eif_quantize_bitnet.c
 * @brief BitNet b1.58 Ternary Quantization & Multiplication-Free Matrix Kernels.
 */

#include "eif_quantize_bitnet.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <alloca.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

int8_t eif_unpack_ternary(uint8_t byte, int pos)
{
    uint8_t code = (uint8_t)((byte >> (pos * 2)) & 0x03u);
    if (code == 1u) return 1;
    if (code == 2u) return -1;
    return 0;
}

size_t eif_pack_ternary(const int8_t *ternary, uint8_t *packed, size_t count)
{
    size_t num_bytes = (count + 3u) / 4u;
    for (size_t b = 0; b < num_bytes; b++) {
        uint8_t byte_val = 0;
        for (int pos = 0; pos < 4; pos++) {
            size_t idx = b * 4u + (size_t)pos;
            uint8_t code = 0;
            if (idx < count) {
                int8_t val = ternary[idx];
                if (val > 0) code = 1u;
                else if (val < 0) code = 2u;
            }
            byte_val |= (uint8_t)(code << (pos * 2));
        }
        packed[b] = byte_val;
    }
    return num_bytes;
}

/* Precomputed fast 2-bit decoding table: 256 bytes * 4 float weights (4 KB in L1 cache) */
static float s_byte_weights[256][4] __attribute__((aligned(64)));
static int8_t s_byte_weights_i8[256][4] __attribute__((aligned(64)));
static int s_byte_weights_init = 0;

static void ensure_byte_weights(void)
{
    if (s_byte_weights_init)
        return;
    for (int b = 0; b < 256; b++) {
        for (int pos = 0; pos < 4; pos++) {
            uint8_t code = (uint8_t)((b >> (pos * 2)) & 0x03u);
            if (code == 1u) {
                s_byte_weights[b][pos] = 1.0f;
                s_byte_weights_i8[b][pos] = 1;
            } else if (code == 2u) {
                s_byte_weights[b][pos] = -1.0f;
                s_byte_weights_i8[b][pos] = -1;
            } else {
                s_byte_weights[b][pos] = 0.0f;
                s_byte_weights_i8[b][pos] = 0;
            }
        }
    }
    s_byte_weights_init = 1;
}

/* Precomputed fast base-3 LUT decoding table: 256 bytes * 5 weights (int8 and float) */
static int8_t s_dense_lut[256][5];
static float s_dense_lut_f32[256][5];
static int s_dense_lut_init = 0;

static void ensure_dense_lut(void)
{
    if (s_dense_lut_init)
        return;
    for (int b = 0; b < 256; b++) {
        int temp = b;
        for (int i = 0; i < 5; i++) {
            int d = temp % 3;
            temp /= 3;
            int8_t val = (d == 1) ? 1 : ((d == 2) ? -1 : 0);
            s_dense_lut[b][i] = val;
            s_dense_lut_f32[b][i] = (float)val;
        }
    }
    s_dense_lut_init = 1;
}

size_t eif_pack_ternary_dense(const int8_t *ternary, uint8_t *packed, size_t count)
{
    size_t num_bytes = (count + 4u) / 5u;
    for (size_t b = 0; b < num_bytes; b++) {
        uint8_t byte_val = 0;
        uint8_t multiplier = 1;
        for (int pos = 0; pos < 5; pos++) {
            size_t idx = b * 5u + (size_t)pos;
            uint8_t code = 0;
            if (idx < count) {
                int8_t val = ternary[idx];
                if (val > 0) code = 1u;
                else if (val < 0) code = 2u;
            }
            byte_val += (uint8_t)(code * multiplier);
            multiplier = (uint8_t)(multiplier * 3u);
        }
        packed[b] = byte_val;
    }
    return num_bytes;
}

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define EIF_HAS_AVX2 1

#define AVX2_HSUM(acc, s) do { \
    __m128 l = _mm256_castps256_ps128(acc); \
    __m128 h = _mm256_extractf128_ps(acc, 1); \
    __m128 sm = _mm_add_ps(l, h); \
    sm = _mm_hadd_ps(sm, sm); \
    sm = _mm_hadd_ps(sm, sm); \
    s = _mm_cvtss_f32(sm); \
} while(0)

__attribute__((target("avx2,fma")))
static inline void compute_8rows_avx2(const uint8_t *w0, const uint8_t *w1,
                                     const uint8_t *w2, const uint8_t *w3,
                                     const uint8_t *w4, const uint8_t *w5,
                                     const uint8_t *w6, const uint8_t *w7,
                                     const float *input,
                                     float *s0, float *s1, float *s2, float *s3,
                                     float *s4, float *s5, float *s6, float *s7,
                                     int cols)
{
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps();
    __m256 a4 = _mm256_setzero_ps();
    __m256 a5 = _mm256_setzero_ps();
    __m256 a6 = _mm256_setzero_ps();
    __m256 a7 = _mm256_setzero_ps();

    int col = 0;
    int byte_idx = 0;

    for (; col <= cols - 8; col += 8, byte_idx += 2) {
        __m256 in8 = _mm256_loadu_ps(&input[col]);

        __m256 wv0 = _mm256_set_m128(_mm_loadu_ps(s_byte_weights[w0[byte_idx + 1]]), _mm_loadu_ps(s_byte_weights[w0[byte_idx]]));
        __m256 wv1 = _mm256_set_m128(_mm_loadu_ps(s_byte_weights[w1[byte_idx + 1]]), _mm_loadu_ps(s_byte_weights[w1[byte_idx]]));
        __m256 wv2 = _mm256_set_m128(_mm_loadu_ps(s_byte_weights[w2[byte_idx + 1]]), _mm_loadu_ps(s_byte_weights[w2[byte_idx]]));
        __m256 wv3 = _mm256_set_m128(_mm_loadu_ps(s_byte_weights[w3[byte_idx + 1]]), _mm_loadu_ps(s_byte_weights[w3[byte_idx]]));
        __m256 wv4 = _mm256_set_m128(_mm_loadu_ps(s_byte_weights[w4[byte_idx + 1]]), _mm_loadu_ps(s_byte_weights[w4[byte_idx]]));
        __m256 wv5 = _mm256_set_m128(_mm_loadu_ps(s_byte_weights[w5[byte_idx + 1]]), _mm_loadu_ps(s_byte_weights[w5[byte_idx]]));
        __m256 wv6 = _mm256_set_m128(_mm_loadu_ps(s_byte_weights[w6[byte_idx + 1]]), _mm_loadu_ps(s_byte_weights[w6[byte_idx]]));
        __m256 wv7 = _mm256_set_m128(_mm_loadu_ps(s_byte_weights[w7[byte_idx + 1]]), _mm_loadu_ps(s_byte_weights[w7[byte_idx]]));

        a0 = _mm256_fmadd_ps(in8, wv0, a0);
        a1 = _mm256_fmadd_ps(in8, wv1, a1);
        a2 = _mm256_fmadd_ps(in8, wv2, a2);
        a3 = _mm256_fmadd_ps(in8, wv3, a3);
        a4 = _mm256_fmadd_ps(in8, wv4, a4);
        a5 = _mm256_fmadd_ps(in8, wv5, a5);
        a6 = _mm256_fmadd_ps(in8, wv6, a6);
        a7 = _mm256_fmadd_ps(in8, wv7, a7);
    }

    float sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;
    AVX2_HSUM(a0, sum0);
    AVX2_HSUM(a1, sum1);
    AVX2_HSUM(a2, sum2);
    AVX2_HSUM(a3, sum3);
    AVX2_HSUM(a4, sum4);
    AVX2_HSUM(a5, sum5);
    AVX2_HSUM(a6, sum6);
    AVX2_HSUM(a7, sum7);

    for (int c = col; c < cols; c++) {
        int b_idx = c / 4;
        int pos = c % 4;
        uint8_t c0 = (uint8_t)((w0[b_idx] >> (pos * 2)) & 0x03u);
        if (c0 == 1u) sum0 += input[c]; else if (c0 == 2u) sum0 -= input[c];
        uint8_t c1 = (uint8_t)((w1[b_idx] >> (pos * 2)) & 0x03u);
        if (c1 == 1u) sum1 += input[c]; else if (c1 == 2u) sum1 -= input[c];
        uint8_t c2 = (uint8_t)((w2[b_idx] >> (pos * 2)) & 0x03u);
        if (c2 == 1u) sum2 += input[c]; else if (c2 == 2u) sum2 -= input[c];
        uint8_t c3 = (uint8_t)((w3[b_idx] >> (pos * 2)) & 0x03u);
        if (c3 == 1u) sum3 += input[c]; else if (c3 == 2u) sum3 -= input[c];
        uint8_t c4 = (uint8_t)((w4[b_idx] >> (pos * 2)) & 0x03u);
        if (c4 == 1u) sum4 += input[c]; else if (c4 == 2u) sum4 -= input[c];
        uint8_t c5 = (uint8_t)((w5[b_idx] >> (pos * 2)) & 0x03u);
        if (c5 == 1u) sum5 += input[c]; else if (c5 == 2u) sum5 -= input[c];
        uint8_t c6 = (uint8_t)((w6[b_idx] >> (pos * 2)) & 0x03u);
        if (c6 == 1u) sum6 += input[c]; else if (c6 == 2u) sum6 -= input[c];
        uint8_t c7 = (uint8_t)((w7[b_idx] >> (pos * 2)) & 0x03u);
        if (c7 == 1u) sum7 += input[c]; else if (c7 == 2u) sum7 -= input[c];
    }

    *s0 = sum0; *s1 = sum1; *s2 = sum2; *s3 = sum3;
    *s4 = sum4; *s5 = sum5; *s6 = sum6; *s7 = sum7;
}

__attribute__((target("avx2,fma")))
static inline void compute_4rows_avx2(const uint8_t *w0, const uint8_t *w1,
                                     const uint8_t *w2, const uint8_t *w3,
                                     const float *input,
                                     float *s0, float *s1, float *s2, float *s3,
                                     int cols)
{
    __m256 a0_0 = _mm256_setzero_ps(), a0_1 = _mm256_setzero_ps();
    __m256 a1_0 = _mm256_setzero_ps(), a1_1 = _mm256_setzero_ps();
    __m256 a2_0 = _mm256_setzero_ps(), a2_1 = _mm256_setzero_ps();
    __m256 a3_0 = _mm256_setzero_ps(), a3_1 = _mm256_setzero_ps();

    int col = 0;
    int byte_idx = 0;

    for (; col <= cols - 16; col += 16, byte_idx += 4) {
        __m256 in0 = _mm256_loadu_ps(&input[col]);
        __m256 in1 = _mm256_loadu_ps(&input[col + 8]);

        __m256 wv0_0 = _mm256_set_m128(_mm_loadu_ps(s_byte_weights[w0[byte_idx + 1]]), _mm_loadu_ps(s_byte_weights[w0[byte_idx]]));
        __m256 wv0_1 = _mm256_set_m128(_mm_loadu_ps(s_byte_weights[w0[byte_idx + 3]]), _mm_loadu_ps(s_byte_weights[w0[byte_idx + 2]]));

        __m256 wv1_0 = _mm256_set_m128(_mm_loadu_ps(s_byte_weights[w1[byte_idx + 1]]), _mm_loadu_ps(s_byte_weights[w1[byte_idx]]));
        __m256 wv1_1 = _mm256_set_m128(_mm_loadu_ps(s_byte_weights[w1[byte_idx + 3]]), _mm_loadu_ps(s_byte_weights[w1[byte_idx + 2]]));

        __m256 wv2_0 = _mm256_set_m128(_mm_loadu_ps(s_byte_weights[w2[byte_idx + 1]]), _mm_loadu_ps(s_byte_weights[w2[byte_idx]]));
        __m256 wv2_1 = _mm256_set_m128(_mm_loadu_ps(s_byte_weights[w2[byte_idx + 3]]), _mm_loadu_ps(s_byte_weights[w2[byte_idx + 2]]));

        __m256 wv3_0 = _mm256_set_m128(_mm_loadu_ps(s_byte_weights[w3[byte_idx + 1]]), _mm_loadu_ps(s_byte_weights[w3[byte_idx]]));
        __m256 wv3_1 = _mm256_set_m128(_mm_loadu_ps(s_byte_weights[w3[byte_idx + 3]]), _mm_loadu_ps(s_byte_weights[w3[byte_idx + 2]]));

        a0_0 = _mm256_fmadd_ps(in0, wv0_0, a0_0);
        a0_1 = _mm256_fmadd_ps(in1, wv0_1, a0_1);

        a1_0 = _mm256_fmadd_ps(in0, wv1_0, a1_0);
        a1_1 = _mm256_fmadd_ps(in1, wv1_1, a1_1);

        a2_0 = _mm256_fmadd_ps(in0, wv2_0, a2_0);
        a2_1 = _mm256_fmadd_ps(in1, wv2_1, a2_1);

        a3_0 = _mm256_fmadd_ps(in0, wv3_0, a3_0);
        a3_1 = _mm256_fmadd_ps(in1, wv3_1, a3_1);
    }

    __m256 acc0 = _mm256_add_ps(a0_0, a0_1);
    __m256 acc1 = _mm256_add_ps(a1_0, a1_1);
    __m256 acc2 = _mm256_add_ps(a2_0, a2_1);
    __m256 acc3 = _mm256_add_ps(a3_0, a3_1);

    for (; col <= cols - 8; col += 8, byte_idx += 2) {
        __m256 in8 = _mm256_loadu_ps(&input[col]);
        __m256 wv0 = _mm256_set_m128(_mm_loadu_ps(s_byte_weights[w0[byte_idx + 1]]), _mm_loadu_ps(s_byte_weights[w0[byte_idx]]));
        __m256 wv1 = _mm256_set_m128(_mm_loadu_ps(s_byte_weights[w1[byte_idx + 1]]), _mm_loadu_ps(s_byte_weights[w1[byte_idx]]));
        __m256 wv2 = _mm256_set_m128(_mm_loadu_ps(s_byte_weights[w2[byte_idx + 1]]), _mm_loadu_ps(s_byte_weights[w2[byte_idx]]));
        __m256 wv3 = _mm256_set_m128(_mm_loadu_ps(s_byte_weights[w3[byte_idx + 1]]), _mm_loadu_ps(s_byte_weights[w3[byte_idx]]));

        acc0 = _mm256_fmadd_ps(in8, wv0, acc0);
        acc1 = _mm256_fmadd_ps(in8, wv1, acc1);
        acc2 = _mm256_fmadd_ps(in8, wv2, acc2);
        acc3 = _mm256_fmadd_ps(in8, wv3, acc3);
    }

    float sum0, sum1, sum2, sum3;
    AVX2_HSUM(acc0, sum0);
    AVX2_HSUM(acc1, sum1);
    AVX2_HSUM(acc2, sum2);
    AVX2_HSUM(acc3, sum3);

    for (int c = col; c < cols; c++) {
        int b_idx = c / 4;
        int pos = c % 4;
        uint8_t c0 = (uint8_t)((w0[b_idx] >> (pos * 2)) & 0x03u);
        if (c0 == 1u) sum0 += input[c]; else if (c0 == 2u) sum0 -= input[c];
        uint8_t c1 = (uint8_t)((w1[b_idx] >> (pos * 2)) & 0x03u);
        if (c1 == 1u) sum1 += input[c]; else if (c1 == 2u) sum1 -= input[c];
        uint8_t c2 = (uint8_t)((w2[b_idx] >> (pos * 2)) & 0x03u);
        if (c2 == 1u) sum2 += input[c]; else if (c2 == 2u) sum2 -= input[c];
        uint8_t c3 = (uint8_t)((w3[b_idx] >> (pos * 2)) & 0x03u);
        if (c3 == 1u) sum3 += input[c]; else if (c3 == 2u) sum3 -= input[c];
    }

    *s0 = sum0; *s1 = sum1; *s2 = sum2; *s3 = sum3;
}

__attribute__((target("avx2,fma")))
static void matmul_bitnet_f32_avx2(const uint8_t *weights,
                                   const float *scales,
                                   const float *input,
                                   const float *bias,
                                   float *output,
                                   int rows,
                                   int cols)
{
    ensure_byte_weights();
    const int row_bytes = (cols + 3) / 4;

    /* Primary 8-Row Register Tiling for peak AVX2 throughput */
    #pragma omp parallel for schedule(static) if((int64_t)rows * cols >= 16384)
    for (int row = 0; row < (rows & ~7); row += 8) {
        const uint8_t *w0 = weights + (row * row_bytes);
        const uint8_t *w1 = weights + ((row + 1) * row_bytes);
        const uint8_t *w2 = weights + ((row + 2) * row_bytes);
        const uint8_t *w3 = weights + ((row + 3) * row_bytes);
        const uint8_t *w4 = weights + ((row + 4) * row_bytes);
        const uint8_t *w5 = weights + ((row + 5) * row_bytes);
        const uint8_t *w6 = weights + ((row + 6) * row_bytes);
        const uint8_t *w7 = weights + ((row + 7) * row_bytes);

        float s0, s1, s2, s3, s4, s5, s6, s7;
        compute_8rows_avx2(w0, w1, w2, w3, w4, w5, w6, w7, input,
                           &s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7, cols);

        output[row]     = s0 * (scales ? scales[row] : 1.0f)     + (bias ? bias[row] : 0.0f);
        output[row + 1] = s1 * (scales ? scales[row + 1] : 1.0f) + (bias ? bias[row + 1] : 0.0f);
        output[row + 2] = s2 * (scales ? scales[row + 2] : 1.0f) + (bias ? bias[row + 2] : 0.0f);
        output[row + 3] = s3 * (scales ? scales[row + 3] : 1.0f) + (bias ? bias[row + 3] : 0.0f);
        output[row + 4] = s4 * (scales ? scales[row + 4] : 1.0f) + (bias ? bias[row + 4] : 0.0f);
        output[row + 5] = s5 * (scales ? scales[row + 5] : 1.0f) + (bias ? bias[row + 5] : 0.0f);
        output[row + 6] = s6 * (scales ? scales[row + 6] : 1.0f) + (bias ? bias[row + 6] : 0.0f);
        output[row + 7] = s7 * (scales ? scales[row + 7] : 1.0f) + (bias ? bias[row + 7] : 0.0f);
    }

    /* 4-row fallback for rows % 8 */
    for (int row = (rows & ~7); row < (rows & ~3); row += 4) {
        const uint8_t *w0 = weights + (row * row_bytes);
        const uint8_t *w1 = weights + ((row + 1) * row_bytes);
        const uint8_t *w2 = weights + ((row + 2) * row_bytes);
        const uint8_t *w3 = weights + ((row + 3) * row_bytes);

        float s0, s1, s2, s3;
        compute_4rows_avx2(w0, w1, w2, w3, input, &s0, &s1, &s2, &s3, cols);

        output[row]     = s0 * (scales ? scales[row] : 1.0f)     + (bias ? bias[row] : 0.0f);
        output[row + 1] = s1 * (scales ? scales[row + 1] : 1.0f) + (bias ? bias[row + 1] : 0.0f);
        output[row + 2] = s2 * (scales ? scales[row + 2] : 1.0f) + (bias ? bias[row + 2] : 0.0f);
        output[row + 3] = s3 * (scales ? scales[row + 3] : 1.0f) + (bias ? bias[row + 3] : 0.0f);
    }

    /* Single-row cleanup for rows % 4 remainder */
    for (int row = (rows & ~3); row < rows; row++) {
        const uint8_t *row_w = weights + (row * row_bytes);
        __m256 acc = _mm256_setzero_ps();
        int col = 0;
        int byte_idx = 0;

        for (; col <= cols - 8; col += 8, byte_idx += 2) {
            uint8_t b0 = row_w[byte_idx];
            uint8_t b1 = row_w[byte_idx + 1];
            __m128 w_low = _mm_loadu_ps(s_byte_weights[b0]);
            __m128 w_high = _mm_loadu_ps(s_byte_weights[b1]);
            __m256 w8 = _mm256_set_m128(w_high, w_low);
            __m256 in8 = _mm256_loadu_ps(&input[col]);
            acc = _mm256_fmadd_ps(in8, w8, acc);
        }

        float s_row;
        AVX2_HSUM(acc, s_row);

        for (; col < cols; col++) {
            int b_idx = col / 4;
            int pos = col % 4;
            uint8_t code = (uint8_t)((row_w[b_idx] >> (pos * 2)) & 0x03u);
            if (code == 1u) s_row += input[col];
            else if (code == 2u) s_row -= input[col];
        }

        float scale = scales ? scales[row] : 1.0f;
        output[row] = s_row * scale + (bias ? bias[row] : 0.0f);
    }
}
#elif defined(__ARM_NEON) && !defined(__aarch64__)
#include <arm_neon.h>
#define EIF_HAS_NEON 1

static void matmul_bitnet_f32_neon(const uint8_t *weights,
                                   const float *scales,
                                   const float *input,
                                   const float *bias,
                                   float *output,
                                   int rows,
                                   int cols)
{
    ensure_byte_weights();
    const int row_bytes = (cols + 3) / 4;

    /* Primary 8-Row Register Tiling for ARM NEON */
    #pragma omp parallel for schedule(static) if((int64_t)rows * cols >= 16384)
    for (int row = 0; row < (rows & ~7); row += 8) {
        const uint8_t *w0 = weights + (row * row_bytes);
        const uint8_t *w1 = weights + ((row + 1) * row_bytes);
        const uint8_t *w2 = weights + ((row + 2) * row_bytes);
        const uint8_t *w3 = weights + ((row + 3) * row_bytes);
        const uint8_t *w4 = weights + ((row + 4) * row_bytes);
        const uint8_t *w5 = weights + ((row + 5) * row_bytes);
        const uint8_t *w6 = weights + ((row + 6) * row_bytes);
        const uint8_t *w7 = weights + ((row + 7) * row_bytes);

        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
        float32x4_t a4 = vdupq_n_f32(0.0f), a5 = vdupq_n_f32(0.0f);
        float32x4_t a6 = vdupq_n_f32(0.0f), a7 = vdupq_n_f32(0.0f);

        int col = 0;
        int byte_idx = 0;

        for (; col <= cols - 4; col += 4, byte_idx++) {
            float32x4_t in = vld1q_f32(&input[col]);
            float32x4_t wv0 = vld1q_f32(s_byte_weights[w0[byte_idx]]);
            float32x4_t wv1 = vld1q_f32(s_byte_weights[w1[byte_idx]]);
            float32x4_t wv2 = vld1q_f32(s_byte_weights[w2[byte_idx]]);
            float32x4_t wv3 = vld1q_f32(s_byte_weights[w3[byte_idx]]);
            float32x4_t wv4 = vld1q_f32(s_byte_weights[w4[byte_idx]]);
            float32x4_t wv5 = vld1q_f32(s_byte_weights[w5[byte_idx]]);
            float32x4_t wv6 = vld1q_f32(s_byte_weights[w6[byte_idx]]);
            float32x4_t wv7 = vld1q_f32(s_byte_weights[w7[byte_idx]]);

            #if defined(__ARM_FEATURE_FMA) || defined(__aarch64__)
            a0 = vfmaq_f32(a0, in, wv0); a1 = vfmaq_f32(a1, in, wv1);
            a2 = vfmaq_f32(a2, in, wv2); a3 = vfmaq_f32(a3, in, wv3);
            a4 = vfmaq_f32(a4, in, wv4); a5 = vfmaq_f32(a5, in, wv5);
            a6 = vfmaq_f32(a6, in, wv6); a7 = vfmaq_f32(a7, in, wv7);
            #else
            a0 = vaddq_f32(a0, vmulq_f32(in, wv0)); a1 = vaddq_f32(a1, vmulq_f32(in, wv1));
            a2 = vaddq_f32(a2, vmulq_f32(in, wv2)); a3 = vaddq_f32(a3, vmulq_f32(in, wv3));
            a4 = vaddq_f32(a4, vmulq_f32(in, wv4)); a5 = vaddq_f32(a5, vmulq_f32(in, wv5));
            a6 = vaddq_f32(a6, vmulq_f32(in, wv6)); a7 = vaddq_f32(a7, vmulq_f32(in, wv7));
            #endif
        }

        #if defined(__aarch64__)
        float s0 = vaddvq_f32(a0), s1 = vaddvq_f32(a1);
        float s2 = vaddvq_f32(a2), s3 = vaddvq_f32(a3);
        float s4 = vaddvq_f32(a4), s5 = vaddvq_f32(a5);
        float s6 = vaddvq_f32(a6), s7 = vaddvq_f32(a7);
        #else
        #define NEON_HSUM(acc, s) do { \
            float32x2_t low = vget_low_f32(acc); \
            float32x2_t high = vget_high_f32(acc); \
            float32x2_t pair = vpadd_f32(low, high); \
            pair = vpadd_f32(pair, pair); \
            s = vget_lane_f32(pair, 0); \
        } while(0)
        float s0, s1, s2, s3, s4, s5, s6, s7;
        NEON_HSUM(a0, s0); NEON_HSUM(a1, s1); NEON_HSUM(a2, s2); NEON_HSUM(a3, s3);
        NEON_HSUM(a4, s4); NEON_HSUM(a5, s5); NEON_HSUM(a6, s6); NEON_HSUM(a7, s7);
        #endif

        for (int c = col; c < cols; c++) {
            int b_idx = c / 4;
            int pos = c % 4;
            uint8_t c0 = (uint8_t)((w0[b_idx] >> (pos * 2)) & 0x03u);
            if (c0 == 1u) s0 += input[c]; else if (c0 == 2u) s0 -= input[c];
            uint8_t c1 = (uint8_t)((w1[b_idx] >> (pos * 2)) & 0x03u);
            if (c1 == 1u) s1 += input[c]; else if (c1 == 2u) s1 -= input[c];
            uint8_t c2 = (uint8_t)((w2[b_idx] >> (pos * 2)) & 0x03u);
            if (c2 == 1u) s2 += input[c]; else if (c2 == 2u) s2 -= input[c];
            uint8_t c3 = (uint8_t)((w3[b_idx] >> (pos * 2)) & 0x03u);
            if (c3 == 1u) s3 += input[c]; else if (c3 == 2u) s3 -= input[c];
            uint8_t c4 = (uint8_t)((w4[b_idx] >> (pos * 2)) & 0x03u);
            if (c4 == 1u) s4 += input[c]; else if (c4 == 2u) s4 -= input[c];
            uint8_t c5 = (uint8_t)((w5[b_idx] >> (pos * 2)) & 0x03u);
            if (c5 == 1u) s5 += input[c]; else if (c5 == 2u) s5 -= input[c];
            uint8_t c6 = (uint8_t)((w6[b_idx] >> (pos * 2)) & 0x03u);
            if (c6 == 1u) s6 += input[c]; else if (c6 == 2u) s6 -= input[c];
            uint8_t c7 = (uint8_t)((w7[b_idx] >> (pos * 2)) & 0x03u);
            if (c7 == 1u) s7 += input[c]; else if (c7 == 2u) s7 -= input[c];
        }

        output[row]     = s0 * (scales ? scales[row] : 1.0f)     + (bias ? bias[row] : 0.0f);
        output[row + 1] = s1 * (scales ? scales[row + 1] : 1.0f) + (bias ? bias[row + 1] : 0.0f);
        output[row + 2] = s2 * (scales ? scales[row + 2] : 1.0f) + (bias ? bias[row + 2] : 0.0f);
        output[row + 3] = s3 * (scales ? scales[row + 3] : 1.0f) + (bias ? bias[row + 3] : 0.0f);
        output[row + 4] = s4 * (scales ? scales[row + 4] : 1.0f) + (bias ? bias[row + 4] : 0.0f);
        output[row + 5] = s5 * (scales ? scales[row + 5] : 1.0f) + (bias ? bias[row + 5] : 0.0f);
        output[row + 6] = s6 * (scales ? scales[row + 6] : 1.0f) + (bias ? bias[row + 6] : 0.0f);
        output[row + 7] = s7 * (scales ? scales[row + 7] : 1.0f) + (bias ? bias[row + 7] : 0.0f);
    }

    /* 4-row fallback for rows % 8 */
    for (int row = (rows & ~7); row < (rows & ~3); row += 4) {
        const uint8_t *w0 = weights + (row * row_bytes);
        const uint8_t *w1 = weights + ((row + 1) * row_bytes);
        const uint8_t *w2 = weights + ((row + 2) * row_bytes);
        const uint8_t *w3 = weights + ((row + 3) * row_bytes);

        float32x4_t a0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f);
        float32x4_t a3 = vdupq_n_f32(0.0f);

        int col = 0;
        int byte_idx = 0;

        for (; col <= cols - 4; col += 4, byte_idx++) {
            float32x4_t in = vld1q_f32(&input[col]);
            float32x4_t wv0 = vld1q_f32(s_byte_weights[w0[byte_idx]]);
            float32x4_t wv1 = vld1q_f32(s_byte_weights[w1[byte_idx]]);
            float32x4_t wv2 = vld1q_f32(s_byte_weights[w2[byte_idx]]);
            float32x4_t wv3 = vld1q_f32(s_byte_weights[w3[byte_idx]]);

            #if defined(__ARM_FEATURE_FMA) || defined(__aarch64__)
            a0 = vfmaq_f32(a0, in, wv0);
            a1 = vfmaq_f32(a1, in, wv1);
            a2 = vfmaq_f32(a2, in, wv2);
            a3 = vfmaq_f32(a3, in, wv3);
            #else
            a0 = vaddq_f32(a0, vmulq_f32(in, wv0));
            a1 = vaddq_f32(a1, vmulq_f32(in, wv1));
            a2 = vaddq_f32(a2, vmulq_f32(in, wv2));
            a3 = vaddq_f32(a3, vmulq_f32(in, wv3));
            #endif
        }

        #if defined(__aarch64__)
        float s0 = vaddvq_f32(a0);
        float s1 = vaddvq_f32(a1);
        float s2 = vaddvq_f32(a2);
        float s3 = vaddvq_f32(a3);
        #else
        float s0, s1, s2, s3;
        NEON_HSUM(a0, s0); NEON_HSUM(a1, s1); NEON_HSUM(a2, s2); NEON_HSUM(a3, s3);
        #endif

        for (int c = col; c < cols; c++) {
            int b_idx = c / 4;
            int pos = c % 4;
            uint8_t c0 = (uint8_t)((w0[b_idx] >> (pos * 2)) & 0x03u);
            if (c0 == 1u) s0 += input[c]; else if (c0 == 2u) s0 -= input[c];
            uint8_t c1 = (uint8_t)((w1[b_idx] >> (pos * 2)) & 0x03u);
            if (c1 == 1u) s1 += input[c]; else if (c1 == 2u) s1 -= input[c];
            uint8_t c2 = (uint8_t)((w2[b_idx] >> (pos * 2)) & 0x03u);
            if (c2 == 1u) s2 += input[c]; else if (c2 == 2u) s2 -= input[c];
            uint8_t c3 = (uint8_t)((w3[b_idx] >> (pos * 2)) & 0x03u);
            if (c3 == 1u) s3 += input[c]; else if (c3 == 2u) s3 -= input[c];
        }

        output[row]     = s0 * (scales ? scales[row] : 1.0f)     + (bias ? bias[row] : 0.0f);
        output[row + 1] = s1 * (scales ? scales[row + 1] : 1.0f) + (bias ? bias[row + 1] : 0.0f);
        output[row + 2] = s2 * (scales ? scales[row + 2] : 1.0f) + (bias ? bias[row + 2] : 0.0f);
        output[row + 3] = s3 * (scales ? scales[row + 3] : 1.0f) + (bias ? bias[row + 3] : 0.0f);
    }

    /* Remainder */
    for (int row = (rows & ~3); row < rows; row++) {
        const uint8_t *row_w = weights + (row * row_bytes);
        float32x4_t acc = vdupq_n_f32(0.0f);
        int col = 0;
        int byte_idx = 0;

        for (; col <= cols - 4; col += 4, byte_idx++) {
            float32x4_t in = vld1q_f32(&input[col]);
            float32x4_t w = vld1q_f32(s_byte_weights[row_w[byte_idx]]);
            #if defined(__ARM_FEATURE_FMA) || defined(__aarch64__)
            acc = vfmaq_f32(acc, in, w);
            #else
            acc = vaddq_f32(acc, vmulq_f32(in, w));
            #endif
        }

        #if defined(__aarch64__)
        float sum = vaddvq_f32(acc);
        #else
        float32x2_t low = vget_low_f32(acc);
        float32x2_t high = vget_high_f32(acc);
        float32x2_t pair = vpadd_f32(low, high);
        pair = vpadd_f32(pair, pair);
        float sum = vget_lane_f32(pair, 0);
        #endif

        for (; col < cols; col++) {
            int b_idx = col / 4;
            int pos = col % 4;
            uint8_t code = (uint8_t)((row_w[b_idx] >> (pos * 2)) & 0x03u);
            if (code == 1u) sum += input[col];
            else if (code == 2u) sum -= input[col];
        }

        float scale = scales ? scales[row] : 1.0f;
        output[row] = sum * scale + (bias ? bias[row] : 0.0f);
    }
}
#endif /* EIF_HAS_AVX2 */

/* ============================================================
 * ARM NEON — Multiplication-Free BitNet 1.58 GEMV
 *
 * BitNet 1.58 weights ∈ {-1, 0, +1}, so inner product becomes:
 *   acc += x[i]   if w == +1  (code == 01)
 *   acc -= x[i]   if w == -1  (code == 10)
 *   (skip)        if w ==  0  (code == 00)
 *
 * No floating-point multiplications in the hot path.
 * Decodes 2-bit ternary codes via bitmask, no LUT needed.
 *
 * Strategy (4-row tiling):
 *  - Process 4 output rows simultaneously to amortize input loads
 *  - For each packed byte (4 weights): extract pos/neg masks in integer domain
 *  - float32x4_t add/sub using NEON bitwise-select (VBSL/VAND)
 *
 * Cortex-A76 (RPi 5): FADD throughput 1/cycle, avoids FMLA pipeline pressure.
 * ============================================================ */
#define MAX_TMAC_GROUPS 1024
static __thread const float *s_cached_input = NULL;
static __thread int s_cached_cols = 0;
static __thread float s_act_lut[MAX_TMAC_GROUPS * 256];

#if defined(__aarch64__)
#include <arm_neon.h>
#define EIF_HAS_NEON 1

/* Horizontal sum of a float32x4_t accumulator */
static inline float neon_hsum(float32x4_t v) {
    float32x2_t lo = vget_low_f32(v);
    float32x2_t hi = vget_high_f32(v);
    float32x2_t sum2 = vadd_f32(lo, hi);
    return vget_lane_f32(vpadd_f32(sum2, sum2), 0);
}

/*
 * Decode one packed byte into NEON masks:
 *   2-bit encoding: 00=zero  01=+1  10=-1
 *   pos_mask[i] = 0xFFFFFFFF if code == 01
 *   neg_mask[i] = 0xFFFFFFFF if code == 10
 * No branch, no LUT, pure bitwise.
 */
static inline void decode_byte_masks(uint8_t byte,
                                     uint32x4_t *pos_mask,
                                     uint32x4_t *neg_mask)
{
    /* Isolate bit[0] (low) and bit[1] (high) of each 2-bit pair */
    uint32_t low  =  byte & 0x55u;  /* bits 0,2,4,6 → code0[0], code1[0], ... */
    uint32_t high = (byte >> 1) & 0x55u; /* bits 1,3,5,7 → code0[1], code1[1], ... */

    /* code==01 (+1): low bit set, high bit clear */
    /* code==10 (-1): high bit set, low bit clear */
    uint32_t pos_packed = low  & ~high; /* {0=00,1=01,0=10,0=11} at 2-bit positions */
    uint32_t neg_packed = high & ~low;

    /* Expand each isolated bit to a full-lane mask */
    /* bit positions: 0→lane0, 2→lane1, 4→lane2, 6→lane3 */
    const uint32x4_t bit_sel = {1u, 4u, 16u, 64u}; /* 1 << (2*lane) */
    *pos_mask = vceqq_u32(vandq_u32(vdupq_n_u32(pos_packed), bit_sel), bit_sel);
    *neg_mask = vceqq_u32(vandq_u32(vdupq_n_u32(neg_packed), bit_sel), bit_sel);
}

/* Process 4 output rows × 4 input columns per byte, multiplication-free */
static inline void neon_accum_4rows_1byte(
        uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3,
        float32x4_t xq,
        float32x4_t *acc0, float32x4_t *acc1,
        float32x4_t *acc2, float32x4_t *acc3)
{
    uint32x4_t pos, neg;
    float32x4_t xpos, xneg;

    decode_byte_masks(b0, &pos, &neg);
    xpos = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(xq), pos));
    xneg = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(xq), neg));
    *acc0 = vaddq_f32(vsubq_f32(*acc0, xneg), xpos);

    decode_byte_masks(b1, &pos, &neg);
    xpos = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(xq), pos));
    xneg = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(xq), neg));
    *acc1 = vaddq_f32(vsubq_f32(*acc1, xneg), xpos);

    decode_byte_masks(b2, &pos, &neg);
    xpos = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(xq), pos));
    xneg = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(xq), neg));
    *acc2 = vaddq_f32(vsubq_f32(*acc2, xneg), xpos);

    decode_byte_masks(b3, &pos, &neg);
    xpos = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(xq), pos));
    xneg = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(xq), neg));
    *acc3 = vaddq_f32(vsubq_f32(*acc3, xneg), xpos);
}

static void matmul_bitnet_f32_neon(const uint8_t *weights,
                                    const float *scales,
                                    const float *input,
                                    const float *bias,
                                    float *output,
                                    int rows, int cols)
{
    const int row_bytes = (cols + 3) / 4;

    #pragma omp parallel for schedule(static) if(!omp_in_parallel())
    for (int row = 0; row < (rows & ~3); row += 4) {
        const uint8_t *w0 = weights + (size_t)(row + 0) * row_bytes;
        const uint8_t *w1 = weights + (size_t)(row + 1) * row_bytes;
        const uint8_t *w2 = weights + (size_t)(row + 2) * row_bytes;
        const uint8_t *w3 = weights + (size_t)(row + 3) * row_bytes;

        float32x4_t acc0 = vdupq_n_f32(0.f);
        float32x4_t acc1 = vdupq_n_f32(0.f);
        float32x4_t acc2 = vdupq_n_f32(0.f);
        float32x4_t acc3 = vdupq_n_f32(0.f);

        int col = 0, bidx = 0;

        /* Main loop: 4 weights per byte × 4 rows = 16 adds/subs per NEON step */
        for (; col <= cols - 4; col += 4, bidx++) {
            float32x4_t xq = vld1q_f32(input + col);
            neon_accum_4rows_1byte(w0[bidx], w1[bidx], w2[bidx], w3[bidx],
                                   xq, &acc0, &acc1, &acc2, &acc3);
        }

        /* Horizontal reduce and scale — only multiply is per-row scale factor */
        float s0 = neon_hsum(acc0);
        float s1 = neon_hsum(acc1);
        float s2 = neon_hsum(acc2);
        float s3 = neon_hsum(acc3);

        /* Tail elements (cols % 4 remainder) */
        for (; col < cols; col++) {
            int b_idx = col / 4, pos = col % 4;
            uint8_t c0 = (w0[b_idx] >> (pos*2)) & 0x3u;
            uint8_t c1 = (w1[b_idx] >> (pos*2)) & 0x3u;
            uint8_t c2 = (w2[b_idx] >> (pos*2)) & 0x3u;
            uint8_t c3 = (w3[b_idx] >> (pos*2)) & 0x3u;
            float xi = input[col];
            if (c0 == 1u) s0 += xi; else if (c0 == 2u) s0 -= xi;
            if (c1 == 1u) s1 += xi; else if (c1 == 2u) s1 -= xi;
            if (c2 == 1u) s2 += xi; else if (c2 == 2u) s2 -= xi;
            if (c3 == 1u) s3 += xi; else if (c3 == 2u) s3 -= xi;
        }

        output[row]     = s0 * (scales ? scales[row]     : 1.f) + (bias ? bias[row]     : 0.f);
        output[row + 1] = s1 * (scales ? scales[row + 1] : 1.f) + (bias ? bias[row + 1] : 0.f);
        output[row + 2] = s2 * (scales ? scales[row + 2] : 1.f) + (bias ? bias[row + 2] : 0.f);
        output[row + 3] = s3 * (scales ? scales[row + 3] : 1.f) + (bias ? bias[row + 3] : 0.f);
    }

    /* Scalar cleanup for rows % 4 remainder */
    for (int row = (rows & ~3); row < rows; row++) {
        const uint8_t *rw = weights + (size_t)row * row_bytes;
        float sum = 0.f;
        int col = 0, bidx = 0;
        for (; col <= cols - 4; col += 4, bidx++) {
            uint8_t byte = rw[bidx];
            /* Multiplication-free: decode each 2-bit code and add/subtract */
            for (int p = 0; p < 4; p++) {
                uint8_t code = (byte >> (p*2)) & 0x3u;
                if (code == 1u) sum += input[col + p];
                else if (code == 2u) sum -= input[col + p];
            }
        }
        for (; col < cols; col++) {
            uint8_t code = (rw[col/4] >> ((col%4)*2)) & 0x3u;
            if (code == 1u) sum += input[col];
            else if (code == 2u) sum -= input[col];
        }
        output[row] = sum * (scales ? scales[row] : 1.f) + (bias ? bias[row] : 0.f);
    }
}

/* ============================================================
 * T-MAC (Table Lookup Matrix Multiplication) for ARM NEON
 * Precomputes an activation LUT per 4-weight group, then
 * executes purely via table lookup accumulation (zero multiply, zero add/sub).
 * ============================================================ */
static void matmul_bitnet_f32_tmac_neon(const uint8_t *weights,
                                        const float *scales,
                                        const float *input,
                                        const float *bias,
                                        float *output,
                                        int rows, int cols)
{
    ensure_byte_weights();
    const int num_groups = (cols + 3) / 4;
    int safe_groups = (num_groups < MAX_TMAC_GROUPS) ? num_groups : MAX_TMAC_GROUPS;

    /* Amortized Activation LUT:
     * Transformer repeatedly multiplies the same activation against Q, K, V
     * and W1, W3. We reuse the precomputed LUT across these projections! */
    if (s_cached_input != input || s_cached_cols != cols) {
        for (int g = 0; g < safe_groups; g++) {
            int c0 = g * 4;
            float x0 = (c0 < cols) ? input[c0] : 0.f;
            float x1 = (c0 + 1 < cols) ? input[c0 + 1] : 0.f;
            float x2 = (c0 + 2 < cols) ? input[c0 + 2] : 0.f;
            float x3 = (c0 + 3 < cols) ? input[c0 + 3] : 0.f;
            float *group_lut = s_act_lut + (g * 256);

            for (int b = 0; b < 256; b++) {
                const float *w4 = s_byte_weights[b];
                group_lut[b] = x0 * w4[0] + x1 * w4[1] + x2 * w4[2] + x3 * w4[3];
            }
        }
        s_cached_input = input;
        s_cached_cols = cols;
    }

    #pragma omp parallel for schedule(static) if((int64_t)rows * cols >= 16384)
    for (int row = 0; row < (rows & ~3); row += 4) {
        const uint8_t *w0 = weights + (size_t)(row + 0) * num_groups;
        const uint8_t *w1 = weights + (size_t)(row + 1) * num_groups;
        const uint8_t *w2 = weights + (size_t)(row + 2) * num_groups;
        const uint8_t *w3 = weights + (size_t)(row + 3) * num_groups;

        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;

        for (int g = 0; g < safe_groups; g++) {
            const float *lut = s_act_lut + (g * 256);
            s0 += lut[w0[g]];
            s1 += lut[w1[g]];
            s2 += lut[w2[g]];
            s3 += lut[w3[g]];
        }

        output[row + 0] = s0 * (scales ? scales[row + 0] : 1.f) + (bias ? bias[row + 0] : 0.f);
        output[row + 1] = s1 * (scales ? scales[row + 1] : 1.f) + (bias ? bias[row + 1] : 0.f);
        output[row + 2] = s2 * (scales ? scales[row + 2] : 1.f) + (bias ? bias[row + 2] : 0.f);
        output[row + 3] = s3 * (scales ? scales[row + 3] : 1.f) + (bias ? bias[row + 3] : 0.f);
    }

    for (int row = (rows & ~3); row < rows; row++) {
        const uint8_t *rw = weights + (size_t)row * num_groups;
        float sum = 0.f;
        for (int g = 0; g < safe_groups; g++) {
            sum += s_act_lut[g * 256 + rw[g]];
        }
        output[row] = sum * (scales ? scales[row] : 1.f) + (bias ? bias[row] : 0.f);
    }
}
#endif /* EIF_HAS_NEON */



void eif_matmul_bitnet_f32(const uint8_t *weights,
                           const float *scales,
                           const float *input,
                           const float *bias,
                           float *output,
                           int rows,
                           int cols)
{
#if defined(EIF_HAS_AVX2)
    if (__builtin_cpu_supports("avx2")) {
        matmul_bitnet_f32_avx2(weights, scales, input, bias, output, rows, cols);
        return;
    }
#elif defined(EIF_HAS_NEON)
    static int s_use_tmac = -1;
    if (s_use_tmac < 0) {
        const char *env = getenv("EIF_USE_TMAC");
        s_use_tmac = (env && env[0] == '1') ? 1 : 0;
    }
    if (s_use_tmac) {
        matmul_bitnet_f32_tmac_neon(weights, scales, input, bias, output, rows, cols);
        return;
    }
    matmul_bitnet_f32_neon(weights, scales, input, bias, output, rows, cols);
    return;
#endif

    ensure_byte_weights();
    const int row_bytes = (cols + 3) / 4;

    #pragma omp parallel for schedule(static) if((int64_t)rows * cols >= 16384)
    for (int row = 0; row < rows; row++) {
        const uint8_t *row_w = weights + (row * row_bytes);
        float sum = 0.0f;
        int col = 0;
        int byte_idx = 0;

        /* Branchless lookup via precomputed 4KB L1 cache table */
        for (; col <= cols - 4; col += 4, byte_idx++) {
            const float *w4 = s_byte_weights[row_w[byte_idx]];
            sum += input[col]     * w4[0]
                 + input[col + 1] * w4[1]
                 + input[col + 2] * w4[2]
                 + input[col + 3] * w4[3];
        }

        /* Tail elements */
        for (; col < cols; col++) {
            int b_idx = col / 4;
            int pos = col % 4;
            uint8_t code = (uint8_t)((row_w[b_idx] >> (pos * 2)) & 0x03u);
            if (code == 1u) sum += input[col];
            else if (code == 2u) sum -= input[col];
        }

        float scale = scales ? scales[row] : 1.0f;
        output[row] = sum * scale + (bias ? bias[row] : 0.0f);
    }
}

void eif_matmul_bitnet_f32_w1w3(float *out_w1, float *out_w3,
                                const uint8_t *w1, const uint8_t *w3,
                                const float *scale_w1, const float *scale_w3,
                                const float *input, int hidden_dim, int dim)
{
    ensure_byte_weights();
    const int row_bytes = (dim + 3) / 4;
    int total_rows = hidden_dim * 2;

    #pragma omp parallel for schedule(static) if(!omp_in_parallel())
    for (int row = 0; row < (total_rows & ~3); row += 4) {
        const uint8_t *w_base = (row < hidden_dim) ? w1 : w3;
        const float *scale_base = (row < hidden_dim) ? scale_w1 : scale_w3;
        float *out_base = (row < hidden_dim) ? out_w1 : out_w3;
        int local_r = (row < hidden_dim) ? row : (row - hidden_dim);

        const uint8_t *r0 = w_base + (local_r * row_bytes);
        const uint8_t *r1 = w_base + ((local_r + 1) * row_bytes);
        const uint8_t *r2 = w_base + ((local_r + 2) * row_bytes);
        const uint8_t *r3 = w_base + ((local_r + 3) * row_bytes);

        float s0, s1, s2, s3;
#if defined(EIF_HAS_AVX2)
        if (__builtin_cpu_supports("avx2")) {
            compute_4rows_avx2(r0, r1, r2, r3, input, &s0, &s1, &s2, &s3, dim);
        } else
#endif
        {
            s0 = 0.0f; s1 = 0.0f; s2 = 0.0f; s3 = 0.0f;
            for (int col = 0, byte_idx = 0; col <= dim - 4; col += 4, byte_idx++) {
                const float *tw0 = s_byte_weights[r0[byte_idx]];
                const float *tw1 = s_byte_weights[r1[byte_idx]];
                const float *tw2 = s_byte_weights[r2[byte_idx]];
                const float *tw3 = s_byte_weights[r3[byte_idx]];
                for (int p = 0; p < 4; p++) {
                    s0 += input[col + p] * tw0[p];
                    s1 += input[col + p] * tw1[p];
                    s2 += input[col + p] * tw2[p];
                    s3 += input[col + p] * tw3[p];
                }
            }
        }

        out_base[local_r]     = s0 * scale_base[local_r];
        out_base[local_r + 1] = s1 * scale_base[local_r + 1];
        out_base[local_r + 2] = s2 * scale_base[local_r + 2];
        out_base[local_r + 3] = s3 * scale_base[local_r + 3];
    }
}

void eif_matmul_bitnet_dense_f32(const uint8_t *weights,
                                 const float *scales,
                                 const float *input,
                                 const float *bias,
                                 float *output,
                                 int rows,
                                 int cols)
{
    ensure_dense_lut();
    const int row_bytes = (cols + 4) / 5;

    #pragma omp parallel for schedule(static) if((int64_t)rows * cols >= 16384)
    for (int row = 0; row < rows; row++) {
        const uint8_t *row_w = weights + (row * row_bytes);
        float sum = 0.0f;
        int col = 0;
        int byte_idx = 0;

        /* Branchless 5-element float dot product from L1 cache table */
        for (; col <= cols - 5; col += 5, byte_idx++) {
            const float *w_vals = s_dense_lut_f32[row_w[byte_idx]];
            sum += input[col]     * w_vals[0]
                 + input[col + 1] * w_vals[1]
                 + input[col + 2] * w_vals[2]
                 + input[col + 3] * w_vals[3]
                 + input[col + 4] * w_vals[4];
        }

        if (col < cols) {
            const float *w_vals = s_dense_lut_f32[row_w[byte_idx]];
            int tail_idx = 0;
            for (; col < cols; col++, tail_idx++) {
                sum += input[col] * w_vals[tail_idx];
            }
        }

        float scale = scales ? scales[row] : 1.0f;
        output[row] = sum * scale + (bias ? bias[row] : 0.0f);
    }
}


void eif_matmul_bitnet_int8(const uint8_t *weights,
                            float scale_w,
                            const int8_t *input,
                            float scale_in,
                            const int32_t *bias,
                            int8_t *output,
                            float scale_out,
                            int rows,
                            int cols)
{
    ensure_byte_weights();
    const int row_bytes = (cols + 3) / 4;
    float eff_scale = (scale_w * scale_in) / (scale_out > 1e-12f ? scale_out : 1.0f);

    #pragma omp parallel for schedule(static) if(rows >= 16)
    for (int row = 0; row < rows; row++) {
        const uint8_t *row_w = weights + (row * row_bytes);
        int32_t isum = bias ? bias[row] : 0;
        int col = 0;
        int byte_idx = 0;

        /* Branchless 4-element int8 multiply-accumulate */
        for (; col <= cols - 4; col += 4, byte_idx++) {
            const int8_t *w4 = s_byte_weights_i8[row_w[byte_idx]];
            isum += (int32_t)input[col]     * (int32_t)w4[0]
                  + (int32_t)input[col + 1] * (int32_t)w4[1]
                  + (int32_t)input[col + 2] * (int32_t)w4[2]
                  + (int32_t)input[col + 3] * (int32_t)w4[3];
        }

        for (; col < cols; col++) {
            int b_idx = col / 4;
            int pos = col % 4;
            uint8_t code = (uint8_t)((row_w[b_idx] >> (pos * 2)) & 0x03u);
            if (code == 1u) isum += (int32_t)input[col];
            else if (code == 2u) isum -= (int32_t)input[col];
        }

        float scaled = (float)isum * eff_scale;
        long val = (long)(scaled >= 0.0f ? (scaled + 0.5f) : (scaled - 0.5f));
        if (val > 127) val = 127;
        if (val < -128) val = -128;
        output[row] = (int8_t)val;
    }
}

void eif_quantize_activation_i8(const float *x_f32,
                                int8_t *x_i8,
                                float *out_scale,
                                int n)
{
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = fabsf(x_f32[i]);
        if (v > amax) amax = v;
    }

    if (amax < 1e-12f) {
        memset(x_i8, 0, (size_t)n * sizeof(int8_t));
        if (out_scale) *out_scale = 1.0f;
        return;
    }

    float scale = amax / 127.0f;
    float inv_scale = 127.0f / amax;
    if (out_scale) *out_scale = scale;

    for (int i = 0; i < n; i++) {
        float q = x_f32[i] * inv_scale;
        long val = (long)(q >= 0.0f ? (q + 0.5f) : (q - 0.5f));
        if (val > 127) val = 127;
        if (val < -128) val = -128;
        x_i8[i] = (int8_t)val;
    }
}

#if defined(EIF_HAS_AVX2)
__attribute__((target("avx2")))
static inline int32_t dotprod_i8_avx2(const int8_t *input, const uint8_t *row_w, int cols)
{
    __m256i acc32 = _mm256_setzero_si256();
    int col = 0;
    int byte_idx = 0;
    for (; col <= cols - 16; col += 16, byte_idx += 4) {
        __m128i in_lo_8 = _mm_loadu_si128((const __m128i *)&input[col]);
        __m256i in16 = _mm256_cvtepi8_epi16(in_lo_8);

        int8_t w_buf[16];
        memcpy(w_buf + 0,  s_byte_weights_i8[row_w[byte_idx + 0]], 4);
        memcpy(w_buf + 4,  s_byte_weights_i8[row_w[byte_idx + 1]], 4);
        memcpy(w_buf + 8,  s_byte_weights_i8[row_w[byte_idx + 2]], 4);
        memcpy(w_buf + 12, s_byte_weights_i8[row_w[byte_idx + 3]], 4);
        __m128i w_lo_8 = _mm_loadu_si128((const __m128i *)w_buf);
        __m256i w16 = _mm256_cvtepi8_epi16(w_lo_8);

        __m256i prod32 = _mm256_madd_epi16(in16, w16);
        acc32 = _mm256_add_epi32(acc32, prod32);
    }
    __m128i hi128 = _mm256_extracti128_si256(acc32, 1);
    __m128i lo128 = _mm256_castsi256_si128(acc32);
    __m128i s128 = _mm_add_epi32(lo128, hi128);
    s128 = _mm_hadd_epi32(s128, s128);
    s128 = _mm_hadd_epi32(s128, s128);
    int32_t isum = _mm_cvtsi128_si32(s128);

    for (; col <= cols - 4; col += 4, byte_idx++) {
        const int8_t *w4 = s_byte_weights_i8[row_w[byte_idx]];
        isum += (int32_t)input[col]     * (int32_t)w4[0]
              + (int32_t)input[col + 1] * (int32_t)w4[1]
              + (int32_t)input[col + 2] * (int32_t)w4[2]
              + (int32_t)input[col + 3] * (int32_t)w4[3];
    }
    for (; col < cols; col++) {
        int b_idx = col / 4;
        int pos = col % 4;
        uint8_t code = (uint8_t)((row_w[b_idx] >> (pos * 2)) & 0x03u);
        if (code == 1u) isum += (int32_t)input[col];
        else if (code == 2u) isum -= (int32_t)input[col];
    }
    return isum;
}
#endif

#if defined(__ARM_FEATURE_DOTPROD)
static inline int32_t dotprod_i8_neon(const int8_t *input, const uint8_t *row_w, int cols)
{
    int32x4_t acc = vdupq_n_s32(0);
    int col = 0;
    int byte_idx = 0;
    for (; col <= cols - 16; col += 16, byte_idx += 4) {
        int8x16_t in16 = vld1q_s8(&input[col]);
        int8_t w_buf[16];
        memcpy(w_buf + 0,  s_byte_weights_i8[row_w[byte_idx + 0]], 4);
        memcpy(w_buf + 4,  s_byte_weights_i8[row_w[byte_idx + 1]], 4);
        memcpy(w_buf + 8,  s_byte_weights_i8[row_w[byte_idx + 2]], 4);
        memcpy(w_buf + 12, s_byte_weights_i8[row_w[byte_idx + 3]], 4);
        int8x16_t w16 = vld1q_s8(w_buf);
        acc = vdotq_s32(acc, in16, w16);
    }
    int32_t isum = vaddvq_s32(acc);
    for (; col <= cols - 4; col += 4, byte_idx++) {
        const int8_t *w4 = s_byte_weights_i8[row_w[byte_idx]];
        isum += (int32_t)input[col]     * (int32_t)w4[0]
              + (int32_t)input[col + 1] * (int32_t)w4[1]
              + (int32_t)input[col + 2] * (int32_t)w4[2]
              + (int32_t)input[col + 3] * (int32_t)w4[3];
    }
    for (; col < cols; col++) {
        int b_idx = col / 4;
        int pos = col % 4;
        uint8_t code = (uint8_t)((row_w[b_idx] >> (pos * 2)) & 0x03u);
        if (code == 1u) isum += (int32_t)input[col];
        else if (code == 2u) isum -= (int32_t)input[col];
    }
    return isum;
}
#endif

void eif_matmul_bitnet_i8xternary_f32(const uint8_t *weights,
                                      const float *scales,
                                      const int8_t *input_i8,
                                      float scale_in,
                                      const float *bias,
                                      float *output,
                                      int rows,
                                      int cols)
{
    ensure_byte_weights();
    const int row_bytes = (cols + 3) / 4;

    #pragma omp parallel for schedule(static) if((int64_t)rows * cols >= 16384)
    for (int row = 0; row < rows; row++) {
        const uint8_t *row_w = weights + (row * row_bytes);
        int32_t isum = 0;

#if defined(EIF_HAS_AVX2)
        if (__builtin_cpu_supports("avx2")) {
            isum = dotprod_i8_avx2(input_i8, row_w, cols);
        } else {
            int col = 0, byte_idx = 0;
            for (; col <= cols - 4; col += 4, byte_idx++) {
                const int8_t *w4 = s_byte_weights_i8[row_w[byte_idx]];
                isum += (int32_t)input_i8[col]     * (int32_t)w4[0]
                      + (int32_t)input_i8[col + 1] * (int32_t)w4[1]
                      + (int32_t)input_i8[col + 2] * (int32_t)w4[2]
                      + (int32_t)input_i8[col + 3] * (int32_t)w4[3];
            }
            for (; col < cols; col++) {
                int b_idx = col / 4, pos = col % 4;
                uint8_t code = (uint8_t)((row_w[b_idx] >> (pos * 2)) & 0x03u);
                if (code == 1u) isum += (int32_t)input_i8[col];
                else if (code == 2u) isum -= (int32_t)input_i8[col];
            }
        }
#elif defined(__ARM_FEATURE_DOTPROD)
        isum = dotprod_i8_neon(input_i8, row_w, cols);
#else
        {
            int col = 0, byte_idx = 0;
            for (; col <= cols - 4; col += 4, byte_idx++) {
                const int8_t *w4 = s_byte_weights_i8[row_w[byte_idx]];
                isum += (int32_t)input_i8[col]     * (int32_t)w4[0]
                      + (int32_t)input_i8[col + 1] * (int32_t)w4[1]
                      + (int32_t)input_i8[col + 2] * (int32_t)w4[2]
                      + (int32_t)input_i8[col + 3] * (int32_t)w4[3];
            }
            for (; col < cols; col++) {
                int b_idx = col / 4, pos = col % 4;
                uint8_t code = (uint8_t)((row_w[b_idx] >> (pos * 2)) & 0x03u);
                if (code == 1u) isum += (int32_t)input_i8[col];
                else if (code == 2u) isum -= (int32_t)input_i8[col];
            }
        }
#endif

        float total_scale = (scales ? scales[row] : 1.0f) * scale_in;
        output[row] = ((float)isum * total_scale) + (bias ? bias[row] : 0.0f);
    }
}

void eif_matmul_bitnet_act_quant_f32(const uint8_t *weights,
                                     const float *scales,
                                     const float *input_f32,
                                     const float *bias,
                                     float *output,
                                     int rows,
                                     int cols)
{
    /* Dynamically allocate stack buffer if small, or heap if large */
    int8_t stack_buf[4096];
    int8_t *i8_buf = (cols <= 4096) ? stack_buf : (int8_t *)malloc((size_t)cols * sizeof(int8_t));
    float scale_in = 1.0f;

    eif_quantize_activation_i8(input_f32, i8_buf, &scale_in, cols);
    eif_matmul_bitnet_i8xternary_f32(weights, scales, i8_buf, scale_in, bias, output, rows, cols);

    if (cols > 4096 && i8_buf != stack_buf) {
        free(i8_buf);
    }
}

void eif_matmul_bitnet_block_f32(const uint8_t *weights,
                                 const float *block_scales,
                                 int block_size,
                                 const float *input,
                                 const float *bias,
                                 float *output,
                                 int rows,
                                 int cols)
{
    ensure_byte_weights();
    if (block_size <= 0) block_size = 64;
    const int num_blocks_per_row = (cols + block_size - 1) / block_size;
    const int row_bytes = (cols + 3) / 4;

    #pragma omp parallel for schedule(static) if((int64_t)rows * cols >= 16384)
    for (int row = 0; row < rows; row++) {
        const uint8_t *row_w = weights + (row * row_bytes);
        const float *scales_row = block_scales ? (block_scales + row * num_blocks_per_row) : NULL;
        float total_sum = 0.0f;

        for (int b = 0; b < num_blocks_per_row; b++) {
            int start_col = b * block_size;
            int end_col = start_col + block_size;
            if (end_col > cols) end_col = cols;

            float block_sum = 0.0f;
            int col = start_col;
            int byte_idx = start_col / 4;

            /* If start_col is aligned to 4 */
            if ((start_col % 4) == 0) {
                for (; col <= end_col - 4; col += 4, byte_idx++) {
                    const float *w4 = s_byte_weights[row_w[byte_idx]];
                    block_sum += input[col]     * w4[0]
                               + input[col + 1] * w4[1]
                               + input[col + 2] * w4[2]
                               + input[col + 3] * w4[3];
                }
            }

            for (; col < end_col; col++) {
                int b_idx = col / 4;
                int pos = col % 4;
                uint8_t code = (uint8_t)((row_w[b_idx] >> (pos * 2)) & 0x03u);
                if (code == 1u) block_sum += input[col];
                else if (code == 2u) block_sum -= input[col];
            }

            float b_scale = scales_row ? scales_row[b] : 1.0f;
            total_sum += block_sum * b_scale;
        }

        output[row] = total_sum + (bias ? bias[row] : 0.0f);
    }
}

size_t eif_interleave_weights_4rows(const uint8_t *src_row_major,
                                    uint8_t *dst_interleaved,
                                    int rows,
                                    int cols)
{
    if (!src_row_major || !dst_interleaved || rows <= 0 || cols <= 0)
        return 0;

    const int num_groups = (cols + 3) / 4;
    int r_block = 0;
    size_t dst_offset = 0;

    for (; r_block < (rows & ~3); r_block += 4) {
        const uint8_t *r0 = src_row_major + (size_t)(r_block + 0) * num_groups;
        const uint8_t *r1 = src_row_major + (size_t)(r_block + 1) * num_groups;
        const uint8_t *r2 = src_row_major + (size_t)(r_block + 2) * num_groups;
        const uint8_t *r3 = src_row_major + (size_t)(r_block + 3) * num_groups;

        for (int g = 0; g < num_groups; g++) {
            dst_interleaved[dst_offset + 0] = r0[g];
            dst_interleaved[dst_offset + 1] = r1[g];
            dst_interleaved[dst_offset + 2] = r2[g];
            dst_interleaved[dst_offset + 3] = r3[g];
            dst_offset += 4;
        }
    }

    for (; r_block < rows; r_block++) {
        const uint8_t *r = src_row_major + (size_t)r_block * num_groups;
        memcpy(dst_interleaved + dst_offset, r, (size_t)num_groups);
        dst_offset += (size_t)num_groups;
    }

    return dst_offset;
}

void eif_matmul_bitnet_tmac_interleaved_f32(const uint8_t *interleaved_weights,
                                           const float *scales,
                                           const float *input,
                                           const float *bias,
                                           float *output,
                                           int rows,
                                           int cols)
{
    ensure_byte_weights();
    const int num_groups = (cols + 3) / 4;
    int safe_groups = (num_groups < MAX_TMAC_GROUPS) ? num_groups : MAX_TMAC_GROUPS;

    /* Populate or amortize activation LUT */
    if (s_cached_input != input || s_cached_cols != cols) {
        for (int g = 0; g < safe_groups; g++) {
            int c0 = g * 4;
            float x0 = (c0 < cols) ? input[c0] : 0.f;
            float x1 = (c0 + 1 < cols) ? input[c0 + 1] : 0.f;
            float x2 = (c0 + 2 < cols) ? input[c0 + 2] : 0.f;
            float x3 = (c0 + 3 < cols) ? input[c0 + 3] : 0.f;
            float *group_lut = s_act_lut + (g * 256);

            for (int b = 0; b < 256; b++) {
                const float *w4 = s_byte_weights[b];
                group_lut[b] = x0 * w4[0] + x1 * w4[1] + x2 * w4[2] + x3 * w4[3];
            }
        }
        s_cached_input = input;
        s_cached_cols = cols;
    }

    #pragma omp parallel for schedule(static) if((int64_t)rows * cols >= 16384)
    for (int r_block = 0; r_block < (rows & ~3); r_block += 4) {
        const uint8_t *tile = interleaved_weights + (size_t)r_block * num_groups;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;

        /* Sequential 32-bit load fetches weights for all 4 rows in a single memory op */
        for (int g = 0; g < safe_groups; g++) {
            const float *lut = s_act_lut + (g * 256);
            uint32_t w4 = *(const uint32_t *)(tile + (g * 4));
            s0 += lut[(uint8_t)(w4)];
            s1 += lut[(uint8_t)(w4 >> 8)];
            s2 += lut[(uint8_t)(w4 >> 16)];
            s3 += lut[(uint8_t)(w4 >> 24)];
        }

        output[r_block + 0] = s0 * (scales ? scales[r_block + 0] : 1.f) + (bias ? bias[r_block + 0] : 0.f);
        output[r_block + 1] = s1 * (scales ? scales[r_block + 1] : 1.f) + (bias ? bias[r_block + 1] : 0.f);
        output[r_block + 2] = s2 * (scales ? scales[r_block + 2] : 1.f) + (bias ? bias[r_block + 2] : 0.f);
        output[r_block + 3] = s3 * (scales ? scales[r_block + 3] : 1.f) + (bias ? bias[r_block + 3] : 0.f);
    }

    /* Remainder rows */
    size_t rem_offset = (size_t)(rows & ~3) * num_groups;
    for (int row = (rows & ~3); row < rows; row++) {
        const uint8_t *rw = interleaved_weights + rem_offset + (size_t)(row - (rows & ~3)) * num_groups;
        float sum = 0.f;
        for (int g = 0; g < safe_groups; g++) {
            sum += s_act_lut[g * 256 + rw[g]];
        }
        output[row] = sum * (scales ? scales[row] : 1.f) + (bias ? bias[row] : 0.f);
    }
}


/* ===========================================================================
 * Hadamard rotation + group scales (Bonsai-style ternary, BitLinear(group_size, hadamard) in the SDK)
 * =========================================================================== */

int eif_hadamard_block_size(int n)
{
    int b = 1;
    while (n > 0 && n % (b * 2) == 0 && b * 2 <= 1024) b *= 2;
    return b;
}

int eif_bitnet_group_size(int n, int max_group)
{
    if (max_group <= 0 || max_group >= n) return 0;
    for (int g = max_group - max_group % 32; g >= 32; g -= 32) {
        if (n % g == 0) return g;
    }
    return 0;
}

void eif_hadamard_rotate_f32(float *x, const float *signs, int n, int block)
{
    if (!x || block <= 0 || n % block != 0) return;
    if (signs) {
        for (int i = 0; i < n; i++) x[i] *= signs[i];
    }
    const float norm = 1.0f / sqrtf((float)block);
    for (int base = 0; base < n; base += block) {
        float *v = x + base;
        for (int h = 1; h < block; h *= 2) {
            for (int i = 0; i < block; i += 2 * h) {
                for (int j = i; j < i + h; j++) {
                    float a = v[j], b = v[j + h];
                    v[j] = a + b;
                    v[j + h] = a - b;
                }
            }
        }
        for (int i = 0; i < block; i++) v[i] *= norm;
    }
}
