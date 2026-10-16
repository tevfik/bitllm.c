/**
 * @file eif_bert.c
 * @brief High-Performance Cross-Platform (x86_64 AVX2 / ARM64 NEON) BERT Runtime
 */

#if defined(__linux__)
#define _GNU_SOURCE
#endif

#include "eif_bert.h"
#include "eif_gguf.h"
#include "eif_bpe_tokenizer.h"
#include "eif_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>

/* =============================================================================
 * Cross-Platform SIMD Setup & Reduction Helpers
 * ============================================================================= */

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__)
#include <immintrin.h>
#define EIF_ARCH_X86_64 1
#endif
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(__ARM_NEON)
#include <arm_neon.h>
#define EIF_ARCH_ARM64 1
#endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(EIF_ARCH_X86_64)
/* Fast horizontal addition of 8 single-precision floats (AVX) without haddps stalls */
static inline float hsum_float_8(__m256 x)
{
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline int32_t hsum_epi32_8(__m256i x)
{
    __m128i l = _mm256_castsi256_si128(x);
    __m128i h = _mm256_extracti128_si256(x, 1);
    __m128i s = _mm_add_epi32(l, h);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}
#endif

#if defined(EIF_ARCH_ARM64)
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC target("+dotprod")
#endif
static inline int32_t arm_dot32(const int8_t *wb, const int8_t *xb)
{
    int32x4_t vacc = vdupq_n_s32(0);
    vacc = vdotq_s32(vacc, vld1q_s8(wb), vld1q_s8(xb));
    vacc = vdotq_s32(vacc, vld1q_s8(wb + 16), vld1q_s8(xb + 16));
    return vaddvq_s32(vacc);
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
#endif

/* Standard ALiBi geometric slope sequence (Press et al., 2021) for H heads */
static inline float bert_alibi_slope(int h, int n_heads)
{
    int n_head_log2 = 1;
    while ((n_head_log2 << 1) <= n_heads) {
        n_head_log2 <<= 1;
    }
    if (n_head_log2 == n_heads) {
        return powf(2.0f, -8.0f * (float)(h + 1) / (float)n_heads);
    }
    float m0 = powf(2.0f, -8.0f / (float)n_head_log2);
    float m1 = powf(2.0f, -4.0f / (float)n_head_log2);
    if (h < n_head_log2) {
        return powf(m0, (float)(h + 1));
    } else {
        return powf(m1, (float)(2 * (h - n_head_log2) + 1));
    }
}

/* 1. LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta */
static void bert_layernorm(float *out, const float *in, const float *gamma, const float *beta, int dim, float eps)
{
    float mean = 0.0f;
    float var = 0.0f;

#if defined(EIF_ARCH_X86_64)
    __m256 vmean = _mm256_setzero_ps();
    for (int i = 0; i <= dim - 8; i += 8) {
        vmean = _mm256_add_ps(vmean, _mm256_loadu_ps(&in[i]));
    }
    mean = hsum_float_8(vmean);
    for (int i = (dim & ~7); i < dim; i++) mean += in[i];
#elif defined(EIF_ARCH_ARM64)
    float32x4_t vmean = vdupq_n_f32(0.0f);
    for (int i = 0; i <= dim - 4; i += 4) {
        vmean = vaddq_f32(vmean, vld1q_f32(&in[i]));
    }
    mean = vaddvq_f32(vmean);
    for (int i = (dim & ~3); i < dim; i++) mean += in[i];
#else
    for (int i = 0; i < dim; i++) mean += in[i];
#endif
    mean /= (float)dim;

#if defined(EIF_ARCH_X86_64)
    __m256 vvm = _mm256_set1_ps(mean);
    __m256 vvar = _mm256_setzero_ps();
    for (int i = 0; i <= dim - 8; i += 8) {
        __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(&in[i]), vvm);
        vvar = _mm256_fmadd_ps(diff, diff, vvar);
    }
    var = hsum_float_8(vvar);
    for (int i = (dim & ~7); i < dim; i++) {
        float d = in[i] - mean;
        var += d * d;
    }
#elif defined(EIF_ARCH_ARM64)
    float32x4_t vvm = vdupq_n_f32(mean);
    float32x4_t vvar = vdupq_n_f32(0.0f);
    for (int i = 0; i <= dim - 4; i += 4) {
        float32x4_t diff = vsubq_f32(vld1q_f32(&in[i]), vvm);
        vvar = vfmaq_f32(vvar, diff, diff);
    }
    var = vaddvq_f32(vvar);
    for (int i = (dim & ~3); i < dim; i++) {
        float d = in[i] - mean;
        var += d * d;
    }
#else
    for (int i = 0; i < dim; i++) {
        float d = in[i] - mean;
        var += d * d;
    }
#endif
    var /= (float)dim;
    float inv_std = 1.0f / sqrtf(var + eps);

#if defined(EIF_ARCH_X86_64)
    __m256 vinv = _mm256_set1_ps(inv_std);
    for (int i = 0; i <= dim - 8; i += 8) {
        __m256 x = _mm256_loadu_ps(&in[i]);
        __m256 g = _mm256_loadu_ps(&gamma[i]);
        __m256 b = beta ? _mm256_loadu_ps(&beta[i]) : _mm256_setzero_ps();
        __m256 norm = _mm256_mul_ps(_mm256_sub_ps(x, vvm), vinv);
        _mm256_storeu_ps(&out[i], _mm256_fmadd_ps(norm, g, b));
    }
    for (int i = (dim & ~7); i < dim; i++) {
        out[i] = ((in[i] - mean) * inv_std) * gamma[i] + (beta ? beta[i] : 0.0f);
    }
#elif defined(EIF_ARCH_ARM64)
    float32x4_t vinv = vdupq_n_f32(inv_std);
    for (int i = 0; i <= dim - 4; i += 4) {
        float32x4_t x = vld1q_f32(&in[i]);
        float32x4_t g = vld1q_f32(&gamma[i]);
        float32x4_t b = beta ? vld1q_f32(&beta[i]) : vdupq_n_f32(0.0f);
        float32x4_t norm = vmulq_f32(vsubq_f32(x, vvm), vinv);
        vst1q_f32(&out[i], vfmaq_f32(b, norm, g));
    }
    for (int i = (dim & ~3); i < dim; i++) {
        out[i] = ((in[i] - mean) * inv_std) * gamma[i] + (beta ? beta[i] : 0.0f);
    }
#else
    for (int i = 0; i < dim; i++) {
        out[i] = ((in[i] - mean) * inv_std) * gamma[i] + (beta ? beta[i] : 0.0f);
    }
#endif
}

/* 2. Fast GELU Activation */
static inline float gelu_f32(float x)
{
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

static void bert_gelu(float *out, const float *in, int n)
{
#pragma omp parallel for schedule(static) if(n >= 1024)
    for (int i = 0; i < n; i++) {
        out[i] = gelu_f32(in[i]);
    }
}

/* 3. High-Performance Batched GEMM: Y[T, rows] = X[T, cols] * W[rows, cols]^T + bias */
static void bert_gemm_int8(
    float *Y,               /* Output: [T, rows] */
    const int8_t *W,        /* Weights: [rows, cols] */
    const float *W_scales,  /* Scales:  [rows, cols / 32] */
    const float *X,         /* Input:   [T, cols] */
    const float *bias,      /* Bias:    [rows] (optional) */
    int T,
    int rows,
    int cols)
{
    const int blocks_per_row = cols / 32;
    const int TILE_R = 64;
    const int TILE_B = (blocks_per_row >= 32) ? 8 : blocks_per_row;

#pragma omp parallel for schedule(static) if((int64_t)rows * cols >= 4096)
    for (int r_tile = 0; r_tile < rows; r_tile += TILE_R) {
        int r_end = (r_tile + TILE_R <= rows) ? r_tile + TILE_R : rows;

        if (TILE_B == blocks_per_row) {
            for (int r = r_tile; r < r_end; r++) {
                const int8_t *w_row = W + (size_t)r * cols;
                const float *scales_row = W_scales + (size_t)r * blocks_per_row;
                const float b = bias ? bias[r] : 0.0f;
                int t = 0;

#if defined(EIF_ARCH_X86_64)
                for (; t <= T - 4; t += 4) {
                    const float *x0 = X + (size_t)(t + 0) * cols;
                    const float *x1 = X + (size_t)(t + 1) * cols;
                    const float *x2 = X + (size_t)(t + 2) * cols;
                    const float *x3 = X + (size_t)(t + 3) * cols;

                    __m256 vacc0 = _mm256_setzero_ps();
                    __m256 vacc1 = _mm256_setzero_ps();
                    __m256 vacc2 = _mm256_setzero_ps();
                    __m256 vacc3 = _mm256_setzero_ps();

                    for (int b_idx = 0; b_idx < blocks_per_row; b_idx++) {
                        const int8_t *wb = w_row + (size_t)b_idx * 32;
                        float scale = scales_row[b_idx];

                        __m256i raw32 = _mm256_loadu_si256((const __m256i *)wb);
                        __m128i rlo = _mm256_castsi256_si128(raw32);
                        __m128i rhi = _mm256_extracti128_si256(raw32, 1);

                        __m256 vs = _mm256_set1_ps(scale);
                        __m256 w0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(rlo)), vs);
                        __m256 w1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(rlo, 8))), vs);
                        __m256 w2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(rhi)), vs);
                        __m256 w3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(rhi, 8))), vs);

                        size_t off = (size_t)b_idx * 32;

                        vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + off +  0), w0, vacc0);
                        vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + off +  8), w1, vacc0);
                        vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + off + 16), w2, vacc0);
                        vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + off + 24), w3, vacc0);

                        vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + off +  0), w0, vacc1);
                        vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + off +  8), w1, vacc1);
                        vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + off + 16), w2, vacc1);
                        vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + off + 24), w3, vacc1);

                        vacc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x2 + off +  0), w0, vacc2);
                        vacc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x2 + off +  8), w1, vacc2);
                        vacc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x2 + off + 16), w2, vacc2);
                        vacc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x2 + off + 24), w3, vacc2);

                        vacc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x3 + off +  0), w0, vacc3);
                        vacc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x3 + off +  8), w1, vacc3);
                        vacc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x3 + off + 16), w2, vacc3);
                        vacc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x3 + off + 24), w3, vacc3);
                    }

                    Y[(t + 0) * rows + r] = hsum_float_8(vacc0) + b;
                    Y[(t + 1) * rows + r] = hsum_float_8(vacc1) + b;
                    Y[(t + 2) * rows + r] = hsum_float_8(vacc2) + b;
                    Y[(t + 3) * rows + r] = hsum_float_8(vacc3) + b;
                }
                for (; t <= T - 2; t += 2) {
                    const float *x0 = X + (size_t)(t + 0) * cols;
                    const float *x1 = X + (size_t)(t + 1) * cols;

                    __m256 vacc0 = _mm256_setzero_ps();
                    __m256 vacc1 = _mm256_setzero_ps();

                    for (int b_idx = 0; b_idx < blocks_per_row; b_idx++) {
                        const int8_t *wb = w_row + (size_t)b_idx * 32;
                        float scale = scales_row[b_idx];

                        __m256i raw32 = _mm256_loadu_si256((const __m256i *)wb);
                        __m128i rlo = _mm256_castsi256_si128(raw32);
                        __m128i rhi = _mm256_extracti128_si256(raw32, 1);

                        __m256 vs = _mm256_set1_ps(scale);
                        __m256 w0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(rlo)), vs);
                        __m256 w1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(rlo, 8))), vs);
                        __m256 w2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(rhi)), vs);
                        __m256 w3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(rhi, 8))), vs);

                        size_t off = (size_t)b_idx * 32;

                        vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + off +  0), w0, vacc0);
                        vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + off +  8), w1, vacc0);
                        vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + off + 16), w2, vacc0);
                        vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + off + 24), w3, vacc0);

                        vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + off +  0), w0, vacc1);
                        vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + off +  8), w1, vacc1);
                        vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + off + 16), w2, vacc1);
                        vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + off + 24), w3, vacc1);
                    }

                    Y[(t + 0) * rows + r] = hsum_float_8(vacc0) + b;
                    Y[(t + 1) * rows + r] = hsum_float_8(vacc1) + b;
                }
                for (; t < T; t++) {
                    const float *x0 = X + (size_t)t * cols;
                    __m256 vacc0 = _mm256_setzero_ps();

                    for (int b_idx = 0; b_idx < blocks_per_row; b_idx++) {
                        const int8_t *wb = w_row + (size_t)b_idx * 32;
                        float scale = scales_row[b_idx];

                        __m256i raw32 = _mm256_loadu_si256((const __m256i *)wb);
                        __m128i rlo = _mm256_castsi256_si128(raw32);
                        __m128i rhi = _mm256_extracti128_si256(raw32, 1);

                        __m256 vs = _mm256_set1_ps(scale);
                        __m256 w0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(rlo)), vs);
                        __m256 w1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(rlo, 8))), vs);
                        __m256 w2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(rhi)), vs);
                        __m256 w3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(rhi, 8))), vs);

                        size_t off = (size_t)b_idx * 32;

                        vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + off +  0), w0, vacc0);
                        vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + off +  8), w1, vacc0);
                        vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + off + 16), w2, vacc0);
                        vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + off + 24), w3, vacc0);
                    }

                    Y[t * rows + r] = hsum_float_8(vacc0) + b;
                }
#elif defined(EIF_ARCH_ARM64)
                for (; t <= T - 4; t += 4) {
                    const float *x0 = X + (size_t)(t + 0) * cols;
                    const float *x1 = X + (size_t)(t + 1) * cols;
                    const float *x2 = X + (size_t)(t + 2) * cols;
                    const float *x3 = X + (size_t)(t + 3) * cols;

                    float32x4_t vacc0 = vdupq_n_f32(0.0f);
                    float32x4_t vacc1 = vdupq_n_f32(0.0f);
                    float32x4_t vacc2 = vdupq_n_f32(0.0f);
                    float32x4_t vacc3 = vdupq_n_f32(0.0f);

                    for (int b_idx = 0; b_idx < blocks_per_row; b_idx++) {
                        const int8_t *wb = w_row + (size_t)b_idx * 32;
                        float scale = scales_row[b_idx];
                        float32x4_t vs = vdupq_n_f32(scale);

                        int8x16_t b0 = vld1q_s8(wb);
                        int8x16_t b1 = vld1q_s8(wb + 16);
                        int16x8_t s0 = vmovl_s8(vget_low_s8(b0));
                        int16x8_t s1 = vmovl_s8(vget_high_s8(b0));
                        int16x8_t s2 = vmovl_s8(vget_low_s8(b1));
                        int16x8_t s3 = vmovl_s8(vget_high_s8(b1));

                        float32x4_t wf0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s0))), vs);
                        float32x4_t wf1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s0))), vs);
                        float32x4_t wf2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s1))), vs);
                        float32x4_t wf3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s1))), vs);
                        float32x4_t wf4 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s2))), vs);
                        float32x4_t wf5 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s2))), vs);
                        float32x4_t wf6 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s3))), vs);
                        float32x4_t wf7 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s3))), vs);

                        size_t off = (size_t)b_idx * 32;

                        vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off +  0), wf0);
                        vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off +  4), wf1);
                        vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off +  8), wf2);
                        vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 12), wf3);
                        vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 16), wf4);
                        vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 20), wf5);
                        vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 24), wf6);
                        vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 28), wf7);

                        vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off +  0), wf0);
                        vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off +  4), wf1);
                        vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off +  8), wf2);
                        vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off + 12), wf3);
                        vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off + 16), wf4);
                        vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off + 20), wf5);
                        vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off + 24), wf6);
                        vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off + 28), wf7);

                        vacc2 = vfmaq_f32(vacc2, vld1q_f32(x2 + off +  0), wf0);
                        vacc2 = vfmaq_f32(vacc2, vld1q_f32(x2 + off +  4), wf1);
                        vacc2 = vfmaq_f32(vacc2, vld1q_f32(x2 + off +  8), wf2);
                        vacc2 = vfmaq_f32(vacc2, vld1q_f32(x2 + off + 12), wf3);
                        vacc2 = vfmaq_f32(vacc2, vld1q_f32(x2 + off + 16), wf4);
                        vacc2 = vfmaq_f32(vacc2, vld1q_f32(x2 + off + 20), wf5);
                        vacc2 = vfmaq_f32(vacc2, vld1q_f32(x2 + off + 24), wf6);
                        vacc2 = vfmaq_f32(vacc2, vld1q_f32(x2 + off + 28), wf7);

                        vacc3 = vfmaq_f32(vacc3, vld1q_f32(x3 + off +  0), wf0);
                        vacc3 = vfmaq_f32(vacc3, vld1q_f32(x3 + off +  4), wf1);
                        vacc3 = vfmaq_f32(vacc3, vld1q_f32(x3 + off +  8), wf2);
                        vacc3 = vfmaq_f32(vacc3, vld1q_f32(x3 + off + 12), wf3);
                        vacc3 = vfmaq_f32(vacc3, vld1q_f32(x3 + off + 16), wf4);
                        vacc3 = vfmaq_f32(vacc3, vld1q_f32(x3 + off + 20), wf5);
                        vacc3 = vfmaq_f32(vacc3, vld1q_f32(x3 + off + 24), wf6);
                        vacc3 = vfmaq_f32(vacc3, vld1q_f32(x3 + off + 28), wf7);
                    }

                    Y[(t + 0) * rows + r] = vaddvq_f32(vacc0) + b;
                    Y[(t + 1) * rows + r] = vaddvq_f32(vacc1) + b;
                    Y[(t + 2) * rows + r] = vaddvq_f32(vacc2) + b;
                    Y[(t + 3) * rows + r] = vaddvq_f32(vacc3) + b;
                }
                for (; t <= T - 2; t += 2) {
                    const float *x0 = X + (size_t)(t + 0) * cols;
                    const float *x1 = X + (size_t)(t + 1) * cols;

                    float32x4_t vacc0 = vdupq_n_f32(0.0f);
                    float32x4_t vacc1 = vdupq_n_f32(0.0f);

                    for (int b_idx = 0; b_idx < blocks_per_row; b_idx++) {
                        const int8_t *wb = w_row + (size_t)b_idx * 32;
                        float scale = scales_row[b_idx];
                        float32x4_t vs = vdupq_n_f32(scale);

                        int8x16_t b0 = vld1q_s8(wb);
                        int8x16_t b1 = vld1q_s8(wb + 16);
                        int16x8_t s0 = vmovl_s8(vget_low_s8(b0));
                        int16x8_t s1 = vmovl_s8(vget_high_s8(b0));
                        int16x8_t s2 = vmovl_s8(vget_low_s8(b1));
                        int16x8_t s3 = vmovl_s8(vget_high_s8(b1));

                        float32x4_t wf0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s0))), vs);
                        float32x4_t wf1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s0))), vs);
                        float32x4_t wf2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s1))), vs);
                        float32x4_t wf3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s1))), vs);
                        float32x4_t wf4 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s2))), vs);
                        float32x4_t wf5 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s2))), vs);
                        float32x4_t wf6 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s3))), vs);
                        float32x4_t wf7 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s3))), vs);

                        size_t off = (size_t)b_idx * 32;

                        vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off +  0), wf0);
                        vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off +  4), wf1);
                        vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off +  8), wf2);
                        vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 12), wf3);
                        vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 16), wf4);
                        vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 20), wf5);
                        vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 24), wf6);
                        vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 28), wf7);

                        vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off +  0), wf0);
                        vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off +  4), wf1);
                        vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off +  8), wf2);
                        vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off + 12), wf3);
                        vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off + 16), wf4);
                        vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off + 20), wf5);
                        vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off + 24), wf6);
                        vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off + 28), wf7);
                    }

                    Y[(t + 0) * rows + r] = vaddvq_f32(vacc0) + b;
                    Y[(t + 1) * rows + r] = vaddvq_f32(vacc1) + b;
                }
                for (; t < T; t++) {
                    const float *x0 = X + (size_t)t * cols;
                    float32x4_t vacc0 = vdupq_n_f32(0.0f);

                    for (int b_idx = 0; b_idx < blocks_per_row; b_idx++) {
                        const int8_t *wb = w_row + (size_t)b_idx * 32;
                        float scale = scales_row[b_idx];
                        float32x4_t vs = vdupq_n_f32(scale);

                        int8x16_t b0 = vld1q_s8(wb);
                        int8x16_t b1 = vld1q_s8(wb + 16);
                        int16x8_t s0 = vmovl_s8(vget_low_s8(b0));
                        int16x8_t s1 = vmovl_s8(vget_high_s8(b0));
                        int16x8_t s2 = vmovl_s8(vget_low_s8(b1));
                        int16x8_t s3 = vmovl_s8(vget_high_s8(b1));

                        float32x4_t wf0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s0))), vs);
                        float32x4_t wf1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s0))), vs);
                        float32x4_t wf2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s1))), vs);
                        float32x4_t wf3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s1))), vs);
                        float32x4_t wf4 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s2))), vs);
                        float32x4_t wf5 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s2))), vs);
                        float32x4_t wf6 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s3))), vs);
                        float32x4_t wf7 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s3))), vs);

                        size_t off = (size_t)b_idx * 32;

                        vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off +  0), wf0);
                        vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off +  4), wf1);
                        vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off +  8), wf2);
                        vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 12), wf3);
                        vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 16), wf4);
                        vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 20), wf5);
                        vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 24), wf6);
                        vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 28), wf7);
                    }

                    Y[t * rows + r] = vaddvq_f32(vacc0) + b;
                }
#else
                for (; t < T; t++) {
                    const float *x0 = X + (size_t)t * cols;
                    float sum0 = 0.0f;
                    for (int b_idx = 0; b_idx < blocks_per_row; b_idx++) {
                        const int8_t *wb = w_row + (size_t)b_idx * 32;
                        float scale = scales_row[b_idx];
                        size_t off = (size_t)b_idx * 32;
                        float bsum = 0.0f;
                        for (int c = 0; c < 32; c++) {
                            bsum += (float)wb[c] * x0[off + c];
                        }
                        sum0 += bsum * scale;
                    }
                    Y[t * rows + r] = sum0 + b;
                }
#endif
            }
        } else {
            /* Tiled multi-pass path for large dimensions */
            for (int r = r_tile; r < r_end; r++) {
                const float b = bias ? bias[r] : 0.0f;
                for (int t = 0; t < T; t++) {
                    Y[t * rows + r] = b;
                }
            }

            for (int b_start = 0; b_start < blocks_per_row; b_start += TILE_B) {
                int b_end = (b_start + TILE_B <= blocks_per_row) ? b_start + TILE_B : blocks_per_row;

                for (int r = r_tile; r < r_end; r++) {
                    const int8_t *w_row = W + (size_t)r * cols;
                    const float *scales_row = W_scales + (size_t)r * blocks_per_row;
                    int t = 0;

#if defined(EIF_ARCH_X86_64)
                    for (; t <= T - 4; t += 4) {
                        const float *x0 = X + (size_t)(t + 0) * cols;
                        const float *x1 = X + (size_t)(t + 1) * cols;
                        const float *x2 = X + (size_t)(t + 2) * cols;
                        const float *x3 = X + (size_t)(t + 3) * cols;

                        __m256 vacc0 = _mm256_setzero_ps();
                        __m256 vacc1 = _mm256_setzero_ps();
                        __m256 vacc2 = _mm256_setzero_ps();
                        __m256 vacc3 = _mm256_setzero_ps();

                        for (int b_idx = b_start; b_idx < b_end; b_idx++) {
                            const int8_t *wb = w_row + (size_t)b_idx * 32;
                            float scale = scales_row[b_idx];

                            __m256i raw32 = _mm256_loadu_si256((const __m256i *)wb);
                            __m128i rlo = _mm256_castsi256_si128(raw32);
                            __m128i rhi = _mm256_extracti128_si256(raw32, 1);

                            __m256 vs = _mm256_set1_ps(scale);
                            __m256 w0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(rlo)), vs);
                            __m256 w1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(rlo, 8))), vs);
                            __m256 w2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(rhi)), vs);
                            __m256 w3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(rhi, 8))), vs);

                            size_t off = (size_t)b_idx * 32;

                            vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + off +  0), w0, vacc0);
                            vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + off +  8), w1, vacc0);
                            vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + off + 16), w2, vacc0);
                            vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + off + 24), w3, vacc0);

                            vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + off +  0), w0, vacc1);
                            vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + off +  8), w1, vacc1);
                            vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + off + 16), w2, vacc1);
                            vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + off + 24), w3, vacc1);

                            vacc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x2 + off +  0), w0, vacc2);
                            vacc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x2 + off +  8), w1, vacc2);
                            vacc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x2 + off + 16), w2, vacc2);
                            vacc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x2 + off + 24), w3, vacc2);

                            vacc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x3 + off +  0), w0, vacc3);
                            vacc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x3 + off +  8), w1, vacc3);
                            vacc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x3 + off + 16), w2, vacc3);
                            vacc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x3 + off + 24), w3, vacc3);
                        }

                        Y[(t + 0) * rows + r] += hsum_float_8(vacc0);
                        Y[(t + 1) * rows + r] += hsum_float_8(vacc1);
                        Y[(t + 2) * rows + r] += hsum_float_8(vacc2);
                        Y[(t + 3) * rows + r] += hsum_float_8(vacc3);
                    }
                    for (; t <= T - 2; t += 2) {
                        const float *x0 = X + (size_t)(t + 0) * cols;
                        const float *x1 = X + (size_t)(t + 1) * cols;

                        __m256 vacc0 = _mm256_setzero_ps();
                        __m256 vacc1 = _mm256_setzero_ps();

                        for (int b_idx = b_start; b_idx < b_end; b_idx++) {
                            const int8_t *wb = w_row + (size_t)b_idx * 32;
                            float scale = scales_row[b_idx];

                            __m256i raw32 = _mm256_loadu_si256((const __m256i *)wb);
                            __m128i rlo = _mm256_castsi256_si128(raw32);
                            __m128i rhi = _mm256_extracti128_si256(raw32, 1);

                            __m256 vs = _mm256_set1_ps(scale);
                            __m256 w0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(rlo)), vs);
                            __m256 w1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(rlo, 8))), vs);
                            __m256 w2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(rhi)), vs);
                            __m256 w3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(rhi, 8))), vs);

                            size_t off = (size_t)b_idx * 32;

                            vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + off +  0), w0, vacc0);
                            vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + off +  8), w1, vacc0);
                            vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + off + 16), w2, vacc0);
                            vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + off + 24), w3, vacc0);

                            vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + off +  0), w0, vacc1);
                            vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + off +  8), w1, vacc1);
                            vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + off + 16), w2, vacc1);
                            vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + off + 24), w3, vacc1);
                        }

                        Y[(t + 0) * rows + r] += hsum_float_8(vacc0);
                        Y[(t + 1) * rows + r] += hsum_float_8(vacc1);
                    }
                    for (; t < T; t++) {
                        const float *x0 = X + (size_t)t * cols;
                        __m256 vacc0 = _mm256_setzero_ps();

                        for (int b_idx = b_start; b_idx < b_end; b_idx++) {
                            const int8_t *wb = w_row + (size_t)b_idx * 32;
                            float scale = scales_row[b_idx];

                            __m256i raw32 = _mm256_loadu_si256((const __m256i *)wb);
                            __m128i rlo = _mm256_castsi256_si128(raw32);
                            __m128i rhi = _mm256_extracti128_si256(raw32, 1);

                            __m256 vs = _mm256_set1_ps(scale);
                            __m256 w0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(rlo)), vs);
                            __m256 w1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(rlo, 8))), vs);
                            __m256 w2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(rhi)), vs);
                            __m256 w3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(rhi, 8))), vs);

                            size_t off = (size_t)b_idx * 32;

                            vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + off +  0), w0, vacc0);
                            vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + off +  8), w1, vacc0);
                            vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + off + 16), w2, vacc0);
                            vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + off + 24), w3, vacc0);
                        }

                        Y[t * rows + r] += hsum_float_8(vacc0);
                    }
#elif defined(EIF_ARCH_ARM64)
                    for (; t <= T - 4; t += 4) {
                        const float *x0 = X + (size_t)(t + 0) * cols;
                        const float *x1 = X + (size_t)(t + 1) * cols;
                        const float *x2 = X + (size_t)(t + 2) * cols;
                        const float *x3 = X + (size_t)(t + 3) * cols;

                        float32x4_t vacc0 = vdupq_n_f32(0.0f);
                        float32x4_t vacc1 = vdupq_n_f32(0.0f);
                        float32x4_t vacc2 = vdupq_n_f32(0.0f);
                        float32x4_t vacc3 = vdupq_n_f32(0.0f);

                        for (int b_idx = b_start; b_idx < b_end; b_idx++) {
                            const int8_t *wb = w_row + (size_t)b_idx * 32;
                            float scale = scales_row[b_idx];
                            float32x4_t vs = vdupq_n_f32(scale);

                            int8x16_t b0 = vld1q_s8(wb);
                            int8x16_t b1 = vld1q_s8(wb + 16);
                            int16x8_t s0 = vmovl_s8(vget_low_s8(b0));
                            int16x8_t s1 = vmovl_s8(vget_high_s8(b0));
                            int16x8_t s2 = vmovl_s8(vget_low_s8(b1));
                            int16x8_t s3 = vmovl_s8(vget_high_s8(b1));

                            float32x4_t wf0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s0))), vs);
                            float32x4_t wf1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s0))), vs);
                            float32x4_t wf2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s1))), vs);
                            float32x4_t wf3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s1))), vs);
                            float32x4_t wf4 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s2))), vs);
                            float32x4_t wf5 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s2))), vs);
                            float32x4_t wf6 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s3))), vs);
                            float32x4_t wf7 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s3))), vs);

                            size_t off = (size_t)b_idx * 32;

                            vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off +  0), wf0);
                            vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off +  4), wf1);
                            vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off +  8), wf2);
                            vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 12), wf3);
                            vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 16), wf4);
                            vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 20), wf5);
                            vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 24), wf6);
                            vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 28), wf7);

                            vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off +  0), wf0);
                            vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off +  4), wf1);
                            vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off +  8), wf2);
                            vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off + 12), wf3);
                            vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off + 16), wf4);
                            vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off + 20), wf5);
                            vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off + 24), wf6);
                            vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off + 28), wf7);

                            vacc2 = vfmaq_f32(vacc2, vld1q_f32(x2 + off +  0), wf0);
                            vacc2 = vfmaq_f32(vacc2, vld1q_f32(x2 + off +  4), wf1);
                            vacc2 = vfmaq_f32(vacc2, vld1q_f32(x2 + off +  8), wf2);
                            vacc2 = vfmaq_f32(vacc2, vld1q_f32(x2 + off + 12), wf3);
                            vacc2 = vfmaq_f32(vacc2, vld1q_f32(x2 + off + 16), wf4);
                            vacc2 = vfmaq_f32(vacc2, vld1q_f32(x2 + off + 20), wf5);
                            vacc2 = vfmaq_f32(vacc2, vld1q_f32(x2 + off + 24), wf6);
                            vacc2 = vfmaq_f32(vacc2, vld1q_f32(x2 + off + 28), wf7);

                            vacc3 = vfmaq_f32(vacc3, vld1q_f32(x3 + off +  0), wf0);
                            vacc3 = vfmaq_f32(vacc3, vld1q_f32(x3 + off +  4), wf1);
                            vacc3 = vfmaq_f32(vacc3, vld1q_f32(x3 + off +  8), wf2);
                            vacc3 = vfmaq_f32(vacc3, vld1q_f32(x3 + off + 12), wf3);
                            vacc3 = vfmaq_f32(vacc3, vld1q_f32(x3 + off + 16), wf4);
                            vacc3 = vfmaq_f32(vacc3, vld1q_f32(x3 + off + 20), wf5);
                            vacc3 = vfmaq_f32(vacc3, vld1q_f32(x3 + off + 24), wf6);
                            vacc3 = vfmaq_f32(vacc3, vld1q_f32(x3 + off + 28), wf7);
                        }

                        Y[(t + 0) * rows + r] += vaddvq_f32(vacc0);
                        Y[(t + 1) * rows + r] += vaddvq_f32(vacc1);
                        Y[(t + 2) * rows + r] += vaddvq_f32(vacc2);
                        Y[(t + 3) * rows + r] += vaddvq_f32(vacc3);
                    }
                    for (; t <= T - 2; t += 2) {
                        const float *x0 = X + (size_t)(t + 0) * cols;
                        const float *x1 = X + (size_t)(t + 1) * cols;

                        float32x4_t vacc0 = vdupq_n_f32(0.0f);
                        float32x4_t vacc1 = vdupq_n_f32(0.0f);

                        for (int b_idx = b_start; b_idx < b_end; b_idx++) {
                            const int8_t *wb = w_row + (size_t)b_idx * 32;
                            float scale = scales_row[b_idx];
                            float32x4_t vs = vdupq_n_f32(scale);

                            int8x16_t b0 = vld1q_s8(wb);
                            int8x16_t b1 = vld1q_s8(wb + 16);
                            int16x8_t s0 = vmovl_s8(vget_low_s8(b0));
                            int16x8_t s1 = vmovl_s8(vget_high_s8(b0));
                            int16x8_t s2 = vmovl_s8(vget_low_s8(b1));
                            int16x8_t s3 = vmovl_s8(vget_high_s8(b1));

                            float32x4_t wf0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s0))), vs);
                            float32x4_t wf1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s0))), vs);
                            float32x4_t wf2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s1))), vs);
                            float32x4_t wf3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s1))), vs);
                            float32x4_t wf4 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s2))), vs);
                            float32x4_t wf5 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s2))), vs);
                            float32x4_t wf6 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s3))), vs);
                            float32x4_t wf7 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s3))), vs);

                            size_t off = (size_t)b_idx * 32;

                            vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off +  0), wf0);
                            vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off +  4), wf1);
                            vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off +  8), wf2);
                            vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 12), wf3);
                            vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 16), wf4);
                            vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 20), wf5);
                            vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 24), wf6);
                            vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 28), wf7);

                            vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off +  0), wf0);
                            vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off +  4), wf1);
                            vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off +  8), wf2);
                            vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off + 12), wf3);
                            vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off + 16), wf4);
                            vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off + 20), wf5);
                            vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off + 24), wf6);
                            vacc1 = vfmaq_f32(vacc1, vld1q_f32(x1 + off + 28), wf7);
                        }

                        Y[(t + 0) * rows + r] += vaddvq_f32(vacc0);
                        Y[(t + 1) * rows + r] += vaddvq_f32(vacc1);
                    }
                    for (; t < T; t++) {
                        const float *x0 = X + (size_t)t * cols;
                        float32x4_t vacc0 = vdupq_n_f32(0.0f);

                        for (int b_idx = b_start; b_idx < b_end; b_idx++) {
                            const int8_t *wb = w_row + (size_t)b_idx * 32;
                            float scale = scales_row[b_idx];
                            float32x4_t vs = vdupq_n_f32(scale);

                            int8x16_t b0 = vld1q_s8(wb);
                            int8x16_t b1 = vld1q_s8(wb + 16);
                            int16x8_t s0 = vmovl_s8(vget_low_s8(b0));
                            int16x8_t s1 = vmovl_s8(vget_high_s8(b0));
                            int16x8_t s2 = vmovl_s8(vget_low_s8(b1));
                            int16x8_t s3 = vmovl_s8(vget_high_s8(b1));

                            float32x4_t wf0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s0))), vs);
                            float32x4_t wf1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s0))), vs);
                            float32x4_t wf2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s1))), vs);
                            float32x4_t wf3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s1))), vs);
                            float32x4_t wf4 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s2))), vs);
                            float32x4_t wf5 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s2))), vs);
                            float32x4_t wf6 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s3))), vs);
                            float32x4_t wf7 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s3))), vs);

                            size_t off = (size_t)b_idx * 32;

                            vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off +  0), wf0);
                            vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off +  4), wf1);
                            vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off +  8), wf2);
                            vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 12), wf3);
                            vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 16), wf4);
                            vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 20), wf5);
                            vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 24), wf6);
                            vacc0 = vfmaq_f32(vacc0, vld1q_f32(x0 + off + 28), wf7);
                        }

                        Y[t * rows + r] += vaddvq_f32(vacc0);
                    }
#else
                    for (; t < T; t++) {
                        const float *x0 = X + (size_t)t * cols;
                        float sum0 = 0.0f;
                        for (int b_idx = b_start; b_idx < b_end; b_idx++) {
                            const int8_t *wb = w_row + (size_t)b_idx * 32;
                            float scale = scales_row[b_idx];
                            size_t off = (size_t)b_idx * 32;
                            float bsum = 0.0f;
                            for (int c = 0; c < 32; c++) {
                                bsum += (float)wb[c] * x0[off + c];
                            }
                            sum0 += bsum * scale;
                        }
                        Y[t * rows + r] += sum0;
                    }
#endif
                }
            }
        }
    }
}

static void bert_gemm_f32(
    float *Y,
    const float *W,
    const float *X,
    const float *bias,
    int T,
    int rows,
    int cols)
{
#pragma omp parallel for schedule(static) if((int64_t)rows * cols >= 4096)
    for (int r = 0; r < rows; r++) {
        const float *w_row = W + (size_t)r * cols;
        const float b = bias ? bias[r] : 0.0f;
        int t = 0;

#if defined(EIF_ARCH_X86_64)
        for (; t <= T - 4; t += 4) {
            const float *x0 = X + (size_t)(t + 0) * cols;
            const float *x1 = X + (size_t)(t + 1) * cols;
            const float *x2 = X + (size_t)(t + 2) * cols;
            const float *x3 = X + (size_t)(t + 3) * cols;

            __m256 vacc0 = _mm256_setzero_ps();
            __m256 vacc1 = _mm256_setzero_ps();
            __m256 vacc2 = _mm256_setzero_ps();
            __m256 vacc3 = _mm256_setzero_ps();

            for (int c = 0; c <= cols - 8; c += 8) {
                __m256 wv = _mm256_loadu_ps(&w_row[c]);
                vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(&x0[c]), wv, vacc0);
                vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(&x1[c]), wv, vacc1);
                vacc2 = _mm256_fmadd_ps(_mm256_loadu_ps(&x2[c]), wv, vacc2);
                vacc3 = _mm256_fmadd_ps(_mm256_loadu_ps(&x3[c]), wv, vacc3);
            }

            float s0 = hsum_float_8(vacc0);
            float s1 = hsum_float_8(vacc1);
            float s2 = hsum_float_8(vacc2);
            float s3 = hsum_float_8(vacc3);

            for (int c = (cols & ~7); c < cols; c++) {
                float w = w_row[c];
                s0 += w * x0[c];
                s1 += w * x1[c];
                s2 += w * x2[c];
                s3 += w * x3[c];
            }

            Y[(t + 0) * rows + r] = s0 + b;
            Y[(t + 1) * rows + r] = s1 + b;
            Y[(t + 2) * rows + r] = s2 + b;
            Y[(t + 3) * rows + r] = s3 + b;
        }

        for (; t <= T - 2; t += 2) {
            const float *x0 = X + (size_t)(t + 0) * cols;
            const float *x1 = X + (size_t)(t + 1) * cols;

            __m256 vacc0 = _mm256_setzero_ps();
            __m256 vacc1 = _mm256_setzero_ps();

            for (int c = 0; c <= cols - 8; c += 8) {
                __m256 wv = _mm256_loadu_ps(&w_row[c]);
                vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(&x0[c]), wv, vacc0);
                vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(&x1[c]), wv, vacc1);
            }

            float s0 = hsum_float_8(vacc0);
            float s1 = hsum_float_8(vacc1);

            for (int c = (cols & ~7); c < cols; c++) {
                float w = w_row[c];
                s0 += w * x0[c];
                s1 += w * x1[c];
            }

            Y[(t + 0) * rows + r] = s0 + b;
            Y[(t + 1) * rows + r] = s1 + b;
        }

        for (; t < T; t++) {
            const float *x0 = X + (size_t)t * cols;
            __m256 vacc0 = _mm256_setzero_ps();

            for (int c = 0; c <= cols - 8; c += 8) {
                __m256 wv = _mm256_loadu_ps(&w_row[c]);
                vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(&x0[c]), wv, vacc0);
            }

            float s0 = hsum_float_8(vacc0);
            for (int c = (cols & ~7); c < cols; c++) {
                s0 += w_row[c] * x0[c];
            }

            Y[t * rows + r] = s0 + b;
        }
#elif defined(EIF_ARCH_ARM64)
        for (; t <= T - 4; t += 4) {
            const float *x0 = X + (size_t)(t + 0) * cols;
            const float *x1 = X + (size_t)(t + 1) * cols;
            const float *x2 = X + (size_t)(t + 2) * cols;
            const float *x3 = X + (size_t)(t + 3) * cols;

            float32x4_t vacc0 = vdupq_n_f32(0.0f);
            float32x4_t vacc1 = vdupq_n_f32(0.0f);
            float32x4_t vacc2 = vdupq_n_f32(0.0f);
            float32x4_t vacc3 = vdupq_n_f32(0.0f);

            for (int c = 0; c <= cols - 4; c += 4) {
                float32x4_t wv = vld1q_f32(&w_row[c]);
                vacc0 = vfmaq_f32(vacc0, vld1q_f32(&x0[c]), wv);
                vacc1 = vfmaq_f32(vacc1, vld1q_f32(&x1[c]), wv);
                vacc2 = vfmaq_f32(vacc2, vld1q_f32(&x2[c]), wv);
                vacc3 = vfmaq_f32(vacc3, vld1q_f32(&x3[c]), wv);
            }

            float s0 = vaddvq_f32(vacc0);
            float s1 = vaddvq_f32(vacc1);
            float s2 = vaddvq_f32(vacc2);
            float s3 = vaddvq_f32(vacc3);

            for (int c = (cols & ~3); c < cols; c++) {
                float w = w_row[c];
                s0 += w * x0[c];
                s1 += w * x1[c];
                s2 += w * x2[c];
                s3 += w * x3[c];
            }

            Y[(t + 0) * rows + r] = s0 + b;
            Y[(t + 1) * rows + r] = s1 + b;
            Y[(t + 2) * rows + r] = s2 + b;
            Y[(t + 3) * rows + r] = s3 + b;
        }

        for (; t <= T - 2; t += 2) {
            const float *x0 = X + (size_t)(t + 0) * cols;
            const float *x1 = X + (size_t)(t + 1) * cols;
            float32x4_t vacc0 = vdupq_n_f32(0.0f);
            float32x4_t vacc1 = vdupq_n_f32(0.0f);

            for (int c = 0; c <= cols - 4; c += 4) {
                float32x4_t wv = vld1q_f32(&w_row[c]);
                vacc0 = vfmaq_f32(vacc0, vld1q_f32(&x0[c]), wv);
                vacc1 = vfmaq_f32(vacc1, vld1q_f32(&x1[c]), wv);
            }

            float s0 = vaddvq_f32(vacc0);
            float s1 = vaddvq_f32(vacc1);

            for (int c = (cols & ~3); c < cols; c++) {
                float w = w_row[c];
                s0 += w * x0[c];
                s1 += w * x1[c];
            }

            Y[(t + 0) * rows + r] = s0 + b;
            Y[(t + 1) * rows + r] = s1 + b;
        }

        for (; t < T; t++) {
            const float *x0 = X + (size_t)t * cols;
            float32x4_t vacc0 = vdupq_n_f32(0.0f);
            for (int c = 0; c <= cols - 4; c += 4) {
                float32x4_t wv = vld1q_f32(&w_row[c]);
                vacc0 = vfmaq_f32(vacc0, vld1q_f32(&x0[c]), wv);
            }
            float s0 = vaddvq_f32(vacc0);
            for (int c = (cols & ~3); c < cols; c++) {
                s0 += w_row[c] * x0[c];
            }
            Y[t * rows + r] = s0 + b;
        }
#else
        for (; t < T; t++) {
            const float *x0 = X + (size_t)t * cols;
            float sum0 = 0.0f;
            for (int c = 0; c < cols; c++) {
                sum0 += w_row[c] * x0[c];
            }
            Y[t * rows + r] = sum0 + b;
        }
#endif
    }
}

/* Hardware-Accelerated INT8 Dot-Product GEMM with 4-Token Unroll & Cache Tiling */
static void bert_gemm_int8_gguf(
    float *Y,               /* Output: [T, rows] */
    const uint8_t *W,       /* Interleaved Q8_0: [rows, (cols / 32) * 34] */
    const float *X,         /* Input:   [T, cols] */
    const float *bias,      /* Bias:    [rows] (optional) */
    int T,
    int rows,
    int cols)
{
    const int blocks_per_row = cols / 32;
    const size_t row_stride_bytes = (size_t)blocks_per_row * 34;
    const int TILE_R = 64;

    /* Dynamic INT8 Block Quantization of input X */
    size_t xq_sz = (size_t)T * cols * sizeof(int8_t);
    size_t xs_sz = (size_t)T * blocks_per_row * sizeof(float);
    int8_t stack_xq[8192];
    float  stack_xs[1024];

    int8_t *xq = (xq_sz <= sizeof(stack_xq)) ? stack_xq : (int8_t *)malloc(xq_sz);
    float  *xs = (xs_sz <= sizeof(stack_xs)) ? stack_xs : (float *)malloc(xs_sz);
    if (!xq || !xs) {
        if (xq && xq != stack_xq) free(xq);
        if (xs && xs != stack_xs) free(xs);
        return;
    }

    for (int t = 0; t < T; t++) {
        const float *xt = X + (size_t)t * cols;
        int8_t *xqt = xq + (size_t)t * cols;
        float *xst = xs + (size_t)t * blocks_per_row;
        for (int b = 0; b < blocks_per_row; b++) {
            const float *xb = xt + b * 32;
            float max_val = 0.0f;
            for (int i = 0; i < 32; i++) {
                float a = fabsf(xb[i]);
                if (a > max_val) max_val = a;
            }
            float d = max_val / 127.0f;
            float id = d > 1e-12f ? (127.0f / max_val) : 0.0f;
            xst[b] = d;
            for (int i = 0; i < 32; i++) {
                int v = (int)roundf(xb[i] * id);
                xqt[b * 32 + i] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
            }
        }
    }

#ifdef _OPENMP
    int max_th = omp_get_max_threads();
    int gemm_th = max_th;
#else
    int gemm_th = 1;
#endif
#pragma omp parallel for schedule(static) if((int64_t)rows * cols >= 4096) num_threads(gemm_th)
    for (int r_tile = 0; r_tile < rows; r_tile += TILE_R) {
        int r_end = (r_tile + TILE_R <= rows) ? r_tile + TILE_R : rows;

        for (int r = r_tile; r < r_end; r++) {
            const uint8_t *w_row = W + (size_t)r * row_stride_bytes;
            const float b = bias ? bias[r] : 0.0f;
            int t = 0;

            /* 4-way token unroll: Load W once and multiply against 4 tokens */
            for (; t <= T - 4; t += 4) {
                const int8_t *xq0 = xq + (size_t)(t + 0) * cols;
                const int8_t *xq1 = xq + (size_t)(t + 1) * cols;
                const int8_t *xq2 = xq + (size_t)(t + 2) * cols;
                const int8_t *xq3 = xq + (size_t)(t + 3) * cols;
                const float *xs0 = xs + (size_t)(t + 0) * blocks_per_row;
                const float *xs1 = xs + (size_t)(t + 1) * blocks_per_row;
                const float *xs2 = xs + (size_t)(t + 2) * blocks_per_row;
                const float *xs3 = xs + (size_t)(t + 3) * blocks_per_row;
                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

                for (int b_idx = 0; b_idx < blocks_per_row; b_idx++) {
                    const uint8_t *blk = w_row + (size_t)b_idx * 34;
                    float w_scale = fp16_to_fp32(*(const uint16_t *)blk);
                    const int8_t *wb = (const int8_t *)(blk + 2);
                    size_t off = (size_t)b_idx * 32;

#if defined(EIF_ARCH_ARM64)
                    int32_t s0 = arm_dot32(wb, xq0 + off);
                    int32_t s1 = arm_dot32(wb, xq1 + off);
                    int32_t s2 = arm_dot32(wb, xq2 + off);
                    int32_t s3 = arm_dot32(wb, xq3 + off);

                    acc0 += (float)s0 * (w_scale * xs0[b_idx]);
                    acc1 += (float)s1 * (w_scale * xs1[b_idx]);
                    acc2 += (float)s2 * (w_scale * xs2[b_idx]);
                    acc3 += (float)s3 * (w_scale * xs3[b_idx]);
#elif defined(EIF_ARCH_X86_64)
                    __m128i w_raw0 = _mm_loadu_si128((const __m128i *)wb);
                    __m128i w_raw1 = _mm_loadu_si128((const __m128i *)(wb + 16));
                    __m256i w16_0 = _mm256_cvtepi8_epi16(w_raw0);
                    __m256i w16_1 = _mm256_cvtepi8_epi16(w_raw1);

                    __m256i p0 = _mm256_add_epi32(_mm256_madd_epi16(w16_0, _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(xq0 + off)))),
                                                  _mm256_madd_epi16(w16_1, _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(xq0 + off + 16)))));
                    __m256i p1 = _mm256_add_epi32(_mm256_madd_epi16(w16_0, _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(xq1 + off)))),
                                                  _mm256_madd_epi16(w16_1, _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(xq1 + off + 16)))));
                    __m256i p2 = _mm256_add_epi32(_mm256_madd_epi16(w16_0, _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(xq2 + off)))),
                                                  _mm256_madd_epi16(w16_1, _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(xq2 + off + 16)))));
                    __m256i p3 = _mm256_add_epi32(_mm256_madd_epi16(w16_0, _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(xq3 + off)))),
                                                  _mm256_madd_epi16(w16_1, _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(xq3 + off + 16)))));

                    acc0 += (float)hsum_epi32_8(p0) * (w_scale * xs0[b_idx]);
                    acc1 += (float)hsum_epi32_8(p1) * (w_scale * xs1[b_idx]);
                    acc2 += (float)hsum_epi32_8(p2) * (w_scale * xs2[b_idx]);
                    acc3 += (float)hsum_epi32_8(p3) * (w_scale * xs3[b_idx]);
#else
                    int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                    for (int i = 0; i < 32; i++) {
                        int32_t w_val = (int32_t)wb[i];
                        s0 += w_val * (int32_t)xq0[off + i];
                        s1 += w_val * (int32_t)xq1[off + i];
                        s2 += w_val * (int32_t)xq2[off + i];
                        s3 += w_val * (int32_t)xq3[off + i];
                    }
                    acc0 += (float)s0 * (w_scale * xs0[b_idx]);
                    acc1 += (float)s1 * (w_scale * xs1[b_idx]);
                    acc2 += (float)s2 * (w_scale * xs2[b_idx]);
                    acc3 += (float)s3 * (w_scale * xs3[b_idx]);
#endif
                }
                Y[(t + 0) * rows + r] = acc0 + b;
                Y[(t + 1) * rows + r] = acc1 + b;
                Y[(t + 2) * rows + r] = acc2 + b;
                Y[(t + 3) * rows + r] = acc3 + b;
            }

            /* Remainder tokens */
            for (; t < T; t++) {
                const int8_t *xqt = xq + (size_t)t * cols;
                const float *xst = xs + (size_t)t * blocks_per_row;
                float acc = 0.0f;

                for (int b_idx = 0; b_idx < blocks_per_row; b_idx++) {
                    const uint8_t *blk = w_row + (size_t)b_idx * 34;
                    float w_scale = fp16_to_fp32(*(const uint16_t *)blk);
                    const int8_t *wb = (const int8_t *)(blk + 2);
                    const int8_t *xb = xqt + b_idx * 32;

#if defined(EIF_ARCH_ARM64)
                    int32_t sum_int = arm_dot32(wb, xb);
#elif defined(EIF_ARCH_X86_64)
                    __m128i w_raw0 = _mm_loadu_si128((const __m128i *)wb);
                    __m128i w_raw1 = _mm_loadu_si128((const __m128i *)(wb + 16));
                    __m128i x_raw0 = _mm_loadu_si128((const __m128i *)xb);
                    __m128i x_raw1 = _mm_loadu_si128((const __m128i *)(xb + 16));

                    __m256i w16_0 = _mm256_cvtepi8_epi16(w_raw0);
                    __m256i w16_1 = _mm256_cvtepi8_epi16(w_raw1);
                    __m256i x16_0 = _mm256_cvtepi8_epi16(x_raw0);
                    __m256i x16_1 = _mm256_cvtepi8_epi16(x_raw1);

                    __m256i p0 = _mm256_madd_epi16(w16_0, x16_0);
                    __m256i p1 = _mm256_madd_epi16(w16_1, x16_1);
                    int32_t sum_int = hsum_epi32_8(_mm256_add_epi32(p0, p1));
#else
                    int32_t sum_int = 0;
                    for (int i = 0; i < 32; i++) {
                        sum_int += (int32_t)wb[i] * (int32_t)xb[i];
                    }
#endif
                    acc += (float)sum_int * (w_scale * xst[b_idx]);
                }
                Y[t * rows + r] = acc + b;
            }
        }
    }

    if (xq != stack_xq) free(xq);
    if (xs != stack_xs) free(xs);
}

/* Dispatcher for Batched GEMM */
static inline void bert_gemm(
    float *Y,
    const void *W,
    const float *W_scales,
    const float *X,
    const float *bias,
    int T,
    int rows,
    int cols,
    int qtype)
{
    if (qtype == 2) {
        bert_gemm_int8_gguf(Y, (const uint8_t *)W, X, bias, T, rows, cols);
    } else if (qtype == 1) {
        bert_gemm_int8(Y, (const int8_t *)W, W_scales, X, bias, T, rows, cols);
    } else {
        bert_gemm_f32(Y, (const float *)W, X, bias, T, rows, cols);
    }
}

/* =============================================================================
 * WordPiece / SentencePiece Subword Tokenizer
 * ============================================================================= */

static inline uint32_t hash_str(const char *s) {
    uint32_t h = 2166136261u;
    while (*s) {
        h = (h ^ (uint8_t)*s++) * 16777619u;
    }
    return h;
}

#define EBERT_HASH_MASK 0x1FFFF /* 131,072 slots */

static int find_token_id(const eif_bert_vocab_t *vocab, const char *token_str)
{
    if (!vocab || !vocab->tokens) return -1;
    if (vocab->hash_table) {
        uint32_t mask = vocab->hash_mask ? vocab->hash_mask : (uint32_t)EBERT_HASH_MASK;
        uint32_t idx = hash_str(token_str) & mask;
        for (int step = 0; step < 256; step++) {
            int32_t tid = vocab->hash_table[(idx + step) & mask];
            if (tid == -1) return -1;
            if (strcmp(vocab->tokens[tid], token_str) == 0) return (int)tid;
        }
        return -1;
    }
    for (int i = 0; i < vocab->vocab_size; i++) {
        if (vocab->tokens[i] && strcmp(vocab->tokens[i], token_str) == 0) {
            return i;
        }
    }
    return -1;
}

int eif_bert_tokenize(const eif_bert_t *bert, const char *text, int32_t *out_tokens, int max_tokens)
{
    if (!bert || !text || !out_tokens || max_tokens < 3) return -1;
    const eif_bert_vocab_t *v = &bert->vocab;

    if (v->tok_type == EBERT_TOKENIZER_BPE) {
        int bos = (v->cls_id >= 0) ? 1 : 0;
        int eos = (v->sep_id >= 0) ? 1 : 0;
        return eif_bpe_tokenizer_encode(&v->bpe_tok, text, (int8_t)bos, (int8_t)eos, out_tokens, max_tokens);
    }

    if (v->tok_type == EBERT_TOKENIZER_SPM) {
        int n_out = 0;
        out_tokens[n_out++] = (v->cls_id >= 0) ? v->cls_id : 0;

        const char *p = text;
        char word[256];

        while (*p && n_out < max_tokens - 1) {
            while (*p && isspace((unsigned char)*p)) p++;
            if (!*p) break;

            if (ispunct((unsigned char)*p)) {
                char pchar = *p++;
                char punct_str[8];
                punct_str[0] = (char)0xE2; punct_str[1] = (char)0x96; punct_str[2] = (char)0x81;
                punct_str[3] = pchar; punct_str[4] = '\0';
                int tid = find_token_id(v, punct_str);
                if (tid < 0) {
                    punct_str[0] = pchar; punct_str[1] = '\0';
                    tid = find_token_id(v, punct_str);
                }
                if (tid >= 0) out_tokens[n_out++] = tid;
                continue;
            }

            int wlen = 0;
            while (*p && !isspace((unsigned char)*p) && !ispunct((unsigned char)*p) && wlen < 200) {
                word[wlen++] = *p++;
            }
            word[wlen] = '\0';
            if (wlen == 0) continue;

            int start = 0;
            while (start < wlen && n_out < max_tokens - 1) {
                int end = wlen;
                int found_id = -1;
                while (end > start) {
                    int sub_len = end - start;
                    char sub[256];

                    if (start == 0) {
                        sub[0] = (char)0xE2; sub[1] = (char)0x96; sub[2] = (char)0x81;
                        memcpy(sub + 3, word + start, sub_len);
                        sub[sub_len + 3] = '\0';
                        found_id = find_token_id(v, sub);
                        if (found_id < 0) {
                            memcpy(sub, word + start, sub_len);
                            sub[sub_len] = '\0';
                            found_id = find_token_id(v, sub);
                        }
                    } else {
                        sub[0] = (char)0xE2; sub[1] = (char)0x96; sub[2] = (char)0x81;
                        sub[3] = (char)0xE2; sub[4] = (char)0x96; sub[5] = (char)0x81;
                        memcpy(sub + 6, word + start, sub_len);
                        sub[sub_len + 6] = '\0';
                        found_id = find_token_id(v, sub);

                        if (found_id < 0) {
                            sub[0] = (char)0xE2; sub[1] = (char)0x96; sub[2] = (char)0x81;
                            memcpy(sub + 3, word + start, sub_len);
                            sub[sub_len + 3] = '\0';
                            found_id = find_token_id(v, sub);
                        }
                        if (found_id < 0) {
                            memcpy(sub, word + start, sub_len);
                            sub[sub_len] = '\0';
                            found_id = find_token_id(v, sub);
                        }
                    }

                    if (found_id >= 0) {
                        out_tokens[n_out++] = found_id;
                        start = end;
                        break;
                    }
                    end--;
                }
                if (found_id < 0) {
                    out_tokens[n_out++] = (v->unk_id >= 0) ? v->unk_id : 3;
                    start++;
                }
            }
        }

        if (n_out < max_tokens) {
            out_tokens[n_out++] = (v->sep_id >= 0) ? v->sep_id : 2;
        }
        return n_out;
    }

    int n_out = 0;
    out_tokens[n_out++] = (v->cls_id >= 0) ? v->cls_id : 101; /* [CLS] */

    char word[256];
    const char *p = text;

    while (*p && n_out < max_tokens - 1) {
        /* Skip whitespace */
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;

        /* Punctuation handling: isolate ASCII punctuation */
        if (ispunct((unsigned char)*p)) {
            char pchar = *p++;
            char punct_str[8];
            int tid = -1;
            if (v->is_spm_prefix) {
                punct_str[0] = (char)0xE2;
                punct_str[1] = (char)0x96;
                punct_str[2] = (char)0x81;
                punct_str[3] = pchar;
                punct_str[4] = '\0';
                tid = find_token_id(v, punct_str);
            }
            if (tid < 0) {
                punct_str[0] = pchar;
                punct_str[1] = '\0';
                tid = find_token_id(v, punct_str);
            }
            if (tid >= 0) {
                out_tokens[n_out++] = tid;
            }
            continue;
        }

        /* Read word until space or punctuation */
        int wlen = 0;
        while (*p && !isspace((unsigned char)*p) && !ispunct((unsigned char)*p) && wlen < 200) {
            word[wlen++] = (char)tolower((unsigned char)*p++);
        }
        word[wlen] = '\0';
        if (wlen == 0) continue;

        /* Subword matching */
        int start = 0;
        while (start < wlen && n_out < max_tokens - 1) {
            int end = wlen;
            int found_id = -1;
            while (end > start) {
                char sub[256];
                int sub_len = end - start;
                if (start == 0) {
                    if (v->is_spm_prefix) {
                        sub[0] = (char)0xE2;
                        sub[1] = (char)0x96;
                        sub[2] = (char)0x81;
                        memcpy(sub + 3, word + start, sub_len);
                        sub[sub_len + 3] = '\0';
                    } else {
                        memcpy(sub, word + start, sub_len);
                        sub[sub_len] = '\0';
                    }
                } else {
                    if (v->is_spm_prefix) {
                        memcpy(sub, word + start, sub_len);
                        sub[sub_len] = '\0';
                    } else {
                        sub[0] = '#';
                        sub[1] = '#';
                        memcpy(sub + 2, word + start, sub_len);
                        sub[sub_len + 2] = '\0';
                    }
                }
                found_id = find_token_id(v, sub);
                if (found_id >= 0) {
                    out_tokens[n_out++] = found_id;
                    start = end;
                    break;
                }
                end--;
            }
            if (found_id < 0) {
                /* If start == 0 and spm prefix didn't match, fallback without spm prefix */
                if (start == 0 && v->is_spm_prefix) {
                    end = wlen;
                    while (end > start) {
                        char sub[256];
                        int sub_len = end - start;
                        memcpy(sub, word + start, sub_len);
                        sub[sub_len] = '\0';
                        found_id = find_token_id(v, sub);
                        if (found_id >= 0) {
                            out_tokens[n_out++] = found_id;
                            start = end;
                            break;
                        }
                        end--;
                    }
                }
                if (found_id < 0) {
                    out_tokens[n_out++] = (v->unk_id >= 0) ? v->unk_id : 100;
                    break;
                }
            }
        }
    }

    if (n_out < max_tokens) {
        out_tokens[n_out++] = (v->sep_id >= 0) ? v->sep_id : 102; /* [SEP] */
    }

    return n_out;
}

/* =============================================================================
 * Model Loading (Zero-Copy mmap)
 * ============================================================================= */

static inline size_t align64(size_t sz)
{
    return (sz + 63) & ~63;
}


/* Common setup for vocab hash table and scratch memory */
static int bert_setup_vocab_and_scratch(eif_bert_t *bert)
{
    int dim = bert->config.dim;
    int inter_dim = bert->config.intermediate_dim;
    int max_t = bert->config.max_seq_len;

    /* Detect tokenizer type:
     * 1. BPE (RoBERTa / GPT-2 / Granite): presence of byte-level Ġ (\xc4\xa0) tokens
     * 2. SentencePiece (XLM-RoBERTa / Multilingual-E5): presence of \xe2\x96\x81 tokens
     * 3. WordPiece (Classic MiniLM / BERT): ## subwords
     */
    int bpe_prefix_count = 0;
    int spm_prefix_count = 0;
    for (int t = 0; t < bert->vocab.vocab_size && t < 5000; t++) {
        if (!bert->vocab.tokens[t]) continue;
        const uint8_t *tk = (const uint8_t *)bert->vocab.tokens[t];
        if (tk[0] == 0xC4 && tk[1] == 0xA0) {
            bpe_prefix_count++;
        } else if (tk[0] == 0xE2 && tk[1] == 0x96 && tk[2] == 0x81) {
            spm_prefix_count++;
        }
    }

    if (bpe_prefix_count > 50) {
        bert->vocab.tok_type = EBERT_TOKENIZER_BPE;
        bert->vocab.is_spm_prefix = false;
        eif_bpe_tokenizer_init_from_vocab(&bert->vocab.bpe_tok, bert->vocab.tokens, NULL, bert->vocab.vocab_size, bert->vocab.cls_id, bert->vocab.sep_id);
    } else if (spm_prefix_count > 50) {
        bert->vocab.tok_type = EBERT_TOKENIZER_SPM;
        bert->vocab.is_spm_prefix = true;
    } else {
        bert->vocab.tok_type = EBERT_TOKENIZER_WORDPIECE;
        bert->vocab.is_spm_prefix = false;
    }

    /* Build fast O(1) hash lookup table dynamically sized to prevent collisions */
    int hash_size = 131072;
    while (hash_size < bert->vocab.vocab_size * 2) {
        hash_size <<= 1;
    }
    bert->vocab.hash_size = hash_size;
    bert->vocab.hash_mask = (uint32_t)hash_size - 1;

    bert->vocab.hash_table = (int32_t *)malloc((size_t)hash_size * sizeof(int32_t));
    if (!bert->vocab.hash_table) return -7;
    for (int i = 0; i < hash_size; i++) bert->vocab.hash_table[i] = -1;
    for (int t = 0; t < bert->vocab.vocab_size; t++) {
        if (!bert->vocab.tokens[t]) continue;
        uint32_t idx = hash_str(bert->vocab.tokens[t]) & bert->vocab.hash_mask;
        for (int step = 0; step < 256; step++) {
            uint32_t slot = (idx + step) & bert->vocab.hash_mask;
            if (bert->vocab.hash_table[slot] == -1) {
                bert->vocab.hash_table[slot] = (int32_t)t;
                break;
            }
        }
    }

    /* Allocate Scratch Activation Memory */
    bert->scratch_seq_x  = (float *)malloc((size_t)max_t * dim * sizeof(float));
    bert->scratch_seq_xb = (float *)malloc((size_t)max_t * dim * sizeof(float));
    bert->scratch_q      = (float *)malloc((size_t)max_t * dim * sizeof(float));
    bert->scratch_k      = (float *)malloc((size_t)max_t * dim * sizeof(float));
    bert->scratch_v      = (float *)malloc((size_t)max_t * dim * sizeof(float));
    bert->scratch_att    = (float *)malloc((size_t)bert->config.n_heads * max_t * max_t * sizeof(float));
    /* Gated FFN (GEGLU) needs a second [T x inter] buffer for the gate branch */
    size_t inter_mult = bert->weights.ffn_gate_w ? 2 : 1;
    bert->scratch_inter  = (float *)malloc((size_t)max_t * inter_dim * inter_mult * sizeof(float));
    bert->scratch_proj   = (float *)malloc((size_t)max_t * dim * sizeof(float));

    if (!bert->scratch_seq_x || !bert->scratch_seq_xb || !bert->scratch_q ||
        !bert->scratch_k || !bert->scratch_v || !bert->scratch_att ||
        !bert->scratch_inter || !bert->scratch_proj) {
        return -8;
    }

    bert->is_initialized = true;
    return 0;
}

/* 1. Native GGUF BERT Loader */
static int eif_bert_load_gguf(eif_bert_t *bert, const uint8_t *addr, size_t file_sz)
{
    const uint8_t *p = addr;
    uint32_t magic = gguf_read_u32(&p);
    uint32_t ver   = gguf_read_u32(&p);
    uint64_t n_tensors = gguf_read_u64(&p);
    uint64_t n_kv      = gguf_read_u64(&p);

    if (magic != GGUF_MAGIC) return -5;
    (void)ver;
    (void)file_sz;

    uint32_t alignment = 32;
    int dim = 384, inter_dim = 1536, n_layers = 6, n_heads = 12, max_seq = 512;

    for (uint64_t i = 0; i < n_kv; i++) {
        char key[128];
        gguf_read_str(&p, key, sizeof(key));
        uint32_t vtype = gguf_read_u32(&p);

        if (strcmp(key, "general.alignment") == 0) {
            alignment = gguf_read_u32(&p);
        } else if (strstr(key, "embedding_length")) {
            dim = (int)gguf_read_u32(&p);
        } else if (strstr(key, "feed_forward_length")) {
            inter_dim = (int)gguf_read_u32(&p);
        } else if (strstr(key, "block_count")) {
            n_layers = (int)gguf_read_u32(&p);
        } else if (strstr(key, "head_count")) {
            n_heads = (int)gguf_read_u32(&p);
        } else if (strstr(key, "context_length")) {
            max_seq = (int)gguf_read_u32(&p);
        } else if (strstr(key, "attention.alibi")) {
            if (vtype == 7) {
                bert->config.use_alibi = (*p++ != 0);
            } else if (vtype == 4) {
                bert->config.use_alibi = (gguf_read_u32(&p) != 0);
            } else {
                p += 1;
            }
        } else if (strstr(key, "pooling_type")) {
            bert->config.pooling_type = (int)gguf_read_u32(&p);
        } else if (strcmp(key, "tokenizer.ggml.bos_token_id") == 0) {
            bert->vocab.cls_id = (int)gguf_read_u32(&p);
        } else if (strcmp(key, "tokenizer.ggml.eos_token_id") == 0) {
            bert->vocab.sep_id = (int)gguf_read_u32(&p);
        } else if (strcmp(key, "tokenizer.ggml.unknown_token_id") == 0) {
            bert->vocab.unk_id = (int)gguf_read_u32(&p);
        } else if (strcmp(key, "tokenizer.ggml.padding_token_id") == 0) {
            bert->vocab.pad_id = (int)gguf_read_u32(&p);
        } else if (strcmp(key, "tokenizer.ggml.tokens") == 0) {
            uint32_t itype = gguf_read_u32(&p);
            uint64_t count = gguf_read_u64(&p);
            (void)itype;

            bert->vocab.vocab_size = (int)count;
            bert->vocab.tokens = (char **)malloc((size_t)count * sizeof(char *));
            if (bert->vocab.cls_id < 0) bert->vocab.cls_id = 101;
            if (bert->vocab.sep_id < 0) bert->vocab.sep_id = 102;
            if (bert->vocab.unk_id < 0) bert->vocab.unk_id = 100;
            if (bert->vocab.pad_id < 0) bert->vocab.pad_id = 0;

            for (uint64_t t = 0; t < count; t++) {
                uint64_t slen = gguf_read_u64(&p);
                bert->vocab.tokens[t] = (char *)malloc(slen + 1);
                memcpy(bert->vocab.tokens[t], p, slen);
                bert->vocab.tokens[t][slen] = '\0';
                p += slen;

                if (strcmp(bert->vocab.tokens[t], "[CLS]") == 0 || strcmp(bert->vocab.tokens[t], "<s>") == 0) bert->vocab.cls_id = (int)t;
                else if (strcmp(bert->vocab.tokens[t], "[SEP]") == 0 || strcmp(bert->vocab.tokens[t], "</s>") == 0) bert->vocab.sep_id = (int)t;
                else if (strcmp(bert->vocab.tokens[t], "[UNK]") == 0 || strcmp(bert->vocab.tokens[t], "<unk>") == 0) bert->vocab.unk_id = (int)t;
                else if (strcmp(bert->vocab.tokens[t], "[PAD]") == 0 || strcmp(bert->vocab.tokens[t], "<pad>") == 0) bert->vocab.pad_id = (int)t;
            }
        } else {
            /* Skip metadata value */
            switch (vtype) {
                case 0: case 1: case 7: p += 1; break;
                case 2: case 3: p += 2; break;
                case 4: case 5: case 6: p += 4; break;
                case 8: { uint64_t l = gguf_read_u64(&p); p += l; break; }
                case 10: case 11: case 12: p += 8; break;
                case 9: {
                    uint32_t atype = gguf_read_u32(&p);
                    uint64_t acount = gguf_read_u64(&p);
                    if (atype == 8) {
                        for (uint64_t k = 0; k < acount; k++) { uint64_t sl = gguf_read_u64(&p); p += sl; }
                    } else if (atype == 0 || atype == 1 || atype == 7) p += acount;
                    else if (atype == 2 || atype == 3) p += acount * 2;
                    else if (atype == 4 || atype == 5 || atype == 6) p += acount * 4;
                    else if (atype == 10 || atype == 11 || atype == 12) p += acount * 8;
                    break;
                }
                default:
                    return -6;
            }
        }
    }

    bert->config.dim = dim;
    bert->config.intermediate_dim = inter_dim;
    bert->config.n_layers = n_layers;
    bert->config.n_heads = n_heads;
    bert->config.max_seq_len = max_seq;

    /* Parse tensor directory */
    gguf_tensor_desc_t *tdescs = (gguf_tensor_desc_t *)malloc((size_t)n_tensors * sizeof(gguf_tensor_desc_t));
    if (!tdescs) return -7;

    for (uint64_t t = 0; t < n_tensors; t++) {
        gguf_read_str(&p, tdescs[t].name, sizeof(tdescs[t].name));
        uint32_t ndims = gguf_read_u32(&p);
        for (uint32_t d = 0; d < ndims; d++) gguf_read_u64(&p);
        tdescs[t].type = gguf_read_u32(&p);
        tdescs[t].offset = gguf_read_u64(&p);
    }

    size_t header_len = (size_t)(p - addr);
    size_t data_start_offset = (header_len + alignment - 1) & ~(size_t)(alignment - 1);
    const uint8_t *data_start = addr + data_start_offset;

    uint32_t tok_type = 0;
    bert->weights.token_emb = find_gguf_tensor(tdescs, (int)n_tensors, data_start, "token_embd.weight", &tok_type);
    bert->config.qtype = (tok_type == 8) ? 2 : 0; /* 2 = Native GGUF Q8_0 interleaved */

    uint32_t pos_type = 0;
    const void *raw_pos = find_gguf_tensor(tdescs, (int)n_tensors, data_start, "position_embd.weight", &pos_type);
    if (pos_type == 8 && raw_pos) {
        float *dequant_pos = (float *)malloc((size_t)max_seq * dim * sizeof(float));
        int blocks_dim = dim / 32;
        const uint8_t *w_bytes = (const uint8_t *)raw_pos;
        for (int t = 0; t < max_seq; t++) {
            const uint8_t *w_row = w_bytes + (size_t)t * (blocks_dim * 34);
            for (int b = 0; b < blocks_dim; b++) {
                const uint8_t *blk = w_row + b * 34;
                float scale = fp16_to_fp32(*(const uint16_t *)blk);
                const int8_t *wb = (const int8_t *)(blk + 2);
                for (int c = 0; c < 32; c++) {
                    dequant_pos[t * dim + b * 32 + c] = (float)wb[c] * scale;
                }
            }
        }
        bert->weights.pos_emb = dequant_pos;
        bert->weights.pos_emb_allocated = true;
    } else {
        bert->weights.pos_emb = (const float *)raw_pos;
        bert->weights.pos_emb_allocated = false;
    }
    if (raw_pos == NULL) {
        bert->config.use_alibi = true;
    }
    bert->weights.type_emb   = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, "token_types.weight", NULL);
    bert->weights.emb_norm_w = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, "token_embd_norm.weight", NULL);
    bert->weights.emb_norm_b = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, "token_embd_norm.bias", NULL);

    size_t arr_sz = (size_t)n_layers * sizeof(void *);
    bert->weights.q_w            = (const void **)malloc(arr_sz);
    bert->weights.q_w_scales     = NULL;
    bert->weights.q_b            = (const float **)malloc(arr_sz);
    bert->weights.k_w            = (const void **)malloc(arr_sz);
    bert->weights.k_w_scales     = NULL;
    bert->weights.k_b            = (const float **)malloc(arr_sz);
    bert->weights.v_w            = (const void **)malloc(arr_sz);
    bert->weights.v_w_scales     = NULL;
    bert->weights.v_b            = (const float **)malloc(arr_sz);
    bert->weights.out_w          = (const void **)malloc(arr_sz);
    bert->weights.out_w_scales   = NULL;
    bert->weights.out_b          = (const float **)malloc(arr_sz);
    bert->weights.att_norm_w     = (const float **)malloc(arr_sz);
    bert->weights.att_norm_b     = (const float **)malloc(arr_sz);
    bert->weights.ffn_up_w       = (const void **)malloc(arr_sz);
    bert->weights.ffn_up_w_scales = NULL;
    bert->weights.ffn_up_b       = (const float **)malloc(arr_sz);
    bert->weights.ffn_down_w     = (const void **)malloc(arr_sz);
    bert->weights.ffn_down_w_scales = NULL;
    bert->weights.ffn_down_b     = (const float **)malloc(arr_sz);
    bert->weights.ffn_norm_w     = (const float **)malloc(arr_sz);
    bert->weights.ffn_norm_b     = (const float **)malloc(arr_sz);
    bert->weights.ffn_gate_w     = NULL;
    bert->weights.q_norm_w = bert->weights.q_norm_b = NULL;
    bert->weights.k_norm_w = bert->weights.k_norm_b = NULL;
    bert->weights.att_norm2_w = bert->weights.att_norm2_b = NULL;

    /* Optional tensors: probe layer 0 to decide whether to allocate the arrays */
    {
        int has_gate  = find_gguf_tensor(tdescs, (int)n_tensors, data_start, "blk.0.ffn_gate.weight", NULL) != NULL;
        int has_qkn   = find_gguf_tensor(tdescs, (int)n_tensors, data_start, "blk.0.attn_q_norm.weight", NULL) != NULL;
        int has_norm2 = find_gguf_tensor(tdescs, (int)n_tensors, data_start, "blk.0.attn_norm_2.weight", NULL) != NULL;
        if (has_gate) bert->weights.ffn_gate_w = (const void **)calloc((size_t)n_layers, sizeof(void *));
        if (has_qkn) {
            bert->weights.q_norm_w = (const float **)calloc((size_t)n_layers, sizeof(void *));
            bert->weights.q_norm_b = (const float **)calloc((size_t)n_layers, sizeof(void *));
            bert->weights.k_norm_w = (const float **)calloc((size_t)n_layers, sizeof(void *));
            bert->weights.k_norm_b = (const float **)calloc((size_t)n_layers, sizeof(void *));
        }
        if (has_norm2) {
            bert->weights.att_norm2_w = (const float **)calloc((size_t)n_layers, sizeof(void *));
            bert->weights.att_norm2_b = (const float **)calloc((size_t)n_layers, sizeof(void *));
        }
    }

    char tname[128];
    for (int l = 0; l < n_layers; l++) {
        snprintf(tname, sizeof(tname), "blk.%d.attn_q.weight", l);
        bert->weights.q_w[l] = find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        snprintf(tname, sizeof(tname), "blk.%d.attn_q.bias", l);
        bert->weights.q_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);

        snprintf(tname, sizeof(tname), "blk.%d.attn_k.weight", l);
        bert->weights.k_w[l] = find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        snprintf(tname, sizeof(tname), "blk.%d.attn_k.bias", l);
        bert->weights.k_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);

        snprintf(tname, sizeof(tname), "blk.%d.attn_v.weight", l);
        bert->weights.v_w[l] = find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        snprintf(tname, sizeof(tname), "blk.%d.attn_v.bias", l);
        bert->weights.v_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);

        snprintf(tname, sizeof(tname), "blk.%d.attn_output.weight", l);
        bert->weights.out_w[l] = find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        snprintf(tname, sizeof(tname), "blk.%d.attn_output.bias", l);
        bert->weights.out_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);

        snprintf(tname, sizeof(tname), "blk.%d.attn_output_norm.weight", l);
        bert->weights.att_norm_w[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        snprintf(tname, sizeof(tname), "blk.%d.attn_output_norm.bias", l);
        bert->weights.att_norm_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);

        snprintf(tname, sizeof(tname), "blk.%d.ffn_up.weight", l);
        bert->weights.ffn_up_w[l] = find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        snprintf(tname, sizeof(tname), "blk.%d.ffn_up.bias", l);
        bert->weights.ffn_up_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);

        snprintf(tname, sizeof(tname), "blk.%d.ffn_down.weight", l);
        bert->weights.ffn_down_w[l] = find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        snprintf(tname, sizeof(tname), "blk.%d.ffn_down.bias", l);
        bert->weights.ffn_down_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);

        snprintf(tname, sizeof(tname), "blk.%d.layer_output_norm.weight", l);
        bert->weights.ffn_norm_w[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        snprintf(tname, sizeof(tname), "blk.%d.layer_output_norm.bias", l);
        bert->weights.ffn_norm_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);

        if (bert->weights.ffn_gate_w) {
            snprintf(tname, sizeof(tname), "blk.%d.ffn_gate.weight", l);
            bert->weights.ffn_gate_w[l] = find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        }
        if (bert->weights.q_norm_w) {
            snprintf(tname, sizeof(tname), "blk.%d.attn_q_norm.weight", l);
            bert->weights.q_norm_w[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
            snprintf(tname, sizeof(tname), "blk.%d.attn_q_norm.bias", l);
            bert->weights.q_norm_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
            snprintf(tname, sizeof(tname), "blk.%d.attn_k_norm.weight", l);
            bert->weights.k_norm_w[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
            snprintf(tname, sizeof(tname), "blk.%d.attn_k_norm.bias", l);
            bert->weights.k_norm_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        }
        if (bert->weights.att_norm2_w) {
            snprintf(tname, sizeof(tname), "blk.%d.attn_norm_2.weight", l);
            bert->weights.att_norm2_w[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
            snprintf(tname, sizeof(tname), "blk.%d.attn_norm_2.bias", l);
            bert->weights.att_norm2_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        }
    }
    free(tdescs);

    return bert_setup_vocab_and_scratch(bert);
}

/* 2. EIFM Binary Loader */
static int eif_bert_load_eifm(eif_bert_t *bert, const uint8_t *addr, size_t file_sz)
{
    const uint32_t *h32 = (const uint32_t *)addr;
    if (h32[0] != EBERT_MAGIC) return -5;

    bert->config.dim              = (int)h32[2];
    bert->config.intermediate_dim = (int)h32[3];
    bert->config.n_layers         = (int)h32[4];
    bert->config.n_heads          = (int)h32[5];
    bert->config.max_seq_len      = (int)h32[6];
    bert->config.vocab_size       = (int)h32[7];
    bert->config.qtype            = (int)h32[8];

    int dim = bert->config.dim;
    int inter_dim = bert->config.intermediate_dim;
    int n_layers = bert->config.n_layers;
    int vocab_size = bert->config.vocab_size;
    int qtype = bert->config.qtype;
    int blocks_dim = dim / 32;
    int blocks_inter = inter_dim / 32;

    const uint8_t *ptr = addr + 64;
    #define ADVANCE_PTR(p, bytes) do { p += align64(bytes); } while(0)

    bert->weights.token_emb = ptr;
    size_t token_emb_bytes = (qtype == 1) ? ((size_t)vocab_size * dim) : ((size_t)vocab_size * dim * sizeof(float));
    ADVANCE_PTR(ptr, token_emb_bytes);

    if (qtype == 1) {
        bert->weights.token_emb_scales = (const float *)ptr;
        ADVANCE_PTR(ptr, (size_t)vocab_size * blocks_dim * sizeof(float));
    }

    bert->weights.pos_emb = (const float *)ptr;
    bert->weights.pos_emb_allocated = false;
    ADVANCE_PTR(ptr, (size_t)bert->config.max_seq_len * dim * sizeof(float));

    bert->weights.type_emb = (const float *)ptr;
    ADVANCE_PTR(ptr, (size_t)2 * dim * sizeof(float));

    bert->weights.emb_norm_w = (const float *)ptr;
    ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));

    bert->weights.emb_norm_b = (const float *)ptr;
    ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));

    size_t arr_sz = (size_t)n_layers * sizeof(void *);
    bert->weights.q_w            = (const void **)malloc(arr_sz);
    bert->weights.q_w_scales     = (const float **)malloc(arr_sz);
    bert->weights.q_b            = (const float **)malloc(arr_sz);
    bert->weights.k_w            = (const void **)malloc(arr_sz);
    bert->weights.k_w_scales     = (const float **)malloc(arr_sz);
    bert->weights.k_b            = (const float **)malloc(arr_sz);
    bert->weights.v_w            = (const void **)malloc(arr_sz);
    bert->weights.v_w_scales     = (const float **)malloc(arr_sz);
    bert->weights.v_b            = (const float **)malloc(arr_sz);
    bert->weights.out_w          = (const void **)malloc(arr_sz);
    bert->weights.out_w_scales   = (const float **)malloc(arr_sz);
    bert->weights.out_b          = (const float **)malloc(arr_sz);
    bert->weights.att_norm_w     = (const float **)malloc(arr_sz);
    bert->weights.att_norm_b     = (const float **)malloc(arr_sz);
    bert->weights.ffn_up_w       = (const void **)malloc(arr_sz);
    bert->weights.ffn_up_w_scales = (const float **)malloc(arr_sz);
    bert->weights.ffn_up_b       = (const float **)malloc(arr_sz);
    bert->weights.ffn_down_w     = (const void **)malloc(arr_sz);
    bert->weights.ffn_down_w_scales = (const float **)malloc(arr_sz);
    bert->weights.ffn_down_b     = (const float **)malloc(arr_sz);
    bert->weights.ffn_norm_w     = (const float **)malloc(arr_sz);
    bert->weights.ffn_norm_b     = (const float **)malloc(arr_sz);
    bert->weights.att_norm2_w    = bert->weights.att_norm2_b = NULL;

    size_t lin_dim_bytes = (qtype == 1) ? ((size_t)dim * dim) : ((size_t)dim * dim * sizeof(float));
    size_t lin_dim_scales_bytes = (size_t)dim * blocks_dim * sizeof(float);
    size_t up_bytes = (qtype == 1) ? ((size_t)inter_dim * dim) : ((size_t)inter_dim * dim * sizeof(float));
    size_t up_scales_bytes = (size_t)inter_dim * blocks_dim * sizeof(float);
    size_t down_bytes = (qtype == 1) ? ((size_t)dim * inter_dim) : ((size_t)dim * inter_dim * sizeof(float));
    size_t down_scales_bytes = (size_t)dim * blocks_inter * sizeof(float);

    for (int l = 0; l < n_layers; l++) {
        bert->weights.q_w[l] = ptr; ADVANCE_PTR(ptr, lin_dim_bytes);
        if (qtype == 1) { bert->weights.q_w_scales[l] = (const float *)ptr; ADVANCE_PTR(ptr, lin_dim_scales_bytes); }
        bert->weights.q_b[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));

        bert->weights.k_w[l] = ptr; ADVANCE_PTR(ptr, lin_dim_bytes);
        if (qtype == 1) { bert->weights.k_w_scales[l] = (const float *)ptr; ADVANCE_PTR(ptr, lin_dim_scales_bytes); }
        bert->weights.k_b[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));

        bert->weights.v_w[l] = ptr; ADVANCE_PTR(ptr, lin_dim_bytes);
        if (qtype == 1) { bert->weights.v_w_scales[l] = (const float *)ptr; ADVANCE_PTR(ptr, lin_dim_scales_bytes); }
        bert->weights.v_b[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));

        bert->weights.out_w[l] = ptr; ADVANCE_PTR(ptr, lin_dim_bytes);
        if (qtype == 1) { bert->weights.out_w_scales[l] = (const float *)ptr; ADVANCE_PTR(ptr, lin_dim_scales_bytes); }
        bert->weights.out_b[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));

        bert->weights.att_norm_w[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));
        bert->weights.att_norm_b[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));

        bert->weights.ffn_up_w[l] = ptr; ADVANCE_PTR(ptr, up_bytes);
        if (qtype == 1) { bert->weights.ffn_up_w_scales[l] = (const float *)ptr; ADVANCE_PTR(ptr, up_scales_bytes); }
        bert->weights.ffn_up_b[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)inter_dim * sizeof(float));

        bert->weights.ffn_down_w[l] = ptr; ADVANCE_PTR(ptr, down_bytes);
        if (qtype == 1) { bert->weights.ffn_down_w_scales[l] = (const float *)ptr; ADVANCE_PTR(ptr, down_scales_bytes); }
        bert->weights.ffn_down_b[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));

        bert->weights.ffn_norm_w[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));
        bert->weights.ffn_norm_b[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));
    }

    /* Vocabulary Table */
    if (ptr < addr + file_sz) {
        uint32_t n_tok = *(const uint32_t *)ptr;
        ptr += 4;
        bert->vocab.vocab_size = (int)n_tok;
        bert->vocab.tokens = (char **)malloc((size_t)n_tok * sizeof(char *));
        bert->vocab.cls_id = 101;
        bert->vocab.sep_id = 102;
        bert->vocab.unk_id = 100;
        bert->vocab.pad_id = 0;

        for (uint32_t t = 0; t < n_tok && ptr < addr + file_sz; t++) {
            uint16_t tlen = *(const uint16_t *)ptr;
            ptr += 2;
            bert->vocab.tokens[t] = (char *)malloc(tlen + 1);
            memcpy(bert->vocab.tokens[t], ptr, tlen);
            bert->vocab.tokens[t][tlen] = '\0';
            ptr += tlen;

            if (strcmp(bert->vocab.tokens[t], "[CLS]") == 0) bert->vocab.cls_id = (int)t;
            else if (strcmp(bert->vocab.tokens[t], "[SEP]") == 0) bert->vocab.sep_id = (int)t;
            else if (strcmp(bert->vocab.tokens[t], "[UNK]") == 0) bert->vocab.unk_id = (int)t;
            else if (strcmp(bert->vocab.tokens[t], "[PAD]") == 0) bert->vocab.pad_id = (int)t;
        }
    }

    return bert_setup_vocab_and_scratch(bert);
}

/* 3. Polymorphic BERT Loader (GGUF or EIFM) */
int eif_bert_load(eif_bert_t *bert, const char *model_path)
{
    if (!bert || !model_path) return -1;
    memset(bert, 0, sizeof(eif_bert_t));

#if defined(_WIN32)
    HANDLE hFile = CreateFileA(model_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "eif_bert_load: failed to open '%s'\n", model_path);
        return -2;
    }

    LARGE_INTEGER liSize;
    if (!GetFileSizeEx(hFile, &liSize)) {
        CloseHandle(hFile);
        return -3;
    }
    size_t file_sz = (size_t)liSize.QuadPart;
    if (file_sz < 64) {
        CloseHandle(hFile);
        return -4;
    }

    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) {
        CloseHandle(hFile);
        return -4;
    }

    void *addr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, file_sz);
    CloseHandle(hMap);
    CloseHandle(hFile);
    if (!addr) {
        fprintf(stderr, "eif_bert_load MapViewOfFile failed for '%s'\n", model_path);
        return -4;
    }
#else
    int fd = open(model_path, O_RDONLY);
    if (fd < 0) {
        perror("eif_bert_load open");
        return -2;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return -3;
    }
    size_t file_sz = (size_t)st.st_size;
    if (file_sz < 64) {
        close(fd);
        return -4;
    }

    void *addr = mmap(NULL, file_sz, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) {
        perror("eif_bert_load mmap");
        return -4;
    }
#endif

    bert->mmap_addr = addr;
    bert->mmap_size = file_sz;
    bert->is_mmap = true;

    uint32_t magic = *(const uint32_t *)addr;
    if (magic == GGUF_MAGIC) {
        int rc = eif_bert_load_gguf(bert, (const uint8_t *)addr, file_sz);
        if (rc != 0) {
            eif_bert_free(bert);
            return rc;
        }
        return 0;
    } else if (magic == EBERT_MAGIC) {
        int rc = eif_bert_load_eifm(bert, (const uint8_t *)addr, file_sz);
        if (rc != 0) {
            eif_bert_free(bert);
            return rc;
        }
        return 0;
    } else {
        fprintf(stderr, "eif_bert_load: Unsupported model magic 0x%08X in '%s' (expected GGUF or BERT)\n", magic, model_path);
        eif_bert_free(bert);
        return -5;
    }
}

void eif_bert_free(eif_bert_t *bert)
{
    if (!bert || !bert->is_initialized) return;

    if (bert->weights.pos_emb_allocated && bert->weights.pos_emb) {
        free((void *)bert->weights.pos_emb);
        bert->weights.pos_emb = NULL;
        bert->weights.pos_emb_allocated = false;
    }

    if (bert->weights.q_w) free((void *)bert->weights.q_w);
    if (bert->weights.q_w_scales) free((void *)bert->weights.q_w_scales);
    if (bert->weights.q_b) free((void *)bert->weights.q_b);
    if (bert->weights.k_w) free((void *)bert->weights.k_w);
    if (bert->weights.k_w_scales) free((void *)bert->weights.k_w_scales);
    if (bert->weights.k_b) free((void *)bert->weights.k_b);
    if (bert->weights.v_w) free((void *)bert->weights.v_w);
    if (bert->weights.v_w_scales) free((void *)bert->weights.v_w_scales);
    if (bert->weights.v_b) free((void *)bert->weights.v_b);
    if (bert->weights.out_w) free((void *)bert->weights.out_w);
    if (bert->weights.out_w_scales) free((void *)bert->weights.out_w_scales);
    if (bert->weights.out_b) free((void *)bert->weights.out_b);
    if (bert->weights.att_norm_w) free((void *)bert->weights.att_norm_w);
    if (bert->weights.att_norm_b) free((void *)bert->weights.att_norm_b);
    if (bert->weights.ffn_up_w) free((void *)bert->weights.ffn_up_w);
    if (bert->weights.ffn_up_w_scales) free((void *)bert->weights.ffn_up_w_scales);
    if (bert->weights.ffn_up_b) free((void *)bert->weights.ffn_up_b);
    if (bert->weights.ffn_down_w) free((void *)bert->weights.ffn_down_w);
    if (bert->weights.ffn_down_w_scales) free((void *)bert->weights.ffn_down_w_scales);
    if (bert->weights.ffn_down_b) free((void *)bert->weights.ffn_down_b);
    if (bert->weights.ffn_norm_w) free((void *)bert->weights.ffn_norm_w);
    if (bert->weights.ffn_norm_b) free((void *)bert->weights.ffn_norm_b);
    if (bert->weights.ffn_gate_w) free((void *)bert->weights.ffn_gate_w);
    if (bert->weights.q_norm_w) free((void *)bert->weights.q_norm_w);
    if (bert->weights.q_norm_b) free((void *)bert->weights.q_norm_b);
    if (bert->weights.k_norm_w) free((void *)bert->weights.k_norm_w);
    if (bert->weights.k_norm_b) free((void *)bert->weights.k_norm_b);
    if (bert->weights.att_norm2_w) free((void *)bert->weights.att_norm2_w);
    if (bert->weights.att_norm2_b) free((void *)bert->weights.att_norm2_b);

    if (bert->vocab.tokens) {
        for (int i = 0; i < bert->vocab.vocab_size; i++) {
            if (bert->vocab.tokens[i]) free(bert->vocab.tokens[i]);
        }
        free(bert->vocab.tokens);
    }
    if (bert->vocab.hash_table) {
        free(bert->vocab.hash_table);
    }
    if (bert->vocab.tok_type == EBERT_TOKENIZER_BPE) {
        bert->vocab.bpe_tok.vocab = NULL;
        eif_bpe_tokenizer_free(&bert->vocab.bpe_tok);
    }

    if (bert->scratch_seq_x) free(bert->scratch_seq_x);
    if (bert->scratch_seq_xb) free(bert->scratch_seq_xb);
    if (bert->scratch_q) free(bert->scratch_q);
    if (bert->scratch_k) free(bert->scratch_k);
    if (bert->scratch_v) free(bert->scratch_v);
    if (bert->scratch_att) free(bert->scratch_att);
    if (bert->scratch_inter) free(bert->scratch_inter);
    if (bert->scratch_proj) free(bert->scratch_proj);

    if (bert->is_mmap && bert->mmap_addr) {
#if defined(_WIN32)
        UnmapViewOfFile(bert->mmap_addr);
#else
        munmap(bert->mmap_addr, bert->mmap_size);
#endif
    }

    bert->is_initialized = false;
}

/* =============================================================================
 * Forward Pass (Bidirectional Multi-Head Attention + FFN)
 * ============================================================================= */

int eif_bert_embed(eif_bert_t *bert, const char *text, float *out_embedding)
{
    if (!bert || !bert->is_initialized || !text || !out_embedding) return -1;

    int max_tokens = bert->config.max_seq_len;
    int32_t tokens[512];
    int T = eif_bert_tokenize(bert, text, tokens, (max_tokens > 512) ? 512 : max_tokens);
    if (T <= 0) return -2;

    int dim = bert->config.dim;
    int inter_dim = bert->config.intermediate_dim;
    int n_layers = bert->config.n_layers;
    int n_heads = bert->config.n_heads;
    int head_dim = dim / n_heads;
    int qtype = bert->config.qtype;
    int blocks_dim = dim / 32;
    float inv_sqrt_head = 1.0f / sqrtf((float)head_dim);

    float *X = bert->scratch_seq_x;
    float *XB = bert->scratch_seq_xb;
    float *Q = bert->scratch_q;
    float *K = bert->scratch_k;
    float *V = bert->scratch_v;
    float *ATT = bert->scratch_att;
    float *H_INTER = bert->scratch_inter;
    float *PROJ = bert->scratch_proj;

    /* 1. Embedding Layer: Word + Pos + Token_Type -> LayerNorm */
    for (int t = 0; t < T; t++) {
        int tid = tokens[t];
        float *x_t = X + t * dim;

        if (qtype == 2) {
            /* Native GGUF Q8_0 interleaved (2B FP16 scale + 32B INT8 weights) */
            const uint8_t *w_row = (const uint8_t *)bert->weights.token_emb + (size_t)tid * (blocks_dim * 34);
            for (int b = 0; b < blocks_dim; b++) {
                const uint8_t *blk = w_row + b * 34;
                float scale = fp16_to_fp32(*(const uint16_t *)blk);
                const int8_t *wb = (const int8_t *)(blk + 2);
                for (int c = 0; c < 32; c++) {
                    x_t[b * 32 + c] = (float)wb[c] * scale;
                }
            }
        } else if (qtype == 1) {
            const int8_t *w_tok = (const int8_t *)bert->weights.token_emb + (size_t)tid * dim;
            const float *s_tok = bert->weights.token_emb_scales + (size_t)tid * blocks_dim;
            for (int b = 0; b < blocks_dim; b++) {
                float scale = s_tok[b];
                for (int c = 0; c < 32; c++) {
                    x_t[b * 32 + c] = (float)w_tok[b * 32 + c] * scale;
                }
            }
        } else {
            const float *w_tok = (const float *)bert->weights.token_emb + (size_t)tid * dim;
            memcpy(x_t, w_tok, dim * sizeof(float));
        }

        /* Add Position and Segment Embeddings */
        if (bert->weights.pos_emb) {
            const float *pos_row = bert->weights.pos_emb + (size_t)t * dim;
            for (int d = 0; d < dim; d++) {
                x_t[d] += pos_row[d];
            }
        }
        if (bert->weights.type_emb) {
            const float *type_row = bert->weights.type_emb; /* segment 0 */
            for (int d = 0; d < dim; d++) {
                x_t[d] += type_row[d];
            }
        }

        /* Embedding LayerNorm */
        bert_layernorm(x_t, x_t, bert->weights.emb_norm_w, bert->weights.emb_norm_b, dim, 1e-12f);
    }

    /* 2. Transformer Layers */
    bool use_alibi = bert->config.use_alibi || (bert->weights.pos_emb == NULL);

    for (int l = 0; l < n_layers; l++) {
        /* 2a. Batched Q, K, V Projections for all T tokens */
        bert_gemm(Q, bert->weights.q_w[l], bert->weights.q_w_scales ? bert->weights.q_w_scales[l] : NULL,
                  X, bert->weights.q_b[l], T, dim, dim, qtype);
        bert_gemm(K, bert->weights.k_w[l], bert->weights.k_w_scales ? bert->weights.k_w_scales[l] : NULL,
                  X, bert->weights.k_b[l], T, dim, dim, qtype);
        bert_gemm(V, bert->weights.v_w[l], bert->weights.v_w_scales ? bert->weights.v_w_scales[l] : NULL,
                  X, bert->weights.v_b[l], T, dim, dim, qtype);

        /* 2a'. Optional Q/K LayerNorm over the full hidden vector (Jina v2 code) */
        if (bert->weights.q_norm_w) {
            for (int t = 0; t < T; t++) {
                bert_layernorm(Q + t * dim, Q + t * dim, bert->weights.q_norm_w[l], bert->weights.q_norm_b[l], dim, 1e-12f);
                bert_layernorm(K + t * dim, K + t * dim, bert->weights.k_norm_w[l], bert->weights.k_norm_b[l], dim, 1e-12f);
            }
        }

        /* 2b. Multi-Head Bidirectional Self-Attention */
#ifdef _OPENMP
        int max_th_heads = omp_get_max_threads();
        int head_th = (n_heads < max_th_heads) ? n_heads : max_th_heads;
#else
        int head_th = 1;
#endif
        #pragma omp parallel for schedule(static) num_threads(head_th)
        for (int h = 0; h < n_heads; h++) {
            float *att_h = ATT + h * T * T;
            float slope = use_alibi ? bert_alibi_slope(h, n_heads) : 0.0f;

            for (int i = 0; i < T; i++) {
                const float *q_i = Q + i * dim + h * head_dim;
                float max_val = -1e9f;

                /* Dot-product Q_i and K_j across all j (no causal mask!) */
                for (int j = 0; j < T; j++) {
                    const float *k_j = K + j * dim + h * head_dim;
                    float score = 0.0f;
#if defined(EIF_ARCH_X86_64)
                    __m256 sacc = _mm256_setzero_ps();
                    for (int d = 0; d <= head_dim - 8; d += 8) {
                        sacc = _mm256_fmadd_ps(_mm256_loadu_ps(&q_i[d]), _mm256_loadu_ps(&k_j[d]), sacc);
                    }
                    score = hsum_float_8(sacc);
                    for (int d = (head_dim & ~7); d < head_dim; d++) {
                        score += q_i[d] * k_j[d];
                    }
#elif defined(EIF_ARCH_ARM64)
                    float32x4_t sacc = vdupq_n_f32(0.0f);
                    for (int d = 0; d <= head_dim - 4; d += 4) {
                        sacc = vfmaq_f32(sacc, vld1q_f32(&q_i[d]), vld1q_f32(&k_j[d]));
                    }
                    score = vaddvq_f32(sacc);
                    for (int d = (head_dim & ~3); d < head_dim; d++) {
                        score += q_i[d] * k_j[d];
                    }
#else
                    for (int d = 0; d < head_dim; d++) {
                        score += q_i[d] * k_j[d];
                    }
#endif
                    score *= inv_sqrt_head;

                    if (use_alibi) {
                        float dist = (float)(i > j ? i - j : j - i);
                        score -= slope * dist;
                    }

                    att_h[i * T + j] = score;
                    if (score > max_val) max_val = score;
                }

                /* Softmax */
                float sum_exp = 0.0f;
                for (int j = 0; j < T; j++) {
                    float ex = expf(att_h[i * T + j] - max_val);
                    att_h[i * T + j] = ex;
                    sum_exp += ex;
                }
                float inv_sum = 1.0f / (sum_exp > 1e-12f ? sum_exp : 1.0f);
                for (int j = 0; j < T; j++) {
                    att_h[i * T + j] *= inv_sum;
                }
            }
        }

        /* 2c. Context Accumulation: C = Attn * V */
        memset(XB, 0, (size_t)T * dim * sizeof(float));
        #pragma omp parallel for schedule(static) num_threads(head_th)
        for (int h = 0; h < n_heads; h++) {
            const float *att_h = ATT + h * T * T;
            for (int i = 0; i < T; i++) {
                float *c_ih = XB + i * dim + h * head_dim;
                for (int j = 0; j < T; j++) {
                    float p_ij = att_h[i * T + j];
                    const float *v_jh = V + j * dim + h * head_dim;
#if defined(EIF_ARCH_X86_64)
                    __m256 vp = _mm256_set1_ps(p_ij);
                    for (int d = 0; d <= head_dim - 8; d += 8) {
                        __m256 cv = _mm256_loadu_ps(&c_ih[d]);
                        __m256 vv = _mm256_loadu_ps(&v_jh[d]);
                        _mm256_storeu_ps(&c_ih[d], _mm256_fmadd_ps(vp, vv, cv));
                    }
                    for (int d = (head_dim & ~7); d < head_dim; d++) {
                        c_ih[d] += p_ij * v_jh[d];
                    }
#elif defined(EIF_ARCH_ARM64)
                    float32x4_t vp = vdupq_n_f32(p_ij);
                    for (int d = 0; d <= head_dim - 4; d += 4) {
                        float32x4_t cv = vld1q_f32(&c_ih[d]);
                        float32x4_t vv = vld1q_f32(&v_jh[d]);
                        vst1q_f32(&c_ih[d], vfmaq_f32(cv, vp, vv));
                    }
                    for (int d = (head_dim & ~3); d < head_dim; d++) {
                        c_ih[d] += p_ij * v_jh[d];
                    }
#else
                    for (int d = 0; d < head_dim; d++) {
                        c_ih[d] += p_ij * v_jh[d];
                    }
#endif
                }
            }
        }

        /* 2d. Output Projection: PROJ = XB * W_out + b */
        bert_gemm(PROJ, bert->weights.out_w[l], bert->weights.out_w_scales ? bert->weights.out_w_scales[l] : NULL,
                  XB, bert->weights.out_b[l], T, dim, dim, qtype);

        /* Residual Connection + LayerNorm */
        if (bert->weights.att_norm2_w) {
            for (int t = 0; t < T; t++) {
                float *x_t = X + t * dim;
                const float *p_t = PROJ + t * dim;
                float *c_t = XB + t * dim;
                for (int d = 0; d < dim; d++) {
                    c_t[d] = x_t[d] + p_t[d];
                }
                bert_layernorm(c_t, c_t, bert->weights.att_norm_w[l], bert->weights.att_norm_b[l], dim, 1e-12f);
                for (int d = 0; d < dim; d++) {
                    c_t[d] += x_t[d];
                }
                bert_layernorm(x_t, c_t, bert->weights.att_norm2_w[l], bert->weights.att_norm2_b[l], dim, 1e-12f);
            }
        } else {
            for (int t = 0; t < T; t++) {
                float *x_t = X + t * dim;
                const float *p_t = PROJ + t * dim;
                for (int d = 0; d < dim; d++) {
                    x_t[d] += p_t[d];
                }
                bert_layernorm(x_t, x_t, bert->weights.att_norm_w[l], bert->weights.att_norm_b[l], dim, 1e-12f);
            }
        }

        /* 2e. Feed-Forward Network: H_INTER = GELU(X * W_up + b) */
        bert_gemm(H_INTER, bert->weights.ffn_up_w[l], bert->weights.ffn_up_w_scales ? bert->weights.ffn_up_w_scales[l] : NULL,
                  X, bert->weights.ffn_up_b[l], T, inter_dim, dim, qtype);

        if (bert->weights.ffn_gate_w) {
            /* GEGLU: H = GELU(X * W_gate) * (X * W_up + b) */
            float *H_GATE = H_INTER + (size_t)T * inter_dim;
            bert_gemm(H_GATE, bert->weights.ffn_gate_w[l], NULL,
                      X, NULL, T, inter_dim, dim, qtype);
            bert_gelu(H_GATE, H_GATE, T * inter_dim);
            for (int i = 0; i < T * inter_dim; i++) H_INTER[i] *= H_GATE[i];
        } else {
            bert_gelu(H_INTER, H_INTER, T * inter_dim);
        }

        /* FFN Down: PROJ = H_INTER * W_down + b */
        bert_gemm(PROJ, bert->weights.ffn_down_w[l], bert->weights.ffn_down_w_scales ? bert->weights.ffn_down_w_scales[l] : NULL,
                  H_INTER, bert->weights.ffn_down_b[l], T, dim, inter_dim, qtype);

        /* Residual Connection + LayerNorm */
        for (int t = 0; t < T; t++) {
            float *x_t = X + t * dim;
            const float *f_t = PROJ + t * dim;
            for (int d = 0; d < dim; d++) {
                x_t[d] += f_t[d];
            }
            bert_layernorm(x_t, x_t, bert->weights.ffn_norm_w[l], bert->weights.ffn_norm_b[l], dim, 1e-12f);
        }
    }

    /* 3. Pooling: CLS (token 0) or Mean (all tokens) */
    if (bert->config.pooling_type == 2) {
        /* CLS Pooling (e.g. Granite-30M) */
        memcpy(out_embedding, X, (size_t)dim * sizeof(float));
    } else {
        /* Mean Pooling (e.g. Multilingual-E5, MiniLM) */
        memset(out_embedding, 0, (size_t)dim * sizeof(float));
        for (int t = 0; t < T; t++) {
            const float *x_t = X + t * dim;
            for (int d = 0; d < dim; d++) {
                out_embedding[d] += x_t[d];
            }
        }
        float inv_T = 1.0f / (float)T;
        for (int d = 0; d < dim; d++) {
            out_embedding[d] *= inv_T;
        }
    }

    /* 4. L2 Normalization */
    float sum_sq = 0.0f;
    for (int d = 0; d < dim; d++) {
        sum_sq += out_embedding[d] * out_embedding[d];
    }
    float norm = sqrtf(sum_sq);
    if (norm > 1e-12f) {
        float inv_norm = 1.0f / norm;
        for (int d = 0; d < dim; d++) {
            out_embedding[d] *= inv_norm;
        }
    }

    return 0;
}

int eif_bert_embed_batch(eif_bert_t *bert, const char **texts, int n_texts, float *out_embeddings)
{
    if (!bert || !bert->is_initialized || !texts || n_texts <= 0 || !out_embeddings) return -1;
    int dim = bert->config.dim;
    for (int i = 0; i < n_texts; i++) {
        if (!texts[i]) return -1;
        int rc = eif_bert_embed(bert, texts[i], out_embeddings + (size_t)i * dim);
        if (rc != 0) return rc;
    }
    return 0;
}
