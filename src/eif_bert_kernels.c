/**
 * @file eif_bert_kernels.c
 * @brief LayerNorm, GELU and batched GEMM (F32 / INT8 / GGUF Q8_0; AVX2 and NEON paths) of the BERT runtime.
 */
#if defined(__linux__)
#define _GNU_SOURCE
#endif

#include "eif_bert_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* 1. LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta */
void bert_layernorm(float *out, const float *in, const float *gamma, const float *beta, int dim, float eps)
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

/* Exact GELU (erf form) — gemma-embedding uses LLM_FFN_GELU in llama.cpp */
static inline float gelu_erf_f32(float x)
{
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
}

void bert_gelu_exact(float *out, const float *in, int n)
{
#pragma omp parallel for schedule(static) if(n >= 1024)
    for (int i = 0; i < n; i++) {
        out[i] = gelu_erf_f32(in[i]);
    }
}

void bert_gelu(float *out, const float *in, int n)
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
#if defined(EIF_ARCH_ARM64)
                float32x4_t va0 = vdupq_n_f32(0.0f), va1 = va0, va2 = va0, va3 = va0;
#endif

                for (int b_idx = 0; b_idx < blocks_per_row; b_idx++) {
                    const uint8_t *blk = w_row + (size_t)b_idx * 34;
                    float w_scale = fp16_to_fp32(*(const uint16_t *)blk);
                    const int8_t *wb = (const int8_t *)(blk + 2);
                    size_t off = (size_t)b_idx * 32;

#if defined(EIF_ARCH_ARM64)
                    va0 = vfmaq_n_f32(va0, vcvtq_f32_s32(arm_dot32_v(wb, xq0 + off)), w_scale * xs0[b_idx]);
                    va1 = vfmaq_n_f32(va1, vcvtq_f32_s32(arm_dot32_v(wb, xq1 + off)), w_scale * xs1[b_idx]);
                    va2 = vfmaq_n_f32(va2, vcvtq_f32_s32(arm_dot32_v(wb, xq2 + off)), w_scale * xs2[b_idx]);
                    va3 = vfmaq_n_f32(va3, vcvtq_f32_s32(arm_dot32_v(wb, xq3 + off)), w_scale * xs3[b_idx]);
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
#if defined(EIF_ARCH_ARM64)
                acc0 = vaddvq_f32(va0); acc1 = vaddvq_f32(va1); acc2 = vaddvq_f32(va2); acc3 = vaddvq_f32(va3);
#endif
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
void bert_gemm(
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

/* NeoX-style rotary embedding: rotates the half-split pairs (x[i], x[hd/2+i]) of one head at position pos. */
void bert_rope_neox(float *x, int pos, int hd, float freq_base)
{
    static int disable = -1;
    if (disable < 0) disable = getenv("EIF_NO_ROPE") ? 1 : 0;
    if (disable || pos == 0) return;
    /* NEOX style (gemma/every LLM rope in llama.cpp): rotate half-split pairs.
     * Pair (x[i], x[hd/2 + i]), freq = base^(-2i/hd) — NOT interleaved pairs. */
    int half = hd / 2;
    for (int i = 0; i < half; i++) {
        float freq = powf(freq_base, -2.0f * (float)i / (float)hd);
        float ang = (float)pos * freq;
        float c = cosf(ang), s = sinf(ang);
        float x0 = x[i], x1 = x[half + i];
        x[i]        = x0 * c - x1 * s;
        x[half + i] = x0 * s + x1 * c;
    }
}
