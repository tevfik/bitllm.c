/**
 * @file eif_bert_internal.h
 * @brief Shared SIMD setup, kernels and entry points of the BERT runtime (not a public API).
 */
#ifndef EIF_BERT_INTERNAL_H
#define EIF_BERT_INTERNAL_H

#include "eif_bert.h"
#include "eif_gguf.h"
#include <math.h>
#include <stdint.h>

/* Cross-platform SIMD setup and reduction helpers */

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
/* Same dot product, left as 4 partial sums so callers reduce once per row instead of once per block. */
static inline int32x4_t arm_dot32_v(const int8_t *wb, const int8_t *xb)
{
    int32x4_t vacc = vdupq_n_s32(0);
    vacc = vdotq_s32(vacc, vld1q_s8(wb), vld1q_s8(xb));
    return vdotq_s32(vacc, vld1q_s8(wb + 16), vld1q_s8(xb + 16));
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

/* eif_bert_kernels.c */
void bert_layernorm(float *out, const float *in, const float *gamma, const float *beta, int dim, float eps);
void bert_gelu(float *out, const float *in, int n);
void bert_gelu_exact(float *out, const float *in, int n);
/** Y[T, rows] = X[T, cols] * W[rows, cols]^T + bias; qtype 0 = F32, 1 = INT8 + scales, 2 = GGUF Q8_0 blocks. */
void bert_gemm(float *Y, const void *W, const float *W_scales, const float *X, const float *bias,
               int T, int rows, int cols, int qtype);

/* eif_bert_gpu.c: optional Vulkan offload (BITLLM_BERT_GPU=1). bert_gpu_gemm returns 1 when it computed Y. */
void eif_bert_gpu_attach(eif_bert_t *bert);
void eif_bert_gpu_detach(eif_bert_t *bert);
int bert_gpu_layers(eif_bert_t *bert, int T); /* 1 when the layers ran on the GPU (hidden states in scratch_seq_x) */
int bert_gpu_gemm(float *Y, const void *W, const float *X, const float *bias, int T, int rows, int cols);

void bert_rope_neox(float *x, int pos, int hd, float freq_base);

/* eif_bert.c */
int bert_token_cap(const eif_bert_t *bert);

/* eif_bert_modern.c */
int eif_bert_forward_modern(eif_bert_t *bert, const int32_t *tokens, int T);

/* eif_bert_gemma.c: Gemma-embedding encoders (norm_style 1) */
int eif_bert_embed_gemma(eif_bert_t *bert, const char *text, float *out_embedding);
int eif_bert_embed_gemma2(eif_bert_t *bert, const char *text, float *out_embedding);

#endif
