/**
 * @file eif_siglip.c
 * @brief Native Pure C99 SigLIP2 Vision Encoder & Connector Implementation.
 */

#include "eif_siglip.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SIGLIP_PATCH_SIZE   16
#define SIGLIP_IMG_SIZE     512
#define SIGLIP_NUM_LAYERS   12
#define SIGLIP_HIDDEN_SIZE  768
#define SIGLIP_NUM_HEADS    12
#define SIGLIP_HEAD_DIM     64
#define SIGLIP_INTERM_SIZE  3072
#define SIGLIP_NUM_PATCHES  1024
#define SIGLIP_CONN_IN      12288
#define SIGLIP_CONN_OUT     576
#define SIGLIP_NUM_TOKENS   64

int eif_siglip_init_from_buffer(eif_siglip_t *model, const void *buffer, size_t buffer_size)
{
    if (!model || !buffer) return -1;
    memset(model, 0, sizeof(*model));

    if (buffer_size < sizeof(eif_siglip_header_t)) {
        fprintf(stderr, "[EIF SigLIP Error] Buffer too small for header\n");
        return -2;
    }

    const eif_siglip_header_t *hdr = (const eif_siglip_header_t *)buffer;
    if (hdr->magic != EIF_SIGLIP_MAGIC) {
        fprintf(stderr, "[EIF SigLIP Error] Invalid magic: 0x%08X (expected 0x%08X)\n",
                hdr->magic, EIF_SIGLIP_MAGIC);
        return -3;
    }

    model->header = *hdr;

    const uint8_t *ptr = (const uint8_t *)buffer + sizeof(eif_siglip_header_t);

    /* 1. Embeddings */
    model->pe_w = (const float *)ptr;
    ptr += (size_t)SIGLIP_HIDDEN_SIZE * 3 * SIGLIP_PATCH_SIZE * SIGLIP_PATCH_SIZE * sizeof(float);

    model->pe_b = (const float *)ptr;
    ptr += (size_t)SIGLIP_HIDDEN_SIZE * sizeof(float);

    model->pos_w = (const float *)ptr;
    ptr += (size_t)SIGLIP_NUM_PATCHES * SIGLIP_HIDDEN_SIZE * sizeof(float);

    /* 2. 12 Transformer Layers */
    for (int l = 0; l < SIGLIP_NUM_LAYERS; l++) {
        eif_siglip_layer_t *layer = &model->layers[l];

        layer->ln1_w = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);
        layer->ln1_b = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);

        layer->q_w = (const float *)ptr; ptr += (size_t)SIGLIP_HIDDEN_SIZE * SIGLIP_HIDDEN_SIZE * sizeof(float);
        layer->q_b = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);

        layer->k_w = (const float *)ptr; ptr += (size_t)SIGLIP_HIDDEN_SIZE * SIGLIP_HIDDEN_SIZE * sizeof(float);
        layer->k_b = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);

        layer->v_w = (const float *)ptr; ptr += (size_t)SIGLIP_HIDDEN_SIZE * SIGLIP_HIDDEN_SIZE * sizeof(float);
        layer->v_b = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);

        layer->out_w = (const float *)ptr; ptr += (size_t)SIGLIP_HIDDEN_SIZE * SIGLIP_HIDDEN_SIZE * sizeof(float);
        layer->out_b = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);

        layer->ln2_w = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);
        layer->ln2_b = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);

        layer->fc1_w = (const float *)ptr; ptr += (size_t)SIGLIP_INTERM_SIZE * SIGLIP_HIDDEN_SIZE * sizeof(float);
        layer->fc1_b = (const float *)ptr; ptr += SIGLIP_INTERM_SIZE * sizeof(float);

        layer->fc2_w = (const float *)ptr; ptr += (size_t)SIGLIP_HIDDEN_SIZE * SIGLIP_INTERM_SIZE * sizeof(float);
        layer->fc2_b = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);
    }

    /* 3. Post LayerNorm */
    model->post_ln_w = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);
    model->post_ln_b = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);

    /* 4. Connector */
    model->conn_w = (const float *)ptr; ptr += (size_t)SIGLIP_CONN_OUT * SIGLIP_CONN_IN * sizeof(float);

    size_t consumed = (size_t)(ptr - (const uint8_t *)buffer);
    if (consumed > buffer_size) {
        fprintf(stderr, "[EIF SigLIP Error] Expected at least %zu bytes, got %zu\n", consumed, buffer_size);
        return -4;
    }

    model->is_initialized = true;
    eif_gpu_init(&model->gpu);
    if (model->gpu.is_available) {
        fprintf(stderr, "  • Vision Compute: Vulkan GPU Acceleration ACTIVE [%s]\n", model->gpu.device_name);
        fflush(stderr);
    }
    return 0;
}

int eif_siglip_load_from_file(eif_siglip_t *model, const char *filepath, size_t offset, size_t size)
{
    if (!model || !filepath) return -1;
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "[EIF SigLIP Error] Failed to open %s\n", filepath);
        return -1;
    }

    if (fseek(f, (long)offset, SEEK_SET) != 0) {
        fclose(f);
        return -2;
    }

    uint8_t *buf = (uint8_t *)malloc(size);
    if (!buf) {
        fclose(f);
        return -3;
    }

    if (fread(buf, 1, size, f) != size) {
        free(buf);
        fclose(f);
        return -4;
    }
    fclose(f);

    int rc = eif_siglip_init_from_buffer(model, buf, size);
    if (rc != 0) {
        free(buf);
        return rc;
    }

    model->allocated_weights = buf;
    return 0;
}

void eif_siglip_free(eif_siglip_t *model)
{
    if (!model) return;
    if (model->gpu.is_available) {
        eif_gpu_cleanup(&model->gpu);
    }
    if (model->allocated_weights) {
        free(model->allocated_weights);
        model->allocated_weights = NULL;
    }
    model->is_initialized = false;
}

static void layer_norm_f32(const float *x, const float *w, const float *b, float *out, int n, float eps)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i];
    float mean = sum / (float)n;

    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = x[i] - mean;
        sum_sq += diff * diff;
    }
    float var = sum_sq / (float)n;
    float inv_std = 1.0f / sqrtf(var + eps);

    for (int i = 0; i < n; i++) {
        out[i] = (x[i] - mean) * inv_std * w[i] + b[i];
    }
}

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX512F__)
#include <immintrin.h>
static inline float dot_f32_simd(const float *a, const float *b, int n)
{
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    int i = 0;
    for (; i <= n - 32; i += 32) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), acc1);
    }
    for (; i <= n - 16; i += 16) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
    }
    acc0 = _mm512_add_ps(acc0, acc1);
    float s = _mm512_reduce_add_ps(acc0);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}
#elif defined(__AVX2__)
#include <immintrin.h>
static inline float dot_f32_simd(const float *a, const float *b, int n)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 16; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    for (; i <= n - 8; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float s = _mm_cvtss_f32(sum);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}
#elif defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
static inline float dot_f32_simd(const float *a, const float *b, int n)
{
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i <= n - 16; i += 16) {
        sum0 = vfmaq_f32(sum0, vld1q_f32(a + i), vld1q_f32(b + i));
        sum1 = vfmaq_f32(sum1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        sum2 = vfmaq_f32(sum2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        sum3 = vfmaq_f32(sum3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i <= n - 4; i += 4) {
        sum0 = vfmaq_f32(sum0, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum0 = vaddq_f32(sum0, sum2);
    float s = vaddvq_f32(sum0);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}
#else
static inline float dot_f32_simd(const float *a, const float *b, int n)
{
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}
#endif

static void siglip_gemm_f32(
    eif_gpu_context_t *gpu,
    const float *A,        /* [M, K] */
    const float *W,        /* [N, K] - row-major */
    const float *bias,     /* [N] (optional) */
    float *C,              /* [M, N] */
    int M, int N, int K,
    bool residual_add
)
{
    if (gpu && gpu->is_available) {
        if (eif_gpu_gemm_f32(gpu, A, W, bias, C, M, N, K, 0, residual_add)) {
            return;
        }
    }

    #pragma omp parallel for schedule(static) if(!omp_in_parallel())
    for (int p = 0; p < M; p += 4) {
        const float *a0 = A + (size_t)(p + 0) * K;
        const float *a1 = A + (size_t)(p + 1) * K;
        const float *a2 = A + (size_t)(p + 2) * K;
        const float *a3 = A + (size_t)(p + 3) * K;
        float *c0 = C + (size_t)(p + 0) * N;
        float *c1 = C + (size_t)(p + 1) * N;
        float *c2 = C + (size_t)(p + 2) * N;
        float *c3 = C + (size_t)(p + 3) * N;

        for (int d = 0; d < N; d++) {
            const float *w = W + (size_t)d * K;
            float b_val = bias ? bias[d] : 0.0f;

#if defined(__AVX512F__)
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();
            int k = 0;
            for (; k <= K - 16; k += 16) {
                __m512 vw = _mm512_loadu_ps(w + k);
                acc0 = _mm512_fmadd_ps(vw, _mm512_loadu_ps(a0 + k), acc0);
                acc1 = _mm512_fmadd_ps(vw, _mm512_loadu_ps(a1 + k), acc1);
                acc2 = _mm512_fmadd_ps(vw, _mm512_loadu_ps(a2 + k), acc2);
                acc3 = _mm512_fmadd_ps(vw, _mm512_loadu_ps(a3 + k), acc3);
            }
            float d0 = _mm512_reduce_add_ps(acc0);
            float d1 = _mm512_reduce_add_ps(acc1);
            float d2 = _mm512_reduce_add_ps(acc2);
            float d3 = _mm512_reduce_add_ps(acc3);
            for (; k < K; k++) {
                float wv = w[k];
                d0 += a0[k] * wv; d1 += a1[k] * wv; d2 += a2[k] * wv; d3 += a3[k] * wv;
            }
#elif defined(__AVX2__)
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            int k = 0;
            for (; k <= K - 8; k += 8) {
                __m256 vw = _mm256_loadu_ps(w + k);
                acc0 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(a0 + k), acc0);
                acc1 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(a1 + k), acc1);
                acc2 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(a2 + k), acc2);
                acc3 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(a3 + k), acc3);
            }
            __m128 lo0 = _mm256_castps256_ps128(acc0), hi0 = _mm256_extractf128_ps(acc0, 1);
            __m128 s0 = _mm_add_ps(lo0, hi0); s0 = _mm_hadd_ps(s0, s0); s0 = _mm_hadd_ps(s0, s0);
            float d0 = _mm_cvtss_f32(s0);
            __m128 lo1 = _mm256_castps256_ps128(acc1), hi1 = _mm256_extractf128_ps(acc1, 1);
            __m128 s1 = _mm_add_ps(lo1, hi1); s1 = _mm_hadd_ps(s1, s1); s1 = _mm_hadd_ps(s1, s1);
            float d1 = _mm_cvtss_f32(s1);
            __m128 lo2 = _mm256_castps256_ps128(acc2), hi2 = _mm256_extractf128_ps(acc2, 1);
            __m128 s2 = _mm_add_ps(lo2, hi2); s2 = _mm_hadd_ps(s2, s2); s2 = _mm_hadd_ps(s2, s2);
            float d2 = _mm_cvtss_f32(s2);
            __m128 lo3 = _mm256_castps256_ps128(acc3), hi3 = _mm256_extractf128_ps(acc3, 1);
            __m128 s3 = _mm_add_ps(lo3, hi3); s3 = _mm_hadd_ps(s3, s3); s3 = _mm_hadd_ps(s3, s3);
            float d3 = _mm_cvtss_f32(s3);
            for (; k < K; k++) {
                float wv = w[k];
                d0 += a0[k] * wv; d1 += a1[k] * wv; d2 += a2[k] * wv; d3 += a3[k] * wv;
            }
#elif defined(__ARM_NEON) || defined(__aarch64__)
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k <= K - 8; k += 8) {
                float32x4_t vw0 = vld1q_f32(w + k);
                float32x4_t vw1 = vld1q_f32(w + k + 4);
                acc0 = vfmaq_f32(acc0, vw0, vld1q_f32(a0 + k));
                acc0 = vfmaq_f32(acc0, vw1, vld1q_f32(a0 + k + 4));
                acc1 = vfmaq_f32(acc1, vw0, vld1q_f32(a1 + k));
                acc1 = vfmaq_f32(acc1, vw1, vld1q_f32(a1 + k + 4));
                acc2 = vfmaq_f32(acc2, vw0, vld1q_f32(a2 + k));
                acc2 = vfmaq_f32(acc2, vw1, vld1q_f32(a2 + k + 4));
                acc3 = vfmaq_f32(acc3, vw0, vld1q_f32(a3 + k));
                acc3 = vfmaq_f32(acc3, vw1, vld1q_f32(a3 + k + 4));
            }
            float d0 = vaddvq_f32(acc0);
            float d1 = vaddvq_f32(acc1);
            float d2 = vaddvq_f32(acc2);
            float d3 = vaddvq_f32(acc3);
            for (; k < K; k++) {
                float wv = w[k];
                d0 += a0[k] * wv; d1 += a1[k] * wv; d2 += a2[k] * wv; d3 += a3[k] * wv;
            }
#else
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
            for (int k = 0; k < K; k++) {
                float wv = w[k];
                d0 += a0[k] * wv; d1 += a1[k] * wv; d2 += a2[k] * wv; d3 += a3[k] * wv;
            }
#endif
            if (residual_add) {
                c0[d] += b_val + d0;
                c1[d] += b_val + d1;
                c2[d] += b_val + d2;
                c3[d] += b_val + d3;
            } else {
                c0[d] = b_val + d0;
                c1[d] = b_val + d1;
                c2[d] = b_val + d2;
                c3[d] = b_val + d3;
            }
        }
    }
}

static void siglip_gemm_f32_gelu(
    eif_gpu_context_t *gpu,
    const float *A,        /* [M, K] */
    const float *W,        /* [N, K] */
    const float *bias,     /* [N] */
    float *C,              /* [M, N] */
    int M, int N, int K
)
{
    if (gpu && gpu->is_available) {
        if (eif_gpu_gemm_f32(gpu, A, W, bias, C, M, N, K, 1 /* GELU */, false)) {
            return;
        }
    }

    const float sqrt_2_over_pi = 0.7978845608028654f;
    #pragma omp parallel for schedule(static) if(!omp_in_parallel())
    for (int p = 0; p < M; p += 4) {
        const float *a0 = A + (size_t)(p + 0) * K;
        const float *a1 = A + (size_t)(p + 1) * K;
        const float *a2 = A + (size_t)(p + 2) * K;
        const float *a3 = A + (size_t)(p + 3) * K;
        float *c0 = C + (size_t)(p + 0) * N;
        float *c1 = C + (size_t)(p + 1) * N;
        float *c2 = C + (size_t)(p + 2) * N;
        float *c3 = C + (size_t)(p + 3) * N;

        for (int d = 0; d < N; d++) {
            const float *w = W + (size_t)d * K;
            float b_val = bias ? bias[d] : 0.0f;

#if defined(__AVX512F__)
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();
            int k = 0;
            for (; k <= K - 16; k += 16) {
                __m512 vw = _mm512_loadu_ps(w + k);
                acc0 = _mm512_fmadd_ps(vw, _mm512_loadu_ps(a0 + k), acc0);
                acc1 = _mm512_fmadd_ps(vw, _mm512_loadu_ps(a1 + k), acc1);
                acc2 = _mm512_fmadd_ps(vw, _mm512_loadu_ps(a2 + k), acc2);
                acc3 = _mm512_fmadd_ps(vw, _mm512_loadu_ps(a3 + k), acc3);
            }
            float d0 = _mm512_reduce_add_ps(acc0);
            float d1 = _mm512_reduce_add_ps(acc1);
            float d2 = _mm512_reduce_add_ps(acc2);
            float d3 = _mm512_reduce_add_ps(acc3);
            for (; k < K; k++) {
                float wv = w[k];
                d0 += a0[k] * wv; d1 += a1[k] * wv; d2 += a2[k] * wv; d3 += a3[k] * wv;
            }
#elif defined(__AVX2__)
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            int k = 0;
            for (; k <= K - 8; k += 8) {
                __m256 vw = _mm256_loadu_ps(w + k);
                acc0 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(a0 + k), acc0);
                acc1 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(a1 + k), acc1);
                acc2 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(a2 + k), acc2);
                acc3 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(a3 + k), acc3);
            }
            __m128 lo0 = _mm256_castps256_ps128(acc0), hi0 = _mm256_extractf128_ps(acc0, 1);
            __m128 s0 = _mm_add_ps(lo0, hi0); s0 = _mm_hadd_ps(s0, s0); s0 = _mm_hadd_ps(s0, s0);
            float d0 = _mm_cvtss_f32(s0);
            __m128 lo1 = _mm256_castps256_ps128(acc1), hi1 = _mm256_extractf128_ps(acc1, 1);
            __m128 s1 = _mm_add_ps(lo1, hi1); s1 = _mm_hadd_ps(s1, s1); s1 = _mm_hadd_ps(s1, s1);
            float d1 = _mm_cvtss_f32(s1);
            __m128 lo2 = _mm256_castps256_ps128(acc2), hi2 = _mm256_extractf128_ps(acc2, 1);
            __m128 s2 = _mm_add_ps(lo2, hi2); s2 = _mm_hadd_ps(s2, s2); s2 = _mm_hadd_ps(s2, s2);
            float d2 = _mm_cvtss_f32(s2);
            __m128 lo3 = _mm256_castps256_ps128(acc3), hi3 = _mm256_extractf128_ps(acc3, 1);
            __m128 s3 = _mm_add_ps(lo3, hi3); s3 = _mm_hadd_ps(s3, s3); s3 = _mm_hadd_ps(s3, s3);
            float d3 = _mm_cvtss_f32(s3);
            for (; k < K; k++) {
                float wv = w[k];
                d0 += a0[k] * wv; d1 += a1[k] * wv; d2 += a2[k] * wv; d3 += a3[k] * wv;
            }
#elif defined(__ARM_NEON) || defined(__aarch64__)
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k <= K - 8; k += 8) {
                float32x4_t vw0 = vld1q_f32(w + k);
                float32x4_t vw1 = vld1q_f32(w + k + 4);
                acc0 = vfmaq_f32(acc0, vw0, vld1q_f32(a0 + k));
                acc0 = vfmaq_f32(acc0, vw1, vld1q_f32(a0 + k + 4));
                acc1 = vfmaq_f32(acc1, vw0, vld1q_f32(a1 + k));
                acc1 = vfmaq_f32(acc1, vw1, vld1q_f32(a1 + k + 4));
                acc2 = vfmaq_f32(acc2, vw0, vld1q_f32(a2 + k));
                acc2 = vfmaq_f32(acc2, vw1, vld1q_f32(a2 + k + 4));
                acc3 = vfmaq_f32(acc3, vw0, vld1q_f32(a3 + k));
                acc3 = vfmaq_f32(acc3, vw1, vld1q_f32(a3 + k + 4));
            }
            float d0 = vaddvq_f32(acc0);
            float d1 = vaddvq_f32(acc1);
            float d2 = vaddvq_f32(acc2);
            float d3 = vaddvq_f32(acc3);
            for (; k < K; k++) {
                float wv = w[k];
                d0 += a0[k] * wv; d1 += a1[k] * wv; d2 += a2[k] * wv; d3 += a3[k] * wv;
            }
#else
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
            for (int k = 0; k < K; k++) {
                float wv = w[k];
                d0 += a0[k] * wv; d1 += a1[k] * wv; d2 += a2[k] * wv; d3 += a3[k] * wv;
            }
#endif
            float val0 = b_val + d0;
            float val1 = b_val + d1;
            float val2 = b_val + d2;
            float val3 = b_val + d3;

            float u0 = sqrt_2_over_pi * (val0 + 0.044715f * val0 * val0 * val0);
            float u1 = sqrt_2_over_pi * (val1 + 0.044715f * val1 * val1 * val1);
            float u2 = sqrt_2_over_pi * (val2 + 0.044715f * val2 * val2 * val2);
            float u3 = sqrt_2_over_pi * (val3 + 0.044715f * val3 * val3 * val3);

            c0[d] = 0.5f * val0 * (1.0f + tanhf(u0));
            c1[d] = 0.5f * val1 * (1.0f + tanhf(u1));
            c2[d] = 0.5f * val2 * (1.0f + tanhf(u2));
            c3[d] = 0.5f * val3 * (1.0f + tanhf(u3));
        }
    }
}

static void siglip_gemm_qkv(
    eif_gpu_context_t *gpu,
    const float *A,         /* [M, K] */
    const float *W_q, const float *B_q,
    const float *W_k, const float *B_k,
    const float *W_v, const float *B_v,
    float *Q, float *K_out, float *V_out,
    int M, int N, int K
)
{
    if (gpu && gpu->is_available) {
        bool ok_q = eif_gpu_gemm_f32(gpu, A, W_q, B_q, Q, M, N, K, 0, false);
        bool ok_k = eif_gpu_gemm_f32(gpu, A, W_k, B_k, K_out, M, N, K, 0, false);
        bool ok_v = eif_gpu_gemm_f32(gpu, A, W_v, B_v, V_out, M, N, K, 0, false);
        if (ok_q && ok_k && ok_v) return;
    }

    #pragma omp parallel for schedule(static) if(!omp_in_parallel())
    for (int p = 0; p < M; p += 4) {
        const float *a0 = A + (size_t)(p + 0) * K;
        const float *a1 = A + (size_t)(p + 1) * K;
        const float *a2 = A + (size_t)(p + 2) * K;
        const float *a3 = A + (size_t)(p + 3) * K;

        float *q0 = Q + (size_t)(p + 0) * N;
        float *q1 = Q + (size_t)(p + 1) * N;
        float *q2 = Q + (size_t)(p + 2) * N;
        float *q3 = Q + (size_t)(p + 3) * N;

        float *k0 = K_out + (size_t)(p + 0) * N;
        float *k1 = K_out + (size_t)(p + 1) * N;
        float *k2 = K_out + (size_t)(p + 2) * N;
        float *k3 = K_out + (size_t)(p + 3) * N;

        float *v0 = V_out + (size_t)(p + 0) * N;
        float *v1 = V_out + (size_t)(p + 1) * N;
        float *v2 = V_out + (size_t)(p + 2) * N;
        float *v3 = V_out + (size_t)(p + 3) * N;

        for (int d = 0; d < N; d++) {
            const float *wq = W_q + (size_t)d * K;
            const float *wk = W_k + (size_t)d * K;
            const float *wv = W_v + (size_t)d * K;
            float bq = B_q[d];
            float bk = B_k[d];
            float bv = B_v[d];

#if defined(__ARM_NEON) || defined(__aarch64__)
            float32x4_t q_acc0 = vdupq_n_f32(0.0f), q_acc1 = vdupq_n_f32(0.0f);
            float32x4_t q_acc2 = vdupq_n_f32(0.0f), q_acc3 = vdupq_n_f32(0.0f);

            float32x4_t k_acc0 = vdupq_n_f32(0.0f), k_acc1 = vdupq_n_f32(0.0f);
            float32x4_t k_acc2 = vdupq_n_f32(0.0f), k_acc3 = vdupq_n_f32(0.0f);

            float32x4_t v_acc0 = vdupq_n_f32(0.0f), v_acc1 = vdupq_n_f32(0.0f);
            float32x4_t v_acc2 = vdupq_n_f32(0.0f), v_acc3 = vdupq_n_f32(0.0f);

            int k = 0;
            for (; k <= K - 4; k += 4) {
                float32x4_t va0 = vld1q_f32(a0 + k);
                float32x4_t va1 = vld1q_f32(a1 + k);
                float32x4_t va2 = vld1q_f32(a2 + k);
                float32x4_t va3 = vld1q_f32(a3 + k);

                float32x4_t v_wq = vld1q_f32(wq + k);
                float32x4_t v_wk = vld1q_f32(wk + k);
                float32x4_t v_wv = vld1q_f32(wv + k);

                q_acc0 = vfmaq_f32(q_acc0, v_wq, va0);
                q_acc1 = vfmaq_f32(q_acc1, v_wq, va1);
                q_acc2 = vfmaq_f32(q_acc2, v_wq, va2);
                q_acc3 = vfmaq_f32(q_acc3, v_wq, va3);

                k_acc0 = vfmaq_f32(k_acc0, v_wk, va0);
                k_acc1 = vfmaq_f32(k_acc1, v_wk, va1);
                k_acc2 = vfmaq_f32(k_acc2, v_wk, va2);
                k_acc3 = vfmaq_f32(k_acc3, v_wk, va3);

                v_acc0 = vfmaq_f32(v_acc0, v_wv, va0);
                v_acc1 = vfmaq_f32(v_acc1, v_wv, va1);
                v_acc2 = vfmaq_f32(v_acc2, v_wv, va2);
                v_acc3 = vfmaq_f32(v_acc3, v_wv, va3);
            }

            float q_d0 = vaddvq_f32(q_acc0), q_d1 = vaddvq_f32(q_acc1), q_d2 = vaddvq_f32(q_acc2), q_d3 = vaddvq_f32(q_acc3);
            float k_d0 = vaddvq_f32(k_acc0), k_d1 = vaddvq_f32(k_acc1), k_d2 = vaddvq_f32(k_acc2), k_d3 = vaddvq_f32(k_acc3);
            float v_d0 = vaddvq_f32(v_acc0), v_d1 = vaddvq_f32(v_acc1), v_d2 = vaddvq_f32(v_acc2), v_d3 = vaddvq_f32(v_acc3);

            for (; k < K; k++) {
                float a0_v = a0[k], a1_v = a1[k], a2_v = a2[k], a3_v = a3[k];
                float wq_v = wq[k], wk_v = wk[k], wv_v = wv[k];

                q_d0 += a0_v * wq_v; q_d1 += a1_v * wq_v; q_d2 += a2_v * wq_v; q_d3 += a3_v * wq_v;
                k_d0 += a0_v * wk_v; k_d1 += a1_v * wk_v; k_d2 += a2_v * wk_v; k_d3 += a3_v * wk_v;
                v_d0 += a0_v * wv_v; v_d1 += a1_v * wv_v; v_d2 += a2_v * wv_v; v_d3 += a3_v * wv_v;
            }
#else
            float q_d0 = 0.0f, q_d1 = 0.0f, q_d2 = 0.0f, q_d3 = 0.0f;
            float k_d0 = 0.0f, k_d1 = 0.0f, k_d2 = 0.0f, k_d3 = 0.0f;
            float v_d0 = 0.0f, v_d1 = 0.0f, v_d2 = 0.0f, v_d3 = 0.0f;
            for (int k = 0; k < K; k++) {
                float a0_v = a0[k], a1_v = a1[k], a2_v = a2[k], a3_v = a3[k];
                float wq_v = wq[k], wk_v = wk[k], wv_v = wv[k];
                q_d0 += a0_v * wq_v; q_d1 += a1_v * wq_v; q_d2 += a2_v * wq_v; q_d3 += a3_v * wq_v;
                k_d0 += a0_v * wk_v; k_d1 += a1_v * wk_v; k_d2 += a2_v * wk_v; k_d3 += a3_v * wk_v;
                v_d0 += a0_v * wv_v; v_d1 += a1_v * wv_v; v_d2 += a2_v * wv_v; v_d3 += a3_v * wv_v;
            }
#endif
            q0[d] = bq + q_d0; q1[d] = bq + q_d1; q2[d] = bq + q_d2; q3[d] = bq + q_d3;
            k0[d] = bk + k_d0; k1[d] = bk + k_d1; k2[d] = bk + k_d2; k3[d] = bk + k_d3;
            v0[d] = bv + v_d0; v1[d] = bv + v_d1; v2[d] = bv + v_d2; v3[d] = bv + v_d3;
        }
    }
}

int eif_siglip_encode_tile(const eif_siglip_t *model, const float *tile_pixels, float *out_tokens)
{
    if (!model || !model->is_initialized || !tile_pixels || !out_tokens) return -1;
    eif_gpu_context_t *gpu = (eif_gpu_context_t *)&model->gpu;

    /* Temporary scratch buffers */
    float *x = (float *)malloc(SIGLIP_NUM_PATCHES * SIGLIP_HIDDEN_SIZE * sizeof(float));
    float *x_norm = (float *)malloc(SIGLIP_NUM_PATCHES * SIGLIP_HIDDEN_SIZE * sizeof(float));
    float *q = (float *)malloc(SIGLIP_NUM_PATCHES * SIGLIP_HIDDEN_SIZE * sizeof(float));
    float *k = (float *)malloc(SIGLIP_NUM_PATCHES * SIGLIP_HIDDEN_SIZE * sizeof(float));
    float *v = (float *)malloc(SIGLIP_NUM_PATCHES * SIGLIP_HIDDEN_SIZE * sizeof(float));
    float *attn_out = (float *)malloc(SIGLIP_NUM_PATCHES * SIGLIP_HIDDEN_SIZE * sizeof(float));
    float *mlp_h1 = (float *)malloc(SIGLIP_NUM_PATCHES * SIGLIP_INTERM_SIZE * sizeof(float));
    float *grouped = (float *)malloc(SIGLIP_NUM_TOKENS * SIGLIP_CONN_IN * sizeof(float));

    if (!x || !x_norm || !q || !k || !v || !attn_out || !mlp_h1 || !grouped) {
        if (x) free(x);
        if (x_norm) free(x_norm);
        if (q) free(q);
        if (k) free(k);
        if (v) free(v);
        if (attn_out) free(attn_out);
        if (mlp_h1) free(mlp_h1);
        if (grouped) free(grouped);
        return -2;
    }

    /* 1. Patch Embedding Conv2D + Position Embedding */
    /* tile_pixels: [3, 512, 512], pe_w: [768, 3, 16, 16] */
    const int k_spatial = 3 * SIGLIP_PATCH_SIZE * SIGLIP_PATCH_SIZE; /* 3 * 256 = 768 */
    #pragma omp parallel for collapse(2) schedule(static) if(!omp_in_parallel())
    for (int py = 0; py < 32; py++) {
        for (int px = 0; px < 32; px++) {
            int p_idx = py * 32 + px;
            float *out_p = x + (size_t)p_idx * SIGLIP_HIDDEN_SIZE;
            const float *pos_p = model->pos_w + (size_t)p_idx * SIGLIP_HIDDEN_SIZE;

            /* Unpack patch pixels into contiguous 768-element buffer */
            float patch_pix[768];
            int p_pix_idx = 0;
            for (int c = 0; c < 3; c++) {
                const float *c_pixels = tile_pixels + (size_t)c * (SIGLIP_IMG_SIZE * SIGLIP_IMG_SIZE);
                for (int ky = 0; ky < SIGLIP_PATCH_SIZE; ky++) {
                    const float *row = c_pixels + (size_t)(py * SIGLIP_PATCH_SIZE + ky) * SIGLIP_IMG_SIZE + (px * SIGLIP_PATCH_SIZE);
                    memcpy(patch_pix + p_pix_idx, row, SIGLIP_PATCH_SIZE * sizeof(float));
                    p_pix_idx += SIGLIP_PATCH_SIZE;
                }
            }

            for (int d = 0; d < SIGLIP_HIDDEN_SIZE; d++) {
                out_p[d] = model->pe_b[d] + pos_p[d] + dot_f32_simd(patch_pix, model->pe_w + (size_t)d * k_spatial, k_spatial);
            }
        }
    }

    /* 2. 12 Transformer Layers */
    for (int l = 0; l < SIGLIP_NUM_LAYERS; l++) {
        const eif_siglip_layer_t *layer = &model->layers[l];

        /* LayerNorm 1 */
        #pragma omp parallel for schedule(static) if(!omp_in_parallel())
        for (int p = 0; p < SIGLIP_NUM_PATCHES; p++) {
            layer_norm_f32(x + (size_t)p * SIGLIP_HIDDEN_SIZE,
                           layer->ln1_w, layer->ln1_b,
                           x_norm + (size_t)p * SIGLIP_HIDDEN_SIZE,
                           SIGLIP_HIDDEN_SIZE, 1e-6f);
        }

        /* Q, K, V projections */
        siglip_gemm_qkv(gpu, x_norm,
                        layer->q_w, layer->q_b,
                        layer->k_w, layer->k_b,
                        layer->v_w, layer->v_b,
                        q, k, v,
                        SIGLIP_NUM_PATCHES, SIGLIP_HIDDEN_SIZE, SIGLIP_HIDDEN_SIZE);

        /* Multi-Head Attention: 12 heads, head_dim 64, scale = 1/8 = 0.125f */
        const float attn_scale = 1.0f / sqrtf((float)SIGLIP_HEAD_DIM);
        #pragma omp parallel for collapse(2) schedule(dynamic, 8) if(!omp_in_parallel())
        for (int h = 0; h < SIGLIP_NUM_HEADS; h++) {
            for (int i = 0; i < SIGLIP_NUM_PATCHES; i++) {
                int h_off = h * SIGLIP_HEAD_DIM;
                float local_scores[SIGLIP_NUM_PATCHES];
                const float *qi = q + (size_t)i * SIGLIP_HIDDEN_SIZE + h_off;

#if defined(__AVX512F__)
                __m512 qv[4];
                for (int d = 0; d < 4; d++) qv[d] = _mm512_loadu_ps(qi + d * 16);

                float max_s = -1e30f;
                for (int j = 0; j < SIGLIP_NUM_PATCHES; j++) {
                    const float *kj = k + (size_t)j * SIGLIP_HIDDEN_SIZE + h_off;
                    __m512 acc0 = _mm512_mul_ps(qv[0], _mm512_loadu_ps(kj + 0));
                    acc0 = _mm512_fmadd_ps(qv[1], _mm512_loadu_ps(kj + 16), acc0);
                    acc0 = _mm512_fmadd_ps(qv[2], _mm512_loadu_ps(kj + 32), acc0);
                    acc0 = _mm512_fmadd_ps(qv[3], _mm512_loadu_ps(kj + 48), acc0);
                    float s = _mm512_reduce_add_ps(acc0) * attn_scale;
                    local_scores[j] = s;
                    if (s > max_s) max_s = s;
                }

                float sum_exp = 0.0f;
                for (int j = 0; j < SIGLIP_NUM_PATCHES; j++) {
                    float ev = expf(local_scores[j] - max_s);
                    local_scores[j] = ev;
                    sum_exp += ev;
                }
                float inv_sum = 1.0f / sum_exp;

                __m512 o[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
                for (int j = 0; j < SIGLIP_NUM_PATCHES; j++) {
                    __m512 vs = _mm512_set1_ps(local_scores[j] * inv_sum);
                    const float *vj = v + (size_t)j * SIGLIP_HIDDEN_SIZE + h_off;
                    for (int d = 0; d < 4; d++) {
                        o[d] = _mm512_fmadd_ps(vs, _mm512_loadu_ps(vj + d * 16), o[d]);
                    }
                }
                float *out_ih = attn_out + (size_t)i * SIGLIP_HIDDEN_SIZE + h_off;
                for (int d = 0; d < 4; d++) {
                    _mm512_storeu_ps(out_ih + d * 16, o[d]);
                }

#elif defined(__AVX2__)
                __m256 qv[8];
                for (int d = 0; d < 8; d++) qv[d] = _mm256_loadu_ps(qi + d * 8);

                float max_s = -1e30f;
                for (int j = 0; j < SIGLIP_NUM_PATCHES; j++) {
                    const float *kj = k + (size_t)j * SIGLIP_HIDDEN_SIZE + h_off;
                    __m256 acc0 = _mm256_mul_ps(qv[0], _mm256_loadu_ps(kj + 0));
                    __m256 acc1 = _mm256_mul_ps(qv[1], _mm256_loadu_ps(kj + 8));
                    for (int d = 2; d < 8; d += 2) {
                        acc0 = _mm256_fmadd_ps(qv[d + 0], _mm256_loadu_ps(kj + (d + 0) * 8), acc0);
                        acc1 = _mm256_fmadd_ps(qv[d + 1], _mm256_loadu_ps(kj + (d + 1) * 8), acc1);
                    }
                    acc0 = _mm256_add_ps(acc0, acc1);
                    __m128 lo = _mm256_castps256_ps128(acc0);
                    __m128 hi = _mm256_extractf128_ps(acc0, 1);
                    __m128 sum = _mm_add_ps(lo, hi);
                    sum = _mm_hadd_ps(sum, sum);
                    sum = _mm_hadd_ps(sum, sum);
                    float s = _mm_cvtss_f32(sum) * attn_scale;
                    local_scores[j] = s;
                    if (s > max_s) max_s = s;
                }

                float sum_exp = 0.0f;
                for (int j = 0; j < SIGLIP_NUM_PATCHES; j++) {
                    float ev = expf(local_scores[j] - max_s);
                    local_scores[j] = ev;
                    sum_exp += ev;
                }
                float inv_sum = 1.0f / sum_exp;

                __m256 o[8];
                for (int d = 0; d < 8; d++) o[d] = _mm256_setzero_ps();
                for (int j = 0; j < SIGLIP_NUM_PATCHES; j++) {
                    __m256 vs = _mm256_set1_ps(local_scores[j] * inv_sum);
                    const float *vj = v + (size_t)j * SIGLIP_HIDDEN_SIZE + h_off;
                    for (int d = 0; d < 8; d++) {
                        o[d] = _mm256_fmadd_ps(vs, _mm256_loadu_ps(vj + d * 8), o[d]);
                    }
                }
                float *out_ih = attn_out + (size_t)i * SIGLIP_HIDDEN_SIZE + h_off;
                for (int d = 0; d < 8; d++) {
                    _mm256_storeu_ps(out_ih + d * 8, o[d]);
                }

#elif defined(__ARM_NEON) || defined(__aarch64__)
                float32x4_t qv[16];
                for (int d = 0; d < 16; d++) qv[d] = vld1q_f32(qi + d * 4);

                float max_s = -1e30f;
                for (int j = 0; j < SIGLIP_NUM_PATCHES; j++) {
                    const float *kj = k + (size_t)j * SIGLIP_HIDDEN_SIZE + h_off;
                    float32x4_t acc0 = vmulq_f32(qv[0], vld1q_f32(kj + 0));
                    float32x4_t acc1 = vmulq_f32(qv[1], vld1q_f32(kj + 4));
                    float32x4_t acc2 = vmulq_f32(qv[2], vld1q_f32(kj + 8));
                    float32x4_t acc3 = vmulq_f32(qv[3], vld1q_f32(kj + 12));
                    for (int d = 4; d < 16; d += 4) {
                        acc0 = vfmaq_f32(acc0, qv[d + 0], vld1q_f32(kj + (d + 0) * 4));
                        acc1 = vfmaq_f32(acc1, qv[d + 1], vld1q_f32(kj + (d + 1) * 4));
                        acc2 = vfmaq_f32(acc2, qv[d + 2], vld1q_f32(kj + (d + 2) * 4));
                        acc3 = vfmaq_f32(acc3, qv[d + 3], vld1q_f32(kj + (d + 3) * 4));
                    }
                    acc0 = vaddq_f32(acc0, acc1);
                    acc2 = vaddq_f32(acc2, acc3);
                    acc0 = vaddq_f32(acc0, acc2);
                    float s = vaddvq_f32(acc0) * attn_scale;
                    local_scores[j] = s;
                    if (s > max_s) max_s = s;
                }

                float sum_exp = 0.0f;
                for (int j = 0; j < SIGLIP_NUM_PATCHES; j++) {
                    float ev = expf(local_scores[j] - max_s);
                    local_scores[j] = ev;
                    sum_exp += ev;
                }
                float inv_sum = 1.0f / sum_exp;

                float32x4_t o[16];
                for (int d = 0; d < 16; d++) o[d] = vdupq_n_f32(0.0f);

                for (int j = 0; j < SIGLIP_NUM_PATCHES; j++) {
                    float32x4_t vs = vdupq_n_f32(local_scores[j] * inv_sum);
                    const float *vj = v + (size_t)j * SIGLIP_HIDDEN_SIZE + h_off;
                    for (int d = 0; d < 16; d++) {
                        o[d] = vfmaq_f32(o[d], vs, vld1q_f32(vj + d * 4));
                    }
                }

                float *out_ih = attn_out + (size_t)i * SIGLIP_HIDDEN_SIZE + h_off;
                for (int d = 0; d < 16; d++) {
                    vst1q_f32(out_ih + d * 4, o[d]);
                }
#else
                float max_s = -1e30f;
                for (int j = 0; j < SIGLIP_NUM_PATCHES; j++) {
                    const float *kj = k + (size_t)j * SIGLIP_HIDDEN_SIZE + h_off;
                    float s = dot_f32_simd(qi, kj, SIGLIP_HEAD_DIM) * attn_scale;
                    local_scores[j] = s;
                    if (s > max_s) max_s = s;
                }

                float sum_exp = 0.0f;
                for (int j = 0; j < SIGLIP_NUM_PATCHES; j++) {
                    float ev = expf(local_scores[j] - max_s);
                    local_scores[j] = ev;
                    sum_exp += ev;
                }
                float inv_sum = 1.0f / sum_exp;

                float *out_ih = attn_out + (size_t)i * SIGLIP_HIDDEN_SIZE + h_off;
                memset(out_ih, 0, SIGLIP_HEAD_DIM * sizeof(float));
                for (int j = 0; j < SIGLIP_NUM_PATCHES; j++) {
                    float s = local_scores[j] * inv_sum;
                    const float *vj = v + (size_t)j * SIGLIP_HIDDEN_SIZE + h_off;
                    for (int d = 0; d < SIGLIP_HEAD_DIM; d++) {
                        out_ih[d] += s * vj[d];
                    }
                }
#endif
            }
        }

        /* Out Projection + Residual Add */
        siglip_gemm_f32(gpu, attn_out, layer->out_w, layer->out_b, x,
                        SIGLIP_NUM_PATCHES, SIGLIP_HIDDEN_SIZE, SIGLIP_HIDDEN_SIZE, true);

        /* LayerNorm 2 */
        #pragma omp parallel for schedule(static) if(!omp_in_parallel())
        for (int p = 0; p < SIGLIP_NUM_PATCHES; p++) {
            layer_norm_f32(x + (size_t)p * SIGLIP_HIDDEN_SIZE,
                           layer->ln2_w, layer->ln2_b,
                           x_norm + (size_t)p * SIGLIP_HIDDEN_SIZE,
                           SIGLIP_HIDDEN_SIZE, 1e-6f);
        }

        /* MLP FC1 + GELU */
        siglip_gemm_f32_gelu(gpu, x_norm, layer->fc1_w, layer->fc1_b, mlp_h1,
                             SIGLIP_NUM_PATCHES, SIGLIP_INTERM_SIZE, SIGLIP_HIDDEN_SIZE);

        /* MLP FC2 + Residual Add */
        siglip_gemm_f32(gpu, mlp_h1, layer->fc2_w, layer->fc2_b, x,
                        SIGLIP_NUM_PATCHES, SIGLIP_HIDDEN_SIZE, SIGLIP_INTERM_SIZE, true);
    }

    /* 3. Post LayerNorm */
    #pragma omp parallel for schedule(static) if(!omp_in_parallel())
    for (int p = 0; p < SIGLIP_NUM_PATCHES; p++) {
        layer_norm_f32(x + (size_t)p * SIGLIP_HIDDEN_SIZE,
                       model->post_ln_w, model->post_ln_b,
                       x + (size_t)p * SIGLIP_HIDDEN_SIZE,
                       SIGLIP_HIDDEN_SIZE, 1e-6f);
    }

    /* 4. Spatial Patch Merging: 32x32 -> 8x8 blocks of 4x4 patches (16 * 768 = 12,288) */
    #pragma omp parallel for collapse(2) schedule(static) if(!omp_in_parallel())
    for (int by = 0; by < 8; by++) {
        for (int bx = 0; bx < 8; bx++) {
            int block_idx = by * 8 + bx;
            float *grp = grouped + (size_t)block_idx * SIGLIP_CONN_IN;

            for (int py = 0; py < 4; py++) {
                for (int px = 0; px < 4; px++) {
                    int p_idx = (by * 4 + py) * 32 + (bx * 4 + px);
                    int local_p = py * 4 + px;
                    memcpy(grp + (size_t)local_p * SIGLIP_HIDDEN_SIZE,
                           x + (size_t)p_idx * SIGLIP_HIDDEN_SIZE,
                           SIGLIP_HIDDEN_SIZE * sizeof(float));
                }
            }
        }
    }

    /* 5. Connector Modality Projection: [64, 12288] @ W_conn.T [12288, 576] -> [64, 576] */
    siglip_gemm_f32(gpu, grouped, model->conn_w, NULL, out_tokens,
                    SIGLIP_NUM_TOKENS, SIGLIP_CONN_OUT, SIGLIP_CONN_IN, false);

    free(x);
    free(x_norm);
    free(q);
    free(k);
    free(v);
    free(attn_out);
    free(mlp_h1);
    free(grouped);

    return 0;
}

static void resize_bilinear_rgb(const uint8_t *src, int src_w, int src_h,
                                float *dst_chw, int dst_w, int dst_h)
{
    float scale_x = (float)src_w / (float)dst_w;
    float scale_y = (float)src_h / (float)dst_h;

    for (int y = 0; y < dst_h; y++) {
        float src_y = (y + 0.5f) * scale_y - 0.5f;
        int y0 = (int)floorf(src_y);
        int y1 = y0 + 1;
        float dy = src_y - (float)y0;
        if (y0 < 0) { y0 = 0; dy = 0.0f; }
        if (y1 >= src_h) { y1 = src_h - 1; }

        for (int x = 0; x < dst_w; x++) {
            float src_x = (x + 0.5f) * scale_x - 0.5f;
            int x0 = (int)floorf(src_x);
            int x1 = x0 + 1;
            float dx = src_x - (float)x0;
            if (x0 < 0) { x0 = 0; dx = 0.0f; }
            if (x1 >= src_w) { x1 = src_w - 1; }

            for (int c = 0; c < 3; c++) {
                float v00 = (float)src[(y0 * src_w + x0) * 3 + c];
                float v01 = (float)src[(y0 * src_w + x1) * 3 + c];
                float v10 = (float)src[(y1 * src_w + x0) * 3 + c];
                float v11 = (float)src[(y1 * src_w + x1) * 3 + c];
                float val = (1.0f - dy) * ((1.0f - dx) * v00 + dx * v01) +
                            dy * ((1.0f - dx) * v10 + dx * v11);
                /* SigLIP2 normalization: val / 127.5f - 1.0f */
                dst_chw[c * (dst_h * dst_w) + y * dst_w + x] = (val / 127.5f) - 1.0f;
            }
        }
    }
}

int eif_siglip_encode_image(const eif_siglip_t *model, const char *image_path, float *out_tokens)
{
    if (!model || !model->is_initialized || !image_path || !out_tokens) return -1;

    int img_w = 0, img_h = 0, channels = 0;
    uint8_t *img_data = stbi_load(image_path, &img_w, &img_h, &channels, 3);
    if (!img_data) {
        fprintf(stderr, "[EIF SigLIP Error] Failed to load image: %s\n", image_path);
        return -2;
    }

    fprintf(stderr, "  • Vision Pipeline: Processing '%s' (%dx%d RGB)\n", image_path, img_w, img_h);
    fflush(stderr);

    /* 1. Resize entire image to fit in 4x3 grid (2048 x 1536) */
    const int grid_rows = 4;
    const int grid_cols = 3;
    const int target_h = grid_rows * SIGLIP_IMG_SIZE; /* 2048 */
    const int target_w = grid_cols * SIGLIP_IMG_SIZE; /* 1536 */

    float *full_canvas = (float *)malloc(3 * target_h * target_w * sizeof(float));
    const size_t tile_size_floats = 3 * SIGLIP_IMG_SIZE * SIGLIP_IMG_SIZE;
    float *tiles_all = (float *)malloc(13 * tile_size_floats * sizeof(float));
    if (!full_canvas || !tiles_all) {
        stbi_image_free(img_data);
        if (full_canvas) free(full_canvas);
        if (tiles_all) free(tiles_all);
        return -3;
    }

    /* Fill canvas with normalized white (+1.0) */
    for (int i = 0; i < 3 * target_h * target_w; i++) full_canvas[i] = 1.0f;

    /* Compute scaling preserving aspect ratio to fit inside target_w x target_h */
    float scale = fminf((float)target_w / (float)img_w, (float)target_h / (float)img_h);
    int scaled_w = (int)((float)img_w * scale);
    int scaled_h = (int)((float)img_h * scale);
    if (scaled_w > target_w) scaled_w = target_w;
    if (scaled_h > target_h) scaled_h = target_h;

    float *scaled_img = (float *)malloc(3 * scaled_h * scaled_w * sizeof(float));
    if (!scaled_img) {
        stbi_image_free(img_data);
        free(full_canvas);
        free(tiles_all);
        return -3;
    }

    resize_bilinear_rgb(img_data, img_w, img_h, scaled_img, scaled_w, scaled_h);

    /* Place scaled image into full canvas (top-left aligned) */
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < scaled_h; y++) {
            memcpy(full_canvas + (size_t)c * (target_h * target_w) + (size_t)y * target_w,
                   scaled_img + (size_t)c * (scaled_h * scaled_w) + (size_t)y * scaled_w,
                   scaled_w * sizeof(float));
        }
    }
    free(scaled_img);

    /* 2. Crop 12 local tiles into tiles_all */
    for (int r = 0; r < grid_rows; r++) {
        for (int c = 0; c < grid_cols; c++) {
            int tidx = r * grid_cols + c;
            float *dst_tile = tiles_all + (size_t)tidx * tile_size_floats;
            for (int ch = 0; ch < 3; ch++) {
                const float *src_plane = full_canvas + (size_t)ch * (target_h * target_w);
                float *dst_plane = dst_tile + (size_t)ch * (SIGLIP_IMG_SIZE * SIGLIP_IMG_SIZE);
                for (int ty = 0; ty < SIGLIP_IMG_SIZE; ty++) {
                    const float *src_row = src_plane + (size_t)(r * SIGLIP_IMG_SIZE + ty) * target_w + (c * SIGLIP_IMG_SIZE);
                    memcpy(dst_plane + (size_t)ty * SIGLIP_IMG_SIZE, src_row, SIGLIP_IMG_SIZE * sizeof(float));
                }
            }
        }
    }
    free(full_canvas);

    /* 3. Prepare 13th tile: Global thumbnail into tiles_all[12] */
    float *thumb_tile = tiles_all + (size_t)12 * tile_size_floats;
    resize_bilinear_rgb(img_data, img_w, img_h, thumb_tile, SIGLIP_IMG_SIZE, SIGLIP_IMG_SIZE);
    stbi_image_free(img_data);

    /* 4. Encode all 13 tiles across CPU cores */
    fprintf(stderr, "  • Vision ViT: Processing 13 tiles across CPU cores...\n");
    fflush(stderr);

    for (int t = 0; t < 13; t++) {
        const float *tile_pix = tiles_all + (size_t)t * tile_size_floats;
        float *tile_out = out_tokens + (size_t)t * SIGLIP_NUM_TOKENS * SIGLIP_CONN_OUT;

        /* Deduplication: If tile is identical to an earlier tile (e.g. blank margin), reuse embeddings */
        int match_idx = -1;
        for (int prev = 0; prev < t; prev++) {
            const float *prev_pix = tiles_all + (size_t)prev * tile_size_floats;
            if (memcmp(tile_pix, prev_pix, tile_size_floats * sizeof(float)) == 0) {
                match_idx = prev;
                break;
            }
        }

        if (match_idx >= 0) {
            float *prev_out = out_tokens + (size_t)match_idx * SIGLIP_NUM_TOKENS * SIGLIP_CONN_OUT;
            memcpy(tile_out, prev_out, (size_t)SIGLIP_NUM_TOKENS * SIGLIP_CONN_OUT * sizeof(float));
        } else {
            eif_siglip_encode_tile(model, tile_pix, tile_out);
        }
    }

    free(tiles_all);

    fprintf(stderr, "  • Vision Pipeline: Encoded 13 tiles -> 832 visual tokens (dim=576) in parallel\n");
    fflush(stderr);
    return 0;
}
