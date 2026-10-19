/**
 * @file eif_siglip.c
 * @brief Native Pure C99 SigLIP2 Vision Encoder & Connector Implementation.
 */

#include "eif_siglip.h"
#include "eif_quantize_bitnet.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

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

/* Bytes of one stored ViT projection [N, K] for the header's qtype (codes + scales). */
static size_t siglip_mat_bytes(const eif_siglip_header_t *h, int N, int K)
{
    if (h->qtype == EIF_SIGLIP_Q_INT8) return (size_t)N * K + (size_t)N * sizeof(float);
    if (h->qtype == EIF_SIGLIP_Q_INT4) {
        int g = h->reserved[0] > 0 ? (int)h->reserved[0] : K;
        return (size_t)N * K / 2 + (size_t)N * (K / g) * sizeof(float);
    }
    return (size_t)N * K * sizeof(float);
}

/* Point *w (FP32) or *qm (quantised) at the matrix stored at ptr; returns the pointer past it. */
static const uint8_t *siglip_take_mat(const eif_siglip_header_t *h, const uint8_t *ptr, int N, int K,
                                      const float **w, eif_siglip_qmat_t *qm)
{
    if (h->qtype == EIF_SIGLIP_Q_FP32) {
        *w = (const float *)ptr;
        qm->codes = NULL;
        qm->scales = NULL;
    } else {
        *w = NULL;
        qm->codes = ptr;
        qm->scales = (const float *)(ptr + (h->qtype == EIF_SIGLIP_Q_INT8 ? (size_t)N * K : (size_t)N * K / 2));
    }
    return ptr + siglip_mat_bytes(h, N, K);
}

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
    if (hdr->qtype != EIF_SIGLIP_Q_FP32 && hdr->qtype != EIF_SIGLIP_Q_INT8 && hdr->qtype != EIF_SIGLIP_Q_INT4) {
        fprintf(stderr, "[EIF SigLIP Error] Unsupported vision qtype %d\n", hdr->qtype);
        return -5;
    }
    if (hdr->qtype == EIF_SIGLIP_Q_INT4 && hdr->reserved[0] > 0 &&
        (SIGLIP_HIDDEN_SIZE % hdr->reserved[0] != 0 || SIGLIP_INTERM_SIZE % hdr->reserved[0] != 0)) {
        fprintf(stderr, "[EIF SigLIP Error] INT4 group %u does not divide 768 and 3072\n", hdr->reserved[0]);
        return -5;
    }

    const uint8_t *ptr = (const uint8_t *)buffer + sizeof(eif_siglip_header_t);
    if (hdr->reserved[1]) {
        model->had_signs_768 = (const float *)ptr;
        ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);
        model->had_signs_3072 = (const float *)ptr;
        ptr += SIGLIP_INTERM_SIZE * sizeof(float);
    }

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

        ptr = siglip_take_mat(hdr, ptr, SIGLIP_HIDDEN_SIZE, SIGLIP_HIDDEN_SIZE, &layer->q_w, &layer->qm[0]);
        layer->q_b = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);

        ptr = siglip_take_mat(hdr, ptr, SIGLIP_HIDDEN_SIZE, SIGLIP_HIDDEN_SIZE, &layer->k_w, &layer->qm[1]);
        layer->k_b = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);

        ptr = siglip_take_mat(hdr, ptr, SIGLIP_HIDDEN_SIZE, SIGLIP_HIDDEN_SIZE, &layer->v_w, &layer->qm[2]);
        layer->v_b = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);

        ptr = siglip_take_mat(hdr, ptr, SIGLIP_HIDDEN_SIZE, SIGLIP_HIDDEN_SIZE, &layer->out_w, &layer->qm[3]);
        layer->out_b = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);

        layer->ln2_w = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);
        layer->ln2_b = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);

        ptr = siglip_take_mat(hdr, ptr, SIGLIP_INTERM_SIZE, SIGLIP_HIDDEN_SIZE, &layer->fc1_w, &layer->qm[4]);
        layer->fc1_b = (const float *)ptr; ptr += SIGLIP_INTERM_SIZE * sizeof(float);

        ptr = siglip_take_mat(hdr, ptr, SIGLIP_HIDDEN_SIZE, SIGLIP_INTERM_SIZE, &layer->fc2_w, &layer->qm[5]);
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

size_t eif_siglip_struct_size(void) { return sizeof(eif_siglip_t); }

int eif_siglip_dequant_weight(const eif_siglip_t *model, int l, int m, float *out)
{
    if (!model || !model->is_initialized || !out || l < 0 || l >= SIGLIP_NUM_LAYERS || m < 0 || m > 5) return -1;
    const eif_siglip_layer_t *L = &model->layers[l];
    const float *fp[6] = {L->q_w, L->k_w, L->v_w, L->out_w, L->fc1_w, L->fc2_w};
    const int N = (m == 4) ? SIGLIP_INTERM_SIZE : SIGLIP_HIDDEN_SIZE;
    const int K = (m == 5) ? SIGLIP_INTERM_SIZE : SIGLIP_HIDDEN_SIZE;
    const eif_siglip_header_t *h = &model->header;
    if (h->qtype == EIF_SIGLIP_Q_FP32) {
        memcpy(out, fp[m], (size_t)N * K * sizeof(float));
        return 0;
    }
    const eif_siglip_qmat_t *q = &L->qm[m];
    const int g = (h->qtype == EIF_SIGLIP_Q_INT4 && h->reserved[0] > 0) ? (int)h->reserved[0] : K;
    const int ng = K / g;
    const float *signs = (K == SIGLIP_HIDDEN_SIZE) ? model->had_signs_768 : model->had_signs_3072;
    const int block = eif_hadamard_block_size(K);
    #pragma omp parallel for schedule(static) if(!omp_in_parallel())
    for (int r = 0; r < N; r++) {
        float *row = out + (size_t)r * K;
        if (h->qtype == EIF_SIGLIP_Q_INT8) {
            const int8_t *c = (const int8_t *)q->codes + (size_t)r * K;
            const float s = q->scales[r];
            for (int i = 0; i < K; i++) row[i] = (float)c[i] * s;
        } else {
            const uint8_t *c = q->codes + (size_t)r * K / 2;
            const float *s = q->scales + (size_t)r * ng;
            for (int i = 0; i < K; i += 2) {
                const uint8_t b = c[i / 2];
                const int lo = (int)(b & 0x0F) - ((b & 0x08) ? 16 : 0);
                const int hi = (int)(b >> 4) - ((b & 0x80) ? 16 : 0);
                row[i] = (float)lo * s[i / g];
                row[i + 1] = (float)hi * s[(i + 1) / g];
            }
        }
        if (h->reserved[1]) {
            /* stored W' = W·D·H  ->  W = (W' H) D   (H is symmetric and orthonormal) */
            eif_hadamard_rotate_f32(row, NULL, K, block);
            for (int i = 0; i < K; i++) row[i] *= signs[i];
        }
    }
    return 0;
}

/* FP32 view of projection m of layer l: the stored weight, or dequantised into dst. */
static const float *siglip_weight(const eif_siglip_t *model, int l, int m, float *dst)
{
    const eif_siglip_layer_t *L = &model->layers[l];
    const float *fp[6] = {L->q_w, L->k_w, L->v_w, L->out_w, L->fc1_w, L->fc2_w};
    if (model->header.qtype == EIF_SIGLIP_Q_FP32) return fp[m];
    eif_siglip_dequant_weight(model, l, m, dst);
    return dst;
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

/* ---------------------------------------------------------------------------------------------
 * INT8 x INT8 -> INT32 ViT projections on Armv8.6 i8mm (SMMLA), used for INT8-exported weights.
 * Activations are quantised per row (absmax / 127), weights keep their per-row export scales, so
 *   C[m][n] = sa[m] * sw[n] * sum_k a8[m][k] * w8[n][k] + b[n].
 * Both operands are packed in row pairs, 8 bytes of each row per k-step: one 16-byte vector feeds SMMLA,
 * which adds the 2x2 block a(2x8) . b(2x8)^T. BITLLM_VIT_FP32=1 forces the FP32 path.
 * ------------------------------------------------------------------------------------------- */
#if defined(__aarch64__) && defined(__linux__) && defined(__GNUC__)
#include <arm_neon.h>
#include <sys/auxv.h>
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1 << 13)
#endif
#define SIGLIP_I8MM_BUILD 1
#ifndef SIGLIP_I8_DEFAULT_MASK
#define SIGLIP_I8_DEFAULT_MASK 0x3F /* all six; fc2 uses grouped activation scales */
#endif

/* bit m set -> projection m (q, k, v, out, fc1, fc2) runs as INT8 GEMM; BITLLM_VIT_I8_MASK overrides */
static int siglip_i8_mask(void)
{
    static int mask = -1;
    if (mask < 0) {
        const char *e = getenv("BITLLM_VIT_I8_MASK");
        mask = e ? (int)strtol(e, NULL, 0) : SIGLIP_I8_DEFAULT_MASK;
    }
    return mask;
}

int eif_siglip_int8_gemm_available(void)
{
    static int avail = -1;
    if (avail < 0) {
        const char *e = getenv("BITLLM_VIT_FP32");
        avail = (e && e[0] == '1') ? 0 : ((getauxval(AT_HWCAP2) & HWCAP2_I8MM) ? 1 : 0);
    }
    return avail;
}

/* rows [M, K] -> pair-packed int8 + one scale per (row, group of G inputs): sa[m * (K / G) + g].
 * G = K is per-row scaling; a smaller G keeps small activations resolvable next to outlier channels
 * (the GELU output feeding fc2). */
static void siglip_pack_a_i8(const float *A, int M, int K, int G, int8_t *Ap, float *sa)
{
    const int KB = K / 8, ng = K / G;
    #pragma omp parallel for schedule(static) if(!omp_in_parallel())
    for (int m = 0; m < M; m++) {
        const float *a = A + (size_t)m * K;
        int8_t *dst = Ap + (size_t)(m / 2) * KB * 16 + (m & 1) * 8;
        for (int g = 0; g < ng; g++) {
            float amax = 0.0f;
            for (int k = g * G; k < (g + 1) * G; k++) { float v = fabsf(a[k]); if (v > amax) amax = v; }
            const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
            const float inv = 1.0f / s;
            sa[(size_t)m * ng + g] = s;
            for (int kb = g * G / 8; kb < (g + 1) * G / 8; kb++)
                for (int j = 0; j < 8; j++) {
                    float q = a[kb * 8 + j] * inv;
                    int qi = (int)(q + (q >= 0.0f ? 0.5f : -0.5f));
                    dst[(size_t)kb * 16 + j] = (int8_t)(qi > 127 ? 127 : (qi < -127 ? -127 : qi));
                }
        }
    }
}

/* int8 weight rows [N, K] (export layout) -> pair-packed */
static void siglip_pack_w_i8(const int8_t *W, int N, int K, int8_t *Wp)
{
    const int KB = K / 8;
    #pragma omp parallel for schedule(static) if(!omp_in_parallel())
    for (int n = 0; n < N; n++) {
        const int8_t *w = W + (size_t)n * K;
        int8_t *dst = Wp + (size_t)(n / 2) * KB * 16 + (n & 1) * 8;
        for (int kb = 0; kb < KB; kb++) memcpy(dst + (size_t)kb * 16, w + kb * 8, 8);
    }
}

static inline float siglip_gelu_tanh(float v)
{
    const float u = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + tanhf(u));
}

/* mode: 0 = store, 1 = GELU then store, 2 = add to C (residual) */
__attribute__((target("+i8mm")))
static void siglip_gemm_i8_packed(const int8_t *Ap, const float *sa, const int8_t *Wp, const float *sw,
                                  const float *bias, float *C, int M, int N, int K, int G, int mode)
{
    const int KB = K / 8, ng = K / G, gkb = G / 8;
    #pragma omp parallel for schedule(static) if(!omp_in_parallel())
    for (int mb = 0; mb < M; mb += 8) {
        const int8_t *a0 = Ap + (size_t)(mb / 2 + 0) * KB * 16;
        const int8_t *a1 = a0 + (size_t)KB * 16;
        const int8_t *a2 = a1 + (size_t)KB * 16;
        const int8_t *a3 = a2 + (size_t)KB * 16;
        for (int nb = 0; nb < N; nb += 8) {
            const int8_t *b0 = Wp + (size_t)(nb / 2 + 0) * KB * 16;
            const int8_t *b1 = b0 + (size_t)KB * 16;
            const int8_t *b2 = b1 + (size_t)KB * 16;
            const int8_t *b3 = b2 + (size_t)KB * 16;
            /* facc[i][j] = 2x2 block (rows 2i, 2i+1 x cols 2j, 2j+1) of sum_g sa[row][g] * int32 partial sum */
            float32x4_t facc[4][4];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) facc[i][j] = vdupq_n_f32(0.0f);
            for (int g = 0; g < ng; g++) {
                int32x4_t acc[4][4];
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++) acc[i][j] = vdupq_n_s32(0);
                for (int kb = g * gkb; kb < (g + 1) * gkb; kb++) {
                    const size_t o = (size_t)kb * 16;
                    const int8x16_t va[4] = {vld1q_s8(a0 + o), vld1q_s8(a1 + o), vld1q_s8(a2 + o), vld1q_s8(a3 + o)};
                    const int8x16_t vb[4] = {vld1q_s8(b0 + o), vld1q_s8(b1 + o), vld1q_s8(b2 + o), vld1q_s8(b3 + o)};
                    for (int i = 0; i < 4; i++)
                        for (int j = 0; j < 4; j++) acc[i][j] = vmmlaq_s32(acc[i][j], va[i], vb[j]);
                }
                for (int i = 0; i < 4; i++) {
                    const int r0 = mb + 2 * i;
                    const float s0 = sa[(size_t)r0 * ng + g], s1 = sa[(size_t)(r0 + 1) * ng + g];
                    const float sv[4] = {s0, s0, s1, s1}; /* lanes: (r0,c0) (r0,c1) (r1,c0) (r1,c1) */
                    const float32x4_t vs = vld1q_f32(sv);
                    for (int j = 0; j < 4; j++) facc[i][j] = vfmaq_f32(facc[i][j], vcvtq_f32_s32(acc[i][j]), vs);
                }
            }
            for (int i = 0; i < 4; i++) {
                const int r0 = mb + 2 * i, r1 = r0 + 1;
                for (int j = 0; j < 4; j++) {
                    const int c0 = nb + 2 * j, c1 = c0 + 1;
                    float v[4];
                    vst1q_f32(v, facc[i][j]);
                    float o00 = v[0] * sw[c0] + (bias ? bias[c0] : 0.0f);
                    float o01 = v[1] * sw[c1] + (bias ? bias[c1] : 0.0f);
                    float o10 = v[2] * sw[c0] + (bias ? bias[c0] : 0.0f);
                    float o11 = v[3] * sw[c1] + (bias ? bias[c1] : 0.0f);
                    float *p0 = C + (size_t)r0 * N, *p1 = C + (size_t)r1 * N;
                    if (mode == 1) {
                        p0[c0] = siglip_gelu_tanh(o00); p0[c1] = siglip_gelu_tanh(o01);
                        p1[c0] = siglip_gelu_tanh(o10); p1[c1] = siglip_gelu_tanh(o11);
                    } else if (mode == 2) {
                        p0[c0] += o00; p0[c1] += o01; p1[c0] += o10; p1[c1] += o11;
                    } else {
                        p0[c0] = o00; p0[c1] = o01; p1[c0] = o10; p1[c1] = o11;
                    }
                }
            }
        }
    }
}

/* one projection: weights of matrix m of layer l, A already packed */
static void siglip_proj_i8(const eif_siglip_t *model, int l, int m, const int8_t *Ap, const float *sa, int G,
                           const float *bias, float *C, int M, int N, int K, int mode, int8_t *wscratch)
{
    const eif_siglip_qmat_t *q = &model->layers[l].qm[m];
    siglip_pack_w_i8((const int8_t *)q->codes, N, K, wscratch);
    siglip_gemm_i8_packed(Ap, sa, wscratch, q->scales, bias, C, M, N, K, G, mode);
}

/* activation group size of the fc2 input (K = 3072); BITLLM_VIT_FC2_GROUP overrides (3072 = per row) */
static int siglip_fc2_group(void)
{
    static int g = -1;
    if (g < 0) {
        const char *e = getenv("BITLLM_VIT_FC2_GROUP");
        g = e ? atoi(e) : 64;
        if (g < 8 || g % 8 != 0 || SIGLIP_INTERM_SIZE % g != 0) g = 64;
    }
    return g;
}
int eif_siglip_int8_gemm(const float *A, int M, int K, int G, const int8_t *W8, const float *sw, int N,
                         const float *bias, float *C, int mode)
{
    if (!eif_siglip_int8_gemm_available()) return -1;
    if (!A || !W8 || !sw || !C || M % 8 || N % 8 || K % 8 || G % 8 || K % G) return -2;
    int8_t *ap = (int8_t *)malloc((size_t)M * K), *wp = (int8_t *)malloc((size_t)N * K);
    float *sa = (float *)malloc((size_t)M * (K / G) * sizeof(float));
    if (!ap || !wp || !sa) { free(ap); free(wp); free(sa); return -3; }
    siglip_pack_a_i8(A, M, K, G, ap, sa);
    siglip_pack_w_i8(W8, N, K, wp);
    siglip_gemm_i8_packed(ap, sa, wp, sw, bias, C, M, N, K, G, mode);
    free(ap); free(wp); free(sa);
    return 0;
}
#else
int eif_siglip_int8_gemm_available(void) { return 0; }
int eif_siglip_int8_gemm(const float *A, int M, int K, int G, const int8_t *W8, const float *sw, int N,
                         const float *bias, float *C, int mode)
{
    (void)A; (void)M; (void)K; (void)G; (void)W8; (void)sw; (void)N; (void)bias; (void)C; (void)mode;
    return -1;
}
#endif

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
    /* dequantised weights of the current GEMM (3 x 768x768 for q/k/v fits in 3072x768) */
    float *wbuf = (model->header.qtype != EIF_SIGLIP_Q_FP32)
                      ? (float *)malloc((size_t)SIGLIP_INTERM_SIZE * SIGLIP_HIDDEN_SIZE * sizeof(float))
                      : NULL;
    const size_t wsq = (size_t)SIGLIP_HIDDEN_SIZE * SIGLIP_HIDDEN_SIZE;
#ifdef SIGLIP_I8MM_BUILD
    /* INT8 weights without Hadamard rotation can run as true INT8 GEMMs (unless a GPU takes the GEMMs) */
    int use_i8 = model->header.qtype == EIF_SIGLIP_Q_INT8 && !model->header.reserved[1] &&
                       !(gpu && gpu->is_available) && eif_siglip_int8_gemm_available();
    int8_t *a8 = use_i8 ? (int8_t *)malloc((size_t)SIGLIP_NUM_PATCHES * SIGLIP_INTERM_SIZE) : NULL;
    float *a_scale = use_i8 ? (float *)malloc((size_t)SIGLIP_NUM_PATCHES * (SIGLIP_INTERM_SIZE / 8) * sizeof(float)) : NULL;
    int8_t *w8 = use_i8 ? (int8_t *)malloc((size_t)SIGLIP_INTERM_SIZE * SIGLIP_HIDDEN_SIZE) : NULL;
    if (use_i8 && (!a8 || !a_scale || !w8)) {
        free(a8); free(a_scale); free(w8);
        a8 = NULL; a_scale = NULL; w8 = NULL;
        use_i8 = 0;
    }
#define SIGLIP_I8_FREE() do { free(a8); free(a_scale); free(w8); } while (0)
#else
#define SIGLIP_I8_FREE() do { } while (0)
#endif

    if (!x || !x_norm || !q || !k || !v || !attn_out || !mlp_h1 || !grouped ||
        (model->header.qtype != EIF_SIGLIP_Q_FP32 && !wbuf)) {
        if (wbuf) free(wbuf);
        SIGLIP_I8_FREE();
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
#ifdef SIGLIP_I8MM_BUILD
        if (use_i8 && (siglip_i8_mask() & 0x7) == 0x7) {
            siglip_pack_a_i8(x_norm, SIGLIP_NUM_PATCHES, SIGLIP_HIDDEN_SIZE, SIGLIP_HIDDEN_SIZE, a8, a_scale);
            siglip_proj_i8(model, l, 0, a8, a_scale, SIGLIP_HIDDEN_SIZE, layer->q_b, q, SIGLIP_NUM_PATCHES, SIGLIP_HIDDEN_SIZE, SIGLIP_HIDDEN_SIZE, 0, w8);
            siglip_proj_i8(model, l, 1, a8, a_scale, SIGLIP_HIDDEN_SIZE, layer->k_b, k, SIGLIP_NUM_PATCHES, SIGLIP_HIDDEN_SIZE, SIGLIP_HIDDEN_SIZE, 0, w8);
            siglip_proj_i8(model, l, 2, a8, a_scale, SIGLIP_HIDDEN_SIZE, layer->v_b, v, SIGLIP_NUM_PATCHES, SIGLIP_HIDDEN_SIZE, SIGLIP_HIDDEN_SIZE, 0, w8);
        } else
#endif
        siglip_gemm_qkv(gpu, x_norm,
                        siglip_weight(model, l, 0, wbuf), layer->q_b,
                        siglip_weight(model, l, 1, wbuf ? wbuf + wsq : NULL), layer->k_b,
                        siglip_weight(model, l, 2, wbuf ? wbuf + 2 * wsq : NULL), layer->v_b,
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
#ifdef SIGLIP_I8MM_BUILD
        if (use_i8 && (siglip_i8_mask() & 0x8)) {
            siglip_pack_a_i8(attn_out, SIGLIP_NUM_PATCHES, SIGLIP_HIDDEN_SIZE, SIGLIP_HIDDEN_SIZE, a8, a_scale);
            siglip_proj_i8(model, l, 3, a8, a_scale, SIGLIP_HIDDEN_SIZE, layer->out_b, x, SIGLIP_NUM_PATCHES, SIGLIP_HIDDEN_SIZE, SIGLIP_HIDDEN_SIZE, 2, w8);
        } else
#endif
        siglip_gemm_f32(gpu, attn_out, siglip_weight(model, l, 3, wbuf), layer->out_b, x,
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
#ifdef SIGLIP_I8MM_BUILD
        if (use_i8 && (siglip_i8_mask() & 0x10)) {
            siglip_pack_a_i8(x_norm, SIGLIP_NUM_PATCHES, SIGLIP_HIDDEN_SIZE, SIGLIP_HIDDEN_SIZE, a8, a_scale);
            siglip_proj_i8(model, l, 4, a8, a_scale, SIGLIP_HIDDEN_SIZE, layer->fc1_b, mlp_h1, SIGLIP_NUM_PATCHES, SIGLIP_INTERM_SIZE, SIGLIP_HIDDEN_SIZE, 1, w8);
        } else
#endif
        siglip_gemm_f32_gelu(gpu, x_norm, siglip_weight(model, l, 4, wbuf), layer->fc1_b, mlp_h1,
                             SIGLIP_NUM_PATCHES, SIGLIP_INTERM_SIZE, SIGLIP_HIDDEN_SIZE);

        /* MLP FC2 + Residual Add */
#ifdef SIGLIP_I8MM_BUILD
        if (use_i8 && (siglip_i8_mask() & 0x20)) {
            siglip_pack_a_i8(mlp_h1, SIGLIP_NUM_PATCHES, SIGLIP_INTERM_SIZE, siglip_fc2_group(), a8, a_scale);
            siglip_proj_i8(model, l, 5, a8, a_scale, siglip_fc2_group(), layer->fc2_b, x, SIGLIP_NUM_PATCHES, SIGLIP_HIDDEN_SIZE, SIGLIP_INTERM_SIZE, 2, w8);
        } else
#endif
        siglip_gemm_f32(gpu, mlp_h1, siglip_weight(model, l, 5, wbuf), layer->fc2_b, x,
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
    if (wbuf) free(wbuf);
    SIGLIP_I8_FREE();

    return 0;
}

/* ---------------------------------------------------------------------------------------------
 * Idefics3 image preprocessing (matches transformers' Idefics3ImageProcessor, size.longest_edge
 * = 2048, max_image_size.longest_edge = 512, resample = LANCZOS, do_image_splitting = true):
 *   1. scale the longest edge to 2048 (small images are enlarged), keeping the aspect ratio;
 *   2. round both sides up to multiples of 512 (this stretches the image slightly);
 *   3. cut it into rows x cols tiles of 512 x 512 (row-major) and append a 512 x 512 global view.
 * The number of tiles is therefore rows * cols + 1 (5 .. 17), not a constant 13.
 * ------------------------------------------------------------------------------------------- */

static int ceil_div_i(int a, int b) { return (a + b - 1) / b; }

/* _resize_output_size_rescale_to_max_len(min_len = 1, max_len = 2048) */
static void rescale_to_longest_edge(int w, int h, int max_len, int *out_w, int *out_h)
{
    double aspect = (double)w / (double)h;
    int nw = w, nh = h;
    if (w >= h) {
        nw = max_len;
        nh = (int)((double)nw / aspect);
        if (nh % 2 != 0) nh++;
    } else {
        nh = max_len;
        nw = (int)((double)nh * aspect);
        if (nw % 2 != 0) nw++;
    }
    *out_w = nw < 1 ? 1 : nw;
    *out_h = nh < 1 ? 1 : nh;
}

/* resize_for_vision_encoder: sides rounded up to multiples of `m` */
static void round_up_to_multiple(int w, int h, int m, int *out_w, int *out_h)
{
    double aspect = (double)w / (double)h;
    int nw = w, nh = h;
    if (w >= h) {
        nw = ceil_div_i(w, m) * m;
        nh = (int)((double)nw / aspect);
        nh = ceil_div_i(nh, m) * m;
    } else {
        nh = ceil_div_i(h, m) * m;
        nw = (int)((double)nh * aspect);
        nw = ceil_div_i(nw, m) * m;
    }
    *out_w = nw;
    *out_h = nh;
}

int eif_siglip_plan_tiles(int img_w, int img_h, int *rows, int *cols)
{
    if (img_w <= 0 || img_h <= 0) return -1;
    int w1, h1, w2, h2;
    rescale_to_longest_edge(img_w, img_h, 2048, &w1, &h1);
    round_up_to_multiple(w1, h1, SIGLIP_IMG_SIZE, &w2, &h2);
    if (rows) *rows = h2 / SIGLIP_IMG_SIZE;
    if (cols) *cols = w2 / SIGLIP_IMG_SIZE;
    return 0;
}

/* Pillow-compatible Lanczos (a = 3) resampling of an interleaved uint8 RGB image: separable,
 * anti-aliased when shrinking, horizontal pass first, 8-bit rounding between the passes. */
static double lanczos3(double x)
{
    if (x <= -3.0 || x >= 3.0) return 0.0;
    if (x > -1e-9 && x < 1e-9) return 1.0;
    double px = 3.14159265358979323846 * x;
    return (sin(px) / px) * (sin(px / 3.0) / (px / 3.0));
}

static uint8_t clip_u8(double v)
{
    int r = (int)floor(v + 0.5);
    return (uint8_t)(r < 0 ? 0 : (r > 255 ? 255 : r));
}

static void resample_axis(const uint8_t *src, int src_w, int src_h, uint8_t *dst, int dst_len, int horizontal)
{
    int in_len = horizontal ? src_w : src_h;
    double scale = (double)in_len / (double)dst_len;
    double filterscale = scale < 1.0 ? 1.0 : scale;
    double support = 3.0 * filterscale;
    int kmax = (int)ceil(support) * 2 + 1;
    double *w = (double *)malloc((size_t)kmax * sizeof(double));
    if (!w) return;

    for (int o = 0; o < dst_len; o++) {
        double center = ((double)o + 0.5) * scale;
        int xmin = (int)(center - support + 0.5);
        int xmax = (int)(center + support + 0.5);
        if (xmin < 0) xmin = 0;
        if (xmax > in_len) xmax = in_len;
        int n = xmax - xmin;
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            w[k] = lanczos3(((double)(xmin + k) - center + 0.5) / filterscale);
            sum += w[k];
        }
        if (sum != 0.0) for (int k = 0; k < n; k++) w[k] /= sum;

        if (horizontal) {
            for (int y = 0; y < src_h; y++) {
                for (int c = 0; c < 3; c++) {
                    double acc = 0.0;
                    for (int k = 0; k < n; k++) acc += w[k] * src[((size_t)y * src_w + xmin + k) * 3 + c];
                    dst[((size_t)y * dst_len + o) * 3 + c] = clip_u8(acc);
                }
            }
        } else {
            for (int x = 0; x < src_w; x++) {
                for (int c = 0; c < 3; c++) {
                    double acc = 0.0;
                    for (int k = 0; k < n; k++) acc += w[k] * src[((size_t)(xmin + k) * src_w + x) * 3 + c];
                    dst[((size_t)o * src_w + x) * 3 + c] = clip_u8(acc);
                }
            }
        }
    }
    free(w);
}

/* Returns a new malloc'd dst_w x dst_h RGB image (or NULL). */
static uint8_t *resize_lanczos_rgb(const uint8_t *src, int src_w, int src_h, int dst_w, int dst_h)
{
    uint8_t *cur = (uint8_t *)malloc((size_t)src_w * src_h * 3);
    if (!cur) return NULL;
    memcpy(cur, src, (size_t)src_w * src_h * 3);
    int cw = src_w, ch = src_h;
    if (dst_w != cw) {
        uint8_t *tmp = (uint8_t *)malloc((size_t)dst_w * ch * 3);
        if (!tmp) { free(cur); return NULL; }
        resample_axis(cur, cw, ch, tmp, dst_w, 1);
        free(cur); cur = tmp; cw = dst_w;
    }
    if (dst_h != ch) {
        uint8_t *tmp = (uint8_t *)malloc((size_t)cw * dst_h * 3);
        if (!tmp) { free(cur); return NULL; }
        resample_axis(cur, cw, ch, tmp, dst_h, 0);
        free(cur); cur = tmp; ch = dst_h;
    }
    return cur;
}

/* uint8 HWC crop -> normalised float CHW tile ((v / 255 - 0.5) / 0.5) */
static void to_chw_tile(const uint8_t *img, int img_w, int x0, int y0, int tw, int th, float *dst)
{
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < th; y++)
            for (int x = 0; x < tw; x++)
                dst[(size_t)c * th * tw + (size_t)y * tw + x] =
                    (float)img[((size_t)(y0 + y) * img_w + x0 + x) * 3 + c] / 127.5f - 1.0f;
}

int eif_siglip_preprocess_image(const char *image_path, float **out_tiles, int *rows, int *cols)
{
    if (!image_path || !out_tiles || !rows || !cols) return -1;
    int img_w = 0, img_h = 0, channels = 0;
    uint8_t *img = stbi_load(image_path, &img_w, &img_h, &channels, 3);
    if (!img) {
        fprintf(stderr, "[EIF SigLIP Error] Failed to load image: %s\n", image_path);
        return -2;
    }
    fprintf(stderr, "  • Vision Pipeline: Processing '%s' (%dx%d RGB)\n", image_path, img_w, img_h);

    int w1, h1, w2, h2;
    rescale_to_longest_edge(img_w, img_h, 2048, &w1, &h1);
    round_up_to_multiple(w1, h1, SIGLIP_IMG_SIZE, &w2, &h2);
    int nr = h2 / SIGLIP_IMG_SIZE, nc = w2 / SIGLIP_IMG_SIZE;

    uint8_t *step1 = resize_lanczos_rgb(img, img_w, img_h, w1, h1);
    stbi_image_free(img);
    uint8_t *step2 = step1 ? resize_lanczos_rgb(step1, w1, h1, w2, h2) : NULL;
    uint8_t *glob = step2 ? resize_lanczos_rgb(step2, w2, h2, SIGLIP_IMG_SIZE, SIGLIP_IMG_SIZE) : NULL;
    free(step1);

    const size_t tile_floats = 3 * (size_t)SIGLIP_IMG_SIZE * SIGLIP_IMG_SIZE;
    float *tiles = glob ? (float *)malloc((size_t)(nr * nc + 1) * tile_floats * sizeof(float)) : NULL;
    if (!tiles) { free(step2); free(glob); return -3; }

    for (int r = 0; r < nr; r++)
        for (int c = 0; c < nc; c++)
            to_chw_tile(step2, w2, c * SIGLIP_IMG_SIZE, r * SIGLIP_IMG_SIZE, SIGLIP_IMG_SIZE, SIGLIP_IMG_SIZE,
                        tiles + (size_t)(r * nc + c) * tile_floats);
    to_chw_tile(glob, SIGLIP_IMG_SIZE, 0, 0, SIGLIP_IMG_SIZE, SIGLIP_IMG_SIZE, tiles + (size_t)(nr * nc) * tile_floats);
    free(step2);
    free(glob);

    *out_tiles = tiles;
    *rows = nr;
    *cols = nc;
    return 0;
}

int eif_siglip_encode_image(const eif_siglip_t *model, const char *image_path, float *out_tokens,
                            int *out_rows, int *out_cols)
{
    if (!model || !model->is_initialized || !image_path || !out_tokens) return -1;

    float *tiles_all = NULL;
    int rows = 0, cols = 0;
    int rc = eif_siglip_preprocess_image(image_path, &tiles_all, &rows, &cols);
    if (rc != 0) return rc;

    const int n_tiles = rows * cols + 1;
    const size_t tile_size_floats = 3 * (size_t)SIGLIP_IMG_SIZE * SIGLIP_IMG_SIZE;
    fprintf(stderr, "  • Vision ViT: Processing %d tiles (%d x %d grid + global) across CPU cores...\n",
            n_tiles, rows, cols);
    fflush(stderr);

    for (int t = 0; t < n_tiles; t++) {
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

    fprintf(stderr, "  • Vision Pipeline: Encoded %d tiles -> %d visual tokens (dim=%d)\n",
            n_tiles, n_tiles * SIGLIP_NUM_TOKENS, SIGLIP_CONN_OUT);
    fflush(stderr);
    if (out_rows) *out_rows = rows;
    if (out_cols) *out_cols = cols;
    return 0;
}
