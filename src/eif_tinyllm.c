/**
 * @file eif_tinyllm.c
 * @brief Tiny LLM Implementation
 *
 * Minimal transformer for microcontroller deployment.
 * Supports INT4/INT8 quantization for memory efficiency.
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE 1
#endif

#include "eif_tinyllm.h"
#include "eif_quantize_bitnet.h"
#include "eif_gguf.h"

#include <time.h>
#include <math.h>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>

// =============================================================================
// Memory Management
// =============================================================================

size_t tinyllm_memory_size(const tinyllm_config_t *config)
{
    size_t size = 0;

    int head_dim = (config->head_dim > 0) ? config->head_dim : (config->dim / config->n_heads);
    int q_dim = config->n_heads * head_dim;
    int kv_dim = config->n_kv_heads * head_dim;
    int max_dim = (q_dim > config->dim) ? q_dim : config->dim;

    // Activation buffers
    size += config->dim * sizeof(float);        // x
    size += max_dim * sizeof(float);            // xb [max(dim, q_dim)]
    size += config->dim * sizeof(float);        // xb2
    size += q_dim * sizeof(float);              // q [q_dim]
    size += kv_dim * sizeof(float);             // k (kv_dim)
    size += kv_dim * sizeof(float);             // v (kv_dim)
    size += config->hidden_dim * sizeof(float); // hb
    size += config->hidden_dim * sizeof(float); // hb2
    size += config->vocab_size * sizeof(float); // logits

    // KV cache (per layer: seq_len * kv_dim * 2 for k and v)
    size += config->n_layers * config->seq_len * kv_dim * 2 * sizeof(float);

    // Attention scores
    size += config->n_heads * config->seq_len * sizeof(float);

    return size;
}

/* BitNet matmul honouring the optional grouped scales (config.bitnet_group). Grouped scales are stored per matrix
 * kind as [n_layers, rows, in_dim / g]; a width with no admissible group falls back to g = in_dim (one per row). */
static void tl_bitnet_mm(const tinyllm_t *llm, int mat, int l, float *out, const uint8_t *w_layer, const float *in,
                         int in_dim, int out_dim, const float *row_scales)
{
    const float *gs = llm->weights.bitnet_gscale[mat];
    if (llm->config.bitnet_group > 0 && gs) {
        int g = eif_bitnet_group_size(in_dim, llm->config.bitnet_group);
        if (g <= 0) g = in_dim;
        eif_matmul_bitnet_block_f32(w_layer, gs + (size_t)l * out_dim * (in_dim / g), g, in, NULL, out, out_dim,
                                    in_dim);
        return;
    }
    tinyllm_matmul_bitnet_158(out, w_layer, in, in_dim, out_dim, row_scales);
}

/* Rotate a BitNet input in place when the weights are Hadamard-rotated (config.bitnet_hadamard). */
static void tl_bitnet_rotate(const tinyllm_t *llm, float *x, const float *signs, int n)
{
    if (llm->config.bitnet_hadamard && signs) eif_hadamard_rotate_f32(x, signs, n, eif_hadamard_block_size(n));
}

int tinyllm_init(tinyllm_t *llm, const tinyllm_config_t *config, void *memory_pool,
                 size_t pool_size)
{
    memset(llm, 0, sizeof(tinyllm_t));
    memcpy(&llm->config, config, sizeof(tinyllm_config_t));

    size_t required = tinyllm_memory_size(config);
    if (pool_size < required) {
        return -1;
    }

    llm->memory_pool = memory_pool;
    llm->memory_size = pool_size;

    int head_dim = (config->head_dim > 0) ? config->head_dim : (config->dim / config->n_heads);
    int q_dim = config->n_heads * head_dim;
    int kv_dim = config->n_kv_heads * head_dim;
    int max_dim = (q_dim > config->dim) ? q_dim : config->dim;

    float *ptr = (float *)memory_pool;

    llm->state.x = ptr;
    ptr += config->dim;
    llm->state.xb = ptr;
    ptr += max_dim;
    llm->state.xb2 = ptr;
    ptr += config->dim;
    llm->state.q = ptr;
    ptr += q_dim;
    llm->state.k = ptr;
    ptr += kv_dim;
    llm->state.v = ptr;
    ptr += kv_dim;
    llm->state.hb = ptr;
    ptr += config->hidden_dim;
    llm->state.hb2 = ptr;
    ptr += config->hidden_dim;
    llm->state.logits = ptr;
    ptr += config->vocab_size;

    llm->state.att = ptr;
    ptr += config->n_heads * config->seq_len;

    llm->state.key_cache = ptr;
    ptr += config->n_layers * config->seq_len * kv_dim;
    llm->state.value_cache = ptr;
    ptr += config->n_layers * config->seq_len * kv_dim;

    if (config->kv_type == 1) {
        llm->state.key_cache_i8 = (int8_t *)llm->state.key_cache;
        llm->state.value_cache_i8 = (int8_t *)llm->state.value_cache;
        size_t scale_elems = (size_t)config->n_layers * config->seq_len * config->n_kv_heads;
        llm->state.key_scale = (float *)malloc(scale_elems * sizeof(float));
        llm->state.value_scale = (float *)malloc(scale_elems * sizeof(float));
        if (llm->state.key_scale) memset(llm->state.key_scale, 0, scale_elems * sizeof(float));
        if (llm->state.value_scale) memset(llm->state.value_scale, 0, scale_elems * sizeof(float));
    } else {
        llm->state.key_cache_i8 = NULL;
        llm->state.value_cache_i8 = NULL;
        llm->state.key_scale = NULL;
        llm->state.value_scale = NULL;
    }

    return 0;
}

void tinyllm_free(tinyllm_t *llm)
{
    if (llm->state.key_scale) {
        free(llm->state.key_scale);
        llm->state.key_scale = NULL;
    }
    if (llm->state.value_scale) {
        free(llm->state.value_scale);
        llm->state.value_scale = NULL;
    }
    // Memory pool is externally managed
    memset(llm, 0, sizeof(tinyllm_t));
}

void tinyllm_reset(tinyllm_t *llm)
{
    llm->state.pos = 0;

    int head_dim = (llm->config.head_dim > 0) ? llm->config.head_dim : (llm->config.dim / llm->config.n_heads);
    int kv_dim = llm->config.n_kv_heads * head_dim;

    memset(llm->state.key_cache, 0,
           llm->config.n_layers * llm->config.seq_len * kv_dim * sizeof(float));
    memset(llm->state.value_cache, 0,
           llm->config.n_layers * llm->config.seq_len * kv_dim * sizeof(float));

    if (llm->state.key_scale) {
        size_t scale_elems = (size_t)llm->config.n_layers * llm->config.seq_len * llm->config.n_kv_heads;
        memset(llm->state.key_scale, 0, scale_elems * sizeof(float));
    }
    if (llm->state.value_scale) {
        size_t scale_elems = (size_t)llm->config.n_layers * llm->config.seq_len * llm->config.n_kv_heads;
        memset(llm->state.value_scale, 0, scale_elems * sizeof(float));
    }
}

// =============================================================================
// Core Operations
// =============================================================================

#if defined(__AVX512F__)
#include <immintrin.h>
static void rmsnorm(float *out, const float *x, const float *weight, int size, float eps)
{
    __m512 ss_vec = _mm512_setzero_ps();
    int i = 0;
    for (; i <= size - 16; i += 16) {
        __m512 vx = _mm512_loadu_ps(&x[i]);
        ss_vec = _mm512_fmadd_ps(vx, vx, ss_vec);
    }
    float ss = _mm512_reduce_add_ps(ss_vec);
    for (; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / size + eps);

    __m512 vss = _mm512_set1_ps(ss);
    i = 0;
    if (weight) {
        for (; i <= size - 16; i += 16) {
            __m512 vx = _mm512_loadu_ps(&x[i]);
            __m512 vw = _mm512_loadu_ps(&weight[i]);
            _mm512_storeu_ps(&out[i], _mm512_mul_ps(_mm512_mul_ps(vx, vss), vw));
        }
        for (; i < size; i++) out[i] = x[i] * ss * weight[i];
    } else {
        for (; i <= size - 16; i += 16) {
            __m512 vx = _mm512_loadu_ps(&x[i]);
            _mm512_storeu_ps(&out[i], _mm512_mul_ps(vx, vss));
        }
        for (; i < size; i++) out[i] = x[i] * ss;
    }
}
#elif defined(__AVX2__)
#include <immintrin.h>
static void rmsnorm(float *out, const float *x, const float *weight, int size, float eps)
{
    __m256 ss_vec = _mm256_setzero_ps();
    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(&x[i]);
        ss_vec = _mm256_fmadd_ps(vx, vx, ss_vec);
    }
    __m128 low = _mm256_castps256_ps128(ss_vec);
    __m128 high = _mm256_extractf128_ps(ss_vec, 1);
    __m128 sum128 = _mm_add_ps(low, high);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float ss = _mm_cvtss_f32(sum128);
    for (; i < size; i++) {
        ss += x[i] * x[i];
    }
    ss = ss / size + eps;
    ss = 1.0f / sqrtf(ss);

    __m256 vss = _mm256_set1_ps(ss);
    i = 0;
    if (weight) {
        for (; i <= size - 8; i += 8) {
            __m256 vx = _mm256_loadu_ps(&x[i]);
            __m256 vw = _mm256_loadu_ps(&weight[i]);
            _mm256_storeu_ps(&out[i], _mm256_mul_ps(_mm256_mul_ps(vx, vss), vw));
        }
        for (; i < size; i++) {
            out[i] = x[i] * ss * weight[i];
        }
    } else {
        for (; i <= size - 8; i += 8) {
            __m256 vx = _mm256_loadu_ps(&x[i]);
            _mm256_storeu_ps(&out[i], _mm256_mul_ps(vx, vss));
        }
        for (; i < size; i++) {
            out[i] = x[i] * ss;
        }
    }
}
#elif defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
static void rmsnorm(float *out, const float *x, const float *weight, int size, float eps)
{
    float32x4_t ss_vec0 = vdupq_n_f32(0.0f);
    float32x4_t ss_vec1 = vdupq_n_f32(0.0f);
    float32x4_t ss_vec2 = vdupq_n_f32(0.0f);
    float32x4_t ss_vec3 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i <= size - 16; i += 16) {
        float32x4_t vx0 = vld1q_f32(&x[i]);
        float32x4_t vx1 = vld1q_f32(&x[i + 4]);
        float32x4_t vx2 = vld1q_f32(&x[i + 8]);
        float32x4_t vx3 = vld1q_f32(&x[i + 12]);
        ss_vec0 = vfmaq_f32(ss_vec0, vx0, vx0);
        ss_vec1 = vfmaq_f32(ss_vec1, vx1, vx1);
        ss_vec2 = vfmaq_f32(ss_vec2, vx2, vx2);
        ss_vec3 = vfmaq_f32(ss_vec3, vx3, vx3);
    }
    for (; i <= size - 4; i += 4) {
        float32x4_t vx = vld1q_f32(&x[i]);
        ss_vec0 = vfmaq_f32(ss_vec0, vx, vx);
    }
    ss_vec0 = vaddq_f32(vaddq_f32(ss_vec0, ss_vec1), vaddq_f32(ss_vec2, ss_vec3));
    float ss = vaddvq_f32(ss_vec0);
    for (; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / size + eps);

    float32x4_t vss = vdupq_n_f32(ss);
    i = 0;
    if (weight) {
        for (; i <= size - 8; i += 8) {
            float32x4_t vx0 = vld1q_f32(&x[i]);
            float32x4_t vx1 = vld1q_f32(&x[i + 4]);
            float32x4_t vw0 = vld1q_f32(&weight[i]);
            float32x4_t vw1 = vld1q_f32(&weight[i + 4]);
            vst1q_f32(&out[i], vmulq_f32(vmulq_f32(vx0, vss), vw0));
            vst1q_f32(&out[i + 4], vmulq_f32(vmulq_f32(vx1, vss), vw1));
        }
        for (; i < size; i++) out[i] = x[i] * ss * weight[i];
    } else {
        for (; i <= size - 8; i += 8) {
            float32x4_t vx0 = vld1q_f32(&x[i]);
            float32x4_t vx1 = vld1q_f32(&x[i + 4]);
            vst1q_f32(&out[i], vmulq_f32(vx0, vss));
            vst1q_f32(&out[i + 4], vmulq_f32(vx1, vss));
        }
        for (; i < size; i++) out[i] = x[i] * ss;
    }
}
#else
static void rmsnorm(float *out, const float *x, const float *weight, int size, float eps)
{
    float ss = 0.0f;
    for (int i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }
    ss = ss / size + eps;
    ss = 1.0f / sqrtf(ss);

    for (int i = 0; i < size; i++) {
        out[i] = x[i] * ss * (weight ? weight[i] : 1.0f);
    }
}
#endif

static void softmax(float *x, int size)
{
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val)
            max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}

static void matmul(float *xout, const float *x, const float *w, int n, int d)
{
    // W (d,n) @ x (n,) -> xout (d,)
    // W is stored row-major as (d, n)
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

// =============================================================================
// Forward Pass
// =============================================================================

float *tinyllm_forward_embedding_ex(tinyllm_t *llm, const float *embedding, int pos, bool compute_logits)
{
    tinyllm_config_t *p = &llm->config;
    tinyllm_weights_t *w = &llm->weights;
    tinyllm_state_t *s = &llm->state;

    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_dim = (p->head_dim > 0) ? p->head_dim : (dim / p->n_heads);
    int q_dim = p->n_heads * head_dim;
    int kv_dim = p->n_kv_heads * head_dim;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int attn_scales_per_layer = (p->head_dim > 0 && p->head_dim != dim / p->n_heads) ?
                                (q_dim + kv_dim * 2 + dim) :
                                (dim * 4 + kv_dim * 2);

    if (pos < 0 || pos >= p->seq_len) {
        return NULL; /* the KV cache has seq_len slots per layer; writing further corrupts the next layer */
    }

    if (embedding != s->x) {
        memcpy(s->x, embedding, dim * sizeof(float));
    }

    /* Precompute RoPE cos/sin for this position (once for all layers) */
    int half_head = head_dim / 2;
    float rope_cos[128];
    float rope_sin[128];
    float rope_theta = (p->rope_theta > 0.0f) ? p->rope_theta : ((p->vocab_size > 120000) ? 5000000.0f : ((p->vocab_size > 60000) ? 100000.0f : 10000.0f));
    for (int j = 0; j < half_head && j < 128; j++) {
        float freq = 1.0f / powf(rope_theta, (float)(2 * j) / (float)head_dim);
        float val = (float)pos * freq;
        rope_cos[j] = cosf(val);
        rope_sin[j] = sinf(val);
    }
    float inv_sqrt_head_dim = 1.0f / sqrtf((float)head_dim);

    // Process each layer
    for (int l = 0; l < p->n_layers; l++) {
        float rms_eps = (p->vocab_size > 120000) ? 1e-6f : 1e-5f;
        // Attention RMSNorm
        rmsnorm(s->xb, s->x, w->rms_att_weight + l * dim, dim, rms_eps);

        // QKV matmuls
        int loff = l * p->seq_len * kv_dim;

        float *k = s->key_cache + loff + pos * kv_dim;
        float *v = s->value_cache + loff + pos * kv_dim;

        int wk_offset = (p->head_dim > 0 && p->head_dim != dim / p->n_heads) ? q_dim : dim;
        int wv_offset = (p->head_dim > 0 && p->head_dim != dim / p->n_heads) ? (q_dim + kv_dim) : (dim * 2);
        int wo_offset = (p->head_dim > 0 && p->head_dim != dim / p->n_heads) ? (q_dim + kv_dim * 2) : (dim * 2 + kv_dim);

        // matmul(xout, x, w, n, d): W(d,n) @ x(n,) -> xout(d,)
        if (p->qtype == TINYLLM_QTYPE_INT8) {
            const int8_t *w_wq = (const int8_t *)w->wq;
            const int8_t *w_wk = (const int8_t *)w->wk;
            const int8_t *w_wv = (const int8_t *)w->wv;
            tinyllm_matmul_int8_rowscale(s->q, w_wq + l * q_dim * dim, s->xb, dim, q_dim,
                                         w->attn_scale + l * attn_scales_per_layer);
            tinyllm_matmul_int8_rowscale(k, w_wk + l * kv_dim * dim, s->xb, dim, kv_dim,
                                         w->attn_scale + l * attn_scales_per_layer + wk_offset);
            tinyllm_matmul_int8_rowscale(v, w_wv + l * kv_dim * dim, s->xb, dim, kv_dim,
                                         w->attn_scale + l * attn_scales_per_layer + wv_offset);
        } else if (p->qtype == TINYLLM_QTYPE_INT4) {
            uint8_t *w_wq = (uint8_t *)w->wq;
            uint8_t *w_wk = (uint8_t *)w->wk;
            uint8_t *w_wv = (uint8_t *)w->wv;
            int dim_packed = (dim + 1) / 2;
            tinyllm_matmul_int4(s->q, w_wq + l * q_dim * dim_packed, s->xb, dim, q_dim,
                                w->attn_scale + l * attn_scales_per_layer);
            tinyllm_matmul_int4(k, w_wk + l * kv_dim * dim_packed, s->xb, dim, kv_dim,
                                w->attn_scale + l * attn_scales_per_layer + wk_offset);
            tinyllm_matmul_int4(v, w_wv + l * kv_dim * dim_packed, s->xb, dim, kv_dim,
                                w->attn_scale + l * attn_scales_per_layer + wv_offset);
        } else if (p->qtype == TINYLLM_QTYPE_BITNET_158) {
            uint8_t *w_wq = (uint8_t *)w->wq;
            uint8_t *w_wk = (uint8_t *)w->wk;
            uint8_t *w_wv = (uint8_t *)w->wv;
            int dim_packed = (dim + 3) / 4;
            tl_bitnet_rotate(llm, s->xb, w->had_signs_dim, dim);
            tl_bitnet_mm(llm, TL_BN_WQ, l, s->q, w_wq + l * q_dim * dim_packed, s->xb, dim, q_dim,
                                     w->attn_scale + l * attn_scales_per_layer);
            tl_bitnet_mm(llm, TL_BN_WK, l, k, w_wk + l * kv_dim * dim_packed, s->xb, dim, kv_dim,
                                     w->attn_scale + l * attn_scales_per_layer + wk_offset);
            tl_bitnet_mm(llm, TL_BN_WV, l, v, w_wv + l * kv_dim * dim_packed, s->xb, dim, kv_dim,
                                     w->attn_scale + l * attn_scales_per_layer + wv_offset);
        } else {
            float *w_wq = (float *)w->wq;
            float *w_wk = (float *)w->wk;
            float *w_wv = (float *)w->wv;
            matmul(s->q, s->xb, w_wq + l * q_dim * dim, dim, q_dim);
            matmul(k, s->xb, w_wk + l * kv_dim * dim, dim, kv_dim);
            matmul(v, s->xb, w_wv + l * kv_dim * dim, dim, kv_dim);
        }

        // Add QKV biases if present (Qwen / Qwen2 architecture)
        if (w->bq) {
            const float *bq = w->bq + (size_t)l * q_dim;
            for (int i = 0; i < q_dim; i++) s->q[i] += bq[i];
        }
        if (w->bk) {
            const float *bk = w->bk + (size_t)l * kv_dim;
            for (int i = 0; i < kv_dim; i++) k[i] += bk[i];
        }
        if (w->bv) {
            const float *bv = w->bv + (size_t)l * kv_dim;
            for (int i = 0; i < kv_dim; i++) v[i] += bv[i];
        }

        // RoPE positional encoding (HuggingFace LLaMA rotate_half standard)
        for (int h = 0; h < p->n_heads; h++) {
            float *qh = s->q + h * head_dim;
            for (int j = 0; j < half_head; j++) {
                float q0 = qh[j];
                float q1 = qh[j + half_head];
                float c = rope_cos[j];
                float s_val = rope_sin[j];
                qh[j]             = q0 * c - q1 * s_val;
                qh[j + half_head] = q1 * c + q0 * s_val;
            }
        }
        for (int h = 0; h < p->n_kv_heads; h++) {
            float *kh = k + h * head_dim;
            for (int j = 0; j < half_head; j++) {
                float k0 = kh[j];
                float k1 = kh[j + half_head];
                float c = rope_cos[j];
                float s_val = rope_sin[j];
                kh[j]             = k0 * c - k1 * s_val;
                kh[j + half_head] = k1 * c + k0 * s_val;
            }
        }

        // Quantize key and value to INT8 if enabled (75% memory bandwidth reduction)
        if (p->kv_type == 1 && s->key_cache_i8) {
            int loff_scale = l * p->seq_len * p->n_kv_heads;
            for (int h_kv = 0; h_kv < p->n_kv_heads; h_kv++) {
                float *kh = k + h_kv * head_dim;
                int8_t *kh_i8 = s->key_cache_i8 + loff + pos * kv_dim + h_kv * head_dim;
                float amax_k = 0.0f;
                for (int j = 0; j < head_dim; j++) {
                    float v_abs = fabsf(kh[j]);
                    if (v_abs > amax_k) amax_k = v_abs;
                }
                float scale_k = (amax_k < 1e-12f) ? 1.0f : (amax_k / 127.0f);
                float inv_scale_k = 127.0f / (amax_k < 1e-12f ? 1.0f : amax_k);
                s->key_scale[loff_scale + pos * p->n_kv_heads + h_kv] = scale_k;
                for (int j = 0; j < head_dim; j++) {
                    long qv = (long)(kh[j] * inv_scale_k + (kh[j] >= 0.0f ? 0.5f : -0.5f));
                    if (qv > 127) {
                        qv = 127;
                    } else if (qv < -128) {
                        qv = -128;
                    }
                    kh_i8[j] = (int8_t)qv;
                }

                float *vh = v + h_kv * head_dim;
                int8_t *vh_i8 = s->value_cache_i8 + loff + pos * kv_dim + h_kv * head_dim;
                float amax_v = 0.0f;
                for (int j = 0; j < head_dim; j++) {
                    float v_abs = fabsf(vh[j]);
                    if (v_abs > amax_v) amax_v = v_abs;
                }
                float scale_v = (amax_v < 1e-12f) ? 1.0f : (amax_v / 127.0f);
                float inv_scale_v = 127.0f / (amax_v < 1e-12f ? 1.0f : amax_v);
                s->value_scale[loff_scale + pos * p->n_kv_heads + h_kv] = scale_v;
                for (int j = 0; j < head_dim; j++) {
                    long qv = (long)(vh[j] * inv_scale_v + (vh[j] >= 0.0f ? 0.5f : -0.5f));
                    if (qv > 127) {
                        qv = 127;
                    } else if (qv < -128) {
                        qv = -128;
                    }
                    vh_i8[j] = (int8_t)qv;
                }
            }
        }

        // Multi-head attention with AVX2 vectorization. Heads are independent (own att / xb slices), so they
        // run in parallel once the context is long enough for the KV scan to dominate a decode step.
        #pragma omp parallel for schedule(static) if(!omp_in_parallel() && pos >= 64)
        for (int h = 0; h < p->n_heads; h++) {
            float *q = s->q + h * head_dim;
            float *att = s->att + h * p->seq_len;

            for (int t = 0; t <= pos; t++) {
                if (p->kv_type == 1 && s->key_cache_i8) {
                    int8_t *k_t_i8 = s->key_cache_i8 + loff + t * kv_dim + (h / kv_mul) * head_dim;
                    int loff_scale = l * p->seq_len * p->n_kv_heads;
                    float k_scale = s->key_scale[loff_scale + t * p->n_kv_heads + (h / kv_mul)];
                    float score = 0.0f;
                    for (int i = 0; i < head_dim; i++) {
                        score += q[i] * (float)k_t_i8[i];
                    }
                    att[t] = (score * k_scale) * inv_sqrt_head_dim;
                } else {
                    float *k_t = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
#if defined(__AVX512F__)
                    __m512 acc512 = _mm512_setzero_ps();
                    for (int i = 0; i < head_dim; i += 16) {
                        __m512 vq = _mm512_loadu_ps(&q[i]);
                        __m512 vk = _mm512_loadu_ps(&k_t[i]);
                        acc512 = _mm512_fmadd_ps(vq, vk, acc512);
                    }
                    att[t] = _mm512_reduce_add_ps(acc512) * inv_sqrt_head_dim;
#elif defined(__AVX2__)
                    __m256 acc = _mm256_setzero_ps();
                    for (int i = 0; i < head_dim; i += 8) {
                        __m256 vq = _mm256_loadu_ps(&q[i]);
                        __m256 vk = _mm256_loadu_ps(&k_t[i]);
                        acc = _mm256_fmadd_ps(vq, vk, acc);
                    }
                    __m128 l_v = _mm256_castps256_ps128(acc);
                    __m128 hi_v = _mm256_extractf128_ps(acc, 1);
                    __m128 sm_v = _mm_add_ps(l_v, hi_v);
                    sm_v = _mm_hadd_ps(sm_v, sm_v);
                    sm_v = _mm_hadd_ps(sm_v, sm_v);
                    att[t] = _mm_cvtss_f32(sm_v) * inv_sqrt_head_dim;
#elif defined(__ARM_NEON) || defined(__aarch64__)
                    float32x4_t acc0 = vdupq_n_f32(0.0f);
                    float32x4_t acc1 = vdupq_n_f32(0.0f);
                    for (int i = 0; i < head_dim; i += 8) {
                        acc0 = vfmaq_f32(acc0, vld1q_f32(&q[i]), vld1q_f32(&k_t[i]));
                        acc1 = vfmaq_f32(acc1, vld1q_f32(&q[i + 4]), vld1q_f32(&k_t[i + 4]));
                    }
                    att[t] = vaddvq_f32(vaddq_f32(acc0, acc1)) * inv_sqrt_head_dim;
#else
                    float score = 0.0f;
                    for (int i = 0; i < head_dim; i++) {
                        score += q[i] * k_t[i];
                    }
                    att[t] = score * inv_sqrt_head_dim;
#endif
                }
            }

            softmax(att, pos + 1);

            float *xb = s->xb + h * head_dim;
            memset(xb, 0, head_dim * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                if (p->kv_type == 1 && s->value_cache_i8) {
                    int8_t *v_t_i8 = s->value_cache_i8 + loff + t * kv_dim + (h / kv_mul) * head_dim;
                    int loff_scale = l * p->seq_len * p->n_kv_heads;
                    float v_scale = s->value_scale[loff_scale + t * p->n_kv_heads + (h / kv_mul)];
                    float a = att[t] * v_scale;
                    for (int i = 0; i < head_dim; i++) {
                        xb[i] += a * (float)v_t_i8[i];
                    }
                } else {
                    float *v_t = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
                    float a = att[t];
#if defined(__AVX512F__)
                    __m512 va512 = _mm512_set1_ps(a);
                    for (int i = 0; i < head_dim; i += 16) {
                        __m512 vxb = _mm512_loadu_ps(&xb[i]);
                        __m512 vv = _mm512_loadu_ps(&v_t[i]);
                        _mm512_storeu_ps(&xb[i], _mm512_fmadd_ps(va512, vv, vxb));
                    }
#elif defined(__AVX2__)
                    __m256 va = _mm256_set1_ps(a);
                    for (int i = 0; i < head_dim; i += 8) {
                        __m256 vxb = _mm256_loadu_ps(&xb[i]);
                        __m256 vv = _mm256_loadu_ps(&v_t[i]);
                        _mm256_storeu_ps(&xb[i], _mm256_fmadd_ps(va, vv, vxb));
                    }
#elif defined(__ARM_NEON) || defined(__aarch64__)
                    float32x4_t va = vdupq_n_f32(a);
                    for (int i = 0; i < head_dim; i += 8) {
                        vst1q_f32(&xb[i], vfmaq_f32(vld1q_f32(&xb[i]), va, vld1q_f32(&v_t[i])));
                        vst1q_f32(&xb[i + 4], vfmaq_f32(vld1q_f32(&xb[i + 4]), va, vld1q_f32(&v_t[i + 4])));
                    }
#else
                    for (int i = 0; i < head_dim; i++) {
                        xb[i] += a * v_t[i];
                    }
#endif
                }
            }
        }

        // Output projection
        if (p->qtype == TINYLLM_QTYPE_INT8) {
            const int8_t *w_wo = (const int8_t *)w->wo;
            tinyllm_matmul_int8_rowscale(s->xb2, w_wo + l * dim * q_dim, s->xb, q_dim, dim,
                                         w->attn_scale + l * attn_scales_per_layer + wo_offset);
        } else if (p->qtype == TINYLLM_QTYPE_INT4) {
            uint8_t *w_wo = (uint8_t *)w->wo;
            int q_dim_packed = (q_dim + 1) / 2;
            tinyllm_matmul_int4(s->xb2, w_wo + l * dim * q_dim_packed, s->xb, q_dim, dim,
                                w->attn_scale + l * attn_scales_per_layer + wo_offset);
        } else if (p->qtype == TINYLLM_QTYPE_BITNET_158) {
            float *sub_w = w->rms_sub_att_weight ? (w->rms_sub_att_weight + l * q_dim) : NULL;
            rmsnorm(s->xb, s->xb, sub_w, q_dim, 1e-5f);
            uint8_t *w_wo = (uint8_t *)w->wo;
            int q_dim_packed = (q_dim + 3) / 4;
            tl_bitnet_rotate(llm, s->xb, w->had_signs_q, q_dim);
            tl_bitnet_mm(llm, TL_BN_WO, l, s->xb2, w_wo + l * dim * q_dim_packed, s->xb, q_dim, dim,
                                     w->attn_scale + l * attn_scales_per_layer + wo_offset);
        } else {
            float *w_wo = (float *)w->wo;
            matmul(s->xb2, s->xb, w_wo + l * dim * q_dim, q_dim, dim);
        }

        // Residual with AVX-512 / AVX2 / NEON vectorization
#if defined(__AVX512F__)
        for (int i = 0; i <= dim - 16; i += 16) {
            __m512 vx = _mm512_loadu_ps(&s->x[i]);
            __m512 vxb2 = _mm512_loadu_ps(&s->xb2[i]);
            _mm512_storeu_ps(&s->x[i], _mm512_add_ps(vx, vxb2));
        }
#elif defined(__AVX2__)
        for (int i = 0; i <= dim - 8; i += 8) {
            __m256 vx = _mm256_loadu_ps(&s->x[i]);
            __m256 vxb2 = _mm256_loadu_ps(&s->xb2[i]);
            _mm256_storeu_ps(&s->x[i], _mm256_add_ps(vx, vxb2));
        }
#elif defined(__ARM_NEON) || defined(__aarch64__)
        for (int i = 0; i <= dim - 8; i += 8) {
            vst1q_f32(&s->x[i], vaddq_f32(vld1q_f32(&s->x[i]), vld1q_f32(&s->xb2[i])));
            vst1q_f32(&s->x[i + 4], vaddq_f32(vld1q_f32(&s->x[i + 4]), vld1q_f32(&s->xb2[i + 4])));
        }
#else
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb2[i];
        }
#endif

        // FFN RMSNorm
        rmsnorm(s->xb, s->x, w->rms_ffn_weight + l * dim, dim, rms_eps);

        // FFN: SwiGLU
        if (p->qtype == TINYLLM_QTYPE_INT8) {
            const int8_t *w_w1 = (const int8_t *)w->w1;
            const int8_t *w_w3 = (const int8_t *)w->w3;
            tinyllm_matmul_int8_rowscale(s->hb, w_w1 + l * hidden_dim * dim, s->xb, dim, hidden_dim,
                                         w->ffn_scale + l * (hidden_dim * 2 + dim));
            tinyllm_matmul_int8_rowscale(s->hb2, w_w3 + l * hidden_dim * dim, s->xb, dim, hidden_dim,
                                         w->ffn_scale + l * (hidden_dim * 2 + dim) + hidden_dim);
        } else if (p->qtype == TINYLLM_QTYPE_INT4) {
            uint8_t *w_w1 = (uint8_t *)w->w1;
            uint8_t *w_w3 = (uint8_t *)w->w3;
            int dim_packed = (dim + 1) / 2;
            tinyllm_matmul_int4(s->hb, w_w1 + l * hidden_dim * dim_packed, s->xb, dim, hidden_dim,
                                w->ffn_scale + l * (hidden_dim * 2 + dim));
            tinyllm_matmul_int4(s->hb2, w_w3 + l * hidden_dim * dim_packed, s->xb, dim, hidden_dim,
                                w->ffn_scale + l * (hidden_dim * 2 + dim) + hidden_dim);
        } else if (p->qtype == TINYLLM_QTYPE_BITNET_158) {
            uint8_t *w_w1 = (uint8_t *)w->w1;
            uint8_t *w_w3 = (uint8_t *)w->w3;
            int dim_packed = (dim + 3) / 4;
            tl_bitnet_rotate(llm, s->xb, w->had_signs_dim, dim);
            tl_bitnet_mm(llm, TL_BN_W1, l, s->hb, w_w1 + l * hidden_dim * dim_packed, s->xb, dim, hidden_dim,
                                     w->ffn_scale + l * (hidden_dim * 2 + dim));
            tl_bitnet_mm(llm, TL_BN_W3, l, s->hb2, w_w3 + l * hidden_dim * dim_packed, s->xb, dim, hidden_dim,
                                     w->ffn_scale + l * (hidden_dim * 2 + dim) + hidden_dim);
        } else {
            float *w_w1 = (float *)w->w1;
            float *w_w3 = (float *)w->w3;
            matmul(s->hb, s->xb, w_w1 + l * dim * hidden_dim, dim, hidden_dim);
            matmul(s->hb2, s->xb, w_w3 + l * dim * hidden_dim, dim, hidden_dim);
        }

#if defined(__AVX2__)
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 two = _mm256_set1_ps(2.0f);
        __m256 log2ef = _mm256_set1_ps(1.44269504088896341f);
        __m256 c1 = _mm256_set1_ps(0.6931471805599453f);
        __m256 poly_p = _mm256_set1_ps(0.16666666666666666f);
        __m256 poly_half = _mm256_set1_ps(0.5f);
        __m256 min_x = _mm256_set1_ps(-88.3762626647949f);
        __m256 max_x = _mm256_set1_ps(88.3762626647949f);
        __m256i bias127 = _mm256_set1_epi32(127);

        for (int i = 0; i <= hidden_dim - 8; i += 8) {
            __m256 val = _mm256_loadu_ps(&s->hb[i]);
            __m256 val2 = _mm256_loadu_ps(&s->hb2[i]);
            __m256 neg_val = _mm256_sub_ps(_mm256_setzero_ps(), val);

            __m256 z = _mm256_max_ps(neg_val, min_x);
            z = _mm256_min_ps(z, max_x);
            __m256 fx = _mm256_fmadd_ps(z, log2ef, poly_half);
            __m256i emm0 = _mm256_cvttps_epi32(_mm256_floor_ps(fx));
            __m256 n = _mm256_cvtepi32_ps(emm0);
            __m256 x_red = _mm256_fnmadd_ps(n, c1, z);
            __m256 p = _mm256_fmadd_ps(poly_p, x_red, poly_half);
            p = _mm256_fmadd_ps(p, x_red, one);
            p = _mm256_fmadd_ps(p, x_red, one);
            __m256i pow2n = _mm256_slli_epi32(_mm256_add_epi32(emm0, bias127), 23);
            __m256 exp_neg = _mm256_mul_ps(p, _mm256_castsi256_ps(pow2n));

            __m256 denom = _mm256_add_ps(one, exp_neg);
            __m256 r0 = _mm256_rcp_ps(denom);
            __m256 sig = _mm256_mul_ps(r0, _mm256_fnmadd_ps(denom, r0, two));

            __m256 res = _mm256_mul_ps(_mm256_mul_ps(val, sig), val2);
            _mm256_storeu_ps(&s->hb[i], res);
        }
#else
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }
#endif

        if (p->qtype == TINYLLM_QTYPE_INT8) {
            const int8_t *w_w2 = (const int8_t *)w->w2;
            tinyllm_matmul_int8_rowscale(s->xb, w_w2 + l * dim * hidden_dim, s->hb, hidden_dim, dim,
                                         w->ffn_scale + l * (hidden_dim * 2 + dim) + hidden_dim * 2);
        } else if (p->qtype == TINYLLM_QTYPE_INT4) {
            uint8_t *w_w2 = (uint8_t *)w->w2;
            int hidden_dim_packed = (hidden_dim + 1) / 2;
            tinyllm_matmul_int4(s->xb, w_w2 + l * dim * hidden_dim_packed, s->hb, hidden_dim, dim,
                                w->ffn_scale + l * (hidden_dim * 2 + dim) + hidden_dim * 2);
        } else if (p->qtype == TINYLLM_QTYPE_BITNET_158) {
            float *sub_w = w->rms_sub_ffn_weight ? (w->rms_sub_ffn_weight + l * hidden_dim) : NULL;
            rmsnorm(s->hb, s->hb, sub_w, hidden_dim, 1e-5f);
            uint8_t *w_w2 = (uint8_t *)w->w2;
            int hidden_dim_packed = (hidden_dim + 3) / 4;
            tl_bitnet_rotate(llm, s->hb, w->had_signs_hidden, hidden_dim);
            tl_bitnet_mm(llm, TL_BN_W2, l, s->xb, w_w2 + l * dim * hidden_dim_packed, s->hb, hidden_dim, dim,
                                     w->ffn_scale + l * (hidden_dim * 2 + dim) + hidden_dim * 2);
        } else {
            float *w_w2 = (float *)w->w2;
            matmul(s->xb, s->hb, w_w2 + l * hidden_dim * dim, hidden_dim, dim);
        }

        // Residual with AVX-512 / AVX2 / NEON vectorization
#if defined(__AVX512F__)
        for (int i = 0; i <= dim - 16; i += 16) {
            __m512 vx = _mm512_loadu_ps(&s->x[i]);
            __m512 vxb = _mm512_loadu_ps(&s->xb[i]);
            _mm512_storeu_ps(&s->x[i], _mm512_add_ps(vx, vxb));
        }
#elif defined(__AVX2__)
        for (int i = 0; i <= dim - 8; i += 8) {
            __m256 vx = _mm256_loadu_ps(&s->x[i]);
            __m256 vxb = _mm256_loadu_ps(&s->xb[i]);
            _mm256_storeu_ps(&s->x[i], _mm256_add_ps(vx, vxb));
        }
#elif defined(__ARM_NEON) || defined(__aarch64__)
        for (int i = 0; i <= dim - 8; i += 8) {
            vst1q_f32(&s->x[i], vaddq_f32(vld1q_f32(&s->x[i]), vld1q_f32(&s->xb[i])));
            vst1q_f32(&s->x[i + 4], vaddq_f32(vld1q_f32(&s->x[i + 4]), vld1q_f32(&s->xb[i + 4])));
        }
#else
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb[i];
        }
#endif
    }

    // Final RMSNorm
    float rms_eps_final = (p->vocab_size > 120000) ? 1e-6f : 1e-5f;
    rmsnorm(s->x, s->x, w->rms_final_weight, dim, rms_eps_final);

    if (!compute_logits) {
        return s->x;
    }

    // Classifier (weight tying with token embeddings)
    if (p->qtype == TINYLLM_QTYPE_INT8) {
        const int8_t *w_wcls = (const int8_t *)w->wcls;
        tinyllm_matmul_int8_rowscale(s->logits, w_wcls, s->x, dim, p->vocab_size, w->wcls_scale);
    } else if (p->qtype == TINYLLM_QTYPE_INT4) {
        uint8_t *w_wcls = (uint8_t *)w->wcls;
        tinyllm_matmul_int4(s->logits, w_wcls, s->x, dim, p->vocab_size, w->wcls_scale);
    } else if (p->qtype == TINYLLM_QTYPE_BITNET_158) {
        const int8_t *w_wcls = (const int8_t *)w->wcls;
        tinyllm_matmul_int8_rowscale(s->logits, w_wcls, s->x, dim, p->vocab_size, w->wcls_scale);
    } else {
        float *w_wcls = (float *)w->wcls;
        float *w_token_embedding = (float *)w->token_embedding;
        float *wcls = w_wcls ? w_wcls : w_token_embedding;
        matmul(s->logits, s->x, wcls, dim, p->vocab_size);
    }

    return s->logits;
}

float *tinyllm_forward_embedding(tinyllm_t *llm, const float *embedding, int pos)
{
    return tinyllm_forward_embedding_ex(llm, embedding, pos, true);
}

float *tinyllm_get_hidden_state(tinyllm_t *llm)
{
    return llm ? llm->state.x : NULL;
}

void tinyllm_get_token_embedding(tinyllm_t *llm, tinyllm_token_t token, float *out_emb)
{
    if (!llm || !out_emb) return;
    tinyllm_config_t *p = &llm->config;
    tinyllm_weights_t *w = &llm->weights;
    int dim = p->dim;

    if (p->qtype == TINYLLM_QTYPE_INT8) {
        const int8_t *w_token_embedding = (const int8_t *)w->token_embedding;
        const int8_t *row = w_token_embedding + (size_t)token * dim;
        float scale = w->token_embedding_scale ? w->token_embedding_scale[token] : 1.0f;
#if defined(__AVX2__)
        __m256 vscale = _mm256_set1_ps(scale);
        for (int i = 0; i <= dim - 8; i += 8) {
            __m128i r8 = _mm_loadl_epi64((const __m128i*)&row[i]);
            __m256 f8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r8));
            _mm256_storeu_ps(&out_emb[i], _mm256_mul_ps(f8, vscale));
        }
#else
        for (int i = 0; i < dim; i++) {
            out_emb[i] = (float)row[i] * scale;
        }
#endif
    } else if (p->qtype == TINYLLM_QTYPE_INT4) {
        uint8_t *w_token_embedding = (uint8_t *)w->token_embedding;
        int dim_packed = (dim + 1) / 2;
        for (int i = 0; i < dim; i++) {
            int packed_idx = i / 2;
            int nibble = i % 2;
            uint8_t packed = w_token_embedding[token * dim_packed + packed_idx];
            int8_t val = nibble == 0 ? ((packed >> 4) & 0x0F) - 8 : (packed & 0x0F) - 8;
            out_emb[i] = val * w->token_embedding_scale[token];
        }
    } else if (p->qtype == TINYLLM_QTYPE_BITNET_158) {
        const int8_t *w_token_embedding = (const int8_t *)w->token_embedding;
        const int8_t *row = w_token_embedding + (size_t)token * dim;
        float scale = w->token_embedding_scale ? w->token_embedding_scale[token] : 1.0f;
#if defined(__AVX2__)
        __m256 vscale = _mm256_set1_ps(scale);
        for (int i = 0; i <= dim - 8; i += 8) {
            __m128i r8 = _mm_loadl_epi64((const __m128i*)&row[i]);
            __m256 f8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r8));
            _mm256_storeu_ps(&out_emb[i], _mm256_mul_ps(f8, vscale));
        }
#else
        for (int i = 0; i < dim; i++) {
            out_emb[i] = (float)row[i] * scale;
        }
#endif
    } else {
        float *w_token_embedding = (float *)w->token_embedding;
        float *content_row = w_token_embedding + token * dim;
        memcpy(out_emb, content_row, dim * sizeof(float));
    }
}

static void tinyllm_embed_lookup(tinyllm_t *llm, tinyllm_token_t token)
{
    tinyllm_get_token_embedding(llm, token, llm->state.x);
}

float *tinyllm_forward_batch_embeddings(tinyllm_t *llm, const float *embeddings, int n_tokens, int start_pos, bool compute_last_logits)
{
    if (!llm || !embeddings || n_tokens <= 0) {
        return NULL;
    }

    tinyllm_config_t *p = &llm->config;
    tinyllm_weights_t *w = &llm->weights;
    tinyllm_state_t *s = &llm->state;

    if (start_pos + n_tokens > p->seq_len) {
        fprintf(stderr, "Error: tinyllm_forward_batch_embeddings: batch exceeds seq_len (%d + %d > %d)\n",
                start_pos, n_tokens, p->seq_len);
        return NULL;
    }

    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_dim = (p->head_dim > 0) ? p->head_dim : (dim / p->n_heads);
    int q_dim = p->n_heads * head_dim;
    int kv_dim = p->n_kv_heads * head_dim;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int attn_scales_per_layer = (p->head_dim > 0 && p->head_dim != dim / p->n_heads) ?
                                (q_dim + kv_dim * 2 + dim) :
                                (dim * 4 + kv_dim * 2);
    int max_act_dim = (dim > q_dim) ? dim : q_dim;

    size_t seq_x_sz = (size_t)n_tokens * dim * sizeof(float);
    size_t seq_xb_sz = (size_t)n_tokens * max_act_dim * sizeof(float);
    size_t seq_xb2_sz = (size_t)n_tokens * dim * sizeof(float);
    size_t seq_hb_sz = (size_t)n_tokens * hidden_dim * sizeof(float);
    size_t seq_hb2_sz = (size_t)n_tokens * hidden_dim * sizeof(float);
    size_t seq_q_sz = (size_t)n_tokens * q_dim * sizeof(float);

    int half_head = head_dim / 2;
    size_t rope_table_sz = (size_t)n_tokens * half_head * sizeof(float);

    float *seq_x = (float *)malloc(seq_x_sz);
    float *seq_xb = (float *)malloc(seq_xb_sz);
    float *seq_xb2 = (float *)malloc(seq_xb2_sz);
    float *seq_hb = (float *)malloc(seq_hb_sz);
    float *seq_hb2 = (float *)malloc(seq_hb2_sz);
    float *seq_q = (float *)malloc(seq_q_sz);
    float *rope_cos_table = (float *)malloc(rope_table_sz);
    float *rope_sin_table = (float *)malloc(rope_table_sz);

    int n_threads = 1;
#if defined(_OPENMP)
    n_threads = omp_get_max_threads();
#endif
    if (n_threads < 1) n_threads = 1;
    float *thread_att_pool = (float *)malloc((size_t)n_threads * p->seq_len * sizeof(float));

    if (!seq_x || !seq_xb || !seq_xb2 || !seq_hb || !seq_hb2 || !seq_q ||
        !rope_cos_table || !rope_sin_table || !thread_att_pool) {
        free(seq_x); free(seq_xb); free(seq_xb2); free(seq_hb);
        free(seq_hb2); free(seq_q); free(rope_cos_table); free(rope_sin_table);
        free(thread_att_pool);
        fprintf(stderr, "Error: tinyllm_forward_batch_embeddings: failed to allocate sequence scratch buffers\n");
        return NULL;
    }

    memcpy(seq_x, embeddings, seq_x_sz);

    /* Precompute RoPE table once for all tokens across all layers */
    float rope_theta = (p->rope_theta > 0.0f) ? p->rope_theta : ((p->vocab_size > 120000) ? 5000000.0f : ((p->vocab_size > 60000) ? 100000.0f : 10000.0f));
    for (int i = 0; i < n_tokens; i++) {
        int pos = start_pos + i;
        float *rcos = rope_cos_table + i * half_head;
        float *rsin = rope_sin_table + i * half_head;
        for (int j = 0; j < half_head; j++) {
            float freq = 1.0f / powf(rope_theta, (float)(2 * j) / (float)head_dim);
            float val = (float)pos * freq;
            rcos[j] = cosf(val);
            rsin[j] = sinf(val);
        }
    }

    float inv_sqrt_head_dim = 1.0f / sqrtf((float)head_dim);
    float rms_eps = (p->vocab_size > 120000) ? 1e-6f : 1e-5f;

    int wk_offset = (p->head_dim > 0 && p->head_dim != dim / p->n_heads) ? q_dim : dim;
    int wv_offset = (p->head_dim > 0 && p->head_dim != dim / p->n_heads) ? (q_dim + kv_dim) : (dim * 2);
    int wo_offset = (p->head_dim > 0 && p->head_dim != dim / p->n_heads) ? (q_dim + kv_dim * 2) : (dim * 2 + kv_dim);

    /* Process Layer-by-Layer: Layer weights stay in CPU cache across all sequence tokens */
    for (int l = 0; l < p->n_layers; l++) {
        int loff = l * p->seq_len * kv_dim;

        /* Step 1: RMSNorm + QKV Projection + RoPE + store into KV cache */
        #pragma omp parallel for schedule(dynamic, 8) if(!omp_in_parallel() && n_tokens >= 8)
        for (int i = 0; i < n_tokens; i++) {
            int pos = start_pos + i;
            float *x_tok = seq_x + i * dim;
            float *xb_tok = seq_xb + i * max_act_dim;
            float *q_tok = seq_q + i * q_dim;
            float *k_tok = s->key_cache + loff + pos * kv_dim;
            float *v_tok = s->value_cache + loff + pos * kv_dim;

            // RMSNorm
            rmsnorm(xb_tok, x_tok, w->rms_att_weight + l * dim, dim, rms_eps);

            // QKV Matmuls
            if (p->qtype == TINYLLM_QTYPE_INT8) {
                const int8_t *w_wq = (const int8_t *)w->wq;
                const int8_t *w_wk = (const int8_t *)w->wk;
                const int8_t *w_wv = (const int8_t *)w->wv;
                tinyllm_matmul_int8_rowscale(q_tok, w_wq + l * q_dim * dim, xb_tok, dim, q_dim,
                                             w->attn_scale + l * attn_scales_per_layer);
                tinyllm_matmul_int8_rowscale(k_tok, w_wk + l * kv_dim * dim, xb_tok, dim, kv_dim,
                                             w->attn_scale + l * attn_scales_per_layer + wk_offset);
                tinyllm_matmul_int8_rowscale(v_tok, w_wv + l * kv_dim * dim, xb_tok, dim, kv_dim,
                                             w->attn_scale + l * attn_scales_per_layer + wv_offset);
            } else if (p->qtype == TINYLLM_QTYPE_INT4) {
                uint8_t *w_wq = (uint8_t *)w->wq;
                uint8_t *w_wk = (uint8_t *)w->wk;
                uint8_t *w_wv = (uint8_t *)w->wv;
                int dim_packed = (dim + 1) / 2;
                tinyllm_matmul_int4(q_tok, w_wq + l * q_dim * dim_packed, xb_tok, dim, q_dim,
                                    w->attn_scale + l * attn_scales_per_layer);
                tinyllm_matmul_int4(k_tok, w_wk + l * kv_dim * dim_packed, xb_tok, dim, kv_dim,
                                    w->attn_scale + l * attn_scales_per_layer + wk_offset);
                tinyllm_matmul_int4(v_tok, w_wv + l * kv_dim * dim_packed, xb_tok, dim, kv_dim,
                                    w->attn_scale + l * attn_scales_per_layer + wv_offset);
            } else if (p->qtype == TINYLLM_QTYPE_BITNET_158) {
                uint8_t *w_wq = (uint8_t *)w->wq;
                uint8_t *w_wk = (uint8_t *)w->wk;
                uint8_t *w_wv = (uint8_t *)w->wv;
                int dim_packed = (dim + 3) / 4;
                tl_bitnet_rotate(llm, xb_tok, w->had_signs_dim, dim);
                tl_bitnet_mm(llm, TL_BN_WQ, l, q_tok, w_wq + l * q_dim * dim_packed, xb_tok, dim, q_dim,
                                         w->attn_scale + l * attn_scales_per_layer);
                tl_bitnet_mm(llm, TL_BN_WK, l, k_tok, w_wk + l * kv_dim * dim_packed, xb_tok, dim, kv_dim,
                                         w->attn_scale + l * attn_scales_per_layer + wk_offset);
                tl_bitnet_mm(llm, TL_BN_WV, l, v_tok, w_wv + l * kv_dim * dim_packed, xb_tok, dim, kv_dim,
                                         w->attn_scale + l * attn_scales_per_layer + wv_offset);
            } else {
                float *w_wq = (float *)w->wq;
                float *w_wk = (float *)w->wk;
                float *w_wv = (float *)w->wv;
                matmul(q_tok, xb_tok, w_wq + l * q_dim * dim, dim, q_dim);
                matmul(k_tok, xb_tok, w_wk + l * kv_dim * dim, dim, kv_dim);
                matmul(v_tok, xb_tok, w_wv + l * kv_dim * dim, dim, kv_dim);
            }

            // Add QKV biases if present
            if (w->bq) {
                const float *bq = w->bq + (size_t)l * q_dim;
                for (int j = 0; j < q_dim; j++) q_tok[j] += bq[j];
            }
            if (w->bk) {
                const float *bk = w->bk + (size_t)l * kv_dim;
                for (int j = 0; j < kv_dim; j++) k_tok[j] += bk[j];
            }
            if (w->bv) {
                const float *bv = w->bv + (size_t)l * kv_dim;
                for (int j = 0; j < kv_dim; j++) v_tok[j] += bv[j];
            }

            // RoPE
            const float *rcos = rope_cos_table + i * half_head;
            const float *rsin = rope_sin_table + i * half_head;
            for (int h = 0; h < p->n_heads; h++) {
                float *qh = q_tok + h * head_dim;
                for (int j = 0; j < half_head; j++) {
                    float q0 = qh[j];
                    float q1 = qh[j + half_head];
                    float c = rcos[j];
                    float s_val = rsin[j];
                    qh[j]             = q0 * c - q1 * s_val;
                    qh[j + half_head] = q1 * c + q0 * s_val;
                }
            }
            for (int h = 0; h < p->n_kv_heads; h++) {
                float *kh = k_tok + h * head_dim;
                for (int j = 0; j < half_head; j++) {
                    float k0 = kh[j];
                    float k1 = kh[j + half_head];
                    float c = rcos[j];
                    float s_val = rsin[j];
                    kh[j]             = k0 * c - k1 * s_val;
                    kh[j + half_head] = k1 * c + k0 * s_val;
                }
            }

            // INT8 KV cache quantization if enabled
            if (p->kv_type == 1 && s->key_cache_i8) {
                int loff_scale = l * p->seq_len * p->n_kv_heads;
                for (int h_kv = 0; h_kv < p->n_kv_heads; h_kv++) {
                    float *kh = k_tok + h_kv * head_dim;
                    int8_t *kh_i8 = s->key_cache_i8 + loff + pos * kv_dim + h_kv * head_dim;
                    float amax_k = 0.0f;
                    for (int j = 0; j < head_dim; j++) {
                        float v_abs = fabsf(kh[j]);
                        if (v_abs > amax_k) amax_k = v_abs;
                    }
                    float scale_k = (amax_k < 1e-12f) ? 1.0f : (amax_k / 127.0f);
                    float inv_scale_k = 127.0f / (amax_k < 1e-12f ? 1.0f : amax_k);
                    s->key_scale[loff_scale + pos * p->n_kv_heads + h_kv] = scale_k;
                    for (int j = 0; j < head_dim; j++) {
                        long qv = (long)(kh[j] * inv_scale_k + (kh[j] >= 0.0f ? 0.5f : -0.5f));
                        if (qv > 127) qv = 127; else if (qv < -128) qv = -128;
                        kh_i8[j] = (int8_t)qv;
                    }

                    float *vh = v_tok + h_kv * head_dim;
                    int8_t *vh_i8 = s->value_cache_i8 + loff + pos * kv_dim + h_kv * head_dim;
                    float amax_v = 0.0f;
                    for (int j = 0; j < head_dim; j++) {
                        float v_abs = fabsf(vh[j]);
                        if (v_abs > amax_v) amax_v = v_abs;
                    }
                    float scale_v = (amax_v < 1e-12f) ? 1.0f : (amax_v / 127.0f);
                    float inv_scale_v = 127.0f / (amax_v < 1e-12f ? 1.0f : amax_v);
                    s->value_scale[loff_scale + pos * p->n_kv_heads + h_kv] = scale_v;
                    for (int j = 0; j < head_dim; j++) {
                        long qv = (long)(vh[j] * inv_scale_v + (vh[j] >= 0.0f ? 0.5f : -0.5f));
                        if (qv > 127) qv = 127; else if (qv < -128) qv = -128;
                        vh_i8[j] = (int8_t)qv;
                    }
                }
            }
        }

        /* Step 2: Multi-head Attention across sequence */
        #pragma omp parallel for schedule(dynamic, 4) if(!omp_in_parallel() && n_tokens >= 4)
        for (int i = 0; i < n_tokens; i++) {
            int tid = 0;
#if defined(_OPENMP)
            tid = omp_get_thread_num();
#endif
            float *local_att = thread_att_pool + tid * p->seq_len;
            int pos = start_pos + i;
            float *q_tok = seq_q + i * q_dim;
            float *xb_tok = seq_xb + i * max_act_dim;

            for (int h = 0; h < p->n_heads; h++) {
                float *q = q_tok + h * head_dim;

                for (int t = 0; t <= pos; t++) {
                    if (p->kv_type == 1 && s->key_cache_i8) {
                        int8_t *k_t_i8 = s->key_cache_i8 + loff + t * kv_dim + (h / kv_mul) * head_dim;
                        int loff_scale = l * p->seq_len * p->n_kv_heads;
                        float k_scale = s->key_scale[loff_scale + t * p->n_kv_heads + (h / kv_mul)];
                        float score = 0.0f;
                        for (int d = 0; d < head_dim; d++) {
                            score += q[d] * (float)k_t_i8[d];
                        }
                        local_att[t] = (score * k_scale) * inv_sqrt_head_dim;
                    } else {
                        float *k_t = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
#if defined(__AVX512F__)
                        __m512 acc512 = _mm512_setzero_ps();
                        for (int d = 0; d < head_dim; d += 16) {
                            __m512 vq = _mm512_loadu_ps(&q[d]);
                            __m512 vk = _mm512_loadu_ps(&k_t[d]);
                            acc512 = _mm512_fmadd_ps(vq, vk, acc512);
                        }
                        local_att[t] = _mm512_reduce_add_ps(acc512) * inv_sqrt_head_dim;
#elif defined(__AVX2__)
                        __m256 acc = _mm256_setzero_ps();
                        for (int d = 0; d < head_dim; d += 8) {
                            __m256 vq = _mm256_loadu_ps(&q[d]);
                            __m256 vk = _mm256_loadu_ps(&k_t[d]);
                            acc = _mm256_fmadd_ps(vq, vk, acc);
                        }
                        __m128 l_v = _mm256_castps256_ps128(acc);
                        __m128 hi_v = _mm256_extractf128_ps(acc, 1);
                        __m128 sm_v = _mm_add_ps(l_v, hi_v);
                        sm_v = _mm_hadd_ps(sm_v, sm_v);
                        sm_v = _mm_hadd_ps(sm_v, sm_v);
                        local_att[t] = _mm_cvtss_f32(sm_v) * inv_sqrt_head_dim;
#elif defined(__ARM_NEON) || defined(__aarch64__)
                        float32x4_t acc0 = vdupq_n_f32(0.0f);
                        float32x4_t acc1 = vdupq_n_f32(0.0f);
                        for (int d = 0; d < head_dim; d += 8) {
                            acc0 = vfmaq_f32(acc0, vld1q_f32(&q[d]), vld1q_f32(&k_t[d]));
                            acc1 = vfmaq_f32(acc1, vld1q_f32(&q[d + 4]), vld1q_f32(&k_t[d + 4]));
                        }
                        local_att[t] = vaddvq_f32(vaddq_f32(acc0, acc1)) * inv_sqrt_head_dim;
#else
                        float score = 0.0f;
                        for (int d = 0; d < head_dim; d++) {
                            score += q[d] * k_t[d];
                        }
                        local_att[t] = score * inv_sqrt_head_dim;
#endif
                    }
                }

                softmax(local_att, pos + 1);

                float *xb_head = xb_tok + h * head_dim;
                memset(xb_head, 0, head_dim * sizeof(float));
                for (int t = 0; t <= pos; t++) {
                    if (p->kv_type == 1 && s->value_cache_i8) {
                        int8_t *v_t_i8 = s->value_cache_i8 + loff + t * kv_dim + (h / kv_mul) * head_dim;
                        int loff_scale = l * p->seq_len * p->n_kv_heads;
                        float v_scale = s->value_scale[loff_scale + t * p->n_kv_heads + (h / kv_mul)];
                        float a = local_att[t] * v_scale;
                        for (int d = 0; d < head_dim; d++) {
                            xb_head[d] += a * (float)v_t_i8[d];
                        }
                    } else {
                        float *v_t = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
                        float a = local_att[t];
#if defined(__AVX512F__)
                        __m512 va512 = _mm512_set1_ps(a);
                        for (int d = 0; d < head_dim; d += 16) {
                            __m512 vxb = _mm512_loadu_ps(&xb_head[d]);
                            __m512 vv = _mm512_loadu_ps(&v_t[d]);
                            _mm512_storeu_ps(&xb_head[d], _mm512_fmadd_ps(va512, vv, vxb));
                        }
#elif defined(__AVX2__)
                        __m256 va = _mm256_set1_ps(a);
                        for (int d = 0; d < head_dim; d += 8) {
                            __m256 vxb = _mm256_loadu_ps(&xb_head[d]);
                            __m256 vv = _mm256_loadu_ps(&v_t[d]);
                            _mm256_storeu_ps(&xb_head[d], _mm256_fmadd_ps(va, vv, vxb));
                        }
#elif defined(__ARM_NEON) || defined(__aarch64__)
                        float32x4_t va = vdupq_n_f32(a);
                        for (int d = 0; d < head_dim; d += 8) {
                            vst1q_f32(&xb_head[d], vfmaq_f32(vld1q_f32(&xb_head[d]), va, vld1q_f32(&v_t[d])));
                            vst1q_f32(&xb_head[d + 4], vfmaq_f32(vld1q_f32(&xb_head[d + 4]), va, vld1q_f32(&v_t[d + 4])));
                        }
#else
                        for (int d = 0; d < head_dim; d++) {
                            xb_head[d] += a * v_t[d];
                        }
#endif
                    }
                }
            }
        }

        /* Step 3: Output projection + Residual */
        #pragma omp parallel for schedule(dynamic, 8) if(!omp_in_parallel() && n_tokens >= 8)
        for (int i = 0; i < n_tokens; i++) {
            float *xb_tok = seq_xb + i * max_act_dim;
            float *xb2_tok = seq_xb2 + i * dim;
            float *x_tok = seq_x + i * dim;

            if (p->qtype == TINYLLM_QTYPE_INT8) {
                const int8_t *w_wo = (const int8_t *)w->wo;
                tinyllm_matmul_int8_rowscale(xb2_tok, w_wo + l * dim * q_dim, xb_tok, q_dim, dim,
                                             w->attn_scale + l * attn_scales_per_layer + wo_offset);
            } else if (p->qtype == TINYLLM_QTYPE_INT4) {
                uint8_t *w_wo = (uint8_t *)w->wo;
                int q_dim_packed = (q_dim + 1) / 2;
                tinyllm_matmul_int4(xb2_tok, w_wo + l * dim * q_dim_packed, xb_tok, q_dim, dim,
                                    w->attn_scale + l * attn_scales_per_layer + wo_offset);
            } else if (p->qtype == TINYLLM_QTYPE_BITNET_158) {
                float *sub_w = w->rms_sub_att_weight ? (w->rms_sub_att_weight + l * q_dim) : NULL;
                rmsnorm(xb_tok, xb_tok, sub_w, q_dim, 1e-5f);
                uint8_t *w_wo = (uint8_t *)w->wo;
                int q_dim_packed = (q_dim + 3) / 4;
                tl_bitnet_rotate(llm, xb_tok, w->had_signs_q, q_dim);
                tl_bitnet_mm(llm, TL_BN_WO, l, xb2_tok, w_wo + l * dim * q_dim_packed, xb_tok, q_dim, dim,
                                         w->attn_scale + l * attn_scales_per_layer + wo_offset);
            } else {
                float *w_wo = (float *)w->wo;
                matmul(xb2_tok, xb_tok, w_wo + l * dim * q_dim, q_dim, dim);
            }

#if defined(__AVX512F__)
            for (int d = 0; d <= dim - 16; d += 16) {
                __m512 vx = _mm512_loadu_ps(&x_tok[d]);
                __m512 vxb2 = _mm512_loadu_ps(&xb2_tok[d]);
                _mm512_storeu_ps(&x_tok[d], _mm512_add_ps(vx, vxb2));
            }
#elif defined(__AVX2__)
            for (int d = 0; d <= dim - 8; d += 8) {
                __m256 vx = _mm256_loadu_ps(&x_tok[d]);
                __m256 vxb2 = _mm256_loadu_ps(&xb2_tok[d]);
                _mm256_storeu_ps(&x_tok[d], _mm256_add_ps(vx, vxb2));
            }
#elif defined(__ARM_NEON) || defined(__aarch64__)
            for (int d = 0; d <= dim - 8; d += 8) {
                vst1q_f32(&x_tok[d], vaddq_f32(vld1q_f32(&x_tok[d]), vld1q_f32(&xb2_tok[d])));
                vst1q_f32(&x_tok[d + 4], vaddq_f32(vld1q_f32(&x_tok[d + 4]), vld1q_f32(&xb2_tok[d + 4])));
            }
#else
            for (int d = 0; d < dim; d++) {
                x_tok[d] += xb2_tok[d];
            }
#endif
        }

        /* Step 4: FFN RMSNorm + SwiGLU + Down + Residual */
        #pragma omp parallel for schedule(dynamic, 8) if(!omp_in_parallel() && n_tokens >= 8)
        for (int i = 0; i < n_tokens; i++) {
            float *x_tok = seq_x + i * dim;
            float *xb_tok = seq_xb + i * max_act_dim;
            float *hb_tok = seq_hb + i * hidden_dim;
            float *hb2_tok = seq_hb2 + i * hidden_dim;

            // FFN RMSNorm
            rmsnorm(xb_tok, x_tok, w->rms_ffn_weight + l * dim, dim, rms_eps);

            // SwiGLU Projections
            if (p->qtype == TINYLLM_QTYPE_INT8) {
                const int8_t *w_w1 = (const int8_t *)w->w1;
                const int8_t *w_w3 = (const int8_t *)w->w3;
                tinyllm_matmul_int8_rowscale(hb_tok, w_w1 + l * hidden_dim * dim, xb_tok, dim, hidden_dim,
                                             w->ffn_scale + l * (hidden_dim * 2 + dim));
                tinyllm_matmul_int8_rowscale(hb2_tok, w_w3 + l * hidden_dim * dim, xb_tok, dim, hidden_dim,
                                             w->ffn_scale + l * (hidden_dim * 2 + dim) + hidden_dim);
            } else if (p->qtype == TINYLLM_QTYPE_INT4) {
                uint8_t *w_w1 = (uint8_t *)w->w1;
                uint8_t *w_w3 = (uint8_t *)w->w3;
                int dim_packed = (dim + 1) / 2;
                tinyllm_matmul_int4(hb_tok, w_w1 + l * hidden_dim * dim_packed, xb_tok, dim, hidden_dim,
                                    w->ffn_scale + l * (hidden_dim * 2 + dim));
                tinyllm_matmul_int4(hb2_tok, w_w3 + l * hidden_dim * dim_packed, xb_tok, dim, hidden_dim,
                                    w->ffn_scale + l * (hidden_dim * 2 + dim) + hidden_dim);
            } else if (p->qtype == TINYLLM_QTYPE_BITNET_158) {
                uint8_t *w_w1 = (uint8_t *)w->w1;
                uint8_t *w_w3 = (uint8_t *)w->w3;
                int dim_packed = (dim + 3) / 4;
                tl_bitnet_rotate(llm, xb_tok, w->had_signs_dim, dim);
                tl_bitnet_mm(llm, TL_BN_W1, l, hb_tok, w_w1 + l * hidden_dim * dim_packed, xb_tok, dim, hidden_dim,
                                         w->ffn_scale + l * (hidden_dim * 2 + dim));
                tl_bitnet_mm(llm, TL_BN_W3, l, hb2_tok, w_w3 + l * hidden_dim * dim_packed, xb_tok, dim, hidden_dim,
                                         w->ffn_scale + l * (hidden_dim * 2 + dim) + hidden_dim);
            } else {
                float *w_w1 = (float *)w->w1;
                float *w_w3 = (float *)w->w3;
                matmul(hb_tok, xb_tok, w_w1 + l * dim * hidden_dim, dim, hidden_dim);
                matmul(hb2_tok, xb_tok, w_w3 + l * dim * hidden_dim, dim, hidden_dim);
            }

            // SwiGLU activation
#if defined(__AVX2__)
            __m256 one = _mm256_set1_ps(1.0f);
            __m256 two = _mm256_set1_ps(2.0f);
            __m256 log2ef = _mm256_set1_ps(1.44269504088896341f);
            __m256 c1 = _mm256_set1_ps(0.6931471805599453f);
            __m256 poly_p = _mm256_set1_ps(0.16666666666666666f);
            __m256 poly_half = _mm256_set1_ps(0.5f);
            __m256 min_x = _mm256_set1_ps(-88.3762626647949f);
            __m256 max_x = _mm256_set1_ps(88.3762626647949f);
            __m256i bias127 = _mm256_set1_epi32(127);

            for (int d = 0; d <= hidden_dim - 8; d += 8) {
                __m256 val = _mm256_loadu_ps(&hb_tok[d]);
                __m256 val2 = _mm256_loadu_ps(&hb2_tok[d]);
                __m256 neg_val = _mm256_sub_ps(_mm256_setzero_ps(), val);

                __m256 z = _mm256_max_ps(neg_val, min_x);
                z = _mm256_min_ps(z, max_x);
                __m256 fx = _mm256_fmadd_ps(z, log2ef, poly_half);
                __m256i emm0 = _mm256_cvttps_epi32(_mm256_floor_ps(fx));
                __m256 n = _mm256_cvtepi32_ps(emm0);
                __m256 x_red = _mm256_fnmadd_ps(n, c1, z);
                __m256 p_poly = _mm256_fmadd_ps(poly_p, x_red, poly_half);
                p_poly = _mm256_fmadd_ps(p_poly, x_red, one);
                p_poly = _mm256_fmadd_ps(p_poly, x_red, one);
                __m256i pow2n = _mm256_slli_epi32(_mm256_add_epi32(emm0, bias127), 23);
                __m256 exp_neg = _mm256_mul_ps(p_poly, _mm256_castsi256_ps(pow2n));

                __m256 denom = _mm256_add_ps(one, exp_neg);
                __m256 r0 = _mm256_rcp_ps(denom);
                __m256 sig = _mm256_mul_ps(r0, _mm256_fnmadd_ps(denom, r0, two));

                __m256 res = _mm256_mul_ps(_mm256_mul_ps(val, sig), val2);
                _mm256_storeu_ps(&hb_tok[d], res);
            }
#else
            for (int d = 0; d < hidden_dim; d++) {
                float val = hb_tok[d];
                val *= (1.0f / (1.0f + expf(-val)));
                val *= hb2_tok[d];
                hb_tok[d] = val;
            }
#endif

            // Down Projection
            if (p->qtype == TINYLLM_QTYPE_INT8) {
                const int8_t *w_w2 = (const int8_t *)w->w2;
                tinyllm_matmul_int8_rowscale(xb_tok, w_w2 + l * dim * hidden_dim, hb_tok, hidden_dim, dim,
                                             w->ffn_scale + l * (hidden_dim * 2 + dim) + hidden_dim * 2);
            } else if (p->qtype == TINYLLM_QTYPE_INT4) {
                uint8_t *w_w2 = (uint8_t *)w->w2;
                int hidden_dim_packed = (hidden_dim + 1) / 2;
                tinyllm_matmul_int4(xb_tok, w_w2 + l * dim * hidden_dim_packed, hb_tok, hidden_dim, dim,
                                    w->ffn_scale + l * (hidden_dim * 2 + dim) + hidden_dim * 2);
            } else if (p->qtype == TINYLLM_QTYPE_BITNET_158) {
                float *sub_w = w->rms_sub_ffn_weight ? (w->rms_sub_ffn_weight + l * hidden_dim) : NULL;
                rmsnorm(hb_tok, hb_tok, sub_w, hidden_dim, 1e-5f);
                uint8_t *w_w2 = (uint8_t *)w->w2;
                int hidden_dim_packed = (hidden_dim + 3) / 4;
                tl_bitnet_rotate(llm, hb_tok, w->had_signs_hidden, hidden_dim);
                tl_bitnet_mm(llm, TL_BN_W2, l, xb_tok, w_w2 + l * dim * hidden_dim_packed, hb_tok, hidden_dim, dim,
                                         w->ffn_scale + l * (hidden_dim * 2 + dim) + hidden_dim * 2);
            } else {
                float *w_w2 = (float *)w->w2;
                matmul(xb_tok, hb_tok, w_w2 + l * hidden_dim * dim, hidden_dim, dim);
            }

            // Residual addition
#if defined(__AVX512F__)
            for (int d = 0; d <= dim - 16; d += 16) {
                __m512 vx = _mm512_loadu_ps(&x_tok[d]);
                __m512 vxb = _mm512_loadu_ps(&xb_tok[d]);
                _mm512_storeu_ps(&x_tok[d], _mm512_add_ps(vx, vxb));
            }
#elif defined(__AVX2__)
            for (int d = 0; d <= dim - 8; d += 8) {
                __m256 vx = _mm256_loadu_ps(&x_tok[d]);
                __m256 vxb = _mm256_loadu_ps(&xb_tok[d]);
                _mm256_storeu_ps(&x_tok[d], _mm256_add_ps(vx, vxb));
            }
#elif defined(__ARM_NEON) || defined(__aarch64__)
            for (int d = 0; d <= dim - 8; d += 8) {
                vst1q_f32(&x_tok[d], vaddq_f32(vld1q_f32(&x_tok[d]), vld1q_f32(&xb_tok[d])));
                vst1q_f32(&x_tok[d + 4], vaddq_f32(vld1q_f32(&x_tok[d + 4]), vld1q_f32(&xb_tok[d + 4])));
            }
#else
            for (int d = 0; d < dim; d++) {
                x_tok[d] += xb_tok[d];
            }
#endif
        }
    }

    // Copy last token's activation into s->x
    memcpy(s->x, seq_x + (size_t)(n_tokens - 1) * dim, dim * sizeof(float));

    // Final RMSNorm
    float rms_eps_final = (p->vocab_size > 120000) ? 1e-6f : 1e-5f;
    rmsnorm(s->x, s->x, w->rms_final_weight, dim, rms_eps_final);

    // Free scratch buffers
    free(seq_x);
    free(seq_xb);
    free(seq_xb2);
    free(seq_hb);
    free(seq_hb2);
    free(seq_q);
    free(rope_cos_table);
    free(rope_sin_table);
    free(thread_att_pool);

    if (!compute_last_logits) {
        return s->x;
    }

    // Classifier (weight tying with token embeddings)
    if (p->qtype == TINYLLM_QTYPE_INT8) {
        const int8_t *w_wcls = (const int8_t *)w->wcls;
        tinyllm_matmul_int8_rowscale(s->logits, w_wcls, s->x, dim, p->vocab_size, w->wcls_scale);
    } else if (p->qtype == TINYLLM_QTYPE_INT4) {
        uint8_t *w_wcls = (uint8_t *)w->wcls;
        tinyllm_matmul_int4(s->logits, w_wcls, s->x, dim, p->vocab_size, w->wcls_scale);
    } else if (p->qtype == TINYLLM_QTYPE_BITNET_158) {
        const int8_t *w_wcls = (const int8_t *)w->wcls;
        tinyllm_matmul_int8_rowscale(s->logits, w_wcls, s->x, dim, p->vocab_size, w->wcls_scale);
    } else {
        float *w_wcls = (float *)w->wcls;
        float *w_token_embedding = (float *)w->token_embedding;
        float *wcls = w_wcls ? w_wcls : w_token_embedding;
        matmul(s->logits, s->x, wcls, dim, p->vocab_size);
    }

    return s->logits;
}

float *tinyllm_forward(tinyllm_t *llm, tinyllm_token_t token, int pos)
{
    tinyllm_embed_lookup(llm, token);
    return tinyllm_forward_embedding_ex(llm, llm->state.x, pos, true);
}

float *tinyllm_forward_no_logits(tinyllm_t *llm, tinyllm_token_t token, int pos)
{
    tinyllm_embed_lookup(llm, token);
    return tinyllm_forward_embedding_ex(llm, llm->state.x, pos, false);
}


// =============================================================================
// Sampling
// =============================================================================

tinyllm_token_t tinyllm_sample_greedy(const float *logits, int vocab_size)
{
    int max_i = 0;
    float max_v = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_v) {
            max_v = logits[i];
            max_i = i;
        }
    }
    return max_i;
}

tinyllm_token_t tinyllm_sample_temp(const float *logits, int vocab_size, float temperature)
{
    if (temperature == 0.0f) {
        return tinyllm_sample_greedy(logits, vocab_size);
    }

    // Apply temperature
    float *scaled = (float *)malloc(vocab_size * sizeof(float));
    for (int i = 0; i < vocab_size; i++) {
        scaled[i] = logits[i] / temperature;
    }

    // Softmax
    softmax(scaled, vocab_size);

    // Sample from distribution
    float r = (float)rand() / RAND_MAX;
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += scaled[i];
        if (r < cumsum) {
            free(scaled);
            return i;
        }
    }

    free(scaled);
    return vocab_size - 1;
}

tinyllm_token_t tinyllm_sample_top_k(const float *logits, int vocab_size, int k, float temperature)
{
    if (k > 64) k = 64;
    int indices[64];
    float values[64];

    for (int i = 0; i < k; i++) {
        indices[i] = -1;
        values[i] = -INFINITY;
    }

    for (int i = 0; i < vocab_size; i++) {
        float v = logits[i];
        if (v <= values[k - 1]) continue;

        for (int j = 0; j < k; j++) {
            if (v > values[j]) {
                // Shift down
                for (int l = k - 1; l > j; l--) {
                    values[l] = values[l - 1];
                    indices[l] = indices[l - 1];
                }
                values[j] = v;
                indices[j] = i;
                break;
            }
        }
    }

    // Apply temperature and softmax
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        values[i] = expf(values[i] / temperature);
        sum += values[i];
    }

    // Sample
    float r = ((float)rand() / (float)RAND_MAX) * sum;
    float cumsum = 0.0f;
    for (int i = 0; i < k; i++) {
        cumsum += values[i];
        if (r < cumsum) {
            return indices[i];
        }
    }

    return indices[k - 1];
}

// =============================================================================
// Quantized MatMul
// =============================================================================

void tinyllm_matmul_int8(float *xout, const int8_t *w, const float *x, int n, int d, float scale)
{
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += w[i * n + j] * x[j];
        }
        xout[i] = sum * scale;
    }
}

void tinyllm_matmul_int4(float *xout, const uint8_t *w, const float *x, int n, int d,
                         const float *scales)
{
    int n_packed = (n + 1) / 2;
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n_packed; j++) {
            uint8_t packed = w[i * n_packed + j];
            int8_t v0 = ((packed >> 4) & 0x0F) - 8;
            int8_t v1 = (packed & 0x0F) - 8;
            sum += v0 * x[j * 2];
            if (j * 2 + 1 < n) {
                sum += v1 * x[j * 2 + 1];
            }
        }
        xout[i] = sum * scales[i];
    }
}

void tinyllm_matmul_bitnet_158(float *out, const uint8_t *w, const float *in, int in_dim, int out_dim,
                               const float *scales)
{
    eif_matmul_bitnet_f32(w, scales, in, NULL, out, out_dim, in_dim);
}

#if defined(__AVX2__)
__attribute__((target("avx2,fma")))
void tinyllm_matmul_int8_rowscale(float *y, const int8_t *w, const float *x, int cols, int rows, const float *scales)
{
    #pragma omp parallel for schedule(static) if((int64_t)rows * cols >= 16384)
    for (int r = 0; r < (rows & ~3); r += 4) {
        const int8_t *w0 = w + r * cols;
        const int8_t *w1 = w + (r + 1) * cols;
        const int8_t *w2 = w + (r + 2) * cols;
        const int8_t *w3 = w + (r + 3) * cols;

        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps();
        __m256 a3 = _mm256_setzero_ps();

        for (int c = 0; c <= cols - 8; c += 8) {
            __m256 in = _mm256_loadu_ps(&x[c]);

            __m128i r0 = _mm_loadl_epi64((const __m128i*)&w0[c]);
            __m128i r1 = _mm_loadl_epi64((const __m128i*)&w1[c]);
            __m128i r2 = _mm_loadl_epi64((const __m128i*)&w2[c]);
            __m128i r3 = _mm_loadl_epi64((const __m128i*)&w3[c]);

            __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r0));
            __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r1));
            __m256 f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r2));
            __m256 f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r3));

            a0 = _mm256_fmadd_ps(in, f0, a0);
            a1 = _mm256_fmadd_ps(in, f1, a1);
            a2 = _mm256_fmadd_ps(in, f2, a2);
            a3 = _mm256_fmadd_ps(in, f3, a3);
        }

        #define I8_HSUM(acc, s) do { \
            __m128 l = _mm256_castps256_ps128(acc); \
            __m128 h = _mm256_extractf128_ps(acc, 1); \
            __m128 sm = _mm_add_ps(l, h); \
            sm = _mm_hadd_ps(sm, sm); \
            sm = _mm_hadd_ps(sm, sm); \
            s = _mm_cvtss_f32(sm); \
        } while(0)

        float s0, s1, s2, s3;
        I8_HSUM(a0, s0);
        I8_HSUM(a1, s1);
        I8_HSUM(a2, s2);
        I8_HSUM(a3, s3);
        /* columns beyond the last full group of 8 */
        for (int c = cols & ~7; c < cols; c++) {
            s0 += (float)w0[c] * x[c];
            s1 += (float)w1[c] * x[c];
            s2 += (float)w2[c] * x[c];
            s3 += (float)w3[c] * x[c];
        }
        y[r]     = s0 * (scales ? scales[r] : 1.0f);
        y[r + 1] = s1 * (scales ? scales[r + 1] : 1.0f);
        y[r + 2] = s2 * (scales ? scales[r + 2] : 1.0f);
        y[r + 3] = s3 * (scales ? scales[r + 3] : 1.0f);
    }

    for (int r = (rows & ~3); r < rows; r++) {
        float sum = 0.0f;
        const int8_t *wr = w + r * cols;
        for (int c = 0; c < cols; c++) {
            sum += (float)wr[c] * x[c];
        }
        y[r] = sum * (scales ? scales[r] : 1.0f);
    }
}
#elif (defined(__ARM_NEON) || defined(__aarch64__))
#include <arm_neon.h>
#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
#define TL_I8_BLK 32

static int tl_has_dotprod(void)
{
    static int cached = -1;
    if (cached < 0) cached = ((getauxval(AT_HWCAP) & HWCAP_ASIMDDP) && !getenv("BITLLM_INT8_F32")) ? 1 : 0;
    return cached;
}

/* W8A8 GEMV: x is quantised to INT8 with one scale per 32 inputs (Q8_0-style), each block is reduced with
 * SDOT and accumulated in FP32. Requires cols % 32 == 0. */
__attribute__((target("+dotprod")))
static void tl_int8_rowscale_dot(float *y, const int8_t *w, const float *x, int cols, int rows, const float *scales)
{
    int nb = cols / TL_I8_BLK;
    int8_t xq[cols] __attribute__((aligned(16)));
    float xs[nb];
    for (int b = 0; b < nb; b++) {
        const float *xb = x + b * TL_I8_BLK;
        float32x4_t m = vdupq_n_f32(0.0f);
        for (int i = 0; i < TL_I8_BLK; i += 4) m = vmaxq_f32(m, vabsq_f32(vld1q_f32(xb + i)));
        float amax = vmaxvq_f32(m);
        float d = amax / 127.0f;
        float id = (amax > 0.0f) ? 127.0f / amax : 0.0f;
        xs[b] = d;
        for (int i = 0; i < TL_I8_BLK; i += 8) {
            int32x4_t q0 = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(xb + i), id));
            int32x4_t q1 = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(xb + i + 4), id));
            int16x8_t q = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
            vst1_s8(xq + b * TL_I8_BLK + i, vqmovn_s16(q));
        }
    }

    #pragma omp parallel for schedule(static) if((int64_t)rows * cols >= 16384)
    for (int r = 0; r < (rows & ~3); r += 4) {
        const int8_t *w0 = w + (size_t)r * cols;
        const int8_t *w1 = w0 + cols, *w2 = w1 + cols, *w3 = w2 + cols;
        float32x4_t f0 = vdupq_n_f32(0.0f), f1 = f0, f2 = f0, f3 = f0;
        for (int b = 0; b < nb; b++) {
            int o = b * TL_I8_BLK;
            int8x16_t xa = vld1q_s8(xq + o), xb = vld1q_s8(xq + o + 16);
            int32x4_t i0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(w0 + o), xa), vld1q_s8(w0 + o + 16), xb);
            int32x4_t i1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(w1 + o), xa), vld1q_s8(w1 + o + 16), xb);
            int32x4_t i2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(w2 + o), xa), vld1q_s8(w2 + o + 16), xb);
            int32x4_t i3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(w3 + o), xa), vld1q_s8(w3 + o + 16), xb);
            f0 = vfmaq_n_f32(f0, vcvtq_f32_s32(i0), xs[b]);
            f1 = vfmaq_n_f32(f1, vcvtq_f32_s32(i1), xs[b]);
            f2 = vfmaq_n_f32(f2, vcvtq_f32_s32(i2), xs[b]);
            f3 = vfmaq_n_f32(f3, vcvtq_f32_s32(i3), xs[b]);
        }
        y[r]     = vaddvq_f32(f0) * (scales ? scales[r] : 1.0f);
        y[r + 1] = vaddvq_f32(f1) * (scales ? scales[r + 1] : 1.0f);
        y[r + 2] = vaddvq_f32(f2) * (scales ? scales[r + 2] : 1.0f);
        y[r + 3] = vaddvq_f32(f3) * (scales ? scales[r + 3] : 1.0f);
    }
    for (int r = (rows & ~3); r < rows; r++) {
        const int8_t *wr = w + (size_t)r * cols;
        float32x4_t f = vdupq_n_f32(0.0f);
        for (int b = 0; b < nb; b++) {
            int o = b * TL_I8_BLK;
            int32x4_t i = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(wr + o), vld1q_s8(xq + o)),
                                    vld1q_s8(wr + o + 16), vld1q_s8(xq + o + 16));
            f = vfmaq_n_f32(f, vcvtq_f32_s32(i), xs[b]);
        }
        y[r] = vaddvq_f32(f) * (scales ? scales[r] : 1.0f);
    }
}
#endif

void tinyllm_matmul_int8_rowscale(float *y, const int8_t *w, const float *x, int cols, int rows, const float *scales)
{
#if defined(__aarch64__) && defined(__linux__)
    if (cols % TL_I8_BLK == 0 && tl_has_dotprod()) {
        tl_int8_rowscale_dot(y, w, x, cols, rows, scales);
        return;
    }
#endif
    #pragma omp parallel for schedule(static) if((int64_t)rows * cols >= 16384)
    for (int r = 0; r < (rows & ~3); r += 4) {
        const int8_t *w0 = w + r * cols;
        const int8_t *w1 = w + (r + 1) * cols;
        const int8_t *w2 = w + (r + 2) * cols;
        const int8_t *w3 = w + (r + 3) * cols;

        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);

        int c = 0;
        for (; c <= cols - 8; c += 8) {
            float32x4_t in_lo = vld1q_f32(&x[c]), in_hi = vld1q_f32(&x[c + 4]);
            int16x8_t r0 = vmovl_s8(vld1_s8(&w0[c]));
            int16x8_t r1 = vmovl_s8(vld1_s8(&w1[c]));
            int16x8_t r2 = vmovl_s8(vld1_s8(&w2[c]));
            int16x8_t r3 = vmovl_s8(vld1_s8(&w3[c]));
            #define I8_ACC(a, r) do { \
                a = vmlaq_f32(a, in_lo, vcvtq_f32_s32(vmovl_s16(vget_low_s16(r)))); \
                a = vmlaq_f32(a, in_hi, vcvtq_f32_s32(vmovl_s16(vget_high_s16(r)))); \
            } while (0)
            I8_ACC(a0, r0); I8_ACC(a1, r1); I8_ACC(a2, r2); I8_ACC(a3, r3);
            #undef I8_ACC
        }
        float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
        for (; c < cols; c++) {
            t0 += (float)w0[c] * x[c]; t1 += (float)w1[c] * x[c];
            t2 += (float)w2[c] * x[c]; t3 += (float)w3[c] * x[c];
        }
        a0 = vsetq_lane_f32(vgetq_lane_f32(a0, 0) + t0, a0, 0);
        a1 = vsetq_lane_f32(vgetq_lane_f32(a1, 0) + t1, a1, 0);
        a2 = vsetq_lane_f32(vgetq_lane_f32(a2, 0) + t2, a2, 0);
        a3 = vsetq_lane_f32(vgetq_lane_f32(a3, 0) + t3, a3, 0);

        #if defined(__aarch64__)
        y[r]     = vaddvq_f32(a0) * (scales ? scales[r] : 1.0f);
        y[r + 1] = vaddvq_f32(a1) * (scales ? scales[r + 1] : 1.0f);
        y[r + 2] = vaddvq_f32(a2) * (scales ? scales[r + 2] : 1.0f);
        y[r + 3] = vaddvq_f32(a3) * (scales ? scales[r + 3] : 1.0f);
        #else
        y[r]     = (vgetq_lane_f32(a0, 0) + vgetq_lane_f32(a0, 1) + vgetq_lane_f32(a0, 2) + vgetq_lane_f32(a0, 3)) * (scales ? scales[r] : 1.0f);
        y[r + 1] = (vgetq_lane_f32(a1, 0) + vgetq_lane_f32(a1, 1) + vgetq_lane_f32(a1, 2) + vgetq_lane_f32(a1, 3)) * (scales ? scales[r + 1] : 1.0f);
        y[r + 2] = (vgetq_lane_f32(a2, 0) + vgetq_lane_f32(a2, 1) + vgetq_lane_f32(a2, 2) + vgetq_lane_f32(a2, 3)) * (scales ? scales[r + 2] : 1.0f);
        y[r + 3] = (vgetq_lane_f32(a3, 0) + vgetq_lane_f32(a3, 1) + vgetq_lane_f32(a3, 2) + vgetq_lane_f32(a3, 3)) * (scales ? scales[r + 3] : 1.0f);
        #endif
    }

    for (int r = (rows & ~3); r < rows; r++) {
        float sum = 0.0f;
        const int8_t *wr = w + r * cols;
        for (int c = 0; c < cols; c++) {
            sum += (float)wr[c] * x[c];
        }
        y[r] = sum * (scales ? scales[r] : 1.0f);
    }
}
#else
void tinyllm_matmul_int8_rowscale(float *y, const int8_t *w, const float *x, int cols, int rows, const float *scales)
{
    #pragma omp parallel for schedule(static) if((int64_t)rows * cols >= 16384)
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        const int8_t *wr = w + r * cols;
        for (int c = 0; c < cols; c++) {
            sum += (float)wr[c] * x[c];
        }
        y[r] = sum * (scales ? scales[r] : 1.0f);
    }
}
#endif

// =============================================================================
// Weight Quantization
// =============================================================================

float tinyllm_quantize_int8(const float *src, int8_t *dst, int n)
{
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float abs_val = fabsf(src[i]);
        if (abs_val > max_abs)
            max_abs = abs_val;
    }

    float scale = max_abs / 127.0f;
    if (scale < 1e-8f)
        scale = 1e-8f;

    for (int i = 0; i < n; i++) {
        float q = src[i] / scale;
        if (q > 127.0f)
            q = 127.0f;
        if (q < -127.0f)
            q = -127.0f;
        dst[i] = (int8_t)roundf(q);
    }

    return scale;
}

/* Per-row INT8 quantization: scales[r] holds the row scale (used by the LM head). */
static void tinyllm_quantize_int8_rowscale(const float *src, int8_t *dst, int cols, int rows, float *scales)
{
    for (int r = 0; r < rows; r++) {
        scales[r] = tinyllm_quantize_int8(src + (size_t)r * cols, dst + (size_t)r * cols, cols);
    }
}

float tinyllm_quantize_int4(const float *src, uint8_t *dst, int n, int out_dim)
{
    (void)out_dim;
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float abs_val = fabsf(src[i]);
        if (abs_val > max_abs)
            max_abs = abs_val;
    }

    float scale = max_abs / 7.0f;
    if (scale < 1e-8f)
        scale = 1e-8f;

    for (int i = 0; i < n; i += 2) {
        float q0 = src[i] / scale;
        float q1 = (i + 1 < n) ? src[i + 1] / scale : 0.0f;

        int8_t v0 = (int8_t)roundf(fmaxf(-8.0f, fminf(7.0f, q0))) + 8;
        int8_t v1 = (int8_t)roundf(fmaxf(-8.0f, fminf(7.0f, q1))) + 8;

        dst[i / 2] = (v0 << 4) | (v1 & 0x0F);
    }

    return scale;
}

float tinyllm_quantize_bitnet_158(const float *src, uint8_t *dst, int n, int out_dim)
{
    (void)out_dim;
    float sum_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_abs += fabsf(src[i]);
    }
    float gamma = (n > 0) ? (sum_abs / (float)n) : 1e-4f;
    if (gamma < 1e-8f)
        gamma = 1e-8f;

    int n_packed = (n + 3) / 4;
    for (int b = 0; b < n_packed; b++) {
        uint8_t byte_val = 0;
        for (int pos = 0; pos < 4; pos++) {
            int idx = b * 4 + pos;
            uint8_t code = 0;
            if (idx < n) {
                float q = roundf(src[idx] / gamma);
                if (q > 1.0f) q = 1.0f;
                if (q < -1.0f) q = -1.0f;
                int8_t val = (int8_t)q;
                if (val > 0) code = 1u;
                else if (val < 0) code = 2u;
            }
            byte_val |= (uint8_t)(code << (pos * 2));
        }
        dst[b] = byte_val;
    }

    return gamma;
}

int tinyllm_quantize_model(tinyllm_t *llm, tinyllm_qtype_t qtype, void *quant_buffer,
                           size_t buffer_size)
{
    tinyllm_config_t *p = &llm->config;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int kv_dim = (dim * p->n_kv_heads) / p->n_heads;

    // Calculate required buffer size
    size_t required = 0;
    if (qtype == TINYLLM_QTYPE_INT8) {
        required = (p->vocab_size * dim + p->n_layers * dim * dim * 4 +
                    p->n_layers * dim * kv_dim * 2 + p->n_layers * dim * hidden_dim * 3) *
                   sizeof(int8_t);
    } else if (qtype == TINYLLM_QTYPE_INT4) {
        int vocab_packed = (p->vocab_size * dim + 1) / 2;
        int dim_packed = (dim * dim + 1) / 2;
        int hidden_packed = (dim * hidden_dim + 1) / 2;
        required = (vocab_packed + p->n_layers * dim_packed * 4 +
                    p->n_layers * ((dim * kv_dim + 1) / 2) * 2 + p->n_layers * hidden_packed * 3) *
                   sizeof(uint8_t);
    } else if (qtype == TINYLLM_QTYPE_BITNET_158) {
        int dim_packed = (dim + 3) / 4;
        int hidden_packed = (hidden_dim + 3) / 4;
        required = ((size_t)p->vocab_size * dim +
                    (size_t)p->n_layers * (dim * dim_packed * 2 + kv_dim * dim_packed * 2 +
                                           hidden_dim * dim_packed * 2 + dim * hidden_packed) +
                    (size_t)p->vocab_size * dim) *
                   sizeof(uint8_t);
    }

    if (buffer_size < required) {
        fprintf(stderr, "Quant buffer too small: need %zu, got %zu\n", required, buffer_size);
        return -1;
    }

    float *w_emb = (float *)llm->weights.token_embedding;
    float *w_wq = (float *)llm->weights.wq;
    float *w_wk = (float *)llm->weights.wk;
    float *w_wv = (float *)llm->weights.wv;
    float *w_wo = (float *)llm->weights.wo;
    float *w_w1 = (float *)llm->weights.w1;
    float *w_w2 = (float *)llm->weights.w2;
    float *w_w3 = (float *)llm->weights.w3;
    /* lm_head source: separate weight if present, otherwise tied to embeddings. */
    float *w_wcls = llm->weights.wcls ? (float *)llm->weights.wcls
                                      : (float *)llm->weights.token_embedding;

    uint8_t *ptr = (uint8_t *)quant_buffer;

    if (qtype == TINYLLM_QTYPE_INT8) {
        // Allocate scale arrays
        llm->weights.token_embedding_scale = (float *)malloc(sizeof(float));
        llm->weights.attn_scale = (float *)malloc(p->n_layers * 4 * sizeof(float));
        llm->weights.ffn_scale = (float *)malloc(p->n_layers * 3 * sizeof(float));

        // Quantize embeddings
        llm->weights.token_embedding_scale[0] =
            tinyllm_quantize_int8(w_emb, (int8_t *)ptr, p->vocab_size * dim);
        llm->weights.token_embedding = ptr;
        ptr += p->vocab_size * dim * sizeof(int8_t);

        // Quantize attention weights - save base pointers
        llm->weights.wq = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            llm->weights.attn_scale[l * 4 + 0] =
                tinyllm_quantize_int8(w_wq + l * dim * dim, (int8_t *)ptr, dim * dim);
            ptr += dim * dim * sizeof(int8_t);
        }

        llm->weights.wk = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            llm->weights.attn_scale[l * 4 + 1] =
                tinyllm_quantize_int8(w_wk + l * dim * kv_dim, (int8_t *)ptr, dim * kv_dim);
            ptr += dim * kv_dim * sizeof(int8_t);
        }

        llm->weights.wv = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            llm->weights.attn_scale[l * 4 + 2] =
                tinyllm_quantize_int8(w_wv + l * dim * kv_dim, (int8_t *)ptr, dim * kv_dim);
            ptr += dim * kv_dim * sizeof(int8_t);
        }

        llm->weights.wo = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            llm->weights.attn_scale[l * 4 + 3] =
                tinyllm_quantize_int8(w_wo + l * dim * dim, (int8_t *)ptr, dim * dim);
            ptr += dim * dim * sizeof(int8_t);
        }

        // Quantize FFN weights
        llm->weights.w1 = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            llm->weights.ffn_scale[l * 3 + 0] =
                tinyllm_quantize_int8(w_w1 + l * dim * hidden_dim, (int8_t *)ptr, dim * hidden_dim);
            ptr += dim * hidden_dim * sizeof(int8_t);
        }

        llm->weights.w2 = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            llm->weights.ffn_scale[l * 3 + 1] =
                tinyllm_quantize_int8(w_w2 + l * hidden_dim * dim, (int8_t *)ptr, hidden_dim * dim);
            ptr += hidden_dim * dim * sizeof(int8_t);
        }

        llm->weights.w3 = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            llm->weights.ffn_scale[l * 3 + 2] =
                tinyllm_quantize_int8(w_w3 + l * dim * hidden_dim, (int8_t *)ptr, dim * hidden_dim);
            ptr += dim * hidden_dim * sizeof(int8_t);
        }

        llm->weights.wcls = llm->weights.token_embedding;
        llm->weights.wcls_scale = llm->weights.token_embedding_scale;
    } else if (qtype == TINYLLM_QTYPE_INT4) {
        // INT4: Per-row scales for each weight matrix
        int total_scales = p->vocab_size + p->n_layers * (dim * 4 + kv_dim * 2 + hidden_dim * 3);
        llm->weights.token_embedding_scale = (float *)malloc(total_scales * sizeof(float));
        llm->weights.attn_scale = llm->weights.token_embedding_scale + p->vocab_size;
        llm->weights.ffn_scale = llm->weights.attn_scale + p->n_layers * (dim * 4 + kv_dim * 2);

        // Quantize embeddings (per-row scales)
        llm->weights.token_embedding = ptr;
        int vocab_packed = (p->vocab_size * dim + 1) / 2;
        for (int i = 0; i < p->vocab_size; i++) {
            llm->weights.token_embedding_scale[i] = tinyllm_quantize_int4(
                w_emb + i * dim, (uint8_t *)ptr + i * ((dim + 1) / 2), dim, dim);
        }
        ptr += vocab_packed * sizeof(uint8_t);

        // Quantize attention weights
        llm->weights.wq = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < dim; row++) {
                llm->weights.attn_scale[l * (dim * 4 + kv_dim * 2) + row] = tinyllm_quantize_int4(
                    w_wq + l * dim * dim + row * dim,
                    (uint8_t *)ptr + l * dim * ((dim + 1) / 2) + row * ((dim + 1) / 2), dim, dim);
            }
        }
        ptr += p->n_layers * dim * ((dim + 1) / 2) * sizeof(uint8_t);

        llm->weights.wk = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < kv_dim; row++) {
                llm->weights.attn_scale[l * (dim * 4 + kv_dim * 2) + dim + row] =
                    tinyllm_quantize_int4(w_wk + l * dim * kv_dim + row * dim,
                                          (uint8_t *)ptr + l * kv_dim * ((dim + 1) / 2) +
                                              row * ((dim + 1) / 2),
                                          dim, dim);
            }
        }
        ptr += p->n_layers * kv_dim * ((dim + 1) / 2) * sizeof(uint8_t);

        llm->weights.wv = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < kv_dim; row++) {
                llm->weights.attn_scale[l * (dim * 4 + kv_dim * 2) + dim * 2 + row] =
                    tinyllm_quantize_int4(w_wv + l * dim * kv_dim + row * dim,
                                          (uint8_t *)ptr + l * kv_dim * ((dim + 1) / 2) +
                                              row * ((dim + 1) / 2),
                                          dim, dim);
            }
        }
        ptr += p->n_layers * kv_dim * ((dim + 1) / 2) * sizeof(uint8_t);

        llm->weights.wo = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < dim; row++) {
                llm->weights.attn_scale[l * (dim * 4 + kv_dim * 2) + dim * 2 + kv_dim + row] =
                    tinyllm_quantize_int4(w_wo + l * dim * dim + row * dim,
                                          (uint8_t *)ptr + l * dim * ((dim + 1) / 2) +
                                              row * ((dim + 1) / 2),
                                          dim, dim);
            }
        }
        ptr += p->n_layers * dim * ((dim + 1) / 2) * sizeof(uint8_t);

        // Quantize FFN weights
        llm->weights.w1 = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < hidden_dim; row++) {
                llm->weights.ffn_scale[l * (hidden_dim * 2 + dim) + row] = tinyllm_quantize_int4(
                    w_w1 + l * dim * hidden_dim + row * dim,
                    (uint8_t *)ptr + l * hidden_dim * ((dim + 1) / 2) + row * ((dim + 1) / 2), dim,
                    dim);
            }
        }
        ptr += p->n_layers * hidden_dim * ((dim + 1) / 2) * sizeof(uint8_t);

        llm->weights.w2 = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < dim; row++) {
                llm->weights.ffn_scale[l * (hidden_dim * 2 + dim) + hidden_dim + row] =
                    tinyllm_quantize_int4(w_w2 + l * hidden_dim * dim + row * hidden_dim,
                                          (uint8_t *)ptr + l * dim * ((hidden_dim + 1) / 2) +
                                              row * ((hidden_dim + 1) / 2),
                                          hidden_dim, hidden_dim);
            }
        }
        ptr += p->n_layers * dim * ((hidden_dim + 1) / 2) * sizeof(uint8_t);

        llm->weights.w3 = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < hidden_dim; row++) {
                llm->weights.ffn_scale[l * (hidden_dim * 2 + dim) + hidden_dim * 2 + row] =
                    tinyllm_quantize_int4(w_w3 + l * dim * hidden_dim + row * dim,
                                          (uint8_t *)ptr + l * hidden_dim * ((dim + 1) / 2) +
                                              row * ((dim + 1) / 2),
                                          dim, dim);
            }
        }
        ptr += p->n_layers * hidden_dim * ((dim + 1) / 2) * sizeof(uint8_t);

        llm->weights.wcls = llm->weights.token_embedding;
        llm->weights.wcls_scale = llm->weights.token_embedding_scale;
    } else if (qtype == TINYLLM_QTYPE_BITNET_158) {
        int total_scales = p->vocab_size * 2 + p->n_layers * (dim * 4 + kv_dim * 2 + hidden_dim * 2 + dim);
        llm->weights.token_embedding_scale = (float *)malloc(total_scales * sizeof(float));
        llm->weights.attn_scale = llm->weights.token_embedding_scale + p->vocab_size;
        llm->weights.ffn_scale = llm->weights.attn_scale + p->n_layers * (dim * 4 + kv_dim * 2);
        llm->weights.wcls_scale = llm->weights.ffn_scale + p->n_layers * (hidden_dim * 2 + dim);

        int dim_packed = (dim + 3) / 4;
        int hidden_packed = (hidden_dim + 3) / 4;

        llm->weights.token_embedding = ptr;
        tinyllm_quantize_int8_rowscale(w_emb, (int8_t *)ptr, dim, p->vocab_size,
                                       llm->weights.token_embedding_scale);
        ptr += (size_t)p->vocab_size * dim;

        llm->weights.wq = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < dim; row++) {
                llm->weights.attn_scale[l * (dim * 4 + kv_dim * 2) + row] =
                    tinyllm_quantize_bitnet_158(w_wq + l * dim * dim + row * dim,
                                                ptr + l * dim * dim_packed + row * dim_packed, dim, dim);
            }
        }
        ptr += p->n_layers * dim * dim_packed;

        llm->weights.wk = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < kv_dim; row++) {
                llm->weights.attn_scale[l * (dim * 4 + kv_dim * 2) + dim + row] =
                    tinyllm_quantize_bitnet_158(w_wk + l * dim * kv_dim + row * dim,
                                                ptr + l * kv_dim * dim_packed + row * dim_packed, dim, dim);
            }
        }
        ptr += p->n_layers * kv_dim * dim_packed;

        llm->weights.wv = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < kv_dim; row++) {
                llm->weights.attn_scale[l * (dim * 4 + kv_dim * 2) + dim * 2 + row] =
                    tinyllm_quantize_bitnet_158(w_wv + l * dim * kv_dim + row * dim,
                                                ptr + l * kv_dim * dim_packed + row * dim_packed, dim, dim);
            }
        }
        ptr += p->n_layers * kv_dim * dim_packed;

        llm->weights.wo = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < dim; row++) {
                llm->weights.attn_scale[l * (dim * 4 + kv_dim * 2) + dim * 2 + kv_dim + row] =
                    tinyllm_quantize_bitnet_158(w_wo + l * dim * dim + row * dim,
                                                ptr + l * dim * dim_packed + row * dim_packed, dim, dim);
            }
        }
        ptr += p->n_layers * dim * dim_packed;

        llm->weights.w1 = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < hidden_dim; row++) {
                llm->weights.ffn_scale[l * (hidden_dim * 2 + dim) + row] =
                    tinyllm_quantize_bitnet_158(w_w1 + l * dim * hidden_dim + row * dim,
                                                ptr + l * hidden_dim * dim_packed + row * dim_packed, dim, dim);
            }
        }
        ptr += p->n_layers * hidden_dim * dim_packed;

        llm->weights.w2 = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < dim; row++) {
                llm->weights.ffn_scale[l * (hidden_dim * 2 + dim) + hidden_dim * 2 + row] =
                    tinyllm_quantize_bitnet_158(w_w2 + l * hidden_dim * dim + row * hidden_dim,
                                                ptr + l * dim * hidden_packed + row * hidden_packed,
                                                hidden_dim, hidden_dim);
            }
        }
        ptr += p->n_layers * dim * hidden_packed;

        llm->weights.w3 = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < hidden_dim; row++) {
                llm->weights.ffn_scale[l * (hidden_dim * 2 + dim) + hidden_dim + row] =
                    tinyllm_quantize_bitnet_158(w_w3 + l * dim * hidden_dim + row * dim,
                                                ptr + l * hidden_dim * dim_packed + row * dim_packed, dim, dim);
            }
        }
        ptr += p->n_layers * hidden_dim * dim_packed;

        /* Output classifier (lm_head): INT8 with per-row scale */
        llm->weights.wcls = ptr;
        tinyllm_quantize_int8_rowscale(w_wcls, (int8_t *)ptr, dim, p->vocab_size,
                                       llm->weights.wcls_scale);
        ptr += (size_t)p->vocab_size * dim;
    }

    llm->config.qtype = qtype;
    printf("Model quantized to %s\n",
           qtype == TINYLLM_QTYPE_BITNET_158 ? "BitNet b1.58 (Ternary 2-bit)" :
           qtype == TINYLLM_QTYPE_INT8 ? "INT8" : "INT4");
    printf("Quantized size: %zu bytes (%.1f MB)\n", required, required / 1024.0f / 1024.0f);
    printf("Compression ratio: %.1fx\n",
           (float)(p->vocab_size * dim +
                   p->n_layers * (dim * dim * 4 + dim * kv_dim * 2 + dim * hidden_dim * 3)) *
               sizeof(float) / required);

    return 0;
}

// =============================================================================
// Model Save/Load
// =============================================================================

#define TINYLLM_FILE_MAGIC 0x54544D4C // "TTML" in hex
/* v2: BitNet-158 uses INT8 (per-row scaled) tied embeddings/LM head + packed layers.
 * Bump invalidates v1 artifacts written with the old packed-embedding layout. */
#define TINYLLM_FILE_VERSION 2

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t config_size;
    uint32_t weights_size;
    uint32_t scales_size;
    uint32_t rms_weights_size;
    uint32_t reserved[3];
} tinyllm_file_header_t;

// Forward declaration
static void tinyllm_setup_quantized_pointers(tinyllm_t *llm, uint8_t *weights_ptr,
                                             float *scales_ptr, size_t weights_size);

int tinyllm_save_quantized_model(const tinyllm_t *llm, const char *filename)
{
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to create file: %s\n", filename);
        return -1;
    }

    const tinyllm_config_t *p = &llm->config;

    // Calculate sizes
    size_t config_size = sizeof(tinyllm_config_t);
    size_t weights_size = 0;
    size_t scales_size = 0;
    size_t rms_weights_size =
        (p->n_layers * p->dim * 2 + p->dim) * sizeof(float); // rms_att + rms_ffn + rms_final

    if (p->qtype == TINYLLM_QTYPE_INT8) {
        weights_size = (p->vocab_size * p->dim + p->n_layers * p->dim * p->dim * 4 +
                        p->n_layers * p->dim * ((p->dim * p->n_kv_heads) / p->n_heads) * 2 +
                        p->n_layers * p->dim * p->hidden_dim * 3) *
                       sizeof(int8_t);
        scales_size = (1 + p->n_layers * 7) * sizeof(float);
    } else if (p->qtype == TINYLLM_QTYPE_INT4) {
        int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
        weights_size =
            ((p->vocab_size * p->dim + 1) / 2 + p->n_layers * p->dim * ((p->dim + 1) / 2) * 4 +
             p->n_layers * kv_dim * ((p->dim + 1) / 2) * 2 +
             p->n_layers * p->hidden_dim * ((p->dim + 1) / 2) * 2 +
             p->n_layers * p->dim * ((p->hidden_dim + 1) / 2)) *
            sizeof(uint8_t);
        scales_size =
            (p->vocab_size + p->n_layers * (p->dim * 4 + kv_dim * 2 + p->hidden_dim * 2 + p->dim)) *
            sizeof(float);
    } else if (p->qtype == TINYLLM_QTYPE_BITNET_158) {
        int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
        int dim_packed = (p->dim + 3) / 4;
        int hidden_packed = (p->hidden_dim + 3) / 4;
        weights_size = ((size_t)p->vocab_size * p->dim +
                        (size_t)p->n_layers * (p->dim * dim_packed * 2 + kv_dim * dim_packed * 2 +
                                               p->hidden_dim * dim_packed * 2 + p->dim * hidden_packed) +
                        (size_t)p->vocab_size * p->dim) *
                       sizeof(uint8_t);
        scales_size =
            (p->vocab_size * 2 + p->n_layers * (p->dim * 4 + kv_dim * 2 + p->hidden_dim * 2 + p->dim)) *
            sizeof(float);
    } else {
        fprintf(stderr, "Only quantized models can be saved\n");
        fclose(f);
        return -1;
    }

    // Write header
    tinyllm_file_header_t header = {.magic = TINYLLM_FILE_MAGIC,
                                    .version = TINYLLM_FILE_VERSION,
                                    .config_size = (uint32_t)config_size,
                                    .weights_size = (uint32_t)weights_size,
                                    .scales_size = (uint32_t)scales_size,
                                    .rms_weights_size = (uint32_t)rms_weights_size,
                                    .reserved = {0}};

    if (fwrite(&header, sizeof(header), 1, f) != 1)
        goto error;

    // Write config
    if (fwrite(p, config_size, 1, f) != 1)
        goto error;

    // Write weights (pointed to by llm->weights)
    const uint8_t *weights_data = (const uint8_t *)llm->weights.token_embedding;
    if (fwrite(weights_data, weights_size, 1, f) != 1)
        goto error;

    // Write scales - each scale array separately (they may not be contiguous)
    if (p->qtype == TINYLLM_QTYPE_INT8) {
        if (fwrite(llm->weights.token_embedding_scale, sizeof(float), 1, f) != 1)
            goto error;
        if (fwrite(llm->weights.attn_scale, p->n_layers * 4 * sizeof(float), 1, f) != 1)
            goto error;
        if (fwrite(llm->weights.ffn_scale, p->n_layers * 3 * sizeof(float), 1, f) != 1)
            goto error;
    } else {
        int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
        if (fwrite(llm->weights.token_embedding_scale, p->vocab_size * sizeof(float), 1, f) != 1)
            goto error;
        if (fwrite(llm->weights.attn_scale, p->n_layers * (p->dim * 4 + kv_dim * 2) * sizeof(float), 1, f) != 1)
            goto error;
        if (fwrite(llm->weights.ffn_scale, p->n_layers * (p->hidden_dim * 2 + p->dim) * sizeof(float), 1, f) != 1)
            goto error;
        if (p->qtype == TINYLLM_QTYPE_BITNET_158) {
            if (fwrite(llm->weights.wcls_scale, p->vocab_size * sizeof(float), 1, f) != 1)
                goto error;
        }
    }

    // Write RMS weights (not quantized)
    if (fwrite(llm->weights.rms_att_weight, p->n_layers * p->dim * sizeof(float), 1, f) != 1)
        goto error;
    if (fwrite(llm->weights.rms_ffn_weight, p->n_layers * p->dim * sizeof(float), 1, f) != 1)
        goto error;
    if (fwrite(llm->weights.rms_final_weight, p->dim * sizeof(float), 1, f) != 1)
        goto error;

    fclose(f);

    printf("Model saved to %s\n", filename);
    printf("  Config: %zu bytes\n", config_size);
    printf("  Weights: %zu bytes\n", weights_size);
    printf("  Scales: %zu bytes\n", scales_size);
    printf("  RMS weights: %zu bytes\n", rms_weights_size);
    printf("  Total: %zu bytes (%.1f MB)\n",
           sizeof(header) + config_size + weights_size + scales_size + rms_weights_size,
           (sizeof(header) + config_size + weights_size + scales_size + rms_weights_size) /
               1024.0f / 1024.0f);

    return 0;

error:
    fprintf(stderr, "Error writing model file\n");
    fclose(f);
    return -1;
}

int tinyllm_load_quantized_model(tinyllm_t *llm, const char *filename, void *quant_buffer,
                                 size_t buffer_size)
{
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return -1;
    }

    // Read and verify header
    uint32_t magic = 0;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1) {
        fprintf(stderr, "Error reading file magic\n");
        fclose(f);
        return -1;
    }
    fseek(f, 0, SEEK_SET);

    size_t weights_size = 0, scales_size = 0, rms_weights_size = 0;
    long weights_file_offset = 0, scales_file_offset = 0, rms_file_offset = 0;

    if (magic == 0x53564C4D) { // 'SVLM' (Multimodal / Granite Docling format)
        uint32_t svlm_hdr[12];
        if (fread(svlm_hdr, sizeof(uint32_t), 12, f) != 12) {
            fprintf(stderr, "Error reading SVLM header\n");
            fclose(f);
            return -1;
        }
        weights_file_offset = (long)svlm_hdr[3];
        scales_file_offset  = (long)svlm_hdr[4];
        rms_file_offset     = (long)svlm_hdr[5];
        weights_size        = svlm_hdr[7];
        scales_size         = svlm_hdr[8];
        rms_weights_size    = svlm_hdr[9];
    } else if (magic == TINYLLM_FILE_MAGIC || magic == 0x544C4C4D) { // 'TTML' or 'TLLM'
        tinyllm_file_header_t header;
        if (fread(&header, sizeof(header), 1, f) != 1) {
            fprintf(stderr, "Error reading TTML header\n");
            fclose(f);
            return -1;
        }
        weights_file_offset = sizeof(header) + header.config_size;
        scales_file_offset  = weights_file_offset + header.weights_size;
        rms_file_offset     = scales_file_offset + header.scales_size;
        weights_size        = header.weights_size;
        scales_size         = header.scales_size;
        rms_weights_size    = header.rms_weights_size;
    } else {
        fprintf(stderr, "Invalid file format (wrong magic: 0x%08X)\n", magic);
        fclose(f);
        return -1;
    }

    size_t total_size = weights_size + scales_size + rms_weights_size;
    if (buffer_size < total_size) {
        fprintf(stderr, "Buffer too small: need %zu, got %zu\n", total_size, buffer_size);
        fclose(f);
        return -1;
    }

    // Read weights into buffer
    uint8_t *ptr = (uint8_t *)quant_buffer;
    fseek(f, weights_file_offset, SEEK_SET);
    if (fread(ptr, weights_size, 1, f) != 1) {
        fprintf(stderr, "Error reading weights\n");
        fclose(f);
        return -1;
    }

    // Read scales into buffer after weights
    float *scales_ptr = (float *)(ptr + weights_size);
    fseek(f, scales_file_offset, SEEK_SET);
    if (fread(scales_ptr, scales_size, 1, f) != 1) {
        fprintf(stderr, "Error reading scales\n");
        fclose(f);
        return -1;
    }

    // Read RMS weights into buffer after scales
    float *rms_ptr = (float *)((uint8_t *)scales_ptr + scales_size);
    fseek(f, rms_file_offset, SEEK_SET);
    if (fread(rms_ptr, rms_weights_size, 1, f) != 1) {
        fprintf(stderr, "Error reading RMS weights\n");
        fclose(f);
        return -1;
    }

    fclose(f);

    // Set up all pointers
    tinyllm_config_t *p = &llm->config;

    // RMS weights pointers
    llm->weights.rms_att_weight = rms_ptr;
    llm->weights.rms_ffn_weight = rms_ptr + p->n_layers * p->dim;
    llm->weights.rms_final_weight = llm->weights.rms_ffn_weight + p->n_layers * p->dim;

    int q_dim = p->n_heads * ((p->head_dim > 0) ? p->head_dim : (p->dim / p->n_heads));
    size_t standard_rms_count = (size_t)p->n_layers * p->dim * 2 + p->dim;
    size_t extra_rms_count = (size_t)p->n_layers * q_dim + (size_t)p->n_layers * p->hidden_dim;
    if (rms_weights_size >= (standard_rms_count + extra_rms_count) * sizeof(float)) {
        float *extra_rms = llm->weights.rms_final_weight + p->dim;
        llm->weights.rms_sub_att_weight = extra_rms;
        llm->weights.rms_sub_ffn_weight = extra_rms + (size_t)p->n_layers * q_dim;
    } else {
        llm->weights.rms_sub_att_weight = NULL;
        llm->weights.rms_sub_ffn_weight = NULL;
    }

    // Quantized weights pointers
    tinyllm_setup_quantized_pointers(llm, ptr, scales_ptr, weights_size);

    fprintf(stderr, "Model loaded from %s\n", filename);
    const char *qstr = "FP32";
    if (llm->config.qtype == TINYLLM_QTYPE_INT8) qstr = "INT8";
    else if (llm->config.qtype == TINYLLM_QTYPE_INT4) qstr = "INT4";
    else if (llm->config.qtype == TINYLLM_QTYPE_BITNET_158) qstr = "BitNet b1.58 (Ternary 1.58-bit)";
    fprintf(stderr, "  Quantization: %s\n", qstr);
    fflush(stderr);

    return 0;
}

static void tinyllm_setup_quantized_pointers(tinyllm_t *llm, uint8_t *weights_ptr,
                                             float *scales_ptr, size_t weights_size)
{
    tinyllm_config_t *p = &llm->config;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_dim = (p->head_dim > 0) ? p->head_dim : (dim / p->n_heads);
    int q_dim = p->n_heads * head_dim;
    int kv_dim = p->n_kv_heads * head_dim;
    int attn_scales_per_layer = (p->head_dim > 0 && p->head_dim != dim / p->n_heads) ?
                                (q_dim + kv_dim * 2 + dim) :
                                (dim * 4 + kv_dim * 2);

    if (p->qtype == TINYLLM_QTYPE_INT8) {
        uint8_t *ptr = weights_ptr;

        llm->weights.token_embedding = ptr;
        llm->weights.token_embedding_scale = scales_ptr;
        ptr += (size_t)p->vocab_size * dim;

        llm->weights.wq = ptr;
        ptr += (size_t)p->n_layers * q_dim * dim;

        llm->weights.wk = ptr;
        ptr += (size_t)p->n_layers * kv_dim * dim;

        llm->weights.wv = ptr;
        ptr += (size_t)p->n_layers * kv_dim * dim;

        llm->weights.wo = ptr;
        ptr += (size_t)p->n_layers * dim * q_dim;

        llm->weights.w1 = ptr;
        ptr += (size_t)p->n_layers * hidden_dim * dim;

        llm->weights.w2 = ptr;
        ptr += (size_t)p->n_layers * dim * hidden_dim;

        llm->weights.w3 = ptr;
        ptr += (size_t)p->n_layers * hidden_dim * dim;

        size_t consumed = (size_t)(ptr - weights_ptr);
        size_t lm_head_size = (size_t)p->vocab_size * dim;
        if (weights_size >= consumed + lm_head_size) {
            llm->weights.wcls = ptr;
            ptr += lm_head_size;
            llm->weights.wcls_scale = scales_ptr + p->vocab_size + p->n_layers * attn_scales_per_layer + p->n_layers * (hidden_dim * 2 + dim);
        } else {
            llm->weights.wcls = llm->weights.token_embedding;
            llm->weights.wcls_scale = llm->weights.token_embedding_scale;
        }

        llm->weights.attn_scale = scales_ptr + p->vocab_size;
        llm->weights.ffn_scale = llm->weights.attn_scale + p->n_layers * attn_scales_per_layer;
    } else if (p->qtype == TINYLLM_QTYPE_INT4) {
        uint8_t *ptr = weights_ptr;
        int dim_packed = (dim + 1) / 2;
        int q_dim_packed = (q_dim + 1) / 2;
        int hidden_packed = (hidden_dim + 1) / 2;

        llm->weights.token_embedding = ptr;
        llm->weights.token_embedding_scale = scales_ptr;
        ptr += (size_t)p->vocab_size * dim_packed;

        llm->weights.wq = ptr;
        ptr += (size_t)p->n_layers * q_dim * dim_packed;

        llm->weights.wk = ptr;
        ptr += (size_t)p->n_layers * kv_dim * dim_packed;

        llm->weights.wv = ptr;
        ptr += (size_t)p->n_layers * kv_dim * dim_packed;

        llm->weights.wo = ptr;
        ptr += (size_t)p->n_layers * dim * q_dim_packed;

        llm->weights.w1 = ptr;
        ptr += (size_t)p->n_layers * hidden_dim * dim_packed;

        llm->weights.w2 = ptr;
        ptr += (size_t)p->n_layers * dim * hidden_packed;

        llm->weights.w3 = ptr;
        ptr += (size_t)p->n_layers * hidden_dim * dim_packed;

        size_t consumed = (size_t)(ptr - weights_ptr);
        size_t lm_head_size = (size_t)p->vocab_size * dim_packed;
        if (weights_size >= consumed + lm_head_size) {
            llm->weights.wcls = ptr;
            ptr += lm_head_size;
            llm->weights.wcls_scale = scales_ptr + p->vocab_size + p->n_layers * attn_scales_per_layer + p->n_layers * (hidden_dim * 2 + dim);
        } else {
            llm->weights.wcls = llm->weights.token_embedding;
            llm->weights.wcls_scale = llm->weights.token_embedding_scale;
        }

        llm->weights.attn_scale = scales_ptr + p->vocab_size;
        llm->weights.ffn_scale = llm->weights.attn_scale + p->n_layers * attn_scales_per_layer;
    } else if (p->qtype == TINYLLM_QTYPE_BITNET_158) {
        uint8_t *ptr = weights_ptr;
        int dim_packed = (dim + 3) / 4;
        int q_dim_packed = (q_dim + 3) / 4;
        int hidden_packed = (hidden_dim + 3) / 4;

        // 1. Token embeddings: INT8 [vocab_size, dim]
        llm->weights.token_embedding = ptr;
        ptr += (size_t)p->vocab_size * dim * sizeof(int8_t);

        // 2. Transformer layers: BitNet b1.58 Ternary 2-bit
        llm->weights.wq = ptr;
        ptr += (size_t)p->n_layers * q_dim * dim_packed;

        llm->weights.wk = ptr;
        ptr += (size_t)p->n_layers * kv_dim * dim_packed;

        llm->weights.wv = ptr;
        ptr += (size_t)p->n_layers * kv_dim * dim_packed;

        llm->weights.wo = ptr;
        ptr += (size_t)p->n_layers * dim * q_dim_packed;

        llm->weights.w1 = ptr;
        ptr += (size_t)p->n_layers * hidden_dim * dim_packed;

        llm->weights.w2 = ptr;
        ptr += (size_t)p->n_layers * dim * hidden_packed;

        llm->weights.w3 = ptr;
        ptr += (size_t)p->n_layers * hidden_dim * dim_packed;

        // 3. Output classifier (lm_head): INT8 [vocab_size, dim]
        llm->weights.wcls = ptr;
        ptr += (size_t)p->vocab_size * dim * sizeof(int8_t);

        // 4. Scales layout
        llm->weights.token_embedding_scale = scales_ptr;
        llm->weights.attn_scale = scales_ptr + p->vocab_size;
        llm->weights.ffn_scale = llm->weights.attn_scale + p->n_layers * attn_scales_per_layer;
        llm->weights.wcls_scale = llm->weights.ffn_scale + p->n_layers * (hidden_dim * 2 + dim);

        for (int m = 0; m < TL_BN_COUNT; m++) llm->weights.bitnet_gscale[m] = NULL;
        llm->weights.had_signs_dim = llm->weights.had_signs_q = llm->weights.had_signs_hidden = NULL;
        if (p->bitnet_group > 0 || p->bitnet_hadamard) {
            /* Grouped layout (docling_to_eif --bitnet-group/--hadamard): embedding scales [vocab], then per matrix
             * kind q,k,v,o,w1,w3,w2 the scales [n_layers, rows, in/g], then lm_head scales [vocab], then the
             * Hadamard signs for widths dim, q_dim, hidden_dim (only with bitnet_hadamard). Per-row files with only
             * hadamard set store g = in_dim (one scale per row) in the same layout. */
            const int rows[TL_BN_COUNT] = {q_dim, kv_dim, kv_dim, dim, hidden_dim, hidden_dim, dim};
            const int ins[TL_BN_COUNT] = {dim, dim, dim, q_dim, dim, dim, hidden_dim};
            const float *sp = scales_ptr + p->vocab_size;
            for (int m = 0; m < TL_BN_COUNT; m++) {
                int g = (p->bitnet_group > 0) ? eif_bitnet_group_size(ins[m], p->bitnet_group) : 0;
                if (g <= 0) g = ins[m];
                llm->weights.bitnet_gscale[m] = sp;
                sp += (size_t)p->n_layers * rows[m] * (ins[m] / g);
            }
            llm->weights.wcls_scale = (float *)sp;
            sp += p->vocab_size;
            if (p->bitnet_hadamard) {
                llm->weights.had_signs_dim = sp;
                sp += dim;
                llm->weights.had_signs_q = sp;
                sp += q_dim;
                llm->weights.had_signs_hidden = sp;
            }
            if (p->bitnet_group <= 0) p->bitnet_group = 1 << 30; /* route through the scale arrays (g = in_dim) */
        }
    }
}

// =============================================================================
// Utility
// =============================================================================

void tinyllm_print_info(const tinyllm_t *llm)
{
    printf("TinyLLM Model:\n");
    printf("  Dim: %d\n", llm->config.dim);
    printf("  Hidden: %d\n", llm->config.hidden_dim);
    printf("  Layers: %d\n", llm->config.n_layers);
    printf("  Heads: %d\n", llm->config.n_heads);
    printf("  Vocab: %d\n", llm->config.vocab_size);
    printf("  SeqLen: %d\n", llm->config.seq_len);
    printf("  Quantization: %s\n", llm->config.qtype == TINYLLM_QTYPE_BITNET_158 ? "BitNet b1.58 (Ternary 2-bit)"
                                   : llm->config.qtype == TINYLLM_QTYPE_INT4   ? "INT4"
                                   : llm->config.qtype == TINYLLM_QTYPE_INT8   ? "INT8"
                                                                               : "FP32");
    printf("  Memory: %zu bytes\n", llm->memory_size);
}

void tinyllm_memory_stats(const tinyllm_t *llm, size_t *used, size_t *total)
{
    *total = llm->memory_size;
    *used = tinyllm_memory_size(&llm->config);
}

// =============================================================================
// Tokenizer Loader
// =============================================================================

int tinyllm_load_tokenizer(tinyllm_t *llm, const char *tokenizer_path)
{
    if (!llm || !tokenizer_path)
        return -1;

    FILE *f = fopen(tokenizer_path, "rb");
    if (!f)
        return -1;

    int max_token_length = 0;
    if (fread(&max_token_length, sizeof(int), 1, f) != 1) {
        fclose(f);
        return -1;
    }

    int vocab_size = llm->config.vocab_size;
    if (vocab_size <= 0) {
        fclose(f);
        return -1;
    }

    llm->tokenizer.vocab = (char **)malloc(vocab_size * sizeof(char *));
    llm->tokenizer.vocab_scores = (float *)malloc(vocab_size * sizeof(float));
    if (!llm->tokenizer.vocab || !llm->tokenizer.vocab_scores) {
        fclose(f);
        return -1;
    }
    llm->tokenizer.vocab_size = vocab_size;
    llm->tokenizer.bos_token = 1;
    llm->tokenizer.eos_token = 2;

    for (int i = 0; i < vocab_size; i++) {
        if (fread(&llm->tokenizer.vocab_scores[i], sizeof(float), 1, f) != 1) {
            fclose(f);
            return -1;
        }
        int len = 0;
        if (fread(&len, sizeof(int), 1, f) != 1 || len < 0 || len > 2048) {
            fclose(f);
            return -1;
        }
        llm->tokenizer.vocab[i] = (char *)malloc(len + 1);
        if (!llm->tokenizer.vocab[i]) {
            fclose(f);
            return -1;
        }
        if (fread(llm->tokenizer.vocab[i], 1, len, f) != (size_t)len) {
            fclose(f);
            return -1;
        }
        llm->tokenizer.vocab[i][len] = '\0';
    }

    fclose(f);
    return 0;
}

void tinyllm_free_tokenizer(tinyllm_t *llm)
{
    if (!llm || !llm->tokenizer.vocab)
        return;
    for (int i = 0; i < llm->tokenizer.vocab_size; i++) {
        if (llm->tokenizer.vocab[i]) {
            free(llm->tokenizer.vocab[i]);
            llm->tokenizer.vocab[i] = NULL;
        }
    }
    free(llm->tokenizer.vocab);
    free(llm->tokenizer.vocab_scores);
    llm->tokenizer.vocab = NULL;
    llm->tokenizer.vocab_scores = NULL;
    llm->tokenizer.vocab_size = 0;
}

// =============================================================================
// Native GGUF Causal LLM & Embedded Tokenizer Support
// =============================================================================

int tinyllm_read_gguf_config(const char *filename, tinyllm_config_t *cfg,
                             char *out_arch, size_t max_arch)
{
    if (!filename || !cfg) return -1;
    memset(cfg, 0, sizeof(*cfg));

    FILE *f = fopen(filename, "rb");
    if (!f) return -1;

    uint32_t magic = 0, ver = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (fread(&magic, 4, 1, f) != 1 || fread(&ver, 4, 1, f) != 1 ||
        fread(&n_tensors, 8, 1, f) != 1 || fread(&n_kv, 8, 1, f) != 1) {
        fclose(f);
        return -2;
    }
    if (magic != GGUF_MAGIC) {
        fclose(f);
        return -3;
    }

    char arch[64] = "llama";
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t klen = 0;
        if (fread(&klen, 8, 1, f) != 1 || klen > 1024) break;
        char key[256];
        size_t to_read = klen < sizeof(key) - 1 ? klen : sizeof(key) - 1;
        if (fread(key, 1, to_read, f) != to_read) break;
        key[to_read] = '\0';
        if (klen > to_read) fseek(f, (long)(klen - to_read), SEEK_CUR);

        uint32_t vtype = 0;
        if (fread(&vtype, 4, 1, f) != 1) break;

        if (strcmp(key, "general.architecture") == 0 && vtype == 8) {
            uint64_t slen = 0;
            if (fread(&slen, 8, 1, f) != 1) break;
            size_t sread = slen < sizeof(arch) - 1 ? slen : sizeof(arch) - 1;
            if (fread(arch, 1, sread, f) != sread) break;
            arch[sread] = '\0';
            if (slen > sread) fseek(f, (long)(slen - sread), SEEK_CUR);
            if (out_arch && max_arch > 0) {
                snprintf(out_arch, max_arch, "%s", arch);
            }
            continue;
        }

        char k_dim[128], k_inter[128], k_layers[128], k_heads[128], k_kv_heads[128];
        char k_seq[128], k_rope[128], k_head_dim[128];
        snprintf(k_dim, sizeof(k_dim), "%s.embedding_length", arch);
        snprintf(k_inter, sizeof(k_inter), "%s.feed_forward_length", arch);
        snprintf(k_layers, sizeof(k_layers), "%s.block_count", arch);
        snprintf(k_heads, sizeof(k_heads), "%s.attention.head_count", arch);
        snprintf(k_kv_heads, sizeof(k_kv_heads), "%s.attention.head_count_kv", arch);
        snprintf(k_seq, sizeof(k_seq), "%s.context_length", arch);
        snprintf(k_rope, sizeof(k_rope), "%s.rope.freq_base", arch);
        snprintf(k_head_dim, sizeof(k_head_dim), "%s.attention.key_length", arch);

        if (strcmp(key, k_dim) == 0) {
            uint32_t val = 0;
            if (vtype == 4 || vtype == 5) { if (fread(&val, 4, 1, f) == 1) cfg->dim = (int)val; }
            else if (vtype == 10 || vtype == 11) { uint64_t v64 = 0; if (fread(&v64, 8, 1, f) == 1) cfg->dim = (int)v64; }
            continue;
        } else if (strcmp(key, k_inter) == 0) {
            uint32_t val = 0;
            if (vtype == 4 || vtype == 5) { if (fread(&val, 4, 1, f) == 1) cfg->hidden_dim = (int)val; }
            else if (vtype == 10 || vtype == 11) { uint64_t v64 = 0; if (fread(&v64, 8, 1, f) == 1) cfg->hidden_dim = (int)v64; }
            continue;
        } else if (strcmp(key, k_layers) == 0) {
            uint32_t val = 0;
            if (vtype == 4 || vtype == 5) { if (fread(&val, 4, 1, f) == 1) cfg->n_layers = (int)val; }
            else if (vtype == 10 || vtype == 11) { uint64_t v64 = 0; if (fread(&v64, 8, 1, f) == 1) cfg->n_layers = (int)v64; }
            continue;
        } else if (strcmp(key, k_heads) == 0) {
            uint32_t val = 0;
            if (vtype == 4 || vtype == 5) { if (fread(&val, 4, 1, f) == 1) cfg->n_heads = (int)val; }
            else if (vtype == 10 || vtype == 11) { uint64_t v64 = 0; if (fread(&v64, 8, 1, f) == 1) cfg->n_heads = (int)v64; }
            continue;
        } else if (strcmp(key, k_kv_heads) == 0) {
            uint32_t val = 0;
            if (vtype == 4 || vtype == 5) { if (fread(&val, 4, 1, f) == 1) cfg->n_kv_heads = (int)val; }
            else if (vtype == 10 || vtype == 11) { uint64_t v64 = 0; if (fread(&v64, 8, 1, f) == 1) cfg->n_kv_heads = (int)v64; }
            continue;
        } else if (strcmp(key, k_seq) == 0) {
            uint32_t val = 0;
            if (vtype == 4 || vtype == 5) { if (fread(&val, 4, 1, f) == 1) cfg->seq_len = (int)val; }
            else if (vtype == 10 || vtype == 11) { uint64_t v64 = 0; if (fread(&v64, 8, 1, f) == 1) cfg->seq_len = (int)v64; }
            continue;
        } else if (strcmp(key, k_rope) == 0 || strstr(key, "rope.freq_base") != NULL || strstr(key, "rope_freq_base") != NULL) {
            if (vtype == 6) { if (fread(&cfg->rope_theta, 4, 1, f) != 1) cfg->rope_theta = 0.0f; }
            else if (vtype == 12) { double d = 0; if (fread(&d, 8, 1, f) == 1) cfg->rope_theta = (float)d; }
            else if (vtype == 4 || vtype == 5) { uint32_t v = 0; if (fread(&v, 4, 1, f) == 1) cfg->rope_theta = (float)v; }
            else if (vtype == 10 || vtype == 11) { uint64_t v = 0; if (fread(&v, 8, 1, f) == 1) cfg->rope_theta = (float)v; }
            continue;
        } else if (strcmp(key, k_head_dim) == 0) {
            uint32_t val = 0;
            if (vtype == 4 || vtype == 5) { if (fread(&val, 4, 1, f) == 1) cfg->head_dim = (int)val; }
            else if (vtype == 10 || vtype == 11) { uint64_t v64 = 0; if (fread(&v64, 8, 1, f) == 1) cfg->head_dim = (int)v64; }
            continue;
        } else if (strcmp(key, "tokenizer.ggml.tokens") == 0 && vtype == 9) {
            uint32_t atype = 0;
            uint64_t count = 0;
            if (fread(&atype, 4, 1, f) == 1 && fread(&count, 8, 1, f) == 1) {
                cfg->vocab_size = (int)count;
                if (atype == 8) {
                    for (uint64_t t = 0; t < count; t++) {
                        uint64_t sl = 0;
                        if (fread(&sl, 8, 1, f) != 1) break;
                        fseek(f, (long)sl, SEEK_CUR);
                    }
                }
            }
            continue;
        }

        /* Skip other metadata */
        if (vtype == 0 || vtype == 1 || vtype == 7) fseek(f, 1, SEEK_CUR);
        else if (vtype == 2 || vtype == 3) fseek(f, 2, SEEK_CUR);
        else if (vtype == 4 || vtype == 5 || vtype == 6) fseek(f, 4, SEEK_CUR);
        else if (vtype == 8) {
            uint64_t sl = 0;
            if (fread(&sl, 8, 1, f) != 1) break;
            fseek(f, (long)sl, SEEK_CUR);
        } else if (vtype >= 10 && vtype <= 12) fseek(f, 8, SEEK_CUR);
        else if (vtype == 9) {
            uint32_t atype = 0;
            uint64_t acount = 0;
            if (fread(&atype, 4, 1, f) != 1 || fread(&acount, 8, 1, f) != 1) break;
            if (atype == 8) {
                for (uint64_t k = 0; k < acount; k++) {
                    uint64_t sl = 0;
                    if (fread(&sl, 8, 1, f) != 1) break;
                    fseek(f, (long)sl, SEEK_CUR);
                }
            } else if (atype == 0 || atype == 1 || atype == 7) fseek(f, (long)acount, SEEK_CUR);
            else if (atype == 2 || atype == 3) fseek(f, (long)(acount * 2), SEEK_CUR);
            else if (atype == 4 || atype == 5 || atype == 6) fseek(f, (long)(acount * 4), SEEK_CUR);
            else if (atype >= 10 && atype <= 12) fseek(f, (long)(acount * 8), SEEK_CUR);
        }
    }
    fclose(f);

    if (cfg->n_kv_heads == 0) cfg->n_kv_heads = cfg->n_heads;
    if (cfg->head_dim == 0 && cfg->n_heads > 0) cfg->head_dim = cfg->dim / cfg->n_heads;
    if (cfg->seq_len <= 0) cfg->seq_len = 2048;
    if (cfg->seq_len > 4096) cfg->seq_len = 4096;
    if (cfg->rope_theta <= 0.0f) {
        if (strstr(arch, "qwen") != NULL || strstr(filename, "qwen") != NULL) {
            cfg->rope_theta = 1000000.0f;
        } else {
            cfg->rope_theta = 10000.0f;
        }
    }
    if (strcmp(arch, "bitnet") == 0 || strstr(filename, "bitnet") != NULL) {
        cfg->qtype = TINYLLM_QTYPE_BITNET_158;
    } else {
        cfg->qtype = TINYLLM_QTYPE_INT8;
    }

    if (n_tensors == 0 || cfg->dim <= 0 || cfg->n_layers <= 0 || cfg->n_heads <= 0) {
        fprintf(stderr, "[EIF TinyLLM Error] Incomplete model architecture or vocab-only GGUF (tensors=%lu, dim=%d, layers=%d, heads=%d)\n",
                (unsigned long)n_tensors, cfg->dim, cfg->n_layers, cfg->n_heads);
        return -4;
    }

    return 0;
}

int tinyllm_load_gguf_tokenizer(const char *filename, eif_bpe_tokenizer_t *tok)
{
    if (!filename || !tok) return -1;

    FILE *f = fopen(filename, "rb");
    if (!f) return -1;

    uint32_t magic = 0, ver = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (fread(&magic, 4, 1, f) != 1 || fread(&ver, 4, 1, f) != 1 ||
        fread(&n_tensors, 8, 1, f) != 1 || fread(&n_kv, 8, 1, f) != 1) {
        fclose(f);
        return -2;
    }
    if (magic != GGUF_MAGIC) {
        fclose(f);
        return -3;
    }

    int bos_id = -1, eos_id = -1;
    char **tokens = NULL;
    int vocab_size = 0;

    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t klen = 0;
        if (fread(&klen, 8, 1, f) != 1 || klen > 1024) break;
        char key[256];
        size_t to_read = klen < sizeof(key) - 1 ? klen : sizeof(key) - 1;
        if (fread(key, 1, to_read, f) != to_read) break;
        key[to_read] = '\0';
        if (klen > to_read) fseek(f, (long)(klen - to_read), SEEK_CUR);

        uint32_t vtype = 0;
        if (fread(&vtype, 4, 1, f) != 1) break;

        if (strcmp(key, "tokenizer.ggml.bos_token_id") == 0) {
            uint32_t val = 0;
            if (fread(&val, 4, 1, f) == 1) bos_id = (int)val;
            continue;
        } else if (strcmp(key, "tokenizer.ggml.eos_token_id") == 0) {
            uint32_t val = 0;
            if (fread(&val, 4, 1, f) == 1) eos_id = (int)val;
            continue;
        } else if (strcmp(key, "tokenizer.ggml.tokens") == 0 && vtype == 9) {
            uint32_t atype = 0;
            uint64_t count = 0;
            if (fread(&atype, 4, 1, f) == 1 && fread(&count, 8, 1, f) == 1 && atype == 8) {
                vocab_size = (int)count;
                tokens = (char **)malloc((size_t)count * sizeof(char *));
                if (!tokens) break;
                for (uint64_t t = 0; t < count; t++) {
                    uint64_t slen = 0;
                    if (fread(&slen, 8, 1, f) != 1) break;
                    tokens[t] = (char *)malloc(slen + 1);
                    if (tokens[t]) {
                        if (fread(tokens[t], 1, slen, f) == slen) {
                            tokens[t][slen] = '\0';
                        } else {
                            tokens[t][0] = '\0';
                        }
                    } else {
                        fseek(f, (long)slen, SEEK_CUR);
                    }
                }
            }
            continue;
        }

        /* Skip other metadata */
        if (vtype == 0 || vtype == 1 || vtype == 7) fseek(f, 1, SEEK_CUR);
        else if (vtype == 2 || vtype == 3) fseek(f, 2, SEEK_CUR);
        else if (vtype == 4 || vtype == 5 || vtype == 6) fseek(f, 4, SEEK_CUR);
        else if (vtype == 8) {
            uint64_t sl = 0;
            if (fread(&sl, 8, 1, f) != 1) break;
            fseek(f, (long)sl, SEEK_CUR);
        } else if (vtype >= 10 && vtype <= 12) fseek(f, 8, SEEK_CUR);
        else if (vtype == 9) {
            uint32_t atype = 0;
            uint64_t acount = 0;
            if (fread(&atype, 4, 1, f) != 1 || fread(&acount, 8, 1, f) != 1) break;
            if (atype == 8) {
                for (uint64_t k = 0; k < acount; k++) {
                    uint64_t sl = 0;
                    if (fread(&sl, 8, 1, f) != 1) break;
                    fseek(f, (long)sl, SEEK_CUR);
                }
            } else if (atype == 0 || atype == 1 || atype == 7) fseek(f, (long)acount, SEEK_CUR);
            else if (atype == 2 || atype == 3) fseek(f, (long)(acount * 2), SEEK_CUR);
            else if (atype == 4 || atype == 5 || atype == 6) fseek(f, (long)(acount * 4), SEEK_CUR);
            else if (atype >= 10 && atype <= 12) fseek(f, (long)(acount * 8), SEEK_CUR);
        }
    }
    fclose(f);

    if (!tokens || vocab_size <= 0) return -4;

    eif_status_t st = eif_bpe_tokenizer_init_from_vocab(tok, tokens, NULL, vocab_size, bos_id, eos_id);
    return (st == EIF_STATUS_OK) ? 0 : -5;
}

static void load_rmsnorm(float *dest, const void *src, uint32_t type, int count)
{
    if (!dest) return;
    if (!src) {
        for (int i = 0; i < count; i++) dest[i] = 1.0f;
        return;
    }
    if (type == GGUF_TYPE_F32) {
        memcpy(dest, src, (size_t)count * sizeof(float));
    } else if (type == GGUF_TYPE_F16) {
        const uint16_t *f16 = (const uint16_t *)src;
        for (int i = 0; i < count; i++) {
            dest[i] = fp16_to_fp32(f16[i]);
        }
    } else {
        for (int i = 0; i < count; i++) dest[i] = 1.0f;
    }
}

/* ========================================================================= */
/* GGUF Block Dequantization Helpers (Q4_0, Q4_1, Q4_K, Q6_K)                */
/* ========================================================================= */

#define QK4_0 32
#define QK4_1 32
#define QK_K  256

#pragma pack(push, 1)
typedef struct {
    uint16_t d;
    uint8_t  qs[16];
} block_q4_0_t;

typedef struct {
    uint16_t d;
    uint16_t m;
    uint8_t  qs[16];
} block_q4_1_t;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
} block_q4_K_t;

typedef struct {
    uint8_t  ql[128];
    uint8_t  qh[64];
    int8_t   scales[16];
    uint16_t d;
} block_q6_K_t;
#pragma pack(pop)

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4)   | ((q[j - 0] >> 6) << 4));
    }
}

static void dequantize_row_q4_0(const void *src, float *y, int k) {
    const block_q4_0_t *x = (const block_q4_0_t *)src;
    int nb = k / QK4_0;
    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < 16; ++j) {
            int x0 = (x[i].qs[j] & 0x0F) - 8;
            int x1 = (x[i].qs[j] >> 4) - 8;
            y[i * 32 + j]      = x0 * d;
            y[i * 32 + j + 16] = x1 * d;
        }
    }
}

static void dequantize_row_q4_1(const void *src, float *y, int k) {
    const block_q4_1_t *x = (const block_q4_1_t *)src;
    int nb = k / QK4_1;
    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(x[i].d);
        float m = fp16_to_fp32(x[i].m);
        for (int j = 0; j < 16; ++j) {
            int x0 = (x[i].qs[j] & 0x0F);
            int x1 = (x[i].qs[j] >> 4);
            y[i * 32 + j]      = x0 * d + m;
            y[i * 32 + j + 16] = x1 * d + m;
        }
    }
}

static void dequantize_row_q4_K(const void *src, float *y, int k) {
    const block_q4_K_t *x = (const block_q4_K_t *)src;
    int nb = k / QK_K;
    float *py = y;
    for (int i = 0; i < nb; i++) {
        const uint8_t *q = x[i].qs;
        float d = fp16_to_fp32(x[i].d);
        float min = fp16_to_fp32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            float d1 = d * sc;
            float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            float d2 = d * sc;
            float m2 = min * m;
            for (int l = 0; l < 32; ++l) *py++ = d1 * (q[l] & 0x0F) - m1;
            for (int l = 0; l < 32; ++l) *py++ = d2 * (q[l] >> 4)   - m2;
            q += 32;
            is += 2;
        }
    }
}

static void dequantize_row_q6_K(const void *src, float *y, int k) {
    const block_q6_K_t *x = (const block_q6_K_t *)src;
    int nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t  *sc = x[i].scales;
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l +  0]  >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32]  >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

static void quantize_row_f32_to_int8(int8_t *dst, float *dest_scale, const float *row_f, int cols) {
    float max_val = 0.0f;
    for (int c = 0; c < cols; c++) {
        float a = fabsf(row_f[c]);
        if (a > max_val) max_val = a;
    }
    float scale = max_val / 127.0f;
    if (scale < 1e-12f) scale = 1.0f;
    float inv_scale = 1.0f / scale;
    *dest_scale = scale;
    for (int c = 0; c < cols; c++) {
        int q = (int)roundf(row_f[c] * inv_scale);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        dst[c] = (int8_t)q;
    }
}

static void load_embedding_tensor(
    int8_t *dest_w,
    float *dest_scales,
    const void *src_data,
    uint32_t src_type,
    int vocab_size,
    int dim)
{
    if (!src_data || !dest_w || !dest_scales) return;

    if (src_type == GGUF_TYPE_Q8_0) {
        int blocks_per_row = (dim + 31) / 32;
        float *row_buf = (float *)malloc((size_t)blocks_per_row * 32 * sizeof(float));
        if (row_buf) {
            const uint8_t *src_bytes = (const uint8_t *)src_data;
            for (int v = 0; v < vocab_size; v++) {
                const uint8_t *row_src = src_bytes + (size_t)v * blocks_per_row * 34;
                for (int b = 0; b < blocks_per_row; b++) {
                    const uint8_t *blk = row_src + b * 34;
                    uint16_t h16; memcpy(&h16, blk, 2);
                    float sc = fp16_to_fp32(h16);
                    const int8_t *qs = (const int8_t *)(blk + 2);
                    for (int i = 0; i < 32; i++) row_buf[b * 32 + i] = sc * qs[i];
                }
                quantize_row_f32_to_int8(dest_w + (size_t)v * dim, &dest_scales[v], row_buf, dim);
            }
            free(row_buf);
        }
    } else if (src_type == GGUF_TYPE_Q4_0 || src_type == GGUF_TYPE_Q4_1 ||
               src_type == GGUF_TYPE_Q4_K || src_type == GGUF_TYPE_Q6_K) {
        float *row_buf = (float *)malloc((size_t)dim * sizeof(float));
        if (row_buf) {
            size_t block_bytes = (src_type == GGUF_TYPE_Q4_0) ? sizeof(block_q4_0_t) :
                                 ((src_type == GGUF_TYPE_Q4_1) ? sizeof(block_q4_1_t) :
                                 ((src_type == GGUF_TYPE_Q4_K) ? sizeof(block_q4_K_t) : sizeof(block_q6_K_t)));
            int blk_size = (src_type == GGUF_TYPE_Q4_K || src_type == GGUF_TYPE_Q6_K) ? QK_K : 32;
            size_t row_stride = (size_t)(dim / blk_size) * block_bytes;
            const uint8_t *src_bytes = (const uint8_t *)src_data;
            for (int v = 0; v < vocab_size; v++) {
                const void *r_src = src_bytes + (size_t)v * row_stride;
                if (src_type == GGUF_TYPE_Q4_0) dequantize_row_q4_0(r_src, row_buf, dim);
                else if (src_type == GGUF_TYPE_Q4_1) dequantize_row_q4_1(r_src, row_buf, dim);
                else if (src_type == GGUF_TYPE_Q4_K) dequantize_row_q4_K(r_src, row_buf, dim);
                else dequantize_row_q6_K(r_src, row_buf, dim);
                quantize_row_f32_to_int8(dest_w + (size_t)v * dim, &dest_scales[v], row_buf, dim);
            }
            free(row_buf);
        }
    } else if (src_type == GGUF_TYPE_F32 || src_type == GGUF_TYPE_F16) {
        for (int v = 0; v < vocab_size; v++) {
            float max_val = 0.0f;
            for (int c = 0; c < dim; c++) {
                float val = (src_type == GGUF_TYPE_F32) ?
                            ((const float *)src_data)[(size_t)v * dim + c] :
                            fp16_to_fp32(((const uint16_t *)src_data)[(size_t)v * dim + c]);
                float a = fabsf(val);
                if (a > max_val) max_val = a;
            }
            float scale = max_val / 127.0f;
            if (scale < 1e-12f) scale = 1.0f;
            float inv_scale = 1.0f / scale;
            dest_scales[v] = scale;
            for (int c = 0; c < dim; c++) {
                float val = (src_type == GGUF_TYPE_F32) ?
                            ((const float *)src_data)[(size_t)v * dim + c] :
                            fp16_to_fp32(((const uint16_t *)src_data)[(size_t)v * dim + c]);
                int q = (int)roundf(val * inv_scale);
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                dest_w[(size_t)v * dim + c] = (int8_t)q;
            }
        }
    }
}

static void load_projection_tensor(
    void *dest_w,
    float *dest_scales,
    const void *src_data,
    uint32_t src_type,
    int rows,
    int cols,
    tinyllm_qtype_t qtype)
{
    if (!src_data || !dest_w || !dest_scales) return;

    if (qtype == TINYLLM_QTYPE_BITNET_158) {
        uint8_t *dst = (uint8_t *)dest_w;
        int row_packed = (cols + 3) / 4;

        if (src_type == GGUF_TYPE_TL1 || src_type == GGUF_TYPE_TL2) {
            int blocks = (cols + 31) / 32;
            const uint8_t *src_bytes = (const uint8_t *)src_data;
            for (int r = 0; r < rows; r++) {
                const uint8_t *r_src = src_bytes + (size_t)r * blocks * 10;
                uint8_t *r_dst = dst + (size_t)r * row_packed;
                float s_sum = 0.0f;
                for (int b = 0; b < blocks; b++) {
                    const uint8_t *blk = r_src + b * 10;
                    s_sum += fp16_to_fp32(*(const uint16_t *)blk);
                    memcpy(r_dst + b * 8, blk + 2, 8);
                }
                dest_scales[r] = (blocks > 0) ? (s_sum / blocks) : 1.0f;
            }
        } else if (src_type == GGUF_TYPE_Q8_0) {
            int blocks = (cols + 31) / 32;
            const uint8_t *src_bytes = (const uint8_t *)src_data;
            for (int r = 0; r < rows; r++) {
                const uint8_t *r_src = src_bytes + (size_t)r * blocks * 34;
                uint8_t *r_dst = dst + (size_t)r * row_packed;
                memset(r_dst, 0, (size_t)row_packed);
                float s_sum = 0.0f;
                for (int b = 0; b < blocks; b++) {
                    const uint8_t *blk = r_src + b * 34;
                    float sc = fp16_to_fp32(*(const uint16_t *)blk);
                    s_sum += sc;
                    const int8_t *qs = (const int8_t *)(blk + 2);
                    int count = (b == blocks - 1) ? (cols - b * 32) : 32;
                    for (int i = 0; i < count; i++) {
                        int c = b * 32 + i;
                        int8_t v = qs[i];
                        uint8_t code = (v > 0) ? 1 : ((v < 0) ? 2 : 0);
                        r_dst[c / 4] |= (code << ((c % 4) * 2));
                    }
                }
                dest_scales[r] = (blocks > 0) ? (s_sum / blocks) : 1.0f;
            }
        } else if (src_type == GGUF_TYPE_Q4_0 || src_type == GGUF_TYPE_Q4_1 ||
                   src_type == GGUF_TYPE_Q4_K || src_type == GGUF_TYPE_Q6_K) {
            float *row_buf = (float *)malloc((size_t)cols * sizeof(float));
            if (row_buf) {
                size_t block_bytes = (src_type == GGUF_TYPE_Q4_0) ? sizeof(block_q4_0_t) :
                                     ((src_type == GGUF_TYPE_Q4_1) ? sizeof(block_q4_1_t) :
                                     ((src_type == GGUF_TYPE_Q4_K) ? sizeof(block_q4_K_t) : sizeof(block_q6_K_t)));
                int blk_size = (src_type == GGUF_TYPE_Q4_K || src_type == GGUF_TYPE_Q6_K) ? QK_K : 32;
                size_t row_stride = (size_t)(cols / blk_size) * block_bytes;
                const uint8_t *src_bytes = (const uint8_t *)src_data;
                for (int r = 0; r < rows; r++) {
                    const void *r_src = src_bytes + (size_t)r * row_stride;
                    if (src_type == GGUF_TYPE_Q4_0) dequantize_row_q4_0(r_src, row_buf, cols);
                    else if (src_type == GGUF_TYPE_Q4_1) dequantize_row_q4_1(r_src, row_buf, cols);
                    else if (src_type == GGUF_TYPE_Q4_K) dequantize_row_q4_K(r_src, row_buf, cols);
                    else dequantize_row_q6_K(r_src, row_buf, cols);

                    uint8_t *r_dst = dst + (size_t)r * row_packed;
                    memset(r_dst, 0, (size_t)row_packed);
                    float sum_abs = 0.0f;
                    for (int c = 0; c < cols; c++) sum_abs += fabsf(row_buf[c]);
                    float gamma = (cols > 0) ? (sum_abs / cols) : 1.0f;
                    if (gamma < 1e-12f) gamma = 1.0f;
                    dest_scales[r] = gamma;
                    float inv_gamma = 1.0f / gamma;
                    for (int c = 0; c < cols; c++) {
                        float scaled = row_buf[c] * inv_gamma;
                        uint8_t code = (scaled > 0.5f) ? 1 : ((scaled < -0.5f) ? 2 : 0);
                        r_dst[c / 4] |= (code << ((c % 4) * 2));
                    }
                }
                free(row_buf);
            }
        } else if (src_type == GGUF_TYPE_F32 || src_type == GGUF_TYPE_F16) {
            for (int r = 0; r < rows; r++) {
                uint8_t *r_dst = dst + (size_t)r * row_packed;
                memset(r_dst, 0, (size_t)row_packed);
                float sum_abs = 0.0f;
                for (int c = 0; c < cols; c++) {
                    float val = (src_type == GGUF_TYPE_F32) ?
                                ((const float *)src_data)[(size_t)r * cols + c] :
                                fp16_to_fp32(((const uint16_t *)src_data)[(size_t)r * cols + c]);
                    sum_abs += fabsf(val);
                }
                float gamma = (cols > 0) ? (sum_abs / cols) : 1.0f;
                if (gamma < 1e-12f) gamma = 1.0f;
                dest_scales[r] = gamma;
                float inv_gamma = 1.0f / gamma;
                for (int c = 0; c < cols; c++) {
                    float val = (src_type == GGUF_TYPE_F32) ?
                                ((const float *)src_data)[(size_t)r * cols + c] :
                                fp16_to_fp32(((const uint16_t *)src_data)[(size_t)r * cols + c]);
                    float scaled = val * inv_gamma;
                    uint8_t code = (scaled > 0.5f) ? 1 : ((scaled < -0.5f) ? 2 : 0);
                    r_dst[c / 4] |= (code << ((c % 4) * 2));
                }
            }
        }
    } else if (qtype == TINYLLM_QTYPE_INT8) {
        int8_t *dst = (int8_t *)dest_w;
        if (src_type == GGUF_TYPE_Q8_0) {
            int blocks = (cols + 31) / 32;
            float *row_buf = (float *)malloc((size_t)blocks * 32 * sizeof(float));
            if (row_buf) {
                const uint8_t *src_bytes = (const uint8_t *)src_data;
                for (int r = 0; r < rows; r++) {
                    const uint8_t *r_src = src_bytes + (size_t)r * blocks * 34;
                    for (int b = 0; b < blocks; b++) {
                        const uint8_t *blk = r_src + b * 34;
                        uint16_t h16; memcpy(&h16, blk, 2);
                        float sc = fp16_to_fp32(h16);
                        const int8_t *qs = (const int8_t *)(blk + 2);
                        for (int i = 0; i < 32; i++) row_buf[b * 32 + i] = sc * qs[i];
                    }
                    quantize_row_f32_to_int8(dst + (size_t)r * cols, &dest_scales[r], row_buf, cols);
                }
                free(row_buf);
            }
        } else if (src_type == GGUF_TYPE_Q4_0 || src_type == GGUF_TYPE_Q4_1 ||
                   src_type == GGUF_TYPE_Q4_K || src_type == GGUF_TYPE_Q6_K) {
            float *row_buf = (float *)malloc((size_t)cols * sizeof(float));
            if (row_buf) {
                size_t block_bytes = (src_type == GGUF_TYPE_Q4_0) ? sizeof(block_q4_0_t) :
                                     ((src_type == GGUF_TYPE_Q4_1) ? sizeof(block_q4_1_t) :
                                     ((src_type == GGUF_TYPE_Q4_K) ? sizeof(block_q4_K_t) : sizeof(block_q6_K_t)));
                int blk_size = (src_type == GGUF_TYPE_Q4_K || src_type == GGUF_TYPE_Q6_K) ? QK_K : 32;
                size_t row_stride = (size_t)(cols / blk_size) * block_bytes;
                const uint8_t *src_bytes = (const uint8_t *)src_data;
                for (int r = 0; r < rows; r++) {
                    const void *r_src = src_bytes + (size_t)r * row_stride;
                    if (src_type == GGUF_TYPE_Q4_0) dequantize_row_q4_0(r_src, row_buf, cols);
                    else if (src_type == GGUF_TYPE_Q4_1) dequantize_row_q4_1(r_src, row_buf, cols);
                    else if (src_type == GGUF_TYPE_Q4_K) dequantize_row_q4_K(r_src, row_buf, cols);
                    else dequantize_row_q6_K(r_src, row_buf, cols);
                    quantize_row_f32_to_int8(dst + (size_t)r * cols, &dest_scales[r], row_buf, cols);
                }
                free(row_buf);
            }
        } else if (src_type == GGUF_TYPE_F32 || src_type == GGUF_TYPE_F16) {
            for (int r = 0; r < rows; r++) {
                float max_val = 0.0f;
                for (int c = 0; c < cols; c++) {
                    float val = (src_type == GGUF_TYPE_F32) ?
                                ((const float *)src_data)[(size_t)r * cols + c] :
                                fp16_to_fp32(((const uint16_t *)src_data)[(size_t)r * cols + c]);
                    float a = fabsf(val);
                    if (a > max_val) max_val = a;
                }
                float scale = max_val / 127.0f;
                if (scale < 1e-12f) scale = 1.0f;
                float inv_s = 1.0f / scale;
                dest_scales[r] = scale;
                for (int c = 0; c < cols; c++) {
                    float val = (src_type == GGUF_TYPE_F32) ?
                                ((const float *)src_data)[(size_t)r * cols + c] :
                                fp16_to_fp32(((const uint16_t *)src_data)[(size_t)r * cols + c]);
                    int q = (int)roundf(val * inv_s);
                    if (q > 127) q = 127;
                    if (q < -128) q = -128;
                    dst[(size_t)r * cols + c] = (int8_t)q;
                }
            }
        }
    }
}

int tinyllm_load_gguf(tinyllm_t *llm, const char *filename, void *quant_buffer, size_t buffer_size)
{
    if (!llm || !filename || !quant_buffer) return -1;

    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[EIF TinyLLM Error] Cannot open GGUF model: %s\n", filename);
        return -1;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return -1;
    }

    size_t file_sz = (size_t)st.st_size;
    const uint8_t *addr = (const uint8_t *)mmap(NULL, file_sz, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);

    if (addr == MAP_FAILED) {
        fprintf(stderr, "[EIF TinyLLM Error] mmap failed for: %s\n", filename);
        return -1;
    }

    const uint8_t *p = addr;
    uint32_t magic = gguf_read_u32(&p);
    uint32_t ver = gguf_read_u32(&p);
    uint64_t n_tensors = gguf_read_u64(&p);
    uint64_t n_kv = gguf_read_u64(&p);
    (void)ver;

    if (magic != GGUF_MAGIC) {
        munmap((void *)addr, file_sz);
        return -2;
    }

    uint32_t alignment = 32;
    char model_arch[64] = "transformer";
    for (uint64_t i = 0; i < n_kv; i++) {
        char key[128];
        gguf_read_str(&p, key, sizeof(key));
        uint32_t vtype = gguf_read_u32(&p);
        if (strcmp(key, "general.alignment") == 0) {
            alignment = gguf_read_u32(&p);
        } else if (strcmp(key, "general.architecture") == 0 && vtype == 8) {
            uint64_t sl = gguf_read_u64(&p);
            size_t to_copy = (sl < sizeof(model_arch) - 1) ? (size_t)sl : sizeof(model_arch) - 1;
            memcpy(model_arch, p, to_copy);
            model_arch[to_copy] = '\0';
            p += sl;
        } else if (strstr(key, "rope.freq_base") != NULL || strstr(key, "rope_freq_base") != NULL) {
            if (vtype == 6) { llm->config.rope_theta = *(const float *)p; p += 4; }
            else if (vtype == 12) { llm->config.rope_theta = (float)*(const double *)p; p += 8; }
            else if (vtype == 4 || vtype == 5) { llm->config.rope_theta = (float)*(const uint32_t *)p; p += 4; }
            else if (vtype == 10 || vtype == 11) { llm->config.rope_theta = (float)*(const uint64_t *)p; p += 8; }
            else { p += 4; }
        } else {
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
                        for (uint64_t k = 0; k < acount; k++) {
                            uint64_t sl = gguf_read_u64(&p);
                            p += sl;
                        }
                    } else if (atype == 0 || atype == 1 || atype == 7) p += acount;
                    else if (atype == 2 || atype == 3) p += acount * 2;
                    else if (atype == 4 || atype == 5 || atype == 6) p += acount * 4;
                    else if (atype >= 10 && atype <= 12) p += acount * 8;
                    break;
                }
                default:
                    munmap((void *)addr, file_sz);
                    return -3;
            }
        }
    }

    if (llm->config.rope_theta <= 0.0f) {
        if (strstr(model_arch, "qwen") != NULL || strstr(filename, "qwen") != NULL) {
            llm->config.rope_theta = 1000000.0f;
        } else {
            llm->config.rope_theta = 10000.0f;
        }
    }

    if (n_tensors == 0) {
        fprintf(stderr, "[EIF TinyLLM Error] GGUF file has no model tensors (vocab-only container): %s\n", filename);
        munmap((void *)addr, file_sz);
        return -4;
    }

    gguf_tensor_desc_t *tdescs = (gguf_tensor_desc_t *)malloc((size_t)n_tensors * sizeof(gguf_tensor_desc_t));
    if (!tdescs) {
        munmap((void *)addr, file_sz);
        return -4;
    }

    for (uint64_t t = 0; t < n_tensors; t++) {
        gguf_read_str(&p, tdescs[t].name, sizeof(tdescs[t].name));
        uint32_t ndims = gguf_read_u32(&p);
        for (uint32_t d = 0; d < ndims; d++) {
            if (d < 4) tdescs[t].ne[d] = gguf_read_u64(&p);
            else gguf_read_u64(&p);
        }
        tdescs[t].n_dims = ndims;
        tdescs[t].type = gguf_read_u32(&p);
        tdescs[t].offset = gguf_read_u64(&p);
    }

    size_t header_len = (size_t)(p - addr);
    size_t data_start_offset = (header_len + alignment - 1) & ~(size_t)(alignment - 1);
    const uint8_t *data_start = addr + data_start_offset;

    uint32_t probe_type = 0;
    if (!find_gguf_tensor(tdescs, (int)n_tensors, data_start, "blk.0.attn_q.weight", &probe_type)) {
        if (!find_gguf_tensor(tdescs, (int)n_tensors, data_start, "blk.0.ffn_down.weight", &probe_type)) {
            // fallback: find the first block weight to determine quantization type
            for (uint64_t i = 0; i < n_tensors; i++) {
                if (strstr(tdescs[i].name, "blk.0.") && strstr(tdescs[i].name, ".weight")) {
                    probe_type = tdescs[i].type;
                    if (probe_type != 0 && probe_type != 1) break; // break if not F32/F16
                }
            }
        }
    }

    if (probe_type == GGUF_TYPE_TL1 || probe_type == GGUF_TYPE_TL2) {
        llm->config.qtype = TINYLLM_QTYPE_BITNET_158;
    } else if (probe_type == GGUF_TYPE_Q8_0 ||
               probe_type == GGUF_TYPE_Q4_0 || probe_type == GGUF_TYPE_Q4_1 ||
               probe_type == GGUF_TYPE_Q4_K || probe_type == GGUF_TYPE_Q6_K) {
        llm->config.qtype = TINYLLM_QTYPE_INT8; // Runtime maps these to INT8 processing
    } else {
        // Safe fallback for other types that might be treated generically
        llm->config.qtype = TINYLLM_QTYPE_INT8;
    }

    tinyllm_config_t *cp = &llm->config;
    int dim = cp->dim;
    int hidden_dim = cp->hidden_dim;
    int head_dim = (cp->head_dim > 0) ? cp->head_dim : (dim / cp->n_heads);
    int q_dim = cp->n_heads * head_dim;
    int kv_dim = cp->n_kv_heads * head_dim;
    int attn_scales_per_layer = (cp->head_dim > 0 && cp->head_dim != dim / cp->n_heads) ?
                                (q_dim + kv_dim * 2 + dim) :
                                (dim * 4 + kv_dim * 2);

    int dim_packed = (cp->qtype == TINYLLM_QTYPE_BITNET_158) ? (dim + 3) / 4 : (cp->qtype == TINYLLM_QTYPE_INT4 ? (dim + 1) / 2 : dim);
    int q_dim_packed = (cp->qtype == TINYLLM_QTYPE_BITNET_158) ? (q_dim + 3) / 4 : (cp->qtype == TINYLLM_QTYPE_INT4 ? (q_dim + 1) / 2 : q_dim);
    int hidden_packed = (cp->qtype == TINYLLM_QTYPE_BITNET_158) ? (hidden_dim + 3) / 4 : (cp->qtype == TINYLLM_QTYPE_INT4 ? (hidden_dim + 1) / 2 : hidden_dim);

    size_t weights_size = 0;
    if (cp->qtype == TINYLLM_QTYPE_BITNET_158) {
        weights_size = (size_t)cp->vocab_size * dim * sizeof(int8_t) +
                       (size_t)cp->n_layers * (
                           (size_t)q_dim * dim_packed +
                           (size_t)kv_dim * dim_packed * 2 +
                           (size_t)dim * q_dim_packed +
                           (size_t)hidden_dim * dim_packed +
                           (size_t)dim * hidden_packed +
                           (size_t)hidden_dim * dim_packed
                       ) +
                       (size_t)cp->vocab_size * dim * sizeof(int8_t);
    } else if (cp->qtype == TINYLLM_QTYPE_INT8) {
        weights_size = (size_t)cp->vocab_size * dim * sizeof(int8_t) +
                       (size_t)cp->n_layers * (
                           (size_t)q_dim * dim +
                           (size_t)kv_dim * dim * 2 +
                           (size_t)dim * q_dim +
                           (size_t)hidden_dim * dim +
                           (size_t)dim * hidden_dim +
                           (size_t)hidden_dim * dim
                       ) +
                       (size_t)cp->vocab_size * dim * sizeof(int8_t);
    }

    size_t scales_count = (size_t)cp->vocab_size +
                          (size_t)cp->n_layers * attn_scales_per_layer +
                          (size_t)cp->n_layers * (hidden_dim * 2 + dim) +
                          (size_t)cp->vocab_size;
    size_t scales_size = scales_count * sizeof(float);
    size_t rms_weights_size = (size_t)(2 * cp->n_layers + 1) * dim * sizeof(float);
    uint32_t b_type = 0;
    bool has_qkv_bias = (find_gguf_tensor(tdescs, (int)n_tensors, data_start, "blk.0.attn_q.bias", &b_type) != NULL);
    size_t bias_size = 0;
    if (has_qkv_bias) {
        bias_size = (size_t)cp->n_layers * (q_dim + kv_dim * 2) * sizeof(float);
    }
    size_t total_required = weights_size + scales_size + rms_weights_size + bias_size;

    if (buffer_size < total_required) {
        fprintf(stderr, "[EIF TinyLLM Error] GGUF buffer too small: need %zu, got %zu\n", total_required, buffer_size);
        free(tdescs);
        munmap((void *)addr, file_sz);
        return -5;
    }

    uint8_t *weights_ptr = (uint8_t *)quant_buffer;
    float *scales_ptr = (float *)(weights_ptr + weights_size);
    float *rms_ptr = (float *)((uint8_t *)scales_ptr + scales_size);
    float *bias_ptr = (float *)((uint8_t *)rms_ptr + rms_weights_size);

    llm->weights.rms_att_weight = rms_ptr;
    llm->weights.rms_ffn_weight = rms_ptr + cp->n_layers * dim;
    llm->weights.rms_final_weight = llm->weights.rms_ffn_weight + cp->n_layers * dim;

    if (has_qkv_bias) {
        llm->weights.bq = bias_ptr;
        llm->weights.bk = bias_ptr + (size_t)cp->n_layers * q_dim;
        llm->weights.bv = llm->weights.bk + (size_t)cp->n_layers * kv_dim;
    } else {
        llm->weights.bq = NULL;
        llm->weights.bk = NULL;
        llm->weights.bv = NULL;
    }

    tinyllm_setup_quantized_pointers(llm, weights_ptr, scales_ptr, weights_size);

    /* Load RMSNorm weights */
    for (int l = 0; l < cp->n_layers; l++) {
        char name[128];
        uint32_t ttype = 0;
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", l);
        const void *raw = find_gguf_tensor(tdescs, (int)n_tensors, data_start, name, &ttype);
        load_rmsnorm(llm->weights.rms_att_weight + l * dim, raw, ttype, dim);

        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
        raw = find_gguf_tensor(tdescs, (int)n_tensors, data_start, name, &ttype);
        load_rmsnorm(llm->weights.rms_ffn_weight + l * dim, raw, ttype, dim);
    }
    uint32_t ttype = 0;
    const void *raw_final = find_gguf_tensor(tdescs, (int)n_tensors, data_start, "output_norm.weight", &ttype);
    if (!raw_final) raw_final = find_gguf_tensor(tdescs, (int)n_tensors, data_start, "norm.weight", &ttype);
    load_rmsnorm(llm->weights.rms_final_weight, raw_final, ttype, dim);

    /* Load token embeddings */
    uint32_t emb_type = 0;
    const void *raw_emb = find_gguf_tensor(tdescs, (int)n_tensors, data_start, "token_embd.weight", &emb_type);
    if (raw_emb) {
        load_embedding_tensor((int8_t *)llm->weights.token_embedding, llm->weights.token_embedding_scale,
                              raw_emb, emb_type, cp->vocab_size, dim);
    }

    /* Load layer projections */
    for (int l = 0; l < cp->n_layers; l++) {
        char name[128];
        const void *raw = NULL;

        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
        raw = find_gguf_tensor(tdescs, (int)n_tensors, data_start, name, &ttype);
        load_projection_tensor((uint8_t *)llm->weights.wq + (size_t)l * q_dim * dim_packed,
                               llm->weights.attn_scale + (size_t)l * attn_scales_per_layer,
                               raw, ttype, q_dim, dim, cp->qtype);

        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
        raw = find_gguf_tensor(tdescs, (int)n_tensors, data_start, name, &ttype);
        load_projection_tensor((uint8_t *)llm->weights.wk + (size_t)l * kv_dim * dim_packed,
                               llm->weights.attn_scale + (size_t)l * attn_scales_per_layer + ((cp->head_dim > 0 && cp->head_dim != dim / cp->n_heads) ? q_dim : dim),
                               raw, ttype, kv_dim, dim, cp->qtype);

        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
        raw = find_gguf_tensor(tdescs, (int)n_tensors, data_start, name, &ttype);
        load_projection_tensor((uint8_t *)llm->weights.wv + (size_t)l * kv_dim * dim_packed,
                               llm->weights.attn_scale + (size_t)l * attn_scales_per_layer + ((cp->head_dim > 0 && cp->head_dim != dim / cp->n_heads) ? (q_dim + kv_dim) : (dim * 2)),
                               raw, ttype, kv_dim, dim, cp->qtype);

        if (llm->weights.bq) {
            snprintf(name, sizeof(name), "blk.%d.attn_q.bias", l);
            const void *raw_bq = find_gguf_tensor(tdescs, (int)n_tensors, data_start, name, &ttype);
            if (raw_bq) load_rmsnorm(llm->weights.bq + (size_t)l * q_dim, raw_bq, ttype, q_dim);
        }
        if (llm->weights.bk) {
            snprintf(name, sizeof(name), "blk.%d.attn_k.bias", l);
            const void *raw_bk = find_gguf_tensor(tdescs, (int)n_tensors, data_start, name, &ttype);
            if (raw_bk) load_rmsnorm(llm->weights.bk + (size_t)l * kv_dim, raw_bk, ttype, kv_dim);
        }
        if (llm->weights.bv) {
            snprintf(name, sizeof(name), "blk.%d.attn_v.bias", l);
            const void *raw_bv = find_gguf_tensor(tdescs, (int)n_tensors, data_start, name, &ttype);
            if (raw_bv) load_rmsnorm(llm->weights.bv + (size_t)l * kv_dim, raw_bv, ttype, kv_dim);
        }

        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
        raw = find_gguf_tensor(tdescs, (int)n_tensors, data_start, name, &ttype);
        if (!raw) {
            snprintf(name, sizeof(name), "blk.%d.attn_out.weight", l);
            raw = find_gguf_tensor(tdescs, (int)n_tensors, data_start, name, &ttype);
        }
        load_projection_tensor((uint8_t *)llm->weights.wo + (size_t)l * dim * q_dim_packed,
                               llm->weights.attn_scale + (size_t)l * attn_scales_per_layer + ((cp->head_dim > 0 && cp->head_dim != dim / cp->n_heads) ? (q_dim + kv_dim * 2) : (dim * 2 + kv_dim)),
                               raw, ttype, dim, q_dim, cp->qtype);

        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", l);
        raw = find_gguf_tensor(tdescs, (int)n_tensors, data_start, name, &ttype);
        load_projection_tensor((uint8_t *)llm->weights.w1 + (size_t)l * hidden_dim * dim_packed,
                               llm->weights.ffn_scale + (size_t)l * (hidden_dim * 2 + dim),
                               raw, ttype, hidden_dim, dim, cp->qtype);

        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", l);
        raw = find_gguf_tensor(tdescs, (int)n_tensors, data_start, name, &ttype);
        load_projection_tensor((uint8_t *)llm->weights.w2 + (size_t)l * dim * hidden_packed,
                               llm->weights.ffn_scale + (size_t)l * (hidden_dim * 2 + dim) + hidden_dim * 2,
                               raw, ttype, dim, hidden_dim, cp->qtype);

        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", l);
        raw = find_gguf_tensor(tdescs, (int)n_tensors, data_start, name, &ttype);
        load_projection_tensor((uint8_t *)llm->weights.w3 + (size_t)l * hidden_dim * dim_packed,
                               llm->weights.ffn_scale + (size_t)l * (hidden_dim * 2 + dim) + hidden_dim,
                               raw, ttype, hidden_dim, dim, cp->qtype);
    }

    /* Output classifier (lm_head) */
    uint32_t out_type = 0;
    const void *raw_out = find_gguf_tensor(tdescs, (int)n_tensors, data_start, "output.weight", &out_type);
    if (!raw_out) raw_out = find_gguf_tensor(tdescs, (int)n_tensors, data_start, "lm_head.weight", &out_type);
    if (raw_out) {
        load_embedding_tensor((int8_t *)llm->weights.wcls, llm->weights.wcls_scale,
                              raw_out, out_type, cp->vocab_size, dim);
    } else {
        llm->weights.wcls = llm->weights.token_embedding;
        llm->weights.wcls_scale = llm->weights.token_embedding_scale;
    }

    free(tdescs);
    munmap((void *)addr, file_sz);

    printf("GGUF Causal LLM loaded from %s\n", filename);
    printf("  Architecture: %s (%s), Layers: %d, Dim: %d, Vocab: %d\n",
           model_arch,
           cp->qtype == TINYLLM_QTYPE_BITNET_158 ? "BitNet b1.58" : "Dense",
           cp->n_layers, cp->dim, cp->vocab_size);

    return 0;
}

