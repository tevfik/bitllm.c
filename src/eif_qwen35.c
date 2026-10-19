/**
 * @file eif_qwen35.c
 * @brief Qwen3.5-0.8B Hybrid LLM Inference Engine (Pure C99)
 *
 * Implements:
 *   - GatedDeltaNet recurrent kernel (O(1) per token, no KV cache growth)
 *   - Full GQA Attention with RoPE (partial_rotary_factor=0.25)
 *   - SwiGLU FFN
 *   - Vision ViT encoder + PatchMerger projector (FP32)
 *   - BitNet 1.58 ternary matrix-vector multiply
 *   - INT8 embedding lookup with per-row dequantization
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

#include "eif_qwen35.h"

#include <alloca.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(_OPENMP)
#include <omp.h>
#endif

/* ============================================================
 * Math utilities
 * ============================================================ */

static inline float silu(float x) { return x / (1.0f + expf(-x)); }
static inline float softplus(float x) { return logf(1.0f + expf(x)); }

static void rmsnorm(float *o, const float *x, const float *w, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i]*x[i];
    ss = 1.0f / sqrtf(ss / n + 1e-6f);
    for (int i = 0; i < n; i++) o[i] = w[i] * ss * x[i];
}

static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i]-mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

static void l2norm(float *x, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i]*x[i];
    float inv = 1.0f / sqrtf(ss + 1e-12f);
    for (int i = 0; i < n; i++) x[i] *= inv;
}

/* ============================================================
 * Multi-Quantization GEMV Suite
 * Supports:
 *   - BitNet 1.58 2-bit packed ternary (4 weights/byte)
 *   - BitNet Dense 1.60-bit base-3 ternary (5 weights/byte via 256-entry LUT)
 *   - Signed INT4 symmetric (2 weights/byte)
 *   - FP32 uncompressed
 * ============================================================ */

static float s_dense_lut_f[256][8] __attribute__((aligned(32)));
static int s_dense_lut_init = 0;

static void init_dense_lut(void) {
    if (s_dense_lut_init) return;
    for (int b = 0; b < 256; b++) {
        int temp = b;
        for (int i = 0; i < 5; i++) {
            int d = temp % 3;
            temp /= 3;
            s_dense_lut_f[b][i] = (d == 1) ? 1.0f : ((d == 2) ? -1.0f : 0.0f);
        }
        for (int i = 5; i < 8; i++) {
            s_dense_lut_f[b][i] = 0.0f;
        }
    }
    s_dense_lut_init = 1;
}

#include "eif_quantize_bitnet.h"

static inline size_t row_bytes(int cols, int quant_type) {
    if (quant_type == QWEN35_QTYPE_BITNET_DENSE) return (size_t)(cols + 4) / 5;
    if (quant_type == QWEN35_QTYPE_INT4)         return (size_t)(cols + 1) / 2;
    if (quant_type == QWEN35_QTYPE_FP32)         return (size_t)cols * sizeof(float);
    return (size_t)(cols + 3) / 4; /* QWEN35_QTYPE_BITNET_2BIT */
}

/* Byte count of one embedding row for the embedding storage types. */
static inline size_t embed_row_bytes(int cols, int eq) {
    if (eq == QWEN35_EMBED_INT8)  return (size_t)cols;
    if (eq == QWEN35_EMBED_INT4)  return (size_t)(cols + 1) / 2;
    if (eq == QWEN35_EMBED_2BIT)  return (size_t)(cols + 3) / 4;
    if (eq == QWEN35_EMBED_DENSE) return (size_t)(cols + 4) / 5;
    return (size_t)cols * sizeof(float); /* QWEN35_EMBED_FP32 */
}

/* Map embedding storage type to the weight quant type used by linear_gemv. */
static inline int embed_to_weight_qtype(int eq) {
    if (eq == QWEN35_EMBED_INT4)  return QWEN35_QTYPE_INT4;
    if (eq == QWEN35_EMBED_2BIT)  return QWEN35_QTYPE_BITNET_2BIT;
    if (eq == QWEN35_EMBED_DENSE) return QWEN35_QTYPE_BITNET_DENSE;
    return QWEN35_QTYPE_FP32;
}

static void bitnet_gemv_2bit(float *out, const uint8_t *W, const float *sc,
                             const float *x, int rows, int cols) {
    eif_matmul_bitnet_f32(W, sc, x, NULL, out, rows, cols);
}

static void bitnet_gemv_dense(float *out, const uint8_t *W, const float *sc,
                              const float *x, int rows, int cols) {
    init_dense_lut();
    int pcols = (cols + 4) / 5;
    int full_bytes = cols / 5;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const uint8_t *row = W + (size_t)r * pcols;
        float acc = 0.0f;
        int c = 0;
        int b = 0;
        for (; b < full_bytes; b++) {
            uint8_t byte = row[b];
            const float *lut = s_dense_lut_f[byte];
            acc += lut[0] * x[c]
                 + lut[1] * x[c + 1]
                 + lut[2] * x[c + 2]
                 + lut[3] * x[c + 3]
                 + lut[4] * x[c + 4];
            c += 5;
        }
        for (; b < pcols; b++) {
            uint8_t byte = row[b];
            const float *lut = s_dense_lut_f[byte];
            for (int p = 0; p < 5 && c + p < cols; p++) {
                acc += lut[p] * x[c + p];
            }
            c += 5;
        }
        out[r] = sc[r] * acc;
    }
}

static void int4_gemv(float *out, const uint8_t *W, const float *sc,
                      const float *x, int rows, int cols) {
    int pcols = (cols + 1) / 2;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const uint8_t *row = W + (size_t)r * pcols;
        float acc = 0.0f;
#if defined(__AVX2__)
        __m256 vacc0 = _mm256_setzero_ps();
        __m256 vacc1 = _mm256_setzero_ps();
        int c = 0;
        for (; c <= cols - 32; c += 32) {
            /* 16 packed bytes -> 32 signed nibbles (even col = high, odd = low) */
            __m128i b   = _mm_loadu_si128((const __m128i *)(row + (c >> 1)));
            __m256i b16 = _mm256_cvtepu8_epi16(b);
            __m256i hi  = _mm256_and_si256(_mm256_srli_epi16(b16, 4), _mm256_set1_epi16(0x0F));
            __m256i lo  = _mm256_and_si256(b16, _mm256_set1_epi16(0x0F));
            __m256i v_lo = _mm256_sub_epi16(_mm256_unpacklo_epi16(hi, lo), _mm256_set1_epi16(8));
            __m256i v_hi = _mm256_sub_epi16(_mm256_unpackhi_epi16(hi, lo), _mm256_set1_epi16(8));
            __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_lo)));   /* cols 0-7   */
            __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_hi)));   /* cols 8-15  */
            __m256 f2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(v_lo, 1))); /* 16-23 */
            __m256 f3 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(v_hi, 1))); /* 24-31 */
            vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + c),      f0, vacc0);
            vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + c + 8),  f1, vacc0);
            vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + c + 16), f2, vacc1);
            vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + c + 24), f3, vacc1);
        }
        __m256 s256 = _mm256_add_ps(vacc0, vacc1);
        __m128 l = _mm256_castps256_ps128(s256);
        __m128 h = _mm256_extractf128_ps(s256, 1);
        __m128 sm = _mm_add_ps(l, h);
        sm = _mm_hadd_ps(sm, sm); sm = _mm_hadd_ps(sm, sm);
        acc = _mm_cvtss_f32(sm);
        for (int cc = c; cc < cols; cc++) {
            uint8_t byte = row[cc >> 1];
            int8_t w = (cc & 1) ? (int8_t)(byte & 0x0F) - 8
                                : (int8_t)((byte >> 4) & 0x0F) - 8;
            acc += (float)w * x[cc];
        }
#else
        int c = 0;
        for (int b = 0; b < pcols; b++) {
            uint8_t byte = row[b];
            int8_t w0 = (int8_t)((byte >> 4) & 0x0F) - 8;
            int8_t w1 = (int8_t)(byte & 0x0F) - 8;
            if (c < cols) acc += (float)w0 * x[c];
            if (c + 1 < cols) acc += (float)w1 * x[c + 1];
            c += 2;
        }
#endif
        out[r] = sc[r] * acc;
    }
}

/* FP32 GEMV */
static void f32_gemv(float *out, const float *W, const float *b_vec,
                     const float *x, int rows, int cols) {
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const float *row = W + (size_t)r * cols;
        float acc = b_vec ? b_vec[r] : 0.0f;
#if defined(__AVX2__)
        __m256 vacc = _mm256_setzero_ps();
        int c = 0;
        for (; c <= cols - 8; c += 8) {
            __m256 wr = _mm256_loadu_ps(row + c);
            __m256 xr = _mm256_loadu_ps(x + c);
            vacc = _mm256_fmadd_ps(wr, xr, vacc);
        }
        __m128 l = _mm256_castps256_ps128(vacc);
        __m128 h = _mm256_extractf128_ps(vacc, 1);
        __m128 sm = _mm_add_ps(l, h);
        sm = _mm_hadd_ps(sm, sm);
        sm = _mm_hadd_ps(sm, sm);
        acc += _mm_cvtss_f32(sm);
        for (; c < cols; c++) acc += row[c] * x[c];
#else
        for (int c = 0; c < cols; c++) acc += row[c] * x[c];
#endif
        out[r] = acc;
    }
}

/* INT8 GEMV (LM head): signed per-row scaled integer weights. */
static void int8_gemv(float *out, const int8_t *W, const float *sc,
                      const float *x, int rows, int cols) {
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const int8_t *row = W + (size_t)r * cols;
#if defined(__AVX2__)
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        for (int i = 0; i <= cols - 16; i += 16) {
            __m128i r16 = _mm_loadu_si128((const __m128i *)(row + i));
            __m256i r32_0 = _mm256_cvtepi8_epi32(r16);
            __m256i r32_1 = _mm256_cvtepi8_epi32(_mm_srli_si128(r16, 8));
            __m256 f0 = _mm256_cvtepi32_ps(r32_0);
            __m256 f1 = _mm256_cvtepi32_ps(r32_1);
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), f0, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8), f1, acc1);
        }
        __m256 s256 = _mm256_add_ps(acc0, acc1);
        __m128 l = _mm256_castps256_ps128(s256);
        __m128 h = _mm256_extractf128_ps(s256, 1);
        __m128 sm = _mm_add_ps(l, h);
        sm = _mm_hadd_ps(sm, sm);
        sm = _mm_hadd_ps(sm, sm);
        float acc = _mm_cvtss_f32(sm);
        for (int i = (cols & ~15); i < cols; i++) acc += x[i] * (float)row[i];
#else
        float acc = 0.0f;
        for (int i = 0; i < cols; i++) acc += x[i] * (float)row[i];
#endif
        out[r] = acc * sc[r];
    }
}

/* Unified GEMV dispatcher */
static void linear_gemv(float *out, const void *W, const float *sc,
                        const float *x, int rows, int cols, int quant_type) {
    if (quant_type == QWEN35_QTYPE_BITNET_DENSE) {
        bitnet_gemv_dense(out, (const uint8_t *)W, sc, x, rows, cols);
    } else if (quant_type == QWEN35_QTYPE_INT4) {
        int4_gemv(out, (const uint8_t *)W, sc, x, rows, cols);
    } else if (quant_type == QWEN35_QTYPE_FP32) {
        f32_gemv(out, (const float *)W, NULL, x, rows, cols);
    } else {
        bitnet_gemv_2bit(out, (const uint8_t *)W, sc, x, rows, cols);
    }
}

/* Quantized embedding lookup (input token -> dequantized hidden row). */
static void embed_lookup(float *out, const uint8_t *T, const float *sc,
                         int tok, int dim, int eq) {
    const uint8_t *row = T + (size_t)tok * embed_row_bytes(dim, eq);
    float s = sc[tok];
    if (eq == QWEN35_EMBED_INT8) {
        for (int i = 0; i < dim; i++) out[i] = (float)((const int8_t *)row)[i] * s;
    } else if (eq == QWEN35_EMBED_INT4) {
        for (int i = 0; i < dim; i++) {
            uint8_t byte = row[i >> 1];
            int q = (i & 1) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
            out[i] = (float)(q - 8) * s;
        }
    } else if (eq == QWEN35_EMBED_2BIT) {
        for (int i = 0; i < dim; i++) {
            uint8_t code = (uint8_t)((row[i >> 2] >> ((i & 3) * 2)) & 0x03u);
            float w = (code == 1u) ? 1.0f : ((code == 2u) ? -1.0f : 0.0f);
            out[i] = w * s;
        }
    } else if (eq == QWEN35_EMBED_DENSE) {
        for (int i = 0; i < dim; i++) {
            uint8_t byte = row[i / 5];
            int temp = byte;
            int d = 0;
            for (int k = 0; k <= (i % 5); k++) { d = temp % 3; temp /= 3; }
            float w = (d == 1) ? 1.0f : ((d == 2) ? -1.0f : 0.0f);
            out[i] = w * s;
        }
    } else {
        const float *frow = (const float *)row;
        for (int i = 0; i < dim; i++) out[i] = frow[i];
    }
}

/* ============================================================
 * Partial RoPE  (partial_rotary_factor=0.25, rope_theta=1e7)
 * Only first n_rot dimensions of each head are rotated.
 * n_rot = head_dim * 0.25 = 256*0.25 = 64  ->  32 pairs
 * ============================================================ */
#define ROPE_THETA 10000000.0f
#define ROPE_PARTIAL 0.25f

static void rope_partial(float *x, int pos, int n_heads, int head_dim) {
    int n_rot = (int)(head_dim * ROPE_PARTIAL);   /* 64 */
    int half  = n_rot / 2;                         /* 32 */
    if (half > 128) half = 128;

    /* Inverse frequencies depend only on (i, n_rot): compute once, not per head. */
    static float s_inv[128];
    static int   s_inv_nrot = -1;
    if (s_inv_nrot != n_rot) {
        for (int i = 0; i < half; i++)
            s_inv[i] = 1.0f / powf(ROPE_THETA, 2.0f * i / (float)n_rot);
        s_inv_nrot = n_rot;
    }

    /* Rotation angles depend only on (pos, i): compute once for all heads. */
    float c[128], s[128];
    for (int i = 0; i < half; i++) {
        float theta = pos * s_inv[i];
        c[i] = cosf(theta);
        s[i] = sinf(theta);
    }

    for (int h = 0; h < n_heads; h++) {
        float *xh = x + h * head_dim;
        for (int i = 0; i < half; i++) {
            float x0 = xh[i], x1 = xh[i + half];
            xh[i]      = x0*c[i] - x1*s[i];
            xh[i+half] = x0*s[i] + x1*c[i];
        }
    }
}

/* ============================================================
 * GatedDeltaNet forward (single token, autoregressive)
 *
 * All buffers are allocated on the heap via the run state pool.
 * Uses stack-alloca for per-call scratch to avoid malloc overhead.
 * ============================================================ */
static void deltanet_fwd(
    float *out, const float *x,
    qwen35_delta_weights_t *w,
    qwen35_delta_state_t *st,
    const qwen35_config_t *cfg,
    float *scratch)   /* scratch: caller provides [conv_dim + 3*val_dim + 2*nv] */
{
    int dim      = cfg->dim;
    int nk       = cfg->lin_n_k_heads;   /* 16 */
    int kd       = cfg->lin_k_head_dim;  /* 128 */
    int nv2      = cfg->lin_n_v_heads;   /* 16 */
    int vd       = cfg->lin_v_head_dim;  /* 128 */
    int key_dim  = nk * kd;              /* 2048 */
    int val_dim  = nv2 * vd;             /* 2048 */
    int conv_dim = key_dim*2 + val_dim;  /* 6144 */
    int conv_k   = cfg->conv_k_size;     /* 4 */

    float *qkv_raw = scratch;
    float *z_buf   = scratch + conv_dim;
    float *b_buf   = scratch + conv_dim + val_dim;
    float *a_buf   = scratch + conv_dim + val_dim + nv2;
    float *o_buf   = scratch + conv_dim + val_dim + nv2*2;
    /* o_buf size = val_dim, then we need a retrieve scratch of vd */
    float *e_buf   = scratch + conv_dim + val_dim*2 + nv2*2;  /* [vd] */

    /* Projections */
    linear_gemv(qkv_raw, w->qkv,    w->qkv_s, x, conv_dim, dim, cfg->quant_lm);
    linear_gemv(z_buf,   w->z,      w->z_s,   x, val_dim,  dim, cfg->quant_lm);
    linear_gemv(b_buf,   w->proj_b, w->b_s,   x, nv2, dim,      cfg->quant_lm);
    linear_gemv(a_buf,   w->proj_a, w->a_s,   x, nv2, dim,      cfg->quant_lm);


    /* Causal Conv1d sliding window update */
    {
        float *cs = st->conv_state;         /* [conv_dim, conv_k] */
        const float *cw = w->conv;
        for (int c = 0; c < conv_dim; c++) {
            float *cs_row = cs + c * conv_k;
            for (int i = 0; i < conv_k-1; i++) cs_row[i] = cs_row[i+1];
            cs_row[conv_k-1] = qkv_raw[c];
            float acc = 0.0f;
            for (int i = 0; i < conv_k; i++) acc += cs_row[i] * cw[c*conv_k + i];
            qkv_raw[c] = silu(acc);
        }
    }

    /* Split Q/K/V */
    float *q = qkv_raw;                /* [key_dim] = first 2048 */
    float *k = qkv_raw + key_dim;      /* [key_dim] */
    float *v = qkv_raw + key_dim*2;    /* [val_dim] */

    /* L2-normalize Q and K per head, and scale Q by 1/sqrt(head_k_dim) */
    float q_scale = 1.0f / sqrtf((float)kd);
    for (int h = 0; h < nk; h++) {
        l2norm(q + h*kd, kd);
        for (int i = 0; i < kd; i++) q[h*kd + i] *= q_scale;
    }
    for (int h = 0; h < nk; h++) l2norm(k + h*kd, kd);

    /* Compute beta and g per v_head */
    for (int h = 0; h < nv2; h++) {
        b_buf[h] = 1.0f / (1.0f + expf(-b_buf[h]));          /* beta = sigmoid */
        float A  = w->a_exp[h];
        a_buf[h] = -A * softplus(a_buf[h] + w->dt_bias[h]);  /* g = -exp(A)*softplus(...) */
    }

    /* Gated Delta Rule update per head (parallelized across heads)
     * S[h]: [kd, vd]  stored row-major
     * e  = S k          (retrieve)
     * S  = exp(g)*S + beta * k^T (v - e)  (update)
     * o  = S q          (output)
     */
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < nv2; h++) {
        float *Sh = st->S + (size_t)h * kd * vd;
        float *kh = k + h * kd;
        float *qh = q + h * kd;
        float *vh = v + h * vd;
        float *oh = o_buf + h * vd;
        float decay = expf(a_buf[h]);   /* exp(g) */
        float bh    = b_buf[h];
        float *eh   = e_buf + h * vd;

        /* 1. Decay recurrent state: S = S * exp(g) */
        for (int i = 0; i < kd; i++) {
            for (int j = 0; j < vd; j++) {
                Sh[i*vd + j] *= decay;
            }
        }

        /* 2. Retrieve with decayed state: e = S^T k  [vd] */
        for (int j = 0; j < vd; j++) {
            float acc = 0.0f;
            for (int i = 0; i < kd; i++) acc += Sh[i*vd + j] * kh[i];
            eh[j] = acc;
        }

        /* 3. Update recurrent state: S = S + beta * outer(k, v - e) */
        for (int i = 0; i < kd; i++) {
            float ki = bh * kh[i];
            for (int j = 0; j < vd; j++) {
                Sh[i*vd + j] += ki * (vh[j] - eh[j]);
            }
        }

        /* 4. Attention output: o = S q  [vd] */
        for (int j = 0; j < vd; j++) {
            float acc = 0.0f;
            for (int i = 0; i < kd; i++) acc += Sh[i*vd + j] * qh[i];
            oh[j] = acc;
        }
    }


    /* Gated RMSNorm: out[h] = RMSNorm(o[h]) * silu(z[h]) * norm_w */
    for (int h = 0; h < nv2; h++) {
        float *oh = o_buf + h * vd;
        float *zh = z_buf + h * vd;
        float ss = 0.0f;
        for (int i = 0; i < vd; i++) ss += oh[i]*oh[i];
        float rms = 1.0f / sqrtf(ss/vd + 1e-6f);
        for (int i = 0; i < vd; i++) {
            float gz = silu(zh[i]);
            oh[i] = oh[i] * rms * w->norm_w[i] * gz;
        }
    }


    /* Output projection */
    linear_gemv(out, w->out, w->out_s, o_buf, dim, val_dim, cfg->quant_lm);
}

/* ============================================================
 * Full Attention forward (GQA + partial RoPE + output gate)
 * ============================================================ */
static void fullatt_fwd(
    float *out, const float *x,
    qwen35_attn_weights_t *w,
    qwen35_attn_state_t *st,
    int pos, const qwen35_config_t *cfg,
    float *q_buf, float *k_buf, float *v_buf, float *att_buf)
{
    int dim     = cfg->dim;
    int nh      = cfg->n_heads;       /* 8 */
    int nkv     = cfg->n_kv_heads;    /* 2 */
    int hd      = cfg->head_dim;      /* 256 */
    int q_dim   = nh  * hd;           /* 2048 */
    int qg_dim  = q_dim * 2;          /* 4096 (query + gate) */
    int kv_dim  = nkv * hd;           /* 512 */
    int ng      = nh / nkv;           /* 4  (GQA groups) */
    float scale = 1.0f / sqrtf((float)hd);

    float *qg_buf    = (float *)alloca(qg_dim * sizeof(float));
    float *gate_buf  = (float *)alloca(q_dim * sizeof(float));
    /* Attention output is q_dim wide (nh*hd), which is larger than the caller's
     * residual buffer (dim). Keep it in a dedicated q_dim scratch and only
     * project into `out` at the end. */
    float *attn_out  = (float *)alloca(q_dim * sizeof(float));

    linear_gemv(qg_buf, w->q, w->q_s, x, qg_dim,  dim, cfg->quant_lm);
    linear_gemv(k_buf,  w->k, w->k_s, x, kv_dim,  dim, cfg->quant_lm);
    linear_gemv(v_buf,  w->v, w->v_s, x, kv_dim,  dim, cfg->quant_lm);

    /* Unpack interleaved [q, gate] per head: each head has 2*hd = 512 elements */
    for (int h = 0; h < nh; h++) {
        const float *src = qg_buf + h * (hd * 2);
        memcpy(q_buf + h * hd, src, hd * sizeof(float));
        memcpy(gate_buf + h * hd, src + hd, hd * sizeof(float));
    }

    /* QK norms per head */
    for (int h = 0; h < nh;  h++) {
        float *qh = q_buf + h*hd;
        float ss = 0.0f;
        for (int i = 0; i < hd; i++) ss += qh[i]*qh[i];
        float inv = 1.0f / sqrtf(ss/hd + 1e-6f);
        for (int i = 0; i < hd; i++) qh[i] = qh[i]*inv*w->q_norm[i];
    }
    for (int h = 0; h < nkv; h++) {
        float *kh = k_buf + h*hd;
        float ss = 0.0f;
        for (int i = 0; i < hd; i++) ss += kh[i]*kh[i];
        float inv = 1.0f / sqrtf(ss/hd + 1e-6f);
        for (int i = 0; i < hd; i++) kh[i] = kh[i]*inv*w->k_norm[i];
    }


    /* Partial RoPE */
    rope_partial(q_buf, pos, nh,  hd);
    rope_partial(k_buf, pos, nkv, hd);

    /* Store in KV cache */
    memcpy(st->k_cache + (size_t)pos * kv_dim, k_buf, kv_dim * sizeof(float));
    memcpy(st->v_cache + (size_t)pos * kv_dim, v_buf, kv_dim * sizeof(float));

    /* Multi-head attention (GQA) */
    for (int h = 0; h < nh; h++) {
        const float *qh = q_buf + h*hd;
        int kv_h = h / ng;
        float *att = att_buf + h * (pos+1);
        for (int t = 0; t <= pos; t++) {
            const float *kt = st->k_cache + (size_t)t * kv_dim + kv_h * hd;
            float s = 0.0f;
            for (int i = 0; i < hd; i++) s += qh[i]*kt[i];
            att[t] = s * scale;
        }
        softmax(att, pos+1);
        float *oh = attn_out + h*hd;
        memset(oh, 0, hd * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            const float *vt = st->v_cache + (size_t)t * kv_dim + kv_h * hd;
            float a = att[t];
            for (int i = 0; i < hd; i++) oh[i] += a * vt[i];
        }
    }

    /* Output gate: attn_output = attn_output * sigmoid(gate) */
    for (int i = 0; i < q_dim; i++) {
        float g = 1.0f / (1.0f + expf(-gate_buf[i]));
        attn_out[i] *= g;
    }

    /* Output projection with SubLN */
    float ss_o = 0.0f;
    for (int i = 0; i < q_dim; i++) ss_o += attn_out[i] * attn_out[i];
    float inv_rms_o = 1.0f / sqrtf(ss_o / q_dim + 1e-6f);
    if (w->sub_norm) {
        for (int i = 0; i < q_dim; i++) q_buf[i] = attn_out[i] * inv_rms_o * w->sub_norm[i];
    } else {
        for (int i = 0; i < q_dim; i++) q_buf[i] = attn_out[i] * inv_rms_o;
    }
    linear_gemv(out, w->o, w->o_s, q_buf, dim, q_dim, cfg->quant_lm);
}

/* ============================================================
 * SwiGLU FFN
 * ============================================================ */
static void ffn_swiglu(float *out, const float *x,
                       qwen35_ffn_weights_t *w,
                       float *hb, float *hb2, int dim, int h_dim, int quant_type) {
    if (quant_type == QWEN35_QTYPE_BITNET_2BIT) {
        /* Fused gate+up in a single OpenMP region (same kernel, one fork/join). */
        eif_matmul_bitnet_f32_w1w3(hb, hb2, w->gate, w->up, w->gate_s, w->up_s,
                                   x, h_dim, dim);
    } else {
        linear_gemv(hb,  w->gate, w->gate_s, x, h_dim, dim, quant_type);
        linear_gemv(hb2, w->up,   w->up_s,   x, h_dim, dim, quant_type);
    }

    for (int i = 0; i < h_dim; i++) hb[i] = silu(hb[i]) * hb2[i];

    /* SubLN (RMSNorm) for BitLinear down_proj */
    float ss_d = 0.0f;
    for (int i = 0; i < h_dim; i++) ss_d += hb[i] * hb[i];
    float inv_rms_d = 1.0f / sqrtf(ss_d / h_dim + 1e-6f);
    if (w->sub_norm) {
        for (int i = 0; i < h_dim; i++) hb[i] = hb[i] * inv_rms_d * w->sub_norm[i];
    } else {
        for (int i = 0; i < h_dim; i++) hb[i] *= inv_rms_d;
    }

    linear_gemv(out, w->down, w->down_s, hb, dim, h_dim, quant_type);
}


/* ============================================================
 * Forward pass
 * ============================================================ */
float *qwen35_forward_ex(qwen35_t *m, int tok, int pos, bool compute_logits) {
    const qwen35_config_t *cfg = &m->config;
    qwen35_weights_t *w = &m->weights;
    qwen35_state_t   *s = &m->state;
    int dim    = cfg->dim;
    int n_lay  = cfg->n_layers;
    int nv2    = cfg->lin_n_v_heads;
    int kd     = cfg->lin_k_head_dim;
    int vd     = cfg->lin_v_head_dim;
    int key_dim = nv2 * kd;  /* NOTE: nk == nv for Qwen3.5, both 16 */
    int val_dim = nv2 * vd;
    int conv_dim = key_dim*2 + val_dim;

    /* Scratch for deltanet (stack alloc is fine for 6144+4096+32+32+2048 floats) */
    size_t dn_scratch_n = conv_dim + val_dim*3 + nv2*2;
    float *dn_scratch = (float *)alloca(dn_scratch_n * sizeof(float));

    if (pos < 0 || pos >= cfg->max_seq_len) return NULL;

    /* 1. Embed */
    embed_lookup(s->x, (const uint8_t *)w->embed, w->embed_s, tok, dim, cfg->quant_embed);

    int dn_idx = 0, fa_idx = 0;
    for (int l = 0; l < n_lay; l++) {
        rmsnorm(s->xb, s->x, w->rms_att[l], dim);

        if (m->layer_types[l] == QWEN35_LAYER_DELTANET) {
            deltanet_fwd(s->xb2, s->xb,
                         &w->delta[dn_idx], &s->delta[dn_idx], cfg, dn_scratch);
            dn_idx++;
        } else {
            /* use xb2 as output scratch, xb stays as input */
            memset(s->xb2, 0, dim * sizeof(float));
            fullatt_fwd(s->xb2, s->xb,
                        &w->full[fa_idx], &s->attn[fa_idx],
                        pos, cfg, s->q, s->k, s->v, s->att);
            fa_idx++;
        }

        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];

        rmsnorm(s->xb, s->x, w->rms_ffn[l], dim);
        ffn_swiglu(s->xb2, s->xb, &w->ffn[l], s->hb, s->hb2, dim, cfg->hidden_dim, cfg->quant_lm);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];
    }

    /* Final norm */
    rmsnorm(s->xb, s->x, w->rms_final, dim);

    if (!compute_logits) {
        s->pos = pos + 1;
        return s->xb;
    }

    /* LM head (tied with embed_tokens): dispatch on the embedding storage type. */
    const uint8_t *E  = (const uint8_t *)w->embed;
    const float   *Es = w->embed_s;
    int vocab = cfg->vocab_size;
    if (cfg->quant_embed == QWEN35_EMBED_INT8) {
        int8_gemv(s->logits, (const int8_t *)E, Es, s->xb, vocab, dim);
    } else {
        linear_gemv(s->logits, E, Es, s->xb, vocab, dim,
                    embed_to_weight_qtype(cfg->quant_embed));
    }

    s->pos = pos + 1;
    return s->logits;
}

float *qwen35_forward(qwen35_t *m, int tok, int pos) {
    return qwen35_forward_ex(m, tok, pos, true);
}

float *qwen35_forward_no_logits(qwen35_t *m, int tok, int pos) {
    return qwen35_forward_ex(m, tok, pos, false);
}

float *qwen35_get_hidden_state(qwen35_t *m) {
    return m ? m->state.xb : NULL;
}

/* ============================================================
 * Inject precomputed visual embeddings (from the Python bridge)
 * ============================================================ */
void qwen35_inject_visual(qwen35_t *m, const float *embs, int n_tok, int start_pos) {
    const qwen35_config_t *cfg = &m->config;
    qwen35_state_t *s = &m->state;
    int dim    = cfg->dim;
    int nv2    = cfg->lin_n_v_heads;
    int kd     = cfg->lin_k_head_dim;
    int vd     = cfg->lin_v_head_dim;
    int key_dim = nv2 * kd;
    int val_dim = nv2 * vd;
    int conv_dim = key_dim*2 + val_dim;
    size_t dn_scratch_n = conv_dim + val_dim*3 + nv2*2;
    float *dn_scratch = (float *)alloca(dn_scratch_n * sizeof(float));

    if (start_pos < 0 || n_tok < 0 || start_pos + n_tok > cfg->max_seq_len) {
        fprintf(stderr, "qwen35_inject_visual: %d tokens at %d exceed max_seq_len %d\n", n_tok, start_pos,
                cfg->max_seq_len);
        return;
    }
    for (int t = 0; t < n_tok; t++) {
        memcpy(s->x, embs + (size_t)t * dim, dim * sizeof(float));
        int dn_idx = 0, fa_idx = 0;
        for (int l = 0; l < cfg->n_layers; l++) {
            rmsnorm(s->xb, s->x, m->weights.rms_att[l], dim);
            if (m->layer_types[l] == QWEN35_LAYER_DELTANET) {
                deltanet_fwd(s->xb2, s->xb,
                             &m->weights.delta[dn_idx], &s->delta[dn_idx], cfg, dn_scratch);
                dn_idx++;
            } else {
                memset(s->xb2, 0, dim * sizeof(float));
                fullatt_fwd(s->xb2, s->xb,
                            &m->weights.full[fa_idx], &s->attn[fa_idx],
                            start_pos + t, cfg, s->q, s->k, s->v, s->att);
                fa_idx++;
            }
            for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];
            rmsnorm(s->xb, s->x, m->weights.rms_ffn[l], dim);
            ffn_swiglu(s->xb2, s->xb, &m->weights.ffn[l], s->hb, s->hb2, dim, cfg->hidden_dim, cfg->quant_lm);
            for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];
        }
    }
    s->pos = start_pos + n_tok;
}

/* ============================================================
 * Memory sizing
 * ============================================================ */
size_t qwen35_state_size(const qwen35_config_t *c) {
    int nh = c->n_heads, nkv = c->n_kv_heads, hd = c->head_dim;
    int nv = c->lin_n_v_heads, kd = c->lin_k_head_dim, vd = c->lin_v_head_dim;
    int conv = (c->lin_n_k_heads * kd * 2 + nv * vd);
    int q_dim  = nh  * hd;
    int kv_dim = nkv * hd;
    size_t sz = 0;
    sz += (size_t)c->dim         * sizeof(float) * 3;  /* x, xb, xb2 */
    sz += (size_t)(q_dim > conv ? q_dim : conv) * sizeof(float);  /* q */
    sz += (size_t)kv_dim         * sizeof(float);       /* k */
    sz += (size_t)nv * vd        * sizeof(float);       /* v */
    sz += (size_t)c->hidden_dim  * sizeof(float) * 2;  /* hb, hb2 */
    sz += (size_t)c->vocab_size  * sizeof(float);       /* logits */
    sz += (size_t)nh * c->max_seq_len * sizeof(float);  /* att */
    for (int i = 0; i < c->n_delta; i++) {
        sz += (size_t)nv * kd * vd * sizeof(float);     /* S */
        sz += (size_t)conv * c->conv_k_size * sizeof(float); /* conv_state */
    }
    for (int i = 0; i < c->n_full; i++) {
        sz += (size_t)c->max_seq_len * kv_dim * 2 * sizeof(float); /* k+v cache */
    }
    sz += 256 * 30;  /* alignment padding */
    return sz;
}

/* ============================================================
 * Load from EIF binary (mmap)
 * ============================================================ */
int qwen35_load(qwen35_t *m, const char *path, void *state_mem, size_t state_sz) {
    memset(m, 0, sizeof(qwen35_t));

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("qwen35_load"); return -1; }
    struct stat st; fstat(fd, &st);
    void *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) return -2;

    m->file_data = data;  m->file_size = st.st_size;  m->file_is_mmap = 1;

    const uint8_t *base = (const uint8_t *)data;
    const qwen35_file_header_t *hdr = (const qwen35_file_header_t *)base;

    if (hdr->magic != QWEN35_MAGIC || hdr->version != QWEN35_VERSION) {
        fprintf(stderr, "qwen35_load: bad magic/version\n"); return -3;
    }

    memcpy(&m->config, base + hdr->config_offset, sizeof(qwen35_config_t));
    /* Layer types: immediately after the 128-byte config block */
    const uint8_t *lt = base + hdr->config_offset + 128;
    memcpy(m->layer_types, lt, m->config.n_layers);

    const qwen35_config_t *cfg = &m->config;
    qwen35_weights_t *W = &m->weights;

    /* ---- Validate the (potentially untrusted) header before use ---- */
    {
        size_t fs = (size_t)m->file_size;
        const qwen35_config_t *c = cfg;
        if ((size_t)hdr->config_offset + 128 + (size_t)c->n_layers > fs ||
            (size_t)hdr->weights_offset > fs || (size_t)hdr->scales_offset > fs ||
            (size_t)hdr->norms_offset > fs) {
            fprintf(stderr, "qwen35_load: header region offsets out of file bounds\n");
            return -4;
        }
        if (c->dim <= 0 || c->vocab_size <= 0 || c->hidden_dim <= 0 ||
            c->n_heads <= 0 || c->head_dim <= 0 || c->n_layers <= 0 ||
            c->n_layers > 24 || c->n_delta < 0 || c->n_delta > 18 ||
            c->n_full < 0 || c->n_full > 6 ||
            (c->n_delta + c->n_full) != c->n_layers ||
            c->lin_n_k_heads <= 0 || c->lin_n_v_heads <= 0 ||
            c->lin_k_head_dim <= 0 || c->lin_v_head_dim <= 0 ||
            c->max_seq_len <= 0 || c->conv_k_size <= 0) {
            fprintf(stderr, "qwen35_load: invalid model configuration\n");
            return -5;
        }
    }

    int dim    = cfg->dim,    vocab = cfg->vocab_size;
    int hidden = cfg->hidden_dim;
    int nh = cfg->n_heads, nkv = cfg->n_kv_heads, hd = cfg->head_dim;
    int nk = cfg->lin_n_k_heads, kd = cfg->lin_k_head_dim;
    int nv = cfg->lin_n_v_heads, vd = cfg->lin_v_head_dim;
    int ck = cfg->conv_k_size;
    int n_dn = cfg->n_delta, n_fa = cfg->n_full;

    int key_dim  = nk * kd, val_dim = nv * vd, conv_dim = key_dim*2 + val_dim;
    int q_dim    = nh  * hd, kv_dim = nkv * hd;
    int qg_dim   = q_dim * 2; /* 4096: Query (2048) + Gate (2048) */

    const uint8_t *wp = base + hdr->weights_offset;

    /* Embed */
    W->embed = (const int8_t *)wp;
    wp += (size_t)vocab * embed_row_bytes(dim, cfg->quant_embed);

    /* Compute packed sizes */
#define PSZ(r, c) (row_bytes((c), cfg->quant_lm) * (size_t)(r))
    const uint8_t *dn_qkv = wp; wp += (size_t)n_dn * PSZ(conv_dim, dim);
    const uint8_t *dn_z   = wp; wp += (size_t)n_dn * PSZ(val_dim,  dim);
    const uint8_t *dn_out = wp; wp += (size_t)n_dn * PSZ(dim, val_dim);
    const uint8_t *dn_pb  = wp; wp += (size_t)n_dn * PSZ(nv, dim);
    const uint8_t *dn_pa  = wp; wp += (size_t)n_dn * PSZ(nv, dim);
    const uint8_t *fa_q   = wp; wp += (size_t)n_fa * PSZ(qg_dim, dim);
    const uint8_t *fa_k   = wp; wp += (size_t)n_fa * PSZ(kv_dim, dim);
    const uint8_t *fa_v   = wp; wp += (size_t)n_fa * PSZ(kv_dim, dim);
    const uint8_t *fa_o   = wp; wp += (size_t)n_fa * PSZ(dim, q_dim);
    const uint8_t *ff_g   = wp; wp += (size_t)24   * PSZ(hidden, dim);
    const uint8_t *ff_u   = wp; wp += (size_t)24   * PSZ(hidden, dim);
    const uint8_t *ff_d   = wp; wp += (size_t)24   * PSZ(dim, hidden);
    const float   *dn_cv  = (const float *)wp; wp += (size_t)n_dn * conv_dim * ck * 4;
    const float   *dn_dt  = (const float *)wp; wp += (size_t)n_dn * nv * 4;
    const float   *dn_al  = (const float *)wp;
#undef PSZ

    /* Assign per-layer pointers */
    int di = 0, fi = 0;
    for (int l = 0; l < cfg->n_layers; l++) {
        W->ffn[l].gate   = ff_g + (size_t)l * (row_bytes(dim, cfg->quant_lm) * (size_t)hidden);
        W->ffn[l].up     = ff_u + (size_t)l * (row_bytes(dim, cfg->quant_lm) * (size_t)hidden);
        W->ffn[l].down   = ff_d + (size_t)l * (row_bytes(hidden, cfg->quant_lm) * (size_t)dim);
        if (m->layer_types[l] == QWEN35_LAYER_DELTANET) {
            W->delta[di].qkv    = dn_qkv + (size_t)di * (row_bytes(dim, cfg->quant_lm) * (size_t)conv_dim);
            W->delta[di].z      = dn_z   + (size_t)di * (row_bytes(dim, cfg->quant_lm) * (size_t)val_dim);
            W->delta[di].out    = dn_out + (size_t)di * (row_bytes(val_dim, cfg->quant_lm) * (size_t)dim);
            W->delta[di].proj_b = dn_pb  + (size_t)di * (row_bytes(dim, cfg->quant_lm) * (size_t)nv);
            W->delta[di].proj_a = dn_pa  + (size_t)di * (row_bytes(dim, cfg->quant_lm) * (size_t)nv);
            W->delta[di].conv   = dn_cv  + (size_t)di * conv_dim * ck;
            W->delta[di].dt_bias= dn_dt  + (size_t)di * nv;
            W->delta[di].A_log  = dn_al  + (size_t)di * nv;
            for (int h = 0; h < nv; h++) W->delta[di].a_exp[h] = expf(W->delta[di].A_log[h]);
            di++;
        } else {
            W->full[fi].q    = fa_q + (size_t)fi * (row_bytes(dim, cfg->quant_lm) * (size_t)qg_dim);
            W->full[fi].k    = fa_k + (size_t)fi * (row_bytes(dim, cfg->quant_lm) * (size_t)kv_dim);
            W->full[fi].v    = fa_v + (size_t)fi * (row_bytes(dim, cfg->quant_lm) * (size_t)kv_dim);
            W->full[fi].o    = fa_o + (size_t)fi * (row_bytes(q_dim, cfg->quant_lm) * (size_t)dim);
            fi++;
        }
    }

    /* Scales */
    const float *sc = (const float *)(base + hdr->scales_offset);
    W->embed_s = sc; sc += vocab;
    for (int i = 0; i < n_dn; i++) {
        W->delta[i].qkv_s = sc; sc += conv_dim;
        W->delta[i].z_s   = sc; sc += val_dim;
        W->delta[i].out_s = sc; sc += dim;
        W->delta[i].b_s   = sc; sc += nv;
        W->delta[i].a_s   = sc; sc += nv;
    }
    for (int i = 0; i < n_fa; i++) {
        W->full[i].q_s    = sc; sc += qg_dim;
        W->full[i].k_s    = sc; sc += kv_dim;
        W->full[i].v_s    = sc; sc += kv_dim;
        W->full[i].o_s    = sc; sc += dim;
    }
    for (int l = 0; l < 24; l++) {
        W->ffn[l].gate_s  = sc; sc += hidden;
        W->ffn[l].up_s    = sc; sc += hidden;
        W->ffn[l].down_s  = sc; sc += dim;
    }

    /* Norms */
    const float *np = (const float *)(base + hdr->norms_offset);
    for (int l = 0; l < 24; l++) { W->rms_att[l] = np; np += dim; }
    for (int l = 0; l < 24; l++) { W->rms_ffn[l] = np; np += dim; }
    W->rms_final = np; np += dim;
    /* GatedSSM norms + QK norms */
    di = 0; fi = 0;
    for (int l = 0; l < cfg->n_layers; l++) {
        if (m->layer_types[l] == QWEN35_LAYER_DELTANET) {
            W->delta[di].norm_w = np; np += vd; di++;
        } else {
            W->full[fi].q_norm = np; np += hd;
            W->full[fi].k_norm = np; np += hd; fi++;
        }
    }

    /* SubLN scales (if serialized in norms buffer) */
    const uint8_t *norms_end = base + hdr->norms_offset + hdr->norms_size;
    fi = 0;
    for (int l = 0; l < cfg->n_layers; l++) {
        if (m->layer_types[l] == QWEN35_LAYER_FULLATT) {
            if ((const uint8_t *)(np + q_dim) <= norms_end) {
                W->full[fi].sub_norm = np; np += q_dim;
            } else {
                W->full[fi].sub_norm = NULL;
            }
            fi++;
        }
        if ((const uint8_t *)(np + hidden) <= norms_end) {
            W->ffn[l].sub_norm = np; np += hidden;
        } else {
            W->ffn[l].sub_norm = NULL;
        }
    }

    /* State pool */
    if (state_sz < qwen35_state_size(cfg)) {
        fprintf(stderr, "State buffer too small\n"); return -4;
    }
    uint8_t *pool = (uint8_t *)(((uintptr_t)state_mem + 63) & ~(uintptr_t)63);
    m->state.pool = pool; m->state.pool_size = state_sz;

#define A(n) ((float *)pool); pool += ((size_t)(n)*sizeof(float)+63)&~63
    m->state.x      = A(dim);
    m->state.xb     = A(dim);
    m->state.xb2    = A(dim);
    m->state.q      = A(q_dim > conv_dim ? q_dim : conv_dim);
    m->state.k      = A(kv_dim);
    m->state.v      = A(nv * vd);
    m->state.hb     = A(hidden);
    m->state.hb2    = A(hidden);
    m->state.logits = A(vocab);
    m->state.att    = A(nh * cfg->max_seq_len);
    for (int i = 0; i < n_dn; i++) {
        m->state.delta[i].S          = A(nv * kd * vd);
        m->state.delta[i].conv_state = A(conv_dim * ck);
    }
    for (int i = 0; i < n_fa; i++) {
        m->state.attn[i].k_cache = A(cfg->max_seq_len * kv_dim);
        m->state.attn[i].v_cache = A(cfg->max_seq_len * kv_dim);
    }
#undef A

    qwen35_reset(m);
    printf("qwen35: loaded  dim=%d layers=%d(%dDN+%dFA) vocab=%d  seq=%d  lm_q=%d vis_q=%d emb_q=%d\n",
           dim, cfg->n_layers, n_dn, n_fa, vocab, cfg->max_seq_len,
           cfg->quant_lm, cfg->quant_vision, cfg->quant_embed);
    return 0;
}

void qwen35_reset(qwen35_t *m) {
    const qwen35_config_t *c = &m->config;
    int nv = c->lin_n_v_heads, kd = c->lin_k_head_dim, vd = c->lin_v_head_dim;
    int conv = c->lin_n_k_heads * kd * 2 + nv * vd;
    for (int i = 0; i < c->n_delta; i++) {
        memset(m->state.delta[i].S,          0, (size_t)nv*kd*vd*sizeof(float));
        memset(m->state.delta[i].conv_state,  0, (size_t)conv*c->conv_k_size*sizeof(float));
    }
    for (int i = 0; i < c->n_full; i++) {
        int kv_dim = c->n_kv_heads * c->head_dim;
        memset(m->state.attn[i].k_cache, 0, (size_t)c->max_seq_len * kv_dim * sizeof(float));
        memset(m->state.attn[i].v_cache, 0, (size_t)c->max_seq_len * kv_dim * sizeof(float));
    }
    m->state.pos = 0;
}

void qwen35_free(qwen35_t *m) {
    if (m->file_is_mmap && m->file_data)
        munmap(m->file_data, m->file_size);
    memset(m, 0, sizeof(qwen35_t));
}

/* ============================================================
 * Generation
 * ============================================================ */
int qwen35_generate(qwen35_t *m,
                    const int *prompt, int n_prompt,
                    const float *vis_emb, int n_vis,
                    int *out, int max_new, float rep_pen, int eos_id) {
    qwen35_reset(m);
    int pos = 0;

    if (vis_emb && n_vis > 0) { qwen35_inject_visual(m, vis_emb, n_vis, 0); pos = n_vis; }

    for (int i = 0; i < n_prompt - 1; i++) {
        qwen35_forward(m, prompt[i], pos++);
    }

    int tok = n_prompt > 0 ? prompt[n_prompt-1] : eos_id;
    int n = 0;
    for (int step = 0; step < max_new; step++) {
        float *lg = qwen35_forward(m, tok, pos++);

        if (rep_pen > 1.0f) {
            for (int i = 0; i < n; i++) {
                if (lg[out[i]] > 0) lg[out[i]] /= rep_pen;
                else                lg[out[i]] *= rep_pen;
            }
        }

        int next = 0; float best = lg[0];
        for (int v = 1; v < m->config.vocab_size; v++)
            if (lg[v] > best) { best = lg[v]; next = v; }

        if (next == eos_id) break;
        out[n++] = next;
        tok = next;
    }
    return n;
}
