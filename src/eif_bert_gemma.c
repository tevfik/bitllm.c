/**
 * @file eif_bert_gemma.c
 * @brief Gemma-embedding encoders (EmbeddingGemma 300m and 2): RMSNorm, NeoX RoPE, SWA, GQA, PLE.
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

/* RMS norm with plain weight multiply (llama.cpp convention: converters bake
 * the trained (1+w) gemma values into the stored weights at conversion time). */
static void gemma_rmsnorm(float *out, const float *in, const float *w, int n, float eps)
{
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += in[i] * in[i];
    float scale = 1.0f / sqrtf(ss / (float)n + eps);
    if (w) {
        for (int i = 0; i < n; i++) out[i] = in[i] * scale * w[i];
    } else {
        for (int i = 0; i < n; i++) out[i] = in[i] * scale;
    }
}

static void gemma_head_rmsnorm(float *p, const float *w /* head slice */, int hd, float eps)
{
    float ss = 0.0f;
    for (int d = 0; d < hd; d++) ss += p[d] * p[d];
    float scale = 1.0f / sqrtf(ss / (float)hd + eps);
    for (int d = 0; d < hd; d++) p[d] *= scale * w[d];
}

static void gemma_dump_tensor(const char *name, const float *data, int dim, int T)
{
    const char *dir = getenv("EIF_DUMP_DIR");
    if (!dir) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    int64_t ne[4] = { dim, T, 1, 1 };
    fwrite(ne, sizeof(ne), 1, f);
    fwrite(data, sizeof(float), (size_t)dim * T, f);
    fclose(f);
}

int eif_bert_embed_gemma(eif_bert_t *bert, const char *text, float *out_embedding)
{
    if (!bert || !bert->is_initialized || !text || !out_embedding) return -1;

    int max_tokens = bert_token_cap(bert);
    int32_t *tokens = (int32_t *)malloc((size_t)max_tokens * sizeof(int32_t));
    if (!tokens) return -1;
    int T = eif_bert_tokenize(bert, text, tokens, max_tokens);
    if (T <= 0) { free(tokens); return -2; }

    int dim = bert->config.dim_model > 0 ? bert->config.dim_model : bert->config.dim;
    int inter_dim = bert->config.intermediate_dim;
    int n_layers = bert->config.n_layers;
    int n_heads = bert->config.n_heads;
    int n_kv = bert->config.n_kv_heads > 0 ? bert->config.n_kv_heads : n_heads;
    int head_dim = bert->config.head_dim > 0 ? bert->config.head_dim : dim / n_heads;
    int kv_dim = n_kv * head_dim;
    int qtype = bert->config.qtype;
    int blocks_dim = dim / 32;
    float eps = bert->config.norm_eps > 0.0f ? bert->config.norm_eps : 1e-6f;
    float freq_base = bert->config.rope_freq_base > 0.0f ? bert->config.rope_freq_base : 1000000.0f;
    float emb_scale = bert->config.emb_scale > 0.0f ? bert->config.emb_scale : 1.0f;
    if (!bert->weights.pre_attn_norm_w || !bert->weights.attn_post_norm_w ||
        !bert->weights.pre_ffn_norm_w || !bert->weights.post_ffw_norm_w || !bert->weights.ffn_gate_w) {
        free(tokens);
        return -3;
    }
    float *X = bert->scratch_seq_x;
    float *XB = bert->scratch_seq_xb;
    float *Q = bert->scratch_q;
    float *K = bert->scratch_k;
    float *V = bert->scratch_v;
    float *ATT = bert->scratch_att;
    float *H_INTER = bert->scratch_inter;
    float *PROJ = bert->scratch_proj;

    /* Embedding layer: dequant + optional scale + RMS*(1+w) */
    for (int t = 0; t < T; t++) {
        int tid = tokens[t];
        float *x_t = X + t * dim;
        if (qtype == 2) {
            const uint8_t *w_row = (const uint8_t *)bert->weights.token_emb + (size_t)tid * (blocks_dim * 34);
            for (int b = 0; b < blocks_dim; b++) {
                const uint8_t *blk = w_row + b * 34;
                float scale = fp16_to_fp32(*(const uint16_t *)blk);
                const int8_t *wb = (const int8_t *)(blk + 2);
                for (int c = 0; c < 32; c++)
                    x_t[b * 32 + c] = (float)wb[c] * scale;
            }
        } else {
            memcpy(x_t, (const float *)bert->weights.token_emb + (size_t)tid * dim, (size_t)dim * sizeof(float));
        }
    }
    if (emb_scale != 1.0f) {
        for (int i = 0; i < T * dim; i++) X[i] *= emb_scale;
    }
    if (bert->weights.emb_norm_w) {
        for (int t = 0; t < T; t++)
            gemma_rmsnorm(X + t * dim, X + t * dim, bert->weights.emb_norm_w, dim, eps);
    }
    gemma_dump_tensor("inp_scaled", X, dim, T);

    /* gemma3-style layered rope: layers il%6==5 are GLOBAL (dense/full attn, global theta);
     * the rest are SWA layers with their own (usually smaller) theta. llama.cpp:
     * is_swa(il) ? rope_freq_base_train_swa : rope.freq_base; default swa base = 10000. */
    float freq_base_swa = 10000.0f;

    float inv_sqrt_head = bert->config.attn_scale > 0.0f
                               ? bert->config.attn_scale
                               : 1.0f / sqrtf((float)head_dim);
    int kv_group = n_heads / n_kv;

    for (int l = 0; l < n_layers; l++) {
        for (int t = 0; t < T; t++)
            gemma_rmsnorm(XB + (size_t)t * dim, X + (size_t)t * dim, bert->weights.pre_attn_norm_w[l], dim, eps);

        bert_gemm(Q, bert->weights.q_w[l],
                  bert->weights.q_w_scales ? bert->weights.q_w_scales[l] : NULL, XB,
                  bert->weights.q_b ? bert->weights.q_b[l] : NULL, T, dim, dim, qtype);
        bert_gemm(K, bert->weights.k_w[l],
                  bert->weights.k_w_scales ? bert->weights.k_w_scales[l] : NULL, XB,
                  bert->weights.k_b ? bert->weights.k_b[l] : NULL, T, kv_dim, dim, qtype);
        bert_gemm(V, bert->weights.v_w[l],
                  bert->weights.v_w_scales ? bert->weights.v_w_scales[l] : NULL, XB,
                  bert->weights.v_b ? bert->weights.v_b[l] : NULL, T, kv_dim, dim, qtype);

        /* per-head QK RMS *(1+w), then RoPE */
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < n_heads; h++) {
                float *qc = Q + (size_t)t * dim + (size_t)h * head_dim;
                if (bert->weights.q_norm_w[l])
                    gemma_head_rmsnorm(qc, bert->weights.q_norm_w[l], head_dim, eps);
            }
            for (int kvh = 0; kvh < n_kv; kvh++) {
                float *kc = K + (size_t)t * kv_dim + (size_t)kvh * head_dim;
                if (bert->weights.k_norm_w[l])
                    gemma_head_rmsnorm(kc, bert->weights.k_norm_w[l], head_dim, eps);
            }
        }
        if (l == 0) gemma_dump_tensor("Qcur_normed-0", Q, dim, T);
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < n_heads; h++)
                bert_rope_neox(Q + (size_t)t * dim + (size_t)h * head_dim, t, head_dim, l % 6 == 5 ? freq_base : freq_base_swa);
            for (int kvh = 0; kvh < n_kv; kvh++)
                bert_rope_neox(K + (size_t)t * kv_dim + (size_t)kvh * head_dim, t, head_dim, l % 6 == 5 ? freq_base : freq_base_swa);
        }
        if (l == 0) {
            gemma_dump_tensor("Qcur-0", Q, dim, T);
            gemma_dump_tensor("Kcur-0", K, kv_dim, T);
            gemma_dump_tensor("Vcur-0", V, kv_dim, T);
        }
        /* bidirectional full attention, KV heads shared across groups */
        const int swa_on = bert->config.sliding_window > 0 && (l % 6 != 5);
        const int swa_half = bert->config.sliding_window / 2;
        memset(XB, 0, (size_t)T * dim * sizeof(float));
        #pragma omp parallel for collapse(2) schedule(dynamic,1) if(T >= 32)
        for (int h = 0; h < n_heads; h++) {
            for (int i = 0; i < T; i++) {
                int kvh = h / kv_group;
                float *att_row = ATT + (size_t)h * T * T + (size_t)i * T;
                const float *q_i = Q + (size_t)i * dim + (size_t)h * head_dim;
                float max_val = -1e30f;
                for (int j = 0; j < T; j++) {
                    const float *k_j = K + (size_t)j * kv_dim + (size_t)kvh * head_dim;
                    float score = 0.0f;
#if defined(EIF_ARCH_X86_64)
                    __m256 sacc = _mm256_setzero_ps();
                    for (int d = 0; d <= head_dim - 8; d += 8)
                        sacc = _mm256_fmadd_ps(_mm256_loadu_ps(&q_i[d]), _mm256_loadu_ps(&k_j[d]), sacc);
                    score = hsum_float_8(sacc);
                    for (int d = (head_dim & ~7); d < head_dim; d++) score += q_i[d] * k_j[d];
#elif defined(EIF_ARCH_ARM64)
                    float32x4_t sacc = vdupq_n_f32(0.0f);
                    for (int d = 0; d <= head_dim - 4; d += 4)
                        sacc = vfmaq_f32(sacc, vld1q_f32(&q_i[d]), vld1q_f32(&k_j[d]));
                    score = vaddvq_f32(sacc);
                    for (int d = (head_dim & ~3); d < head_dim; d++) score += q_i[d] * k_j[d];
#else
                    for (int d = 0; d < head_dim; d++) score += q_i[d] * k_j[d];
#endif
                    score *= inv_sqrt_head;
                    /* sliding-window layers (5 of every 6): symmetric window, llama.cpp masks |i-j| > n_swa/2 */
                    if (swa_on && abs(i - j) > swa_half) { att_row[j] = -INFINITY; continue; }
                    att_row[j] = score;
                    if (score > max_val) max_val = score;
                }
                float sum_exp = 0.0f;
                for (int j = 0; j < T; j++) {
                    float ex = att_row[j] == -INFINITY ? 0.0f : expf(att_row[j] - max_val);
                    att_row[j] = ex;
                    sum_exp += ex;
                }
                float inv_sum = 1.0f / (sum_exp > 1e-12f ? sum_exp : 1.0f);
                for (int j = 0; j < T; j++) att_row[j] *= inv_sum;
            }
        }
        #pragma omp parallel for collapse(2) schedule(dynamic,1) if(T >= 32)
        for (int h = 0; h < n_heads; h++) {
            for (int i = 0; i < T; i++) {
                int kvh = h / kv_group;
                const float *att_row = ATT + (size_t)h * T * T + (size_t)i * T;
                float *c_h = XB + (size_t)i * dim + (size_t)h * head_dim;
                for (int j = 0; j < T; j++) {
                    float p_ij = att_row[j];
                    const float *v_j = V + (size_t)j * kv_dim + (size_t)kvh * head_dim;
#if defined(EIF_ARCH_X86_64)
                    __m256 vp = _mm256_set1_ps(p_ij);
                    for (int d = 0; d <= head_dim - 8; d += 8) {
                        _mm256_storeu_ps(&c_h[d], _mm256_fmadd_ps(vp, _mm256_loadu_ps(&v_j[d]), _mm256_loadu_ps(&c_h[d])));
                    }
                    for (int d = (head_dim & ~7); d < head_dim; d++) c_h[d] += p_ij * v_j[d];
#elif defined(EIF_ARCH_ARM64)
                    float32x4_t vp = vdupq_n_f32(p_ij);
                    for (int d = 0; d <= head_dim - 4; d += 4)
                        vst1q_f32(&c_h[d], vfmaq_f32(vld1q_f32(&c_h[d]), vp, vld1q_f32(&v_j[d])));
                    for (int d = (head_dim & ~3); d < head_dim; d++) c_h[d] += p_ij * v_j[d];
#else
                    for (int d = 0; d < head_dim; d++) c_h[d] += p_ij * v_j[d];
#endif
                }
            }
        }

        bert_gemm(PROJ, bert->weights.out_w[l], NULL, XB, NULL, T, dim, dim, qtype);
        /* post-attention norm on attn output, THEN residual (gemma sandwich) */
        for (int t = 0; t < T; t++)
            gemma_rmsnorm(PROJ + (size_t)t * dim, PROJ + (size_t)t * dim, bert->weights.attn_post_norm_w[l], dim, eps);
        if (l == 0) gemma_dump_tensor("attn_post_norm-0", PROJ, dim, T);
        for (int i = 0; i < T * dim; i++) X[i] += PROJ[i];
        if (l == 0) gemma_dump_tensor("sa_out-0", X, dim, T);

        /* GEGLU MLP: pre-FFN norm = ffn_norm; down( gelu(gate(x)) * up(x) ) */
        for (int t = 0; t < T; t++)
            gemma_rmsnorm(XB + (size_t)t * dim, X + (size_t)t * dim, bert->weights.pre_ffn_norm_w[l], dim, eps);
        if (l == 0) gemma_dump_tensor("ffn_norm-0", XB, dim, T);
        bert_gemm(H_INTER, bert->weights.ffn_up_w[l], NULL, XB, NULL, T, inter_dim, dim, qtype);
        float *H_GATE = H_INTER + (size_t)T * inter_dim;
        bert_gemm(H_GATE, bert->weights.ffn_gate_w[l], NULL, XB, NULL, T, inter_dim, dim, qtype);
        /* gemma-embedding uses exact erf gelu (LLM_FFN_GELU), not tanh approx */
        bert_gelu_exact(H_GATE, H_GATE, T * inter_dim);
        for (int i = 0; i < T * inter_dim; i++) H_INTER[i] *= H_GATE[i];

        bert_gemm(PROJ, bert->weights.ffn_down_w[l], NULL, H_INTER, NULL, T, dim, inter_dim, qtype);
        if (l == 0) gemma_dump_tensor("ffn_out-0", PROJ, dim, T);
        for (int t = 0; t < T; t++)
            gemma_rmsnorm(PROJ + (size_t)t * dim, PROJ + (size_t)t * dim, bert->weights.post_ffw_norm_w[l], dim, eps);
        for (int i = 0; i < T * dim; i++) X[i] += PROJ[i];
        if (l == 0) gemma_dump_tensor("l_out-0", X, dim, T);
        if (l == 1) gemma_dump_tensor("l_out-1", X, dim, T);
    }

    if (bert->weights.output_norm_w) {
        for (int t = 0; t < T; t++)
            gemma_rmsnorm(X + t * dim, X + t * dim, bert->weights.output_norm_w, dim, eps);
    }
    gemma_dump_tensor("result_norm", X, dim, T);

    /* Mean pooling (EmbeddingGemma uses mean over tokens) */
    memset(out_embedding, 0, (size_t)dim * sizeof(float));
    for (int t = 0; t < T; t++) {
        const float *x_t = X + t * dim;
        for (int d = 0; d < dim; d++) out_embedding[d] += x_t[d];
    }
    float inv_T = 1.0f / (float)T;
    float ss = 0.0f;
    for (int d = 0; d < dim; d++) {
        out_embedding[d] *= inv_T;
        ss += out_embedding[d] * out_embedding[d];
    }
    /* L2-normalize: same contract as llama.cpp's llama-embedding (unit vectors) */
    float inv_norm = ss > 1e-24f ? 1.0f / sqrtf(ss) : 0.0f;
    for (int d = 0; d < dim; d++) out_embedding[d] *= inv_norm;
    free(tokens);
    return 0;
}

/* EmbeddingGemma-2 (gemma-embedding2 arch): Gemma4 backbone with per-layer
 * embeddings (PLE), per-layer GQA head config (SWA layers: 4x256 heads / 2 KV,
 * rope base swa; global layers: 4x512 heads / 1 KV, rope base), and a final
 * projection to embedding_length_out. Faithful port of llama.cpp PR #30054
 * (src/models/gemma-embedding2.cpp). */
int eif_bert_embed_gemma2(eif_bert_t *bert, const char *text, float *out_embedding)
{
    if (!bert || !bert->is_initialized || !text || !out_embedding) return -1;
    if (!bert->weights.g2_pl_model_proj || !bert->weights.g2_pl_proj_norm ||
        !bert->weights.g2_inp_gate || !bert->weights.g2_out) return -3;

    int max_tokens = bert_token_cap(bert);
    int32_t *tokens = (int32_t *)malloc((size_t)max_tokens * sizeof(int32_t));
    if (!tokens) return -1;
    int T = eif_bert_tokenize(bert, text, tokens, max_tokens);
    if (T <= 0) { free(tokens); return -2; }
    if (T > max_tokens) T = max_tokens;
    { /* parity/probe override: EIF_FORCED_TOKENS="2,3823,..." */
        const char *ft = getenv("EIF_FORCED_TOKENS");
        if (ft && ft[0]) {
            int n = 0; const char *p = ft;
            while (*p && n < max_tokens) {
                char *e; long v = strtol(p, &e, 10);
                if (e == p) { if (*p == ',') p++; else break; continue; }
                tokens[n++] = (int32_t)v;
                p = e;
                while (*p == ',') p++;
            }
            if (n > 0) { T = n; fprintf(stderr, "eif_bert: forced %d tokens\n", n); }
        }
    }

    const int dim_m     = bert->config.dim_model;
    const int dim_out   = bert->config.dim;
    const int n_layers  = bert->config.n_layers;
    const int n_heads   = bert->config.n_heads;
    const int inter_dim = bert->config.intermediate_dim;
    const int qtype     = bert->config.qtype;
    int blocks_dim = dim_m / 32;
    const float eps   = bert->config.norm_eps > 0.0f ? bert->config.norm_eps : 1e-6f;
    const float freq_base     = bert->config.rope_freq_base > 0.0f ? bert->config.rope_freq_base : 1000000.0f;
    const float freq_base_swa = bert->config.rope_freq_base_swa > 0.0f ? bert->config.rope_freq_base_swa : 10000.0f;
    const float emb_scale     = bert->config.emb_scale > 0.0f ? bert->config.emb_scale : 1.0f;
    const int   swa_window    = bert->config.sliding_window > 0 ? bert->config.sliding_window : 1024;

    float *X       = bert->scratch_seq_x;   /* layer hidden state            [T x dim_m]   */
    float *XB      = bert->scratch_seq_xb;  /* attn_out                      [T x dim_m]   */
    float *Q       = bert->scratch_q;       /* per-layer Q                   [T x max_q]   */
    float *K       = bert->scratch_k;       /* per-layer K (kv_dim)          [T x dim_m]   */
    float *V       = bert->scratch_v;       /* per-layer V (kv_dim)          [T x dim_m]   */
    float *ATT     = bert->scratch_att;
    float *H_INTER = bert->scratch_inter;
    float *PROJ    = bert->scratch_proj;    /* [T x max(dim_m, dim_out)]     */
    float *G2      = bert->scratch_g2;      /* PLE gate scratch              [T x dim_m]   */
    float *PL      = bert->scratch_pl;      /* per-layer input table [T x dim_m * n_layers] */
    float *AO      = bert->scratch_attout;  /* attention output accum        [T x max_q]   */

    /* Token embeddings * sqrt(n_embd) (no input normalization in EG2) */
    for (int t = 0; t < T; t++) {
        int tid = tokens[t];
        float *x_t = X + t * dim_m;
        if (qtype == 2) {
            const uint8_t *w_row = (const uint8_t *)bert->weights.token_emb + (size_t)tid * (blocks_dim * 34);
            for (int b = 0; b < blocks_dim; b++) {
                const uint8_t *blk = w_row + b * 34;
                float scale = fp16_to_fp32(*(const uint16_t *)blk);
                const int8_t *wb = (const int8_t *)(blk + 2);
                for (int c = 0; c < 32; c++)
                    x_t[b * 32 + c] = (float)wb[c] * scale;
            }
        } else {
            memcpy(x_t, (const float *)bert->weights.token_emb + (size_t)tid * dim_m, (size_t)dim_m * sizeof(float));
        }
    }
    if (emb_scale != 1.0f) {
        for (int i = 0; i < T * dim_m; i++) X[i] *= emb_scale;
    }
    gemma_dump_tensor("inp_scaled", X, dim_m, T);

    /* Precompute per-layer inputs:
     * PL[t][il][d] = rms_norm(per_layer_proj_norm, (per_layer_model_proj @ X[t]) * 1/sqrt(n_embd))[il][d]
     * Token-major layout; RMS-norm over each 512-d chunk (llama.cpp: reshape
     * [n_embd_per_layer, n_layer, T] then rms_norm over ne[0]). Equivalent to
     * EmbeddingGemma2TextPLE in the reference python code. */
    for (int t = 0; t < T; t++)
        bert_gemm(PL + (size_t)t * (dim_m * n_layers), bert->weights.g2_pl_model_proj, NULL,
                  X + (size_t)t * dim_m, NULL, 1, dim_m * n_layers, dim_m, 0 /* dequantized FP32 weights */);
    {
        float inv_sqrt_e = 1.0f / sqrtf((float)dim_m);
        for (int i = 0; i < T * dim_m * n_layers; i++) PL[i] *= inv_sqrt_e;
    }
    for (int t = 0; t < T; t++)
        for (int il = 0; il < n_layers; il++)
            gemma_rmsnorm(PL + ((size_t)t * n_layers + il) * dim_m,
                          PL + ((size_t)t * n_layers + il) * dim_m,
                          bert->weights.g2_pl_proj_norm, dim_m, eps);
    if (getenv("EIF_DUMP_DIR")) { /* parity dump in llama [d + dim*l + dim*L*t] order */
        const char *dir = getenv("EIF_DUMP_DIR");
        char path[1024]; snprintf(path, sizeof(path), "%s/inp_per_layer.bin", dir);
        FILE *f = fopen(path, "wb");
        if (f) {
            int64_t ne[4] = { dim_m, n_layers, T, 1 };
            fwrite(ne, sizeof(ne), 1, f);
            float *buf = (float *)malloc(sizeof(float) * (size_t)dim_m * n_layers * T);
            for (int il = 0; il < n_layers; il++)
                for (int d = 0; d < dim_m; d++)
                    for (int t = 0; t < T; t++)
                        buf[(size_t)il * dim_m + (size_t)d + (size_t)dim_m * n_layers * t] =
                            PL[((size_t)t * n_layers + il) * dim_m + d];
            fwrite(buf, sizeof(float), (size_t)dim_m * n_layers * T, f);
            free(buf); fclose(f);
        }
    }

    /* Main backbone loop */
    for (int l = 0; l < n_layers; l++) {
        const int head_dim = bert->weights.g2_head_dim[l]; /* 256 SWA / 512 global */
        const int n_kv     = bert->weights.g2_n_kv[l];     /* 2 SWA   / 1 global  */
        const int kv_dim   = n_kv * head_dim;
        const int q_dim    = n_heads * head_dim;
        const int kv_group = n_heads / n_kv;
        const int is_global = (head_dim == bert->config.head_dim); /* il % 6 == 5 pattern */
        const float fbase = is_global ? freq_base : freq_base_swa;

        /* pre-attention norm */
        for (int t = 0; t < T; t++)
            gemma_rmsnorm(XB + (size_t)t * dim_m, X + (size_t)t * dim_m,
                          bert->weights.pre_attn_norm_w[l], dim_m, eps);
        if (l == 0) gemma_dump_tensor("attn_norm-0", XB, dim_m, T);

        bert_gemm(Q, bert->weights.q_w[l], NULL, XB, NULL, T, q_dim, dim_m, qtype);
        bert_gemm(K, bert->weights.k_w[l], NULL, XB, NULL, T, kv_dim, dim_m, qtype);
        bert_gemm(V, bert->weights.v_w[l], NULL, XB, NULL, T, kv_dim, dim_m, qtype);
        if (l == 0) {
            gemma_dump_tensor("Qcur-0", Q, q_dim, T);
            gemma_dump_tensor("Kcur-0", K, kv_dim, T);
            gemma_dump_tensor("Vcur-0", V, kv_dim, T);
        }

        /* QK RMS-norm (per head, with weights); V gets plain RMS (no weights) */
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < n_heads; h++) {
                float *qc = Q + (size_t)t * q_dim + (size_t)h * head_dim;
                gemma_head_rmsnorm(qc, bert->weights.q_norm_w[l], head_dim, eps);
            }
            for (int kvh = 0; kvh < n_kv; kvh++) {
                float *kc = K + (size_t)t * kv_dim + (size_t)kvh * head_dim;
                gemma_head_rmsnorm(kc, bert->weights.k_norm_w[l], head_dim, eps);
            }
        }
        /* V RMS-norm over each head slice + store back (separate loop, no weights) */
        for (int t = 0; t < T; t++)
            for (int kvh = 0; kvh < n_kv; kvh++)
                gemma_rmsnorm(V + (size_t)t * kv_dim + (size_t)kvh * head_dim,
                              V + (size_t)t * kv_dim + (size_t)kvh * head_dim, NULL, head_dim, eps);
        if (l == 0) gemma_dump_tensor("Qcur_normed-0", Q, q_dim, T);

        /* RoPE (NEOX half-split pairs; whole head rotated: n_rot == head_dim) */
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < n_heads; h++)
                bert_rope_neox(Q + (size_t)t * q_dim + (size_t)h * head_dim, t, head_dim, fbase);
            for (int kvh = 0; kvh < n_kv; kvh++)
                bert_rope_neox(K + (size_t)t * kv_dim + (size_t)kvh * head_dim, t, head_dim, fbase);
        }
        if (l == 0) gemma_dump_tensor("Qcur_pos-0", Q, q_dim, T);

        /* bidirectional attention (SWA layers use a symmetric 1024-token window);
         * output accumulates into AO ([T x q_dim] — global layers q_dim=2048
         * exceeds dim_model, so XB must NOT be used as the accumulation target) */
        memset(AO, 0, (size_t)T * q_dim * sizeof(float));
        memset(XB, 0, (size_t)T * dim_m * sizeof(float));
        #pragma omp parallel for collapse(2) schedule(dynamic,1) if(T >= 32)
        for (int h = 0; h < n_heads; h++) {
            for (int i = 0; i < T; i++) {
                int kvh = h / kv_group;
                float *att_row = ATT + (size_t)h * T * T + (size_t)i * T;
                const float *q_i = Q + (size_t)i * q_dim + (size_t)h * head_dim;
                float max_val = -1e30f;
                for (int j = 0; j < T; j++) {
                    const float *k_j = K + (size_t)j * kv_dim + (size_t)kvh * head_dim;
                    float score = 0.0f;
#if defined(EIF_ARCH_X86_64)
                    __m256 sacc = _mm256_setzero_ps();
                    for (int d = 0; d <= head_dim - 8; d += 8)
                        sacc = _mm256_fmadd_ps(_mm256_loadu_ps(&q_i[d]), _mm256_loadu_ps(&k_j[d]), sacc);
                    score = hsum_float_8(sacc);
                    for (int d = (head_dim & ~7); d < head_dim; d++) score += q_i[d] * k_j[d];
#elif defined(EIF_ARCH_ARM64)
                    float32x4_t sacc = vdupq_n_f32(0.0f);
                    for (int d = 0; d <= head_dim - 4; d += 4)
                        sacc = vfmaq_f32(sacc, vld1q_f32(&q_i[d]), vld1q_f32(&k_j[d]));
                    score = vaddvq_f32(sacc);
                    for (int d = (head_dim & ~3); d < head_dim; d++) score += q_i[d] * k_j[d];
#else
                    for (int d = 0; d < head_dim; d++) score += q_i[d] * k_j[d];
#endif
                    /* EG2: full attention scale (q_norm makes scaling unnecessary) */
                    if (!is_global && abs(i - j) > swa_window / 2) { att_row[j] = -INFINITY; continue; } /* SWA mask */
                    att_row[j] = score;
                    if (score > max_val) max_val = score;
                }
                float sum_exp = 0.0f;
                for (int j = 0; j < T; j++) {
                    float ex = att_row[j] == -INFINITY ? 0.0f : expf(att_row[j] - max_val);
                    att_row[j] = ex;
                    sum_exp += ex;
                }
                float inv_sum = 1.0f / (sum_exp > 1e-12f ? sum_exp : 1.0f);
                for (int j = 0; j < T; j++) att_row[j] *= inv_sum;
            }
        }
        #pragma omp parallel for collapse(2) schedule(dynamic,1) if(T >= 32)
        for (int h = 0; h < n_heads; h++) {
            for (int i = 0; i < T; i++) {
                int kvh = h / kv_group;
                const float *att_row = ATT + (size_t)h * T * T + (size_t)i * T;
                float *c_h = AO + (size_t)i * q_dim + (size_t)h * head_dim;
                for (int j = 0; j < T; j++) {
                    float p_ij = att_row[j];
                    if (p_ij == 0.0f) continue;
                    const float *v_j = V + (size_t)j * kv_dim + (size_t)kvh * head_dim;
#if defined(EIF_ARCH_X86_64)
                    __m256 vp = _mm256_set1_ps(p_ij);
                    for (int d = 0; d <= head_dim - 8; d += 8) {
                        _mm256_storeu_ps(&c_h[d], _mm256_fmadd_ps(vp, _mm256_loadu_ps(&v_j[d]), _mm256_loadu_ps(&c_h[d])));
                    }
                    for (int d = (head_dim & ~7); d < head_dim; d++) c_h[d] += p_ij * v_j[d];
#elif defined(EIF_ARCH_ARM64)
                    float32x4_t vp = vdupq_n_f32(p_ij);
                    for (int d = 0; d <= head_dim - 4; d += 4)
                        vst1q_f32(&c_h[d], vfmaq_f32(vld1q_f32(&c_h[d]), vp, vld1q_f32(&v_j[d])));
                    for (int d = (head_dim & ~3); d < head_dim; d++) c_h[d] += p_ij * v_j[d];
#else
                    for (int d = 0; d < head_dim; d++) c_h[d] += p_ij * v_j[d];
#endif
                }
            }
        }

        /* attn output + post-norm + residual -> attn_out (XB) */
        bert_gemm(PROJ, bert->weights.out_w[l], NULL, AO, NULL, T, dim_m, q_dim, qtype);
        if (l == 0) gemma_dump_tensor("sa_out-0", PROJ, dim_m, T);
        for (int t = 0; t < T; t++)
            gemma_rmsnorm(PROJ + (size_t)t * dim_m, PROJ + (size_t)t * dim_m,
                          bert->weights.attn_post_norm_w[l], dim_m, eps);
        if (l == 0) gemma_dump_tensor("attn_post_norm-0", PROJ, dim_m, T);
        for (int i = 0; i < T * dim_m; i++) XB[i] = X[i] + PROJ[i]; /* attn_out */
        if (l == 0) gemma_dump_tensor("attn_out-0", XB, dim_m, T);

        /* GEGLU FFN on norm(attn_out); residual = ffn_out * (1+norm) + attn_out */
        for (int t = 0; t < T; t++)
            gemma_rmsnorm(X + (size_t)t * dim_m, XB + (size_t)t * dim_m,
                          bert->weights.pre_ffn_norm_w[l], dim_m, eps);
        if (l == 0) gemma_dump_tensor("ffn_norm-0", X, dim_m, T);
        bert_gemm(H_INTER, bert->weights.ffn_up_w[l], NULL, X, NULL, T, inter_dim, dim_m, qtype);
        float *H_GATE = H_INTER + (size_t)T * inter_dim;
        bert_gemm(H_GATE, bert->weights.ffn_gate_w[l], NULL, X, NULL, T, inter_dim, dim_m, qtype);
        bert_gelu_exact(H_GATE, H_GATE, T * inter_dim);
        for (int i = 0; i < T * inter_dim; i++) H_INTER[i] *= H_GATE[i];
        bert_gemm(PROJ, bert->weights.ffn_down_w[l], NULL, H_INTER, NULL, T, dim_m, inter_dim, qtype);
        if (l == 0) gemma_dump_tensor("ffn_out-0", PROJ, dim_m, T);
        for (int t = 0; t < T; t++)
            gemma_rmsnorm(PROJ + (size_t)t * dim_m, PROJ + (size_t)t * dim_m,
                          bert->weights.post_ffw_norm_w[l], dim_m, eps);
        if (l == 0) gemma_dump_tensor("ffn_post_norm-0", PROJ, dim_m, T);
        for (int i = 0; i < T * dim_m; i++) X[i] = XB[i] + PROJ[i]; /* cur = attn_out + ffn */

        /* PLE block: gate(inp) gelu * inp_per_layer[l] -> proj -> post_norm -> + pe_in */
        bert_gemm(G2, bert->weights.g2_inp_gate[l], NULL, X, NULL, T, dim_m, dim_m, qtype);
        bert_gelu_exact(G2, G2, T * dim_m);
        for (int t = 0; t < T; t++) {
            const float *inpl = PL + ((size_t)t * n_layers + l) * dim_m;
            float *g = G2 + (size_t)t * dim_m;
            for (int d = 0; d < dim_m; d++) g[d] *= inpl[d];
        }
        bert_gemm(PROJ, bert->weights.g2_proj[l], NULL, G2, NULL, T, dim_m, dim_m, qtype);
        for (int t = 0; t < T; t++)
            gemma_rmsnorm(PROJ + (size_t)t * dim_m, PROJ + (size_t)t * dim_m,
                          bert->weights.g2_post_norm[l], dim_m, eps);
        if (l == 0) gemma_dump_tensor("per_layer_embd_out-0", PROJ, dim_m, T);
        float os = bert->weights.g2_out_scale[l][0]; /* layer_output_scale [1] */
        for (int i = 0; i < T * dim_m; i++) X[i] = (X[i] + PROJ[i]) * os;
        if (l == 0) gemma_dump_tensor("l_out-0", X, dim_m, T);
    }

    /* Final: RMS norm * output norm, project to 768, mean-pool, L2-normalize */
    for (int t = 0; t < T; t++)
        gemma_rmsnorm(X + (size_t)t * dim_m, X + (size_t)t * dim_m, bert->weights.output_norm_w, dim_m, eps);
    gemma_dump_tensor("result_norm", X, dim_m, T);
    bert_gemm(PROJ, bert->weights.g2_out, NULL, X, NULL, T, dim_out, dim_m, qtype);

    memset(out_embedding, 0, (size_t)dim_out * sizeof(float));
    for (int t = 0; t < T; t++) {
        const float *x_t = PROJ + t * dim_out;
        for (int d = 0; d < dim_out; d++) out_embedding[d] += x_t[d];
    }
    float inv_T = 1.0f / (float)T;
    float ss = 0.0f;
    for (int d = 0; d < dim_out; d++) out_embedding[d] *= inv_T;
    for (int d = 0; d < dim_out; d++) ss += out_embedding[d] * out_embedding[d];
    float inv_norm = ss > 1e-24f ? 1.0f / sqrtf(ss) : 0.0f;
    for (int d = 0; d < dim_out; d++) out_embedding[d] *= inv_norm;
    free(tokens);
    return 0;
}
