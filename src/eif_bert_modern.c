/**
 * @file eif_bert_modern.c
 * @brief ModernBERT encoder forward pass (granite-embedding-multilingual and other modern-bert GGUFs).
 *
 * Pre-norm layers without biases, fused QKV, NeoX RoPE with a global and a local base, local layers attend
 * within +-sliding_window/2 (llama.cpp's symmetric rule), layer 0 skips its attention norm, the FFN is a
 * fused up projection split into (activation input, gate) halves, and a final LayerNorm closes the stack.
 * Reference: llama.cpp src/models/modern-bert.cpp.
 */
#include "eif_bert_internal.h"
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Hidden states of all T tokens are left in bert->scratch_seq_x [T x dim]. */
int eif_bert_forward_modern(eif_bert_t *bert, const int32_t *tokens, int T)
{
    const int dim = bert->config.dim;
    const int ff = bert->config.intermediate_dim;
    const int n_heads = bert->config.n_heads;
    const int hd = dim / n_heads;
    const int blocks = dim / 32;
    const float eps = bert->config.norm_eps > 0.0f ? bert->config.norm_eps : 1e-5f;
    const float scale = 1.0f / sqrtf((float)hd);
    const int half_win = bert->config.sliding_window > 0 ? bert->config.sliding_window / 2 : 0;
    float *X = bert->scratch_seq_x, *XB = bert->scratch_seq_xb, *Q = bert->scratch_q, *K = bert->scratch_k, *V = bert->scratch_v;
    float *PROJ = bert->scratch_proj, *UP = bert->scratch_inter, *GATED = bert->scratch_inter + (size_t)T * 2 * ff;

#pragma omp parallel for schedule(dynamic, 8) if(T >= 32)
    for (int t = 0; t < T; t++) {
        const uint8_t *row = (const uint8_t *)bert->weights.token_emb + (size_t)tokens[t] * (size_t)blocks * 34;
        float *x = X + (size_t)t * dim;
        for (int b = 0; b < blocks; b++) {
            const uint8_t *blk = row + (size_t)b * 34;
            float sc = fp16_to_fp32(*(const uint16_t *)blk);
            const int8_t *w = (const int8_t *)(blk + 2);
            for (int c = 0; c < 32; c++) x[b * 32 + c] = (float)w[c] * sc;
        }
        bert_layernorm(x, x, bert->weights.emb_norm_w, NULL, dim, eps);
    }

    for (int l = 0; l < bert->config.n_layers; l++) {
        const int is_global = (l % bert->config.swa_pattern) == 0 || half_win == 0;
        const float base = is_global ? bert->config.rope_freq_base : bert->config.rope_freq_base_swa;
        const float *an = bert->weights.pre_attn_norm_w[l];
        const float *xin = X;
        if (an) {
#pragma omp parallel for schedule(dynamic, 8) if(T >= 32)
            for (int t = 0; t < T; t++) bert_layernorm(XB + (size_t)t * dim, X + (size_t)t * dim, an, NULL, dim, eps);
            xin = XB;
        }
        bert_gemm(Q, bert->weights.q_w[l], NULL, xin, NULL, T, dim, dim, 2);
        bert_gemm(K, bert->weights.k_w[l], NULL, xin, NULL, T, dim, dim, 2);
        bert_gemm(V, bert->weights.v_w[l], NULL, xin, NULL, T, dim, dim, 2);
#pragma omp parallel for schedule(dynamic, 8) if(T >= 32)
        for (int t = 0; t < T; t++)
            for (int h = 0; h < n_heads; h++) {
                bert_rope_neox(Q + (size_t)t * dim + (size_t)h * hd, t, hd, base);
                bert_rope_neox(K + (size_t)t * dim + (size_t)h * hd, t, hd, base);
            }

        /* attention: one task per (head, query row); local layers only visit |i-j| <= half_win */
#pragma omp parallel if((int64_t)n_heads * T >= 64)
        {
            float *sc = (float *)malloc((size_t)T * sizeof(float));
#pragma omp for schedule(dynamic, 4)
            for (int hi = 0; hi < n_heads * T; hi++) {
                const int h = hi / T, i = hi % T;
                const int j0 = is_global ? 0 : (i - half_win > 0 ? i - half_win : 0);
                const int j1 = is_global ? T - 1 : (i + half_win < T - 1 ? i + half_win : T - 1);
                const float *q = Q + (size_t)i * dim + (size_t)h * hd;
                float mx = -INFINITY;
                for (int j = j0; j <= j1; j++) {
                    const float *k = K + (size_t)j * dim + (size_t)h * hd;
                    float s = 0.0f;
                    for (int d = 0; d < hd; d++) s += q[d] * k[d];
                    s *= scale;
                    sc[j] = s;
                    if (s > mx) mx = s;
                }
                float sum = 0.0f;
                for (int j = j0; j <= j1; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
                const float inv = 1.0f / sum;
                float *o = XB + (size_t)i * dim + (size_t)h * hd;
                for (int d = 0; d < hd; d++) o[d] = 0.0f;
                for (int j = j0; j <= j1; j++) {
                    const float p = sc[j] * inv;
                    const float *v = V + (size_t)j * dim + (size_t)h * hd;
                    for (int d = 0; d < hd; d++) o[d] += p * v[d];
                }
            }
            free(sc);
        }
        bert_gemm(PROJ, bert->weights.out_w[l], NULL, XB, NULL, T, dim, dim, 2);
        for (size_t i = 0; i < (size_t)T * dim; i++) X[i] += PROJ[i];

        /* FFN: x += down(act(a) * g) with [a | g] = up(norm(x)) */
        const float *fn = bert->weights.pre_ffn_norm_w[l];
#pragma omp parallel for schedule(dynamic, 8) if(T >= 32)
        for (int t = 0; t < T; t++) bert_layernorm(XB + (size_t)t * dim, X + (size_t)t * dim, fn, NULL, dim, eps);
        bert_gemm(UP, bert->weights.ffn_up_w[l], NULL, XB, NULL, T, 2 * ff, dim, 2);
        const int silu = bert->config.ffn_silu;
#pragma omp parallel for schedule(static) if((int64_t)T * ff >= 4096)
        for (int t = 0; t < T; t++) {
            const float *a = UP + (size_t)t * 2 * ff, *g = a + ff;
            float *o = GATED + (size_t)t * ff;
            for (int i = 0; i < ff; i++) {
                float x = a[i];
                float act = silu ? x / (1.0f + expf(-x)) : 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
                o[i] = act * g[i];
            }
        }
        bert_gemm(PROJ, bert->weights.ffn_down_w[l], NULL, GATED, NULL, T, dim, ff, 2);
        for (size_t i = 0; i < (size_t)T * dim; i++) X[i] += PROJ[i];
    }

#pragma omp parallel for schedule(dynamic, 8) if(T >= 32)
    for (int t = 0; t < T; t++)
        bert_layernorm(X + (size_t)t * dim, X + (size_t)t * dim, bert->weights.output_norm_w, NULL, dim, eps);
    return 0;
}
