/**
 * @file eif_bert_gpu.c
 * @brief Optional Vulkan offload of the encoder (BITLLM_BERT_GPU=1).
 *
 * Two levels, both falling back to the CPU kernels when the model is not attached, the sequence is short
 * (BITLLM_BERT_GPU_MIN_T, default 24 tokens for 768+ wide models, 64 otherwise), or anything fails on the GPU:
 *  - bert_gpu_gemm: one Q8_0 GEMM of bert_gemm on the GPU (weights repacked and uploaded on first use);
 *  - bert_gpu_layers: all transformer layers of a plain post-LN BERT/XLM-R (MiniLM, bge, e5, arctic, roberta,
 *    bge-m3, rerankers) in one command buffer: QKV, attention, projections, LayerNorm, GELU FFN.
 * Embedding lookup, pooling and the other architectures stay on the CPU. Not thread-safe.
 */
#include "eif_bert_internal.h"
#include "eif_gpu_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const uint8_t *base;
    size_t size;
    eif_gpu_buf_t *params; /* biases and norm vectors of every layer, FP32 (built on first use) */
    size_t params_floats;
} range_t;
typedef struct { const void *ptr; int N, K; eif_gpu_weight_t *w; } entry_t;

static eif_gpu_context_t g_ctx;
static int g_state = -1; /* -1 not tried, 0 unavailable, 1 ready */
static range_t *g_ranges; static int g_nranges;
static entry_t *g_entries; static int g_nentries, g_cap_entries;
static int g_min_t; /* BITLLM_BERT_GPU_MIN_T; 0 = by model width */

/* Measured on a GB10: the GPU wins from ~20 tokens for 1024-wide models and ~70 for 384-wide ones. */
static int too_short(int T, int dim) { return T < (g_min_t > 0 ? g_min_t : dim >= 768 ? 24 : 64); }
/* activations, grown on demand */
static eif_gpu_buf_t *g_x, *g_qkv, *g_att, *g_inter, *g_scores;
static size_t g_cap_x, g_cap_inter, g_scores_bytes; /* capacities in bytes: x/att hold T*dim floats, inter T*ff */
static unsigned long g_forwards;

static int env_enabled(void)
{
    const char *e = getenv("BITLLM_BERT_GPU");
    return e && *e && *e != '0';
}

static int gpu_ready(void)
{
    if (g_state >= 0) return g_state;
    if (!env_enabled()) return 0; /* not cached: the variable may be set before a later load */
    g_state = 0;
    const char *mt = getenv("BITLLM_BERT_GPU_MIN_T");
    if (mt && atoi(mt) > 0) g_min_t = atoi(mt);
    memset(&g_ctx, 0, sizeof(g_ctx));
    if (eif_gpu_init(&g_ctx)) g_state = 1;
    else fprintf(stderr, "eif_bert: BITLLM_BERT_GPU set but no usable Vulkan device, using the CPU\n");
    return g_state;
}

static range_t *find_range(const void *p)
{
    for (int i = 0; i < g_nranges; i++)
        if ((const uint8_t *)p >= g_ranges[i].base && (const uint8_t *)p < g_ranges[i].base + g_ranges[i].size)
            return &g_ranges[i];
    return NULL;
}

void eif_bert_gpu_attach(eif_bert_t *bert)
{
    if (!env_enabled() || !bert->is_mmap || !bert->mmap_addr || bert->config.qtype != 2 || !gpu_ready()) return; /* the variable is read per load: models loaded without it stay on the CPU */
    range_t *r = (range_t *)realloc(g_ranges, (size_t)(g_nranges + 1) * sizeof(*r));
    if (!r) return;
    g_ranges = r;
    memset(&g_ranges[g_nranges], 0, sizeof(*g_ranges));
    g_ranges[g_nranges].base = (const uint8_t *)bert->mmap_addr;
    g_ranges[g_nranges].size = bert->mmap_size;
    g_nranges++;
}

void eif_bert_gpu_detach(eif_bert_t *bert)
{
    if (!bert->mmap_addr || g_state != 1) return;
    const uint8_t *lo = (const uint8_t *)bert->mmap_addr, *hi = lo + bert->mmap_size;
    int n = 0;
    for (int i = 0; i < g_nentries; i++) {
        if ((const uint8_t *)g_entries[i].ptr >= lo && (const uint8_t *)g_entries[i].ptr < hi)
            eif_gpu_weight_free(&g_ctx, g_entries[i].w);
        else g_entries[n++] = g_entries[i];
    }
    g_nentries = n;
    n = 0;
    for (int i = 0; i < g_nranges; i++) {
        if (g_ranges[i].base == lo) eif_gpu_buf_destroy(&g_ctx, g_ranges[i].params);
        else g_ranges[n++] = g_ranges[i];
    }
    g_nranges = n;
}

static eif_gpu_weight_t *get_weight(const void *W, int rows, int cols)
{
    for (int i = 0; i < g_nentries; i++)
        if (g_entries[i].ptr == W && g_entries[i].N == rows && g_entries[i].K == cols) return g_entries[i].w;
    if (g_nentries == g_cap_entries) {
        int cap = g_cap_entries ? g_cap_entries * 2 : 64;
        entry_t *e = (entry_t *)realloc(g_entries, (size_t)cap * sizeof(*e));
        if (!e) return NULL;
        g_entries = e; g_cap_entries = cap;
    }
    eif_gpu_weight_t *w = eif_gpu_weight_upload_q8_0(&g_ctx, (const uint8_t *)W, rows, cols);
    if (w) g_entries[g_nentries++] = (entry_t){W, rows, cols, w};
    return w;
}

int bert_gpu_gemm(float *Y, const void *W, const float *X, const float *bias, int T, int rows, int cols)
{
    if (g_state != 1 || too_short(T, cols) || cols % 32 != 0 || !find_range(W)) return 0;
    eif_gpu_weight_t *w = get_weight(W, rows, cols);
    return w && eif_gpu_gemm(&g_ctx, X, w, bias, Y, T, 0, false) ? 1 : 0;
}

/* ---- all layers in one command buffer ---------------------------------------------------------- */

static int plain_bert(const eif_bert_t *b)
{
    const eif_bert_config_t *c = &b->config;
    const eif_bert_weights_t *w = &b->weights;
    return c->qtype == 2 && !c->arch_modern && !c->use_rope && !w->q_norm_w && !w->ffn_gate_w && !w->att_norm2_w &&
           w->pos_emb && c->dim % 32 == 0 && c->intermediate_dim % 32 == 0 && c->dim % c->n_heads == 0 &&
           !c->use_alibi;
}

/* biases and norms of layer l start at l * per_layer floats: q,k,v,o biases, up bias, down bias, 4 norm vectors */
static size_t layer_floats(const eif_bert_t *b) { return (size_t)9 * b->config.dim + b->config.intermediate_dim; }

static eif_gpu_buf_t *build_params(const eif_bert_t *b)
{
    const int dim = b->config.dim, ff = b->config.intermediate_dim;
    const eif_bert_weights_t *w = &b->weights;
    const size_t per = layer_floats(b), n = per * (size_t)b->config.n_layers;
    float *h = (float *)calloc(n, sizeof(float));
    if (!h) return NULL;
    for (int l = 0; l < b->config.n_layers; l++) {
        float *p = h + per * (size_t)l;
        const float *src[10] = {w->q_b[l], w->k_b[l], w->v_b[l], w->out_b[l], w->ffn_up_b[l], w->ffn_down_b[l],
                                w->att_norm_w[l], w->att_norm_b[l], w->ffn_norm_w[l], w->ffn_norm_b[l]};
        const int len[10] = {dim, dim, dim, dim, ff, dim, dim, dim, dim, dim};
        for (int i = 0; i < 10; i++) {
            if (src[i]) memcpy(p, src[i], (size_t)len[i] * sizeof(float));
            p += len[i];
        }
    }
    eif_gpu_buf_t *buf = eif_gpu_buf_create(&g_ctx, n * sizeof(float), false);
    if (buf && !eif_gpu_buf_upload(&g_ctx, buf, 0, h, n * sizeof(float))) { eif_gpu_buf_destroy(&g_ctx, buf); buf = NULL; }
    free(h);
    return buf;
}

static int ensure_acts(const eif_bert_t *b, int T, size_t score_bytes)
{
    const size_t dim = (size_t)b->config.dim, ff = (size_t)b->config.intermediate_dim, mb = eif_gpu_max_binding_bytes(&g_ctx);
    if (score_bytes > mb || (size_t)T * ff * 4 > mb || (size_t)T * 3 * dim * 4 > mb) return 0;
    const size_t need_x = (size_t)T * dim * 4, need_i = (size_t)T * ff * 4;
    if (need_x > g_cap_x || need_i > g_cap_inter) { /* another model needs wider rows even for the same T */
        eif_gpu_buf_destroy(&g_ctx, g_x); eif_gpu_buf_destroy(&g_ctx, g_qkv); eif_gpu_buf_destroy(&g_ctx, g_att);
        eif_gpu_buf_destroy(&g_ctx, g_inter);
        g_x = eif_gpu_buf_create(&g_ctx, (size_t)T * dim * 4, true);
        g_qkv = eif_gpu_buf_create(&g_ctx, (size_t)T * 3 * dim * 4, false);
        g_att = eif_gpu_buf_create(&g_ctx, (size_t)T * dim * 4, false);
        g_inter = eif_gpu_buf_create(&g_ctx, (size_t)T * ff * 4, false);
        g_cap_x = g_x && g_qkv && g_att && g_inter ? need_x : 0;
        g_cap_inter = g_cap_x ? need_i : 0;
        if (!g_cap_x) return 0;
    }
    if (score_bytes > g_scores_bytes) {
        eif_gpu_buf_destroy(&g_ctx, g_scores);
        g_scores = eif_gpu_buf_create(&g_ctx, score_bytes, false);
        g_scores_bytes = g_scores ? score_bytes : 0;
        if (!g_scores) return 0;
    }
    return 1;
}

/* C[M, N] (+)= epilogue(A[M, K] W^T): bias from params at bias_off, optional GELU / residual onto C */
static bool rec_gemm(const void *Wp, int N, int K, eif_gpu_buf_t *A, eif_gpu_buf_t *params, uint32_t bias_off,
                     eif_gpu_buf_t *C, uint32_t ldc, uint32_t c_off, int M, uint32_t flags)
{
    eif_gpu_weight_t *w = get_weight(Wp, N, K);
    if (!w) return false;
    eif_gpu_gemm_pc_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.M = (uint32_t)M; pc.N = (uint32_t)N; pc.K = (uint32_t)K;
    pc.lda = (uint32_t)K; pc.ldb = (uint32_t)K; pc.ldc = ldc;
    pc.c_off = c_off; pc.s_off = eif_gpu_weight_scale_off(w); pc.bias_off = bias_off;
    pc.flags = EIF_GPU_GEMM_BIAS | flags;
    pc.zdiv = 1; pc.alpha = 1.0f;
    return eif_gpu_cmd_gemm(&g_ctx, EIF_GPU_K_GEMM_Q8B, A, eif_gpu_weight_buf(w), eif_gpu_weight_buf(w), params, C, &pc, 1);
}

static bool rec_layernorm(eif_gpu_buf_t *x, eif_gpu_buf_t *params, uint32_t w_off, uint32_t b_off, int dim, int T)
{
    struct { uint32_t N, rows, a_off, c_off, w_off, b_off; float eps; } pc = {(uint32_t)dim, (uint32_t)T, 0, 0, w_off, b_off, 1e-12f};
    eif_gpu_buf_t *b[5] = {x, NULL, params, NULL, x};
    return eif_gpu_cmd_dispatch(&g_ctx, EIF_GPU_K_LAYERNORM, b, &pc, sizeof(pc), T > 65535 ? 65535 : (uint32_t)T, 1, 1);
}

static bool rec_attention(const eif_bert_t *bert, int T, bool fused)
{
    const int dim = bert->config.dim, heads = bert->config.n_heads, hd = dim / heads;
    const float scale = 1.0f / sqrtf((float)hd);
    if (fused) {
        eif_gpu_attn_pc_t ap = {(uint32_t)T, (uint32_t)heads, (uint32_t)(3 * dim), (uint32_t)dim, 0, 0,
                                scale * 1.4426950408889634f};
        eif_gpu_buf_t *b[5] = {g_qkv, NULL, NULL, NULL, g_att};
        return eif_gpu_cmd_dispatch(&g_ctx, EIF_GPU_K_ATTENTION, b, &ap, sizeof(ap), ((uint32_t)T + 63) / 64,
                                    (uint32_t)heads, 1);
    }
    eif_gpu_gemm_pc_t pc;
    memset(&pc, 0, sizeof(pc)); /* scores[h] = Q_h K_h^T * scale */
    pc.M = (uint32_t)T; pc.N = (uint32_t)T; pc.K = (uint32_t)hd;
    pc.lda = (uint32_t)(3 * dim); pc.ldb = (uint32_t)(3 * dim); pc.ldc = (uint32_t)T;
    pc.b_off = (uint32_t)dim;
    pc.zdiv = (uint32_t)heads;
    pc.a_s2 = (uint32_t)hd; pc.b_s2 = (uint32_t)hd; pc.c_s2 = (uint32_t)T * (uint32_t)T;
    pc.alpha = scale;
    bool ok = eif_gpu_cmd_gemm(&g_ctx, EIF_GPU_K_GEMM_F32, g_qkv, g_qkv, NULL, NULL, g_scores, &pc, (uint32_t)heads);
    eif_gpu_cmd_barrier(&g_ctx);
    uint32_t rows = (uint32_t)heads * (uint32_t)T, sm[3] = {(uint32_t)T, rows, 0};
    eif_gpu_buf_t *sb[5] = {NULL, NULL, NULL, NULL, g_scores};
    ok = ok && eif_gpu_cmd_dispatch(&g_ctx, EIF_GPU_K_SOFTMAX, sb, sm, sizeof(sm), rows > 65535 ? 65535 : rows, 1, 1);
    eif_gpu_cmd_barrier(&g_ctx);
    memset(&pc, 0, sizeof(pc)); /* att[:, h] = P_h V_h */
    pc.M = (uint32_t)T; pc.N = (uint32_t)hd; pc.K = (uint32_t)T;
    pc.lda = (uint32_t)T; pc.ldb = (uint32_t)(3 * dim); pc.ldc = (uint32_t)dim;
    pc.b_off = (uint32_t)(2 * dim);
    pc.zdiv = (uint32_t)heads;
    pc.a_s2 = (uint32_t)T * (uint32_t)T; pc.b_s2 = (uint32_t)hd; pc.c_s2 = (uint32_t)hd;
    pc.alpha = 1.0f;
    return ok && eif_gpu_cmd_gemm(&g_ctx, EIF_GPU_K_GEMM_KN, g_scores, g_qkv, NULL, NULL, g_att, &pc, (uint32_t)heads);
}

int bert_gpu_layers(eif_bert_t *bert, int T)
{
    if (g_state != 1 || too_short(T, bert->config.dim) || !plain_bert(bert)) return 0;
    range_t *r = find_range(bert->weights.q_w[0]);
    if (!r) return 0;
    const int dim = bert->config.dim, ff = bert->config.intermediate_dim, heads = bert->config.n_heads, hd = dim / heads;
    const eif_bert_weights_t *w = &bert->weights;
    const bool fused = hd == 64 && eif_gpu_kernel_available(&g_ctx, EIF_GPU_K_ATTENTION);
    if (!r->params) {
        /* uploads run their own command buffers, so every weight goes up before recording starts */
        for (int l = 0; l < bert->config.n_layers; l++)
            if (!get_weight(w->q_w[l], dim, dim) || !get_weight(w->k_w[l], dim, dim) || !get_weight(w->v_w[l], dim, dim) ||
                !get_weight(w->out_w[l], dim, dim) || !get_weight(w->ffn_up_w[l], ff, dim) ||
                !get_weight(w->ffn_down_w[l], dim, ff))
                return 0;
        r->params = build_params(bert);
        if (!r->params) return 0;
    }
    if (!ensure_acts(bert, T, fused ? 0 : (size_t)heads * (size_t)T * (size_t)T * sizeof(float))) return 0;

    memcpy(eif_gpu_buf_map(g_x), bert->scratch_seq_x, (size_t)T * dim * sizeof(float));
    if (!eif_gpu_cmd_begin(&g_ctx)) return 0;
    const size_t per = layer_floats(bert);
    bool ok = true;
    for (int l = 0; l < bert->config.n_layers && ok; l++) {
        const uint32_t o = (uint32_t)(per * (size_t)l);
        const uint32_t bq = o, bk = o + dim, bv = o + 2 * dim, bo = o + 3 * dim, bup = o + 4 * dim,
                       bdn = o + 4 * dim + ff, n1w = bdn + dim, n1b = n1w + dim, n2w = n1b + dim, n2b = n2w + dim;
        ok = ok && rec_gemm(w->q_w[l], dim, dim, g_x, r->params, bq, g_qkv, 3 * dim, 0, T, 0);
        ok = ok && rec_gemm(w->k_w[l], dim, dim, g_x, r->params, bk, g_qkv, 3 * dim, dim, T, 0);
        ok = ok && rec_gemm(w->v_w[l], dim, dim, g_x, r->params, bv, g_qkv, 3 * dim, 2 * dim, T, 0);
        eif_gpu_cmd_barrier(&g_ctx);
        ok = ok && rec_attention(bert, T, fused);
        eif_gpu_cmd_barrier(&g_ctx);
        ok = ok && rec_gemm(w->out_w[l], dim, dim, g_att, r->params, bo, g_x, dim, 0, T, EIF_GPU_GEMM_RESIDUAL);
        eif_gpu_cmd_barrier(&g_ctx);
        ok = ok && rec_layernorm(g_x, r->params, n1w, n1b, dim, T);
        eif_gpu_cmd_barrier(&g_ctx);
        ok = ok && rec_gemm(w->ffn_up_w[l], ff, dim, g_x, r->params, bup, g_inter, ff, 0, T, EIF_GPU_GEMM_GELU);
        eif_gpu_cmd_barrier(&g_ctx);
        ok = ok && rec_gemm(w->ffn_down_w[l], dim, ff, g_inter, r->params, bdn, g_x, dim, 0, T, EIF_GPU_GEMM_RESIDUAL);
        eif_gpu_cmd_barrier(&g_ctx);
        ok = ok && rec_layernorm(g_x, r->params, n2w, n2b, dim, T);
        eif_gpu_cmd_barrier(&g_ctx);
    }
    if (!ok) { eif_gpu_cmd_submit(&g_ctx); return 0; } /* closes the recording; the CPU recomputes */
    if (!eif_gpu_cmd_submit(&g_ctx)) return 0;
    memcpy(bert->scratch_seq_x, eif_gpu_buf_map(g_x), (size_t)T * dim * sizeof(float));
    g_forwards++;
    return 1;
}

unsigned long eif_bert_gpu_forwards(void) { return g_forwards; }
