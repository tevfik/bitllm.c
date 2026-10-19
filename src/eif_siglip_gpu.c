/**
 * @file eif_siglip_gpu.c
 * @brief SigLIP2 ViT + connector on the Vulkan backend: weights resident on the GPU (INT8 rows dequantised
 *        in the GEMM kernel, FP32 otherwise, INT4 expanded to FP32 at upload), a batch of tiles runs the
 *        whole encoder (patch embedding, 12 layers incl. layernorm / attention / GELU / residual, post-LN,
 *        patch merge, connector) in a single command buffer submit.
 */

#include "eif_siglip.h"
#include "eif_siglip_gpu.h"
#include "eif_gpu_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define S_P     16      /* patch size */
#define S_IMG   512
#define S_L     12
#define S_H     768
#define S_HEADS 12
#define S_HD    64
#define S_I     3072
#define S_NP    1024
#define S_CIN   12288
#define S_COUT  576
#define S_NT    64

typedef struct {
    uint32_t ln1_w, ln1_b, qkv, qkv_s, qkv_b, o, o_s, o_b, ln2_w, ln2_b, fc1, fc1_s, fc1_b, fc2, fc2_s, fc2_b;
} layer_off_t;

typedef struct {
    eif_gpu_context_t *ctx;
    bool q8;
    int max_batch;
    eif_gpu_buf_t *w;
    layer_off_t L[S_L];
    uint32_t pe_w, pe_b, pos, post_w, post_b, conn;
    eif_gpu_buf_t *x, *xn, *qkv, *attn, *big, *in, *out;
} siglip_gpu_t;

/* Host image of the weight buffer; offsets are recorded in units of 4 bytes (INT8 codes: in bytes). */
typedef struct {
    uint8_t *data;
    size_t used;
} wimg_t;

static size_t wimg_put(wimg_t *w, const void *src, size_t bytes, bool counting)
{
    size_t off = w->used;
    if (!counting) {
        if (src) memcpy(w->data + off, src, bytes);
        else memset(w->data + off, 0, bytes);
    }
    w->used = (off + bytes + 15) & ~(size_t)15;
    return off;
}

/* One projection: FP32 rows (dequantised for INT4 / FP32 storage) or INT8 codes + scales. */
static void put_proj(wimg_t *w, const eif_siglip_t *m, int l, int idx, int N, int K, bool q8, bool counting,
                     float *tmp, uint32_t *off_codes, uint32_t *off_scales)
{
    if (q8) {
        const eif_siglip_qmat_t *q = &m->layers[l].qm[idx];
        size_t c = wimg_put(w, q->codes, (size_t)N * K, counting);
        size_t s = wimg_put(w, q->scales, (size_t)N * sizeof(float), counting);
        if (off_codes) *off_codes = (uint32_t)c;
        if (off_scales) *off_scales = (uint32_t)(s / 4);
    } else {
        if (!counting) eif_siglip_dequant_weight(m, l, idx, tmp);
        size_t c = wimg_put(w, tmp, (size_t)N * K * sizeof(float), counting);
        if (off_codes) *off_codes = (uint32_t)(c / 4);
    }
}

/* q, k, v are stored back to back so one [2304, 768] GEMM computes all three. */
static void put_qkv(wimg_t *w, const eif_siglip_t *m, int l, bool q8, bool counting, float *tmp, layer_off_t *o)
{
    const eif_siglip_layer_t *L = &m->layers[l];
    if (q8) {
        size_t c0 = 0;
        for (int i = 0; i < 3; i++) {
            size_t c = wimg_put(w, L->qm[i].codes, (size_t)S_H * S_H, counting);
            if (i == 0) c0 = c;
        }
        size_t s0 = 0;
        for (int i = 0; i < 3; i++) {
            size_t s = wimg_put(w, L->qm[i].scales, S_H * sizeof(float), counting);
            if (i == 0) s0 = s;
        }
        o->qkv = (uint32_t)c0;
        o->qkv_s = (uint32_t)(s0 / 4);
    } else {
        size_t c0 = 0;
        for (int i = 0; i < 3; i++) {
            if (!counting) eif_siglip_dequant_weight(m, l, i, tmp);
            size_t c = wimg_put(w, tmp, (size_t)S_H * S_H * sizeof(float), counting);
            if (i == 0) c0 = c;
        }
        o->qkv = (uint32_t)(c0 / 4);
    }
    const float *b[3] = {L->q_b, L->k_b, L->v_b};
    size_t b0 = 0;
    for (int i = 0; i < 3; i++) {
        size_t bo = wimg_put(w, b[i], S_H * sizeof(float), counting);
        if (i == 0) b0 = bo;
    }
    o->qkv_b = (uint32_t)(b0 / 4);
}

#define F32OFF(w, p, n) ((uint32_t)(wimg_put((w), (p), (size_t)(n) * sizeof(float), counting) / 4))

static void build_weights(wimg_t *w, siglip_gpu_t *g, const eif_siglip_t *m, bool counting, float *tmp)
{
    const bool q8 = g->q8;
    w->used = 0;
    g->pe_w = F32OFF(w, m->pe_w, S_H * 3 * S_P * S_P);
    g->pe_b = F32OFF(w, m->pe_b, S_H);
    g->pos = F32OFF(w, m->pos_w, S_NP * S_H);
    for (int l = 0; l < S_L; l++) {
        const eif_siglip_layer_t *L = &m->layers[l];
        layer_off_t *o = &g->L[l];
        o->ln1_w = F32OFF(w, L->ln1_w, S_H);
        o->ln1_b = F32OFF(w, L->ln1_b, S_H);
        put_qkv(w, m, l, q8, counting, tmp, o);
        put_proj(w, m, l, 3, S_H, S_H, q8, counting, tmp, &o->o, &o->o_s);
        o->o_b = F32OFF(w, L->out_b, S_H);
        o->ln2_w = F32OFF(w, L->ln2_w, S_H);
        o->ln2_b = F32OFF(w, L->ln2_b, S_H);
        put_proj(w, m, l, 4, S_I, S_H, q8, counting, tmp, &o->fc1, &o->fc1_s);
        o->fc1_b = F32OFF(w, L->fc1_b, S_I);
        put_proj(w, m, l, 5, S_H, S_I, q8, counting, tmp, &o->fc2, &o->fc2_s);
        o->fc2_b = F32OFF(w, L->fc2_b, S_H);
    }
    g->post_w = F32OFF(w, m->post_ln_w, S_H);
    g->post_b = F32OFF(w, m->post_ln_b, S_H);
    g->conn = F32OFF(w, m->conn_w, (size_t)S_COUT * S_CIN);
}

static void free_acts(siglip_gpu_t *g)
{
    eif_gpu_buf_t **b[7] = {&g->x, &g->xn, &g->qkv, &g->attn, &g->big, &g->in, &g->out};
    for (int i = 0; i < 7; i++) {
        eif_gpu_buf_destroy(g->ctx, *b[i]);
        *b[i] = NULL;
    }
}

static bool alloc_acts(siglip_gpu_t *g, int B)
{
    const size_t f = sizeof(float), rows = (size_t)B * S_NP;
    size_t big = (size_t)B * S_HEADS * S_NP * S_NP;                  /* attention scores */
    if (big < rows * S_I) big = rows * S_I;                            /* MLP hidden */
    if (big < (size_t)B * S_NT * S_CIN) big = (size_t)B * S_NT * S_CIN; /* merged patches */
    if (big * f > eif_gpu_max_binding_bytes(g->ctx)) return false;
    g->x = eif_gpu_buf_create(g->ctx, rows * S_H * f, false);
    g->xn = eif_gpu_buf_create(g->ctx, rows * S_H * f, false);
    g->qkv = eif_gpu_buf_create(g->ctx, rows * 3 * S_H * f, false);
    g->attn = eif_gpu_buf_create(g->ctx, rows * S_H * f, false);
    g->big = eif_gpu_buf_create(g->ctx, big * f, false);
    g->in = eif_gpu_buf_create(g->ctx, rows * S_H * f, true);
    g->out = eif_gpu_buf_create(g->ctx, (size_t)B * S_NT * S_COUT * f, true);
    if (g->x && g->xn && g->qkv && g->attn && g->big && g->in && g->out) return true;
    free_acts(g);
    return false;
}

void eif_siglip_gpu_release(eif_siglip_t *model)
{
    if (!model || !model->gpu_vit) return;
    siglip_gpu_t *g = (siglip_gpu_t *)model->gpu_vit;
    free_acts(g);
    eif_gpu_buf_destroy(g->ctx, g->w);
    free(g);
    model->gpu_vit = NULL;
}

int eif_siglip_gpu_attach(eif_siglip_t *model, eif_gpu_context_t *ctx, int max_batch)
{
    if (!model || !model->is_initialized || !ctx || !ctx->is_available) return -1;
    eif_siglip_gpu_release(model);
    siglip_gpu_t *g = (siglip_gpu_t *)calloc(1, sizeof(*g));
    if (!g) return -2;
    g->ctx = ctx;
    g->q8 = (model->header.qtype == EIF_SIGLIP_Q_INT8);

    wimg_t w = {NULL, 0};
    build_weights(&w, g, model, true, NULL);
    const size_t wbytes = w.used;
    float *tmp = g->q8 ? NULL : (float *)malloc((size_t)S_I * S_H * sizeof(float));
    w.data = (uint8_t *)malloc(wbytes);
    if (!w.data || (!g->q8 && !tmp) || wbytes > eif_gpu_max_binding_bytes(ctx)) {
        free(w.data); free(tmp); free(g);
        return -3;
    }
    build_weights(&w, g, model, false, tmp);
    free(tmp);
    g->w = eif_gpu_buf_create(ctx, wbytes, false);
    bool ok = g->w && eif_gpu_buf_upload(ctx, g->w, 0, w.data, wbytes);
    free(w.data);
    if (!ok) {
        eif_gpu_buf_destroy(ctx, g->w);
        free(g);
        return -4;
    }

    if (max_batch < 1) max_batch = 1;
    if (max_batch > EIF_SIGLIP_MAX_TILES) max_batch = EIF_SIGLIP_MAX_TILES;
    while (max_batch >= 1 && !alloc_acts(g, max_batch)) max_batch /= 2;
    if (max_batch < 1) {
        eif_gpu_buf_destroy(ctx, g->w);
        free(g);
        return -5;
    }
    g->max_batch = max_batch;
    model->gpu_vit = g;
    return 0;
}

int eif_siglip_gpu_batch(const eif_siglip_t *model) { return model && model->gpu_vit ? ((siglip_gpu_t *)model->gpu_vit)->max_batch : 0; }

/* ---- command recording ------------------------------------------------------------------- */

static uint32_t groups_for(size_t count) { size_t g = (count + 255) / 256; return (uint32_t)(g > 65535 ? 65535 : g); }

static bool rec_gemm(siglip_gpu_t *g, bool wq8, eif_gpu_buf_t *A, uint32_t a_off, uint32_t lda, uint32_t b_off,
                     uint32_t s_off, uint32_t bias_off, eif_gpu_buf_t *C, uint32_t ldc,
                     uint32_t M, uint32_t N, uint32_t K, uint32_t flags)
{
    eif_gpu_gemm_pc_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.M = M; pc.N = N; pc.K = K;
    pc.lda = lda; pc.ldb = K; pc.ldc = ldc;
    pc.a_off = a_off; pc.b_off = b_off;
    pc.s_off = s_off; pc.bias_off = bias_off;
    pc.flags = flags | EIF_GPU_GEMM_BIAS | (wq8 ? EIF_GPU_GEMM_SCALES : 0);
    pc.zdiv = 1;
    pc.alpha = 1.0f;
    return eif_gpu_cmd_gemm(g->ctx, wq8 ? EIF_GPU_K_GEMM_Q8 : EIF_GPU_K_GEMM_F32, A, g->w, g->w, g->w, C, &pc, 1);
}

static bool rec_layernorm(siglip_gpu_t *g, eif_gpu_buf_t *in, eif_gpu_buf_t *out, uint32_t rows,
                          uint32_t w_off, uint32_t b_off)
{
    struct { uint32_t N, rows, a_off, c_off, w_off, b_off; float eps; } pc = {S_H, rows, 0, 0, w_off, b_off, 1e-6f};
    eif_gpu_buf_t *b[5] = {in, NULL, g->w, NULL, out};
    return eif_gpu_cmd_dispatch(g->ctx, EIF_GPU_K_LAYERNORM, b, &pc, sizeof(pc), rows > 65535 ? 65535 : rows, 1, 1);
}

static bool record_batch(siglip_gpu_t *g, int B)
{
    eif_gpu_context_t *ctx = g->ctx;
    const uint32_t rows = (uint32_t)B * S_NP;
    bool ok = true;

    /* x = pos (per tile) ; x += patches @ pe_w^T + pe_b */
    {
        uint32_t pc[4] = {rows * S_H, S_NP * S_H, g->pos, 0};
        eif_gpu_buf_t *b[5] = {g->w, NULL, NULL, NULL, g->x};
        ok = ok && eif_gpu_cmd_dispatch(ctx, EIF_GPU_K_BCAST, b, pc, sizeof(pc), groups_for(pc[0]), 1, 1);
    }
    eif_gpu_cmd_barrier(ctx);
    ok = ok && rec_gemm(g, false, g->in, 0, S_H, g->pe_w, 0, g->pe_b, g->x, S_H, rows, S_H, S_H,
                        EIF_GPU_GEMM_RESIDUAL);
    eif_gpu_cmd_barrier(ctx);

    for (int l = 0; l < S_L && ok; l++) {
        const layer_off_t *o = &g->L[l];
        ok = ok && rec_layernorm(g, g->x, g->xn, rows, o->ln1_w, o->ln1_b);
        eif_gpu_cmd_barrier(ctx);
        ok = ok && rec_gemm(g, g->q8, g->xn, 0, S_H, o->qkv, o->qkv_s, o->qkv_b, g->qkv, 3 * S_H,
                            rows, 3 * S_H, S_H, 0);
        eif_gpu_cmd_barrier(ctx);

        /* scores[t, h] = Q_h K_h^T / 8 */
        eif_gpu_gemm_pc_t pc;
        memset(&pc, 0, sizeof(pc));
        pc.M = S_NP; pc.N = S_NP; pc.K = S_HD;
        pc.lda = 3 * S_H; pc.ldb = 3 * S_H; pc.ldc = S_NP;
        pc.a_off = 0; pc.b_off = S_H;
        pc.zdiv = S_HEADS;
        pc.a_s1 = S_NP * 3 * S_H; pc.a_s2 = S_HD;
        pc.b_s1 = S_NP * 3 * S_H; pc.b_s2 = S_HD;
        pc.c_s1 = S_HEADS * S_NP * S_NP; pc.c_s2 = S_NP * S_NP;
        pc.alpha = 0.125f;
        ok = ok && eif_gpu_cmd_gemm(ctx, EIF_GPU_K_GEMM_F32, g->qkv, g->qkv, NULL, NULL, g->big, &pc,
                                    (uint32_t)B * S_HEADS);
        eif_gpu_cmd_barrier(ctx);
        {
            uint32_t sm[3] = {S_NP, (uint32_t)B * S_HEADS * S_NP, 0};
            eif_gpu_buf_t *b[5] = {NULL, NULL, NULL, NULL, g->big};
            ok = ok && eif_gpu_cmd_dispatch(ctx, EIF_GPU_K_SOFTMAX, b, sm, sizeof(sm),
                                            sm[1] > 65535 ? 65535 : sm[1], 1, 1);
        }
        eif_gpu_cmd_barrier(ctx);
        /* attn[t, :, h] = P V_h */
        memset(&pc, 0, sizeof(pc));
        pc.M = S_NP; pc.N = S_HD; pc.K = S_NP;
        pc.lda = S_NP; pc.ldb = 3 * S_H; pc.ldc = S_H;
        pc.a_off = 0; pc.b_off = 2 * S_H;
        pc.zdiv = S_HEADS;
        pc.a_s1 = S_HEADS * S_NP * S_NP; pc.a_s2 = S_NP * S_NP;
        pc.b_s1 = S_NP * 3 * S_H; pc.b_s2 = S_HD;
        pc.c_s1 = S_NP * S_H; pc.c_s2 = S_HD;
        pc.alpha = 1.0f;
        ok = ok && eif_gpu_cmd_gemm(ctx, EIF_GPU_K_GEMM_KN, g->big, g->qkv, NULL, NULL, g->attn, &pc,
                                    (uint32_t)B * S_HEADS);
        eif_gpu_cmd_barrier(ctx);

        ok = ok && rec_gemm(g, g->q8, g->attn, 0, S_H, o->o, o->o_s, o->o_b, g->x, S_H, rows, S_H, S_H,
                            EIF_GPU_GEMM_RESIDUAL);
        eif_gpu_cmd_barrier(ctx);
        ok = ok && rec_layernorm(g, g->x, g->xn, rows, o->ln2_w, o->ln2_b);
        eif_gpu_cmd_barrier(ctx);
        ok = ok && rec_gemm(g, g->q8, g->xn, 0, S_H, o->fc1, o->fc1_s, o->fc1_b, g->big, S_I, rows, S_I, S_H,
                            EIF_GPU_GEMM_GELU);
        eif_gpu_cmd_barrier(ctx);
        ok = ok && rec_gemm(g, g->q8, g->big, 0, S_I, o->fc2, o->fc2_s, o->fc2_b, g->x, S_H, rows, S_H, S_I,
                            EIF_GPU_GEMM_RESIDUAL);
        eif_gpu_cmd_barrier(ctx);
    }

    ok = ok && rec_layernorm(g, g->x, g->x, rows, g->post_w, g->post_b);
    eif_gpu_cmd_barrier(ctx);
    {
        uint32_t pc[4] = {(uint32_t)B * S_NT * S_CIN, S_H, 0, 0};
        eif_gpu_buf_t *b[5] = {g->x, NULL, NULL, NULL, g->big};
        ok = ok && eif_gpu_cmd_dispatch(ctx, EIF_GPU_K_MERGE, b, pc, sizeof(pc), groups_for(pc[0]), 1, 1);
    }
    eif_gpu_cmd_barrier(ctx);
    {
        eif_gpu_gemm_pc_t pc;
        memset(&pc, 0, sizeof(pc));
        pc.M = (uint32_t)B * S_NT; pc.N = S_COUT; pc.K = S_CIN;
        pc.lda = S_CIN; pc.ldb = S_CIN; pc.ldc = S_COUT;
        pc.b_off = g->conn;
        pc.zdiv = 1;
        pc.alpha = 1.0f;
        ok = ok && eif_gpu_cmd_gemm(ctx, EIF_GPU_K_GEMM_F32, g->big, g->w, NULL, NULL, g->out, &pc, 1);
    }
    return ok;
}

/* [3, 512, 512] tile -> [1024 patches, 768] rows in Conv2D weight order (c, ky, kx). */
static void im2col_tile(const float *pix, float *dst)
{
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < S_NP; p++) {
        const int py = p / 32, px = p % 32;
        float *row = dst + (size_t)p * S_H;
        for (int c = 0; c < 3; c++)
            for (int ky = 0; ky < S_P; ky++)
                memcpy(row + c * S_P * S_P + ky * S_P,
                       pix + (size_t)c * S_IMG * S_IMG + (size_t)(py * S_P + ky) * S_IMG + px * S_P,
                       S_P * sizeof(float));
    }
}

int eif_siglip_gpu_encode(const eif_siglip_t *model, const float *const *tiles, int n_tiles, float *const *outs)
{
    if (!model || !model->gpu_vit || !tiles || !outs || n_tiles < 0) return -1;
    siglip_gpu_t *g = (siglip_gpu_t *)model->gpu_vit;
    float *in = (float *)eif_gpu_buf_map(g->in);
    const float *out = (const float *)eif_gpu_buf_map(g->out);
    for (int t0 = 0; t0 < n_tiles; t0 += g->max_batch) {
        const int B = (n_tiles - t0) < g->max_batch ? (n_tiles - t0) : g->max_batch;
        for (int t = 0; t < B; t++) im2col_tile(tiles[t0 + t], in + (size_t)t * S_NP * S_H);
        if (!eif_gpu_cmd_begin(g->ctx)) return -2;
        bool ok = record_batch(g, B);
        if (!eif_gpu_cmd_submit(g->ctx) || !ok) return -3;
        for (int t = 0; t < B; t++)
            memcpy(outs[t0 + t], out + (size_t)t * S_NT * S_COUT, (size_t)S_NT * S_COUT * sizeof(float));
    }
    return 0;
}
