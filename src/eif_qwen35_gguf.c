/**
 * @file eif_qwen35_gguf.c
 * @brief Load a standard llama.cpp style Qwen3.5 text GGUF (F32/F16/BF16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K) into the QWEN35 runtime.
 *
 * The GGUF is dequantised row by row and repacked in memory into the exact .eifm image qwen35_load() already
 * understands (INT4 row-scaled linears, INT8 embedding, FP32 norms/conv, plain_linear = 1), so no other runtime
 * code changes. GGUF conventions undone here: RMSNorm weights already carry the +1 (same as the runtime expects),
 * ssm_a = -exp(A_log) -> A_log = log(-ssm_a). Limits: 24 layers, tied LM head, num_k_heads == num_v_heads,
 * rms eps 1e-6, no vision tower (it is a separate mmproj file in GGUF).
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "eif_qwen35.h"
#include "eif_gguf.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define QK_K 256

/* ---------------- GGUF header parsing ---------------- */

typedef struct { const uint8_t *p, *end; int bad; } rd_t;

static uint64_t rd_u(rd_t *r, int n) {
    if (r->bad || r->p + n > r->end) { r->bad = 1; return 0; }
    uint64_t v = 0;
    memcpy(&v, r->p, (size_t)n);
    r->p += n;
    return v;
}

static size_t scalar_size(uint32_t t) {
    switch (t) {
    case 0: case 1: case 7: return 1;
    case 2: case 3: return 2;
    case 4: case 5: case 6: return 4;
    case 10: case 11: case 12: return 8;
    default: return 0;
    }
}

static void skip_value(rd_t *r, uint32_t t) {
    if (t == 8) { uint64_t n = rd_u(r, 8); if (!r->bad) { if (r->p + n > r->end) r->bad = 1; else r->p += n; } return; }
    if (t == 9) {
        uint32_t et = (uint32_t)rd_u(r, 4);
        uint64_t n = rd_u(r, 8);
        size_t ss = scalar_size(et);
        if (ss) { if (r->p + n * ss > r->end) r->bad = 1; else r->p += n * ss; return; }
        for (uint64_t i = 0; i < n && !r->bad; i++) skip_value(r, et);
        return;
    }
    size_t ss = scalar_size(t);
    if (!ss) { r->bad = 1; return; }
    if (r->p + ss > r->end) r->bad = 1; else r->p += ss;
}

static double read_number(rd_t *r, uint32_t t) {
    union { uint64_t u; float f; double d; } v; v.u = rd_u(r, (int)scalar_size(t));
    switch (t) {
    case 0: return (uint8_t)v.u;  case 1: return (int8_t)v.u;
    case 2: return (uint16_t)v.u; case 3: return (int16_t)v.u;
    case 4: return (uint32_t)v.u; case 5: return (int32_t)v.u;
    case 6: return v.f;           case 10: return (double)v.u;
    case 11: return (double)(int64_t)v.u; case 12: return v.d;
    default: return 0;
    }
}

/* ---------------- dequantisation (one row) ---------------- */

#pragma pack(push, 1)
typedef struct { uint16_t d; int8_t qs[32]; } blk_q8_0;
typedef struct { uint16_t d; uint8_t qs[16]; } blk_q4_0;
typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; } blk_q4_K;
typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } blk_q5_K;
typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; } blk_q6_K;
#pragma pack(pop)

static void sc_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (uint8_t)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

/* Block geometry: elements per block and bytes per block; 0 when the type is unsupported. */
static int type_geom(uint32_t t, size_t *bytes) {
    switch (t) {
    case GGUF_TYPE_F32: *bytes = 4; return 1;
    case GGUF_TYPE_F16: *bytes = 2; return 1;
    case 30:            *bytes = 2; return 1; /* BF16 */
    case GGUF_TYPE_Q8_0: *bytes = sizeof(blk_q8_0); return 32;
    case GGUF_TYPE_Q4_0: *bytes = sizeof(blk_q4_0); return 32;
    case GGUF_TYPE_Q4_K: *bytes = sizeof(blk_q4_K); return QK_K;
    case GGUF_TYPE_Q5_K: *bytes = sizeof(blk_q5_K); return QK_K;
    case GGUF_TYPE_Q6_K: *bytes = sizeof(blk_q6_K); return QK_K;
    default: return 0;
    }
}

static void dequant_row(uint32_t t, const void *src, float *y, int n) {
    switch (t) {
    case GGUF_TYPE_F32: memcpy(y, src, (size_t)n * 4); break;
    case GGUF_TYPE_F16: { const uint16_t *h = src; for (int i = 0; i < n; i++) y[i] = fp16_to_fp32(h[i]); break; }
    case 30: { const uint16_t *h = src; for (int i = 0; i < n; i++) { uint32_t u = (uint32_t)h[i] << 16; memcpy(&y[i], &u, 4); } break; }
    case GGUF_TYPE_Q8_0: {
        const blk_q8_0 *b = src;
        for (int i = 0; i < n / 32; i++) { float d = fp16_to_fp32(b[i].d); for (int j = 0; j < 32; j++) y[i * 32 + j] = d * b[i].qs[j]; }
        break;
    }
    case GGUF_TYPE_Q4_0: {
        const blk_q4_0 *b = src;
        for (int i = 0; i < n / 32; i++) {
            float d = fp16_to_fp32(b[i].d);
            for (int j = 0; j < 16; j++) { y[i * 32 + j] = ((b[i].qs[j] & 0x0F) - 8) * d; y[i * 32 + j + 16] = ((b[i].qs[j] >> 4) - 8) * d; }
        }
        break;
    }
    case GGUF_TYPE_Q4_K: {
        const blk_q4_K *b = src;
        for (int i = 0; i < n / QK_K; i++) {
            const uint8_t *q = b[i].qs; float d = fp16_to_fp32(b[i].d), mn = fp16_to_fp32(b[i].dmin);
            int is = 0; uint8_t sc, m;
            for (int j = 0; j < QK_K; j += 64) {
                sc_min_k4(is, b[i].scales, &sc, &m); float d1 = d * sc, m1 = mn * m;
                sc_min_k4(is + 1, b[i].scales, &sc, &m); float d2 = d * sc, m2 = mn * m;
                for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0x0F) - m1;
                for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;
                q += 32; is += 2;
            }
        }
        break;
    }
    case GGUF_TYPE_Q5_K: {
        const blk_q5_K *b = src;
        for (int i = 0; i < n / QK_K; i++) {
            const uint8_t *ql = b[i].qs, *qh = b[i].qh; float d = fp16_to_fp32(b[i].d), mn = fp16_to_fp32(b[i].dmin);
            int is = 0; uint8_t sc, m, u1 = 1, u2 = 2;
            for (int j = 0; j < QK_K; j += 64) {
                sc_min_k4(is, b[i].scales, &sc, &m); float d1 = d * sc, m1 = mn * m;
                sc_min_k4(is + 1, b[i].scales, &sc, &m); float d2 = d * sc, m2 = mn * m;
                for (int l = 0; l < 32; l++) *y++ = d1 * ((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1;
                for (int l = 0; l < 32; l++) *y++ = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
                ql += 32; is += 2; u1 = (uint8_t)(u1 << 2); u2 = (uint8_t)(u2 << 2);
            }
        }
        break;
    }
    case GGUF_TYPE_Q6_K: {
        const blk_q6_K *b = src;
        for (int i = 0; i < n / QK_K; i++) {
            float d = fp16_to_fp32(b[i].d); const uint8_t *ql = b[i].ql, *qh = b[i].qh; const int8_t *sc = b[i].scales;
            for (int h = 0; h < QK_K; h += 128) {
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    int q1 = ((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    y[l] = d * sc[is] * q1; y[l + 32] = d * sc[is + 2] * q2;
                    y[l + 64] = d * sc[is + 4] * q3; y[l + 96] = d * sc[is + 6] * q4;
                }
                y += 128; ql += 64; qh += 32; sc += 8;
            }
        }
        break;
    }
    default: break;
    }
}

/* ---------------- tensor lookup ---------------- */

typedef struct {
    const gguf_tensor_desc_t *t; int n; const uint8_t *data; size_t file_size; const uint8_t *file;
} tmap_t;

static const gguf_tensor_desc_t *find_t(const tmap_t *m, const char *name) {
    for (int i = 0; i < m->n; i++) if (strcmp(m->t[i].name, name) == 0) return &m->t[i];
    return NULL;
}

/* Fetch tensor `name` as rows x cols (cols = ne[0]); checks shape, type and bounds. NULL + message on failure. */
static const gguf_tensor_desc_t *need(const tmap_t *m, const char *name, int rows, int cols, int *err) {
    const gguf_tensor_desc_t *t = find_t(m, name);
    size_t bytes = 0;
    if (!t) { fprintf(stderr, "qwen35 gguf: missing tensor %s\n", name); *err = 1; return NULL; }
    int blk = type_geom(t->type, &bytes);
    uint64_t r = t->n_dims >= 2 ? t->ne[1] : 1;
    if (!blk || (int)t->ne[0] != cols || (int)r != rows || cols % blk) {
        fprintf(stderr, "qwen35 gguf: tensor %s has unsupported type %u or shape [%llu,%llu] (want [%d,%d])\n", name, t->type,
                (unsigned long long)t->ne[0], (unsigned long long)r, cols, rows);
        *err = 1; return NULL;
    }
    size_t need_b = (size_t)rows * (size_t)(cols / blk) * bytes;
    if ((size_t)(m->data - m->file) + t->offset + need_b > m->file_size) {
        fprintf(stderr, "qwen35 gguf: tensor %s is outside the file\n", name); *err = 1; return NULL;
    }
    return t;
}

static const uint8_t *row_ptr(const tmap_t *m, const gguf_tensor_desc_t *t, int row, int cols) {
    size_t bytes; int blk = type_geom(t->type, &bytes);
    return m->data + t->offset + (size_t)row * (size_t)(cols / blk) * bytes;
}

/* Dense F32 copy of a (small) tensor. */
static void copy_f32(const tmap_t *m, const char *name, int rows, int cols, float *dst, int *err) {
    const gguf_tensor_desc_t *t = need(m, name, rows, cols, err);
    if (!t) return;
    for (int r = 0; r < rows; r++) dequant_row(t->type, row_ptr(m, t, r, cols), dst + (size_t)r * cols, cols);
}

/* Row-wise symmetric quantisation, same maths as quant_pack.py (scale = absmax / qmax). */
static void quant_rows(const tmap_t *m, const char *name, int rows, int cols, int bits, uint8_t *dst, float *scales, int *err) {
    const gguf_tensor_desc_t *t = need(m, name, rows, cols, err);
    if (!t) return;
    size_t rb = bits == 8 ? (size_t)cols : (size_t)(cols + 1) / 2;
    float qmax = bits == 8 ? 127.0f : 7.0f;
    #pragma omp parallel
    {
        float *buf = (float *)malloc((size_t)(cols + 1) * sizeof(float));
        #pragma omp for schedule(static)
        for (int r = 0; r < rows; r++) {
            dequant_row(t->type, row_ptr(m, t, r, cols), buf, cols);
            float amax = 1e-8f;
            for (int c = 0; c < cols; c++) { float a = fabsf(buf[c]); if (a > amax) amax = a; }
            float s = amax / qmax, inv = 1.0f / s;
            scales[r] = s;
            uint8_t *o = dst + (size_t)r * rb;
            if (bits == 8) {
                for (int c = 0; c < cols; c++) {
                    int q = (int)nearbyintf(buf[c] * inv); q = q > 127 ? 127 : (q < -128 ? -128 : q);
                    ((int8_t *)o)[c] = (int8_t)q;
                }
            } else {
                buf[cols] = 0.0f;
                for (int c = 0; c < cols; c += 2) {
                    int a = (int)nearbyintf(buf[c] * inv), b = (int)nearbyintf(buf[c + 1] * inv);
                    a = a > 7 ? 7 : (a < -8 ? -8 : a); b = b > 7 ? 7 : (b < -8 ? -8 : b);
                    o[c / 2] = (uint8_t)(((a + 8) << 4) | ((b + 8) & 0x0F));
                }
            }
        }
        free(buf);
    }
}

/* ---------------- image builder ---------------- */

int qwen35_gguf_build_image(const uint8_t *file, size_t file_size, void **out_img, size_t *out_size) {
    rd_t r = {file, file + file_size, 0};
    if (rd_u(&r, 4) != GGUF_MAGIC) return -3;
    rd_u(&r, 4);
    uint64_t n_tensors = rd_u(&r, 8), n_kv = rd_u(&r, 8);

    char arch[64] = "";
    uint64_t alignment = 32;
    double v_block = 0, v_dim = 0, v_ff = 0, v_heads = 0, v_kv = 0, v_keylen = 0, v_ctx = 0, v_interval = 4, v_eps = 1e-6;
    double s_conv = 0, s_state = 0, s_group = 0, s_rank = 0, s_inner = 0;
    for (uint64_t i = 0; i < n_kv && !r.bad; i++) {
        char key[160];
        uint64_t kl = rd_u(&r, 8);
        if (r.bad || kl >= sizeof key || r.p + kl > r.end) return -3;
        memcpy(key, r.p, kl); key[kl] = 0; r.p += kl;
        uint32_t vt = (uint32_t)rd_u(&r, 4);
        if (strcmp(key, "general.architecture") == 0 && vt == 8) {
            uint64_t sl = rd_u(&r, 8);
            if (r.bad || r.p + sl > r.end) return -3;
            size_t n = sl < sizeof arch - 1 ? (size_t)sl : sizeof arch - 1; memcpy(arch, r.p, n); arch[n] = 0; r.p += sl;
        } else if (strcmp(key, "general.alignment") == 0 && scalar_size(vt)) {
            alignment = (uint64_t)read_number(&r, vt);
        } else if (scalar_size(vt) && strncmp(key, "qwen35.", 7) == 0) {
            double v = read_number(&r, vt); const char *k = key + 7;
            if (!strcmp(k, "block_count")) v_block = v;
            else if (!strcmp(k, "embedding_length")) v_dim = v;
            else if (!strcmp(k, "feed_forward_length")) v_ff = v;
            else if (!strcmp(k, "attention.head_count")) v_heads = v;
            else if (!strcmp(k, "attention.head_count_kv")) v_kv = v;
            else if (!strcmp(k, "attention.key_length")) v_keylen = v;
            else if (!strcmp(k, "context_length")) v_ctx = v;
            else if (!strcmp(k, "full_attention_interval")) v_interval = v;
            else if (!strcmp(k, "attention.layer_norm_rms_epsilon")) v_eps = v;
            else if (!strcmp(k, "ssm.conv_kernel")) s_conv = v;
            else if (!strcmp(k, "ssm.state_size")) s_state = v;
            else if (!strcmp(k, "ssm.group_count")) s_group = v;
            else if (!strcmp(k, "ssm.time_step_rank")) s_rank = v;
            else if (!strcmp(k, "ssm.inner_size")) s_inner = v;
        } else {
            skip_value(&r, vt);
        }
    }
    if (r.bad) return -3;
    if (strcmp(arch, "qwen35") != 0) { fprintf(stderr, "qwen35 gguf: architecture '%s' is not qwen35\n", arch); return -3; }
    if (n_tensors == 0 || n_tensors > 100000) return -3;

    gguf_tensor_desc_t *td = (gguf_tensor_desc_t *)calloc((size_t)n_tensors, sizeof *td);
    if (!td) return -4;
    for (uint64_t i = 0; i < n_tensors && !r.bad; i++) {
        uint64_t nl = rd_u(&r, 8);
        if (r.bad || nl >= sizeof td[i].name || r.p + nl > r.end) { free(td); return -3; }
        memcpy(td[i].name, r.p, nl); td[i].name[nl] = 0; r.p += nl;
        td[i].n_dims = (uint32_t)rd_u(&r, 4);
        for (uint32_t d = 0; d < td[i].n_dims; d++) { uint64_t v = rd_u(&r, 8); if (d < 4) td[i].ne[d] = v; }
        td[i].type = (uint32_t)rd_u(&r, 4);
        td[i].offset = rd_u(&r, 8);
    }
    if (r.bad) { free(td); return -3; }
    if (alignment == 0) alignment = 32;
    size_t hdr_len = (size_t)(r.p - file);
    tmap_t tm = {td, (int)n_tensors, file + ((hdr_len + alignment - 1) / alignment) * alignment, file_size, file};

    int dim = (int)v_dim, hidden = (int)v_ff, L = (int)v_block, nh = (int)v_heads, nkv = (int)v_kv, hd = (int)v_keylen;
    int ck = (int)s_conv, kd = (int)s_state, nk = (int)s_group, nv = (int)s_rank, interval = (int)v_interval;
    int vd = nv > 0 ? (int)s_inner / nv : 0;
    int err = 0;
    if (L != 24) { fprintf(stderr, "qwen35 gguf: the runtime supports exactly 24 layers (got %d)\n", L); err = 1; }
    if (nk != nv) { fprintf(stderr, "qwen35 gguf: num_k_heads != num_v_heads is not supported\n"); err = 1; }
    if (fabs(v_eps - 1e-6) > 1e-8) { fprintf(stderr, "qwen35 gguf: the runtime hard-codes rms eps 1e-6 (got %g)\n", v_eps); err = 1; }
    if (find_t(&tm, "output.weight")) { fprintf(stderr, "qwen35 gguf: untied output.weight is not supported\n"); err = 1; }
    if (dim <= 0 || hidden <= 0 || nh <= 0 || nkv <= 0 || hd <= 0 || ck <= 0 || kd <= 0 || vd <= 0 || interval <= 0) err = 1;
    const gguf_tensor_desc_t *te = err ? NULL : find_t(&tm, "token_embd.weight");
    if (err || !te) { free(td); return -4; }
    int vocab = (int)te->ne[1];
    if (nh * hd != dim || nv * vd != dim) { fprintf(stderr, "qwen35 gguf: runtime needs q_dim == val_dim == dim\n"); free(td); return -4; }

    uint8_t layer_full[24];
    int n_fa = 0, n_dn = 0;
    for (int l = 0; l < L; l++) { layer_full[l] = ((l + 1) % interval == 0); if (layer_full[l]) n_fa++; else n_dn++; }

    int key_dim = nk * kd, val_dim = nv * vd, conv_dim = key_dim * 2 + val_dim, q_dim = nh * hd, kv_dim = nkv * hd, qg_dim = q_dim * 2;
    size_t rbd = (size_t)(dim + 1) / 2, rbh = (size_t)(hidden + 1) / 2, rbv = (size_t)(val_dim + 1) / 2, rbq = (size_t)(q_dim + 1) / 2;

    size_t w_embed = (size_t)vocab * dim;
    size_t w_total = w_embed
        + (size_t)n_dn * (rbd * conv_dim + rbd * val_dim + rbv * dim + 2 * rbd * nv)
        + (size_t)n_fa * (rbd * qg_dim + rbd * kv_dim * 2 + rbq * dim)
        + (size_t)L * (rbd * hidden * 2 + rbh * dim)
        + (size_t)n_dn * ((size_t)conv_dim * ck + 2 * nv) * 4;
    size_t s_total = ((size_t)vocab
        + (size_t)n_dn * (conv_dim + val_dim + dim + 2 * nv)
        + (size_t)n_fa * (qg_dim + 2 * kv_dim + dim)
        + (size_t)L * (hidden * 2 + dim)) * 4;
    size_t n_total = ((size_t)(2 * L + 1) * dim + (size_t)n_dn * vd + (size_t)n_fa * 2 * hd) * 4;

    const size_t HDR = 48, CFG = 128, LT = 24;
    size_t total = HDR + CFG + LT + w_total + s_total + n_total;
    if (total > 0xFFFFFFFFu) { fprintf(stderr, "qwen35 gguf: image larger than the 32-bit .eifm offsets\n"); free(td); return -4; }
    uint8_t *img = (uint8_t *)calloc(1, total + 64);
    if (!img) { free(td); return -4; }

    qwen35_file_header_t *h = (qwen35_file_header_t *)img;
    h->magic = QWEN35_MAGIC; h->version = QWEN35_VERSION; h->config_offset = (uint32_t)HDR;
    h->weights_offset = (uint32_t)(HDR + CFG + LT); h->scales_offset = h->weights_offset + (uint32_t)w_total;
    h->norms_offset = h->scales_offset + (uint32_t)s_total; h->vision_offset = h->norms_offset + (uint32_t)n_total;
    h->weights_size = (uint32_t)w_total; h->scales_size = (uint32_t)s_total; h->norms_size = (uint32_t)n_total;

    qwen35_config_t *c = (qwen35_config_t *)(img + HDR);
    memset(c, 0, sizeof *c);
    c->dim = dim; c->hidden_dim = hidden; c->n_layers = L; c->n_heads = nh; c->n_kv_heads = nkv; c->head_dim = hd;
    c->vocab_size = vocab; c->max_seq_len = (int)(v_ctx > 0 && v_ctx < 4096 ? v_ctx : 4096);
    c->n_delta = n_dn; c->n_full = n_fa; c->lin_n_k_heads = nk; c->lin_k_head_dim = kd; c->lin_n_v_heads = nv;
    c->lin_v_head_dim = vd; c->conv_k_size = ck;
    c->lm_head_tied = 1; c->quant_lm = QWEN35_QTYPE_INT4; c->quant_embed = QWEN35_EMBED_INT8; c->plain_linear = 1;
    memcpy(img + HDR + CFG, layer_full, L);

    uint8_t *wp = img + h->weights_offset;
    float *sc = (float *)(img + h->scales_offset);
    float *np = (float *)(img + h->norms_offset);
    char nm[96];

    /* weights, in loader order: embed, dn_qkv, dn_z, dn_out, dn_pb, dn_pa, fa_q/k/v/o, ffn gate/up/down, conv, dt, A_log */
    uint8_t *w_embed_p = wp; wp += w_embed;
    uint8_t *w_dnqkv = wp; wp += (size_t)n_dn * rbd * conv_dim;
    uint8_t *w_dnz = wp;   wp += (size_t)n_dn * rbd * val_dim;
    uint8_t *w_dnout = wp; wp += (size_t)n_dn * rbv * dim;
    uint8_t *w_dnpb = wp;  wp += (size_t)n_dn * rbd * nv;
    uint8_t *w_dnpa = wp;  wp += (size_t)n_dn * rbd * nv;
    uint8_t *w_faq = wp;   wp += (size_t)n_fa * rbd * qg_dim;
    uint8_t *w_fak = wp;   wp += (size_t)n_fa * rbd * kv_dim;
    uint8_t *w_fav = wp;   wp += (size_t)n_fa * rbd * kv_dim;
    uint8_t *w_fao = wp;   wp += (size_t)n_fa * rbq * dim;
    uint8_t *w_ffg = wp;   wp += (size_t)L * rbd * hidden;
    uint8_t *w_ffu = wp;   wp += (size_t)L * rbd * hidden;
    uint8_t *w_ffd = wp;   wp += (size_t)L * rbh * dim;
    float *w_conv = (float *)wp; wp += (size_t)n_dn * conv_dim * ck * 4;
    float *w_dt = (float *)wp;   wp += (size_t)n_dn * nv * 4;
    float *w_al = (float *)wp;

    /* scales, in loader order */
    float *s_embed = sc; sc += vocab;
    quant_rows(&tm, "token_embd.weight", vocab, dim, 8, w_embed_p, s_embed, &err);
    /* the scale buffer holds all DeltaNet scales, then all FullAttn scales, then all FFN scales */
    float *sc_dn = sc;
    float *sc_fa = sc_dn + (size_t)n_dn * (conv_dim + val_dim + dim + 2 * nv);
    float *sc_ffn = sc_fa + (size_t)n_fa * (qg_dim + 2 * kv_dim + dim);
    int di = 0, fi = 0;
    for (int l = 0; l < L && !err; l++) {
        if (!layer_full[l]) {
            float *s_qkv = sc_dn; sc_dn += conv_dim; float *s_z = sc_dn; sc_dn += val_dim; float *s_out = sc_dn; sc_dn += dim;
            float *s_b = sc_dn; sc_dn += nv; float *s_a = sc_dn; sc_dn += nv;
            snprintf(nm, sizeof nm, "blk.%d.attn_qkv.weight", l); quant_rows(&tm, nm, conv_dim, dim, 4, w_dnqkv + (size_t)di * rbd * conv_dim, s_qkv, &err);
            snprintf(nm, sizeof nm, "blk.%d.attn_gate.weight", l); quant_rows(&tm, nm, val_dim, dim, 4, w_dnz + (size_t)di * rbd * val_dim, s_z, &err);
            snprintf(nm, sizeof nm, "blk.%d.ssm_out.weight", l); quant_rows(&tm, nm, dim, val_dim, 4, w_dnout + (size_t)di * rbv * dim, s_out, &err);
            snprintf(nm, sizeof nm, "blk.%d.ssm_beta.weight", l); quant_rows(&tm, nm, nv, dim, 4, w_dnpb + (size_t)di * rbd * nv, s_b, &err);
            snprintf(nm, sizeof nm, "blk.%d.ssm_alpha.weight", l); quant_rows(&tm, nm, nv, dim, 4, w_dnpa + (size_t)di * rbd * nv, s_a, &err);
            snprintf(nm, sizeof nm, "blk.%d.ssm_conv1d.weight", l); copy_f32(&tm, nm, conv_dim, ck, w_conv + (size_t)di * conv_dim * ck, &err);
            snprintf(nm, sizeof nm, "blk.%d.ssm_dt.bias", l); copy_f32(&tm, nm, 1, nv, w_dt + (size_t)di * nv, &err);
            snprintf(nm, sizeof nm, "blk.%d.ssm_a", l); copy_f32(&tm, nm, 1, nv, w_al + (size_t)di * nv, &err);
            if (!err) for (int i = 0; i < nv; i++) w_al[(size_t)di * nv + i] = logf(-w_al[(size_t)di * nv + i]);
            di++;
        } else {
            float *s_q = sc_fa; sc_fa += qg_dim; float *s_k = sc_fa; sc_fa += kv_dim; float *s_v = sc_fa; sc_fa += kv_dim; float *s_o = sc_fa; sc_fa += dim;
            snprintf(nm, sizeof nm, "blk.%d.attn_q.weight", l); quant_rows(&tm, nm, qg_dim, dim, 4, w_faq + (size_t)fi * rbd * qg_dim, s_q, &err);
            snprintf(nm, sizeof nm, "blk.%d.attn_k.weight", l); quant_rows(&tm, nm, kv_dim, dim, 4, w_fak + (size_t)fi * rbd * kv_dim, s_k, &err);
            snprintf(nm, sizeof nm, "blk.%d.attn_v.weight", l); quant_rows(&tm, nm, kv_dim, dim, 4, w_fav + (size_t)fi * rbd * kv_dim, s_v, &err);
            snprintf(nm, sizeof nm, "blk.%d.attn_output.weight", l); quant_rows(&tm, nm, dim, q_dim, 4, w_fao + (size_t)fi * rbq * dim, s_o, &err);
            fi++;
        }
    }
    for (int l = 0; l < L && !err; l++) {
        float *s_g = sc_ffn; sc_ffn += hidden; float *s_u = sc_ffn; sc_ffn += hidden; float *s_d = sc_ffn; sc_ffn += dim;
        snprintf(nm, sizeof nm, "blk.%d.ffn_gate.weight", l); quant_rows(&tm, nm, hidden, dim, 4, w_ffg + (size_t)l * rbd * hidden, s_g, &err);
        snprintf(nm, sizeof nm, "blk.%d.ffn_up.weight", l); quant_rows(&tm, nm, hidden, dim, 4, w_ffu + (size_t)l * rbd * hidden, s_u, &err);
        snprintf(nm, sizeof nm, "blk.%d.ffn_down.weight", l); quant_rows(&tm, nm, dim, hidden, 4, w_ffd + (size_t)l * rbh * dim, s_d, &err);
    }

    /* norms: GGUF already holds the +1 form the runtime uses */
    for (int l = 0; l < L && !err; l++) { snprintf(nm, sizeof nm, "blk.%d.attn_norm.weight", l); copy_f32(&tm, nm, 1, dim, np, &err); np += dim; }
    for (int l = 0; l < L && !err; l++) { snprintf(nm, sizeof nm, "blk.%d.post_attention_norm.weight", l); copy_f32(&tm, nm, 1, dim, np, &err); np += dim; }
    if (!err) { copy_f32(&tm, "output_norm.weight", 1, dim, np, &err); np += dim; }
    for (int l = 0; l < L && !err; l++) {
        if (!layer_full[l]) { snprintf(nm, sizeof nm, "blk.%d.ssm_norm.weight", l); copy_f32(&tm, nm, 1, vd, np, &err); np += vd; }
        else {
            snprintf(nm, sizeof nm, "blk.%d.attn_q_norm.weight", l); copy_f32(&tm, nm, 1, hd, np, &err); np += hd;
            snprintf(nm, sizeof nm, "blk.%d.attn_k_norm.weight", l); copy_f32(&tm, nm, 1, hd, np, &err); np += hd;
        }
    }

    free(td);
    if (err) { free(img); return -4; }
    *out_img = img; *out_size = total;
    return 0;
}
