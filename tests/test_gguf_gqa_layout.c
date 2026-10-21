/* A GQA model whose hidden_dim != dim (the Qwen2.5 shape) must give the same logits as a float
 * reference after tinyllm_load_gguf: the loader's INT8 row-scale layout has to match the forward pass. */
#include "eif_tinyllm.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { DIM = 64, HID = 128, NH = 4, NKV = 2, HD = DIM / NH, KV = NKV * HD, VOC = 64 };

static FILE *g;
static void u32(uint32_t v) { fwrite(&v, 4, 1, g); }
static void u64(uint64_t v) { fwrite(&v, 8, 1, g); }
static void str(const char *s) { u64(strlen(s)); fwrite(s, 1, strlen(s), g); }
static void kv_u32(const char *k, uint32_t v) { str(k); u32(4); u32(v); }

typedef struct { const char *name; int rows, cols; float *d; } tens_t;
static tens_t T[16];
static int nT;

static float *mk(const char *name, int rows, int cols, float scale, unsigned seed)
{
    float *d = malloc((size_t)rows * cols * sizeof(float));
    for (long i = 0; i < (long)rows * cols; i++) {
        seed = seed * 1664525u + 1013904223u;
        d[i] = scale * ((float)(seed >> 8) / 8388608.0f - 1.0f);
    }
    T[nT++] = (tens_t){name, rows, cols, d};
    return d;
}

static void mv(float *y, const float *w, const float *x, int rows, int cols)
{
    for (int r = 0; r < rows; r++) { double s = 0; for (int c = 0; c < cols; c++) s += w[(size_t)r * cols + c] * x[c]; y[r] = (float)s; }
}
static void rms(float *y, const float *x, const float *w, int n)
{
    double ss = 0; for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float k = 1.0f / sqrtf((float)(ss / n) + 1e-5f);
    for (int i = 0; i < n; i++) y[i] = x[i] * k * w[i];
}

int main(void)
{
    const char *path = "test_gguf_gqa_layout.gguf";
    float *emb = mk("token_embd.weight", VOC, DIM, 0.5f, 1);
    float *an = mk("blk.0.attn_norm.weight", 1, DIM, 1.0f, 2);
    float *wq = mk("blk.0.attn_q.weight", DIM, DIM, 0.3f, 3);
    float *wk = mk("blk.0.attn_k.weight", KV, DIM, 0.3f, 4);
    float *wv = mk("blk.0.attn_v.weight", KV, DIM, 1.5f, 5);
    float *wo = mk("blk.0.attn_output.weight", DIM, DIM, 0.1f, 6);
    float *fn = mk("blk.0.ffn_norm.weight", 1, DIM, 1.0f, 7);
    float *wg = mk("blk.0.ffn_gate.weight", HID, DIM, 0.4f, 8);
    float *wd = mk("blk.0.ffn_down.weight", DIM, HID, 0.03f, 9);
    float *wu = mk("blk.0.ffn_up.weight", HID, DIM, 2.0f, 10);
    float *on = mk("output_norm.weight", 1, DIM, 1.0f, 11);
    float *out = mk("output.weight", VOC, DIM, 0.5f, 12);
    for (int i = 0; i < DIM; i++) { an[i] = 1 + 0.2f * an[i]; fn[i] = 1 + 0.2f * fn[i]; on[i] = 1 + 0.2f * on[i]; }

    g = fopen(path, "wb");
    if (!g) return 1;
    u32(0x46554747); u32(3); u64(nT); u64(10);
    str("general.architecture"); u32(8); str("qwen2");
    kv_u32("qwen2.embedding_length", DIM); kv_u32("qwen2.feed_forward_length", HID);
    kv_u32("qwen2.block_count", 1); kv_u32("qwen2.attention.head_count", NH);
    kv_u32("qwen2.attention.head_count_kv", NKV); kv_u32("qwen2.context_length", 32);
    str("qwen2.rope.freq_base"); u32(6); { float f = 1e6f; fwrite(&f, 4, 1, g); }
    str("tokenizer.ggml.tokens"); u32(9); u32(8); u64(VOC);
    for (int i = 0; i < VOC; i++) { char b[16]; snprintf(b, sizeof b, "t%d", i); str(b); }
    str("general.alignment"); u32(4); u32(32);
    uint64_t off = 0;
    for (int i = 0; i < nT; i++) {
        str(T[i].name);
        int two = T[i].rows > 1;
        u32(two ? 2 : 1); u64(T[i].cols); if (two) u64(T[i].rows);
        u32(0); u64(off);
        off += ((uint64_t)T[i].rows * T[i].cols * 4 + 31) & ~31ull;
    }
    long pos = ftell(g);
    for (long p = pos; p % 32; p++) fputc(0, g);
    for (int i = 0; i < nT; i++) {
        size_t n = (size_t)T[i].rows * T[i].cols * 4, pad = ((n + 31) & ~31ull) - n;
        fwrite(T[i].d, 1, n, g);
        for (size_t p = 0; p < pad; p++) fputc(0, g);
    }
    fclose(g);

    tinyllm_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    char arch[64];
    if (tinyllm_read_gguf_config(path, &cfg, arch, sizeof arch) != 0) { printf("config failed\n"); return 1; }
    size_t mem = tinyllm_memory_size(&cfg), wsz = 8u << 20;
    uint8_t *buf = calloc(1, mem + wsz);
    tinyllm_t llm;
    if (tinyllm_init(&llm, &cfg, buf, mem) != 0 || tinyllm_load_gguf(&llm, path, buf + mem, wsz) != 0) { printf("load failed\n"); return 1; }
    remove(path);

    const int tok = 5;
    float x[DIM], xb[DIM], q[DIM], v[KV], ao[DIM], t[DIM], g1[HID], u1[HID], h[HID], lg[VOC];
    memcpy(x, emb + tok * DIM, sizeof x);
    rms(xb, x, an, DIM);
    mv(v, wv, xb, KV, DIM);
    (void)q; (void)wq; (void)wk;
    for (int hh = 0; hh < NH; hh++) memcpy(ao + hh * HD, v + (hh / (NH / NKV)) * HD, HD * sizeof(float));
    mv(t, wo, ao, DIM, DIM);
    for (int i = 0; i < DIM; i++) x[i] += t[i];
    rms(xb, x, fn, DIM);
    mv(g1, wg, xb, HID, DIM); mv(u1, wu, xb, HID, DIM);
    for (int i = 0; i < HID; i++) h[i] = g1[i] / (1.0f + expf(-g1[i])) * u1[i];
    mv(t, wd, h, DIM, HID);
    for (int i = 0; i < DIM; i++) x[i] += t[i];
    rms(xb, x, on, DIM);
    mv(lg, out, xb, VOC, DIM);

    const float *got = tinyllm_forward(&llm, tok, 0);
    if (!got) { printf("forward failed\n"); return 1; }
    double num = 0, den = 0;
    for (int i = 0; i < VOC; i++) { num += (got[i] - lg[i]) * (got[i] - lg[i]); den += lg[i] * lg[i]; }
    double rel = sqrt(num / den);
    printf("GQA GGUF logits relative error vs float reference: %.4f\n", rel);
    for (int i = 0; i < nT; i++) free(T[i].d);
    free(buf);
    if (rel > 0.05) { printf("FAIL\n"); return 1; }
    printf("PASS\n");
    return 0;
}
