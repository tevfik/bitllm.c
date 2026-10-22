/* Two-token logits of a synthetic GGUF vs a float reference for both RoPE layouts:
 *   arch "qwen2": HuggingFace half-split Q/K rows (pairs j, j+HD/2)
 *   arch "llama": llama.cpp stores Q/K rows permuted for adjacent-pair RoPE (pairs 2j, 2j+1)
 * Position 1 is where the layouts differ; a loader that rotates the wrong pairs gives a wrong logit vector
 * (SmolLM2 / Llama GGUF used to answer with <|im_end|> for every prompt). */
#include "eif_tinyllm.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { DIM = 64, HID = 128, NH = 4, NKV = 2, HD = DIM / NH, KV = NKV * HD, VOC = 64 };
static const float THETA = 2.0f; /* small base: large rotation angles at position 1 */

static FILE *g;
static void u32(uint32_t v) { fwrite(&v, 4, 1, g); }
static void u64(uint64_t v) { fwrite(&v, 8, 1, g); }
static void str(const char *s) { u64(strlen(s)); fwrite(s, 1, strlen(s), g); }
typedef struct { char name[48]; int rows, cols; float *d; } tens_t;
static tens_t T[16];
static int nT;

static float *mk(const char *name, int rows, int cols, float scale, unsigned seed)
{
    float *d = malloc((size_t)rows * cols * sizeof(float));
    for (long i = 0; i < (long)rows * cols; i++) { seed = seed * 1664525u + 1013904223u; d[i] = scale * ((float)(seed >> 8) / 8388608.0f - 1.0f); }
    snprintf(T[nT].name, sizeof T[nT].name, "%s", name);
    T[nT].rows = rows; T[nT].cols = cols; T[nT].d = d; nT++;
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
static void rope_half(float *x, int nh, int pos)
{
    for (int h = 0; h < nh; h++) for (int j = 0; j < HD / 2; j++) {
        float a = pos * powf(THETA, -2.0f * j / HD), c = cosf(a), s = sinf(a);
        float *p = x + h * HD, x0 = p[j], x1 = p[j + HD / 2];
        p[j] = x0 * c - x1 * s; p[j + HD / 2] = x1 * c + x0 * s;
    }
}
/* HF row j of a head -> GGUF row 2j, HF row j+HD/2 -> GGUF row 2j+1 (llama.cpp convert permute) */
static void permute_rows(float *dst, const float *src, int nh)
{
    for (int h = 0; h < nh; h++) for (int j = 0; j < HD / 2; j++) {
        memcpy(dst + ((size_t)h * HD + 2 * j) * DIM, src + ((size_t)h * HD + j) * DIM, DIM * sizeof(float));
        memcpy(dst + ((size_t)h * HD + 2 * j + 1) * DIM, src + ((size_t)h * HD + j + HD / 2) * DIM, DIM * sizeof(float));
    }
}

static double run(const char *arch)
{
    nT = 0;
    int llama = strcmp(arch, "llama") == 0;
    float *emb = mk("token_embd.weight", VOC, DIM, 0.5f, 1);
    float *an = mk("blk.0.attn_norm.weight", 1, DIM, 1.0f, 2);
    float *wq = mk("blk.0.attn_q.weight", DIM, DIM, 3.0f, 3);
    float *wk = mk("blk.0.attn_k.weight", KV, DIM, 3.0f, 4);
    float *wv = mk("blk.0.attn_v.weight", KV, DIM, 1.5f, 5);
    float *wo = mk("blk.0.attn_output.weight", DIM, DIM, 0.1f, 6);
    float *fn = mk("blk.0.ffn_norm.weight", 1, DIM, 1.0f, 7);
    float *wg = mk("blk.0.ffn_gate.weight", HID, DIM, 0.4f, 8);
    float *wd = mk("blk.0.ffn_down.weight", DIM, HID, 0.03f, 9);
    float *wu = mk("blk.0.ffn_up.weight", HID, DIM, 2.0f, 10);
    float *on = mk("output_norm.weight", 1, DIM, 1.0f, 11);
    float *out = mk("output.weight", VOC, DIM, 0.5f, 12);
    for (int i = 0; i < DIM; i++) { an[i] = 1 + 0.2f * an[i]; fn[i] = 1 + 0.2f * fn[i]; on[i] = 1 + 0.2f * on[i]; }

    /* file content: llama stores permuted copies of Q and K; the reference keeps the HF layout */
    float *wq_f = malloc(sizeof(float) * DIM * DIM), *wk_f = malloc(sizeof(float) * KV * DIM);
    if (llama) { permute_rows(wq_f, wq, NH); permute_rows(wk_f, wk, NKV); } else { memcpy(wq_f, wq, sizeof(float) * DIM * DIM); memcpy(wk_f, wk, sizeof(float) * KV * DIM); }
    float *keep_q = T[2].d, *keep_k = T[3].d;
    T[2].d = wq_f; T[3].d = wk_f;

    char path[64], key[96];
    snprintf(path, sizeof path, "test_gguf_rope_%s.gguf", arch);
    g = fopen(path, "wb");
    u32(0x46554747); u32(3); u64(nT); u64(10);
    str("general.architecture"); u32(8); str(arch);
    #define KVU(suffix, v) do { snprintf(key, sizeof key, "%s.%s", arch, suffix); str(key); u32(4); u32(v); } while (0)
    KVU("embedding_length", DIM); KVU("feed_forward_length", HID); KVU("block_count", 1);
    KVU("attention.head_count", NH); KVU("attention.head_count_kv", NKV); KVU("context_length", 32);
    snprintf(key, sizeof key, "%s.rope.freq_base", arch); str(key); u32(6); { float f = THETA; fwrite(&f, 4, 1, g); }
    str("tokenizer.ggml.tokens"); u32(9); u32(8); u64(VOC);
    for (int i = 0; i < VOC; i++) { char b[16]; snprintf(b, sizeof b, "t%d", i); str(b); }
    str("general.alignment"); u32(4); u32(32);
    uint64_t off = 0;
    for (int i = 0; i < nT; i++) {
        str(T[i].name); int two = T[i].rows > 1;
        u32(two ? 2 : 1); u64(T[i].cols); if (two) u64(T[i].rows);
        u32(0); u64(off);
        off += ((uint64_t)T[i].rows * T[i].cols * 4 + 31) & ~31ull;
    }
    for (long p = ftell(g); p % 32; p++) fputc(0, g);
    for (int i = 0; i < nT; i++) {
        size_t n = (size_t)T[i].rows * T[i].cols * 4, pad = ((n + 31) & ~31ull) - n;
        fwrite(T[i].d, 1, n, g); for (size_t p = 0; p < pad; p++) fputc(0, g);
    }
    fclose(g);

    tinyllm_config_t cfg; memset(&cfg, 0, sizeof cfg);
    char a2[64];
    if (tinyllm_read_gguf_config(path, &cfg, a2, sizeof a2) != 0) { printf("config failed\n"); return 9; }
    size_t mem = tinyllm_memory_size(&cfg), wsz = 8u << 20;
    uint8_t *buf = calloc(1, mem + wsz);
    tinyllm_t llm;
    if (tinyllm_init(&llm, &cfg, buf, mem) != 0 || tinyllm_load_gguf(&llm, path, buf + mem, wsz) != 0) { printf("load failed\n"); return 9; }
    remove(path);

    /* float reference, HF layout, two positions */
    const int toks[2] = {5, 9};
    float x[2][DIM], xb[DIM], q[2][DIM], k[2][KV], v[2][KV];
    for (int p = 0; p < 2; p++) {
        memcpy(x[p], emb + toks[p] * DIM, sizeof x[p]);
        rms(xb, x[p], an, DIM);
        mv(q[p], wq, xb, DIM, DIM); mv(k[p], wk, xb, KV, DIM); mv(v[p], wv, xb, KV, DIM);
        rope_half(q[p], NH, p); rope_half(k[p], NKV, p);
    }
    float ao[DIM], t[DIM], g1[HID], u1[HID], h[HID], lg[VOC];
    for (int hh = 0; hh < NH; hh++) {
        int kh = hh / (NH / NKV);
        double sc[2], mx = -1e30, sum = 0;
        for (int j = 0; j < 2; j++) { double d = 0; for (int e = 0; e < HD; e++) d += q[1][hh * HD + e] * k[j][kh * HD + e]; sc[j] = d / sqrt(HD); if (sc[j] > mx) mx = sc[j]; }
        for (int j = 0; j < 2; j++) { sc[j] = exp(sc[j] - mx); sum += sc[j]; }
        for (int e = 0; e < HD; e++) ao[hh * HD + e] = (float)((sc[0] * v[0][kh * HD + e] + sc[1] * v[1][kh * HD + e]) / sum);
    }
    mv(t, wo, ao, DIM, DIM);
    for (int i = 0; i < DIM; i++) x[1][i] += t[i];
    rms(xb, x[1], fn, DIM);
    mv(g1, wg, xb, HID, DIM); mv(u1, wu, xb, HID, DIM);
    for (int i = 0; i < HID; i++) h[i] = g1[i] / (1.0f + expf(-g1[i])) * u1[i];
    mv(t, wd, h, DIM, HID);
    for (int i = 0; i < DIM; i++) x[1][i] += t[i];
    rms(xb, x[1], on, DIM);
    mv(lg, out, xb, VOC, DIM);

    tinyllm_forward(&llm, toks[0], 0);
    const float *got = tinyllm_forward(&llm, toks[1], 1);
    double num = 0, den = 0;
    for (int i = 0; i < VOC; i++) { num += (got[i] - lg[i]) * (got[i] - lg[i]); den += lg[i] * lg[i]; }
    T[2].d = keep_q; T[3].d = keep_k;
    for (int i = 0; i < nT; i++) free(T[i].d);
    free(wq_f); free(wk_f); free(buf);
    return sqrt(num / den);
}

int main(void)
{
    int fail = 0;
    const char *archs[] = {"qwen2", "llama"};
    for (int i = 0; i < 2; i++) {
        double rel = run(archs[i]);
        printf("%-6s two-token logits, relative error vs float reference: %.4f  %s\n", archs[i], rel, rel < 0.05 ? "ok" : "FAIL");
        if (rel >= 0.05) fail = 1;
    }
    printf(fail ? "FAIL\n" : "PASS\n");
    return fail;
}
