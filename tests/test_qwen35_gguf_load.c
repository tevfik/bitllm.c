/* A llama.cpp style Qwen3.5 GGUF must load through qwen35_load(): the repacked image keeps the .eifm scale-buffer
 * order (all DeltaNet, then all FullAttn, then all FFN), INT4 row scales, A_log = log(-ssm_a), norms unchanged. */
#include "eif_qwen35.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { L = 24, DIM = 32, HID = 64, NH = 2, NKV = 1, HD = 16, NK = 2, KD = 16, VD = 16, CK = 4, VOC = 16,
       CONV = NK * KD * 2 + NH * VD, QG = NH * HD * 2, KVD = NKV * HD };

static FILE *g;
static void u32(uint32_t v) { fwrite(&v, 4, 1, g); }
static void u64(uint64_t v) { fwrite(&v, 8, 1, g); }
static void str(const char *s) { u64(strlen(s)); fwrite(s, 1, strlen(s), g); }
static void kv_u32(const char *k, uint32_t v) { str(k); u32(4); u32(v); }

typedef struct { char name[64]; int rows, cols; float *d; } tens_t;
static tens_t T[400];
static int nT;
static unsigned seed = 7;

static float *mk(int rows, int cols, float scale, const char *fmt, int l)
{
    tens_t *t = &T[nT++];
    snprintf(t->name, sizeof t->name, fmt, l);
    t->rows = rows; t->cols = cols;
    t->d = malloc((size_t)rows * cols * sizeof(float));
    for (long i = 0; i < (long)rows * cols; i++) {
        seed = seed * 1664525u + 1013904223u;
        t->d[i] = scale * ((float)(seed >> 8) / 8388608.0f - 1.0f);
    }
    return t->d;
}

static int fails;
#define CHECK(c, m) do { if (!(c)) { printf("FAIL: %s\n", m); fails++; } } while (0)

static float row_absmax(const float *d, int cols) { float m = 1e-8f; for (int i = 0; i < cols; i++) m = fmaxf(m, fabsf(d[i])); return m; }

int main(void)
{
    const char *path = "test_qwen35_gguf_load.gguf";
    mk(VOC, DIM, 0.5f, "token_embd.weight", 0);
    mk(1, DIM, 1.0f, "output_norm.weight", 0);
    float *attn_q[L] = {0}, *ssm_z[L] = {0}, *up[L] = {0}, *ssm_a[L] = {0};
    for (int l = 0; l < L; l++) {
        int full = (l + 1) % 4 == 0;
        mk(1, DIM, 1.0f, "blk.%d.attn_norm.weight", l);
        mk(1, DIM, 1.0f, "blk.%d.post_attention_norm.weight", l);
        up[l] = mk(HID, DIM, 0.1f * (l + 1), "blk.%d.ffn_up.weight", l);
        mk(HID, DIM, 0.3f, "blk.%d.ffn_gate.weight", l);
        mk(DIM, HID, 0.2f, "blk.%d.ffn_down.weight", l);
        if (!full) {
            mk(CONV, DIM, 0.4f, "blk.%d.attn_qkv.weight", l);
            ssm_z[l] = mk(NH * VD, DIM, 0.05f * (l + 1), "blk.%d.attn_gate.weight", l);
            mk(DIM, NH * VD, 0.3f, "blk.%d.ssm_out.weight", l);
            mk(NH, DIM, 0.2f, "blk.%d.ssm_beta.weight", l);
            mk(NH, DIM, 0.2f, "blk.%d.ssm_alpha.weight", l);
            mk(CONV, CK, 0.5f, "blk.%d.ssm_conv1d.weight", l);
            mk(1, NH, 0.5f, "blk.%d.ssm_dt.bias", l);
            ssm_a[l] = mk(1, NH, 1.0f, "blk.%d.ssm_a", l);
            for (int i = 0; i < NH; i++) ssm_a[l][i] = -(1.5f + ssm_a[l][i]);
            mk(1, VD, 1.0f, "blk.%d.ssm_norm.weight", l);
        } else {
            attn_q[l] = mk(QG, DIM, 0.07f * (l + 1), "blk.%d.attn_q.weight", l);
            mk(KVD, DIM, 0.3f, "blk.%d.attn_k.weight", l);
            mk(KVD, DIM, 0.3f, "blk.%d.attn_v.weight", l);
            mk(DIM, NH * HD, 0.3f, "blk.%d.attn_output.weight", l);
            mk(1, HD, 1.0f, "blk.%d.attn_q_norm.weight", l);
            mk(1, HD, 1.0f, "blk.%d.attn_k_norm.weight", l);
        }
    }
    g = fopen(path, "wb");
    if (!g) return 1;
    u32(0x46554747); u32(3); u64(nT); u64(15);
    str("general.architecture"); u32(8); str("qwen35");
    kv_u32("qwen35.block_count", L); kv_u32("qwen35.embedding_length", DIM); kv_u32("qwen35.feed_forward_length", HID);
    kv_u32("qwen35.attention.head_count", NH); kv_u32("qwen35.attention.head_count_kv", NKV);
    kv_u32("qwen35.attention.key_length", HD); kv_u32("qwen35.context_length", 64);
    kv_u32("qwen35.ssm.conv_kernel", CK); kv_u32("qwen35.ssm.state_size", KD); kv_u32("qwen35.ssm.group_count", NK);
    kv_u32("qwen35.ssm.time_step_rank", NH); kv_u32("qwen35.ssm.inner_size", NH * VD);
    kv_u32("qwen35.full_attention_interval", 4);
    str("qwen35.attention.layer_norm_rms_epsilon"); u32(6); { float e = 1e-6f; fwrite(&e, 4, 1, g); }
    uint64_t off = 0;
    for (int i = 0; i < nT; i++) {
        str(T[i].name);
        int two = T[i].rows > 1;
        u32(two ? 2 : 1); u64(T[i].cols); if (two) u64(T[i].rows);
        u32(0); u64(off);
        off += ((uint64_t)T[i].rows * T[i].cols * 4 + 31) & ~31ull;
    }
    for (long p = ftell(g); p % 32; p++) fputc(0, g);
    for (int i = 0; i < nT; i++) {
        size_t n = (size_t)T[i].rows * T[i].cols * 4, pad = ((n + 31) & ~31ull) - n;
        fwrite(T[i].d, 1, n, g);
        for (size_t p = 0; p < pad; p++) fputc(0, g);
    }
    fclose(g);

    qwen35_t m;
    qwen35_config_t probe;
    memset(&probe, 0, sizeof probe);
    probe.dim = DIM; probe.hidden_dim = HID; probe.n_layers = L; probe.n_heads = NH; probe.n_kv_heads = NKV; probe.head_dim = HD;
    probe.vocab_size = VOC; probe.max_seq_len = 64; probe.n_delta = 18; probe.n_full = 6; probe.lin_n_k_heads = NK;
    probe.lin_k_head_dim = KD; probe.lin_n_v_heads = NH; probe.lin_v_head_dim = VD; probe.conv_k_size = CK;
    size_t state_sz = qwen35_state_size(&probe) + 4096;
    void *state = malloc(state_sz);
    int rc = qwen35_load(&m, path, state, state_sz);
    remove(path);
    CHECK(rc == 0, "qwen35_load accepts the GGUF");
    if (rc != 0) return 1;
    const qwen35_weights_t *W = &m.weights;
    CHECK(m.config.plain_linear == 1 && m.config.quant_lm == QWEN35_QTYPE_INT4, "plain INT4 image");
    CHECK(m.config.n_delta == 18 && m.config.n_full == 6, "layer counts");

    /* scales: row scale = absmax / qmax, found where the loader looks for them (this is what mis-ordered once) */
    int di = 0, fi = 0;
    for (int l = 0; l < L; l++) {
        float want_up = row_absmax(up[l], DIM) / 7.0f;
        CHECK(fabsf(W->ffn[l].up_s[0] - want_up) < 1e-6f * (1 + want_up), "ffn up scale");
        if ((l + 1) % 4 == 0) {
            float want = row_absmax(attn_q[l], DIM) / 7.0f;
            CHECK(fabsf(W->full[fi].q_s[0] - want) < 1e-6f * (1 + want), "full-attention q scale");
            fi++;
        } else {
            float want = row_absmax(ssm_z[l], DIM) / 7.0f;
            CHECK(fabsf(W->delta[di].z_s[0] - want) < 1e-6f * (1 + want), "deltanet z scale");
            CHECK(fabsf(W->delta[di].A_log[0] - logf(-ssm_a[l][0])) < 1e-6f, "A_log = log(-ssm_a)");
            di++;
        }
    }
    CHECK(fabsf(W->embed_s[3] - row_absmax(T[0].d + 3 * DIM, DIM) / 127.0f) < 1e-7f, "embedding row scale");
    CHECK(W->rms_final[5] == T[1].d[5], "final norm copied as is");

    qwen35_free(&m);
    free(state);
    for (int i = 0; i < nT; i++) free(T[i].d);
    printf(fails ? "FAIL\n" : "PASS\n");
    return fails ? 1 : 0;
}
