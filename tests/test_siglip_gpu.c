/**
 * @file test_siglip_gpu.c
 * @brief SigLIP2 encoder on the Vulkan GPU vs the CPU path: synthetic INT8 and INT4 g128 + Hadamard models
 *        (3 tiles with 2 tiles per submit, so batching and a partial batch are covered) and, when available,
 *        the real Granite-Docling model (argv[1] or BITLLM_DOCLING_EIFM). Passes with a SKIP message when no
 *        Vulkan device is present.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eif_siglip.h"

#define H 768
#define I 3072
#define L 12
#define NP 1024
#define TILE (3 * 512 * 512)
#define TOK (64 * 576)

static int g_fail = 0;
static unsigned s_seed = 4242;
static float frand(void) { s_seed = s_seed * 1103515245u + 12345u; return ((s_seed >> 8) & 0xFFFF) / 32768.0f - 1.0f; }

static size_t mat_bytes(int qtype, int g, int n, int k)
{
    if (qtype == EIF_SIGLIP_Q_INT8) return (size_t)n * k + (size_t)n * 4;
    if (qtype == EIF_SIGLIP_Q_INT4) return (size_t)n * k / 2 + (size_t)n * (k / g) * 4;
    return (size_t)n * k * 4;
}

static float *put_f(uint8_t **p, size_t n, float base, float amp)
{
    float *f = (float *)*p;
    for (size_t i = 0; i < n; i++) f[i] = base + amp * frand();
    *p += n * sizeof(float);
    return f;
}

static void put_mat(uint8_t **p, int qtype, int g, int n, int k)
{
    const float amp = 1.4f / sqrtf((float)k); /* keeps activations O(1) */
    if (qtype == EIF_SIGLIP_Q_INT8) {
        int8_t *c = (int8_t *)*p;
        for (size_t i = 0; i < (size_t)n * k; i++) c[i] = (int8_t)(frand() * 127.0f);
        *p += (size_t)n * k;
        put_f(p, n, amp / 127.0f, 0.2f * amp / 127.0f);
    } else if (qtype == EIF_SIGLIP_Q_INT4) {
        for (size_t i = 0; i < (size_t)n * k / 2; i++) {
            int lo = (int)(frand() * 7.99f), hi = (int)(frand() * 7.99f);
            (*p)[i] = (uint8_t)((lo & 0x0F) | ((hi & 0x0F) << 4));
        }
        *p += (size_t)n * k / 2;
        put_f(p, (size_t)n * (k / g), amp / 7.0f, 0.2f * amp / 7.0f);
    } else {
        put_f(p, (size_t)n * k, 0.0f, amp);
    }
}

static uint8_t *make_payload(int qtype, int g, int had, size_t *size)
{
    size_t sz = sizeof(eif_siglip_header_t) + (had ? (H + I) * 4 : 0) + (size_t)H * H * 4 + H * 4 + (size_t)NP * H * 4;
    for (int l = 0; l < L; l++)
        sz += 2 * H * 4 + 4 * (mat_bytes(qtype, g, H, H) + H * 4) + 2 * H * 4 + mat_bytes(qtype, g, I, H) + I * 4 +
              mat_bytes(qtype, g, H, I) + H * 4;
    sz += 2 * H * 4 + (size_t)576 * 12288 * 4;
    uint8_t *buf = (uint8_t *)calloc(1, sz), *p = buf;
    if (!buf) return NULL;
    eif_siglip_header_t *h = (eif_siglip_header_t *)p;
    h->magic = EIF_SIGLIP_MAGIC; h->version = 1; h->patch_size = 16; h->image_size = 512; h->num_layers = L;
    h->hidden_size = H; h->num_heads = 12; h->head_dim = 64; h->intermediate_dim = I; h->num_patches = NP;
    h->conn_in_dim = 12288; h->conn_out_dim = 576; h->qtype = qtype; h->total_payload = (uint32_t)sz;
    h->reserved[0] = (uint32_t)g; h->reserved[1] = (uint32_t)had;
    p += sizeof(*h);
    if (had)
        for (int i = 0; i < H + I; i++) { ((float *)p)[i] = frand() < 0 ? -1.0f : 1.0f; }
    if (had) p += (H + I) * 4;
    put_f(&p, (size_t)H * H, 0.0f, 0.06f); /* patch embedding */
    put_f(&p, H, 0.0f, 0.1f);
    put_f(&p, (size_t)NP * H, 0.0f, 0.3f);
    for (int l = 0; l < L; l++) {
        put_f(&p, H, 1.0f, 0.2f); put_f(&p, H, 0.0f, 0.1f);
        for (int m = 0; m < 4; m++) { put_mat(&p, qtype, g, H, H); put_f(&p, H, 0.0f, 0.1f); }
        put_f(&p, H, 1.0f, 0.2f); put_f(&p, H, 0.0f, 0.1f);
        put_mat(&p, qtype, g, I, H); put_f(&p, I, 0.0f, 0.1f);
        put_mat(&p, qtype, g, H, I); put_f(&p, H, 0.0f, 0.1f);
    }
    put_f(&p, H, 1.0f, 0.2f); put_f(&p, H, 0.0f, 0.1f);
    put_f(&p, (size_t)576 * 12288, 0.0f, 0.01f);
    *size = sz;
    return buf;
}

static void compare(const char *name, const float *gpu, const float *cpu, size_t n)
{
    double dot = 0, ng = 0, nc = 0, maxd = 0, maxc = 0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)gpu[i] * cpu[i];
        ng += (double)gpu[i] * gpu[i];
        nc += (double)cpu[i] * cpu[i];
        double d = fabs((double)gpu[i] - cpu[i]);
        if (d > maxd || d != d) maxd = d;
        if (fabs(cpu[i]) > maxc) maxc = fabs(cpu[i]);
    }
    double cos = dot / (sqrt(ng * nc) + 1e-30);
    bool ok = cos > 0.99999 && maxd <= 1e-3 * maxc;
    printf("  %s  %-40s cos %.7f  max|d| %.3e (max|ref| %.3f)\n", ok ? "PASS" : "FAIL", name, cos, maxd, maxc);
    if (!ok) g_fail++;
}

/* Encode n tiles on the CPU, then on the GPU (batch 2), and compare. */
static void run_model(const char *name, eif_siglip_t *m, const float *tiles, int n)
{
    float *cpu = (float *)malloc((size_t)n * TOK * sizeof(float));
    float *gpu = (float *)malloc((size_t)n * TOK * sizeof(float));
    if (eif_siglip_encode_tiles(m, tiles, n, cpu) != 0) { printf("  FAIL  %s: CPU encode\n", name); g_fail++; goto out; }
    if (eif_siglip_use_gpu(m, NULL) != 0) { printf("  FAIL  %s: GPU attach\n", name); g_fail++; goto out; }
    if (eif_siglip_encode_tiles(m, tiles, n, gpu) != 0) { printf("  FAIL  %s: GPU encode\n", name); g_fail++; goto out; }
    char label[96];
    for (int t = 0; t < n; t++) {
        snprintf(label, sizeof(label), "%s tile %d", name, t);
        compare(label, gpu + (size_t)t * TOK, cpu + (size_t)t * TOK, TOK);
    }
    if (eif_siglip_encode_tile(m, tiles, gpu) != 0) { printf("  FAIL  %s: single tile\n", name); g_fail++; goto out; }
    snprintf(label, sizeof(label), "%s single-tile API", name);
    compare(label, gpu, cpu, TOK);
out:
    free(cpu);
    free(gpu);
}

static void run_synthetic(const char *name, int qtype, int g, int had, const float *tiles, int n)
{
    size_t sz = 0;
    uint8_t *buf = make_payload(qtype, g, had, &sz);
    eif_siglip_t m;
    if (!buf || eif_siglip_init_from_buffer(&m, buf, sz) != 0) {
        printf("  FAIL  %s: payload\n", name);
        g_fail++;
        free(buf);
        return;
    }
    run_model(name, &m, tiles, n);
    eif_siglip_free(&m);
    free(buf);
}

int main(int argc, char **argv)
{
    printf("====================================================\n");
    printf("  Test: SigLIP2 encoder, Vulkan GPU vs CPU\n");
    printf("====================================================\n");
    eif_gpu_context_t probe;
    if (!eif_gpu_init(&probe)) {
        printf("  [SKIP] No Vulkan device (%s): GPU encoder not tested.\n", probe.device_name);
        return 0;
    }
    printf("  Device: %s\n", probe.device_name);
    eif_gpu_cleanup(&probe);
    /* the CPU reference is the exact FP32-activation path (the i8mm W8A8 path also quantises activations) */
#if defined(_WIN32)
    _putenv_s("BITLLM_GPU_TILE_BATCH", "2");
    _putenv_s("BITLLM_VIT_FP32", "1");
#else
    setenv("BITLLM_GPU_TILE_BATCH", "2", 1);
    setenv("BITLLM_VIT_FP32", "1", 1);
#endif

    const int n = 3;
    float *tiles = (float *)malloc((size_t)n * TILE * sizeof(float));
    for (size_t i = 0; i < (size_t)n * TILE; i++) tiles[i] = frand();
    for (size_t i = 0; i < TILE; i++) tiles[2 * (size_t)TILE + i] = (i % 512 < 256) ? 0.9f : -0.4f; /* flat tile */

    run_synthetic("synthetic int8", EIF_SIGLIP_Q_INT8, 0, 0, tiles, n);
    run_synthetic("synthetic int4-g128-had", EIF_SIGLIP_Q_INT4, 128, 1, tiles, n);

    const char *model = argc > 1 ? argv[1] : getenv("BITLLM_DOCLING_EIFM");
    FILE *f = model ? fopen(model, "rb") : NULL;
    if (!f) {
        printf("  [SKIP] real model: pass the Granite-Docling .eifm as argv[1] or BITLLM_DOCLING_EIFM\n");
    } else {
        uint32_t hdr[12] = {0};
        size_t got = fread(hdr, sizeof(uint32_t), 12, f);
        fclose(f);
        eif_siglip_t m;
        if (got != 12 || hdr[0] != 0x53564C4D || hdr[10] == 0 ||
            eif_siglip_load_from_file(&m, model, hdr[6], hdr[10]) != 0) {
            printf("  FAIL  real model: load %s\n", model);
            g_fail++;
        } else {
            run_model("real model", &m, tiles, n);
            eif_siglip_free(&m);
        }
    }
    free(tiles);
    printf("%s\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
    return g_fail ? 1 : 0;
}
