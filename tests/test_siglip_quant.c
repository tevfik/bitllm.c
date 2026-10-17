/**
 * @file test_siglip_quant.c
 * @brief Quantised SigLIP2 vision weights: INT8 per-row, INT4 per-group (g128) and INT4 g128 + Hadamard rotation.
 *
 * A synthetic vision payload (header + [Hadamard signs] + the 12-layer weight stream, layout in eif_siglip.h) is
 * built in memory, loaded with eif_siglip_init_from_buffer and each quantised projection is dequantised with
 * eif_siglip_dequant_weight. The reference is computed here independently (explicit Sylvester matrix for the rotation).
 * Only layer 0 `q` and layer 11 `fc2` carry random codes; everything else is zero.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eif_siglip.h"

#define H 768
#define I 3072
#define L 12

static int g_fail = 0;
#define CHECK(c, msg)                                                   \
    do {                                                                \
        if (c) printf("  PASS  %s\n", msg);                             \
        else { printf("  FAIL  %s\n", msg); g_fail++; }                 \
    } while (0)

static unsigned s_seed = 12345;
static float frand(void) { s_seed = s_seed * 1103515245u + 12345u; return ((s_seed >> 8) & 0xFFFF) / 32768.0f - 1.0f; }

static size_t mat_bytes(int qtype, int g, int n, int k)
{
    if (qtype == EIF_SIGLIP_Q_INT8) return (size_t)n * k + (size_t)n * 4;
    if (qtype == EIF_SIGLIP_Q_INT4) { int gg = g > 0 ? g : k; return (size_t)n * k / 2 + (size_t)n * (k / gg) * 4; }
    return (size_t)n * k * 4;
}

/* Sylvester Hadamard, normalised: H[i][j] = (-1)^popcount(i&j) / sqrt(b) */
static float had(int i, int j, int b) { return (__builtin_popcount(i & j) & 1 ? -1.0f : 1.0f) / sqrtf((float)b); }

/* Fill one stored matrix with random codes/scales; return the reference dequantised value of rows 0..2 in `ref`. */
static void fill_mat(uint8_t *p, int qtype, int g, int n, int k, const float *signs, int block, float *ref)
{
    int gg = (qtype == EIF_SIGLIP_Q_INT4 && g > 0) ? g : k;
    int ng = k / gg;
    float *wrot = (float *)malloc((size_t)3 * k * sizeof(float));
    if (qtype == EIF_SIGLIP_Q_INT8) {
        int8_t *c = (int8_t *)p;
        float *s = (float *)(p + (size_t)n * k);
        for (int r = 0; r < n; r++) {
            s[r] = 0.002f + 0.001f * fabsf(frand());
            for (int i = 0; i < k; i++) c[(size_t)r * k + i] = (int8_t)(frand() * 127.0f);
        }
        for (int r = 0; r < 3; r++)
            for (int i = 0; i < k; i++) wrot[(size_t)r * k + i] = c[(size_t)r * k + i] * s[r];
    } else {
        uint8_t *c = p;
        float *s = (float *)(p + (size_t)n * k / 2);
        for (int r = 0; r < n; r++) {
            for (int q = 0; q < ng; q++) s[(size_t)r * ng + q] = 0.003f + 0.002f * fabsf(frand());
            for (int i = 0; i < k; i += 2) {
                int lo = (int)(frand() * 7.99f), hi = (int)(frand() * 7.99f); /* -7..7 */
                c[(size_t)r * k / 2 + i / 2] = (uint8_t)((lo & 0x0F) | ((hi & 0x0F) << 4));
                if (r < 3) {
                    wrot[(size_t)r * k + i] = lo * s[(size_t)r * ng + i / gg];
                    wrot[(size_t)r * k + i + 1] = hi * s[(size_t)r * ng + (i + 1) / gg];
                }
            }
        }
    }
    for (int r = 0; r < 3; r++) {
        for (int i = 0; i < k; i++) {
            if (!signs) { ref[(size_t)r * k + i] = wrot[(size_t)r * k + i]; continue; }
            /* un-rotate: W = (W' H) D, block-wise */
            int b0 = (i / block) * block, j = i - b0;
            float acc = 0;
            for (int t = 0; t < block; t++) acc += wrot[(size_t)r * k + b0 + t] * had(t, j, block);
            ref[(size_t)r * k + i] = acc * signs[i];
        }
    }
    free(wrot);
}

static void run_mode(const char *name, int qtype, int g, int hadamard)
{
    printf("%s\n", name);
    size_t per_layer = 2 * H * 4 + mat_bytes(qtype, g, H, H) * 4 + 4 * H * 4 /* q,k,v,out biases */ + 2 * H * 4 +
                       mat_bytes(qtype, g, I, H) + I * 4 + mat_bytes(qtype, g, H, I) + H * 4;
    size_t total = sizeof(eif_siglip_header_t) + (hadamard ? (H + I) * 4 : 0) + (size_t)H * 3 * 256 * 4 + H * 4 +
                   (size_t)1024 * H * 4 + per_layer * L + 2 * H * 4 + (size_t)576 * 12288 * 4;
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    eif_siglip_header_t *hdr = (eif_siglip_header_t *)buf;
    hdr->magic = EIF_SIGLIP_MAGIC; hdr->version = 1; hdr->patch_size = 16; hdr->image_size = 512;
    hdr->num_layers = L; hdr->hidden_size = H; hdr->num_heads = 12; hdr->head_dim = 64; hdr->intermediate_dim = I;
    hdr->num_patches = 1024; hdr->conn_in_dim = 12288; hdr->conn_out_dim = 576; hdr->qtype = qtype;
    hdr->reserved[0] = (uint32_t)g; hdr->reserved[1] = (uint32_t)hadamard;
    uint8_t *p = buf + sizeof(*hdr);
    float *sg768 = NULL, *sg3072 = NULL;
    if (hadamard) {
        sg768 = (float *)p; for (int i = 0; i < H; i++) sg768[i] = frand() > 0 ? 1.0f : -1.0f; p += H * 4;
        sg3072 = (float *)p; for (int i = 0; i < I; i++) sg3072[i] = frand() > 0 ? 1.0f : -1.0f; p += I * 4;
    }
    p += ((size_t)H * 3 * 256 + H + (size_t)1024 * H) * 4; /* patch embedding, position embedding (zeros) */

    float *ref_q = (float *)malloc((size_t)3 * H * sizeof(float));
    float *ref_fc2 = (float *)malloc((size_t)3 * I * sizeof(float));
    for (int l = 0; l < L; l++) {
        p += 2 * H * 4;                              /* ln1 */
        if (l == 0) fill_mat(p, qtype, g, H, H, sg768, 256, ref_q);
        p += mat_bytes(qtype, g, H, H) + H * 4;      /* q, bias */
        p += mat_bytes(qtype, g, H, H) + H * 4;      /* k */
        p += mat_bytes(qtype, g, H, H) + H * 4;      /* v */
        p += mat_bytes(qtype, g, H, H) + H * 4;      /* out */
        p += 2 * H * 4;                              /* ln2 */
        p += mat_bytes(qtype, g, I, H) + I * 4;      /* fc1 */
        if (l == L - 1) fill_mat(p, qtype, g, H, I, sg3072, 1024, ref_fc2);
        p += mat_bytes(qtype, g, H, I) + H * 4;      /* fc2 */
    }

    eif_siglip_t model;
    CHECK(eif_siglip_init_from_buffer(&model, buf, total) == 0, "payload loads");
    float *out = (float *)malloc((size_t)H * I * sizeof(float));
    float err = 0;
    CHECK(eif_siglip_dequant_weight(&model, 0, 0, out) == 0, "dequant layer 0 q");
    for (int i = 0; i < 3 * H; i++) err = fmaxf(err, fabsf(out[i] - ref_q[i]));
    CHECK(err < 1e-5f, "layer 0 q rows 0-2 match the independent reference");
    err = 0;
    CHECK(eif_siglip_dequant_weight(&model, L - 1, 5, out) == 0, "dequant layer 11 fc2");
    for (int i = 0; i < 3 * I; i++) err = fmaxf(err, fabsf(out[i] - ref_fc2[i]));
    CHECK(err < 1e-5f, "layer 11 fc2 rows 0-2 match the independent reference");
    CHECK(eif_siglip_dequant_weight(&model, L, 0, out) < 0 && eif_siglip_dequant_weight(&model, 0, 6, out) < 0,
          "out-of-range layer / matrix rejected");
    eif_siglip_free(&model);
    free(out); free(ref_q); free(ref_fc2); free(buf);
}

int main(void)
{
    printf("test_siglip_quant\n");
    run_mode("INT8 per-row", EIF_SIGLIP_Q_INT8, 0, 0);
    run_mode("INT4 g128", EIF_SIGLIP_Q_INT4, 128, 0);
    run_mode("INT4 g128 + Hadamard", EIF_SIGLIP_Q_INT4, 128, 1);

    /* unsupported qtype and bad group are rejected, not read as FP32 */
    eif_siglip_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = EIF_SIGLIP_MAGIC; hdr.qtype = 7;
    eif_siglip_t m;
    CHECK(eif_siglip_init_from_buffer(&m, &hdr, sizeof(hdr)) < 0, "unknown qtype rejected");
    hdr.qtype = EIF_SIGLIP_Q_INT4; hdr.reserved[0] = 100;
    CHECK(eif_siglip_init_from_buffer(&m, &hdr, sizeof(hdr)) < 0, "INT4 group that does not divide 768/3072 rejected");

    printf(g_fail ? "FAILED (%d)\n" : "ALL TESTS PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
