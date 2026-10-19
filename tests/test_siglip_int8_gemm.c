/**
 * @file test_siglip_int8_gemm.c
 * @brief INT8 ViT projection kernel (i8mm) against an FP32 reference.
 *
 * Exactness: the kernel must equal the FP32 product of the dequantised (per-row/group quantised) operands.
 * Accuracy: with one outlier channel, per-group activation scales must stay much closer to the exact FP32
 * product than per-row scales (the fc2 failure mode).
 */
#include "eif_siglip.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); failures++; } } while (0)

static uint32_t rng = 12345u;
static float frand(void) { rng = rng * 1664525u + 1013904223u; return ((rng >> 8) & 0xFFFF) / 32768.0f - 1.0f; }

static float gelu(float v) { return 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v))); }

/* quantise row segments exactly like the kernel and return the dequantised copy */
static void fake_quant(const float *A, int M, int K, int G, float *out)
{
    for (int m = 0; m < M; m++)
        for (int g = 0; g < K / G; g++) {
            float amax = 0.0f;
            for (int k = g * G; k < (g + 1) * G; k++) amax = fmaxf(amax, fabsf(A[m * K + k]));
            float s = amax > 0.0f ? amax / 127.0f : 1.0f;
            for (int k = g * G; k < (g + 1) * G; k++) {
                float q = A[m * K + k] / s;
                int qi = (int)(q + (q >= 0.0f ? 0.5f : -0.5f));
                qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
                out[m * K + k] = qi * s;
            }
        }
}

static void ref_gemm(const float *A, const int8_t *W, const float *sw, const float *b, float *C, int M, int N, int K, int mode)
{
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++) acc += (double)A[m * K + k] * W[n * K + k];
            float v = (float)acc * sw[n] + b[n];
            if (mode == 1) C[m * N + n] = gelu(v);
            else if (mode == 2) C[m * N + n] += v;
            else C[m * N + n] = v;
        }
}

static double rel_err(const float *a, const float *b, int n)
{
    double num = 0, den = 0;
    for (int i = 0; i < n; i++) { num += (a[i] - b[i]) * (double)(a[i] - b[i]); den += (double)b[i] * b[i]; }
    return sqrt(num / (den + 1e-30));
}

int main(void)
{
    if (!eif_siglip_int8_gemm_available()) {
        printf("test_siglip_int8_gemm: SKIP (no i8mm on this CPU / build)\n");
        return 0;
    }
    enum { M = 16, N = 24, K = 256 };
    static float A[M * K], Aq[M * K], C[M * N], R[M * N], bias[N], sw[N];
    static int8_t W[N * K];
    for (int i = 0; i < M * K; i++) A[i] = frand();
    for (int i = 0; i < N * K; i++) W[i] = (int8_t)(frand() * 127.0f);
    for (int n = 0; n < N; n++) { sw[n] = 0.01f + 0.001f * n; bias[n] = 0.1f * frand(); }

    /* 1. exactness against the dequantised operands, every mode, per-row and grouped scales */
    const int groups[2] = {K, 64};
    for (int gi = 0; gi < 2; gi++)
        for (int mode = 0; mode < 3; mode++) {
            const int G = groups[gi];
            fake_quant(A, M, K, G, Aq);
            for (int i = 0; i < M * N; i++) C[i] = R[i] = 0.5f;
            CHECK(eif_siglip_int8_gemm(A, M, K, G, W, sw, N, bias, C, mode) == 0, "call G=%d mode=%d", G, mode);
            ref_gemm(Aq, W, sw, bias, R, M, N, K, mode);
            double e = rel_err(C, R, M * N);
            CHECK(e < 1e-5, "G=%d mode=%d: rel err vs dequantised reference %.2e", G, mode, e);
        }

    /* 2. an outlier channel: grouped scales stay accurate, per-row scales lose the small values */
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) A[m * K + k] = 0.05f * frand();
        A[m * K + 7] = 40.0f;
    }
    ref_gemm(A, W, sw, bias, R, M, N, K, 0);
    eif_siglip_int8_gemm(A, M, K, K, W, sw, N, bias, C, 0);
    double e_row = rel_err(C, R, M * N);
    eif_siglip_int8_gemm(A, M, K, 64, W, sw, N, bias, C, 0);
    double e_grp = rel_err(C, R, M * N);
    CHECK(e_grp < 0.5 * e_row && e_grp < 0.01, "outlier: grouped %.4f vs per-row %.4f", e_grp, e_row);

    CHECK(eif_siglip_int8_gemm(A, 12, K, 64, W, sw, N, bias, C, 0) == -2, "M not a multiple of 8 must be rejected");
    if (!failures) printf("test_siglip_int8_gemm: all checks passed (outlier rel err: per-row %.4f, group-64 %.4f)\n", e_row, e_grp);
    return failures ? 1 : 0;
}
