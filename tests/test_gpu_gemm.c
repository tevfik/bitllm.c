/**
 * @file test_gpu_gemm.c
 * @brief Vulkan GEMM kernels vs a double-precision CPU reference: FP32 and INT8 (per-row scales) weights,
 *        odd sizes, bias / GELU / residual epilogues, both tile sizes, and the batched strided QK^T / PV
 *        kernels used by attention. Passes with a SKIP message when no Vulkan device is present.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eif_gpu.h"
#include "../src/eif_gpu_internal.h"

static int g_fail = 0;
static unsigned s_seed = 777;
static float frand(void) { s_seed = s_seed * 1103515245u + 12345u; return ((s_seed >> 8) & 0xFFFF) / 32768.0f - 1.0f; }

static double gelu(double v) { return 0.5 * v * (1.0 + tanh(0.7978845608028654 * (v + 0.044715 * v * v * v))); }

/* max |gpu - ref| / (sum_k |a_k w_k| + 1e-3), the scale of the rounding error of a K-term dot product */
static void check(const char *name, const float *got, const double *ref, const double *mag, size_t n, double tol)
{
    double worst = 0.0;
    size_t bad = 0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs((double)got[i] - ref[i]) / (mag[i] + 1e-3);
        if (!(e <= tol)) bad++;
        if (e > worst || e != e) worst = e;
    }
    if (bad == 0) printf("  PASS  %-44s rel err %.2e\n", name, worst);
    else { printf("  FAIL  %-44s rel err %.2e (%zu bad)\n", name, worst, bad); g_fail++; }
}

static void ref_gemm(const float *A, const float *W, const int8_t *Wq, const float *sc, const float *bias,
                     const float *C0, int M, int N, int K, int act, double *ref, double *mag)
{
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double acc = 0.0, mg = 0.0;
            for (int k = 0; k < K; k++) {
                double w = W ? W[(size_t)n * K + k] : (double)Wq[(size_t)n * K + k] * sc[n];
                acc += (double)A[(size_t)m * K + k] * w;
                mg += fabs((double)A[(size_t)m * K + k] * w);
            }
            if (bias) acc += bias[n];
            if (act) acc = gelu(acc);
            if (C0) { acc += C0[(size_t)m * N + n]; mg += fabs(C0[(size_t)m * N + n]); }
            ref[(size_t)m * N + n] = acc;
            mag[(size_t)m * N + n] = mg;
        }
}

static void run_case(eif_gpu_context_t *gpu, int M, int N, int K, bool q8, bool use_bias, int act, bool residual)
{
    float *A = malloc((size_t)M * K * sizeof(float));
    float *W = malloc((size_t)N * K * sizeof(float));
    int8_t *Wq = malloc((size_t)N * K);
    float *sc = malloc((size_t)N * sizeof(float));
    float *bias = malloc((size_t)N * sizeof(float));
    float *C = malloc((size_t)M * N * sizeof(float));
    float *C0 = malloc((size_t)M * N * sizeof(float));
    double *ref = malloc((size_t)M * N * sizeof(double));
    double *mag = malloc((size_t)M * N * sizeof(double));
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = frand();
    for (size_t i = 0; i < (size_t)N * K; i++) { W[i] = frand() * 0.1f; Wq[i] = (int8_t)(frand() * 127.0f); }
    for (int i = 0; i < N; i++) { sc[i] = 0.001f + 0.002f * fabsf(frand()); bias[i] = frand(); }
    for (size_t i = 0; i < (size_t)M * N; i++) C0[i] = C[i] = frand();

    ref_gemm(A, q8 ? NULL : W, Wq, sc, use_bias ? bias : NULL, residual ? C0 : NULL, M, N, K, act, ref, mag);

    bool ok;
    eif_gpu_weight_t *w = q8 ? eif_gpu_weight_upload_q8(gpu, Wq, sc, N, K) : NULL;
    if (q8) ok = w && eif_gpu_gemm(gpu, A, w, use_bias ? bias : NULL, C, M, act, residual);
    else ok = eif_gpu_gemm_f32(gpu, A, W, use_bias ? bias : NULL, C, M, N, K, act, residual);
    eif_gpu_weight_free(gpu, w);

    char name[128];
    snprintf(name, sizeof(name), "%s M=%d N=%d K=%d%s%s%s", q8 ? "int8" : "fp32", M, N, K,
             use_bias ? " +bias" : "", act ? " +gelu" : "", residual ? " +res" : "");
    if (!ok) { printf("  FAIL  %s (dispatch failed)\n", name); g_fail++; }
    else check(name, C, ref, mag, (size_t)M * N, 2e-5);
    free(A); free(W); free(Wq); free(sc); free(bias); free(C); free(C0); free(ref); free(mag);
}

static double half_to_double(uint16_t h)
{
    int e = (h >> 10) & 31, m = h & 1023;
    double v = e == 0 ? ldexp((double)m, -24) : ldexp((double)(1024 + m), e - 25);
    return (h & 0x8000) ? -v : v;
}

/* GGUF Q8_0 weights (interleaved 34-byte blocks: fp16 scale + 32 codes) with one scale per block. */
static void run_q8_0(eif_gpu_context_t *gpu, int M, int N, int K, bool use_bias)
{
    const int nbk = K / 32;
    uint8_t *G = malloc((size_t)N * nbk * 34);
    float *A = malloc((size_t)M * K * sizeof(float)), *bias = malloc((size_t)N * sizeof(float));
    float *C = malloc((size_t)M * N * sizeof(float));
    double *ref = malloc((size_t)M * N * sizeof(double)), *mag = malloc((size_t)M * N * sizeof(double));
    double *Wd = malloc((size_t)N * K * sizeof(double));
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = frand();
    for (int i = 0; i < N; i++) bias[i] = frand();
    for (int n = 0; n < N; n++)
        for (int b = 0; b < nbk; b++) {
            uint8_t *blk = G + ((size_t)n * nbk + b) * 34;
            /* scales spanning several binades (0.0005 .. 0.03) so a wrongly indexed scale shows up */
            uint16_t h = (uint16_t)(0x1000 + ((s_seed = s_seed * 1103515245u + 12345u) >> 16) % 0x0c00);
            blk[0] = (uint8_t)(h & 255); blk[1] = (uint8_t)(h >> 8);
            for (int c = 0; c < 32; c++) {
                int8_t q = (int8_t)(frand() * 127.0f);
                blk[2 + c] = (uint8_t)q;
                Wd[(size_t)n * K + b * 32 + c] = (double)q * half_to_double(h);
            }
        }
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double acc = 0, mg = 0;
            for (int k = 0; k < K; k++) {
                double t = (double)A[(size_t)m * K + k] * Wd[(size_t)n * K + k];
                acc += t; mg += fabs(t);
            }
            ref[(size_t)m * N + n] = acc + (use_bias ? bias[n] : 0.0);
            mag[(size_t)m * N + n] = mg;
        }
    eif_gpu_weight_t *w = eif_gpu_weight_upload_q8_0(gpu, G, N, K);
    bool ok = w && eif_gpu_gemm(gpu, A, w, use_bias ? bias : NULL, C, M, 0, false);
    eif_gpu_weight_free(gpu, w);
    char name[96];
    snprintf(name, sizeof(name), "q8_0 blocks M=%d N=%d K=%d%s", M, N, K, use_bias ? " +bias" : "");
    if (!ok) { printf("  FAIL  %s (dispatch failed)\n", name); g_fail++; }
    else check(name, C, ref, mag, (size_t)M * N, 2e-5);
    free(G); free(A); free(bias); free(C); free(ref); free(mag); free(Wd);
}

/* Attention-shaped batched GEMMs: T tiles x H heads, S tokens, head dim D, rows of width 3*H*D (q|k|v). */
static void run_attention(eif_gpu_context_t *gpu, int T, int H, int S, int D)
{
    const int ld = 3 * H * D;
    const size_t nq = (size_t)T * S * ld, ns = (size_t)T * H * S * S, no = (size_t)T * S * H * D;
    float *qkv = malloc(nq * sizeof(float));
    for (size_t i = 0; i < nq; i++) qkv[i] = frand();
    eif_gpu_buf_t *bq = eif_gpu_buf_create(gpu, nq * sizeof(float), true);
    eif_gpu_buf_t *bs = eif_gpu_buf_create(gpu, ns * sizeof(float), true);
    eif_gpu_buf_t *bo = eif_gpu_buf_create(gpu, no * sizeof(float), true);
    if (!bq || !bs || !bo) { printf("  FAIL  attention buffers\n"); g_fail++; goto out; }
    memcpy(eif_gpu_buf_map(bq), qkv, nq * sizeof(float));

    eif_gpu_gemm_pc_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.M = S; pc.N = S; pc.K = D; pc.lda = ld; pc.ldb = ld; pc.ldc = S;
    pc.a_off = 0; pc.b_off = H * D; pc.zdiv = H;
    pc.a_s1 = S * ld; pc.a_s2 = D; pc.b_s1 = S * ld; pc.b_s2 = D;
    pc.c_s1 = H * S * S; pc.c_s2 = S * S; pc.alpha = 0.125f;
    eif_gpu_gemm_pc_t pv = pc;
    pv.M = S; pv.N = D; pv.K = S; pv.lda = S; pv.ldb = ld; pv.ldc = H * D;
    pv.a_off = 0; pv.b_off = 2 * H * D; pv.a_s1 = H * S * S; pv.a_s2 = S * S;
    pv.c_s1 = S * H * D; pv.c_s2 = D; pv.alpha = 1.0f;
    uint32_t sm[3] = {(uint32_t)S, (uint32_t)(T * H * S), 0};
    eif_gpu_buf_t *smb[5] = {NULL, NULL, NULL, NULL, bs};

    bool ok = eif_gpu_cmd_begin(gpu);
    ok = ok && eif_gpu_cmd_gemm(gpu, EIF_GPU_K_GEMM_F32, bq, bq, NULL, NULL, bs, &pc, T * H);
    eif_gpu_cmd_barrier(gpu);
    ok = ok && eif_gpu_cmd_dispatch(gpu, EIF_GPU_K_SOFTMAX, smb, sm, sizeof(sm), T * H * S < 65535 ? T * H * S : 65535, 1, 1);
    eif_gpu_cmd_barrier(gpu);
    ok = ok && eif_gpu_cmd_gemm(gpu, EIF_GPU_K_GEMM_KN, bs, bq, NULL, NULL, bo, &pv, T * H);
    ok = eif_gpu_cmd_submit(gpu) && ok;
    if (!ok) { printf("  FAIL  attention dispatch\n"); g_fail++; goto out; }

    double *ref = malloc(no * sizeof(double)), *mag = malloc(no * sizeof(double));
    double *p = malloc((size_t)S * sizeof(double));
    for (int t = 0; t < T; t++)
        for (int h = 0; h < H; h++)
            for (int i = 0; i < S; i++) {
                const float *q = qkv + (size_t)t * S * ld + (size_t)i * ld + h * D;
                double mx = -1e300, sum = 0;
                for (int j = 0; j < S; j++) {
                    const float *k = qkv + (size_t)t * S * ld + (size_t)j * ld + H * D + h * D;
                    double s = 0;
                    for (int d = 0; d < D; d++) s += (double)q[d] * k[d];
                    p[j] = s * 0.125;
                    if (p[j] > mx) mx = p[j];
                }
                for (int j = 0; j < S; j++) { p[j] = exp(p[j] - mx); sum += p[j]; }
                for (int d = 0; d < D; d++) {
                    double acc = 0, mg = 0;
                    for (int j = 0; j < S; j++) {
                        double v = qkv[(size_t)t * S * ld + (size_t)j * ld + 2 * H * D + h * D + d];
                        acc += p[j] / sum * v;
                        mg += fabs(p[j] / sum * v);
                    }
                    size_t o = (size_t)t * S * H * D + (size_t)i * H * D + h * D + d;
                    ref[o] = acc;
                    mag[o] = mg;
                }
            }
    char name[96];
    snprintf(name, sizeof(name), "attention T=%d H=%d S=%d D=%d", T, H, S, D);
    check(name, (const float *)eif_gpu_buf_map(bo), ref, mag, no, 1e-4);

    if (D == 64 && eif_gpu_kernel_available(gpu, EIF_GPU_K_ATTENTION)) {
        memset(eif_gpu_buf_map(bo), 0, no * sizeof(float));
        eif_gpu_attn_pc_t ap = {(uint32_t)S, (uint32_t)H, (uint32_t)ld, (uint32_t)(H * D), 0, 0,
                                0.125f * 1.4426950408889634f};
        eif_gpu_buf_t *fb[5] = {bq, NULL, NULL, NULL, bo};
        ok = eif_gpu_cmd_begin(gpu) &&
             eif_gpu_cmd_dispatch(gpu, EIF_GPU_K_ATTENTION, fb, &ap, sizeof(ap), (S + 63) / 64, H, T);
        ok = eif_gpu_cmd_submit(gpu) && ok;
        snprintf(name, sizeof(name), "fused attention T=%d H=%d S=%d D=%d", T, H, S, D);
        if (!ok) { printf("  FAIL  %s (dispatch failed)\n", name); g_fail++; }
        else check(name, (const float *)eif_gpu_buf_map(bo), ref, mag, no, 1e-4);
    }
    free(ref); free(mag); free(p);
out:
    eif_gpu_buf_destroy(gpu, bq);
    eif_gpu_buf_destroy(gpu, bs);
    eif_gpu_buf_destroy(gpu, bo);
    free(qkv);
}

int main(void)
{
    printf("====================================================\n");
    printf("  Test: Vulkan GEMM kernels vs CPU reference\n");
    printf("====================================================\n");
    eif_gpu_context_t gpu;
    if (!eif_gpu_init(&gpu)) {
        printf("  [SKIP] No Vulkan device (%s): GPU kernels not tested.\n", gpu.device_name);
        return 0;
    }
    printf("  Device: %s\n", gpu.device_name);

    run_case(&gpu, 1, 1, 1, false, false, 0, false);
    run_case(&gpu, 37, 53, 29, false, true, 0, false);
    run_case(&gpu, 130, 67, 771, false, true, 1, false);
    run_case(&gpu, 65, 129, 17, false, false, 0, true);
    run_case(&gpu, 1024, 768, 768, false, true, 0, true);
    run_case(&gpu, 1031, 1543, 300, false, true, 1, false); /* 128x128 tiles, ragged edges */
    run_case(&gpu, 37, 53, 29, true, true, 0, false);       /* K % 4 != 0: unaligned INT8 rows */
    run_case(&gpu, 3, 7, 1, true, false, 0, false);
    run_case(&gpu, 129, 255, 770, true, true, 1, false);
    run_case(&gpu, 1024, 2304, 768, true, true, 0, false);  /* 128x128 tiles */
    run_case(&gpu, 1100, 768, 3072, true, true, 0, true);
    run_q8_0(&gpu, 5, 7, 32, false);                        /* one block per row */
    run_q8_0(&gpu, 37, 53, 96, true);
    run_q8_0(&gpu, 1540, 1301, 160, true);                  /* 128x128 tiles, ragged edges */
    run_q8_0(&gpu, 130, 90, 800, true);                     /* 64x64 tiles, aligned K */
    run_attention(&gpu, 2, 3, 70, 64);
    run_attention(&gpu, 1, 2, 129, 16);
    run_attention(&gpu, 3, 2, 31, 64);
    run_attention(&gpu, 1, 12, 1024, 64);

    eif_gpu_cleanup(&gpu);
    printf("%s\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
    return g_fail ? 1 : 0;
}
