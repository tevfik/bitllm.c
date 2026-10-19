/* eif_gemm_f32_nt vs a double reference: tail shapes (M % 8, N % 12, K % 256), bias, GELU, accumulate. */
#include "eif_gemm.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static float frand(void) { return (float)rand() / RAND_MAX * 2.0f - 1.0f; }

static int check(int M, int N, int K, int act, int acc)
{
    float *A = malloc(sizeof(float) * M * K), *W = malloc(sizeof(float) * N * K);
    float *b = malloc(sizeof(float) * N), *C = malloc(sizeof(float) * M * N), *C0 = malloc(sizeof(float) * M * N);
    for (int i = 0; i < M * K; i++) A[i] = frand();
    for (int i = 0; i < N * K; i++) W[i] = frand();
    for (int i = 0; i < N; i++) b[i] = frand();
    for (int i = 0; i < M * N; i++) C[i] = C0[i] = frand();
    eif_gemm_f32_nt(A, W, b, C, M, N, K, act, acc);
    double worst = 0.0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            double v = b[j], mag = fabs(b[j]);
            for (int k = 0; k < K; k++) {
                v += (double)A[(size_t)i * K + k] * W[(size_t)j * K + k];
                mag += fabs((double)A[(size_t)i * K + k] * W[(size_t)j * K + k]);
            }
            if (acc) v += C0[(size_t)i * N + j];
            else if (act == EIF_GEMM_ACT_GELU_TANH) v = 0.5 * v * (1.0 + tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));
            worst = fmax(worst, fabs(v - C[(size_t)i * N + j]) / (mag + 1.0));
        }
    int ok = worst < 1e-5;
    printf("  M=%4d N=%4d K=%4d act=%d acc=%d  max err %.1e %s\n", M, N, K, act, acc, worst, ok ? "ok" : "FAIL");
    free(A); free(W); free(b); free(C); free(C0);
    return ok;
}

static int check_softmax(int rows, int cols, float scale)
{
    int ld = cols + 3;
    float *S = malloc(sizeof(float) * rows * ld), *S0 = malloc(sizeof(float) * rows * ld);
    for (int i = 0; i < rows * ld; i++) S[i] = S0[i] = frand() * 40.0f;
    eif_softmax_rows_f32(S, rows, cols, ld, scale);
    double worst = 0.0;
    for (int r = 0; r < rows; r++) {
        double mx = -1e300, sum = 0.0;
        for (int c = 0; c < cols; c++) mx = fmax(mx, S0[r * ld + c]);
        for (int c = 0; c < cols; c++) sum += exp((S0[r * ld + c] - mx) * scale);
        for (int c = 0; c < cols; c++) {
            double ref = exp((S0[r * ld + c] - mx) * scale) / sum;
            worst = fmax(worst, fabs(ref - S[r * ld + c]) / (ref + 1e-30) * (ref > 1e-30));
        }
        for (int c = cols; c < ld; c++) worst = fmax(worst, S[r * ld + c] != S0[r * ld + c]);  /* padding untouched */
    }
    int ok = worst < 1e-5;
    printf("  softmax rows=%3d cols=%4d scale=%.3f  max rel err %.1e %s\n", rows, cols, scale, worst, ok ? "ok" : "FAIL");
    free(S); free(S0);
    return ok;
}

int main(void)
{
    int ok = 1;
    ok &= check_softmax(64, 1024, 0.125f);
    ok &= check_softmax(5, 7, 1.0f);
    ok &= check_softmax(3, 1, 2.0f);
    ok &= check(1024, 768, 768, EIF_GEMM_ACT_NONE, 0);
    ok &= check(64, 3072, 768, EIF_GEMM_ACT_GELU_TANH, 0);
    ok &= check(67, 770, 3072, EIF_GEMM_ACT_NONE, 1);
    ok &= check(1, 1, 1, EIF_GEMM_ACT_NONE, 0);
    ok &= check(13, 25, 257, EIF_GEMM_ACT_GELU_TANH, 0);
    ok &= check(9, 11, 5, EIF_GEMM_ACT_NONE, 1);
    printf(ok ? "test_gemm: ALL PASSED\n" : "test_gemm: FAILED\n");
    return ok ? 0 : 1;
}
