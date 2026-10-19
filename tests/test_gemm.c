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

int main(void)
{
    int ok = 1;
    ok &= check(1024, 768, 768, EIF_GEMM_ACT_NONE, 0);
    ok &= check(64, 3072, 768, EIF_GEMM_ACT_GELU_TANH, 0);
    ok &= check(67, 770, 3072, EIF_GEMM_ACT_NONE, 1);
    ok &= check(1, 1, 1, EIF_GEMM_ACT_NONE, 0);
    ok &= check(13, 25, 257, EIF_GEMM_ACT_GELU_TANH, 0);
    ok &= check(9, 11, 5, EIF_GEMM_ACT_NONE, 1);
    printf(ok ? "test_gemm: ALL PASSED\n" : "test_gemm: FAILED\n");
    return ok ? 0 : 1;
}
