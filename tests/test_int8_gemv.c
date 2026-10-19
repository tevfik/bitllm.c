/* tinyllm_matmul_int8_rowscale vs a double-precision reference: odd shapes (tail rows/columns),
 * the W8A8 SDOT path (cols % 32 == 0) and the FP32-activation path. */
#include "eif_tinyllm.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* W8A8 (cols % 32 == 0): x is rounded to 1/254 of its 32-block absmax, so the per-row error is bounded by
 * scale * sum_b amax_b / 254 * sum_{c in b} |w_c|. Otherwise the FP32-activation path must be exact to rounding. */
static int check(int rows, int cols, double tol)
{
    int8_t *w = malloc((size_t)rows * cols);
    float *x = malloc(sizeof(float) * cols), *sc = malloc(sizeof(float) * rows), *y = malloc(sizeof(float) * rows);
    srand(rows * 7919 + cols);
    for (int i = 0; i < rows * cols; i++) w[i] = (int8_t)(rand() % 255 - 127);
    for (int i = 0; i < cols; i++) x[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f + (i % 97 == 0 ? 6.0f : 0.0f);
    for (int r = 0; r < rows; r++) sc[r] = 0.001f * (1 + r % 5);
    tinyllm_matmul_int8_rowscale(y, w, x, cols, rows, sc);
    int w8a8 = cols % 32 == 0 && !getenv("BITLLM_INT8_F32");
    double worst = 0.0;
    for (int r = 0; r < rows; r++) {
        double ref = 0.0, abs_sum = 0.0, bound = 0.0;
        for (int c = 0; c < cols; c++) {
            ref += (double)w[(size_t)r * cols + c] * x[c];
            abs_sum += fabs((double)w[(size_t)r * cols + c] * x[c]);
        }
        for (int b = 0; w8a8 && b < cols / 32; b++) {
            double amax = 0.0, wsum = 0.0;
            for (int c = b * 32; c < b * 32 + 32; c++) {
                amax = fmax(amax, fabs(x[c]));
                wsum += fabs((double)w[(size_t)r * cols + c]);
            }
            bound += amax / 254.0 * wsum;
        }
        bound = bound * 1.001 + tol * abs_sum;  /* + FP32 accumulation rounding */
        double ratio = fabs(ref - y[r] / sc[r]) / bound;
        worst = fmax(worst, ratio);
    }
    int ok = worst <= 1.0;
    printf("  rows=%4d cols=%4d  %s  error / bound %.3f %s\n", rows, cols, w8a8 ? "W8A8" : "FP32", worst,
           ok ? "ok" : "FAIL");
    free(w); free(x); free(sc); free(y);
    return ok;
}

int main(void)
{
    int ok = 1;
    /* W8A8 path where available (activation quantisation error ~1e-3 of the output range) */
    ok &= check(1536, 576, 1e-6);
    ok &= check(577, 1536, 1e-6);
    ok &= check(3, 64, 1e-6);
    /* FP32-activation path: columns not a multiple of 32 / 8 / 4 must still use every column */
    ok &= check(7, 579, 1e-5);
    ok &= check(9, 6, 1e-5);
    ok &= check(4, 13, 1e-5);
    printf(ok ? "test_int8_gemv: ALL PASSED\n" : "test_int8_gemv: FAILED\n");
    return ok ? 0 : 1;
}
