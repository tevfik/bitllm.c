/* Qwen3.5 INT4 GEMV: the SDOT path (Q8 activations per 32 inputs) against a double reference, checked with its
 * exact activation-rounding bound; the scalar path (BITLLM_INT8_F32=1, read on first use) must be exact. */
#include "../src/eif_qwen35.c"

static int check(int rows, int cols, int dot)
{
    int pcols = (cols + 1) / 2;
    uint8_t *W = malloc((size_t)rows * pcols);
    float *x = malloc(sizeof(float) * cols), *sc = malloc(sizeof(float) * rows), *y = malloc(sizeof(float) * rows);
    srand(rows * 31 + cols);
    for (int i = 0; i < rows * pcols; i++) W[i] = (uint8_t)(rand() & 0xFF);
    for (int i = 0; i < cols; i++) x[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f + (i % 61 == 0 ? 9.0f : 0.0f);
    for (int r = 0; r < rows; r++) sc[r] = 0.01f * (1 + r % 3);
    int4_gemv(y, W, sc, x, rows, cols);
    double worst = 0.0;
    for (int r = 0; r < rows; r++) {
        double ref = 0.0, abs_sum = 0.0, bound = 0.0;
        for (int c = 0; c < cols; c++) {
            uint8_t byte = W[(size_t)r * pcols + c / 2];
            int w = (c & 1) ? (byte & 0x0F) - 8 : (byte >> 4) - 8;
            ref += (double)w * x[c];
            abs_sum += fabs((double)w * x[c]);
        }
        for (int b = 0; dot && b < cols / 32; b++) {
            double amax = 0.0, wsum = 0.0;
            for (int c = b * 32; c < b * 32 + 32; c++) {
                uint8_t byte = W[(size_t)r * pcols + c / 2];
                amax = fmax(amax, fabs(x[c]));
                wsum += abs((c & 1) ? (byte & 0x0F) - 8 : (byte >> 4) - 8);
            }
            bound += amax / 254.0 * wsum;
        }
        bound = bound * 1.001 + 1e-6 * abs_sum;
        worst = fmax(worst, fabs(ref - y[r] / sc[r]) / bound);
    }
    int ok = worst <= 1.0;
    printf("  rows=%4d cols=%4d %s  error / bound %.3f %s\n", rows, cols, dot ? "SDOT" : "FP32", worst, ok ? "ok" : "FAIL");
    free(W); free(x); free(sc); free(y);
    return ok;
}

int main(void)
{
    int dot = 0;
#if defined(__aarch64__) && defined(__linux__)
    dot = q35_has_dotprod() && !getenv("BITLLM_INT8_F32");
#endif
    int ok = 1;
    ok &= check(1024, 1024, dot);
    ok &= check(37, 3584, dot);
    ok &= check(5, 33, 0);   /* cols % 32 != 0 always takes the scalar path */
    printf(ok ? "test_q35_int4: ALL PASSED\n" : "test_q35_int4: FAILED\n");
    return ok ? 0 : 1;
}
