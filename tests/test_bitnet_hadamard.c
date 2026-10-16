/**
 * @file test_bitnet_hadamard.c
 * @brief Hadamard rotation + group-scaled ternary matmul (Bonsai-style BitLinear(group_size, hadamard)).
 *
 * 1. eif_hadamard_rotate_f32 equals multiplication by the Sylvester matrix (H[i][j] = (-1)^popcount(i&j)/sqrt(b)),
 *    the same matrix the Python SDK uses, blockwise, after the sign flip D.
 * 2. Rotating twice with no signs is the identity (H is symmetric and orthonormal).
 * 3. End to end: W ternary-quantised per group IN THE ROTATED DOMAIN, input rotated at runtime, grouped matmul
 *    == dense float reference built from the same codes/scales.
 * 4. Size helpers match the SDK (576 -> block 64 / g96, 1536 -> block 512 / g128).
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eif_quantize_bitnet.h"

static int g_fail = 0;
#define CHECK(c, msg)                                                   \
    do {                                                                \
        if (c) printf("  PASS  %s\n", msg);                             \
        else { printf("  FAIL  %s\n", msg); g_fail++; }                 \
    } while (0)

static float frand(unsigned *s) { *s = *s * 1103515245u + 12345u; return ((*s >> 8) & 0xFFFF) / 32768.0f - 1.0f; }

static float sylvester(int i, int j, int b) { return (__builtin_popcount(i & j) & 1 ? -1.0f : 1.0f) / sqrtf((float)b); }

int main(void)
{
    printf("test_bitnet_hadamard\n");
    CHECK(eif_hadamard_block_size(576) == 64 && eif_hadamard_block_size(1536) == 512, "block sizes 576->64, 1536->512");
    CHECK(eif_bitnet_group_size(576, 128) == 96 && eif_bitnet_group_size(1536, 128) == 128 &&
          eif_bitnet_group_size(576, 0) == 0, "group sizes 576->96, 1536->128, 0->per-row");

    enum { N = 192, B = 64, ROWS = 7 };
    unsigned seed = 1;
    float x[N], signs[N], ref[N], y[N];
    for (int i = 0; i < N; i++) { x[i] = frand(&seed); signs[i] = (frand(&seed) > 0) ? 1.0f : -1.0f; }

    /* 1. against the explicit matrix */
    for (int blk = 0; blk < N; blk += B)
        for (int j = 0; j < B; j++) {
            float s = 0;
            for (int i = 0; i < B; i++) s += x[blk + i] * signs[blk + i] * sylvester(i, j, B);
            ref[blk + j] = s;
        }
    memcpy(y, x, sizeof(x));
    eif_hadamard_rotate_f32(y, signs, N, B);
    float err = 0;
    for (int i = 0; i < N; i++) err = fmaxf(err, fabsf(y[i] - ref[i]));
    CHECK(err < 1e-5f, "rotation == x*D*H (Sylvester, blockwise)");

    /* 2. involution */
    memcpy(y, x, sizeof(x));
    eif_hadamard_rotate_f32(y, NULL, N, B);
    eif_hadamard_rotate_f32(y, NULL, N, B);
    err = 0;
    for (int i = 0; i < N; i++) err = fmaxf(err, fabsf(y[i] - x[i]));
    CHECK(err < 1e-5f, "H(H(x)) == x");

    /* 3. rotated grouped ternary matmul */
    const int G = 96, NG = N / G, RB = (N + 3) / 4;
    float w[ROWS][N], wr[ROWS][N], scales[ROWS * (N / 96)];
    uint8_t packed[ROWS * ((N + 3) / 4)];
    memset(packed, 0, sizeof(packed));
    for (int r = 0; r < ROWS; r++) {
        for (int i = 0; i < N; i++) w[r][i] = frand(&seed) * (i == 5 ? 8.0f : 1.0f); /* one outlier column */
        memcpy(wr[r], w[r], sizeof(w[r]));
        eif_hadamard_rotate_f32(wr[r], signs, N, B); /* W' = W·D·H (row-wise) */
    }
    float deq[ROWS][N]; /* dense reference from the same codes/scales */
    for (int r = 0; r < ROWS; r++)
        for (int g = 0; g < NG; g++) {
            float m = 0;
            for (int i = 0; i < G; i++) m += fabsf(wr[r][g * G + i]);
            m = m / G > 1e-8f ? m / G : 1e-8f;
            scales[r * NG + g] = m;
            for (int i = 0; i < G; i++) {
                int c = (int)roundf(wr[r][g * G + i] / m);
                c = c > 1 ? 1 : (c < -1 ? -1 : c);
                deq[r][g * G + i] = c * m;
                int col = g * G + i;
                packed[r * RB + col / 4] |= (uint8_t)((c == 1 ? 1 : c == -1 ? 2 : 0) << ((col % 4) * 2));
            }
        }
    float xr[N], out[ROWS], want[ROWS];
    memcpy(xr, x, sizeof(x));
    eif_hadamard_rotate_f32(xr, signs, N, B);
    eif_matmul_bitnet_block_f32(packed, scales, G, xr, NULL, out, ROWS, N);
    err = 0;
    for (int r = 0; r < ROWS; r++) {
        want[r] = 0;
        for (int i = 0; i < N; i++) want[r] += deq[r][i] * xr[i];
        err = fmaxf(err, fabsf(out[r] - want[r]));
    }
    CHECK(err < 1e-4f, "grouped ternary matmul on rotated input == dense reference");

    /* and the rotation itself preserves the float product: W x == W' x' */
    err = 0;
    for (int r = 0; r < ROWS; r++) {
        float a = 0, b = 0;
        for (int i = 0; i < N; i++) { a += w[r][i] * x[i]; b += wr[r][i] * xr[i]; }
        err = fmaxf(err, fabsf(a - b));
    }
    CHECK(err < 1e-3f, "W x == (W D H)(x D H)");

    printf(g_fail ? "FAILED (%d)\n" : "ALL TESTS PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
