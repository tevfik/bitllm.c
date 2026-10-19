/**
 * @file eif_gemm.c
 * @brief Blocked FP32 GEMM (C = A W^T): packed MC x KC / NC x KC panels, 8 x 12 register tile.
 */

#include "eif_gemm.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#if defined(_OPENMP)
#include <omp.h>
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#define GEMM_MR 8
#define GEMM_NR 12
#define GEMM_KC 256
#define GEMM_MC 64
#define GEMM_NC 96

/* A rows [m0, m0+mc) x cols [k0, k0+kc) -> panels of MR rows, k-major; rows past M are zero. */
static void pack_a(const float *A, int lda, int M, int m0, int mc, int k0, int kc, float *Ap)
{
    for (int i0 = 0; i0 < mc; i0 += GEMM_MR) {
        float *dst = Ap + (size_t)i0 * kc;
        for (int i = 0; i < GEMM_MR; i++) {
            int m = m0 + i0 + i;
            if (i0 + i < mc && m < M) {
                const float *src = A + (size_t)m * lda + k0;
                for (int k = 0; k < kc; k++) dst[k * GEMM_MR + i] = src[k];
            } else {
                for (int k = 0; k < kc; k++) dst[k * GEMM_MR + i] = 0.0f;
            }
        }
    }
}

/* W rows [n0, n0+nc) x cols [k0, k0+kc) -> panels of NR rows, k-major; rows past N are zero. */
static void pack_b(const float *W, int ldw, int N, int n0, int nc, int k0, int kc, float *Bp)
{
    for (int j0 = 0; j0 < nc; j0 += GEMM_NR) {
        float *dst = Bp + (size_t)j0 * kc;
        for (int j = 0; j < GEMM_NR; j++) {
            int n = n0 + j0 + j;
            if (j0 + j < nc && n < N) {
                const float *src = W + (size_t)n * ldw + k0;
                for (int k = 0; k < kc; k++) dst[k * GEMM_NR + j] = src[k];
            } else {
                for (int k = 0; k < kc; k++) dst[k * GEMM_NR + j] = 0.0f;
            }
        }
    }
}

/* Ct[8, 12] (row stride ldc) += Ap[kc, 8]^T Bp[kc, 12] */
static void ukr_8x12(int kc, const float *Ap, const float *Bp, float *Ct, int ldc)
{
#if defined(__aarch64__)
    float32x4_t c[GEMM_MR][3];
    for (int i = 0; i < GEMM_MR; i++)
        for (int j = 0; j < 3; j++) c[i][j] = vld1q_f32(Ct + i * ldc + 4 * j);
    for (int k = 0; k < kc; k++) {
        float32x4_t b0 = vld1q_f32(Bp), b1 = vld1q_f32(Bp + 4), b2 = vld1q_f32(Bp + 8);
        float32x4_t a0 = vld1q_f32(Ap), a1 = vld1q_f32(Ap + 4);
#define GEMM_ROW(i, a, l) \
        c[i][0] = vfmaq_laneq_f32(c[i][0], b0, a, l); \
        c[i][1] = vfmaq_laneq_f32(c[i][1], b1, a, l); \
        c[i][2] = vfmaq_laneq_f32(c[i][2], b2, a, l);
        GEMM_ROW(0, a0, 0) GEMM_ROW(1, a0, 1) GEMM_ROW(2, a0, 2) GEMM_ROW(3, a0, 3)
        GEMM_ROW(4, a1, 0) GEMM_ROW(5, a1, 1) GEMM_ROW(6, a1, 2) GEMM_ROW(7, a1, 3)
#undef GEMM_ROW
        Ap += GEMM_MR;
        Bp += GEMM_NR;
    }
    for (int i = 0; i < GEMM_MR; i++)
        for (int j = 0; j < 3; j++) vst1q_f32(Ct + i * ldc + 4 * j, c[i][j]);
#else
    float c[GEMM_MR][GEMM_NR];
    for (int i = 0; i < GEMM_MR; i++)
        for (int j = 0; j < GEMM_NR; j++) c[i][j] = Ct[i * ldc + j];
    for (int k = 0; k < kc; k++) {
        for (int i = 0; i < GEMM_MR; i++) {
            float a = Ap[i];
            for (int j = 0; j < GEMM_NR; j++) c[i][j] += a * Bp[j];
        }
        Ap += GEMM_MR;
        Bp += GEMM_NR;
    }
    for (int i = 0; i < GEMM_MR; i++)
        for (int j = 0; j < GEMM_NR; j++) Ct[i * ldc + j] = c[i][j];
#endif
}

static inline float gelu_tanh(float x)
{
    const float k = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

void eif_gemm_f32_nt(const float *A, const float *W, const float *bias, float *C,
                     int M, int N, int K, int act, int accumulate)
{
    eif_gemm_f32_nt_ld(A, K, W, K, bias, C, N, M, N, K, act, accumulate);
}

void eif_gemm_f32_nt_ld(const float *A, int lda, const float *W, int ldw, const float *bias, float *C, int ldc,
                        int M, int N, int K, int act, int accumulate)
{
    if (M <= 0 || N <= 0) return;
    const int mb = (M + GEMM_MC - 1) / GEMM_MC;
    const int nb = (N + GEMM_NC - 1) / GEMM_NC;
    const int jobs = mb * nb;

    #pragma omp parallel if(!omp_in_parallel() && jobs > 1)
    {
        float *Ap = (float *)malloc(sizeof(float) * GEMM_MC * GEMM_KC);
        float *Bp = (float *)malloc(sizeof(float) * GEMM_NC * GEMM_KC);
        float *Ct = (float *)malloc(sizeof(float) * GEMM_MC * GEMM_NC);
        int ok = Ap && Bp && Ct;

        #pragma omp for schedule(dynamic, 1)
        for (int job = 0; job < jobs; job++) {
            const int m0 = (job / nb) * GEMM_MC, n0 = (job % nb) * GEMM_NC;
            const int mc = (M - m0 < GEMM_MC) ? M - m0 : GEMM_MC;
            const int nc = (N - n0 < GEMM_NC) ? N - n0 : GEMM_NC;
            if (!ok) { /* out of memory: unblocked fallback for this tile */
                for (int i = m0; i < m0 + mc; i++)
                    for (int j = n0; j < n0 + nc; j++) {
                        float v = bias ? bias[j] : 0.0f;
                        for (int k = 0; k < K; k++) v += A[(size_t)i * lda + k] * W[(size_t)j * ldw + k];
                        if (accumulate) C[(size_t)i * ldc + j] += v;
                        else C[(size_t)i * ldc + j] = (act == EIF_GEMM_ACT_GELU_TANH) ? gelu_tanh(v) : v;
                    }
                continue;
            }
            const int mc_r = (mc + GEMM_MR - 1) / GEMM_MR * GEMM_MR;
            const int nc_r = (nc + GEMM_NR - 1) / GEMM_NR * GEMM_NR;
            memset(Ct, 0, sizeof(float) * GEMM_MC * GEMM_NC);

            for (int k0 = 0; k0 < K; k0 += GEMM_KC) {
                const int kc = (K - k0 < GEMM_KC) ? K - k0 : GEMM_KC;
                pack_a(A, lda, M, m0, mc_r, k0, kc, Ap);
                pack_b(W, ldw, N, n0, nc_r, k0, kc, Bp);
                for (int i0 = 0; i0 < mc_r; i0 += GEMM_MR)
                    for (int j0 = 0; j0 < nc_r; j0 += GEMM_NR)
                        ukr_8x12(kc, Ap + (size_t)i0 * kc, Bp + (size_t)j0 * kc, Ct + i0 * GEMM_NC + j0, GEMM_NC);
            }

            for (int i = 0; i < mc; i++) {
                float *c = C + (size_t)(m0 + i) * ldc + n0;
                const float *t = Ct + i * GEMM_NC;
                for (int j = 0; j < nc; j++) {
                    float v = t[j] + (bias ? bias[n0 + j] : 0.0f);
                    if (accumulate) c[j] += v;
                    else c[j] = (act == EIF_GEMM_ACT_GELU_TANH) ? gelu_tanh(v) : v;
                }
            }
        }
        free(Ap);
        free(Bp);
        free(Ct);
    }
}

#if defined(__aarch64__)
/* exp(x) for x <= 0-ish softmax inputs (Cephes polynomial, ~1 ulp on [-87, 88]) */
static inline float32x4_t exp_f32x4(float32x4_t x)
{
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.3f)), vdupq_n_f32(88.3f));
    float32x4_t n = vrndnq_f32(vmulq_n_f32(x, 1.44269504088896341f));
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693359375f));
    r = vfmsq_f32(r, n, vdupq_n_f32(-2.12194440e-4f));
    float32x4_t p = vdupq_n_f32(1.9875691500e-4f);
    p = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), p, r);
    p = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), p, r);
    p = vfmaq_f32(vaddq_f32(r, vdupq_n_f32(1.0f)), p, vmulq_f32(r, r));
    int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(e));
}
#endif

void eif_softmax_rows_f32(float *S, int rows, int cols, int ld, float scale)
{
    #pragma omp parallel for schedule(static) if(!omp_in_parallel() && rows > 1)
    for (int r = 0; r < rows; r++) {
        float *x = S + (size_t)r * ld;
        int c = 0;
        float mx = -INFINITY, sum = 0.0f;
#if defined(__aarch64__)
        float32x4_t vm = vdupq_n_f32(-INFINITY);
        for (; c <= cols - 4; c += 4) vm = vmaxq_f32(vm, vld1q_f32(x + c));
        mx = vmaxvq_f32(vm);
#endif
        for (; c < cols; c++) mx = fmaxf(mx, x[c]);
        c = 0;
#if defined(__aarch64__)
        float32x4_t vs = vdupq_n_f32(0.0f), vsc = vdupq_n_f32(scale), vmx = vdupq_n_f32(mx * scale);
        for (; c <= cols - 4; c += 4) {
            float32x4_t e = exp_f32x4(vfmsq_f32(vnegq_f32(vmx), vnegq_f32(vsc), vld1q_f32(x + c)));
            vst1q_f32(x + c, e);
            vs = vaddq_f32(vs, e);
        }
        sum = vaddvq_f32(vs);
#endif
        for (; c < cols; c++) {
            x[c] = expf((x[c] - mx) * scale);
            sum += x[c];
        }
        float inv = 1.0f / sum;
        for (c = 0; c < cols; c++) x[c] *= inv;
    }
}
