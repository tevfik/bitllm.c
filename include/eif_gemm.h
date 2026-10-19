/**
 * @file eif_gemm.h
 * @brief Blocked FP32 GEMM for the batched (ViT / prefill) paths.
 */

#ifndef EIF_GEMM_H
#define EIF_GEMM_H

#ifdef __cplusplus
extern "C" {
#endif

#define EIF_GEMM_ACT_NONE 0
#define EIF_GEMM_ACT_GELU_TANH 1

/**
 * C[M, N] = act(A[M, K] @ W[N, K]^T + bias[N])      (accumulate == 0)
 * C[M, N] += A[M, K] @ W[N, K]^T + bias[N]           (accumulate != 0, act must be NONE)
 *
 * Row-major, contiguous rows. bias may be NULL. Parallelised with OpenMP unless already inside a parallel region.
 */
void eif_gemm_f32_nt(const float *A, const float *W, const float *bias, float *C,
                     int M, int N, int K, int act, int accumulate);

#ifdef __cplusplus
}
#endif

#endif /* EIF_GEMM_H */
