/**
 * @file eif_gpu.h
 * @brief EIF Optional GPU Compute Acceleration Backend (Vulkan / Graceful CPU Fallback)
 *
 * Provides a portable, cross-platform GPU compute interface designed for
 * zero build failures across diverse edge targets (ARM Linux, x86, embedded).
 *
 * Resilience Guarantees:
 * 1. Zero Dependency on headless/baremetal targets: If compiled without Vulkan,
 *    compiles 100% cleanly with zero external libraries or headers.
 * 2. Graceful Runtime Fallback: If Vulkan is enabled but no GPU driver or
 *    compatible physical device is present, initialization safely returns false
 *    and inference falls back automatically to CPU T-MAC / NEON kernels.
 */

#ifndef EIF_GPU_H
#define EIF_GPU_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    bool is_available;
    char backend_name[64];
    char device_name[256];
    void *internal_handle;
} eif_gpu_context_t;

/**
 * @brief Initialize GPU compute runtime.
 *
 * @param ctx GPU context structure to populate.
 * @return true if GPU acceleration was successfully initialized, false if unavailable.
 */
bool eif_gpu_init(eif_gpu_context_t *ctx);

/**
 * @brief Check if GPU compute acceleration is currently available.
 */
bool eif_gpu_is_available(const eif_gpu_context_t *ctx);

/**
 * @brief Clean up GPU resources and free handles.
 */
void eif_gpu_cleanup(eif_gpu_context_t *ctx);

/**
 * @brief GPU-accelerated BitLinear Matrix-Vector Multiplication.
 *
 * Computes: output[rows] = scales[row] * (W_ternary[rows, cols] @ input[cols]) + bias[row]
 *
 * @return true if executed successfully on GPU, false if caller must execute CPU fallback.
 */
bool eif_gpu_matmul_bitnet_f32(eif_gpu_context_t *ctx,
                               const uint8_t *weights,
                               const float *scales,
                               const float *input,
                               const float *bias,
                               float *output,
                               int rows,
                               int cols);

/**
 * @brief GPU-accelerated General Matrix Multiplication (GEMM) for FP32.
 *
 * Computes: C[M, N] = A[M, K] @ W[N, K]^T + (bias[N] if present)
 * Supports optional GELU activation and residual accumulation.
 *
 * @param activation_mode 0 = linear/identity, 1 = GELU (tanh approximation)
 * @param residual_add If true, adds result to existing C[M, N], else overwrites C.
 * @return true if executed successfully on GPU, false if caller must execute CPU fallback.
 */
bool eif_gpu_gemm_f32(eif_gpu_context_t *ctx,
                      const float *A,
                      const float *W,
                      const float *bias,
                      float *C,
                      int M, int N, int K,
                      int activation_mode,
                      bool residual_add);

#ifdef __cplusplus
}
#endif

#endif /* EIF_GPU_H */
