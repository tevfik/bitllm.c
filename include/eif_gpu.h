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
 * 3. No link-time dependency: the Vulkan loader (libvulkan.so.1 / vulkan-1.dll) is opened at runtime.
 *
 * Environment: BITLLM_NO_GPU=1 disables the backend, BITLLM_GPU_ALLOW_CPU=1 accepts software devices
 * (llvmpipe), BITLLM_VULKAN_LIB overrides the loader path.
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

/** Weight matrix resident in GPU memory (see eif_gpu_weight_upload_*). */
typedef struct eif_gpu_weight eif_gpu_weight_t;

/** @brief Upload an FP32 weight W[N, K] once; returns NULL if the GPU is unavailable or out of memory. */
eif_gpu_weight_t *eif_gpu_weight_upload_f32(eif_gpu_context_t *ctx, const float *W, int N, int K);

/** @brief Upload an INT8 weight (codes[N, K], per-row scales[N]: W = codes * scale); dequantised in the kernel. */
eif_gpu_weight_t *eif_gpu_weight_upload_q8(eif_gpu_context_t *ctx, const int8_t *codes, const float *scales,
                                           int N, int K);

/** @brief Release a resident weight (must be called before eif_gpu_cleanup). */
void eif_gpu_weight_free(eif_gpu_context_t *ctx, eif_gpu_weight_t *w);

/**
 * @brief C[M, N] = A[M, K] @ W^T (+ bias) with a resident weight; same epilogue options as eif_gpu_gemm_f32.
 * @return false if the caller must execute the CPU fallback.
 */
bool eif_gpu_gemm(eif_gpu_context_t *ctx, const float *A, const eif_gpu_weight_t *W, const float *bias,
                  float *C, int M, int activation_mode, bool residual_add);

/**
 * @brief GPU-accelerated General Matrix Multiplication (GEMM) for FP32 (weights copied on every call;
 *        prefer eif_gpu_weight_upload_f32 + eif_gpu_gemm for repeated use).
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
