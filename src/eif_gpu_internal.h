/**
 * @file eif_gpu_internal.h
 * @brief Low-level Vulkan compute interface shared by the GPU encoders (not installed).
 *
 * Usage: create buffers once, then record a command stream with eif_gpu_cmd_begin(), any number of
 * eif_gpu_cmd_*() dispatches separated by eif_gpu_cmd_barrier(), and finish with eif_gpu_cmd_submit()
 * (one submit, waits for completion). A context is not thread-safe.
 */
#ifndef EIF_GPU_INTERNAL_H
#define EIF_GPU_INTERNAL_H

#include "eif_gpu.h"

typedef struct eif_gpu_buf eif_gpu_buf_t;

enum {
    EIF_GPU_K_GEMM_F32 = 0,
    EIF_GPU_K_GEMM_Q8,
    EIF_GPU_K_GEMM_KN,
    EIF_GPU_K_GEMM_F32_BIG,
    EIF_GPU_K_GEMM_Q8_BIG,
    EIF_GPU_K_LAYERNORM,
    EIF_GPU_K_SOFTMAX,
    EIF_GPU_K_BCAST,
    EIF_GPU_K_MERGE,
    EIF_GPU_K_COUNT
};

#define EIF_GPU_GEMM_BIAS     1u
#define EIF_GPU_GEMM_GELU     2u
#define EIF_GPU_GEMM_RESIDUAL 4u
#define EIF_GPU_GEMM_SCALES   8u

/* Mirrors the push constants of shaders/gemm.comp. */
typedef struct {
    uint32_t M, N, K;
    uint32_t lda, ldb, ldc;
    uint32_t a_off, b_off, c_off;
    uint32_t s_off, bias_off;
    uint32_t flags;
    uint32_t zdiv;
    uint32_t a_s1, a_s2, b_s1, b_s2, c_s1, c_s2;
    float alpha;
} eif_gpu_gemm_pc_t;

/* host_visible: mapped, coherent (prefers device-local); otherwise device-local, filled via staging. */
eif_gpu_buf_t *eif_gpu_buf_create(eif_gpu_context_t *ctx, size_t bytes, bool host_visible);
void eif_gpu_buf_destroy(eif_gpu_context_t *ctx, eif_gpu_buf_t *b);
void *eif_gpu_buf_map(eif_gpu_buf_t *b);
bool eif_gpu_buf_upload(eif_gpu_context_t *ctx, eif_gpu_buf_t *b, size_t offset, const void *src, size_t bytes);
/* Largest size of one storage buffer binding. */
size_t eif_gpu_max_binding_bytes(const eif_gpu_context_t *ctx);

bool eif_gpu_cmd_begin(eif_gpu_context_t *ctx);
/* bufs: bindings 0 A, 1 B, 2 scales/params, 3 bias, 4 C (NULL entries are bound to a dummy buffer). */
bool eif_gpu_cmd_dispatch(eif_gpu_context_t *ctx, int kernel, eif_gpu_buf_t *const bufs[5],
                          const void *push, uint32_t push_bytes, uint32_t gx, uint32_t gy, uint32_t gz);
/* Picks the tile size and grid; kernel is EIF_GPU_K_GEMM_F32, _Q8 or _KN; gz = batch count. */
bool eif_gpu_cmd_gemm(eif_gpu_context_t *ctx, int kernel, eif_gpu_buf_t *A, eif_gpu_buf_t *B, eif_gpu_buf_t *S,
                      eif_gpu_buf_t *bias, eif_gpu_buf_t *C, const eif_gpu_gemm_pc_t *pc, uint32_t gz);
void eif_gpu_cmd_barrier(eif_gpu_context_t *ctx);
bool eif_gpu_cmd_submit(eif_gpu_context_t *ctx);

#endif /* EIF_GPU_INTERNAL_H */
