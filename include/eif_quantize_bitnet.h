/**
 * @file eif_quantize_bitnet.h
 * @brief BitNet b1.58 Ternary {-1, 0, +1} Quantization and BitLinear Engine.
 *
 * Implements Microsoft BitNet b1.58 (1.58-bit ternary weight) arithmetic:
 * - 2-bit packing: 4 ternary weights per byte
 *   00 -> 0
 *   01 -> +1
 *   10 -> -1
 *   11 -> reserved / 0
 * - Multiplication-free matrix multiplication: inner loop contains ONLY
 *   additions and subtractions, zero hardware multipliers required!
 * - C99 compliant, MISRA friendly, zero heap allocations.
 */

#ifndef EIF_QUANTIZE_BITNET_H
#define EIF_QUANTIZE_BITNET_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Unpack a single ternary value from a packed byte.
 * @param byte Byte containing 4 ternary values.
 * @param pos Position within byte (0, 1, 2, or 3).
 * @return Decoded weight in {-1, 0, +1}.
 */
int8_t eif_unpack_ternary(uint8_t byte, int pos);

/**
 * @brief Pack an array of ternary values ({-1, 0, +1}) into 2-bit packed bytes.
 * @param ternary Input array of int8 values (-1, 0, 1).
 * @param packed Output buffer for packed bytes.
 * @param count Number of ternary elements.
 * @return Number of bytes written ((count + 3) / 4).
 */
size_t eif_pack_ternary(const int8_t *ternary, uint8_t *packed, size_t count);

/**
 * @brief Dense Base-3 Ternary Packing (5 ternary weights per byte, 1.60 bits/weight).
 *
 * Utilizes 3^5 = 243 <= 255 to store 5 ternary values per byte, achieving 20.0x
 * compression over FP32 (20% denser than standard 2-bit packing).
 *
 * @param ternary Input array of int8 values (-1, 0, +1).
 * @param packed Output buffer for packed bytes.
 * @param count Number of ternary elements.
 * @return Number of bytes written ((count + 4) / 5).
 */
size_t eif_pack_ternary_dense(const int8_t *ternary, uint8_t *packed, size_t count);

/**
 * @brief Multiplication-free BitLinear Matrix-Vector Multiplication for FP32 inputs.
 *
 * Computes: output[rows] = scales[row] * (W_ternary[rows, cols] @ input[cols]) + bias[row]
 *
 * The inner loop uses ONLY additions and subtractions. No floating-point or integer
 * multiplications occur across columns.
 *
 * @param weights Packed 2-bit ternary weights (rows * ((cols + 3) / 4) bytes)
 * @param scales Per-row scale factors (gamma)
 * @param input FP32 input vector of length cols
 * @param bias Optional FP32 bias vector of length rows (can be NULL)
 * @param output Output FP32 vector of length rows
 * @param rows Number of output features
 * @param cols Number of input features
 */
void eif_matmul_bitnet_f32(const uint8_t *weights,
                           const float *scales,
                           const float *input,
                           const float *bias,
                           float *output,
                           int rows,
                           int cols);

/**
 * @brief Fast Multiplication-free BitLinear Matrix-Vector Multiplication with Dense 1.60-bit weights.
 *
 * Decodes 5 weights per byte using a 256-entry L1 lookup table (1.25 KB),
 * performing pure additions and subtractions at 20.0x compression.
 *
 * @param weights Packed base-3 ternary weights (rows * ((cols + 4) / 5) bytes)
 * @param scales Per-row scale factors (gamma)
 * @param input FP32 input vector of length cols
 * @param bias Optional FP32 bias vector of length rows (can be NULL)
 * @param output Output FP32 vector of length rows
 * @param rows Number of output features
 * @param cols Number of input features
 */
void eif_matmul_bitnet_dense_f32(const uint8_t *weights,
                                 const float *scales,
                                 const float *input,
                                 const float *bias,
                                 float *output,
                                 int rows,
                                 int cols);


/**
 * @brief Pure Integer BitLinear Matrix-Vector Multiplication for INT8 inputs.
 *
 * Computes:
 *   isum = sum(W_ternary[i, j] * input[j])  (ONLY additions and subtractions)
 *   output[i] = quantize(isum * effective_scale + bias)
 *
 * @param weights Packed 2-bit ternary weights
 * @param scale_w Weight scale gamma
 * @param input INT8 input vector
 * @param scale_in Input activation scale
 * @param bias Optional INT32 bias (can be NULL)
 * @param output Output INT8 vector
 * @param scale_out Output activation scale
 * @param rows Number of output features
 * @param cols Number of input features
 */
void eif_matmul_bitnet_int8(const uint8_t *weights,
                            float scale_w,
                            const int8_t *input,
                            float scale_in,
                            const int32_t *bias,
                            int8_t *output,
                            float scale_out,
                            int rows,
                            int cols);

/**
 * @brief Fused W1 (Gate) and W3 (Up) BitLinear Matrix-Vector Multiplication.
 * Evaluates W1 and W3 in a single OpenMP parallel region.
 */
void eif_matmul_bitnet_f32_w1w3(float *out_w1, float *out_w3,
                                const uint8_t *w1, const uint8_t *w3,
                                const float *scale_w1, const float *scale_w3,
                                const float *input, int hidden_dim, int dim);

/**
 * @brief Quantize an FP32 activation vector to INT8 with dynamic absmax scaling.
 *
 * Implements BitNet b1.58 activation quantization:
 *   scale = max(|x|) / 127.0
 *   x_i8[i] = clip(round(x[i] / scale), -128, 127)
 *
 * @param x_f32     Input FP32 array of length n
 * @param x_i8      Output INT8 array of length n
 * @param out_scale Output float scale (absmax / 127.0f)
 * @param n         Number of elements
 */
void eif_quantize_activation_i8(const float *x_f32,
                                int8_t *x_i8,
                                float *out_scale,
                                int n);

/**
 * @brief Pure Integer BitLinear GEMV with INT8 activation inputs.
 *
 * Performs matrix-vector multiplication where weights are 2-bit ternary {-1, 0, +1}
 * and activations are INT8. The inner loop contains zero floating-point operations.
 *
 * @param weights   Packed 2-bit ternary weights
 * @param scales    Per-row scale factors (gamma)
 * @param input_i8  INT8 activation vector
 * @param scale_in  Dynamic input activation scale
 * @param bias      Optional FP32 bias vector (can be NULL)
 * @param output    Output FP32 vector
 * @param rows      Number of output features
 * @param cols      Number of input features
 */
void eif_matmul_bitnet_i8xternary_f32(const uint8_t *weights,
                                      const float *scales,
                                      const int8_t *input_i8,
                                      float scale_in,
                                      const float *bias,
                                      float *output,
                                      int rows,
                                      int cols);

/**
 * @brief Auto-quantizing BitLinear GEMV (FP32 in -> INT8 on-the-fly -> FP32 out).
 *
 * Dynamically quantizes FP32 input to INT8 in L1 cache and computes multiplication-free
 * integer additions/subtractions across all rows.
 */
void eif_matmul_bitnet_act_quant_f32(const uint8_t *weights,
                                     const float *scales,
                                     const float *input_f32,
                                     const float *bias,
                                     float *output,
                                     int rows,
                                     int cols);

/**
 * @brief Sub-byte Block-Quantized BitLinear Matrix-Vector Multiplication.
 *
 * Partition rows into blocks of block_size (e.g. 32, 64, or 128 elements),
 * each having an individual scale factor. Matches llama.cpp block quantization (Q2_K / IQ2).
 *
 * @param weights      Packed 2-bit ternary weights
 * @param block_scales Block scale factors [rows * ((cols + block_size - 1) / block_size)]
 * @param block_size   Size of each quantization block (e.g. 64 or 128)
 * @param input        FP32 input vector
 * @param bias         Optional FP32 bias vector
 * @param output       Output FP32 vector
 * @param rows         Number of rows
 * @param cols         Number of columns
 */
void eif_matmul_bitnet_block_f32(const uint8_t *weights,
                                 const float *block_scales,
                                 int block_size,
                                 const float *input,
                                 const float *bias,
                                 float *output,
                                 int rows,
                                 int cols);

/**
 * @brief Interleaves row-major 2-bit packed weights into groups of 4 rows for T-MAC cache optimization.
 *
 * Storage order: For each 4-row block, weights for column group g are contiguous:
 * [W0[g], W1[g], W2[g], W3[g]]. Enables single 32-bit loads per group across 4 output rows.
 *
 * @param src_row_major Input packed 2-bit weights in standard row-major layout
 * @param dst_interleaved Output buffer for interleaved weights (same total bytes: rows * ((cols + 3)/4))
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Total number of bytes written
 */
size_t eif_interleave_weights_4rows(const uint8_t *src_row_major,
                                    uint8_t *dst_interleaved,
                                    int rows,
                                    int cols);

/**
 * @brief High-performance T-MAC GEMV using pre-interleaved 4-row weight layout.
 *
 * Eliminates cache-line striding by reading 4 rows per group in a single 32-bit memory access.
 */
void eif_matmul_bitnet_tmac_interleaved_f32(const uint8_t *interleaved_weights,
                                           const float *scales,
                                           const float *input,
                                           const float *bias,
                                           float *output,
                                           int rows,
                                           int cols);

/**
 * @brief Largest power of two <= 1024 that divides n (576 -> 64, 1536 -> 512).
 *        Block size of the Hadamard rotation used by BitLinear(hadamard=True) in the Python SDK.
 */
int eif_hadamard_block_size(int n);

/**
 * @brief Largest group size <= max_group that divides n and is a multiple of 32 (0 = one scale per row).
 *        Mirrors resolve_group_size() in the Python SDK (576 with 128 -> 96, 1536 with 128 -> 128).
 */
int eif_bitnet_group_size(int n, int max_group);

/**
 * @brief In-place rotation x <- FWHT_blocks(x * signs), orthonormal (1/sqrt(block) per block).
 *
 * Applies the activation side of a Hadamard-rotated BitLinear: the weights were stored as W·D·H, so
 * (W·D·H)(x·D·H)^T = W x^T. Uses additions/subtractions only plus one scale per element; no allocation.
 * Butterfly order matches fwht_blocks() in sdk/eif_tools/quantize/bitnet.py.
 *
 * @param x      Vector of length n (rotated in place)
 * @param signs  +-1 per element (D), length n; NULL means all +1
 * @param n      Length (multiple of block)
 * @param block  Power-of-two block size
 */
void eif_hadamard_rotate_f32(float *x, const float *signs, int n, int block);

#ifdef __cplusplus
}
#endif

#endif /* EIF_QUANTIZE_BITNET_H */
