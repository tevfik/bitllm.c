/**
 * @file eif_tinyllm.h
 * @brief Tiny LLM Inference Engine for Microcontrollers
 *
 * Minimal transformer inference for models like TinyStories.
 * Designed for ESP32-S3 with PSRAM or similar MCUs.
 *
 * Features:
 * - INT4/INT8 weight quantization
 * - Efficient memory usage
 * - Character-level or BPE tokenization
 * - Greedy and top-k sampling
 *
 * Based on llama2.c by Andrej Karpathy
 */

#ifndef EIF_TINYLLM_H
#define EIF_TINYLLM_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include "eif_bpe_tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Configuration
// =============================================================================

#ifndef TINYLLM_MAX_SEQ_LEN
#define TINYLLM_MAX_SEQ_LEN 256
#endif

#ifndef TINYLLM_MAX_VOCAB
#define TINYLLM_MAX_VOCAB 4096
#endif

// =============================================================================
// Types
// =============================================================================

/**
 * @brief Token type
 */
typedef int32_t tinyllm_token_t;

/**
 * @brief Quantization type
 */
typedef enum {
    TINYLLM_QTYPE_FP32 = 0,
    TINYLLM_QTYPE_INT8 = 1,
    TINYLLM_QTYPE_INT4 = 2,
    TINYLLM_QTYPE_BITNET_158 = 3
} tinyllm_qtype_t;

/**
 * @brief Transformer configuration
 */
typedef struct {
    int dim;               // Transformer dimension (hidden size)
    int hidden_dim;        // FFN hidden dimension
    int n_layers;          // Number of transformer layers
    int n_heads;           // Number of attention heads
    int n_kv_heads;        // Number of KV heads (for GQA)
    int vocab_size;        // Vocabulary size
    int seq_len;           // Maximum sequence length
    tinyllm_qtype_t qtype; // Weight quantization type
    int head_dim;          // Head dimension (if 0, defaults to dim / n_heads)
    int   rope_interleaved; // 1 = RoPE rotates adjacent pairs (llama.cpp llama-family GGUF), 0 = half-split (HF)
    float rope_theta;      // RoPE base frequency (if 0.0f, defaults to 10000.0f or 100000.0f)
    int kv_type;           // KV cache type: 0 = FP32 (default), 1 = INT8 quantized
    int bitnet_group;      // BitNet: max ternary scale group (0 = one scale per row); per matrix the largest
                           //   multiple of 32 <= this dividing the input width (eif_bitnet_group_size)
    int bitnet_hadamard;   // BitNet: weights stored as W·D·H; inputs are rotated (eif_hadamard_rotate_f32)
} tinyllm_config_t;

/** BitNet matrix ids for tinyllm_weights_t.bitnet_gscale */
enum { TL_BN_WQ = 0, TL_BN_WK, TL_BN_WV, TL_BN_WO, TL_BN_W1, TL_BN_W3, TL_BN_W2, TL_BN_COUNT };

/**
 * @brief Transformer weights
 */
typedef struct {
    // Token embedding table [vocab_size, dim]
    void *token_embedding;
    float *token_embedding_scale;

    // RMSNorm weights [dim]
    float *rms_att_weight;
    float *rms_ffn_weight;

    // Attention weights
    void *wq; // Query projection [dim, dim]
    void *wk; // Key projection [dim, dim]
    void *wv; // Value projection [dim, dim]
    void *wo; // Output projection [dim, dim]
    float *attn_scale;

    // FFN weights
    void *w1; // Gate projection [dim, hidden_dim]
    void *w2; // Down projection [hidden_dim, dim]
    void *w3; // Up projection [dim, hidden_dim]
    float *ffn_scale;

    // Output weights
    float *rms_final_weight;
    void *wcls; // Classifier weights (can share with token_embedding)
    float *wcls_scale;

    // Optional Attention QKV biases (Qwen / Qwen2 architecture)
    float *bq; // [n_layers, q_dim]
    float *bk; // [n_layers, kv_dim]
    float *bv; // [n_layers, kv_dim]

    // Optional BitNet b1.58 SubLN RMSNorm scales
    float *rms_sub_att_weight; // [n_layers, q_dim]
    float *rms_sub_ffn_weight; // [n_layers, hidden_dim]

    // Optional grouped BitNet scales, one array per matrix kind: [n_layers, rows, in_dim / group]
    // (set when config.bitnet_group > 0; attn_scale/ffn_scale are then unused for BitNet matmuls)
    const float *bitnet_gscale[TL_BN_COUNT];
    // Optional Hadamard signs (config.bitnet_hadamard): one +-1 vector per input width
    const float *had_signs_dim;    // [dim]     inputs of wq/wk/wv/w1/w3
    const float *had_signs_q;      // [q_dim]   input of wo
    const float *had_signs_hidden; // [hidden_dim] input of w2
} tinyllm_weights_t;

/**
 * @brief Runtime state for transformer
 */
typedef struct {
    // Current token position
    int pos;

    // Activation buffers
    float *x;      // Current activation [dim]
    float *xb;     // Buffer [dim]
    float *xb2;    // Buffer [dim]
    float *hb;     // Buffer [hidden_dim]
    float *hb2;    // Buffer [hidden_dim]
    float *q;      // Query [dim]
    float *k;      // Key cache [n_layers, seq_len, dim]
    float *v;      // Value cache [n_layers, seq_len, dim]
    float *att;    // Attention scores [n_heads, seq_len]
    float *logits; // Output logits [vocab_size]

    // Key-value cache (FP32)
    float *key_cache;
    float *value_cache;

    // Optional INT8 Quantized Key-value cache
    int8_t *key_cache_i8;
    int8_t *value_cache_i8;
    float  *key_scale;
    float  *value_scale;
} tinyllm_state_t;

/**
 * @brief Tokenizer
 */
typedef struct {
    char **vocab;
    float *vocab_scores;
    int vocab_size;
    int bos_token;
    int eos_token;
} tinyllm_tokenizer_t;

/**
 * @brief Complete LLM context
 */
typedef struct {
    tinyllm_config_t config;
    tinyllm_weights_t weights;
    tinyllm_state_t state;
    tinyllm_tokenizer_t tokenizer;
    void *memory_pool;
    size_t memory_size;
} tinyllm_t;

// =============================================================================
// Memory Management
// =============================================================================

/**
 * @brief Calculate memory required for transformer
 */
size_t tinyllm_memory_size(const tinyllm_config_t *config);

/**
 * @brief Initialize transformer with memory pool
 */
int tinyllm_init(tinyllm_t *llm, const tinyllm_config_t *config, void *memory_pool,
                 size_t pool_size);

/**
 * @brief Free transformer resources
 */
void tinyllm_free(tinyllm_t *llm);

// =============================================================================
// Model Loading
// =============================================================================

/**
 * @brief Load model from binary file
 *
 * Format:
 * - Header: config (28 bytes)
 * - Token embeddings
 * - Layer weights (x n_layers)
 * - Output weights
 *
 * @param llm LLM context
 * @param model_path Path to model file
 * @return 0 on success, negative on error
 */
int tinyllm_load_model(tinyllm_t *llm, const char *model_path);

/**
 * @brief Load model from memory buffer
 */
int tinyllm_load_model_from_memory(tinyllm_t *llm, const uint8_t *data, size_t size);

/**
 * @brief Load tokenizer from file
 */
int tinyllm_load_tokenizer(tinyllm_t *llm, const char *tokenizer_path);

/**
 * @brief Free allocated tokenizer memory
 */
void tinyllm_free_tokenizer(tinyllm_t *llm);

// =============================================================================
// Inference
// =============================================================================

/**
 * @brief Reset transformer state for new generation
 */
void tinyllm_reset(tinyllm_t *llm);

/**
 * @brief Forward pass for single token
 *
 * @param llm LLM context
 * @param token Input token
 * @param pos Position in sequence
 * @return Pointer to logits (vocab_size floats)
 */
float *tinyllm_forward(tinyllm_t *llm, tinyllm_token_t token, int pos);

/**
 * @brief Forward pass with a precomputed embedding vector (e.g. from VLM multimodal projector)
 *
 * @param llm LLM context
 * @param embedding Input embedding vector [dim floats]
 * @param pos Position in sequence
 * @return Pointer to logits (vocab_size floats)
 */
float *tinyllm_forward_embedding(tinyllm_t *llm, const float *embedding, int pos);
float *tinyllm_forward_embedding_ex(tinyllm_t *llm, const float *embedding, int pos, bool compute_logits);
float *tinyllm_forward_no_logits(tinyllm_t *llm, tinyllm_token_t token, int pos);

/**
 * @brief Extract embedding for a token into destination buffer [dim floats]
 */
void tinyllm_get_token_embedding(tinyllm_t *llm, tinyllm_token_t token, float *out_emb);

/**
 * @brief Forward pass for a batch of embeddings processed layer-by-layer
 *
 * Keeps each transformer layer's weights in CPU cache across all sequence tokens,
 * eliminating repeated DRAM traffic during prefill.
 *
 * @param llm LLM context
 * @param embeddings Input embeddings array [n_tokens * dim floats]
 * @param n_tokens Number of tokens in batch
 * @param start_pos Starting sequence position for the batch
 * @param compute_last_logits If true, computes logits for the last token in batch
 * @return Pointer to logits if compute_last_logits is true, else hidden state
 */
float *tinyllm_forward_batch_embeddings(tinyllm_t *llm, const float *embeddings, int n_tokens, int start_pos, bool compute_last_logits);

/**
 * @brief Get pointer to last computed normalized hidden state vector [dim].
 *
 * @param llm LLM context
 * @return Pointer to hidden state (dim floats)
 */
float *tinyllm_get_hidden_state(tinyllm_t *llm);


/**
 * @brief Sample next token from logits (greedy)
 */
tinyllm_token_t tinyllm_sample_greedy(const float *logits, int vocab_size);

/**
 * @brief Sample next token with temperature
 */
tinyllm_token_t tinyllm_sample_temp(const float *logits, int vocab_size, float temperature);

/**
 * @brief Sample next token with top-k sampling
 */
tinyllm_token_t tinyllm_sample_top_k(const float *logits, int vocab_size, int k, float temperature);

/**
 * @brief Generate tokens
 *
 * @param llm LLM context
 * @param prompt Input prompt (NULL for empty)
 * @param output Output buffer for generated tokens
 * @param max_tokens Maximum tokens to generate
 * @param temperature Sampling temperature (0.0 for greedy)
 * @return Number of tokens generated
 */
int tinyllm_generate(tinyllm_t *llm, const char *prompt, char *output, int max_tokens,
                     float temperature);

// =============================================================================
// Tokenization
// =============================================================================

/**
 * @brief Encode text to tokens
 */
int tinyllm_encode(tinyllm_t *llm, const char *text, tinyllm_token_t *tokens, int max_tokens);

/**
 * @brief Decode tokens to text
 */
int tinyllm_decode(tinyllm_t *llm, const tinyllm_token_t *tokens, int n_tokens, char *text,
                   int max_len);

// =============================================================================
// Quantization Helpers
// =============================================================================

/**
 * @brief Dequantize INT4 weight
 */
static inline float tinyllm_dequant_int4(uint8_t packed, int idx, float scale)
{
    int8_t val = (idx == 0) ? ((packed >> 4) & 0x0F) - 8 : (packed & 0x0F) - 8;
    return val * scale;
}

/**
 * @brief Dequantize INT8 weight
 */
static inline float tinyllm_dequant_int8(int8_t val, float scale)
{
    return val * scale;
}

/**
 * @brief Matrix-vector multiply with INT4 weights
 */
void tinyllm_matmul_int4(float *out, const uint8_t *w, const float *in, int in_dim, int out_dim,
                         const float *scales);

/**
 * @brief Multiplication-free Matrix-vector multiply with BitNet b1.58 ternary weights
 */
void tinyllm_matmul_bitnet_158(float *out, const uint8_t *w, const float *in, int in_dim, int out_dim,
                               const float *scales);

/**
 * @brief Matrix-vector multiply with INT8 weights
 */
void tinyllm_matmul_int8(float *out, const int8_t *w, const float *in, int in_dim, int out_dim,
                         float scale);

/**
 * @brief Matrix-vector multiply with INT8 weights and per-row scales
 */
void tinyllm_matmul_int8_rowscale(float *y, const int8_t *w, const float *x, int in_dim, int out_dim,
                                  const float *scales);

/**
 * @brief Quantize FP32 weights to INT8
 * @param src Source FP32 weights
 * @param dst Destination INT8 weights
 * @param n Number of elements
 * @return Scale factor for dequantization
 */
float tinyllm_quantize_int8(const float *src, int8_t *dst, int n);

/**
 * @brief Quantize FP32 weights to INT4 (packed)
 * @param src Source FP32 weights
 * @param dst Destination packed INT4 weights
 * @param n Number of elements
 * @param out_dim Output dimension for per-row scales
 * @return Scale factor for dequantization
 */
float tinyllm_quantize_int4(const float *src, uint8_t *dst, int n, int out_dim);

/**
 * @brief Quantize FP32 weights to BitNet b1.58 ternary format (2-bit packed)
 * @param src Source FP32 weights
 * @param dst Destination packed ternary weights
 * @param n Number of elements
 * @param out_dim Output dimension for per-row scales
 * @return Scale factor for dequantization
 */
float tinyllm_quantize_bitnet_158(const float *src, uint8_t *dst, int n, int out_dim);

/**
 * @brief Quantize all model weights
 * @param llm LLM context with FP32 weights
 * @param qtype Target quantization type
 * @param quant_buffer Buffer to store quantized weights
 * @param buffer_size Size of quant buffer
 * @return 0 on success, negative on error
 */
int tinyllm_quantize_model(tinyllm_t *llm, tinyllm_qtype_t qtype, void *quant_buffer,
                           size_t buffer_size);

/**
 * @brief Save quantized model to disk
 * @param llm LLM context with quantized weights
 * @param filename Output filename
 * @return 0 on success, negative on error
 */
int tinyllm_save_quantized_model(const tinyllm_t *llm, const char *filename);

/**
 * @brief Load quantized model from disk
 * @param llm LLM context to load into (must be initialized with same config)
 * @param filename Model filename
 * @param quant_buffer Buffer to load weights into
 * @param buffer_size Size of buffer
 * @return 0 on success, negative on error
 */
int tinyllm_load_quantized_model(tinyllm_t *llm, const char *filename, void *quant_buffer,
                                 size_t buffer_size);

/**
 * @brief Load Causal LLM directly from GGUF file (v2/v3).
 *
 * Supports Qwen2/Qwen2.5, LLaMA, Granite, and BitNet Causal LLM models in GGUF format.
 *
 * @param llm LLM context to load into
 * @param filename Model filename (.gguf)
 * @param quant_buffer Buffer to load weights into
 * @param buffer_size Size of buffer
 * @return 0 on success, negative on error
 */
int tinyllm_load_gguf(tinyllm_t *llm, const char *filename, void *quant_buffer,
                      size_t buffer_size);

/**
 * @brief Read model config and architecture type from GGUF file metadata.
 */
int tinyllm_read_gguf_config(const char *filename, tinyllm_config_t *cfg,
                             char *out_arch, size_t max_arch);

/**
 * @brief Load embedded vocabulary / BPE tokenizer from GGUF metadata.
 */
int tinyllm_load_gguf_tokenizer(const char *filename, eif_bpe_tokenizer_t *tok);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Print model info
 */
void tinyllm_print_info(const tinyllm_t *llm);

/**
 * @brief Get memory usage statistics
 */
void tinyllm_memory_stats(const tinyllm_t *llm, size_t *used, size_t *total);

#ifdef __cplusplus
}
#endif

#endif // EIF_TINYLLM_H
