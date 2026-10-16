/**
 * @file eif_llm.h
 * @brief EIF Large Language Model (LLM / SLM) Unified C99 Runtime Engine
 *
 * Provides a clean, polymorphic, zero-heap C99 facade that unifies diverse
 * edge language model architectures behind a single, consistent client API:
 *
 * Supported Architectures:
 *   1. LLaMA / SmolLM2 / Granite-Docling (Standard Decoder-Only Transformer):
 *      - BitNet b1.58 ternary weights {-1, 0, +1} (2-bit & 1.60-bit dense base-3)
 *      - RoPE (Rotary Position Embeddings), GQA (Grouped Query Attention)
 *      - SwiGLU FFN activation
 *      - Backed by: eif_tinyllm
 *
 *   2. Qwen3.5 (Hybrid Gated DeltaNet + Full Attention):
 *      - Linear-time O(1) recurrent SSM state (DeltaNet) + standard GQA Attention
 *      - Selectable embedding & LM head quantization (INT8, INT4, 2-bit, FP32)
 *      - BitNet b1.58 linear layers with fused SwiGLU
 *      - Backed by: eif_qwen35
 *
 * Zero-Heap Guarantee:
 *   All allocations can be supplied via a single user-provided memory arena,
 *   enabling deterministic embedded deployment without runtime malloc().
 */

#ifndef EIF_LLM_H
#define EIF_LLM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "eif_bpe_tokenizer.h"
#include "eif_gpu.h"
#include "eif_qwen35.h"
#include "eif_tinyllm.h"
#include "eif_bert.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Supported LLM Architecture Types in EIF Engine */
typedef enum {
    EIF_LLM_ARCH_UNKNOWN = 0,
    EIF_LLM_ARCH_SMOLLM2 = 1,   /**< LLaMA / SmolLM2 / Granite decoder-only transformer */
    EIF_LLM_ARCH_QWEN35  = 2,   /**< Qwen3.5 hybrid Gated DeltaNet + Full Attention */
    EIF_LLM_ARCH_BERT    = 3,   /**< BERT / MiniLM / Granite-107M encoder transformer */
    EIF_LLM_ARCH_QWEN2   = 4,   /**< Qwen2 / Qwen2.5 causal transformer */
} eif_llm_arch_t;

/** Generation sampling configuration */
typedef struct {
    int max_new_tokens;       /**< Maximum number of tokens to generate (default: 64) */
    float temperature;        /**< Sampling temperature: 0.0 = greedy argmax, >0.0 = stochastic (default: 0.7) */
    float top_p;              /**< Nucleus sampling threshold (default: 0.9) */
    float repetition_penalty; /**< Repetition penalty: 1.0 = disabled, >1.0 = penalize seen tokens (default: 1.15) */
    int eos_token_id;         /**< Stop generation if this token is produced (-1 = use model default) */
    int kv_type;              /**< KV Cache: 0 = FP32, 1 = INT8 Quantized (default: 0) */
    bool use_gpu;             /**< Request GPU acceleration (auto falls back to CPU if unavailable) */
} eif_llm_gen_config_t;

/** Callback function for token generation stream */
typedef void (*eif_llm_token_callback_fn)(const char *piece, int token_id, void *user_data);

/**
 * @brief Unified LLM Engine Handle
 *
 * Encapsulates model weights, runtime state, and tokenizer behind
 * a single polymorphic C99 runtime.
 */
typedef struct {
    eif_llm_arch_t arch;
    char model_path[512];
    char tokenizer_path[512];

    /* Common model configuration */
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
    int current_pos;

    /* Polymorphic backends */
    union {
        tinyllm_t  tinyllm;
        qwen35_t   qwen35;
        eif_bert_t bert;
    } backend;

    /* Embedded tokenizer */
    eif_bpe_tokenizer_t tokenizer;
    bool tokenizer_loaded;

    /* Optional GPU Acceleration Context (Vulkan or CPU Fallback) */
    eif_gpu_context_t gpu;

    /* Memory buffers */
    uint8_t *buffer;
    size_t   buffer_size;
    bool     owns_buffer;
    bool     is_initialized;
} eif_llm_t;

/**
 * @brief Auto-detect model architecture from binary file header.
 *
 * @param model_path Path to model file (.eifm or .bin)
 * @return Detected architecture enum
 */
eif_llm_arch_t eif_llm_detect_arch(const char *model_path);

/**
 * @brief Calculate recommended buffer size (RAM) for model execution.
 *
 * @param model_path Path to model file
 * @return Recommended memory size in bytes, or 0 on error
 */
size_t eif_llm_compute_buffer_size(const char *model_path);

/**
 * @brief Load model and tokenizer into unified LLM runtime.
 *
 * @param llm            Engine handle to initialize
 * @param model_path     Path to model binary (.eifm or .bin)
 * @param tokenizer_path Path to tokenizer.bin (if NULL, auto-discovered in model directory)
 * @param buffer         Memory buffer for model weights & state (if NULL, allocates buffer)
 * @param buffer_size    Size of memory buffer
 * @return 0 on success, negative error code on failure
 */
int eif_llm_load(eif_llm_t *llm, const char *model_path, const char *tokenizer_path,
                 void *buffer, size_t buffer_size);

/**
 * @brief Execute forward pass for a single token at position pos.
 *
 * @param llm   Engine handle
 * @param token Token ID
 * @param pos   Sequence position (0-based)
 * @return 0 on success, negative error code on failure
 */
int eif_llm_forward(eif_llm_t *llm, int token, int pos);

/**
 * @brief Execute forward pass for a single token without computing vocabulary logits.
 *
 * Runs transformer layers and updates internal hidden state and KV cache,
 * skipping the expensive [dim x vocab_size] classifier projection.
 *
 * @param llm   Engine handle
 * @param token Token ID
 * @param pos   Sequence position (0-based)
 * @return 0 on success, negative error code on failure
 */
int eif_llm_forward_no_logits(eif_llm_t *llm, int token, int pos);

/**
 * @brief Get pointer to output logits array [vocab_size].
 *
 * @param llm Engine handle
 * @return Float pointer to logits, or NULL on error
 */
float* eif_llm_get_logits(eif_llm_t *llm);

/**
 * @brief Sample next token from current logits using temperature and top-p.
 *
 * @param llm         Engine handle
 * @param temperature Sampling temperature (0.0 = greedy argmax)
 * @param top_p       Top-p threshold (1.0 = full distribution)
 * @return Sampled token ID
 */
int eif_llm_sample(eif_llm_t *llm, float temperature, float top_p);

/**
 * @brief High-level text generation from prompt string.
 *
 * Encodes prompt, runs prefill, and samples autoregressively,
 * streaming decoded tokens via callback or stdout.
 *
 * @param llm       Engine handle
 * @param prompt    Input text prompt
 * @param cfg       Generation config (NULL for defaults: 64 tokens, temp 0.7, top_p 0.9)
 * @param cb        Token callback function (NULL prints directly to stdout)
 * @param user_data User context pointer passed to callback
 * @return Number of generated tokens on success, negative on error
 */
int eif_llm_generate(eif_llm_t *llm, const char *prompt, const eif_llm_gen_config_t *cfg,
                     eif_llm_token_callback_fn cb, void *user_data);

/**
 * @brief High-level multimodal vision generation for Docling.
 *
 * Prefills 13 tiles (12 local grid + 1 global thumbnail) with spatial tokens,
 * applies sampling with repetition penalty, and streams Docling markdown tokens.
 *
 * @param llm        Engine handle
 * @param image_path Path to input document image
 * @param prompt     Prompt text (e.g. "Convert this page to docling.")
 * @param cfg        Generation config
 * @param cb         Token callback function (NULL prints directly to stdout)
 * @param user_data  User data for callback
 * @return Number of generated tokens
 */
int eif_llm_generate_vision_docling(eif_llm_t *llm, const char *image_path, const char *prompt,
                                    const eif_llm_gen_config_t *cfg,
                                    eif_llm_token_callback_fn cb, void *user_data);

/**
 * @brief Reset KV cache / recurrent state for a new conversation.
 *
 * @param llm Engine handle
 */
void eif_llm_reset(eif_llm_t *llm);

/**
 * @brief Release resources associated with unified LLM engine.
 *
 * @param llm Engine handle
 */
void eif_llm_free(eif_llm_t *llm);

/** Pooling strategy for dense text embedding vectors */
typedef enum {
    EIF_LLM_POOL_MEAN = 0,  /**< Mean pooling over all tokens (standard for embeddings) */
    EIF_LLM_POOL_LAST = 1,  /**< Last token hidden state (decoder / causal models) */
    EIF_LLM_POOL_CLS  = 2,  /**< First token / CLS representation */
} eif_llm_pool_mode_t;

/**
 * @brief Get pointer to last computed normalized hidden state vector [dim].
 *
 * @param llm Engine handle
 * @return Float pointer to hidden vector of size llm->dim, or NULL on error
 */
float *eif_llm_get_hidden_state(eif_llm_t *llm);

/**
 * @brief Compute dense text embedding vector from prompt text.
 *
 * Runs prompt tokens through transformer without generating tokens,
 * pools the final hidden state representations according to pool_mode,
 * and L2-normalizes the resulting vector.
 *
 * @param llm           Engine handle
 * @param prompt        Input text to embed
 * @param out_embedding Buffer of at least [llm->dim] floats to receive normalized embedding
 * @param pool_mode     Pooling mode (EIF_LLM_POOL_MEAN or EIF_LLM_POOL_LAST)
 * @return 0 on success, negative error code on failure
 */
int eif_llm_embed(eif_llm_t *llm, const char *prompt, float *out_embedding, eif_llm_pool_mode_t pool_mode);

/**
 * @brief Compute cosine similarity between two float vectors.
 *
 * @param a   First vector [dim]
 * @param b   Second vector [dim]
 * @param dim Dimension of vectors
 * @return Cosine similarity in range [-1.0, 1.0], or 0.0 on error
 */
float eif_llm_cosine_similarity(const float *a, const float *b, int dim);

#ifdef __cplusplus
}
#endif

#endif /* EIF_LLM_H */
