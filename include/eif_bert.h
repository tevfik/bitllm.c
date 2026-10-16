/**
 * @file eif_bert.h
 * @brief EIF-Runtime High-Performance BERT Encoder for Dense Text Embeddings
 *
 * Supports:
 * - Architectures: all-MiniLM, Granite-107M, BGE, E5, Arctic
 * - Precisions: INT8 / Q8_0, FP32
 * - Hardware SIMD Acceleration:
 *     - Intel/AMD (x86_64): AVX2 + FMA
 *     - ARM64 (aarch64): ARM NEON + DotProd
 *     - Generic C99 fallback with OpenMP
 * - Embedded WordPiece Tokenization: Standalone C99 text-to-id mapping
 * - Bidirectional Full-Sequence Self-Attention
 */

#ifndef EIF_BERT_H
#define EIF_BERT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EBERT_MAGIC 0x54524542 /* 'BERT' in Little Endian */
#ifndef GGUF_MAGIC
#define GGUF_MAGIC  0x46554747u /* 'GGUF' in Little Endian */
#endif

typedef struct {
    int dim;              /**< Hidden dimension (e.g. 384) */
    int intermediate_dim; /**< FFN intermediate dimension (e.g. 1536) */
    int n_layers;         /**< Number of transformer encoder layers (e.g. 6) */
    int n_heads;          /**< Number of attention heads (e.g. 12) */
    int max_seq_len;      /**< Maximum sequence context length (e.g. 512) */
    int vocab_size;       /**< Vocabulary size (e.g. 30522) */
    int qtype;            /**< 0=FP32, 1=INT8 EIFM separated, 2=INT8 GGUF interleaved */
    bool use_alibi;       /**< Attention with Linear Biases (e.g. Jina Embeddings v2) */
    int pooling_type;     /**< 0/1=Mean pooling, 2=CLS pooling */
} eif_bert_config_t;

typedef struct {
    /* Embedding table pointers */
    const void  *token_emb;
    const float *token_emb_scales;
    const float *pos_emb;
    bool         pos_emb_allocated;
    const float *type_emb;
    const float *emb_norm_w;
    const float *emb_norm_b;

    /* Layer pointers arrays (size n_layers) */
    const void  **q_w;
    const float **q_w_scales;
    const float **q_b;

    const void  **k_w;
    const float **k_w_scales;
    const float **k_b;

    const void  **v_w;
    const float **v_w_scales;
    const float **v_b;

    const void  **out_w;
    const float **out_w_scales;
    const float **out_b;

    const float **att_norm_w;
    const float **att_norm_b;
    const float **att_norm2_w;
    const float **att_norm2_b;

    const void  **ffn_up_w;
    const float **ffn_up_w_scales;
    const float **ffn_up_b;

    const void  **ffn_down_w;
    const float **ffn_down_w_scales;
    const float **ffn_down_b;

    const float **ffn_norm_w;
    const float **ffn_norm_b;

    /* Optional (NULL arrays when absent): GEGLU gate projection and Q/K LayerNorm
     * (e.g. Jina Embeddings v2 code). */
    const void  **ffn_gate_w;
    const float **q_norm_w;
    const float **q_norm_b;
    const float **k_norm_w;
    const float **k_norm_b;
} eif_bert_weights_t;

#include "eif_bpe_tokenizer.h"

typedef enum {
    EBERT_TOKENIZER_WORDPIECE = 0, /**< Traditional BERT / MiniLM / BGE (## prefix) */
    EBERT_TOKENIZER_BPE       = 1, /**< Byte-Level BPE / RoBERTa / Granite (Ġ prefix) */
    EBERT_TOKENIZER_SPM       = 2  /**< SentencePiece / XLM-RoBERTa / Multilingual-E5 (  prefix) */
} eif_bert_tokenizer_type_t;

typedef struct {
    char **tokens;
    int32_t *hash_table;
    int hash_size;
    uint32_t hash_mask;
    int vocab_size;
    int cls_id;
    int sep_id;
    int unk_id;
    int pad_id;
    bool is_spm_prefix;
    eif_bert_tokenizer_type_t tok_type;
    eif_bpe_tokenizer_t bpe_tok;
} eif_bert_vocab_t;

typedef struct {
    eif_bert_config_t  config;
    eif_bert_weights_t weights;
    eif_bert_vocab_t   vocab;

    /* File backing (mmap) */
    void   *mmap_addr;
    size_t  mmap_size;
    bool    is_mmap;

    /* Dynamic scratch activation buffers */
    float *scratch_seq_x;    /**< [max_seq_len * dim] */
    float *scratch_seq_xb;   /**< [max_seq_len * dim] */
    float *scratch_q;        /**< [max_seq_len * dim] */
    float *scratch_k;        /**< [max_seq_len * dim] */
    float *scratch_v;        /**< [max_seq_len * dim] */
    float *scratch_att;      /**< [n_heads * max_seq_len * max_seq_len] */
    float *scratch_inter;    /**< [max_seq_len * intermediate_dim] */
    float *scratch_proj;     /**< [max_seq_len * dim] */

    bool is_initialized;
} eif_bert_t;

/**
 * @brief Load EIF BERT binary model (zero-copy mmap).
 *
 * @param bert       Output BERT engine handle
 * @param model_path Path to .eifm binary model file
 * @return 0 on success, negative error code on failure
 */
int eif_bert_load(eif_bert_t *bert, const char *model_path);

/**
 * @brief Tokenize input text string using embedded WordPiece vocabulary.
 *
 * @param bert       Loaded BERT engine
 * @param text       Input text string
 * @param out_tokens Buffer for token IDs
 * @param max_tokens Maximum number of tokens to store
 * @return Number of tokens written, or negative error code
 */
int eif_bert_tokenize(const eif_bert_t *bert, const char *text, int32_t *out_tokens, int max_tokens);

/**
 * @brief Compute dense semantic embedding vector for input text.
 *
 * Performs tokenization, multi-head bidirectional attention, FFN,
 * mean pooling, and L2 normalization.
 *
 * @param bert          Loaded BERT engine
 * @param text          Input text to embed
 * @param out_embedding Output float buffer of size bert->config.dim
 * @return 0 on success, negative error code on failure
 */
int eif_bert_embed(eif_bert_t *bert, const char *text, float *out_embedding);

/**
 * @brief Compute dense semantic embeddings for a batch of input texts.
 *
 * @param bert           Loaded BERT engine
 * @param texts          Array of input text strings (size n_texts)
 * @param n_texts        Number of texts in batch
 * @param out_embeddings Output buffer of size (n_texts * bert->config.dim)
 * @return 0 on success, negative error code on failure
 */
int eif_bert_embed_batch(eif_bert_t *bert, const char **texts, int n_texts, float *out_embeddings);

/**
 * @brief Release resources associated with BERT engine.
 *
 * @param bert Engine handle
 */
void eif_bert_free(eif_bert_t *bert);

#ifdef __cplusplus
}
#endif

#endif /* EIF_BERT_H */
