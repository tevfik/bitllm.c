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
    int dim;              /**< OUTPUT embedding dimension (post-projection; e.g. 384) */
    int dim_model;        /**< Backbone hidden dimension (0 = dim; EmbeddingGemma2: 512 backbone -> 768 out) */
    int intermediate_dim; /**< FFN intermediate dimension (e.g. 1536) */
    int n_layers;         /**< Number of transformer encoder layers (e.g. 6) */
    int n_heads;          /**< Number of attention heads (e.g. 12) */
    int max_seq_len;      /**< Maximum sequence context length (e.g. 512) */
    int vocab_size;       /**< Vocabulary size (e.g. 30522) */
    int qtype;            /**< 0=FP32, 1=INT8 EIFM separated, 2=INT8 GGUF interleaved */
    bool arch_modern;     /**< ModernBERT: pre-norm, fused GLU FFN, local/global attention with two RoPE bases */
    int swa_pattern;      /**< ModernBERT: every swa_pattern-th layer (starting at 0) is global, the rest are local */
    bool use_rope;        /**< rotary Q/K (NeoX half-split) instead of absolute positions/ALiBi (nomic-bert) */
    bool ffn_silu;        /**< gated FFN activation is SiLU (SwiGLU) instead of GELU (GEGLU) */
    bool use_alibi;       /**< Attention with Linear Biases (e.g. Jina Embeddings v2) */
    int pooling_type;     /**< 0/1=Mean pooling, 2=CLS pooling */

    /* Extended attention/normalization controls (0/legacy when absent) */
    int   att_type;        /**< 0=absolute pos, 1=ALiBi, 2=RoPE per-head (gemma-embedding) */
    int   n_kv_heads;      /**< KV heads; 0 = same as n_heads (MHA) */
    int   head_dim;        /**< Per-head dim; 0 = dim / n_heads */
    float rope_freq_base;  /**< RoPE theta (att_type==2, e.g. 1e6) */
    float rope_freq_base_swa; /**< RoPE theta for SWA layers (gemma-embedding2: 1e4) */
    int   sliding_window;  /**< SWA window from GGUF (0 = unknown) */
    int   norm_style;      /**< 0=LayerNorm, 1=Gemma RMS * (1+w) */
    float norm_eps;        /**< LN eps (1e-12 default) / RMS eps (1e-6) */
    float emb_scale;       /**< 0=none; else embeddings *= this before first norm (gemma: sqrt(dim)) */
    float attn_scale;      /**< 0=legacy 1/sqrt(head_dim); else fixed attention scale
                             *   (gemma-embedding2 uses 1.0: QK-norm makes scaling redundant) */
    int   n_embd_per_layer;/**< Per-layer embedding width (gemma-embedding2: 512; 0 = unsupported) */
} eif_bert_config_t;

typedef struct {
    /* Embedding table pointers */
    const void  *token_emb;
    const float *token_emb_scales;
    const float *pos_emb;
    bool         pos_emb_allocated;
    int          pos_rows;  /**< actual row count of position_embd (0 = full context) */
    const float *type_emb;
    int          type_rows;   /**< segment-embedding rows (BERT 2, XLM-R 1) */
    /* Cross-encoder head (cls.*), NULL for plain embedding models */
    const void  *cls_w;       /**< [dim x dim] dense layer, Q8_0 (cls_qtype 2) or FP32 (0) */
    int          cls_qtype;
    float       *cls_w_owned; /**< FP32 copy when the file stores FP16 */
    const float *cls_b;
    const float *cls_out_w;   /**< [dim] output projection */
    float        cls_out_b;
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

    /* Gemma-embedding style (norm_style==1): four per-layer RMS norms +
     * final output norm. Q/K norms above are reused (RMS, biasless).
     * NOTE: like llama.cpp converters, stored norm weights are already the
     * trained (1+w) values -> runtime is plain rms_norm(x) * w. */
    const float **pre_attn_norm_w;    /**< blk.N.attn_norm.weight (pre-attention) */
    const float **attn_post_norm_w;   /**< blk.N.post_attention_norm.weight (attention OUTPUT, before residual) */
    const float **pre_ffn_norm_w;     /**< blk.N.ffn_norm.weight (pre-FFN) */
    const float **post_ffw_norm_w;    /**< blk.N.post_ffw_norm.weight (FFN output, before residual) */
    const float  *output_norm_w;      /**< output_norm.weight (final) */

    /* gemma-embedding2 (EmbeddingGemma-2): per-layer embeddings (PLE) +
     * per-layer head config + final output projection. */
    const void  **g2_inp_gate;        /**< blk.N.inp_gate.weight [dim,dim] */
    const void  **g2_proj;            /**< blk.N.proj.weight [dim,dim] */
    const float **g2_post_norm;       /**< blk.N.post_norm.weight [dim] */
    const float **g2_out_scale;       /**< blk.N.layer_output_scale.weight [1] */
    const float  *g2_pl_model_proj;   /**< per_layer_model_proj [dim, dim*n_layers] (dequantized BF16) */
    const float  *g2_pl_proj_norm;    /**< per_layer_proj_norm [dim] */
    const void  *g2_out;              /**< output.weight [dim_model, dim_out] final projection */
    int          *g2_head_dim;        /**< per-layer head dim (256 SWA / 512 global) */
    int          *g2_n_kv;            /**< per-layer KV head count (2 SWA / 1 global) */
    int           g2_max_q_dim;       /**< max over layers of n_heads*head_dim (scratch sizing) */
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
    float *scores;       /**< tokenizer.ggml.scores (Unigram models), NULL otherwise */
    char **merge_strs;      /**< tokenizer.ggml.merges ("left right", rank = index) for BPE vocabularies */
    int32_t *merge_hash;    /**< open-addressing table: merge rank + 1, 0 = empty */
    int n_merges;
    uint32_t merge_mask;
    bool bpe_merges;        /**< Gemma4-style SPM BPE (merge ranks over raw UTF-8, spaces escaped to U+2581) */
    bool spm_bigram;     /**< llama.cpp SentencePiece merge tokenisation (tokenizer.ggml.model == "llama") */
    bool add_space_prefix; /**< prepend U+2581 to the first word (tokenizer.ggml.add_space_prefix) */
    int byte_bpe;         /**< byte-level BPE by merge rank: 0 off, 1 GPT-2 pre-tokenizer, 2 Granite multilingual (+ignore_merges) */
    bool unigram;        /**< Viterbi Unigram tokenisation (tokenizer.ggml.model == "t5") */
    bool lowercase;      /**< lowercase input before WordPiece lookup (uncased BERT vocab) */
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
    float *scratch_proj;     /**< [max_seq_len * max(dim, dim_out)] */
    float *scratch_pl;       /**< gemma-embedding2: per-layer input table [max_t * dim_model * n_layers] (precomputed, RMS-normed) */
    float *scratch_g2;       /**< gemma-embedding2: PLE gate scratch [max_t * dim_model] */
    float *scratch_attout;   /**< gemma-embedding2: attention output accum [max_t * max_q_dim] (heads can exceed dim_model) */

    int max_t_alloc;         /**< token capacity of the scratch buffers */

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
 * @param out_embedding Output float buffer of size bert->config.dim (the
 *                      OUTPUT embedding dim; gemma-embedding2: 768)
 * @return 0 on success, negative error code on failure
 */
int eif_bert_embed(eif_bert_t *bert, const char *text, float *out_embedding);

/** Tokenise a (query, document) pair: RoBERTa/XLM-R style `<s> a </s></s> b </s>`, BERT style
 *  `[CLS] a [SEP] b [SEP]` with segment ids in @p types (0/1). The document is truncated to fit @p max_tokens.
 *  @return number of tokens, negative on error. */
int eif_bert_tokenize_pair(const eif_bert_t *bert, const char *a, const char *b, int32_t *out_tokens, uint8_t *out_types, int max_tokens);

/** 1 if the model carries a cross-encoder head (cls.weight / cls.output.weight), e.g. bge-reranker-v2-m3. */
int eif_bert_is_ranker(const eif_bert_t *bert);

/** Relevance logit of @p doc for @p query (higher = more relevant; sigmoid() gives the 0..1 score). 0 on success. */
int eif_bert_rank(eif_bert_t *bert, const char *query, const char *doc, float *score);

/** Scores for several documents against one query. */
int eif_bert_rank_batch(eif_bert_t *bert, const char *query, const char **docs, int n_docs, float *scores);

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
