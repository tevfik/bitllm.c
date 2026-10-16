/**
 * @file eif_qwen35.h
 * @brief Qwen3.5-0.8B Hybrid LLM Inference Engine (Pure C99)
 *
 * Supports the Qwen3_5ForConditionalGeneration architecture:
 *   - 18x GatedDeltaNet (linear_attention) layers  -- O(1) recurrent SSM
 *   - 6x  Full Attention (full_attention) layers    -- GQA with RoPE
 *   - 24x SwiGLU FFN layers
 *   - Vision ViT encoder + PatchMerger projector
 *
 * All nn.Linear weights are BitNet 1.58 ternary (2-bit packed).
 * Conv1d, dt_bias, A_log, norms, vision encoder: FP32.
 *
 * Memory layout mirrors qwen35_to_eif.py output format (QWEN35 v1).
 */

#ifndef EIF_QWEN35_H
#define EIF_QWEN35_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// File format
// =============================================================================

#define QWEN35_MAGIC   0x51574E35u  /* 'QWN5' */
#define QWEN35_VERSION 1u

/** Layer type codes (1 byte per layer, 24 bytes total in binary) */
#define QWEN35_LAYER_DELTANET 0
#define QWEN35_LAYER_FULLATT  1

/** Binary file header (48 bytes) */
typedef struct {
    uint32_t magic;          /* QWEN35_MAGIC */
    uint32_t version;        /* QWEN35_VERSION */
    uint32_t config_offset;  /* offset to config block (== 48) */
    uint32_t weights_offset;
    uint32_t scales_offset;
    uint32_t norms_offset;
    uint32_t vision_offset;
    uint32_t weights_size;
    uint32_t scales_size;
    uint32_t norms_size;
    uint32_t vision_size;
    uint32_t reserved;
} qwen35_file_header_t;

/** Config block (128 bytes = 32 × int32) */
typedef struct {
    /* Language model */
    int32_t dim;             /* hidden_size = 1024 */
    int32_t hidden_dim;      /* intermediate_size = 3584 */
    int32_t n_layers;        /* total layers = 24 */
    int32_t n_heads;         /* full-attn heads = 8 */
    int32_t n_kv_heads;      /* GQA KV heads = 2 */
    int32_t head_dim;        /* full-attn head_dim = 256 */
    int32_t vocab_size;      /* 248320 */
    int32_t max_seq_len;     /* 2048 (compiled) */
    /* Layer type counts */
    int32_t n_delta;         /* DeltaNet layers = 18 */
    int32_t n_full;          /* FullAttn layers = 6 */
    /* DeltaNet dimensions */
    int32_t lin_n_k_heads;   /* 16 */
    int32_t lin_k_head_dim;  /* 128 */
    int32_t lin_n_v_heads;   /* 16 */
    int32_t lin_v_head_dim;  /* 128 */
    int32_t conv_k_size;     /* conv1d kernel = 4 */
    /* Vision encoder */
    int32_t vis_depth;       /* 12 */
    int32_t vis_hidden;      /* 768 */
    int32_t vis_heads;       /* 12 */
    int32_t vis_patch_size;  /* 16 */
    int32_t vis_merge_size;  /* 2 */
    int32_t vis_out_dim;     /* 1024 */
    int32_t vis_n_pos;       /* 2304 */
#define QWEN35_QTYPE_FP32         0
#define QWEN35_QTYPE_INT4         1
#define QWEN35_QTYPE_BITNET_2BIT  2
#define QWEN35_QTYPE_BITNET_DENSE 3

    /* Embedding / LM-head storage type (tied weights). INT8 is the legacy
     * default so v1 binaries (pad value 0) keep working unchanged. */
#define QWEN35_EMBED_INT8   0
#define QWEN35_EMBED_INT4   1
#define QWEN35_EMBED_2BIT   2   /* ternary {-1,0,+1}, 4 weights/byte */
#define QWEN35_EMBED_DENSE  3   /* ternary base-3, 5 weights/byte */
#define QWEN35_EMBED_FP32   4

    int32_t lm_head_tied;    /* 1 = tied with embed_tokens */
    int32_t quant_lm;        /* 0=FP32, 1=INT4, 2=BitNet 2-bit, 3=BitNet Dense 1.60b */
    int32_t quant_vision;    /* 0=FP32, 1=INT4, 2=BitNet 2-bit, 3=BitNet Dense 1.60b */
    int32_t quant_embed;     /* QWEN35_EMBED_* for embed_tokens / tied LM head */
    int32_t _pad[6];         /* zero-padded to 128 bytes */
} qwen35_config_t;

// =============================================================================
// Weight pointers (mapped directly into mmap'd or loaded binary)
// =============================================================================

/** Weights for a single GatedDeltaNet layer (BitNet 1.58 ternary) */
typedef struct {
    const uint8_t *qkv;         /* [conv_dim, dim] 2-bit packed */
    const uint8_t *z;           /* [val_dim, dim]  2-bit packed */
    const uint8_t *out;         /* [dim, val_dim]  2-bit packed */
    const uint8_t *proj_b;      /* [n_v_heads, dim] 2-bit packed */
    const uint8_t *proj_a;      /* [n_v_heads, dim] 2-bit packed */
    /* FP32 SSM params */
    const float   *conv;        /* [conv_dim, conv_k_size] */
    const float   *dt_bias;     /* [n_v_heads] */
    const float   *A_log;       /* [n_v_heads] */
    /* Runtime cache (not serialized): expf(A_log[h]) precomputed at load. */
    float          a_exp[64];
    /* Scales for ternary */
    const float   *qkv_s;       /* [conv_dim] */
    const float   *z_s;         /* [val_dim] */
    const float   *out_s;       /* [dim] */
    const float   *b_s;         /* [n_v_heads] */
    const float   *a_s;         /* [n_v_heads] */
    /* GatedRMSNorm weight */
    const float   *norm_w;      /* [lin_v_head_dim] */
} qwen35_delta_weights_t;

/** Weights for a single Full Attention layer (BitNet 1.58 ternary) */
typedef struct {
    const uint8_t *q;           /* [n_heads*head_dim, dim] 2-bit */
    const uint8_t *k;           /* [n_kv*head_dim, dim]  2-bit */
    const uint8_t *v;           /* [n_kv*head_dim, dim]  2-bit */
    const uint8_t *o;           /* [dim, n_heads*head_dim] 2-bit */
    /* Scales */
    const float   *q_s;         /* [n_heads*head_dim] */
    const float   *k_s;         /* [n_kv*head_dim] */
    const float   *v_s;         /* [n_kv*head_dim] */
    const float   *o_s;         /* [dim] */
    /* QK norms */
    const float   *q_norm;      /* [head_dim] */
    const float   *k_norm;      /* [head_dim] */
    /* SubLN */
    const float   *sub_norm;    /* [n_heads*head_dim] SubLN scale before o_proj */
} qwen35_attn_weights_t;

/** Weights for a single FFN layer (SwiGLU, BitNet 1.58) */
typedef struct {
    const uint8_t *gate;        /* [hidden_dim, dim] 2-bit */
    const uint8_t *up;          /* [hidden_dim, dim] 2-bit */
    const uint8_t *down;        /* [dim, hidden_dim] 2-bit */
    const float   *gate_s;      /* [hidden_dim] */
    const float   *up_s;        /* [hidden_dim] */
    const float   *down_s;      /* [dim] */
    /* SubLN */
    const float   *sub_norm;    /* [hidden_dim] SubLN scale before down_proj */
} qwen35_ffn_weights_t;

/** All model weights */
typedef struct {
    /* Token embeddings: INT8 [vocab_size, dim] */
    const int8_t  *embed;
    const float   *embed_s;     /* [vocab_size] per-row scales */

    /* Per-layer weights (indexed by layer index) */
    qwen35_delta_weights_t delta[18]; /* DeltaNet layers */
    qwen35_attn_weights_t  full[6];   /* Full attention layers */
    qwen35_ffn_weights_t   ffn[24];   /* FFN (all layers, index matches layer_types) */

    /* RMSNorm: input_layernorm for all 24 layers, then post_attention for all 24 */
    const float   *rms_att[24]; /* input_layernorm.weight [dim] each */
    const float   *rms_ffn[24]; /* post_attn_layernorm.weight [dim] each */
    const float   *rms_final;   /* model.norm.weight [dim] */
} qwen35_weights_t;

// =============================================================================
// Run state (allocated in RAM)
// =============================================================================

/** Per-DeltaNet-layer recurrent states (in RAM) */
typedef struct {
    float *S;           /* recurrent state [n_v_heads, head_k_dim, head_v_dim] */
    float *conv_state;  /* conv1d sliding window [conv_dim, conv_k_size] */
} qwen35_delta_state_t;

/** Per-FullAttn-layer KV cache (in RAM) */
typedef struct {
    float *k_cache;     /* [max_seq_len, n_kv_heads, head_dim] */
    float *v_cache;     /* [max_seq_len, n_kv_heads, head_dim] */
} qwen35_attn_state_t;

/** Complete model run state */
typedef struct {
    qwen35_delta_state_t delta[18];
    qwen35_attn_state_t  attn[6];
    int                  pos;       /* current position in sequence */

    /* Activation scratch buffers */
    float *x;           /* [dim] current hidden state */
    float *xb;          /* [dim] scratch */
    float *xb2;         /* [dim] scratch */
    float *q;           /* [max(n_heads*head_dim, conv_dim)] */
    float *k;           /* [n_kv_heads * head_dim] */
    float *v;           /* [val_dim] */
    float *z;           /* [val_dim] (DeltaNet gate) */
    float *hb;          /* [hidden_dim] FFN scratch */
    float *hb2;         /* [hidden_dim] FFN scratch */
    float *logits;      /* [vocab_size] */
    float *att;         /* [n_heads * max_seq_len] (full-attn scores) */

    /* Memory pool (all above are views into this) */
    void  *pool;
    size_t pool_size;
} qwen35_state_t;

// =============================================================================
// Model handle
// =============================================================================

typedef struct {
    qwen35_config_t  config;
    uint8_t          layer_types[24];   /* QWEN35_LAYER_DELTANET or _FULLATT */
    qwen35_weights_t weights;
    qwen35_state_t   state;

    /* Raw memory backing (mmap or malloc'd) */
    void   *file_data;
    size_t  file_size;
    bool    file_is_mmap;
} qwen35_t;

// =============================================================================
// API
// =============================================================================

/**
 * @brief Compute required RAM for run state.
 */
size_t qwen35_state_size(const qwen35_config_t *cfg);

/**
 * @brief Load model from EIF binary file (zero-copy mmap).
 *
 * @param model      Output handle.
 * @param path       Path to .eifm binary.
 * @param state_mem  RAM buffer for run state (size >= qwen35_state_size()).
 * @param state_sz   Size of state_mem.
 * @return 0 on success, <0 on error.
 */
int qwen35_load(qwen35_t *model, const char *path, void *state_mem, size_t state_sz);

/**
 * @brief Free resources (unmap file, free state if needed).
 */
void qwen35_free(qwen35_t *model);

/**
 * @brief Reset recurrent states and KV cache for new generation.
 */
void qwen35_reset(qwen35_t *model);

/**
 * @brief Run one forward pass for token `tok` at position `pos`.
 *
 * @param model     Loaded model.
 * @param tok       Input token id.
 * @param pos       Position in sequence (used for RoPE in full-attn layers).
 * @return Pointer to logits buffer [vocab_size]. Valid until next call.
 */
float *qwen35_forward(qwen35_t *model, int tok, int pos);
float *qwen35_forward_ex(qwen35_t *model, int tok, int pos, bool compute_logits);
float *qwen35_forward_no_logits(qwen35_t *model, int tok, int pos);

/**
 * @brief Get pointer to last computed normalized hidden state vector [dim].
 *
 * @param model Loaded model.
 * @return Pointer to hidden state buffer [dim].
 */
float *qwen35_get_hidden_state(qwen35_t *model);

/**
 * @brief Inject visual embeddings before text generation.
 *
 * Runs the DeltaNet/FullAttn layers over each visual embedding
 * without sampling, updating the recurrent state.
 *
 * @param model      Loaded model.
 * @param embeddings Visual token embeddings [n_tokens, dim].
 * @param n_tokens   Number of visual tokens.
 * @param start_pos  Position offset (usually 0).
 */
void qwen35_inject_visual(
    qwen35_t *model,
    const float *embeddings,
    int n_tokens,
    int start_pos
);

/**
 * @brief Greedy decode with repetition penalty.
 *
 * Generates up to max_new_tokens tokens, stopping at eos_token_id.
 *
 * @param model          Loaded model.
 * @param prompt_tokens  Input token ids.
 * @param n_prompt       Number of prompt tokens.
 * @param visual_emb     Optional visual embeddings (NULL if text-only).
 * @param n_visual       Number of visual tokens (0 if text-only).
 * @param out_tokens     Output token buffer (caller allocates).
 * @param max_new_tokens Max tokens to generate.
 * @param rep_penalty    Repetition penalty (1.0 = disabled, 1.1 typical).
 * @param eos_id         Token id that stops generation.
 * @return Number of tokens generated.
 */
int qwen35_generate(
    qwen35_t *model,
    const int *prompt_tokens,
    int n_prompt,
    const float *visual_emb,
    int n_visual,
    int *out_tokens,
    int max_new_tokens,
    float rep_penalty,
    int eos_id
);

#ifdef __cplusplus
}
#endif

#endif /* EIF_QWEN35_H */
