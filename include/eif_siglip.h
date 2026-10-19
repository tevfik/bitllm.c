/**
 * @file eif_siglip.h
 * @brief Native Pure C99 SigLIP2 Vision Encoder & Connector for EIF Multimodal Runtime.
 *
 * Implements Google/IBM SigLIP2 Vision Transformer (ViT) and Modality Projection Connector:
 * - 512x512 tile ingestion with bicubic/bilinear interpolation
 * - 16x16 Conv2D patch projection to hidden dim (768)
 * - 1024 learned position embeddings
 * - 12 Transformer encoder layers (Multi-Head Self-Attention + MLP with tanh-GELU)
 * - Post LayerNorm
 * - 4x4 spatial patch merging (12,288 features) -> 576-dim linear connector projection
 * - Generates exact 64 visual tokens per tile (64 per tile; 5 to 17 tiles per image)
 *
 * Fully self-contained pure C99 with ZERO Python, PyTorch, or external dependencies.
 */

#ifndef EIF_SIGLIP_H
#define EIF_SIGLIP_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "eif_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

#define EIF_SIGLIP_MAGIC 0x56495332 /* 'VIS2' */

typedef struct {
    uint32_t magic;           /* 0x56495332 ('VIS2') */
    int32_t  version;         /* 1 */
    int32_t  patch_size;      /* 16 */
    int32_t  image_size;      /* 512 */
    int32_t  num_layers;      /* 12 */
    int32_t  hidden_size;     /* 768 */
    int32_t  num_heads;       /* 12 */
    int32_t  head_dim;        /* 64 */
    int32_t  intermediate_dim;/* 3072 */
    int32_t  num_patches;     /* 1024 */
    int32_t  conn_in_dim;     /* 12288 (16 * 768) */
    int32_t  conn_out_dim;    /* 576 (Language model dimension) */
    int32_t  qtype;           /* ViT linear weights: EIF_SIGLIP_Q_* */
    uint32_t total_payload;   /* Total bytes of vision weights */
    uint32_t reserved[2];     /* [0] INT4 group size (0 = per row), [1] 1 = Hadamard-rotated weights */
} eif_siglip_header_t;

/* Storage of the 72 ViT projection matrices (q, k, v, out, fc1, fc2 of each layer); everything else stays FP32.
 * Quantised matrices are dequantised into a scratch buffer right before each GEMM, so the FP32 SIMD / Vulkan GEMMs
 * are reused unchanged and RAM holds only the compressed weights. Layout of one quantised [N, K] matrix:
 *   INT8: int8 codes [N*K], then FP32 scales [N] (symmetric, absmax / 127 per row)
 *   INT4: packed codes [N*K/2] (two's-complement nibbles, low nibble = even column), then FP32 scales
 *         [N * K/G] (absmax / 7 per group of G input weights; G = reserved[0], 0 means one group per row)
 * With reserved[1] = 1 the stored matrix is W' = W·D·H (blockwise Walsh-Hadamard, eif_hadamard_block_size(K));
 * the sign vectors D for K = 768 and K = 3072 follow the header (FP32 +-1) and the loader un-rotates on dequant.
 * Matches sdk/eif_tools/quantize/vit_weight_quant.py. */
#define EIF_SIGLIP_Q_FP32 0
#define EIF_SIGLIP_Q_INT8 1
#define EIF_SIGLIP_Q_INT4 2

typedef struct {
    const uint8_t *codes;     /* NULL for FP32 matrices */
    const float *scales;
} eif_siglip_qmat_t;

typedef struct {
    /* LayerNorm 1 */
    const float *ln1_w;       /* [768] */
    const float *ln1_b;       /* [768] */
    /* Self-Attention Q, K, V, Out */
    const float *q_w;         /* [768, 768] */
    const float *q_b;         /* [768] */
    const float *k_w;         /* [768, 768] */
    const float *k_b;         /* [768] */
    const float *v_w;         /* [768, 768] */
    const float *v_b;         /* [768] */
    const float *out_w;       /* [768, 768] */
    const float *out_b;       /* [768] */
    /* LayerNorm 2 */
    const float *ln2_w;       /* [768] */
    const float *ln2_b;       /* [768] */
    /* MLP FC1 & FC2 */
    const float *fc1_w;       /* [3072, 768] */
    const float *fc1_b;       /* [3072] */
    const float *fc2_w;       /* [768, 3072] */
    const float *fc2_b;       /* [768] */
    /* Quantised storage (header.qtype != FP32): q, k, v, out, fc1, fc2 */
    eif_siglip_qmat_t qm[6];
} eif_siglip_layer_t;

typedef struct {
    eif_siglip_header_t header;

    /* Embeddings */
    const float *pe_w;        /* [768, 3, 16, 16] */
    const float *pe_b;        /* [768] */
    const float *pos_w;       /* [1024, 768] */

    /* 12 ViT Layers */
    eif_siglip_layer_t layers[12];

    /* Post LayerNorm */
    const float *post_ln_w;   /* [768] */
    const float *post_ln_b;   /* [768] */

    /* Connector */
    const float *conn_w;      /* [576, 12288] */

    /* Hadamard signs (header.reserved[1] = 1) */
    const float *had_signs_768;
    const float *had_signs_3072;

    bool is_initialized;
    uint8_t *allocated_weights; /* Non-NULL if loaded from file and owns memory */
    eif_gpu_context_t gpu;      /* Vulkan context owned by the model (eif_siglip_use_gpu with ctx = NULL) */
    void *gpu_vit;              /* GPU-resident weights and activations, NULL when encoding on the CPU */
} eif_siglip_t;

/**
 * @brief Initialize SigLIP2 model from raw memory buffer containing vision payload.
 *
 * @param model Target SigLIP model struct.
 * @param buffer Pointer to vision data (starting with eif_siglip_header_t).
 * @param buffer_size Size of buffer in bytes.
 * @return 0 on success, negative error code on failure.
 */
int eif_siglip_init_from_buffer(eif_siglip_t *model, const void *buffer, size_t buffer_size);

/**
 * @brief Initialize SigLIP2 model from an EIF binary file (offset at SVLM vision section).
 *
 * @param model Target SigLIP model struct.
 * @param filepath Path to .eifm file.
 * @param offset Byte offset of vision section in file.
 * @param size Byte size of vision section.
 * @return 0 on success, negative error code on failure.
 */
int eif_siglip_load_from_file(eif_siglip_t *model, const char *filepath, size_t offset, size_t size);

/** @brief sizeof(eif_siglip_t), for language bindings that allocate the struct. */
size_t eif_siglip_struct_size(void);

/**
 * @brief Dequantise one ViT projection (layer l, matrix m: 0 q, 1 k, 2 v, 3 out, 4 fc1, 5 fc2) into FP32 [N, K].
 *        For FP32 models this copies the stored weight. Exposed for verification.
 * @return 0 on success, negative on bad arguments.
 */
int eif_siglip_dequant_weight(const eif_siglip_t *model, int l, int m, float *out);

/**
 * @brief Run the encoder on a Vulkan GPU: uploads the ViT + connector weights once (INT8 weights stay INT8)
 *        and encodes batches of tiles with one submit each.
 *
 * @param ctx Initialised GPU context to share (must outlive the model), or NULL to create one owned by the model.
 * @return 0 if the GPU path is active; negative if unavailable (the model keeps encoding on the CPU).
 *         BITLLM_GPU_TILE_BATCH sets the tiles per submit (default 8; reduced to fit device memory limits).
 *         A GPU context is not thread-safe: do not encode concurrently with models sharing one context.
 */
int eif_siglip_use_gpu(eif_siglip_t *model, eif_gpu_context_t *ctx);

/** @brief true if eif_siglip_use_gpu succeeded for this model. */
bool eif_siglip_gpu_active(const eif_siglip_t *model);

/**
 * @brief Encode n_tiles tiles ([3, 512, 512] each, contiguous) into out_tokens ([n_tiles * 64, 576]).
 *        Uses the GPU in batches when active, the CPU otherwise.
 * @return 0 on success, negative error code on failure.
 */
int eif_siglip_encode_tiles(const eif_siglip_t *model, const float *tiles, int n_tiles, float *out_tokens);

/**
 * @brief Free any dynamically allocated weights in SigLIP model.
 */
void eif_siglip_free(eif_siglip_t *model);

/**
 * @brief Encode a single 512x512 tile [3, 512, 512] into 64 visual tokens [64, 576].
 *
 * @param model Initialized SigLIP model.
 * @param tile_pixels Normalized CHW pixel float array [3, 512, 512], range [-1.0, 1.0].
 * @param out_tokens Output buffer of size 64 * 576 * sizeof(float).
 * @return 0 on success, negative error code on failure.
 */
int eif_siglip_encode_tile(const eif_siglip_t *model, const float *tile_pixels, float *out_tokens);

/** Largest tile count the Idefics3 pipeline can produce: 4 x 4 grid + 1 global view. */
#define EIF_SIGLIP_MAX_TILES 17

/**
 * @brief Grid chosen by the Idefics3 image processor for an image of the given size.
 *
 * The longest edge is scaled to 2048 (small images are enlarged) and both sides are rounded up to
 * multiples of 512, so rows, cols are in 1..4 and the tile count is rows * cols + 1 (5..17).
 * Examples: 768x1024 -> 4 x 3, 449x92 -> 1 x 4, 1275x1650 -> 4 x 4.
 *
 * @return 0 on success, -1 for a non-positive size.
 */
int eif_siglip_plan_tiles(int img_w, int img_h, int *rows, int *cols);

/**
 * @brief Load an image and produce the normalised 512x512 tiles ((v/255 - 0.5) / 0.5, CHW, row-major
 *        tiles followed by the global view), identical in layout to transformers' Idefics3ImageProcessor.
 *
 * @param out_tiles Receives a malloc'd buffer of (rows * cols + 1) * 3 * 512 * 512 floats (caller frees).
 * @return 0 on success, negative error code on failure.
 */
int eif_siglip_preprocess_image(const char *image_path, float **out_tiles, int *rows, int *cols);

/**
 * @brief Encode an arbitrary image file (.png, .jpg, etc.) directly into visual tokens.
 *
 * Implements the full Idefics3/Docling document image pipeline:
 * - Reads image via stb_image
 * - Splits it into a rows x cols grid of tiles chosen from the aspect ratio (see eif_siglip_plan_tiles)
 *   plus one global overview tile
 * - Runs ViT and patch merge + connector on each tile (64 tokens per tile, dim 576)
 *
 * @param model Initialized SigLIP model.
 * @param image_path Path to image file.
 * @param out_tokens Output buffer of EIF_SIGLIP_MAX_TILES * 64 * 576 floats (about 2.5 MB); the first
 *        (rows * cols + 1) * 64 tokens are written, the global tile last.
 * @param out_rows Receives the grid rows (may be NULL).
 * @param out_cols Receives the grid columns (may be NULL).
 * @return 0 on success, negative error code on failure.
 */
int eif_siglip_encode_image(const eif_siglip_t *model, const char *image_path, float *out_tokens,
                            int *out_rows, int *out_cols);

#ifdef __cplusplus
}
#endif

#endif /* EIF_SIGLIP_H */
