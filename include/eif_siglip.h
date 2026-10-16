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
 * - Generates exact 64 visual tokens per tile (832 visual tokens for 13 tiles)
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
    int32_t  qtype;           /* 0 = FP32 */
    uint32_t total_payload;   /* Total bytes of vision weights */
    uint32_t reserved[2];
} eif_siglip_header_t;

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

    bool is_initialized;
    uint8_t *allocated_weights; /* Non-NULL if loaded from file and owns memory */
    eif_gpu_context_t gpu;      /* Optional Vulkan GPU backend */
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

/**
 * @brief Encode an arbitrary image file (.png, .jpg, etc.) directly into 832 visual tokens [832, 576].
 *
 * Implements the full Idefics3/Docling document image pipeline:
 * - Reads image via stb_image
 * - Generates 12 local tiles (4 rows x 3 columns grid) + 1 global overview tile = 13 tiles
 * - Runs ViT and patch merge + connector on each tile
 * - Output shape: 832 tokens x 576 dimensions
 *
 * @param model Initialized SigLIP model.
 * @param image_path Path to image file.
 * @param out_tokens Output buffer of size 832 * 576 * sizeof(float) (approx 1.83 MB).
 * @return 0 on success, negative error code on failure.
 */
int eif_siglip_encode_image(const eif_siglip_t *model, const char *image_path, float *out_tokens);

#ifdef __cplusplus
}
#endif

#endif /* EIF_SIGLIP_H */
