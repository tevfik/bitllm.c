/**
 * @file eif_gguf.h
 * @brief Unified C99 GGUF (v2/v3) Binary Container Parser for EIF Engine
 *
 * Provides fast, zero-copy parsing of GGUF models for both:
 * 1. Encoder models: BERT, MiniLM, BGE embeddings
 * 2. Decoder/Causal models: Qwen2/2.5, LLaMA, Granite, BitNet b1.58
 */

#ifndef EIF_GGUF_H
#define EIF_GGUF_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef GGUF_MAGIC
#define GGUF_MAGIC 0x46554747u /* 'GGUF' in Little Endian */
#endif

/** Standard GGML Tensor Quantization Types */
typedef enum {
    GGUF_TYPE_F32     = 0,
    GGUF_TYPE_F16     = 1,
    GGUF_TYPE_Q4_0    = 2,
    GGUF_TYPE_Q4_1    = 3,
    GGUF_TYPE_Q5_0    = 6,
    GGUF_TYPE_Q5_1    = 7,
    GGUF_TYPE_Q8_0    = 8,
    GGUF_TYPE_Q8_1    = 9,
    GGUF_TYPE_Q2_K    = 10,
    GGUF_TYPE_Q3_K    = 11,
    GGUF_TYPE_Q4_K    = 12,
    GGUF_TYPE_Q5_K    = 13,
    GGUF_TYPE_Q6_K    = 14,
    GGUF_TYPE_IQ2_XXS = 16,
    GGUF_TYPE_IQ2_XS  = 17,
    GGUF_TYPE_IQ1_S   = 19,
    GGUF_TYPE_TL1     = 34, /* BitNet b1.58 Ternary {-1, 0, +1} in bitnet.cpp / ggml */
    GGUF_TYPE_TL2     = 35,
} gguf_tensor_type_t;

typedef struct {
    char name[128];
    uint32_t type;
    uint64_t offset;
    uint32_t n_dims;
    uint64_t ne[4];
} gguf_tensor_desc_t;

static inline float fp16_to_fp32(uint16_t h) {
#if (defined(__aarch64__) || defined(_M_ARM64)) && (defined(__ARM_FP) || defined(__ARM_NEON))
    __fp16 f16;
    memcpy(&f16, &h, 2);
    return (float)f16;
#else
    uint32_t sign = ((uint32_t)(h & 0x8000)) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    if (exp == 0) {
        if (mant == 0) return (sign) ? -0.0f : 0.0f;
        while (!(mant & 0x0400)) { mant <<= 1; exp--; }
        exp++;
        mant &= ~0x0400;
    } else if (exp == 31) {
        return (mant == 0) ? ((sign) ? -1.0f/0.0f : 1.0f/0.0f) : 0.0f/0.0f;
    }
    exp = exp + (127 - 15);
    uint32_t f32_u = sign | (exp << 23) | (mant << 13);
    float out;
    memcpy(&out, &f32_u, 4);
    return out;
#endif
}

static inline uint32_t gguf_read_u32(const uint8_t **p) {
    uint32_t v;
    memcpy(&v, *p, 4);
    *p += 4;
    return v;
}

static inline uint64_t gguf_read_u64(const uint8_t **p) {
    uint64_t v;
    memcpy(&v, *p, 8);
    *p += 8;
    return v;
}

static inline void gguf_read_str(const uint8_t **p, char *out, size_t max_out) {
    uint64_t len = gguf_read_u64(p);
    size_t copy_len = (len < max_out - 1) ? len : (max_out - 1);
    memcpy(out, *p, copy_len);
    out[copy_len] = '\0';
    *p += len;
}

static inline const void *find_gguf_tensor(
    const gguf_tensor_desc_t *tensors,
    int n_tensors,
    const uint8_t *data_start,
    const char *name,
    uint32_t *out_type)
{
    for (int i = 0; i < n_tensors; i++) {
        if (strcmp(tensors[i].name, name) == 0) {
            if (out_type) *out_type = tensors[i].type;
            return data_start + tensors[i].offset;
        }
    }
    return NULL;
}

#ifdef __cplusplus
}
#endif

#endif /* EIF_GGUF_H */
