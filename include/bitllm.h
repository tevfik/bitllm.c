/**
 * @file bitllm.h
 * @brief bitllm.c: High-Performance, Pure C99 Inference Engine for 1.58-bit LLMs & VLMs
 *
 * Designed for embedded integration into Gleann and standalone edge deployment.
 * Supports:
 * - 1.58-bit Ternary BitNet {-1, 0, +1} (grouped scales + Walsh-Hadamard outlier rotation)
 * - Hybrid Gated DeltaNet Linear Attention + Sliding-Window / Full Attention (Qwen3.5 0.8B/2B)
 * - Multimodal Vision Encoders (SigLIP2, Docling OCR/DocVQA)
 * - Pure C99 Byte-Level BPE Tokenizer with regex pre-tokenization
 * - GGUF & EIFM model format loading
 * - Optional Vulkan GPU compute acceleration
 */

#ifndef BITLLM_H
#define BITLLM_H

#ifdef __cplusplus
extern "C" {
#endif

#include "eif_status.h"
#include "eif_bpe_tokenizer.h"
#include "eif_quantize_bitnet.h"
#include "eif_gpu.h"
#include "eif_qwen35.h"
#include "eif_tinyllm.h"
#include "eif_siglip.h"
#include "eif_gguf.h"
#include "eif_bert.h"
#include "eif_llm.h"

#define BITLLM_VERSION_MAJOR 1
#define BITLLM_VERSION_MINOR 0
#define BITLLM_VERSION_PATCH 0
#define BITLLM_VERSION_STRING "1.0.0"

/* Backward compatibility aliases */
#define EIF_RUNTIME_VERSION_MAJOR  BITLLM_VERSION_MAJOR
#define EIF_RUNTIME_VERSION_MINOR  BITLLM_VERSION_MINOR
#define EIF_RUNTIME_VERSION_PATCH  BITLLM_VERSION_PATCH
#define EIF_RUNTIME_VERSION_STRING BITLLM_VERSION_STRING

/* BitLLM Type Aliases for clean integration */
typedef eif_status_t bitllm_status_t;
typedef eif_bpe_tokenizer_t bitllm_tokenizer_t;
typedef qwen35_t bitllm_qwen35_t;
typedef tinyllm_t bitllm_tinyllm_t;
typedef eif_siglip_t bitllm_siglip_t;
typedef eif_bert_t bitllm_bert_t;

#ifdef __cplusplus
}
#endif

#endif /* BITLLM_H */
