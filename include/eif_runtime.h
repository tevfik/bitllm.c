/**
 * @file eif_runtime.h
 * @brief EIF-Runtime: High-Performance, Zero-Heap C99 Inference Engine
 *
 * Unified C99 runtime supporting:
 * - Hybrid Gated DeltaNet + Full Attention (Qwen3.5 1.58b / 2b)
 * - Decoder-Only Transformers (Granite, SmolLM, LLaMA)
 * - Multiplication-Free BitNet b1.58 Ternary Weights {-1, 0, +1}
 * - Sub-10ms token generation latency on standard CPUs
 */

#ifndef EIF_RUNTIME_H
#define EIF_RUNTIME_H

#include "eif_status.h"
#include "eif_bpe_tokenizer.h"
#include "eif_quantize_bitnet.h"
#include "eif_gpu.h"
#include "eif_qwen35.h"
#include "eif_tinyllm.h"
#include "eif_gguf.h"
#include "eif_bert.h"
#include "eif_llm.h"

#define EIF_RUNTIME_VERSION_MAJOR 1
#define EIF_RUNTIME_VERSION_MINOR 0
#define EIF_RUNTIME_VERSION_PATCH 0
#define EIF_RUNTIME_VERSION_STRING "1.0.0"

#endif /* EIF_RUNTIME_H */
