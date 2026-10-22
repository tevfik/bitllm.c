# bitllm.c ⚡

A high-performance, pure C99 inference engine for **1.58-bit BitNet** ternary LLMs, **Gated DeltaNet** hybrid models, and **Multimodal VLMs** on the edge.

Designed from the ground up for seamless embedding into applications (such as [gleann](https://github.com/tevfik/gleann)), standalone CLI execution, and resource-constrained edge computing.

---

## 🌟 Key Features

- **Pure C99 & Zero External Dependencies**: Runs anywhere standard C99 compiles (Linux, macOS, Windows, Android, Raspberry Pi, ARM servers).
- **Multiplication-Free 1.58-bit Ternary BitNet**: Native support for ternary weights `{-1, 0, +1}`. Integer additions/subtractions replace heavy floating-point matrix multiplications.
- **Grouped Scaling & Walsh-Hadamard Transform**: Blockwise Walsh-Hadamard outlier rotation and fine-grained group scales (g128/g96) preserving model quality at extreme quantization.
- **Hybrid Gated DeltaNet + Full Attention**: First-class support for linear attention architectures like Qwen3.5 (0.8B / 2B) with $O(1)$ memory consumption in linear layers.
- **Native Vision Language Model (VLM) Support**: Integrated SigLIP2 vision encoder and Docling OCR/DocVQA vision projection.
- **Embedded Byte-Level BPE Tokenizer**: 100% byte-exact parity with HuggingFace tokenizers via C99 pre-tokenization regex and rank-based BPE merges.
- **Flexible Memory Model**: Designed for embedding without rigid MCU limitations; supports system allocators or custom user-provided memory pools.
- **Dual Format Support**: Reads `.eifm` (Edge Intelligence Format Model) and `.gguf` weights directly. Qwen2/2.5 GGUFs run through the dense INT8 path; llama.cpp style Qwen3.5 text GGUFs (F16/Q8_0/Q4_K/Q5_K/Q6_K, 24 layers, tied head) are repacked in memory to INT4 row-scaled linears on load (`Q8_0` keeps the best quality; `Q4_K_M` is re-quantised twice and degrades on long answers). Vision, untied heads and other layer counts are not supported.
- **Optional Vulkan GPU Compute**: the SigLIP2 vision encoder runs on any Vulkan GPU (`--gpu`); the loader is opened at runtime, so builds need no Vulkan SDK and machines without a GPU fall back to the AVX2/NEON/OpenMP CPU path.

---

---

## ✅ Supported models and how they are verified

| Interface | Models (GGUF unless noted) | Reference / status |
|---|---|---|
| Text generation | Qwen2/2.5 (INT8 dense), Qwen3.5 0.8B/2B text (INT4 repack, 24 layers, tied head), LLaMA family (SmolLM2, Granite, MiniCPM; interleaved RoPE), BitNet, `.eifm` | greedy "Paris" check, repeatability, long prompt (`test_gen_matrix`) |
| Embeddings | BERT/MiniLM/BGE/E5/Arctic (WordPiece), granite-embedding-97m-multilingual (modern-bert: pre-norm, GLU FFN, local/global attention, Granite BPE), nomic-embed-text-v1.5 (nomic-bert: fused QKV, RoPE, SwiGLU; add the `search_document: ` task prefix yourself), bge-m3 (Unigram), jina-v2-code (BPE), EmbeddingGemma 300m / 2 (SPM merge / Gemma4 BPE) | cosine vs llama.cpp on short, code and ~1400-token inputs (`test_embed_matrix`) |
| Reranking | bge-reranker-v2-m3 (`--rerank Q --doc D ...`, `eif_llm_rerank`; ~0.26 s per 300-token pair on 20 Arm cores) | logit within 0.1 of HF fp32, same token ids (`test_rerank_matrix`) |
| Vision + text | Granite-Docling (SigLIP2), Qwen3.5 `.eifm` | `test_docling_native`, `test_siglip_*` |
| Audio | not supported | – |

Refused at load with a clear error (no crash, no silent LLaMA fallback): unknown `general.architecture` values and layouts
the loader cannot map. Known gaps against llama.cpp are listed in `KNOWN_GAPS` in `tests/test_embed_matrix.c` (currently none).
The reference texts include Turkish and Japanese/accented input, so WordPiece/BPE/Unigram handling of non-ASCII text is checked too.

Model-dependent tests are skipped unless pointed at the files:

```bash
BITLLM_MODELS=/path/to/embedding_and_reranker_ggufs BITLLM_GEN_MODELS=/path/to/chat_ggufs ctest --test-dir build
# references: tests/data/embed_ref/*.ref (llama.cpp), tests/data/rerank_ref/*.tsv (HF); regenerate with tests/tools/
```

## 🏗️ Architecture & Embedding

`bitllm.c` builds both a standalone CLI (`bitllm-run`) and static/shared libraries (`libbitllm.a` / `libbitllm.so`) intended for direct integration into C/C++, Rust, Go, or Python hosts.

```
bitllm.c/
├── include/
│   ├── bitllm.h              # Unified master header for hosts (Gleann, etc.)
│   ├── eif_qwen35.h          # Qwen3.5 hybrid DeltaNet + Attention engine
│   ├── eif_tinyllm.h         # Decoder-only LLM & BitNet 1.58b engine
│   ├── eif_siglip.h          # SigLIP2 vision encoder
│   ├── eif_bpe_tokenizer.h   # Byte-Level BPE Tokenizer
│   ├── eif_quantize_bitnet.h # Walsh-Hadamard transform & BitNet quantization
│   └── eif_gpu.h             # Optional Vulkan compute abstraction
├── src/                      # C99 kernel implementations
│   ├── eif_bert.c            # encoder: GGUF/EIFM loader, forward pass, pooling, reranker head
│   ├── eif_bert_kernels.c    # LayerNorm, GELU, RoPE, batched GEMM (F32 / INT8 / Q8_0; AVX2, NEON)
│   ├── eif_bert_gemma.c      # Gemma-embedding encoders (300m, 2)
│   ├── eif_bert_tokenizer.c  # WordPiece, SPM merge, Unigram, Gemma4 BPE, pair encoding
│   └── eif_bert_internal.h   # shared by the four files above (not public)
├── cli/
│   └── main.c                # Interactive / one-shot CLI runner (bitllm-run)
└── tests/                    # Feature verification & benchmark suites
```

---

## 🚀 Quick Start

### 1. Build

```bash
git clone https://github.com/tevfik/bitllm.c.git
cd bitllm.c

# Configure and compile with optimizations (AVX2/FMA/OpenMP auto-detected)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### 2. Run an LLM

```bash
./build/bitllm-run \
  --model path/to/model.eifm \
  --tokenizer path/to/tokenizer.bin \
  --prompt "What is edge intelligence?" \
  --max-tokens 128 \
  --threads 4
```

### 3. Run Multimodal Inference (Vision + Language)

```bash
./build/bitllm-run \
  --model path/to/docling_vlm.eifm \
  --tokenizer path/to/tokenizer.bin \
  --image path/to/document.png \
  --prompt "Extract the total amount from this invoice."
```

**Granite-Docling (document OCR / tables):** the ready-made ternary model is published at
[`tevfikk/granite-docling-258M-bitnet`](https://huggingface.co/tevfikk/granite-docling-258M-bitnet)
(`granite_docling_bitnet_dense.eifm` + `tokenizer.bin`). Its vision tower is stored as **INT8 per-row weights**
(`SVLM` config, SigLIP header `qtype = 1`); INT4 per group, optionally with a Walsh-Hadamard rotation, is supported too
(`qtype = 2`, see `include/eif_siglip.h`). On the CPU each ViT matrix is dequantised into a scratch buffer right before its GEMM.
Runtimes older than this support reading the model silently produce no output ("Generated Tokens: -3").
This project was previously published as `eif-runtime`.

### 4. GPU vision encoder (Vulkan)

```bash
./build/bitllm-run path/to/granite_docling_bitnet_dense.eifm --image page.png --gpu
```

With `--gpu` (or, from C, an initialised `llm.gpu` / `eif_siglip_use_gpu()`), the whole ViT + connector runs on the
GPU: weights are uploaded once (INT8 ViT weights stay INT8 and are dequantised inside the GEMM kernel), and a batch of
tiles goes through patch embedding, the 12 layers (layernorm, fused attention, GELU MLP, residuals) and the connector in
one command-buffer submit. The language model still runs on the CPU. GPU and CPU visual tokens agree to cosine
1.0000000 (max abs difference ~1e-3 on values up to ~200).

- Build: `BITLLM_ENABLE_VULKAN` defaults to ON on Linux/Windows/macOS/Android and only needs `dl`; there is no link
  against libvulkan. Embedded/bare-metal targets (`CMAKE_SYSTEM_NAME` Generic) keep it OFF.
- Runtime: needs a Vulkan loader (`libvulkan.so.1`, `vulkan-1.dll`) and a GPU driver; otherwise `eif_gpu_init()` returns
  false and everything runs on the CPU unchanged.
- Environment: `BITLLM_NO_GPU=1` disables the backend, `BITLLM_GPU_TILE_BATCH=n` sets tiles per submit (default 8,
  ~35 MB of GPU memory per tile plus the weights), `BITLLM_GPU_FLASH=0` forces the unfused attention path,
  `BITLLM_GPU_ALLOW_CPU=1` accepts software devices such as llvmpipe (testing), `BITLLM_VULKAN_LIB` overrides the loader.
- Shaders: sources in `shaders/*.comp`; `scripts/gen_spv.sh [glslangValidator]` regenerates the committed
  `src/eif_vk_shaders.h`, so normal builds need no shader compiler.

---

## 🧩 Embedding into Gleann / C Applications

To embed `bitllm.c` into [gleann](https://github.com/tevfik/gleann) or any C/C++ project:

### Using CMake `FetchContent` or Git Submodule

```cmake
add_subdirectory(path/to/bitllm.c)
target_link_libraries(your_app PRIVATE bitllm_static)
```

### C API Example

```c
#include <bitllm.h>
#include <stdio.h>

int main(void) {
    // 1. Initialize tokenizer
    eif_bpe_tokenizer_t tok;
    if (eif_bpe_tokenizer_init(&tok, "tokenizer.bin") != EIF_STATUS_OK) {
        fprintf(stderr, "Failed to load tokenizer\n");
        return 1;
    }

    // 2. Load model (e.g., Qwen3.5 0.8B BitNet or TinyLLM)
    eif_tinyllm_model_t model;
    if (eif_tinyllm_load(&model, "model.eifm") != EIF_STATUS_OK) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // 3. Encode prompt & generate
    const char *prompt = "Explain quantum computing in one sentence:";
    int32_t prompt_tokens[256];
    int n_tokens = eif_bpe_tokenize(&tok, prompt, prompt_tokens, 256);

    printf("Generating: ");
    eif_tinyllm_generate(&model, prompt_tokens, n_tokens, 64, stdout);
    printf("\n");

    // 4. Cleanup
    eif_tinyllm_free(&model);
    eif_bpe_tokenizer_free(&tok);
    return 0;
}
```

---

## 🧪 Testing & Validation

Run the comprehensive test suite:

```bash
ctest --test-dir build --output-on-failure
```

The test suite covers:
- **`test_bitnet_hadamard`**: Walsh-Hadamard input rotation, group scale precision, and numeric parity.
- **`test_siglip_native`**: Patch embedding, self-attention, and pooling for vision encoders.
- **`test_all_features`**: Tokenizer parity, hybrid DeltaNet states, and KV-cache validation.

---

## 📄 License

Licensed under the [Apache License, Version 2.0](LICENSE).
