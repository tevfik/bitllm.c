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
- **Dual Format Support**: Reads `.eifm` (Edge Intelligence Format Model) and `.gguf` weights directly.
- **Optional Vulkan GPU Compute**: Hardware acceleration via Vulkan compute shaders when available, gracefully falling back to AVX2/NEON/OpenMP CPU execution.

---

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
(`qtype = 2`, see `include/eif_siglip.h`). Each ViT matrix is dequantised into a scratch buffer right before its GEMM.
Runtimes older than this support reading the model silently produce no output ("Generated Tokens: -3").
This project was previously published as `eif-runtime`.

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
