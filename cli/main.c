#define _GNU_SOURCE
/**
 * @file main.c
 * @brief bitllm.c standalone CLI runner (bitllm-run; formerly EIF-Runtime eif-run)
 *
 * Fast on-device inference for BitNet b1.58 ternary language models.
 * Supports:
 * - Hybrid Gated DeltaNet (Qwen3.5)
 * - Decoder-Only Transformers (Docling, SmolLM, LLaMA)
 */

#define _POSIX_C_SOURCE 199309L
#define _DEFAULT_SOURCE

#include "bitllm.h"
#include "eif_bert.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "stb_image.h"

static double get_time_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void print_usage(const char *prog)
{
    printf("bitllm.c Standalone CLI Runner (v%s)\n", BITLLM_VERSION_STRING);
    printf("High-Performance 1.58-bit BitNet & DeltaNet Inference Engine\n\n");
    printf("Usage:\n");
    printf("  %s <model_path> [prompt] [max_tokens] [tokenizer.bin] [options]\n\n", prog);
    printf("Options:\n");
    printf("  -m, --model <path>       Model file (same as the first positional argument)\n");
    printf("  -i, --image <path>       Document image for vision models (Granite-Docling)\n");
    printf("  -p, --prompt <str>       Input prompt text for generative LLMs\n");
    printf("  -e, --embed <str>        Compute dense embeddings for text (BERT / Granite / E5)\n");
    printf("  -n, --max-tokens <int>   Maximum new tokens to generate (default: 64)\n");
    printf("  -t, --tokenizer <path>   Path to custom tokenizer.bin\n");
    printf("  -T, --temp <float>       Sampling temperature (default: 0.7, 0.0=argmax; 0.0 with --image)\n");
    printf("  --top-p <float>          Nucleus top-p threshold (default: 0.9)\n");
    printf("  -r, --rep-penalty <val>  Repetition penalty factor (default: 1.0 = none)\n");
    printf("  --no-loop-guard          Do not stop DocTags generation that repeats one pattern (--image)\n");
    printf("  --raw                    Do not wrap the prompt in the ChatML template (Qwen2 instruct models)\n");
    printf("  --gpu                    Run the vision encoder on a Vulkan GPU (graceful CPU fallback)\n");
    printf("  --kv-int8                Enable INT8 quantized KV cache (75%% memory reduction)\n\n");
    printf("Examples:\n");
    printf("  %s models/dist/granite-30m-q8_0.gguf --embed \"Artificial intelligence and edge computing\"\n", prog);
    printf("  %s models/dist/minilm.gguf -e \"Semantic similarity search\"\n", prog);
    printf("  %s artifacts/minicpm5/minicpm5_1b_bitnet.eifm \"Explain edge AI\"\n", prog);
    printf("  %s artifacts/granite_docling/granite_docling_bitnet_dense.eifm --image page.png\n\n", prog);
}

#if defined(_OPENMP) && defined(__linux__)
/* On big.LITTLE CPUs (e.g. Cortex-X925 + A725) statically scheduled GEMVs wait for the slowest core:
 * default to the fastest cores only. Returns their count (0 = homogeneous or unknown). */
static int pin_to_fastest_cores(void)
{
    enum { MAX_CPUS = 1024 };
    static long cap[MAX_CPUS];
    long best = 0;
    int n = 0;
    for (int c = 0; c < MAX_CPUS; c++) {
        char path[96];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpu_capacity", c);
        FILE *f = fopen(path, "r");
        if (!f) {
            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", c);
            f = fopen(path, "r");
        }
        cap[c] = -1;
        if (f) {
            if (fscanf(f, "%ld", &cap[c]) != 1) cap[c] = -1;
            fclose(f);
        }
        if (cap[c] > best) best = cap[c];
        if (cap[c] >= 0) n = c + 1;
    }
    char places[4096] = "";
    int n_fast = 0, n_slow = 0;
    size_t len = 0;
    for (int c = 0; c < n; c++) {
        if (cap[c] < 0) continue;
        if (cap[c] * 10 < best * 9) { n_slow++; continue; } /* same core type can differ by a few % */
        int w = snprintf(places + len, sizeof(places) - len, "%s{%d}", n_fast ? "," : "", c);
        if (w < 0 || (size_t)w >= sizeof(places) - len) return 0;
        len += (size_t)w;
        n_fast++;
    }
    if (n_fast == 0 || n_slow == 0) return 0;
    setenv("OMP_PLACES", places, 0);
    return n_fast;
}
#endif

int main(int argc, char **argv)
{
    if (argc < 2 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        print_usage(argv[0]);
        return (argc < 2) ? 1 : 0;
    }

#ifdef _OPENMP
    /* libgomp reads OMP_* when the library is initialised, before main(): set the defaults, then re-exec once
     * so they take effect (spinning workers matter for the many small decode GEMVs). */
    if (!getenv("BITLLM_OMP_ENV")) {
        setenv("BITLLM_OMP_ENV", "1", 1);
        setenv("OMP_WAIT_POLICY", "ACTIVE", 0);
        setenv("OMP_PROC_BIND", "close", 0);
        setenv("GOMP_SPINCOUNT", "100000", 0);
        int n_fast = 0;
#if defined(__linux__)
        if (!getenv("OMP_PLACES") && !getenv("GOMP_CPU_AFFINITY")) n_fast = pin_to_fastest_cores();
#endif
        if (!getenv("OMP_NUM_THREADS")) {
            int procs = omp_get_num_procs();
            char nt[16];
            snprintf(nt, sizeof(nt), "%d", n_fast > 0 ? n_fast : (procs >= 8 ? 8 : procs));
            setenv("OMP_NUM_THREADS", nt, 1);
        }
#if defined(__linux__)
        execv("/proc/self/exe", argv); /* only returns on failure: continue with the current settings */
#endif
        omp_set_num_threads(atoi(getenv("OMP_NUM_THREADS")));
    }
#endif

    const char *model_path = NULL;
    const char *prompt = "Hello world!";
    const char *embed_text = NULL;
    const char *image_path = NULL;
    int max_tokens = 64;
    const char *tokenizer_path = NULL;
    float temperature = 0.7f;
    float top_p = 0.9f;
    float rep_penalty = 1.0f;
    bool temp_set = false, rep_set = false; /* explicit -T / -r win over the image-mode defaults */
    bool use_gpu = false;
    bool raw_prompt = false;
    int loop_guard = 0;
    int kv_type = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) {
            if (i + 1 < argc) prompt = argv[++i];
        } else if (strcmp(argv[i], "-e") == 0 || strcmp(argv[i], "--embed") == 0) {
            if (i + 1 < argc) embed_text = argv[++i];
        } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (i + 1 < argc) model_path = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--image") == 0) {
            if (i + 1 < argc) image_path = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--max-tokens") == 0 || strcmp(argv[i], "n") == 0) {
            if (i + 1 < argc) max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--tokenizer") == 0) {
            if (i + 1 < argc) tokenizer_path = argv[++i];
        } else if (strcmp(argv[i], "-T") == 0 || strcmp(argv[i], "--temp") == 0 || strcmp(argv[i], "--temperature") == 0) {
            if (i + 1 < argc) { temperature = (float)atof(argv[++i]); temp_set = true; }
        } else if (strcmp(argv[i], "--top-p") == 0) {
            if (i + 1 < argc) top_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--rep-penalty") == 0 || strcmp(argv[i], "--repetition-penalty") == 0) {
            if (i + 1 < argc) { rep_penalty = (float)atof(argv[++i]); rep_set = true; }
        } else if (strcmp(argv[i], "--no-loop-guard") == 0) {
            loop_guard = -1;
        } else if (strcmp(argv[i], "--raw") == 0) {
            raw_prompt = true;
        } else if (strcmp(argv[i], "--gpu") == 0) {
            use_gpu = true;
        } else if (strcmp(argv[i], "--kv-int8") == 0) {
            kv_type = 1;
        } else if (argv[i][0] != '-') {
            if (!model_path) {
                model_path = argv[i];
            } else if (!prompt || strcmp(prompt, "Hello world!") == 0) {
                prompt = argv[i];
            } else if (max_tokens == 64) {
                max_tokens = atoi(argv[i]);
            } else if (!tokenizer_path) {
                tokenizer_path = argv[i];
            }
        }
    }

    if (!model_path) {
        print_usage(argv[0]);
        return 1;
    }

    /* Document images (DocTags / OTSL) need deterministic decoding: sampling noise and the repetition
     * penalty (which punishes the many repeated <loc_N> / cell tags) corrupt the structured output. */
    if (image_path != NULL) {
        if (!temp_set) temperature = 0.0f;
        if (!rep_set) rep_penalty = 1.0f;
    }

    /* Dedicated Embedding Inference Mode */
    if (embed_text != NULL) {
        printf("=================================================================\n");
        printf("  bitllm.c (formerly EIF-Runtime) v%s\n", EIF_RUNTIME_VERSION_STRING);
        printf("=================================================================\n");
        printf("  • Model File  : %s\n", model_path);
        printf("  • Mode        : Dense Semantic Vector Embedding\n");

        eif_bert_t bert;
        memset(&bert, 0, sizeof(bert));
        if (eif_bert_load(&bert, model_path) != 0) {
            fprintf(stderr, "\n[Error] Failed to load embedding model '%s'\n", model_path);
            return 1;
        }

        const char *tok_desc = "WordPiece (MiniLM / BERT)";
        if (bert.vocab.tok_type == EBERT_TOKENIZER_BPE) tok_desc = "Byte-Level BPE (RoBERTa / Granite)";
        else if (bert.vocab.tok_type == EBERT_TOKENIZER_SPM) tok_desc = "SentencePiece (XLM-RoBERTa / E5)";

        printf("  • Architecture: %s\n", tok_desc);
        printf("  • Parameters  : Dim=%d, Intermediate=%d, Layers=%d, Heads=%d, Vocab=%d\n",
               bert.config.dim, bert.config.intermediate_dim, bert.config.n_layers,
               bert.config.n_heads, bert.vocab.vocab_size);
        printf("  • Pooling     : %s\n", (bert.config.pooling_type == 2) ? "CLS (Token 0)" : "Mean (All Tokens)");

        float *emb = (float *)malloc((size_t)bert.config.dim * sizeof(float));
        if (!emb) {
            eif_bert_free(&bert);
            return 1;
        }

        double t0 = get_time_sec();
        int rc = eif_bert_embed(&bert, embed_text, emb);
        double dt_ms = (get_time_sec() - t0) * 1000.0;
        if (rc != 0) {
            fprintf(stderr, "\n[Error] Embedding inference failed with code %d\n", rc);
            free(emb);
            eif_bert_free(&bert);
            return 1;
        }

        printf("\nInput Text: \"%s\"\n", embed_text);
        printf("Embedding Vector [%d dims] (Latency: %.2f ms | Throughput: %.1f emb/sec):\n[",
               bert.config.dim, dt_ms, 1000.0 / (dt_ms > 1e-4 ? dt_ms : 1e-4));
        int print_dims = bert.config.dim > 10 ? 10 : bert.config.dim;
        for (int d = 0; d < print_dims; d++) {
            printf("%.6f%s", emb[d], (d + 1 < bert.config.dim) ? ", " : "");
        }
        if (bert.config.dim > 10) printf("... (+%d more dimensions)", bert.config.dim - 10);
        printf("]\n");

        free(emb);
        eif_bert_free(&bert);
        return 0;
    }

    /* Native C99 Image Ingestion */
    int img_w = 0, img_h = 0, img_c = 0;
    uint8_t *img_pixels = NULL;
    if (image_path != NULL) {
        img_pixels = stbi_load(image_path, &img_w, &img_h, &img_c, 3);
        if (!img_pixels) {
            fprintf(stderr, "\n[Error] Failed to load image '%s': %s\n", image_path, stbi_failure_reason());
            return 1;
        }
        if (!prompt || strcmp(prompt, "Hello world!") == 0) {
            prompt = "Convert this page to docling.";
        }
    }

    printf("=================================================================\n");
    printf("  bitllm.c (formerly EIF-Runtime) v%s\n", EIF_RUNTIME_VERSION_STRING);
    printf("=================================================================\n");
    printf("  • Model File  : %s\n", model_path);
    if (image_path != NULL) {
        printf("  • Modality    : Multimodal VLM (Vision-Language)\n");
        printf("  • Input Image : %s (%dx%d RGB)\n", image_path, img_w, img_h);
    }

    /* 1. Load model via polymorphic facade */
    eif_llm_t llm;
    int rc = eif_llm_load(&llm, model_path, tokenizer_path, NULL, 0);
    if (rc != 0) {
        if (img_pixels) stbi_image_free(img_pixels);
        fprintf(stderr, "\n[Error] Failed to load model '%s' (code: %d)\n", model_path, rc);
        return 1;
    }

    const char *arch_name = "Unknown";
    if (llm.arch == EIF_LLM_ARCH_SMOLLM2) {
        arch_name = "Decoder-Only Transformer (Granite / SmolLM / LLaMA)";
    } else if (llm.arch == EIF_LLM_ARCH_QWEN2) {
        arch_name = "Qwen2 / Qwen2.5 Causal Transformer";
    } else if (llm.arch == EIF_LLM_ARCH_QWEN35) {
        arch_name = "Hybrid Gated DeltaNet + Full Attention (Qwen3.5)";
    }

    /* Initialize GPU if requested */
    if (use_gpu) {
        eif_gpu_init(&llm.gpu);
        if (eif_gpu_is_available(&llm.gpu)) {
            printf("  • Compute Unit: GPU [%s] (%s) for the vision encoder, CPU for the LM\n", llm.gpu.device_name,
                   llm.gpu.backend_name);
        } else {
            printf("  • Compute Unit: CPU Native [T-MAC / NEON] (GPU requested but unavailable on this device)\n");
        }
    } else {
        printf("  • Compute Unit: CPU Native [T-MAC / NEON]\n");
    }

    printf("  • Architecture: %s\n", arch_name);
    printf("  • Parameters  : Dim=%d, Hidden=%d, Layers=%d, Heads=%d/%d, Vocab=%d\n",
           llm.dim, llm.hidden_dim, llm.n_layers, llm.n_heads, llm.n_kv_heads, llm.vocab_size);
    printf("  • Tokenizer   : %s (%s)\n",
           llm.tokenizer_path[0] ? llm.tokenizer_path : "Auto-detected",
           llm.tokenizer_loaded ? "Loaded" : "Not Found");
    printf("  • KV Cache    : %s\n", (kv_type == 1) ? "INT8 Quantized (4x Bandwidth Compression)" : "FP32 Native");
    printf("  • Sampling    : Temp=%.2f, Top-p=%.2f, RepPenalty=%.2f\n",
           temperature, top_p, rep_penalty);
    printf("=================================================================\n\n");

    /* 2. Configure generation */
    eif_llm_gen_config_t cfg = {
        .max_new_tokens = max_tokens,
        .temperature = temperature,
        .top_p = top_p,
        .repetition_penalty = rep_penalty,
        .eos_token_id = -1,
        .kv_type = kv_type,
        .use_gpu = use_gpu,
        .loop_guard_tokens = loop_guard,
    };

    /* Auto-wrap prompt depending on model architecture and multimodal modality */
    char formatted_prompt[8192];
    const char *prompt_to_use = prompt;
    if (llm.arch == EIF_LLM_ARCH_QWEN35) {
        if (strstr(prompt, "<|im_start|>") == NULL) {
            if (image_path != NULL) {
                snprintf(formatted_prompt, sizeof(formatted_prompt),
                         "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>%s<|im_end|>\n<|im_start|>assistant\n", prompt);
            } else {
                snprintf(formatted_prompt, sizeof(formatted_prompt),
                         "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt);
            }
            prompt_to_use = formatted_prompt;
        }
    } else if (llm.arch == EIF_LLM_ARCH_QWEN2 && !raw_prompt && image_path == NULL &&
               strstr(prompt, "<|im_start|>") == NULL &&
               (strcasestr(model_path, "instruct") || strcasestr(model_path, "chat"))) {
        snprintf(formatted_prompt, sizeof(formatted_prompt),
                 "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt);
        prompt_to_use = formatted_prompt;
    } else if (image_path != NULL) {
        if (strstr(prompt, "<|image|>") == NULL) {
            snprintf(formatted_prompt, sizeof(formatted_prompt),
                     "<|image|><|begin_of_text|>%s", prompt);
            prompt_to_use = formatted_prompt;
        }
    }


    printf("--- Prompt ---\n%s\n\n", prompt);
    printf("--- Output Streaming ---\n");

    /* 3. Execute streaming generation */
    double t_start = get_time_sec();
    int gen_tokens = 0;
    if (image_path != NULL && llm.arch == EIF_LLM_ARCH_SMOLLM2) {
        gen_tokens = eif_llm_generate_vision_docling(&llm, image_path, prompt, &cfg, NULL, NULL);
    } else {
        gen_tokens = eif_llm_generate(&llm, prompt_to_use, &cfg, NULL, NULL);
    }
    double t_end = get_time_sec();

    double elapsed = t_end - t_start;
    double tok_per_sec = (elapsed > 0.0) ? (gen_tokens / elapsed) : 0.0;
    double ms_per_tok = (gen_tokens > 0) ? (elapsed * 1000.0 / gen_tokens) : 0.0;

    printf("\n\n-----------------------------------------------------------------\n");
    printf("  Performance Summary:\n");
    printf("  • Generated Tokens : %d tokens\n", gen_tokens);
    printf("  • Total Time       : %.3f seconds (wall-clock)\n", elapsed);
    printf("  • Latency          : %.2f ms / token\n", ms_per_tok);
    printf("  • Throughput       : %.1f tokens/sec\n", tok_per_sec);
    printf("-----------------------------------------------------------------\n");

    if (img_pixels) {
        stbi_image_free(img_pixels);
    }
    eif_llm_free(&llm);
    return 0;
}
