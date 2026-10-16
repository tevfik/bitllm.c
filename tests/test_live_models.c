/**
 * @file test_live_models.c
 * @brief Live End-to-End Validation of EIF-Runtime with Real BitNet LLM and Embedding Models.
 *
 * Validates:
 *  1. BERT GGUF & EIFM Embedding Model:
 *     - GGUF and EIFM model parsing
 *     - WordPiece tokenization
 *     - 384-dimensional dense embedding generation
 *     - L2 unit-norm verification
 *     - Pairwise semantic cosine similarity matrix
 *  2. BitNet b1.58 Causal LLM (Granite Docling):
 *     - Model load and parameter verification (dim=576, layers=30)
 *     - Forward pass & logits check
 *     - Autoregressive generation with FP32 KV-Cache
 *     - Autoregressive generation with INT8 Quantized KV-Cache
 *     - Last-token and Mean-pooled text embeddings from Causal LLM
 *     - Throughput (tokens/sec) and latency (ms/token) profiling
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#include "eif_runtime.h"

#define COLOR_GREEN  "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_RED    "\033[31m"
#define COLOR_BLUE   "\033[34m"
#define COLOR_CYAN   "\033[36m"
#define COLOR_BOLD   "\033[1m"
#define COLOR_RESET  "\033[0m"

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { \
        printf("  [" COLOR_GREEN "PASS" COLOR_RESET "] %s\n", msg); \
        g_pass++; \
    } else { \
        printf("  [" COLOR_RED "FAIL" COLOR_RESET "] %s (line %d)\n", msg, __LINE__); \
        g_fail++; \
    } \
} while(0)

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static const char* find_file(const char *subpath) {
    static char buf[1024];
    const char *prefixes[] = {
        "",
        "../../../../",
        "../../../",
        "../../",
        NULL
    };
    for (int i = 0; prefixes[i] != NULL; i++) {
        snprintf(buf, sizeof(buf), "%s%s", prefixes[i], subpath);
        if (access(buf, R_OK) == 0) {
            return buf;
        }
    }
    return NULL;
}

/* ========================================================================= */
/* 1. Embedding Model Validation (GGUF & EIFM)                               */
/* ========================================================================= */
static void validate_embedding_models(void) {
    printf("\n" COLOR_BOLD "=================================================================" COLOR_RESET "\n");
    printf(COLOR_BOLD "  [1/2] Live Validation: BERT Embedding Models (GGUF & EIFM)" COLOR_RESET "\n");
    printf(COLOR_BOLD "=================================================================" COLOR_RESET "\n");

    const char *gguf_path = find_file("models/dist/minilm.gguf");
    const char *eifm_path = find_file("models/dist/minilm.eifm");

    if (!gguf_path || !eifm_path) {
        printf("  [" COLOR_YELLOW "SKIP" COLOR_RESET "] Embedding models not found (models/dist/minilm.gguf/eifm). Skipping live test.\n");
        return;
    }

    CHECK(gguf_path != NULL, "Found models/dist/minilm.gguf");
    CHECK(eifm_path != NULL, "Found models/dist/minilm.eifm");

    /* A. Test GGUF Loading */
    eif_bert_t bert_gguf;
    memset(&bert_gguf, 0, sizeof(bert_gguf));
    int rc_gguf = eif_bert_load(&bert_gguf, gguf_path);
    CHECK(rc_gguf == 0, "eif_bert_load loaded GGUF format model (Q8_0 INT8)");
    CHECK(bert_gguf.config.dim == 384, "GGUF model embedding dimension == 384");
    CHECK(bert_gguf.config.n_layers == 6, "GGUF model layer count == 6");

    /* B. Test EIFM Loading */
    eif_bert_t bert_eifm;
    memset(&bert_eifm, 0, sizeof(bert_eifm));
    int rc_eifm = eif_bert_load(&bert_eifm, eifm_path);
    CHECK(rc_eifm == 0, "eif_bert_load loaded EIFM binary format model");
    CHECK(bert_eifm.config.dim == 384, "EIFM model embedding dimension == 384");
    CHECK(bert_eifm.config.n_layers == 6, "EIFM model layer count == 6");

    /* C. Semantic Similarity Matrix Test with 4 Distinct Sentences */
    const char *sentences[4] = {
        "Edge intelligence enables local deep learning inference without cloud.",
        "Microcontrollers and edge sensors process neural networks on-device.",
        "Traditional cloud computing relies on centralized hyperscale data centers.",
        "A sweet recipe for chocolate strawberry cheesecake with sugar syrup."
    };

    float emb[4][384];
    double t_start = get_time_sec();
    for (int i = 0; i < 4; i++) {
        int r = eif_bert_embed(&bert_gguf, sentences[i], emb[i]);
        CHECK(r == 0, "Generated 384-dim embedding for test sentence");

        /* L2 Normalization Check */
        float norm = 0.0f;
        for (int d = 0; d < 384; d++) norm += emb[i][d] * emb[i][d];
        norm = sqrtf(norm);
        CHECK(fabsf(norm - 1.0f) < 1e-3f, "Embedding vector is strictly L2 normalized (|norm - 1.0| < 1e-3)");
    }
    double t_embed = get_time_sec() - t_start;
    printf(COLOR_CYAN "  ⚡ Embedding Speed: %.2f ms / sentence (%.1f sentences/sec)" COLOR_RESET "\n",
           (t_embed / 4.0) * 1000.0, 4.0 / t_embed);

    /* Cosine Similarity Matrix */
    printf("\n  📊 Pairwise Cosine Similarity Matrix:\n");
    printf("         [0: Edge AI] [1: MCU AI] [2: Cloud]   [3: Cake]\n");
    for (int i = 0; i < 4; i++) {
        printf("    [%d]  ", i);
        for (int j = 0; j < 4; j++) {
            float sim = eif_llm_cosine_similarity(emb[i], emb[j], 384);
            printf("%8.4f   ", sim);
        }
        printf("\n");
    }

    float sim_edge_edge = eif_llm_cosine_similarity(emb[0], emb[1], 384);
    float sim_edge_cloud = eif_llm_cosine_similarity(emb[0], emb[2], 384);
    float sim_edge_cake = eif_llm_cosine_similarity(emb[0], emb[3], 384);

    printf("\n  • Sim(Edge AI, MCU AI)   = %.4f (Expected: Strong Correlation)\n", sim_edge_edge);
    printf("  • Sim(Edge AI, Cloud)    = %.4f (Expected: Moderate/Opposite Domain)\n", sim_edge_cloud);
    printf("  • Sim(Edge AI, Cake)     = %.4f (Expected: Unrelated / Low)\n", sim_edge_cake);

    CHECK(sim_edge_edge > sim_edge_cloud, "Semantic Check: Edge AI is closer to MCU AI than to Cloud");
    CHECK(sim_edge_edge > sim_edge_cake + 0.20f, "Semantic Check: Edge AI similarity is significantly higher than Cake recipe");

    /* D. Cross-Format Consistency: EIFM vs GGUF Embeddings */
    float emb_eifm_0[384];
    eif_bert_embed(&bert_eifm, sentences[0], emb_eifm_0);
    float format_agreement = eif_llm_cosine_similarity(emb[0], emb_eifm_0, 384);
    printf("  • Format Agreement (GGUF vs EIFM output) = %.6f\n", format_agreement);
    CHECK(format_agreement > 0.999f, "GGUF and EIFM produce identical mathematical embeddings (cos > 0.999)");

    eif_bert_free(&bert_gguf);
    eif_bert_free(&bert_eifm);
}

/* ========================================================================= */
/* 2. BitNet b1.58 Causal LLM Validation (Granite Docling)                   */
/* ========================================================================= */
static void validate_bitnet_llm(void) {
    printf("\n" COLOR_BOLD "=================================================================" COLOR_RESET "\n");
    printf(COLOR_BOLD "  [2/2] Live Validation: BitNet b1.58 Causal LLM (Granite Docling)" COLOR_RESET "\n");
    printf(COLOR_BOLD "=================================================================" COLOR_RESET "\n");

    const char *model_path = find_file("artifacts/granite_docling/granite_docling_bitnet_dense.eifm");
    if (!model_path) {
        printf("  [" COLOR_YELLOW "SKIP" COLOR_RESET "] BitNet model not found (artifacts/granite_docling/...). Skipping.\n");
        return;
    }
    CHECK(model_path != NULL, "Found artifacts/granite_docling/granite_docling_bitnet_dense.eifm");

    eif_llm_t llm;
    memset(&llm, 0, sizeof(llm));
    int rc = eif_llm_load(&llm, model_path, NULL, NULL, 0);
    CHECK(rc == 0, "eif_llm_load parsed and mapped BitNet b1.58 ternary model");
    CHECK(llm.arch == EIF_LLM_ARCH_SMOLLM2, "Auto-detected Transformer Causal LLM architecture");
    CHECK(llm.dim == 576, "Model hidden dimension == 576");
    CHECK(llm.n_layers == 30, "Model layer count == 30 layers");
    CHECK(llm.vocab_size == 100352, "Tokenizer vocabulary size == 100,352 tokens");

    /* A. Single Token Forward Pass & Logits Validation */
    rc = eif_llm_forward(&llm, 1, 0);
    CHECK(rc == 0, "eif_llm_forward single step forward pass executed");
    float *logits = eif_llm_get_logits(&llm);
    CHECK(logits != NULL, "eif_llm_get_logits returned non-NULL pointer");

    /* B. Autoregressive Generation with FP32 KV-Cache */
    printf("\n  ▶ Autoregressive Generation (FP32 KV-Cache):\n");
    eif_llm_gen_config_t cfg_fp32 = {
        .max_new_tokens = 16,
        .temperature = 0.7f,
        .top_p = 0.9f,
        .repetition_penalty = 1.15f,
        .kv_type = 0,
        .eos_token_id = -1
    };

    double t0 = get_time_sec();
    int n_gen_fp32 = eif_llm_generate(&llm, "Docling", &cfg_fp32, NULL, NULL);
    double dt_fp32 = get_time_sec() - t0;
    CHECK(n_gen_fp32 == 16, "Generated requested 16 tokens with FP32 KV-Cache");
    double tps_fp32 = (double)n_gen_fp32 / dt_fp32;
    printf(COLOR_CYAN "  ⚡ FP32 KV Throughput: %.1f tokens/sec (Latency: %.2f ms/token, Total: %.3fs)" COLOR_RESET "\n",
           tps_fp32, (dt_fp32 / n_gen_fp32) * 1000.0, dt_fp32);

    /* C. Autoregressive Generation with INT8 Quantized KV-Cache */
    printf("\n  ▶ Autoregressive Generation (INT8 Quantized KV-Cache, 4x Memory Reduction):\n");
    eif_llm_gen_config_t cfg_int8 = {
        .max_new_tokens = 16,
        .temperature = 0.7f,
        .top_p = 0.9f,
        .repetition_penalty = 1.15f,
        .kv_type = 1,
        .eos_token_id = -1
    };

    t0 = get_time_sec();
    int n_gen_int8 = eif_llm_generate(&llm, "Docling", &cfg_int8, NULL, NULL);
    double dt_int8 = get_time_sec() - t0;
    CHECK(n_gen_int8 == 16, "Generated requested 16 tokens with INT8 KV-Cache");
    double tps_int8 = (double)n_gen_int8 / dt_int8;
    printf(COLOR_CYAN "  ⚡ INT8 KV Throughput: %.1f tokens/sec (Latency: %.2f ms/token, Total: %.3fs)" COLOR_RESET "\n",
           tps_int8, (dt_int8 / n_gen_int8) * 1000.0, dt_int8);

    /* D. Causal LLM Dense Text Embeddings */
    float emb_last[576] = {0};
    float emb_mean[576] = {0};
    int r_last = eif_llm_embed(&llm, "Invoice processing on embedded core", emb_last, EIF_LLM_POOL_LAST);
    int r_mean = eif_llm_embed(&llm, "Invoice processing on embedded core", emb_mean, EIF_LLM_POOL_MEAN);
    CHECK(r_last == 0 && r_mean == 0, "eif_llm_embed generated both LAST and MEAN pooled embeddings");

    /* Validate L2 norm of Causal LLM embeddings */
    float norm_last = 0.f, norm_mean = 0.f;
    for (int d = 0; d < 576; d++) {
        norm_last += emb_last[d] * emb_last[d];
        norm_mean += emb_mean[d] * emb_mean[d];
    }
    CHECK(fabsf(sqrtf(norm_last) - 1.0f) < 1e-3f, "LLM POOL_LAST embedding is L2 normalized");
    CHECK(fabsf(sqrtf(norm_mean) - 1.0f) < 1e-3f, "LLM POOL_MEAN embedding is L2 normalized");

    eif_llm_free(&llm);
    CHECK(!llm.is_initialized, "eif_llm_free cleanly deallocated all BitNet LLM resources");
}

int main(void) {
    printf(COLOR_BOLD "=================================================================\n" COLOR_RESET);
    printf(COLOR_BOLD "    EIF-Runtime Live BitNet LLM & Embedding Model Test Suite\n" COLOR_RESET);
    printf(COLOR_BOLD "=================================================================\n" COLOR_RESET);

    validate_embedding_models();
    validate_bitnet_llm();

    printf("\n" COLOR_BOLD "=================================================================" COLOR_RESET "\n");
    printf(COLOR_BOLD "  Live Verification Summary:\n" COLOR_RESET);
    printf("    Passed: " COLOR_GREEN "%d" COLOR_RESET "\n", g_pass);
    printf("    Failed: %s%d" COLOR_RESET "\n", g_fail > 0 ? COLOR_RED : COLOR_GREEN, g_fail);
    printf(COLOR_BOLD "=================================================================\n" COLOR_RESET);

    return (g_fail == 0) ? 0 : 1;
}
