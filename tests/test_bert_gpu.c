/**
 * @file test_bert_gpu.c
 * @brief Vulkan encoder path (BITLLM_BERT_GPU=1) against the CPU kernels on real GGUF models.
 *
 * For every model found in $BITLLM_MODELS (all-MiniLM: head dim 32, unfused attention; bge-small; granite-30m;
 * bge-m3 and the reranker: head dim 64, fused attention; nomic: GEMM offload only) the same texts are embedded
 * once with the GPU disabled and once with it enabled, over lengths that are not multiples of any tile size.
 * The cosine must stay >= 0.997 (the CPU kernels quantise activations to INT8, so they sit ~1e-3 away from the
 * FP32-activation GPU result; both are within 0.9995 of llama.cpp, see test_embed_matrix with BITLLM_BERT_GPU=1;
 * a wrong attention, norm or scale gives cosines of 0.5-0.95) and the GPU path must really have run (eif_bert_gpu_forwards()), otherwise
 * the test would pass vacuously. Exit code 77 (ctest SKIP) without models or without a Vulkan device.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "eif_bert.h"
#include "eif_gpu.h"

static int g_fail = 0, g_pass = 0;
#define CHECK(c, ...) do { if (c) g_pass++; else { g_fail++; printf("    FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static const char *MODELS[] = {
    "all-MiniLM-L6-v2-Q8_0", "bge-small-en-v1.5-q8_0", "granite-embedding-30m-english-Q8_0",
    "bge-m3-q8_0", "bge-reranker-v2-m3-Q8_0", "nomic-embed-text-v1.5-Q8_0", NULL};

static char *make_text(int words, unsigned seed)
{
    static const char *W[] = {"edge", "computing", "runs", "models", "locally", "the", "quick", "brown", "fox",
                              "jumps", "over", "lazy", "dogs", "while", "sensors", "stream", "data", "Istanbul"};
    char *t = malloc((size_t)words * 12 + 4);
    t[0] = 0;
    for (int i = 0; i < words; i++) {
        seed = seed * 1103515245u + 12345u;
        strcat(t, W[(seed >> 16) % 18]);
        strcat(t, " ");
    }
    return t;
}

static double cosine(const float *a, const float *b, int n)
{
    double d = 0, x = 0, y = 0;
    for (int i = 0; i < n; i++) { d += (double)a[i] * b[i]; x += (double)a[i] * a[i]; y += (double)b[i] * b[i]; }
    return d / (sqrt(x * y) + 1e-30);
}

int main(void)
{
    const char *dir = getenv("BITLLM_MODELS");
    if (!dir) { printf("BITLLM_MODELS not set: SKIP\n"); return 77; }
    eif_gpu_context_t probe;
    if (!eif_gpu_init(&probe)) { printf("no Vulkan device: SKIP\n"); return 77; }
    eif_gpu_cleanup(&probe);
    const int lens[] = {3, 17, 40, 101, 233};
    int ran = 0;
    for (int m = 0; MODELS[m]; m++) {
        char path[1024];
        snprintf(path, sizeof path, "%s/%s.gguf", dir, MODELS[m]);
        FILE *f = fopen(path, "rb");
        if (!f) continue;
        fclose(f);
        unsetenv("BITLLM_BERT_GPU");
        static eif_bert_t cpu, gpu;
        memset(&cpu, 0, sizeof cpu); memset(&gpu, 0, sizeof gpu);
        if (eif_bert_load(&cpu, path) != 0) { printf("[%s] load failed\n", MODELS[m]); g_fail++; continue; }
        setenv("BITLLM_BERT_GPU", "1", 1);
        setenv("BITLLM_BERT_GPU_MIN_T", "1", 1);
        if (eif_bert_load(&gpu, path) != 0) { printf("[%s] load failed\n", MODELS[m]); g_fail++; eif_bert_free(&cpu); continue; }
        ran++;
        printf("[%s]\n", MODELS[m]);
        const int dim = cpu.config.dim;
        float *a = malloc(sizeof(float) * (size_t)dim * 2), *b = malloc(sizeof(float) * (size_t)dim * 2);
        double worst = 1.0;
        const unsigned long before = eif_bert_gpu_forwards();
        for (size_t i = 0; i < sizeof lens / sizeof *lens; i++) {
            char *t = make_text(lens[i], (unsigned)(7 + i));
            if (eif_bert_is_ranker(&cpu)) {
                float sc_c = 0, sc_g = 0;
                CHECK(eif_bert_rank(&cpu, "what runs models locally", t, &sc_c) == 0 &&
                      eif_bert_rank(&gpu, "what runs models locally", t, &sc_g) == 0, "rank failed");
                CHECK(fabsf(sc_c - sc_g) < 0.2f, "len %d: logit cpu %.4f gpu %.4f", lens[i], sc_c, sc_g);
            } else {
                CHECK(eif_bert_embed(&cpu, t, a) == 0 && eif_bert_embed(&gpu, t, b) == 0, "embed failed");
                double c = cosine(a, b, dim);
                if (c < worst) worst = c;
                CHECK(c >= 0.997, "len %d: cosine %.6f", lens[i], c);
            }
            free(t);
        }
        const unsigned long ran_gpu = eif_bert_gpu_forwards() - before;
        const int whole = (cpu.config.dim % 32 == 0) && !cpu.config.arch_modern && !cpu.config.use_rope;
        printf("    min cosine %.6f, %lu forward passes with the layers on the GPU\n", worst, ran_gpu);
        if (whole) CHECK(ran_gpu >= sizeof lens / sizeof *lens, "the GPU layer path did not run (%lu)", ran_gpu);
        free(a); free(b);
        eif_bert_free(&cpu);
        eif_bert_free(&gpu);
    }
    if (ran == 0) { printf("no model available: SKIP\n"); return 77; }
    printf("\nmodels run: %d, checks passed: %d, failed: %d\n", ran, g_pass, g_fail);
    return g_fail ? 1 : 0;
}
