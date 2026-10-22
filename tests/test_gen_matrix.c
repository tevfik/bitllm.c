/**
 * @file test_gen_matrix.c
 * @brief Text-generation matrix through the public eif_llm_* API.
 *
 * Every *.gguf / *.eifm in $BITLLM_GEN_MODELS is loaded and checked for:
 *   - the expected runtime (eif_llm_detect_arch, derived from the file name)
 *   - greedy answer to a factual question contains "Paris" (ChatML wrapped like the CLI does)
 *   - greedy decoding is repeatable after eif_llm_reset (state leaks show up here)
 *   - the callback saw exactly as many pieces as eif_llm_generate returned tokens
 *   - first logits are finite, a ~600 token prompt neither crashes nor returns an error
 * Files that look like neither Qwen nor SmolLM/LLaMA are reported, never silently accepted.
 * Exit code 77 (ctest SKIP) when $BITLLM_GEN_MODELS is unset or empty.
 */
#include <ctype.h>
#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "eif_llm.h"

static int g_fail = 0, g_pass = 0;
#define CHECK(c, ...) do { if (c) g_pass++; else { g_fail++; printf("    FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

typedef struct { char text[4096]; int pieces; } sink_t;
static void on_piece(const char *piece, int id, void *u)
{
    (void)id;
    sink_t *s = (sink_t *)u;
    size_t n = strlen(s->text);
    size_t m = strlen(piece);
    if (n + m < sizeof(s->text) - 1) memcpy(s->text + n, piece, m + 1);
    s->pieces++;
}

static int icontains(const char *h, const char *needle)
{
    size_t n = strlen(needle);
    for (; *h; h++) if (strncasecmp(h, needle, n) == 0) return 1;
    return 0;
}

static int gen(eif_llm_t *llm, const char *prompt, int max_new, sink_t *s)
{
    eif_llm_gen_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.max_new_tokens = max_new;
    cfg.temperature = 0.0f;
    cfg.top_p = 1.0f;
    cfg.repetition_penalty = 1.0f;
    cfg.eos_token_id = -1;
    memset(s, 0, sizeof(*s));
    return eif_llm_generate(llm, prompt, &cfg, on_piece, s);
}

static void run(const char *dir, const char *name)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    eif_llm_arch_t want = EIF_LLM_ARCH_UNKNOWN;
    if (icontains(name, "qwen3")) want = EIF_LLM_ARCH_QWEN35;
    else if (icontains(name, "qwen2")) want = EIF_LLM_ARCH_QWEN2;
    else if (icontains(name, "smollm") || icontains(name, "llama")) want = EIF_LLM_ARCH_SMOLLM2;
    printf("[%s]\n", name);
    if (want == EIF_LLM_ARCH_UNKNOWN) { printf("    no expectation for this file name (add one)\n"); return; }

    CHECK(eif_llm_detect_arch(path) == want, "detect_arch=%d expected %d", (int)eif_llm_detect_arch(path), (int)want);
    static eif_llm_t llm;
    if (eif_llm_load(&llm, path, NULL, NULL, 0) != 0) { printf("    FAIL: load\n"); g_fail++; return; }

    /* Qwen3.5 thinks by default; the empty think block is the chat template's enable_thinking=false form */
    const char *p = want == EIF_LLM_ARCH_QWEN35
        ? "<|im_start|>user\nWhat is the capital of France? Answer with one word.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        : "<|im_start|>user\nWhat is the capital of France? Answer with one word.<|im_end|>\n<|im_start|>assistant\n";
    sink_t a, b;
    int na = gen(&llm, p, 24, &a);
    CHECK(na > 0, "generate rc=%d", na);
    CHECK(icontains(a.text, "Paris"), "answer lacks 'Paris': \"%s\"", a.text);
    CHECK(na == a.pieces || na == a.pieces + 1, "callback pieces %d vs tokens %d", a.pieces, na);

    eif_llm_reset(&llm);
    int nb = gen(&llm, p, 24, &b);
    CHECK(nb == na && strcmp(a.text, b.text) == 0, "greedy rerun differs: \"%s\" vs \"%s\"", a.text, b.text);

    eif_llm_reset(&llm);
    char *longp = (char *)malloc(8192);
    strcpy(longp, "<|im_start|>user\nSummarise: ");
    for (int i = 0; i < 40; i++) strcat(longp, "The engineers reviewed the bridge design and wrote a long report. ");
    strcat(longp, "<|im_end|>\n<|im_start|>assistant\n");
    sink_t c;
    int nc = gen(&llm, longp, 8, &c);
    CHECK(nc >= 0, "long prompt rc=%d", nc);
    float *lg = eif_llm_get_logits(&llm);
    int finite = lg != NULL;
    for (int i = 0; lg && i < llm.vocab_size; i++) if (!isfinite(lg[i])) { finite = 0; break; }
    CHECK(finite, "logits not finite after long prompt");
    printf("    answer: \"%.60s\"  (%d tokens)\n", a.text, na);
    free(longp);
    eif_llm_free(&llm);
}

int main(void)
{
    const char *dir = getenv("BITLLM_GEN_MODELS");
    if (!dir) { printf("BITLLM_GEN_MODELS not set: SKIP\n"); return 77; }
    DIR *d = opendir(dir);
    if (!d) return 77;
    int n = 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        size_t l = strlen(e->d_name);
        if ((l > 5 && !strcmp(e->d_name + l - 5, ".gguf")) || (l > 5 && !strcmp(e->d_name + l - 5, ".eifm"))) { run(dir, e->d_name); n++; }
    }
    closedir(d);
    printf("\nmodels: %d, checks passed: %d, failed: %d\n", n, g_pass, g_fail);
    return n == 0 ? 77 : (g_fail ? 1 : 0);
}
