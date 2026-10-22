/**
 * @file test_rerank_matrix.c
 * @brief Cross-encoder reranker (bge-reranker-v2-m3 GGUF) against the HuggingFace fp32 reference.
 *
 * tests/data/rerank_ref/<stem>.tsv holds the reference logit and token count per (query, document) pair
 * (tests/tools/gen_rerank_ref.py). With $BITLLM_MODELS/<stem>.gguf present this checks:
 *   - the pair tokenisation produces exactly the reference number of tokens (<s> q </s></s> d </s>)
 *   - the Q8_0 logit is within 0.30 of the fp32 logit and the ranking order of the pairs agrees
 *   - eif_llm_rerank (the public API) gives the same scores as the encoder call
 *   - an embedding-only model is refused with a clean error
 * Exit code 77 (ctest SKIP) without the model.
 */
#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "eif_llm.h"

#ifndef RERANK_REF_DIR
#define RERANK_REF_DIR "tests/data/rerank_ref"
#endif

static int g_fail = 0, g_pass = 0;
#define CHECK(c, ...) do { if (c) g_pass++; else { g_fail++; printf("    FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

typedef struct { float logit; int ntok; char q[512], d[1024]; } pair_t;

static int load_pairs(const char *path, pair_t *p, int max)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char line[2048];
    int n = 0;
    while (n < max && fgets(line, sizeof line, f)) {
        if (line[0] == '#') continue;
        char *t1 = strchr(line, '\t'), *t2 = t1 ? strchr(t1 + 1, '\t') : NULL, *t3 = t2 ? strchr(t2 + 1, '\t') : NULL;
        if (!t3) continue;
        *t1 = *t2 = *t3 = 0;
        char *nl = strchr(t3 + 1, '\n'); if (nl) *nl = 0;
        p[n].logit = (float)atof(line); p[n].ntok = atoi(t1 + 1);
        snprintf(p[n].q, sizeof p[n].q, "%s", t2 + 1); snprintf(p[n].d, sizeof p[n].d, "%s", t3 + 1);
        n++;
    }
    fclose(f);
    return n;
}

int main(void)
{
    const char *models = getenv("BITLLM_MODELS");
    if (!models) { printf("BITLLM_MODELS not set: SKIP\n"); return 77; }
    DIR *d = opendir(RERANK_REF_DIR);
    if (!d) return 77;
    int ran = 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        size_t n = strlen(e->d_name);
        if (n < 5 || strcmp(e->d_name + n - 4, ".tsv")) continue;
        char stem[256], gguf[1024], ref[1024];
        snprintf(stem, sizeof stem, "%.*s", (int)(n - 4), e->d_name);
        snprintf(gguf, sizeof gguf, "%s/%s.gguf", models, stem);
        snprintf(ref, sizeof ref, "%s/%s", RERANK_REF_DIR, e->d_name);
        FILE *g = fopen(gguf, "rb");
        if (!g) { printf("[%s] SKIP (no GGUF)\n", stem); continue; }
        fclose(g);
        static pair_t P[64];
        int np = load_pairs(ref, P, 64);
        printf("[%s] %d pairs\n", stem, np);
        static eif_llm_t llm;
        if (eif_llm_load(&llm, gguf, NULL, NULL, 0) != 0) { printf("    FAIL: load\n"); g_fail++; continue; }
        float *sc = calloc(np, sizeof(float));
        const char *docs[64];
        for (int i = 0; i < np; i++) docs[i] = P[i].d;
        double max_err = 0;
        for (int i = 0; i < np; i++) {
            int32_t tok[1024]; uint8_t ty[1024];
            int nt = eif_bert_tokenize_pair(&llm.backend.bert, P[i].q, P[i].d, tok, ty, 1024);
            CHECK(nt == P[i].ntok, "pair %d: %d tokens, reference %d", i, nt, P[i].ntok);
            float s;
            int rc = eif_bert_rank(&llm.backend.bert, P[i].q, P[i].d, &s);
            CHECK(rc == 0, "pair %d: rank rc=%d", i, rc);
            if (rc) continue;
            sc[i] = s;
            double err = fabs(s - P[i].logit);
            if (err > max_err) max_err = err;
            CHECK(err < 0.30, "pair %d: logit %.3f vs reference %.3f", i, s, P[i].logit);
        }
        for (int i = 0; i < np; i++) for (int j = i + 1; j < np; j++)
            if (fabsf(P[i].logit - P[j].logit) > 1.0f)
                CHECK((P[i].logit > P[j].logit) == (sc[i] > sc[j]), "order of pairs %d/%d differs from the reference", i, j);
        /* the public API must agree for a shared query (pairs 0..1 share one) */
        float via_api[2] = {0, 0};
        int rc = eif_llm_rerank(&llm, P[0].q, docs, 2, via_api);
        CHECK(rc == 0 && fabsf(via_api[0] - sc[0]) < 1e-4f && fabsf(via_api[1] - sc[1]) < 1e-4f, "eif_llm_rerank disagrees (rc=%d)", rc);
        printf("    max |logit error| %.3f\n", max_err);
        eif_llm_free(&llm);
        free(sc);
        ran++;

        /* an embedding-only model must be refused cleanly */
        char mini[1024];
        snprintf(mini, sizeof mini, "%s/all-MiniLM-L6-v2-Q8_0.gguf", models);
        FILE *m = fopen(mini, "rb");
        if (m) {
            fclose(m);
            static eif_llm_t emb;
            if (eif_llm_load(&emb, mini, NULL, NULL, 0) == 0) {
                float s = 0;
                CHECK(eif_llm_rerank(&emb, "q", docs, 1, &s) != 0, "embedding model must not be rerankable");
                eif_llm_free(&emb);
            }
        }
    }
    closedir(d);
    printf("\nmodels: %d, checks passed: %d, failed: %d\n", ran, g_pass, g_fail);
    return ran == 0 ? 77 : (g_fail ? 1 : 0);
}
