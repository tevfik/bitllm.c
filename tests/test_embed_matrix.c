/**
 * @file test_embed_matrix.c
 * @brief Embedding-model matrix through the public eif_llm_* API (what hosts such as gleann call).
 *
 * For every *.ref fixture (llama.cpp reference vectors, see tests/tools/gen_embed_ref.py) whose
 * GGUF exists in $BITLLM_MODELS this checks:
 *   - eif_llm_detect_arch routes the file to the BERT/encoder runtime (not the LLaMA fallback)
 *   - load succeeds and the output dimension matches
 *   - cosine to the llama.cpp vector for short texts (incl. word-shuffled and code input)
 *   - cosine for a ~1400 token input when the model context allows it (silent truncation shows here)
 *   - finite, unit-norm vectors for inputs of 1..8000 characters (NaN / out-of-range position rows)
 *   - empty / NULL input neither crashes nor yields NaN
 * Models that are not present are reported as SKIP; BITLLM_REQUIRE_MODELS=1 turns that into a failure.
 * Exit code 77 (ctest SKIP) when no model at all is available.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <stdint.h>
#include "eif_llm.h"

#ifndef EMBED_REF_DIR
#define EMBED_REF_DIR "tests/data/embed_ref"
#endif

/* Files that must be refused cleanly at load (no crash, no garbage): fused attn_qkv layouts etc. */
static const char *const KNOWN_UNSUPPORTED[] = {
    NULL};

/* Documented gaps against llama.cpp. The floors keep them from getting worse; fix the cause and delete the row. */
static const struct { const char *stem; float short_floor, long_floor; const char *why; } KNOWN_GAPS[] = {
    
    {NULL, 0, 0, NULL}};

#define COS_SHORT 0.98f
#define COS_LONG  0.97f

static const char *SHORT[7] = {
    "The quick brown fox jumps over the lazy dog.",
    "A fast brown fox leaps over a sleepy dog.",
    "Quarterly revenue grew by twelve percent in the European market.",
    "dog lazy the over jumps fox brown quick The.",
    "def add(a, b):\n    return a + b",
    "Türkiye'nin başkenti Ankara'dır; İstanbul çok büyük bir şehirdir. Ünlü şair Yunus Emre'yi okudum.",
    "日本語のテキストを埋め込みます。Café naïve résumé 123456 !!! 東京タワー"};
static const char *BASE[5] = {
    "The committee reviewed the annual budget and approved several infrastructure projects across the region.",
    "Engineers measured the vibration of the bridge under heavy traffic and recorded the results in a report.",
    "A small bakery on the corner of the street sells fresh bread, croissants and coffee every morning.",
    "Researchers trained a neural network to classify satellite images of forests, rivers and cities.",
    "The museum opened a new exhibition about ancient trade routes connecting Asia, Africa and Europe."};

static int g_fail = 0, g_pass = 0;
#define CHECK(c, ...) do { if (c) g_pass++; else { g_fail++; printf("    FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static float cosine(const float *a, const float *b, int n)
{
    double d = 0, x = 0, y = 0;
    for (int i = 0; i < n; i++) { d += (double)a[i] * b[i]; x += (double)a[i] * a[i]; y += (double)b[i] * b[i]; }
    return (x > 0 && y > 0) ? (float)(d / sqrt(x * y)) : 0.0f;
}

static int finite_unit(const float *v, int n, float *norm)
{
    double s = 0;
    for (int i = 0; i < n; i++) { if (!isfinite(v[i])) return 0; s += (double)v[i] * v[i]; }
    *norm = (float)sqrt(s);
    return fabsf(*norm - 1.0f) < 1e-2f;
}

static int run_model(const char *models, const char *ref_path, const char *stem)
{
    char gguf[1024];
    snprintf(gguf, sizeof(gguf), "%s/%s.gguf", models, stem);
    FILE *g = fopen(gguf, "rb");
    if (!g) return 0;
    fclose(g);

    FILE *f = fopen(ref_path, "rb");
    char magic[4]; int32_t hdr[3];
    if (!f || fread(magic, 1, 4, f) != 4 || memcmp(magic, "ERF1", 4) || fread(hdr, 4, 3, f) != 3) {
        printf("[%s] bad reference file\n", stem); g_fail++; if (f) fclose(f); return 1;
    }
    int dim = hdr[0], ns = hdr[1], has_long = hdr[2];
    float *ref = (float *)malloc((size_t)dim * (ns + has_long) * sizeof(float));
    if (fread(ref, 4, (size_t)dim * (ns + has_long), f) != (size_t)dim * (ns + has_long)) { g_fail++; fclose(f); free(ref); return 1; }
    fclose(f);

    int fail0 = g_fail;
    printf("[%s]\n", stem);
    CHECK(eif_llm_detect_arch(gguf) == EIF_LLM_ARCH_BERT, "detect_arch != BERT (would load through the LLaMA path)");

    static eif_llm_t llm;
    int rc_load = eif_llm_load(&llm, gguf, NULL, NULL, 0);
    for (int u = 0; KNOWN_UNSUPPORTED[u]; u++) {
        if (strcmp(stem, KNOWN_UNSUPPORTED[u]) != 0) continue;
        CHECK(rc_load != 0, "listed as unsupported but it loads now: remove it from KNOWN_UNSUPPORTED");
        printf("    unsupported architecture, refused at load (rc=%d) instead of crashing: %s\n", rc_load, rc_load != 0 ? "OK" : "FAILED");
        if (rc_load == 0) eif_llm_free(&llm);
        free(ref);
        return 1;
    }
    if (rc_load != 0) { printf("    FAIL: load\n"); g_fail++; free(ref); return 1; }
    CHECK(llm.dim == dim, "dim %d != reference %d", llm.dim, dim);
    float *v = (float *)calloc((size_t)dim > 0 ? dim : 1, sizeof(float));

    float short_th = COS_SHORT, long_th = COS_LONG;
    for (int u = 0; KNOWN_GAPS[u].stem; u++)
        if (strcmp(stem, KNOWN_GAPS[u].stem) == 0) { short_th = KNOWN_GAPS[u].short_floor; long_th = KNOWN_GAPS[u].long_floor; printf("    known gap (floor %.2f/%.2f): %s\n", short_th, long_th, KNOWN_GAPS[u].why); }
    float min_short = 2.0f;
    for (int i = 0; i < ns; i++) {
        int rc = eif_llm_embed(&llm, SHORT[i], v, EIF_LLM_POOL_MEAN);
        float nrm; CHECK(rc == 0, "embed short[%d] rc=%d", i, rc);
        if (rc) continue;
        CHECK(finite_unit(v, dim, &nrm), "short[%d] not finite/unit (norm %.4f)", i, nrm);
        float c = cosine(v, ref + (size_t)i * dim, dim);
        if (c < min_short) min_short = c;
        CHECK(c >= short_th, "short[%d] cosine to llama.cpp %.4f < %.2f", i, c, short_th);
    }

    char *lng = (char *)malloc(16384);
    lng[0] = 0;
    for (int i = 0; i < 75; i++) { char b[256]; snprintf(b, sizeof(b), "%s%s (%d)", i ? " " : "", BASE[i % 5], i); strcat(lng, b); }
    float c_long = -2.0f;
    if (has_long) {
        int rc = eif_llm_embed(&llm, lng, v, EIF_LLM_POOL_MEAN);
        CHECK(rc == 0, "embed long rc=%d", rc);
        if (rc == 0) {
            c_long = cosine(v, ref + (size_t)ns * dim, dim);
            CHECK(c_long >= long_th, "long input cosine %.4f < %.2f (ctx %d: silent truncation?)", c_long, long_th, llm.seq_len);
        }
    }

    static const int lens[] = {1, 20, 200, 800, 2000, 4000, 8000};
    size_t full = strlen(lng);
    for (size_t k = 0; k < sizeof(lens) / sizeof(lens[0]); k++) {
        char *t = (char *)malloc(16384);
        size_t n = 0;
        while (n < (size_t)lens[k]) { size_t m = full < (size_t)lens[k] - n ? full : (size_t)lens[k] - n; memcpy(t + n, lng, m); n += m; }
        t[n] = 0;
        float nrm = 0;
        int rc = eif_llm_embed(&llm, t, v, EIF_LLM_POOL_MEAN);
        CHECK(rc == 0 && finite_unit(v, dim, &nrm), "length %d chars: rc=%d norm=%.4f (NaN or bad position row)", lens[k], rc, nrm);
        free(t);
    }
    { /* empty text: either a clean error or a finite unit vector (CLS+SEP only), never NaN/crash */
        float nrm = 1.0f;
        int rc = eif_llm_embed(&llm, "", v, EIF_LLM_POOL_MEAN);
        CHECK(rc != 0 || finite_unit(v, dim, &nrm), "empty string gave rc=0 with norm %.4f", nrm);
    }
    CHECK(eif_llm_embed(&llm, NULL, v, EIF_LLM_POOL_MEAN) != 0, "NULL text must fail cleanly");

    printf("    dim=%d ctx=%d  min short cos=%.4f  long cos=%s%.4f  %s\n", dim, llm.seq_len, min_short,
           has_long ? "" : "(n/a) ", has_long ? c_long : 0.0f, g_fail == fail0 ? "OK" : "FAILED");
    eif_llm_free(&llm);
    free(lng); free(v); free(ref);
    return 1;
}

int main(void)
{
    const char *models = getenv("BITLLM_MODELS");
    if (!models) { printf("BITLLM_MODELS not set: SKIP\n"); return 77; }
    int require = getenv("BITLLM_REQUIRE_MODELS") != NULL;
    DIR *d = opendir(EMBED_REF_DIR);
    if (!d) { printf("no reference dir %s\n", EMBED_REF_DIR); return 1; }
    int ran = 0, missing = 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        size_t n = strlen(e->d_name);
        if (n < 5 || strcmp(e->d_name + n - 4, ".ref")) continue;
        char stem[256], path[1024];
        snprintf(stem, sizeof(stem), "%.*s", (int)(n - 4), e->d_name);
        snprintf(path, sizeof(path), "%s/%s", EMBED_REF_DIR, e->d_name);
        if (run_model(models, path, stem)) ran++;
        else { missing++; printf("[%s] SKIP (no GGUF in %s)\n", stem, models); }
    }
    closedir(d);
    printf("\nmodels run: %d, skipped: %d, checks passed: %d, failed: %d\n", ran, missing, g_pass, g_fail);
    if (ran == 0) return 77;
    return (g_fail || (require && missing)) ? 1 : 0;
}
