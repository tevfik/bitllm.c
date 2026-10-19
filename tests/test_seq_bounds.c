/* tinyllm_forward at pos >= seq_len must refuse (NULL) instead of writing past the KV cache. */
#include "eif_tinyllm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CANARY 4096

int main(void)
{
    tinyllm_config_t cfg = {.dim = 8, .hidden_dim = 16, .n_layers = 2, .n_heads = 2, .n_kv_heads = 1,
                            .vocab_size = 10, .seq_len = 4, .qtype = TINYLLM_QTYPE_FP32};
    size_t pool_sz = tinyllm_memory_size(&cfg);
    unsigned char *pool = malloc(pool_sz + CANARY);
    memset(pool + pool_sz, 0xAB, CANARY);
    tinyllm_t llm;
    if (tinyllm_init(&llm, &cfg, pool, pool_sz) != 0) { printf("init failed\n"); return 1; }

    static float wts[4096];
    for (int i = 0; i < 4096; i++) wts[i] = 0.01f * ((i % 7) - 3);
    static float ones[64];
    for (int i = 0; i < 64; i++) ones[i] = 1.0f;
    tinyllm_weights_t *w = &llm.weights;
    w->token_embedding = wts; w->wcls = wts;
    w->wq = wts; w->wk = wts; w->wv = wts; w->wo = wts; w->w1 = wts; w->w2 = wts; w->w3 = wts;
    w->rms_att_weight = ones; w->rms_ffn_weight = ones; w->rms_final_weight = ones;

    int ok = 1;
    for (int pos = 0; pos < cfg.seq_len; pos++)
        if (!tinyllm_forward(&llm, pos % 10, pos)) { printf("  forward at pos %d failed\n", pos); ok = 0; }
    for (int pos = cfg.seq_len; pos < cfg.seq_len + 3; pos++)
        if (tinyllm_forward(&llm, 1, pos)) { printf("  forward at pos %d (>= seq_len) accepted\n", pos); ok = 0; }
    for (int i = 0; i < CANARY; i++)
        if (pool[pool_sz + i] != 0xAB) { printf("  memory after the state pool overwritten at +%d\n", i); ok = 0; break; }
    printf(ok ? "test_seq_bounds: ALL PASSED\n" : "test_seq_bounds: FAILED\n");
    free(pool);
    return ok ? 0 : 1;
}
