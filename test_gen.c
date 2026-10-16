#include <stdio.h>
#include <stdlib.h>
#include "eif_runtime.h"

int main() {
    eif_llm_t llm;
    int rc = eif_llm_load(&llm, "artifacts/qwen35/qwen35_bitnet_dense.eifm", "artifacts/qwen35/qwen_tokenizer.bin", NULL, 0);
    printf("Model loaded rc=%d, arch=%d, vocab=%d\n", rc, llm.arch, llm.vocab_size);
    printf("Tokenizer: bos=%d ('%s'), eos=%d ('%s')\n",
           llm.tokenizer.bos_token_id, llm.tokenizer.vocab[llm.tokenizer.bos_token_id],
           llm.tokenizer.eos_token_id, llm.tokenizer.vocab[llm.tokenizer.eos_token_id]);

    eif_llm_gen_config_t cfg = {
        .max_new_tokens = 64,
        .temperature = 0.7f,
        .top_p = 0.9f,
        .repetition_penalty = 1.15f,
        .eos_token_id = -1,
    };

    printf("Calling eif_llm_generate with 'Edge computing nedir?'...\n");
    int count = eif_llm_generate(&llm, "Edge computing nedir?", &cfg, NULL, NULL);
    printf("\nReturned token count: %d\n", count);
    return 0;
}
