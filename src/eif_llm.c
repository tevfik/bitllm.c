/**
 * @file eif_llm.c
 * @brief Unified C99 Runtime Implementation for Large Language Models
 */

#if defined(__linux__)
#define _GNU_SOURCE
#endif

#include "eif_llm.h"
#include "eif_siglip.h"

#include <ctype.h>
#include <libgen.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define DEFAULT_BUFFER_SIZE (256 * 1024 * 1024) /* 256 MB */

static eif_llm_arch_t detect_gguf_arch(const char *model_path)
{
    FILE *f = fopen(model_path, "rb");
    if (!f) return EIF_LLM_ARCH_BERT;

    uint32_t magic = 0, ver = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (fread(&magic, 4, 1, f) != 1 || fread(&ver, 4, 1, f) != 1 ||
        fread(&n_tensors, 8, 1, f) != 1 || fread(&n_kv, 8, 1, f) != 1) {
        fclose(f);
        return EIF_LLM_ARCH_BERT;
    }
    if (magic != 0x46554747) {
        fclose(f);
        return EIF_LLM_ARCH_UNKNOWN;
    }

    eif_llm_arch_t result = EIF_LLM_ARCH_UNKNOWN;
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t klen = 0;
        if (fread(&klen, 8, 1, f) != 1 || klen > 1024) break;
        char key[256];
        size_t to_read = klen < sizeof(key) - 1 ? klen : sizeof(key) - 1;
        if (fread(key, 1, to_read, f) != to_read) break;
        key[to_read] = '\0';
        if (klen > to_read) fseek(f, (long)(klen - to_read), SEEK_CUR);

        uint32_t vtype = 0;
        if (fread(&vtype, 4, 1, f) != 1) break;

        if (strcmp(key, "general.architecture") == 0 && vtype == 8) {
            uint64_t slen = 0;
            if (fread(&slen, 8, 1, f) != 1) break;
            char arch_val[128];
            size_t sread = slen < sizeof(arch_val) - 1 ? slen : sizeof(arch_val) - 1;
            if (fread(arch_val, 1, sread, f) != sread) break;
            arch_val[sread] = '\0';

            if (strcmp(arch_val, "bert") == 0 || strcmp(arch_val, "nomic-bert") == 0) {
                result = EIF_LLM_ARCH_BERT;
            } else if (strcmp(arch_val, "qwen3") == 0 || strcmp(arch_val, "qwen35") == 0) {
                result = EIF_LLM_ARCH_QWEN35;
            } else if (strcmp(arch_val, "qwen2") == 0 || strcmp(arch_val, "qwen2.5") == 0) {
                result = EIF_LLM_ARCH_QWEN2;
            } else {
                result = EIF_LLM_ARCH_SMOLLM2;
            }
            break;
        }

        /* Skip other metadata */
        if (vtype == 0 || vtype == 1 || vtype == 7) fseek(f, 1, SEEK_CUR);
        else if (vtype == 2 || vtype == 3) fseek(f, 2, SEEK_CUR);
        else if (vtype == 4 || vtype == 5 || vtype == 6) fseek(f, 4, SEEK_CUR);
        else if (vtype == 8) {
            uint64_t sl = 0;
            if (fread(&sl, 8, 1, f) != 1) break;
            fseek(f, (long)sl, SEEK_CUR);
        } else if (vtype >= 10 && vtype <= 12) fseek(f, 8, SEEK_CUR);
        else if (vtype == 9) {
            uint32_t atype = 0;
            uint64_t acount = 0;
            if (fread(&atype, 4, 1, f) != 1 || fread(&acount, 8, 1, f) != 1) break;
            if (atype == 8) {
                for (uint64_t k = 0; k < acount; k++) {
                    uint64_t sl = 0;
                    if (fread(&sl, 8, 1, f) != 1) break;
                    fseek(f, (long)sl, SEEK_CUR);
                }
            } else if (atype == 0 || atype == 1 || atype == 7) fseek(f, (long)acount, SEEK_CUR);
            else if (atype == 2 || atype == 3) fseek(f, (long)(acount * 2), SEEK_CUR);
            else if (atype == 4 || atype == 5 || atype == 6) fseek(f, (long)(acount * 4), SEEK_CUR);
            else if (atype >= 10 && atype <= 12) fseek(f, (long)(acount * 8), SEEK_CUR);
        }
    }
    fclose(f);
    return (result != EIF_LLM_ARCH_UNKNOWN) ? result : EIF_LLM_ARCH_BERT;
}

eif_llm_arch_t eif_llm_detect_arch(const char *model_path)
{
    if (!model_path) return EIF_LLM_ARCH_UNKNOWN;

    FILE *f = fopen(model_path, "rb");
    if (!f) return EIF_LLM_ARCH_UNKNOWN;

    uint32_t magic = 0;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return EIF_LLM_ARCH_UNKNOWN;
    }
    fclose(f);

    /* 0x51574E35 = 'QWN5' */
    if (magic == 0x51574E35) {
        return EIF_LLM_ARCH_QWEN35;
    }
    /* 0x53564C4D = 'SVLM', 0x54544D4C = 'TTML', 0x544C4C4D = 'TLLM' */
    if (magic == 0x53564C4D || magic == 0x54544D4C || magic == 0x544C4C4D) {
        return EIF_LLM_ARCH_SMOLLM2;
    }
    /* 0x54524542 = 'BERT' */
    if (magic == 0x54524542) {
        return EIF_LLM_ARCH_BERT;
    }
    /* 0x46554747 = 'GGUF' */
    if (magic == 0x46554747) {
        return detect_gguf_arch(model_path);
    }

    /* Fallback heuristic by filename */
    if (strstr(model_path, "bert") != NULL || strstr(model_path, "minilm") != NULL) {
        return EIF_LLM_ARCH_BERT;
    }
    if (strstr(model_path, "qwen2") != NULL || strstr(model_path, "qwen2.5") != NULL) {
        return EIF_LLM_ARCH_QWEN2;
    }
    if (strstr(model_path, "qwen") != NULL) {
        return EIF_LLM_ARCH_QWEN35;
    }
    if (strstr(model_path, "smol") != NULL || strstr(model_path, "docling") != NULL ||
        strstr(model_path, "granite") != NULL || strstr(model_path, "llama") != NULL ||
        strstr(model_path, "minicpm") != NULL) {
        return EIF_LLM_ARCH_SMOLLM2;
    }

    return EIF_LLM_ARCH_UNKNOWN;
}

size_t eif_llm_compute_buffer_size(const char *model_path)
{
    if (!model_path) return DEFAULT_BUFFER_SIZE;
    FILE *f = fopen(model_path, "rb");
    if (!f) return DEFAULT_BUFFER_SIZE;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fclose(f);
    if (sz <= 0) return DEFAULT_BUFFER_SIZE;

    tinyllm_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    char arch_buf[64] = {0};
    size_t mem_sz = 0;
    if (tinyllm_read_gguf_config(model_path, &cfg, arch_buf, sizeof(arch_buf)) == 0 && cfg.dim > 0) {
        mem_sz = tinyllm_memory_size(&cfg);
    } else {
        /* Default activation/KV cache memory heuristic (e.g. 512MB) */
        mem_sz = 512 * 1024 * 1024;
    }

    size_t weight_estimate = (size_t)sz * 2;
    size_t needed = weight_estimate + mem_sz + (128 * 1024 * 1024);
    return needed > DEFAULT_BUFFER_SIZE ? needed : DEFAULT_BUFFER_SIZE;
}

static bool find_tokenizer_path(const char *model_path, char *out_tok_path, size_t max_len)
{
    if (!model_path || !out_tok_path) return false;

    char dir_buf[512];
    snprintf(dir_buf, sizeof(dir_buf), "%s", model_path);
    char *dir = dirname(dir_buf);

    /* Check in model directory */
    char candidate[512];
    snprintf(candidate, sizeof(candidate), "%s/tokenizer.bin", dir);
    if (access(candidate, F_OK) == 0) {
        snprintf(out_tok_path, max_len, "%s", candidate);
        return true;
    }

    snprintf(candidate, sizeof(candidate), "%s/qwen_tokenizer.bin", dir);
    if (access(candidate, F_OK) == 0) {
        snprintf(out_tok_path, max_len, "%s", candidate);
        return true;
    }

    /* Known global artifact paths */
    const char *defaults[] = {
        "artifacts/minicpm5/tokenizer.bin",
        "artifacts/granite_docling/tokenizer.bin",
        "artifacts/smolvlm2_bitnet/tokenizer.bin",
        "artifacts/qwen35/qwen_tokenizer.bin",
        NULL
    };

    for (int i = 0; defaults[i] != NULL; i++) {
        if (access(defaults[i], F_OK) == 0) {
            snprintf(out_tok_path, max_len, "%s", defaults[i]);
            return true;
        }
    }

    return false;
}

int eif_llm_load(eif_llm_t *llm, const char *model_path, const char *tokenizer_path,
                 void *buffer, size_t buffer_size)
{
    if (!llm || !model_path) return -1;
    memset(llm, 0, sizeof(eif_llm_t));

    snprintf(llm->model_path, sizeof(llm->model_path), "%s", model_path);
    llm->arch = eif_llm_detect_arch(model_path);
    if (llm->arch == EIF_LLM_ARCH_UNKNOWN) {
        fprintf(stderr, "[EIF LLM Error] Unknown model architecture: %s\n", model_path);
        return -2;
    }

    /* Buffer allocation */
    if (buffer && buffer_size > 0) {
        llm->buffer = (uint8_t *)buffer;
        llm->buffer_size = buffer_size;
        llm->owns_buffer = false;
    } else {
        llm->buffer_size = eif_llm_compute_buffer_size(model_path);
        llm->buffer = (uint8_t *)malloc(llm->buffer_size);
        if (!llm->buffer) {
            fprintf(stderr, "[EIF LLM Error] Failed to allocate %zu bytes\n", llm->buffer_size);
            return -3;
        }
        llm->owns_buffer = true;
    }

    /* Architecture-specific backend initialization */
    if (llm->arch == EIF_LLM_ARCH_BERT) {
        if (eif_bert_load(&llm->backend.bert, model_path) != 0) {
            fprintf(stderr, "[EIF LLM Error] Failed to load BERT model: %s\n", model_path);
            if (llm->owns_buffer && llm->buffer) free(llm->buffer);
            return -4;
        }
        llm->dim = llm->backend.bert.config.dim;
        llm->hidden_dim = llm->backend.bert.config.intermediate_dim;
        llm->n_layers = llm->backend.bert.config.n_layers;
        llm->n_heads = llm->backend.bert.config.n_heads;
        llm->n_kv_heads = llm->backend.bert.config.n_heads;
        llm->vocab_size = llm->backend.bert.config.vocab_size;
        llm->seq_len = llm->backend.bert.config.max_seq_len;
        llm->tokenizer_loaded = (llm->backend.bert.vocab.tokens != NULL);
        llm->is_initialized = true;
        return 0;
    } else if (llm->arch == EIF_LLM_ARCH_QWEN35) {
        size_t state_sz = 128 * 1024 * 1024;
        if (qwen35_load(&llm->backend.qwen35, model_path, llm->buffer, state_sz) != 0) {
            fprintf(stderr, "[EIF LLM Error] Failed to load Qwen3.5: %s\n", model_path);
            if (llm->owns_buffer) free(llm->buffer);
            return -4;
        }
        llm->dim = llm->backend.qwen35.config.dim;
        llm->hidden_dim = llm->backend.qwen35.config.hidden_dim;
        llm->n_layers = llm->backend.qwen35.config.n_layers;
        llm->n_heads = llm->backend.qwen35.config.n_heads;
        llm->n_kv_heads = llm->backend.qwen35.config.n_kv_heads;
        llm->vocab_size = llm->backend.qwen35.config.vocab_size;
        llm->seq_len = llm->backend.qwen35.config.max_seq_len;
    } else {
        /* LLaMA / SmolLM2 / Granite-Docling (eif_tinyllm backend) */
        FILE *f = fopen(model_path, "rb");
        if (!f) {
            if (llm->owns_buffer) free(llm->buffer);
            return -5;
        }

        uint32_t magic = 0;
        if (fread(&magic, sizeof(uint32_t), 1, f) != 1) {
            fclose(f);
            if (llm->owns_buffer) free(llm->buffer);
            return -5;
        }

        tinyllm_config_t cfg;
        memset(&cfg, 0, sizeof(cfg));

        if (magic == 0x46554747) {
            /* Native GGUF Causal Model */
            fclose(f);
            char arch_str[64] = {0};
            if (tinyllm_read_gguf_config(model_path, &cfg, arch_str, sizeof(arch_str)) != 0) {
                fprintf(stderr, "[EIF LLM Error] Failed to read GGUF config from %s\n", model_path);
                if (llm->owns_buffer) free(llm->buffer);
                return -6;
            }
        } else if (magic == 0x53564C4D) {
            /* SVLM format: 48 bytes header + 128 bytes config (32 x int32) */
            int32_t cfg_ints[32];
            fseek(f, 48, SEEK_SET);
            if (fread(cfg_ints, sizeof(int32_t), 32, f) != 32) {
                fclose(f);
                if (llm->owns_buffer) free(llm->buffer);
                return -6;
            }
            fclose(f);
            cfg.dim = cfg_ints[0];
            cfg.hidden_dim = cfg_ints[1];
            cfg.n_layers = cfg_ints[2];
            cfg.n_heads = cfg_ints[3];
            cfg.n_kv_heads = cfg_ints[4];
            cfg.head_dim = cfg_ints[5];
            cfg.vocab_size = cfg_ints[6];
            cfg.seq_len = cfg_ints[7] > 0 ? cfg_ints[7] : 512;
            cfg.rope_theta = 100000.0f;
            cfg.qtype = (cfg_ints[12] == 3) ? TINYLLM_QTYPE_BITNET_158 : (tinyllm_qtype_t)cfg_ints[12];
            /* [13] vision qtype; [14] BitNet max scale group (0 = per row); [15] Hadamard-rotated weights */
            cfg.bitnet_group = cfg_ints[14];
            cfg.bitnet_hadamard = cfg_ints[15];
        } else {
            /* TTML / TLLM format: 36 bytes header + tinyllm_config_t */
            fseek(f, 36, SEEK_SET);
            /* Files predate the BitNet group/Hadamard fields (not supported by TTML): read up to them only. */
            if (fread(&cfg, offsetof(tinyllm_config_t, bitnet_group), 1, f) != 1) {
                fclose(f);
                if (llm->owns_buffer) free(llm->buffer);
                return -6;
            }
            fclose(f);
        }

        size_t mem_size = tinyllm_memory_size(&cfg);

        /* Ensure buffer has enough space for BOTH runtime state (KV cache/activations)
         * AND model weights. If owns_buffer is true, dynamically expand if needed. */
        struct stat st_f;
        size_t f_size = 0;
        if (stat(model_path, &st_f) == 0 && st_f.st_size > 0) {
            f_size = (size_t)st_f.st_size;
        }
        size_t weight_estimate = (magic == 0x46554747 && cfg.qtype == TINYLLM_QTYPE_INT8) ? (size_t)(f_size * 2) : f_size;
        size_t required_total = mem_size + weight_estimate + (128 * 1024 * 1024);
        if (llm->owns_buffer && llm->buffer_size < required_total) {
            uint8_t *new_buf = (uint8_t *)realloc(llm->buffer, required_total);
            if (new_buf) {
                llm->buffer = new_buf;
                llm->buffer_size = required_total;
            }
        }

        if (tinyllm_init(&llm->backend.tinyllm, &cfg, llm->buffer, mem_size) != 0) {
            fprintf(stderr, "[EIF LLM Error] Failed to init TinyLLM state\n");
            if (llm->owns_buffer) free(llm->buffer);
            return -7;
        }

        uint8_t *weights_buf = llm->buffer + mem_size;
        size_t weights_buf_size = llm->buffer_size - mem_size;
        if (magic == 0x46554747) {
            if (tinyllm_load_gguf(&llm->backend.tinyllm, model_path, weights_buf, weights_buf_size) != 0) {
                fprintf(stderr, "[EIF LLM Error] Failed to load TinyLLM GGUF weights\n");
                if (llm->owns_buffer) free(llm->buffer);
                return -8;
            }
        } else {
            if (tinyllm_load_quantized_model(&llm->backend.tinyllm, model_path, weights_buf, weights_buf_size) != 0) {
                fprintf(stderr, "[EIF LLM Error] Failed to load TinyLLM quantized weights\n");
                if (llm->owns_buffer) free(llm->buffer);
                return -8;
            }
        }

        llm->dim = cfg.dim;
        llm->hidden_dim = cfg.hidden_dim;
        llm->n_layers = cfg.n_layers;
        llm->n_heads = cfg.n_heads;
        llm->n_kv_heads = cfg.n_kv_heads;
        llm->vocab_size = cfg.vocab_size;
        llm->seq_len = cfg.seq_len;
    }

    /* Tokenizer resolution & loading */
    char resolved_tok[512] = {0};
    if (tokenizer_path && access(tokenizer_path, F_OK) == 0) {
        snprintf(resolved_tok, sizeof(resolved_tok), "%s", tokenizer_path);
    } else {
        find_tokenizer_path(model_path, resolved_tok, sizeof(resolved_tok));
    }

    if (strlen(resolved_tok) > 0) {
        snprintf(llm->tokenizer_path, sizeof(llm->tokenizer_path), "%s", resolved_tok);
        FILE *tf = fopen(resolved_tok, "rb");
        uint32_t tmagic = 0;
        if (tf) {
            if (fread(&tmagic, sizeof(uint32_t), 1, tf) != 1) tmagic = 0;
            fclose(tf);
        }
        if (tmagic == 0x46554747) {
            if (tinyllm_load_gguf_tokenizer(resolved_tok, &llm->tokenizer) == 0) {
                llm->tokenizer_loaded = true;
            } else {
                fprintf(stderr, "[EIF LLM Warning] GGUF Tokenizer failed to load from %s\n", resolved_tok);
            }
        } else {
            if (eif_bpe_tokenizer_load(&llm->tokenizer, resolved_tok, llm->vocab_size) == EIF_STATUS_OK) {
                llm->tokenizer_loaded = true;
            } else {
                fprintf(stderr, "[EIF LLM Warning] Tokenizer failed to load from %s\n", resolved_tok);
            }
        }
    } else {
        /* Check if model itself has embedded GGUF tokenizer */
        FILE *mf = fopen(model_path, "rb");
        if (mf) {
            uint32_t mmagic = 0;
            if (fread(&mmagic, 4, 1, mf) == 1 && mmagic == 0x46554747) {
                if (tinyllm_load_gguf_tokenizer(model_path, &llm->tokenizer) == 0) {
                    llm->tokenizer_loaded = true;
                    snprintf(llm->tokenizer_path, sizeof(llm->tokenizer_path), "%s (embedded)", model_path);
                }
            }
            fclose(mf);
        }
    }

    llm->current_pos = 0;
    llm->is_initialized = true;
    return 0;
}

int eif_llm_forward(eif_llm_t *llm, int token, int pos)
{
    if (!llm || !llm->is_initialized) return -1;

    float *logits = NULL;
    if (llm->arch == EIF_LLM_ARCH_QWEN35) {
        logits = qwen35_forward(&llm->backend.qwen35, token, pos);
    } else {
        logits = tinyllm_forward(&llm->backend.tinyllm, token, pos);
    }

    llm->current_pos = pos + 1;
    return (logits != NULL) ? 0 : -2;
}

int eif_llm_forward_no_logits(eif_llm_t *llm, int token, int pos)
{
    if (!llm || !llm->is_initialized) return -1;

    float *hidden = NULL;
    if (llm->arch == EIF_LLM_ARCH_QWEN35) {
        hidden = qwen35_forward_no_logits(&llm->backend.qwen35, token, pos);
    } else {
        hidden = tinyllm_forward_no_logits(&llm->backend.tinyllm, token, pos);
    }

    llm->current_pos = pos + 1;
    return (hidden != NULL) ? 0 : -2;
}

float* eif_llm_get_logits(eif_llm_t *llm)
{
    if (!llm || !llm->is_initialized) return NULL;
    if (llm->arch == EIF_LLM_ARCH_QWEN35) {
        return llm->backend.qwen35.state.logits;
    } else {
        return llm->backend.tinyllm.state.logits;
    }
}

typedef struct {
    int id;
    float prob;
} eif_token_prob_t;

static int eif_compare_token_prob_desc(const void *a, const void *b) {
    const eif_token_prob_t *ta = (const eif_token_prob_t *)a;
    const eif_token_prob_t *tb = (const eif_token_prob_t *)b;
    if (tb->prob > ta->prob) return 1;
    if (tb->prob < ta->prob) return -1;
    return 0;
}

int eif_llm_sample(eif_llm_t *llm, float temperature, float top_p)
{
    float *logits = eif_llm_get_logits(llm);
    if (!logits) return 0;
    int vocab_size = llm->vocab_size;

    /* Greedy argmax */
    if (temperature <= 0.0001f) {
        int best_id = 0;
        float best_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_id = i;
            }
        }
        return best_id;
    }

    /* Find maximum logit */
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    /* Collect top candidates within 12*temperature of max_logit */
    float threshold = max_logit - 12.0f * temperature;
    #define MAX_SAMPLE_CANDS 256
    eif_token_prob_t candidates[MAX_SAMPLE_CANDS];
    int n_cands = 0;

    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] >= threshold) {
            float p = expf((logits[i] - max_logit) / temperature);
            if (n_cands < MAX_SAMPLE_CANDS) {
                candidates[n_cands].id = i;
                candidates[n_cands].prob = p;
                n_cands++;
            } else {
                int min_k = 0;
                for (int k = 1; k < MAX_SAMPLE_CANDS; k++) {
                    if (candidates[k].prob < candidates[min_k].prob) min_k = k;
                }
                if (p > candidates[min_k].prob) {
                    candidates[min_k].id = i;
                    candidates[min_k].prob = p;
                }
            }
        }
    }

    if (n_cands == 0) {
        int best_id = 0;
        float best_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_id = i;
            }
        }
        return best_id;
    }

    /* Sort candidates descending by probability */
    qsort(candidates, n_cands, sizeof(eif_token_prob_t), eif_compare_token_prob_desc);

    float total_cand_prob = 0.0f;
    for (int i = 0; i < n_cands; i++) {
        total_cand_prob += candidates[i].prob;
    }

    /* Nucleus cutoff (Top-P) */
    float p_limit = (top_p > 0.0f && top_p <= 1.0f) ? top_p : 1.0f;
    float cdf = 0.0f;
    int cutoff = n_cands;
    for (int i = 0; i < n_cands; i++) {
        cdf += candidates[i].prob;
        if (cdf >= p_limit * total_cand_prob) {
            cutoff = i + 1;
            break;
        }
    }

    /* Draw random number within the nucleus mass */
    float coin = (((float)rand()) / (float)RAND_MAX) * cdf;
    float acc = 0.0f;
    for (int i = 0; i < cutoff; i++) {
        acc += candidates[i].prob;
        if (coin <= acc) {
            return candidates[i].id;
        }
    }
    return candidates[0].id;
    #undef MAX_SAMPLE_CANDS
}

int eif_llm_generate(eif_llm_t *llm, const char *prompt, const eif_llm_gen_config_t *cfg,
                     eif_llm_token_callback_fn cb, void *user_data)
{
    if (!llm || !llm->is_initialized || !prompt) return -1;
    if (!llm->tokenizer_loaded) {
        fprintf(stderr, "[EIF LLM Error] Tokenizer is required for generate()\n");
        return -2;
    }

    eif_llm_gen_config_t conf;
    memset(&conf, 0, sizeof(conf));
    if (cfg) {
        conf = *cfg;
    } else {
        conf.max_new_tokens = 64;
        conf.temperature = 0.7f;
        conf.top_p = 0.9f;
        conf.repetition_penalty = 1.15f;
        conf.eos_token_id = llm->tokenizer.eos_token_id;
        conf.kv_type = 0;
        conf.use_gpu = false;
    }
    if (conf.repetition_penalty <= 0.0f) {
        conf.repetition_penalty = 1.0f;
    }

    if (llm->arch == EIF_LLM_ARCH_SMOLLM2 || llm->arch == EIF_LLM_ARCH_QWEN2) {
        llm->backend.tinyllm.config.kv_type = conf.kv_type;
    }

    /* Reset internal model state (KV cache, recurrent SSM state) */
    eif_llm_reset(llm);

    /* Tokenize prompt */
    int32_t prompt_tokens[2048];
    int n_prompt = eif_bpe_tokenizer_encode(&llm->tokenizer, prompt, 0, 0, prompt_tokens, 2048);
    if (n_prompt <= 0) return -3;
    fprintf(stderr, "[Tokenizer Debug] Prompt tokens (%d): [", n_prompt);
    for (int i = 0; i < n_prompt; i++) {
        fprintf(stderr, "%d%s", prompt_tokens[i], (i < n_prompt - 1) ? ", " : "");
    }
    fprintf(stderr, "]\n");

    /* Prefill phase */
    int current_token = prompt_tokens[0];
    for (int i = 0; i < n_prompt; i++) {
        current_token = prompt_tokens[i];
        if (eif_llm_forward(llm, current_token, i) != 0) {
            return -4;
        }
    }

    int32_t history_tokens[4096];
    int total_history = 0;
    for (int i = 0; i < n_prompt && total_history < 4096; i++) {
        history_tokens[total_history++] = prompt_tokens[i];
    }

    int generated_count = 0;
    int pos = n_prompt;

    /* Autoregressive generation phase */
    for (int g = 0; g < conf.max_new_tokens; g++) {
        /* Apply repetition penalty to logits of previously seen tokens */
        if (conf.repetition_penalty > 1.0f && total_history > 0) {
            float *logits = eif_llm_get_logits(llm);
            if (logits) {
                for (int h = 0; h < total_history; h++) {
                    int tid = history_tokens[h];
                    if (tid >= 0 && tid < llm->vocab_size) {
                        if (logits[tid] > 0.0f) {
                            logits[tid] /= conf.repetition_penalty;
                        } else {
                            logits[tid] *= conf.repetition_penalty;
                        }
                    }
                }
            }
        }

        int next_token = eif_llm_sample(llm, conf.temperature, conf.top_p);

        if (total_history < 4096) {
            history_tokens[total_history++] = next_token;
        }

        if (next_token == conf.eos_token_id || next_token == llm->tokenizer.eos_token_id ||
            next_token == 248046 || next_token == 248044 || next_token == 2) {
            break;
        }

        const char *piece = eif_bpe_tokenizer_decode(&llm->tokenizer, next_token);
        if (cb) {
            cb(piece, next_token, user_data);
        } else {
            /* Default stdout streaming */
            if (piece) {
                /* Strip BPE prefix symbol Ġ if present */
                if ((unsigned char)piece[0] == 0xC4 && (unsigned char)piece[1] == 0xA0) {
                    putchar(' ');
                    fputs(piece + 2, stdout);
                } else if ((unsigned char)piece[0] == 0xC4 && (unsigned char)piece[1] == 0x8A) {
                    putchar('\n');
                    fputs(piece + 2, stdout);
                } else {
                    fputs(piece, stdout);
                }
                fflush(stdout);
            }
        }

        generated_count++;
        if (eif_llm_forward(llm, next_token, pos++) != 0) {
            break;
        }
    }

    return generated_count;
}

int eif_llm_repetition_span(const int32_t *toks, int n, int max_period)
{
    if (!toks || n <= 0) return 0;
    int best = 0;
    const int scan = n < 1024 ? n : 1024; /* only the recent tail matters */
    const int32_t *t = toks + (n - scan);
    for (int p = 1; p <= max_period && 4 * p <= scan; p++) {
        int run = 0; /* tokens (from the end) equal to the token one period earlier */
        for (int i = scan - 1; i >= p && t[i] == t[i - p]; i--) run++;
        int span = run + p;
        if (span >= 4 * p && span > best) best = span;
    }
    return best;
}

int eif_llm_encode_vision_docling(const eif_llm_t *llm, const char *image_path, eif_llm_vision_input_t *out)
{
    if (!llm || !llm->is_initialized || !image_path || !out) return -1;
    memset(out, 0, sizeof(*out));

    /* 1. Try precomputed visual embeddings (.vis.bin), or encode directly via native SigLIP2 */
    float *vis_embeds = NULL;
    int num_vis_tokens = 0;
    int grid_rows = 4, grid_cols = 3; /* precomputed .vis.bin files use the legacy 4 x 3 + global layout */

    char vis_path[2048];
    snprintf(vis_path, sizeof(vis_path), "%s.vis.bin", image_path);
    FILE *fv = fopen(vis_path, "rb");
    if (!fv) {
        char base[1024];
        strncpy(base, image_path, sizeof(base) - 1);
        base[sizeof(base) - 1] = '\0';
        char *dot = strrchr(base, '.');
        if (dot) {
            *dot = '\0';
            snprintf(vis_path, sizeof(vis_path), "%s.vis.bin", base);
            fv = fopen(vis_path, "rb");
        }
    }

    if (fv) {
        fseek(fv, 0, SEEK_END);
        long fsz = ftell(fv);
        fseek(fv, 0, SEEK_SET);
        num_vis_tokens = (int)(fsz / (576 * sizeof(float)));
        vis_embeds = (float *)malloc(fsz);
        if (vis_embeds && fread(vis_embeds, 1, fsz, fv) == (size_t)fsz) {
            fprintf(stderr, "  • Vision Source: Precomputed SigLIP2 embeddings (%d tokens, dim=576)\n", num_vis_tokens);
        } else {
            if (vis_embeds) free(vis_embeds);
            vis_embeds = NULL;
        }
        fclose(fv);
    } else {
        /* On-the-fly native C99 SigLIP2 Vision Encoder */
        FILE *fm = fopen(llm->model_path, "rb");
        if (fm) {
            uint32_t svlm_hdr[12];
            if (fread(svlm_hdr, sizeof(uint32_t), 12, fm) == 12 && svlm_hdr[0] == 0x53564C4D) {
                uint32_t v_off = svlm_hdr[6];
                uint32_t v_sz = svlm_hdr[10];
                if (v_sz > 0) {
                    eif_siglip_t siglip;
                    struct timespec t_v0, t_v1;
                    clock_gettime(CLOCK_MONOTONIC, &t_v0);
                    if (eif_siglip_load_from_file(&siglip, llm->model_path, v_off, v_sz) == 0) {
                        if (llm->gpu.is_available) eif_siglip_use_gpu(&siglip, (eif_gpu_context_t *)&llm->gpu);
                        clock_gettime(CLOCK_MONOTONIC, &t_v1);
                        double load_sec = (t_v1.tv_sec - t_v0.tv_sec) + (t_v1.tv_nsec - t_v0.tv_nsec) * 1e-9;
                        fprintf(stderr, "  • Vision Model Loaded in %.2fs\n", load_sec);
                        fflush(stderr);

                        clock_gettime(CLOCK_MONOTONIC, &t_v0);
                        vis_embeds = (float *)malloc((size_t)EIF_SIGLIP_MAX_TILES * 64 * 576 * sizeof(float));
                        if (vis_embeds && eif_siglip_encode_image(&siglip, image_path, vis_embeds, &grid_rows, &grid_cols) == 0) {
                            clock_gettime(CLOCK_MONOTONIC, &t_v1);
                            double enc_sec = (t_v1.tv_sec - t_v0.tv_sec) + (t_v1.tv_nsec - t_v0.tv_nsec) * 1e-9;
                            int n_tiles = grid_rows * grid_cols + 1;
                            fprintf(stderr, "  • Vision ViT Encode Complete in %.2fs (%.2fs / tile)\n", enc_sec, enc_sec / n_tiles);
                            fflush(stderr);
                            num_vis_tokens = n_tiles * 64;
                        } else {
                            if (vis_embeds) { free(vis_embeds); vis_embeds = NULL; }
                        }
                        eif_siglip_free(&siglip);
                    }
                }
            }
            fclose(fm);
        }
    }

    if (!vis_embeds || num_vis_tokens != (grid_rows * grid_cols + 1) * 64 || grid_rows < 1 || grid_rows > 4 ||
        grid_cols < 1 || grid_cols > 4) {
        fprintf(stderr, "[EIF LLM Error] Visual embeddings could not be generated or loaded for '%s'\n", image_path);
        if (vis_embeds) free(vis_embeds);
        return -3;
    }

    out->embeds = vis_embeds;
    out->n_tokens = num_vis_tokens;
    out->rows = grid_rows;
    out->cols = grid_cols;
    return 0;
}

void eif_llm_vision_input_free(eif_llm_vision_input_t *v)
{
    if (!v) return;
    free(v->embeds);
    memset(v, 0, sizeof(*v));
}

int eif_llm_generate_vision_docling_from(eif_llm_t *llm, const eif_llm_vision_input_t *vin, const char *prompt,
                                         const eif_llm_gen_config_t *cfg,
                                         eif_llm_token_callback_fn cb, void *user_data)
{
    if (!llm || !llm->is_initialized || !vin || !vin->embeds) return -1;
    if (!llm->tokenizer_loaded) {
        fprintf(stderr, "[EIF LLM Error] Tokenizer is required for generate()\n");
        return -2;
    }
    const float *vis_embeds = vin->embeds;
    const int grid_rows = vin->rows, grid_cols = vin->cols;
    if (grid_rows < 1 || grid_rows > 4 || grid_cols < 1 || grid_cols > 4 ||
        vin->n_tokens != (grid_rows * grid_cols + 1) * 64)
        return -3;

    eif_llm_gen_config_t conf;
    memset(&conf, 0, sizeof(conf));
    if (cfg) {
        conf = *cfg;
    } else {
        conf.max_new_tokens = 256;
        conf.temperature = 0.0f;
        conf.top_p = 0.9f;
        conf.repetition_penalty = 1.15f;
        conf.eos_token_id = 100257;
        conf.kv_type = 0;
        conf.use_gpu = false;
    }
    if (conf.repetition_penalty <= 0.0f) conf.repetition_penalty = 1.15f;

    /* Reset internal model state */
    eif_llm_reset(llm);

    int pos = 0;
    int32_t history_tokens[4096];
    int total_history = 0;
    int dim = llm->backend.tinyllm.config.dim;

    /* Assemble all prompt tokens and visual embeddings into a single batch */
    float *batch_embeddings = (float *)malloc((size_t)4096 * dim * sizeof(float));
    if (!batch_embeddings) {
        return -1;
    }
    int total_prompt_tokens = 0;

    /* 1. Prefix: <|start_of_role|>user<|end_of_role|> */
    int prefix_tokens[] = {100264, 882, 100265};
    for (int i = 0; i < 3; i++) {
        tinyllm_get_token_embedding(&llm->backend.tinyllm, prefix_tokens[i], batch_embeddings + (size_t)total_prompt_tokens * dim);
        if (total_history < 4096) history_tokens[total_history++] = prefix_tokens[i];
        total_prompt_tokens++;
    }

    /* 2. Local tiles, row-major: <fake_token_around_image><row_r_col_c> + 64 image tokens, "\n" after each row */
    static const int row_col_tokens[4][4] = {
        {100258, 100259, 100261, 100262},
        {100263, 100267, 100268, 100341},
        {100342, 100343, 100344, 100345},
        {100346, 100347, 100348, 100349}
    };

    int tile_idx = 0;
    for (int r = 0; r < grid_rows; r++) {
        for (int c = 0; c < grid_cols; c++) {
            tinyllm_get_token_embedding(&llm->backend.tinyllm, 100339, batch_embeddings + (size_t)total_prompt_tokens * dim);
            total_prompt_tokens++;

            tinyllm_get_token_embedding(&llm->backend.tinyllm, row_col_tokens[r][c], batch_embeddings + (size_t)total_prompt_tokens * dim);
            total_prompt_tokens++;

            const float *emb_tile = vis_embeds + (size_t)tile_idx * 64 * 576;
            memcpy(batch_embeddings + (size_t)total_prompt_tokens * dim, emb_tile, (size_t)64 * 576 * sizeof(float));
            total_prompt_tokens += 64;

            tile_idx++;
        }
        if (r < grid_rows - 1) {
            tinyllm_get_token_embedding(&llm->backend.tinyllm, 198, batch_embeddings + (size_t)total_prompt_tokens * dim);
            total_prompt_tokens++;
        }
    }

    /* 3. Global tile */
    tinyllm_get_token_embedding(&llm->backend.tinyllm, 271, batch_embeddings + (size_t)total_prompt_tokens * dim);
    total_prompt_tokens++;

    tinyllm_get_token_embedding(&llm->backend.tinyllm, 100339, batch_embeddings + (size_t)total_prompt_tokens * dim);
    total_prompt_tokens++;

    tinyllm_get_token_embedding(&llm->backend.tinyllm, 100340, batch_embeddings + (size_t)total_prompt_tokens * dim);
    total_prompt_tokens++;

    const float *emb_global = vis_embeds + (size_t)(grid_rows * grid_cols) * 64 * 576;
    memcpy(batch_embeddings + (size_t)total_prompt_tokens * dim, emb_global, (size_t)64 * 576 * sizeof(float));
    total_prompt_tokens += 64;

    tinyllm_get_token_embedding(&llm->backend.tinyllm, 100339, batch_embeddings + (size_t)total_prompt_tokens * dim);
    total_prompt_tokens++;

    /* 4. Suffix: <prompt text><|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>
     * The prompt selects the task (e.g. "Convert this page to docling.", "Convert table to OTSL.",
     * "Convert chart to table."); an empty prompt falls back to the page prompt. */
    int32_t suffix_tokens[1024];
    int n_suffix = 0;
    const char *task_prompt = (prompt && prompt[0]) ? prompt : "Convert this page to docling.";
    int n_prompt = eif_bpe_tokenizer_encode(&llm->tokenizer, task_prompt, 0, 0, suffix_tokens, 1024 - 8);
    if (n_prompt <= 0) {
        fprintf(stderr, "[EIF LLM Error] Could not tokenize prompt '%s'\n", task_prompt);
        free(batch_embeddings);
        return -4;
    }
    n_suffix = n_prompt;
    static const int32_t suffix_tail[] = {100257, 198, 100264, 78191, 100265};
    for (size_t i = 0; i < sizeof(suffix_tail) / sizeof(suffix_tail[0]); i++) suffix_tokens[n_suffix++] = suffix_tail[i];
    for (int i = 0; i < n_suffix; i++) {
        tinyllm_get_token_embedding(&llm->backend.tinyllm, suffix_tokens[i], batch_embeddings + (size_t)total_prompt_tokens * dim);
        if (total_history < 4096) history_tokens[total_history++] = suffix_tokens[i];
        total_prompt_tokens++;
    }

    fprintf(stderr, "  • LLM Batched Prefill: Processing %d prompt tokens layer-by-layer across %d layers...\n",
           total_prompt_tokens, llm->backend.tinyllm.config.n_layers);
    fflush(stderr);

    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    tinyllm_forward_batch_embeddings(&llm->backend.tinyllm, batch_embeddings, total_prompt_tokens, 0, true);
    pos = total_prompt_tokens;

    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double prefill_sec = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
    fprintf(stderr, "  • LLM Prefill Complete: %d tokens in %.2fs (%.1f tok/s)\n",
           total_prompt_tokens, prefill_sec, (prefill_sec > 0) ? (total_prompt_tokens / prefill_sec) : 0.0);
    fflush(stderr);

    free(batch_embeddings);

    /* 5. Autoregressive generation phase with repetition penalty */
    int generated_count = 0;
    static _Thread_local int32_t gen_tokens[8192];
    int n_gen = 0;
    const int loop_guard = conf.loop_guard_tokens == 0 ? 64 : conf.loop_guard_tokens;
    for (int g = 0; g < conf.max_new_tokens; g++) {
        float *logits = eif_llm_get_logits(llm);
        if (conf.repetition_penalty > 1.0f && total_history > 0 && logits) {
            for (int h = 0; h < total_history; h++) {
                int tid = history_tokens[h];
                if (tid >= 0 && tid < llm->vocab_size) {
                    if (logits[tid] > 0.0f) {
                        logits[tid] /= conf.repetition_penalty;
                    } else {
                        logits[tid] *= conf.repetition_penalty;
                    }
                }
            }
        }

        int next_token = eif_llm_sample(llm, conf.temperature, conf.top_p);
        if (total_history < 4096) {
            history_tokens[total_history++] = next_token;
        }

        if (next_token == 100257 || next_token == 100265 || next_token == conf.eos_token_id) {
            break;
        }

        if (n_gen < (int)(sizeof(gen_tokens) / sizeof(gen_tokens[0]))) gen_tokens[n_gen++] = next_token;
        if (loop_guard > 0 && n_gen >= loop_guard) {
            int span = eif_llm_repetition_span(gen_tokens, n_gen, 64);
            if (span >= loop_guard) {
                fprintf(stderr, "  • Loop guard: last %d tokens repeat one pattern, stopping after %d tokens\n", span, n_gen);
                fflush(stderr);
                break;
            }
        }

        const char *piece = eif_bpe_tokenizer_decode(&llm->tokenizer, next_token);
        if (cb) {
            cb(piece, next_token, user_data);
        } else {
            if (piece) {
                fputs(piece, stdout);
                fflush(stdout);
            }
        }
        generated_count++;

        tinyllm_forward(&llm->backend.tinyllm, next_token, pos++);
    }

    return generated_count;
}

int eif_llm_generate_vision_docling(eif_llm_t *llm, const char *image_path, const char *prompt,
                                    const eif_llm_gen_config_t *cfg,
                                    eif_llm_token_callback_fn cb, void *user_data)
{
    if (!llm || !llm->is_initialized || !image_path) return -1;
    if (!llm->tokenizer_loaded) {
        fprintf(stderr, "[EIF LLM Error] Tokenizer is required for generate()\n");
        return -2;
    }
    eif_llm_vision_input_t vin;
    int rc = eif_llm_encode_vision_docling(llm, image_path, &vin);
    if (rc != 0) return rc;
    rc = eif_llm_generate_vision_docling_from(llm, &vin, prompt, cfg, cb, user_data);
    eif_llm_vision_input_free(&vin);
    return rc;
}

void eif_llm_reset(eif_llm_t *llm)
{
    if (!llm || !llm->is_initialized) return;

    if (llm->arch == EIF_LLM_ARCH_QWEN35) {
        qwen35_reset(&llm->backend.qwen35);
    } else {
        tinyllm_reset(&llm->backend.tinyllm);
    }
    llm->current_pos = 0;
}

void eif_llm_free(eif_llm_t *llm)
{
    if (!llm || !llm->is_initialized) return;

    if (llm->arch == EIF_LLM_ARCH_BERT) {
        eif_bert_free(&llm->backend.bert);
    } else if (llm->arch == EIF_LLM_ARCH_QWEN35) {
        qwen35_free(&llm->backend.qwen35);
    } else {
        tinyllm_free(&llm->backend.tinyllm);
    }

    if (llm->tokenizer_loaded && llm->arch != EIF_LLM_ARCH_BERT) {
        eif_bpe_tokenizer_free(&llm->tokenizer);
        llm->tokenizer_loaded = false;
    }

    if (llm->owns_buffer && llm->buffer) {
        free(llm->buffer);
        llm->buffer = NULL;
    }

    eif_gpu_cleanup(&llm->gpu);

    llm->is_initialized = false;
}

float *eif_llm_get_hidden_state(eif_llm_t *llm)
{
    if (!llm || !llm->is_initialized) return NULL;
    if (llm->arch == EIF_LLM_ARCH_BERT) {
        return llm->backend.bert.scratch_seq_x;
    } else if (llm->arch == EIF_LLM_ARCH_QWEN35) {
        return qwen35_get_hidden_state(&llm->backend.qwen35);
    } else {
        return tinyllm_get_hidden_state(&llm->backend.tinyllm);
    }
}

int eif_llm_embed(eif_llm_t *llm, const char *prompt, float *out_embedding, eif_llm_pool_mode_t pool_mode)
{
    if (!llm || !llm->is_initialized || !prompt || !out_embedding) return -1;

    if (llm->arch == EIF_LLM_ARCH_BERT) {
        return eif_bert_embed(&llm->backend.bert, prompt, out_embedding);
    }

    if (!llm->tokenizer_loaded) return -2;

    int32_t prompt_tokens[2048];
    int n_prompt = eif_bpe_tokenizer_encode(&llm->tokenizer, prompt, 1, 0, prompt_tokens, 2048);
    if (n_prompt <= 0) return -3;

    int dim = llm->dim;
    memset(out_embedding, 0, (size_t)dim * sizeof(float));

    eif_llm_reset(llm);

    for (int i = 0; i < n_prompt; i++) {
        int token = prompt_tokens[i];
        if (eif_llm_forward_no_logits(llm, token, i) != 0) {
            return -4;
        }

        float *hidden = eif_llm_get_hidden_state(llm);
        if (!hidden) return -5;

        if (pool_mode == EIF_LLM_POOL_MEAN) {
            /* Accumulate for mean pooling */
            for (int d = 0; d < dim; d++) {
                out_embedding[d] += hidden[d];
            }
        } else if (pool_mode == EIF_LLM_POOL_LAST) {
            /* Keep last token */
            memcpy(out_embedding, hidden, (size_t)dim * sizeof(float));
        } else if (pool_mode == EIF_LLM_POOL_CLS && i == 0) {
            /* First token / CLS */
            memcpy(out_embedding, hidden, (size_t)dim * sizeof(float));
        }
    }

    if (pool_mode == EIF_LLM_POOL_MEAN && n_prompt > 1) {
        float inv_n = 1.0f / (float)n_prompt;
        for (int d = 0; d < dim; d++) {
            out_embedding[d] *= inv_n;
        }
    }

    /* L2 Normalization */
    float sum_sq = 0.0f;
    for (int d = 0; d < dim; d++) {
        sum_sq += out_embedding[d] * out_embedding[d];
    }
    float norm = sqrtf(sum_sq);
    if (norm > 1e-12f) {
        float inv_norm = 1.0f / norm;
        for (int d = 0; d < dim; d++) {
            out_embedding[d] *= inv_norm;
        }
    }

    return 0;
}

float eif_llm_cosine_similarity(const float *a, const float *b, int dim)
{
    if (!a || !b || dim <= 0) return 0.0f;
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float denom = sqrtf(norm_a) * sqrtf(norm_b);
    return (denom > 1e-12f) ? (dot / denom) : 0.0f;
}
