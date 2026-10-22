/**
 * @file eif_bert.c
 * @brief High-Performance Cross-Platform (x86_64 AVX2 / ARM64 NEON) BERT Runtime
 */

#if defined(__linux__)
#define _GNU_SOURCE
#endif

#include "eif_bert_internal.h"
#include "eif_bert_tokenizer.h"
#include "eif_gguf.h"
#include "eif_bpe_tokenizer.h"
#include "eif_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>

/* Longest token sequence the scratch buffers can hold (never more than the model context). */
int bert_token_cap(const eif_bert_t *bert)
{
    int cap = bert->config.max_seq_len > 0 ? bert->config.max_seq_len : 512;
    if (bert->max_t_alloc > 0 && cap > bert->max_t_alloc) cap = bert->max_t_alloc;
    return cap;
}

/* =============================================================================
 * Model Loading (Zero-Copy mmap)
 * ============================================================================= */

static inline size_t align64(size_t sz)
{
    return (sz + 63) & ~63;
}


/* Common setup for vocab hash table and scratch memory */
static int bert_setup_vocab_and_scratch(eif_bert_t *bert)
{
    int dim = bert->config.dim;
    int inter_dim = bert->config.intermediate_dim;
    int max_t = bert->config.max_seq_len;

    /* Detect tokenizer type:
     * 1. BPE (RoBERTa / GPT-2 / Granite): presence of byte-level Ġ (\xc4\xa0) tokens
     * 2. SentencePiece (XLM-RoBERTa / Multilingual-E5): presence of \xe2\x96\x81 tokens
     * 3. WordPiece (Classic MiniLM / BERT): ## subwords
     */
    int bpe_prefix_count = 0;
    int spm_prefix_count = 0;
    for (int t = 0; t < bert->vocab.vocab_size && t < 5000; t++) {
        if (!bert->vocab.tokens[t]) continue;
        const uint8_t *tk = (const uint8_t *)bert->vocab.tokens[t];
        if (tk[0] == 0xC4 && tk[1] == 0xA0) {
            bpe_prefix_count++;
        } else if (tk[0] == 0xE2 && tk[1] == 0x96 && tk[2] == 0x81) {
            spm_prefix_count++;
        }
    }

    if (bpe_prefix_count > 50) {
        bert->vocab.tok_type = EBERT_TOKENIZER_BPE;
        bert->vocab.is_spm_prefix = false;
        eif_bpe_tokenizer_init_from_vocab(&bert->vocab.bpe_tok, bert->vocab.tokens, NULL, bert->vocab.vocab_size, bert->vocab.cls_id, bert->vocab.sep_id);
        bert->vocab.bpe_tok.pretok_gpt2 = true;
    } else if (spm_prefix_count > 50) {
        bert->vocab.tok_type = EBERT_TOKENIZER_SPM;
        bert->vocab.is_spm_prefix = true;
    } else {
        bert->vocab.tok_type = EBERT_TOKENIZER_WORDPIECE;
        bert->vocab.is_spm_prefix = false;
    }

    /* Build fast O(1) hash lookup table dynamically sized to prevent collisions */
    int hash_size = 131072;
    while (hash_size < bert->vocab.vocab_size * 2) {
        hash_size <<= 1;
    }
    bert->vocab.hash_size = hash_size;
    bert->vocab.hash_mask = (uint32_t)hash_size - 1;

    bert->vocab.hash_table = (int32_t *)malloc((size_t)hash_size * sizeof(int32_t));
    if (!bert->vocab.hash_table) return -7;
    for (int i = 0; i < hash_size; i++) bert->vocab.hash_table[i] = -1;
    for (int t = 0; t < bert->vocab.vocab_size; t++) {
        if (!bert->vocab.tokens[t]) continue;
        uint32_t idx = ebert_hash_str(bert->vocab.tokens[t]) & bert->vocab.hash_mask;
        for (int step = 0; step < 256; step++) {
            uint32_t slot = (idx + step) & bert->vocab.hash_mask;
            if (bert->vocab.hash_table[slot] == -1) {
                bert->vocab.hash_table[slot] = (int32_t)t;
                break;
            }
        }
    }

    /* Allocate Scratch Activation Memory */
    /* Backbone dim (gemma-embedding2 embeds 512-d activations, outputs 768) */
    int dim_a = bert->config.dim_model > 0 ? bert->config.dim_model : dim;
    /* Hard cap on activation length: published RoPE encoders may advertise huge
     * contexts (EmbeddingGemma-2: 262144) -> scratch would be terabytes. */
    if (max_t > 4096) max_t = 2048;
    bert->max_t_alloc = max_t;
    /* gemma-embedding2 global layers use up to 512-d heads: Q scratch widens */
    int q_dim = (bert->config.n_embd_per_layer > 0 && bert->weights.g2_max_q_dim > dim_a)
                    ? bert->weights.g2_max_q_dim : dim_a;
    int proj_dim = (dim > dim_a) ? dim : dim_a; /* output projection may widen */
    bert->scratch_seq_x  = (float *)malloc((size_t)max_t * dim_a * sizeof(float));
    bert->scratch_seq_xb = (float *)malloc((size_t)max_t * dim_a * sizeof(float));
    bert->scratch_q      = (float *)malloc((size_t)max_t * q_dim * sizeof(float));
    bert->scratch_k      = (float *)malloc((size_t)max_t * dim_a * sizeof(float));
    bert->scratch_v      = (float *)malloc((size_t)max_t * dim_a * sizeof(float));
    bert->scratch_att    = (float *)malloc((size_t)bert->config.n_heads * max_t * max_t * sizeof(float));
    /* Gated FFN (GEGLU) needs a second [T x inter] buffer for the gate branch */
    size_t inter_mult = bert->config.arch_modern ? 3 : (bert->weights.ffn_gate_w ? 2 : 1); /* modern: fused up [2 ff] + gated result [ff] */
    bert->scratch_inter  = (float *)malloc((size_t)max_t * inter_dim * inter_mult * sizeof(float));
    bert->scratch_proj   = (float *)malloc((size_t)max_t * proj_dim * sizeof(float));
    if (bert->config.n_embd_per_layer > 0) {
        int dim_pl = bert->config.dim_model > 0 ? bert->config.dim_model : dim;
        bert->scratch_pl = (float *)malloc((size_t)max_t * (size_t)dim_pl * (size_t)bert->config.n_layers * sizeof(float));
        bert->scratch_g2 = (float *)malloc((size_t)max_t * dim_pl * sizeof(float));
        bert->scratch_attout = (float *)malloc((size_t)max_t * (size_t)bert->weights.g2_max_q_dim * sizeof(float));
    } else {
        bert->scratch_pl = NULL;
        bert->scratch_g2 = NULL;
        bert->scratch_attout = NULL;
    }

    if (!bert->scratch_seq_x || !bert->scratch_seq_xb || !bert->scratch_q ||
        !bert->scratch_k || !bert->scratch_v || !bert->scratch_att ||
        !bert->scratch_inter || !bert->scratch_proj ||
        (bert->config.n_embd_per_layer > 0 && (!bert->scratch_pl || !bert->scratch_g2 || !bert->scratch_attout))) {
        return -8;
    }

    bert->is_initialized = true;
    return 0;
}

/* Generic GGUF metadata value skipper: advances *pp past a value of the given
 * type (including arrays of any element type). Returns 0 on success, -1 if
 * the type is unknown (caller treats it as a corrupt file). */
static int gguf_skip_value(const uint8_t **pp, uint32_t vtype)
{
    const uint8_t *p = *pp;
    switch (vtype) {
        case 0: case 1: case 7: p += 1; break;
        case 2: case 3: p += 2; break;
        case 4: case 5: case 6: p += 4; break;
        case 8: { uint64_t l = gguf_read_u64(&p); p += l; break; }
        case 10: case 11: case 12: p += 8; break;
        case 9: {
            uint32_t atype = gguf_read_u32(&p);
            uint64_t acount = gguf_read_u64(&p);
            if (atype == 8) {
                for (uint64_t k = 0; k < acount; k++) { uint64_t sl = gguf_read_u64(&p); p += sl; }
            } else if (atype == 0 || atype == 1 || atype == 7) p += acount;
            else if (atype == 2 || atype == 3) p += acount * 2;
            else if (atype == 4 || atype == 5 || atype == 6) p += acount * 4;
            else if (atype == 10 || atype == 11 || atype == 12) p += acount * 8;
            else return -1;
            break;
        }
        default: return -1;
    }
    *pp = p;
    return 0;
}

/* 1. Native GGUF BERT Loader */
static int eif_bert_load_gguf(eif_bert_t *bert, const uint8_t *addr, size_t file_sz)
{
    const uint8_t *p = addr;
    uint32_t magic = gguf_read_u32(&p);
    uint32_t ver   = gguf_read_u32(&p);
    uint64_t n_tensors = gguf_read_u64(&p);
    uint64_t n_kv      = gguf_read_u64(&p);

    if (magic != GGUF_MAGIC) return -5;
    (void)ver;
    (void)file_sz;

    uint32_t alignment = 32;
    int dim = 384, inter_dim = 1536, n_layers = 6, n_heads = 12, max_seq = 512;
    int n_kv_heads = 0, n_embd_pl = 0, dim_out = 0;
    float rope_freq = 0.0f, rope_freq_swa = 0.0f, rms_eps = 0.0f;
    int sliding_window = 0;
    int tok_model_bert = 0, tok_model_known = 0, tok_model_unigram = 0, tok_model_spm = 0, tok_model_gemma4 = 0, add_space_prefix = 1, pre_kind = 0, tok_model_gpt2 = 0, swa_pattern = 0;
    char hidden_act[16] = "";
    char arch_name[64] = {0};

    for (uint64_t i = 0; i < n_kv; i++) {
        char key[128];
        gguf_read_str(&p, key, sizeof(key));
        uint32_t vtype = gguf_read_u32(&p);

        if (strcmp(key, "general.architecture") == 0 && vtype == 8) {
            gguf_read_str(&p, arch_name, sizeof(arch_name));
            continue;
        } else if (strcmp(key, "tokenizer.ggml.model") == 0 && vtype == 8) {
            char tm[32];
            gguf_read_str(&p, tm, sizeof(tm));
            tok_model_bert = (strcmp(tm, "bert") == 0);
            tok_model_unigram = (strcmp(tm, "t5") == 0);
            tok_model_spm = (strcmp(tm, "llama") == 0 || strcmp(tm, "gemma4") == 0);
            tok_model_gemma4 = (strcmp(tm, "gemma4") == 0);
            tok_model_gpt2 = (strcmp(tm, "gpt2") == 0);
            tok_model_known = 1;
        } else if (strcmp(key, "tokenizer.ggml.pre") == 0 && vtype == 8) {
            char pre[64];
            gguf_read_str(&p, pre, sizeof(pre));
            pre_kind = strcmp(pre, "granite-embed-multi-97m") == 0 ? 2 : (strcmp(pre, "roberta-bpe") == 0 || strcmp(pre, "gpt-2") == 0 || strcmp(pre, "jina-v2-code") == 0 || strcmp(pre, "jina-v2-de") == 0 || strcmp(pre, "jina-v2-es") == 0 || strcmp(pre, "default") == 0) ? 1 : 0;
        } else if (strstr(key, "attention.sliding_window_pattern") && vtype == 4) {
            swa_pattern = (int)gguf_read_u32(&p);
        } else if (strstr(key, ".hidden_activation") && vtype == 8) {
            gguf_read_str(&p, hidden_act, sizeof(hidden_act));
        } else if (strcmp(key, "general.alignment") == 0) {
            alignment = gguf_read_u32(&p);
        } else if (strstr(key, ".embedding_length") && strcmp(strstr(key, ".embedding_length"), ".embedding_length") == 0 && vtype == 4) {
            /* exact-suffix match: must not swallow "..._per_layer_input" / "..._out" */
            dim = (int)gguf_read_u32(&p);
        } else if (strstr(key, ".embedding_length_per_layer_input") && vtype == 4) {
            n_embd_pl = (int)gguf_read_u32(&p);
        } else if (strstr(key, ".embedding_length_out") && vtype == 4) {
            dim_out = (int)gguf_read_u32(&p);
        } else if (strstr(key, "feed_forward_length") && !strstr(key, "expert") && vtype == 4) {
            inter_dim = (int)gguf_read_u32(&p);
        } else if (strstr(key, "block_count") && vtype == 4) {
            n_layers = (int)gguf_read_u32(&p);
        } else if (strstr(key, "head_count_kv")) {
            if (vtype == 4 || vtype == 5) n_kv_heads = (int)gguf_read_u32(&p);
            else if (vtype == 0 || vtype == 1 || vtype == 7) n_kv_heads = *p++;
            else if (gguf_skip_value(&p, vtype) != 0) return -6; /* array (gemma-embedding2): per-layer values derived from tensor dims */
        } else if (strstr(key, "head_count") && vtype == 4) {
            n_heads = (int)gguf_read_u32(&p);
        } else if (strstr(key, "attention.sliding_window") && !strstr(key, "pattern") && vtype == 4) {
            sliding_window = (int)gguf_read_u32(&p);
        } else if (strstr(key, "rope.freq_base_swa")) {
            if (vtype == 6) { float f; memcpy(&f, p, 4); p += 4; rope_freq_swa = f; } else if (gguf_skip_value(&p, vtype) != 0) return -6;
        } else if (strstr(key, ".rope.freq_base") && vtype == 6) {
            float f; memcpy(&f, p, 4); p += 4; rope_freq = f;
        } else if ((strstr(key, "rms_epsilon") || strstr(key, "norm_epsilon")) && vtype == 6) {
            float f; memcpy(&f, p, 4); p += 4; rms_eps = f;
        } else if (strstr(key, "context_length") && vtype == 4) {
            max_seq = (int)gguf_read_u32(&p);
        } else if (strstr(key, "attention.alibi")) {
            if (vtype == 7) {
                bert->config.use_alibi = (*p++ != 0);
            } else if (vtype == 4) {
                bert->config.use_alibi = (gguf_read_u32(&p) != 0);
            } else if (gguf_skip_value(&p, vtype) != 0) return -6;
        } else if (strstr(key, "pooling_type") && vtype == 4) {
            bert->config.pooling_type = (int)gguf_read_u32(&p);
        } else if (strcmp(key, "tokenizer.ggml.bos_token_id") == 0) {
            bert->vocab.cls_id = (int)gguf_read_u32(&p);
        } else if (strcmp(key, "tokenizer.ggml.eos_token_id") == 0) {
            bert->vocab.sep_id = (int)gguf_read_u32(&p);
        } else if (strcmp(key, "tokenizer.ggml.unknown_token_id") == 0) {
            bert->vocab.unk_id = (int)gguf_read_u32(&p);
        } else if (strcmp(key, "tokenizer.ggml.padding_token_id") == 0) {
            bert->vocab.pad_id = (int)gguf_read_u32(&p);
        } else if (strcmp(key, "tokenizer.ggml.add_space_prefix") == 0 && vtype == 7) {
            add_space_prefix = (*p++ != 0);
        } else if (strcmp(key, "tokenizer.ggml.merges") == 0 && vtype == 9) {
            uint32_t itype = gguf_read_u32(&p);
            uint64_t count = gguf_read_u64(&p);
            if (itype != 8) { fprintf(stderr, "eif_bert: tokenizer.ggml.merges is not a string array\n"); return -6; }
            bert->vocab.merge_strs = (char **)calloc((size_t)count, sizeof(char *));
            if (!bert->vocab.merge_strs) return -8;
            bert->vocab.n_merges = (int)count;
            for (uint64_t mi = 0; mi < count; mi++) {
                uint64_t slen = gguf_read_u64(&p);
                bert->vocab.merge_strs[mi] = (char *)malloc(slen + 1);
                if (!bert->vocab.merge_strs[mi]) return -8;
                memcpy(bert->vocab.merge_strs[mi], p, slen);
                bert->vocab.merge_strs[mi][slen] = '\0';
                p += slen;
            }
        } else if (strcmp(key, "tokenizer.ggml.scores") == 0 && vtype == 9) {
            uint32_t itype = gguf_read_u32(&p);
            uint64_t count = gguf_read_u64(&p);
            if (itype != 6) { fprintf(stderr, "eif_bert: tokenizer.ggml.scores is not float32\n"); return -6; }
            bert->vocab.scores = (float *)malloc((size_t)count * sizeof(float));
            if (!bert->vocab.scores) return -8;
            memcpy(bert->vocab.scores, p, (size_t)count * sizeof(float));
            p += count * 4;
        } else if (strcmp(key, "tokenizer.ggml.tokens") == 0) {
            if (vtype != 9) { fprintf(stderr, "eif_bert: tokenizer.ggml.tokens wrong type %u\n", vtype); return -6; }
            uint32_t itype = gguf_read_u32(&p);
            uint64_t count = gguf_read_u64(&p);
            (void)itype;

            bert->vocab.vocab_size = (int)count;
            bert->vocab.tokens = (char **)malloc((size_t)count * sizeof(char *));
            if (bert->vocab.cls_id < 0) bert->vocab.cls_id = 101;
            if (bert->vocab.sep_id < 0) bert->vocab.sep_id = 102;
            if (bert->vocab.unk_id < 0) bert->vocab.unk_id = 100;
            if (bert->vocab.pad_id < 0) bert->vocab.pad_id = 0;

            for (uint64_t t = 0; t < count; t++) {
                uint64_t slen = gguf_read_u64(&p);
                bert->vocab.tokens[t] = (char *)malloc(slen + 1);
                memcpy(bert->vocab.tokens[t], p, slen);
                bert->vocab.tokens[t][slen] = '\0';
                p += slen;

                if (strcmp(bert->vocab.tokens[t], "[CLS]") == 0 || strcmp(bert->vocab.tokens[t], "<s>") == 0) bert->vocab.cls_id = (int)t;
                else if (strcmp(bert->vocab.tokens[t], "[SEP]") == 0 || strcmp(bert->vocab.tokens[t], "</s>") == 0) bert->vocab.sep_id = (int)t;
                else if (strcmp(bert->vocab.tokens[t], "[UNK]") == 0 || strcmp(bert->vocab.tokens[t], "<unk>") == 0) bert->vocab.unk_id = (int)t;
                else if (strcmp(bert->vocab.tokens[t], "[PAD]") == 0 || strcmp(bert->vocab.tokens[t], "<pad>") == 0) bert->vocab.pad_id = (int)t;
            }
        } else {
            /* Skip metadata value */
            switch (vtype) {
                case 0: case 1: case 7: p += 1; break;
                case 2: case 3: p += 2; break;
                case 4: case 5: case 6: p += 4; break;
                case 8: { uint64_t l = gguf_read_u64(&p); p += l; break; }
                case 10: case 11: case 12: p += 8; break;
                case 9: {
                    uint32_t atype = gguf_read_u32(&p);
                    uint64_t acount = gguf_read_u64(&p);
                    if (atype == 8) {
                        for (uint64_t k = 0; k < acount; k++) { uint64_t sl = gguf_read_u64(&p); p += sl; }
                    } else if (atype == 0 || atype == 1 || atype == 7) p += acount;
                    else if (atype == 2 || atype == 3) p += acount * 2;
                    else if (atype == 4 || atype == 5 || atype == 6) p += acount * 4;
                    else if (atype == 10 || atype == 11 || atype == 12) p += acount * 8;
                    break;
                }
                default:
                    fprintf(stderr, "eif_bert: unsupported KV type %u for key '%s'\n", vtype, key);
                    return -6;
            }
        }
    }

    bert->config.dim = dim;
    bert->config.dim_model = 0;
    bert->config.intermediate_dim = inter_dim;
    bert->config.n_layers = n_layers;
    bert->config.n_heads = n_heads;
    bert->config.max_seq_len = max_seq;
    bert->config.n_kv_heads = n_kv_heads;
    bert->config.rope_freq_base = rope_freq > 0.0f ? rope_freq : 10000.0f;
    bert->config.rope_freq_base_swa = rope_freq_swa > 0.0f ? rope_freq_swa : 10000.0f;
    bert->config.sliding_window = sliding_window;
    /* llama.cpp lowercases for tokenizer.ggml.model == "bert" (uncased WordPiece, stored with a
     * "\xe2\x96\x81" word-start marker) and not for SentencePiece models; the SPM branch used to
     * skip lowercasing altogether, so every capitalised word of MiniLM/BGE/E5/Arctic became [UNK]. */
    bert->vocab.unigram = tok_model_unigram && bert->vocab.scores != NULL;
    bert->vocab.spm_bigram = tok_model_spm && bert->vocab.scores != NULL && !(tok_model_gemma4 && bert->vocab.n_merges > 0);
    bert->vocab.bpe_merges = tok_model_gemma4 && bert->vocab.n_merges > 0;
    bert->vocab.byte_bpe = (tok_model_gpt2 && bert->vocab.n_merges > 0) ? pre_kind : 0;
    if ((bert->vocab.bpe_merges || bert->vocab.byte_bpe) && ebert_build_merge_table(&bert->vocab) != 0) return -8;
    bert->vocab.add_space_prefix = add_space_prefix != 0;
    bert->vocab.lowercase = tok_model_known ? (tok_model_bert != 0) : true;
    { const char *kc = getenv("EIF_SPM_KEEP_CASE"); if (kc && kc[0] == '1') bert->vocab.lowercase = false; }
    bert->config.norm_eps = rms_eps > 0.0f ? rms_eps : 1e-12f;
    bert->config.head_dim = dim / n_heads;
    bert->config.att_type = 0;
    bert->config.norm_style = 0;
    bert->config.use_rope = false;
    bert->config.ffn_silu = false;
    if (strcmp(arch_name, "nomic-bert") == 0) { bert->config.use_rope = true; bert->config.ffn_silu = true; }
    bert->config.arch_modern = strcmp(arch_name, "modern-bert") == 0;
    bert->config.swa_pattern = swa_pattern > 0 ? swa_pattern : 3;
    if (bert->config.arch_modern) bert->config.ffn_silu = strcmp(hidden_act, "silu") == 0;
    bert->config.emb_scale = 0.0f;
    bert->config.attn_scale = 0.0f;
    bert->config.n_embd_per_layer = 0;

    if (strstr(arch_name, "gemma-embedding") != NULL) {
        /* Gemma-embedding family (EmbeddingGemma 1/2 text encoder):
         * bidirectional RoPE attention + GQA + QK-RMSNorm + gated-GELU FFN,
         * Gemma RMS * (1+w) normalization everywhere, mean pooling. */
        bert->config.att_type = 2;
        bert->config.norm_style = 1;
        bert->config.emb_scale = sqrtf((float)dim); /* gemma3: embd *= sqrt(n_embd) */
        {
            const char *env = getenv("EIF_GEMMA_SCALE");
            if (env && env[0] == '0') bert->config.emb_scale = 0.0f;
        }
    }
    if (strstr(arch_name, "gemma-embedding2") != NULL) {
        /* EmbeddingGemma-2 (Gemma4 backbone, 0.3B text): per-layer embeddings +
         * per-layer GQA head config + mixed rope bases + final 768-d projection.
         * QK-norm makes attention scaling unnecessary (attn_scale == 1.0). */
        bert->config.n_embd_per_layer = n_embd_pl > 0 ? n_embd_pl : dim;
        bert->config.dim_model = dim;
        /* config.dim becomes the OUTPUT embedding dim (embedding_length_out) */
        if (dim_out > 0 && dim_out != dim) dim = dim_out;
        bert->config.dim = dim;
        bert->config.attn_scale = 1.0f;
    }

    /* Parse tensor directory */
    gguf_tensor_desc_t *tdescs = (gguf_tensor_desc_t *)malloc((size_t)n_tensors * sizeof(gguf_tensor_desc_t));
    if (!tdescs) return -7;

    for (uint64_t t = 0; t < n_tensors; t++) {
        gguf_read_str(&p, tdescs[t].name, sizeof(tdescs[t].name));
        uint32_t ndims = gguf_read_u32(&p);
        tdescs[t].n_dims = ndims > 4 ? 4 : ndims;
        for (uint32_t d = 0; d < ndims; d++)
            tdescs[t].ne[d] = gguf_read_u64(&p);
        tdescs[t].type = gguf_read_u32(&p);
        tdescs[t].offset = gguf_read_u64(&p);
    }

    size_t header_len = (size_t)(p - addr);
    size_t data_start_offset = (header_len + alignment - 1) & ~(size_t)(alignment - 1);
    const uint8_t *data_start = addr + data_start_offset;

    uint32_t tok_type = 0;
    bert->weights.token_emb = find_gguf_tensor(tdescs, (int)n_tensors, data_start, "token_embd.weight", &tok_type);
    bert->config.qtype = (tok_type == 8) ? 2 : 0; /* 2 = Native GGUF Q8_0 interleaved */

    uint32_t pos_type = 0;
    const void *raw_pos = find_gguf_tensor(tdescs, (int)n_tensors, data_start, "position_embd.weight", &pos_type);
    /* Actual row count of the position table: never trust max_seq blindly —
     * several published GGUF encoders store fewer positional rows than
     * bert.context_length, and reading past the table produces NaN. */
    bert->weights.pos_rows = max_seq;
    for (int ti = 0; ti < (int)n_tensors; ti++) {
        if (strcmp(tdescs[ti].name, "position_embd.weight") == 0) {
            /* GGUF ne[0] is the hidden width, ne[1] the number of positions */
            if (tdescs[ti].n_dims > 1 && tdescs[ti].ne[1] > 0) bert->weights.pos_rows = (int)tdescs[ti].ne[1];
        }
    }
    if (pos_type == 8 && raw_pos) {
        /* only the rows that exist in the file (see pos_rows above) */
        float *dequant_pos = (float *)malloc((size_t)bert->weights.pos_rows * dim * sizeof(float));
        if (!dequant_pos) { free(tdescs); return -8; }
        int blocks_dim = dim / 32;
        const uint8_t *w_bytes = (const uint8_t *)raw_pos;
        for (int t = 0; t < bert->weights.pos_rows; t++) {
            const uint8_t *w_row = w_bytes + (size_t)t * (blocks_dim * 34);
            for (int b = 0; b < blocks_dim; b++) {
                const uint8_t *blk = w_row + b * 34;
                float scale = fp16_to_fp32(*(const uint16_t *)blk);
                const int8_t *wb = (const int8_t *)(blk + 2);
                for (int c = 0; c < 32; c++) {
                    dequant_pos[t * dim + b * 32 + c] = (float)wb[c] * scale;
                }
            }
        }
        bert->weights.pos_emb = dequant_pos;
        bert->weights.pos_emb_allocated = true;
    } else if (pos_type == 1 && raw_pos) {
        /* FP16 table stored directly — expand to FP32 instead of
         * reinterpret-casting the raw bytes as floats. */
        float *dequant_pos = (float *)malloc((size_t)bert->weights.pos_rows * dim * sizeof(float));
        if (dequant_pos) {
            const uint16_t *w16 = (const uint16_t *)raw_pos;
            for (int t = 0; t < bert->weights.pos_rows * dim; t++)
                dequant_pos[t] = fp16_to_fp32(w16[t]);
            bert->weights.pos_emb = dequant_pos;
            bert->weights.pos_emb_allocated = true;
        } else {
            bert->weights.pos_emb = (const float *)raw_pos;
            bert->weights.pos_emb_allocated = false;
        }
    } else {
        bert->weights.pos_emb = (const float *)raw_pos;
        bert->weights.pos_emb_allocated = false;
    }
    if (raw_pos == NULL) {
        /* RoPE models (gemma-embedding) have no absolute position table on
         * purpose — they use rotary embeddings, not ALiBi. */
        if (bert->config.norm_style == 0 && !bert->config.use_rope && !bert->config.arch_modern) bert->config.use_alibi = true;
    }
    bert->weights.type_emb   = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, "token_types.weight", NULL);
    bert->weights.type_rows  = 1;
    for (int ti = 0; ti < (int)n_tensors; ti++)
        if (strcmp(tdescs[ti].name, "token_types.weight") == 0 && tdescs[ti].n_dims > 1 && tdescs[ti].ne[1] > 0)
            bert->weights.type_rows = (int)tdescs[ti].ne[1];
    {   /* cross-encoder / reranker head: cls.weight (dense) -> tanh -> cls.output (1 logit) */
        uint32_t ct = 0;
        const void *cw = find_gguf_tensor(tdescs, (int)n_tensors, data_start, "cls.weight", &ct);
        const float *cb = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, "cls.bias", NULL);
        const float *co = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, "cls.output.weight", NULL);
        const float *cob = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, "cls.output.bias", NULL);
        if (cw && cb && co) {
            if (ct == 8) { bert->weights.cls_w = cw; bert->weights.cls_qtype = 2; }
            else if (ct == 0) { bert->weights.cls_w = cw; bert->weights.cls_qtype = 0; }
            else if (ct == 1) {
                float *f = (float *)malloc((size_t)dim * dim * sizeof(float));
                if (!f) { free(tdescs); return -8; }
                for (size_t i = 0; i < (size_t)dim * dim; i++) f[i] = fp16_to_fp32(((const uint16_t *)cw)[i]);
                bert->weights.cls_w_owned = f; bert->weights.cls_w = f; bert->weights.cls_qtype = 0;
            } else { fprintf(stderr, "eif_bert: cls.weight type %u unsupported\n", ct); free(tdescs); return -10; }
            bert->weights.cls_b = cb; bert->weights.cls_out_w = co; bert->weights.cls_out_b = cob ? cob[0] : 0.0f;
        }
    }
    bert->weights.emb_norm_w = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, "token_embd_norm.weight", NULL);
    bert->weights.emb_norm_b = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, "token_embd_norm.bias", NULL);

    size_t arr_sz = (size_t)n_layers * sizeof(void *);
    bert->weights.q_w            = (const void **)malloc(arr_sz);
    bert->weights.q_w_scales     = NULL;
    bert->weights.q_b            = (const float **)malloc(arr_sz);
    bert->weights.k_w            = (const void **)malloc(arr_sz);
    bert->weights.k_w_scales     = NULL;
    bert->weights.k_b            = (const float **)malloc(arr_sz);
    bert->weights.v_w            = (const void **)malloc(arr_sz);
    bert->weights.v_w_scales     = NULL;
    bert->weights.v_b            = (const float **)malloc(arr_sz);
    bert->weights.out_w          = (const void **)malloc(arr_sz);
    bert->weights.out_w_scales   = NULL;
    bert->weights.out_b          = (const float **)malloc(arr_sz);
    bert->weights.att_norm_w     = (const float **)malloc(arr_sz);
    bert->weights.att_norm_b     = (const float **)malloc(arr_sz);
    bert->weights.ffn_up_w       = (const void **)malloc(arr_sz);
    bert->weights.ffn_up_w_scales = NULL;
    bert->weights.ffn_up_b       = (const float **)malloc(arr_sz);
    bert->weights.ffn_down_w     = (const void **)malloc(arr_sz);
    bert->weights.ffn_down_w_scales = NULL;
    bert->weights.ffn_down_b     = (const float **)malloc(arr_sz);
    bert->weights.ffn_norm_w     = (const float **)malloc(arr_sz);
    bert->weights.ffn_norm_b     = (const float **)malloc(arr_sz);
    bert->weights.ffn_gate_w     = NULL;
    bert->weights.q_norm_w = bert->weights.q_norm_b = NULL;
    bert->weights.k_norm_w = bert->weights.k_norm_b = NULL;
    bert->weights.att_norm2_w = bert->weights.att_norm2_b = NULL;

    /* Optional tensors: probe layer 0 to decide whether to allocate the arrays */
    {
        int has_gate  = find_gguf_tensor(tdescs, (int)n_tensors, data_start, "blk.0.ffn_gate.weight", NULL) != NULL;
        int has_qkn   = find_gguf_tensor(tdescs, (int)n_tensors, data_start, "blk.0.attn_q_norm.weight", NULL) != NULL;
        int has_norm2 = find_gguf_tensor(tdescs, (int)n_tensors, data_start, "blk.0.attn_norm_2.weight", NULL) != NULL;
        if (has_gate) bert->weights.ffn_gate_w = (const void **)calloc((size_t)n_layers, sizeof(void *));
        if (has_qkn) {
            bert->weights.q_norm_w = (const float **)calloc((size_t)n_layers, sizeof(void *));
            bert->weights.q_norm_b = (const float **)calloc((size_t)n_layers, sizeof(void *));
            bert->weights.k_norm_w = (const float **)calloc((size_t)n_layers, sizeof(void *));
            bert->weights.k_norm_b = (const float **)calloc((size_t)n_layers, sizeof(void *));
        }
        if (has_norm2) {
            bert->weights.att_norm2_w = (const float **)calloc((size_t)n_layers, sizeof(void *));
            bert->weights.att_norm2_b = (const float **)calloc((size_t)n_layers, sizeof(void *));
        }
    }

    char tname[128];
    for (int l = 0; l < n_layers; l++) {
        snprintf(tname, sizeof(tname), "blk.%d.attn_q.weight", l);
        bert->weights.q_w[l] = find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        snprintf(tname, sizeof(tname), "blk.%d.attn_q.bias", l);
        bert->weights.q_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);

        snprintf(tname, sizeof(tname), "blk.%d.attn_k.weight", l);
        bert->weights.k_w[l] = find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        snprintf(tname, sizeof(tname), "blk.%d.attn_k.bias", l);
        bert->weights.k_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);

        snprintf(tname, sizeof(tname), "blk.%d.attn_v.weight", l);
        bert->weights.v_w[l] = find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        snprintf(tname, sizeof(tname), "blk.%d.attn_v.bias", l);
        bert->weights.v_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);

        snprintf(tname, sizeof(tname), "blk.%d.attn_output.weight", l);
        bert->weights.out_w[l] = find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        snprintf(tname, sizeof(tname), "blk.%d.attn_output.bias", l);
        bert->weights.out_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);

        snprintf(tname, sizeof(tname), "blk.%d.attn_output_norm.weight", l);
        bert->weights.att_norm_w[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        snprintf(tname, sizeof(tname), "blk.%d.attn_output_norm.bias", l);
        bert->weights.att_norm_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);

        snprintf(tname, sizeof(tname), "blk.%d.ffn_up.weight", l);
        bert->weights.ffn_up_w[l] = find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        snprintf(tname, sizeof(tname), "blk.%d.ffn_up.bias", l);
        bert->weights.ffn_up_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);

        snprintf(tname, sizeof(tname), "blk.%d.ffn_down.weight", l);
        bert->weights.ffn_down_w[l] = find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        snprintf(tname, sizeof(tname), "blk.%d.ffn_down.bias", l);
        bert->weights.ffn_down_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);

        snprintf(tname, sizeof(tname), "blk.%d.layer_output_norm.weight", l);
        bert->weights.ffn_norm_w[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        snprintf(tname, sizeof(tname), "blk.%d.layer_output_norm.bias", l);
        bert->weights.ffn_norm_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);

        if (bert->weights.ffn_gate_w) {
            snprintf(tname, sizeof(tname), "blk.%d.ffn_gate.weight", l);
            bert->weights.ffn_gate_w[l] = find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        }
        if (bert->weights.q_norm_w) {
            snprintf(tname, sizeof(tname), "blk.%d.attn_q_norm.weight", l);
            bert->weights.q_norm_w[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
            snprintf(tname, sizeof(tname), "blk.%d.attn_q_norm.bias", l);
            bert->weights.q_norm_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
            snprintf(tname, sizeof(tname), "blk.%d.attn_k_norm.weight", l);
            bert->weights.k_norm_w[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
            snprintf(tname, sizeof(tname), "blk.%d.attn_k_norm.bias", l);
            bert->weights.k_norm_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        }
        if (bert->weights.att_norm2_w) {
            snprintf(tname, sizeof(tname), "blk.%d.attn_norm_2.weight", l);
            bert->weights.att_norm2_w[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
            snprintf(tname, sizeof(tname), "blk.%d.attn_norm_2.bias", l);
            bert->weights.att_norm2_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        }
    }

    /* Fused attn_qkv (nomic-bert): rows [0,dim) are Q, [dim,2dim) K, [2dim,3dim) V, so the three projections
     * are plain row offsets into the same tensor (Q8_0 / F32 only, the layouts bert_gemm reads). */
    for (int l = 0; l < n_layers; l++) {
        if (bert->weights.q_w[l] || !(bert->config.use_rope || bert->config.arch_modern)) continue; /* the two fused layouts we know */
        uint32_t ft = 0;
        snprintf(tname, sizeof(tname), "blk.%d.attn_qkv.weight", l);
        const uint8_t *f = (const uint8_t *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, &ft);
        size_t row = (bert->config.qtype == 2 && ft == 8) ? (size_t)(dim / 32) * 34 : (bert->config.qtype == 0 && ft == 0) ? (size_t)dim * 4 : 0;
        if (!f || !row) continue;
        bert->weights.q_w[l] = f;
        bert->weights.k_w[l] = f + row * (size_t)dim;
        bert->weights.v_w[l] = f + 2 * row * (size_t)dim;
        snprintf(tname, sizeof(tname), "blk.%d.attn_qkv.bias", l);
        const float *fb = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        if (fb) { bert->weights.q_b[l] = fb; bert->weights.k_b[l] = fb + dim; bert->weights.v_b[l] = fb + 2 * dim; }
    }

    /* A missing projection used to load "successfully" and crash in the first GEMM. Refuse such files up front. */
    for (int l = 0; l < n_layers; l++) {
        if (!bert->weights.q_w[l] || !bert->weights.k_w[l] || !bert->weights.v_w[l] || !bert->weights.out_w[l] ||
            !bert->weights.ffn_up_w[l] || !bert->weights.ffn_down_w[l]) {
            fprintf(stderr, "eif_bert: architecture '%s' is not supported: layer %d lacks separate attn_q/k/v/output or ffn_up/down tensors "
                            "(unsupported fused layout)\n", arch_name, l);
            free(tdescs);
            return -9;
        }
    }

    /* ModernBERT: pre-norm weights (layer 0 has no attention norm) and the final norm */
    if (bert->config.arch_modern) {
        bert->weights.pre_attn_norm_w = (const float **)calloc((size_t)n_layers, sizeof(float *));
        bert->weights.pre_ffn_norm_w  = (const float **)calloc((size_t)n_layers, sizeof(float *));
        if (!bert->weights.pre_attn_norm_w || !bert->weights.pre_ffn_norm_w) { free(tdescs); return -8; }
        for (int l = 0; l < n_layers; l++) {
            snprintf(tname, sizeof(tname), "blk.%d.attn_norm.weight", l);
            bert->weights.pre_attn_norm_w[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
            snprintf(tname, sizeof(tname), "blk.%d.ffn_norm.weight", l);
            bert->weights.pre_ffn_norm_w[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
            if (!bert->weights.pre_ffn_norm_w[l] || (l > 0 && !bert->weights.pre_attn_norm_w[l])) {
                fprintf(stderr, "eif_bert: modern-bert layer %d lacks its norm weights\n", l);
                free(tdescs);
                return -9;
            }
        }
        bert->weights.output_norm_w = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, "output_norm.weight", NULL);
        if (!bert->weights.output_norm_w || !bert->weights.emb_norm_w || bert->config.qtype != 2 ||
            bert->weights.token_emb == NULL) {
            fprintf(stderr, "eif_bert: modern-bert needs token_embd_norm, output_norm and Q8_0 weights\n");
            free(tdescs);
            return -9;
        }
    }

    /* gemma-embedding family: dedicated norm arrays + final norm + QK norms */
    if (bert->config.norm_style == 1) {
        bert->weights.pre_attn_norm_w = (const float **)calloc((size_t)n_layers, sizeof(float *));
        bert->weights.attn_post_norm_w = (const float **)calloc((size_t)n_layers, sizeof(float *));
        bert->weights.pre_ffn_norm_w  = (const float **)calloc((size_t)n_layers, sizeof(float *));
        bert->weights.post_ffw_norm_w = (const float **)calloc((size_t)n_layers, sizeof(float *));
        if (bert->weights.ffn_gate_w == NULL)
            bert->weights.ffn_gate_w = (const void **)calloc((size_t)n_layers, sizeof(void *));
        if (bert->weights.q_norm_w == NULL) {
            bert->weights.q_norm_w = (const float **)calloc((size_t)n_layers, sizeof(float *));
            bert->weights.k_norm_w = (const float **)calloc((size_t)n_layers, sizeof(float *));
        }
        for (int l = 0; l < n_layers; l++) {
            snprintf(tname, sizeof(tname), "blk.%d.attn_norm.weight", l);
            bert->weights.pre_attn_norm_w[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
            snprintf(tname, sizeof(tname), "blk.%d.post_attention_norm.weight", l);
            bert->weights.attn_post_norm_w[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
            snprintf(tname, sizeof(tname), "blk.%d.ffn_norm.weight", l);
            bert->weights.pre_ffn_norm_w[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
            snprintf(tname, sizeof(tname), "blk.%d.post_ffw_norm.weight", l);
            bert->weights.post_ffw_norm_w[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
            snprintf(tname, sizeof(tname), "blk.%d.attn_q_norm.weight", l);
            bert->weights.q_norm_w[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
            snprintf(tname, sizeof(tname), "blk.%d.attn_k_norm.weight", l);
            bert->weights.k_norm_w[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
            snprintf(tname, sizeof(tname), "blk.%d.ffn_gate.weight", l);
            bert->weights.ffn_gate_w[l] = find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        }
        bert->weights.output_norm_w = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, "output_norm.weight", NULL);
        /* Q/K RoPE norms carry no bias in this family */
        memset(bert->weights.q_norm_b, 0, (size_t)n_layers * sizeof(float *));
        memset(bert->weights.k_norm_b, 0, (size_t)n_layers * sizeof(float *));

        /* gemma-embedding2 (EmbeddingGemma-2): per-layer embeddings (PLE),
         * per-layer GQA head dims derived from tensor shapes, final output
         * projection. Reference: llama.cpp PR #30054 (models/gemma-embedding2.cpp). */
        if (bert->config.n_embd_per_layer > 0) {
            int dim_m = bert->config.dim_model;
            int max_head_dim = 0, min_kv = 1 << 30, max_q_dim = 0;
            bert->weights.g2_inp_gate  = (const void  **)calloc((size_t)n_layers, sizeof(void *));
            bert->weights.g2_proj      = (const void  **)calloc((size_t)n_layers, sizeof(void *));
            bert->weights.g2_post_norm = (const float **)calloc((size_t)n_layers, sizeof(float *));
            bert->weights.g2_out_scale = (const float **)calloc((size_t)n_layers, sizeof(float *));
            bert->weights.g2_head_dim  = (int *)calloc((size_t)n_layers, sizeof(int));
            bert->weights.g2_n_kv      = (int *)calloc((size_t)n_layers, sizeof(int));
            if (!bert->weights.g2_inp_gate || !bert->weights.g2_proj ||
                !bert->weights.g2_post_norm || !bert->weights.g2_out_scale ||
                !bert->weights.g2_head_dim || !bert->weights.g2_n_kv) {
                free(tdescs);
                return -7;
            }
            for (int l = 0; l < n_layers; l++) {
                uint64_t q_rows = 0, k_rows = 0;
                uint32_t qt = 0, kt = 0;
                char qname[128], kname[128];
                snprintf(qname, sizeof(qname), "blk.%d.attn_q.weight", l);
                snprintf(kname, sizeof(kname), "blk.%d.attn_k.weight", l);
                if (!find_gguf_tensor_dims(tdescs, (int)n_tensors, data_start, qname, &qt, NULL, &q_rows) ||
                    !find_gguf_tensor_dims(tdescs, (int)n_tensors, data_start, kname, &kt, NULL, &k_rows) ||
                    q_rows == 0 || k_rows == 0) {
                    fprintf(stderr, "eif_bert: gemma-embedding2 missing Q/K dims for layer %d\n", l);
                    free(tdescs);
                    return -7;
                }
                int hd = (int)(q_rows / (uint64_t)n_heads);
                int nkv = hd > 0 ? (int)(k_rows / (uint64_t)hd) : 0;
                if (hd <= 0 || nkv <= 0 || n_heads % nkv != 0) {
                    fprintf(stderr, "eif_bert: gemma-embedding2 bad head layout in layer %d (q_rows=%llu k_rows=%llu heads=%d)\n",
                            l, (unsigned long long)q_rows, (unsigned long long)k_rows, n_heads);
                    free(tdescs);
                    return -7;
                }
                bert->weights.g2_head_dim[l] = hd;
                bert->weights.g2_n_kv[l] = nkv;
                if (hd > max_head_dim) max_head_dim = hd;
                if (nkv < min_kv) min_kv = nkv;
                int q_dim = n_heads * hd;
                if (q_dim > max_q_dim) max_q_dim = q_dim;

                snprintf(tname, sizeof(tname), "blk.%d.inp_gate.weight", l);
                bert->weights.g2_inp_gate[l] = find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
                snprintf(tname, sizeof(tname), "blk.%d.proj.weight", l);
                bert->weights.g2_proj[l] = find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
                snprintf(tname, sizeof(tname), "blk.%d.post_norm.weight", l);
                bert->weights.g2_post_norm[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
                snprintf(tname, sizeof(tname), "blk.%d.layer_output_scale.weight", l);
                bert->weights.g2_out_scale[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
                if (!bert->weights.g2_inp_gate[l] || !bert->weights.g2_proj[l] ||
                    !bert->weights.g2_post_norm[l] || !bert->weights.g2_out_scale[l]) {
                    fprintf(stderr, "eif_bert: gemma-embedding2 missing PLE tensors in layer %d\n", l);
                    free(tdescs);
                    return -7;
                }
            }
            bert->config.head_dim = max_head_dim;   /* 512 (global layers) */
            bert->config.n_kv_heads = min_kv;       /* 1 (global layers) */
            bert->weights.g2_max_q_dim = max_q_dim; /* 4*512 = 2048 */

            /* per-layer input table projection: per_layer_model_proj
             * [dim_model, dim_model*n_layers], stored as BF16 here -> dequantize. */
            uint32_t plm_type = 0;
            uint64_t plm_ne0 = 0, plm_ne1 = 0;
            const void *plm = find_gguf_tensor_dims(tdescs, (int)n_tensors, data_start,
                                                    "per_layer_model_proj.weight", &plm_type, &plm_ne0, &plm_ne1);
            if (!plm || plm_ne0 != (uint64_t)dim_m) {
                fprintf(stderr, "eif_bert: gemma-embedding2 requires per_layer_model_proj [%d rows x %d*n_layers]\n",
                        dim_m, dim_m);
                free(tdescs);
                return -7;
            }
            size_t plm_elems = (size_t)plm_ne0 * (size_t)plm_ne1;
            float *plm_f = (float *)malloc(plm_elems * sizeof(float));
            if (!plm_f) { free(tdescs); return -8; }
            if (plm_type == 30) { /* BF16 */
                const uint16_t *src = (const uint16_t *)plm;
                for (size_t i = 0; i < plm_elems; i++) {
                    uint32_t bits = ((uint32_t)src[i]) << 16;
                    memcpy(&plm_f[i], &bits, sizeof(float));
                }
            } else if (plm_type == 1) { /* F16 */
                const uint16_t *src = (const uint16_t *)plm;
                for (size_t i = 0; i < plm_elems; i++) plm_f[i] = fp16_to_fp32(src[i]);
            } else if (plm_type == 0) {
                memcpy(plm_f, plm, plm_elems * sizeof(float));
            } else {
                fprintf(stderr, "eif_bert: per_layer_model_proj type %u unsupported\n", plm_type);
                free(plm_f);
                free(tdescs);
                return -10;
            }
            bert->weights.g2_pl_model_proj = plm_f; /* freed in eif_bert_free */

            bert->weights.g2_pl_proj_norm = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, "per_layer_proj_norm.weight", NULL);
            bert->weights.g2_out = find_gguf_tensor(tdescs, (int)n_tensors, data_start, "output.weight", NULL);
            if (!bert->weights.g2_pl_proj_norm || !bert->weights.g2_out) {
                fprintf(stderr, "eif_bert: gemma-embedding2 missing per_layer_proj_norm / output projection\n");
                free(tdescs);
                return -7;
            }
        }
    }
    free(tdescs);

    return bert_setup_vocab_and_scratch(bert);
}

/* 2. EIFM Binary Loader */
static int eif_bert_load_eifm(eif_bert_t *bert, const uint8_t *addr, size_t file_sz)
{
    const uint32_t *h32 = (const uint32_t *)addr;
    if (h32[0] != EBERT_MAGIC) return -5;

    bert->config.dim              = (int)h32[2];
    bert->config.intermediate_dim = (int)h32[3];
    bert->config.n_layers         = (int)h32[4];
    bert->config.n_heads          = (int)h32[5];
    bert->config.max_seq_len      = (int)h32[6];
    bert->config.vocab_size       = (int)h32[7];
    bert->config.qtype            = (int)h32[8];

    int dim = bert->config.dim;
    int inter_dim = bert->config.intermediate_dim;
    int n_layers = bert->config.n_layers;
    int vocab_size = bert->config.vocab_size;
    int qtype = bert->config.qtype;
    int blocks_dim = dim / 32;
    int blocks_inter = inter_dim / 32;

    const uint8_t *ptr = addr + 64;
    #define ADVANCE_PTR(p, bytes) do { p += align64(bytes); } while(0)

    bert->weights.token_emb = ptr;
    size_t token_emb_bytes = (qtype == 1) ? ((size_t)vocab_size * dim) : ((size_t)vocab_size * dim * sizeof(float));
    ADVANCE_PTR(ptr, token_emb_bytes);

    if (qtype == 1) {
        bert->weights.token_emb_scales = (const float *)ptr;
        ADVANCE_PTR(ptr, (size_t)vocab_size * blocks_dim * sizeof(float));
    }

    bert->weights.pos_emb = (const float *)ptr;
    bert->weights.pos_emb_allocated = false;
    ADVANCE_PTR(ptr, (size_t)bert->config.max_seq_len * dim * sizeof(float));

    bert->weights.type_emb = (const float *)ptr;
    bert->weights.type_rows = 2;
    ADVANCE_PTR(ptr, (size_t)2 * dim * sizeof(float));

    bert->weights.emb_norm_w = (const float *)ptr;
    ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));

    bert->weights.emb_norm_b = (const float *)ptr;
    ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));

    size_t arr_sz = (size_t)n_layers * sizeof(void *);
    bert->weights.q_w            = (const void **)malloc(arr_sz);
    bert->weights.q_w_scales     = (const float **)malloc(arr_sz);
    bert->weights.q_b            = (const float **)malloc(arr_sz);
    bert->weights.k_w            = (const void **)malloc(arr_sz);
    bert->weights.k_w_scales     = (const float **)malloc(arr_sz);
    bert->weights.k_b            = (const float **)malloc(arr_sz);
    bert->weights.v_w            = (const void **)malloc(arr_sz);
    bert->weights.v_w_scales     = (const float **)malloc(arr_sz);
    bert->weights.v_b            = (const float **)malloc(arr_sz);
    bert->weights.out_w          = (const void **)malloc(arr_sz);
    bert->weights.out_w_scales   = (const float **)malloc(arr_sz);
    bert->weights.out_b          = (const float **)malloc(arr_sz);
    bert->weights.att_norm_w     = (const float **)malloc(arr_sz);
    bert->weights.att_norm_b     = (const float **)malloc(arr_sz);
    bert->weights.ffn_up_w       = (const void **)malloc(arr_sz);
    bert->weights.ffn_up_w_scales = (const float **)malloc(arr_sz);
    bert->weights.ffn_up_b       = (const float **)malloc(arr_sz);
    bert->weights.ffn_down_w     = (const void **)malloc(arr_sz);
    bert->weights.ffn_down_w_scales = (const float **)malloc(arr_sz);
    bert->weights.ffn_down_b     = (const float **)malloc(arr_sz);
    bert->weights.ffn_norm_w     = (const float **)malloc(arr_sz);
    bert->weights.ffn_norm_b     = (const float **)malloc(arr_sz);
    bert->weights.att_norm2_w    = bert->weights.att_norm2_b = NULL;

    size_t lin_dim_bytes = (qtype == 1) ? ((size_t)dim * dim) : ((size_t)dim * dim * sizeof(float));
    size_t lin_dim_scales_bytes = (size_t)dim * blocks_dim * sizeof(float);
    size_t up_bytes = (qtype == 1) ? ((size_t)inter_dim * dim) : ((size_t)inter_dim * dim * sizeof(float));
    size_t up_scales_bytes = (size_t)inter_dim * blocks_dim * sizeof(float);
    size_t down_bytes = (qtype == 1) ? ((size_t)dim * inter_dim) : ((size_t)dim * inter_dim * sizeof(float));
    size_t down_scales_bytes = (size_t)dim * blocks_inter * sizeof(float);

    for (int l = 0; l < n_layers; l++) {
        bert->weights.q_w[l] = ptr; ADVANCE_PTR(ptr, lin_dim_bytes);
        if (qtype == 1) { bert->weights.q_w_scales[l] = (const float *)ptr; ADVANCE_PTR(ptr, lin_dim_scales_bytes); }
        bert->weights.q_b[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));

        bert->weights.k_w[l] = ptr; ADVANCE_PTR(ptr, lin_dim_bytes);
        if (qtype == 1) { bert->weights.k_w_scales[l] = (const float *)ptr; ADVANCE_PTR(ptr, lin_dim_scales_bytes); }
        bert->weights.k_b[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));

        bert->weights.v_w[l] = ptr; ADVANCE_PTR(ptr, lin_dim_bytes);
        if (qtype == 1) { bert->weights.v_w_scales[l] = (const float *)ptr; ADVANCE_PTR(ptr, lin_dim_scales_bytes); }
        bert->weights.v_b[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));

        bert->weights.out_w[l] = ptr; ADVANCE_PTR(ptr, lin_dim_bytes);
        if (qtype == 1) { bert->weights.out_w_scales[l] = (const float *)ptr; ADVANCE_PTR(ptr, lin_dim_scales_bytes); }
        bert->weights.out_b[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));

        bert->weights.att_norm_w[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));
        bert->weights.att_norm_b[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));

        bert->weights.ffn_up_w[l] = ptr; ADVANCE_PTR(ptr, up_bytes);
        if (qtype == 1) { bert->weights.ffn_up_w_scales[l] = (const float *)ptr; ADVANCE_PTR(ptr, up_scales_bytes); }
        bert->weights.ffn_up_b[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)inter_dim * sizeof(float));

        bert->weights.ffn_down_w[l] = ptr; ADVANCE_PTR(ptr, down_bytes);
        if (qtype == 1) { bert->weights.ffn_down_w_scales[l] = (const float *)ptr; ADVANCE_PTR(ptr, down_scales_bytes); }
        bert->weights.ffn_down_b[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));

        bert->weights.ffn_norm_w[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));
        bert->weights.ffn_norm_b[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));
    }

    /* Vocabulary Table */
    if (ptr < addr + file_sz) {
        uint32_t n_tok = *(const uint32_t *)ptr;
        ptr += 4;
        bert->vocab.vocab_size = (int)n_tok;
        bert->vocab.tokens = (char **)malloc((size_t)n_tok * sizeof(char *));
        bert->vocab.cls_id = 101;
        bert->vocab.sep_id = 102;
        bert->vocab.unk_id = 100;
        bert->vocab.pad_id = 0;

        for (uint32_t t = 0; t < n_tok && ptr < addr + file_sz; t++) {
            uint16_t tlen = *(const uint16_t *)ptr;
            ptr += 2;
            bert->vocab.tokens[t] = (char *)malloc(tlen + 1);
            memcpy(bert->vocab.tokens[t], ptr, tlen);
            bert->vocab.tokens[t][tlen] = '\0';
            ptr += tlen;

            if (strcmp(bert->vocab.tokens[t], "[CLS]") == 0) bert->vocab.cls_id = (int)t;
            else if (strcmp(bert->vocab.tokens[t], "[SEP]") == 0) bert->vocab.sep_id = (int)t;
            else if (strcmp(bert->vocab.tokens[t], "[UNK]") == 0) bert->vocab.unk_id = (int)t;
            else if (strcmp(bert->vocab.tokens[t], "[PAD]") == 0) bert->vocab.pad_id = (int)t;
        }
    }

    return bert_setup_vocab_and_scratch(bert);
}

/* 3. Polymorphic BERT Loader (GGUF or EIFM) */
int eif_bert_load(eif_bert_t *bert, const char *model_path)
{
    if (!bert || !model_path) return -1;
    memset(bert, 0, sizeof(eif_bert_t));

#if defined(_WIN32)
    HANDLE hFile = CreateFileA(model_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "eif_bert_load: failed to open '%s'\n", model_path);
        return -2;
    }

    LARGE_INTEGER liSize;
    if (!GetFileSizeEx(hFile, &liSize)) {
        CloseHandle(hFile);
        return -3;
    }
    size_t file_sz = (size_t)liSize.QuadPart;
    if (file_sz < 64) {
        CloseHandle(hFile);
        return -4;
    }

    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) {
        CloseHandle(hFile);
        return -4;
    }

    void *addr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, file_sz);
    CloseHandle(hMap);
    CloseHandle(hFile);
    if (!addr) {
        fprintf(stderr, "eif_bert_load MapViewOfFile failed for '%s'\n", model_path);
        return -4;
    }
#else
    int fd = open(model_path, O_RDONLY);
    if (fd < 0) {
        perror("eif_bert_load open");
        return -2;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return -3;
    }
    size_t file_sz = (size_t)st.st_size;
    if (file_sz < 64) {
        close(fd);
        return -4;
    }

    void *addr = mmap(NULL, file_sz, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) {
        perror("eif_bert_load mmap");
        return -4;
    }
#endif

    bert->mmap_addr = addr;
    bert->mmap_size = file_sz;
    bert->is_mmap = true;

    uint32_t magic = *(const uint32_t *)addr;
    if (magic == GGUF_MAGIC) {
        int rc = eif_bert_load_gguf(bert, (const uint8_t *)addr, file_sz);
        if (rc != 0) {
            eif_bert_free(bert);
            return rc;
        }
        return 0;
    } else if (magic == EBERT_MAGIC) {
        int rc = eif_bert_load_eifm(bert, (const uint8_t *)addr, file_sz);
        if (rc != 0) {
            eif_bert_free(bert);
            return rc;
        }
        return 0;
    } else {
        fprintf(stderr, "eif_bert_load: Unsupported model magic 0x%08X in '%s' (expected GGUF or BERT)\n", magic, model_path);
        eif_bert_free(bert);
        return -5;
    }
}

void eif_bert_free(eif_bert_t *bert)
{
    if (!bert || !bert->is_initialized) return;

    if (bert->weights.pos_emb_allocated && bert->weights.pos_emb) {
        free((void *)bert->weights.pos_emb);
        bert->weights.pos_emb = NULL;
        bert->weights.pos_emb_allocated = false;
    }

    if (bert->weights.q_w) free((void *)bert->weights.q_w);
    if (bert->weights.q_w_scales) free((void *)bert->weights.q_w_scales);
    if (bert->weights.q_b) free((void *)bert->weights.q_b);
    if (bert->weights.k_w) free((void *)bert->weights.k_w);
    if (bert->weights.k_w_scales) free((void *)bert->weights.k_w_scales);
    if (bert->weights.k_b) free((void *)bert->weights.k_b);
    if (bert->weights.v_w) free((void *)bert->weights.v_w);
    if (bert->weights.v_w_scales) free((void *)bert->weights.v_w_scales);
    if (bert->weights.v_b) free((void *)bert->weights.v_b);
    if (bert->weights.out_w) free((void *)bert->weights.out_w);
    if (bert->weights.out_w_scales) free((void *)bert->weights.out_w_scales);
    if (bert->weights.out_b) free((void *)bert->weights.out_b);
    if (bert->weights.att_norm_w) free((void *)bert->weights.att_norm_w);
    if (bert->weights.att_norm_b) free((void *)bert->weights.att_norm_b);
    if (bert->weights.ffn_up_w) free((void *)bert->weights.ffn_up_w);
    if (bert->weights.ffn_up_w_scales) free((void *)bert->weights.ffn_up_w_scales);
    if (bert->weights.ffn_up_b) free((void *)bert->weights.ffn_up_b);
    if (bert->weights.ffn_down_w) free((void *)bert->weights.ffn_down_w);
    if (bert->weights.ffn_down_w_scales) free((void *)bert->weights.ffn_down_w_scales);
    if (bert->weights.ffn_down_b) free((void *)bert->weights.ffn_down_b);
    if (bert->weights.ffn_norm_w) free((void *)bert->weights.ffn_norm_w);
    if (bert->weights.ffn_norm_b) free((void *)bert->weights.ffn_norm_b);
    if (bert->weights.ffn_gate_w) free((void *)bert->weights.ffn_gate_w);
    if (bert->weights.q_norm_w) free((void *)bert->weights.q_norm_w);
    if (bert->weights.q_norm_b) free((void *)bert->weights.q_norm_b);
    if (bert->weights.k_norm_w) free((void *)bert->weights.k_norm_w);
    if (bert->weights.k_norm_b) free((void *)bert->weights.k_norm_b);
    if (bert->weights.att_norm2_w) free((void *)bert->weights.att_norm2_w);
    if (bert->weights.att_norm2_b) free((void *)bert->weights.att_norm2_b);

    if (bert->vocab.tokens) {
        for (int i = 0; i < bert->vocab.vocab_size; i++) {
            if (bert->vocab.tokens[i]) free(bert->vocab.tokens[i]);
        }
        free(bert->vocab.tokens);
        free(bert->vocab.scores);
        bert->vocab.scores = NULL;
        for (int mi = 0; mi < bert->vocab.n_merges; mi++) free(bert->vocab.merge_strs ? bert->vocab.merge_strs[mi] : NULL);
        free(bert->vocab.merge_strs);
        free(bert->vocab.merge_hash);
        bert->vocab.merge_strs = NULL; bert->vocab.merge_hash = NULL; bert->vocab.n_merges = 0;
    }
    if (bert->vocab.hash_table) {
        free(bert->vocab.hash_table);
    }
    if (bert->vocab.tok_type == EBERT_TOKENIZER_BPE) {
        bert->vocab.bpe_tok.vocab = NULL;
        eif_bpe_tokenizer_free(&bert->vocab.bpe_tok);
    }

    if (bert->scratch_seq_x) free(bert->scratch_seq_x);
    if (bert->scratch_seq_xb) free(bert->scratch_seq_xb);
    if (bert->scratch_q) free(bert->scratch_q);
    if (bert->scratch_k) free(bert->scratch_k);
    if (bert->scratch_v) free(bert->scratch_v);
    if (bert->scratch_att) free(bert->scratch_att);
    if (bert->scratch_inter) free(bert->scratch_inter);
    if (bert->scratch_proj) free(bert->scratch_proj);
    if (bert->scratch_pl) free(bert->scratch_pl);
    if (bert->scratch_g2) free(bert->scratch_g2);
    if (bert->scratch_attout) free(bert->scratch_attout);
    if (bert->weights.cls_w_owned) free(bert->weights.cls_w_owned);
    if (bert->weights.g2_pl_model_proj) free((void *)bert->weights.g2_pl_model_proj);
    free((void *)bert->weights.pre_attn_norm_w); free((void *)bert->weights.attn_post_norm_w);
    free((void *)bert->weights.pre_ffn_norm_w); free((void *)bert->weights.post_ffw_norm_w);
    if (bert->weights.g2_inp_gate) free((void *)bert->weights.g2_inp_gate);
    if (bert->weights.g2_proj) free((void *)bert->weights.g2_proj);
    if (bert->weights.g2_post_norm) free((void *)bert->weights.g2_post_norm);
    if (bert->weights.g2_out_scale) free((void *)bert->weights.g2_out_scale);
    if (bert->weights.g2_head_dim) free((void *)bert->weights.g2_head_dim);
    if (bert->weights.g2_n_kv) free((void *)bert->weights.g2_n_kv);

    if (bert->is_mmap && bert->mmap_addr) {
#if defined(_WIN32)
        UnmapViewOfFile(bert->mmap_addr);
#else
        munmap(bert->mmap_addr, bert->mmap_size);
#endif
    }

    bert->is_initialized = false;
}

/* =============================================================================
 * Forward Pass (Bidirectional Multi-Head Attention + FFN)
 * ============================================================================= */

/* Encoder forward pass over T tokens (optional segment ids): the hidden states of all tokens are left in
 * bert->scratch_seq_x [T x dim]. Shared by embedding (pooling) and ranking (CLS head). */
static int bert_forward(eif_bert_t *bert, const int32_t *tokens, const uint8_t *type_ids, int T)
{
    if (bert->config.arch_modern) return eif_bert_forward_modern(bert, tokens, T);
    int dim = bert->config.dim;
    int inter_dim = bert->config.intermediate_dim;
    int n_layers = bert->config.n_layers;
    int n_heads = bert->config.n_heads;
    int head_dim = dim / n_heads;
    int qtype = bert->config.qtype;
    int blocks_dim = dim / 32;
    float inv_sqrt_head = 1.0f / sqrtf((float)head_dim);

    float *X = bert->scratch_seq_x;
    float *XB = bert->scratch_seq_xb;
    float *Q = bert->scratch_q;
    float *K = bert->scratch_k;
    float *V = bert->scratch_v;
    float *ATT = bert->scratch_att;
    float *H_INTER = bert->scratch_inter;
    float *PROJ = bert->scratch_proj;

    /* 1. Embedding Layer: Word + Pos + Token_Type -> LayerNorm */
    #pragma omp parallel for schedule(dynamic, 8) if(T >= 32)
    for (int t = 0; t < T; t++) {
        int tid = tokens[t];
        float *x_t = X + t * dim;

        if (qtype == 2) {
            /* Native GGUF Q8_0 interleaved (2B FP16 scale + 32B INT8 weights) */
            const uint8_t *w_row = (const uint8_t *)bert->weights.token_emb + (size_t)tid * (blocks_dim * 34);
            for (int b = 0; b < blocks_dim; b++) {
                const uint8_t *blk = w_row + b * 34;
                float scale = fp16_to_fp32(*(const uint16_t *)blk);
                const int8_t *wb = (const int8_t *)(blk + 2);
                for (int c = 0; c < 32; c++) {
                    x_t[b * 32 + c] = (float)wb[c] * scale;
                }
            }
        } else if (qtype == 1) {
            const int8_t *w_tok = (const int8_t *)bert->weights.token_emb + (size_t)tid * dim;
            const float *s_tok = bert->weights.token_emb_scales + (size_t)tid * blocks_dim;
            for (int b = 0; b < blocks_dim; b++) {
                float scale = s_tok[b];
                for (int c = 0; c < 32; c++) {
                    x_t[b * 32 + c] = (float)w_tok[b * 32 + c] * scale;
                }
            }
        } else {
            const float *w_tok = (const float *)bert->weights.token_emb + (size_t)tid * dim;
            memcpy(x_t, w_tok, dim * sizeof(float));
        }

        /* Add Position and Segment Embeddings */
        if (bert->weights.pos_emb) {
            const float *pos_row = bert->weights.pos_emb + (size_t)t * dim;
            for (int d = 0; d < dim; d++) {
                x_t[d] += pos_row[d];
            }
        }
        if (bert->weights.type_emb) {
            int seg = type_ids ? type_ids[t] : 0;
            if (seg >= bert->weights.type_rows) seg = bert->weights.type_rows - 1;
            const float *type_row = bert->weights.type_emb + (size_t)seg * dim;
            for (int d = 0; d < dim; d++) {
                x_t[d] += type_row[d];
            }
        }

        /* Embedding LayerNorm */
        bert_layernorm(x_t, x_t, bert->weights.emb_norm_w, bert->weights.emb_norm_b, dim, 1e-12f);
    }

    /* 2. Transformer Layers */
    bool use_alibi = !bert->config.use_rope && (bert->config.use_alibi || (bert->weights.pos_emb == NULL));

    for (int l = 0; l < n_layers; l++) {
        /* 2a. Batched Q, K, V Projections for all T tokens */
        bert_gemm(Q, bert->weights.q_w[l], bert->weights.q_w_scales ? bert->weights.q_w_scales[l] : NULL,
                  X, bert->weights.q_b[l], T, dim, dim, qtype);
        bert_gemm(K, bert->weights.k_w[l], bert->weights.k_w_scales ? bert->weights.k_w_scales[l] : NULL,
                  X, bert->weights.k_b[l], T, dim, dim, qtype);
        bert_gemm(V, bert->weights.v_w[l], bert->weights.v_w_scales ? bert->weights.v_w_scales[l] : NULL,
                  X, bert->weights.v_b[l], T, dim, dim, qtype);

        /* 2a'. Optional Q/K LayerNorm over the full hidden vector (Jina v2 code) */
        if (bert->weights.q_norm_w) {
            #pragma omp parallel for schedule(dynamic, 8) if(T >= 32)
            for (int t = 0; t < T; t++) {
                bert_layernorm(Q + t * dim, Q + t * dim, bert->weights.q_norm_w[l], bert->weights.q_norm_b[l], dim, 1e-12f);
                bert_layernorm(K + t * dim, K + t * dim, bert->weights.k_norm_w[l], bert->weights.k_norm_b[l], dim, 1e-12f);
            }
        }

        if (bert->config.use_rope) {
            for (int t = 0; t < T; t++)
                for (int h = 0; h < n_heads; h++) {
                    bert_rope_neox(Q + (size_t)t * dim + (size_t)h * head_dim, t, head_dim, bert->config.rope_freq_base);
                    bert_rope_neox(K + (size_t)t * dim + (size_t)h * head_dim, t, head_dim, bert->config.rope_freq_base);
                }
        }

        /* 2b. Multi-Head Bidirectional Self-Attention */
        /* One task per (head, query row): heads alone (16 on 20 cores of two speeds) left cores idle, and rows
         * are independent. Dynamic scheduling keeps the slower cores from setting the pace. */
        #pragma omp parallel for schedule(dynamic, 4) if((int64_t)n_heads * T >= 64)
        for (int hi = 0; hi < n_heads * T; hi++) {
            const int h = hi / T, i = hi % T;
            float *att_h = ATT + (size_t)h * T * T;
            float slope = use_alibi ? bert_alibi_slope(h, n_heads) : 0.0f;
            {
                const float *q_i = Q + i * dim + h * head_dim;
                float max_val = -1e9f;

                /* Dot-product Q_i and K_j across all j (no causal mask!) */
                for (int j = 0; j < T; j++) {
                    const float *k_j = K + j * dim + h * head_dim;
                    float score = 0.0f;
#if defined(EIF_ARCH_X86_64)
                    __m256 sacc = _mm256_setzero_ps();
                    for (int d = 0; d <= head_dim - 8; d += 8) {
                        sacc = _mm256_fmadd_ps(_mm256_loadu_ps(&q_i[d]), _mm256_loadu_ps(&k_j[d]), sacc);
                    }
                    score = hsum_float_8(sacc);
                    for (int d = (head_dim & ~7); d < head_dim; d++) {
                        score += q_i[d] * k_j[d];
                    }
#elif defined(EIF_ARCH_ARM64)
                    float32x4_t sacc = vdupq_n_f32(0.0f);
                    for (int d = 0; d <= head_dim - 4; d += 4) {
                        sacc = vfmaq_f32(sacc, vld1q_f32(&q_i[d]), vld1q_f32(&k_j[d]));
                    }
                    score = vaddvq_f32(sacc);
                    for (int d = (head_dim & ~3); d < head_dim; d++) {
                        score += q_i[d] * k_j[d];
                    }
#else
                    for (int d = 0; d < head_dim; d++) {
                        score += q_i[d] * k_j[d];
                    }
#endif
                    score *= inv_sqrt_head;

                    if (use_alibi) {
                        float dist = (float)(i > j ? i - j : j - i);
                        score -= slope * dist;
                    }

                    att_h[i * T + j] = score;
                    if (score > max_val) max_val = score;
                }

                /* Softmax */
                float sum_exp = 0.0f;
                for (int j = 0; j < T; j++) {
                    float ex = expf(att_h[i * T + j] - max_val);
                    att_h[i * T + j] = ex;
                    sum_exp += ex;
                }
                float inv_sum = 1.0f / (sum_exp > 1e-12f ? sum_exp : 1.0f);
                for (int j = 0; j < T; j++) {
                    att_h[i * T + j] *= inv_sum;
                }
            }
        }

        /* 2c. Context Accumulation: C = Attn * V */
        memset(XB, 0, (size_t)T * dim * sizeof(float));
        #pragma omp parallel for schedule(dynamic, 4) if((int64_t)n_heads * T >= 64)
        for (int hi = 0; hi < n_heads * T; hi++) {
            const int h = hi / T, i = hi % T;
            const float *att_h = ATT + (size_t)h * T * T;
            {
                float *c_ih = XB + i * dim + h * head_dim;
                for (int j = 0; j < T; j++) {
                    float p_ij = att_h[i * T + j];
                    const float *v_jh = V + j * dim + h * head_dim;
#if defined(EIF_ARCH_X86_64)
                    __m256 vp = _mm256_set1_ps(p_ij);
                    for (int d = 0; d <= head_dim - 8; d += 8) {
                        __m256 cv = _mm256_loadu_ps(&c_ih[d]);
                        __m256 vv = _mm256_loadu_ps(&v_jh[d]);
                        _mm256_storeu_ps(&c_ih[d], _mm256_fmadd_ps(vp, vv, cv));
                    }
                    for (int d = (head_dim & ~7); d < head_dim; d++) {
                        c_ih[d] += p_ij * v_jh[d];
                    }
#elif defined(EIF_ARCH_ARM64)
                    float32x4_t vp = vdupq_n_f32(p_ij);
                    for (int d = 0; d <= head_dim - 4; d += 4) {
                        float32x4_t cv = vld1q_f32(&c_ih[d]);
                        float32x4_t vv = vld1q_f32(&v_jh[d]);
                        vst1q_f32(&c_ih[d], vfmaq_f32(cv, vp, vv));
                    }
                    for (int d = (head_dim & ~3); d < head_dim; d++) {
                        c_ih[d] += p_ij * v_jh[d];
                    }
#else
                    for (int d = 0; d < head_dim; d++) {
                        c_ih[d] += p_ij * v_jh[d];
                    }
#endif
                }
            }
        }

        /* 2d. Output Projection: PROJ = XB * W_out + b */
        bert_gemm(PROJ, bert->weights.out_w[l], bert->weights.out_w_scales ? bert->weights.out_w_scales[l] : NULL,
                  XB, bert->weights.out_b[l], T, dim, dim, qtype);

        /* Residual Connection + LayerNorm */
        if (bert->weights.att_norm2_w) {
            #pragma omp parallel for schedule(dynamic, 8) if(T >= 32)
            for (int t = 0; t < T; t++) {
                float *x_t = X + t * dim;
                const float *p_t = PROJ + t * dim;
                float *c_t = XB + t * dim;
                for (int d = 0; d < dim; d++) {
                    c_t[d] = x_t[d] + p_t[d];
                }
                bert_layernorm(c_t, c_t, bert->weights.att_norm_w[l], bert->weights.att_norm_b[l], dim, 1e-12f);
                for (int d = 0; d < dim; d++) {
                    c_t[d] += x_t[d];
                }
                bert_layernorm(x_t, c_t, bert->weights.att_norm2_w[l], bert->weights.att_norm2_b[l], dim, 1e-12f);
            }
        } else {
            #pragma omp parallel for schedule(dynamic, 8) if(T >= 32)
            for (int t = 0; t < T; t++) {
                float *x_t = X + t * dim;
                const float *p_t = PROJ + t * dim;
                for (int d = 0; d < dim; d++) {
                    x_t[d] += p_t[d];
                }
                bert_layernorm(x_t, x_t, bert->weights.att_norm_w[l], bert->weights.att_norm_b[l], dim, 1e-12f);
            }
        }

        /* 2e. Feed-Forward Network: H_INTER = GELU(X * W_up + b) */
        bert_gemm(H_INTER, bert->weights.ffn_up_w[l], bert->weights.ffn_up_w_scales ? bert->weights.ffn_up_w_scales[l] : NULL,
                  X, bert->weights.ffn_up_b[l], T, inter_dim, dim, qtype);

        if (bert->weights.ffn_gate_w) {
            /* GEGLU: H = GELU(X * W_gate) * (X * W_up + b) */
            float *H_GATE = H_INTER + (size_t)T * inter_dim;
            bert_gemm(H_GATE, bert->weights.ffn_gate_w[l], NULL,
                      X, NULL, T, inter_dim, dim, qtype);
            if (bert->config.ffn_silu) {
                for (int i = 0; i < T * inter_dim; i++) H_GATE[i] = H_GATE[i] / (1.0f + expf(-H_GATE[i]));
            } else {
                bert_gelu(H_GATE, H_GATE, T * inter_dim);
            }
            for (int i = 0; i < T * inter_dim; i++) H_INTER[i] *= H_GATE[i];
        } else {
            bert_gelu(H_INTER, H_INTER, T * inter_dim);
        }

        /* FFN Down: PROJ = H_INTER * W_down + b */
        bert_gemm(PROJ, bert->weights.ffn_down_w[l], bert->weights.ffn_down_w_scales ? bert->weights.ffn_down_w_scales[l] : NULL,
                  H_INTER, bert->weights.ffn_down_b[l], T, dim, inter_dim, qtype);

        /* Residual Connection + LayerNorm */
        #pragma omp parallel for schedule(dynamic, 8) if(T >= 32)
        for (int t = 0; t < T; t++) {
            float *x_t = X + t * dim;
            const float *f_t = PROJ + t * dim;
            for (int d = 0; d < dim; d++) {
                x_t[d] += f_t[d];
            }
            bert_layernorm(x_t, x_t, bert->weights.ffn_norm_w[l], bert->weights.ffn_norm_b[l], dim, 1e-12f);
        }
    }
    return 0;
}

int eif_bert_embed(eif_bert_t *bert, const char *text, float *out_embedding)
{
    if (!bert || !bert->is_initialized || !text || !out_embedding) return -1;
    if (bert->config.norm_style == 1) {
        if (bert->config.n_embd_per_layer > 0) return eif_bert_embed_gemma2(bert, text, out_embedding);
        return eif_bert_embed_gemma(bert, text, out_embedding);
    }


    /* Inputs longer than 512 tokens used to be cut silently (fixed int32_t tokens[512]) although encoders such as
     * bge-m3 / jina advertise 8192; the cap is the scratch capacity actually allocated. */
    int max_tokens = bert_token_cap(bert);
    int32_t *tokens = (int32_t *)malloc((size_t)max_tokens * sizeof(int32_t));
    if (!tokens) return -1;
    int T = eif_bert_tokenize(bert, text, tokens, max_tokens);
    if (T <= 0) { free(tokens); return -2; }

    /* Position table may be shorter than the advertised context (see loader);
     * clamping prevents out-of-bounds positional reads -> NaN vectors. */
    if (bert->weights.pos_emb && bert->weights.pos_rows > 0 && T > bert->weights.pos_rows)
        T = bert->weights.pos_rows;

    if (bert_forward(bert, tokens, NULL, T) != 0) { free(tokens); return -3; }
    const int dim = bert->config.dim;
    const float *X = bert->scratch_seq_x;

    /* 3. Pooling: CLS (token 0) or Mean (all tokens) */
    if (bert->config.pooling_type == 2) {
        /* CLS Pooling (e.g. Granite-30M) */
        memcpy(out_embedding, X, (size_t)dim * sizeof(float));
    } else {
        /* Mean Pooling (e.g. Multilingual-E5, MiniLM) */
        memset(out_embedding, 0, (size_t)dim * sizeof(float));
        for (int t = 0; t < T; t++) {
            const float *x_t = X + t * dim;
            for (int d = 0; d < dim; d++) {
                out_embedding[d] += x_t[d];
            }
        }
        float inv_T = 1.0f / (float)T;
        for (int d = 0; d < dim; d++) {
            out_embedding[d] *= inv_T;
        }
    }

    /* 4. L2 Normalization */
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

    free(tokens);
    return 0;
}

/* ---------------------------------------------------------------------------
 * Cross-encoder ranking (bge-reranker-v2-m3 and friends)
 * score = cls.output . tanh(cls.weight . h[CLS] + cls.bias) + cls.output.bias, on the pair
 * "<s> query </s></s> doc </s>" (RoBERTa/XLM-R) or "[CLS] query [SEP] doc [SEP]" with segment ids (BERT).
 * ------------------------------------------------------------------------- */
int eif_bert_is_ranker(const eif_bert_t *bert)
{
    return bert && bert->is_initialized && bert->weights.cls_w && bert->weights.cls_out_w;
}

int eif_bert_rank(eif_bert_t *bert, const char *query, const char *doc, float *score)
{
    if (!bert || !bert->is_initialized || !query || !doc || !score) return -1;
    if (!eif_bert_is_ranker(bert) || bert->config.norm_style == 1) return -3;
    int max_tokens = bert_token_cap(bert);
    if (bert->weights.pos_emb && bert->weights.pos_rows > 0 && max_tokens > bert->weights.pos_rows)
        max_tokens = bert->weights.pos_rows;
    int32_t *tokens = (int32_t *)malloc((size_t)max_tokens * sizeof(int32_t));
    uint8_t *types = (uint8_t *)malloc((size_t)max_tokens);
    if (!tokens || !types) { free(tokens); free(types); return -1; }
    int T = eif_bert_tokenize_pair(bert, query, doc, tokens, types, max_tokens);
    if (T <= 0) { free(tokens); free(types); return -2; }
    int rc = bert_forward(bert, tokens, types, T);
    free(tokens); free(types);
    if (rc != 0) return -3;

    const int dim = bert->config.dim;
    float *hid = bert->scratch_proj;                 /* [dim], free after the forward pass */
    bert_gemm(hid, bert->weights.cls_w, NULL, bert->scratch_seq_x, bert->weights.cls_b, 1, dim, dim, bert->weights.cls_qtype);
    double acc = bert->weights.cls_out_b;
    for (int d = 0; d < dim; d++) acc += (double)tanhf(hid[d]) * bert->weights.cls_out_w[d];
    *score = (float)acc;
    return 0;
}

int eif_bert_rank_batch(eif_bert_t *bert, const char *query, const char **docs, int n_docs, float *scores)
{
    if (!bert || !query || !docs || n_docs <= 0 || !scores) return -1;
    for (int i = 0; i < n_docs; i++) {
        if (!docs[i]) return -1;
        int rc = eif_bert_rank(bert, query, docs[i], &scores[i]);
        if (rc != 0) return rc;
    }
    return 0;
}

int eif_bert_embed_batch(eif_bert_t *bert, const char **texts, int n_texts, float *out_embeddings)
{
    if (!bert || !bert->is_initialized || !texts || n_texts <= 0 || !out_embeddings) return -1;
    int dim = bert->config.dim;
    for (int i = 0; i < n_texts; i++) {
        if (!texts[i]) return -1;
        int rc = eif_bert_embed(bert, texts[i], out_embeddings + (size_t)i * dim);
        if (rc != 0) return rc;
    }
    return 0;
}
