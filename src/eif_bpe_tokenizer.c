/**
 * @file eif_bpe_tokenizer.c
 * @brief Zero-Dependency C99 Byte-Pair Encoding (BPE) Tokenizer Implementation
 */

#include "eif_bpe_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int compare_token_index(const void *a, const void *b)
{
    const eif_bpe_token_index_t *tok_a = (const eif_bpe_token_index_t *)a;
    const eif_bpe_token_index_t *tok_b = (const eif_bpe_token_index_t *)b;
    return strcmp(tok_a->str, tok_b->str);
}

static int str_lookup(const char *str, const eif_bpe_token_index_t *sorted_vocab, int vocab_size)
{
    eif_bpe_token_index_t key;
    key.str = (char *)str;
    key.id = 0;

    eif_bpe_token_index_t *res = (eif_bpe_token_index_t *)bsearch(
        &key, sorted_vocab, vocab_size, sizeof(eif_bpe_token_index_t), compare_token_index);

    return res ? res->id : -1;
}

static char g_b2u[256][4];
static uint8_t g_u2b[512];
static bool g_b2u_initialized = false;

static void init_b2u_table(void)
{
    if (g_b2u_initialized) return;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        bool in_bs = ((b >= '!' && b <= '~') || (b >= 161 && b <= 172) || (b >= 174 && b <= 255));
        if (in_bs) {
            if (b < 128) {
                g_b2u[b][0] = (char)b;
                g_b2u[b][1] = '\0';
            } else {
                g_b2u[b][0] = (char)(0xC0 | (b >> 6));
                g_b2u[b][1] = (char)(0x80 | (b & 0x3F));
                g_b2u[b][2] = '\0';
            }
            g_u2b[b] = (uint8_t)b;
        } else {
            int cp = 256 + n;
            n++;
            g_b2u[b][0] = (char)(0xC0 | (cp >> 6));
            g_b2u[b][1] = (char)(0x80 | (cp & 0x3F));
            g_b2u[b][2] = '\0';
            g_u2b[cp] = (uint8_t)b;
        }
    }
    g_b2u_initialized = true;
}

eif_status_t eif_bpe_tokenizer_load(eif_bpe_tokenizer_t *tok, const char *filename, int vocab_size)
{
    if (!tok || !filename || vocab_size <= 0) {
        return EIF_STATUS_INVALID_ARGUMENT;
    }

    FILE *f = fopen(filename, "rb");
    if (!f) {
        return EIF_STATUS_ERROR;
    }

    int max_token_length = 0;
    if (fread(&max_token_length, sizeof(int), 1, f) != 1) {
        fclose(f);
        return EIF_STATUS_ERROR;
    }

    tok->max_token_length = max_token_length;
    tok->vocab_size = vocab_size;
    tok->bos_token_id = (vocab_size > 100000) ? 100264 : 1;
    tok->eos_token_id = (vocab_size > 100000) ? 100257 : 2;

    tok->vocab = (char **)malloc(vocab_size * sizeof(char *));
    tok->scores = (float *)malloc(vocab_size * sizeof(float));
    tok->sorted_vocab = (eif_bpe_token_index_t *)malloc(vocab_size * sizeof(eif_bpe_token_index_t));

    if (!tok->vocab || !tok->scores || !tok->sorted_vocab) {
        fclose(f);
        eif_bpe_tokenizer_free(tok);
        return EIF_STATUS_OUT_OF_MEMORY;
    }

    /* Initialize byte pieces for single bytes */
    for (int i = 0; i < 256; i++) {
        tok->byte_pieces[i * 2] = (char)i;
        tok->byte_pieces[i * 2 + 1] = '\0';
    }

    int actual_tokens = 0;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(&tok->scores[i], sizeof(float), 1, f) != 1) {
            /* Clean EOF reached (e.g. padded vocabulary in model embedding) */
            break;
        }

        int len = 0;
        if (fread(&len, sizeof(int), 1, f) != 1 || len < 0) {
            fclose(f);
            eif_bpe_tokenizer_free(tok);
            return EIF_STATUS_ERROR;
        }

        tok->vocab[i] = (char *)malloc(len + 1);
        if (!tok->vocab[i]) {
            fclose(f);
            eif_bpe_tokenizer_free(tok);
            return EIF_STATUS_OUT_OF_MEMORY;
        }

        if (fread(tok->vocab[i], 1, len, f) != (size_t)len) {
            fclose(f);
            eif_bpe_tokenizer_free(tok);
            return EIF_STATUS_ERROR;
        }
        tok->vocab[i][len] = '\0';

        tok->sorted_vocab[i].str = tok->vocab[i];
        tok->sorted_vocab[i].id = i;
        actual_tokens++;
    }

    fclose(f);

    tok->vocab_size = actual_tokens;

    /* Auto-detect BOS and EOS tokens from vocabulary strings */
    tok->bos_token_id = -1;
    tok->eos_token_id = -1;
    for (int i = 0; i < actual_tokens; i++) {
        if (tok->bos_token_id < 0 && (strcmp(tok->vocab[i], "<s>") == 0 || strcmp(tok->vocab[i], "<|begin_of_text|>") == 0)) {
            tok->bos_token_id = i;
        }
        if (tok->eos_token_id < 0 && (strcmp(tok->vocab[i], "</s>") == 0 || strcmp(tok->vocab[i], "<|im_end|>") == 0 || strcmp(tok->vocab[i], "<|endoftext|>") == 0)) {
            tok->eos_token_id = i;
        }
    }
    /* Sort vocab for fast O(log V) binary search */
    qsort(tok->sorted_vocab, actual_tokens, sizeof(eif_bpe_token_index_t), compare_token_index);

    init_b2u_table();
    tok->is_byte_bpe = (str_lookup("\xC4\xA0", tok->sorted_vocab, actual_tokens) != -1 ||
                        str_lookup("Ġ", tok->sorted_vocab, actual_tokens) != -1);

    return EIF_STATUS_OK;
}


/* ---- Qwen-style pre-tokenizer ------------------------------------------------
 * Splits text into the chunks BPE is applied to, following
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|
 *   ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
 * Unicode classes are approximated by code point ranges (no tables): everything
 * non-ASCII is a letter unless it falls in the punctuation/symbol/space ranges below. */
typedef enum { CP_LETTER, CP_MARK, CP_NUMBER, CP_SPACE, CP_OTHER } cp_class_t;

static int utf8_next(const char *p, const char *end, uint32_t *cp)
{
    unsigned char c = (unsigned char)p[0];
    int n = (c < 0x80) ? 1 : (c >= 0xF0 ? 4 : (c >= 0xE0 ? 3 : (c >= 0xC0 ? 2 : 1)));
    if (n == 1 || p + n > end) { *cp = c; return 1; }
    uint32_t v = (n == 2) ? (c & 0x1Fu) : (n == 3) ? (c & 0x0Fu) : (c & 0x07u);
    for (int i = 1; i < n; i++) {
        unsigned char cc = (unsigned char)p[i];
        if ((cc & 0xC0) != 0x80) { *cp = c; return 1; }
        v = (v << 6) | (cc & 0x3Fu);
    }
    *cp = v;
    return n;
}

static cp_class_t cp_class(uint32_t c)
{
    if (c < 0x80) {
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) return CP_LETTER;
        if (c >= '0' && c <= '9') return CP_NUMBER;
        if (c == ' ' || (c >= 9 && c <= 13)) return CP_SPACE;
        return CP_OTHER;
    }
    if (c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 ||
        c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000) return CP_SPACE;
    if (c < 0xC0) {
        if (c == 0xAA || c == 0xB5 || c == 0xBA) return CP_LETTER;
        if (c == 0xB2 || c == 0xB3 || c == 0xB9 || (c >= 0xBC && c <= 0xBE)) return CP_NUMBER;
        return CP_OTHER;
    }
    if (c == 0xD7 || c == 0xF7) return CP_OTHER;
    if ((c >= 0x300 && c <= 0x36F) || (c >= 0x483 && c <= 0x489) || (c >= 0x64B && c <= 0x65F) ||
        (c >= 0x93A && c <= 0x94F) || (c >= 0xE31 && c <= 0xE3A && c != 0xE32 && c != 0xE33) ||
        (c >= 0xE47 && c <= 0xE4E) || (c >= 0x20D0 && c <= 0x20FF) || (c >= 0xFE00 && c <= 0xFE0F) ||
        (c >= 0xFE20 && c <= 0xFE2F)) return CP_MARK;
    if ((c >= 0x660 && c <= 0x669) || (c >= 0x6F0 && c <= 0x6F9) || (c >= 0x966 && c <= 0x96F) ||
        (c >= 0x2070 && c <= 0x2079) || (c >= 0x2080 && c <= 0x2089) || (c >= 0x2150 && c <= 0x218B) ||
        (c >= 0x2460 && c <= 0x249B) || (c >= 0xFF10 && c <= 0xFF19)) return CP_NUMBER;
    if ((c >= 0x2000 && c <= 0x206F) || (c >= 0x20A0 && c <= 0x20CF) || (c >= 0x2190 && c <= 0x2BFF) ||
        (c >= 0x2E00 && c <= 0x2E7F) || (c >= 0x3001 && c <= 0x3004) || (c >= 0x3008 && c <= 0x303F) ||
        (c >= 0xE000 && c <= 0xF8FF) || (c >= 0xFE30 && c <= 0xFE6F) || (c >= 0xFF01 && c <= 0xFF0F) ||
        (c >= 0xFF1A && c <= 0xFF20) || (c >= 0xFF3B && c <= 0xFF40) || (c >= 0xFF5B && c <= 0xFF65) ||
        (c >= 0xFFF0 && c <= 0xFFFF) || (c >= 0x1F000 && c <= 0x1FAFF) || (c >= 0x2600 && c <= 0x27BF)) return CP_OTHER;
    return CP_LETTER;
}

static cp_class_t class_at(const char *p, const char *end, int *len, uint32_t *cp)
{
    *len = utf8_next(p, end, cp);
    return cp_class(*cp);
}

/* Length in bytes of the next pre-token starting at p (p < end). */
static size_t pretoken_len(const char *p, const char *end)
{
    uint32_t c0;
    int l0;
    cp_class_t k0 = class_at(p, end, &l0, &c0);

    /* 1. contractions */
    if (c0 == '\'' && p + 1 < end) {
        char a = (char)(p[1] | 0x20);
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
        if (p + 2 < end) {
            char b = (char)(p[2] | 0x20);
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
        }
    }

    /* 2. optional non-letter/number lead + letters/marks */
    {
        const char *q = p;
        bool lead_ok = (c0 != '\r' && c0 != '\n' && k0 != CP_LETTER && k0 != CP_NUMBER);
        if (lead_ok) q = p + l0;
        uint32_t c;
        int l;
        cp_class_t k;
        if (q < end && ((k = class_at(q, end, &l, &c)) == CP_LETTER || k == CP_MARK)) {
            while (q < end && ((k = class_at(q, end, &l, &c)) == CP_LETTER || k == CP_MARK)) q += l;
            return (size_t)(q - p);
        }
        if (k0 == CP_LETTER || k0 == CP_MARK) {
            /* unreachable for letters (consumed above); a lone mark with no follower */
            return (size_t)l0;
        }
    }

    /* 3. single number */
    if (k0 == CP_NUMBER) return (size_t)l0;

    /* 4. optional space + symbols + newlines */
    {
        const char *q = p;
        uint32_t c;
        int l;
        cp_class_t k;
        if (c0 == ' ') q = p + 1;
        const char *s0 = q;
        while (q < end && (k = class_at(q, end, &l, &c)) == CP_OTHER) q += l;
        if (q > s0) {
            while (q < end && (*q == '\r' || *q == '\n')) q++;
            return (size_t)(q - p);
        }
    }

    /* 5-7. whitespace */
    if (k0 == CP_SPACE) {
        const char *q = p, *last_nl_end = NULL, *prev = p;
        uint32_t c;
        int l;
        while (q < end && class_at(q, end, &l, &c) == CP_SPACE) {
            prev = q;
            q += l;
            if (c == '\r' || c == '\n') last_nl_end = q;
        }
        if (last_nl_end) return (size_t)(last_nl_end - p);
        if (q >= end) return (size_t)(q - p);
        if (prev > p) return (size_t)(prev - p); /* leave last space for the next token */
        return (size_t)(q - p);
    }
    return (size_t)l0;
}

/* GPT-2 / RoBERTa pre-tokenizer (what llama.cpp uses for jina-v2-code and plain gpt2 vocabularies):
 *   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
 * Differs from the Qwen split above: digits group, whitespace runs keep all but the last blank. */
size_t eif_pretoken_len_gpt2(const char *p, const char *end)
{
    uint32_t c0;
    int l0;
    cp_class_t k0 = class_at(p, end, &l0, &c0);
    if (c0 == '\'' && p + 1 < end) {
        char a = p[1];
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
        if (p + 2 < end) {
            char b = p[2];
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
        }
    }
    const char *q = p;
    if (c0 == ' ' && p + 1 < end) q = p + 1;
    uint32_t c;
    int l;
    cp_class_t k = class_at(q, end, &l, &c);
    if (k == CP_LETTER || k == CP_MARK) {
        while (q < end && ((k = class_at(q, end, &l, &c)) == CP_LETTER || k == CP_MARK)) q += l;
        return (size_t)(q - p);
    }
    if (k == CP_NUMBER) {
        while (q < end && class_at(q, end, &l, &c) == CP_NUMBER) q += l;
        return (size_t)(q - p);
    }
    if (k == CP_OTHER) {
        while (q < end && class_at(q, end, &l, &c) == CP_OTHER) q += l;
        return (size_t)(q - p);
    }
    if (k0 == CP_SPACE) {
        const char *e = p, *prev = p;
        while (e < end && class_at(e, end, &l, &c) == CP_SPACE) { prev = e; e += l; }
        if (e >= end) return (size_t)(e - p);
        if (prev > p) return (size_t)(prev - p);
        return (size_t)(e - p);
    }
    return (size_t)l0;
}

/* Granite multilingual embedding pre-tokenizer (llama.cpp LLAMA_VOCAB_PRE_TYPE_GRANITE_EMB_MULTI):
 *   [^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])[^a-z])*((?=[\p{L}\p{M}])[^A-Z])+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?
 *  |[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])[^a-z])+((?=[\p{L}\p{M}])[^A-Z])*(?:'...)?
 *  |\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
 * "Upper" letters are letters/marks outside a-z, "lower" letters those outside A-Z (non-ASCII letters are both). */
static bool gr_letter(cp_class_t k) { return k == CP_LETTER || k == CP_MARK; }

static size_t gr_contraction(const char *q, const char *end)
{
    if (q >= end || *q != '\'' || q + 1 >= end) return 0;
    char a = (char)(q[1] | 0x20);
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    if (q + 2 < end) {
        char b = (char)(q[2] | 0x20);
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
    }
    return 0;
}

/* Word alternatives 1 and 2 starting at s (after the optional lead); 0 when neither matches. */
static size_t gr_word(const char *s, const char *end)
{
    uint32_t c; int l;
    const char *q = s;
    int u_max = 0;
    while (q < end) {                                   /* leading "upper" letters */
        cp_class_t k = class_at(q, end, &l, &c);
        if (!gr_letter(k) || (c >= 'a' && c <= 'z')) break;
        u_max++; q += l;
    }
    const char *after_u = q;
    int best = -1;
    q = s;
    for (int idx = 0; idx <= u_max && q < end; idx++) { /* alt 1: largest k <= u_max whose char is "lower" */
        cp_class_t k = class_at(q, end, &l, &c);
        if (!gr_letter(k)) break;
        if (!(c >= 'A' && c <= 'Z')) best = idx;
        q += l;
    }
    const char *e;
    if (best >= 0) {
        q = s;
        for (int idx = 0; idx < best; idx++) { class_at(q, end, &l, &c); q += l; }
        while (q < end) {                                /* L+ from char best */
            cp_class_t k = class_at(q, end, &l, &c);
            if (!gr_letter(k) || (c >= 'A' && c <= 'Z')) break;
            q += l;
        }
        e = q;
    } else if (u_max > 0) {
        e = after_u;                                     /* alt 2: U+ with an empty L* */
    } else {
        return 0;
    }
    e += gr_contraction(e, end);
    return (size_t)(e - s);
}

size_t eif_pretoken_len_granite(const char *p, const char *end)
{
    uint32_t c0; int l0;
    cp_class_t k0 = class_at(p, end, &l0, &c0);
    if (k0 != CP_LETTER && k0 != CP_NUMBER && c0 != '\r' && c0 != '\n') {
        size_t w = gr_word(p + l0, end);
        if (w) return (size_t)l0 + w;
    }
    {
        size_t w = gr_word(p, end);
        if (w) return w;
    }
    if (k0 == CP_NUMBER) {
        const char *q = p; uint32_t c; int l, n = 0;
        while (q < end && n < 3 && class_at(q, end, &l, &c) == CP_NUMBER) { q += l; n++; }
        return (size_t)(q - p);
    }
    {
        const char *q = (c0 == ' ') ? p + 1 : p;
        const char *s0 = q;
        uint32_t c; int l; cp_class_t k;
        while (q < end && (k = class_at(q, end, &l, &c)) != CP_SPACE && k != CP_LETTER && k != CP_NUMBER) q += l;
        if (q > s0) {
            while (q < end && (*q == '\r' || *q == '\n' || *q == '/')) q++;
            return (size_t)(q - p);
        }
    }
    if (k0 == CP_SPACE) {
        const char *q = p, *last_nl_end = NULL, *prev = p;
        uint32_t c; int l;
        while (q < end && class_at(q, end, &l, &c) == CP_SPACE) {
            prev = q; q += l;
            if (c == '\r' || c == '\n') last_nl_end = q;
        }
        if (last_nl_end) return (size_t)(last_nl_end - p);
        if (q >= end) return (size_t)(q - p);
        if (prev > p) return (size_t)(prev - p);
        return (size_t)(q - p);
    }
    return (size_t)l0;
}

/* GPT-2 byte -> printable UTF-8 character (the table used by every byte-level BPE vocabulary). */
const char *eif_bpe_byte_to_unicode(uint8_t b)
{
    init_b2u_table();
    return g_b2u[b];
}

static void bpe_merge_range(const eif_bpe_tokenizer_t *tok, int32_t *work, int start, int *n_tokens,
                            bool skip_special)
{
    char merge_buf[512];
    while (1) {
        float best_score = -1e10f;
        int best_id = -1;
        int best_idx = -1;

        for (int i = start; i < *n_tokens - 1; i++) {
            const char *first = tok->vocab[work[i]];
            const char *second = tok->vocab[work[i + 1]];

            /* Never merge special control tokens */
            if (skip_special && (first[0] == '<' || second[0] == '<')) {
                continue;
            }

            size_t len1 = strlen(first);
            size_t len2 = strlen(second);
            if (len1 + len2 >= sizeof(merge_buf)) {
                continue;
            }

            memcpy(merge_buf, first, len1);
            memcpy(merge_buf + len1, second, len2);
            merge_buf[len1 + len2] = '\0';

            int id = str_lookup(merge_buf, tok->sorted_vocab, tok->vocab_size);
            if (id != -1 && tok->scores[id] > best_score) {
                best_score = tok->scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; /* No more merges possible */
        }

        work[best_idx] = best_id;
        for (int i = best_idx + 1; i < *n_tokens - 1; i++) {
            work[i] = work[i + 1];
        }
        (*n_tokens)--;
    }
}

int eif_bpe_tokenizer_encode(const eif_bpe_tokenizer_t *tok, const char *text,
                             int8_t bos, int8_t eos, int32_t *tokens, int max_tokens)
{
    if (!tok || !text || !tokens || max_tokens <= 0) {
        return -1;
    }

    size_t text_len = strlen(text);
    if (text_len == 0) {
        int n = 0;
        if (bos && tok->bos_token_id >= 0 && n < max_tokens) tokens[n++] = tok->bos_token_id;
        if (eos && tok->eos_token_id >= 0 && n < max_tokens) tokens[n++] = tok->eos_token_id;
        return n;
    }

    /* Allocate work buffer so character-level phase is not prematurely truncated by max_tokens */
    size_t work_cap = text_len + 32;
    int32_t stack_work[512];
    int32_t *work = (work_cap <= 512) ? stack_work : (int32_t *)malloc(work_cap * sizeof(int32_t));
    if (!work) return -1;

    int n_tokens = 0;
    if (bos && tok->bos_token_id >= 0) {
        work[n_tokens++] = tok->bos_token_id;
    }

    /* 1. Initial character-level tokenization with special control token detection */
    char str_buf[8];
    for (const char *c = text; *c != '\0'; c++) {
        if ((size_t)n_tokens + 2 >= work_cap) {
            break;
        }

        /* Check for special control tokens enclosed in <...> (e.g. <|im_start|>, <|im_end|>, <think>, </think>) */
        if (*c == '<') {
            const char *end_bracket = strchr(c, '>');
            if (end_bracket && (end_bracket - c) < 64) {
                char tag[64];
                size_t tag_len = (size_t)(end_bracket - c + 1);
                memcpy(tag, c, tag_len);
                tag[tag_len] = '\0';
                int tag_id = str_lookup(tag, tok->sorted_vocab, tok->vocab_size);
                if (tag_id != -1) {
                    work[n_tokens++] = tag_id;
                    c = end_bracket;
                    continue;
                }
            }
        }

        if (tok->is_byte_bpe) {
            /* Pre-tokenize, then byte-map and merge each chunk independently */
            const char *end = text + text_len;
            size_t plen = tok->pretok_gpt2 ? eif_pretoken_len_gpt2(c, end) : pretoken_len(c, end);
            /* stop the chunk at the next special tag so it is handled by the check above */
            for (size_t k = 1; k < plen; k++) {
                if (c[k] != '<') continue;
                const char *eb = strchr(c + k, '>');
                if (eb && (eb - (c + k)) < 64) {
                    char tag[64];
                    size_t tl = (size_t)(eb - (c + k) + 1);
                    memcpy(tag, c + k, tl);
                    tag[tl] = '\0';
                    if (str_lookup(tag, tok->sorted_vocab, tok->vocab_size) != -1) { plen = k; break; }
                }
            }
            if (plen == 0) plen = 1;
            int start = n_tokens;
            for (size_t k = 0; k < plen && (size_t)n_tokens + 2 < work_cap; k++) {
                int id = str_lookup(g_b2u[(uint8_t)c[k]], tok->sorted_vocab, tok->vocab_size);
                if (id != -1) work[n_tokens++] = id;
            }
            bpe_merge_range(tok, work, start, &n_tokens, false);
            c += plen - 1;
        } else {
            str_buf[0] = *c;
            str_buf[1] = '\0';
            int id = str_lookup(str_buf, tok->sorted_vocab, tok->vocab_size);

            if (id != -1) {
                work[n_tokens++] = id;
            } else {
                /* Byte fallback token */
                unsigned char byte_val = (unsigned char)*c;
                int byte_id = byte_val + 3; /* Standard Llama byte offset */
                if (byte_id < tok->vocab_size) {
                    work[n_tokens++] = byte_id;
                }
            }
        }
    }

    /* 2. Merge best consecutive pairs iteratively (SentencePiece-style vocab) */
    if (!tok->is_byte_bpe) {
        bpe_merge_range(tok, work, 0, &n_tokens, true);
    }

    if (eos && tok->eos_token_id >= 0) {
        work[n_tokens++] = tok->eos_token_id;
    }

    /* Copy final merged tokens to output buffer */
    int out_count = (n_tokens < max_tokens) ? n_tokens : max_tokens;
    memcpy(tokens, work, (size_t)out_count * sizeof(int32_t));

    if (work != stack_work) {
        free(work);
    }

    return out_count;
}

const char *eif_bpe_tokenizer_decode(const eif_bpe_tokenizer_t *tok, int32_t token_id)
{
    if (!tok || !tok->vocab || token_id < 0 || token_id >= tok->vocab_size) {
        return "";
    }
    if (token_id == tok->eos_token_id || token_id == 248044 || token_id == 248046) {
        return "";
    }
    const char *raw = tok->vocab[token_id];
    if (!tok->is_byte_bpe || !raw) {
        return raw ? raw : "";
    }

    /* Convert Byte-Level BPE chars (e.g. Ġ -> ' ', Ċ -> '\n') back to original bytes */
    static char decode_buf[512];
    size_t out_idx = 0;
    for (size_t i = 0; raw[i] != '\0' && out_idx + 1 < sizeof(decode_buf); ) {
        unsigned char b1 = (unsigned char)raw[i];
        if (b1 >= 0xC2 && b1 <= 0xC5 && raw[i + 1] != '\0') {
            unsigned char b2 = (unsigned char)raw[i + 1];
            int cp = ((b1 & 0x1F) << 6) | (b2 & 0x3F);
            if (cp >= 161 && cp < 256 + 68) {
                decode_buf[out_idx++] = (char)g_u2b[cp];
                i += 2;
                continue;
            }
        }
        decode_buf[out_idx++] = raw[i++];
    }
    decode_buf[out_idx] = '\0';
    return decode_buf;
}

void eif_bpe_tokenizer_free(eif_bpe_tokenizer_t *tok)
{
    if (!tok) return;

    if (tok->vocab) {
        for (int i = 0; i < tok->vocab_size; i++) {
            if (tok->vocab[i]) {
                free(tok->vocab[i]);
            }
        }
        free(tok->vocab);
        tok->vocab = NULL;
    }

    if (tok->scores) {
        free(tok->scores);
        tok->scores = NULL;
    }

    if (tok->sorted_vocab) {
        free(tok->sorted_vocab);
        tok->sorted_vocab = NULL;
    }

    tok->vocab_size = 0;
}

eif_status_t eif_bpe_tokenizer_init_from_vocab(eif_bpe_tokenizer_t *tok,
                                               char **tokens,
                                               const float *scores,
                                               int vocab_size,
                                               int bos_id,
                                               int eos_id)
{
    if (!tok || !tokens || vocab_size <= 0) {
        return EIF_STATUS_INVALID_ARGUMENT;
    }

    memset(tok, 0, sizeof(*tok));
    tok->vocab = tokens;
    tok->vocab_size = vocab_size;
    tok->scores = (float *)malloc((size_t)vocab_size * sizeof(float));
    tok->sorted_vocab = (eif_bpe_token_index_t *)malloc((size_t)vocab_size * sizeof(eif_bpe_token_index_t));
    if (!tok->sorted_vocab) {
        return EIF_STATUS_OUT_OF_MEMORY;
    }

    if (scores && tok->scores) {
        memcpy(tok->scores, scores, (size_t)vocab_size * sizeof(float));
    } else if (tok->scores) {
        for (int i = 0; i < vocab_size; i++) {
            tok->scores[i] = -(float)i;
        }
    }

    int max_len = 0;
    for (int i = 0; i < vocab_size; i++) {
        tok->sorted_vocab[i].str = tok->vocab[i];
        tok->sorted_vocab[i].id = i;
        if (tok->vocab[i]) {
            int len = (int)strlen(tok->vocab[i]);
            if (len > max_len) max_len = len;
        }
    }
    tok->max_token_length = max_len > 0 ? max_len : 128;

    /* Initialize byte pieces */
    for (int i = 0; i < 256; i++) {
        tok->byte_pieces[i * 2] = (char)i;
        tok->byte_pieces[i * 2 + 1] = '\0';
    }

    /* Set or auto-detect special tokens */
    tok->bos_token_id = bos_id;
    tok->eos_token_id = eos_id;
    if (tok->bos_token_id < 0 || tok->eos_token_id < 0) {
        for (int i = 0; i < vocab_size; i++) {
            if (!tok->vocab[i]) continue;
            if (tok->bos_token_id < 0 && (strcmp(tok->vocab[i], "<s>") == 0 || strcmp(tok->vocab[i], "<|begin_of_text|>") == 0 || strcmp(tok->vocab[i], "<|im_start|>") == 0)) {
                tok->bos_token_id = i;
            }
            if (tok->eos_token_id < 0 && (strcmp(tok->vocab[i], "</s>") == 0 || strcmp(tok->vocab[i], "<|end_of_text|>") == 0 || strcmp(tok->vocab[i], "<|im_end|>") == 0 || strcmp(tok->vocab[i], "<|endoftext|>") == 0)) {
                tok->eos_token_id = i;
            }
        }
    }

    /* Sort vocab for fast O(log V) binary search */
    qsort(tok->sorted_vocab, (size_t)vocab_size, sizeof(eif_bpe_token_index_t), compare_token_index);

    init_b2u_table();
    tok->is_byte_bpe = (str_lookup("\xC4\xA0", tok->sorted_vocab, vocab_size) != -1 ||
                        str_lookup("Ġ", tok->sorted_vocab, vocab_size) != -1);

    return EIF_STATUS_OK;
}

