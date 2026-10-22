/**
 * @file eif_bert_tokenizer.c
 * @brief Text -> token ids for the encoder runtime: WordPiece (uncased BERT), SentencePiece merge ("llama"),
 *        Unigram/Viterbi ("t5", XLM-R) and byte-level BPE. Split out of eif_bert.c; all of it works on the
 *        vocabulary loaded into eif_bert_vocab_t.
 */
#include "eif_bert_tokenizer.h"
#include "eif_bpe_tokenizer.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =============================================================================
 * WordPiece / SentencePiece Subword Tokenizer
 * ============================================================================= */

uint32_t ebert_hash_str(const char *s) {
    uint32_t h = 2166136261u;
    while (*s) {
        h = (h ^ (uint8_t)*s++) * 16777619u;
    }
    return h;
}


int ebert_find_token(const eif_bert_vocab_t *vocab, const char *token_str)
{
    if (!vocab || !vocab->tokens) return -1;
    if (vocab->hash_table) {
        uint32_t mask = vocab->hash_mask ? vocab->hash_mask : (uint32_t)EBERT_HASH_MASK;
        uint32_t idx = ebert_hash_str(token_str) & mask;
        for (int step = 0; step < 256; step++) {
            int32_t tid = vocab->hash_table[(idx + step) & mask];
            if (tid == -1) return -1;
            if (strcmp(vocab->tokens[tid], token_str) == 0) return (int)tid;
        }
        return -1;
    }
    for (int i = 0; i < vocab->vocab_size; i++) {
        if (vocab->tokens[i] && strcmp(vocab->tokens[i], token_str) == 0) {
            return i;
        }
    }
    return -1;
}

/* SentencePiece "llama" tokenisation as in llama.cpp: characters are merged pairwise, best vocabulary score first
 * (ties: leftmost), instead of splitting at punctuation and matching greedily; bytes without a token fall back to
 * <0xXX>. add_space_prefix == 0 (Gemma) leaves the first word without the U+2581 prefix. */
typedef struct { float score; int left, right, size; } spm_bigram_t;
static int spm_bigram_before(const spm_bigram_t *a, const spm_bigram_t *b)
{
    return a->score > b->score || (a->score == b->score && a->left < b->left);
}
static void spm_heap_push(spm_bigram_t *h, int *n, spm_bigram_t x)
{
    int i = (*n)++;
    while (i > 0) { int par = (i - 1) / 2; if (!spm_bigram_before(&x, &h[par])) break; h[i] = h[par]; i = par; }
    h[i] = x;
}
static spm_bigram_t spm_heap_pop(spm_bigram_t *h, int *n)
{
    spm_bigram_t top = h[0], x = h[--(*n)];
    int i = 0;
    for (;;) {
        int c = 2 * i + 1;
        if (c >= *n) break;
        if (c + 1 < *n && spm_bigram_before(&h[c + 1], &h[c])) c++;
        if (!spm_bigram_before(&h[c], &x)) break;
        h[i] = h[c]; i = c;
    }
    if (*n > 0) h[i] = x;
    return top;
}

/* Core merge pass over one NUL-terminated segment: appends tokens (no CLS/SEP) at out[*n_out]. */
static int spm_core(const eif_bert_vocab_t *v, const char *text, int prefix, int32_t *out, int *n_out_p, int max_tokens)
{
    size_t n = strlen(text);
    char *norm = (char *)malloc(n * 3 + 8);
    if (!norm) return -1;
    size_t m = 0;
    if (prefix) { norm[m++] = (char)0xE2; norm[m++] = (char)0x96; norm[m++] = (char)0x81; }
    for (const char *p = text; *p; p++) {
        if (*p == ' ') { norm[m++] = (char)0xE2; norm[m++] = (char)0x96; norm[m++] = (char)0x81; }
        else norm[m++] = *p;
    }
    int ns = 0;
    int *start = (int *)malloc((m + 1) * sizeof(int)), *len = (int *)malloc((m + 1) * sizeof(int));
    int *prev = (int *)malloc((m + 1) * sizeof(int)), *next = (int *)malloc((m + 1) * sizeof(int));
    spm_bigram_t *heap = (spm_bigram_t *)malloc((2 * m + 8) * sizeof(spm_bigram_t));
    if (!start || !len || !prev || !next || !heap) { free(norm); free(start); free(len); free(prev); free(next); free(heap); return -1; }
    for (size_t i = 0; i < m;) {
        unsigned char c0 = (unsigned char)norm[i];
        size_t cl = c0 >= 0xF0 ? 4 : c0 >= 0xE0 ? 3 : c0 >= 0xC0 ? 2 : 1;
        if (i + cl > m) cl = m - i;
        start[ns] = (int)i; len[ns] = (int)cl; prev[ns] = ns - 1; next[ns] = ns + 1; ns++;
        i += cl;
    }
    if (ns > 0) next[ns - 1] = -1;
    int nh = 0;
    char buf[256];
#define SPM_TRY(L, R) do { int l_ = (L), r_ = (R); if (l_ >= 0 && r_ >= 0 && len[l_] + len[r_] < (int)sizeof(buf)) { \
        memcpy(buf, norm + start[l_], len[l_]); memcpy(buf + len[l_], norm + start[r_], len[r_]); buf[len[l_] + len[r_]] = 0; \
        int id_ = ebert_find_token(v, buf); \
        if (id_ >= 0 && nh < (int)(2 * m + 8)) spm_heap_push(heap, &nh, (spm_bigram_t){v->scores[id_], l_, r_, len[l_] + len[r_]}); } } while (0)
    for (int i = 0; i + 1 < ns; i++) SPM_TRY(i, i + 1);
    while (nh > 0) {
        spm_bigram_t b = spm_heap_pop(heap, &nh);
        if (len[b.left] == 0 || len[b.right] == 0 || len[b.left] + len[b.right] != b.size) continue;
        len[b.left] += len[b.right];
        len[b.right] = 0;
        next[b.left] = next[b.right];
        if (next[b.right] >= 0) prev[next[b.right]] = b.left;
        SPM_TRY(prev[b.left], b.left);
        SPM_TRY(b.left, next[b.left]);
    }
#undef SPM_TRY
    int n_out = *n_out_p;
    for (int i = 0; i >= 0 && i < ns && n_out < max_tokens - 1; i = next[i]) {
        if (len[i] == 0) continue;
        memcpy(buf, norm + start[i], len[i]); buf[len[i]] = 0;
        int id = ebert_find_token(v, buf);
        if (id >= 0) { out[n_out++] = id; continue; }
        for (int k = 0; k < len[i] && n_out < max_tokens - 1; k++) {
            char bt[8]; snprintf(bt, sizeof bt, "<0x%02X>", (unsigned char)norm[start[i] + k]);
            int bid = ebert_find_token(v, bt);
            out[n_out++] = bid >= 0 ? bid : ((v->unk_id >= 0) ? v->unk_id : 3);
        }
    }
    *n_out_p = n_out;
    free(norm); free(start); free(len); free(prev); free(next); free(heap);
    return 0;
}

/* Gemma vocabularies hold runs of raw spaces as tokens (2..31). llama.cpp emits those for a run of two or more
 * blanks and the word after the run carries no U+2581; a single blank stays a U+2581 prefix. Without this
 * indented code (newline + 4 spaces) tokenised as four U+2581 and the embedding drifted (cos 0.88). */
static int spm_tokenize(const eif_bert_vocab_t *v, const char *text, int32_t *out, int max_tokens)
{
    int n_out = 0, has_runs = ebert_find_token(v, "  ") >= 0;
    out[n_out++] = (v->cls_id >= 0) ? v->cls_id : 0;
    size_t n = strlen(text);
    char *seg = (char *)malloc(n + 1);
    if (!seg) return -1;
    size_t i = 0;
    int first = 1;
    while (i <= n) {
        size_t j = i;
        while (text[j] && !(has_runs && text[j] == ' ' && text[j + 1] == ' ')) j++;
        if (j > i || first) {
            memcpy(seg, text + i, j - i); seg[j - i] = 0;
            if (spm_core(v, seg, first && v->add_space_prefix, out, &n_out, max_tokens) != 0) { free(seg); return -1; }
        }
        first = 0;
        if (!text[j]) break;
        size_t k = j;
        while (text[k] == ' ') k++;
        size_t run = k - j;
        while (run >= 2 && n_out < max_tokens - 1) {
            size_t take = run > 31 ? 31 : run;
            int id = -1;
            for (; take >= 2; take--) {
                char sp[40];
                memset(sp, ' ', take); sp[take] = 0;
                if ((id = ebert_find_token(v, sp)) >= 0) break;
            }
            if (id < 0) break;
            out[n_out++] = id; run -= take;
        }
        if (run == 1 && n_out < max_tokens - 1) { int id = ebert_find_token(v, "\xE2\x96\x81"); if (id >= 0) out[n_out++] = id; }
        i = k;
    }
    free(seg);
    out[n_out++] = (v->sep_id >= 0) ? v->sep_id : 2;
    return n_out;
}

/* Unigram (SentencePiece "t5"/XLM-R) tokenisation: the segmentation with the highest total score over the
 * vocabulary scores (Viterbi), as llama.cpp does. The old greedy longest-match differed from it for many words
 * (bge-m3 / multilingual encoders), shifting every embedding. Input is normalised like llama.cpp's UGM:
 * leading/trailing blanks dropped, blank runs collapsed to U+2581, dummy prefix U+2581. */
static int ugm_tokenize(const eif_bert_vocab_t *v, const char *text, int32_t *out, int max_tokens)
{
    size_t n = strlen(text);
    char *norm = (char *)malloc(n * 3 + 8);
    if (!norm) return -1;
    size_t m = 0;
    int need_space = 1;
    for (const char *p = text; *p; p++) {
        if (isspace((unsigned char)*p)) { need_space = 1; continue; }
        if (need_space) { norm[m++] = (char)0xE2; norm[m++] = (char)0x96; norm[m++] = (char)0x81; need_space = 0; }
        norm[m++] = *p;
    }
    int n_out = 0;
    out[n_out++] = (v->cls_id >= 0) ? v->cls_id : 0;
    if (m > 0) {
        double *best = (double *)malloc((m + 1) * sizeof(double));
        int *prev = (int *)malloc((m + 1) * sizeof(int));
        int *tid = (int *)malloc((m + 1) * sizeof(int));
        float min_score = 0.0f;
        for (int i = 0; i < v->vocab_size; i++) if (v->scores[i] < min_score) min_score = v->scores[i];
        if (!best || !prev || !tid) { free(best); free(prev); free(tid); free(norm); return -1; }
        for (size_t i = 0; i <= m; i++) { best[i] = -1e30; prev[i] = -1; tid[i] = -1; }
        best[0] = 0.0;
        char sub[128];
        for (size_t i = 0; i < m; i++) {
            if (best[i] < -1e29) continue;
            if (((unsigned char)norm[i] & 0xC0) == 0x80) continue; /* not a character start */
            int found_single = 0;
            for (size_t len = 1; len <= 96 && i + len <= m; len++) {
                size_t j = i + len;
                if (j < m && ((unsigned char)norm[j] & 0xC0) == 0x80) continue; /* must end on a boundary */
                memcpy(sub, norm + i, len); sub[len] = '\0';
                int id = ebert_find_token(v, sub);
                if (id < 0) continue;
                double sc = best[i] + v->scores[id];
                if (sc > best[j]) { best[j] = sc; prev[j] = (int)i; tid[j] = id; }
                if (len == 1 || (((unsigned char)norm[i] >= 0xC0) && j == i + 1)) found_single = 1;
            }
            (void)found_single;
            /* unknown character: consume one UTF-8 character as <unk> with a penalised score */
            size_t cl = 1;
            unsigned char c0 = (unsigned char)norm[i];
            if (c0 >= 0xF0) cl = 4; else if (c0 >= 0xE0) cl = 3; else if (c0 >= 0xC0) cl = 2;
            if (i + cl > m) cl = m - i;
            double sc = best[i] + min_score - 10.0;
            if (sc > best[i + cl] && best[i + cl] < -1e29) { best[i + cl] = sc; prev[i + cl] = (int)i; tid[i + cl] = (v->unk_id >= 0) ? v->unk_id : 3; }
        }
        int *rev = (int *)malloc((m + 1) * sizeof(int));
        int nr = 0;
        for (int pos = (int)m; pos > 0 && rev; pos = prev[pos]) { if (prev[pos] < 0) break; rev[nr++] = tid[pos]; }
        for (int i = nr - 1; i >= 0 && n_out < max_tokens - 1; i--) out[n_out++] = rev[i];
        free(rev); free(best); free(prev); free(tid);
    }
    free(norm);
    out[n_out++] = (v->sep_id >= 0) ? v->sep_id : 2;
    return n_out;
}

/* ---------------------------------------------------------------------------
 * Gemma4 BPE (llama.cpp LLAMA_VOCAB_PRE_TYPE_GEMMA4): spaces become U+2581, the text is cut at newlines
 * ([^\n]+ | \n+), every piece is split into UTF-8 characters and merged by merge rank (lowest rank first,
 * ties leftmost); bytes with no token fall back to <0xXX>. No GPT-2 byte encoding.
 * ------------------------------------------------------------------------- */
int ebert_build_merge_table(eif_bert_vocab_t *v)
{
    uint32_t size = 1024;
    while (size < (uint32_t)v->n_merges * 2u) size <<= 1;
    v->merge_hash = (int32_t *)calloc(size, sizeof(int32_t));
    if (!v->merge_hash) return -1;
    v->merge_mask = size - 1;
    for (int i = 0; i < v->n_merges; i++) {
        uint32_t idx = ebert_hash_str(v->merge_strs[i]) & v->merge_mask;
        while (v->merge_hash[idx]) idx = (idx + 1) & v->merge_mask;
        v->merge_hash[idx] = i + 1;
    }
    return 0;
}

static int merge_rank(const eif_bert_vocab_t *v, const char *l, int ln, const char *r, int rn)
{
    char buf[256];
    if (ln + rn + 1 >= (int)sizeof(buf)) return -1;
    memcpy(buf, l, ln); buf[ln] = ' '; memcpy(buf + ln + 1, r, rn); buf[ln + 1 + rn] = '\0';
    uint32_t idx = ebert_hash_str(buf) & v->merge_mask;
    while (v->merge_hash[idx]) {
        int rank = v->merge_hash[idx] - 1;
        if (strcmp(v->merge_strs[rank], buf) == 0) return rank;
        idx = (idx + 1) & v->merge_mask;
    }
    return -1;
}

typedef struct { int rank, left, right, size; } bpe_bigram_t;
static int bpe_before(const bpe_bigram_t *a, const bpe_bigram_t *b) { return a->rank < b->rank || (a->rank == b->rank && a->left < b->left); }
static void bpe_push(bpe_bigram_t *h, int *n, bpe_bigram_t x)
{
    int i = (*n)++;
    while (i > 0) { int par = (i - 1) / 2; if (!bpe_before(&x, &h[par])) break; h[i] = h[par]; i = par; }
    h[i] = x;
}
static bpe_bigram_t bpe_pop(bpe_bigram_t *h, int *n)
{
    bpe_bigram_t top = h[0], x = h[--(*n)];
    int i = 0;
    for (;;) {
        int c = 2 * i + 1;
        if (c >= *n) break;
        if (c + 1 < *n && bpe_before(&h[c + 1], &h[c])) c++;
        if (!bpe_before(&h[c], &x)) break;
        h[i] = h[c]; i = c;
    }
    if (*n > 0) h[i] = x;
    return top;
}

static int bpe_emit_piece(const eif_bert_vocab_t *v, const char *piece, size_t m, int32_t *out, int *n_out, int max_out)
{
    if (m == 0) return 0;
    int *start = (int *)malloc((m + 1) * sizeof(int)), *len = (int *)malloc((m + 1) * sizeof(int));
    int *next = (int *)malloc((m + 1) * sizeof(int)), *prev = (int *)malloc((m + 1) * sizeof(int));
    bpe_bigram_t *heap = (bpe_bigram_t *)malloc((2 * m + 8) * sizeof(bpe_bigram_t));
    if (!start || !len || !next || !prev || !heap) { free(start); free(len); free(next); free(prev); free(heap); return -1; }
    int ns = 0;
    for (size_t i = 0; i < m;) {
        unsigned char c0 = (unsigned char)piece[i];
        size_t cl = c0 >= 0xF0 ? 4 : c0 >= 0xE0 ? 3 : c0 >= 0xC0 ? 2 : 1;
        if (i + cl > m) cl = m - i;
        start[ns] = (int)i; len[ns] = (int)cl; prev[ns] = ns - 1; next[ns] = ns + 1; ns++;
        i += cl;
    }
    next[ns - 1] = -1;
    int nh = 0, cap_h = (int)(2 * m + 8);
#define BPE_TRY(L, R) do { int l_ = (L), r_ = (R); if (l_ >= 0 && r_ >= 0) { \
        int rk_ = merge_rank(v, piece + start[l_], len[l_], piece + start[r_], len[r_]); \
        if (rk_ >= 0 && nh < cap_h) bpe_push(heap, &nh, (bpe_bigram_t){rk_, l_, r_, len[l_] + len[r_]}); } } while (0)
    for (int i = 0; i + 1 < ns; i++) BPE_TRY(i, i + 1);
    while (nh > 0) {
        bpe_bigram_t b = bpe_pop(heap, &nh);
        if (len[b.left] == 0 || len[b.right] == 0 || len[b.left] + len[b.right] != b.size) continue;
        len[b.left] += len[b.right]; len[b.right] = 0;
        next[b.left] = next[b.right];
        if (next[b.right] >= 0) prev[next[b.right]] = b.left;
        BPE_TRY(prev[b.left], b.left);
        BPE_TRY(b.left, next[b.left]);
    }
#undef BPE_TRY
    char buf[256];
    for (int i = 0; i >= 0 && i < ns && *n_out < max_out; i = next[i]) {
        if (len[i] == 0) continue;
        if (len[i] >= (int)sizeof(buf)) continue;
        memcpy(buf, piece + start[i], len[i]); buf[len[i]] = 0;
        int id = ebert_find_token(v, buf);
        if (id >= 0) { out[(*n_out)++] = id; continue; }
        for (int k = 0; k < len[i] && *n_out < max_out; k++) {
            char bt[8]; snprintf(bt, sizeof bt, "<0x%02X>", (unsigned char)piece[start[i] + k]);
            int bid = ebert_find_token(v, bt);
            out[(*n_out)++] = bid >= 0 ? bid : ((v->unk_id >= 0) ? v->unk_id : 3);
        }
    }
    free(start); free(len); free(next); free(prev); free(heap);
    return 0;
}

static int gemma4_bpe_tokenize(const eif_bert_vocab_t *v, const char *text, int32_t *out, int max_tokens)
{
    size_t n = strlen(text);
    char *norm = (char *)malloc(n * 3 + 8);
    if (!norm) return -1;
    size_t m = 0;
    if (v->add_space_prefix) { norm[m++] = (char)0xE2; norm[m++] = (char)0x96; norm[m++] = (char)0x81; }
    for (const char *p = text; *p; p++) {
        if (*p == ' ') { norm[m++] = (char)0xE2; norm[m++] = (char)0x96; norm[m++] = (char)0x81; }
        else norm[m++] = *p;
    }
    int n_out = 0;
    out[n_out++] = (v->cls_id >= 0) ? v->cls_id : 0;
    int max_body = max_tokens - 1;
    for (size_t i = 0; i < m && n_out < max_body;) {
        size_t j = i;
        int nl = norm[i] == '\n';
        while (j < m && (norm[j] == '\n') == nl) j++;
        if (nl) {                                   /* run of newlines: one token when the vocabulary has it */
            char tmp[128]; size_t L = j - i < sizeof(tmp) - 1 ? j - i : sizeof(tmp) - 1;
            memcpy(tmp, norm + i, L); tmp[L] = 0;
            int id = ebert_find_token(v, tmp);
            if (id >= 0 && L == j - i) out[n_out++] = id;
            else if (bpe_emit_piece(v, norm + i, j - i, out, &n_out, max_body) != 0) { free(norm); return -1; }
        } else if (bpe_emit_piece(v, norm + i, j - i, out, &n_out, max_body) != 0) { free(norm); return -1; }
        i = j;
    }
    out[n_out++] = (v->sep_id >= 0) ? v->sep_id : 2;
    free(norm);
    return n_out;
}

int eif_bert_tokenize(const eif_bert_t *bert, const char *text, int32_t *out_tokens, int max_tokens)
{
    if (!bert || !text || !out_tokens || max_tokens < 3) return -1;
    const eif_bert_vocab_t *v = &bert->vocab;

    if (v->tok_type == EBERT_TOKENIZER_BPE) {
        int bos = (v->cls_id >= 0) ? 1 : 0;
        int eos = (v->sep_id >= 0) ? 1 : 0;
        return eif_bpe_tokenizer_encode(&v->bpe_tok, text, (int8_t)bos, (int8_t)eos, out_tokens, max_tokens);
    }

    if (v->tok_type == EBERT_TOKENIZER_SPM && v->bpe_merges) return gemma4_bpe_tokenize(v, text, out_tokens, max_tokens);
    if (v->tok_type == EBERT_TOKENIZER_SPM && v->unigram) return ugm_tokenize(v, text, out_tokens, max_tokens);
    if (v->tok_type == EBERT_TOKENIZER_SPM && v->spm_bigram) return spm_tokenize(v, text, out_tokens, max_tokens);

    if (v->tok_type == EBERT_TOKENIZER_SPM) {
        int n_out = 0;
        out_tokens[n_out++] = (v->cls_id >= 0) ? v->cls_id : 0;

        const char *p = text;
        char word[256];

        while (*p && n_out < max_tokens - 1) {
            while (*p && isspace((unsigned char)*p)) p++;
            if (!*p) break;

            if (ispunct((unsigned char)*p)) {
                char pchar = *p++;
                char punct_str[8];
                punct_str[0] = (char)0xE2; punct_str[1] = (char)0x96; punct_str[2] = (char)0x81;
                punct_str[3] = pchar; punct_str[4] = '\0';
                int tid = ebert_find_token(v, punct_str);
                if (tid < 0) {
                    punct_str[0] = pchar; punct_str[1] = '\0';
                    tid = ebert_find_token(v, punct_str);
                }
                if (tid >= 0) out_tokens[n_out++] = tid;
                continue;
            }

            int wlen = 0;
            while (*p && !isspace((unsigned char)*p) && !ispunct((unsigned char)*p) && wlen < 200) {
                word[wlen++] = v->lowercase ? (char)tolower((unsigned char)*p) : *p;
                p++;
            }
            word[wlen] = '\0';
            if (wlen == 0) continue;

            int start = 0;
            while (start < wlen && n_out < max_tokens - 1) {
                int end = wlen;
                int found_id = -1;
                while (end > start) {
                    int sub_len = end - start;
                    char sub[256];

                    if (start == 0) {
                        sub[0] = (char)0xE2; sub[1] = (char)0x96; sub[2] = (char)0x81;
                        memcpy(sub + 3, word + start, sub_len);
                        sub[sub_len + 3] = '\0';
                        found_id = ebert_find_token(v, sub);
                        if (found_id < 0) {
                            memcpy(sub, word + start, sub_len);
                            sub[sub_len] = '\0';
                            found_id = ebert_find_token(v, sub);
                        }
                    } else {
                        sub[0] = (char)0xE2; sub[1] = (char)0x96; sub[2] = (char)0x81;
                        sub[3] = (char)0xE2; sub[4] = (char)0x96; sub[5] = (char)0x81;
                        memcpy(sub + 6, word + start, sub_len);
                        sub[sub_len + 6] = '\0';
                        found_id = ebert_find_token(v, sub);

                        if (found_id < 0) {
                            sub[0] = (char)0xE2; sub[1] = (char)0x96; sub[2] = (char)0x81;
                            memcpy(sub + 3, word + start, sub_len);
                            sub[sub_len + 3] = '\0';
                            found_id = ebert_find_token(v, sub);
                        }
                        if (found_id < 0) {
                            memcpy(sub, word + start, sub_len);
                            sub[sub_len] = '\0';
                            found_id = ebert_find_token(v, sub);
                        }
                    }

                    if (found_id >= 0) {
                        out_tokens[n_out++] = found_id;
                        start = end;
                        break;
                    }
                    end--;
                }
                if (found_id < 0) {
                    out_tokens[n_out++] = (v->unk_id >= 0) ? v->unk_id : 3;
                    start++;
                }
            }
        }

        if (n_out < max_tokens) {
            out_tokens[n_out++] = (v->sep_id >= 0) ? v->sep_id : 2;
        }
        return n_out;
    }

    int n_out = 0;
    out_tokens[n_out++] = (v->cls_id >= 0) ? v->cls_id : 101; /* [CLS] */

    char word[256];
    const char *p = text;

    while (*p && n_out < max_tokens - 1) {
        /* Skip whitespace */
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;

        /* Punctuation handling: isolate ASCII punctuation */
        if (ispunct((unsigned char)*p)) {
            char pchar = *p++;
            char punct_str[8];
            int tid = -1;
            if (v->is_spm_prefix) {
                punct_str[0] = (char)0xE2;
                punct_str[1] = (char)0x96;
                punct_str[2] = (char)0x81;
                punct_str[3] = pchar;
                punct_str[4] = '\0';
                tid = ebert_find_token(v, punct_str);
            }
            if (tid < 0) {
                punct_str[0] = pchar;
                punct_str[1] = '\0';
                tid = ebert_find_token(v, punct_str);
            }
            if (tid >= 0) {
                out_tokens[n_out++] = tid;
            }
            continue;
        }

        /* Read word until space or punctuation.
         * Gemma/SPM vocabs are case-sensitive: EIF_SPM_KEEP_CASE=1 disables
         * the historical lowercase pass (used by mE5's lowercase-only vocab). */
        int wlen = 0;
        while (*p && !isspace((unsigned char)*p) && !ispunct((unsigned char)*p) && wlen < 200) {
            word[wlen++] = v->lowercase ? (char)tolower((unsigned char)*p) : *p;
            p++;
        }
        word[wlen] = '\0';
        if (wlen == 0) continue;

        /* Subword matching */
        int start = 0;
        while (start < wlen && n_out < max_tokens - 1) {
            int end = wlen;
            int found_id = -1;
            while (end > start) {
                char sub[256];
                int sub_len = end - start;
                if (start == 0) {
                    if (v->is_spm_prefix) {
                        sub[0] = (char)0xE2;
                        sub[1] = (char)0x96;
                        sub[2] = (char)0x81;
                        memcpy(sub + 3, word + start, sub_len);
                        sub[sub_len + 3] = '\0';
                    } else {
                        memcpy(sub, word + start, sub_len);
                        sub[sub_len] = '\0';
                    }
                } else {
                    if (v->is_spm_prefix) {
                        memcpy(sub, word + start, sub_len);
                        sub[sub_len] = '\0';
                    } else {
                        sub[0] = '#';
                        sub[1] = '#';
                        memcpy(sub + 2, word + start, sub_len);
                        sub[sub_len + 2] = '\0';
                    }
                }
                found_id = ebert_find_token(v, sub);
                if (found_id >= 0) {
                    out_tokens[n_out++] = found_id;
                    start = end;
                    break;
                }
                end--;
            }
            if (found_id < 0) {
                /* If start == 0 and spm prefix didn't match, fallback without spm prefix */
                if (start == 0 && v->is_spm_prefix) {
                    end = wlen;
                    while (end > start) {
                        char sub[256];
                        int sub_len = end - start;
                        memcpy(sub, word + start, sub_len);
                        sub[sub_len] = '\0';
                        found_id = ebert_find_token(v, sub);
                        if (found_id >= 0) {
                            out_tokens[n_out++] = found_id;
                            start = end;
                            break;
                        }
                        end--;
                    }
                }
                if (found_id < 0) {
                    out_tokens[n_out++] = (v->unk_id >= 0) ? v->unk_id : 100;
                    break;
                }
            }
        }
    }

    if (n_out < max_tokens) {
        out_tokens[n_out++] = (v->sep_id >= 0) ? v->sep_id : 102; /* [SEP] */
    }

    return n_out;
}


int eif_bert_tokenize_pair(const eif_bert_t *bert, const char *a, const char *b, int32_t *out, uint8_t *types, int max_tokens)
{
    if (!bert || !a || !b || !out || max_tokens < 8) return -1;
    const eif_bert_vocab_t *v = &bert->vocab;
    int cap = max_tokens;
    int32_t *ta = (int32_t *)malloc((size_t)cap * sizeof(int32_t));
    int32_t *tb = (int32_t *)malloc((size_t)cap * sizeof(int32_t));
    if (!ta || !tb) { free(ta); free(tb); return -1; }
    /* eif_bert_tokenize adds the leading/trailing special tokens: strip them, we place our own */
    int na = eif_bert_tokenize(bert, a, ta, cap);
    int nb = eif_bert_tokenize(bert, b, tb, cap);
    if (na < 2 || nb < 2) { free(ta); free(tb); return -2; }
    na -= 2; nb -= 2;
    int cls = (v->cls_id >= 0) ? v->cls_id : 0, sep = (v->sep_id >= 0) ? v->sep_id : 2;
    int roberta = v->unigram || v->tok_type == EBERT_TOKENIZER_BPE;     /* "</s></s>" between segments */
    int specials = roberta ? 4 : 3;
    if (na > cap - specials - 1) na = cap - specials - 1;               /* query first, but always leave room */
    if (na < 0) na = 0;
    if (nb > cap - specials - na) nb = cap - specials - na;             /* longest-first: the document gives way */
    int n = 0;
    out[n] = cls; if (types) types[n] = 0; n++;
    for (int i = 0; i < na; i++) { out[n] = ta[1 + i]; if (types) types[n] = 0; n++; }
    out[n] = sep; if (types) types[n] = 0; n++;
    if (roberta) { out[n] = sep; if (types) types[n] = 0; n++; }
    for (int i = 0; i < nb; i++) { out[n] = tb[1 + i]; if (types) types[n] = roberta ? 0 : 1; n++; }
    out[n] = sep; if (types) types[n] = roberta ? 0 : 1; n++;
    free(ta); free(tb);
    return n;
}
