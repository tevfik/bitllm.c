#ifndef EIF_BERT_TOKENIZER_H
#define EIF_BERT_TOKENIZER_H
/* Internal helpers shared by eif_bert.c and eif_bert_tokenizer.c (not part of the public API). */
#include "eif_bert.h"

#define EBERT_HASH_MASK 0x1FFFF /* default hash table: 131,072 slots */

uint32_t ebert_hash_str(const char *s);
/** Exact vocabulary lookup, -1 if absent. */
int ebert_find_token(const eif_bert_vocab_t *vocab, const char *token_str);

/** Build the merge-rank hash table of a vocabulary that carries tokenizer.ggml.merges. 0 on success. */
int ebert_build_merge_table(eif_bert_vocab_t *vocab);

#endif
