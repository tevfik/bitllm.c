/**
 * @file eif_bpe_tokenizer.h
 * @brief Zero-Dependency C99 Byte-Pair Encoding (BPE) Tokenizer
 *
 * Implements high-performance BPE tokenization compatible with Hugging Face,
 * Llama2, and SmolLM2/SmolVLM2 models. Loads binary `tokenizer.bin` format (Karpathy format),
 * uses O(log V) binary search lookup, and provides zero-Python encode & decode in C.
 */

#ifndef EIF_BPE_TOKENIZER_H
#define EIF_BPE_TOKENIZER_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include "eif_status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char *str;
    int id;
} eif_bpe_token_index_t;

typedef struct {
    char **vocab;
    float *scores;
    eif_bpe_token_index_t *sorted_vocab;
    int vocab_size;
    int max_token_length;
    char byte_pieces[512];
    int bos_token_id;
    int eos_token_id;
    bool is_byte_bpe;
} eif_bpe_tokenizer_t;

/**
 * @brief Load BPE tokenizer from binary file (Karpathy tokenizer.bin format).
 *
 * @param tok            Tokenizer handle to initialize
 * @param filename       Path to tokenizer.bin
 * @param vocab_size     Expected vocabulary size (e.g., 49280 for SmolLM2, 32000 for LLaMA)
 * @return EIF_STATUS_OK on success
 */
eif_status_t eif_bpe_tokenizer_load(eif_bpe_tokenizer_t *tok, const char *filename, int vocab_size);

/**
 * @brief Initialize BPE tokenizer directly from an in-memory vocabulary array (e.g. extracted from GGUF).
 *
 * @param tok        Tokenizer handle to initialize
 * @param tokens     Array of allocated string tokens (ownership transferred to tokenizer)
 * @param scores     Array of float token scores (can be NULL)
 * @param vocab_size Number of tokens
 * @param bos_id     BOS token id (-1 for auto-detect)
 * @param eos_id     EOS token id (-1 for auto-detect)
 * @return EIF_STATUS_OK on success
 */
eif_status_t eif_bpe_tokenizer_init_from_vocab(eif_bpe_tokenizer_t *tok,
                                               char **tokens,
                                               const float *scores,
                                               int vocab_size,
                                               int bos_id,
                                               int eos_id);

/**
 * @brief Encode a UTF-8 text string into an array of token IDs using BPE.
 *
 * @param tok            Loaded tokenizer
 * @param text           Input text string
 * @param bos            Prepend BOS token (1 = yes, 0 = no)
 * @param eos            Append EOS token (1 = yes, 0 = no)
 * @param tokens         Output buffer for token IDs
 * @param max_tokens     Maximum capacity of tokens buffer
 * @return Number of tokens written, or -1 on error
 */
int eif_bpe_tokenizer_encode(const eif_bpe_tokenizer_t *tok, const char *text,
                             int8_t bos, int8_t eos, int32_t *tokens, int max_tokens);

/**
 * @brief Decode a single token ID into its string representation.
 *
 * @param tok            Loaded tokenizer
 * @param token_id       Token ID to decode
 * @return String slice pointer (null-terminated), or empty string if invalid
 */
const char *eif_bpe_tokenizer_decode(const eif_bpe_tokenizer_t *tok, int32_t token_id);

/**
 * @brief Free all memory allocated by the tokenizer.
 *
 * @param tok            Tokenizer handle to free
 */
void eif_bpe_tokenizer_free(eif_bpe_tokenizer_t *tok);

#ifdef __cplusplus
}
#endif

#endif /* EIF_BPE_TOKENIZER_H */
