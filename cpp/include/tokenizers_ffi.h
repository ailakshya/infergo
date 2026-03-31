// cpp/include/tokenizers_ffi.h
// C FFI interface to the infergo Rust tokenizer library (libinfer_tokenizer.so).
// This header is included by C++ code only — never by Go directly.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a loaded tokenizer.
typedef struct TokenizerHandle TokenizerHandle;

// Load a tokenizer from a tokenizer.json file path.
// On failure returns NULL and writes an error message into err_buf.
TokenizerHandle* tokenizer_load(
    const char* path,
    char*       err_buf,
    int         buf_len
);

// Free a tokenizer handle. Safe to call with NULL.
void tokenizer_free(TokenizerHandle* handle);

// Encode text into token IDs and attention mask.
// add_special_tokens: non-zero to prepend/append BOS/EOS tokens.
// out_ids:  caller-allocated array of at least max_tokens ints.
// out_mask: caller-allocated array of at least max_tokens ints (may be NULL).
// Returns the number of tokens written, or -1 on error.
int tokenizer_encode(
    const TokenizerHandle* handle,
    const char*            text,
    int                    add_special_tokens,
    int*                   out_ids,
    int*                   out_mask,
    int                    max_tokens,
    char*                  err_buf,
    int                    buf_len
);

// Decode token IDs back to text.
// skip_special_tokens: non-zero to omit BOS/EOS/PAD in the output.
// out_buf: caller-allocated buffer of at least buf_len bytes.
// Returns 0 on success, -1 on error.
int tokenizer_decode(
    const TokenizerHandle* handle,
    const int*             ids,
    int                    n_ids,
    int                    skip_special_tokens,
    char*                  out_buf,
    int                    buf_len,
    char*                  err_buf,
    int                    err_len
);

// Returns the vocabulary size, or 0 if handle is NULL.
int tokenizer_vocab_size(const TokenizerHandle* handle);

#ifdef __cplusplus
} // extern "C"
#endif
