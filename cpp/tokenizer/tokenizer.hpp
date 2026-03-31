#pragma once

#include "tokenizers_ffi.h"

#include <cstdint>
#include <string>
#include <vector>

namespace infergo {

/// Encoding result returned by TokenizerWrapper::encode().
struct Encoding {
    std::vector<int32_t> ids;
    std::vector<int32_t> attention_mask;
};

/// C++ wrapper around the HuggingFace tokenizers Rust FFI.
/// Owns the TokenizerHandle lifetime. Not copyable, movable.
class TokenizerWrapper {
public:
    /// Load a tokenizer from a tokenizer.json file.
    /// Throws std::runtime_error on failure.
    explicit TokenizerWrapper(const std::string& path);

    ~TokenizerWrapper();

    TokenizerWrapper(const TokenizerWrapper&)            = delete;
    TokenizerWrapper& operator=(const TokenizerWrapper&) = delete;
    TokenizerWrapper(TokenizerWrapper&&) noexcept;
    TokenizerWrapper& operator=(TokenizerWrapper&&) noexcept;

    /// Encode text into token IDs and attention mask.
    /// add_special_tokens: prepend/append BOS/EOS when true.
    /// max_tokens: hard cap on output length (default 4096).
    /// Throws std::runtime_error on failure.
    Encoding encode(const std::string& text,
                    bool add_special_tokens = true,
                    int  max_tokens         = 4096) const;

    /// Decode token IDs back to text.
    /// skip_special_tokens: omit BOS/EOS/PAD in output when true.
    /// Throws std::runtime_error on failure.
    std::string decode(const std::vector<int32_t>& ids,
                       bool skip_special_tokens = true) const;

    /// Decode a single token ID to its string piece.
    /// Useful for streaming token-by-token output.
    std::string decode_token(int32_t id) const;

    /// Returns the vocabulary size.
    int vocab_size() const noexcept;

private:
    TokenizerHandle* handle_ = nullptr;
};

} // namespace infergo
