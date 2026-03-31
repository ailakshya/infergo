#include "tokenizer.hpp"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace infergo {

// ─── Constructor / Destructor ─────────────────────────────────────────────────

TokenizerWrapper::TokenizerWrapper(const std::string& path) {
    char err[512] = {};
    handle_ = tokenizer_load(path.c_str(), err, static_cast<int>(sizeof(err)));
    if (handle_ == nullptr) {
        throw std::runtime_error(
            std::string("TokenizerWrapper: failed to load '") + path + "': " + err);
    }
}

TokenizerWrapper::~TokenizerWrapper() {
    tokenizer_free(handle_);
    handle_ = nullptr;
}

TokenizerWrapper::TokenizerWrapper(TokenizerWrapper&& other) noexcept
    : handle_(other.handle_)
{
    other.handle_ = nullptr;
}

TokenizerWrapper& TokenizerWrapper::operator=(TokenizerWrapper&& other) noexcept {
    if (this != &other) {
        tokenizer_free(handle_);
        handle_       = other.handle_;
        other.handle_ = nullptr;
    }
    return *this;
}

// ─── encode ──────────────────────────────────────────────────────────────────

Encoding TokenizerWrapper::encode(const std::string& text,
                                  bool add_special_tokens,
                                  int  max_tokens) const
{
    std::vector<int32_t> ids(static_cast<size_t>(max_tokens));
    std::vector<int32_t> mask(static_cast<size_t>(max_tokens));

    char err[512] = {};
    const int n = tokenizer_encode(
        handle_,
        text.c_str(),
        add_special_tokens ? 1 : 0,
        ids.data(),
        mask.data(),
        max_tokens,
        err,
        static_cast<int>(sizeof(err))
    );

    if (n < 0) {
        throw std::runtime_error(
            std::string("TokenizerWrapper::encode failed: ") + err);
    }

    ids.resize(static_cast<size_t>(n));
    mask.resize(static_cast<size_t>(n));
    return Encoding{std::move(ids), std::move(mask)};
}

// ─── decode ──────────────────────────────────────────────────────────────────

std::string TokenizerWrapper::decode(const std::vector<int32_t>& ids,
                                     bool skip_special_tokens) const
{
    if (ids.empty()) return {};

    // Output buffer: allow up to 8 bytes per token on average
    const int buf_len = static_cast<int>(ids.size()) * 8 + 64;
    std::string out(static_cast<size_t>(buf_len), '\0');

    char err[512] = {};
    const int rc = tokenizer_decode(
        handle_,
        ids.data(),
        static_cast<int>(ids.size()),
        skip_special_tokens ? 1 : 0,
        out.data(),
        buf_len,
        err,
        static_cast<int>(sizeof(err))
    );

    if (rc != 0) {
        throw std::runtime_error(
            std::string("TokenizerWrapper::decode failed: ") + err);
    }

    // Trim to actual null-terminated length
    out.resize(std::strlen(out.c_str()));
    return out;
}

// ─── decode_token ─────────────────────────────────────────────────────────────

std::string TokenizerWrapper::decode_token(int32_t id) const {
    return decode({id}, /*skip_special_tokens=*/false);
}

// ─── vocab_size ───────────────────────────────────────────────────────────────

int TokenizerWrapper::vocab_size() const noexcept {
    return tokenizer_vocab_size(handle_);
}

} // namespace infergo
