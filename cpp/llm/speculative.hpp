#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>
#include <functional>

namespace infergo {

/// SpeculativeDecoder implements speculative decoding using a draft model.
///
/// Algorithm per step:
///   1. Draft: run draft model N times autoregressively, collect N candidate tokens
///   2. Verify: feed [last_accepted, d0..dN-1] to target model in ONE batch decode
///   3. Accept: compare target logits with draft tokens, accept matching prefix
///   4. Cleanup: remove rejected KV entries from both models
///
/// The draft model must share the same vocabulary as the target model.
class SpeculativeDecoder {
public:
    /// Callback for streaming tokens. Return false to stop generation.
    using TokenCallback = std::function<bool(int32_t token, const char* piece)>;

    SpeculativeDecoder();
    ~SpeculativeDecoder();

    // Not copyable
    SpeculativeDecoder(const SpeculativeDecoder&) = delete;
    SpeculativeDecoder& operator=(const SpeculativeDecoder&) = delete;

    /// Load the draft model. target_ctx is borrowed (not owned).
    /// Returns false on failure (incompatible vocab, load error, etc.)
    bool Init(llama_model* target_model,
              llama_context* target_ctx,
              const std::string& draft_path,
              int n_gpu_layers,
              int n_draft);

    /// Run the full speculative generation loop.
    /// prompt_tokens: tokenized prompt (including BOS)
    /// max_tokens: max generation length
    /// temperature: sampling temperature (0 = greedy)
    /// grammar_str: GBNF grammar (empty = no constraint)
    /// callback: called for each accepted token (nullptr = collect all)
    ///
    /// Returns generated text. Sets n_predict, n_drafted, n_accepted for stats.
    std::string Generate(const std::vector<int32_t>& prompt_tokens,
                         int max_tokens,
                         float temperature,
                         const std::string& grammar_str,
                         TokenCallback callback,
                         int* out_n_predict,
                         int* out_n_drafted,
                         int* out_n_accepted);

    /// Check if initialized
    bool IsReady() const noexcept { return draft_model_ != nullptr; }

    int NDraft() const noexcept { return n_draft_; }

private:
    llama_model*   target_model_  = nullptr;  // borrowed
    llama_context* target_ctx_    = nullptr;  // borrowed
    llama_model*   draft_model_   = nullptr;  // owned
    llama_context* draft_ctx_     = nullptr;  // owned
    int            n_draft_       = 5;
    int            vocab_size_    = 0;

    /// Sample a token greedily or with temperature from logits at batch index idx
    int32_t SampleFromCtx(llama_context* ctx, int batch_idx, float temperature);
};

} // namespace infergo
