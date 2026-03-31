#pragma once

#include "kv_cache.hpp"

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

namespace infergo {

/// Input for one sequence in a BatchDecode call.
/// Represents the tokens to process in this decode step.
struct SequenceInput {
    int                   seq_id;        // slot ID (from KVCacheSlotManager)
    std::vector<int32_t>  tokens;        // token IDs to process
    int                   pos;           // KV position of the first token
    bool                  want_logits;   // true → output logits for last token
};

/// Output logits for one sequence after BatchDecode.
struct SequenceLogits {
    int                seq_id;           // matches SequenceInput::seq_id
    std::vector<float> logits;           // [vocab_size] for the last token
};

/// LLMEngine wraps a llama_model + llama_context and exposes a simple
/// multi-sequence batch decode interface.
///
/// Typical lifecycle:
///   engine.LoadModel(path, n_gpu_layers, ctx_size, n_seq_max, n_batch);
///   while (serving) {
///       auto out = engine.BatchDecode(inputs);   // one decode step
///   }
///   // destructor frees context and model
class LLMEngine {
public:
    LLMEngine();
    ~LLMEngine();

    // Not copyable or movable — owns C resources
    LLMEngine(const LLMEngine&)            = delete;
    LLMEngine& operator=(const LLMEngine&) = delete;

    /// Load a GGUF model and create the llama context.
    /// n_gpu_layers: how many transformer layers to offload to GPU (≥0, large value = all)
    /// ctx_size:     total KV cache token budget across all sequences
    /// n_seq_max:    max number of concurrent sequences
    /// n_batch:      max tokens per llama_decode call
    void LoadModel(const std::string& path,
                   int  n_gpu_layers = 99,
                   int  ctx_size     = 4096,
                   int  n_seq_max    = 16,
                   int  n_batch      = 512);

    /// Process one decode step for a set of sequences.
    /// Each entry in inputs describes which tokens to process and at what position.
    /// Returns logits for every sequence that had want_logits=true.
    /// Throws std::runtime_error if the model is not loaded or the batch fails.
    std::vector<SequenceLogits> BatchDecode(const std::vector<SequenceInput>& inputs);

    /// Returns the vocabulary size (valid after LoadModel).
    int VocabSize() const noexcept;

    /// BOS / EOS token IDs (valid after LoadModel).
    int32_t BOS() const noexcept;
    int32_t EOS() const noexcept;

    /// Returns true if token signals end-of-generation (EOS, EOT, etc.).
    bool IsEOG(int32_t token) const noexcept;

private:
    llama_model*   model_   = nullptr;
    llama_context* ctx_     = nullptr;
    llama_batch    batch_   = {};
    bool           batch_initialized_ = false;
    int            n_batch_ = 0;
    int            vocab_size_ = 0;
};

} // namespace infergo
