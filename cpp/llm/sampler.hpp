#pragma once

#include <cstdint>
#include <vector>

namespace infergo {

/// Sampling parameters — passed to SampleToken().
struct SamplerParams {
    float    temperature    = 1.0f;  // 0 = greedy; >0 = stochastic
    float    top_p          = 1.0f;  // nucleus: keep tokens whose cumulative prob ≤ top_p
    int      top_k          = 0;     // keep only top-k logits (0 = disabled)
    float    repeat_penalty = 1.0f;  // penalise already-generated tokens (1.0 = no penalty)
};

/// Sample the next token from a logit distribution.
///
/// Applies the enabled filters in order:
///   1. Repetition penalty (on raw logits)
///   2. Temperature scaling
///   3. Softmax → probability distribution
///   4. Top-k truncation
///   5. Top-p (nucleus) truncation
///   6. Multinomial sample  (or argmax if temperature == 0)
///
/// logits:      [vocab_size] raw logits from the model
/// vocab_size:  length of the logits array
/// prev_tokens: already-generated token IDs used for repetition penalty
/// params:      sampling configuration
/// Returns the sampled token ID.
int32_t SampleToken(const float*             logits,
                    int                      vocab_size,
                    const std::vector<int32_t>& prev_tokens,
                    const SamplerParams&     params);

/// Convenience: greedy argmax with no penalties.
int32_t SampleGreedy(const float* logits, int vocab_size);

} // namespace infergo
