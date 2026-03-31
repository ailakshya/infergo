#include "sampler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace infergo {

// ─── SampleGreedy ─────────────────────────────────────────────────────────────

int32_t SampleGreedy(const float* logits, int vocab_size) {
    if (logits == nullptr || vocab_size <= 0) {
        throw std::invalid_argument("SampleGreedy: invalid logits or vocab_size");
    }
    return static_cast<int32_t>(
        std::max_element(logits, logits + vocab_size) - logits
    );
}

// ─── SampleToken ──────────────────────────────────────────────────────────────

int32_t SampleToken(const float*                logits,
                    int                         vocab_size,
                    const std::vector<int32_t>& prev_tokens,
                    const SamplerParams&        params)
{
    if (logits == nullptr || vocab_size <= 0) {
        throw std::invalid_argument("SampleToken: invalid logits or vocab_size");
    }

    // Step 1: copy logits so we can mutate them
    std::vector<float> scores(logits, logits + vocab_size);

    // Step 2: repetition penalty — divide logits of previously seen tokens
    if (params.repeat_penalty != 1.0f && !prev_tokens.empty()) {
        for (int32_t id : prev_tokens) {
            if (id >= 0 && id < vocab_size) {
                // Standard formulation: positive logits are divided, negative multiplied
                if (scores[static_cast<size_t>(id)] > 0.0f) {
                    scores[static_cast<size_t>(id)] /= params.repeat_penalty;
                } else {
                    scores[static_cast<size_t>(id)] *= params.repeat_penalty;
                }
            }
        }
    }

    // Step 3: greedy path
    if (params.temperature <= 0.0f) {
        return static_cast<int32_t>(
            std::max_element(scores.begin(), scores.end()) - scores.begin()
        );
    }

    // Step 4: temperature scaling
    const float inv_temp = 1.0f / params.temperature;
    for (float& s : scores) s *= inv_temp;

    // Step 5: build (index, score) pairs for top-k / top-p filtering
    std::vector<std::pair<float, int32_t>> candidates;
    candidates.reserve(static_cast<size_t>(vocab_size));
    for (int i = 0; i < vocab_size; ++i) {
        candidates.emplace_back(scores[static_cast<size_t>(i)], static_cast<int32_t>(i));
    }

    // Step 6: top-k — keep only top-k by score
    int effective_k = vocab_size;
    if (params.top_k > 0 && params.top_k < vocab_size) {
        effective_k = params.top_k;
    }
    // Partial sort: bring the top `effective_k` to the front
    std::partial_sort(candidates.begin(),
                      candidates.begin() + effective_k,
                      candidates.end(),
                      [](const auto& a, const auto& b){ return a.first > b.first; });
    candidates.resize(static_cast<size_t>(effective_k));

    // Step 7: softmax over remaining candidates
    const float max_score = candidates[0].first;  // sorted descending
    float sum = 0.0f;
    for (auto& [s, _] : candidates) {
        s = std::exp(s - max_score);
        sum += s;
    }
    for (auto& [s, _] : candidates) s /= sum;

    // Step 8: top-p (nucleus) — keep smallest prefix whose cumulative prob ≥ top_p
    if (params.top_p < 1.0f) {
        float cumsum = 0.0f;
        size_t cutoff = candidates.size();
        for (size_t i = 0; i < candidates.size(); ++i) {
            cumsum += candidates[i].first;
            if (cumsum >= params.top_p) {
                cutoff = i + 1;
                break;
            }
        }
        candidates.resize(cutoff);
        // Re-normalise after truncation
        sum = 0.0f;
        for (auto& [s, _] : candidates) sum += s;
        for (auto& [s, _] : candidates) s /= sum;
    }

    // Step 9: multinomial sample using thread-local RNG
    thread_local std::mt19937 rng(std::random_device{}());
    std::vector<float> probs;
    probs.reserve(candidates.size());
    for (const auto& [s, _] : candidates) probs.push_back(s);
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return candidates[static_cast<size_t>(dist(rng))].second;
}

} // namespace infergo
