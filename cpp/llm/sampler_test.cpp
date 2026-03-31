#include "sampler.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <gtest/gtest.h>

using namespace infergo;

// ─── Helpers ──────────────────────────────────────────────────────────────────

// Build a logit array where token `winner` has a very high score.
static std::vector<float> SpikedLogits(int vocab_size, int winner, float spike = 100.0f) {
    std::vector<float> logits(static_cast<size_t>(vocab_size), 0.0f);
    logits[static_cast<size_t>(winner)] = spike;
    return logits;
}

// ─── SampleGreedy ─────────────────────────────────────────────────────────────

TEST(SampleGreedy, ReturnsArgmax) {
    std::vector<float> logits = {0.1f, 0.5f, 0.9f, 0.2f};
    EXPECT_EQ(SampleGreedy(logits.data(), 4), 2);
}

TEST(SampleGreedy, SingleToken) {
    std::vector<float> logits = {3.14f};
    EXPECT_EQ(SampleGreedy(logits.data(), 1), 0);
}

TEST(SampleGreedy, NullThrows) {
    EXPECT_THROW(SampleGreedy(nullptr, 4), std::invalid_argument);
}

TEST(SampleGreedy, ZeroVocabThrows) {
    std::vector<float> logits = {1.0f};
    EXPECT_THROW(SampleGreedy(logits.data(), 0), std::invalid_argument);
}

// ─── SampleToken — greedy (temperature=0) ────────────────────────────────────

TEST(SampleToken, GreedyIsDeterministic) {
    std::vector<float> logits = {0.1f, 0.9f, 0.3f, 0.7f};
    SamplerParams p;
    p.temperature = 0.0f;

    const int first = SampleToken(logits.data(), 4, {}, p);
    EXPECT_EQ(first, 1);
    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(SampleToken(logits.data(), 4, {}, p), first);
    }
}

TEST(SampleToken, GreedyPicksHighestLogit) {
    const int vocab = 100;
    auto logits = SpikedLogits(vocab, 42);
    SamplerParams p;
    p.temperature = 0.0f;
    EXPECT_EQ(SampleToken(logits.data(), vocab, {}, p), 42);
}

// ─── SampleToken — temperature ────────────────────────────────────────────────

TEST(SampleToken, TemperatureGivesVariedOutputs) {
    // Uniform-ish logits — with temperature > 0, repeated sampling should
    // produce more than one distinct token across many trials.
    const int vocab = 10;
    std::vector<float> logits(static_cast<size_t>(vocab), 1.0f);
    SamplerParams p;
    p.temperature = 1.0f;

    std::set<int32_t> seen;
    for (int i = 0; i < 200; ++i) {
        seen.insert(SampleToken(logits.data(), vocab, {}, p));
    }
    EXPECT_GT(static_cast<int>(seen.size()), 1)
        << "temperature=1 with uniform logits should sample multiple tokens";
}

TEST(SampleToken, HighTemperatureMoreDiverse) {
    const int vocab = 10;
    std::vector<float> logits(static_cast<size_t>(vocab), 1.0f);

    SamplerParams low;
    low.temperature = 0.1f;

    SamplerParams high;
    high.temperature = 10.0f;

    std::set<int32_t> low_seen, high_seen;
    for (int i = 0; i < 500; ++i) {
        low_seen.insert(SampleToken(logits.data(), vocab, {}, low));
        high_seen.insert(SampleToken(logits.data(), vocab, {}, high));
    }
    // High temperature must be at least as diverse as low temperature
    EXPECT_GE(high_seen.size(), low_seen.size());
}

TEST(SampleToken, SpikedTemperatureConverges) {
    // Even with temperature=1, a very spiked distribution should almost always
    // pick the winner (spike=100 → e^100 >> e^0)
    const int vocab = 100;
    auto logits = SpikedLogits(vocab, 7, 100.0f);
    SamplerParams p;
    p.temperature = 1.0f;

    int hits = 0;
    for (int i = 0; i < 50; ++i) {
        if (SampleToken(logits.data(), vocab, {}, p) == 7) ++hits;
    }
    EXPECT_EQ(hits, 50) << "spike=100 should always pick token 7";
}

// ─── SampleToken — top-k ──────────────────────────────────────────────────────

TEST(SampleToken, TopKLimitsVocab) {
    // Logits: tokens 0-9 have score 1, token 5 has score 10 (best)
    const int vocab = 10;
    std::vector<float> logits(static_cast<size_t>(vocab), 1.0f);
    logits[5] = 10.0f;

    SamplerParams p;
    p.temperature = 1.0f;
    p.top_k       = 3;  // only top-3 tokens are candidates

    // With top-k=3, tokens outside the top-3 must never appear
    std::set<int32_t> seen;
    for (int i = 0; i < 500; ++i) {
        seen.insert(SampleToken(logits.data(), vocab, {}, p));
    }
    EXPECT_LE(static_cast<int>(seen.size()), 3)
        << "top-k=3 should produce at most 3 distinct tokens";
}

TEST(SampleToken, TopK1IsGreedy) {
    const int vocab = 20;
    auto logits = SpikedLogits(vocab, 13);
    SamplerParams p;
    p.temperature = 1.0f;
    p.top_k       = 1;

    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(SampleToken(logits.data(), vocab, {}, p), 13);
    }
}

// ─── SampleToken — top-p ──────────────────────────────────────────────────────

TEST(SampleToken, TopPLimitsVocab) {
    // Spike token 0 heavily — top-p=0.99 should almost always pick only token 0
    const int vocab = 100;
    auto logits = SpikedLogits(vocab, 0, 50.0f);
    SamplerParams p;
    p.temperature = 1.0f;
    p.top_p       = 0.99f;

    for (int i = 0; i < 30; ++i) {
        EXPECT_EQ(SampleToken(logits.data(), vocab, {}, p), 0)
            << "spike=50 with top-p=0.99 should always pick token 0";
    }
}

// ─── SampleToken — repetition penalty ────────────────────────────────────────

TEST(SampleToken, RepetitionPenaltyReducesPreviousToken) {
    // Token 0 has the highest logit initially.
    // With a heavy penalty and token 0 in prev_tokens, token 1 should win.
    std::vector<float> logits = {10.0f, 5.0f, 1.0f};
    SamplerParams p;
    p.temperature    = 0.0f;   // greedy
    p.repeat_penalty = 100.0f; // heavy penalty

    // Without penalty: token 0
    EXPECT_EQ(SampleToken(logits.data(), 3, {}, p), 0);

    // With token 0 penalised: logits[0] = 10/100 = 0.1 < 5 → token 1 wins
    EXPECT_EQ(SampleToken(logits.data(), 3, {0}, p), 1);
}

TEST(SampleToken, NoPenaltyOnUnseenTokens) {
    std::vector<float> logits = {10.0f, 5.0f, 1.0f};
    SamplerParams p;
    p.temperature    = 0.0f;
    p.repeat_penalty = 100.0f;

    // Token 2 penalised but token 0 not in prev → token 0 still wins
    EXPECT_EQ(SampleToken(logits.data(), 3, {2}, p), 0);
}

// ─── Edge cases ───────────────────────────────────────────────────────────────

TEST(SampleToken, NullThrows) {
    SamplerParams p;
    EXPECT_THROW(SampleToken(nullptr, 4, {}, p), std::invalid_argument);
}

TEST(SampleToken, SingleTokenAlwaysReturnsZero) {
    std::vector<float> logits = {42.0f};
    SamplerParams p;
    p.temperature = 1.0f;
    EXPECT_EQ(SampleToken(logits.data(), 1, {}, p), 0);
}
