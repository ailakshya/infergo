#include "postprocess.hpp"
#include "tensor.hpp"
#include "infer_api.h"
#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>

// Helper: allocate a 1-D float32 CPU tensor from a vector
static infergo::Tensor* make_logits(const std::vector<float>& vals) {
    const int shape[] = {static_cast<int>(vals.size())};
    infergo::Tensor* t = infergo::tensor_alloc_cpu(shape, 1, INFER_DTYPE_FLOAT32);
    std::copy(vals.begin(), vals.end(), static_cast<float*>(t->data));
    return t;
}

// ─── Classify ────────────────────────────────────────────────────────────────

TEST(Classify, NullTensorThrows) {
    EXPECT_THROW(infergo::classify(nullptr, 1), std::invalid_argument);
}

TEST(Classify, NegativeTopKThrows) {
    auto* t = make_logits({1.0f, 2.0f, 3.0f});
    EXPECT_THROW(infergo::classify(t, 0), std::invalid_argument);
    infergo::tensor_free(t);
}

TEST(Classify, UniformLogitsEqualProbs) {
    // All same logit → softmax gives 1/N each
    const int N = 4;
    auto* t = make_logits({0.0f, 0.0f, 0.0f, 0.0f});
    auto results = infergo::classify(t, N);
    infergo::tensor_free(t);

    ASSERT_EQ(static_cast<int>(results.size()), N);
    for (const auto& r : results)
        EXPECT_NEAR(r.confidence, 1.0f / N, 1e-5f);
}

TEST(Classify, DominantLogitHighConfidence) {
    // One very large logit → that class gets near-1.0 probability
    auto* t = make_logits({100.0f, 0.0f, 0.0f, 0.0f});
    auto results = infergo::classify(t, 1);
    infergo::tensor_free(t);

    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].label_idx, 0);
    EXPECT_NEAR(results[0].confidence, 1.0f, 1e-5f);
}

TEST(Classify, ResultsSortedDescending) {
    auto* t = make_logits({1.0f, 5.0f, 3.0f, 2.0f});
    auto results = infergo::classify(t, 4);
    infergo::tensor_free(t);

    ASSERT_EQ(results.size(), 4u);
    EXPECT_EQ(results[0].label_idx, 1);  // highest logit
    EXPECT_EQ(results[1].label_idx, 2);
    EXPECT_EQ(results[2].label_idx, 3);
    EXPECT_EQ(results[3].label_idx, 0);

    // Probabilities must be non-increasing
    for (int i = 1; i < 4; ++i)
        EXPECT_GE(results[i-1].confidence, results[i].confidence);
}

TEST(Classify, TopKClipsToN) {
    // Ask for 10 but only 3 classes → get 3
    auto* t = make_logits({1.0f, 2.0f, 3.0f});
    auto results = infergo::classify(t, 10);
    infergo::tensor_free(t);

    EXPECT_EQ(results.size(), 3u);
}

TEST(Classify, ProbsSumToOne) {
    auto* t = make_logits({0.5f, 1.5f, -0.5f, 2.0f, 0.0f});
    auto results = infergo::classify(t, 5);
    infergo::tensor_free(t);

    float sum = 0.0f;
    for (const auto& r : results) sum += r.confidence;
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}
