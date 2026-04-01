#include "postprocess.hpp"
#include "tensor.hpp"
#include "infer_api.h"
#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>

// Helper: allocate 1-D float32 tensor from a vector
static infergo::Tensor* make_vec(const std::vector<float>& vals) {
    const int shape[] = {static_cast<int>(vals.size())};
    infergo::Tensor* t = infergo::tensor_alloc_cpu(shape, 1, INFER_DTYPE_FLOAT32);
    std::copy(vals.begin(), vals.end(), static_cast<float*>(t->data));
    return t;
}

static float l2_norm(const infergo::Tensor* t) {
    const float* p = static_cast<const float*>(t->data);
    const int n = static_cast<int>(t->nelements());
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += static_cast<double>(p[i]) * static_cast<double>(p[i]);
    return static_cast<float>(std::sqrt(s));
}

// ─── NormalizeEmbedding ───────────────────────────────────────────────────────

TEST(NormalizeEmbedding, NullTensorThrows) {
    EXPECT_THROW(infergo::normalize_embedding(nullptr), std::invalid_argument);
}

TEST(NormalizeEmbedding, L2NormIsOne) {
    auto* t = make_vec({3.0f, 4.0f});  // norm = 5 → [0.6, 0.8]
    infergo::normalize_embedding(t);
    EXPECT_NEAR(l2_norm(t), 1.0f, 1e-6f);
    infergo::tensor_free(t);
}

TEST(NormalizeEmbedding, ValuesScaledCorrectly) {
    auto* t = make_vec({3.0f, 4.0f});
    infergo::normalize_embedding(t);
    const float* p = static_cast<const float*>(t->data);
    EXPECT_NEAR(p[0], 0.6f, 1e-6f);
    EXPECT_NEAR(p[1], 0.8f, 1e-6f);
    infergo::tensor_free(t);
}

TEST(NormalizeEmbedding, ZeroVectorUnchanged) {
    auto* t = make_vec({0.0f, 0.0f, 0.0f});
    infergo::normalize_embedding(t);  // must not crash or NaN
    const float* p = static_cast<const float*>(t->data);
    EXPECT_FLOAT_EQ(p[0], 0.0f);
    EXPECT_FLOAT_EQ(p[1], 0.0f);
    EXPECT_FLOAT_EQ(p[2], 0.0f);
    infergo::tensor_free(t);
}

TEST(NormalizeEmbedding, UnitVectorUnchanged) {
    auto* t = make_vec({1.0f, 0.0f, 0.0f});
    infergo::normalize_embedding(t);
    EXPECT_NEAR(l2_norm(t), 1.0f, 1e-6f);
    const float* p = static_cast<const float*>(t->data);
    EXPECT_NEAR(p[0], 1.0f, 1e-6f);
    infergo::tensor_free(t);
}

TEST(NormalizeEmbedding, LargeEmbeddingNormIsOne) {
    // 512-dim random-ish embedding
    const int N = 512;
    const int shape[] = {N};
    infergo::Tensor* t = infergo::tensor_alloc_cpu(shape, 1, INFER_DTYPE_FLOAT32);
    float* p = static_cast<float*>(t->data);
    for (int i = 0; i < N; ++i) p[i] = static_cast<float>((i % 7) - 3);
    infergo::normalize_embedding(t);
    EXPECT_NEAR(l2_norm(t), 1.0f, 1e-5f);
    infergo::tensor_free(t);
}
