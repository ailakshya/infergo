#include "postprocess.hpp"
#include "tensor.hpp"
#include "infer_api.h"
#include <gtest/gtest.h>
#include <stdexcept>

// Helper: build a [1, N, 4+C] float32 tensor from a flat row-major vector
// rows[i] = {cx, cy, w, h, cls0, cls1, ...}
static infergo::Tensor* make_preds(const std::vector<std::vector<float>>& rows) {
    const int N = static_cast<int>(rows.size());
    const int stride = static_cast<int>(rows[0].size());
    const int shape[] = {1, N, stride};
    infergo::Tensor* t = infergo::tensor_alloc_cpu(shape, 3, INFER_DTYPE_FLOAT32);
    float* p = static_cast<float*>(t->data);
    for (const auto& row : rows)
        for (float v : row) *p++ = v;
    return t;
}

// ─── NMS ─────────────────────────────────────────────────────────────────────

TEST(NMS, NullTensorThrows) {
    EXPECT_THROW(infergo::nms(nullptr, 0.5f, 0.45f), std::invalid_argument);
}

TEST(NMS, NoDetectionsAboveThresh) {
    // All scores below 0.5
    auto* t = make_preds({
        {100, 100, 50, 50, 0.1f, 0.2f},
        {200, 200, 60, 60, 0.3f, 0.1f},
    });
    auto result = infergo::nms(t, 0.5f, 0.45f);
    infergo::tensor_free(t);
    EXPECT_TRUE(result.empty());
}

TEST(NMS, SingleBoxKept) {
    auto* t = make_preds({
        {100, 100, 50, 50, 0.9f, 0.1f},
    });
    auto result = infergo::nms(t, 0.5f, 0.45f);
    infergo::tensor_free(t);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].class_idx, 0);
    EXPECT_NEAR(result[0].confidence, 0.9f, 1e-5f);
}

TEST(NMS, OverlappingBoxesSuppressed) {
    // Two near-identical boxes → only the higher-confidence one kept
    auto* t = make_preds({
        {100, 100, 80, 80, 0.95f, 0.1f},  // box A (higher conf)
        {102, 102, 80, 80, 0.80f, 0.1f},  // box B (almost same position)
    });
    auto result = infergo::nms(t, 0.5f, 0.45f);
    infergo::tensor_free(t);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0].confidence, 0.95f, 1e-5f);
}

TEST(NMS, NonOverlappingBoxesBothKept) {
    // Two boxes far apart → both kept
    auto* t = make_preds({
        {100,  100, 40, 40, 0.9f, 0.1f},
        {500,  500, 40, 40, 0.8f, 0.1f},
    });
    auto result = infergo::nms(t, 0.5f, 0.45f);
    infergo::tensor_free(t);
    EXPECT_EQ(result.size(), 2u);
}

TEST(NMS, DifferentClassesNotSuppressed) {
    // Same position but different classes → both kept
    auto* t = make_preds({
        {100, 100, 80, 80, 0.95f, 0.05f},  // class 0
        {100, 100, 80, 80, 0.05f, 0.90f},  // class 1
    });
    auto result = infergo::nms(t, 0.5f, 0.45f);
    infergo::tensor_free(t);
    ASSERT_EQ(result.size(), 2u);
    // Must have one of each class
    bool has0 = false, has1 = false;
    for (const auto& b : result) {
        if (b.class_idx == 0) has0 = true;
        if (b.class_idx == 1) has1 = true;
    }
    EXPECT_TRUE(has0);
    EXPECT_TRUE(has1);
}

TEST(NMS, CxCyWhConvertsToX1Y1X2Y2) {
    // cx=100, cy=100, w=60, h=40 → x1=70, y1=80, x2=130, y2=120
    auto* t = make_preds({
        {100, 100, 60, 40, 0.9f},
    });
    auto result = infergo::nms(t, 0.5f, 0.45f);
    infergo::tensor_free(t);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0].x1, 70.0f,  1e-4f);
    EXPECT_NEAR(result[0].y1, 80.0f,  1e-4f);
    EXPECT_NEAR(result[0].x2, 130.0f, 1e-4f);
    EXPECT_NEAR(result[0].y2, 120.0f, 1e-4f);
}

TEST(NMS, SortedByConfidenceDescending) {
    auto* t = make_preds({
        {100, 100, 30, 30, 0.7f},
        {200, 200, 30, 30, 0.9f},
        {300, 300, 30, 30, 0.8f},
    });
    auto result = infergo::nms(t, 0.5f, 0.45f);
    infergo::tensor_free(t);
    ASSERT_EQ(result.size(), 3u);
    EXPECT_GE(result[0].confidence, result[1].confidence);
    EXPECT_GE(result[1].confidence, result[2].confidence);
}
