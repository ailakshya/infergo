#include "preprocess.hpp"
#include "tensor.hpp"

#include <gtest/gtest.h>
#include <cstring>

using namespace infergo;

// Helper: create a [H, W, 3] float32 tensor filled with value
static Tensor* make_hwc(int H, int W, float fill) {
    const int shape[] = {H, W, 3};
    Tensor* t = tensor_alloc_cpu(shape, 3, INFER_DTYPE_FLOAT32);
    if (t == nullptr) return nullptr;
    float* data = static_cast<float*>(t->data);
    const int n = H * W * 3;
    for (int i = 0; i < n; ++i) data[i] = fill;
    return t;
}

TEST(Letterbox, NullInputThrows) {
    EXPECT_THROW(letterbox(nullptr, 640, 640), std::runtime_error);
}

TEST(Letterbox, InvalidTargetThrows) {
    Tensor* t = make_hwc(4, 4, 100.0f);
    ASSERT_NE(t, nullptr);
    EXPECT_THROW(letterbox(t, 0, 640), std::runtime_error);
    EXPECT_THROW(letterbox(t, 640, 0), std::runtime_error);
    tensor_free(t);
}

TEST(Letterbox, OutputShape) {
    Tensor* src = make_hwc(480, 640, 50.0f);
    ASSERT_NE(src, nullptr);
    Tensor* out = letterbox(src, 640, 640);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->ndim, 3);
    EXPECT_EQ(out->shape[0], 640);
    EXPECT_EQ(out->shape[1], 640);
    EXPECT_EQ(out->shape[2], 3);
    tensor_free(src);
    tensor_free(out);
}

TEST(Letterbox, PaddingIs114) {
    // 1×1 image letterboxed to 3×3 — corners must be padding (114)
    Tensor* src = make_hwc(1, 1, 200.0f);
    ASSERT_NE(src, nullptr);
    Tensor* out = letterbox(src, 3, 3);
    ASSERT_NE(out, nullptr);
    const float* data = static_cast<const float*>(out->data);
    // Top-left pixel (0,0) is padding
    EXPECT_FLOAT_EQ(data[0], 114.0f);
    tensor_free(src);
    tensor_free(out);
}

TEST(Letterbox, SquareImageUnchangedSize) {
    // Square input → no padding, just resize
    Tensor* src = make_hwc(100, 100, 77.0f);
    ASSERT_NE(src, nullptr);
    Tensor* out = letterbox(src, 200, 200);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->shape[0], 200);
    EXPECT_EQ(out->shape[1], 200);
    tensor_free(src);
    tensor_free(out);
}
