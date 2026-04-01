#include "preprocess.hpp"
#include "tensor.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace infergo;

static Tensor* make_hwc(int H, int W, float r, float g, float b) {
    const int shape[] = {H, W, 3};
    Tensor* t = tensor_alloc_cpu(shape, 3, INFER_DTYPE_FLOAT32);
    if (t == nullptr) return nullptr;
    float* data = static_cast<float*>(t->data);
    for (int i = 0; i < H * W; ++i) {
        data[i * 3 + 0] = r;
        data[i * 3 + 1] = g;
        data[i * 3 + 2] = b;
    }
    return t;
}

TEST(Normalize, NullInputThrows) {
    const float mean[] = {0.f, 0.f, 0.f};
    const float std[]  = {1.f, 1.f, 1.f};
    EXPECT_THROW(normalize(nullptr, 255.0f, mean, std), std::runtime_error);
}

TEST(Normalize, OutputIsCHW) {
    Tensor* src = make_hwc(2, 3, 128.0f, 64.0f, 32.0f);
    ASSERT_NE(src, nullptr);

    const float mean[]   = {0.485f, 0.456f, 0.406f};
    const float std_dev[] = {0.229f, 0.224f, 0.225f};
    Tensor* out = normalize(src, 255.0f, mean, std_dev);
    ASSERT_NE(out, nullptr);

    // Output must be [3, H, W]
    EXPECT_EQ(out->ndim, 3);
    EXPECT_EQ(out->shape[0], 3);
    EXPECT_EQ(out->shape[1], 2);
    EXPECT_EQ(out->shape[2], 3);

    tensor_free(src);
    tensor_free(out);
}

TEST(Normalize, KnownValues) {
    // Single pixel [H=1, W=1, C=3], values [255, 0, 128]
    Tensor* src = make_hwc(1, 1, 255.0f, 0.0f, 128.0f);
    ASSERT_NE(src, nullptr);

    // mean=0, std=1, scale=255 → output = pixel/255
    const float mean[]   = {0.f, 0.f, 0.f};
    const float std_dev[] = {1.f, 1.f, 1.f};
    Tensor* out = normalize(src, 255.0f, mean, std_dev);
    ASSERT_NE(out, nullptr);

    const float* data = static_cast<const float*>(out->data);
    EXPECT_NEAR(data[0], 1.0f,           1e-5f);   // R channel
    EXPECT_NEAR(data[1], 0.0f,           1e-5f);   // G channel
    EXPECT_NEAR(data[2], 128.0f / 255.f, 1e-5f);   // B channel

    tensor_free(src);
    tensor_free(out);
}

TEST(Normalize, ImageNetNormalization) {
    // Known ImageNet norm: pixel=128, scale=255, mean=0.485, std=0.229
    // expected = (128/255 - 0.485) / 0.229
    Tensor* src = make_hwc(1, 1, 128.0f, 128.0f, 128.0f);
    ASSERT_NE(src, nullptr);

    const float mean[]    = {0.485f, 0.456f, 0.406f};
    const float std_dev[] = {0.229f, 0.224f, 0.225f};
    Tensor* out = normalize(src, 255.0f, mean, std_dev);
    ASSERT_NE(out, nullptr);

    const float* data = static_cast<const float*>(out->data);
    const float expected_r = (128.0f / 255.0f - 0.485f) / 0.229f;
    EXPECT_NEAR(data[0], expected_r, 1e-4f);

    tensor_free(src);
    tensor_free(out);
}
