#include "preprocess.hpp"
#include "tensor.hpp"

#include <gtest/gtest.h>

using namespace infergo;

static Tensor* make_chw(int C, int H, int W, float fill) {
    const int shape[] = {C, H, W};
    Tensor* t = tensor_alloc_cpu(shape, 3, INFER_DTYPE_FLOAT32);
    if (t == nullptr) return nullptr;
    float* data = static_cast<float*>(t->data);
    const int n = C * H * W;
    for (int i = 0; i < n; ++i) data[i] = fill;
    return t;
}

TEST(StackBatch, EmptyThrows) {
    EXPECT_THROW(stack_batch({}), std::runtime_error);
}

TEST(StackBatch, NullElementThrows) {
    std::vector<const Tensor*> v = {nullptr};
    EXPECT_THROW(stack_batch(v), std::runtime_error);
}

TEST(StackBatch, SingleTensor) {
    Tensor* t = make_chw(3, 4, 5, 1.0f);
    ASSERT_NE(t, nullptr);
    Tensor* out = stack_batch({t});
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->ndim, 4);
    EXPECT_EQ(out->shape[0], 1);
    EXPECT_EQ(out->shape[1], 3);
    EXPECT_EQ(out->shape[2], 4);
    EXPECT_EQ(out->shape[3], 5);
    tensor_free(t);
    tensor_free(out);
}

TEST(StackBatch, FourTensors) {
    Tensor* t0 = make_chw(3, 640, 640, 0.0f);
    Tensor* t1 = make_chw(3, 640, 640, 1.0f);
    Tensor* t2 = make_chw(3, 640, 640, 2.0f);
    Tensor* t3 = make_chw(3, 640, 640, 3.0f);
    ASSERT_NE(t0, nullptr);

    Tensor* out = stack_batch({t0, t1, t2, t3});
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->ndim, 4);
    EXPECT_EQ(out->shape[0], 4);   // N
    EXPECT_EQ(out->shape[1], 3);   // C
    EXPECT_EQ(out->shape[2], 640); // H
    EXPECT_EQ(out->shape[3], 640); // W

    // Verify that each batch slot has correct fill value
    const float* data = static_cast<const float*>(out->data);
    const int per = 3 * 640 * 640;
    for (int n = 0; n < 4; ++n) {
        EXPECT_FLOAT_EQ(data[n * per], static_cast<float>(n)) << "batch[" << n << "]";
    }

    tensor_free(t0); tensor_free(t1); tensor_free(t2); tensor_free(t3);
    tensor_free(out);
}

TEST(StackBatch, MismatchedShapeThrows) {
    Tensor* a = make_chw(3, 4, 5, 0.f);
    Tensor* b = make_chw(3, 4, 6, 0.f);  // different W
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_THROW(stack_batch({a, b}), std::runtime_error);
    tensor_free(a);
    tensor_free(b);
}
