#include <gtest/gtest.h>
#include "tensor.hpp"

using infergo::Tensor;

// ─── dtype_size ──────────────────────────────────────────────────────────────

TEST(TensorDtypeSize, KnownDtypes) {
    EXPECT_EQ(Tensor::dtype_size(0), static_cast<size_t>(4));  // FLOAT32
    EXPECT_EQ(Tensor::dtype_size(1), static_cast<size_t>(2));  // FLOAT16
    EXPECT_EQ(Tensor::dtype_size(2), static_cast<size_t>(2));  // BFLOAT16
    EXPECT_EQ(Tensor::dtype_size(3), static_cast<size_t>(4));  // INT32
    EXPECT_EQ(Tensor::dtype_size(4), static_cast<size_t>(8));  // INT64
    EXPECT_EQ(Tensor::dtype_size(5), static_cast<size_t>(1));  // UINT8
    EXPECT_EQ(Tensor::dtype_size(6), static_cast<size_t>(1));  // BOOL
}

TEST(TensorDtypeSize, UnknownDtype) {
    EXPECT_EQ(Tensor::dtype_size(99), static_cast<size_t>(0));
}

// ─── nelements ───────────────────────────────────────────────────────────────

TEST(TensorNelements, Uninitialised) {
    Tensor t;
    EXPECT_EQ(t.nelements(), static_cast<size_t>(0));
}

TEST(TensorNelements, NullShape) {
    Tensor t;
    t.ndim = 3;
    t.shape = nullptr;
    EXPECT_EQ(t.nelements(), static_cast<size_t>(0));
}

TEST(TensorNelements, ThreeDims) {
    int shape[] = {2, 3, 4};
    Tensor t;
    t.shape = shape;
    t.ndim  = 3;
    EXPECT_EQ(t.nelements(), static_cast<size_t>(24));
}

TEST(TensorNelements, OneDim) {
    int shape[] = {100};
    Tensor t;
    t.shape = shape;
    t.ndim  = 1;
    EXPECT_EQ(t.nelements(), static_cast<size_t>(100));
}

TEST(TensorNelements, ZeroDimension) {
    int shape[] = {2, 0, 4};
    Tensor t;
    t.shape = shape;
    t.ndim  = 3;
    EXPECT_EQ(t.nelements(), static_cast<size_t>(0));
}

// ─── compute_nbytes ──────────────────────────────────────────────────────────

TEST(TensorComputeNbytes, Float32_234) {
    int shape[] = {2, 3, 4};
    // 2*3*4 = 24 elements, 4 bytes each = 96
    EXPECT_EQ(Tensor::compute_nbytes(shape, 3, 0), static_cast<size_t>(96));
}

TEST(TensorComputeNbytes, Int64_10) {
    int shape[] = {10};
    // 10 elements, 8 bytes each = 80
    EXPECT_EQ(Tensor::compute_nbytes(shape, 1, 4), static_cast<size_t>(80));
}

TEST(TensorComputeNbytes, NullShape) {
    EXPECT_EQ(Tensor::compute_nbytes(nullptr, 3, 0), static_cast<size_t>(0));
}

TEST(TensorComputeNbytes, ZeroNdim) {
    int shape[] = {2, 3};
    EXPECT_EQ(Tensor::compute_nbytes(shape, 0, 0), static_cast<size_t>(0));
}

TEST(TensorComputeNbytes, UnknownDtype) {
    int shape[] = {2, 3};
    EXPECT_EQ(Tensor::compute_nbytes(shape, 2, 99), static_cast<size_t>(0));
}

TEST(TensorComputeNbytes, NegativeDimension) {
    int shape[] = {2, -1, 4};
    EXPECT_EQ(Tensor::compute_nbytes(shape, 3, 0), static_cast<size_t>(0));
}
