#include <gtest/gtest.h>
#include "tensor.hpp"

using infergo::Tensor;

// T-02+ will add allocation and correctness tests here.
// These tests verify dtype_size() which is implemented in T-01's stub.

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
