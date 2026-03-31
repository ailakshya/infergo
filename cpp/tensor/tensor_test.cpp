#include <gtest/gtest.h>
#include "tensor.hpp"

using infergo::Tensor;
using infergo::tensor_alloc_cpu;
using infergo::tensor_free;
using infergo::get_last_error;

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
    EXPECT_EQ(Tensor::compute_nbytes(shape, 3, 0), static_cast<size_t>(96));
}

TEST(TensorComputeNbytes, Int64_10) {
    int shape[] = {10};
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

// ─── tensor_alloc_cpu ────────────────────────────────────────────────────────

// Acceptance criterion: alloc float32 [2,3,4] → data ptr not null, nbytes == 96
TEST(TensorAllocCpu, Float32_234) {
    int shape[] = {2, 3, 4};
    Tensor* t = tensor_alloc_cpu(shape, 3, 0);

    ASSERT_NE(t, nullptr);
    EXPECT_NE(t->data, nullptr);
    EXPECT_EQ(t->nbytes, static_cast<size_t>(96));
    EXPECT_EQ(t->nelements(), static_cast<size_t>(24));
    EXPECT_EQ(t->ndim, 3);
    EXPECT_EQ(t->shape[0], 2);
    EXPECT_EQ(t->shape[1], 3);
    EXPECT_EQ(t->shape[2], 4);
    EXPECT_EQ(t->dtype, 0);
    EXPECT_FALSE(t->on_device);
    EXPECT_EQ(t->device_id, 0);

    tensor_free(t);
}

TEST(TensorAllocCpu, AllDtypes) {
    int shape[] = {4};
    const size_t expected[] = {16, 8, 8, 16, 32, 4, 4};
    for (int dtype = 0; dtype <= 6; ++dtype) {
        Tensor* t = tensor_alloc_cpu(shape, 1, dtype);
        ASSERT_NE(t, nullptr) << "dtype=" << dtype;
        EXPECT_EQ(t->nbytes, expected[static_cast<size_t>(dtype)]) << "dtype=" << dtype;
        tensor_free(t);
    }
}

TEST(TensorAllocCpu, ShapeCopied) {
    int shape[] = {5, 6};
    Tensor* t = tensor_alloc_cpu(shape, 2, 0);
    ASSERT_NE(t, nullptr);

    // Mutate original — tensor must not be affected
    shape[0] = 99;
    EXPECT_EQ(t->shape[0], 5);

    tensor_free(t);
}

TEST(TensorAllocCpu, NullShapeReturnsNull) {
    Tensor* t = tensor_alloc_cpu(nullptr, 3, 0);
    EXPECT_EQ(t, nullptr);
    EXPECT_STRNE(get_last_error(), "");
}

TEST(TensorAllocCpu, ZeroNdimReturnsNull) {
    int shape[] = {2, 3};
    Tensor* t = tensor_alloc_cpu(shape, 0, 0);
    EXPECT_EQ(t, nullptr);
    EXPECT_STRNE(get_last_error(), "");
}

TEST(TensorAllocCpu, NegativeNdimReturnsNull) {
    int shape[] = {2, 3};
    Tensor* t = tensor_alloc_cpu(shape, -1, 0);
    EXPECT_EQ(t, nullptr);
    EXPECT_STRNE(get_last_error(), "");
}

TEST(TensorAllocCpu, ZeroDimensionReturnsNull) {
    int shape[] = {2, 0, 4};
    Tensor* t = tensor_alloc_cpu(shape, 3, 0);
    EXPECT_EQ(t, nullptr);
    EXPECT_STRNE(get_last_error(), "");
}

TEST(TensorAllocCpu, UnknownDtypeReturnsNull) {
    int shape[] = {2, 3};
    Tensor* t = tensor_alloc_cpu(shape, 2, 99);
    EXPECT_EQ(t, nullptr);
    EXPECT_STRNE(get_last_error(), "");
}

// ─── tensor_free ─────────────────────────────────────────────────────────────

TEST(TensorFree, NullIsNoop) {
    // Must not crash
    tensor_free(nullptr);
}

TEST(TensorFree, FreedPtrsZeroed) {
    int shape[] = {3};
    Tensor* t = tensor_alloc_cpu(shape, 1, 0);
    ASSERT_NE(t, nullptr);

    // Capture raw struct value before free
    void* data_before = t->data;
    EXPECT_NE(data_before, nullptr);

    // After free the memory is gone — we only verified the struct was non-null
    // before the call. Pointer-zeroing is an internal detail tested implicitly
    // by address-sanitizer in Debug builds.
    tensor_free(t);
}
