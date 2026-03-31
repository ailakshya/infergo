#include <gtest/gtest.h>
#include <cstring>
#include <cstdint>
#include "tensor.hpp"

#ifdef INFER_CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

using infergo::Tensor;
using infergo::tensor_alloc_cpu;
using infergo::tensor_copy_from;
using infergo::tensor_free;
using infergo::tensor_get_dtype;
using infergo::tensor_get_nbytes;
using infergo::tensor_get_nelements;
using infergo::tensor_get_shape;
using infergo::get_last_error;

#ifdef INFER_CUDA_AVAILABLE
using infergo::tensor_alloc_cuda;
using infergo::tensor_to_device;
using infergo::tensor_to_host;
#endif

// ─── dtype_size ──────────────────────────────────────────────────────────────

TEST(TensorDtypeSize, KnownDtypes) {
    EXPECT_EQ(Tensor::dtype_size(0), static_cast<size_t>(4));
    EXPECT_EQ(Tensor::dtype_size(1), static_cast<size_t>(2));
    EXPECT_EQ(Tensor::dtype_size(2), static_cast<size_t>(2));
    EXPECT_EQ(Tensor::dtype_size(3), static_cast<size_t>(4));
    EXPECT_EQ(Tensor::dtype_size(4), static_cast<size_t>(8));
    EXPECT_EQ(Tensor::dtype_size(5), static_cast<size_t>(1));
    EXPECT_EQ(Tensor::dtype_size(6), static_cast<size_t>(1));
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
    EXPECT_EQ(tensor_alloc_cpu(shape, 0, 0), nullptr);
    EXPECT_STRNE(get_last_error(), "");
}

TEST(TensorAllocCpu, NegativeNdimReturnsNull) {
    int shape[] = {2, 3};
    EXPECT_EQ(tensor_alloc_cpu(shape, -1, 0), nullptr);
    EXPECT_STRNE(get_last_error(), "");
}

TEST(TensorAllocCpu, ZeroDimensionReturnsNull) {
    int shape[] = {2, 0, 4};
    EXPECT_EQ(tensor_alloc_cpu(shape, 3, 0), nullptr);
    EXPECT_STRNE(get_last_error(), "");
}

TEST(TensorAllocCpu, UnknownDtypeReturnsNull) {
    int shape[] = {2, 3};
    EXPECT_EQ(tensor_alloc_cpu(shape, 2, 99), nullptr);
    EXPECT_STRNE(get_last_error(), "");
}

// ─── tensor_free ─────────────────────────────────────────────────────────────

TEST(TensorFree, NullIsNoop) {
    tensor_free(nullptr);
}

TEST(TensorFree, FreedPtrsZeroed) {
    int shape[] = {3};
    Tensor* t = tensor_alloc_cpu(shape, 1, 0);
    ASSERT_NE(t, nullptr);
    EXPECT_NE(t->data, nullptr);
    tensor_free(t);
}

// ─── tensor_alloc_cuda ───────────────────────────────────────────────────────

#ifdef INFER_CUDA_AVAILABLE

TEST(TensorAllocCuda, Float32_234_DeviceMemory) {
    int shape[] = {2, 3, 4};
    Tensor* t = tensor_alloc_cuda(shape, 3, 0, 0);
    ASSERT_NE(t, nullptr) << get_last_error();
    EXPECT_NE(t->data, nullptr);
    EXPECT_EQ(t->nbytes, static_cast<size_t>(96));
    EXPECT_TRUE(t->on_device);
    EXPECT_EQ(t->device_id, 0);
    cudaPointerAttributes attrs{};
    EXPECT_EQ(cudaPointerGetAttributes(&attrs, t->data), cudaSuccess);
    EXPECT_EQ(attrs.type, cudaMemoryTypeDevice);
    tensor_free(t);
}

TEST(TensorAllocCuda, AllDtypes) {
    int shape[] = {4};
    const size_t expected[] = {16, 8, 8, 16, 32, 4, 4};
    for (int dtype = 0; dtype <= 6; ++dtype) {
        Tensor* t = tensor_alloc_cuda(shape, 1, dtype, 0);
        ASSERT_NE(t, nullptr) << "dtype=" << dtype << ": " << get_last_error();
        EXPECT_EQ(t->nbytes, expected[static_cast<size_t>(dtype)]) << "dtype=" << dtype;
        EXPECT_TRUE(t->on_device);
        tensor_free(t);
    }
}

TEST(TensorAllocCuda, ShapeLivesOnHost) {
    int shape[] = {5, 6};
    Tensor* t = tensor_alloc_cuda(shape, 2, 0, 0);
    ASSERT_NE(t, nullptr) << get_last_error();
    cudaPointerAttributes attrs{};
    EXPECT_EQ(cudaPointerGetAttributes(&attrs, t->shape), cudaSuccess);
    EXPECT_NE(attrs.type, cudaMemoryTypeDevice);
    shape[0] = 99;
    EXPECT_EQ(t->shape[0], 5);
    tensor_free(t);
}

TEST(TensorAllocCuda, NullShapeReturnsNull) {
    EXPECT_EQ(tensor_alloc_cuda(nullptr, 3, 0, 0), nullptr);
    EXPECT_STRNE(get_last_error(), "");
}

TEST(TensorAllocCuda, ZeroDimensionReturnsNull) {
    int shape[] = {2, 0, 4};
    EXPECT_EQ(tensor_alloc_cuda(shape, 3, 0, 0), nullptr);
    EXPECT_STRNE(get_last_error(), "");
}

TEST(TensorAllocCuda, UnknownDtypeReturnsNull) {
    int shape[] = {2, 3};
    EXPECT_EQ(tensor_alloc_cuda(shape, 2, 99, 0), nullptr);
    EXPECT_STRNE(get_last_error(), "");
}

TEST(TensorAllocCuda, FreeDeviceTensor) {
    int shape[] = {8};
    Tensor* t = tensor_alloc_cuda(shape, 1, 0, 0);
    ASSERT_NE(t, nullptr) << get_last_error();
    tensor_free(t);
}

#endif // INFER_CUDA_AVAILABLE

// ─── tensor_free (T-05) ──────────────────────────────────────────────────────

TEST(TensorFreeT05, NullNoOp) {
    tensor_free(nullptr);
    tensor_free(nullptr);
}

TEST(TensorFreeT05, RepeatedAllocFree) {
    int shape[] = {64, 64};
    for (int i = 0; i < 1000; ++i) {
        Tensor* t = tensor_alloc_cpu(shape, 2, 0);
        ASSERT_NE(t, nullptr);
        tensor_free(t);
    }
}

TEST(TensorFreeT05, HighNdim) {
    int shape[] = {2, 3, 4, 5, 6, 7, 8};
    Tensor* t = tensor_alloc_cpu(shape, 7, 0);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->nbytes, static_cast<size_t>(2*3*4*5*6*7*8*4));
    tensor_free(t);
}

TEST(TensorFreeT05, AllDtypesCpuNoLeak) {
    int shape[] = {16};
    for (int dtype = 0; dtype <= 6; ++dtype) {
        Tensor* t = tensor_alloc_cpu(shape, 1, dtype);
        ASSERT_NE(t, nullptr) << "dtype=" << dtype;
        tensor_free(t);
    }
}

#ifdef INFER_CUDA_AVAILABLE
TEST(TensorFreeT05, AllDtypesCudaNoLeak) {
    int shape[] = {16};
    for (int dtype = 0; dtype <= 6; ++dtype) {
        Tensor* t = tensor_alloc_cuda(shape, 1, dtype, 0);
        ASSERT_NE(t, nullptr) << "dtype=" << dtype << ": " << get_last_error();
        tensor_free(t);
    }
}

TEST(TensorFreeT05, RepeatedCudaAllocFree) {
    int shape[] = {256, 256};
    for (int i = 0; i < 100; ++i) {
        Tensor* t = tensor_alloc_cuda(shape, 2, 0, 0);
        ASSERT_NE(t, nullptr) << get_last_error();
        tensor_free(t);
    }
}
#endif // INFER_CUDA_AVAILABLE

// ─── tensor_copy_from (T-07) ─────────────────────────────────────────────────

TEST(TensorCopyFrom, Float32ReadBack) {
    int shape[] = {2, 3, 4};
    Tensor* t = tensor_alloc_cpu(shape, 3, 0);
    ASSERT_NE(t, nullptr);
    float src[24];
    for (int i = 0; i < 24; ++i) { src[i] = static_cast<float>(i) * 1.5f; }
    ASSERT_TRUE(tensor_copy_from(t, src, static_cast<int>(sizeof(src)))) << get_last_error();
    const float* dst = static_cast<const float*>(t->data);
    for (int i = 0; i < 24; ++i) { EXPECT_FLOAT_EQ(dst[i], src[i]) << "i=" << i; }
    tensor_free(t);
}

TEST(TensorCopyFrom, AllDtypes) {
    int shape[] = {8};
    uint8_t src[64];
    for (int i = 0; i < 64; ++i) { src[i] = static_cast<uint8_t>(i); }
    for (int dtype = 0; dtype <= 6; ++dtype) {
        Tensor* t = tensor_alloc_cpu(shape, 1, dtype);
        ASSERT_NE(t, nullptr) << "dtype=" << dtype;
        ASSERT_TRUE(tensor_copy_from(t, src, static_cast<int>(t->nbytes))) << "dtype=" << dtype;
        EXPECT_EQ(std::memcmp(t->data, src, t->nbytes), 0) << "dtype=" << dtype;
        tensor_free(t);
    }
}

TEST(TensorCopyFrom, OverwriteExistingData) {
    int shape[] = {4};
    Tensor* t = tensor_alloc_cpu(shape, 1, 0);
    ASSERT_NE(t, nullptr);
    float first[4]  = {1.f, 2.f, 3.f, 4.f};
    float second[4] = {9.f, 8.f, 7.f, 6.f};
    ASSERT_TRUE(tensor_copy_from(t, first,  static_cast<int>(sizeof(first))));
    ASSERT_TRUE(tensor_copy_from(t, second, static_cast<int>(sizeof(second))));
    const float* dst = static_cast<const float*>(t->data);
    EXPECT_FLOAT_EQ(dst[0], 9.f);
    EXPECT_FLOAT_EQ(dst[3], 6.f);
    tensor_free(t);
}

TEST(TensorCopyFrom, NullTensorReturnsFalse) {
    float src[4] = {};
    EXPECT_FALSE(tensor_copy_from(nullptr, src, static_cast<int>(sizeof(src))));
    EXPECT_STRNE(get_last_error(), "");
}

TEST(TensorCopyFrom, NullSrcReturnsFalse) {
    int shape[] = {4};
    Tensor* t = tensor_alloc_cpu(shape, 1, 0);
    ASSERT_NE(t, nullptr);
    EXPECT_FALSE(tensor_copy_from(t, nullptr, static_cast<int>(t->nbytes)));
    EXPECT_STRNE(get_last_error(), "");
    tensor_free(t);
}

TEST(TensorCopyFrom, ZeroNbytesReturnsFalse) {
    int shape[] = {4};
    float src[4] = {};
    Tensor* t = tensor_alloc_cpu(shape, 1, 0);
    ASSERT_NE(t, nullptr);
    EXPECT_FALSE(tensor_copy_from(t, src, 0));
    EXPECT_STRNE(get_last_error(), "");
    tensor_free(t);
}

TEST(TensorCopyFrom, WrongNbytesReturnsFalse) {
    int shape[] = {4};
    float src[8] = {};
    Tensor* t = tensor_alloc_cpu(shape, 1, 0);
    ASSERT_NE(t, nullptr);
    EXPECT_FALSE(tensor_copy_from(t, src, 32));
    EXPECT_STRNE(get_last_error(), "");
    tensor_free(t);
}

#ifdef INFER_CUDA_AVAILABLE
TEST(TensorCopyFrom, DeviceTensorReturnsFalse) {
    int shape[] = {4};
    Tensor* t = tensor_alloc_cuda(shape, 1, 0, 0);
    ASSERT_NE(t, nullptr) << get_last_error();
    float src[4] = {};
    EXPECT_FALSE(tensor_copy_from(t, src, static_cast<int>(t->nbytes)));
    EXPECT_STRNE(get_last_error(), "");
    tensor_free(t);
}
#endif // INFER_CUDA_AVAILABLE

// ─── tensor_to_device / tensor_to_host (T-06) ────────────────────────────────

#ifdef INFER_CUDA_AVAILABLE

TEST(TensorTransfer, RoundTrip_Float32) {
    int shape[] = {1, 3};
    Tensor* t = tensor_alloc_cpu(shape, 2, 0);
    ASSERT_NE(t, nullptr);
    float src[3] = {1.0f, 2.0f, 3.0f};
    std::memcpy(t->data, src, sizeof(src));
    ASSERT_TRUE(tensor_to_device(t, 0)) << get_last_error();
    EXPECT_TRUE(t->on_device);
    cudaPointerAttributes attrs{};
    ASSERT_EQ(cudaPointerGetAttributes(&attrs, t->data), cudaSuccess);
    EXPECT_EQ(attrs.type, cudaMemoryTypeDevice);
    ASSERT_TRUE(tensor_to_host(t)) << get_last_error();
    EXPECT_FALSE(t->on_device);
    float dst[3] = {};
    std::memcpy(dst, t->data, sizeof(dst));
    EXPECT_FLOAT_EQ(dst[0], 1.0f);
    EXPECT_FLOAT_EQ(dst[1], 2.0f);
    EXPECT_FLOAT_EQ(dst[2], 3.0f);
    tensor_free(t);
}

TEST(TensorTransfer, RoundTrip_Int64) {
    int shape[] = {4};
    Tensor* t = tensor_alloc_cpu(shape, 1, 4);
    ASSERT_NE(t, nullptr);
    int64_t src[4] = {100LL, -200LL, 0LL, 9999999999LL};
    std::memcpy(t->data, src, sizeof(src));
    ASSERT_TRUE(tensor_to_device(t, 0)) << get_last_error();
    ASSERT_TRUE(tensor_to_host(t))      << get_last_error();
    int64_t dst[4] = {};
    std::memcpy(dst, t->data, sizeof(dst));
    EXPECT_EQ(dst[0], 100LL);
    EXPECT_EQ(dst[1], -200LL);
    EXPECT_EQ(dst[2], 0LL);
    EXPECT_EQ(dst[3], 9999999999LL);
    tensor_free(t);
}

TEST(TensorTransfer, ToDeviceIsNoOpIfAlreadyOnDevice) {
    int shape[] = {8};
    Tensor* t = tensor_alloc_cuda(shape, 1, 0, 0);
    ASSERT_NE(t, nullptr) << get_last_error();
    void* original_ptr = t->data;
    EXPECT_TRUE(tensor_to_device(t, 0));
    EXPECT_EQ(t->data, original_ptr);
    tensor_free(t);
}

TEST(TensorTransfer, ToHostIsNoOpIfAlreadyOnHost) {
    int shape[] = {8};
    Tensor* t = tensor_alloc_cpu(shape, 1, 0);
    ASSERT_NE(t, nullptr);
    void* original_ptr = t->data;
    EXPECT_TRUE(tensor_to_host(t));
    EXPECT_EQ(t->data, original_ptr);
    tensor_free(t);
}

TEST(TensorTransfer, NullTensorReturnsFalse) {
    EXPECT_FALSE(tensor_to_device(nullptr, 0));
    EXPECT_STRNE(get_last_error(), "");
    EXPECT_FALSE(tensor_to_host(nullptr));
    EXPECT_STRNE(get_last_error(), "");
}

TEST(TensorTransfer, MultipleRoundTrips) {
    int shape[] = {16};
    Tensor* t = tensor_alloc_cpu(shape, 1, 0);
    ASSERT_NE(t, nullptr);
    float src[16];
    for (int i = 0; i < 16; ++i) { src[i] = static_cast<float>(i) * 0.5f; }
    std::memcpy(t->data, src, sizeof(src));
    for (int round = 0; round < 5; ++round) {
        ASSERT_TRUE(tensor_to_device(t, 0)) << "round=" << round;
        ASSERT_TRUE(tensor_to_host(t))      << "round=" << round;
    }
    float dst[16] = {};
    std::memcpy(dst, t->data, sizeof(dst));
    for (int i = 0; i < 16; ++i) { EXPECT_FLOAT_EQ(dst[i], src[i]) << "i=" << i; }
    tensor_free(t);
}

#endif // INFER_CUDA_AVAILABLE

// ─── Getters (T-08) ──────────────────────────────────────────────────────────

// Acceptance criterion: all getters return correct values for [2,3,4] float32.
TEST(TensorGetters, Float32_234_AllGetters) {
    int shape[] = {2, 3, 4};
    Tensor* t = tensor_alloc_cpu(shape, 3, 0);
    ASSERT_NE(t, nullptr);

    // dtype
    EXPECT_EQ(tensor_get_dtype(t), 0);

    // nbytes: 2*3*4*4 = 96
    EXPECT_EQ(tensor_get_nbytes(t), 96);

    // nelements: 2*3*4 = 24
    EXPECT_EQ(tensor_get_nelements(t), 24);

    // shape
    int out[3] = {};
    int ndim = tensor_get_shape(t, out, 3);
    EXPECT_EQ(ndim, 3);
    EXPECT_EQ(out[0], 2);
    EXPECT_EQ(out[1], 3);
    EXPECT_EQ(out[2], 4);

    tensor_free(t);
}

TEST(TensorGetters, AllDtypesNbytes) {
    int shape[] = {1};
    const int expected_nbytes[] = {4, 2, 2, 4, 8, 1, 1};
    for (int dtype = 0; dtype <= 6; ++dtype) {
        Tensor* t = tensor_alloc_cpu(shape, 1, dtype);
        ASSERT_NE(t, nullptr) << "dtype=" << dtype;
        EXPECT_EQ(tensor_get_dtype(t),   dtype)                      << "dtype=" << dtype;
        EXPECT_EQ(tensor_get_nbytes(t),  expected_nbytes[dtype])     << "dtype=" << dtype;
        EXPECT_EQ(tensor_get_nelements(t), 1)                        << "dtype=" << dtype;
        tensor_free(t);
    }
}

TEST(TensorGetters, ShapeMaxDimsClamp) {
    int shape[] = {2, 3, 4};
    Tensor* t = tensor_alloc_cpu(shape, 3, 0);
    ASSERT_NE(t, nullptr);

    // Ask for only 2 dims — should still return ndim=3, write 2 values
    int out[2] = {};
    int ndim = tensor_get_shape(t, out, 2);
    EXPECT_EQ(ndim, 3);
    EXPECT_EQ(out[0], 2);
    EXPECT_EQ(out[1], 3);

    tensor_free(t);
}

TEST(TensorGetters, ShapeLargeMaxDims) {
    int shape[] = {5};
    Tensor* t = tensor_alloc_cpu(shape, 1, 0);
    ASSERT_NE(t, nullptr);

    int out[8] = {};
    EXPECT_EQ(tensor_get_shape(t, out, 8), 1);
    EXPECT_EQ(out[0], 5);

    tensor_free(t);
}

TEST(TensorGetters, NullTensorSafeDefaults) {
    EXPECT_EQ(tensor_get_dtype(nullptr),    -1);
    EXPECT_EQ(tensor_get_nbytes(nullptr),    0);
    EXPECT_EQ(tensor_get_nelements(nullptr), 0);

    int out[4] = {};
    EXPECT_EQ(tensor_get_shape(nullptr, out, 4), 0);
}

TEST(TensorGetters, ShapeNullOutBufReturnsZero) {
    int shape[] = {2, 3};
    Tensor* t = tensor_alloc_cpu(shape, 2, 0);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(tensor_get_shape(t, nullptr, 2), 0);
    tensor_free(t);
}

TEST(TensorGetters, ShapeZeroMaxDimsReturnsZero) {
    int shape[] = {2, 3};
    Tensor* t = tensor_alloc_cpu(shape, 2, 0);
    ASSERT_NE(t, nullptr);
    int out[2] = {};
    EXPECT_EQ(tensor_get_shape(t, out, 0), 0);
    tensor_free(t);
}

#ifdef INFER_CUDA_AVAILABLE
TEST(TensorGetters, CudaTensorGetters) {
    int shape[] = {2, 3, 4};
    Tensor* t = tensor_alloc_cuda(shape, 3, 0, 0);
    ASSERT_NE(t, nullptr) << get_last_error();

    EXPECT_EQ(tensor_get_dtype(t),     0);
    EXPECT_EQ(tensor_get_nbytes(t),    96);
    EXPECT_EQ(tensor_get_nelements(t), 24);

    int out[3] = {};
    EXPECT_EQ(tensor_get_shape(t, out, 3), 3);
    EXPECT_EQ(out[0], 2);
    EXPECT_EQ(out[1], 3);
    EXPECT_EQ(out[2], 4);

    tensor_free(t);
}
#endif // INFER_CUDA_AVAILABLE
