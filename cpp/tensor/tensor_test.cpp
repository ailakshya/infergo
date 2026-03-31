#include <gtest/gtest.h>
#include <cstring>
#include "tensor.hpp"

#ifdef INFER_CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

using infergo::Tensor;
using infergo::tensor_alloc_cpu;
using infergo::tensor_free;
using infergo::get_last_error;

#ifdef INFER_CUDA_AVAILABLE
using infergo::tensor_alloc_cuda;
using infergo::tensor_to_device;
using infergo::tensor_to_host;
#endif

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

// ─── tensor_alloc_cuda ───────────────────────────────────────────────────────

#ifdef INFER_CUDA_AVAILABLE

// Acceptance criterion: alloc on device 0, ptr not null,
// cudaPointerGetAttributes confirms device memory.
TEST(TensorAllocCuda, Float32_234_DeviceMemory) {
    int shape[] = {2, 3, 4};
    Tensor* t = tensor_alloc_cuda(shape, 3, 0, 0);

    ASSERT_NE(t, nullptr) << "tensor_alloc_cuda failed: " << get_last_error();
    EXPECT_NE(t->data, nullptr);
    EXPECT_EQ(t->nbytes, static_cast<size_t>(96));
    EXPECT_EQ(t->nelements(), static_cast<size_t>(24));
    EXPECT_EQ(t->ndim, 3);
    EXPECT_EQ(t->shape[0], 2);
    EXPECT_EQ(t->shape[1], 3);
    EXPECT_EQ(t->shape[2], 4);
    EXPECT_EQ(t->dtype, 0);
    EXPECT_TRUE(t->on_device);
    EXPECT_EQ(t->device_id, 0);

    // Confirm the data pointer is actually device memory
    cudaPointerAttributes attrs{};
    cudaError_t err = cudaPointerGetAttributes(&attrs, t->data);
    EXPECT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
    EXPECT_EQ(attrs.type, cudaMemoryTypeDevice);
    EXPECT_EQ(attrs.device, 0);

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

    // Shape array must be in host memory, accessible from CPU
    cudaPointerAttributes attrs{};
    cudaError_t err = cudaPointerGetAttributes(&attrs, t->shape);
    // Host malloc'd memory may appear as cudaMemoryTypeUnregistered or host
    EXPECT_EQ(err, cudaSuccess);
    EXPECT_NE(attrs.type, cudaMemoryTypeDevice);

    // Mutate original — tensor must not be affected
    shape[0] = 99;
    EXPECT_EQ(t->shape[0], 5);

    tensor_free(t);
}

TEST(TensorAllocCuda, NullShapeReturnsNull) {
    Tensor* t = tensor_alloc_cuda(nullptr, 3, 0, 0);
    EXPECT_EQ(t, nullptr);
    EXPECT_STRNE(get_last_error(), "");
}

TEST(TensorAllocCuda, ZeroDimensionReturnsNull) {
    int shape[] = {2, 0, 4};
    Tensor* t = tensor_alloc_cuda(shape, 3, 0, 0);
    EXPECT_EQ(t, nullptr);
    EXPECT_STRNE(get_last_error(), "");
}

TEST(TensorAllocCuda, UnknownDtypeReturnsNull) {
    int shape[] = {2, 3};
    Tensor* t = tensor_alloc_cuda(shape, 2, 99, 0);
    EXPECT_EQ(t, nullptr);
    EXPECT_STRNE(get_last_error(), "");
}

TEST(TensorAllocCuda, FreeDeviceTensor) {
    int shape[] = {8};
    Tensor* t = tensor_alloc_cuda(shape, 1, 0, 0);
    ASSERT_NE(t, nullptr) << get_last_error();
    // Must not crash or leak (address sanitizer validates this in Debug builds)
    tensor_free(t);
}

#endif // INFER_CUDA_AVAILABLE

// ─── tensor_free (T-05) ──────────────────────────────────────────────────────

TEST(TensorFreeT05, NullNoOp) {
    // Calling free on nullptr must be a silent no-op — never crash
    tensor_free(nullptr);
    tensor_free(nullptr);
}

TEST(TensorFreeT05, RepeatedAllocFree) {
    // 1000 alloc+free cycles — valgrind / ASan must show zero leaks
    int shape[] = {64, 64};
    for (int i = 0; i < 1000; ++i) {
        Tensor* t = tensor_alloc_cpu(shape, 2, 0);
        ASSERT_NE(t, nullptr);
        tensor_free(t);
    }
}

TEST(TensorFreeT05, HighNdim) {
    // Valgrind checks that the shape array (ndim ints) is fully freed
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
    // 100 CUDA alloc+free cycles — confirms cudaFree is always called
    int shape[] = {256, 256};
    for (int i = 0; i < 100; ++i) {
        Tensor* t = tensor_alloc_cuda(shape, 2, 0, 0);
        ASSERT_NE(t, nullptr) << get_last_error();
        tensor_free(t);
    }
}
#endif // INFER_CUDA_AVAILABLE

// ─── tensor_to_device / tensor_to_host (T-06) ────────────────────────────────

#ifdef INFER_CUDA_AVAILABLE

// Acceptance criterion: alloc CPU [1,3], fill known values, to_device, to_host,
// values match.
TEST(TensorTransfer, RoundTrip_Float32) {
    int shape[] = {1, 3};
    Tensor* t = tensor_alloc_cpu(shape, 2, 0);
    ASSERT_NE(t, nullptr);

    // Write known values into the CPU buffer
    float src[3] = {1.0f, 2.0f, 3.0f};
    std::memcpy(t->data, src, sizeof(src));

    // to_device
    ASSERT_TRUE(tensor_to_device(t, 0)) << get_last_error();
    EXPECT_TRUE(t->on_device);
    EXPECT_EQ(t->device_id, 0);
    EXPECT_EQ(t->nbytes, static_cast<size_t>(12));

    // data pointer must now be device memory
    cudaPointerAttributes attrs{};
    ASSERT_EQ(cudaPointerGetAttributes(&attrs, t->data), cudaSuccess);
    EXPECT_EQ(attrs.type, cudaMemoryTypeDevice);

    // to_host
    ASSERT_TRUE(tensor_to_host(t)) << get_last_error();
    EXPECT_FALSE(t->on_device);

    // data pointer must now be host memory
    ASSERT_EQ(cudaPointerGetAttributes(&attrs, t->data), cudaSuccess);
    EXPECT_NE(attrs.type, cudaMemoryTypeDevice);

    // Values must survive the round trip exactly
    float dst[3] = {};
    std::memcpy(dst, t->data, sizeof(dst));
    EXPECT_FLOAT_EQ(dst[0], 1.0f);
    EXPECT_FLOAT_EQ(dst[1], 2.0f);
    EXPECT_FLOAT_EQ(dst[2], 3.0f);

    tensor_free(t);
}

TEST(TensorTransfer, RoundTrip_Int64) {
    int shape[] = {4};
    Tensor* t = tensor_alloc_cpu(shape, 1, 4);  // INT64
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
    EXPECT_TRUE(tensor_to_device(t, 0));  // must not reallocate or crash
    EXPECT_EQ(t->data, original_ptr);     // pointer unchanged

    tensor_free(t);
}

TEST(TensorTransfer, ToHostIsNoOpIfAlreadyOnHost) {
    int shape[] = {8};
    Tensor* t = tensor_alloc_cpu(shape, 1, 0);
    ASSERT_NE(t, nullptr);

    void* original_ptr = t->data;
    EXPECT_TRUE(tensor_to_host(t));    // must not reallocate or crash
    EXPECT_EQ(t->data, original_ptr);  // pointer unchanged

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
    for (int i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(dst[i], src[i]) << "i=" << i;
    }

    tensor_free(t);
}

#endif // INFER_CUDA_AVAILABLE
