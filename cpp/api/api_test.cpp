#include "infer_api.h"

#include <cstring>
#include <cstdint>
#include <gtest/gtest.h>

// ─── infer_last_error_string ──────────────────────────────────────────────────

TEST(ApiErrorString, ReturnsCharPointer) {
    const char* err = infer_last_error_string();
    ASSERT_NE(err, nullptr);
}

// ─── infer_tensor_alloc_cpu ───────────────────────────────────────────────────

TEST(ApiAllocCpu, Float32_234) {
    const int shape[] = {2, 3, 4};
    InferTensor t = infer_tensor_alloc_cpu(shape, 3, INFER_DTYPE_FLOAT32);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(infer_tensor_dtype(t), INFER_DTYPE_FLOAT32);
    EXPECT_EQ(infer_tensor_nbytes(t), 96);   // 2*3*4*4
    EXPECT_EQ(infer_tensor_nelements(t), 24);
    infer_tensor_free(t);
}

TEST(ApiAllocCpu, AllDtypes) {
    struct Case { int dtype; int elem_size; };
    const Case cases[] = {
        {INFER_DTYPE_FLOAT32,  4},
        {INFER_DTYPE_FLOAT16,  2},
        {INFER_DTYPE_BFLOAT16, 2},
        {INFER_DTYPE_INT32,    4},
        {INFER_DTYPE_INT64,    8},
        {INFER_DTYPE_UINT8,    1},
        {INFER_DTYPE_BOOL,     1},
    };
    const int shape[] = {4};
    for (const auto& c : cases) {
        InferTensor t = infer_tensor_alloc_cpu(shape, 1, c.dtype);
        ASSERT_NE(t, nullptr) << "dtype=" << c.dtype;
        EXPECT_EQ(infer_tensor_nbytes(t), 4 * c.elem_size) << "dtype=" << c.dtype;
        infer_tensor_free(t);
    }
}

TEST(ApiAllocCpu, NullShapeReturnsNull) {
    InferTensor t = infer_tensor_alloc_cpu(nullptr, 1, INFER_DTYPE_FLOAT32);
    EXPECT_EQ(t, nullptr);
    EXPECT_NE(infer_last_error_string()[0], '\0');
}

TEST(ApiAllocCpu, ZeroNdimReturnsNull) {
    const int shape[] = {4};
    InferTensor t = infer_tensor_alloc_cpu(shape, 0, INFER_DTYPE_FLOAT32);
    EXPECT_EQ(t, nullptr);
}

TEST(ApiAllocCpu, UnknownDtypeReturnsNull) {
    const int shape[] = {4};
    InferTensor t = infer_tensor_alloc_cpu(shape, 1, 999);
    EXPECT_EQ(t, nullptr);
}

// ─── infer_tensor_free ────────────────────────────────────────────────────────

TEST(ApiTensorFree, NullIsNoop) {
    infer_tensor_free(nullptr);  // must not crash
}

// ─── infer_tensor_data_ptr ────────────────────────────────────────────────────

TEST(ApiDataPtr, CpuTensorNotNull) {
    const int shape[] = {8};
    InferTensor t = infer_tensor_alloc_cpu(shape, 1, INFER_DTYPE_FLOAT32);
    ASSERT_NE(t, nullptr);
    EXPECT_NE(infer_tensor_data_ptr(t), nullptr);
    infer_tensor_free(t);
}

TEST(ApiDataPtr, NullTensorReturnsNull) {
    EXPECT_EQ(infer_tensor_data_ptr(nullptr), nullptr);
}

// ─── infer_tensor_shape ───────────────────────────────────────────────────────

TEST(ApiShape, CorrectDims) {
    const int shape[] = {2, 3, 4};
    InferTensor t = infer_tensor_alloc_cpu(shape, 3, INFER_DTYPE_FLOAT32);
    ASSERT_NE(t, nullptr);
    int out[8] = {};
    int ndim = infer_tensor_shape(t, out, 8);
    EXPECT_EQ(ndim, 3);
    EXPECT_EQ(out[0], 2);
    EXPECT_EQ(out[1], 3);
    EXPECT_EQ(out[2], 4);
    infer_tensor_free(t);
}

TEST(ApiShape, NullTensorReturnsZero) {
    int out[4] = {};
    EXPECT_EQ(infer_tensor_shape(nullptr, out, 4), 0);
}

TEST(ApiShape, NullOutBufReturnsZero) {
    const int shape[] = {4};
    InferTensor t = infer_tensor_alloc_cpu(shape, 1, INFER_DTYPE_FLOAT32);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(infer_tensor_shape(t, nullptr, 4), 0);
    infer_tensor_free(t);
}

// ─── infer_tensor_copy_from ───────────────────────────────────────────────────

TEST(ApiCopyFrom, RoundTrip) {
    const int shape[] = {4};
    InferTensor t = infer_tensor_alloc_cpu(shape, 1, INFER_DTYPE_FLOAT32);
    ASSERT_NE(t, nullptr);

    const float src[] = {1.0f, 2.0f, 3.0f, 4.0f};
    InferError err = infer_tensor_copy_from(t, src, static_cast<int>(sizeof(src)));
    EXPECT_EQ(err, INFER_OK);

    const float* data = static_cast<const float*>(infer_tensor_data_ptr(t));
    ASSERT_NE(data, nullptr);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(data[i], src[i]);
    }
    infer_tensor_free(t);
}

TEST(ApiCopyFrom, NullTensorReturnsErrNull) {
    const float src[] = {1.0f};
    InferError err = infer_tensor_copy_from(nullptr, src, 4);
    EXPECT_EQ(err, INFER_ERR_NULL);
}

TEST(ApiCopyFrom, NullSrcReturnsErrNull) {
    const int shape[] = {4};
    InferTensor t = infer_tensor_alloc_cpu(shape, 1, INFER_DTYPE_FLOAT32);
    ASSERT_NE(t, nullptr);
    InferError err = infer_tensor_copy_from(t, nullptr, 16);
    EXPECT_EQ(err, INFER_ERR_NULL);
    infer_tensor_free(t);
}

TEST(ApiCopyFrom, WrongNbytesReturnsErrInvalid) {
    const int shape[] = {4};
    InferTensor t = infer_tensor_alloc_cpu(shape, 1, INFER_DTYPE_FLOAT32);
    ASSERT_NE(t, nullptr);
    const float src[] = {1.0f, 2.0f};
    InferError err = infer_tensor_copy_from(t, src, 8);  // tensor needs 16 bytes
    EXPECT_EQ(err, INFER_ERR_INVALID);
    infer_tensor_free(t);
}

// ─── CUDA tests (compiled only when CUDA is available) ───────────────────────

#ifdef INFER_CUDA_AVAILABLE

TEST(ApiAllocCuda, Float32_234) {
    const int shape[] = {2, 3, 4};
    InferTensor t = infer_tensor_alloc_cuda(shape, 3, INFER_DTYPE_FLOAT32, 0);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(infer_tensor_dtype(t), INFER_DTYPE_FLOAT32);
    EXPECT_EQ(infer_tensor_nbytes(t), 96);
    EXPECT_EQ(infer_tensor_nelements(t), 24);
    infer_tensor_free(t);
}

TEST(ApiCudaTransfer, RoundTrip) {
    const int shape[] = {4};
    InferTensor t = infer_tensor_alloc_cpu(shape, 1, INFER_DTYPE_FLOAT32);
    ASSERT_NE(t, nullptr);

    const float src[] = {10.0f, 20.0f, 30.0f, 40.0f};
    ASSERT_EQ(infer_tensor_copy_from(t, src, 16), INFER_OK);

    EXPECT_EQ(infer_tensor_to_device(t, 0), INFER_OK);
    EXPECT_EQ(infer_tensor_to_host(t), INFER_OK);

    const float* data = static_cast<const float*>(infer_tensor_data_ptr(t));
    ASSERT_NE(data, nullptr);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(data[i], src[i]);
    }
    infer_tensor_free(t);
}

TEST(ApiCudaTransfer, NullToDeviceReturnsErrNull) {
    EXPECT_EQ(infer_tensor_to_device(nullptr, 0), INFER_ERR_NULL);
}

TEST(ApiCudaTransfer, NullToHostReturnsErrNull) {
    EXPECT_EQ(infer_tensor_to_host(nullptr), INFER_ERR_NULL);
}

#endif // INFER_CUDA_AVAILABLE

// ─── Session API (T-15) ───────────────────────────────────────────────────────

#ifndef TEST_MODEL_PATH
#define TEST_MODEL_PATH "/tmp/scale.onnx"
#endif

TEST(ApiSession, CreateCpuNotNull) {
    InferSession s = infer_session_create("cpu", 0);
    ASSERT_NE(s, nullptr);
    infer_session_destroy(s);
}

TEST(ApiSession, CreateNullProviderDefaultsCpu) {
    InferSession s = infer_session_create(nullptr, 0);
    ASSERT_NE(s, nullptr);
    infer_session_destroy(s);
}

TEST(ApiSession, DestroyNullIsNoop) {
    infer_session_destroy(nullptr); // must not crash
}

TEST(ApiSession, LoadModel) {
    InferSession s = infer_session_create("cpu", 0);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(infer_session_load(s, TEST_MODEL_PATH), INFER_OK);
    EXPECT_EQ(infer_session_num_inputs(s),  1);
    EXPECT_EQ(infer_session_num_outputs(s), 1);
    infer_session_destroy(s);
}

TEST(ApiSession, LoadBadPathReturnsErrLoad) {
    InferSession s = infer_session_create("cpu", 0);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(infer_session_load(s, "/no/such/model.onnx"), INFER_ERR_LOAD);
    EXPECT_NE(infer_last_error_string()[0], '\0');
    infer_session_destroy(s);
}

TEST(ApiSession, LoadNullSessionReturnsErrNull) {
    EXPECT_EQ(infer_session_load(nullptr, TEST_MODEL_PATH), INFER_ERR_NULL);
}

TEST(ApiSession, InputOutputNames) {
    InferSession s = infer_session_create("cpu", 0);
    ASSERT_NE(s, nullptr);
    ASSERT_EQ(infer_session_load(s, TEST_MODEL_PATH), INFER_OK);

    char buf[64];
    EXPECT_EQ(infer_session_input_name(s, 0, buf, sizeof(buf)), INFER_OK);
    EXPECT_STREQ(buf, "x");

    EXPECT_EQ(infer_session_output_name(s, 0, buf, sizeof(buf)), INFER_OK);
    EXPECT_STREQ(buf, "y");

    infer_session_destroy(s);
}

TEST(ApiSession, RunScaleModel) {
    InferSession s = infer_session_create("cpu", 0);
    ASSERT_NE(s, nullptr);
    ASSERT_EQ(infer_session_load(s, TEST_MODEL_PATH), INFER_OK);

    // Input [1,4] float32 = {1,2,3,4}
    const int shape[] = {1, 4};
    InferTensor in = infer_tensor_alloc_cpu(shape, 2, INFER_DTYPE_FLOAT32);
    ASSERT_NE(in, nullptr);
    const float src[] = {1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT_EQ(infer_tensor_copy_from(in, src, sizeof(src)), INFER_OK);

    InferTensor out = nullptr;
    EXPECT_EQ(infer_session_run(s, &in, 1, &out, 1), INFER_OK);
    ASSERT_NE(out, nullptr);

    // y = x * 2: expect [2,4,6,8]
    EXPECT_EQ(infer_tensor_nbytes(out), 16);
    const float* data = static_cast<const float*>(infer_tensor_data_ptr(out));
    EXPECT_FLOAT_EQ(data[0], 2.0f);
    EXPECT_FLOAT_EQ(data[1], 4.0f);
    EXPECT_FLOAT_EQ(data[2], 6.0f);
    EXPECT_FLOAT_EQ(data[3], 8.0f);

    infer_tensor_free(out);
    infer_tensor_free(in);
    infer_session_destroy(s);
}

TEST(ApiSession, RunNullSessionReturnsErrNull) {
    InferTensor out = nullptr;
    EXPECT_EQ(infer_session_run(nullptr, nullptr, 0, &out, 1), INFER_ERR_NULL);
}

TEST(ApiSession, RunWrongInputCountReturnsErr) {
    InferSession s = infer_session_create("cpu", 0);
    ASSERT_NE(s, nullptr);
    ASSERT_EQ(infer_session_load(s, TEST_MODEL_PATH), INFER_OK);

    // Pass 0 inputs instead of 1
    InferTensor out = nullptr;
    EXPECT_NE(infer_session_run(s, nullptr, 0, &out, 1), INFER_OK);
    infer_session_destroy(s);
}
