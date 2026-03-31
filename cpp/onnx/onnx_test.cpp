#include "onnx_session.hpp"
#include "../tensor/tensor.hpp"

#include <cstring>
#include <gtest/gtest.h>

// Path to the tiny scale.onnx model (y = x * 2, input [1,4] float32)
// Generated on gpu_dev: /tmp/scale.onnx → copied to test fixtures dir
#ifndef TEST_MODEL_PATH
#define TEST_MODEL_PATH "/tmp/scale.onnx"
#endif

using namespace infergo;

// ─── Construction ─────────────────────────────────────────────────────────────

TEST(OnnxSession, ConstructCpuNoThrow) {
    EXPECT_NO_THROW(OnnxSession s("cpu", 0));
}

TEST(OnnxSession, MoveConstruct) {
    OnnxSession a("cpu", 0);
    EXPECT_NO_THROW(OnnxSession b(std::move(a)));
}

// ─── load_model ───────────────────────────────────────────────────────────────

TEST(OnnxSession, LoadModel) {
    OnnxSession s("cpu", 0);
    EXPECT_NO_THROW(s.load_model(TEST_MODEL_PATH));
    EXPECT_EQ(s.num_inputs(),  1);
    EXPECT_EQ(s.num_outputs(), 1);
}

TEST(OnnxSession, LoadModelInputOutputNames) {
    OnnxSession s("cpu", 0);
    s.load_model(TEST_MODEL_PATH);
    EXPECT_EQ(s.input_name(0),  "x");
    EXPECT_EQ(s.output_name(0), "y");
}

TEST(OnnxSession, LoadBadPathThrows) {
    OnnxSession s("cpu", 0);
    EXPECT_THROW(s.load_model("/nonexistent/model.onnx"), std::runtime_error);
}

TEST(OnnxSession, ReloadReplacesSession) {
    OnnxSession s("cpu", 0);
    s.load_model(TEST_MODEL_PATH);
    EXPECT_NO_THROW(s.load_model(TEST_MODEL_PATH)); // reload same file
    EXPECT_EQ(s.num_inputs(), 1);
}

// ─── run ──────────────────────────────────────────────────────────────────────

TEST(OnnxSession, RunScaleModel) {
    OnnxSession s("cpu", 0);
    s.load_model(TEST_MODEL_PATH);

    // Input: [1.0, 2.0, 3.0, 4.0]
    const int shape[] = {1, 4};
    Tensor* in = tensor_alloc_cpu(shape, 2, 0 /*float32*/);
    ASSERT_NE(in, nullptr);

    const float src[] = {1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT_TRUE(tensor_copy_from(in, src, sizeof(src)));

    std::vector<Tensor*> outputs;
    EXPECT_NO_THROW(outputs = s.run({in}));
    ASSERT_EQ(outputs.size(), 1u);

    Tensor* out = outputs[0];
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->ndim, 2);
    EXPECT_EQ(out->shape[0], 1);
    EXPECT_EQ(out->shape[1], 4);

    // y = x * 2: expect [2, 4, 6, 8]
    const float* data = static_cast<const float*>(out->data);
    EXPECT_FLOAT_EQ(data[0], 2.0f);
    EXPECT_FLOAT_EQ(data[1], 4.0f);
    EXPECT_FLOAT_EQ(data[2], 6.0f);
    EXPECT_FLOAT_EQ(data[3], 8.0f);

    tensor_free(out);
    tensor_free(in);
}

TEST(OnnxSession, RunWithoutLoadThrows) {
    OnnxSession s("cpu", 0);
    EXPECT_THROW(s.run({}), std::runtime_error);
}

TEST(OnnxSession, RunWrongInputCountThrows) {
    OnnxSession s("cpu", 0);
    s.load_model(TEST_MODEL_PATH);
    EXPECT_THROW(s.run({}), std::runtime_error);
}

TEST(OnnxSession, RunNullInputThrows) {
    OnnxSession s("cpu", 0);
    s.load_model(TEST_MODEL_PATH);
    EXPECT_THROW(s.run({nullptr}), std::runtime_error);
}

TEST(OnnxSession, RunDeviceTensorThrows) {
    OnnxSession s("cpu", 0);
    s.load_model(TEST_MODEL_PATH);

    const int shape[] = {1, 4};
    Tensor* t = tensor_alloc_cpu(shape, 2, 0);
    ASSERT_NE(t, nullptr);

#ifdef INFER_CUDA_AVAILABLE
    ASSERT_TRUE(tensor_to_device(t, 0));
    EXPECT_THROW(s.run({t}), std::runtime_error);
#else
    // Without CUDA mark it manually to test the guard
    t->on_device = true;
    EXPECT_THROW(s.run({t}), std::runtime_error);
    t->on_device = false;
#endif

    tensor_free(t);
}

TEST(OnnxSession, OutputsOwnedByCaller) {
    // Run twice — verify no double-free / use-after-free
    OnnxSession s("cpu", 0);
    s.load_model(TEST_MODEL_PATH);

    const int shape[] = {1, 4};
    const float src[] = {1.0f, 1.0f, 1.0f, 1.0f};

    for (int i = 0; i < 3; ++i) {
        Tensor* in = tensor_alloc_cpu(shape, 2, 0);
        ASSERT_NE(in, nullptr);
        ASSERT_TRUE(tensor_copy_from(in, src, sizeof(src)));
        auto outs = s.run({in});
        ASSERT_EQ(outs.size(), 1u);
        tensor_free(outs[0]);
        tensor_free(in);
    }
}

// ─── T-14: Execution provider selection ──────────────────────────────────────

TEST(OnnxSessionProvider, CpuProviderActive) {
    OnnxSession s("cpu", 0);
    EXPECT_EQ(s.provider(), "cpu");
}

TEST(OnnxSessionProvider, CpuProviderLoadsAndRuns) {
    OnnxSession s("cpu", 0);
    ASSERT_EQ(s.provider(), "cpu");
    s.load_model(TEST_MODEL_PATH);

    const int shape[] = {1, 4};
    Tensor* in = tensor_alloc_cpu(shape, 2, 0);
    ASSERT_NE(in, nullptr);
    const float src[] = {3.0f, 3.0f, 3.0f, 3.0f};
    ASSERT_TRUE(tensor_copy_from(in, src, sizeof(src)));

    auto outs = s.run({in});
    ASSERT_EQ(outs.size(), 1u);
    const float* data = static_cast<const float*>(outs[0]->data);
    EXPECT_FLOAT_EQ(data[0], 6.0f);
    tensor_free(outs[0]);
    tensor_free(in);
}

TEST(OnnxSessionProvider, UnknownProviderFallsBackToCpu) {
    // "xpu" is not a real provider — must fall back to CPU without throwing
    OnnxSession s("xpu", 0);
    EXPECT_EQ(s.provider(), "cpu");
    // Must still be usable
    s.load_model(TEST_MODEL_PATH);
    EXPECT_EQ(s.num_inputs(), 1);
}

TEST(OnnxSessionProvider, CudaProviderFallsBackOrSucceeds) {
    // Either CUDA is available (provider stays "cuda") or it falls back to CPU.
    // Either way: no exception, and the session is usable.
    OnnxSession s("cuda", 0);
    EXPECT_TRUE(s.provider() == "cuda" || s.provider() == "cpu");
    s.load_model(TEST_MODEL_PATH);
    EXPECT_EQ(s.num_inputs(), 1);
}

TEST(OnnxSessionProvider, TensorRTProviderFallsBackOrSucceeds) {
    OnnxSession s("tensorrt", 0);
    EXPECT_TRUE(s.provider() == "tensorrt" || s.provider() == "cpu");
    s.load_model(TEST_MODEL_PATH);
    EXPECT_EQ(s.num_inputs(), 1);
}
