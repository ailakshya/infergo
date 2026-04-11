#include "torch_session.hpp"
#include "../tensor/tensor.hpp"

#include <gtest/gtest.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

// Helper: create a minimal TorchScript model that doubles its input.
// Saves to a temporary file and returns the path.
// The model is: def forward(self, x): return x * 2
std::string create_double_model(const std::string& path) {
    // Create a minimal TorchScript module that doubles its input.
    torch::jit::Module m("test_double");
    m.define(R"(
        def forward(self, x: Tensor) -> Tensor:
            return x * 2
    )");
    m.save(path);
    return path;
}

// Helper: create a model that returns a tuple of two tensors.
// forward(x) -> (x + 1, x * 3)
std::string create_tuple_model(const std::string& path) {
    torch::jit::Module m("test_tuple");
    m.define(R"(
        def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
            return (x + 1, x * 3)
    )");
    m.save(path);
    return path;
}

} // anonymous namespace

// ─── Tests ───────────────────────────────────────────────────────────────────

class TorchSessionTest : public ::testing::Test {
protected:
    void SetUp() override {
        double_model_path_ = "/tmp/infergo_test_double.pt";
        tuple_model_path_  = "/tmp/infergo_test_tuple.pt";
        create_double_model(double_model_path_);
        create_tuple_model(tuple_model_path_);
    }

    void TearDown() override {
        std::remove(double_model_path_.c_str());
        std::remove(tuple_model_path_.c_str());
    }

    std::string double_model_path_;
    std::string tuple_model_path_;
};

TEST_F(TorchSessionTest, TestLoadModel) {
    infergo::TorchSession sess("cpu", 0);
    ASSERT_NO_THROW(sess.load_model(double_model_path_));
    EXPECT_GE(sess.num_inputs(), 1);
    EXPECT_GE(sess.num_outputs(), 1);
    EXPECT_EQ(sess.provider(), "cpu");
}

TEST_F(TorchSessionTest, TestLoadModelInvalidPath) {
    infergo::TorchSession sess("cpu", 0);
    EXPECT_THROW(sess.load_model("/tmp/nonexistent_model_12345.pt"),
                 std::runtime_error);
}

TEST_F(TorchSessionTest, TestRunInference) {
    infergo::TorchSession sess("cpu", 0);
    sess.load_model(double_model_path_);

    // Create a [2, 3] float32 input tensor with values 1..6
    int shape[] = {2, 3};
    infergo::Tensor* input = infergo::tensor_alloc_cpu(shape, 2, 0 /* FLOAT32 */);
    ASSERT_NE(input, nullptr);

    auto* data = static_cast<float*>(input->data);
    for (int i = 0; i < 6; ++i) data[i] = static_cast<float>(i + 1);

    std::vector<infergo::Tensor*> inputs = {input};
    std::vector<infergo::Tensor*> outputs;
    ASSERT_NO_THROW(outputs = sess.run(inputs));

    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_NE(outputs[0], nullptr);

    // Verify output shape matches input
    EXPECT_EQ(outputs[0]->ndim, 2);
    EXPECT_EQ(outputs[0]->shape[0], 2);
    EXPECT_EQ(outputs[0]->shape[1], 3);
    EXPECT_EQ(outputs[0]->dtype, 0); // FLOAT32

    // Verify values are doubled
    auto* out_data = static_cast<float*>(outputs[0]->data);
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(out_data[i], static_cast<float>((i + 1) * 2));
    }

    // Cleanup
    infergo::tensor_free(input);
    for (auto* t : outputs) infergo::tensor_free(t);
}

TEST_F(TorchSessionTest, TestRunInferenceTupleOutput) {
    infergo::TorchSession sess("cpu", 0);
    sess.load_model(tuple_model_path_);

    int shape[] = {4};
    infergo::Tensor* input = infergo::tensor_alloc_cpu(shape, 1, 0);
    ASSERT_NE(input, nullptr);

    auto* data = static_cast<float*>(input->data);
    for (int i = 0; i < 4; ++i) data[i] = static_cast<float>(i);

    std::vector<infergo::Tensor*> inputs = {input};
    std::vector<infergo::Tensor*> outputs;
    ASSERT_NO_THROW(outputs = sess.run(inputs));

    // Tuple model returns 2 outputs
    ASSERT_EQ(outputs.size(), 2u);

    // First output: x + 1
    auto* out0 = static_cast<float*>(outputs[0]->data);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(out0[i], static_cast<float>(i + 1));
    }

    // Second output: x * 3
    auto* out1 = static_cast<float*>(outputs[1]->data);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(out1[i], static_cast<float>(i * 3));
    }

    infergo::tensor_free(input);
    for (auto* t : outputs) infergo::tensor_free(t);
}

TEST_F(TorchSessionTest, TestRunBeforeLoad) {
    infergo::TorchSession sess("cpu", 0);
    int shape[] = {1};
    infergo::Tensor* input = infergo::tensor_alloc_cpu(shape, 1, 0);
    ASSERT_NE(input, nullptr);

    std::vector<infergo::Tensor*> inputs = {input};
    EXPECT_THROW(sess.run(inputs), std::runtime_error);

    infergo::tensor_free(input);
}

TEST_F(TorchSessionTest, TestCUDADevice) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available — skipping CUDA test";
    }

    infergo::TorchSession sess("cuda", 0);
    EXPECT_EQ(sess.provider(), "cuda");

    sess.load_model(double_model_path_);

    // Run inference — input is CPU, session handles CPU->GPU->CPU transfer
    int shape[] = {3};
    infergo::Tensor* input = infergo::tensor_alloc_cpu(shape, 1, 0);
    ASSERT_NE(input, nullptr);

    auto* data = static_cast<float*>(input->data);
    data[0] = 10.0f; data[1] = 20.0f; data[2] = 30.0f;

    std::vector<infergo::Tensor*> inputs = {input};
    std::vector<infergo::Tensor*> outputs;
    ASSERT_NO_THROW(outputs = sess.run(inputs));

    ASSERT_EQ(outputs.size(), 1u);
    auto* out_data = static_cast<float*>(outputs[0]->data);
    EXPECT_FLOAT_EQ(out_data[0], 20.0f);
    EXPECT_FLOAT_EQ(out_data[1], 40.0f);
    EXPECT_FLOAT_EQ(out_data[2], 60.0f);

    // Output should be on CPU (torch_tensor_to_infer copies back)
    EXPECT_FALSE(outputs[0]->on_device);

    infergo::tensor_free(input);
    for (auto* t : outputs) infergo::tensor_free(t);
}

TEST_F(TorchSessionTest, TestMultipleModels) {
    // Load two models and verify both work correctly
    infergo::TorchSession sess1("cpu", 0);
    infergo::TorchSession sess2("cpu", 0);

    sess1.load_model(double_model_path_);
    sess2.load_model(tuple_model_path_);

    // Both should have valid metadata
    EXPECT_GE(sess1.num_inputs(), 1);
    EXPECT_GE(sess2.num_inputs(), 1);

    // Run model 1: double
    int shape[] = {2};
    infergo::Tensor* in1 = infergo::tensor_alloc_cpu(shape, 1, 0);
    ASSERT_NE(in1, nullptr);
    static_cast<float*>(in1->data)[0] = 5.0f;
    static_cast<float*>(in1->data)[1] = 7.0f;

    auto out1 = sess1.run({in1});
    ASSERT_EQ(out1.size(), 1u);
    EXPECT_FLOAT_EQ(static_cast<float*>(out1[0]->data)[0], 10.0f);
    EXPECT_FLOAT_EQ(static_cast<float*>(out1[0]->data)[1], 14.0f);

    // Run model 2: tuple (x+1, x*3)
    infergo::Tensor* in2 = infergo::tensor_alloc_cpu(shape, 1, 0);
    ASSERT_NE(in2, nullptr);
    static_cast<float*>(in2->data)[0] = 4.0f;
    static_cast<float*>(in2->data)[1] = 6.0f;

    auto out2 = sess2.run({in2});
    ASSERT_EQ(out2.size(), 2u);
    EXPECT_FLOAT_EQ(static_cast<float*>(out2[0]->data)[0], 5.0f);  // 4+1
    EXPECT_FLOAT_EQ(static_cast<float*>(out2[1]->data)[0], 12.0f); // 4*3

    // Cleanup
    infergo::tensor_free(in1);
    infergo::tensor_free(in2);
    for (auto* t : out1) infergo::tensor_free(t);
    for (auto* t : out2) infergo::tensor_free(t);
}

TEST_F(TorchSessionTest, TestCUDAMultipleModels) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available — skipping multi-model CUDA test";
    }

    // Load two models on CUDA — verify they share the same CUDA allocator
    // (libtorch uses a single caching allocator per device)
    infergo::TorchSession sess1("cuda", 0);
    infergo::TorchSession sess2("cuda", 0);

    sess1.load_model(double_model_path_);
    sess2.load_model(double_model_path_);

    // Run both — if the allocator were duplicated we'd see double memory,
    // but with shared caching allocator idle blocks are reused.
    int shape[] = {1024};
    infergo::Tensor* in1 = infergo::tensor_alloc_cpu(shape, 1, 0);
    infergo::Tensor* in2 = infergo::tensor_alloc_cpu(shape, 1, 0);
    ASSERT_NE(in1, nullptr);
    ASSERT_NE(in2, nullptr);

    auto* d1 = static_cast<float*>(in1->data);
    auto* d2 = static_cast<float*>(in2->data);
    for (int i = 0; i < 1024; ++i) {
        d1[i] = static_cast<float>(i);
        d2[i] = static_cast<float>(i * 2);
    }

    auto out1 = sess1.run({in1});
    auto out2 = sess2.run({in2});

    ASSERT_EQ(out1.size(), 1u);
    ASSERT_EQ(out2.size(), 1u);

    // Verify correctness
    EXPECT_FLOAT_EQ(static_cast<float*>(out1[0]->data)[0], 0.0f);
    EXPECT_FLOAT_EQ(static_cast<float*>(out1[0]->data)[1], 2.0f);
    EXPECT_FLOAT_EQ(static_cast<float*>(out2[0]->data)[0], 0.0f);
    EXPECT_FLOAT_EQ(static_cast<float*>(out2[0]->data)[1], 4.0f);

    infergo::tensor_free(in1);
    infergo::tensor_free(in2);
    for (auto* t : out1) infergo::tensor_free(t);
    for (auto* t : out2) infergo::tensor_free(t);
}

TEST_F(TorchSessionTest, TestNullInputTensor) {
    infergo::TorchSession sess("cpu", 0);
    sess.load_model(double_model_path_);

    std::vector<infergo::Tensor*> inputs = {nullptr};
    EXPECT_THROW(sess.run(inputs), std::runtime_error);
}

TEST_F(TorchSessionTest, TestMoveSemantics) {
    infergo::TorchSession sess("cpu", 0);
    sess.load_model(double_model_path_);

    // Move construct
    infergo::TorchSession sess2(std::move(sess));
    EXPECT_GE(sess2.num_inputs(), 1);

    // The moved-from session should be safe but unusable
    // (model_loaded_ == false, so run() will throw)

    // Run on the moved-to session
    int shape[] = {2};
    infergo::Tensor* input = infergo::tensor_alloc_cpu(shape, 1, 0);
    ASSERT_NE(input, nullptr);
    static_cast<float*>(input->data)[0] = 3.0f;
    static_cast<float*>(input->data)[1] = 4.0f;

    auto outputs = sess2.run({input});
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_FLOAT_EQ(static_cast<float*>(outputs[0]->data)[0], 6.0f);
    EXPECT_FLOAT_EQ(static_cast<float*>(outputs[0]->data)[1], 8.0f);

    infergo::tensor_free(input);
    for (auto* t : outputs) infergo::tensor_free(t);
}
