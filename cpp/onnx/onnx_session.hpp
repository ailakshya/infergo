#pragma once

#include "../tensor/tensor.hpp"
#include "onnxruntime_c_api.h"

#include <string>
#include <vector>

namespace infergo {

/// Internal C++ class wrapping an ONNX Runtime session.
/// Not exposed through the C API — Go never sees this type.
///
/// Lifecycle:
///   OnnxSession s("cpu", 0);
///   s.load_model("/path/to/model.onnx");
///   auto outputs = s.run(inputs);
class OnnxSession {
public:
    /// Construct a session for the given execution provider.
    /// provider: "cpu" | "cuda" | "tensorrt" | "coreml" | "openvino"
    /// device_id: CUDA/TensorRT device index (ignored for CPU/CoreML)
    /// Throws std::runtime_error on failure.
    OnnxSession(const std::string& provider, int device_id);

    ~OnnxSession();

    // Non-copyable, movable.
    OnnxSession(const OnnxSession&) = delete;
    OnnxSession& operator=(const OnnxSession&) = delete;
    OnnxSession(OnnxSession&&) noexcept;
    OnnxSession& operator=(OnnxSession&&) noexcept;

    /// Load an ONNX model file.
    /// Throws std::runtime_error on failure.
    void load_model(const std::string& model_path);

    /// Run inference.
    /// inputs:  one Tensor* per model input, in order.
    ///          Each tensor must be a CPU tensor (on_device == false).
    /// Returns: one heap-allocated CPU Tensor* per model output.
    ///          Caller owns the returned tensors and must call tensor_free().
    /// Throws std::runtime_error on shape/dtype mismatch or ORT error.
    std::vector<Tensor*> run(const std::vector<Tensor*>& inputs);

    // ─── Metadata (valid after load_model) ───────────────────────────────────

    int num_inputs()  const noexcept { return static_cast<int>(input_names_.size()); }
    int num_outputs() const noexcept { return static_cast<int>(output_names_.size()); }

    const std::string& input_name(int idx)  const { return input_names_.at(idx); }
    const std::string& output_name(int idx) const { return output_names_.at(idx); }

private:
    void throw_on_error(OrtStatus* status) const;
    int  ort_dtype_to_infer(ONNXTensorElementDataType ort_type) const;

    const OrtApi*        api_      = nullptr;
    OrtEnv*              env_      = nullptr;
    OrtSessionOptions*   options_  = nullptr;
    OrtSession*          session_  = nullptr;
    OrtAllocator*        allocator_ = nullptr;

    std::string provider_;
    int         device_id_ = 0;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

} // namespace infergo
