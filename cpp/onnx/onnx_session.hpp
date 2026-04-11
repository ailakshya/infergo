#pragma once

#include "../tensor/tensor.hpp"
#include "onnxruntime_c_api.h"

#include <string>
#include <vector>
#include <mutex>

namespace infergo {

class OnnxSession {
public:
    OnnxSession(const std::string& provider, int device_id);
    ~OnnxSession();

    OnnxSession(const OnnxSession&) = delete;
    OnnxSession& operator=(const OnnxSession&) = delete;
    OnnxSession(OnnxSession&&) noexcept;
    OnnxSession& operator=(OnnxSession&&) noexcept;

    void load_model(const std::string& model_path);

    /// Run inference. inputs must be CPU tensors.
    /// Returns heap-allocated CPU output tensors (caller frees).
    std::vector<Tensor*> run(const std::vector<Tensor*>& inputs);

    int num_inputs()  const noexcept { return static_cast<int>(input_names_.size()); }
    int num_outputs() const noexcept { return static_cast<int>(output_names_.size()); }

    const std::string& input_name(int idx)  const { return input_names_.at(idx); }
    const std::string& output_name(int idx) const { return output_names_.at(idx); }

    const std::string& provider() const noexcept { return provider_; }

private:
    void throw_on_error(OrtStatus* status) const;
    int  ort_dtype_to_infer(ONNXTensorElementDataType ort_type) const;

    // Setup IO Binding for GPU-resident I/O (called once after load_model).
    void setup_io_binding();

    const OrtApi*        api_      = nullptr;
    OrtEnv*              env_      = nullptr;
    OrtSessionOptions*   options_  = nullptr;
    OrtSession*          session_  = nullptr;
    OrtAllocator*        allocator_ = nullptr;

    std::string provider_;
    int         device_id_ = 0;
    bool        owns_env_  = false;
    bool        use_cuda_  = false;  // true if CUDA provider is active

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

    // ─── IO Binding state (CUDA only) ─────────────────────────────────
    // Pre-allocated GPU input/output buffers to avoid per-call CPU↔GPU copies.
    // Protected by mu_ because multiple goroutines may call run() concurrently.
    std::mutex            mu_;
    OrtIoBinding*         io_binding_ = nullptr;
    OrtMemoryInfo*        cuda_mem_info_ = nullptr;
    void*                 gpu_input_buf_ = nullptr;   // pre-allocated GPU input
    size_t                gpu_input_bytes_ = 0;
    OrtValue*             gpu_input_val_ = nullptr;   // OrtValue wrapping gpu_input_buf_
    // Input shape cached from first run
    std::vector<int64_t>  input_shape_;
    bool                  io_bound_ = false;
};

} // namespace infergo
