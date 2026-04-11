#pragma once

#include "../tensor/tensor.hpp"

#include <torch/script.h>
#include <torch/torch.h>

#include <string>
#include <vector>

namespace infergo {

/// TorchSession — inference session for TorchScript (.pt) models.
///
/// Mirrors the OnnxSession interface: load a model, run inference with CPU
/// Tensor* inputs, get CPU Tensor* outputs.  libtorch handles CUDA
/// synchronisation internally so no external GPU semaphore is needed.
class TorchSession {
public:
    /// @param provider "cuda" or "cpu"
    /// @param device_id CUDA device index (ignored for CPU)
    TorchSession(const std::string& provider, int device_id);
    ~TorchSession();

    TorchSession(const TorchSession&) = delete;
    TorchSession& operator=(const TorchSession&) = delete;
    TorchSession(TorchSession&&) noexcept;
    TorchSession& operator=(TorchSession&&) noexcept;

    /// Load a TorchScript model (.pt file exported via torch.jit.trace or torch.jit.script).
    /// Moves the model to the configured device and sets eval mode.
    /// Performs a warm-up forward pass with a small dummy tensor.
    void load_model(const std::string& model_path);

    /// Run inference.  inputs must be CPU tensors (from tensor_alloc_cpu).
    /// Returns heap-allocated CPU output tensors (caller frees via tensor_free).
    std::vector<Tensor*> run(const std::vector<Tensor*>& inputs);

    /// GPU-optimized run: takes CPU input, uploads to GPU ONCE, runs inference,
    /// and only copies the small output back. Avoids per-layer CPU↔GPU copies.
    /// For YOLO: input is 4.9MB (uploaded once), output is 2.7MB (downloaded once).
    /// On CPU provider, this is identical to run().
    std::vector<Tensor*> run_gpu(const std::vector<Tensor*>& inputs);

    int num_inputs()  const noexcept { return num_inputs_; }
    int num_outputs() const noexcept { return num_outputs_; }

    const std::string& input_name(int idx)  const { return input_names_.at(idx); }
    const std::string& output_name(int idx) const { return output_names_.at(idx); }

    const std::string& provider() const noexcept { return provider_; }

    /// Access the underlying TorchScript module (needed by gpu_preprocess).
    torch::jit::Module& model() { return model_; }

    /// Access the configured device (needed by gpu_preprocess).
    const torch::Device& device() const { return device_; }

private:
    /// Convert an infergo dtype constant to a torch::ScalarType.
    static torch::ScalarType infer_dtype_to_torch(int dtype);

    /// Convert a torch::ScalarType to an infergo dtype constant.
    static int torch_dtype_to_infer(torch::ScalarType st);

    /// Convert a single torch::Tensor output to our Tensor* (CPU, heap-allocated).
    static Tensor* torch_tensor_to_infer(const torch::Tensor& t);

    torch::jit::Module model_;
    torch::Device       device_;
    std::string         provider_;
    int                 device_id_ = 0;
    bool                model_loaded_ = false;

    int num_inputs_  = 0;
    int num_outputs_ = 0;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

} // namespace infergo
