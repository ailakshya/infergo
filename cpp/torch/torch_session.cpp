#include "torch_session.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace infergo {

// ─── Dtype conversion helpers ────────────────────────────────────────────────

torch::ScalarType TorchSession::infer_dtype_to_torch(int dtype) {
    switch (dtype) {
        case 0: return torch::kFloat32;
        case 1: return torch::kFloat16;
        case 2: return torch::kBFloat16;
        case 3: return torch::kInt32;
        case 4: return torch::kInt64;
        case 5: return torch::kUInt8;
        case 6: return torch::kBool;
        default:
            throw std::runtime_error("TorchSession: unsupported infergo dtype: " +
                                     std::to_string(dtype));
    }
}

int TorchSession::torch_dtype_to_infer(torch::ScalarType st) {
    switch (st) {
        case torch::kFloat32:  return 0;
        case torch::kFloat16:  return 1;
        case torch::kBFloat16: return 2;
        case torch::kInt32:    return 3;
        case torch::kInt64:    return 4;
        case torch::kUInt8:    return 5;
        case torch::kBool:     return 6;
        default:
            throw std::runtime_error(
                "TorchSession: unsupported torch ScalarType: " +
                std::to_string(static_cast<int>(st)));
    }
}

// ─── Constructor / Destructor ────────────────────────────────────────────────

TorchSession::TorchSession(const std::string& provider, int device_id)
    : device_(torch::kCPU)
    , provider_(provider)
    , device_id_(device_id)
{
    if (provider == "cuda") {
        if (torch::cuda::is_available()) {
            device_ = torch::Device(torch::kCUDA, device_id);
        } else {
            std::fprintf(stderr,
                "[infergo] CUDA not available in libtorch — falling back to CPU\n");
            provider_ = "cpu";
        }
    } else if (provider != "cpu") {
        std::fprintf(stderr,
            "[infergo] TorchSession: unknown provider '%s' — falling back to CPU\n",
            provider.c_str());
        provider_ = "cpu";
    }
}

TorchSession::~TorchSession() {
    // torch::jit::Module destructor handles cleanup.
    // Explicitly mark as unloaded.
    model_loaded_ = false;
}

TorchSession::TorchSession(TorchSession&& other) noexcept
    : model_(std::move(other.model_))
    , device_(other.device_)
    , provider_(std::move(other.provider_))
    , device_id_(other.device_id_)
    , model_loaded_(other.model_loaded_)
    , num_inputs_(other.num_inputs_)
    , num_outputs_(other.num_outputs_)
    , input_names_(std::move(other.input_names_))
    , output_names_(std::move(other.output_names_))
{
    other.model_loaded_ = false;
    other.num_inputs_   = 0;
    other.num_outputs_  = 0;
}

TorchSession& TorchSession::operator=(TorchSession&& other) noexcept {
    if (this != &other) {
        model_        = std::move(other.model_);
        device_       = other.device_;
        provider_     = std::move(other.provider_);
        device_id_    = other.device_id_;
        model_loaded_ = other.model_loaded_;
        num_inputs_   = other.num_inputs_;
        num_outputs_  = other.num_outputs_;
        input_names_  = std::move(other.input_names_);
        output_names_ = std::move(other.output_names_);

        other.model_loaded_ = false;
        other.num_inputs_   = 0;
        other.num_outputs_  = 0;
    }
    return *this;
}

// ─── load_model ──────────────────────────────────────────────────────────────

void TorchSession::load_model(const std::string& model_path) {
    try {
        model_ = torch::jit::load(model_path, device_);
    } catch (const c10::Error& e) {
        throw std::runtime_error(
            "TorchSession::load_model: failed to load '" + model_path +
            "': " + e.what());
    }

    model_.eval();

    // Freeze: marks module as immutable, enables constant propagation + dead code elimination.
    // This is free optimization — no downsides, typically 3-7% faster inference.
    try {
        model_ = torch::jit::freeze(model_);
    } catch (...) {
        // Some models don't support freeze (e.g., with dynamic control flow).
        // Non-fatal — continue with unfrozen model.
    }

    // Optimize for inference: fuses conv+bn, removes autograd overhead.
    // Typically 7-17% faster for models with BatchNorm layers (all YOLO models).
    try {
        model_ = torch::jit::optimize_for_inference(model_);
    } catch (...) {
        // Non-fatal — continue with unoptimized model.
    }

    model_loaded_ = true;

    // TorchScript models don't expose named input/output metadata the way
    // ONNX does.  We inspect the forward method's schema to extract argument
    // names (excluding 'self') as input names.  Output names are synthetic
    // ("output_0", "output_1", ...) since TorchScript return types don't
    // carry names.
    input_names_.clear();
    output_names_.clear();

    const auto& method = model_.get_method("forward");
    const auto& schema = method.function().getSchema();

    // Input names: skip argument 0 ("self")
    const auto& args = schema.arguments();
    num_inputs_ = 0;
    for (size_t i = 1; i < args.size(); ++i) {
        input_names_.push_back(args[i].name());
        ++num_inputs_;
    }
    // If we couldn't extract any names, provide a fallback
    if (num_inputs_ == 0) {
        num_inputs_ = 1;
        input_names_.push_back("input_0");
    }

    // Warm-up forward pass with a small dummy tensor.
    // This triggers JIT compilation and CUDA kernel caching so the first
    // real inference call is not penalised.
    {
        torch::NoGradGuard no_grad;
        try {
            auto dummy = torch::zeros({1}, torch::TensorOptions()
                                                .dtype(torch::kFloat32)
                                                .device(device_));
            std::vector<torch::jit::IValue> dummy_inputs;
            dummy_inputs.emplace_back(dummy);
            auto result = model_.forward(dummy_inputs);

            // Determine number of outputs from the warm-up result
            if (result.isTuple()) {
                const auto& elems = result.toTuple()->elements();
                num_outputs_ = static_cast<int>(elems.size());
            } else if (result.isTensorList()) {
                num_outputs_ = static_cast<int>(result.toTensorList().size());
            } else {
                // Single tensor output
                num_outputs_ = 1;
            }
        } catch (...) {
            // Warm-up may fail if the model expects a specific input shape.
            // That is fine — we just won't know the output count until run().
            // Default to 1 output.
            if (num_outputs_ == 0) {
                num_outputs_ = 1;
            }
        }
    }

    // Generate synthetic output names
    for (int i = 0; i < num_outputs_; ++i) {
        output_names_.push_back("output_" + std::to_string(i));
    }
}

// ─── torch_tensor_to_infer ───────────────────────────────────────────────────

Tensor* TorchSession::torch_tensor_to_infer(const torch::Tensor& t) {
    // Move to CPU and make contiguous
    torch::Tensor cpu_t = t.to(torch::kCPU).contiguous();

    const int ndim = static_cast<int>(cpu_t.dim());
    if (ndim <= 0 || ndim > 8) {
        throw std::runtime_error(
            "TorchSession: output tensor has unsupported ndim=" +
            std::to_string(ndim));
    }

    int shape[8];
    for (int d = 0; d < ndim; ++d) {
        shape[d] = static_cast<int>(cpu_t.size(d));
    }

    const int dtype = torch_dtype_to_infer(cpu_t.scalar_type());
    Tensor* out = tensor_alloc_cpu(shape, ndim, dtype);
    if (out == nullptr) {
        throw std::runtime_error("TorchSession: output tensor alloc failed");
    }

    std::memcpy(out->data, cpu_t.data_ptr(), out->nbytes);
    return out;
}

// ─── run ─────────────────────────────────────────────────────────────────────

std::vector<Tensor*> TorchSession::run(const std::vector<Tensor*>& inputs) {
    if (!model_loaded_) {
        throw std::runtime_error("TorchSession::run: no model loaded");
    }

    // Convert infergo Tensor* inputs to torch::Tensor IValues
    std::vector<torch::jit::IValue> torch_inputs;
    torch_inputs.reserve(inputs.size());

    for (const Tensor* t : inputs) {
        if (t == nullptr) {
            throw std::runtime_error("TorchSession::run: null input tensor");
        }

        // Build shape as int64_t
        std::vector<int64_t> shape(t->ndim);
        for (int d = 0; d < t->ndim; ++d) {
            shape[d] = static_cast<int64_t>(t->shape[d]);
        }

        const torch::ScalarType torch_dtype = infer_dtype_to_torch(t->dtype);

        // Create a torch::Tensor that wraps the existing CPU data (zero-copy).
        // The from_blob tensor does NOT own the data, which is fine because
        // we copy to device below and the input Tensor* outlives this call.
        torch::Tensor input_t = torch::from_blob(
            t->data, shape,
            torch::TensorOptions().dtype(torch_dtype));

        // Move to target device (no-op if already on the right device)
        input_t = input_t.to(device_);
        torch_inputs.emplace_back(input_t);
    }

    // Run inference with autograd disabled
    torch::NoGradGuard no_grad;

    torch::jit::IValue result;
    try {
        result = model_.forward(torch_inputs);
    } catch (const c10::Error& e) {
        throw std::runtime_error(
            std::string("TorchSession::run: forward failed: ") + e.what());
    }

    // Convert outputs to Tensor*
    std::vector<Tensor*> outputs;

    if (result.isTuple()) {
        const auto& elems = result.toTuple()->elements();
        outputs.reserve(elems.size());
        for (const auto& elem : elems) {
            if (elem.isTensor()) {
                outputs.push_back(torch_tensor_to_infer(elem.toTensor()));
            }
        }
    } else if (result.isTensorList()) {
        const auto& list = result.toTensorList();
        outputs.reserve(list.size());
        for (const torch::Tensor& tensor : list) {
            outputs.push_back(torch_tensor_to_infer(tensor));
        }
    } else if (result.isTensor()) {
        outputs.push_back(torch_tensor_to_infer(result.toTensor()));
    } else {
        throw std::runtime_error(
            "TorchSession::run: model returned unsupported IValue type "
            "(expected Tensor, Tuple of Tensors, or TensorList)");
    }

    // Update num_outputs_ from actual result (in case warm-up didn't determine it)
    num_outputs_ = static_cast<int>(outputs.size());

    return outputs;
}

// ─── run_gpu ────────────────────────────────────────────────────────────────
// Optimized path: upload inputs to GPU in one shot via non_blocking=true,
// then run inference entirely on GPU, then download only outputs.
// This avoids any intermediate CPU↔GPU synchronization that the standard
// run() path might trigger via from_blob + to(device) per input.

std::vector<Tensor*> TorchSession::run_gpu(const std::vector<Tensor*>& inputs) {
    if (!model_loaded_) {
        throw std::runtime_error("TorchSession::run_gpu: no model loaded");
    }

    // If on CPU, just delegate to regular run
    if (device_ == torch::Device(torch::kCPU)) {
        return run(inputs);
    }

    torch::NoGradGuard no_grad;

    // Build all input tensors and upload to GPU with non_blocking=true
    // so the H2D transfers overlap on the CUDA copy stream.
    std::vector<torch::jit::IValue> torch_inputs;
    torch_inputs.reserve(inputs.size());

    for (const Tensor* t : inputs) {
        if (t == nullptr) {
            throw std::runtime_error("TorchSession::run_gpu: null input tensor");
        }

        std::vector<int64_t> shape(t->ndim);
        for (int d = 0; d < t->ndim; ++d)
            shape[d] = static_cast<int64_t>(t->shape[d]);

        torch::Tensor input_t = torch::from_blob(
            t->data, shape,
            torch::TensorOptions().dtype(infer_dtype_to_torch(t->dtype)));

        // non_blocking=true: queue the H2D copy without waiting.
        // The compute stream will synchronize automatically before using the data.
        input_t = input_t.to(device_, /*non_blocking=*/true);
        torch_inputs.emplace_back(std::move(input_t));
    }

    // Run inference — libtorch ensures H2D copies complete before compute starts.
    torch::jit::IValue result;
    try {
        result = model_.forward(torch_inputs);
    } catch (const c10::Error& e) {
        throw std::runtime_error(
            std::string("TorchSession::run_gpu: forward failed: ") + e.what());
    }

    // Convert outputs to CPU Tensor*
    std::vector<Tensor*> outputs;

    if (result.isTuple()) {
        const auto& elems = result.toTuple()->elements();
        outputs.reserve(elems.size());
        for (const auto& elem : elems) {
            if (elem.isTensor()) {
                outputs.push_back(torch_tensor_to_infer(elem.toTensor()));
            }
        }
    } else if (result.isTensorList()) {
        const auto& list = result.toTensorList();
        outputs.reserve(list.size());
        for (const torch::Tensor& tensor : list) {
            outputs.push_back(torch_tensor_to_infer(tensor));
        }
    } else if (result.isTensor()) {
        outputs.push_back(torch_tensor_to_infer(result.toTensor()));
    } else {
        throw std::runtime_error(
            "TorchSession::run_gpu: unsupported output type");
    }

    num_outputs_ = static_cast<int>(outputs.size());
    return outputs;
}

} // namespace infergo
