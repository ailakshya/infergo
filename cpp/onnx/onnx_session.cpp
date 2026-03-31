#include "onnx_session.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace infergo {

// ─── Helpers ─────────────────────────────────────────────────────────────────

void OnnxSession::throw_on_error(OrtStatus* status) const {
    if (status == nullptr) return;
    const char* msg = api_->GetErrorMessage(status);
    std::string err(msg ? msg : "unknown ORT error");
    api_->ReleaseStatus(status);
    throw std::runtime_error(err);
}

int OnnxSession::ort_dtype_to_infer(ONNXTensorElementDataType ort_type) const {
    switch (ort_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return 0; // FLOAT32
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return 1; // FLOAT16
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:return 2; // BFLOAT16
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return 3; // INT32
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return 4; // INT64
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return 5; // UINT8
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return 6; // BOOL
        default:
            throw std::runtime_error("OnnxSession: unsupported ORT element type: " +
                                     std::to_string(static_cast<int>(ort_type)));
    }
}

static ONNXTensorElementDataType infer_dtype_to_ort(int dtype) {
    switch (dtype) {
        case 0: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case 1: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        case 2: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
        case 3: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        case 4: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        case 5: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        case 6: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
        default:
            throw std::runtime_error("OnnxSession: unsupported infergo dtype: " +
                                     std::to_string(dtype));
    }
}

// ─── Constructor / Destructor ─────────────────────────────────────────────────

OnnxSession::OnnxSession(const std::string& provider, int device_id)
    : provider_(provider), device_id_(device_id)
{
    api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (api_ == nullptr) {
        throw std::runtime_error("OnnxSession: failed to get ORT API");
    }

    throw_on_error(api_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "infergo", &env_));
    throw_on_error(api_->CreateSessionOptions(&options_));

    // Performance defaults
    throw_on_error(api_->SetSessionGraphOptimizationLevel(options_, ORT_ENABLE_ALL));
    throw_on_error(api_->SetIntraOpNumThreads(options_, 0)); // 0 = use all cores
}

OnnxSession::~OnnxSession() {
    if (session_)  { api_->ReleaseSession(session_);        session_  = nullptr; }
    if (options_)  { api_->ReleaseSessionOptions(options_); options_  = nullptr; }
    if (env_)      { api_->ReleaseEnv(env_);                env_      = nullptr; }
}

OnnxSession::OnnxSession(OnnxSession&& other) noexcept
    : api_(other.api_)
    , env_(other.env_)
    , options_(other.options_)
    , session_(other.session_)
    , allocator_(other.allocator_)
    , provider_(std::move(other.provider_))
    , device_id_(other.device_id_)
    , input_names_(std::move(other.input_names_))
    , output_names_(std::move(other.output_names_))
{
    other.api_      = nullptr;
    other.env_      = nullptr;
    other.options_  = nullptr;
    other.session_  = nullptr;
    other.allocator_= nullptr;
}

OnnxSession& OnnxSession::operator=(OnnxSession&& other) noexcept {
    if (this != &other) {
        if (session_) api_->ReleaseSession(session_);
        if (options_) api_->ReleaseSessionOptions(options_);
        if (env_)     api_->ReleaseEnv(env_);

        api_          = other.api_;
        env_          = other.env_;
        options_      = other.options_;
        session_      = other.session_;
        allocator_    = other.allocator_;
        provider_     = std::move(other.provider_);
        device_id_    = other.device_id_;
        input_names_  = std::move(other.input_names_);
        output_names_ = std::move(other.output_names_);

        other.api_      = nullptr;
        other.env_      = nullptr;
        other.options_  = nullptr;
        other.session_  = nullptr;
        other.allocator_= nullptr;
    }
    return *this;
}

// ─── load_model ──────────────────────────────────────────────────────────────

void OnnxSession::load_model(const std::string& model_path) {
    if (session_) {
        api_->ReleaseSession(session_);
        session_ = nullptr;
        input_names_.clear();
        output_names_.clear();
    }

    throw_on_error(api_->CreateSession(env_, model_path.c_str(), options_, &session_));
    throw_on_error(api_->GetAllocatorWithDefaultOptions(&allocator_));

    // Read input names
    size_t n_inputs = 0;
    throw_on_error(api_->SessionGetInputCount(session_, &n_inputs));
    input_names_.resize(n_inputs);
    for (size_t i = 0; i < n_inputs; ++i) {
        char* name = nullptr;
        throw_on_error(api_->SessionGetInputName(session_, i, allocator_, &name));
        input_names_[i] = name;
        allocator_->Free(allocator_, name);
    }

    // Read output names
    size_t n_outputs = 0;
    throw_on_error(api_->SessionGetOutputCount(session_, &n_outputs));
    output_names_.resize(n_outputs);
    for (size_t i = 0; i < n_outputs; ++i) {
        char* name = nullptr;
        throw_on_error(api_->SessionGetOutputName(session_, i, allocator_, &name));
        output_names_[i] = name;
        allocator_->Free(allocator_, name);
    }
}

// ─── run ─────────────────────────────────────────────────────────────────────

std::vector<Tensor*> OnnxSession::run(const std::vector<Tensor*>& inputs) {
    if (session_ == nullptr) {
        throw std::runtime_error("OnnxSession::run: no model loaded");
    }
    if (inputs.size() != input_names_.size()) {
        throw std::runtime_error(
            "OnnxSession::run: expected " + std::to_string(input_names_.size()) +
            " inputs, got " + std::to_string(inputs.size()));
    }

    OrtMemoryInfo* cpu_mem_info = nullptr;
    throw_on_error(api_->CreateCpuMemoryInfo(
        OrtArenaAllocator, OrtMemTypeDefault, &cpu_mem_info));

    // Build OrtValue* for each input
    std::vector<OrtValue*> ort_inputs(inputs.size(), nullptr);
    for (size_t i = 0; i < inputs.size(); ++i) {
        const Tensor* t = inputs[i];
        if (t == nullptr) {
            api_->ReleaseMemoryInfo(cpu_mem_info);
            throw std::runtime_error("OnnxSession::run: input[" +
                                     std::to_string(i) + "] is null");
        }
        if (t->on_device) {
            api_->ReleaseMemoryInfo(cpu_mem_info);
            throw std::runtime_error("OnnxSession::run: input[" +
                                     std::to_string(i) + "] is on GPU — call tensor_to_host() first");
        }

        // Build int64_t shape for ORT
        std::vector<int64_t> ort_shape(t->ndim);
        for (int d = 0; d < t->ndim; ++d) {
            ort_shape[d] = static_cast<int64_t>(t->shape[d]);
        }

        OrtStatus* st = api_->CreateTensorWithDataAsOrtValue(
            cpu_mem_info,
            t->data,
            t->nbytes,
            ort_shape.data(),
            static_cast<size_t>(t->ndim),
            infer_dtype_to_ort(t->dtype),
            &ort_inputs[i]
        );
        if (st != nullptr) {
            for (size_t j = 0; j < i; ++j) api_->ReleaseValue(ort_inputs[j]);
            api_->ReleaseMemoryInfo(cpu_mem_info);
            throw_on_error(st);
        }
    }

    api_->ReleaseMemoryInfo(cpu_mem_info);

    // Build C-string pointer arrays for ORT
    std::vector<const char*> in_names(input_names_.size());
    for (size_t i = 0; i < input_names_.size(); ++i) {
        in_names[i] = input_names_[i].c_str();
    }
    std::vector<const char*> out_names(output_names_.size());
    for (size_t i = 0; i < output_names_.size(); ++i) {
        out_names[i] = output_names_[i].c_str();
    }

    std::vector<OrtValue*> ort_outputs(output_names_.size(), nullptr);

    OrtStatus* run_st = api_->Run(
        session_, nullptr,
        in_names.data(),  ort_inputs.data(),  ort_inputs.size(),
        out_names.data(), ort_outputs.size(),  ort_outputs.data()
    );

    for (auto* v : ort_inputs) api_->ReleaseValue(v);

    if (run_st != nullptr) {
        for (auto* v : ort_outputs) { if (v) api_->ReleaseValue(v); }
        throw_on_error(run_st);
    }

    // Convert OrtValue* outputs → Tensor*
    std::vector<Tensor*> results;
    results.reserve(ort_outputs.size());

    for (size_t i = 0; i < ort_outputs.size(); ++i) {
        OrtValue* ov = ort_outputs[i];

        OrtTensorTypeAndShapeInfo* info = nullptr;
        throw_on_error(api_->GetTensorTypeAndShape(ov, &info));

        size_t ndim = 0;
        throw_on_error(api_->GetDimensionsCount(info, &ndim));

        std::vector<int64_t> dims(ndim);
        throw_on_error(api_->GetDimensions(info, dims.data(), ndim));

        ONNXTensorElementDataType ort_dtype;
        throw_on_error(api_->GetTensorElementType(info, &ort_dtype));
        api_->ReleaseTensorTypeAndShapeInfo(info);

        // Convert shape
        std::vector<int> shape(ndim);
        for (size_t d = 0; d < ndim; ++d) {
            shape[d] = static_cast<int>(dims[d]);
        }

        int infer_dtype = ort_dtype_to_infer(ort_dtype);
        Tensor* out_t = tensor_alloc_cpu(shape.data(), static_cast<int>(ndim), infer_dtype);
        if (out_t == nullptr) {
            api_->ReleaseValue(ov);
            for (size_t j = i + 1; j < ort_outputs.size(); ++j) {
                if (ort_outputs[j]) api_->ReleaseValue(ort_outputs[j]);
            }
            for (auto* r : results) tensor_free(r);
            throw std::runtime_error("OnnxSession::run: failed to allocate output tensor");
        }

        // Copy ORT output data into our tensor
        void* ort_data = nullptr;
        throw_on_error(api_->GetTensorMutableData(ov, &ort_data));
        std::memcpy(out_t->data, ort_data, out_t->nbytes);

        api_->ReleaseValue(ov);
        results.push_back(out_t);
    }

    return results;
}

} // namespace infergo
