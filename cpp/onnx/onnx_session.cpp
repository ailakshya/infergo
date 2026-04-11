#include "onnx_session.hpp"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <string>
#include <cstdio>

namespace infergo {

// ─── GPU inference semaphore ─────────────────────────────────────────────────
// Limits concurrent ONNX Run() calls to avoid GPU OOM when multiple models
// are loaded. Only N inference calls proceed simultaneously; the rest queue.
// This keeps idle sessions' BFC arenas small (no workspace allocated).
static constexpr int kMaxConcurrentGpuRuns = 3;
static std::mutex g_gpu_mu;
static std::condition_variable g_gpu_cv;
static int g_gpu_active = 0;

struct GpuSlot {
    GpuSlot() {
        std::unique_lock<std::mutex> lk(g_gpu_mu);
        g_gpu_cv.wait(lk, [] { return g_gpu_active < kMaxConcurrentGpuRuns; });
        ++g_gpu_active;
    }
    ~GpuSlot() {
        std::lock_guard<std::mutex> lk(g_gpu_mu);
        --g_gpu_active;
        g_gpu_cv.notify_one();
    }
    GpuSlot(const GpuSlot&) = delete;
    GpuSlot& operator=(const GpuSlot&) = delete;
};

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

// ─── Global shared OrtEnv ────────────────────────────────────────────────────
// All sessions share one OrtEnv to reduce CUDA context duplication.
// Note: ORT 1.24 does not support shared CUDA allocators via
// CreateAndRegisterAllocatorV2 — each session gets its own BFC arena.
static OrtEnv* g_shared_env = nullptr;
static const OrtApi* g_api = nullptr;

static OrtEnv* get_shared_env(const OrtApi* api) {
    if (g_shared_env == nullptr) {
        g_api = api;
        OrtStatus* st = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "infergo", &g_shared_env);
        if (st != nullptr) {
            const char* msg = api->GetErrorMessage(st);
            std::string err(msg ? msg : "unknown");
            api->ReleaseStatus(st);
            throw std::runtime_error("OnnxSession: CreateEnv failed: " + err);
        }
    }
    return g_shared_env;
}

OnnxSession::OnnxSession(const std::string& provider, int device_id)
    : provider_(provider), device_id_(device_id)
{
    api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (api_ == nullptr) {
        throw std::runtime_error("OnnxSession: failed to get ORT API");
    }

    env_ = get_shared_env(api_);  // shared, NOT owned — do not release in destructor
    owns_env_ = false;
    throw_on_error(api_->CreateSessionOptions(&options_));

    // Performance defaults
    throw_on_error(api_->SetSessionGraphOptimizationLevel(options_, ORT_ENABLE_ALL));
    throw_on_error(api_->SetIntraOpNumThreads(options_, 4));
    throw_on_error(api_->SetSessionExecutionMode(options_, ORT_SEQUENTIAL));
    throw_on_error(api_->EnableMemPattern(options_));
    throw_on_error(api_->EnableCpuMemArena(options_));


    // ── Execution provider selection ──────────────────────────────────────────
    // Each provider is attempted and falls back to CPU on failure (RULE: never
    // crash when a provider is unavailable — log and degrade gracefully).

    if (provider == "cuda") {
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.device_id = device_id;
        // Limit per-session GPU memory to avoid OOM when multiple models are loaded.
        // 0 = unlimited (default). We set a 2GB limit so 4+ models can coexist on
        // a 16GB GPU. The BFC arena will allocate within this budget and fall back
        // to CPU for any overflow, rather than crashing.
        cuda_opts.gpu_mem_limit = 512ULL * 1024 * 1024; // 512 MB per session (TRT needs far less than CUDA EP)
        // Use kSameAsRequested (1) instead of kNextPowerOfTwo (0) to avoid
        // the arena pre-allocating double the requested memory.
        cuda_opts.arena_extend_strategy = 1;
        // Allow CUDA to use all available CUDA streams for better overlap.
        cuda_opts.do_copy_in_default_stream = 0;
        // Enable cuDNN conv algorithm search for faster convolutions (key for YOLO).
        cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        OrtStatus* st = api_->SessionOptionsAppendExecutionProvider_CUDA(options_, &cuda_opts);
        if (st != nullptr) {
            const char* msg = api_->GetErrorMessage(st);
            std::fprintf(stderr,
                "[infergo] CUDA provider unavailable (%s) — falling back to CPU\n",
                msg ? msg : "unknown");
            api_->ReleaseStatus(st);
            provider_ = "cpu";
        }
    } else if (provider == "tensorrt") {
        OrtTensorRTProviderOptions trt_opts{};
        trt_opts.device_id = device_id;
        OrtStatus* st = api_->SessionOptionsAppendExecutionProvider_TensorRT(options_, &trt_opts);
        if (st != nullptr) {
            const char* msg = api_->GetErrorMessage(st);
            std::fprintf(stderr,
                "[infergo] TensorRT provider unavailable (%s) — falling back to CPU\n",
                msg ? msg : "unknown");
            api_->ReleaseStatus(st);
            provider_ = "cpu";
        }
    } else if (provider == "coreml") {
        // CoreML is only available on Apple platforms. ORT exposes it via the
        // generic AppendExecutionProvider() call using the provider name string.
        OrtStatus* st = api_->SessionOptionsAppendExecutionProvider(
            options_, "CoreML", nullptr, nullptr, 0);
        if (st != nullptr) {
            const char* msg = api_->GetErrorMessage(st);
            std::fprintf(stderr,
                "[infergo] CoreML provider unavailable (%s) — falling back to CPU\n",
                msg ? msg : "unknown");
            api_->ReleaseStatus(st);
            provider_ = "cpu";
        }
    } else if (provider == "openvino") {
        OrtStatus* st = api_->SessionOptionsAppendExecutionProvider(
            options_, "OpenVINO", nullptr, nullptr, 0);
        if (st != nullptr) {
            const char* msg = api_->GetErrorMessage(st);
            std::fprintf(stderr,
                "[infergo] OpenVINO provider unavailable (%s) — falling back to CPU\n",
                msg ? msg : "unknown");
            api_->ReleaseStatus(st);
            provider_ = "cpu";
        }
    } else if (provider != "cpu") {
        std::fprintf(stderr,
            "[infergo] Unknown provider '%s' — falling back to CPU\n",
            provider.c_str());
        provider_ = "cpu";
    }
    // "cpu" needs no explicit registration — it is always the ORT default.
}

OnnxSession::~OnnxSession() {
    if (session_)  { api_->ReleaseSession(session_);        session_  = nullptr; }
    if (options_)  { api_->ReleaseSessionOptions(options_); options_  = nullptr; }
    // env_ is shared globally — only release if we own it (never in current design).
    if (owns_env_ && env_) { api_->ReleaseEnv(env_); }
    env_ = nullptr;
}

OnnxSession::OnnxSession(OnnxSession&& other) noexcept
    : api_(other.api_)
    , env_(other.env_)
    , options_(other.options_)
    , session_(other.session_)
    , allocator_(other.allocator_)
    , provider_(std::move(other.provider_))
    , device_id_(other.device_id_)
    , owns_env_(other.owns_env_)
    , input_names_(std::move(other.input_names_))
    , output_names_(std::move(other.output_names_))
{
    other.api_      = nullptr;
    other.env_      = nullptr;
    other.options_  = nullptr;
    other.session_  = nullptr;
    other.allocator_= nullptr;
    other.owns_env_ = false;
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
        owns_env_     = other.owns_env_;
        input_names_  = std::move(other.input_names_);
        output_names_ = std::move(other.output_names_);

        other.api_      = nullptr;
        other.env_      = nullptr;
        other.options_  = nullptr;
        other.session_  = nullptr;
        other.allocator_= nullptr;
        other.owns_env_ = false;
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

// Thread-local scratch buffers to avoid per-call heap allocations.
// Each calling thread gets its own set — no locking needed.
static thread_local OrtMemoryInfo* tl_cpu_mem = nullptr;
static thread_local std::vector<OrtValue*> tl_ort_inputs;
static thread_local std::vector<OrtValue*> tl_ort_outputs;
static thread_local std::vector<const char*> tl_in_names;
static thread_local std::vector<const char*> tl_out_names;
static thread_local std::vector<int64_t> tl_shape;

std::vector<Tensor*> OnnxSession::run(const std::vector<Tensor*>& inputs) {
    if (session_ == nullptr) {
        throw std::runtime_error("OnnxSession::run: no model loaded");
    }
    const size_t n_in = input_names_.size();
    const size_t n_out = output_names_.size();
    if (inputs.size() != n_in) {
        throw std::runtime_error(
            "OnnxSession::run: expected " + std::to_string(n_in) +
            " inputs, got " + std::to_string(inputs.size()));
    }

    // Reuse thread-local CPU memory info (never freed — lives until thread exits).
    if (tl_cpu_mem == nullptr) {
        throw_on_error(api_->CreateCpuMemoryInfo(
            OrtArenaAllocator, OrtMemTypeDefault, &tl_cpu_mem));
    }

    // Resize thread-local scratch buffers (no alloc after first call).
    tl_ort_inputs.resize(n_in);
    tl_ort_outputs.assign(n_out, nullptr);
    tl_in_names.resize(n_in);
    tl_out_names.resize(n_out);

    for (size_t i = 0; i < n_in; ++i) tl_in_names[i] = input_names_[i].c_str();
    for (size_t i = 0; i < n_out; ++i) tl_out_names[i] = output_names_[i].c_str();

    // Wrap input tensors as OrtValues (zero-copy — wraps existing CPU data).
    for (size_t i = 0; i < n_in; ++i) {
        const Tensor* t = inputs[i];
        tl_shape.resize(t->ndim);
        for (int d = 0; d < t->ndim; ++d)
            tl_shape[d] = static_cast<int64_t>(t->shape[d]);

        throw_on_error(api_->CreateTensorWithDataAsOrtValue(
            tl_cpu_mem, t->data, t->nbytes,
            tl_shape.data(), static_cast<size_t>(t->ndim),
            infer_dtype_to_ort(t->dtype), &tl_ort_inputs[i]));
    }

    // ── GPU inference (semaphore-gated) ──────────────────────────────────────
    GpuSlot gpu_slot;

    OrtStatus* run_st = api_->Run(
        session_, nullptr,
        tl_in_names.data(), tl_ort_inputs.data(), n_in,
        tl_out_names.data(), n_out, tl_ort_outputs.data());

    // Release input OrtValues immediately (they just wrap our CPU data).
    for (size_t i = 0; i < n_in; ++i) {
        api_->ReleaseValue(tl_ort_inputs[i]);
        tl_ort_inputs[i] = nullptr;
    }

    if (run_st != nullptr) {
        for (auto* v : tl_ort_outputs) { if (v) api_->ReleaseValue(v); }
        throw_on_error(run_st);
    }

    // ── Convert outputs → Tensor* ────────────────────────────────────────────
    std::vector<Tensor*> results;
    results.reserve(n_out);

    for (size_t i = 0; i < n_out; ++i) {
        OrtValue* ov = tl_ort_outputs[i];

        OrtTensorTypeAndShapeInfo* info = nullptr;
        throw_on_error(api_->GetTensorTypeAndShape(ov, &info));

        size_t ndim = 0;
        throw_on_error(api_->GetDimensionsCount(info, &ndim));

        int64_t dims[8]; // max 8 dims — stack-allocated, no heap
        throw_on_error(api_->GetDimensions(info, dims, ndim));

        ONNXTensorElementDataType ort_dtype;
        throw_on_error(api_->GetTensorElementType(info, &ort_dtype));
        api_->ReleaseTensorTypeAndShapeInfo(info);

        int shape[8];
        for (size_t d = 0; d < ndim; ++d) shape[d] = static_cast<int>(dims[d]);

        Tensor* out_t = tensor_alloc_cpu(shape, static_cast<int>(ndim),
                                          ort_dtype_to_infer(ort_dtype));
        if (out_t == nullptr) {
            api_->ReleaseValue(ov);
            for (size_t j = i + 1; j < n_out; ++j)
                if (tl_ort_outputs[j]) api_->ReleaseValue(tl_ort_outputs[j]);
            for (auto* r : results) tensor_free(r);
            throw std::runtime_error("OnnxSession::run: output tensor alloc failed");
        }

        void* ort_data = nullptr;
        throw_on_error(api_->GetTensorMutableData(ov, &ort_data));
        std::memcpy(out_t->data, ort_data, out_t->nbytes);

        api_->ReleaseValue(ov);
        results.push_back(out_t);
    }

    return results;
}

} // namespace infergo
