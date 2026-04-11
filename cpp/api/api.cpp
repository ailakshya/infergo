#include "infer_api.h"
#include "../tensor/tensor.hpp"
#include "../onnx/onnx_session.hpp"
#ifdef INFER_TORCH_AVAILABLE
#include "../torch/torch_session.hpp"
#include "../torch/gpu_preprocess.hpp"
#endif
#include "../tokenizer/tokenizer.hpp"
#include "../llm/kv_cache.hpp"
#include "../llm/kv_paged.hpp"
#include "../llm/llm_engine.hpp"
#include "../llm/infer_sequence.hpp"
#include "../../vendor/llama.cpp/include/llama.h"
#ifdef INFER_PREPROCESS_AVAILABLE
#include "../preprocess/preprocess.hpp"
#endif
#include "../postprocess/postprocess.hpp"
#ifdef INFER_CUDA_AVAILABLE
#include "../cuda/vram_monitor.hpp"
#endif
#ifdef INFER_VIDEO_AVAILABLE
#include "../video/decoder.hpp"
#include "../video/encoder.hpp"
#include "../video/frame_annotator.hpp"
#ifdef INFER_OPENCV_AVAILABLE
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif
#endif

#include <cstring>
#include <exception>
#include <vector>

// ─── Error string ─────────────────────────────────────────────────────────────

const char* infer_last_error_string(void) {
    return infergo::get_last_error();
}

// ─── Tensor API ───────────────────────────────────────────────────────────────

InferTensor infer_tensor_alloc_cpu(const int* shape, int ndim, int dtype) {
    try {
        return static_cast<InferTensor>(
            infergo::tensor_alloc_cpu(shape, ndim, dtype)
        );
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return nullptr;
    } catch (...) {
        infergo::set_last_error("infer_tensor_alloc_cpu: unknown exception");
        return nullptr;
    }
}

InferTensor infer_tensor_alloc_cuda(const int* shape, int ndim, int dtype, int device_id) {
#ifdef INFER_CUDA_AVAILABLE
    try {
        return static_cast<InferTensor>(
            infergo::tensor_alloc_cuda(shape, ndim, dtype, device_id)
        );
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return nullptr;
    } catch (...) {
        infergo::set_last_error("infer_tensor_alloc_cuda: unknown exception");
        return nullptr;
    }
#else
    (void)shape; (void)ndim; (void)dtype; (void)device_id;
    infergo::set_last_error("infer_tensor_alloc_cuda: CUDA not available in this build");
    return nullptr;
#endif
}

void infer_tensor_free(InferTensor t) {
    try {
        infergo::tensor_free(static_cast<infergo::Tensor*>(t));
    } catch (...) {
        // noexcept by contract — swallow silently
    }
}

void* infer_tensor_data_ptr(InferTensor t) {
    if (t == nullptr) { return nullptr; }
    return static_cast<infergo::Tensor*>(t)->data;
}

int infer_tensor_nbytes(InferTensor t) {
    try {
        return infergo::tensor_get_nbytes(static_cast<const infergo::Tensor*>(t));
    } catch (...) {
        return 0;
    }
}

int infer_tensor_nelements(InferTensor t) {
    try {
        return infergo::tensor_get_nelements(static_cast<const infergo::Tensor*>(t));
    } catch (...) {
        return 0;
    }
}

int infer_tensor_shape(InferTensor t, int* out_shape, int max_dims) {
    try {
        return infergo::tensor_get_shape(
            static_cast<const infergo::Tensor*>(t), out_shape, max_dims
        );
    } catch (...) {
        return 0;
    }
}

int infer_tensor_dtype(InferTensor t) {
    try {
        return infergo::tensor_get_dtype(static_cast<const infergo::Tensor*>(t));
    } catch (...) {
        return -1;
    }
}

InferError infer_tensor_to_device(InferTensor t, int device_id) {
#ifdef INFER_CUDA_AVAILABLE
    try {
        if (t == nullptr) {
            infergo::set_last_error("infer_tensor_to_device: null tensor");
            return INFER_ERR_NULL;
        }
        const bool ok = infergo::tensor_to_device(
            static_cast<infergo::Tensor*>(t), device_id
        );
        return ok ? INFER_OK : INFER_ERR_CUDA;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_RUNTIME;
    } catch (...) {
        return INFER_ERR_UNKNOWN;
    }
#else
    (void)t; (void)device_id;
    infergo::set_last_error("infer_tensor_to_device: CUDA not available in this build");
    return INFER_ERR_CUDA;
#endif
}

InferError infer_tensor_to_host(InferTensor t) {
#ifdef INFER_CUDA_AVAILABLE
    try {
        if (t == nullptr) {
            infergo::set_last_error("infer_tensor_to_host: null tensor");
            return INFER_ERR_NULL;
        }
        const bool ok = infergo::tensor_to_host(static_cast<infergo::Tensor*>(t));
        return ok ? INFER_OK : INFER_ERR_CUDA;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_RUNTIME;
    } catch (...) {
        return INFER_ERR_UNKNOWN;
    }
#else
    (void)t;
    infergo::set_last_error("infer_tensor_to_host: CUDA not available in this build");
    return INFER_ERR_CUDA;
#endif
}

InferError infer_tensor_copy_from(InferTensor t, const void* src, int nbytes) {
    try {
        if (t == nullptr) {
            infergo::set_last_error("infer_tensor_copy_from: null tensor");
            return INFER_ERR_NULL;
        }
        if (src == nullptr) {
            infergo::set_last_error("infer_tensor_copy_from: null src");
            return INFER_ERR_NULL;
        }
        if (nbytes <= 0) {
            infergo::set_last_error("infer_tensor_copy_from: nbytes must be > 0");
            return INFER_ERR_INVALID;
        }
        const bool ok = infergo::tensor_copy_from(
            static_cast<infergo::Tensor*>(t), src, nbytes
        );
        if (!ok) {
            // error string already set by tensor_copy_from
            return INFER_ERR_INVALID;
        }
        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_RUNTIME;
    } catch (...) {
        return INFER_ERR_UNKNOWN;
    }
}

// ─── Session API ──────────────────────────────────────────────────────────────

InferSession infer_session_create(const char* provider, int device_id) {
    try {
        const std::string p = (provider != nullptr) ? provider : "cpu";
        auto* s = new infergo::OnnxSession(p, device_id);
        return static_cast<InferSession>(s);
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return nullptr;
    } catch (...) {
        infergo::set_last_error("infer_session_create: unknown exception");
        return nullptr;
    }
}

InferError infer_session_load(InferSession s, const char* model_path) {
    try {
        if (s == nullptr) {
            infergo::set_last_error("infer_session_load: null session");
            return INFER_ERR_NULL;
        }
        if (model_path == nullptr) {
            infergo::set_last_error("infer_session_load: null model_path");
            return INFER_ERR_NULL;
        }
        static_cast<infergo::OnnxSession*>(s)->load_model(model_path);
        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_LOAD;
    } catch (...) {
        return INFER_ERR_UNKNOWN;
    }
}

int infer_session_num_inputs(InferSession s) {
    if (s == nullptr) return 0;
    return static_cast<infergo::OnnxSession*>(s)->num_inputs();
}

int infer_session_num_outputs(InferSession s) {
    if (s == nullptr) return 0;
    return static_cast<infergo::OnnxSession*>(s)->num_outputs();
}

InferError infer_session_input_name(InferSession s, int idx, char* out_buf, int buf_size) {
    try {
        if (s == nullptr || out_buf == nullptr || buf_size <= 0) {
            infergo::set_last_error("infer_session_input_name: invalid argument");
            return INFER_ERR_NULL;
        }
        const std::string& name =
            static_cast<infergo::OnnxSession*>(s)->input_name(idx);
        std::strncpy(out_buf, name.c_str(), static_cast<size_t>(buf_size) - 1);
        out_buf[buf_size - 1] = '\0';
        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_INVALID;
    } catch (...) {
        return INFER_ERR_UNKNOWN;
    }
}

InferError infer_session_output_name(InferSession s, int idx, char* out_buf, int buf_size) {
    try {
        if (s == nullptr || out_buf == nullptr || buf_size <= 0) {
            infergo::set_last_error("infer_session_output_name: invalid argument");
            return INFER_ERR_NULL;
        }
        const std::string& name =
            static_cast<infergo::OnnxSession*>(s)->output_name(idx);
        std::strncpy(out_buf, name.c_str(), static_cast<size_t>(buf_size) - 1);
        out_buf[buf_size - 1] = '\0';
        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_INVALID;
    } catch (...) {
        return INFER_ERR_UNKNOWN;
    }
}

InferError infer_session_run(
    InferSession  s,
    InferTensor*  inputs,  int n_inputs,
    InferTensor*  outputs, int n_outputs)
{
    try {
        if (s == nullptr) {
            infergo::set_last_error("infer_session_run: null session");
            return INFER_ERR_NULL;
        }
        if ((n_inputs > 0 && inputs == nullptr) || (n_outputs > 0 && outputs == nullptr)) {
            infergo::set_last_error("infer_session_run: null inputs/outputs array");
            return INFER_ERR_NULL;
        }

        std::vector<infergo::Tensor*> in_tensors(static_cast<size_t>(n_inputs));
        for (int i = 0; i < n_inputs; ++i) {
            in_tensors[i] = static_cast<infergo::Tensor*>(inputs[i]);
        }

        std::vector<infergo::Tensor*> out_tensors =
            static_cast<infergo::OnnxSession*>(s)->run(in_tensors);

        const int actual = static_cast<int>(out_tensors.size());
        const int copy_n = (actual < n_outputs) ? actual : n_outputs;
        for (int i = 0; i < copy_n; ++i) {
            outputs[i] = static_cast<InferTensor>(out_tensors[i]);
        }
        // Free any extra outputs not fitting in the caller's array
        for (int i = copy_n; i < actual; ++i) {
            infergo::tensor_free(out_tensors[i]);
        }

        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_RUNTIME;
    } catch (...) {
        return INFER_ERR_UNKNOWN;
    }
}

void infer_session_destroy(InferSession s) {
    if (s == nullptr) return;
    try {
        delete static_cast<infergo::OnnxSession*>(s);
    } catch (...) {
        // destructor must not throw
    }
}

// ─── Torch Session API ───────────────────────────────────────────────────────

#ifdef INFER_TORCH_AVAILABLE

InferTorchSession infer_torch_session_create(const char* provider, int device_id) {
    try {
        const std::string p = (provider != nullptr) ? provider : "cpu";
        auto* s = new infergo::TorchSession(p, device_id);
        return static_cast<InferTorchSession>(s);
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return nullptr;
    } catch (...) {
        infergo::set_last_error("infer_torch_session_create: unknown exception");
        return nullptr;
    }
}

InferError infer_torch_session_load(InferTorchSession s, const char* model_path) {
    try {
        if (s == nullptr) {
            infergo::set_last_error("infer_torch_session_load: null session");
            return INFER_ERR_NULL;
        }
        if (model_path == nullptr) {
            infergo::set_last_error("infer_torch_session_load: null model_path");
            return INFER_ERR_NULL;
        }
        static_cast<infergo::TorchSession*>(s)->load_model(model_path);
        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_LOAD;
    } catch (...) {
        return INFER_ERR_UNKNOWN;
    }
}

int infer_torch_session_num_inputs(InferTorchSession s) {
    if (s == nullptr) return 0;
    return static_cast<infergo::TorchSession*>(s)->num_inputs();
}

int infer_torch_session_num_outputs(InferTorchSession s) {
    if (s == nullptr) return 0;
    return static_cast<infergo::TorchSession*>(s)->num_outputs();
}

InferError infer_torch_session_run(
    InferTorchSession  s,
    InferTensor*       inputs,  int n_inputs,
    InferTensor*       outputs, int n_outputs)
{
    try {
        if (s == nullptr) {
            infergo::set_last_error("infer_torch_session_run: null session");
            return INFER_ERR_NULL;
        }
        if ((n_inputs > 0 && inputs == nullptr) || (n_outputs > 0 && outputs == nullptr)) {
            infergo::set_last_error("infer_torch_session_run: null inputs/outputs array");
            return INFER_ERR_NULL;
        }

        std::vector<infergo::Tensor*> in_tensors(static_cast<size_t>(n_inputs));
        for (int i = 0; i < n_inputs; ++i) {
            in_tensors[i] = static_cast<infergo::Tensor*>(inputs[i]);
        }

        std::vector<infergo::Tensor*> out_tensors =
            static_cast<infergo::TorchSession*>(s)->run(in_tensors);

        const int actual = static_cast<int>(out_tensors.size());
        const int copy_n = (actual < n_outputs) ? actual : n_outputs;
        for (int i = 0; i < copy_n; ++i) {
            outputs[i] = static_cast<InferTensor>(out_tensors[i]);
        }
        // Free any extra outputs not fitting in the caller's array
        for (int i = copy_n; i < actual; ++i) {
            infergo::tensor_free(out_tensors[i]);
        }

        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_RUNTIME;
    } catch (...) {
        return INFER_ERR_UNKNOWN;
    }
}

InferError infer_torch_session_run_gpu(
    InferTorchSession  s,
    InferTensor*       inputs,   int n_inputs,
    InferTensor*       outputs,  int n_outputs)
{
    try {
        if (s == nullptr) {
            infergo::set_last_error("infer_torch_session_run_gpu: null session");
            return INFER_ERR_NULL;
        }
        auto* sess = static_cast<infergo::TorchSession*>(s);

        std::vector<infergo::Tensor*> in_vec;
        in_vec.reserve(n_inputs);
        for (int i = 0; i < n_inputs; ++i)
            in_vec.push_back(static_cast<infergo::Tensor*>(inputs[i]));

        auto out_vec = sess->run_gpu(in_vec);

        int n = std::min(static_cast<int>(out_vec.size()), n_outputs);
        for (int i = 0; i < n; ++i)
            outputs[i] = static_cast<InferTensor>(out_vec[i]);
        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_RUNTIME;
    }
}

int infer_torch_detect_gpu_raw(
    void*          handle,
    const void*    rgb_data,
    int            width, int height,
    float          conf_thresh, float iou_thresh,
    InferBox*      out_boxes, int max_boxes)
{
    // Fast-path validation (no try/catch overhead for these cheap checks)
    if (!handle) { infergo::set_last_error("null session"); return -1; }
    if (!rgb_data || width <= 0 || height <= 0) { infergo::set_last_error("invalid RGB data"); return -1; }
    if (!out_boxes || max_boxes <= 0) { infergo::set_last_error("null output"); return -1; }

    // Use zero-copy path: writes directly to out_boxes, no vector alloc,
    // no Detection→InferBox field copy, no try/catch on the hot path.
    char error_buf[256] = {0};
    auto* sess = static_cast<infergo::TorchSession*>(handle);
    int count = infergo::torch_detect_gpu_raw_into(
        *sess, static_cast<const uint8_t*>(rgb_data), width, height,
        conf_thresh, iou_thresh,
        out_boxes, max_boxes,
        error_buf, sizeof(error_buf));

    if (count < 0 && error_buf[0] != '\0') {
        infergo::set_last_error(error_buf);
    }
    return count;
}

int infer_torch_detect_gpu_yuv(
    void* handle, const void* yuv_data, int width, int height, int linesize,
    float conf_thresh, float iou_thresh, InferBox* out_boxes, int max_boxes)
{
    try {
        if (!handle) { infergo::set_last_error("null session"); return -1; }
        if (!yuv_data || width <= 0 || height <= 0) { infergo::set_last_error("invalid YUV data"); return -1; }
        if (!out_boxes || max_boxes <= 0) { infergo::set_last_error("null output"); return -1; }

        auto* sess = static_cast<infergo::TorchSession*>(handle);
        auto dets = infergo::torch_detect_gpu_yuv(
            *sess, static_cast<const uint8_t*>(yuv_data), width, height, linesize,
            conf_thresh, iou_thresh);

        const int count = static_cast<int>(std::min(dets.size(), static_cast<size_t>(max_boxes)));
        // Detection and InferBox have identical layout — use memcpy
        if (count > 0) {
            std::memcpy(out_boxes, dets.data(),
                        static_cast<size_t>(count) * sizeof(InferBox));
        }
        return count;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return -1;
    }
}

void infer_torch_session_destroy(InferTorchSession s) {
    if (s == nullptr) return;
    try {
        delete static_cast<infergo::TorchSession*>(s);
    } catch (...) {
        // destructor must not throw
    }
}

int infer_torch_detect_gpu(
    void*          handle,
    const void*    jpeg_data,
    int            nbytes,
    float          conf_thresh,
    float          iou_thresh,
    InferBox*      out_boxes,
    int            max_boxes)
{
    try {
        if (handle == nullptr) {
            infergo::set_last_error("infer_torch_detect_gpu: null session");
            return -1;
        }
        if (jpeg_data == nullptr || nbytes <= 0) {
            infergo::set_last_error("infer_torch_detect_gpu: null or empty JPEG data");
            return -1;
        }
        if (out_boxes == nullptr || max_boxes <= 0) {
            infergo::set_last_error("infer_torch_detect_gpu: null or zero-capacity output array");
            return -1;
        }

        auto* sess = static_cast<infergo::TorchSession*>(handle);
        auto dets = infergo::torch_detect_gpu(
            *sess,
            static_cast<const uint8_t*>(jpeg_data), nbytes,
            conf_thresh, iou_thresh);

        const int count = static_cast<int>(
            std::min(dets.size(), static_cast<size_t>(max_boxes)));
        // Detection and InferBox have identical memory layout (verified by
        // static_assert in gpu_preprocess.cpp), so use memcpy instead of
        // field-by-field copy (~0.5ms saved for large detection counts).
        if (count > 0) {
            std::memcpy(out_boxes, dets.data(),
                        static_cast<size_t>(count) * sizeof(InferBox));
        }
        return count;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return -1;
    } catch (...) {
        infergo::set_last_error("infer_torch_detect_gpu: unknown exception");
        return -1;
    }
}

int infer_torch_detect_gpu_batch(
    void*          handle,
    const void**   jpeg_data_array,
    const int*     nbytes_array,
    int            batch_size,
    float          conf_thresh,
    float          iou_thresh,
    InferBox**     out_boxes_array,
    int*           out_counts,
    int            max_boxes_per_image)
{
    try {
        if (handle == nullptr) {
            infergo::set_last_error("infer_torch_detect_gpu_batch: null session");
            return -1;
        }
        if (jpeg_data_array == nullptr || nbytes_array == nullptr || batch_size <= 0) {
            infergo::set_last_error("infer_torch_detect_gpu_batch: invalid input arrays or batch_size");
            return -1;
        }
        if (out_boxes_array == nullptr || out_counts == nullptr || max_boxes_per_image <= 0) {
            infergo::set_last_error("infer_torch_detect_gpu_batch: null or zero-capacity output arrays");
            return -1;
        }

        auto* sess = static_cast<infergo::TorchSession*>(handle);

        // Build uint8_t** from void** for the C++ function.
        std::vector<const uint8_t*> data_ptrs(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            data_ptrs[i] = static_cast<const uint8_t*>(jpeg_data_array[i]);
        }

        auto all_dets = infergo::torch_detect_gpu_batch(
            *sess,
            data_ptrs.data(), nbytes_array, batch_size,
            conf_thresh, iou_thresh);

        for (int i = 0; i < batch_size; ++i) {
            const auto& dets = all_dets[i];
            const int count = static_cast<int>(
                std::min(dets.size(), static_cast<size_t>(max_boxes_per_image)));
            out_counts[i] = count;
            // Detection and InferBox have identical layout — use memcpy
            if (count > 0) {
                std::memcpy(out_boxes_array[i], dets.data(),
                            static_cast<size_t>(count) * sizeof(InferBox));
            }
        }
        return 0;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return -1;
    } catch (...) {
        infergo::set_last_error("infer_torch_detect_gpu_batch: unknown exception");
        return -1;
    }
}

#else // INFER_TORCH_AVAILABLE not defined — stubs

InferTorchSession infer_torch_session_create(const char* /*provider*/, int /*device_id*/) {
    infergo::set_last_error("infer_torch_session_create: libtorch not available in this build");
    return nullptr;
}

InferError infer_torch_session_load(InferTorchSession /*s*/, const char* /*model_path*/) {
    infergo::set_last_error("infer_torch_session_load: libtorch not available in this build");
    return INFER_ERR_RUNTIME;
}

int infer_torch_session_num_inputs(InferTorchSession /*s*/) {
    return 0;
}

int infer_torch_session_num_outputs(InferTorchSession /*s*/) {
    return 0;
}

InferError infer_torch_session_run(
    InferTorchSession /*s*/,
    InferTensor*      /*inputs*/,  int /*n_inputs*/,
    InferTensor*      /*outputs*/, int /*n_outputs*/)
{
    infergo::set_last_error("infer_torch_session_run: libtorch not available in this build");
    return INFER_ERR_RUNTIME;
}

InferError infer_torch_session_run_gpu(
    InferTorchSession /*s*/,
    InferTensor*      /*inputs*/,  int /*n_inputs*/,
    InferTensor*      /*outputs*/, int /*n_outputs*/)
{
    infergo::set_last_error("infer_torch_session_run_gpu: libtorch not available in this build");
    return INFER_ERR_RUNTIME;
}

void infer_torch_session_destroy(InferTorchSession /*s*/) {
    // no-op
}

int infer_torch_detect_gpu(
    void*          /*handle*/,
    const void*    /*jpeg_data*/,
    int            /*nbytes*/,
    float          /*conf_thresh*/,
    float          /*iou_thresh*/,
    InferBox*      /*out_boxes*/,
    int            /*max_boxes*/)
{
    infergo::set_last_error("infer_torch_detect_gpu: libtorch not available in this build");
    return -1;
}

int infer_torch_detect_gpu_batch(
    void*          /*handle*/,
    const void**   /*jpeg_data_array*/,
    const int*     /*nbytes_array*/,
    int            /*batch_size*/,
    float          /*conf_thresh*/,
    float          /*iou_thresh*/,
    InferBox**     /*out_boxes_array*/,
    int*           /*out_counts*/,
    int            /*max_boxes_per_image*/)
{
    infergo::set_last_error("infer_torch_detect_gpu_batch: libtorch not available in this build");
    return -1;
}

int infer_torch_detect_gpu_raw(
    void* /*handle*/, const void* /*rgb*/, int /*w*/, int /*h*/,
    float /*conf*/, float /*iou*/, InferBox* /*boxes*/, int /*max*/)
{
    infergo::set_last_error("infer_torch_detect_gpu_raw: libtorch not available");
    return -1;
}

int infer_torch_detect_gpu_yuv(
    void* /*handle*/, const void* /*yuv*/, int /*w*/, int /*h*/, int /*ls*/,
    float /*conf*/, float /*iou*/, InferBox* /*boxes*/, int /*max*/)
{
    infergo::set_last_error("infer_torch_detect_gpu_yuv: libtorch not available");
    return -1;
}

#endif // INFER_TORCH_AVAILABLE

// ─── Tokenizer API ────────────────────────────────────────────────────────────

InferTokenizer infer_tokenizer_load(const char* path) {
    try {
        if (path == nullptr) {
            infergo::set_last_error("infer_tokenizer_load: null path");
            return nullptr;
        }
        return static_cast<InferTokenizer>(
            new infergo::TokenizerWrapper(path)
        );
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return nullptr;
    } catch (...) {
        infergo::set_last_error("infer_tokenizer_load: unknown exception");
        return nullptr;
    }
}

int infer_tokenizer_encode(
    InferTokenizer  tok,
    const char*     text,
    int             add_special_tokens,
    int*            out_ids,
    int*            out_mask,
    int             max_tokens)
{
    try {
        if (tok == nullptr || text == nullptr || out_ids == nullptr || out_mask == nullptr) {
            infergo::set_last_error("infer_tokenizer_encode: null argument");
            return -1;
        }
        auto& t = *static_cast<infergo::TokenizerWrapper*>(tok);
        const infergo::Encoding enc = t.encode(
            text, add_special_tokens != 0, max_tokens
        );
        const int n = static_cast<int>(enc.ids.size());
        std::memcpy(out_ids,  enc.ids.data(),            static_cast<size_t>(n) * sizeof(int));
        std::memcpy(out_mask, enc.attention_mask.data(), static_cast<size_t>(n) * sizeof(int));
        return n;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return -1;
    } catch (...) {
        infergo::set_last_error("infer_tokenizer_encode: unknown exception");
        return -1;
    }
}

int infer_tokenizer_decode(
    InferTokenizer  tok,
    const int*      ids,
    int             n_ids,
    int             skip_special_tokens,
    char*           out_buf,
    int             buf_size)
{
    try {
        if (tok == nullptr || out_buf == nullptr || buf_size <= 0) {
            infergo::set_last_error("infer_tokenizer_decode: null argument");
            return -1;
        }
        std::vector<int32_t> id_vec;
        if (n_ids > 0 && ids != nullptr) {
            id_vec.assign(ids, ids + n_ids);
        }
        auto& t = *static_cast<infergo::TokenizerWrapper*>(tok);
        const std::string text = t.decode(id_vec, skip_special_tokens != 0);
        std::strncpy(out_buf, text.c_str(), static_cast<size_t>(buf_size) - 1);
        out_buf[buf_size - 1] = '\0';
        return 0;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return -1;
    } catch (...) {
        infergo::set_last_error("infer_tokenizer_decode: unknown exception");
        return -1;
    }
}

int infer_tokenizer_decode_token(
    InferTokenizer  tok,
    int             id,
    char*           out_buf,
    int             buf_size)
{
    try {
        if (tok == nullptr || out_buf == nullptr || buf_size <= 0) {
            infergo::set_last_error("infer_tokenizer_decode_token: null argument");
            return -1;
        }
        auto& t = *static_cast<infergo::TokenizerWrapper*>(tok);
        const std::string piece = t.decode_token(static_cast<int32_t>(id));
        std::strncpy(out_buf, piece.c_str(), static_cast<size_t>(buf_size) - 1);
        out_buf[buf_size - 1] = '\0';
        return 0;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return -1;
    } catch (...) {
        infergo::set_last_error("infer_tokenizer_decode_token: unknown exception");
        return -1;
    }
}

int infer_tokenizer_vocab_size(InferTokenizer tok) {
    if (tok == nullptr) return 0;
    try {
        return static_cast<infergo::TokenizerWrapper*>(tok)->vocab_size();
    } catch (...) {
        return 0;
    }
}

void infer_tokenizer_destroy(InferTokenizer tok) {
    if (tok == nullptr) return;
    try {
        delete static_cast<infergo::TokenizerWrapper*>(tok);
    } catch (...) {
        // destructor must not throw
    }
}

// ─── LLM Engine API ───────────────────────────────────────────────────────────

// Internal handle: owns the engine and the paged KV allocator together.
struct LLMHandle {
    infergo::LLMEngine       engine;
    infergo::KVPageAllocator pages;  // replaces KVCacheSlotManager
    int n_ctx = 0;

    LLMHandle(int n_seq_max, int ctx_size)
        : pages(infergo::KVPageAllocator::kDefaultPageSize,
                n_seq_max,
                ctx_size / infergo::KVPageAllocator::kDefaultPageSize)
        , n_ctx(ctx_size)
    {}
};

// Internal handle: owns one InferSequence + its last decoded logits.
struct SeqHandle {
    infergo::InferSequence  seq;
    std::vector<float>      logits;  // updated by infer_llm_batch_decode
    llama_context*          ctx;     // needed to clear KV cache on destroy

    SeqHandle(infergo::KVPageAllocator& alloc,
              std::vector<int32_t> tokens,
              int32_t eos,
              llama_context* ctx_)
        : seq(alloc, std::move(tokens), eos), ctx(ctx_) {}

    ~SeqHandle() {
        // Remove this sequence's KV cache entries so the slot can be reused
        // by future requests without stale positional data.
        if (ctx != nullptr) {
            llama_memory_seq_rm(llama_get_memory(ctx),
                                static_cast<llama_seq_id>(seq.SlotID()),
                                -1, -1);
        }
    }
};

InferLLM infer_llm_create(const char* path,
                           int         n_gpu_layers,
                           int         ctx_size,
                           int         n_seq_max,
                           int         n_batch)
{
    try {
        if (path == nullptr) {
            infergo::set_last_error("infer_llm_create: null path");
            return nullptr;
        }
        if (n_seq_max <= 0) {
            infergo::set_last_error("infer_llm_create: n_seq_max must be > 0");
            return nullptr;
        }
        auto* h = new LLMHandle(n_seq_max, ctx_size);
        h->engine.LoadModel(path, n_gpu_layers, ctx_size, n_seq_max, n_batch);
        return static_cast<InferLLM>(h);
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return nullptr;
    } catch (...) {
        infergo::set_last_error("infer_llm_create: unknown exception");
        return nullptr;
    }
}

InferLLM infer_llm_create_split(const char* path,
                                 int         n_gpu_layers,
                                 int         ctx_size,
                                 int         n_seq_max,
                                 int         n_batch,
                                 const float* tensor_split,
                                 int          n_split)
{
    try {
        if (path == nullptr) {
            infergo::set_last_error("infer_llm_create_split: null path");
            return nullptr;
        }
        if (n_seq_max <= 0) {
            infergo::set_last_error("infer_llm_create_split: n_seq_max must be > 0");
            return nullptr;
        }
        auto* h = new LLMHandle(n_seq_max, ctx_size);
        h->engine.LoadModelSplit(path, n_gpu_layers, ctx_size, n_seq_max, n_batch,
                                 tensor_split, n_split);
        return static_cast<InferLLM>(h);
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return nullptr;
    } catch (...) {
        infergo::set_last_error("infer_llm_create_split: unknown exception");
        return nullptr;
    }
}

InferLLM infer_llm_create_pipeline(const char* path,
                                    int         n_gpu_layers,
                                    int         ctx_size,
                                    int         n_seq_max,
                                    int         n_batch,
                                    int         n_stages)
{
    try {
        if (path == nullptr) {
            infergo::set_last_error("infer_llm_create_pipeline: null path");
            return nullptr;
        }
        if (n_seq_max <= 0) {
            infergo::set_last_error("infer_llm_create_pipeline: n_seq_max must be > 0");
            return nullptr;
        }
        if (n_stages < 1) {
            infergo::set_last_error("infer_llm_create_pipeline: n_stages must be >= 1");
            return nullptr;
        }
        auto* h = new LLMHandle(n_seq_max, ctx_size);
        h->engine.LoadModelPipeline(path, n_gpu_layers, ctx_size, n_seq_max, n_batch, n_stages);
        return static_cast<InferLLM>(h);
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return nullptr;
    } catch (...) {
        infergo::set_last_error("infer_llm_create_pipeline: unknown exception");
        return nullptr;
    }
}

void infer_llm_destroy(InferLLM llm) {
    if (llm == nullptr) return;
    try { delete static_cast<LLMHandle*>(llm); } catch (...) {}
}

int infer_llm_vocab_size(InferLLM llm) {
    if (llm == nullptr) return 0;
    return static_cast<LLMHandle*>(llm)->engine.VocabSize();
}

int infer_llm_bos(InferLLM llm) {
    if (llm == nullptr) return -1;
    return static_cast<int>(static_cast<LLMHandle*>(llm)->engine.BOS());
}

int infer_llm_eos(InferLLM llm) {
    if (llm == nullptr) return -1;
    return static_cast<int>(static_cast<LLMHandle*>(llm)->engine.EOS());
}

int infer_llm_is_eog(InferLLM llm, int token) {
    if (llm == nullptr) return 0;
    return static_cast<LLMHandle*>(llm)->engine.IsEOG(static_cast<int32_t>(token)) ? 1 : 0;
}

int infer_llm_kv_pages_free(InferLLM llm) {
    if (llm == nullptr) return 0;
    return static_cast<LLMHandle*>(llm)->pages.FreePages();
}

int infer_llm_kv_pages_total(InferLLM llm) {
    if (llm == nullptr) return 0;
    return static_cast<LLMHandle*>(llm)->pages.TotalPages();
}

int infer_llm_kv_page_size(InferLLM llm) {
    if (llm == nullptr) return 0;
    return static_cast<LLMHandle*>(llm)->pages.PageSize();
}

int infer_llm_tokenize(InferLLM llm, const char* text, int add_bos,
                        int* out_ids, int max_tokens) {
    try {
        if (llm == nullptr || text == nullptr || out_ids == nullptr || max_tokens <= 0) {
            infergo::set_last_error("infer_llm_tokenize: invalid argument");
            return -1;
        }
        auto* h = static_cast<LLMHandle*>(llm);
        const auto tokens = h->engine.Tokenize(text, add_bos != 0);
        const int n = std::min(static_cast<int>(tokens.size()), max_tokens);
        for (int i = 0; i < n; ++i) out_ids[i] = static_cast<int>(tokens[i]);
        return n;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return -1;
    } catch (...) {
        return -1;
    }
}

int infer_llm_token_to_piece(InferLLM llm, int token, char* out_buf, int buf_size) {
    try {
        if (llm == nullptr || out_buf == nullptr || buf_size <= 0) {
            infergo::set_last_error("infer_llm_token_to_piece: invalid argument");
            return -1;
        }
        auto* h = static_cast<LLMHandle*>(llm);
        const std::string piece = h->engine.TokenToPiece(static_cast<int32_t>(token));
        const int n = std::min(static_cast<int>(piece.size()), buf_size - 1);
        std::memcpy(out_buf, piece.data(), static_cast<size_t>(n));
        out_buf[n] = '\0';
        return n;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return -1;
    } catch (...) {
        return -1;
    }
}

InferSeq infer_seq_create(InferLLM llm, const int* tokens, int n_tokens) {
    try {
        if (llm == nullptr || tokens == nullptr || n_tokens <= 0) {
            infergo::set_last_error("infer_seq_create: invalid argument");
            return nullptr;
        }
        auto* h = static_cast<LLMHandle*>(llm);
        std::vector<int32_t> tok_vec(tokens, tokens + n_tokens);
        auto* s = new SeqHandle(h->pages, std::move(tok_vec),
                                static_cast<int32_t>(h->engine.EOS()),
                                h->engine.Context());
        return static_cast<InferSeq>(s);
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return nullptr;
    } catch (...) {
        infergo::set_last_error("infer_seq_create: unknown exception");
        return nullptr;
    }
}

void infer_seq_destroy(InferSeq seq) {
    if (seq == nullptr) return;
    try { delete static_cast<SeqHandle*>(seq); } catch (...) {}
}

int infer_seq_is_done(InferSeq seq) {
    if (seq == nullptr) return 1;
    return static_cast<SeqHandle*>(seq)->seq.IsDone() ? 1 : 0;
}

int infer_seq_position(InferSeq seq) {
    if (seq == nullptr) return 0;
    return static_cast<SeqHandle*>(seq)->seq.Position();
}

int infer_seq_slot_id(InferSeq seq) {
    if (seq == nullptr) return -1;
    return static_cast<SeqHandle*>(seq)->seq.SlotID();
}

void infer_seq_append_token(InferSeq seq, int token) {
    if (seq == nullptr) return;
    static_cast<SeqHandle*>(seq)->seq.AppendToken(static_cast<int32_t>(token));
}

int infer_seq_next_tokens(InferSeq seq, int* out_ids, int max_tokens) {
    try {
        if (seq == nullptr || out_ids == nullptr || max_tokens <= 0) {
            infergo::set_last_error("infer_seq_next_tokens: invalid argument");
            return -1;
        }
        const auto tokens = static_cast<SeqHandle*>(seq)->seq.NextTokens();
        const int n = std::min(static_cast<int>(tokens.size()), max_tokens);
        for (int i = 0; i < n; ++i) out_ids[i] = static_cast<int>(tokens[i]);
        return n;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return -1;
    } catch (...) {
        return -1;
    }
}

InferError infer_llm_batch_decode(InferLLM llm, InferSeq* seqs, int n_seqs) {
    try {
        if (llm == nullptr) {
            infergo::set_last_error("infer_llm_batch_decode: null llm");
            return INFER_ERR_NULL;
        }
        if (seqs == nullptr || n_seqs <= 0) {
            infergo::set_last_error("infer_llm_batch_decode: empty sequence list");
            return INFER_ERR_INVALID;
        }
        auto* h = static_cast<LLMHandle*>(llm);

        // Build SequenceInputs
        std::vector<infergo::SequenceInput> inputs;
        inputs.reserve(static_cast<size_t>(n_seqs));
        for (int i = 0; i < n_seqs; ++i) {
            if (seqs[i] == nullptr) continue;
            auto* sh = static_cast<SeqHandle*>(seqs[i]);
            if (sh->seq.IsDone()) continue;

            infergo::SequenceInput inp;
            inp.seq_id      = sh->seq.SlotID();
            inp.tokens      = sh->seq.NextTokens();
            inp.pos         = sh->seq.Position();
            inp.want_logits = true;
            inputs.push_back(std::move(inp));
        }
        if (inputs.empty()) return INFER_OK;

        // Run batch decode
        auto results = h->engine.BatchDecode(inputs);

        // Write logits back into each SeqHandle by matching seq_id
        for (const auto& r : results) {
            for (int i = 0; i < n_seqs; ++i) {
                if (seqs[i] == nullptr) continue;
                auto* sh = static_cast<SeqHandle*>(seqs[i]);
                if (sh->seq.SlotID() == r.seq_id) {
                    sh->logits = r.logits;
                    break;
                }
            }
        }
        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_RUNTIME;
    } catch (...) {
        return INFER_ERR_UNKNOWN;
    }
}

InferError infer_seq_get_logits(InferSeq seq, float* out_logits, int vocab_size) {
    try {
        if (seq == nullptr || out_logits == nullptr || vocab_size <= 0) {
            infergo::set_last_error("infer_seq_get_logits: invalid argument");
            return INFER_ERR_NULL;
        }
        auto* sh = static_cast<SeqHandle*>(seq);
        if (sh->logits.empty()) {
            infergo::set_last_error("infer_seq_get_logits: no logits available (call batch_decode first)");
            return INFER_ERR_INVALID;
        }
        const int n = std::min(vocab_size, static_cast<int>(sh->logits.size()));
        std::memcpy(out_logits, sh->logits.data(), static_cast<size_t>(n) * sizeof(float));
        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_RUNTIME;
    } catch (...) {
        return INFER_ERR_UNKNOWN;
    }
}

// ─── KV cache serialization API ───────────────────────────────────────────────

int infer_llm_kv_serialize(InferLLM llm, int seq_id,
                            uint8_t* out_buf, int out_buf_size) {
    try {
        if (llm == nullptr) {
            infergo::set_last_error("infer_llm_kv_serialize: null llm");
            return -1;
        }
        auto* h = static_cast<LLMHandle*>(llm);

        if (out_buf == nullptr) {
            // Size query: call SerializeKV and return size.
            const auto buf = h->engine.SerializeKV(seq_id);
            return static_cast<int>(buf.size());
        }

        if (out_buf_size <= 0) {
            infergo::set_last_error("infer_llm_kv_serialize: out_buf_size must be > 0");
            return -1;
        }

        const auto buf = h->engine.SerializeKV(seq_id);
        if (buf.empty()) {
            infergo::set_last_error("infer_llm_kv_serialize: SerializeKV returned empty");
            return -1;
        }
        const int n = std::min(static_cast<int>(buf.size()), out_buf_size);
        std::memcpy(out_buf, buf.data(), static_cast<size_t>(n));
        return n;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return -1;
    } catch (...) {
        infergo::set_last_error("infer_llm_kv_serialize: unknown exception");
        return -1;
    }
}

int infer_llm_kv_deserialize(InferLLM llm, int seq_id,
                              const uint8_t* data, int nbytes) {
    try {
        if (llm == nullptr) {
            infergo::set_last_error("infer_llm_kv_deserialize: null llm");
            return -1;
        }
        if (data == nullptr || nbytes <= 0) {
            infergo::set_last_error("infer_llm_kv_deserialize: invalid data or nbytes");
            return -1;
        }
        auto* h = static_cast<LLMHandle*>(llm);
        const bool ok = h->engine.DeserializeKV(seq_id, data, static_cast<size_t>(nbytes));
        return ok ? 0 : -1;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return -1;
    } catch (...) {
        infergo::set_last_error("infer_llm_kv_deserialize: unknown exception");
        return -1;
    }
}

// ─── Preprocessing API ────────────────────────────────────────────────────────

#ifdef INFER_PREPROCESS_AVAILABLE

InferTensor infer_preprocess_decode_image(const void* data, int nbytes) {
    try {
        if (data == nullptr || nbytes <= 0) {
            infergo::set_last_error("infer_preprocess_decode_image: null or empty input");
            return nullptr;
        }
        return static_cast<InferTensor>(
            infergo::decode_image(static_cast<const uint8_t*>(data), nbytes)
        );
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return nullptr;
    } catch (...) {
        infergo::set_last_error("infer_preprocess_decode_image: unknown exception");
        return nullptr;
    }
}

InferTensor infer_preprocess_letterbox(InferTensor src, int target_w, int target_h) {
    try {
        if (src == nullptr) {
            infergo::set_last_error("infer_preprocess_letterbox: null tensor");
            return nullptr;
        }
        if (target_w <= 0 || target_h <= 0) {
            infergo::set_last_error("infer_preprocess_letterbox: target dimensions must be positive");
            return nullptr;
        }
        return static_cast<InferTensor>(
            infergo::letterbox(static_cast<infergo::Tensor*>(src), target_w, target_h)
        );
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return nullptr;
    } catch (...) {
        infergo::set_last_error("infer_preprocess_letterbox: unknown exception");
        return nullptr;
    }
}

InferTensor infer_preprocess_normalize(InferTensor src, float scale,
                                        const float* mean, const float* std) {
    try {
        if (src == nullptr) {
            infergo::set_last_error("infer_preprocess_normalize: null tensor");
            return nullptr;
        }
        if (mean == nullptr || std == nullptr) {
            infergo::set_last_error("infer_preprocess_normalize: null mean or std");
            return nullptr;
        }
        if (scale <= 0.0f) {
            infergo::set_last_error("infer_preprocess_normalize: scale must be positive");
            return nullptr;
        }
        return static_cast<InferTensor>(
            infergo::normalize(static_cast<infergo::Tensor*>(src), scale, mean, std)
        );
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return nullptr;
    } catch (...) {
        infergo::set_last_error("infer_preprocess_normalize: unknown exception");
        return nullptr;
    }
}

InferTensor infer_preprocess_stack_batch(const InferTensor* tensors, int n) {
    try {
        if (tensors == nullptr || n <= 0) {
            infergo::set_last_error("infer_preprocess_stack_batch: null or empty tensor array");
            return nullptr;
        }
        for (int i = 0; i < n; ++i) {
            if (tensors[i] == nullptr) {
                infergo::set_last_error("infer_preprocess_stack_batch: null tensor in array");
                return nullptr;
            }
        }
        std::vector<const infergo::Tensor*> v;
        v.reserve(n);
        for (int i = 0; i < n; ++i)
            v.push_back(static_cast<const infergo::Tensor*>(tensors[i]));
        return static_cast<InferTensor>(infergo::stack_batch(v));
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return nullptr;
    } catch (...) {
        infergo::set_last_error("infer_preprocess_stack_batch: unknown exception");
        return nullptr;
    }
}

#else

InferTensor infer_preprocess_decode_image(const void* /*data*/, int /*nbytes*/) {
    infergo::set_last_error("infer_preprocess_decode_image: OpenCV not available in this build");
    return nullptr;
}

InferTensor infer_preprocess_letterbox(InferTensor /*src*/, int /*target_w*/, int /*target_h*/) {
    infergo::set_last_error("infer_preprocess_letterbox: OpenCV not available in this build");
    return nullptr;
}

InferTensor infer_preprocess_normalize(InferTensor /*src*/, float /*scale*/,
                                        const float* /*mean*/, const float* /*std*/) {
    infergo::set_last_error("infer_preprocess_normalize: OpenCV not available in this build");
    return nullptr;
}

InferTensor infer_preprocess_stack_batch(const InferTensor* /*tensors*/, int /*n*/) {
    infergo::set_last_error("infer_preprocess_stack_batch: OpenCV not available in this build");
    return nullptr;
}

#endif // INFER_PREPROCESS_AVAILABLE

// ─── Postprocessing API ───────────────────────────────────────────────────────

int infer_postprocess_classify(InferTensor logits, int top_k,
                               InferClassResult* out_results) {
    try {
        if (logits == nullptr) {
            infergo::set_last_error("infer_postprocess_classify: null logits tensor");
            return -1;
        }
        if (top_k <= 0) {
            infergo::set_last_error("infer_postprocess_classify: top_k must be positive");
            return -1;
        }
        if (out_results == nullptr) {
            infergo::set_last_error("infer_postprocess_classify: null out_results");
            return -1;
        }
        auto results = infergo::classify(
            static_cast<infergo::Tensor*>(logits), top_k
        );
        for (int i = 0; i < static_cast<int>(results.size()); ++i) {
            out_results[i].label_idx  = results[i].label_idx;
            out_results[i].confidence = results[i].confidence;
        }
        return static_cast<int>(results.size());
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return -1;
    } catch (...) {
        infergo::set_last_error("infer_postprocess_classify: unknown exception");
        return -1;
    }
}

int infer_postprocess_nms(InferTensor predictions,
                          float conf_thresh, float iou_thresh,
                          InferBox* out_boxes, int max_boxes) {
    try {
        if (predictions == nullptr) {
            infergo::set_last_error("infer_postprocess_nms: null predictions tensor");
            return -1;
        }
        if (out_boxes == nullptr || max_boxes <= 0) {
            infergo::set_last_error("infer_postprocess_nms: null or zero out_boxes/max_boxes");
            return -1;
        }
        auto boxes = infergo::nms(
            static_cast<infergo::Tensor*>(predictions),
            conf_thresh, iou_thresh
        );
        const int n = std::min(static_cast<int>(boxes.size()), max_boxes);
        for (int i = 0; i < n; ++i) {
            out_boxes[i].x1         = boxes[i].x1;
            out_boxes[i].y1         = boxes[i].y1;
            out_boxes[i].x2         = boxes[i].x2;
            out_boxes[i].y2         = boxes[i].y2;
            out_boxes[i].class_idx  = boxes[i].class_idx;
            out_boxes[i].confidence = boxes[i].confidence;
        }
        return n;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return -1;
    } catch (...) {
        infergo::set_last_error("infer_postprocess_nms: unknown exception");
        return -1;
    }
}

InferError infer_postprocess_normalize_embedding(InferTensor t) {
    try {
        if (t == nullptr) {
            infergo::set_last_error("infer_postprocess_normalize_embedding: null tensor");
            return INFER_ERR_NULL;
        }
        infergo::normalize_embedding(static_cast<infergo::Tensor*>(t));
        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_INVALID;
    } catch (...) {
        infergo::set_last_error("infer_postprocess_normalize_embedding: unknown exception");
        return INFER_ERR_UNKNOWN;
    }
}

// ─── VRAM Monitoring ─────────────────────────────────────────────────────────

size_t infer_cuda_vram_free(void) {
#ifdef INFER_CUDA_AVAILABLE
    return infergo::cuda_vram_free();
#else
    return 0;
#endif
}

size_t infer_cuda_vram_total(void) {
#ifdef INFER_CUDA_AVAILABLE
    return infergo::cuda_vram_total();
#else
    return 0;
#endif
}

int infer_cuda_vram_used_pct(void) {
#ifdef INFER_CUDA_AVAILABLE
    return infergo::cuda_vram_used_pct();
#else
    return 0;
#endif
}

// ─── Video Decoder / Encoder ────────────────────────────────────────────────

void* infer_video_decoder_open(const char* url, int hw_accel) {
#ifdef INFER_VIDEO_AVAILABLE
    if (!url) {
        infergo::set_last_error("infer_video_decoder_open: null url");
        return nullptr;
    }
    try {
        auto* dec = new infergo::VideoDecoder(url, hw_accel != 0);
        if (!dec->is_open()) {
            infergo::set_last_error("infer_video_decoder_open: failed to open video");
            delete dec;
            return nullptr;
        }
        return static_cast<void*>(dec);
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return nullptr;
    } catch (...) {
        infergo::set_last_error("infer_video_decoder_open: unknown exception");
        return nullptr;
    }
#else
    (void)url; (void)hw_accel;
    infergo::set_last_error("infer_video_decoder_open: video not available in this build");
    return nullptr;
#endif
}

int infer_video_decoder_next_frame(void* dec, uint8_t** out_rgb,
                                   int* w, int* h, int64_t* pts, int* frame_num) {
#ifdef INFER_VIDEO_AVAILABLE
    if (!dec) return 0;
    try {
        auto* d = static_cast<infergo::VideoDecoder*>(dec);
        infergo::FrameInfo info{};
        uint8_t* data = nullptr;
        if (!d->next_frame(&data, &info)) return 0;
        if (out_rgb)   *out_rgb   = data;
        if (w)         *w         = info.width;
        if (h)         *h         = info.height;
        if (pts)       *pts       = info.pts;
        if (frame_num) *frame_num = info.frame_number;
        return 1;
    } catch (...) {
        return 0;
    }
#else
    (void)dec; (void)out_rgb; (void)w; (void)h; (void)pts; (void)frame_num;
    return 0;
#endif
}

int infer_video_decoder_width(void* dec) {
#ifdef INFER_VIDEO_AVAILABLE
    if (!dec) return 0;
    return static_cast<infergo::VideoDecoder*>(dec)->width();
#else
    (void)dec; return 0;
#endif
}

int infer_video_decoder_height(void* dec) {
#ifdef INFER_VIDEO_AVAILABLE
    if (!dec) return 0;
    return static_cast<infergo::VideoDecoder*>(dec)->height();
#else
    (void)dec; return 0;
#endif
}

double infer_video_decoder_fps(void* dec) {
#ifdef INFER_VIDEO_AVAILABLE
    if (!dec) return 0.0;
    return static_cast<infergo::VideoDecoder*>(dec)->fps();
#else
    (void)dec; return 0.0;
#endif
}

int infer_video_decoder_is_hw(void* dec) {
#ifdef INFER_VIDEO_AVAILABLE
    if (!dec) return 0;
    return static_cast<infergo::VideoDecoder*>(dec)->is_hw_accelerated() ? 1 : 0;
#else
    (void)dec; return 0;
#endif
}

int infer_video_decoder_next_frame_resized(void* dec, int target_w, int target_h,
                                            uint8_t** out_rgb, int* out_w, int* out_h,
                                            int64_t* out_pts, int* out_frame_num) {
#ifdef INFER_VIDEO_AVAILABLE
    if (!dec) return 0;
    try {
        auto* d = static_cast<infergo::VideoDecoder*>(dec);
        infergo::FrameInfo info{};
        uint8_t* data = nullptr;
        if (!d->next_frame(&data, &info)) return 0;

        // Resize using OpenCV if dimensions differ.
        static thread_local std::vector<uint8_t> resize_buf;
        if (target_w > 0 && target_h > 0 &&
            (target_w != info.width || target_h != info.height)) {
#ifdef INFER_OPENCV_AVAILABLE
            fprintf(stderr, "[resize] %dx%d -> %dx%d\n", info.width, info.height, target_w, target_h);
            cv::Mat src(info.height, info.width, CV_8UC3, data);
            cv::Mat dst(target_h, target_w, CV_8UC3);
            cv::resize(src, dst, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);
            size_t nb = static_cast<size_t>(target_w) * target_h * 3;
            resize_buf.resize(nb);
            memcpy(resize_buf.data(), dst.data, nb);
            if (out_rgb)   *out_rgb = resize_buf.data();
            if (out_w)     *out_w = target_w;
            if (out_h)     *out_h = target_h;
#else
            // No OpenCV — return original size.
            if (out_rgb) *out_rgb = data;
            if (out_w)   *out_w = info.width;
            if (out_h)   *out_h = info.height;
#endif
        } else {
            if (out_rgb) *out_rgb = data;
            if (out_w)   *out_w = info.width;
            if (out_h)   *out_h = info.height;
        }
        if (out_pts)       *out_pts = info.pts;
        if (out_frame_num) *out_frame_num = info.frame_number;
        return 1;
    } catch (...) { return 0; }
#else
    (void)dec;(void)target_w;(void)target_h;(void)out_rgb;(void)out_w;(void)out_h;(void)out_pts;(void)out_frame_num;
    return 0;
#endif
}

int infer_pipeline_detect_frame(
    void* decoder, void* detector,
    float conf_thresh, float iou_thresh,
    const InferLine* lines, int n_lines,
    const InferPolygonOverlay* polygons, int n_polygons,
    const InferTextOverlay* texts, int n_texts,
    const InferFilledRect* rects, int n_rects,
    int jpeg_w, int jpeg_h, int jpeg_quality,
    uint8_t** out_jpeg, int* out_jpeg_size,
    InferBox* out_boxes, int* out_nboxes, int max_boxes)
{
#if defined(INFER_VIDEO_AVAILABLE) && defined(INFER_TORCH_AVAILABLE)
    if (!decoder || !detector || !out_jpeg || !out_jpeg_size) return 0;
    try {
        auto* dec = static_cast<infergo::VideoDecoder*>(decoder);

        // 1. Decode next frame (internal C buffer — zero copy).
        infergo::FrameInfo fi{};
        uint8_t* rgb = nullptr;
        if (!dec->next_frame(&rgb, &fi)) return 0;
        int w = fi.width, h = fi.height;

        // 2. Detect with YOLO (GPU inference on the C buffer directly).
        int nboxes = infer_torch_detect_gpu_raw(
            detector, rgb, w, h, conf_thresh, iou_thresh, out_boxes, max_boxes);
        if (out_nboxes) *out_nboxes = nboxes > 0 ? nboxes : 0;

        // 3. Convert InferBox to InferAnnotateBox for the annotator.
        std::vector<infergo::AnnotateBox> aboxes(nboxes > 0 ? nboxes : 0);
        for (int i = 0; i < nboxes && i < max_boxes; i++) {
            aboxes[i].x1 = out_boxes[i].x1;
            aboxes[i].y1 = out_boxes[i].y1;
            aboxes[i].x2 = out_boxes[i].x2;
            aboxes[i].y2 = out_boxes[i].y2;
            aboxes[i].class_id = out_boxes[i].class_idx;
            aboxes[i].confidence = out_boxes[i].confidence;
            aboxes[i].track_id = i;
            snprintf(aboxes[i].label, 64, "cls%d %.0f%%",
                     out_boxes[i].class_idx, out_boxes[i].confidence * 100);
        }

        // 4. Annotate + resize + JPEG encode in one call.
        auto* cpp_lines = reinterpret_cast<const infergo::FrameAnnotator::Line*>(lines);
        auto* cpp_polys = reinterpret_cast<const infergo::FrameAnnotator::PolygonOverlay*>(polygons);
        auto* cpp_texts = reinterpret_cast<const infergo::FrameAnnotator::TextOverlay*>(texts);
        auto* cpp_rects = reinterpret_cast<const infergo::FrameAnnotator::FilledRect*>(rects);

        int out_w = (jpeg_w > 0) ? jpeg_w : w;
        int out_h = (jpeg_h > 0) ? jpeg_h : h;

        int size = 0;
        uint8_t* jpeg = infergo::FrameAnnotator::annotate_full(
            rgb, w, h,
            aboxes.data(), static_cast<int>(aboxes.size()),
            cpp_lines, n_lines,
            cpp_polys, n_polygons,
            cpp_texts, n_texts,
            cpp_rects, n_rects,
            out_w, out_h, jpeg_quality, &size);

        if (!jpeg) return 0;
        *out_jpeg = jpeg;
        *out_jpeg_size = size;
        return 1;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return 0;
    } catch (...) {
        return 0;
    }
#else
    (void)decoder;(void)detector;(void)conf_thresh;(void)iou_thresh;
    (void)lines;(void)n_lines;(void)polygons;(void)n_polygons;
    (void)texts;(void)n_texts;(void)rects;(void)n_rects;
    (void)jpeg_w;(void)jpeg_h;(void)jpeg_quality;
    (void)out_jpeg;(void)out_jpeg_size;(void)out_boxes;(void)out_nboxes;(void)max_boxes;
    return 0;
#endif
}

void infer_video_decoder_set_output_size(void* dec, int target_w, int target_h) {
#ifdef INFER_VIDEO_AVAILABLE
    if (!dec) return;
    static_cast<infergo::VideoDecoder*>(dec)->set_output_size(target_w, target_h);
#else
    (void)dec; (void)target_w; (void)target_h;
#endif
}

void infer_video_decoder_close(void* dec) {
#ifdef INFER_VIDEO_AVAILABLE
    if (!dec) return;
    auto* d = static_cast<infergo::VideoDecoder*>(dec);
    d->close();
    delete d;
#else
    (void)dec;
#endif
}

void* infer_video_encoder_open(const char* path, int width, int height,
                               int fps, const char* codec) {
#ifdef INFER_VIDEO_AVAILABLE
    if (!path) {
        infergo::set_last_error("infer_video_encoder_open: null path");
        return nullptr;
    }
    try {
        std::string codec_str = codec ? codec : "h264_nvenc";
        auto* enc = new infergo::VideoEncoder(path, width, height, fps, codec_str);
        if (!enc->is_open()) {
            infergo::set_last_error("infer_video_encoder_open: failed to open encoder");
            delete enc;
            return nullptr;
        }
        return static_cast<void*>(enc);
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return nullptr;
    } catch (...) {
        infergo::set_last_error("infer_video_encoder_open: unknown exception");
        return nullptr;
    }
#else
    (void)path; (void)width; (void)height; (void)fps; (void)codec;
    infergo::set_last_error("infer_video_encoder_open: video not available in this build");
    return nullptr;
#endif
}

int infer_video_encoder_write(void* enc, const uint8_t* rgb, int width, int height) {
#ifdef INFER_VIDEO_AVAILABLE
    if (!enc || !rgb) return 0;
    (void)width; (void)height;  // dimensions set at open time
    try {
        return static_cast<infergo::VideoEncoder*>(enc)->write_frame(rgb) ? 1 : 0;
    } catch (...) {
        return 0;
    }
#else
    (void)enc; (void)rgb; (void)width; (void)height;
    return 0;
#endif
}

int infer_video_encoder_is_hw(void* enc) {
#ifdef INFER_VIDEO_AVAILABLE
    if (!enc) return 0;
    return static_cast<infergo::VideoEncoder*>(enc)->is_hw_accelerated() ? 1 : 0;
#else
    (void)enc; return 0;
#endif
}

void infer_video_encoder_close(void* enc) {
#ifdef INFER_VIDEO_AVAILABLE
    if (!enc) return;
    auto* e = static_cast<infergo::VideoEncoder*>(enc);
    e->close();
    delete e;
#else
    (void)enc;
#endif
}

// ─── Frame Annotator ─────────────────────────────────────────────────────────

InferError infer_frame_annotate_jpeg(const uint8_t* rgb, int w, int h,
                                     const InferAnnotateBox* boxes, int n,
                                     int quality,
                                     uint8_t** out_jpeg, int* out_size) {
#ifdef INFER_VIDEO_AVAILABLE
    if (!rgb || !out_jpeg || !out_size) {
        infergo::set_last_error("infer_frame_annotate_jpeg: null pointer");
        return INFER_ERR_NULL;
    }
    try {
        // Convert InferAnnotateBox -> infergo::AnnotateBox (same layout)
        auto* ab = reinterpret_cast<const infergo::AnnotateBox*>(boxes);
        uint8_t* jpeg = infergo::FrameAnnotator::annotate_jpeg(rgb, w, h, ab, n, quality, out_size);
        if (!jpeg) {
            infergo::set_last_error("infer_frame_annotate_jpeg: JPEG encode failed (TurboJPEG unavailable or error)");
            return INFER_ERR_RUNTIME;
        }
        *out_jpeg = jpeg;
        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_RUNTIME;
    } catch (...) {
        infergo::set_last_error("infer_frame_annotate_jpeg: unknown exception");
        return INFER_ERR_UNKNOWN;
    }
#else
    (void)rgb; (void)w; (void)h; (void)boxes; (void)n;
    (void)quality; (void)out_jpeg; (void)out_size;
    infergo::set_last_error("infer_frame_annotate_jpeg: video not available in this build");
    return INFER_ERR_RUNTIME;
#endif
}

InferError infer_frame_resize_jpeg(const uint8_t* rgb, int sw, int sh,
                                   int dw, int dh, int quality,
                                   uint8_t** out_jpeg, int* out_size) {
#ifdef INFER_VIDEO_AVAILABLE
    if (!rgb || !out_jpeg || !out_size) {
        infergo::set_last_error("infer_frame_resize_jpeg: null pointer");
        return INFER_ERR_NULL;
    }
    try {
        uint8_t* jpeg = infergo::FrameAnnotator::resize_jpeg(rgb, sw, sh, dw, dh, quality, out_size);
        if (!jpeg) {
            infergo::set_last_error("infer_frame_resize_jpeg: JPEG encode failed");
            return INFER_ERR_RUNTIME;
        }
        *out_jpeg = jpeg;
        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_RUNTIME;
    } catch (...) {
        infergo::set_last_error("infer_frame_resize_jpeg: unknown exception");
        return INFER_ERR_UNKNOWN;
    }
#else
    (void)rgb; (void)sw; (void)sh; (void)dw; (void)dh;
    (void)quality; (void)out_jpeg; (void)out_size;
    infergo::set_last_error("infer_frame_resize_jpeg: video not available in this build");
    return INFER_ERR_RUNTIME;
#endif
}

InferError infer_frame_combine_jpeg(const uint8_t* rgb1, int w1, int h1,
                                    const uint8_t* rgb2, int w2, int h2,
                                    const char* status,
                                    int tw, int th, int quality,
                                    uint8_t** out_jpeg, int* out_size) {
#ifdef INFER_VIDEO_AVAILABLE
    if (!rgb1 || !rgb2 || !out_jpeg || !out_size) {
        infergo::set_last_error("infer_frame_combine_jpeg: null pointer");
        return INFER_ERR_NULL;
    }
    try {
        uint8_t* jpeg = infergo::FrameAnnotator::combine_jpeg(
            rgb1, w1, h1, rgb2, w2, h2, status, tw, th, quality, out_size);
        if (!jpeg) {
            infergo::set_last_error("infer_frame_combine_jpeg: JPEG encode failed");
            return INFER_ERR_RUNTIME;
        }
        *out_jpeg = jpeg;
        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_RUNTIME;
    } catch (...) {
        infergo::set_last_error("infer_frame_combine_jpeg: unknown exception");
        return INFER_ERR_UNKNOWN;
    }
#else
    (void)rgb1; (void)w1; (void)h1; (void)rgb2; (void)w2; (void)h2;
    (void)status; (void)tw; (void)th; (void)quality; (void)out_jpeg; (void)out_size;
    infergo::set_last_error("infer_frame_combine_jpeg: video not available in this build");
    return INFER_ERR_RUNTIME;
#endif
}

InferError infer_frame_draw_line(uint8_t* rgb, int w, int h,
                                 int x1, int y1, int x2, int y2,
                                 uint8_t r, uint8_t g, uint8_t b, int thickness) {
#ifdef INFER_VIDEO_AVAILABLE
    if (!rgb) {
        infergo::set_last_error("infer_frame_draw_line: null pointer");
        return INFER_ERR_NULL;
    }
    try {
        infergo::FrameAnnotator::draw_line(rgb, w, h, x1, y1, x2, y2, r, g, b, thickness);
        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_RUNTIME;
    } catch (...) {
        infergo::set_last_error("infer_frame_draw_line: unknown exception");
        return INFER_ERR_UNKNOWN;
    }
#else
    (void)rgb; (void)w; (void)h; (void)x1; (void)y1; (void)x2; (void)y2;
    (void)r; (void)g; (void)b; (void)thickness;
    infergo::set_last_error("infer_frame_draw_line: video not available in this build");
    return INFER_ERR_RUNTIME;
#endif
}

InferError infer_frame_draw_polygon(uint8_t* rgb, int w, int h,
                                    const InferPoint* pts, int n,
                                    uint8_t r, uint8_t g, uint8_t b, uint8_t alpha) {
#ifdef INFER_VIDEO_AVAILABLE
    if (!rgb || !pts) {
        infergo::set_last_error("infer_frame_draw_polygon: null pointer");
        return INFER_ERR_NULL;
    }
    try {
        auto* p = reinterpret_cast<const infergo::Point*>(pts);
        infergo::FrameAnnotator::draw_polygon(rgb, w, h, p, n, r, g, b, alpha);
        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_RUNTIME;
    } catch (...) {
        infergo::set_last_error("infer_frame_draw_polygon: unknown exception");
        return INFER_ERR_UNKNOWN;
    }
#else
    (void)rgb; (void)w; (void)h; (void)pts; (void)n;
    (void)r; (void)g; (void)b; (void)alpha;
    infergo::set_last_error("infer_frame_draw_polygon: video not available in this build");
    return INFER_ERR_RUNTIME;
#endif
}

InferError infer_frame_draw_text(uint8_t* rgb, int w, int h,
                                 int x, int y, const char* text,
                                 uint8_t r, uint8_t g, uint8_t b, int scale) {
#ifdef INFER_VIDEO_AVAILABLE
    if (!rgb) {
        infergo::set_last_error("infer_frame_draw_text: null pointer");
        return INFER_ERR_NULL;
    }
    try {
        infergo::FrameAnnotator::draw_text(rgb, w, h, x, y, text, r, g, b, scale);
        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_RUNTIME;
    } catch (...) {
        infergo::set_last_error("infer_frame_draw_text: unknown exception");
        return INFER_ERR_UNKNOWN;
    }
#else
    (void)rgb; (void)w; (void)h; (void)x; (void)y; (void)text;
    (void)r; (void)g; (void)b; (void)scale;
    infergo::set_last_error("infer_frame_draw_text: video not available in this build");
    return INFER_ERR_RUNTIME;
#endif
}

void infer_frame_jpeg_free(uint8_t* buf) {
#ifdef INFER_VIDEO_AVAILABLE
    infergo::FrameAnnotator::jpeg_free(buf);
#else
    (void)buf;
#endif
}

InferError infer_frame_annotate_full(
    const uint8_t* rgb, int w, int h,
    const InferAnnotateBox* boxes, int n_boxes,
    const InferLine* lines, int n_lines,
    const InferPolygonOverlay* polygons, int n_polygons,
    const InferTextOverlay* texts, int n_texts,
    const InferFilledRect* rects, int n_rects,
    int out_w, int out_h, int quality,
    uint8_t** out_jpeg, int* out_size)
{
#ifdef INFER_VIDEO_AVAILABLE
    if (!rgb || !out_jpeg || !out_size) {
        infergo::set_last_error("infer_frame_annotate_full: null pointer");
        return INFER_ERR_NULL;
    }
    try {
        // Convert C structs to C++ structs (binary-compatible layout).
        auto* cpp_boxes = reinterpret_cast<const infergo::AnnotateBox*>(boxes);
        auto* cpp_lines = reinterpret_cast<const infergo::FrameAnnotator::Line*>(lines);
        auto* cpp_polygons = reinterpret_cast<const infergo::FrameAnnotator::PolygonOverlay*>(polygons);
        auto* cpp_texts = reinterpret_cast<const infergo::FrameAnnotator::TextOverlay*>(texts);
        auto* cpp_rects = reinterpret_cast<const infergo::FrameAnnotator::FilledRect*>(rects);

        uint8_t* jpeg = infergo::FrameAnnotator::annotate_full(
            rgb, w, h,
            cpp_boxes, n_boxes,
            cpp_lines, n_lines,
            cpp_polygons, n_polygons,
            cpp_texts, n_texts,
            cpp_rects, n_rects,
            out_w, out_h, quality, out_size);

        if (!jpeg) {
            infergo::set_last_error("infer_frame_annotate_full: encode failed");
            return INFER_ERR_RUNTIME;
        }
        *out_jpeg = jpeg;
        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_RUNTIME;
    } catch (...) {
        infergo::set_last_error("infer_frame_annotate_full: unknown error");
        return INFER_ERR_UNKNOWN;
    }
#else
    (void)rgb;(void)w;(void)h;(void)boxes;(void)n_boxes;(void)lines;(void)n_lines;
    (void)polygons;(void)n_polygons;(void)texts;(void)n_texts;(void)rects;(void)n_rects;
    (void)out_w;(void)out_h;(void)quality;(void)out_jpeg;(void)out_size;
    infergo::set_last_error("infer_frame_annotate_full: video not available");
    return INFER_ERR_RUNTIME;
#endif
}
