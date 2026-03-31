#include "infer_api.h"
#include "../tensor/tensor.hpp"
#include "../onnx/onnx_session.hpp"
#include "../tokenizer/tokenizer.hpp"

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
