#include "infer_api.h"
#include "../tensor/tensor.hpp"

#include <exception>

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
