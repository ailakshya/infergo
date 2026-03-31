#include "tensor.hpp"

#include <cstdlib>   // malloc, free
#include <cstring>   // memcpy, strncpy

namespace infergo {

// ─── Thread-local error string ───────────────────────────────────────────────

namespace {
    thread_local char g_last_error[512] = {};
}

void set_last_error(const char* msg) noexcept {
    if (msg == nullptr) {
        g_last_error[0] = '\0';
        return;
    }
    std::strncpy(g_last_error, msg, sizeof(g_last_error) - 1);
    g_last_error[sizeof(g_last_error) - 1] = '\0';
}

const char* get_last_error() noexcept {
    return g_last_error;
}

// ─── dtype_size ──────────────────────────────────────────────────────────────

size_t Tensor::dtype_size(int dtype) noexcept {
    switch (dtype) {
        case 0: return 4;  // INFER_DTYPE_FLOAT32
        case 1: return 2;  // INFER_DTYPE_FLOAT16
        case 2: return 2;  // INFER_DTYPE_BFLOAT16
        case 3: return 4;  // INFER_DTYPE_INT32
        case 4: return 8;  // INFER_DTYPE_INT64
        case 5: return 1;  // INFER_DTYPE_UINT8
        case 6: return 1;  // INFER_DTYPE_BOOL
        default: return 0;
    }
}

// ─── nelements / compute_nbytes ──────────────────────────────────────────────

size_t Tensor::nelements() const noexcept {
    if (shape == nullptr || ndim <= 0) {
        return 0;
    }
    size_t n = 1;
    for (int i = 0; i < ndim; ++i) {
        if (shape[i] <= 0) {
            return 0;
        }
        n *= static_cast<size_t>(shape[i]);
    }
    return n;
}

size_t Tensor::compute_nbytes(const int* shape, int ndim, int dtype) noexcept {
    const size_t elem_size = dtype_size(dtype);
    if (shape == nullptr || ndim <= 0 || elem_size == 0) {
        return 0;
    }
    size_t n = 1;
    for (int i = 0; i < ndim; ++i) {
        if (shape[i] <= 0) {
            return 0;
        }
        n *= static_cast<size_t>(shape[i]);
    }
    return n * elem_size;
}

// ─── tensor_alloc_cpu ────────────────────────────────────────────────────────

Tensor* tensor_alloc_cpu(const int* shape, int ndim, int dtype) noexcept {
    if (shape == nullptr) {
        set_last_error("tensor_alloc_cpu: shape is null");
        return nullptr;
    }
    if (ndim <= 0) {
        set_last_error("tensor_alloc_cpu: ndim must be > 0");
        return nullptr;
    }
    for (int i = 0; i < ndim; ++i) {
        if (shape[i] <= 0) {
            set_last_error("tensor_alloc_cpu: all shape dimensions must be > 0");
            return nullptr;
        }
    }

    const size_t nb = Tensor::compute_nbytes(shape, ndim, dtype);
    if (nb == 0) {
        set_last_error("tensor_alloc_cpu: unsupported dtype");
        return nullptr;
    }

    // Allocate the struct itself via malloc — never new (RULE: C ABI compatibility)
    Tensor* t = static_cast<Tensor*>(std::malloc(sizeof(Tensor)));
    if (t == nullptr) {
        set_last_error("tensor_alloc_cpu: OOM allocating Tensor struct");
        return nullptr;
    }
    *t = Tensor{};  // zero-initialise all fields

    // Allocate and copy shape array
    t->shape = static_cast<int*>(
        std::malloc(static_cast<size_t>(ndim) * sizeof(int))
    );
    if (t->shape == nullptr) {
        std::free(t);
        set_last_error("tensor_alloc_cpu: OOM allocating shape array");
        return nullptr;
    }
    std::memcpy(t->shape, shape, static_cast<size_t>(ndim) * sizeof(int));

    // Allocate data buffer
    t->data = std::malloc(nb);
    if (t->data == nullptr) {
        std::free(t->shape);
        std::free(t);
        set_last_error("tensor_alloc_cpu: OOM allocating data buffer");
        return nullptr;
    }

    t->ndim      = ndim;
    t->dtype     = dtype;
    t->on_device = false;
    t->device_id = 0;
    t->nbytes    = nb;

    return t;
}

// ─── tensor_free ─────────────────────────────────────────────────────────────

void tensor_free(Tensor* t) noexcept {
    if (t == nullptr) {
        return;
    }

    // Save → zero → free for every pointer field (RULE 5).
    // Zeroing before freeing means a double-free of `t` produces a harmless
    // free(nullptr) rather than a heap corruption.

    void* data_ptr  = t->data;
    int*  shape_ptr = t->shape;

    t->data  = nullptr;
    t->shape = nullptr;

    if (t->on_device) {
#ifdef INFER_CUDA_AVAILABLE
        tensor_cuda_free_data(data_ptr);  // RULE 7: CUDA memory freed by CUDA code
#endif
    } else {
        std::free(data_ptr);
    }

    std::free(shape_ptr);
    std::free(t);
}

// ─── tensor_copy_from ────────────────────────────────────────────────────────

bool tensor_copy_from(Tensor* t, const void* src, int nbytes) noexcept {
    if (t == nullptr) {
        set_last_error("tensor_copy_from: null tensor");
        return false;
    }
    if (src == nullptr) {
        set_last_error("tensor_copy_from: null src");
        return false;
    }
    if (nbytes <= 0) {
        set_last_error("tensor_copy_from: nbytes must be > 0");
        return false;
    }
    if (t->on_device) {
        set_last_error("tensor_copy_from: tensor is on device — use tensor_to_host first");
        return false;
    }
    if (static_cast<size_t>(nbytes) != t->nbytes) {
        set_last_error("tensor_copy_from: nbytes does not match tensor size");
        return false;
    }

    std::memcpy(t->data, src, t->nbytes);
    return true;
}

// ─── Getters (T-08) ──────────────────────────────────────────────────────────

int tensor_get_shape(const Tensor* t, int* out_shape, int max_dims) noexcept {
    if (t == nullptr || out_shape == nullptr || max_dims <= 0) {
        return 0;
    }
    const int dims = (t->ndim < max_dims) ? t->ndim : max_dims;
    for (int i = 0; i < dims; ++i) {
        out_shape[i] = t->shape[i];
    }
    return t->ndim;
}

int tensor_get_dtype(const Tensor* t) noexcept {
    if (t == nullptr) { return -1; }
    return t->dtype;
}

int tensor_get_nbytes(const Tensor* t) noexcept {
    if (t == nullptr) { return 0; }
    return static_cast<int>(t->nbytes);
}

int tensor_get_nelements(const Tensor* t) noexcept {
    if (t == nullptr) { return 0; }
    return static_cast<int>(t->nelements());
}

} // namespace infergo
