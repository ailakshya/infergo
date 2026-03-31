// tensor_cuda.cu — CUDA allocation and device-memory helpers for libinfer_tensor.
// T-04: tensor_alloc_cuda, tensor_cuda_free_data
// T-06: tensor_to_device, tensor_to_host  (added there)

#include "tensor.hpp"

#include <cuda_runtime.h>
#include <cstdlib>   // malloc, free
#include <cstring>   // memcpy

namespace infergo {

// ─── tensor_cuda_free_data ───────────────────────────────────────────────────

void tensor_cuda_free_data(void* ptr) noexcept {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

// ─── tensor_alloc_cuda ───────────────────────────────────────────────────────

Tensor* tensor_alloc_cuda(const int* shape, int ndim, int dtype, int device_id) noexcept {
    if (shape == nullptr) {
        set_last_error("tensor_alloc_cuda: shape is null");
        return nullptr;
    }
    if (ndim <= 0) {
        set_last_error("tensor_alloc_cuda: ndim must be > 0");
        return nullptr;
    }
    for (int i = 0; i < ndim; ++i) {
        if (shape[i] <= 0) {
            set_last_error("tensor_alloc_cuda: all shape dimensions must be > 0");
            return nullptr;
        }
    }

    const size_t nb = Tensor::compute_nbytes(shape, ndim, dtype);
    if (nb == 0) {
        set_last_error("tensor_alloc_cuda: unsupported dtype");
        return nullptr;
    }

    // Select the target device before allocating
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        set_last_error(cudaGetErrorString(err));
        return nullptr;
    }

    // Tensor struct lives on CPU — malloc, never new (C ABI compatibility)
    Tensor* t = static_cast<Tensor*>(std::malloc(sizeof(Tensor)));
    if (t == nullptr) {
        set_last_error("tensor_alloc_cuda: OOM allocating Tensor struct");
        return nullptr;
    }
    *t = Tensor{};

    // Shape array lives on CPU
    t->shape = static_cast<int*>(
        std::malloc(static_cast<size_t>(ndim) * sizeof(int))
    );
    if (t->shape == nullptr) {
        std::free(t);
        set_last_error("tensor_alloc_cuda: OOM allocating shape array");
        return nullptr;
    }
    std::memcpy(t->shape, shape, static_cast<size_t>(ndim) * sizeof(int));

    // Data buffer lives on the GPU — RULE 7: allocated by CUDA, freed by CUDA
    err = cudaMalloc(&t->data, nb);
    if (err != cudaSuccess) {
        std::free(t->shape);
        std::free(t);
        set_last_error(cudaGetErrorString(err));
        return nullptr;
    }

    t->ndim      = ndim;
    t->dtype     = dtype;
    t->on_device = true;
    t->device_id = device_id;
    t->nbytes    = nb;

    return t;
}

} // namespace infergo
