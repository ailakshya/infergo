#include "tensor.hpp"

namespace infergo {

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

} // namespace infergo
