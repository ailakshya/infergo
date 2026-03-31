#pragma once

#include <cstddef>

namespace infergo {

/// Internal C++ representation of a tensor (CPU or CUDA).
/// Never exposed across the C API — Go only ever sees void* (InferTensor).
struct Tensor {
    void*   data      = nullptr;  ///< Raw data pointer (CPU heap or CUDA device)
    int*    shape     = nullptr;  ///< Heap-allocated int array, length == ndim
    int     ndim      = 0;        ///< Number of dimensions
    int     dtype     = 0;        ///< Data type (INFER_DTYPE_* constants)
    bool    on_device = false;    ///< true when data lives on a CUDA device
    int     device_id = 0;        ///< CUDA device index (0 = first GPU)
    size_t  nbytes    = 0;        ///< Total byte size of the data buffer

    /// Returns the byte size of one element for the given dtype.
    /// Returns 0 for unrecognised dtypes.
    static size_t dtype_size(int dtype) noexcept;

    /// Returns the total number of elements (product of all shape dimensions).
    /// Returns 0 if the tensor is uninitialised (ndim == 0 or shape == nullptr).
    size_t nelements() const noexcept;

    /// Computes nbytes for a tensor with the given shape, ndim, and dtype.
    /// Returns 0 if any argument is invalid (null shape, ndim <= 0, unknown dtype,
    /// or any non-positive dimension).
    static size_t compute_nbytes(const int* shape, int ndim, int dtype) noexcept;
};

} // namespace infergo
