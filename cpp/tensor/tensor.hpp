#pragma once

#include <cstddef>

namespace infergo {

/// Internal C++ representation of a tensor.
/// Never exposed across the C API — Go only ever sees void* (InferTensor).
/// Populated in T-02; this header exists so CMake can build a minimal library
/// target while T-01 is in progress.
struct Tensor {
    void*   data      = nullptr;  ///< Raw data pointer (CPU heap or CUDA device)
    int*    shape     = nullptr;  ///< Heap-allocated shape array, length == ndim
    int     ndim      = 0;        ///< Number of dimensions
    int     dtype     = 0;        ///< Data type (INFER_DTYPE_* constants)
    bool    on_device = false;    ///< true when data lives on a CUDA device
    int     device_id = 0;        ///< CUDA device index (0 = first GPU)
    size_t  nbytes    = 0;        ///< Total byte size of the data buffer

    /// Returns the byte size of one element for the given dtype constant.
    /// Returns 0 for unknown dtypes.
    static size_t dtype_size(int dtype) noexcept;
};

} // namespace infergo
