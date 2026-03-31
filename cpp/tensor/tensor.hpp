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

// ─── Allocation ──────────────────────────────────────────────────────────────

/// Allocate a CPU tensor via malloc.
/// Validates shape, ndim, and dtype. Returns nullptr on any failure and sets
/// the thread-local error string via set_last_error().
/// All three allocations (Tensor struct, shape array, data buffer) use malloc,
/// never operator new, to preserve C ABI compatibility.
Tensor* tensor_alloc_cpu(const int* shape, int ndim, int dtype) noexcept;

/// Free a tensor allocated by tensor_alloc_cpu (or tensor_alloc_cuda in T-04).
/// Safe to call with nullptr. Zeroes all pointer fields before freeing the
/// struct to catch use-after-free (RULE 5).
/// CUDA path (on_device == true) is completed in T-05.
void tensor_free(Tensor* t) noexcept;

// ─── Error string ─────────────────────────────────────────────────────────────

/// Set the thread-local last-error string. Used internally by alloc/free
/// and exposed to Go via infer_last_error_string() in T-09.
void set_last_error(const char* msg) noexcept;

/// Return the thread-local last-error string. Empty string if no error.
const char* get_last_error() noexcept;

} // namespace infergo
