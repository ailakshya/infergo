// cpp/include/infer_api.h
// THE ONLY HEADER THAT GO EVER INCLUDES
// All functions are C-compatible — no C++ types cross this boundary

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// ─────────────────────────────────────────────────────────────────────────────
// ERROR CODES
// ─────────────────────────────────────────────────────────────────────────────

typedef int InferError;

#define INFER_OK               0
#define INFER_ERR_NULL         1   // null pointer passed
#define INFER_ERR_INVALID      2   // invalid argument
#define INFER_ERR_OOM          3   // out of memory
#define INFER_ERR_CUDA         4   // CUDA error
#define INFER_ERR_LOAD         5   // model/file load failure
#define INFER_ERR_RUNTIME      6   // inference runtime error
#define INFER_ERR_SHAPE        7   // tensor shape mismatch
#define INFER_ERR_DTYPE        8   // unsupported dtype
#define INFER_ERR_CANCELLED    9   // operation cancelled
#define INFER_ERR_UNKNOWN      99  // unknown error

// Get human-readable error string for last error on this thread.
// Returns a pointer to a thread-local buffer — valid until the next API call
// on this thread. Never free this pointer.
const char* infer_last_error_string(void);

// ─────────────────────────────────────────────────────────────────────────────
// DTYPE CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────

#define INFER_DTYPE_FLOAT32   0
#define INFER_DTYPE_FLOAT16   1
#define INFER_DTYPE_BFLOAT16  2
#define INFER_DTYPE_INT32     3
#define INFER_DTYPE_INT64     4
#define INFER_DTYPE_UINT8     5
#define INFER_DTYPE_BOOL      6

// ─────────────────────────────────────────────────────────────────────────────
// TENSOR API
// ─────────────────────────────────────────────────────────────────────────────

typedef void* InferTensor;

// Allocate tensor on CPU heap.
// Returns NULL on failure; call infer_last_error_string() for details.
InferTensor infer_tensor_alloc_cpu(const int* shape, int ndim, int dtype);

// Allocate tensor on CUDA device (device_id = 0 for first GPU).
// Returns NULL on failure; call infer_last_error_string() for details.
InferTensor infer_tensor_alloc_cuda(const int* shape, int ndim, int dtype, int device_id);

// Free tensor memory (works for both CPU and CUDA tensors).
// Safe to call with NULL.
void infer_tensor_free(InferTensor t);

// Get raw data pointer.
// CPU tensors: pointer to heap memory — safe to read/write from Go.
// CUDA tensors: device pointer — do NOT dereference from Go.
// Returns NULL if t is NULL.
void* infer_tensor_data_ptr(InferTensor t);

// Get tensor size in bytes. Returns 0 if t is NULL.
int infer_tensor_nbytes(InferTensor t);

// Get number of elements. Returns 0 if t is NULL.
int infer_tensor_nelements(InferTensor t);

// Get shape (writes up to max_dims integers into out_shape, returns ndim).
// Returns 0 if t or out_shape is NULL, or max_dims <= 0.
int infer_tensor_shape(InferTensor t, int* out_shape, int max_dims);

// Get dtype constant (INFER_DTYPE_*). Returns -1 if t is NULL.
int infer_tensor_dtype(InferTensor t);

// Copy CPU tensor data to CUDA device.
// No-op if already on the requested device.
InferError infer_tensor_to_device(InferTensor t, int device_id);

// Copy CUDA tensor data back to CPU.
// No-op if already on host.
InferError infer_tensor_to_host(InferTensor t);

// Copy nbytes from src (host pointer) into a CPU tensor's data buffer.
// src must point to at least nbytes bytes. nbytes must equal the tensor's nbytes.
InferError infer_tensor_copy_from(InferTensor t, const void* src, int nbytes);

#ifdef __cplusplus
} // extern "C"
#endif
