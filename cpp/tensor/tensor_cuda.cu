// tensor_cuda.cu
// CUDA kernel implementations for libinfer_tensor.
// Populated in T-04 (infer_tensor_alloc_cuda) and T-06 (infer_tensor_to_device/to_host).
// This stub exists so the shared library compiles while T-04 is in progress.

#include "tensor.hpp"

// Silence "empty translation unit" warning from nvcc
namespace infergo { namespace cuda { } }
