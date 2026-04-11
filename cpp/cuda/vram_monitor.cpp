// cpp/cuda/vram_monitor.cpp
// VRAM monitoring via CUDA runtime API (cudaMemGetInfo).
// Guarded with INFER_CUDA_AVAILABLE — returns 0 on CPU-only builds.

#include "vram_monitor.hpp"

#ifdef INFER_CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

namespace infergo {

size_t cuda_vram_free() {
#ifdef INFER_CUDA_AVAILABLE
    if (cudaSetDevice(0) != cudaSuccess) {
        return 0;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        return 0;
    }
    return free_bytes;
#else
    return 0;
#endif
}

size_t cuda_vram_total() {
#ifdef INFER_CUDA_AVAILABLE
    if (cudaSetDevice(0) != cudaSuccess) {
        return 0;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        return 0;
    }
    return total_bytes;
#else
    return 0;
#endif
}

int cuda_vram_used_pct() {
#ifdef INFER_CUDA_AVAILABLE
    if (cudaSetDevice(0) != cudaSuccess) {
        return 0;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        return 0;
    }
    if (total_bytes == 0) {
        return 0;
    }
    size_t used = total_bytes - free_bytes;
    return static_cast<int>((used * 100) / total_bytes);
#else
    return 0;
#endif
}

} // namespace infergo
