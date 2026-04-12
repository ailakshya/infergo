#pragma once

#ifdef INFER_CUDA_AVAILABLE

#include <nvjpeg.h>
#include <torch/torch.h>
#include <cstdint>

namespace infergo {

/// GPU JPEG decoder using nvJPEG — decode directly to GPU memory.
/// Eliminates the CPU JPEG decode + upload pipeline (~1ms saved).
class NvJpegDecoder {
public:
    NvJpegDecoder();
    ~NvJpegDecoder();

    /// Decode JPEG bytes directly to a GPU tensor [H, W, 3] uint8.
    /// Returns empty tensor on failure (falls back to CPU decode).
    torch::Tensor Decode(const uint8_t* jpeg_data, int nbytes, torch::Device device);

    bool IsAvailable() const { return handle_ != nullptr; }

private:
    nvjpegHandle_t handle_ = nullptr;
    nvjpegJpegState_t state_ = nullptr;
    cudaStream_t stream_ = nullptr;
};

} // namespace infergo

#endif // INFER_CUDA_AVAILABLE
