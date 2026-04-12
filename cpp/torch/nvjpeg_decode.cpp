#ifdef INFER_CUDA_AVAILABLE

#include "nvjpeg_decode.hpp"
#include <cstdio>

namespace infergo {

NvJpegDecoder::NvJpegDecoder() {
    nvjpegStatus_t st = nvjpegCreateSimple(&handle_);
    if (st != NVJPEG_STATUS_SUCCESS) {
        std::fprintf(stderr, "[infergo] nvJPEG init failed (status=%d) — GPU JPEG decode disabled\n", st);
        handle_ = nullptr;
        return;
    }
    nvjpegJpegStateCreate(handle_, &state_);
    cudaStreamCreate(&stream_);
}

NvJpegDecoder::~NvJpegDecoder() {
    if (state_)  nvjpegJpegStateDestroy(state_);
    if (handle_) nvjpegDestroy(handle_);
    if (stream_) cudaStreamDestroy(stream_);
}

torch::Tensor NvJpegDecoder::Decode(const uint8_t* jpeg_data, int nbytes,
                                     torch::Device device) {
    if (!handle_ || !jpeg_data || nbytes <= 0) return {};

    // Get image dimensions
    int nComponents = 0;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];

    nvjpegStatus_t st = nvjpegGetImageInfo(handle_, jpeg_data, static_cast<size_t>(nbytes),
                                            &nComponents, &subsampling, widths, heights);
    if (st != NVJPEG_STATUS_SUCCESS || nComponents < 3) return {};

    int width = widths[0], height = heights[0];

    // Allocate output: interleaved RGB on GPU
    nvjpegImage_t output;
    size_t pitch = static_cast<size_t>(width) * 3;
    void* gpu_ptr = nullptr;
    cudaMalloc(&gpu_ptr, pitch * static_cast<size_t>(height));
    output.channel[0] = static_cast<unsigned char*>(gpu_ptr);
    output.pitch[0] = static_cast<unsigned int>(pitch);
    for (int i = 1; i < NVJPEG_MAX_COMPONENT; i++) {
        output.channel[i] = nullptr;
        output.pitch[i] = 0;
    }

    // Decode to GPU (interleaved RGB)
    st = nvjpegDecode(handle_, state_, jpeg_data, static_cast<size_t>(nbytes),
                       NVJPEG_OUTPUT_RGBI, &output, stream_);
    cudaStreamSynchronize(stream_);

    if (st != NVJPEG_STATUS_SUCCESS) {
        cudaFree(output.channel[0]);
        return {};
    }

    // Wrap as torch tensor (takes ownership via custom deleter)
    auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(device);
    auto tensor = torch::from_blob(
        output.channel[0], {height, width, 3},
        [](void* ptr) { cudaFree(ptr); },  // custom deleter
        opts);

    return tensor;
}

} // namespace infergo

#endif // INFER_CUDA_AVAILABLE
