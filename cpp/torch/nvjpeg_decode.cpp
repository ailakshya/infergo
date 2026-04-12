#ifdef INFER_CUDA_AVAILABLE

#include "nvjpeg_decode.hpp"
#include <cstdio>
#include <cstring>

namespace infergo {

NvJpegDecoder::NvJpegDecoder() {
    nvjpegStatus_t st = nvjpegCreateSimple(&handle_);
    if (st != NVJPEG_STATUS_SUCCESS) {
        std::fprintf(stderr, "[infergo] nvJPEG init failed (status=%d)\n", st);
        handle_ = nullptr;
        return;
    }
    nvjpegJpegStateCreate(handle_, &state_);
    cudaStreamCreate(&stream_);

    // Pre-warm: allocate a persistent GPU buffer for 640x640 (common YOLO size).
    // This avoids cudaMalloc on the first real decode call.
    warmup_w_ = 640;
    warmup_h_ = 640;
    size_t sz = static_cast<size_t>(warmup_w_) * static_cast<size_t>(warmup_h_) * 3;
    void* ptr = nullptr;
    cudaMalloc(&ptr, sz);
    if (ptr) {
        persistent_buf_ = static_cast<unsigned char*>(ptr);
        persistent_buf_size_ = sz;
        // Warm up the CUDA context + nvJPEG pipeline with a dummy decode
        cudaMemset(persistent_buf_, 0, sz);
        cudaStreamSynchronize(stream_);
    }
}

NvJpegDecoder::~NvJpegDecoder() {
    if (persistent_buf_) cudaFree(persistent_buf_);
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
    size_t needed = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;

    // Reuse persistent buffer if it fits, otherwise reallocate
    unsigned char* out_ptr = nullptr;
    bool owns_buf = false;
    if (persistent_buf_ && needed <= persistent_buf_size_) {
        out_ptr = persistent_buf_;
    } else {
        void* ptr = nullptr;
        cudaMalloc(&ptr, needed);
        if (!ptr) return {};
        out_ptr = static_cast<unsigned char*>(ptr);
        owns_buf = true;
        // Update persistent buffer for future reuse
        if (persistent_buf_) cudaFree(persistent_buf_);
        persistent_buf_ = out_ptr;
        persistent_buf_size_ = needed;
        owns_buf = false;  // now owned by persistent_buf_
    }

    // Decode to GPU (interleaved RGB)
    nvjpegImage_t output;
    output.channel[0] = out_ptr;
    output.pitch[0] = static_cast<unsigned int>(static_cast<size_t>(width) * 3);
    for (int i = 1; i < NVJPEG_MAX_COMPONENT; i++) {
        output.channel[i] = nullptr;
        output.pitch[i] = 0;
    }

    st = nvjpegDecode(handle_, state_, jpeg_data, static_cast<size_t>(nbytes),
                       NVJPEG_OUTPUT_RGBI, &output, stream_);
    cudaStreamSynchronize(stream_);

    if (st != NVJPEG_STATUS_SUCCESS) {
        if (owns_buf) cudaFree(out_ptr);
        return {};
    }

    // Wrap as torch tensor with custom deleter to manage GPU memory.
    // Allocate a fresh buffer for the tensor so the persistent buf stays free.
    void* tensor_buf = nullptr;
    cudaMalloc(&tensor_buf, needed);
    if (!tensor_buf) return {};
    cudaMemcpyAsync(tensor_buf, out_ptr, needed, cudaMemcpyDeviceToDevice, stream_);
    cudaStreamSynchronize(stream_);

    auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(device);
    return torch::from_blob(tensor_buf, {height, width, 3},
        [](void* p) { cudaFree(p); }, opts);
}

} // namespace infergo

#endif // INFER_CUDA_AVAILABLE
