#include "preprocess.hpp"
#include "tensor.hpp"

#include <stdexcept>
#include <cstring>

namespace infergo {

Tensor* normalize(const Tensor* src,
                  float scale,
                  const float mean[3],
                  const float std_dev[3])
{
    if (src == nullptr || mean == nullptr || std_dev == nullptr) {
        throw std::runtime_error("normalize: null argument");
    }
    if (src->dtype != INFER_DTYPE_FLOAT32) {
        throw std::runtime_error("normalize: input must be float32");
    }
    // Accept either [H, W, 3] (HWC) or [3, H, W] (CHW)
    if (src->ndim != 3) {
        throw std::runtime_error("normalize: input must be a 3-dim tensor");
    }

    // Determine layout
    const bool is_hwc = (src->shape[2] == 3);
    const bool is_chw = (src->shape[0] == 3);
    if (!is_hwc && !is_chw) {
        throw std::runtime_error("normalize: channel dim must be 3 (HWC or CHW)");
    }

    int C, H, W;
    if (is_hwc) {
        H = src->shape[0]; W = src->shape[1]; C = src->shape[2];
    } else {
        C = src->shape[0]; H = src->shape[1]; W = src->shape[2];
    }

    // Allocate output [C, H, W]
    const int out_shape[] = {C, H, W};
    Tensor* out = tensor_alloc_cpu(out_shape, 3, INFER_DTYPE_FLOAT32);
    if (out == nullptr) {
        throw std::runtime_error("normalize: tensor_alloc_cpu failed");
    }

    const float* src_data = static_cast<const float*>(src->data);
    float* dst_data = static_cast<float*>(out->data);

    if (is_hwc) {
        // HWC → CHW with scale + normalize
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                for (int c = 0; c < C; ++c) {
                    const float val = src_data[(h * W + w) * C + c] / scale;
                    dst_data[c * H * W + h * W + w] = (val - mean[c]) / std_dev[c];
                }
            }
        }
    } else {
        // CHW → CHW with scale + normalize
        for (int c = 0; c < C; ++c) {
            const int offset = c * H * W;
            for (int i = 0; i < H * W; ++i) {
                const float val = src_data[offset + i] / scale;
                dst_data[offset + i] = (val - mean[c]) / std_dev[c];
            }
        }
    }

    return out;
}

} // namespace infergo
