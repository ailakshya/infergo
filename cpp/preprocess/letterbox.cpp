#include "preprocess.hpp"
#include "tensor.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <stdexcept>

namespace infergo {

Tensor* letterbox(const Tensor* src, int target_w, int target_h) {
    if (src == nullptr) {
        throw std::runtime_error("letterbox: null input tensor");
    }
    if (src->ndim != 3 || src->shape[2] != 3) {
        throw std::runtime_error("letterbox: input must be [H, W, 3]");
    }
    if (target_w <= 0 || target_h <= 0) {
        throw std::runtime_error("letterbox: invalid target dimensions");
    }

    const int src_h = src->shape[0];
    const int src_w = src->shape[1];

    // Compute uniform scale to fit inside target (preserve aspect ratio)
    const float scale = std::min(
        static_cast<float>(target_w) / static_cast<float>(src_w),
        static_cast<float>(target_h) / static_cast<float>(src_h)
    );
    const int new_w = static_cast<int>(static_cast<float>(src_w) * scale);
    const int new_h = static_cast<int>(static_cast<float>(src_h) * scale);

    // Build source cv::Mat view (float32, HWC, no copy)
    const cv::Mat src_mat(src_h, src_w, CV_32FC3,
                          const_cast<void*>(static_cast<const void*>(src->data)));

    // Resize
    cv::Mat resized;
    cv::resize(src_mat, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // Create padded output filled with 114.0
    const int shape[] = {target_h, target_w, 3};
    Tensor* out = tensor_alloc_cpu(shape, 3, INFER_DTYPE_FLOAT32);
    if (out == nullptr) {
        throw std::runtime_error("letterbox: tensor_alloc_cpu failed");
    }
    float* dst_data = static_cast<float*>(out->data);
    const int total = target_h * target_w * 3;
    for (int i = 0; i < total; ++i) dst_data[i] = 114.0f;

    // Compute top-left padding offsets (centre the image)
    const int pad_top  = (target_h - new_h) / 2;
    const int pad_left = (target_w - new_w) / 2;

    // Copy resized image into padded canvas
    cv::Mat dst_mat(target_h, target_w, CV_32FC3, dst_data);
    resized.copyTo(dst_mat(cv::Rect(pad_left, pad_top, new_w, new_h)));

    return out;
}

} // namespace infergo
