#include "preprocess.hpp"
#include "tensor.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <stdexcept>
#include <vector>

namespace infergo {

Tensor* decode_image(const uint8_t* data, int nbytes) {
    if (data == nullptr || nbytes <= 0) {
        throw std::runtime_error("decode_image: null or empty input");
    }

    // Decode via OpenCV (handles JPEG, PNG, WebP, BMP, etc.)
    const std::vector<uint8_t> buf(data, data + nbytes);
    cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);  // BGR, uint8
    if (img.empty()) {
        throw std::runtime_error("decode_image: imdecode failed — unsupported format or corrupt data");
    }

    // Convert BGR → RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    const int H = img.rows;
    const int W = img.cols;
    const int shape[] = {H, W, 3};

    Tensor* t = tensor_alloc_cpu(shape, 3, INFER_DTYPE_FLOAT32);
    if (t == nullptr) {
        throw std::runtime_error("decode_image: tensor_alloc_cpu failed");
    }

    // Copy uint8 pixel data → float32, values stay in [0, 255]
    float* dst = static_cast<float*>(t->data);
    const uint8_t* src = img.data;
    const int n = H * W * 3;
    for (int i = 0; i < n; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }

    return t;
}

} // namespace infergo
