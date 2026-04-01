#include "preprocess.hpp"
#include "tensor.hpp"

#include <cstring>
#include <stdexcept>

namespace infergo {

Tensor* stack_batch(const std::vector<const Tensor*>& tensors) {
    if (tensors.empty()) {
        throw std::runtime_error("stack_batch: empty tensor list");
    }

    // All tensors must be [C, H, W] float32 with identical shape
    const Tensor* first = tensors[0];
    if (first == nullptr) {
        throw std::runtime_error("stack_batch: tensors[0] is null");
    }
    if (first->ndim != 3 || first->dtype != INFER_DTYPE_FLOAT32) {
        throw std::runtime_error("stack_batch: input must be [C, H, W] float32");
    }

    const int C = first->shape[0];
    const int H = first->shape[1];
    const int W = first->shape[2];
    const int per_tensor = C * H * W;
    const int N = static_cast<int>(tensors.size());

    for (int i = 1; i < N; ++i) {
        if (tensors[i] == nullptr) {
            throw std::runtime_error("stack_batch: tensors[" + std::to_string(i) + "] is null");
        }
        if (tensors[i]->ndim != 3 ||
            tensors[i]->shape[0] != C ||
            tensors[i]->shape[1] != H ||
            tensors[i]->shape[2] != W ||
            tensors[i]->dtype != INFER_DTYPE_FLOAT32)
        {
            throw std::runtime_error("stack_batch: all tensors must have identical [C,H,W] float32 shape");
        }
    }

    // Allocate [N, C, H, W]
    const int out_shape[] = {N, C, H, W};
    Tensor* out = tensor_alloc_cpu(out_shape, 4, INFER_DTYPE_FLOAT32);
    if (out == nullptr) {
        throw std::runtime_error("stack_batch: tensor_alloc_cpu failed");
    }

    float* dst = static_cast<float*>(out->data);
    for (int i = 0; i < N; ++i) {
        const float* src = static_cast<const float*>(tensors[i]->data);
        std::memcpy(dst + static_cast<size_t>(i) * static_cast<size_t>(per_tensor),
                    src,
                    static_cast<size_t>(per_tensor) * sizeof(float));
    }

    return out;
}

} // namespace infergo
