#include "tensor.hpp"

namespace infergo {

size_t Tensor::dtype_size(int dtype) noexcept {
    switch (dtype) {
        case 0: return 4;  // INFER_DTYPE_FLOAT32
        case 1: return 2;  // INFER_DTYPE_FLOAT16
        case 2: return 2;  // INFER_DTYPE_BFLOAT16
        case 3: return 4;  // INFER_DTYPE_INT32
        case 4: return 8;  // INFER_DTYPE_INT64
        case 5: return 1;  // INFER_DTYPE_UINT8
        case 6: return 1;  // INFER_DTYPE_BOOL
        default: return 0;
    }
}

} // namespace infergo
