#include "postprocess.hpp"

#include <cmath>
#include <stdexcept>

namespace infergo {

void normalize_embedding(Tensor* t) {
    if (t == nullptr)
        throw std::invalid_argument("normalize_embedding: null tensor");
    if (t->dtype != INFER_DTYPE_FLOAT32)
        throw std::invalid_argument("normalize_embedding: tensor must be float32");

    const int n = static_cast<int>(t->nelements());
    if (n <= 0)
        throw std::invalid_argument("normalize_embedding: empty tensor");

    float* data = static_cast<float*>(t->data);

    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i)
        sum_sq += static_cast<double>(data[i]) * static_cast<double>(data[i]);

    const float norm = static_cast<float>(std::sqrt(sum_sq));
    if (norm == 0.0f) return;  // zero vector — leave unchanged

    for (int i = 0; i < n; ++i)
        data[i] /= norm;
}

} // namespace infergo
