#include "postprocess.hpp"
#include "infer_api.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace infergo {

std::vector<ClassResult> classify(const Tensor* logits, int top_k) {
    if (logits == nullptr)
        throw std::invalid_argument("classify: null logits tensor");

    const int n = static_cast<int>(logits->nelements());
    if (n <= 0)
        throw std::invalid_argument("classify: empty logits tensor");
    if (logits->dtype != INFER_DTYPE_FLOAT32)
        throw std::invalid_argument("classify: logits must be float32");
    if (top_k <= 0)
        throw std::invalid_argument("classify: top_k must be positive");

    const float* data = static_cast<const float*>(logits->data);

    // Numerically stable softmax: subtract max before exp
    float max_val = *std::max_element(data, data + n);
    std::vector<float> probs(n);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        probs[i] = std::exp(data[i] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < n; ++i)
        probs[i] /= sum;

    // Build index array and partial-sort by confidence descending
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) indices[i] = i;

    const int k = std::min(top_k, n);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&](int a, int b) { return probs[a] > probs[b]; });

    std::vector<ClassResult> results(k);
    for (int i = 0; i < k; ++i)
        results[i] = {indices[i], probs[indices[i]]};

    return results;
}

} // namespace infergo
