// cpp/postprocess/postprocess.hpp
#pragma once

#include "tensor.hpp"
#include "infer_api.h"
#include <vector>

namespace infergo {

struct ClassResult {
    int   label_idx;
    float confidence;
};

// Compute softmax over a 1-D float32 logits tensor and return the
// top_k entries sorted by confidence descending.
// Throws std::invalid_argument on bad input.
std::vector<ClassResult> classify(const Tensor* logits, int top_k);

} // namespace infergo
