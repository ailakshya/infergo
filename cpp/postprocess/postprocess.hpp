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

struct Box {
    float x1, y1, x2, y2;
    int   class_idx;
    float confidence;
};

// Compute softmax over a 1-D float32 logits tensor and return the
// top_k entries sorted by confidence descending.
// Throws std::invalid_argument on bad input.
std::vector<ClassResult> classify(const Tensor* logits, int top_k);

// Run NMS on a YOLO output tensor of shape [1, num_detections, 4+num_classes].
// Each row: [cx, cy, w, h, class0_score, class1_score, ...].
// Throws std::invalid_argument on bad input.
std::vector<Box> nms(const Tensor* predictions,
                     float conf_thresh, float iou_thresh);

} // namespace infergo
