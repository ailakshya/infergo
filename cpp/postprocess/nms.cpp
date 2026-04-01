#include "postprocess.hpp"

#include <algorithm>
#include <stdexcept>

namespace infergo {

static float iou(const Box& a, const Box& b) {
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);

    const float iw = std::max(0.0f, ix2 - ix1);
    const float ih = std::max(0.0f, iy2 - iy1);
    const float inter = iw * ih;
    if (inter == 0.0f) return 0.0f;

    const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area_a + area_b - inter);
}

std::vector<Box> nms(const Tensor* predictions,
                     float conf_thresh, float iou_thresh) {
    if (predictions == nullptr)
        throw std::invalid_argument("nms: null predictions tensor");
    if (predictions->dtype != INFER_DTYPE_FLOAT32)
        throw std::invalid_argument("nms: predictions must be float32");
    if (conf_thresh < 0.0f || conf_thresh > 1.0f)
        throw std::invalid_argument("nms: conf_thresh must be in [0, 1]");
    if (iou_thresh < 0.0f || iou_thresh > 1.0f)
        throw std::invalid_argument("nms: iou_thresh must be in [0, 1]");

    // Expect shape [1, num_detections, 4+num_classes]
    if (predictions->ndim != 3)
        throw std::invalid_argument("nms: expected 3-D tensor [1, D, 4+C]");
    if (predictions->shape[0] != 1)
        throw std::invalid_argument("nms: batch dimension must be 1");

    const int num_det    = predictions->shape[1];
    const int row_stride = predictions->shape[2];  // 4 + num_classes
    if (row_stride <= 4)
        throw std::invalid_argument("nms: row stride must be > 4 (4 box coords + >=1 class)");

    const int num_classes = row_stride - 4;
    const float* data = static_cast<const float*>(predictions->data);

    // ── Step 1: decode candidates above conf_thresh ───────────────────────────
    std::vector<Box> candidates;
    candidates.reserve(num_det);

    for (int d = 0; d < num_det; ++d) {
        const float* row = data + d * row_stride;
        const float cx = row[0], cy = row[1], w = row[2], h = row[3];

        // Find best class
        int   best_cls  = 0;
        float best_conf = row[4];
        for (int c = 1; c < num_classes; ++c) {
            if (row[4 + c] > best_conf) {
                best_conf = row[4 + c];
                best_cls  = c;
            }
        }

        if (best_conf < conf_thresh) continue;

        candidates.push_back({
            cx - w * 0.5f, cy - h * 0.5f,   // x1, y1
            cx + w * 0.5f, cy + h * 0.5f,   // x2, y2
            best_cls, best_conf
        });
    }

    // ── Step 2: sort by confidence descending ─────────────────────────────────
    std::sort(candidates.begin(), candidates.end(),
              [](const Box& a, const Box& b) { return a.confidence > b.confidence; });

    // ── Step 3: greedy NMS ────────────────────────────────────────────────────
    std::vector<bool> suppressed(candidates.size(), false);
    std::vector<Box>  kept;

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(candidates[i]);
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) continue;
            if (candidates[i].class_idx != candidates[j].class_idx) continue;
            if (iou(candidates[i], candidates[j]) > iou_thresh)
                suppressed[j] = true;
        }
    }

    return kept;
}

} // namespace infergo
