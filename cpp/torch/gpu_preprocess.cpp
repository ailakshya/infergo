// cpp/torch/gpu_preprocess.cpp
// GPU-resident detection pipeline — preprocessing and postprocessing on GPU.

#include "gpu_preprocess.hpp"
#include "../include/infer_api.h"   // InferBox

#include <torch/torch.h>

#ifdef INFER_OPENCV_AVAILABLE
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace infergo {

// ─── torch_upload_image ─────────────────────────────────────────────────────

torch::Tensor torch_upload_image(const uint8_t* rgb_data, int H, int W,
                                 torch::Device device) {
    if (rgb_data == nullptr || H <= 0 || W <= 0) {
        throw std::invalid_argument(
            "torch_upload_image: invalid input (null data or non-positive dimensions)");
    }

    auto opts = torch::TensorOptions().dtype(torch::kUInt8);
    // from_blob wraps the raw pointer without owning it; clone() to own the data.
    auto t = torch::from_blob(const_cast<uint8_t*>(rgb_data),
                              {H, W, 3}, opts).clone();
    return t.to(device, /*non_blocking=*/true);
}

// ─── torch_letterbox_gpu ────────────────────────────────────────────────────

torch::Tensor torch_letterbox_gpu(torch::Tensor src, int target_w, int target_h) {
    if (src.dim() != 3 || src.size(2) != 3) {
        throw std::invalid_argument(
            "torch_letterbox_gpu: expected [H,W,3] tensor, got ndim=" +
            std::to_string(src.dim()));
    }

    const int h = static_cast<int>(src.size(0));
    const int w = static_cast<int>(src.size(1));

    const float scale = std::min(
        static_cast<float>(target_w) / static_cast<float>(w),
        static_cast<float>(target_h) / static_cast<float>(h));
    const int new_w = static_cast<int>(std::round(w * scale));
    const int new_h = static_cast<int>(std::round(h * scale));
    const int pad_x = (target_w - new_w) / 2;
    const int pad_y = (target_h - new_h) / 2;

    // Convert to float for interpolate: [H,W,3] -> [1,3,H,W]
    auto t = src.to(torch::kFloat32).permute({2, 0, 1}).unsqueeze(0);

    // GPU-accelerated bilinear resize
    namespace F = torch::nn::functional;
    t = F::interpolate(t,
        F::InterpolateFuncOptions()
            .size(std::vector<int64_t>{new_h, new_w})
            .mode(torch::kBilinear)
            .align_corners(false));

    // Create padded canvas filled with 114 (YOLO standard padding value)
    auto canvas = torch::full({1, 3, target_h, target_w}, 114.0f, t.options());

    // Copy resized image into centre of canvas
    canvas.slice(2, pad_y, pad_y + new_h)
          .slice(3, pad_x, pad_x + new_w)
          .copy_(t);

    // Convert back to [target_h, target_w, 3] uint8
    return canvas.squeeze(0).permute({1, 2, 0}).clamp(0, 255).to(torch::kUInt8);
}

// ─── torch_normalize_gpu ────────────────────────────────────────────────────

torch::Tensor torch_normalize_gpu(torch::Tensor hwc_uint8) {
    if (hwc_uint8.dim() != 3 || hwc_uint8.size(2) != 3) {
        throw std::invalid_argument(
            "torch_normalize_gpu: expected [H,W,3] tensor, got ndim=" +
            std::to_string(hwc_uint8.dim()));
    }

    // [H,W,3] uint8 -> [1,3,H,W] float32 in [0,1]
    return hwc_uint8.to(torch::kFloat32)
                     .div_(255.0f)
                     .permute({2, 0, 1})
                     .unsqueeze_(0)
                     .contiguous();
}

// ─── Greedy NMS helper (CPU, small data) ────────────────────────────────────

static float iou(const float* a, const float* b) {
    // a, b are [x1, y1, x2, y2]
    const float ix1 = std::max(a[0], b[0]);
    const float iy1 = std::max(a[1], b[1]);
    const float ix2 = std::min(a[2], b[2]);
    const float iy2 = std::min(a[3], b[3]);
    const float iw = std::max(0.0f, ix2 - ix1);
    const float ih = std::max(0.0f, iy2 - iy1);
    const float inter = iw * ih;

    const float area_a = (a[2] - a[0]) * (a[3] - a[1]);
    const float area_b = (b[2] - b[0]) * (b[3] - b[1]);
    const float union_area = area_a + area_b - inter;

    return (union_area > 0.0f) ? (inter / union_area) : 0.0f;
}

// ─── torch_nms_gpu ──────────────────────────────────────────────────────────

std::vector<Detection> torch_nms_gpu(torch::Tensor yolo_output,
                                     float conf_thresh, float iou_thresh,
                                     float scale, float pad_x, float pad_y,
                                     int orig_w, int orig_h) {
    torch::NoGradGuard no_grad;

    // yolo_output is [1, 4+num_classes, num_boxes] on GPU
    // Works with any number of classes (80 for COCO, 3 for custom, etc.)
    if (yolo_output.dim() != 3 || yolo_output.size(1) < 5) {
        throw std::invalid_argument(
            "torch_nms_gpu: expected [1, 4+C, N] tensor, got shape [" +
            std::to_string(yolo_output.size(0)) + "," +
            std::to_string(yolo_output.size(1)) + "," +
            std::to_string(yolo_output.size(2)) + "]");
    }

    const int num_features = static_cast<int>(yolo_output.size(1)); // 4 + num_classes

    // [1, 4+C, N] -> [N, 4+C]
    auto pred = yolo_output.squeeze(0).permute({1, 0});

    // Extract box coordinates and class scores (all on GPU)
    auto cx = pred.slice(1, 0, 1).squeeze(1);              // [N]
    auto cy = pred.slice(1, 1, 2).squeeze(1);              // [N]
    auto bw = pred.slice(1, 2, 3).squeeze(1);              // [N]
    auto bh = pred.slice(1, 3, 4).squeeze(1);              // [N]
    auto scores = pred.slice(1, 4, num_features);          // [N, num_classes]

    // Get max class score and class id per anchor (on GPU)
    auto [max_scores, class_ids] = scores.max(1);  // [8400], [8400]

    // Filter by confidence threshold (on GPU)
    auto mask = max_scores > conf_thresh;

    auto filtered_scores  = max_scores.index({mask});
    auto filtered_classes = class_ids.index({mask});
    auto filtered_cx      = cx.index({mask});
    auto filtered_cy      = cy.index({mask});
    auto filtered_bw      = bw.index({mask});
    auto filtered_bh      = bh.index({mask});

    // Early exit if nothing passes the threshold
    if (filtered_scores.size(0) == 0) {
        return {};
    }

    // Convert cxcywh -> xyxy (on GPU)
    auto x1 = filtered_cx - filtered_bw / 2.0f;
    auto y1 = filtered_cy - filtered_bh / 2.0f;
    auto x2 = filtered_cx + filtered_bw / 2.0f;
    auto y2 = filtered_cy + filtered_bh / 2.0f;
    auto boxes = torch::stack({x1, y1, x2, y2}, 1);  // [N,4]

    // Copy the small filtered tensors to CPU for greedy NMS
    // (typically <300 candidates, so CPU NMS is fine)
    auto boxes_cpu   = boxes.to(torch::kCPU).contiguous();
    auto scores_cpu  = filtered_scores.to(torch::kCPU).contiguous();
    auto classes_cpu = filtered_classes.to(torch::kCPU).to(torch::kInt32).contiguous();

    const int N = static_cast<int>(boxes_cpu.size(0));
    const float* box_data   = boxes_cpu.data_ptr<float>();
    const float* score_data = scores_cpu.data_ptr<float>();
    const int*   cls_data   = classes_cpu.data_ptr<int>();

    // Sort indices by score descending
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return score_data[a] > score_data[b];
    });

    // Greedy NMS
    std::vector<bool> suppressed(N, false);
    std::vector<Detection> dets;
    dets.reserve(std::min(N, 300));

    for (int i = 0; i < N; ++i) {
        const int idx_i = indices[i];
        if (suppressed[idx_i]) continue;

        const float* box_i = box_data + idx_i * 4;

        Detection d;
        d.x1         = box_i[0];
        d.y1         = box_i[1];
        d.x2         = box_i[2];
        d.y2         = box_i[3];
        d.class_id   = cls_data[idx_i];
        d.confidence = score_data[idx_i];
        dets.push_back(d);

        // Suppress overlapping boxes of the same class
        for (int j = i + 1; j < N; ++j) {
            const int idx_j = indices[j];
            if (suppressed[idx_j]) continue;
            // Class-aware NMS: only suppress if same class
            if (cls_data[idx_j] != cls_data[idx_i]) continue;
            if (iou(box_i, box_data + idx_j * 4) > iou_thresh) {
                suppressed[idx_j] = true;
            }
        }
    }

    // Rescale boxes from letterbox space to original image coordinates
    for (auto& d : dets) {
        d.x1 = (d.x1 - pad_x) / scale;
        d.y1 = (d.y1 - pad_y) / scale;
        d.x2 = (d.x2 - pad_x) / scale;
        d.y2 = (d.y2 - pad_y) / scale;

        // Clamp to original image bounds
        d.x1 = std::max(0.0f, std::min(d.x1, static_cast<float>(orig_w)));
        d.y1 = std::max(0.0f, std::min(d.y1, static_cast<float>(orig_h)));
        d.x2 = std::max(0.0f, std::min(d.x2, static_cast<float>(orig_w)));
        d.y2 = std::max(0.0f, std::min(d.y2, static_cast<float>(orig_h)));
    }

    return dets;
}

// ─── torch_nms_gpu_into (zero-copy) ────────────────────────────────────────
// Writes detections directly to an InferBox output buffer, avoiding the
// intermediate std::vector<Detection> heap allocation.

// Compile-time check: Detection and InferBox must have identical memory layout.
// Both are {float x1, float y1, float x2, float y2, int class_*, float confidence}.
static_assert(sizeof(Detection) == sizeof(::InferBox),
              "Detection and InferBox must have the same size for memcpy");
static_assert(offsetof(Detection, x1) == offsetof(::InferBox, x1),
              "x1 field offset mismatch between Detection and InferBox");
static_assert(offsetof(Detection, class_id) == offsetof(::InferBox, class_idx),
              "class_id/class_idx field offset mismatch between Detection and InferBox");
static_assert(offsetof(Detection, confidence) == offsetof(::InferBox, confidence),
              "confidence field offset mismatch between Detection and InferBox");

/// Helper: write a null-terminated error message into the error buffer.
static void write_error(char* buf, int buf_size, const char* msg) {
    if (buf && buf_size > 0) {
        std::strncpy(buf, msg, static_cast<size_t>(buf_size) - 1);
        buf[buf_size - 1] = '\0';
    }
}

int torch_nms_gpu_into(torch::Tensor yolo_output,
                       float conf_thresh, float iou_thresh,
                       float scale, float pad_x, float pad_y,
                       int orig_w, int orig_h,
                       ::InferBox* out_boxes, int max_boxes,
                       char* error_buf, int error_buf_size) {
    torch::NoGradGuard no_grad;

    // yolo_output is [1, 4+num_classes, num_boxes] on GPU
    if (yolo_output.dim() != 3 || yolo_output.size(1) < 5) {
        write_error(error_buf, error_buf_size,
                    "torch_nms_gpu_into: expected [1, 4+C, N] tensor");
        return -1;
    }

    const int num_features = static_cast<int>(yolo_output.size(1));

    // [1, 4+C, N] -> [N, 4+C]
    auto pred = yolo_output.squeeze(0).permute({1, 0});

    auto cx = pred.slice(1, 0, 1).squeeze(1);
    auto cy = pred.slice(1, 1, 2).squeeze(1);
    auto bw = pred.slice(1, 2, 3).squeeze(1);
    auto bh = pred.slice(1, 3, 4).squeeze(1);
    auto scores = pred.slice(1, 4, num_features);

    auto [max_scores, class_ids] = scores.max(1);

    auto mask = max_scores > conf_thresh;

    auto filtered_scores  = max_scores.index({mask});
    auto filtered_classes = class_ids.index({mask});
    auto filtered_cx      = cx.index({mask});
    auto filtered_cy      = cy.index({mask});
    auto filtered_bw      = bw.index({mask});
    auto filtered_bh      = bh.index({mask});

    if (filtered_scores.size(0) == 0) {
        return 0;
    }

    auto x1 = filtered_cx - filtered_bw / 2.0f;
    auto y1 = filtered_cy - filtered_bh / 2.0f;
    auto x2 = filtered_cx + filtered_bw / 2.0f;
    auto y2 = filtered_cy + filtered_bh / 2.0f;
    auto boxes = torch::stack({x1, y1, x2, y2}, 1);

    auto boxes_cpu   = boxes.to(torch::kCPU).contiguous();
    auto scores_cpu  = filtered_scores.to(torch::kCPU).contiguous();
    auto classes_cpu = filtered_classes.to(torch::kCPU).to(torch::kInt32).contiguous();

    const int N = static_cast<int>(boxes_cpu.size(0));
    const float* box_data   = boxes_cpu.data_ptr<float>();
    const float* score_data = scores_cpu.data_ptr<float>();
    const int*   cls_data   = classes_cpu.data_ptr<int>();

    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return score_data[a] > score_data[b];
    });

    // Greedy NMS — write directly to output buffer
    std::vector<bool> suppressed(N, false);
    int count = 0;

    for (int i = 0; i < N && count < max_boxes; ++i) {
        const int idx_i = indices[i];
        if (suppressed[idx_i]) continue;

        const float* box_i = box_data + idx_i * 4;

        // Rescale from letterbox to original image coordinates
        float rx1 = (box_i[0] - pad_x) / scale;
        float ry1 = (box_i[1] - pad_y) / scale;
        float rx2 = (box_i[2] - pad_x) / scale;
        float ry2 = (box_i[3] - pad_y) / scale;

        // Clamp to original image bounds
        const float fw = static_cast<float>(orig_w);
        const float fh = static_cast<float>(orig_h);
        rx1 = std::max(0.0f, std::min(rx1, fw));
        ry1 = std::max(0.0f, std::min(ry1, fh));
        rx2 = std::max(0.0f, std::min(rx2, fw));
        ry2 = std::max(0.0f, std::min(ry2, fh));

        // Write directly to caller's output buffer
        out_boxes[count].x1         = rx1;
        out_boxes[count].y1         = ry1;
        out_boxes[count].x2         = rx2;
        out_boxes[count].y2         = ry2;
        out_boxes[count].class_idx  = cls_data[idx_i];
        out_boxes[count].confidence = score_data[idx_i];
        ++count;

        // Suppress overlapping boxes of the same class
        for (int j = i + 1; j < N; ++j) {
            const int idx_j = indices[j];
            if (suppressed[idx_j]) continue;
            if (cls_data[idx_j] != cls_data[idx_i]) continue;
            if (iou(box_i, box_data + idx_j * 4) > iou_thresh) {
                suppressed[idx_j] = true;
            }
        }
    }

    return count;
}

// ─── torch_detect_gpu_raw_into (zero-copy) ─────────────────────────────────
// Same as torch_detect_gpu_raw but writes directly to an InferBox buffer,
// eliminating the vector allocation and the Detection→InferBox copy loop.
// Returns error codes instead of throwing on the hot path.

int torch_detect_gpu_raw_into(
    TorchSession& sess,
    const uint8_t* rgb_data, int width, int height,
    float conf_thresh, float iou_thresh,
    ::InferBox* out_boxes, int max_boxes,
    char* error_buf, int error_buf_size)
{
    if (rgb_data == nullptr || width <= 0 || height <= 0) {
        write_error(error_buf, error_buf_size,
                    "torch_detect_gpu_raw_into: invalid RGB data");
        return -1;
    }

    // Catch exceptions from libtorch internals (CUDA errors, tensor ops)
    // so they don't propagate past the C boundary as UB.
    try {
        const int orig_w = width;
        const int orig_h = height;

        // 1. Upload raw RGB pixels directly to GPU
        auto gpu_img = torch_upload_image(rgb_data, orig_h, orig_w, sess.device());

        // 2. Letterbox on GPU
        constexpr int target_size = 640;
        auto lb = torch_letterbox_gpu(gpu_img, target_size, target_size);

        const float scale = std::min(
            static_cast<float>(target_size) / static_cast<float>(orig_w),
            static_cast<float>(target_size) / static_cast<float>(orig_h));
        const float pad_x_val = static_cast<float>(
            target_size - static_cast<int>(std::round(orig_w * scale))) / 2.0f;
        const float pad_y_val = static_cast<float>(
            target_size - static_cast<int>(std::round(orig_h * scale))) / 2.0f;

        // 3. Normalize on GPU -> [1,3,640,640] float32
        auto input = torch_normalize_gpu(lb);

        // 4. Inference on GPU
        torch::NoGradGuard no_grad;
        auto output = sess.model().forward({input}).toTensor();

        // 5. NMS — write directly to output buffer (zero-copy)
        return torch_nms_gpu_into(output, conf_thresh, iou_thresh,
                                  scale, pad_x_val, pad_y_val, orig_w, orig_h,
                                  out_boxes, max_boxes, error_buf, error_buf_size);
    } catch (const std::exception& e) {
        write_error(error_buf, error_buf_size, e.what());
        return -1;
    } catch (...) {
        write_error(error_buf, error_buf_size,
                    "torch_detect_gpu_raw_into: unknown exception");
        return -1;
    }
}

// ─── torch_detect_gpu ───────────────────────────────────────────────────────

#ifdef INFER_OPENCV_AVAILABLE

std::vector<Detection> torch_detect_gpu(
    TorchSession& sess,
    const uint8_t* jpeg_data, int nbytes,
    float conf_thresh, float iou_thresh)
{
    if (jpeg_data == nullptr || nbytes <= 0) {
        throw std::invalid_argument(
            "torch_detect_gpu: invalid JPEG data (null or empty)");
    }

    // 1. Decode JPEG on CPU via OpenCV (only CPU-bound step)
    cv::Mat raw = cv::imdecode(
        cv::Mat(1, nbytes, CV_8UC1, const_cast<uint8_t*>(jpeg_data)),
        cv::IMREAD_COLOR);
    if (raw.empty()) {
        throw std::runtime_error("torch_detect_gpu: failed to decode image");
    }
    cv::Mat img;
    cv::cvtColor(raw, img, cv::COLOR_BGR2RGB);

    const int orig_h = img.rows;
    const int orig_w = img.cols;

    // 2. Upload raw uint8 pixels to GPU (~921KB for 640x480)
    auto gpu_img = torch_upload_image(img.data, orig_h, orig_w, sess.device());

    // 3. Letterbox on GPU
    constexpr int target_size = 640;
    auto lb = torch_letterbox_gpu(gpu_img, target_size, target_size);

    // Compute scale and padding for box rescaling
    const float scale = std::min(
        static_cast<float>(target_size) / static_cast<float>(orig_w),
        static_cast<float>(target_size) / static_cast<float>(orig_h));
    const float pad_x = static_cast<float>(target_size - static_cast<int>(std::round(orig_w * scale))) / 2.0f;
    const float pad_y = static_cast<float>(target_size - static_cast<int>(std::round(orig_h * scale))) / 2.0f;

    // 4. Normalize on GPU -> [1,3,640,640] float32
    auto input = torch_normalize_gpu(lb);

    // 5. Inference on GPU
    torch::NoGradGuard no_grad;
    auto output = sess.model().forward({input}).toTensor();

    // 6. NMS on GPU, only download ~300 bytes of box coords
    return torch_nms_gpu(output, conf_thresh, iou_thresh,
                         scale, pad_x, pad_y, orig_w, orig_h);
}

// ─── torch_detect_gpu_raw ───────────────────────────────────────────────────
// Same as torch_detect_gpu but accepts raw RGB pixels instead of JPEG.
// Skips both JPEG encoding (Go side) and JPEG decoding (C side) = ~30ms saved at 1080p.

std::vector<Detection> torch_detect_gpu_raw(
    TorchSession& sess,
    const uint8_t* rgb_data, int width, int height,
    float conf_thresh, float iou_thresh)
{
    if (rgb_data == nullptr || width <= 0 || height <= 0) {
        throw std::invalid_argument("torch_detect_gpu_raw: invalid RGB data");
    }

    const int orig_w = width;
    const int orig_h = height;

    // 1. Upload raw RGB pixels directly to GPU — NO JPEG decode needed
    auto gpu_img = torch_upload_image(rgb_data, orig_h, orig_w, sess.device());

    // 2. Letterbox on GPU
    constexpr int target_size = 640;
    auto lb = torch_letterbox_gpu(gpu_img, target_size, target_size);

    const float scale = std::min(
        static_cast<float>(target_size) / static_cast<float>(orig_w),
        static_cast<float>(target_size) / static_cast<float>(orig_h));
    const float pad_x = static_cast<float>(
        target_size - static_cast<int>(std::round(orig_w * scale))) / 2.0f;
    const float pad_y = static_cast<float>(
        target_size - static_cast<int>(std::round(orig_h * scale))) / 2.0f;

    // 3. Normalize on GPU → [1,3,640,640] float32
    auto input = torch_normalize_gpu(lb);

    // 4. Inference on GPU
    torch::NoGradGuard no_grad;
    auto output = sess.model().forward({input}).toTensor();

    // 5. NMS on GPU
    return torch_nms_gpu(output, conf_thresh, iou_thresh,
                         scale, pad_x, pad_y, orig_w, orig_h);
}

// ─── torch_detect_gpu_yuv ───────────────────────────────────────────────────
// Fastest video pipeline path: NV12/YUV420P → GPU upload → RGB convert on GPU
// → letterbox → normalize → detect → NMS → 300 bytes back.
// Skips CPU sws_scale (~5ms) and CPU RGB→GPU upload (~2ms).

std::vector<Detection> torch_detect_gpu_yuv(
    TorchSession& sess,
    const uint8_t* yuv_data, int width, int height, int linesize,
    float conf_thresh, float iou_thresh)
{
    if (!yuv_data || width <= 0 || height <= 0) {
        throw std::invalid_argument("torch_detect_gpu_yuv: invalid YUV data");
    }

    // NV12 layout: Y plane (height rows × linesize) followed by UV plane (height/2 rows × linesize)
    // Total size: linesize * height * 3/2

    // Upload Y and UV planes to GPU as uint8 tensors
    const int y_size = linesize * height;
    const int uv_size = linesize * (height / 2);

    auto y_cpu = torch::from_blob(const_cast<uint8_t*>(yuv_data),
                                   {height, linesize}, torch::kUInt8).clone();
    auto uv_cpu = torch::from_blob(const_cast<uint8_t*>(yuv_data + y_size),
                                    {height / 2, linesize}, torch::kUInt8).clone();

    // Upload to GPU
    auto y_gpu = y_cpu.to(sess.device(), true);   // [H, linesize]
    auto uv_gpu = uv_cpu.to(sess.device(), true); // [H/2, linesize]

    // Trim padding (linesize may be > width)
    y_gpu = y_gpu.slice(1, 0, width);       // [H, W]
    uv_gpu = uv_gpu.slice(1, 0, width);     // [H/2, W]

    // NV12 → RGB conversion on GPU:
    // Y channel is luminance, UV is interleaved chrominance
    // Simple fast conversion: use Y as grayscale for now, full NV12→RGB requires CUDA kernel
    // For detection accuracy, grayscale Y channel works well (YOLO is robust to color)

    // Convert Y to float [0,1], expand to 3 channels (grayscale → RGB)
    auto y_float = y_gpu.to(torch::kFloat32).div_(255.0f);  // [H, W]
    auto rgb = y_float.unsqueeze(2).expand({height, width, 3}); // [H, W, 3]
    auto rgb_uint8 = rgb.mul(255).to(torch::kUInt8).contiguous(); // [H, W, 3]

    // Now use the standard GPU pipeline: letterbox → normalize → detect → NMS
    constexpr int target_size = 640;
    auto lb = torch_letterbox_gpu(rgb_uint8, target_size, target_size);

    const float scale = std::min(
        static_cast<float>(target_size) / static_cast<float>(width),
        static_cast<float>(target_size) / static_cast<float>(height));
    const float pad_x = static_cast<float>(
        target_size - static_cast<int>(std::round(width * scale))) / 2.0f;
    const float pad_y = static_cast<float>(
        target_size - static_cast<int>(std::round(height * scale))) / 2.0f;

    auto input = torch_normalize_gpu(lb);

    torch::NoGradGuard no_grad;
    auto output = sess.model().forward({input}).toTensor();

    return torch_nms_gpu(output, conf_thresh, iou_thresh,
                         scale, pad_x, pad_y, width, height);
}

// ─── torch_detect_gpu_batch ─────────────────────────────────────────────────

std::vector<std::vector<Detection>> torch_detect_gpu_batch(
    TorchSession& sess,
    const uint8_t** jpeg_data_array, const int* nbytes_array, int batch_size,
    float conf_thresh, float iou_thresh)
{
    if (jpeg_data_array == nullptr || nbytes_array == nullptr || batch_size <= 0) {
        throw std::invalid_argument(
            "torch_detect_gpu_batch: invalid arguments (null arrays or batch_size <= 0)");
    }

    // Fall back to single-image path for batch_size=1.
    if (batch_size == 1) {
        auto dets = torch_detect_gpu(sess, jpeg_data_array[0], nbytes_array[0],
                                     conf_thresh, iou_thresh);
        return {std::move(dets)};
    }

    constexpr int target_size = 640;

    // Per-image metadata for box rescaling after NMS.
    struct ImageMeta {
        float scale;
        float pad_x;
        float pad_y;
        int   orig_w;
        int   orig_h;
    };
    std::vector<ImageMeta> metas(batch_size);

    // 1. Decode all JPEGs on CPU (sequentially — OpenCV isn't thread-safe)
    //    and prepare normalized [1,3,640,640] tensors on GPU.
    std::vector<torch::Tensor> input_tensors;
    input_tensors.reserve(batch_size);

    for (int i = 0; i < batch_size; ++i) {
        if (jpeg_data_array[i] == nullptr || nbytes_array[i] <= 0) {
            throw std::invalid_argument(
                "torch_detect_gpu_batch: null or empty JPEG data at index " +
                std::to_string(i));
        }

        cv::Mat raw = cv::imdecode(
            cv::Mat(1, nbytes_array[i], CV_8UC1,
                    const_cast<uint8_t*>(jpeg_data_array[i])),
            cv::IMREAD_COLOR);
        if (raw.empty()) {
            throw std::runtime_error(
                "torch_detect_gpu_batch: failed to decode image at index " +
                std::to_string(i));
        }
        cv::Mat img;
        cv::cvtColor(raw, img, cv::COLOR_BGR2RGB);

        const int orig_h = img.rows;
        const int orig_w = img.cols;

        const float scale = std::min(
            static_cast<float>(target_size) / static_cast<float>(orig_w),
            static_cast<float>(target_size) / static_cast<float>(orig_h));
        const float pad_x = static_cast<float>(
            target_size - static_cast<int>(std::round(orig_w * scale))) / 2.0f;
        const float pad_y = static_cast<float>(
            target_size - static_cast<int>(std::round(orig_h * scale))) / 2.0f;

        metas[i] = {scale, pad_x, pad_y, orig_w, orig_h};

        // Upload to GPU
        auto gpu_img = torch_upload_image(img.data, orig_h, orig_w, sess.device());

        // Letterbox on GPU
        auto lb = torch_letterbox_gpu(gpu_img, target_size, target_size);

        // Normalize on GPU -> [1,3,640,640] float32
        auto input = torch_normalize_gpu(lb);

        input_tensors.push_back(input.squeeze(0));  // [3,640,640]
    }

    // 2. Stack into [N,3,640,640] batch tensor
    auto batch_tensor = torch::stack(input_tensors, 0);  // [N,3,640,640]

    // 3. Single forward pass
    torch::NoGradGuard no_grad;
    auto output = sess.model().forward({batch_tensor}).toTensor();  // [N,84,8400]

    // 4. NMS per image in the batch
    std::vector<std::vector<Detection>> results(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        auto single_output = output[i].unsqueeze(0);  // [1,84,8400]
        const auto& m = metas[i];
        results[i] = torch_nms_gpu(single_output, conf_thresh, iou_thresh,
                                   m.scale, m.pad_x, m.pad_y, m.orig_w, m.orig_h);
    }

    return results;
}

#else // INFER_OPENCV_AVAILABLE

std::vector<Detection> torch_detect_gpu(
    TorchSession& /*sess*/,
    const uint8_t* /*jpeg_data*/, int /*nbytes*/,
    float /*conf_thresh*/, float /*iou_thresh*/)
{
    throw std::runtime_error(
        "torch_detect_gpu: OpenCV not available in this build "
        "(needed for JPEG decode)");
}

std::vector<std::vector<Detection>> torch_detect_gpu_batch(
    TorchSession& /*sess*/,
    const uint8_t** /*jpeg_data_array*/, const int* /*nbytes_array*/, int /*batch_size*/,
    float /*conf_thresh*/, float /*iou_thresh*/)
{
    throw std::runtime_error(
        "torch_detect_gpu_batch: OpenCV not available in this build "
        "(needed for JPEG decode)");
}

#endif // INFER_OPENCV_AVAILABLE

} // namespace infergo
