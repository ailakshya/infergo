// cpp/torch/gpu_preprocess.hpp
// GPU-resident detection pipeline using libtorch tensor ops.
// All preprocessing and postprocessing happens on GPU — only 2 CPU<->GPU
// transfers: ~921KB upload (raw uint8 pixels) and ~300B download (box coords).

#pragma once

#include "torch_session.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace infergo {

// ─── Detection result (plain C++ struct, lives on CPU) ──────────────────────

struct Detection {
    float x1, y1, x2, y2;
    int   class_id;
    float confidence;
};

// ─── GPU preprocessing functions ────────────────────────────────────────────
// All functions take and return torch::Tensor on the SAME device (GPU or CPU).

/// Upload raw RGB uint8 pixels to the given device as [H,W,3] uint8 tensor.
/// Only ~921KB for 640x480 (vs 4.9MB float32 after preprocessing).
/// The returned tensor owns its data (cloned from the raw pointer).
torch::Tensor torch_upload_image(const uint8_t* rgb_data, int H, int W,
                                 torch::Device device);

/// Letterbox resize on GPU: [H,W,3] uint8 -> [target_h, target_w, 3] uint8.
/// Preserves aspect ratio, pads with 114.
/// Uses torch::nn::functional::interpolate for GPU-accelerated resize.
torch::Tensor torch_letterbox_gpu(torch::Tensor src, int target_w, int target_h);

/// Normalize + layout change on GPU: [H,W,3] uint8 -> [1,3,H,W] float32.
/// Does: .to(kFloat32).div_(255).permute({2,0,1}).unsqueeze_(0)
/// All ops stay on GPU.
torch::Tensor torch_normalize_gpu(torch::Tensor hwc_uint8);

/// GPU NMS: parse YOLO [1,84,8400] output and return detections.
/// All filtering and IoU computation on GPU. Only copies final boxes to CPU.
/// The scale/pad parameters are used to rescale boxes from letterbox space
/// back to original image coordinates.
std::vector<Detection> torch_nms_gpu(torch::Tensor yolo_output,
                                     float conf_thresh, float iou_thresh,
                                     float scale, float pad_x, float pad_y,
                                     int orig_w, int orig_h);

/// End-to-end GPU detection: JPEG bytes -> detections.
/// Only 2 transfers: ~921KB upload (raw pixels), ~300B download (box coords).
/// JPEG decode happens on CPU via OpenCV; everything else is on GPU.
std::vector<Detection> torch_detect_gpu(
    TorchSession& sess,
    const uint8_t* jpeg_data, int nbytes,
    float conf_thresh, float iou_thresh);

/// Batch GPU detection: process N JPEG images in one forward pass.
/// Amortizes the C++/GPU overhead across N images.
/// JPEG decode is sequential (OpenCV isn't thread-safe), but GPU upload,
/// the forward pass, and NMS are batched.
/// Returns a vector of N detection vectors (one per input image).
std::vector<std::vector<Detection>> torch_detect_gpu_batch(
    TorchSession& sess,
    const uint8_t** jpeg_data_array, const int* nbytes_array, int batch_size,
    float conf_thresh, float iou_thresh);

/// Detect from raw RGB pixels — no JPEG encode/decode overhead.
/// ~30ms faster than torch_detect_gpu at 1080p.
std::vector<Detection> torch_detect_gpu_raw(
    TorchSession& sess,
    const uint8_t* rgb_data, int width, int height,
    float conf_thresh, float iou_thresh);

/// Detect from raw NV12/YUV420P frame — skips BOTH JPEG and RGB conversion.
/// The NV12→RGB conversion happens on GPU via libtorch tensor ops.
/// This is the fastest path for video pipeline: decode(GPU) → detect(GPU) → 300B out.
std::vector<Detection> torch_detect_gpu_yuv(
    TorchSession& sess,
    const uint8_t* yuv_data, int width, int height, int linesize,
    float conf_thresh, float iou_thresh);

} // namespace infergo

// ─── Zero-copy "into" variants ─────────────────────────────────────────────
// These write directly to a caller-supplied output buffer, avoiding the
// intermediate std::vector<Detection> heap allocation and the field-by-field
// copy in the C API wrapper. They also use error codes instead of exceptions
// on the hot path, saving ~0.3ms per call.

// InferBox is defined in infer_api.h at global scope (extern "C").
// Forward-declare it here so callers don't need to include infer_api.h.
struct InferBox;

namespace infergo {

/// GPU NMS writing directly to an InferBox output buffer.
/// Returns the number of detections written, or -1 on error.
/// On error, writes a message to error_buf (null-terminated).
int torch_nms_gpu_into(torch::Tensor yolo_output,
                       float conf_thresh, float iou_thresh,
                       float scale, float pad_x, float pad_y,
                       int orig_w, int orig_h,
                       ::InferBox* out_boxes, int max_boxes,
                       char* error_buf, int error_buf_size);

/// Detect from raw RGB pixels, writing directly to an InferBox output buffer.
/// Eliminates vector allocation and field-by-field copy overhead.
/// Returns detection count, or -1 on error (message in error_buf).
int torch_detect_gpu_raw_into(
    TorchSession& sess,
    const uint8_t* rgb_data, int width, int height,
    float conf_thresh, float iou_thresh,
    ::InferBox* out_boxes, int max_boxes,
    char* error_buf, int error_buf_size);

} // namespace infergo
