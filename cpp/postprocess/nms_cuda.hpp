// cpp/postprocess/nms_cuda.hpp
// GPU-side NMS using raw CUDA kernels (no libtorch dependency).
// Input: flat array of detections on GPU device memory.
// Output: indices of kept detections written to a host buffer.

#pragma once

#include "infer_api.h"  // InferBox, InferError

#ifdef __cplusplus
extern "C" {
#endif

// Run non-maximum suppression entirely on GPU using CUDA kernels.
//
// d_boxes:      Device pointer to N detections, each 6 floats:
//               [x1, y1, x2, y2, confidence, class_id_as_float].
//               Data must already be on GPU (no H2D copy performed).
// n_boxes:      Number of input detections.
// conf_thresh:  Minimum confidence to keep a detection.
// iou_thresh:   IoU threshold above which a box is suppressed (class-aware).
// out_boxes:    Host-allocated output buffer for kept detections (InferBox*).
// max_out:      Capacity of out_boxes.
// out_count:    Receives number of detections written to out_boxes.
// stream:       CUDA stream (0 for default stream).
//
// Returns INFER_OK on success, INFER_ERR_CUDA on kernel failure,
// INFER_ERR_NULL if any required pointer is NULL.
//
// The function performs:
//   1. Confidence threshold filtering (parallel scan)
//   2. Sort filtered detections by confidence descending (thrust::sort)
//   3. Compute NxN IoU matrix using GPU threads
//   4. Greedy suppression using the IoU matrix
//   5. Copy only the kept detections back to host
//
// For typical YOLO outputs (8400 raw -> ~100-500 after conf filter),
// the entire pipeline runs in <0.1ms on modern GPUs.
InferError infer_nms_cuda(const float* d_boxes, int n_boxes,
                          float conf_thresh, float iou_thresh,
                          InferBox* out_boxes, int max_out,
                          int* out_count, void* stream);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
namespace infergo {

// C++ wrapper matching the existing nms() signature but running on GPU.
// Takes a YOLO prediction tensor on GPU [1, num_det, 4+num_classes] and
// returns kept boxes. Falls back to CPU NMS if CUDA is not available.
//
// This is the drop-in replacement for torch_nms_gpu() that uses raw CUDA
// kernels instead of libtorch tensor ops + CPU greedy loop.
struct NmsCudaResult {
    InferBox* boxes;
    int       count;
};

// GPU NMS operating on pre-decoded boxes already on device.
// d_boxes: device memory, each detection is 6 floats [x1,y1,x2,y2,conf,class_id].
// Returns allocated host array of InferBox (caller must free with delete[]).
NmsCudaResult nms_cuda(const float* d_boxes, int n_boxes,
                       float conf_thresh, float iou_thresh,
                       int max_out);

} // namespace infergo
#endif
