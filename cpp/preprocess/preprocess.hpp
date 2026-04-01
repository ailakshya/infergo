#pragma once

// cpp/preprocess/preprocess.hpp
// Internal C++ API for the preprocessing pipeline.
// None of these types cross the C boundary — they are wrapped in api.cpp.

#include "tensor.hpp"     // infergo::Tensor
#include "infer_api.h"    // INFER_DTYPE_* constants

#include <cstdint>
#include <vector>

namespace infergo {

/// Decode raw image bytes (JPEG / PNG / WebP / BMP) into a CPU tensor.
/// Output shape: [H, W, 3], dtype float32, values in [0, 255].
/// Throws std::runtime_error on decode failure.
Tensor* decode_image(const uint8_t* data, int nbytes);

/// Resize and pad an image tensor ([H, W, 3] float32) to [target_h, target_w, 3]
/// using letterboxing (maintains aspect ratio, pads with 114).
/// Throws std::runtime_error on invalid input.
Tensor* letterbox(const Tensor* src, int target_w, int target_h);

/// Normalize a [H, W, 3] or [C, H, W] float32 tensor in-place:
///   1. Converts HWC → CHW (if input is HWC).
///   2. Divides by scale (typically 255.0).
///   3. Subtracts mean[c], divides by std[c] per channel.
/// mean and std must each have exactly 3 elements.
/// Returns a new [C, H, W] float32 tensor.
/// Throws std::runtime_error on invalid input.
Tensor* normalize(const Tensor* src,
                  float scale,
                  const float mean[3],
                  const float std[3]);

/// Stack N tensors of identical shape [C, H, W] into a [N, C, H, W] batch tensor.
/// Throws std::runtime_error if shapes differ or tensors is empty.
Tensor* stack_batch(const std::vector<const Tensor*>& tensors);

} // namespace infergo
