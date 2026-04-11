package main

import (
	"context"
	"fmt"
	"sort"
	"unsafe"

	"github.com/ailakshya/infergo/onnx"
	"github.com/ailakshya/infergo/preprocess"
	"github.com/ailakshya/infergo/server"
	"github.com/ailakshya/infergo/tensor"
)

// cocoClasses holds the 80 standard COCO class names.
// Index i corresponds to ClassID i in DetectedObject.
var cocoClasses = [80]string{
	"person", "bicycle", "car", "motorcycle", "airplane",
	"bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird",
	"cat", "dog", "horse", "sheep", "cow",
	"elephant", "bear", "zebra", "giraffe", "backpack",
	"umbrella", "handbag", "tie", "suitcase", "frisbee",
	"skis", "snowboard", "sports ball", "kite", "baseball bat",
	"baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
	"wine glass", "cup", "fork", "knife", "spoon",
	"bowl", "banana", "apple", "sandwich", "orange",
	"broccoli", "carrot", "hot dog", "pizza", "donut",
	"cake", "chair", "couch", "potted plant", "bed",
	"dining table", "toilet", "tv", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven",
	"toaster", "sink", "refrigerator", "book", "clock",
	"vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}

// detectionAdapter wraps an ONNX session for YOLOv8 object detection.
// It implements server.DetectionModel.
type detectionAdapter struct {
	sess *onnx.Session
}

// Close releases the ONNX session.
// Implements server.Model.
func (a *detectionAdapter) Close() {
	a.sess.Close()
}

// Detect runs the full YOLOv8 detection pipeline on raw image bytes:
//  1. Decode image (JPEG or PNG)
//  2. Letterbox resize to 640×640 preserving aspect ratio
//  3. Build [1,3,640,640] float32 NCHW tensor, normalize to [0,1]
//  4. Run ONNX session → [1, 84, 8400] output
//  5. Parse anchors: filter by confThresh, decode boxes
//  6. Rescale boxes to original image coordinates
//  7. Non-maximum suppression with iouThresh
//
// Implements server.DetectionModel.
func (a *detectionAdapter) Detect(ctx context.Context, imageBytes []byte, confThresh, iouThresh float32) ([]server.DetectedObject, error) {
	const targetW, targetH = 640, 640

	// 1. Decode image via C/OpenCV (SIMD-accelerated, ~1.5ms vs 6ms Go stdlib).
	decoded, err := preprocess.DecodeImage(imageBytes)
	if err != nil {
		return nil, fmt.Errorf("detect: decode image: %w", err)
	}
	defer decoded.Free()

	// 2. Letterbox resize via C/OpenCV (~0.1ms vs 6.5ms Go pixel loop).
	lb, err := preprocess.Letterbox(decoded, targetW, targetH)
	if err != nil {
		return nil, fmt.Errorf("detect: letterbox: %w", err)
	}
	defer lb.Free()

	// 3. Normalize HWC→CHW and scale to [0,1] via C (~0.3ms vs 1.2ms Go loop).
	//    normalize: out[c,h,w] = (in[h,w,c] / scale - mean[c]) / std[c]
	//    With scale=255, mean=[0,0,0], std=[1,1,1] → out = in / 255.
	norm, err := preprocess.Normalize(lb, 255.0, [3]float32{0, 0, 0}, [3]float32{1, 1, 1})
	if err != nil {
		return nil, fmt.Errorf("detect: normalize: %w", err)
	}
	defer norm.Free()

	// 4. Stack to [1, 3, 640, 640] batch.
	inputTensor, err := preprocess.StackBatch([]*tensor.Tensor{norm})
	if err != nil {
		return nil, fmt.Errorf("detect: stack batch: %w", err)
	}
	defer inputTensor.Free()

	// 5. Run ONNX session.
	outputs, err := a.sess.Run([]*tensor.Tensor{inputTensor})
	if err != nil {
		return nil, fmt.Errorf("detect: onnx run: %w", err)
	}
	defer func() {
		for _, o := range outputs {
			o.Free()
		}
	}()
	if len(outputs) == 0 {
		return nil, fmt.Errorf("detect: no outputs from model")
	}

	// 5. Parse raw YOLOv8 output: shape [1, 84, 8400].
	// Layout: output[feat][anchor], where feat 0..3 = cx,cy,w,h and 4..83 = class scores.
	out := outputs[0]
	outShape := out.Shape()
	if len(outShape) != 3 || outShape[0] != 1 || outShape[1] != 84 {
		return nil, fmt.Errorf("detect: unexpected output shape %v (want [1, 84, 8400])", outShape)
	}
	nAnchors := outShape[2] // typically 8400

	rawData := unsafe.Slice((*float32)(out.DataPtr()), out.NElements())
	// rawData layout: [feat0_anch0, feat0_anch1, ..., feat0_anchN, feat1_anch0, ...]
	// i.e., row-major [84][nAnchors]

	type candidate struct {
		x1, y1, x2, y2 float32
		classID         int
		confidence      float32
	}

	var candidates []candidate

	for i := 0; i < nAnchors; i++ {
		cx := rawData[0*nAnchors+i]
		cy := rawData[1*nAnchors+i]
		w := rawData[2*nAnchors+i]
		h := rawData[3*nAnchors+i]

		// Find argmax over 80 class scores.
		bestClass := 0
		bestScore := rawData[4*nAnchors+i]
		for c := 1; c < 80; c++ {
			score := rawData[(4+c)*nAnchors+i]
			if score > bestScore {
				bestScore = score
				bestClass = c
			}
		}

		if bestScore < confThresh {
			continue
		}

		// Decode box to xyxy in 640×640 space.
		x1 := cx - w/2
		y1 := cy - h/2
		x2 := cx + w/2
		y2 := cy + h/2

		candidates = append(candidates, candidate{
			x1:         x1,
			y1:         y1,
			x2:         x2,
			y2:         y2,
			classID:    bestClass,
			confidence: bestScore,
		})
	}

	// 6. Rescale boxes from padded 640×640 space back to original image coordinates.
	// Compute letterbox scale and padding from the original decoded image dimensions.
	decodedShape := decoded.Shape() // [H, W, 3]
	origH, origW := float32(decodedShape[0]), float32(decodedShape[1])
	scale := minF32(float32(targetW)/origW, float32(targetH)/origH)
	padX := (float32(targetW) - origW*scale) / 2.0
	padY := (float32(targetH) - origH*scale) / 2.0

	for idx := range candidates {
		c := &candidates[idx]
		c.x1 = (c.x1 - padX) / scale
		c.y1 = (c.y1 - padY) / scale
		c.x2 = (c.x2 - padX) / scale
		c.y2 = (c.y2 - padY) / scale

		// Clamp to image bounds.
		c.x1 = clampF32(c.x1, 0, origW)
		c.y1 = clampF32(c.y1, 0, origH)
		c.x2 = clampF32(c.x2, 0, origW)
		c.y2 = clampF32(c.y2, 0, origH)
	}

	// 7. Non-maximum suppression — sort by confidence descending, suppress IoU > iouThresh within same class.
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].confidence > candidates[j].confidence
	})

	suppressed := make([]bool, len(candidates))
	var results []server.DetectedObject

	for i := 0; i < len(candidates); i++ {
		if suppressed[i] {
			continue
		}
		a := candidates[i]
		results = append(results, server.DetectedObject{
			X1:         a.x1,
			Y1:         a.y1,
			X2:         a.x2,
			Y2:         a.y2,
			ClassID:    a.classID,
			Confidence: a.confidence,
		})
		for j := i + 1; j < len(candidates); j++ {
			if suppressed[j] {
				continue
			}
			b := candidates[j]
			if a.classID == b.classID && iou(a.x1, a.y1, a.x2, a.y2, b.x1, b.y1, b.x2, b.y2) > iouThresh {
				suppressed[j] = true
			}
		}
	}

	return results, nil
}

// ─── NMS helpers ─────────────────────────────────────────────────────────────

// iou computes intersection-over-union for two axis-aligned boxes.
func iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2 float32) float32 {
	// Intersection.
	ix1 := maxF32(ax1, bx1)
	iy1 := maxF32(ay1, by1)
	ix2 := minF32(ax2, bx2)
	iy2 := minF32(ay2, by2)

	iw := ix2 - ix1
	ih := iy2 - iy1
	if iw <= 0 || ih <= 0 {
		return 0
	}
	intersection := iw * ih

	// Union.
	areaA := (ax2 - ax1) * (ay2 - ay1)
	areaB := (bx2 - bx1) * (by2 - by1)
	union := areaA + areaB - intersection
	if union <= 0 {
		return 0
	}
	return intersection / union
}

// ─── float32 helpers ─────────────────────────────────────────────────────────

func maxF32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func minF32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func clampF32(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
