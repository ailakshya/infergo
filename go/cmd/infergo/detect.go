package main

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	_ "image/png"
	"sort"
	"unsafe"

	"github.com/ailakshya/infergo/onnx"
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
	// 1. Decode image — try JPEG then PNG via stdlib image.Decode.
	src, _, err := image.Decode(bytes.NewReader(imageBytes))
	if err != nil {
		return nil, fmt.Errorf("detect: decode image: %w", err)
	}

	// 2. Letterbox resize to 640×640.
	const targetW, targetH = 640, 640
	lb, scale, padX, padY := letterbox(src, targetW, targetH)

	// 3. Allocate [1, 3, 640, 640] float32 tensor (NCHW).
	inputTensor, err := tensor.NewTensorCPU([]int{1, 3, targetH, targetW}, tensor.Float32)
	if err != nil {
		return nil, fmt.Errorf("detect: alloc input tensor: %w", err)
	}
	defer inputTensor.Free()

	// Fill NCHW layout: data[c*H*W + y*W + x] = pixel channel c at (x,y) / 255.
	nElem := 1 * 3 * targetH * targetW
	data := unsafe.Slice((*float32)(inputTensor.DataPtr()), nElem)
	chSize := targetH * targetW // pixels per channel

	for y := 0; y < targetH; y++ {
		for x := 0; x < targetW; x++ {
			r, g, b, _ := lb.At(x, y).RGBA()
			// RGBA returns 16-bit values (0–65535); shift to 8-bit and normalize.
			idx := y*targetW + x
			data[0*chSize+idx] = float32(r>>8) / 255.0
			data[1*chSize+idx] = float32(g>>8) / 255.0
			data[2*chSize+idx] = float32(b>>8) / 255.0
		}
	}

	// 4. Run ONNX session.
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
	origBounds := src.Bounds()
	for idx := range candidates {
		c := &candidates[idx]
		c.x1 = (c.x1 - padX) / scale
		c.y1 = (c.y1 - padY) / scale
		c.x2 = (c.x2 - padX) / scale
		c.y2 = (c.y2 - padY) / scale

		// Clamp to image bounds.
		c.x1 = clampF32(c.x1, 0, float32(origBounds.Max.X))
		c.y1 = clampF32(c.y1, 0, float32(origBounds.Max.Y))
		c.x2 = clampF32(c.x2, 0, float32(origBounds.Max.X))
		c.y2 = clampF32(c.y2, 0, float32(origBounds.Max.Y))
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

// ─── Image preprocessing ──────────────────────────────────────────────────────

// letterbox scales src to fit inside targetW×targetH while preserving aspect
// ratio, filling the remainder with grey (114, 114, 114).
// Returns the padded image, the scale factor, and the x/y padding offsets.
func letterbox(src image.Image, targetW, targetH int) (image.Image, float32, float32, float32) {
	bounds := src.Bounds()
	srcW := bounds.Max.X - bounds.Min.X
	srcH := bounds.Max.Y - bounds.Min.Y

	scaleW := float32(targetW) / float32(srcW)
	scaleH := float32(targetH) / float32(srcH)
	scale := scaleW
	if scaleH < scale {
		scale = scaleH
	}

	newW := int(float32(srcW) * scale)
	newH := int(float32(srcH) * scale)

	// Padding to center the resized image.
	padX := float32(targetW-newW) / 2.0
	padY := float32(targetH-newH) / 2.0

	// Create output image filled with grey (114, 114, 114).
	dst := image.NewRGBA(image.Rect(0, 0, targetW, targetH))
	grey := color.RGBA{R: 114, G: 114, B: 114, A: 255}
	draw.Draw(dst, dst.Bounds(), &image.Uniform{grey}, image.Point{}, draw.Src)

	// Nearest-neighbor resize: for each dst pixel in the resized region, sample src.
	iPadX := int(padX + 0.5)
	iPadY := int(padY + 0.5)

	for y := 0; y < newH; y++ {
		srcY := int(float32(y)/scale) + bounds.Min.Y
		if srcY >= bounds.Max.Y {
			srcY = bounds.Max.Y - 1
		}
		for x := 0; x < newW; x++ {
			srcX := int(float32(x)/scale) + bounds.Min.X
			if srcX >= bounds.Max.X {
				srcX = bounds.Max.X - 1
			}
			dst.Set(x+iPadX, y+iPadY, src.At(srcX, srcY))
		}
	}

	return dst, scale, float32(iPadX), float32(iPadY)
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
