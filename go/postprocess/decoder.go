// Package postprocess provides Go wrappers for the infergo postprocessing C API.
// It exposes Classify (softmax top-k), NMS (YOLO bounding-box NMS), and
// NormalizeEmbedding (in-place L2 normalization).
package postprocess

/*
#cgo CFLAGS: -I${SRCDIR}/../../cpp/include
#cgo LDFLAGS: -L${SRCDIR}/../../build/cpp/api -linfer_api -Wl,-rpath,${SRCDIR}/../../build/cpp/api

#include "infer_api.h"
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/ailakshya/infergo/tensor"
)

func lastError() error {
	msg := C.infer_last_error_string()
	if msg == nil || C.GoString(msg) == "" {
		return errors.New("unknown C error")
	}
	return errors.New(C.GoString(msg))
}

// ClassResult holds the label index and confidence for one top-k class.
type ClassResult struct {
	ClassID    int
	Confidence float32
}

// Box is an axis-aligned bounding box in absolute pixel coordinates.
type Box struct {
	X1, Y1     float32
	X2, Y2     float32
	ClassID    int
	Confidence float32
}

// Classify runs a softmax over the logits tensor and returns the top-k results
// sorted by confidence descending.
// t must be a 1-D float32 CPU tensor.
func Classify(t *tensor.Tensor, topK int) ([]ClassResult, error) {
	if t == nil {
		return nil, errors.New("postprocess: Classify: nil tensor")
	}
	if topK <= 0 {
		return nil, errors.New("postprocess: Classify: topK must be positive")
	}
	buf := make([]C.InferClassResult, topK)
	n := C.infer_postprocess_classify(
		C.InferTensor(t.UnsafePtr()),
		C.int(topK),
		(*C.InferClassResult)(unsafe.Pointer(&buf[0])),
	)
	if n < 0 {
		return nil, fmt.Errorf("postprocess: Classify failed: %w", lastError())
	}
	out := make([]ClassResult, int(n))
	for i := range out {
		out[i] = ClassResult{
			ClassID:    int(buf[i].label_idx),
			Confidence: float32(buf[i].confidence),
		}
	}
	return out, nil
}

// NMS runs non-maximum suppression on a YOLO predictions tensor of shape
// [1, num_detections, 4+num_classes] and returns kept boxes sorted by
// confidence descending.
// confThresh: minimum class score to keep a detection.
// iouThresh:  IoU threshold above which a box is suppressed.
// maxBoxes:   upper bound on the number of returned boxes.
func NMS(predictions *tensor.Tensor, confThresh, iouThresh float32, maxBoxes int) ([]Box, error) {
	if predictions == nil {
		return nil, errors.New("postprocess: NMS: nil tensor")
	}
	if maxBoxes <= 0 {
		return nil, errors.New("postprocess: NMS: maxBoxes must be positive")
	}
	buf := make([]C.InferBox, maxBoxes)
	n := C.infer_postprocess_nms(
		C.InferTensor(predictions.UnsafePtr()),
		C.float(confThresh),
		C.float(iouThresh),
		(*C.InferBox)(unsafe.Pointer(&buf[0])),
		C.int(maxBoxes),
	)
	if n < 0 {
		return nil, fmt.Errorf("postprocess: NMS failed: %w", lastError())
	}
	out := make([]Box, int(n))
	for i := range out {
		out[i] = Box{
			X1:         float32(buf[i].x1),
			Y1:         float32(buf[i].y1),
			X2:         float32(buf[i].x2),
			Y2:         float32(buf[i].y2),
			ClassID:    int(buf[i].class_idx),
			Confidence: float32(buf[i].confidence),
		}
	}
	return out, nil
}

// NormalizeEmbedding L2-normalizes a float32 tensor in-place.
// Each element is divided by sqrt(sum of squares). No-op on a zero vector.
func NormalizeEmbedding(t *tensor.Tensor) error {
	if t == nil {
		return errors.New("postprocess: NormalizeEmbedding: nil tensor")
	}
	rc := C.infer_postprocess_normalize_embedding(C.InferTensor(t.UnsafePtr()))
	if rc != C.INFER_OK {
		return fmt.Errorf("postprocess: NormalizeEmbedding failed: %w", lastError())
	}
	return nil
}
