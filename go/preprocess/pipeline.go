// Package preprocess provides Go wrappers for the infergo image preprocessing C API.
// It exposes DecodeImage, Letterbox, Normalize, and StackBatch — the four stages
// needed to prepare raw image bytes for ONNX inference (e.g. YOLOv8).
package preprocess

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
	"runtime"
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

func wrap(ptr C.InferTensor) *tensor.Tensor {
	t := tensor.WrapUnsafePtr(unsafe.Pointer(ptr))
	runtime.KeepAlive(ptr)
	return t
}

// DecodeImage decodes raw image bytes (JPEG/PNG/WebP/BMP) into a CPU tensor.
// Output shape: [H, W, 3], dtype float32, pixel values in [0, 255].
func DecodeImage(data []byte) (*tensor.Tensor, error) {
	if len(data) == 0 {
		return nil, errors.New("preprocess: DecodeImage: empty data")
	}
	ptr := C.infer_preprocess_decode_image(unsafe.Pointer(&data[0]), C.int(len(data)))
	if ptr == nil {
		return nil, fmt.Errorf("preprocess: DecodeImage failed: %w", lastError())
	}
	return wrap(ptr), nil
}

// Letterbox resizes a [H, W, 3] float32 tensor to exactly [targetH, targetW, 3]
// by scaling uniformly and padding with 114.0.
func Letterbox(src *tensor.Tensor, targetW, targetH int) (*tensor.Tensor, error) {
	if src == nil {
		return nil, errors.New("preprocess: Letterbox: nil tensor")
	}
	if targetW <= 0 || targetH <= 0 {
		return nil, fmt.Errorf("preprocess: Letterbox: invalid target dims %dx%d", targetW, targetH)
	}
	ptr := C.infer_preprocess_letterbox(
		C.InferTensor(src.UnsafePtr()),
		C.int(targetW),
		C.int(targetH),
	)
	if ptr == nil {
		return nil, fmt.Errorf("preprocess: Letterbox failed: %w", lastError())
	}
	runtime.KeepAlive(src)
	return wrap(ptr), nil
}

// Normalize converts a [H, W, 3] float32 HWC tensor to CHW layout.
// Each channel c: out[c,h,w] = (in[h,w,c] / scale - mean[c]) / std[c].
// mean and std must each have exactly 3 elements.
func Normalize(src *tensor.Tensor, scale float32, mean, std [3]float32) (*tensor.Tensor, error) {
	if src == nil {
		return nil, errors.New("preprocess: Normalize: nil tensor")
	}
	if scale <= 0 {
		return nil, errors.New("preprocess: Normalize: scale must be positive")
	}
	ptr := C.infer_preprocess_normalize(
		C.InferTensor(src.UnsafePtr()),
		C.float(scale),
		(*C.float)(unsafe.Pointer(&mean[0])),
		(*C.float)(unsafe.Pointer(&std[0])),
	)
	if ptr == nil {
		return nil, fmt.Errorf("preprocess: Normalize failed: %w", lastError())
	}
	runtime.KeepAlive(src)
	return wrap(ptr), nil
}

// StackBatch stacks n [C, H, W] float32 tensors into a [N, C, H, W] batch tensor.
// All tensors must have identical shape and dtype.
func StackBatch(tensors []*tensor.Tensor) (*tensor.Tensor, error) {
	if len(tensors) == 0 {
		return nil, errors.New("preprocess: StackBatch: empty tensor slice")
	}
	for i, t := range tensors {
		if t == nil {
			return nil, fmt.Errorf("preprocess: StackBatch: tensor[%d] is nil", i)
		}
	}
	ptrs := make([]C.InferTensor, len(tensors))
	for i, t := range tensors {
		ptrs[i] = C.InferTensor(t.UnsafePtr())
	}
	ptr := C.infer_preprocess_stack_batch(
		(*C.InferTensor)(unsafe.Pointer(&ptrs[0])),
		C.int(len(ptrs)),
	)
	if ptr == nil {
		return nil, fmt.Errorf("preprocess: StackBatch failed: %w", lastError())
	}
	runtime.KeepAlive(tensors)
	return wrap(ptr), nil
}
