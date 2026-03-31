// Package tensor provides a Go wrapper for the infergo tensor C API.
// Tensors may live on CPU heap or on a CUDA device. All memory is
// owned by the C layer; Go wrapper structs hold an opaque pointer and
// register a finalizer so that leaked wrappers are eventually freed.
package tensor

/*
#cgo CFLAGS: -I${SRCDIR}/../../cpp/include
#cgo LDFLAGS: -L${SRCDIR}/../../build/cpp/api -linfer_api -L${SRCDIR}/../../build/cpp/tensor -linfer_tensor -Wl,-rpath,${SRCDIR}/../../build/cpp/api -Wl,-rpath,${SRCDIR}/../../build/cpp/tensor

#include "infer_api.h"
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// DType is the element type of a tensor.
type DType int

const (
	Float32  DType = C.INFER_DTYPE_FLOAT32
	Float16  DType = C.INFER_DTYPE_FLOAT16
	BFloat16 DType = C.INFER_DTYPE_BFLOAT16
	Int32    DType = C.INFER_DTYPE_INT32
	Int64    DType = C.INFER_DTYPE_INT64
	UInt8    DType = C.INFER_DTYPE_UINT8
	Bool     DType = C.INFER_DTYPE_BOOL
)

// Tensor wraps an InferTensor (opaque C pointer). It is safe to copy
// by value, but only one copy should call Free. Use NewTensorCPU or
// NewTensorCUDA to create tensors; always call Free when done.
type Tensor struct {
	ptr C.InferTensor // nil means freed or never allocated
}

// lastError returns the thread-local C error string as a Go error.
func lastError() error {
	msg := C.infer_last_error_string()
	if msg == nil || C.GoString(msg) == "" {
		return errors.New("unknown C error")
	}
	return errors.New(C.GoString(msg))
}

// NewTensorCPU allocates a tensor on the CPU heap with the given shape and dtype.
// Returns an error if allocation fails (null shape, bad ndim/dtype, or OOM).
func NewTensorCPU(shape []int, dtype DType) (*Tensor, error) {
	if len(shape) == 0 {
		return nil, errors.New("tensor: shape must have at least one dimension")
	}
	cShape := make([]C.int, len(shape))
	for i, d := range shape {
		if d <= 0 {
			return nil, fmt.Errorf("tensor: shape[%d]=%d is not positive", i, d)
		}
		cShape[i] = C.int(d)
	}
	ptr := C.infer_tensor_alloc_cpu(
		(*C.int)(unsafe.Pointer(&cShape[0])),
		C.int(len(shape)),
		C.int(dtype),
	)
	if ptr == nil {
		return nil, fmt.Errorf("tensor: alloc_cpu failed: %w", lastError())
	}
	t := &Tensor{ptr: ptr}
	runtime.SetFinalizer(t, (*Tensor).Free)
	return t, nil
}

// NewTensorCUDA allocates a tensor on the specified CUDA device.
// deviceID is the CUDA device index (0 = first GPU).
// Returns an error if CUDA is unavailable or allocation fails.
func NewTensorCUDA(shape []int, dtype DType, deviceID int) (*Tensor, error) {
	if len(shape) == 0 {
		return nil, errors.New("tensor: shape must have at least one dimension")
	}
	cShape := make([]C.int, len(shape))
	for i, d := range shape {
		if d <= 0 {
			return nil, fmt.Errorf("tensor: shape[%d]=%d is not positive", i, d)
		}
		cShape[i] = C.int(d)
	}
	ptr := C.infer_tensor_alloc_cuda(
		(*C.int)(unsafe.Pointer(&cShape[0])),
		C.int(len(shape)),
		C.int(dtype),
		C.int(deviceID),
	)
	if ptr == nil {
		return nil, fmt.Errorf("tensor: alloc_cuda failed: %w", lastError())
	}
	t := &Tensor{ptr: ptr}
	runtime.SetFinalizer(t, (*Tensor).Free)
	return t, nil
}

// Free releases the tensor's memory. Safe to call multiple times.
// After Free, the Tensor is invalid and must not be used.
func (t *Tensor) Free() {
	if t.ptr == nil {
		return
	}
	C.infer_tensor_free(t.ptr)
	t.ptr = nil
	runtime.SetFinalizer(t, nil)
}

// Shape returns a copy of the tensor's shape as a []int.
func (t *Tensor) Shape() []int {
	if t.ptr == nil {
		return nil
	}
	// First call to get ndim.
	ndim := int(C.infer_tensor_shape(t.ptr, nil, 0))
	if ndim <= 0 {
		// ndim=0 with nil buf returns 0 for null-guard; use a large buf.
		var buf [64]C.int
		ndim = int(C.infer_tensor_shape(t.ptr, &buf[0], 64))
		if ndim <= 0 {
			return nil
		}
		out := make([]int, ndim)
		for i := 0; i < ndim; i++ {
			out[i] = int(buf[i])
		}
		return out
	}
	buf := make([]C.int, ndim)
	C.infer_tensor_shape(t.ptr, &buf[0], C.int(ndim))
	out := make([]int, ndim)
	for i, v := range buf {
		out[i] = int(v)
	}
	return out
}

// DType returns the tensor's element type.
func (t *Tensor) DType() DType {
	if t.ptr == nil {
		return -1
	}
	return DType(C.infer_tensor_dtype(t.ptr))
}

// NBytes returns the total byte size of the tensor's data buffer.
func (t *Tensor) NBytes() int {
	if t.ptr == nil {
		return 0
	}
	return int(C.infer_tensor_nbytes(t.ptr))
}

// NElements returns the total number of elements in the tensor.
func (t *Tensor) NElements() int {
	if t.ptr == nil {
		return 0
	}
	return int(C.infer_tensor_nelements(t.ptr))
}

// DataPtr returns an unsafe.Pointer to the raw data buffer.
// For CPU tensors this pointer is safe to read/write from Go.
// For CUDA tensors this is a device pointer — do NOT dereference from Go.
// Returns nil if the tensor is freed or has no data.
func (t *Tensor) DataPtr() unsafe.Pointer {
	if t.ptr == nil {
		return nil
	}
	return C.infer_tensor_data_ptr(t.ptr)
}

// CopyFrom copies len(src)*elemSize bytes from src into the tensor's CPU buffer.
// src must be a Go slice whose total byte size equals t.NBytes().
// Only valid for CPU tensors — call ToHost first if the tensor is on a device.
//
// Example:
//
//	data := []float32{1, 2, 3, 4}
//	err := t.CopyFrom(unsafe.Pointer(&data[0]), len(data)*4)
func (t *Tensor) CopyFrom(src unsafe.Pointer, nbytes int) error {
	if t.ptr == nil {
		return errors.New("tensor: CopyFrom called on freed tensor")
	}
	if src == nil {
		return errors.New("tensor: CopyFrom: src is nil")
	}
	rc := C.infer_tensor_copy_from(t.ptr, src, C.int(nbytes))
	if rc != C.INFER_OK {
		return fmt.Errorf("tensor: CopyFrom failed: %w", lastError())
	}
	return nil
}

// ToDevice copies the tensor data to the specified CUDA device.
// No-op if already on that device.
func (t *Tensor) ToDevice(deviceID int) error {
	if t.ptr == nil {
		return errors.New("tensor: ToDevice called on freed tensor")
	}
	rc := C.infer_tensor_to_device(t.ptr, C.int(deviceID))
	if rc != C.INFER_OK {
		return fmt.Errorf("tensor: ToDevice failed: %w", lastError())
	}
	return nil
}

// ToHost copies the tensor data back to CPU. No-op if already on host.
func (t *Tensor) ToHost() error {
	if t.ptr == nil {
		return errors.New("tensor: ToHost called on freed tensor")
	}
	rc := C.infer_tensor_to_host(t.ptr)
	if rc != C.INFER_OK {
		return fmt.Errorf("tensor: ToHost failed: %w", lastError())
	}
	return nil
}

// UnsafePtr returns the raw C InferTensor pointer as an unsafe.Pointer.
// Used by sibling packages (e.g. onnx) to pass tensors across the C API.
// Do not use outside of the infergo module.
func (t *Tensor) UnsafePtr() unsafe.Pointer {
	return unsafe.Pointer(t.ptr)
}

// WrapUnsafePtr wraps a raw C InferTensor pointer in a Go Tensor.
// The caller is responsible for ensuring ptr is a valid InferTensor.
// Used by sibling packages to wrap tensors returned from C API calls.
// Do not use outside of the infergo module.
func WrapUnsafePtr(ptr unsafe.Pointer) *Tensor {
	t := &Tensor{ptr: C.InferTensor(ptr)}
	runtime.SetFinalizer(t, (*Tensor).Free)
	return t
}
