// Package onnx provides a Go wrapper for the infergo ONNX session C API.
// It lets you load ONNX models and run inference using CPU, CUDA, TensorRT,
// or other execution providers, falling back to CPU when unavailable.
package onnx

/*
#cgo CFLAGS: -I${SRCDIR}/../../cpp/include
#cgo LDFLAGS: -L${SRCDIR}/../../build/cpp/api -linfer_api -L${SRCDIR}/../../build/cpp/onnx -linfer_onnx -L${SRCDIR}/../../build/cpp/tensor -linfer_tensor -Wl,-rpath,${SRCDIR}/../../build/cpp/api -Wl,-rpath,${SRCDIR}/../../build/cpp/onnx -Wl,-rpath,${SRCDIR}/../../build/cpp/tensor

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

// Session wraps an InferSession (opaque C pointer).
// Always call Close when done to release C resources.
type Session struct {
	ptr C.InferSession
}

// lastError returns the thread-local C error string as a Go error.
func lastError() error {
	msg := C.infer_last_error_string()
	if msg == nil || C.GoString(msg) == "" {
		return errors.New("unknown C error")
	}
	return errors.New(C.GoString(msg))
}

// NewSession creates an ONNX inference session for the given execution provider.
// provider: "cpu" | "cuda" | "tensorrt" | "coreml" | "openvino"
// Falls back to CPU if the requested provider is unavailable.
// deviceID is the GPU device index (ignored for CPU).
func NewSession(provider string, deviceID int) (*Session, error) {
	cProvider := C.CString(provider)
	defer C.free(unsafe.Pointer(cProvider))

	ptr := C.infer_session_create(cProvider, C.int(deviceID))
	if ptr == nil {
		return nil, fmt.Errorf("onnx: session_create failed: %w", lastError())
	}
	s := &Session{ptr: ptr}
	runtime.SetFinalizer(s, (*Session).Close)
	return s, nil
}

// Close destroys the session and frees all C resources.
// Safe to call multiple times.
func (s *Session) Close() {
	if s.ptr == nil {
		return
	}
	C.infer_session_destroy(s.ptr)
	s.ptr = nil
	runtime.SetFinalizer(s, nil)
}

// Load reads an ONNX model file from disk and prepares it for inference.
// Must be called before Run.
func (s *Session) Load(modelPath string) error {
	if s.ptr == nil {
		return errors.New("onnx: Load called on closed session")
	}
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	rc := C.infer_session_load(s.ptr, cPath)
	if rc != C.INFER_OK {
		return fmt.Errorf("onnx: Load failed: %w", lastError())
	}
	return nil
}

// NumInputs returns the number of model inputs (valid after Load).
func (s *Session) NumInputs() int {
	if s.ptr == nil {
		return 0
	}
	return int(C.infer_session_num_inputs(s.ptr))
}

// NumOutputs returns the number of model outputs (valid after Load).
func (s *Session) NumOutputs() int {
	if s.ptr == nil {
		return 0
	}
	return int(C.infer_session_num_outputs(s.ptr))
}

// InputName returns the name of the input at index idx.
func (s *Session) InputName(idx int) (string, error) {
	if s.ptr == nil {
		return "", errors.New("onnx: InputName called on closed session")
	}
	var buf [256]C.char
	rc := C.infer_session_input_name(s.ptr, C.int(idx), &buf[0], C.int(len(buf)))
	if rc != C.INFER_OK {
		return "", fmt.Errorf("onnx: InputName(%d) failed: %w", idx, lastError())
	}
	return C.GoString(&buf[0]), nil
}

// OutputName returns the name of the output at index idx.
func (s *Session) OutputName(idx int) (string, error) {
	if s.ptr == nil {
		return "", errors.New("onnx: OutputName called on closed session")
	}
	var buf [256]C.char
	rc := C.infer_session_output_name(s.ptr, C.int(idx), &buf[0], C.int(len(buf)))
	if rc != C.INFER_OK {
		return "", fmt.Errorf("onnx: OutputName(%d) failed: %w", idx, lastError())
	}
	return C.GoString(&buf[0]), nil
}

// Run executes inference with the given input tensors and returns output tensors.
// inputs must match the model's expected number of inputs in order.
// Each returned output tensor is heap-allocated and owned by the caller — call Free() on each.
func (s *Session) Run(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if s.ptr == nil {
		return nil, errors.New("onnx: Run called on closed session")
	}

	nIn := len(inputs)
	nOut := s.NumOutputs()
	if nOut == 0 {
		return nil, errors.New("onnx: Run called before Load (NumOutputs == 0)")
	}

	// Build C array of InferTensor (void*) for inputs — RULE 12
	cInputs := make([]C.InferTensor, nIn)
	for i, t := range inputs {
		if t == nil {
			return nil, fmt.Errorf("onnx: inputs[%d] is nil", i)
		}
		// Access the underlying C pointer via DataPtr trick: we need the
		// InferTensor handle (the Tensor* pointer itself), not the data pointer.
		// Export it through the tensor package's UnsafePtr helper.
		cInputs[i] = C.InferTensor(t.UnsafePtr())
	}

	// Allocate output array
	cOutputs := make([]C.InferTensor, nOut)

	var inPtr *C.InferTensor
	if nIn > 0 {
		inPtr = &cInputs[0]
	}

	rc := C.infer_session_run(
		s.ptr,
		inPtr, C.int(nIn),
		&cOutputs[0], C.int(nOut),
	)
	if rc != C.INFER_OK {
		return nil, fmt.Errorf("onnx: Run failed: %w", lastError())
	}

	// Wrap output C tensors in Go Tensor wrappers
	out := make([]*tensor.Tensor, nOut)
	for i, ct := range cOutputs {
		if ct == nil {
			return nil, fmt.Errorf("onnx: output[%d] is nil after run", i)
		}
		out[i] = tensor.WrapUnsafePtr(unsafe.Pointer(ct))
	}
	return out, nil
}
