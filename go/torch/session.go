//go:build cgo

// Package torch provides a Go wrapper for the infergo PyTorch/libtorch session C API.
// It lets you load TorchScript (.pt) models and run inference using CPU or CUDA,
// falling back to CPU when CUDA is unavailable.
package torch

/*
#cgo CFLAGS: -I${SRCDIR}/../../cpp/include
#cgo LDFLAGS: -L${SRCDIR}/../../build/cpp/api -linfer_api -Wl,-rpath,${SRCDIR}/../../build/cpp/api

#include "infer_api.h"
#include <stdlib.h>

// CGo wrapper to avoid typedef void* type mismatch with InferTorchSession.
static inline int torch_run_gpu_wrap(void* s, void* in, int nin, void* out, int nout) {
    return infer_torch_session_run_gpu((InferTorchSession)s, (InferTensor*)in, nin, (InferTensor*)out, nout);
}

// CGo wrapper for GPU-accelerated detection pipeline.
static inline int torch_detect_gpu_wrap(void* s, const void* data, int nbytes,
    float conf, float iou, void* boxes, int max_boxes) {
    return infer_torch_detect_gpu((InferTorchSession)s, data, nbytes, conf, iou, (InferBox*)boxes, max_boxes);
}

// CGo wrapper for raw RGB detection — no JPEG encode/decode overhead.
static inline int torch_detect_gpu_raw_wrap(void* s, const void* rgb,
    int w, int h, float conf, float iou, void* boxes, int max_boxes) {
    return infer_torch_detect_gpu_raw((InferTorchSession)s, rgb, w, h, conf, iou, (InferBox*)boxes, max_boxes);
}

// CGo wrapper for YUV/NV12 detection — zero CPU color conversion.
static inline int torch_detect_gpu_yuv_wrap(void* s, const void* yuv,
    int w, int h, int linesize, float conf, float iou, void* boxes, int max_boxes) {
    return infer_torch_detect_gpu_yuv((InferTorchSession)s, yuv, w, h, linesize, conf, iou, (InferBox*)boxes, max_boxes);
}

// CGo wrapper for batch GPU detection pipeline.
// Takes void* for pointer-to-pointer args to avoid CGo typedef issues.
static inline int torch_detect_gpu_batch_wrap(void* s,
    void* jpeg_data_array, const int* nbytes_array, int batch_size,
    float conf, float iou,
    void* out_boxes_array, int* out_counts, int max_boxes_per_image) {
    return infer_torch_detect_gpu_batch(
        (InferTorchSession)s,
        (const void**)jpeg_data_array, nbytes_array, batch_size,
        conf, iou,
        (InferBox**)out_boxes_array, out_counts, max_boxes_per_image);
}
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/ailakshya/infergo/tensor"
)

// Session wraps an InferTorchSession (opaque C pointer).
// We store the handle as unsafe.Pointer because Go 1.22 CGo resolves the
// typedef void* to unsafe.Pointer in generated function wrappers.
type Session struct {
	ptr unsafe.Pointer
}

func lastError() error {
	msg := C.infer_last_error_string()
	if msg == nil || C.GoString(msg) == "" {
		return errors.New("unknown C error")
	}
	return errors.New(C.GoString(msg))
}

// NewSession creates a PyTorch inference session.
// provider: "cpu" | "cuda". Falls back to CPU if CUDA is unavailable.
func NewSession(provider string, deviceID int) (*Session, error) {
	cProvider := C.CString(provider)
	defer C.free(unsafe.Pointer(cProvider))

	ptr := C.infer_torch_session_create(cProvider, C.int(deviceID))
	if ptr == nil {
		return nil, fmt.Errorf("torch: session_create failed: %w", lastError())
	}
	s := &Session{ptr: unsafe.Pointer(ptr)}
	runtime.SetFinalizer(s, (*Session).Close)
	return s, nil
}

// Ptr returns the raw C session handle for use with pipeline functions
// like infer_pipeline_detect_frame. The caller must not free the pointer.
func (s *Session) Ptr() unsafe.Pointer { return s.ptr }

// Close destroys the session. Safe to call multiple times.
func (s *Session) Close() {
	if s.ptr == nil {
		return
	}
	C.infer_torch_session_destroy(s.ptr)
	s.ptr = nil
	runtime.SetFinalizer(s, nil)
}

// Load reads a TorchScript model file and prepares it for inference.
func (s *Session) Load(modelPath string) error {
	if s.ptr == nil {
		return errors.New("torch: Load called on closed session")
	}
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	rc := C.infer_torch_session_load(s.ptr, cPath)
	if rc != C.INFER_OK {
		return fmt.Errorf("torch: Load failed: %w", lastError())
	}
	return nil
}

// NumInputs returns the number of model inputs (valid after Load).
func (s *Session) NumInputs() int {
	if s.ptr == nil {
		return 0
	}
	return int(C.infer_torch_session_num_inputs(s.ptr))
}

// NumOutputs returns the number of model outputs (valid after Load).
func (s *Session) NumOutputs() int {
	if s.ptr == nil {
		return 0
	}
	return int(C.infer_torch_session_num_outputs(s.ptr))
}

// Run executes inference. Each returned tensor is owned by the caller (call Free).
func (s *Session) Run(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if s.ptr == nil {
		return nil, errors.New("torch: Run called on closed session")
	}

	nIn := len(inputs)
	nOut := s.NumOutputs()
	if nOut == 0 {
		return nil, errors.New("torch: Run called before Load (NumOutputs == 0)")
	}

	// Build C array of void* for inputs. CGo resolves InferTensor (typedef void*)
	// to unsafe.Pointer in the generated wrappers, so we use unsafe.Pointer directly.
	cInputs := make([]unsafe.Pointer, nIn)
	for i, t := range inputs {
		if t == nil {
			return nil, fmt.Errorf("torch: inputs[%d] is nil", i)
		}
		cInputs[i] = t.UnsafePtr()
	}

	cOutputs := make([]unsafe.Pointer, nOut)

	var inPtr *unsafe.Pointer
	if nIn > 0 {
		inPtr = &cInputs[0]
	}

	rc := C.infer_torch_session_run(
		s.ptr,
		inPtr, C.int(nIn),
		&cOutputs[0], C.int(nOut),
	)
	if rc != C.INFER_OK {
		return nil, fmt.Errorf("torch: Run failed: %w", lastError())
	}

	out := make([]*tensor.Tensor, nOut)
	for i, ct := range cOutputs {
		if ct == nil {
			return nil, fmt.Errorf("torch: output[%d] is nil after run", i)
		}
		out[i] = tensor.WrapUnsafePtr(ct)
	}
	return out, nil
}

// RunGPU is an optimized version of Run that uses non-blocking GPU transfers.
// Input tensors are uploaded to GPU asynchronously, inference runs entirely on GPU,
// and only the small output is copied back to CPU.
func (s *Session) RunGPU(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if s.ptr == nil {
		return nil, errors.New("torch: RunGPU called on closed session")
	}

	nIn := len(inputs)
	nOut := s.NumOutputs()
	if nOut == 0 {
		return nil, errors.New("torch: RunGPU called before Load")
	}

	cInputs := make([]unsafe.Pointer, nIn)
	for i, t := range inputs {
		if t == nil {
			return nil, fmt.Errorf("torch: inputs[%d] is nil", i)
		}
		cInputs[i] = t.UnsafePtr()
	}

	cOutputs := make([]unsafe.Pointer, nOut)

	var inPtr *unsafe.Pointer
	if nIn > 0 {
		inPtr = &cInputs[0]
	}

	rc := C.torch_run_gpu_wrap(
		s.ptr,
		unsafe.Pointer(inPtr), C.int(nIn),
		unsafe.Pointer(&cOutputs[0]), C.int(nOut),
	)
	if rc != C.INFER_OK {
		return nil, fmt.Errorf("torch: RunGPU failed: %w", lastError())
	}

	out := make([]*tensor.Tensor, nOut)
	for i, ct := range cOutputs {
		if ct == nil {
			return nil, fmt.Errorf("torch: output[%d] is nil after run_gpu", i)
		}
		out[i] = tensor.WrapUnsafePtr(ct)
	}
	return out, nil
}

// Detection holds one bounding-box detection result from DetectGPU.
type Detection struct {
	X1, Y1, X2, Y2 float32
	ClassID         int
	Confidence      float32
}

// cInferBox must match the C InferBox layout exactly:
//
//	struct { float x1, y1, x2, y2; int class_idx; float confidence; }
type cInferBox struct {
	x1, y1, x2, y2 C.float
	classIdx        C.int
	confidence      C.float
}

// DetectGPU runs the full GPU-accelerated detection pipeline.
// Takes raw JPEG bytes, returns detected objects.
// Everything stays on GPU after JPEG decode — only box coordinates come back.
func (s *Session) DetectGPU(imageBytes []byte, confThresh, iouThresh float32) ([]Detection, error) {
	if s.ptr == nil {
		return nil, errors.New("torch: DetectGPU called on closed session")
	}
	if len(imageBytes) == 0 {
		return nil, errors.New("torch: DetectGPU called with empty image data")
	}

	const maxBoxes = 300
	var boxes [maxBoxes]cInferBox

	n := C.torch_detect_gpu_wrap(
		s.ptr,
		unsafe.Pointer(&imageBytes[0]), C.int(len(imageBytes)),
		C.float(confThresh), C.float(iouThresh),
		unsafe.Pointer(&boxes[0]), C.int(maxBoxes),
	)
	if n < 0 {
		return nil, fmt.Errorf("torch: DetectGPU failed: %w", lastError())
	}

	dets := make([]Detection, int(n))
	for i := 0; i < int(n); i++ {
		b := &boxes[i]
		dets[i] = Detection{
			X1:         float32(b.x1),
			Y1:         float32(b.y1),
			X2:         float32(b.x2),
			Y2:         float32(b.y2),
			ClassID:    int(b.classIdx),
			Confidence: float32(b.confidence),
		}
	}
	return dets, nil
}

// DetectGPUBatch processes multiple images in one CGo call.
// Amortizes the ~2ms CGo overhead across N images.
// Each image is raw JPEG bytes. All images share the same confidence and IoU thresholds.
func (s *Session) DetectGPUBatch(images [][]byte, confThresh, iouThresh float32) ([][]Detection, error) {
	if s.ptr == nil {
		return nil, errors.New("torch: DetectGPUBatch called on closed session")
	}
	n := len(images)
	if n == 0 {
		return nil, errors.New("torch: DetectGPUBatch called with empty image list")
	}

	// For batch_size=1, use single-image path directly.
	if n == 1 {
		dets, err := s.DetectGPU(images[0], confThresh, iouThresh)
		if err != nil {
			return nil, err
		}
		return [][]Detection{dets}, nil
	}

	const maxBoxes = 300
	boxSize := C.size_t(unsafe.Sizeof(cInferBox{}))

	// Allocate everything in C memory to satisfy CGo pointer rules.
	// CGo forbids Go pointers to Go pointers crossing the C boundary.
	cJpegPtrs := (*[1 << 20]unsafe.Pointer)(C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(unsafe.Pointer(nil)))))[:n:n]
	cNbytes := (*[1 << 20]C.int)(C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(C.int(0)))))[:n:n]
	cBoxPtrs := (*[1 << 20]unsafe.Pointer)(C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(unsafe.Pointer(nil)))))[:n:n]
	cCounts := (*[1 << 20]C.int)(C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(C.int(0)))))[:n:n]
	defer C.free(unsafe.Pointer(&cJpegPtrs[0]))
	defer C.free(unsafe.Pointer(&cNbytes[0]))
	defer C.free(unsafe.Pointer(&cBoxPtrs[0]))
	defer C.free(unsafe.Pointer(&cCounts[0]))

	// Per-image box buffers (C-allocated)
	cBoxBufs := make([]unsafe.Pointer, n)
	for i := 0; i < n; i++ {
		if len(images[i]) == 0 {
			for j := 0; j < i; j++ {
				C.free(cBoxBufs[j])
			}
			return nil, fmt.Errorf("torch: DetectGPUBatch: image[%d] is empty", i)
		}
		cJpegPtrs[i] = unsafe.Pointer(&images[i][0])
		cNbytes[i] = C.int(len(images[i]))
		cBoxBufs[i] = C.malloc(C.size_t(maxBoxes) * boxSize)
		cBoxPtrs[i] = cBoxBufs[i]
	}
	defer func() {
		for i := 0; i < n; i++ {
			C.free(cBoxBufs[i])
		}
	}()

	rc := C.torch_detect_gpu_batch_wrap(
		s.ptr,
		unsafe.Pointer(&cJpegPtrs[0]),
		(*C.int)(unsafe.Pointer(&cNbytes[0])),
		C.int(n),
		C.float(confThresh),
		C.float(iouThresh),
		unsafe.Pointer(&cBoxPtrs[0]),
		(*C.int)(unsafe.Pointer(&cCounts[0])),
		C.int(maxBoxes),
	)
	if rc != 0 {
		return nil, fmt.Errorf("torch: DetectGPUBatch failed: %w", lastError())
	}

	// Convert C results to Go Detection slices.
	results := make([][]Detection, n)
	for i := 0; i < n; i++ {
		cnt := int(cCounts[i])
		boxes := (*[maxBoxes]cInferBox)(cBoxBufs[i])
		dets := make([]Detection, cnt)
		for j := 0; j < cnt; j++ {
			b := &boxes[j]
			dets[j] = Detection{
				X1:         float32(b.x1),
				Y1:         float32(b.y1),
				X2:         float32(b.x2),
				Y2:         float32(b.y2),
				ClassID:    int(b.classIdx),
				Confidence: float32(b.confidence),
			}
		}
		results[i] = dets
	}
	return results, nil
}

// DetectGPURaw runs detection on raw RGB pixels — no JPEG encoding needed.
// This is ~30ms faster than DetectGPU at 1080p because it skips the
// RGB→JPEG→RGB round-trip.
// rgb must be width*height*3 bytes of packed RGB uint8 pixels.
func (s *Session) DetectGPURaw(rgb []byte, width, height int, confThresh, iouThresh float32) ([]Detection, error) {
	if s.ptr == nil {
		return nil, errors.New("torch: DetectGPURaw called on closed session")
	}
	if len(rgb) < width*height*3 {
		return nil, fmt.Errorf("torch: DetectGPURaw: RGB buffer too small: got %d, need %d", len(rgb), width*height*3)
	}

	const maxBoxes = 300
	var boxes [maxBoxes]cInferBox

	n := C.torch_detect_gpu_raw_wrap(
		s.ptr,
		unsafe.Pointer(&rgb[0]),
		C.int(width), C.int(height),
		C.float(confThresh), C.float(iouThresh),
		unsafe.Pointer(&boxes[0]), C.int(maxBoxes),
	)
	if n < 0 {
		return nil, fmt.Errorf("torch: DetectGPURaw failed: %w", lastError())
	}

	dets := make([]Detection, int(n))
	for i := 0; i < int(n); i++ {
		b := &boxes[i]
		dets[i] = Detection{
			X1: float32(b.x1), Y1: float32(b.y1),
			X2: float32(b.x2), Y2: float32(b.y2),
			ClassID: int(b.classIdx), Confidence: float32(b.confidence),
		}
	}
	return dets, nil
}

// DetectGPUYUV runs detection on NV12/YUV420P frame data — zero CPU color conversion.
// NV12→RGB happens on GPU. Fastest path for video pipelines.
// yuv must contain Y plane (width*height) followed by UV plane (width*height/2).
func (s *Session) DetectGPUYUV(yuv []byte, width, height, linesize int, confThresh, iouThresh float32) ([]Detection, error) {
	if s.ptr == nil {
		return nil, errors.New("torch: DetectGPUYUV called on closed session")
	}
	expectedSize := linesize * height * 3 / 2
	if len(yuv) < expectedSize {
		return nil, fmt.Errorf("torch: DetectGPUYUV: buffer too small: got %d, need %d", len(yuv), expectedSize)
	}

	const maxBoxes = 300
	var boxes [maxBoxes]cInferBox

	n := C.torch_detect_gpu_yuv_wrap(
		s.ptr,
		unsafe.Pointer(&yuv[0]),
		C.int(width), C.int(height), C.int(linesize),
		C.float(confThresh), C.float(iouThresh),
		unsafe.Pointer(&boxes[0]), C.int(maxBoxes),
	)
	if n < 0 {
		return nil, fmt.Errorf("torch: DetectGPUYUV failed: %w", lastError())
	}

	dets := make([]Detection, int(n))
	for i := 0; i < int(n); i++ {
		b := &boxes[i]
		dets[i] = Detection{
			X1: float32(b.x1), Y1: float32(b.y1),
			X2: float32(b.x2), Y2: float32(b.y2),
			ClassID: int(b.classIdx), Confidence: float32(b.confidence),
		}
	}
	return dets, nil
}
