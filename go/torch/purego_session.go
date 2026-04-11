//go:build !cgo

// Package torch provides a Go wrapper for the infergo PyTorch/libtorch session C API.
// This file implements the CGo-free path using purego (dlopen/dlsym) to call
// C functions directly from Go without the ~2ms CGo overhead.
//
// Build with CGO_ENABLED=0 to use this path.
package torch

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"unsafe"

	"github.com/ebitengine/purego"
)

// ─── C struct layout ────────────────────────────────────────────────────────

// inferBox matches the C InferBox layout exactly:
//
//	struct InferBox { float x1, y1, x2, y2; int class_idx; float confidence; }
//
// Size: 24 bytes on all platforms (6 x 4-byte fields, no padding).
type inferBox struct {
	X1, Y1, X2, Y2 float32
	ClassIdx        int32
	Confidence      float32
}

// ─── Library handle and function pointers ───────────────────────────────────

var (
	libHandle uintptr

	// Function pointers loaded via dlsym. purego.RegisterLibFunc populates
	// these during init() using the C ABI calling convention.
	fnLastErrorString func() uintptr

	fnTorchCreate     func(provider uintptr, deviceID int32) uintptr
	fnTorchLoad       func(sess uintptr, path uintptr) int32
	fnTorchNumInputs  func(sess uintptr) int32
	fnTorchNumOutputs func(sess uintptr) int32
	fnTorchDestroy    func(sess uintptr)
	fnTorchDetectGPU  func(sess uintptr, data uintptr, nbytes int32,
		conf float32, iou float32, boxes uintptr, maxBoxes int32) int32

	fnTorchRun func(sess uintptr, inputs uintptr, nIn int32,
		outputs uintptr, nOut int32) int32
	fnTorchRunGPU func(sess uintptr, inputs uintptr, nIn int32,
		outputs uintptr, nOut int32) int32

	fnTorchDetectGPUBatch func(sess uintptr,
		jpegDataArray uintptr, nbytesArray uintptr, batchSize int32,
		conf float32, iou float32,
		outBoxesArray uintptr, outCounts uintptr, maxBoxesPerImage int32) int32
)

func init() {
	libPath := findLibrary()
	lib, err := purego.Dlopen(libPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		panic("purego: failed to load libinfer_api.so: " + err.Error())
	}
	libHandle = lib

	purego.RegisterLibFunc(&fnLastErrorString, lib, "infer_last_error_string")

	purego.RegisterLibFunc(&fnTorchCreate, lib, "infer_torch_session_create")
	purego.RegisterLibFunc(&fnTorchLoad, lib, "infer_torch_session_load")
	purego.RegisterLibFunc(&fnTorchNumInputs, lib, "infer_torch_session_num_inputs")
	purego.RegisterLibFunc(&fnTorchNumOutputs, lib, "infer_torch_session_num_outputs")
	purego.RegisterLibFunc(&fnTorchDestroy, lib, "infer_torch_session_destroy")
	purego.RegisterLibFunc(&fnTorchDetectGPU, lib, "infer_torch_detect_gpu")
	purego.RegisterLibFunc(&fnTorchRun, lib, "infer_torch_session_run")
	purego.RegisterLibFunc(&fnTorchRunGPU, lib, "infer_torch_session_run_gpu")
	purego.RegisterLibFunc(&fnTorchDetectGPUBatch, lib, "infer_torch_detect_gpu_batch")
}

// findLibrary searches for libinfer_api.so in well-known locations.
func findLibrary() string {
	// INFERGO_LIB env var takes highest priority.
	if p := os.Getenv("INFERGO_LIB"); p != "" {
		return p
	}
	paths := []string{
		"build/cpp/api/libinfer_api.so",
		"../build/cpp/api/libinfer_api.so",
		"../../build/cpp/api/libinfer_api.so",
		"/usr/local/lib/infergo/libinfer_api.so",
		"/usr/lib/libinfer_api.so",
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	// Fall back to LD_LIBRARY_PATH resolution.
	return "libinfer_api.so"
}

// ─── Helpers ────────────────────────────────────────────────────────────────

// cstring allocates a null-terminated byte slice from a Go string.
// The returned pointer is valid only while the slice is alive.
func cstring(s string) (*byte, []byte) {
	b := append([]byte(s), 0)
	return &b[0], b
}

// goString reads a null-terminated C string from the given pointer.
func goString(ptr uintptr) string {
	if ptr == 0 {
		return ""
	}
	// Walk bytes until NUL. We use a base pointer and index via unsafe.Add
	// to avoid go vet's "misuse of unsafe.Pointer" for pointer arithmetic.
	base := unsafe.Pointer(ptr) //nolint:govet // purego: ptr is a valid C pointer from dlsym
	var length int
	for {
		ch := *(*byte)(unsafe.Add(base, length))
		if ch == 0 {
			break
		}
		length++
		if length > 4096 { // safety limit
			break
		}
	}
	if length == 0 {
		return ""
	}
	return string(unsafe.Slice((*byte)(base), length))
}

// lastError retrieves the last C API error string.
func lastError() error {
	ptr := fnLastErrorString()
	msg := goString(ptr)
	if msg == "" {
		return errors.New("unknown C error")
	}
	return errors.New(msg)
}

// ─── Detection type (shared with CGo path) ─────────────────────────────────

// Detection holds one bounding-box detection result from DetectGPU.
type Detection struct {
	X1, Y1, X2, Y2 float32
	ClassID         int
	Confidence      float32
}

// ─── Session ────────────────────────────────────────────────────────────────

// Session wraps an InferTorchSession (opaque C pointer) using purego.
// No CGo is involved — all C calls go through dlsym function pointers.
type Session struct {
	ptr uintptr
}

// NewSession creates a PyTorch inference session.
// provider: "cpu" | "cuda". Falls back to CPU if CUDA is unavailable.
func NewSession(provider string, deviceID int) (*Session, error) {
	prov, provBuf := cstring(provider)
	_ = provBuf // keep alive
	ptr := fnTorchCreate(uintptr(unsafe.Pointer(prov)), int32(deviceID))
	runtime.KeepAlive(provBuf)
	if ptr == 0 {
		return nil, fmt.Errorf("torch: session_create failed: %w", lastError())
	}
	s := &Session{ptr: ptr}
	runtime.SetFinalizer(s, (*Session).Close)
	return s, nil
}

// Close destroys the session. Safe to call multiple times.
func (s *Session) Close() {
	if s.ptr == 0 {
		return
	}
	fnTorchDestroy(s.ptr)
	s.ptr = 0
	runtime.SetFinalizer(s, nil)
}

// Load reads a TorchScript model file and prepares it for inference.
func (s *Session) Load(modelPath string) error {
	if s.ptr == 0 {
		return errors.New("torch: Load called on closed session")
	}
	p, pBuf := cstring(modelPath)
	_ = pBuf
	rc := fnTorchLoad(s.ptr, uintptr(unsafe.Pointer(p)))
	runtime.KeepAlive(pBuf)
	if rc != 0 { // INFER_OK == 0
		return fmt.Errorf("torch: Load failed: %w", lastError())
	}
	return nil
}

// NumInputs returns the number of model inputs (valid after Load).
func (s *Session) NumInputs() int {
	if s.ptr == 0 {
		return 0
	}
	return int(fnTorchNumInputs(s.ptr))
}

// NumOutputs returns the number of model outputs (valid after Load).
func (s *Session) NumOutputs() int {
	if s.ptr == 0 {
		return 0
	}
	return int(fnTorchNumOutputs(s.ptr))
}

// DetectGPU runs the full GPU-accelerated detection pipeline.
// Takes raw JPEG bytes, returns detected objects.
// Everything stays on GPU after JPEG decode -- only box coordinates come back.
func (s *Session) DetectGPU(imageBytes []byte, confThresh, iouThresh float32) ([]Detection, error) {
	if s.ptr == 0 {
		return nil, errors.New("torch: DetectGPU called on closed session")
	}
	if len(imageBytes) == 0 {
		return nil, errors.New("torch: DetectGPU called with empty image data")
	}

	const maxBoxes = 300
	var boxes [maxBoxes]inferBox

	n := fnTorchDetectGPU(
		s.ptr,
		uintptr(unsafe.Pointer(&imageBytes[0])), int32(len(imageBytes)),
		confThresh, iouThresh,
		uintptr(unsafe.Pointer(&boxes[0])), int32(maxBoxes),
	)
	runtime.KeepAlive(imageBytes)

	if n < 0 {
		return nil, fmt.Errorf("torch: DetectGPU failed: %w", lastError())
	}

	dets := make([]Detection, int(n))
	for i := int32(0); i < n; i++ {
		b := &boxes[i]
		dets[i] = Detection{
			X1:         b.X1,
			Y1:         b.Y1,
			X2:         b.X2,
			Y2:         b.Y2,
			ClassID:    int(b.ClassIdx),
			Confidence: b.Confidence,
		}
	}
	return dets, nil
}

// DetectGPUBatch processes multiple images in one call.
// Each image is raw JPEG bytes. All images share the same confidence and IoU thresholds.
func (s *Session) DetectGPUBatch(images [][]byte, confThresh, iouThresh float32) ([][]Detection, error) {
	if s.ptr == 0 {
		return nil, errors.New("torch: DetectGPUBatch called on closed session")
	}
	n := len(images)
	if n == 0 {
		return nil, errors.New("torch: DetectGPUBatch called with empty image list")
	}

	// For batch_size=1, use single-image path.
	if n == 1 {
		dets, err := s.DetectGPU(images[0], confThresh, iouThresh)
		if err != nil {
			return nil, err
		}
		return [][]Detection{dets}, nil
	}

	const maxBoxes = 300
	boxSize := unsafe.Sizeof(inferBox{})

	// Allocate arrays. In the purego path we don't have CGo pointer rules,
	// so we can use Go-allocated memory directly.
	jpegPtrs := make([]uintptr, n)
	nbytes := make([]int32, n)
	boxBufs := make([][]inferBox, n)
	boxPtrs := make([]uintptr, n)
	counts := make([]int32, n)

	for i := 0; i < n; i++ {
		if len(images[i]) == 0 {
			return nil, fmt.Errorf("torch: DetectGPUBatch: image[%d] is empty", i)
		}
		jpegPtrs[i] = uintptr(unsafe.Pointer(&images[i][0]))
		nbytes[i] = int32(len(images[i]))
		boxBufs[i] = make([]inferBox, maxBoxes)
		boxPtrs[i] = uintptr(unsafe.Pointer(&boxBufs[i][0]))
	}

	rc := fnTorchDetectGPUBatch(
		s.ptr,
		uintptr(unsafe.Pointer(&jpegPtrs[0])),
		uintptr(unsafe.Pointer(&nbytes[0])),
		int32(n),
		confThresh, iouThresh,
		uintptr(unsafe.Pointer(&boxPtrs[0])),
		uintptr(unsafe.Pointer(&counts[0])),
		int32(maxBoxes),
	)
	runtime.KeepAlive(images)
	runtime.KeepAlive(jpegPtrs)
	runtime.KeepAlive(nbytes)
	runtime.KeepAlive(boxBufs)
	runtime.KeepAlive(boxPtrs)
	_ = boxSize

	if rc != 0 {
		return nil, fmt.Errorf("torch: DetectGPUBatch failed: %w", lastError())
	}

	results := make([][]Detection, n)
	for i := 0; i < n; i++ {
		cnt := int(counts[i])
		dets := make([]Detection, cnt)
		for j := 0; j < cnt; j++ {
			b := &boxBufs[i][j]
			dets[j] = Detection{
				X1:         b.X1,
				Y1:         b.Y1,
				X2:         b.X2,
				Y2:         b.Y2,
				ClassID:    int(b.ClassIdx),
				Confidence: b.Confidence,
			}
		}
		results[i] = dets
	}
	return results, nil
}
