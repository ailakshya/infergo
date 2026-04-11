package video

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
)

// Encoder wraps an infergo VideoEncoder (opaque C handle).
type Encoder struct {
	ptr     unsafe.Pointer
	width   int
	height  int
	hwAccel bool
}

// OpenEncoder creates a video encoder writing to the given file path.
// codec: "h264_nvenc" (GPU) or "libx264" (CPU). Falls back to libx264 on failure.
func OpenEncoder(path string, w, h, fps int, codec string) (*Encoder, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	cCodec := C.CString(codec)
	defer C.free(unsafe.Pointer(cCodec))

	ptr := C.infer_video_encoder_open(cPath, C.int(w), C.int(h), C.int(fps), cCodec)
	if ptr == nil {
		return nil, fmt.Errorf("video: encoder open failed: %w", lastError())
	}

	e := &Encoder{
		ptr:     ptr,
		width:   w,
		height:  h,
		hwAccel: C.infer_video_encoder_is_hw(ptr) != 0,
	}
	runtime.SetFinalizer(e, (*Encoder).Close)
	return e, nil
}

// WriteFrame encodes one RGB24 frame. rgb must be exactly width*height*3 bytes.
func (e *Encoder) WriteFrame(rgb []byte) error {
	if e.ptr == nil {
		return errors.New("video: WriteFrame called on closed encoder")
	}
	expected := e.width * e.height * 3
	if len(rgb) < expected {
		return fmt.Errorf("video: WriteFrame: buffer too small (%d < %d)", len(rgb), expected)
	}

	ret := C.infer_video_encoder_write(e.ptr,
		(*C.uint8_t)(unsafe.Pointer(&rgb[0])),
		C.int(e.width), C.int(e.height))
	if ret == 0 {
		return fmt.Errorf("video: encode failed: %w", lastError())
	}
	return nil
}

// IsHWAccelerated returns true if NVENC hardware encoding is active.
func (e *Encoder) IsHWAccelerated() bool { return e.hwAccel }

// Close flushes remaining frames and releases all encoder resources.
// Safe to call multiple times.
func (e *Encoder) Close() {
	if e.ptr == nil {
		return
	}
	C.infer_video_encoder_close(e.ptr)
	e.ptr = nil
	runtime.SetFinalizer(e, nil)
}
