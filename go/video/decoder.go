// Package video provides Go bindings for hardware-accelerated video
// decode (NVDEC) and encode (NVENC) via the infergo C API.
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

// FrameInfo holds metadata for a decoded video frame.
type FrameInfo struct {
	Width       int
	Height      int
	PTS         int64 // presentation timestamp in microseconds
	FrameNumber int
}

// Decoder wraps an infergo VideoDecoder (opaque C handle).
type Decoder struct {
	ptr    unsafe.Pointer
	width  int
	height int
	fps    float64
	hwAccel bool
}

func lastError() error {
	msg := C.infer_last_error_string()
	if msg == nil || C.GoString(msg) == "" {
		return errors.New("unknown C error")
	}
	return errors.New(C.GoString(msg))
}

// OpenDecoder opens a video source for decoding.
// url can be a file path, RTSP URL, or V4L2 device path.
// hwAccel: when true, attempts NVDEC GPU decoding with CPU fallback.
func OpenDecoder(url string, hwAccel bool) (*Decoder, error) {
	cURL := C.CString(url)
	defer C.free(unsafe.Pointer(cURL))

	hwFlag := C.int(0)
	if hwAccel {
		hwFlag = 1
	}

	ptr := C.infer_video_decoder_open(cURL, hwFlag)
	if ptr == nil {
		return nil, fmt.Errorf("video: decoder open failed: %w", lastError())
	}

	d := &Decoder{
		ptr:     ptr,
		width:   int(C.infer_video_decoder_width(ptr)),
		height:  int(C.infer_video_decoder_height(ptr)),
		fps:     float64(C.infer_video_decoder_fps(ptr)),
		hwAccel: C.infer_video_decoder_is_hw(ptr) != 0,
	}
	runtime.SetFinalizer(d, (*Decoder).Close)
	return d, nil
}

// SetOutputSize tells the decoder to resize frames during the sws_scale
// color conversion — zero extra cost (same FFmpeg call). Call once after
// OpenDecoder, before reading frames. Pass (0,0) for native resolution.
func (d *Decoder) SetOutputSize(w, h int) {
	if d.ptr == nil {
		return
	}
	C.infer_video_decoder_set_output_size(d.ptr, C.int(w), C.int(h))
	if w > 0 && h > 0 {
		d.width = w
		d.height = h
	}
}

// NextFrame decodes the next frame and returns RGB24 pixel data.
// The returned byte slice is a copy of the internal buffer and is safe to
// retain after subsequent calls.
// Returns the pixel data and frame info, or an error on EOF/failure.
func (d *Decoder) NextFrame() ([]byte, FrameInfo, error) {
	if d.ptr == nil {
		return nil, FrameInfo{}, errors.New("video: NextFrame called on closed decoder")
	}

	var rgb *C.uint8_t
	var w, h C.int
	var pts C.int64_t
	var frameNum C.int

	ret := C.infer_video_decoder_next_frame(d.ptr, &rgb, &w, &h, &pts, &frameNum)
	if ret == 0 {
		return nil, FrameInfo{}, fmt.Errorf("video: EOF or decode error")
	}

	info := FrameInfo{
		Width:       int(w),
		Height:      int(h),
		PTS:         int64(pts),
		FrameNumber: int(frameNum),
	}

	// Copy pixel data into Go-managed memory so it survives the next call.
	nbytes := int(w) * int(h) * 3
	data := C.GoBytes(unsafe.Pointer(rgb), C.int(nbytes))
	return data, info, nil
}

// NextFrameResized decodes the next frame, resizes to targetW x targetH in C
// (using OpenCV SIMD), and returns the resized RGB24 data. This copies only
// targetW*targetH*3 bytes to Go instead of the full resolution, dramatically
// reducing memcpy overhead for high-resolution video.
func (d *Decoder) NextFrameResized(targetW, targetH int) ([]byte, FrameInfo, error) {
	if d.ptr == nil {
		return nil, FrameInfo{}, errors.New("video: NextFrameResized called on closed decoder")
	}

	var rgb *C.uint8_t
	var w, h C.int
	var pts C.int64_t
	var frameNum C.int

	ret := C.infer_video_decoder_next_frame_resized(d.ptr,
		C.int(targetW), C.int(targetH),
		&rgb, &w, &h, &pts, &frameNum)
	if ret == 0 {
		return nil, FrameInfo{}, fmt.Errorf("video: EOF or decode error")
	}

	info := FrameInfo{
		Width:       int(w),
		Height:      int(h),
		PTS:         int64(pts),
		FrameNumber: int(frameNum),
	}

	nbytes := int(w) * int(h) * 3
	data := C.GoBytes(unsafe.Pointer(rgb), C.int(nbytes))
	return data, info, nil
}

// Width returns the video frame width in pixels.
func (d *Decoder) Width() int { return d.width }

// Height returns the video frame height in pixels.
func (d *Decoder) Height() int { return d.height }

// FPS returns the video frame rate.
func (d *Decoder) FPS() float64 { return d.fps }

// IsHWAccelerated returns true if NVDEC hardware decoding is active.
func (d *Decoder) IsHWAccelerated() bool { return d.hwAccel }

// Ptr returns the raw C decoder handle for use with pipeline functions
// like infer_pipeline_detect_frame. The caller must not free the pointer.
func (d *Decoder) Ptr() unsafe.Pointer { return d.ptr }

// Close releases all decoder resources. Safe to call multiple times.
func (d *Decoder) Close() {
	if d.ptr == nil {
		return
	}
	C.infer_video_decoder_close(d.ptr)
	d.ptr = nil
	runtime.SetFinalizer(d, nil)
}

// NextFrameZeroCopy decodes the next frame and returns a direct pointer to the
// C decoder's internal RGB buffer WITHOUT copying to Go memory.
// WARNING: the returned slice is only valid until the next call to any NextFrame
// method. The caller must NOT retain it.
func (d *Decoder) NextFrameZeroCopy() ([]byte, FrameInfo, error) {
	if d.ptr == nil {
		return nil, FrameInfo{}, errors.New("video: NextFrameZeroCopy called on closed decoder")
	}

	var rgb *C.uint8_t
	var w, h C.int
	var pts C.int64_t
	var frameNum C.int

	ret := C.infer_video_decoder_next_frame(d.ptr, &rgb, &w, &h, &pts, &frameNum)
	if ret == 0 {
		return nil, FrameInfo{}, fmt.Errorf("video: EOF or decode error")
	}

	info := FrameInfo{
		Width:       int(w),
		Height:      int(h),
		PTS:         int64(pts),
		FrameNumber: int(frameNum),
	}

	// Create a Go slice header pointing directly at C memory — zero copy.
	nbytes := int(w) * int(h) * 3
	data := unsafe.Slice((*byte)(unsafe.Pointer(rgb)), nbytes)
	return data, info, nil
}
