// Package video provides C-accelerated frame annotation via the infergo C API.
package video

/*
#cgo CFLAGS: -I${SRCDIR}/../../cpp/include
#cgo LDFLAGS: -L${SRCDIR}/../../build/cpp/api -linfer_api -Wl,-rpath,${SRCDIR}/../../build/cpp/api

#include "infer_api.h"
#include <stdlib.h>
#include <string.h>

// Helper to set label field on InferAnnotateBox (CGo can't handle fixed arrays directly)
static inline void set_annotate_box(InferAnnotateBox* box, float x1, float y1, float x2, float y2,
    int class_id, float confidence, int track_id, const char* label) {
    box->x1 = x1; box->y1 = y1; box->x2 = x2; box->y2 = y2;
    box->class_id = class_id; box->confidence = confidence; box->track_id = track_id;
    if (label) { strncpy(box->label, label, 63); box->label[63] = '\0'; }
    else { box->label[0] = '\0'; }
}

static inline void set_text_overlay(InferTextOverlay* t, int x, int y,
    const char* text, uint8_t r, uint8_t g, uint8_t b, int scale) {
    t->x = x; t->y = y; t->r = r; t->g = g; t->b = b; t->scale = scale;
    if (text) { strncpy(t->text, text, 127); t->text[127] = '\0'; }
    else { t->text[0] = '\0'; }
}
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// AnnotateFast draws bounding boxes and labels on an RGB frame using the C
// frame annotator and returns JPEG-encoded bytes.
// quality controls the JPEG compression quality (1-100).
func AnnotateFast(rgb []byte, width, height int, objects []AnnotateObject, quality int) ([]byte, error) {
	expected := width * height * 3
	if len(rgb) != expected {
		return nil, fmt.Errorf("annotate_fast: expected %d RGB bytes (%dx%dx3), got %d", expected, width, height, len(rgb))
	}

	// Build C box array.
	n := len(objects)
	var boxesPtr *C.InferAnnotateBox
	if n > 0 {
		boxesPtr = (*C.InferAnnotateBox)(C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(C.InferAnnotateBox{}))))
		if boxesPtr == nil {
			return nil, fmt.Errorf("annotate_fast: failed to allocate %d boxes", n)
		}
		defer C.free(unsafe.Pointer(boxesPtr))

		boxes := unsafe.Slice(boxesPtr, n)
		for i, obj := range objects {
			var cLabel *C.char
			if obj.Label != "" {
				cLabel = C.CString(obj.Label)
				defer C.free(unsafe.Pointer(cLabel))
			}
			C.set_annotate_box(&boxes[i],
				C.float(obj.X1), C.float(obj.Y1), C.float(obj.X2), C.float(obj.Y2),
				C.int(obj.ClassID), C.float(obj.Confidence), C.int(obj.TrackID), cLabel)
		}
	}

	var outJPEG *C.uint8_t
	var outSize C.int

	ret := C.infer_frame_annotate_jpeg(
		(*C.uint8_t)(unsafe.Pointer(&rgb[0])),
		C.int(width), C.int(height),
		boxesPtr, C.int(n),
		C.int(quality),
		&outJPEG, &outSize)

	if ret != C.INFER_OK {
		return nil, fmt.Errorf("annotate_fast: C call failed: %w", lastError())
	}

	// Copy to Go-managed memory and free the C buffer.
	result := C.GoBytes(unsafe.Pointer(outJPEG), outSize)
	C.infer_frame_jpeg_free(outJPEG)
	return result, nil
}

// ResizeJPEG resizes an RGB frame and returns the result as JPEG-encoded bytes.
func ResizeJPEG(rgb []byte, srcW, srcH, dstW, dstH, quality int) ([]byte, error) {
	expected := srcW * srcH * 3
	if len(rgb) != expected {
		return nil, fmt.Errorf("resize_jpeg: expected %d RGB bytes (%dx%dx3), got %d", expected, srcW, srcH, len(rgb))
	}

	var outJPEG *C.uint8_t
	var outSize C.int

	ret := C.infer_frame_resize_jpeg(
		(*C.uint8_t)(unsafe.Pointer(&rgb[0])),
		C.int(srcW), C.int(srcH),
		C.int(dstW), C.int(dstH),
		C.int(quality),
		&outJPEG, &outSize)

	if ret != C.INFER_OK {
		return nil, fmt.Errorf("resize_jpeg: C call failed: %w", lastError())
	}

	result := C.GoBytes(unsafe.Pointer(outJPEG), outSize)
	C.infer_frame_jpeg_free(outJPEG)
	return result, nil
}

// CombineJPEG combines two RGB frames side-by-side with a status bar and
// returns the result as JPEG-encoded bytes.
func CombineJPEG(rgb1 []byte, w1, h1 int, rgb2 []byte, w2, h2 int, statusText string, targetW, targetH, quality int) ([]byte, error) {
	expected1 := w1 * h1 * 3
	if len(rgb1) != expected1 {
		return nil, fmt.Errorf("combine_jpeg: frame1 expected %d RGB bytes, got %d", expected1, len(rgb1))
	}
	expected2 := w2 * h2 * 3
	if len(rgb2) != expected2 {
		return nil, fmt.Errorf("combine_jpeg: frame2 expected %d RGB bytes, got %d", expected2, len(rgb2))
	}

	cStatus := C.CString(statusText)
	defer C.free(unsafe.Pointer(cStatus))

	var outJPEG *C.uint8_t
	var outSize C.int

	ret := C.infer_frame_combine_jpeg(
		(*C.uint8_t)(unsafe.Pointer(&rgb1[0])), C.int(w1), C.int(h1),
		(*C.uint8_t)(unsafe.Pointer(&rgb2[0])), C.int(w2), C.int(h2),
		cStatus,
		C.int(targetW), C.int(targetH), C.int(quality),
		&outJPEG, &outSize)

	if ret != C.INFER_OK {
		return nil, fmt.Errorf("combine_jpeg: C call failed: %w", lastError())
	}

	result := C.GoBytes(unsafe.Pointer(outJPEG), outSize)
	C.infer_frame_jpeg_free(outJPEG)
	return result, nil
}

// DrawLineFast draws a line on an RGB buffer in-place using the C frame annotator.
func DrawLineFast(rgb []byte, width, height, x1, y1, x2, y2 int, r, g, b uint8, thickness int) error {
	expected := width * height * 3
	if len(rgb) != expected {
		return fmt.Errorf("draw_line_fast: expected %d RGB bytes (%dx%dx3), got %d", expected, width, height, len(rgb))
	}

	ret := C.infer_frame_draw_line(
		(*C.uint8_t)(unsafe.Pointer(&rgb[0])),
		C.int(width), C.int(height),
		C.int(x1), C.int(y1), C.int(x2), C.int(y2),
		C.uint8_t(r), C.uint8_t(g), C.uint8_t(b), C.int(thickness))

	if ret != C.INFER_OK {
		return fmt.Errorf("draw_line_fast: C call failed: %w", lastError())
	}
	return nil
}

// DrawPolygonFast draws a polygon (outline + alpha fill) on an RGB buffer in-place.
func DrawPolygonFast(rgb []byte, width, height int, points [][2]int, r, g, b, alpha uint8) error {
	expected := width * height * 3
	if len(rgb) != expected {
		return fmt.Errorf("draw_polygon_fast: expected %d RGB bytes (%dx%dx3), got %d", expected, width, height, len(rgb))
	}
	if len(points) < 3 {
		return fmt.Errorf("draw_polygon_fast: need at least 3 points, got %d", len(points))
	}

	n := len(points)
	cPoints := make([]C.InferPoint, n)
	for i, pt := range points {
		cPoints[i].x = C.int(pt[0])
		cPoints[i].y = C.int(pt[1])
	}

	ret := C.infer_frame_draw_polygon(
		(*C.uint8_t)(unsafe.Pointer(&rgb[0])),
		C.int(width), C.int(height),
		&cPoints[0], C.int(n),
		C.uint8_t(r), C.uint8_t(g), C.uint8_t(b), C.uint8_t(alpha))

	if ret != C.INFER_OK {
		return fmt.Errorf("draw_polygon_fast: C call failed: %w", lastError())
	}
	return nil
}

// DrawTextFast draws text on an RGB buffer in-place using the C bitmap font.
func DrawTextFast(rgb []byte, width, height, x, y int, text string, r, g, b uint8, scale int) error {
	expected := width * height * 3
	if len(rgb) != expected {
		return fmt.Errorf("draw_text_fast: expected %d RGB bytes (%dx%dx3), got %d", expected, width, height, len(rgb))
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	ret := C.infer_frame_draw_text(
		(*C.uint8_t)(unsafe.Pointer(&rgb[0])),
		C.int(width), C.int(height),
		C.int(x), C.int(y), cText,
		C.uint8_t(r), C.uint8_t(g), C.uint8_t(b), C.int(scale))

	if ret != C.INFER_OK {
		return fmt.Errorf("draw_text_fast: C call failed: %w", lastError())
	}
	return nil
}

// AnnotateFullInput holds all drawing commands for a single batch AnnotateFull call.
type AnnotateFullInput struct {
	Boxes    []AnnotateObject
	Lines    []AnnotateLine
	Polygons []AnnotatePolygon
	Texts    []AnnotateText
	Rects    []AnnotateRect
}

// AnnotateLine describes a line to draw.
type AnnotateLine struct {
	X1, Y1, X2, Y2 int
	R, G, B         uint8
	Thickness       int
}

// AnnotatePolygon describes a polygon to draw.
type AnnotatePolygon struct {
	Points [][2]int
	R, G, B, Alpha uint8
}

// AnnotateText describes text to draw.
type AnnotateText struct {
	X, Y    int
	Text    string
	R, G, B uint8
	Scale   int
}

// AnnotateRect describes a filled rectangle to draw.
type AnnotateRect struct {
	X1, Y1, X2, Y2 int
	R, G, B, Alpha  uint8
}

// AnnotateFull draws ALL overlays (boxes, lines, polygons, text, rects) on an
// RGB frame, resizes to outW x outH, and encodes to JPEG — all in ONE C call.
// This replaces 20+ individual Go draw calls + Go JPEG encode with a single
// CGo crossing. Returns JPEG bytes.
func AnnotateFull(rgb []byte, w, h int, input AnnotateFullInput, outW, outH, quality int) ([]byte, error) {
	if len(rgb) != w*h*3 {
		return nil, fmt.Errorf("AnnotateFull: RGB size %d != %dx%dx3", len(rgb), w, h)
	}

	// Convert boxes.
	nBoxes := len(input.Boxes)
	var cBoxes *C.InferAnnotateBox
	if nBoxes > 0 {
		boxes := make([]C.InferAnnotateBox, nBoxes)
		for i, obj := range input.Boxes {
			label := obj.Label
			if label == "" {
				label = fmt.Sprintf("cls%d #%d %.0f%%", obj.ClassID, obj.TrackID, obj.Confidence*100)
			}
			cLabel := C.CString(label)
			C.set_annotate_box(&boxes[i],
				C.float(obj.X1), C.float(obj.Y1), C.float(obj.X2), C.float(obj.Y2),
				C.int(obj.ClassID), C.float(obj.Confidence), C.int(obj.TrackID), cLabel)
			C.free(unsafe.Pointer(cLabel))
		}
		cBoxes = &boxes[0]
	}

	// Convert lines.
	nLines := len(input.Lines)
	var cLines *C.InferLine
	if nLines > 0 {
		lines := make([]C.InferLine, nLines)
		for i, l := range input.Lines {
			lines[i] = C.InferLine{
				x1: C.int(l.X1), y1: C.int(l.Y1), x2: C.int(l.X2), y2: C.int(l.Y2),
				r: C.uint8_t(l.R), g: C.uint8_t(l.G), b: C.uint8_t(l.B),
				thickness: C.int(l.Thickness),
			}
		}
		cLines = &lines[0]
	}

	// Convert polygons.
	nPolygons := len(input.Polygons)
	var cPolygons *C.InferPolygonOverlay
	if nPolygons > 0 {
		polys := make([]C.InferPolygonOverlay, nPolygons)
		for i, p := range input.Polygons {
			polys[i].n_pts = C.int(len(p.Points))
			polys[i].r = C.uint8_t(p.R)
			polys[i].g = C.uint8_t(p.G)
			polys[i].b = C.uint8_t(p.B)
			polys[i].alpha = C.uint8_t(p.Alpha)
			for j := 0; j < len(p.Points) && j < 8; j++ {
				polys[i].pts[j].x = C.int(p.Points[j][0])
				polys[i].pts[j].y = C.int(p.Points[j][1])
			}
		}
		cPolygons = &polys[0]
	}

	// Convert texts.
	nTexts := len(input.Texts)
	var cTexts *C.InferTextOverlay
	if nTexts > 0 {
		texts := make([]C.InferTextOverlay, nTexts)
		for i, t := range input.Texts {
			cStr := C.CString(t.Text)
			C.set_text_overlay(&texts[i], C.int(t.X), C.int(t.Y), cStr,
				C.uint8_t(t.R), C.uint8_t(t.G), C.uint8_t(t.B), C.int(t.Scale))
			C.free(unsafe.Pointer(cStr))
		}
		cTexts = &texts[0]
	}

	// Convert rects.
	nRects := len(input.Rects)
	var cRects *C.InferFilledRect
	if nRects > 0 {
		rects := make([]C.InferFilledRect, nRects)
		for i, r := range input.Rects {
			rects[i] = C.InferFilledRect{
				x1: C.int(r.X1), y1: C.int(r.Y1), x2: C.int(r.X2), y2: C.int(r.Y2),
				r: C.uint8_t(r.R), g: C.uint8_t(r.G), b: C.uint8_t(r.B), alpha: C.uint8_t(r.Alpha),
			}
		}
		cRects = &rects[0]
	}

	var outJPEG *C.uint8_t
	var outSize C.int

	ret := C.infer_frame_annotate_full(
		(*C.uint8_t)(unsafe.Pointer(&rgb[0])), C.int(w), C.int(h),
		cBoxes, C.int(nBoxes),
		cLines, C.int(nLines),
		cPolygons, C.int(nPolygons),
		cTexts, C.int(nTexts),
		cRects, C.int(nRects),
		C.int(outW), C.int(outH), C.int(quality),
		&outJPEG, &outSize)

	if ret != C.INFER_OK || outJPEG == nil {
		return nil, fmt.Errorf("AnnotateFull: C call failed: %w", lastError())
	}

	result := C.GoBytes(unsafe.Pointer(outJPEG), outSize)
	C.infer_frame_jpeg_free(outJPEG)
	return result, nil
}
