package preprocess_test

import (
	"testing"

	"github.com/ailakshya/infergo/preprocess"
	"github.com/ailakshya/infergo/tensor"
)

// Minimal 1×1 white JPEG (same bytes used in C++ api_test).
var whiteJpeg = []byte{
	0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
	0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43, 0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08,
	0x07, 0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
	0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20, 0x24, 0x2E, 0x27, 0x20,
	0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29, 0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27,
	0x39, 0x3D, 0x38, 0x32, 0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
	0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01,
	0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04,
	0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
	0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D, 0x01, 0x02, 0x03, 0x00,
	0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32,
	0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
	0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35,
	0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55,
	0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
	0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94,
	0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2,
	0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
	0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6,
	0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA,
	0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD5, 0xFF, 0xD9,
}

// ─── DecodeImage ─────────────────────────────────────────────────────────────

func TestDecodeImage_EmptyData(t *testing.T) {
	_, err := preprocess.DecodeImage(nil)
	if err == nil {
		t.Fatal("expected error for nil data")
	}
}

func TestDecodeImage_InvalidBytes(t *testing.T) {
	_, err := preprocess.DecodeImage([]byte{0x00, 0x01, 0x02, 0x03})
	if err == nil {
		t.Fatal("expected error for invalid image bytes")
	}
}

func TestDecodeImage_ValidJPEG(t *testing.T) {
	img, err := preprocess.DecodeImage(whiteJpeg)
	if err != nil {
		t.Fatalf("DecodeImage: %v", err)
	}
	defer img.Free()

	shape := img.Shape()
	if len(shape) != 3 {
		t.Fatalf("expected ndim=3, got %d", len(shape))
	}
	if shape[0] <= 0 || shape[1] <= 0 {
		t.Errorf("H and W must be positive, got %v", shape)
	}
	if shape[2] != 3 {
		t.Errorf("expected 3 channels, got %d", shape[2])
	}
	if img.DType() != tensor.Float32 {
		t.Errorf("expected Float32, got %v", img.DType())
	}
}

// ─── Letterbox ───────────────────────────────────────────────────────────────

func TestLetterbox_NilTensor(t *testing.T) {
	_, err := preprocess.Letterbox(nil, 640, 640)
	if err == nil {
		t.Fatal("expected error for nil tensor")
	}
}

func TestLetterbox_InvalidDims(t *testing.T) {
	img, err := preprocess.DecodeImage(whiteJpeg)
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	defer img.Free()

	if _, err = preprocess.Letterbox(img, 0, 640); err == nil {
		t.Error("expected error for targetW=0")
	}
	if _, err = preprocess.Letterbox(img, 640, 0); err == nil {
		t.Error("expected error for targetH=0")
	}
}

func TestLetterbox_OutputShape(t *testing.T) {
	img, err := preprocess.DecodeImage(whiteJpeg)
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	defer img.Free()

	const TW, TH = 640, 640
	lb, err := preprocess.Letterbox(img, TW, TH)
	if err != nil {
		t.Fatalf("Letterbox: %v", err)
	}
	defer lb.Free()

	shape := lb.Shape()
	if len(shape) != 3 || shape[0] != TH || shape[1] != TW || shape[2] != 3 {
		t.Errorf("expected [%d,%d,3], got %v", TH, TW, shape)
	}
}

// ─── Normalize ───────────────────────────────────────────────────────────────

var (
	imagenetMean = [3]float32{0.485, 0.456, 0.406}
	imagenetStd  = [3]float32{0.229, 0.224, 0.225}
)

func TestNormalize_NilTensor(t *testing.T) {
	_, err := preprocess.Normalize(nil, 255.0, imagenetMean, imagenetStd)
	if err == nil {
		t.Fatal("expected error for nil tensor")
	}
}

func TestNormalize_OutputIsCHW(t *testing.T) {
	img, err := preprocess.DecodeImage(whiteJpeg)
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	defer img.Free()

	norm, err := preprocess.Normalize(img, 255.0, imagenetMean, imagenetStd)
	if err != nil {
		t.Fatalf("Normalize: %v", err)
	}
	defer norm.Free()

	shape := norm.Shape()
	// Input was [H,W,3] → output must be [3,H,W]
	if len(shape) != 3 || shape[0] != 3 {
		t.Errorf("expected CHW shape [3,H,W], got %v", shape)
	}
	if norm.DType() != tensor.Float32 {
		t.Errorf("expected Float32, got %v", norm.DType())
	}
}

// ─── StackBatch ──────────────────────────────────────────────────────────────

func TestStackBatch_EmptySlice(t *testing.T) {
	_, err := preprocess.StackBatch(nil)
	if err == nil {
		t.Fatal("expected error for empty slice")
	}
}

func TestStackBatch_NilElement(t *testing.T) {
	img, err := preprocess.DecodeImage(whiteJpeg)
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	defer img.Free()

	_, err = preprocess.StackBatch([]*tensor.Tensor{img, nil})
	if err == nil {
		t.Fatal("expected error for nil element")
	}
}

func TestStackBatch_FullPipeline(t *testing.T) {
	// Decode → letterbox → normalize → stack ×3
	const N, TW, TH = 3, 64, 64
	var batch [N]*tensor.Tensor

	for i := range batch {
		img, err := preprocess.DecodeImage(whiteJpeg)
		if err != nil {
			t.Fatalf("DecodeImage[%d]: %v", i, err)
		}
		lb, err := preprocess.Letterbox(img, TW, TH)
		img.Free()
		if err != nil {
			t.Fatalf("Letterbox[%d]: %v", i, err)
		}
		norm, err := preprocess.Normalize(lb, 255.0, imagenetMean, imagenetStd)
		lb.Free()
		if err != nil {
			t.Fatalf("Normalize[%d]: %v", i, err)
		}
		batch[i] = norm
	}

	stacked, err := preprocess.StackBatch(batch[:])
	for _, t2 := range batch {
		t2.Free()
	}
	if err != nil {
		t.Fatalf("StackBatch: %v", err)
	}
	defer stacked.Free()

	shape := stacked.Shape()
	if len(shape) != 4 {
		t.Fatalf("expected 4D NCHW tensor, got ndim=%d shape=%v", len(shape), shape)
	}
	if shape[0] != N || shape[1] != 3 || shape[2] != TH || shape[3] != TW {
		t.Errorf("expected [%d,3,%d,%d], got %v", N, TH, TW, shape)
	}
}
