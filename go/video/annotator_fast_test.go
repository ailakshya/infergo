package video

import (
	"bytes"
	"image/jpeg"
	"testing"
)

// TestAnnotateFastEmpty verifies that annotating with no boxes produces valid JPEG.
func TestAnnotateFastEmpty(t *testing.T) {
	width, height := 320, 240
	rgb := makeTestRGB(width, height)

	jpegData, err := AnnotateFast(rgb, width, height, nil, 75)
	if err != nil {
		t.Fatalf("AnnotateFast with no objects: %v", err)
	}

	// Check JPEG magic bytes.
	if len(jpegData) < 2 || jpegData[0] != 0xFF || jpegData[1] != 0xD8 {
		t.Fatalf("output does not start with JPEG magic (0xFF 0xD8), got %x %x", jpegData[0], jpegData[1])
	}

	// Verify it's a valid JPEG.
	_, err = jpeg.Decode(bytes.NewReader(jpegData))
	if err != nil {
		t.Fatalf("returned data is not valid JPEG: %v", err)
	}
}

// TestAnnotateFastBoxes verifies that annotating with 3 boxes produces valid JPEG.
func TestAnnotateFastBoxes(t *testing.T) {
	width, height := 640, 480
	rgb := makeTestRGB(width, height)

	objects := []AnnotateObject{
		{X1: 10, Y1: 20, X2: 100, Y2: 200, ClassID: 0, Confidence: 0.95, TrackID: 1, Label: "person #1"},
		{X1: 200, Y1: 50, X2: 350, Y2: 300, ClassID: 2, Confidence: 0.88, TrackID: 2, Label: "car #2"},
		{X1: 400, Y1: 100, X2: 500, Y2: 400, ClassID: 15, Confidence: 0.72, TrackID: 3, Label: "cat #3"},
	}

	jpegData, err := AnnotateFast(rgb, width, height, objects, 75)
	if err != nil {
		t.Fatalf("AnnotateFast with 3 objects: %v", err)
	}

	// Check JPEG magic bytes.
	if len(jpegData) < 2 || jpegData[0] != 0xFF || jpegData[1] != 0xD8 {
		t.Fatalf("output does not start with JPEG magic (0xFF 0xD8)")
	}

	// Verify dimensions preserved.
	img, err := jpeg.Decode(bytes.NewReader(jpegData))
	if err != nil {
		t.Fatalf("returned data is not valid JPEG: %v", err)
	}
	bounds := img.Bounds()
	if bounds.Dx() != width || bounds.Dy() != height {
		t.Errorf("expected %dx%d, got %dx%d", width, height, bounds.Dx(), bounds.Dy())
	}
}

// TestResizeJPEG verifies resizing a 640x480 frame to 320x240.
func TestResizeJPEG(t *testing.T) {
	srcW, srcH := 640, 480
	dstW, dstH := 320, 240
	rgb := makeTestRGB(srcW, srcH)

	jpegData, err := ResizeJPEG(rgb, srcW, srcH, dstW, dstH, 75)
	if err != nil {
		t.Fatalf("ResizeJPEG: %v", err)
	}

	// Check JPEG magic bytes.
	if len(jpegData) < 2 || jpegData[0] != 0xFF || jpegData[1] != 0xD8 {
		t.Fatalf("output does not start with JPEG magic (0xFF 0xD8)")
	}

	// Verify output dimensions.
	img, err := jpeg.Decode(bytes.NewReader(jpegData))
	if err != nil {
		t.Fatalf("returned data is not valid JPEG: %v", err)
	}
	bounds := img.Bounds()
	if bounds.Dx() != dstW || bounds.Dy() != dstH {
		t.Errorf("expected %dx%d, got %dx%d", dstW, dstH, bounds.Dx(), bounds.Dy())
	}
}

// TestDrawLineFast verifies that drawing a line modifies the pixel buffer.
func TestDrawLineFast(t *testing.T) {
	width, height := 200, 200
	rgb := makeTestRGB(width, height)

	// Save a copy of the original pixel at the center of the line.
	midIdx := (100*width + 100) * 3
	origR, origG, origB := rgb[midIdx], rgb[midIdx+1], rgb[midIdx+2]

	err := DrawLineFast(rgb, width, height, 0, 100, 199, 100, 255, 0, 0, 3)
	if err != nil {
		t.Fatalf("DrawLineFast: %v", err)
	}

	// Check that the pixel at the line changed.
	newR, newG, newB := rgb[midIdx], rgb[midIdx+1], rgb[midIdx+2]
	if newR == origR && newG == origG && newB == origB {
		t.Error("expected pixel at line center to change, but it did not")
	}
}

// TestCombineJPEG verifies combining two frames into one JPEG.
func TestCombineJPEG(t *testing.T) {
	w1, h1 := 320, 240
	w2, h2 := 320, 240
	rgb1 := makeTestRGB(w1, h1)
	rgb2 := makeTestRGB(w2, h2)

	jpegData, err := CombineJPEG(rgb1, w1, h1, rgb2, w2, h2, "Status: OK", 640, 240, 75)
	if err != nil {
		t.Fatalf("CombineJPEG: %v", err)
	}

	// Check JPEG magic bytes.
	if len(jpegData) < 2 || jpegData[0] != 0xFF || jpegData[1] != 0xD8 {
		t.Fatalf("output does not start with JPEG magic (0xFF 0xD8)")
	}

	// Verify it's a valid JPEG.
	_, err = jpeg.Decode(bytes.NewReader(jpegData))
	if err != nil {
		t.Fatalf("returned data is not valid JPEG: %v", err)
	}
}

// BenchmarkAnnotateFast_1080p benchmarks C-accelerated annotation at 1080p with 5 boxes.
func BenchmarkAnnotateFast_1080p(b *testing.B) {
	width, height := 1920, 1080
	rgb := makeTestRGB(width, height)

	objects := []AnnotateObject{
		{X1: 100, Y1: 100, X2: 400, Y2: 400, ClassID: 0, Confidence: 0.95, TrackID: 1, Label: "person #1"},
		{X1: 500, Y1: 200, X2: 800, Y2: 500, ClassID: 2, Confidence: 0.88, TrackID: 2, Label: "car #2"},
		{X1: 900, Y1: 100, X2: 1200, Y2: 600, ClassID: 15, Confidence: 0.72, TrackID: 3, Label: "cat #3"},
		{X1: 1300, Y1: 300, X2: 1600, Y2: 700, ClassID: 5, Confidence: 0.60, TrackID: 4, Label: "bus #4"},
		{X1: 200, Y1: 600, X2: 700, Y2: 900, ClassID: 7, Confidence: 0.55, TrackID: 5, Label: "truck #5"},
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := AnnotateFast(rgb, width, height, objects, 75)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkAnnotateGo_1080p benchmarks pure Go annotation at 1080p with 5 boxes for comparison.
func BenchmarkAnnotateGo_1080p(b *testing.B) {
	width, height := 1920, 1080
	rgb := makeTestRGB(width, height)

	objects := []AnnotateObject{
		{X1: 100, Y1: 100, X2: 400, Y2: 400, ClassID: 0, Confidence: 0.95, TrackID: 1, Label: "person #1"},
		{X1: 500, Y1: 200, X2: 800, Y2: 500, ClassID: 2, Confidence: 0.88, TrackID: 2, Label: "car #2"},
		{X1: 900, Y1: 100, X2: 1200, Y2: 600, ClassID: 15, Confidence: 0.72, TrackID: 3, Label: "cat #3"},
		{X1: 1300, Y1: 300, X2: 1600, Y2: 700, ClassID: 5, Confidence: 0.60, TrackID: 4, Label: "bus #4"},
		{X1: 200, Y1: 600, X2: 700, Y2: 900, ClassID: 7, Confidence: 0.55, TrackID: 5, Label: "truck #5"},
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := Annotate(rgb, width, height, objects)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkResizeJPEG_1080p benchmarks C-accelerated resize of 1080p to 640x360.
func BenchmarkResizeJPEG_1080p(b *testing.B) {
	width, height := 1920, 1080
	rgb := makeTestRGB(width, height)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := ResizeJPEG(rgb, width, height, 640, 360, 75)
		if err != nil {
			b.Fatal(err)
		}
	}
}
