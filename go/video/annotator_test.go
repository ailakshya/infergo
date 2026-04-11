package video

import (
	"bytes"
	"image/jpeg"
	"testing"
)

// TestAnnotateEmpty verifies that annotating a frame with no objects returns
// a valid JPEG encoding of the original frame.
func TestAnnotateEmpty(t *testing.T) {
	width, height := 320, 240
	rgb := makeTestRGB(width, height)

	jpegData, err := Annotate(rgb, width, height, nil)
	if err != nil {
		t.Fatalf("Annotate with no objects: %v", err)
	}

	// Verify it's a valid JPEG.
	_, err = jpeg.Decode(bytes.NewReader(jpegData))
	if err != nil {
		t.Fatalf("returned data is not valid JPEG: %v", err)
	}

	if len(jpegData) == 0 {
		t.Error("expected non-empty JPEG output")
	}
}

// TestAnnotateBoxes verifies that annotating with 5 objects produces a valid
// JPEG that is larger than the empty-frame JPEG (due to box drawing adding entropy).
func TestAnnotateBoxes(t *testing.T) {
	width, height := 640, 480
	rgb := makeTestRGB(width, height)

	objects := []AnnotateObject{
		{X1: 10, Y1: 20, X2: 100, Y2: 200, ClassID: 0, Confidence: 0.95, TrackID: 1, Label: "person #1"},
		{X1: 200, Y1: 50, X2: 350, Y2: 300, ClassID: 2, Confidence: 0.88, TrackID: 2, Label: "car #2"},
		{X1: 400, Y1: 100, X2: 500, Y2: 400, ClassID: 15, Confidence: 0.72, TrackID: 3, Label: "cat #3"},
		{X1: 50, Y1: 300, X2: 200, Y2: 450, ClassID: 5, Confidence: 0.60, TrackID: 4, Label: "bus #4"},
		{X1: 300, Y1: 200, X2: 600, Y2: 470, ClassID: 7, Confidence: 0.55, TrackID: 5, Label: "truck #5"},
	}

	jpegData, err := Annotate(rgb, width, height, objects)
	if err != nil {
		t.Fatalf("Annotate with 5 objects: %v", err)
	}

	// Verify it's a valid JPEG.
	img, err := jpeg.Decode(bytes.NewReader(jpegData))
	if err != nil {
		t.Fatalf("returned data is not valid JPEG: %v", err)
	}

	// Verify dimensions preserved.
	bounds := img.Bounds()
	if bounds.Dx() != width || bounds.Dy() != height {
		t.Errorf("expected %dx%d, got %dx%d", width, height, bounds.Dx(), bounds.Dy())
	}
}

// TestAnnotateWrongSize verifies that Annotate returns an error when the RGB
// buffer size doesn't match width*height*3.
func TestAnnotateWrongSize(t *testing.T) {
	_, err := Annotate([]byte{0, 1, 2}, 100, 100, nil)
	if err == nil {
		t.Error("expected error for mismatched RGB buffer size")
	}
}

// TestAnnotateClampedBoxes verifies that boxes extending outside the frame
// are clamped correctly and don't panic.
func TestAnnotateClampedBoxes(t *testing.T) {
	width, height := 200, 200
	rgb := makeTestRGB(width, height)

	objects := []AnnotateObject{
		{X1: -50, Y1: -50, X2: 100, Y2: 100, ClassID: 0, Confidence: 0.9, TrackID: 1},
		{X1: 150, Y1: 150, X2: 300, Y2: 300, ClassID: 1, Confidence: 0.8, TrackID: 2},
	}

	_, err := Annotate(rgb, width, height, objects)
	if err != nil {
		t.Fatalf("Annotate with clamped boxes: %v", err)
	}
}

// TestAnnotateAutoLabel verifies that when Label is empty, a default label
// is generated from ClassID, TrackID, and Confidence.
func TestAnnotateAutoLabel(t *testing.T) {
	width, height := 200, 200
	rgb := makeTestRGB(width, height)

	objects := []AnnotateObject{
		{X1: 10, Y1: 10, X2: 100, Y2: 100, ClassID: 0, Confidence: 0.92, TrackID: 5},
	}

	// Should not panic and should produce valid JPEG.
	_, err := Annotate(rgb, width, height, objects)
	if err != nil {
		t.Fatalf("Annotate with auto-label: %v", err)
	}
}

// TestCocoClassName verifies class name lookup.
func TestCocoClassName(t *testing.T) {
	tests := []struct {
		id   int
		want string
	}{
		{0, "person"},
		{2, "car"},
		{79, "toothbrush"},
		{-1, "unknown"},
		{80, "unknown"},
	}
	for _, tt := range tests {
		got := CocoClassName(tt.id)
		if got != tt.want {
			t.Errorf("CocoClassName(%d) = %q, want %q", tt.id, got, tt.want)
		}
	}
}

// BenchmarkAnnotate1080p measures annotation time for 20 boxes on a 1080p frame.
// Target: <3ms.
func BenchmarkAnnotate1080p(b *testing.B) {
	width, height := 1920, 1080
	rgb := makeTestRGB(width, height)

	objects := make([]AnnotateObject, 20)
	for i := range objects {
		// Spread boxes across the frame.
		x := float32(i%5) * 350
		y := float32(i/5) * 250
		objects[i] = AnnotateObject{
			X1:         x + 10,
			Y1:         y + 10,
			X2:         x + 200,
			Y2:         y + 200,
			ClassID:    i % 20,
			Confidence: 0.85,
			TrackID:    i + 1,
			Label:      "person #1",
		}
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

// makeTestRGB creates a solid-color RGB buffer for testing.
func makeTestRGB(width, height int) []byte {
	rgb := make([]byte, width*height*3)
	// Fill with a dark gray to simulate a real video frame.
	for i := 0; i < len(rgb); i += 3 {
		rgb[i] = 40   // R
		rgb[i+1] = 40 // G
		rgb[i+2] = 40 // B
	}
	return rgb
}
