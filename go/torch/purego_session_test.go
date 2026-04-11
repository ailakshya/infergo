//go:build !cgo

package torch_test

import (
	"os"
	"testing"

	"github.com/ailakshya/infergo/torch"
)

// ─── NewSession ─────────────────────────────────────────────────────────────

func TestNewSession(t *testing.T) {
	s, err := torch.NewSession("cpu", 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if s == nil {
		t.Fatal("session is nil")
	}
	s.Close()
}

func TestNewSessionCUDA(t *testing.T) {
	s, err := torch.NewSession("cuda", 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if s == nil {
		t.Fatal("session is nil")
	}
	s.Close()
}

// ─── Close ──────────────────────────────────────────────────────────────────

func TestClose(t *testing.T) {
	s, err := torch.NewSession("cpu", 0)
	if err != nil {
		t.Fatal(err)
	}
	s.Close()
	s.Close() // second call must be a no-op, no panic
}

// ─── DetectGPU ──────────────────────────────────────────────────────────────

func TestDetectGPU(t *testing.T) {
	const modelPath = "/home/lakshya/cgo/models/yolo11n.torchscript.pt"
	const imagePath = "/tmp/test_detect.jpg"

	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("model not found: %s", modelPath)
	}
	imgData, err := os.ReadFile(imagePath)
	if err != nil {
		t.Skipf("test image not found: %s", imagePath)
	}

	sess, err := torch.NewSession("cuda", 0)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer sess.Close()

	if err := sess.Load(modelPath); err != nil {
		t.Fatalf("Load: %v", err)
	}

	dets, err := sess.DetectGPU(imgData, 0.25, 0.45)
	if err != nil {
		t.Fatalf("DetectGPU: %v", err)
	}
	t.Logf("DetectGPU returned %d detections", len(dets))
	for i, d := range dets {
		if i >= 5 {
			break
		}
		t.Logf("  [%d] class=%d conf=%.3f box=(%.1f,%.1f)-(%.1f,%.1f)",
			i, d.ClassID, d.Confidence, d.X1, d.Y1, d.X2, d.Y2)
	}
}

// ─── Benchmarks ─────────────────────────────────────────────────────────────

func BenchmarkPuregoDetectGPU(b *testing.B) {
	const modelPath = "/home/lakshya/cgo/models/yolo11n.torchscript.pt"
	if _, err := os.Stat(modelPath); err != nil {
		b.Skipf("model not found: %s", modelPath)
	}
	imgData, err := os.ReadFile("/tmp/test_detect.jpg")
	if err != nil {
		b.Skipf("test image not found: /tmp/test_detect.jpg")
	}

	sess, err := torch.NewSession("cuda", 0)
	if err != nil {
		b.Fatalf("NewSession: %v", err)
	}
	defer sess.Close()

	if err := sess.Load(modelPath); err != nil {
		b.Fatalf("Load: %v", err)
	}

	// Warm up: 10 iterations to let CUDA initialize fully.
	for i := 0; i < 10; i++ {
		sess.DetectGPU(imgData, 0.25, 0.45)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := sess.DetectGPU(imgData, 0.25, 0.45)
		if err != nil {
			b.Fatalf("DetectGPU: %v", err)
		}
	}
}

func BenchmarkPuregoDetectGPUBatch(b *testing.B) {
	const modelPath = "/home/lakshya/cgo/models/yolo11n.torchscript.pt"
	if _, err := os.Stat(modelPath); err != nil {
		b.Skipf("model not found: %s", modelPath)
	}
	imgData, err := os.ReadFile("/tmp/test_detect.jpg")
	if err != nil {
		b.Skipf("test image not found: /tmp/test_detect.jpg")
	}

	sess, err := torch.NewSession("cuda", 0)
	if err != nil {
		b.Fatalf("NewSession: %v", err)
	}
	defer sess.Close()

	if err := sess.Load(modelPath); err != nil {
		b.Fatalf("Load: %v", err)
	}

	batch := [][]byte{imgData, imgData, imgData, imgData}

	// Warm up.
	for i := 0; i < 10; i++ {
		sess.DetectGPUBatch(batch, 0.25, 0.45)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := sess.DetectGPUBatch(batch, 0.25, 0.45)
		if err != nil {
			b.Fatalf("DetectGPUBatch: %v", err)
		}
	}
}
