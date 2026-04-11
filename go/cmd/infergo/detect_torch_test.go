package main

import (
	"os"
	"testing"

	"github.com/ailakshya/infergo/torch"
)

func loadTorchTestModel(b *testing.B, modelPath string) *torchDetectionAdapter {
	b.Helper()
	sess, err := torch.NewSession("cuda", 0)
	if err != nil {
		b.Fatalf("NewSession: %v", err)
	}
	if err := sess.Load(modelPath); err != nil {
		sess.Close()
		b.Fatalf("Load: %v", err)
	}
	return &torchDetectionAdapter{sess: sess}
}

func BenchmarkDetectTorchGPU(b *testing.B) {
	const modelPath = "/home/lakshya/cgo/models/yolo11n.torchscript.pt"
	if _, err := os.Stat(modelPath); err != nil {
		b.Skipf("model not found: %s", modelPath)
	}
	imgData := makeTestJPEG()
	if imgData == nil {
		b.Skip("No test image at /tmp/test_detect.jpg")
	}

	adapter := loadTorchTestModel(b, modelPath)
	defer adapter.sess.Close()

	// Warm up.
	for i := 0; i < 10; i++ {
		adapter.sess.DetectGPU(imgData, 0.25, 0.45)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := adapter.sess.DetectGPU(imgData, 0.25, 0.45)
		if err != nil {
			b.Fatalf("DetectGPU: %v", err)
		}
	}
}

func BenchmarkDetectTorchFallback(b *testing.B) {
	const modelPath = "/home/lakshya/cgo/models/yolo11n.torchscript.pt"
	if _, err := os.Stat(modelPath); err != nil {
		b.Skipf("model not found: %s", modelPath)
	}
	imgData := makeTestJPEG()
	if imgData == nil {
		b.Skip("No test image at /tmp/test_detect.jpg")
	}

	adapter := loadTorchTestModel(b, modelPath)
	defer adapter.sess.Close()

	// Warm up.
	for i := 0; i < 10; i++ {
		adapter.detectFallback(nil, imgData, 0.25, 0.45)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := adapter.detectFallback(nil, imgData, 0.25, 0.45)
		if err != nil {
			b.Fatalf("detectFallback: %v", err)
		}
	}
}

func benchTorchGPU(b *testing.B, modelPath string) {
	imgData := makeTestJPEG()
	if imgData == nil {
		b.Skip("No test image")
	}
	sess, err := torch.NewSession("cuda", 0)
	if err != nil {
		b.Fatalf("NewSession: %v", err)
	}
	defer sess.Close()
	if err := sess.Load(modelPath); err != nil {
		b.Fatalf("Load: %v", err)
	}
	// warmup
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

func BenchmarkDetectTorchGPU_yolo11n(b *testing.B) {
	benchTorchGPU(b, "/home/lakshya/cgo/models/yolo11n.torchscript.pt")
}
func BenchmarkDetectTorchGPU_yolo11s(b *testing.B) {
	benchTorchGPU(b, "/home/lakshya/cgo/models/yolo11s.torchscript.pt")
}
func BenchmarkDetectTorchGPU_yolo11m(b *testing.B) {
	benchTorchGPU(b, "/home/lakshya/cgo/models/yolo11m.torchscript.pt")
}
func BenchmarkDetectTorchGPU_yolo11l(b *testing.B) {
	benchTorchGPU(b, "/home/lakshya/cgo/models/yolo11l.torchscript.pt")
}
