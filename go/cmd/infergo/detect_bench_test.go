package main

import (
	"os"
	"testing"

	"github.com/ailakshya/infergo/onnx"
)

func loadTestModel(b *testing.B, modelPath string) *detectionAdapter {
	b.Helper()
	sess, err := onnx.NewSession("cuda", 0)
	if err != nil {
		b.Fatalf("NewSession: %v", err)
	}
	if err := sess.Load(modelPath); err != nil {
		sess.Close()
		b.Fatalf("Load: %v", err)
	}
	return &detectionAdapter{sess: sess}
}

func makeTestJPEG() []byte {
	data, err := os.ReadFile("/tmp/test_detect.jpg")
	if err != nil {
		return nil
	}
	return data
}

func benchDirect(b *testing.B, modelPath string) {
	imgData := makeTestJPEG()
	if imgData == nil {
		b.Skip("No test image at /tmp/test_detect.jpg")
	}
	adapter := loadTestModel(b, modelPath)
	defer adapter.sess.Close()

	for i := 0; i < 10; i++ {
		adapter.Detect(nil, imgData, 0.25, 0.45)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := adapter.Detect(nil, imgData, 0.25, 0.45)
		if err != nil {
			b.Fatalf("Detect: %v", err)
		}
	}
}

func BenchmarkDetectDirect_yolo11n(b *testing.B) {
	benchDirect(b, "/home/lakshya/cgo/models/yolo11n.onnx")
}
func BenchmarkDetectDirect_yolo11s(b *testing.B) {
	benchDirect(b, "/home/lakshya/cgo/models/yolo11s.onnx")
}
func BenchmarkDetectDirect_yolo11m(b *testing.B) {
	benchDirect(b, "/home/lakshya/cgo/models/yolo11m.onnx")
}
func BenchmarkDetectDirect_yolo11l(b *testing.B) {
	benchDirect(b, "/home/lakshya/cgo/models/yolo11l.onnx")
}
