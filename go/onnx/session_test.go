package onnx_test

import (
	"testing"
	"unsafe"

	"github.com/ailakshya/infergo/onnx"
	"github.com/ailakshya/infergo/tensor"
)

// scale.onnx: y = x * 2, input "x" [1,4] float32, output "y" [1,4] float32
const testModel = "/tmp/scale.onnx"

// ─── NewSession ───────────────────────────────────────────────────────────────

func TestNewSession_CpuNotNil(t *testing.T) {
	s, err := onnx.NewSession("cpu", 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	s.Close()
}

func TestNewSession_UnknownProviderFallsBack(t *testing.T) {
	// Unknown provider should fall back to CPU without error
	s, err := onnx.NewSession("xpu", 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	s.Close()
}

func TestNewSession_CudaFallsBackOrSucceeds(t *testing.T) {
	s, err := onnx.NewSession("cuda", 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	s.Close()
}

// ─── Close ────────────────────────────────────────────────────────────────────

func TestClose_IdempotentNoPanic(t *testing.T) {
	s, err := onnx.NewSession("cpu", 0)
	if err != nil {
		t.Fatal(err)
	}
	s.Close()
	s.Close() // second call must be a no-op
}

// ─── Load ─────────────────────────────────────────────────────────────────────

func TestLoad_ScaleModel(t *testing.T) {
	s, err := onnx.NewSession("cpu", 0)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	if err := s.Load(testModel); err != nil {
		t.Fatalf("Load: %v", err)
	}
	if s.NumInputs() != 1 {
		t.Errorf("NumInputs: got %d, want 1", s.NumInputs())
	}
	if s.NumOutputs() != 1 {
		t.Errorf("NumOutputs: got %d, want 1", s.NumOutputs())
	}
}

func TestLoad_BadPathErrors(t *testing.T) {
	s, err := onnx.NewSession("cpu", 0)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	if err := s.Load("/no/such/model.onnx"); err == nil {
		t.Fatal("expected error for bad path, got nil")
	}
}

func TestLoad_ClosedSessionErrors(t *testing.T) {
	s, err := onnx.NewSession("cpu", 0)
	if err != nil {
		t.Fatal(err)
	}
	s.Close()
	if err := s.Load(testModel); err == nil {
		t.Fatal("expected error on closed session, got nil")
	}
}

// ─── InputName / OutputName ───────────────────────────────────────────────────

func TestInputOutputNames(t *testing.T) {
	s, err := onnx.NewSession("cpu", 0)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	if err := s.Load(testModel); err != nil {
		t.Fatal(err)
	}

	name, err := s.InputName(0)
	if err != nil {
		t.Fatalf("InputName: %v", err)
	}
	if name != "x" {
		t.Errorf("InputName(0): got %q, want %q", name, "x")
	}

	name, err = s.OutputName(0)
	if err != nil {
		t.Fatalf("OutputName: %v", err)
	}
	if name != "y" {
		t.Errorf("OutputName(0): got %q, want %q", name, "y")
	}
}

// ─── Run ──────────────────────────────────────────────────────────────────────

func TestRun_ScaleModel(t *testing.T) {
	s, err := onnx.NewSession("cpu", 0)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	if err := s.Load(testModel); err != nil {
		t.Fatal(err)
	}

	// Build input [1,4] float32 = {1, 2, 3, 4}
	in, err := tensor.NewTensorCPU([]int{1, 4}, tensor.Float32)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Free()

	src := []float32{1.0, 2.0, 3.0, 4.0}
	if err := in.CopyFrom(unsafe.Pointer(&src[0]), len(src)*4); err != nil {
		t.Fatal(err)
	}

	outputs, err := s.Run([]*tensor.Tensor{in})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(outputs) != 1 {
		t.Fatalf("len(outputs): got %d, want 1", len(outputs))
	}
	out := outputs[0]
	defer out.Free()

	if out.NBytes() != 16 {
		t.Errorf("output NBytes: got %d, want 16", out.NBytes())
	}
	got := (*[4]float32)(out.DataPtr())
	want := [4]float32{2, 4, 6, 8}
	if *got != want {
		t.Errorf("output data: got %v, want %v", *got, want)
	}
}

func TestRun_BeforeLoadErrors(t *testing.T) {
	s, err := onnx.NewSession("cpu", 0)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	_, err = s.Run([]*tensor.Tensor{})
	if err == nil {
		t.Fatal("expected error when running before Load, got nil")
	}
}

func TestRun_NilInputErrors(t *testing.T) {
	s, err := onnx.NewSession("cpu", 0)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	if err := s.Load(testModel); err != nil {
		t.Fatal(err)
	}

	_, err = s.Run([]*tensor.Tensor{nil})
	if err == nil {
		t.Fatal("expected error for nil input, got nil")
	}
}

func TestRun_ClosedSessionErrors(t *testing.T) {
	s, err := onnx.NewSession("cpu", 0)
	if err != nil {
		t.Fatal(err)
	}
	s.Close()

	_, err = s.Run([]*tensor.Tensor{})
	if err == nil {
		t.Fatal("expected error on closed session, got nil")
	}
}

func TestRun_OutputsOwnedByCaller(t *testing.T) {
	s, err := onnx.NewSession("cpu", 0)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	if err := s.Load(testModel); err != nil {
		t.Fatal(err)
	}

	src := []float32{5.0, 5.0, 5.0, 5.0}
	for i := 0; i < 5; i++ {
		in, err := tensor.NewTensorCPU([]int{1, 4}, tensor.Float32)
		if err != nil {
			t.Fatal(err)
		}
		if err := in.CopyFrom(unsafe.Pointer(&src[0]), 16); err != nil {
			t.Fatal(err)
		}
		outs, err := s.Run([]*tensor.Tensor{in})
		if err != nil {
			t.Fatal(err)
		}
		for _, o := range outs {
			o.Free()
		}
		in.Free()
	}
}
