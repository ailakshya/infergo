package onnx_test

import (
	"sync"
	"testing"
	"unsafe"

	"github.com/ailakshya/infergo/onnx"
	"github.com/ailakshya/infergo/tensor"
)

const (
	miniLMModel = "/home/lakshya/cgo/models/all-MiniLM-L6-v2/onnx/model.onnx"
	yoloModel   = "/home/lakshya/cgo/models/yolov8n.onnx"
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

// ─── T2: all-MiniLM-L6-v2 embedding shape ────────────────────────────────────

// TestRun_MiniLM_OutputShape verifies that the all-MiniLM-L6-v2 ONNX model
// runs end-to-end and produces last_hidden_state with the expected last
// dimension of 384.
func TestRun_MiniLM_OutputShape(t *testing.T) {
	s, err := onnx.NewSession("cpu", 0)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	if err := s.Load(miniLMModel); err != nil {
		t.Skipf("miniLM model not available: %v", err)
	}

	// Inputs: input_ids, attention_mask, token_type_ids — all [1, 8] int64
	const batchSize, seqLen = 1, 8
	shape := []int{batchSize, seqLen}
	// Zero-initialize all inputs — tensor_alloc_cpu uses malloc (not calloc),
	// so we must explicitly clear before setting individual elements.
	zeros := make([]int64, batchSize*seqLen)
	inputs := make([]*tensor.Tensor, 3)
	for i := range inputs {
		inp, err := tensor.NewTensorCPU(shape, tensor.Int64)
		if err != nil {
			t.Fatalf("NewTensorCPU input[%d]: %v", i, err)
		}
		if err := inp.CopyFrom(unsafe.Pointer(&zeros[0]), len(zeros)*8); err != nil {
			t.Fatalf("zero-init input[%d]: %v", i, err)
		}
		inputs[i] = inp
	}
	defer func() {
		for _, inp := range inputs {
			inp.Free()
		}
	}()

	// Set input_ids to 101 (CLS) + 102 (SEP), rest zero (padding)
	ids := (*[8]int64)(inputs[0].DataPtr())
	ids[0] = 101
	ids[1] = 102
	// attention_mask: 1 for real tokens, 0 for padding
	mask := (*[8]int64)(inputs[1].DataPtr())
	mask[0] = 1
	mask[1] = 1
	// token_type_ids: all zeros (single-sentence, no segment B)

	outputs, err := s.Run(inputs)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	defer func() {
		for _, o := range outputs {
			o.Free()
		}
	}()

	if len(outputs) != 1 {
		t.Fatalf("expected 1 output, got %d", len(outputs))
	}

	shapeOut := outputs[0].Shape()
	if len(shapeOut) != 3 {
		t.Fatalf("expected 3-dim output, got ndim=%d shape=%v", len(shapeOut), shapeOut)
	}
	if shapeOut[0] != 1 || shapeOut[1] != seqLen || shapeOut[2] != 384 {
		t.Errorf("output shape: got %v, want [1, %d, 384]", shapeOut, seqLen)
	}
}

// ─── T3: yolov8n detection shape ─────────────────────────────────────────────

// TestRun_YOLOv8n_OutputShape verifies that yolov8n on a 640×640 zero image
// produces output shape [1, 84, 8400].
func TestRun_YOLOv8n_OutputShape(t *testing.T) {
	s, err := onnx.NewSession("cpu", 0)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	if err := s.Load(yoloModel); err != nil {
		t.Skipf("yolov8n model not available: %v", err)
	}

	// Input: [1, 3, 640, 640] float32 zeros
	inp, err := tensor.NewTensorCPU([]int{1, 3, 640, 640}, tensor.Float32)
	if err != nil {
		t.Fatal(err)
	}
	defer inp.Free()

	outputs, err := s.Run([]*tensor.Tensor{inp})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	defer func() {
		for _, o := range outputs {
			o.Free()
		}
	}()

	if len(outputs) != 1 {
		t.Fatalf("expected 1 output, got %d", len(outputs))
	}

	shapeOut := outputs[0].Shape()
	if len(shapeOut) != 3 || shapeOut[0] != 1 || shapeOut[1] != 84 || shapeOut[2] != 8400 {
		t.Errorf("output shape: got %v, want [1, 84, 8400]", shapeOut)
	}
}

// ─── T7: Concurrent sessions ──────────────────────────────────────────────────

// TestRun_ConcurrentSessions runs 4 goroutines each creating a session and
// calling Run simultaneously — verifies no crash, no data race.
func TestRun_ConcurrentSessions(t *testing.T) {
	var wg sync.WaitGroup
	errs := make(chan error, 4)

	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			s, err := onnx.NewSession("cpu", 0)
			if err != nil {
				errs <- err
				return
			}
			defer s.Close()

			if err := s.Load(testModel); err != nil {
				errs <- err
				return
			}

			in, err := tensor.NewTensorCPU([]int{1, 4}, tensor.Float32)
			if err != nil {
				errs <- err
				return
			}
			defer in.Free()

			src := []float32{1.0, 2.0, 3.0, 4.0}
			if err := in.CopyFrom(unsafe.Pointer(&src[0]), 16); err != nil {
				errs <- err
				return
			}

			outs, err := s.Run([]*tensor.Tensor{in})
			if err != nil {
				errs <- err
				return
			}
			for _, o := range outs {
				o.Free()
			}
		}()
	}

	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("concurrent run error: %v", err)
	}
}
