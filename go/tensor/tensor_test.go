package tensor_test

import (
	"runtime"
	"testing"
	"unsafe"

	"github.com/ailakshya/infergo/tensor"
)

// ─── NewTensorCPU ─────────────────────────────────────────────────────────────

func TestNewTensorCPU_Float32_234(t *testing.T) {
	shape := []int{2, 3, 4}
	ten, err := tensor.NewTensorCPU(shape, tensor.Float32)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer ten.Free()

	if ten.NBytes() != 96 {
		t.Errorf("NBytes: got %d, want 96", ten.NBytes())
	}
	if ten.NElements() != 24 {
		t.Errorf("NElements: got %d, want 24", ten.NElements())
	}
	if ten.DType() != tensor.Float32 {
		t.Errorf("DType: got %v, want Float32", ten.DType())
	}
}

func TestNewTensorCPU_AllDtypes(t *testing.T) {
	cases := []struct {
		dtype     tensor.DType
		elemBytes int
	}{
		{tensor.Float32, 4},
		{tensor.Float16, 2},
		{tensor.BFloat16, 2},
		{tensor.Int32, 4},
		{tensor.Int64, 8},
		{tensor.UInt8, 1},
		{tensor.Bool, 1},
	}
	shape := []int{8}
	for _, c := range cases {
		ten, err := tensor.NewTensorCPU(shape, c.dtype)
		if err != nil {
			t.Errorf("dtype %v: unexpected error: %v", c.dtype, err)
			continue
		}
		want := 8 * c.elemBytes
		if ten.NBytes() != want {
			t.Errorf("dtype %v: NBytes got %d, want %d", c.dtype, ten.NBytes(), want)
		}
		ten.Free()
	}
}

func TestNewTensorCPU_EmptyShapeErrors(t *testing.T) {
	_, err := tensor.NewTensorCPU([]int{}, tensor.Float32)
	if err == nil {
		t.Fatal("expected error for empty shape, got nil")
	}
}

func TestNewTensorCPU_ZeroDimErrors(t *testing.T) {
	_, err := tensor.NewTensorCPU([]int{0, 4}, tensor.Float32)
	if err == nil {
		t.Fatal("expected error for zero dimension, got nil")
	}
}

func TestNewTensorCPU_NegativeDimErrors(t *testing.T) {
	_, err := tensor.NewTensorCPU([]int{-1}, tensor.Float32)
	if err == nil {
		t.Fatal("expected error for negative dimension, got nil")
	}
}

// ─── Free ─────────────────────────────────────────────────────────────────────

func TestFree_IdempotentNoPanic(t *testing.T) {
	ten, err := tensor.NewTensorCPU([]int{4}, tensor.Float32)
	if err != nil {
		t.Fatal(err)
	}
	ten.Free()
	ten.Free() // second call must be a no-op
}

func TestFree_FinalizerDoesNotPanic(t *testing.T) {
	// Allocate and drop reference; GC + finalizer must not crash.
	for i := 0; i < 10; i++ {
		_, err := tensor.NewTensorCPU([]int{16}, tensor.Float32)
		if err != nil {
			t.Fatal(err)
		}
		// deliberately not calling Free — finalizer should handle it
	}
	runtime.GC()
	runtime.GC()
}

// ─── Shape ────────────────────────────────────────────────────────────────────

func TestShape_CorrectValues(t *testing.T) {
	shape := []int{2, 3, 4}
	ten, err := tensor.NewTensorCPU(shape, tensor.Float32)
	if err != nil {
		t.Fatal(err)
	}
	defer ten.Free()

	got := ten.Shape()
	if len(got) != len(shape) {
		t.Fatalf("Shape len: got %d, want %d", len(got), len(shape))
	}
	for i, v := range shape {
		if got[i] != v {
			t.Errorf("Shape[%d]: got %d, want %d", i, got[i], v)
		}
	}
}

func TestShape_OneDim(t *testing.T) {
	ten, err := tensor.NewTensorCPU([]int{100}, tensor.Int64)
	if err != nil {
		t.Fatal(err)
	}
	defer ten.Free()
	got := ten.Shape()
	if len(got) != 1 || got[0] != 100 {
		t.Errorf("Shape: got %v, want [100]", got)
	}
}

// ─── DataPtr ──────────────────────────────────────────────────────────────────

func TestDataPtr_NotNilForCPU(t *testing.T) {
	ten, err := tensor.NewTensorCPU([]int{4}, tensor.Float32)
	if err != nil {
		t.Fatal(err)
	}
	defer ten.Free()
	if ten.DataPtr() == nil {
		t.Error("DataPtr returned nil for CPU tensor")
	}
}

func TestDataPtr_NilAfterFree(t *testing.T) {
	ten, err := tensor.NewTensorCPU([]int{4}, tensor.Float32)
	if err != nil {
		t.Fatal(err)
	}
	ten.Free()
	if ten.DataPtr() != nil {
		t.Error("DataPtr should return nil after Free")
	}
}

// ─── CopyFrom ─────────────────────────────────────────────────────────────────

func TestCopyFrom_Float32RoundTrip(t *testing.T) {
	src := []float32{1.0, 2.0, 3.0, 4.0}
	ten, err := tensor.NewTensorCPU([]int{4}, tensor.Float32)
	if err != nil {
		t.Fatal(err)
	}
	defer ten.Free()

	if err := ten.CopyFrom(unsafe.Pointer(&src[0]), len(src)*4); err != nil {
		t.Fatalf("CopyFrom: %v", err)
	}

	ptr := (*float32)(ten.DataPtr())
	for i, want := range src {
		got := *(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(ptr)) + uintptr(i)*4))
		if got != want {
			t.Errorf("data[%d]: got %f, want %f", i, got, want)
		}
	}
}

func TestCopyFrom_WrongSizeErrors(t *testing.T) {
	ten, err := tensor.NewTensorCPU([]int{4}, tensor.Float32) // 16 bytes
	if err != nil {
		t.Fatal(err)
	}
	defer ten.Free()
	src := []float32{1.0, 2.0}
	err = ten.CopyFrom(unsafe.Pointer(&src[0]), 8) // wrong size
	if err == nil {
		t.Fatal("expected error for wrong nbytes, got nil")
	}
}

func TestCopyFrom_NilSrcErrors(t *testing.T) {
	ten, err := tensor.NewTensorCPU([]int{4}, tensor.Float32)
	if err != nil {
		t.Fatal(err)
	}
	defer ten.Free()
	if err := ten.CopyFrom(nil, 16); err == nil {
		t.Fatal("expected error for nil src, got nil")
	}
}

// ─── CUDA tests ───────────────────────────────────────────────────────────────

func TestNewTensorCUDA_Float32(t *testing.T) {
	ten, err := tensor.NewTensorCUDA([]int{2, 3, 4}, tensor.Float32, 0)
	if err != nil {
		t.Skipf("CUDA not available: %v", err)
	}
	defer ten.Free()

	if ten.NBytes() != 96 {
		t.Errorf("NBytes: got %d, want 96", ten.NBytes())
	}
	if ten.DType() != tensor.Float32 {
		t.Errorf("DType: got %v, want Float32", ten.DType())
	}
}

func TestToDevice_ToHost_RoundTrip(t *testing.T) {
	src := []float32{10.0, 20.0, 30.0, 40.0}
	ten, err := tensor.NewTensorCPU([]int{4}, tensor.Float32)
	if err != nil {
		t.Fatal(err)
	}
	defer ten.Free()

	if err := ten.CopyFrom(unsafe.Pointer(&src[0]), 16); err != nil {
		t.Fatal(err)
	}

	if err := ten.ToDevice(0); err != nil {
		t.Skipf("CUDA not available: %v", err)
	}
	if err := ten.ToHost(); err != nil {
		t.Fatalf("ToHost: %v", err)
	}

	ptr := (*float32)(ten.DataPtr())
	for i, want := range src {
		got := *(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(ptr)) + uintptr(i)*4))
		if got != want {
			t.Errorf("after round-trip data[%d]: got %f, want %f", i, got, want)
		}
	}
}

func TestToHost_NoOpOnCPU(t *testing.T) {
	ten, err := tensor.NewTensorCPU([]int{4}, tensor.Float32)
	if err != nil {
		t.Fatal(err)
	}
	defer ten.Free()
	if err := ten.ToHost(); err != nil {
		t.Errorf("ToHost on CPU tensor should be no-op, got: %v", err)
	}
}
