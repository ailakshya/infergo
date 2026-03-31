package tensor_test

import (
	"testing"
	"unsafe"

	"github.com/ailakshya/infergo/tensor"
)

// BenchmarkAllocFree measures the cost of allocating and freeing a CPU tensor.
func BenchmarkAllocFree(b *testing.B) {
	shape := []int{1024, 1024} // 4 MB float32
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		t, err := tensor.NewTensorCPU(shape, tensor.Float32)
		if err != nil {
			b.Fatal(err)
		}
		t.Free()
	}
}

// BenchmarkCopyToDevice measures H→D transfer for a 4 MB float32 tensor.
func BenchmarkCopyToDevice(b *testing.B) {
	shape := []int{1024, 1024}
	src := make([]float32, 1024*1024)

	// Warm up: verify CUDA is available before starting the timer.
	probe, err := tensor.NewTensorCPU(shape, tensor.Float32)
	if err != nil {
		b.Fatal(err)
	}
	if err := probe.ToDevice(0); err != nil {
		b.Skipf("CUDA not available: %v", err)
	}
	probe.Free()

	b.ResetTimer()
	b.SetBytes(int64(len(src) * 4))
	for i := 0; i < b.N; i++ {
		t, err := tensor.NewTensorCPU(shape, tensor.Float32)
		if err != nil {
			b.Fatal(err)
		}
		if err := t.CopyFrom(unsafe.Pointer(&src[0]), len(src)*4); err != nil {
			b.Fatal(err)
		}
		if err := t.ToDevice(0); err != nil {
			b.Fatal(err)
		}
		t.Free()
	}
}

// BenchmarkCopyToHost measures D→H transfer for a 4 MB float32 tensor.
func BenchmarkCopyToHost(b *testing.B) {
	shape := []int{1024, 1024}
	src := make([]float32, 1024*1024)

	// Warm up: verify CUDA is available before starting the timer.
	probe, err := tensor.NewTensorCPU(shape, tensor.Float32)
	if err != nil {
		b.Fatal(err)
	}
	if err := probe.ToDevice(0); err != nil {
		b.Skipf("CUDA not available: %v", err)
	}
	probe.Free()

	b.ResetTimer()
	b.SetBytes(int64(len(src) * 4))
	for i := 0; i < b.N; i++ {
		t, err := tensor.NewTensorCPU(shape, tensor.Float32)
		if err != nil {
			b.Fatal(err)
		}
		if err := t.CopyFrom(unsafe.Pointer(&src[0]), len(src)*4); err != nil {
			b.Fatal(err)
		}
		if err := t.ToDevice(0); err != nil {
			b.Fatal(err)
		}
		if err := t.ToHost(); err != nil {
			b.Fatal(err)
		}
		t.Free()
	}
}
