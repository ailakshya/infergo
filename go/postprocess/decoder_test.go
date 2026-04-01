package postprocess_test

import (
	"math"
	"testing"
	"unsafe"

	"github.com/ailakshya/infergo/postprocess"
	"github.com/ailakshya/infergo/tensor"
)

// ─── helpers ─────────────────────────────────────────────────────────────────

func makeFloat32Tensor(shape []int, vals []float32) *tensor.Tensor {
	t, err := tensor.NewTensorCPU(shape, tensor.Float32)
	if err != nil {
		panic(err)
	}
	if err := t.CopyFrom(unsafe.Pointer(&vals[0]), len(vals)*4); err != nil {
		panic(err)
	}
	return t
}

func l2Norm(t *tensor.Tensor) float32 {
	ptr := (*float32)(t.DataPtr())
	n := t.NElements()
	var s float64
	for i := range n {
		v := *(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(ptr)) + uintptr(i)*4))
		s += float64(v) * float64(v)
	}
	return float32(math.Sqrt(s))
}

// ─── Classify ────────────────────────────────────────────────────────────────

func TestClassify_NilTensor(t *testing.T) {
	_, err := postprocess.Classify(nil, 1)
	if err == nil {
		t.Fatal("expected error for nil tensor")
	}
}

func TestClassify_ZeroTopK(t *testing.T) {
	vec := makeFloat32Tensor([]int{4}, []float32{1, 2, 3, 4})
	defer vec.Free()
	_, err := postprocess.Classify(vec, 0)
	if err == nil {
		t.Fatal("expected error for topK=0")
	}
}

func TestClassify_UniformLogitsEqualProbs(t *testing.T) {
	vec := makeFloat32Tensor([]int{4}, []float32{0, 0, 0, 0})
	defer vec.Free()
	results, err := postprocess.Classify(vec, 4)
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if len(results) != 4 {
		t.Fatalf("expected 4 results, got %d", len(results))
	}
	for _, r := range results {
		if math.Abs(float64(r.Confidence)-0.25) > 1e-5 {
			t.Errorf("expected confidence=0.25, got %v", r.Confidence)
		}
	}
}

func TestClassify_DominantLogit(t *testing.T) {
	vec := makeFloat32Tensor([]int{4}, []float32{100, 0, 0, 0})
	defer vec.Free()
	results, err := postprocess.Classify(vec, 1)
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].ClassID != 0 {
		t.Errorf("expected ClassID=0, got %d", results[0].ClassID)
	}
	if math.Abs(float64(results[0].Confidence)-1.0) > 1e-5 {
		t.Errorf("expected confidence≈1.0, got %v", results[0].Confidence)
	}
}

func TestClassify_SortedDescending(t *testing.T) {
	vec := makeFloat32Tensor([]int{4}, []float32{1, 5, 3, 2})
	defer vec.Free()
	results, err := postprocess.Classify(vec, 4)
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if results[0].ClassID != 1 {
		t.Errorf("expected top class=1, got %d", results[0].ClassID)
	}
	for i := 1; i < len(results); i++ {
		if results[i].Confidence > results[i-1].Confidence {
			t.Errorf("results not sorted at position %d", i)
		}
	}
}

// ─── NMS ─────────────────────────────────────────────────────────────────────

// makePredsTensor builds a [1, N, stride] float32 tensor from flat rows.
func makePredsTensor(rows [][]float32) *tensor.Tensor {
	N := len(rows)
	stride := len(rows[0])
	flat := make([]float32, N*stride)
	for i, row := range rows {
		copy(flat[i*stride:], row)
	}
	return makeFloat32Tensor([]int{1, N, stride}, flat)
}

func TestNMS_NilTensor(t *testing.T) {
	_, err := postprocess.NMS(nil, 0.5, 0.45, 10)
	if err == nil {
		t.Fatal("expected error for nil tensor")
	}
}

func TestNMS_OverlappingBoxesSuppressed(t *testing.T) {
	preds := makePredsTensor([][]float32{
		{100, 100, 80, 80, 0.95, 0.05},
		{102, 102, 80, 80, 0.80, 0.05},
	})
	defer preds.Free()

	boxes, err := postprocess.NMS(preds, 0.5, 0.45, 10)
	if err != nil {
		t.Fatalf("NMS: %v", err)
	}
	if len(boxes) != 1 {
		t.Fatalf("expected 1 box after suppression, got %d", len(boxes))
	}
	if math.Abs(float64(boxes[0].Confidence)-0.95) > 1e-5 {
		t.Errorf("expected confidence≈0.95, got %v", boxes[0].Confidence)
	}
}

func TestNMS_NonOverlappingBothKept(t *testing.T) {
	preds := makePredsTensor([][]float32{
		{100, 100, 40, 40, 0.9},
		{500, 500, 40, 40, 0.8},
	})
	defer preds.Free()

	boxes, err := postprocess.NMS(preds, 0.5, 0.45, 10)
	if err != nil {
		t.Fatalf("NMS: %v", err)
	}
	if len(boxes) != 2 {
		t.Errorf("expected 2 boxes, got %d", len(boxes))
	}
}

func TestNMS_CoordsAreX1Y1X2Y2(t *testing.T) {
	// cx=100 cy=100 w=60 h=40 → x1=70 y1=80 x2=130 y2=120
	preds := makePredsTensor([][]float32{{100, 100, 60, 40, 0.9}})
	defer preds.Free()

	boxes, err := postprocess.NMS(preds, 0.5, 0.45, 1)
	if err != nil {
		t.Fatalf("NMS: %v", err)
	}
	if len(boxes) != 1 {
		t.Fatalf("expected 1 box, got %d", len(boxes))
	}
	b := boxes[0]
	if math.Abs(float64(b.X1)-70) > 1e-4 || math.Abs(float64(b.Y1)-80) > 1e-4 ||
		math.Abs(float64(b.X2)-130) > 1e-4 || math.Abs(float64(b.Y2)-120) > 1e-4 {
		t.Errorf("unexpected coords: %+v", b)
	}
}

// ─── NormalizeEmbedding ───────────────────────────────────────────────────────

func TestNormalizeEmbedding_NilTensor(t *testing.T) {
	err := postprocess.NormalizeEmbedding(nil)
	if err == nil {
		t.Fatal("expected error for nil tensor")
	}
}

func TestNormalizeEmbedding_L2NormIsOne(t *testing.T) {
	vec := makeFloat32Tensor([]int{2}, []float32{3, 4}) // norm=5 → [0.6, 0.8]
	defer vec.Free()

	if err := postprocess.NormalizeEmbedding(vec); err != nil {
		t.Fatalf("NormalizeEmbedding: %v", err)
	}
	if math.Abs(float64(l2Norm(vec))-1.0) > 1e-6 {
		t.Errorf("expected L2 norm=1, got %v", l2Norm(vec))
	}
}

func TestNormalizeEmbedding_ZeroVectorUnchanged(t *testing.T) {
	vec := makeFloat32Tensor([]int{3}, []float32{0, 0, 0})
	defer vec.Free()

	if err := postprocess.NormalizeEmbedding(vec); err != nil {
		t.Fatalf("NormalizeEmbedding: %v", err)
	}
	// Should not NaN or crash; norm stays 0
	if math.IsNaN(float64(l2Norm(vec))) {
		t.Error("NaN after normalizing zero vector")
	}
}

func TestNormalizeEmbedding_InPlace(t *testing.T) {
	vec := makeFloat32Tensor([]int{4}, []float32{1, 2, 3, 4})
	defer vec.Free()

	ptrBefore := vec.DataPtr()
	if err := postprocess.NormalizeEmbedding(vec); err != nil {
		t.Fatalf("NormalizeEmbedding: %v", err)
	}
	if vec.DataPtr() != ptrBefore {
		t.Error("expected in-place modification (same data pointer)")
	}
	if math.Abs(float64(l2Norm(vec))-1.0) > 1e-6 {
		t.Errorf("expected L2 norm=1, got %v", l2Norm(vec))
	}
}
