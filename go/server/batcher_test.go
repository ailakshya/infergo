package server_test

import (
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"

	"github.com/ailakshya/infergo/server"
	"github.com/ailakshya/infergo/tensor"
)

// echoProcess is a trivial ProcessFn that returns a new 1-element tensor
// whose value equals the sum of elements in each input tensor.
func echoProcess(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	out := make([]*tensor.Tensor, len(inputs))
	for i, t := range inputs {
		o, err := tensor.NewTensorCPU([]int{1}, tensor.Float32)
		if err != nil {
			return nil, err
		}
		// Read first element from input and copy to output
		src := (*float32)(t.DataPtr())
		dst := (*float32)(o.DataPtr())
		*dst = *src
		out[i] = o
	}
	return out, nil
}

func makeScalarTensor(val float32) *tensor.Tensor {
	t, err := tensor.NewTensorCPU([]int{1}, tensor.Float32)
	if err != nil {
		panic(err)
	}
	if err := t.CopyFrom(unsafe.Pointer(&val), 4); err != nil {
		panic(err)
	}
	return t
}

func readScalar(t *tensor.Tensor) float32 {
	return *(*float32)(t.DataPtr())
}

// ─── Construction ─────────────────────────────────────────────────────────────

func TestNew_InvalidParams(t *testing.T) {
	if _, err := server.New(0, time.Millisecond, echoProcess); err == nil {
		t.Error("expected error for maxBatch=0")
	}
	if _, err := server.New(4, 0, echoProcess); err == nil {
		t.Error("expected error for maxWait=0")
	}
	if _, err := server.New(4, time.Millisecond, nil); err == nil {
		t.Error("expected error for nil process")
	}
}

// ─── Single request ───────────────────────────────────────────────────────────

func TestSubmit_SingleRequest(t *testing.T) {
	s, err := server.New(4, 10*time.Millisecond, echoProcess)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Stop()

	in := makeScalarTensor(42.0)
	defer in.Free()

	out, err := s.Submit(in)
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	defer out.Free()

	if readScalar(out) != 42.0 {
		t.Errorf("expected 42.0, got %v", readScalar(out))
	}
}

// ─── Batch size cap ───────────────────────────────────────────────────────────

func TestBatcher_BatchSizeCap(t *testing.T) {
	const maxBatch = 8
	var maxSeen atomic.Int32

	process := func(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
		n := int32(len(inputs))
		for {
			cur := maxSeen.Load()
			if n <= cur || maxSeen.CompareAndSwap(cur, n) {
				break
			}
		}
		return echoProcess(inputs)
	}

	s, err := server.New(maxBatch, 50*time.Millisecond, process)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Stop()

	const N = 64
	var wg sync.WaitGroup
	wg.Add(N)
	for i := range N {
		go func(v float32) {
			defer wg.Done()
			in := makeScalarTensor(v)
			defer in.Free()
			out, err := s.Submit(in)
			if err != nil {
				t.Errorf("Submit(%v): %v", v, err)
				return
			}
			out.Free()
		}(float32(i))
	}
	wg.Wait()

	if got := maxSeen.Load(); got > maxBatch {
		t.Errorf("batch exceeded maxBatch: got %d, max %d", got, maxBatch)
	}
}

// ─── 100 concurrent requests ─────────────────────────────────────────────────

func TestBatcher_100ConcurrentRequests(t *testing.T) {
	s, err := server.New(16, 20*time.Millisecond, echoProcess)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Stop()

	const N = 100
	errs := make([]error, N)
	outs := make([]*tensor.Tensor, N)
	var wg sync.WaitGroup
	wg.Add(N)

	for i := range N {
		go func(idx int) {
			defer wg.Done()
			val := float32(idx)
			in := makeScalarTensor(val)
			defer in.Free()
			out, err := s.Submit(in)
			errs[idx] = err
			outs[idx] = out
		}(i)
	}
	wg.Wait()

	for i := range N {
		if errs[i] != nil {
			t.Errorf("request[%d] error: %v", i, errs[i])
			continue
		}
		if outs[i] == nil {
			t.Errorf("request[%d]: nil output", i)
			continue
		}
		if readScalar(outs[i]) != float32(i) {
			t.Errorf("request[%d]: expected %v, got %v", i, float32(i), readScalar(outs[i]))
		}
		outs[i].Free()
	}
}

// ─── Timeout flush ────────────────────────────────────────────────────────────

func TestBatcher_TimeoutFlush(t *testing.T) {
	// maxBatch=100 but only 1 request sent — must flush via timeout
	var batchSizes []int
	var mu sync.Mutex

	process := func(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
		mu.Lock()
		batchSizes = append(batchSizes, len(inputs))
		mu.Unlock()
		return echoProcess(inputs)
	}

	s, err := server.New(100, 20*time.Millisecond, process)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Stop()

	in := makeScalarTensor(7.0)
	defer in.Free()
	out, err := s.Submit(in)
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	out.Free()

	mu.Lock()
	defer mu.Unlock()
	if len(batchSizes) != 1 || batchSizes[0] != 1 {
		t.Errorf("expected one batch of size 1, got %v", batchSizes)
	}
}

// ─── Process error propagation ────────────────────────────────────────────────

func TestBatcher_ProcessErrorPropagated(t *testing.T) {
	boom := errors.New("process failed")
	s, err := server.New(4, 10*time.Millisecond, func(_ []*tensor.Tensor) ([]*tensor.Tensor, error) {
		return nil, boom
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Stop()

	in := makeScalarTensor(1.0)
	defer in.Free()
	_, err = s.Submit(in)
	if !errors.Is(err, boom) {
		t.Errorf("expected boom error, got %v", err)
	}
}

// ─── Stop drains pending ─────────────────────────────────────────────────────

func TestBatcher_StopReturnError(t *testing.T) {
	// Slow process so we can stop while requests are in flight
	s, err := server.New(100, 5*time.Millisecond, func(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
		time.Sleep(50 * time.Millisecond)
		return echoProcess(inputs)
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	var wg sync.WaitGroup
	const N = 10
	errs := make([]error, N)
	wg.Add(N)
	for i := range N {
		go func(idx int) {
			defer wg.Done()
			in := makeScalarTensor(float32(idx))
			defer in.Free()
			_, errs[idx] = s.Submit(in)
		}(i)
	}

	time.Sleep(5 * time.Millisecond) // let requests queue up
	s.Stop()
	wg.Wait()

	// At least some requests should have completed (those dispatched before Stop)
	// or received a stop error — none should hang.
	t.Log("all goroutines returned after Stop()")
}
