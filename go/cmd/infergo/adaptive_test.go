package main

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ailakshya/infergo/server"
)

// ─── Mock backend ───────────────────────────────────────────────────────────

// mockDetector is a thread-safe mock that records call counts and returns
// configurable latency and errors.
type mockDetector struct {
	latency time.Duration
	err     error
	calls   atomic.Int64
	closed  atomic.Bool
}

func (m *mockDetector) Detect(ctx context.Context, img []byte, conf, iou float32) ([]server.DetectedObject, error) {
	m.calls.Add(1)
	if m.latency > 0 {
		select {
		case <-time.After(m.latency):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if m.err != nil {
		return nil, m.err
	}
	return []server.DetectedObject{{X1: 1, Y1: 2, X2: 3, Y2: 4, ClassID: 0, Confidence: 0.9}}, nil
}

func (m *mockDetector) Close() {
	m.closed.Store(true)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

// TestAdaptiveSingleRequest verifies that a single isolated request (queue
// depth = 1) is routed to torchSingle.
func TestAdaptiveSingleRequest(t *testing.T) {
	single := &mockDetector{latency: time.Millisecond}
	batch := &mockDetector{latency: time.Millisecond}
	trt := &mockDetector{latency: time.Millisecond}

	ad := NewAdaptiveDetector(AdaptiveOpts{
		TorchSingle:    single,
		TorchBatch:     batch,
		OnnxTRT:        trt,
		BatchThreshold: 3,
	})
	defer ad.Close()

	result, err := ad.Detect(context.Background(), []byte("img"), 0.25, 0.45)
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if len(result) != 1 {
		t.Fatalf("expected 1 detection, got %d", len(result))
	}

	if got := single.calls.Load(); got != 1 {
		t.Errorf("torchSingle calls = %d, want 1", got)
	}
	if got := batch.calls.Load(); got != 0 {
		t.Errorf("torchBatch calls = %d, want 0", got)
	}
	if got := trt.calls.Load(); got != 0 {
		t.Errorf("onnxTRT calls = %d, want 0", got)
	}
}

// TestAdaptiveBurstLoad verifies that under medium concurrency (depth >1
// but <= batchThreshold), requests go to torchBatch.
func TestAdaptiveBurstLoad(t *testing.T) {
	single := &mockDetector{latency: 50 * time.Millisecond}
	batch := &mockDetector{latency: 50 * time.Millisecond}
	trt := &mockDetector{latency: 50 * time.Millisecond}

	ad := NewAdaptiveDetector(AdaptiveOpts{
		TorchSingle:    single,
		TorchBatch:     batch,
		OnnxTRT:        trt,
		BatchThreshold: 3,
		TRTThreshold:   8,
	})
	defer ad.Close()

	const numRequests = 16
	var wg sync.WaitGroup
	wg.Add(numRequests)
	for i := 0; i < numRequests; i++ {
		go func() {
			defer wg.Done()
			_, _ = ad.Detect(context.Background(), []byte("img"), 0.25, 0.45)
		}()
	}
	wg.Wait()

	// With 16 concurrent requests and 50ms latency, most will see depth > 1.
	// The batch backend should get the majority of calls.
	batchCalls := batch.calls.Load()
	trtCalls := trt.calls.Load()
	singleCalls := single.calls.Load()
	total := singleCalls + batchCalls + trtCalls

	if total != numRequests {
		t.Errorf("total calls = %d, want %d", total, numRequests)
	}

	// With 16 goroutines, most should exceed batchThreshold=3,
	// so TRT should get some calls. Batch and TRT together should dominate.
	if batchCalls+trtCalls < int64(numRequests/2) {
		t.Errorf("batch+trt calls = %d, want at least %d (batch=%d, trt=%d, single=%d)",
			batchCalls+trtCalls, numRequests/2, batchCalls, trtCalls, singleCalls)
	}
}

// TestAdaptiveHighLoad verifies that under heavy concurrency (depth > trtThreshold),
// requests are routed to onnxTRT.
func TestAdaptiveHighLoad(t *testing.T) {
	single := &mockDetector{latency: 100 * time.Millisecond}
	batch := &mockDetector{latency: 100 * time.Millisecond}
	trt := &mockDetector{latency: 100 * time.Millisecond}

	ad := NewAdaptiveDetector(AdaptiveOpts{
		TorchSingle:    single,
		TorchBatch:     batch,
		OnnxTRT:        trt,
		BatchThreshold: 3,
		TRTThreshold:   8,
	})
	defer ad.Close()

	const numRequests = 32
	var wg sync.WaitGroup
	wg.Add(numRequests)
	for i := 0; i < numRequests; i++ {
		go func() {
			defer wg.Done()
			_, _ = ad.Detect(context.Background(), []byte("img"), 0.25, 0.45)
		}()
	}
	wg.Wait()

	trtCalls := trt.calls.Load()
	total := single.calls.Load() + batch.calls.Load() + trtCalls

	if total != numRequests {
		t.Errorf("total calls = %d, want %d", total, numRequests)
	}

	// With 32 goroutines and 100ms latency, queue depth will be high.
	// TRT should handle a significant portion.
	if trtCalls < int64(numRequests/4) {
		t.Errorf("onnxTRT calls = %d, want at least %d (25%%)", trtCalls, numRequests/4)
	}
}

// TestAdaptiveFallbackOnError verifies that when the primary backend returns
// an error, the error is propagated (per-request routing, not failover on
// same request). Subsequent requests still get routed normally.
func TestAdaptiveFallbackOnError(t *testing.T) {
	errBackend := errors.New("gpu exploded")
	single := &mockDetector{latency: time.Millisecond, err: errBackend}
	batch := &mockDetector{latency: time.Millisecond}

	ad := NewAdaptiveDetector(AdaptiveOpts{
		TorchSingle:    single,
		TorchBatch:     batch,
		BatchThreshold: 3,
	})
	defer ad.Close()

	// First request: single-image, which errors.
	_, err := ad.Detect(context.Background(), []byte("img"), 0.25, 0.45)
	if !errors.Is(err, errBackend) {
		t.Fatalf("expected errBackend, got: %v", err)
	}

	// Verify error was counted.
	stats := ad.Stats()
	if stats.ErrorCount != 1 {
		t.Errorf("error count = %d, want 1", stats.ErrorCount)
	}
	if stats.RequestCount != 1 {
		t.Errorf("request count = %d, want 1", stats.RequestCount)
	}
}

// TestAdaptiveAllNil verifies that with no backends, Detect returns an error.
func TestAdaptiveAllNil(t *testing.T) {
	ad := NewAdaptiveDetector(AdaptiveOpts{})
	defer ad.Close()

	_, err := ad.Detect(context.Background(), []byte("img"), 0.25, 0.45)
	if err == nil {
		t.Fatal("expected error with no backends, got nil")
	}
	if err.Error() != "adaptive: no detection backends available" {
		t.Errorf("unexpected error message: %s", err.Error())
	}
}

// TestAdaptiveStats verifies queue depth tracking and P50 calculation.
func TestAdaptiveStats(t *testing.T) {
	single := &mockDetector{latency: 10 * time.Millisecond}

	ad := NewAdaptiveDetector(AdaptiveOpts{
		TorchSingle: single,
	})
	defer ad.Close()

	// Run a few requests to establish metrics.
	for i := 0; i < 5; i++ {
		_, err := ad.Detect(context.Background(), []byte("img"), 0.25, 0.45)
		if err != nil {
			t.Fatalf("Detect: %v", err)
		}
	}

	stats := ad.Stats()
	if stats.RequestCount != 5 {
		t.Errorf("request count = %d, want 5", stats.RequestCount)
	}
	if stats.ErrorCount != 0 {
		t.Errorf("error count = %d, want 0", stats.ErrorCount)
	}
	if stats.ErrorRate != 0 {
		t.Errorf("error rate = %f, want 0", stats.ErrorRate)
	}
	// P50 should be in the ballpark of 10ms = 10000us (with some variance).
	if stats.RollingP50Us < 5000 || stats.RollingP50Us > 50000 {
		t.Errorf("rolling P50 = %d us, expected roughly 10000us", stats.RollingP50Us)
	}
	if stats.QueueDepth != 0 {
		t.Errorf("queue depth = %d, want 0 (no in-flight requests)", stats.QueueDepth)
	}
	if stats.ActiveBackend != "torch-single" {
		t.Errorf("active backend = %q, want torch-single (depth=0)", stats.ActiveBackend)
	}
}

// TestAdaptiveStatsQueueDepthDuringFlight verifies that queue depth is
// incremented while requests are in progress.
func TestAdaptiveStatsQueueDepthDuringFlight(t *testing.T) {
	// Use a long latency so we can observe in-flight state.
	single := &mockDetector{latency: 200 * time.Millisecond}

	ad := NewAdaptiveDetector(AdaptiveOpts{
		TorchSingle: single,
	})
	defer ad.Close()

	var wg sync.WaitGroup
	wg.Add(3)
	started := make(chan struct{}, 3)

	for i := 0; i < 3; i++ {
		go func() {
			defer wg.Done()
			started <- struct{}{}
			_, _ = ad.Detect(context.Background(), []byte("img"), 0.25, 0.45)
		}()
	}

	// Wait for all goroutines to start.
	for i := 0; i < 3; i++ {
		<-started
	}
	// Small sleep to let them enter Detect and increment queueDepth.
	time.Sleep(20 * time.Millisecond)

	depth := ad.queueDepth.Load()
	if depth < 1 {
		t.Errorf("queue depth during flight = %d, want >= 1", depth)
	}

	wg.Wait()

	// After completion, depth should be 0.
	if got := ad.queueDepth.Load(); got != 0 {
		t.Errorf("queue depth after completion = %d, want 0", got)
	}
}

// TestAdaptiveClose verifies that Close is called on all non-nil backends.
func TestAdaptiveClose(t *testing.T) {
	single := &mockDetector{}
	batch := &mockDetector{}
	trt := &mockDetector{}
	cuda := &mockDetector{}

	ad := NewAdaptiveDetector(AdaptiveOpts{
		TorchSingle: single,
		TorchBatch:  batch,
		OnnxTRT:     trt,
		OnnxCUDA:    cuda,
	})

	ad.Close()

	if !single.closed.Load() {
		t.Error("torchSingle was not closed")
	}
	if !batch.closed.Load() {
		t.Error("torchBatch was not closed")
	}
	if !trt.closed.Load() {
		t.Error("onnxTRT was not closed")
	}
	if !cuda.closed.Load() {
		t.Error("onnxCUDA was not closed")
	}
}

// TestAdaptiveClosePartial verifies that Close only calls Close on non-nil backends.
func TestAdaptiveClosePartial(t *testing.T) {
	single := &mockDetector{}

	ad := NewAdaptiveDetector(AdaptiveOpts{
		TorchSingle: single,
	})

	// Should not panic even though other backends are nil.
	ad.Close()

	if !single.closed.Load() {
		t.Error("torchSingle was not closed")
	}
}

// TestAdaptiveFallbackChain verifies that when the preferred backend is nil,
// the detector falls through to the next available backend.
func TestAdaptiveFallbackChain(t *testing.T) {
	t.Run("no single, falls to batch", func(t *testing.T) {
		batch := &mockDetector{latency: time.Millisecond}

		ad := NewAdaptiveDetector(AdaptiveOpts{
			TorchBatch:     batch,
			BatchThreshold: 3,
		})
		defer ad.Close()

		_, err := ad.Detect(context.Background(), []byte("img"), 0.25, 0.45)
		if err != nil {
			t.Fatalf("Detect: %v", err)
		}
		if got := batch.calls.Load(); got != 1 {
			t.Errorf("batch calls = %d, want 1", got)
		}
	})

	t.Run("only onnx cuda", func(t *testing.T) {
		cuda := &mockDetector{latency: time.Millisecond}

		ad := NewAdaptiveDetector(AdaptiveOpts{
			OnnxCUDA: cuda,
		})
		defer ad.Close()

		_, err := ad.Detect(context.Background(), []byte("img"), 0.25, 0.45)
		if err != nil {
			t.Fatalf("Detect: %v", err)
		}
		if got := cuda.calls.Load(); got != 1 {
			t.Errorf("onnxCUDA calls = %d, want 1", got)
		}
	})

	t.Run("only onnx trt", func(t *testing.T) {
		trt := &mockDetector{latency: time.Millisecond}

		ad := NewAdaptiveDetector(AdaptiveOpts{
			OnnxTRT: trt,
		})
		defer ad.Close()

		_, err := ad.Detect(context.Background(), []byte("img"), 0.25, 0.45)
		if err != nil {
			t.Fatalf("Detect: %v", err)
		}
		if got := trt.calls.Load(); got != 1 {
			t.Errorf("onnxTRT calls = %d, want 1", got)
		}
	})
}

// TestAdaptiveContextCancellation verifies that a cancelled context is
// respected and does not hang.
func TestAdaptiveContextCancellation(t *testing.T) {
	// Backend with a long latency.
	single := &mockDetector{latency: 5 * time.Second}

	ad := NewAdaptiveDetector(AdaptiveOpts{
		TorchSingle: single,
	})
	defer ad.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := ad.Detect(ctx, []byte("img"), 0.25, 0.45)
	if err == nil {
		t.Fatal("expected context deadline error, got nil")
	}
}

// ─── Single worker tests ────────────────────────────────────────────────────

// TestSingleWorkerRouting verifies that a low-depth request (depth <= 1)
// to torchSingle is routed through the persistent CUDA worker goroutine.
// We verify this by checking that torchSingle gets called and returns the
// correct result.
func TestSingleWorkerRouting(t *testing.T) {
	single := &mockDetector{latency: time.Millisecond}
	batch := &mockDetector{latency: time.Millisecond}

	ad := NewAdaptiveDetector(AdaptiveOpts{
		TorchSingle:    single,
		TorchBatch:     batch,
		BatchThreshold: 3,
	})
	defer ad.Close()

	// Single request at depth 1 → should go through singleWorkerLoop → torchSingle.
	result, err := ad.Detect(context.Background(), []byte("img"), 0.25, 0.45)
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if len(result) != 1 {
		t.Fatalf("expected 1 detection, got %d", len(result))
	}
	if result[0].Confidence != 0.9 {
		t.Errorf("unexpected confidence: %v", result[0].Confidence)
	}

	// Verify torchSingle was called (via the worker).
	if got := single.calls.Load(); got != 1 {
		t.Errorf("torchSingle calls = %d, want 1", got)
	}
	// Verify batch was not called.
	if got := batch.calls.Load(); got != 0 {
		t.Errorf("torchBatch calls = %d, want 0", got)
	}
}

// TestSingleWorkerConcurrency verifies that the persistent worker correctly
// handles 16 concurrent low-depth requests submitted to it.
func TestSingleWorkerConcurrency(t *testing.T) {
	single := &mockDetector{latency: time.Millisecond}

	ad := NewAdaptiveDetector(AdaptiveOpts{
		TorchSingle:    single,
		BatchThreshold: 3,
	})
	defer ad.Close()

	const numRequests = 16
	var wg sync.WaitGroup
	wg.Add(numRequests)

	errs := make(chan error, numRequests)
	results := make(chan []server.DetectedObject, numRequests)

	for i := 0; i < numRequests; i++ {
		go func() {
			defer wg.Done()
			r, err := ad.Detect(context.Background(), []byte("img"), 0.25, 0.45)
			if err != nil {
				errs <- err
			} else {
				results <- r
			}
		}()
	}
	wg.Wait()
	close(errs)
	close(results)

	for err := range errs {
		t.Errorf("unexpected error: %v", err)
	}

	count := 0
	for r := range results {
		count++
		if len(r) != 1 {
			t.Errorf("expected 1 detection, got %d", len(r))
		}
	}
	if count != numRequests {
		t.Errorf("got %d successful results, want %d", count, numRequests)
	}

	// All requests should have been served by torchSingle (some via worker,
	// some directly for depth > 1 which falls through to batch — but since
	// batch is nil, fallback goes to torchSingle directly).
	totalCalls := single.calls.Load()
	if totalCalls != numRequests {
		t.Errorf("torchSingle calls = %d, want %d", totalCalls, numRequests)
	}
}

// TestSingleWorkerClose verifies that the worker goroutine shuts down
// cleanly when Close is called, and does not deadlock.
func TestSingleWorkerClose(t *testing.T) {
	single := &mockDetector{}

	ad := NewAdaptiveDetector(AdaptiveOpts{
		TorchSingle: single,
	})

	// Close should not hang — the worker should exit when the channel is closed.
	done := make(chan struct{})
	go func() {
		ad.Close()
		close(done)
	}()

	select {
	case <-done:
		// success
	case <-time.After(2 * time.Second):
		t.Fatal("Close() timed out — worker goroutine may be deadlocked")
	}
}

// TestSingleWorkerContextCancellation verifies that a cancelled context is
// respected when the request is queued to the worker.
func TestSingleWorkerContextCancellation(t *testing.T) {
	// Backend with a long latency to ensure the context cancels first.
	single := &mockDetector{latency: 5 * time.Second}

	ad := NewAdaptiveDetector(AdaptiveOpts{
		TorchSingle: single,
	})
	defer ad.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := ad.Detect(ctx, []byte("img"), 0.25, 0.45)
	if err == nil {
		t.Fatal("expected context deadline error, got nil")
	}
}

// ─── Batch tuner tests ──────────────────────────────────────────────────────

func TestBatchTunerShrinks(t *testing.T) {
	// Single-image latency 5ms, but batching produces 6ms/image (>10% worse).
	tuner := newBatchTuner(8, 5.0, 1, 16)

	// Feed consistently slow batches.
	for i := 0; i < 20; i++ {
		tuner.recordBatch(4, 24.0) // 6ms/image
	}

	if tuner.maxBatchSize() >= 8 {
		t.Errorf("batch size = %d, expected it to shrink below 8", tuner.maxBatchSize())
	}
}

func TestBatchTunerGrows(t *testing.T) {
	// Single-image latency 5ms, batching produces 3ms/image (>10% better).
	tuner := newBatchTuner(4, 5.0, 1, 16)

	// Feed consistently fast batches.
	for i := 0; i < 20; i++ {
		tuner.recordBatch(4, 12.0) // 3ms/image
	}

	if tuner.maxBatchSize() <= 4 {
		t.Errorf("batch size = %d, expected it to grow above 4", tuner.maxBatchSize())
	}
}

func TestBatchTunerBounds(t *testing.T) {
	tuner := newBatchTuner(1, 5.0, 1, 4)

	// Feed fast batches — should grow but not beyond max.
	for i := 0; i < 50; i++ {
		tuner.recordBatch(4, 8.0) // 2ms/image, well below 5ms single
	}

	if tuner.maxBatchSize() > 4 {
		t.Errorf("batch size = %d, should not exceed max 4", tuner.maxBatchSize())
	}

	// Feed slow batches — should shrink but not below min.
	for i := 0; i < 50; i++ {
		tuner.recordBatch(4, 40.0) // 10ms/image, well above 5ms single
	}

	if tuner.maxBatchSize() < 1 {
		t.Errorf("batch size = %d, should not go below min 1", tuner.maxBatchSize())
	}
}

func TestBatchTunerZeroBatchSize(t *testing.T) {
	tuner := newBatchTuner(4, 5.0, 1, 16)

	// Should not panic or change state.
	tuner.recordBatch(0, 10.0)

	if tuner.maxBatchSize() != 4 {
		t.Errorf("batch size = %d, want 4 (unchanged after zero batch)", tuner.maxBatchSize())
	}
}

func TestBatchTunerStable(t *testing.T) {
	// Batching at roughly the same speed as single — should stay stable.
	tuner := newBatchTuner(4, 5.0, 1, 16)

	for i := 0; i < 20; i++ {
		tuner.recordBatch(4, 20.0) // 5ms/image, same as single
	}

	// Within +-10% of single, so should not change.
	if tuner.maxBatchSize() != 4 {
		t.Errorf("batch size = %d, expected 4 (stable within tolerance)", tuner.maxBatchSize())
	}
}
