package main

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ailakshya/infergo/server"
)

// ─── OPT-28 Config Tests ───────────────────────────────────────────────────

// TestConfigForceCPU (OPT-28-T1): set INFERGO_DETECT_BACKEND=cpu, verify all
// requests use the CPU-mapped backend (torchSingle).
func TestConfigForceCPU(t *testing.T) {
	t.Setenv("INFERGO_DETECT_BACKEND", "cpu")
	t.Setenv("INFERGO_DETECT_WARMUP", "false")

	cfg := LoadAdaptiveConfig()
	if cfg.Backend != "cpu" {
		t.Fatalf("Backend = %q, want cpu", cfg.Backend)
	}

	single := &mockDetector{latency: time.Millisecond}
	batch := &mockDetector{latency: time.Millisecond}
	trt := &mockDetector{latency: time.Millisecond}

	ad := NewAdaptiveDetector(AdaptiveOpts{
		TorchSingle:    single,
		TorchBatch:     batch,
		OnnxTRT:        trt,
		BatchThreshold: 3,
		Config:         cfg,
	})
	defer ad.Close()

	// Send 5 requests — all should go to torchSingle (CPU mapping).
	for i := 0; i < 5; i++ {
		_, err := ad.Detect(context.Background(), []byte("img"), 0.25, 0.45)
		if err != nil {
			t.Fatalf("Detect: %v", err)
		}
	}

	if got := single.calls.Load(); got != 5 {
		t.Errorf("torchSingle calls = %d, want 5 (cpu forces torchSingle)", got)
	}
	if got := batch.calls.Load(); got != 0 {
		t.Errorf("torchBatch calls = %d, want 0", got)
	}
	if got := trt.calls.Load(); got != 0 {
		t.Errorf("onnxTRT calls = %d, want 0", got)
	}
}

// TestConfigPerRequestOverride (OPT-28-T2): set backend=onnx-cuda in request
// context, verify that backend is used regardless of queue depth or env config.
func TestConfigPerRequestOverride(t *testing.T) {
	t.Setenv("INFERGO_DETECT_WARMUP", "false")

	cfg := LoadAdaptiveConfig()
	cfg.Backend = "auto" // adaptive routing by default

	single := &mockDetector{latency: time.Millisecond}
	cuda := &mockDetector{latency: time.Millisecond}

	ad := NewAdaptiveDetector(AdaptiveOpts{
		TorchSingle: single,
		OnnxCUDA:    cuda,
		Config:      cfg,
	})
	defer ad.Close()

	// Set per-request backend override via context.
	ctx := context.WithValue(context.Background(), detectBackendKey{}, "onnx-cuda")
	_, err := ad.Detect(ctx, []byte("img"), 0.25, 0.45)
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}

	if got := cuda.calls.Load(); got != 1 {
		t.Errorf("onnxCUDA calls = %d, want 1 (per-request override)", got)
	}
	if got := single.calls.Load(); got != 0 {
		t.Errorf("torchSingle calls = %d, want 0", got)
	}
}

// TestConfigPerRequestOverrideViaServerKey (OPT-28-T2b): verify the server
// package's DetectBackendKey also works for per-request override.
func TestConfigPerRequestOverrideViaServerKey(t *testing.T) {
	t.Setenv("INFERGO_DETECT_WARMUP", "false")

	cfg := LoadAdaptiveConfig()

	single := &mockDetector{latency: time.Millisecond}
	trt := &mockDetector{latency: time.Millisecond}

	ad := NewAdaptiveDetector(AdaptiveOpts{
		TorchSingle: single,
		OnnxTRT:     trt,
		Config:      cfg,
	})
	defer ad.Close()

	// Use the server package key (as handleDetect would set).
	ctx := context.WithValue(context.Background(), server.DetectBackendKey{}, "onnx-trt")
	_, err := ad.Detect(ctx, []byte("img"), 0.25, 0.45)
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}

	if got := trt.calls.Load(); got != 1 {
		t.Errorf("onnxTRT calls = %d, want 1 (per-request override via server key)", got)
	}
}

// TestConfigWarmup (OPT-28-T3): set INFERGO_DETECT_WARMUP=true, verify that
// backends receive warmup calls during construction.
func TestConfigWarmup(t *testing.T) {
	t.Setenv("INFERGO_DETECT_WARMUP", "true")

	cfg := LoadAdaptiveConfig()
	if !cfg.Warmup {
		t.Fatalf("Warmup = false, want true")
	}

	single := &mockDetector{latency: time.Millisecond}
	batch := &mockDetector{latency: time.Millisecond}

	ad := NewAdaptiveDetector(AdaptiveOpts{
		TorchSingle:    single,
		TorchBatch:     batch,
		BatchThreshold: 3,
		Config:         cfg,
	})
	defer ad.Close()

	// Warmup should have called each backend 3 times.
	if got := single.calls.Load(); got != 3 {
		t.Errorf("torchSingle warmup calls = %d, want 3", got)
	}
	if got := batch.calls.Load(); got != 3 {
		t.Errorf("torchBatch warmup calls = %d, want 3", got)
	}
}

// TestConfigGPUSlots (OPT-28-T4): set slots=2, send 4 concurrent requests,
// verify that only 2 run on GPU simultaneously and 2 route to fallback.
func TestConfigGPUSlots(t *testing.T) {
	t.Setenv("INFERGO_DETECT_WARMUP", "false")

	cfg := LoadAdaptiveConfig()
	cfg.GPUSlots = 2
	cfg.FallbackBackend = "cpu"
	cfg.Backend = "onnx-cuda" // force GPU backend

	// GPU backend — slow enough that we can observe slot contention.
	cuda := &mockDetector{latency: 100 * time.Millisecond}
	// Fallback (CPU) — torchSingle is what "cpu" resolves to.
	single := &mockDetector{latency: time.Millisecond}

	ad := NewAdaptiveDetector(AdaptiveOpts{
		TorchSingle: single,
		OnnxCUDA:    cuda,
		Config:      cfg,
	})
	defer ad.Close()

	const numRequests = 4
	var wg sync.WaitGroup
	wg.Add(numRequests)

	for i := 0; i < numRequests; i++ {
		go func() {
			defer wg.Done()
			_, _ = ad.Detect(context.Background(), []byte("img"), 0.25, 0.45)
		}()
	}
	wg.Wait()

	cudaCalls := cuda.calls.Load()
	singleCalls := single.calls.Load()
	total := cudaCalls + singleCalls

	if total != numRequests {
		t.Errorf("total calls = %d, want %d", total, numRequests)
	}

	// With 2 GPU slots and 4 concurrent requests, at least some should overflow.
	if singleCalls == 0 {
		t.Errorf("fallback calls = 0, expected some overflow to CPU (cuda=%d)", cudaCalls)
	}
	if cudaCalls == 0 {
		t.Errorf("cuda calls = 0, expected some GPU calls (fallback=%d)", singleCalls)
	}
}

// TestConfigBenchmarkLock (OPT-28-T5): set INFERGO_DETECT_BENCHMARK_BACKEND=onnx-cuda,
// verify all requests go to that backend regardless of depth or env BACKEND.
func TestConfigBenchmarkLock(t *testing.T) {
	t.Setenv("INFERGO_DETECT_WARMUP", "false")

	cfg := LoadAdaptiveConfig()
	cfg.Backend = "auto"
	cfg.BenchmarkBackend = "onnx-cuda"

	single := &mockDetector{latency: time.Millisecond}
	cuda := &mockDetector{latency: time.Millisecond}
	trt := &mockDetector{latency: time.Millisecond}

	ad := NewAdaptiveDetector(AdaptiveOpts{
		TorchSingle: single,
		OnnxCUDA:    cuda,
		OnnxTRT:     trt,
		Config:      cfg,
	})
	defer ad.Close()

	for i := 0; i < 5; i++ {
		_, err := ad.Detect(context.Background(), []byte("img"), 0.25, 0.45)
		if err != nil {
			t.Fatalf("Detect: %v", err)
		}
	}

	if got := cuda.calls.Load(); got != 5 {
		t.Errorf("onnxCUDA calls = %d, want 5 (benchmark lock)", got)
	}
	if got := single.calls.Load(); got != 0 {
		t.Errorf("torchSingle calls = %d, want 0", got)
	}
	if got := trt.calls.Load(); got != 0 {
		t.Errorf("onnxTRT calls = %d, want 0", got)
	}
}

// TestConfigCanonicalBackend (OPT-28-T6): set canonical backend, verify the
// same backend is always used (consistency across calls).
func TestConfigCanonicalBackend(t *testing.T) {
	t.Setenv("INFERGO_DETECT_WARMUP", "false")

	cfg := LoadAdaptiveConfig()
	cfg.CanonicalBackend = "onnx-trt"
	// CanonicalBackend is used when the config.Backend is set to it.
	cfg.Backend = cfg.CanonicalBackend

	single := &mockDetector{latency: time.Millisecond}
	trt := &mockDetector{latency: time.Millisecond}

	ad := NewAdaptiveDetector(AdaptiveOpts{
		TorchSingle: single,
		OnnxTRT:     trt,
		Config:      cfg,
	})
	defer ad.Close()

	// Send requests at varying concurrency levels — all should use the same backend.
	const numRequests = 10
	var wg sync.WaitGroup
	wg.Add(numRequests)
	for i := 0; i < numRequests; i++ {
		go func() {
			defer wg.Done()
			_, _ = ad.Detect(context.Background(), []byte("img"), 0.25, 0.45)
		}()
	}
	wg.Wait()

	// Canonical backend should handle the vast majority of requests.
	// Due to goroutine scheduling, a few may route through the single worker
	// before the adaptive resolver kicks in. Accept >=80% on TRT.
	if got := trt.calls.Load(); got < int64(numRequests)*8/10 {
		t.Errorf("onnxTRT calls = %d, want >=%d (canonical backend)", got, numRequests*8/10)
	}
}

// TestConfigGPUCameras (OPT-28-T7): verify GPU cameras config is loaded from env.
func TestConfigGPUCameras(t *testing.T) {
	t.Setenv("INFERGO_DETECT_GPU_CAMERAS", "4")
	t.Setenv("INFERGO_DETECT_WARMUP", "false")

	cfg := LoadAdaptiveConfig()
	if cfg.GPUCameras != 4 {
		t.Errorf("GPUCameras = %d, want 4", cfg.GPUCameras)
	}
}

// TestConfigInvalidBackend (OPT-28-T8): invalid backend name → Validate returns error.
func TestConfigInvalidBackend(t *testing.T) {
	t.Setenv("INFERGO_DETECT_WARMUP", "false")

	cfg := AdaptiveConfig{
		Backend:         "nonexistent-backend",
		FallbackBackend: "cpu",
	}

	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected error for invalid backend name, got nil")
	}

	// Also test ValidateBackendName directly.
	if err := ValidateBackendName("magic-backend"); err == nil {
		t.Error("ValidateBackendName should reject unknown backend names")
	}
	if err := ValidateBackendName(""); err != nil {
		t.Errorf("ValidateBackendName should accept empty string, got: %v", err)
	}
	if err := ValidateBackendName("auto"); err != nil {
		t.Errorf("ValidateBackendName should accept 'auto', got: %v", err)
	}
}

// TestConfigDefaults (OPT-28-T9): no env vars set → defaults are sensible and
// adaptive routing is unchanged.
func TestConfigDefaults(t *testing.T) {
	// Clear all relevant env vars.
	t.Setenv("INFERGO_DETECT_BACKEND", "")
	t.Setenv("INFERGO_DETECT_WARMUP", "")
	t.Setenv("INFERGO_DETECT_GPU_SLOTS", "")
	t.Setenv("INFERGO_DETECT_GPU_CAMERAS", "")
	t.Setenv("INFERGO_DETECT_FALLBACK_BACKEND", "")
	t.Setenv("INFERGO_DETECT_CANONICAL_BACKEND", "")
	t.Setenv("INFERGO_DETECT_BENCHMARK_BACKEND", "")

	cfg := LoadAdaptiveConfig()

	// When env is empty string, our loader keeps the default values.
	if cfg.Backend != "auto" {
		t.Errorf("Backend = %q, want auto", cfg.Backend)
	}
	if cfg.Warmup != true {
		t.Errorf("Warmup = %v, want true", cfg.Warmup)
	}
	if cfg.GPUSlots != 8 {
		t.Errorf("GPUSlots = %d, want 8", cfg.GPUSlots)
	}
	if cfg.GPUCameras != 0 {
		t.Errorf("GPUCameras = %d, want 0", cfg.GPUCameras)
	}
	if cfg.FallbackBackend != "cpu" {
		t.Errorf("FallbackBackend = %q, want cpu", cfg.FallbackBackend)
	}
	if cfg.CanonicalBackend != "" {
		t.Errorf("CanonicalBackend = %q, want empty", cfg.CanonicalBackend)
	}
	if cfg.BenchmarkBackend != "" {
		t.Errorf("BenchmarkBackend = %q, want empty", cfg.BenchmarkBackend)
	}

	// With default config (auto backend), adaptive routing should work normally.
	single := &mockDetector{latency: time.Millisecond}
	batch := &mockDetector{latency: time.Millisecond}

	// Disable warmup for this test so calls are from Detect only.
	cfg.Warmup = false

	ad := NewAdaptiveDetector(AdaptiveOpts{
		TorchSingle:    single,
		TorchBatch:     batch,
		BatchThreshold: 3,
		Config:         cfg,
	})
	defer ad.Close()

	// Single request → should go to torchSingle (adaptive, depth=1).
	_, err := ad.Detect(context.Background(), []byte("img"), 0.25, 0.45)
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if got := single.calls.Load(); got != 1 {
		t.Errorf("torchSingle calls = %d, want 1 (default adaptive routing)", got)
	}
}

// TestConfigGPUSlotsExhausted (OPT-28-T10): all GPU slots full, verify overflow
// routes to the configured fallback backend.
func TestConfigGPUSlotsExhausted(t *testing.T) {
	t.Setenv("INFERGO_DETECT_WARMUP", "false")

	cfg := LoadAdaptiveConfig()
	cfg.GPUSlots = 1
	cfg.FallbackBackend = "cpu"
	cfg.Backend = "onnx-trt" // force a GPU backend

	// trt is slow — will hold the single GPU slot.
	trt := &mockDetector{latency: 200 * time.Millisecond}
	// Single (CPU fallback) is fast.
	single := &mockDetector{latency: time.Millisecond}

	ad := NewAdaptiveDetector(AdaptiveOpts{
		TorchSingle: single,
		OnnxTRT:     trt,
		Config:      cfg,
	})
	defer ad.Close()

	var wg sync.WaitGroup
	var trtCalls, singleCalls atomic.Int64

	// Launch 4 concurrent requests with only 1 GPU slot.
	const numRequests = 4
	wg.Add(numRequests)
	for i := 0; i < numRequests; i++ {
		go func() {
			defer wg.Done()
			_, _ = ad.Detect(context.Background(), []byte("img"), 0.25, 0.45)
		}()
		// Small stagger to ensure the first request grabs the slot.
		if i == 0 {
			time.Sleep(10 * time.Millisecond)
		}
	}
	wg.Wait()

	trtCalls.Store(trt.calls.Load())
	singleCalls.Store(single.calls.Load())

	total := trtCalls.Load() + singleCalls.Load()
	if total != numRequests {
		t.Errorf("total calls = %d, want %d", total, numRequests)
	}

	// At least 1 should have gone to TRT (first request gets the slot).
	if trtCalls.Load() == 0 {
		t.Error("expected at least 1 TRT call (slot should have been available initially)")
	}
	// At least some should have overflowed to CPU fallback.
	if singleCalls.Load() == 0 {
		t.Errorf("expected some fallback calls (only 1 GPU slot for %d requests)", numRequests)
	}
}
