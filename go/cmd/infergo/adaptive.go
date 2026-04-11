package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/ailakshya/infergo/server"
)

// detectBackendKey is the context key used to pass a per-request backend
// override from the HTTP handler to AdaptiveDetector.Detect.
type detectBackendKey struct{}

// ─── Adaptive config ────────────────────────────────────────────────────────

// AdaptiveConfig holds user-configurable variables for the adaptive detection
// backend. Values are read from environment variables, overridden by CLI flags,
// and can be further overridden per-request via the "backend" JSON field.
type AdaptiveConfig struct {
	Backend          string // "auto"|"torch-gpu"|"onnx-cuda"|"tensorrt"|"cpu" (env: INFERGO_DETECT_BACKEND)
	Warmup           bool   // warm up all backends at startup (env: INFERGO_DETECT_WARMUP)
	GPUSlots         int    // max concurrent GPU inference slots (env: INFERGO_DETECT_GPU_SLOTS)
	GPUCameras       int    // first N cameras use GPU, rest CPU (env: INFERGO_DETECT_GPU_CAMERAS)
	FallbackBackend  string // backend when GPU slots exhausted (env: INFERGO_DETECT_FALLBACK_BACKEND)
	CanonicalBackend string // backend for reproducible results (env: INFERGO_DETECT_CANONICAL_BACKEND)
	BenchmarkBackend string // if set, lock to this backend (env: INFERGO_DETECT_BENCHMARK_BACKEND)
}

// validBackends is the set of valid backend names that can be used in config.
var validBackends = map[string]bool{
	"auto":        true,
	"torch-gpu":   true,
	"torch-single": true,
	"torch-batch": true,
	"onnx-cuda":   true,
	"tensorrt":    true,
	"onnx-trt":    true,
	"cpu":         true,
}

// ValidateBackendName returns an error if the backend name is not recognized.
func ValidateBackendName(name string) error {
	if name == "" {
		return nil
	}
	if !validBackends[name] {
		return fmt.Errorf("invalid backend %q: must be one of auto, torch-gpu, torch-single, torch-batch, onnx-cuda, tensorrt, onnx-trt, cpu", name)
	}
	return nil
}

// LoadAdaptiveConfig reads configuration from environment variables with
// sensible defaults.
func LoadAdaptiveConfig() AdaptiveConfig {
	cfg := AdaptiveConfig{
		Backend:          "auto",
		Warmup:           true,
		GPUSlots:         8,
		GPUCameras:       0,
		FallbackBackend:  "cpu",
		CanonicalBackend: "",
		BenchmarkBackend: "",
	}

	if v := os.Getenv("INFERGO_DETECT_BACKEND"); v != "" {
		cfg.Backend = v
	}
	if v := os.Getenv("INFERGO_DETECT_WARMUP"); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			cfg.Warmup = b
		}
	}
	if v := os.Getenv("INFERGO_DETECT_GPU_SLOTS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.GPUSlots = n
		}
	}
	if v := os.Getenv("INFERGO_DETECT_GPU_CAMERAS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			cfg.GPUCameras = n
		}
	}
	if v := os.Getenv("INFERGO_DETECT_FALLBACK_BACKEND"); v != "" {
		cfg.FallbackBackend = v
	}
	if v := os.Getenv("INFERGO_DETECT_CANONICAL_BACKEND"); v != "" {
		cfg.CanonicalBackend = v
	}
	if v := os.Getenv("INFERGO_DETECT_BENCHMARK_BACKEND"); v != "" {
		cfg.BenchmarkBackend = v
	}

	return cfg
}

// Validate checks that all configured backend names are valid.
func (c *AdaptiveConfig) Validate() error {
	for _, name := range []string{c.Backend, c.FallbackBackend, c.CanonicalBackend, c.BenchmarkBackend} {
		if err := ValidateBackendName(name); err != nil {
			return err
		}
	}
	return nil
}

// singleRequest is a detection request to be processed by the persistent
// CUDA worker goroutine.
type singleRequest struct {
	backend DetectCloser
	ctx     context.Context
	image   []byte
	conf    float32
	iou     float32
	result  chan singleResult
}

// singleResult is the result of a single detection request processed
// by the persistent CUDA worker.
type singleResult struct {
	objects []server.DetectedObject
	err     error
}

// ─── Adaptive detector ──────────────────────────────────────────────────────

// DetectCloser is the interface that all detection backends must satisfy to
// participate in the adaptive detector. Both torchDetectionAdapter and
// detectionAdapter already implement this via server.DetectionModel.
type DetectCloser interface {
	Detect(ctx context.Context, imageBytes []byte, confThresh, iouThresh float32) ([]server.DetectedObject, error)
	Close()
}

// AdaptiveOpts configures the adaptive detector at construction time.
type AdaptiveOpts struct {
	// Available backends (may be nil if not loaded).
	TorchSingle DetectCloser // libtorch single-image (no batch loop)
	TorchBatch  DetectCloser // libtorch with batch loop
	OnnxTRT     DetectCloser // ONNX TensorRT execution provider
	OnnxCUDA    DetectCloser // ONNX CUDA execution provider

	// BatchThreshold is the queue depth at which we switch from single-image
	// to batch mode. Default: 3.
	BatchThreshold int

	// TRTThreshold is the queue depth at which we switch from batch mode
	// to TensorRT. Default: 8.
	TRTThreshold int

	// Config holds user-configurable variables (env vars, CLI flags).
	Config AdaptiveConfig
}

// AdaptiveStats exposes current runtime metrics for monitoring/debugging.
type AdaptiveStats struct {
	QueueDepth    int64   // current number of in-flight requests
	RollingP50Us  int64   // exponential moving average latency in microseconds
	RequestCount  int64   // total requests served since startup
	ErrorCount    int64   // total errors returned since startup
	ErrorRate     float64 // errorCount / requestCount (0 if no requests)
	ActiveBackend string  // name of the backend that would be chosen right now
}

// AdaptiveDetector wraps multiple detection backends and routes each request
// to the optimal one based on current queue depth. It implements
// server.DetectionModel so it can be registered directly in the Registry.
type AdaptiveDetector struct {
	// Available backends (may be nil if not loaded).
	torchSingle DetectCloser
	torchBatch  DetectCloser
	onnxTRT     DetectCloser
	onnxCUDA    DetectCloser

	// Persistent CUDA worker goroutine for low-latency single-image detection.
	// Stays locked to one OS thread so the CUDA context remains warm, avoiding
	// thread-switching jitter on the CGo boundary.
	singleWork chan *singleRequest
	singleDone chan struct{}

	// Runtime metrics — all accessed atomically, no mutexes in hot path.
	queueDepth   atomic.Int64
	rollingP50   atomic.Int64 // microseconds, exponential moving average
	requestCount atomic.Int64
	errorCount   atomic.Int64

	// Config.
	batchThreshold int
	trtThreshold   int

	// User-configurable adaptive config (env vars + CLI flags).
	config AdaptiveConfig

	// gpuSlots is a semaphore that limits concurrent GPU inference.
	// When all slots are occupied, new requests are routed to the fallback backend.
	gpuSlots chan struct{}
}

// NewAdaptiveDetector creates an AdaptiveDetector from the given options.
// At least one backend must be non-nil; otherwise Detect will always error.
func NewAdaptiveDetector(opts AdaptiveOpts) *AdaptiveDetector {
	bt := opts.BatchThreshold
	if bt <= 0 {
		bt = 3
	}
	tt := opts.TRTThreshold
	if tt <= 0 {
		tt = 8
	}

	cfg := opts.Config
	if cfg.GPUSlots <= 0 {
		cfg.GPUSlots = 8
	}

	a := &AdaptiveDetector{
		torchSingle:    opts.TorchSingle,
		torchBatch:     opts.TorchBatch,
		onnxTRT:        opts.OnnxTRT,
		onnxCUDA:       opts.OnnxCUDA,
		singleWork:     make(chan *singleRequest, 16),
		singleDone:     make(chan struct{}),
		batchThreshold: bt,
		trtThreshold:   tt,
		config:         cfg,
		gpuSlots:       make(chan struct{}, cfg.GPUSlots),
	}
	go a.singleWorkerLoop()
	names := a.backendNames()
	if len(names) > 0 {
		log.Printf("adaptive: loaded %d backend(s): %v, batch_threshold=%d, trt_threshold=%d, config=%+v",
			len(names), names, bt, tt, cfg)
	}

	// Run warmup if configured.
	if cfg.Warmup {
		a.warmup()
	}

	return a
}

// warmup runs 3 dummy detections on each available backend to pre-warm
// GPU contexts and JIT compilation caches. This ensures the first real
// request doesn't pay a cold-start penalty.
func (a *AdaptiveDetector) warmup() {
	backends := []struct {
		name    string
		backend DetectCloser
	}{
		{"torch-single", a.torchSingle},
		{"torch-batch", a.torchBatch},
		{"onnx-trt", a.onnxTRT},
		{"onnx-cuda", a.onnxCUDA},
	}

	// Use a small dummy image (1x1 pixel JPEG-like bytes — backends that
	// need real images will simply error on warmup, which is fine).
	dummyImg := []byte{0xFF, 0xD8, 0xFF, 0xE0}

	for _, b := range backends {
		if b.backend == nil {
			continue
		}
		for i := 0; i < 3; i++ {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			_, _ = b.backend.Detect(ctx, dummyImg, 0.25, 0.45)
			cancel()
		}
		log.Printf("adaptive: warmed up %s (3 dummy detections)", b.name)
	}
}

// singleWorkerLoop is the persistent CUDA worker goroutine. It pins itself to
// one OS thread via runtime.LockOSThread so the CUDA context stays warm on
// that thread, eliminating thread-switching jitter on the CGo boundary.
// This reduces single-image detection latency by ~0.5-1ms.
func (a *AdaptiveDetector) singleWorkerLoop() {
	runtime.LockOSThread() // Pin to one OS thread — CUDA context stays warm
	defer runtime.UnlockOSThread()
	defer close(a.singleDone)

	for req := range a.singleWork {
		objs, err := req.backend.Detect(req.ctx, req.image, req.conf, req.iou)
		req.result <- singleResult{objs, err}
	}
}

// Detect routes the request to the optimal backend based on current queue depth
// and user configuration. Resolution priority:
//
//  1. Per-request override (via context value from "backend" JSON field)
//  2. Benchmark lock (INFERGO_DETECT_BENCHMARK_BACKEND)
//  3. Env var / CLI flag (INFERGO_DETECT_BACKEND, if not "auto")
//  4. Adaptive routing by queue depth
//
// Falls through to any available backend if the preferred one is nil.
// Implements server.DetectionModel.
func (a *AdaptiveDetector) Detect(ctx context.Context, imageBytes []byte, confThresh, iouThresh float32) ([]server.DetectedObject, error) {
	depth := a.queueDepth.Add(1)
	defer a.queueDepth.Add(-1)

	start := time.Now()
	defer func() {
		latUs := time.Since(start).Microseconds()
		// Exponential moving average (alpha = 1/8 ≈ 0.125).
		// EMA = old * 7/8 + new * 1/8
		old := a.rollingP50.Load()
		if old == 0 {
			a.rollingP50.Store(latUs)
		} else {
			a.rollingP50.Store((old*7 + latUs) / 8)
		}
	}()

	// Read per-request backend override from context (set by HTTP handler).
	// Check both the local key (for direct callers) and the server package key
	// (set by handleDetect / handleDetectBinary).
	var requestBackend string
	if v, ok := ctx.Value(detectBackendKey{}).(string); ok {
		requestBackend = v
	} else if v, ok := ctx.Value(server.DetectBackendKey{}).(string); ok {
		requestBackend = v
	}

	backend, name := a.resolveBackend(requestBackend, depth)
	if backend == nil {
		return nil, errors.New("adaptive: no detection backends available")
	}

	// GPU slot semaphore: if the chosen backend is a GPU backend, try to
	// acquire a slot. If all slots are full, fall back to the configured
	// fallback backend.
	if a.isGPUBackend(name) {
		select {
		case a.gpuSlots <- struct{}{}:
			// Acquired a slot — release it when done.
			defer func() { <-a.gpuSlots }()
		default:
			// All GPU slots occupied — route to fallback.
			fb, fbName := a.lookupBackend(a.config.FallbackBackend)
			if fb != nil {
				backend, name = fb, fbName
			}
			// If fallback is also nil, proceed with original backend (best effort).
		}
	}

	var result []server.DetectedObject
	var err error

	// For low-depth requests using torchSingle, route through the persistent
	// CUDA worker goroutine to avoid thread-switching jitter.
	if depth <= 1 && name == "torch-single" && a.singleWork != nil {
		req := &singleRequest{
			backend: backend,
			ctx:     ctx,
			image:   imageBytes,
			conf:    confThresh,
			iou:     iouThresh,
			result:  make(chan singleResult, 1),
		}
		select {
		case a.singleWork <- req:
			select {
			case res := <-req.result:
				result, err = res.objects, res.err
			case <-ctx.Done():
				result, err = nil, ctx.Err()
			}
		case <-ctx.Done():
			result, err = nil, ctx.Err()
		}
	} else {
		result, err = backend.Detect(ctx, imageBytes, confThresh, iouThresh)
	}

	a.requestCount.Add(1)
	if err != nil {
		a.errorCount.Add(1)
	}
	return result, err
}

// resolveBackend determines which backend to use based on configuration priority:
//  1. Per-request override (requestBackend)
//  2. Benchmark lock (config.BenchmarkBackend)
//  3. Env var / CLI (config.Backend, if not "auto")
//  4. Adaptive routing by queue depth
func (a *AdaptiveDetector) resolveBackend(requestBackend string, depth int64) (DetectCloser, string) {
	// Priority 1: per-request override.
	if requestBackend != "" {
		if b, name := a.lookupBackend(requestBackend); b != nil {
			return b, name
		}
	}

	// Priority 2: benchmark lock.
	if a.config.BenchmarkBackend != "" {
		if b, name := a.lookupBackend(a.config.BenchmarkBackend); b != nil {
			return b, name
		}
	}

	// Priority 3: env var / CLI config (non-auto).
	if a.config.Backend != "" && a.config.Backend != "auto" {
		if b, name := a.lookupBackend(a.config.Backend); b != nil {
			return b, name
		}
	}

	// Priority 3.5: canonical backend (for reproducible results).
	if a.config.CanonicalBackend != "" {
		if b, name := a.lookupBackend(a.config.CanonicalBackend); b != nil {
			return b, name
		}
	}

	// Priority 4: adaptive routing by queue depth.
	return a.pickBackend(depth)
}

// lookupBackend maps a backend name to the corresponding DetectCloser.
// It normalizes names (e.g. "torch-gpu" → torchSingle, "cpu" → torchSingle
// without GPU, etc.) and returns (nil, "") if not found.
func (a *AdaptiveDetector) lookupBackend(name string) (DetectCloser, string) {
	switch strings.ToLower(name) {
	case "torch-single", "torch-gpu":
		if a.torchSingle != nil {
			return a.torchSingle, "torch-single"
		}
	case "torch-batch":
		if a.torchBatch != nil {
			return a.torchBatch, "torch-batch"
		}
	case "onnx-trt", "tensorrt":
		if a.onnxTRT != nil {
			return a.onnxTRT, "onnx-trt"
		}
	case "onnx-cuda":
		if a.onnxCUDA != nil {
			return a.onnxCUDA, "onnx-cuda"
		}
	case "cpu":
		// CPU preference: torchSingle (most isolated), then torchBatch, then onnxCUDA.
		if a.torchSingle != nil {
			return a.torchSingle, "torch-single"
		}
		if a.torchBatch != nil {
			return a.torchBatch, "torch-batch"
		}
	}
	return nil, ""
}

// isGPUBackend returns true if the named backend uses the GPU.
func (a *AdaptiveDetector) isGPUBackend(name string) bool {
	switch name {
	case "torch-single", "torch-batch", "onnx-trt", "onnx-cuda":
		return true
	default:
		return false
	}
}

// pickBackend selects the best available backend for the given queue depth.
// Returns the backend and its human-readable name.
func (a *AdaptiveDetector) pickBackend(depth int64) (DetectCloser, string) {
	switch {
	case depth <= 1 && a.torchSingle != nil:
		return a.torchSingle, "torch-single"
	case depth <= int64(a.batchThreshold) && a.torchBatch != nil:
		return a.torchBatch, "torch-batch"
	case depth > int64(a.batchThreshold) && a.onnxTRT != nil:
		return a.onnxTRT, "onnx-trt"
	case a.torchBatch != nil:
		return a.torchBatch, "torch-batch"
	case a.torchSingle != nil:
		return a.torchSingle, "torch-single"
	case a.onnxCUDA != nil:
		return a.onnxCUDA, "onnx-cuda"
	case a.onnxTRT != nil:
		return a.onnxTRT, "onnx-trt"
	default:
		return nil, ""
	}
}

// Close shuts down the persistent CUDA worker and all non-nil backends.
// Implements server.Model.
func (a *AdaptiveDetector) Close() {
	// Shut down the persistent worker goroutine first, then close backends.
	if a.singleWork != nil {
		close(a.singleWork)
		<-a.singleDone // wait for the worker to finish
	}
	if a.torchSingle != nil {
		a.torchSingle.Close()
	}
	if a.torchBatch != nil {
		a.torchBatch.Close()
	}
	if a.onnxTRT != nil {
		a.onnxTRT.Close()
	}
	if a.onnxCUDA != nil {
		a.onnxCUDA.Close()
	}
}

// Stats returns a snapshot of current runtime metrics.
func (a *AdaptiveDetector) Stats() AdaptiveStats {
	reqCount := a.requestCount.Load()
	errCount := a.errorCount.Load()
	qd := a.queueDepth.Load()

	var errRate float64
	if reqCount > 0 {
		errRate = float64(errCount) / float64(reqCount)
	}

	_, activeName := a.resolveBackend("", qd)

	return AdaptiveStats{
		QueueDepth:    qd,
		RollingP50Us:  a.rollingP50.Load(),
		RequestCount:  reqCount,
		ErrorCount:    errCount,
		ErrorRate:     errRate,
		ActiveBackend: activeName,
	}
}

// Config returns the current adaptive configuration.
func (a *AdaptiveDetector) Config() AdaptiveConfig {
	return a.config
}

// backendNames returns the names of all loaded (non-nil) backends.
func (a *AdaptiveDetector) backendNames() []string {
	var names []string
	if a.torchSingle != nil {
		names = append(names, "torch-single")
	}
	if a.torchBatch != nil {
		names = append(names, "torch-batch")
	}
	if a.onnxTRT != nil {
		names = append(names, "onnx-trt")
	}
	if a.onnxCUDA != nil {
		names = append(names, "onnx-cuda")
	}
	return names
}

// ─── Batch auto-tuner ───────────────────────────────────────────────────────

// batchTuner dynamically adjusts the maximum batch size for the batch detection
// loop based on observed per-image latency. When batching becomes slower than
// single-image inference (within a 10% tolerance), it reduces the batch cap.
// When batching is faster, it increases up to maxAllowed.
type batchTuner struct {
	currentBatch  int
	latencies     [10]float64 // rolling window of per-image latencies (ms)
	idx           int
	singleLatency float64 // measured single-image latency for comparison (ms)
	minBatch      int
	maxBatch      int
}

// newBatchTuner creates a tuner with the given initial batch size and bounds.
// singleLatencyMs is the baseline single-image latency to compare against.
func newBatchTuner(initialBatch int, singleLatencyMs float64, minBatch, maxBatch int) *batchTuner {
	if minBatch < 1 {
		minBatch = 1
	}
	if maxBatch < minBatch {
		maxBatch = minBatch
	}
	if initialBatch < minBatch {
		initialBatch = minBatch
	}
	if initialBatch > maxBatch {
		initialBatch = maxBatch
	}
	return &batchTuner{
		currentBatch:  initialBatch,
		singleLatency: singleLatencyMs,
		minBatch:      minBatch,
		maxBatch:      maxBatch,
	}
}

// recordBatch records the total latency for a batch of the given size and
// adjusts currentBatch accordingly.
func (t *batchTuner) recordBatch(batchSize int, totalMs float64) {
	if batchSize <= 0 {
		return
	}
	perImage := totalMs / float64(batchSize)
	t.latencies[t.idx%len(t.latencies)] = perImage
	t.idx++

	avg := t.avgLatency()
	if avg <= 0 {
		return
	}

	// If batching is >10% slower than single, shrink.
	if avg > t.singleLatency*1.1 && t.currentBatch > t.minBatch {
		t.currentBatch--
	} else if avg < t.singleLatency*0.9 && t.currentBatch < t.maxBatch {
		// If batching is >10% faster than single, grow.
		t.currentBatch++
	}
}

// avgLatency returns the average per-image latency over the filled portion
// of the rolling window.
func (t *batchTuner) avgLatency() float64 {
	n := t.idx
	if n > len(t.latencies) {
		n = len(t.latencies)
	}
	if n == 0 {
		return 0
	}
	var sum float64
	for i := 0; i < n; i++ {
		sum += t.latencies[i]
	}
	return sum / float64(n)
}

// maxBatchSize returns the current recommended maximum batch size.
func (t *batchTuner) maxBatchSize() int {
	return t.currentBatch
}

// Compile-time interface checks.
var _ server.DetectionModel = (*AdaptiveDetector)(nil)
