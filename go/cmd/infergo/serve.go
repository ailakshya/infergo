package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"syscall"
	"time"

	infergogrpc "github.com/ailakshya/infergo/grpc"
	"github.com/ailakshya/infergo/llm"
	"github.com/ailakshya/infergo/onnx"
	"github.com/ailakshya/infergo/server"
	"github.com/ailakshya/infergo/torch"
)

const shutdownTimeout = 15 * time.Second

// modelFlags implements flag.Value for a repeatable --model flag.
// Accepted formats:
//
//	--model path/to/model.gguf           (name inferred from filename)
//	--model name:path/to/model.gguf      (explicit name)
type modelFlags []string

func (m *modelFlags) String() string { return fmt.Sprintf("%v", *m) }
func (m *modelFlags) Set(v string) error {
	*m = append(*m, v)
	return nil
}

// parseModelSpec splits a model spec into name and path.
// If the spec contains a colon that is not part of an absolute path prefix,
// the left side is the name and the right side is the path.
// Otherwise the name is derived from the filename.
func parseModelSpec(spec string) (name, path string) {
	if i := strings.IndexByte(spec, ':'); i > 0 && !filepath.IsAbs(spec[:i]) {
		return spec[:i], spec[i+1:]
	}
	return modelName(spec), spec
}

// Package-level speculative decoding config (set by --draft-model flag).
// Used by loadLLM to configure the scheduler.
var (
	specDraftPath string
	specNDraft    int
)

func runServe(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	var models modelFlags
	fs.Var(&models, "model", "model to load; repeatable. Format: [name:]path.{gguf,onnx,pt,pth}")
	provider  := fs.String("provider", "cpu", "execution provider: cpu|cuda|tensorrt|coreml")
	backend   := fs.String("backend", "auto", "inference backend: auto|onnx|tensorrt|torch")
	port      := fs.Int("port", 9090, "HTTP listen port")
	gpuLayers := fs.Int("gpu-layers", 999, "number of transformer layers to offload to GPU (LLM only)")
	ctxSize   := fs.Int("ctx-size", 16384, "total KV cache tokens (divided by --max-seqs to get per-sequence budget)")
	threads   := fs.Int("threads", 0, "CPU threads for inference (0 = auto: physical cores / 2)")
	maxSeqs   := fs.Int("max-seqs", 16, "max concurrent sequences / KV cache slots (LLM only)")
	minModels := fs.Int("min-models", 1, "minimum models required for /health/ready")
	apiKey    := fs.String("api-key", "", "API key for Bearer auth (empty = open, or set INFERGO_API_KEY env var)")
	rateLimit := fs.Float64("rate-limit", 0, "max requests/second per client IP (0 = unlimited)")
	maxQueue     := fs.Int("max-queue", 100, "max in-flight requests (active+waiting); 503 beyond this")
	maxActive    := fs.Int("max-active", 0, "max concurrent request handlers (0 = same as --max-queue)")
	otlpEndpoint := fs.String("otlp-endpoint", "", "OTLP HTTP endpoint for tracing (e.g. localhost:4318; empty = disabled)")
	grpcPort     := fs.Int("grpc-port", 9091, "gRPC listen port (0 = disabled)")
	tensorSplitStr  := fs.String("tensor-split", "", "comma-separated GPU fractions for tensor parallelism (e.g. 0.5,0.5); empty = single GPU")
	pipelineStages  := fs.Int("pipeline-stages", 1, "number of pipeline stages for layer-split multi-GPU inference (1 = single GPU, N>1 = N GPUs with LLAMA_SPLIT_MODE_LAYER)")
	mode         := fs.String("mode", "combined", "server role: combined|prefill|decode")
	gcInterval   := fs.Int("gc-interval", 100, "call runtime.GC() every N completed requests (0 = disabled)")
	maxBatchSize := fs.Int("max-batch-size", 0, "max sequences per BatchDecode call, 0 = unlimited (set >0 to cap batch size)")
	batchTimeout := fs.Int("batch-timeout-ms", 0, "ms to wait for more requests after the first arrives before firing a batch (0 = no wait)")
	adaptiveFlag := fs.Bool("adaptive", false, "enable adaptive hybrid detection: auto-route requests to the optimal backend based on queue depth")
	safeModeFlag := fs.Bool("safe-mode", false, "disable adaptive detection and batch loop; use single-image libtorch only")
	batchThreshold := fs.Int("batch-threshold", 3, "queue depth at which adaptive detector switches from single-image to batch mode")
	warmupBackends := fs.Bool("warmup-backends", true, "warm up all detection backends with dummy inferences at startup")
	detectGPUSlots := fs.Int("detect-gpu-slots", 8, "max concurrent GPU detection inference slots (overflow routes to fallback)")
	draftModel  := fs.String("draft-model", "", "path to draft GGUF model for speculative decoding (must share vocab with target LLM)")
	nDraft      := fs.Int("n-draft", 5, "number of tokens to draft per speculative step")
	fs.Parse(args)

	// Store speculative config for loadLLM.
	specDraftPath = *draftModel
	specNDraft = *nDraft

	// Apply aggressive GC tuning to reduce heap growth under sustained load.
	// GOGC=50 means GC triggers at 50% heap growth (default is 100%).
	// This trades slightly more frequent GC cycles for a tighter RSS ceiling.
	debug.SetGCPercent(50)
	log.Printf("GC tuning: GOGC=50, gc-interval=%d", *gcInterval)

	if err := validateMode(*mode); err != nil {
		fmt.Fprintf(os.Stderr, "serve: --mode: %v\n", err)
		os.Exit(1)
	}

	if *apiKey == "" {
		*apiKey = os.Getenv("INFERGO_API_KEY")
	}

	// Parse --tensor-split flag into []float32.
	tensorSplit, err := parseTensorSplit(*tensorSplitStr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "serve: --tensor-split: %v\n", err)
		os.Exit(1)
	}

	if len(models) == 0 {
		fmt.Fprintln(os.Stderr, "serve: --model is required")
		fs.Usage()
		os.Exit(1)
	}

	// Initialize OpenTelemetry tracer (no-op if --otlp-endpoint not set).
	tracerShutdown, err := server.InitTracer("infergo", *otlpEndpoint)
	if err != nil {
		log.Fatalf("init tracer: %v", err)
	}
	defer func() {
		if err := tracerShutdown(context.Background()); err != nil {
			log.Printf("tracer shutdown: %v", err)
		}
	}()

	reg := server.NewRegistry()
	health := server.NewHealthChecker(reg, *minModels)
	metrics := server.NewMetrics()

	if err := validateBackend(*backend); err != nil {
		fmt.Fprintf(os.Stderr, "serve: --backend: %v\n", err)
		os.Exit(1)
	}

	// Build adaptive config from env vars, then override with CLI flags.
	adaptiveCfg := LoadAdaptiveConfig()
	// CLI flags override env defaults when explicitly set.
	adaptiveCfg.Warmup = *warmupBackends
	adaptiveCfg.GPUSlots = *detectGPUSlots

	// Load all specified models.
	for _, spec := range models {
		name, path := parseModelSpec(spec)
		if err := loadModel(reg, metrics, name, path, *provider, *backend, *gpuLayers, *ctxSize, *threads, *maxSeqs, tensorSplit, *pipelineStages, *gcInterval, *maxBatchSize, *batchTimeout, *adaptiveFlag, *safeModeFlag, *batchThreshold, adaptiveCfg); err != nil {
			log.Fatalf("failed to load model %q: %v", spec, err)
		}
		log.Printf("loaded model %q (%s)", name, path)
	}

	// Wire up the mux: API routes + health + metrics
	mux := http.NewServeMux()
	apiSrv := server.NewServer(reg)
	apiSrv.SetMode(*mode)

	// Inject the hot-reload function so POST /v1/admin/reload works.
	apiSrv.SetReloader(func(name, path string) error {
		return loadModel(reg, metrics, name, path, *provider, *backend, *gpuLayers, *ctxSize, *threads, *maxSeqs, tensorSplit, *pipelineStages, *gcInterval, *maxBatchSize, *batchTimeout, *adaptiveFlag, *safeModeFlag, *batchThreshold, adaptiveCfg)
	})

	var v1Handler http.Handler = server.WrapTracing(metrics.WrapServer(apiSrv), "infergo.v1")
	if *maxQueue > 0 {
		queueGauge := metrics.QueueDepth
		pq := server.NewQueueMiddleware(*maxActive, *maxQueue, queueGauge)
		v1Handler = pq.Middleware()(v1Handler)
	}
	if *rateLimit > 0 {
		rl := server.NewRateLimiter(*rateLimit)
		v1Handler = rl.Middleware()(v1Handler)
	}
	if *apiKey != "" {
		v1Handler = server.AuthMiddleware(*apiKey)(v1Handler)
	}
	mux.Handle("/v1/", v1Handler)
	health.RegisterRoutes(mux)
	mux.Handle("/metrics", metrics.Handler())

	addr := fmt.Sprintf(":%d", *port)
	httpSrv := &http.Server{Addr: addr, Handler: mux}

	// Start gRPC server if --grpc-port > 0.
	var grpcSrvInst *infergogrpc.Server
	if *grpcPort > 0 {
		grpcSrvInst = infergogrpc.New(newGRPCRegistry(reg))
		go func() {
			grpcAddr := fmt.Sprintf(":%d", *grpcPort)
			log.Printf("gRPC listening on %s", grpcAddr)
			if err := grpcSrvInst.Serve(grpcAddr); err != nil {
				log.Printf("gRPC server error: %v", err)
			}
		}()
	}

	// Graceful shutdown on SIGINT/SIGTERM
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		log.Printf("listening on %s (provider=%s, mode=%s)", addr, *provider, *mode)
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("server error: %v", err)
		}
	}()

	<-stop
	log.Println("shutting down…")
	if grpcSrvInst != nil {
		grpcSrvInst.Stop()
	}
	ctx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer cancel()
	httpSrv.Shutdown(ctx)
	log.Println("bye")
}

// ─── model loading ────────────────────────────────────────────────────────────

func modelName(path string) string {
	base := filepath.Base(path)
	return strings.TrimSuffix(base, filepath.Ext(base))
}

func loadModel(reg *server.Registry, metrics *server.Metrics, name, path, provider, backend string, gpuLayers, ctxSize, threads, maxSeqs int, tensorSplit []float32, pipelineStages, gcInterval, maxBatchSize, batchTimeoutMs int, adaptive, safeMode bool, batchThreshold int, adaptiveCfg AdaptiveConfig) error {
	ext := strings.ToLower(filepath.Ext(path))

	// Resolve effective backend: explicit flag overrides auto-detection.
	effectiveBackend := backend
	if effectiveBackend == "auto" {
		switch ext {
		case ".gguf":
			effectiveBackend = "llm"
		case ".onnx":
			effectiveBackend = "onnx"
		case ".pt", ".pth":
			effectiveBackend = "torch"
		default:
			return fmt.Errorf("unsupported model extension %q (want .gguf, .onnx, .pt, or .pth)", ext)
		}
	}

	// --adaptive backend: build an AdaptiveDetector from all available backends.
	if effectiveBackend == "adaptive" {
		return loadAdaptive(reg, name, path, provider, batchThreshold, adaptiveCfg)
	}

	// --safe-mode: disable batch loop, use single-image libtorch only.
	if safeMode && (effectiveBackend == "torch") {
		return loadTorchSafeMode(reg, name, path, provider)
	}

	// --adaptive flag (not backend): wrap torch detection in adaptive if possible.
	if adaptive && effectiveBackend == "torch" {
		lname := strings.ToLower(filepath.Base(path))
		if strings.Contains(lname, "yolo") {
			return loadAdaptive(reg, name, path, provider, batchThreshold, adaptiveCfg)
		}
	}

	switch effectiveBackend {
	case "llm":
		return loadLLM(reg, metrics, name, path, gpuLayers, ctxSize, threads, maxSeqs, tensorSplit, pipelineStages, gcInterval, maxBatchSize, batchTimeoutMs)
	case "onnx", "tensorrt":
		return loadONNX(reg, name, path, provider)
	case "torch":
		return loadTorch(reg, name, path, provider)
	default:
		// For .gguf files with a non-auto backend, still use LLM.
		if ext == ".gguf" {
			return loadLLM(reg, metrics, name, path, gpuLayers, ctxSize, threads, maxSeqs, tensorSplit, pipelineStages, gcInterval, maxBatchSize, batchTimeoutMs)
		}
		return fmt.Errorf("unsupported backend %q for extension %q", effectiveBackend, ext)
	}
}

func loadLLM(reg *server.Registry, metrics *server.Metrics, name, path string, gpuLayers, ctxSize, threads, maxSeqs int, tensorSplit []float32, pipelineStages, gcInterval, maxBatchSize, batchTimeoutMs int) error {
	if threads <= 0 {
		threads = runtime.NumCPU() / 2
		if threads < 1 {
			threads = 1
		}
	}
	log.Printf("using %d CPU threads, %d max concurrent sequences", threads, maxSeqs)
	// n_batch=2048: max tokens per llama_decode call.
	// 512 was too small for long prompts with chat-template overhead.
	var (
		m   *llm.Model
		err error
	)
	// --pipeline-stages takes priority over --tensor-split when both are set.
	if pipelineStages > 1 {
		log.Printf("pipeline parallelism across %d GPU stage(s) (LLAMA_SPLIT_MODE_LAYER)", pipelineStages)
		m, err = llm.LoadPipeline(path, gpuLayers, ctxSize, maxSeqs, 2048, pipelineStages)
	} else if len(tensorSplit) > 0 {
		log.Printf("tensor split across %d GPU(s): %v", len(tensorSplit), tensorSplit)
		m, err = llm.LoadSplit(path, gpuLayers, ctxSize, maxSeqs, 2048, tensorSplit)
	} else {
		m, err = llm.Load(path, gpuLayers, ctxSize, maxSeqs, 2048)
	}
	if err != nil {
		return err
	}
	log.Printf("scheduler tuning: max-batch-size=%d, batch-timeout-ms=%d, gc-interval=%d", maxBatchSize, batchTimeoutMs, gcInterval)
	sm := newSchedulerModel(m, name, metrics.ActiveSequences, metrics, maxBatchSize, batchTimeoutMs, gcInterval)

	// Enable speculative decoding if a draft model is configured.
	if specDraftPath != "" {
		sd, err := llm.NewSpeculativeDecoder(m, specDraftPath, gpuLayers, specNDraft)
		if err != nil {
			log.Printf("[infergo] WARNING: speculative decoding disabled: %v", err)
		} else {
			sm.specDecoder = sd
			log.Printf("[infergo] speculative decoding enabled: draft=%s, n_draft=%d", filepath.Base(specDraftPath), specNDraft)
		}
	}

	return reg.Load(name, sm)
}

// parseTensorSplit parses a comma-separated string of floats into []float32.
// Returns nil, nil for an empty string (single-GPU mode).
func parseTensorSplit(s string) ([]float32, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	parts := strings.Split(s, ",")
	out := make([]float32, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		v, err := strconv.ParseFloat(p, 32)
		if err != nil {
			return nil, fmt.Errorf("invalid fraction %q: %w", p, err)
		}
		out = append(out, float32(v))
	}
	return out, nil
}

// onnxAdapter is a placeholder for ONNX models that lack a recognized
// pipeline (no tokenizer.json for embedding, no preprocess for detection).
// It registers the model in /v1/models without real inference capability.
type onnxAdapter struct {
	path string
}

func (a *onnxAdapter) Close() {}

func loadONNX(reg *server.Registry, name, path, provider string) error {
	// Verify file exists before registering
	if _, err := os.Stat(path); err != nil {
		return fmt.Errorf("model file not found: %w", err)
	}

	// If tokenizer.json exists near the model, treat as an embedding model.
	if tok := findTokenizerJSON(filepath.Dir(path), 2); tok != "" {
		log.Printf("found tokenizer.json at %s — loading as embedding model", tok)
		adapter, err := loadEmbedding(path, provider)
		if err != nil {
			return fmt.Errorf("load embedding model: %w", err)
		}
		return reg.Load(name, adapter)
	}

	// If filename contains "yolo", treat as detection model.
	lname := strings.ToLower(filepath.Base(path))
	if strings.Contains(lname, "yolo") {
		log.Printf("loading %s as detection model (YOLOv8)", filepath.Base(path))
		adapter, err := loadDetection(path, provider)
		if err != nil {
			return fmt.Errorf("load detection model: %w", err)
		}
		return reg.Load(name, adapter)
	}

	// Fall back to plain ONNX placeholder (appears in /v1/models only).
	return reg.Load(name, &onnxAdapter{path: path})
}

// validateMode returns nil if mode is one of the allowed values.
func validateMode(mode string) error {
	switch mode {
	case "combined", "prefill", "decode":
		return nil
	default:
		return fmt.Errorf("invalid mode %q: must be combined, prefill, or decode", mode)
	}
}

// validateBackend returns nil if backend is one of the allowed values.
func validateBackend(backend string) error {
	switch backend {
	case "auto", "onnx", "tensorrt", "torch", "adaptive":
		return nil
	default:
		return fmt.Errorf("invalid backend %q: must be auto, onnx, tensorrt, torch, or adaptive", backend)
	}
}

// loadDetection creates a YOLOv8 detection model from an ONNX model path.
func loadDetection(modelPath, provider string) (*detectionAdapter, error) {
	sess, err := onnx.NewSession(provider, 0)
	if err != nil {
		return nil, fmt.Errorf("loadDetection: new session: %w", err)
	}
	if err := sess.Load(modelPath); err != nil {
		sess.Close()
		return nil, fmt.Errorf("loadDetection: load model: %w", err)
	}
	return &detectionAdapter{sess: sess}, nil
}

// torchAdapter is a placeholder for torch models that lack a recognized
// pipeline (no YOLO filename pattern). It registers the model in /v1/models
// without real inference capability.
type torchAdapter struct {
	path string
}

func (a *torchAdapter) Close() {}

func loadTorch(reg *server.Registry, name, path, provider string) error {
	// Verify file exists before registering.
	if _, err := os.Stat(path); err != nil {
		return fmt.Errorf("model file not found: %w", err)
	}

	// If filename contains "yolo", treat as detection model.
	lname := strings.ToLower(filepath.Base(path))
	if strings.Contains(lname, "yolo") {
		log.Printf("loading %s as detection model (YOLOv8, torch)", filepath.Base(path))
		adapter, err := loadTorchDetection(path, provider)
		if err != nil {
			return fmt.Errorf("load torch detection model: %w", err)
		}
		return reg.Load(name, adapter)
	}

	// Fall back to plain torch placeholder (appears in /v1/models only).
	return reg.Load(name, &torchAdapter{path: path})
}

// loadTorchDetection creates a YOLOv8 detection model from a TorchScript model path.
// It starts a batch loop goroutine that accumulates concurrent detection requests
// and processes them in one CGo call to amortize the ~2ms Go-to-C overhead.
func loadTorchDetection(modelPath, provider string) (*torchDetectionAdapter, error) {
	sess, err := torch.NewSession(provider, 0)
	if err != nil {
		return nil, fmt.Errorf("loadTorchDetection: new session: %w", err)
	}
	if err := sess.Load(modelPath); err != nil {
		sess.Close()
		return nil, fmt.Errorf("loadTorchDetection: load model: %w", err)
	}

	adapter := &torchDetectionAdapter{
		sess:     sess,
		requests: make(chan *detectRequest, 64),
	}
	go adapter.batchLoop()
	log.Printf("torch detection: batch loop started (max_batch=8, max_wait=2ms)")

	return adapter, nil
}

// loadTorchSingle creates a YOLOv8 torch detection adapter WITHOUT the batch loop.
// Used as the single-image backend in adaptive mode, or as the sole backend
// in safe mode.
func loadTorchSingle(modelPath, provider string) (*torchDetectionAdapter, error) {
	sess, err := torch.NewSession(provider, 0)
	if err != nil {
		return nil, fmt.Errorf("loadTorchSingle: new session: %w", err)
	}
	if err := sess.Load(modelPath); err != nil {
		sess.Close()
		return nil, fmt.Errorf("loadTorchSingle: load model: %w", err)
	}
	// No batch loop — requests is nil, so Detect uses single-image path.
	return &torchDetectionAdapter{sess: sess, requests: nil}, nil
}

// loadTorchSafeMode loads a torch YOLO model with only the single-image path
// (no batch loop, no adaptive). This is the most conservative configuration.
func loadTorchSafeMode(reg *server.Registry, name, path, provider string) error {
	if _, err := os.Stat(path); err != nil {
		return fmt.Errorf("model file not found: %w", err)
	}
	lname := strings.ToLower(filepath.Base(path))
	if !strings.Contains(lname, "yolo") {
		return reg.Load(name, &torchAdapter{path: path})
	}

	log.Printf("safe-mode: loading %s as single-image detection (no batch, no adaptive)", filepath.Base(path))
	adapter, err := loadTorchSingle(path, provider)
	if err != nil {
		return fmt.Errorf("load torch detection (safe-mode): %w", err)
	}
	return reg.Load(name, adapter)
}

// loadAdaptive builds an AdaptiveDetector that wraps all available backends
// for the given model path. It probes for sibling files (.onnx alongside .pt
// and vice versa) to load as many backends as possible.
func loadAdaptive(reg *server.Registry, name, modelPath, provider string, batchThreshold int, adaptiveCfg AdaptiveConfig) error {
	if _, err := os.Stat(modelPath); err != nil {
		return fmt.Errorf("model file not found: %w", err)
	}

	lname := strings.ToLower(filepath.Base(modelPath))
	if !strings.Contains(lname, "yolo") {
		return fmt.Errorf("adaptive backend only supports YOLO models, got %q", filepath.Base(modelPath))
	}

	ext := strings.ToLower(filepath.Ext(modelPath))
	basePath := strings.TrimSuffix(modelPath, filepath.Ext(modelPath))
	// Handle double extensions like .torchscript.pt
	if strings.HasSuffix(strings.ToLower(basePath), ".torchscript") {
		basePath = strings.TrimSuffix(basePath, filepath.Ext(basePath))
	}

	var opts AdaptiveOpts
	opts.BatchThreshold = batchThreshold
	opts.Config = adaptiveCfg

	// Determine which file is the primary and probe for siblings.
	var torchPath, onnxPath string

	switch ext {
	case ".pt", ".pth":
		torchPath = modelPath
		// Probe for sibling ONNX file.
		for _, candidate := range []string{basePath + ".onnx"} {
			if _, err := os.Stat(candidate); err == nil {
				onnxPath = candidate
				break
			}
		}
	case ".onnx":
		onnxPath = modelPath
		// Probe for sibling TorchScript file.
		for _, candidate := range []string{
			basePath + ".torchscript.pt",
			basePath + ".pt",
		} {
			if _, err := os.Stat(candidate); err == nil {
				torchPath = candidate
				break
			}
		}
	default:
		return fmt.Errorf("adaptive: unsupported extension %q (want .pt, .pth, or .onnx)", ext)
	}

	// Load torch backends (single + batch) if available.
	if torchPath != "" {
		single, err := loadTorchSingle(torchPath, provider)
		if err != nil {
			log.Printf("adaptive: torch single-image failed: %v (continuing)", err)
		} else {
			opts.TorchSingle = single
			log.Printf("adaptive: loaded torch single-image from %s", filepath.Base(torchPath))
		}

		batch, err := loadTorchDetection(torchPath, provider)
		if err != nil {
			log.Printf("adaptive: torch batch failed: %v (continuing)", err)
		} else {
			opts.TorchBatch = batch
			log.Printf("adaptive: loaded torch batch from %s", filepath.Base(torchPath))
		}
	}

	// Load ONNX backends if available.
	if onnxPath != "" {
		// Try TensorRT first.
		trt, err := loadDetection(onnxPath, "tensorrt")
		if err != nil {
			log.Printf("adaptive: ONNX TensorRT failed: %v (trying CUDA)", err)
		} else {
			opts.OnnxTRT = trt
			log.Printf("adaptive: loaded ONNX TensorRT from %s", filepath.Base(onnxPath))
		}

		// Also try CUDA EP.
		cuda, err := loadDetection(onnxPath, "cuda")
		if err != nil {
			log.Printf("adaptive: ONNX CUDA failed: %v (continuing)", err)
		} else {
			opts.OnnxCUDA = cuda
			log.Printf("adaptive: loaded ONNX CUDA from %s", filepath.Base(onnxPath))
		}
	}

	// Verify at least one backend loaded.
	if opts.TorchSingle == nil && opts.TorchBatch == nil && opts.OnnxTRT == nil && opts.OnnxCUDA == nil {
		return fmt.Errorf("adaptive: no backends could be loaded for %s", modelPath)
	}

	ad := NewAdaptiveDetector(opts)
	return reg.Load(name, ad)
}
