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
	"strconv"
	"strings"
	"syscall"
	"time"

	infergogrpc "github.com/ailakshya/infergo/grpc"
	"github.com/ailakshya/infergo/llm"
	"github.com/ailakshya/infergo/onnx"
	"github.com/ailakshya/infergo/server"
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

func runServe(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	var models modelFlags
	fs.Var(&models, "model", "model to load; repeatable. Format: [name:]path.{gguf,onnx}")
	provider  := fs.String("provider", "cpu", "execution provider: cpu|cuda|tensorrt|coreml")
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
	mode := fs.String("mode", "combined", "server role: combined|prefill|decode")
	fs.Parse(args)

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

	// Load all specified models.
	for _, spec := range models {
		name, path := parseModelSpec(spec)
		if err := loadModel(reg, metrics, name, path, *provider, *gpuLayers, *ctxSize, *threads, *maxSeqs, tensorSplit, *pipelineStages); err != nil {
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
		return loadModel(reg, metrics, name, path, *provider, *gpuLayers, *ctxSize, *threads, *maxSeqs, tensorSplit, *pipelineStages)
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

func loadModel(reg *server.Registry, metrics *server.Metrics, name, path, provider string, gpuLayers, ctxSize, threads, maxSeqs int, tensorSplit []float32, pipelineStages int) error {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".gguf":
		return loadLLM(reg, metrics, name, path, gpuLayers, ctxSize, threads, maxSeqs, tensorSplit, pipelineStages)
	case ".onnx":
		return loadONNX(reg, name, path, provider)
	default:
		return fmt.Errorf("unsupported model extension %q (want .gguf or .onnx)", ext)
	}
}

func loadLLM(reg *server.Registry, metrics *server.Metrics, name, path string, gpuLayers, ctxSize, threads, maxSeqs int, tensorSplit []float32, pipelineStages int) error {
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
	return reg.Load(name, newSchedulerModel(m, name, metrics.ActiveSequences, metrics))
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
