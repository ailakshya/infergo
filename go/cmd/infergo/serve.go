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
	"strings"
	"syscall"
	"time"

	"github.com/ailakshya/infergo/llm"
	"github.com/ailakshya/infergo/server"
)

const shutdownTimeout = 15 * time.Second

func runServe(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	modelPath := fs.String("model", "", "path to model file (.gguf for LLM, .onnx for ONNX)")
	provider  := fs.String("provider", "cpu", "execution provider: cpu|cuda|tensorrt|coreml")
	port      := fs.Int("port", 9090, "HTTP listen port")
	gpuLayers := fs.Int("gpu-layers", 999, "number of transformer layers to offload to GPU (LLM only)")
	ctxSize   := fs.Int("ctx-size", 16384, "total KV cache tokens (divided by --max-seqs to get per-sequence budget)")
	threads   := fs.Int("threads", 0, "CPU threads for inference (0 = auto: physical cores / 2)")
	maxSeqs   := fs.Int("max-seqs", 16, "max concurrent sequences / KV cache slots (LLM only)")
	minModels := fs.Int("min-models", 1, "minimum models required for /health/ready")
	fs.Parse(args)

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "serve: --model is required")
		fs.Usage()
		os.Exit(1)
	}

	reg := server.NewRegistry()
	health := server.NewHealthChecker(reg, *minModels)
	metrics := server.NewMetrics()

	// Load the model based on file extension
	modelName := modelName(*modelPath)
	if err := loadModel(reg, modelName, *modelPath, *provider, *gpuLayers, *ctxSize, *threads, *maxSeqs); err != nil {
		log.Fatalf("failed to load model %q: %v", *modelPath, err)
	}
	log.Printf("loaded model %q (%s)", modelName, *modelPath)

	// Wire up the mux: API routes + health + metrics
	mux := http.NewServeMux()
	apiSrv := server.NewServer(reg)
	mux.Handle("/v1/", metrics.WrapServer(apiSrv))
	health.RegisterRoutes(mux)
	mux.Handle("/metrics", metrics.Handler())

	addr := fmt.Sprintf(":%d", *port)
	httpSrv := &http.Server{Addr: addr, Handler: mux}

	// Graceful shutdown on SIGINT/SIGTERM
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		log.Printf("listening on %s (provider=%s)", addr, *provider)
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("server error: %v", err)
		}
	}()

	<-stop
	log.Println("shutting down…")
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

func loadModel(reg *server.Registry, name, path, provider string, gpuLayers, ctxSize, threads, maxSeqs int) error {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".gguf":
		return loadLLM(reg, name, path, gpuLayers, ctxSize, threads, maxSeqs)
	case ".onnx":
		return loadONNX(reg, name, path, provider)
	default:
		return fmt.Errorf("unsupported model extension %q (want .gguf or .onnx)", ext)
	}
}

func loadLLM(reg *server.Registry, name, path string, gpuLayers, ctxSize, threads, maxSeqs int) error {
	if threads <= 0 {
		threads = runtime.NumCPU() / 2
		if threads < 1 {
			threads = 1
		}
	}
	log.Printf("using %d CPU threads, %d max concurrent sequences", threads, maxSeqs)
	// n_batch=2048: max tokens per llama_decode call.
	// 512 was too small for long prompts with chat-template overhead.
	m, err := llm.Load(path, gpuLayers, ctxSize, maxSeqs, 2048)
	if err != nil {
		return err
	}
	return reg.Load(name, newSchedulerModel(m))
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

	// Fall back to plain ONNX placeholder (appears in /v1/models only).
	return reg.Load(name, &onnxAdapter{path: path})
}
