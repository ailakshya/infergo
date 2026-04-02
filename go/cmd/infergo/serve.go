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
	"sync"
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
	ctxSize   := fs.Int("ctx-size", 4096, "KV cache token budget (LLM only)")
	threads   := fs.Int("threads", 0, "CPU threads for inference (0 = auto: physical cores / 2)")
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
	if err := loadModel(reg, modelName, *modelPath, *provider, *gpuLayers, *ctxSize, *threads); err != nil {
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

func loadModel(reg *server.Registry, name, path, provider string, gpuLayers, ctxSize, threads int) error {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".gguf":
		return loadLLM(reg, name, path, gpuLayers, ctxSize, threads)
	case ".onnx":
		return loadONNX(reg, name, path, provider)
	default:
		return fmt.Errorf("unsupported model extension %q (want .gguf or .onnx)", ext)
	}
}

// llmAdapter bridges go/llm.Model → server.LLMModel.
// llama.cpp is not thread-safe: BatchDecode must be called from one goroutine
// at a time. mu serializes all Generate calls.
type llmAdapter struct {
	m  *llm.Model
	mu sync.Mutex
}

func (a *llmAdapter) Close() { a.m.Close() }

func (a *llmAdapter) Generate(ctx context.Context, prompt string, maxTokens int, temp float32) (string, int, int, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	tokens, err := a.m.Tokenize(prompt, true, 4096)
	if err != nil {
		return "", 0, 0, fmt.Errorf("tokenize: %w", err)
	}
	promptToks := len(tokens)

	seq, err := a.m.NewSequence(tokens)
	if err != nil {
		return "", 0, 0, fmt.Errorf("new sequence: %w", err)
	}
	defer seq.Close()

	var out strings.Builder
	genToks := 0

	for genToks < maxTokens && !seq.IsDone() {
		select {
		case <-ctx.Done():
			return out.String(), promptToks, genToks, ctx.Err()
		default:
		}
		if err := a.m.BatchDecode([]*llm.Sequence{seq}); err != nil {
			return out.String(), promptToks, genToks, fmt.Errorf("batch decode: %w", err)
		}
		tok, err := seq.SampleToken(temp, 0.9)
		if err != nil {
			return out.String(), promptToks, genToks, fmt.Errorf("sample token: %w", err)
		}
		if a.m.IsEOG(tok) {
			break
		}
		piece, err := a.m.TokenToPiece(tok)
		if err != nil {
			return out.String(), promptToks, genToks, fmt.Errorf("token to piece: %w", err)
		}
		out.WriteString(piece)
		seq.AppendToken(tok)
		genToks++
	}
	return out.String(), promptToks, genToks, nil
}

func loadLLM(reg *server.Registry, name, path string, gpuLayers, ctxSize, threads int) error {
	if threads <= 0 {
		threads = runtime.NumCPU() / 2
		if threads < 1 {
			threads = 1
		}
	}
	log.Printf("using %d CPU threads for inference", threads)
	m, err := llm.Load(path, gpuLayers, ctxSize, threads, 512)
	if err != nil {
		return err
	}
	return reg.Load(name, &llmAdapter{m: m})
}

// onnxAdapter wraps an ONNX session for the server.Model interface.
// Full inference capability requires knowing the model's I/O spec;
// this adapter registers the model so it appears in /v1/models.
type onnxAdapter struct {
	path string
}

func (a *onnxAdapter) Close() {}

func loadONNX(reg *server.Registry, name, path, provider string) error {
	// Verify file exists before registering
	if _, err := os.Stat(path); err != nil {
		return fmt.Errorf("model file not found: %w", err)
	}
	return reg.Load(name, &onnxAdapter{path: path})
}
