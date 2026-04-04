package grpc_test

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	realgrpc "google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	infergogrpc "github.com/ailakshya/infergo/grpc"
	"github.com/ailakshya/infergo/grpc/pb"
)

// ─── Fake models ──────────────────────────────────────────────────────────────

type fakeLLM struct{}

func (f *fakeLLM) Generate(_ context.Context, _ string, _ int, _ float32) (string, int, int, error) {
	return "hello from grpc", 3, 4, nil
}

type fakeEmbedder struct{}

func (f *fakeEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	return []float32{0.1, 0.2, 0.3}, nil
}

type fakeDetector struct{}

func (f *fakeDetector) Detect(_ context.Context, _ []byte, _, _ float32) ([]infergogrpc.DetectedObject, error) {
	return []infergogrpc.DetectedObject{
		{X1: 10, Y1: 20, X2: 30, Y2: 40, ClassID: 1, Confidence: 0.9},
	}, nil
}

// ─── Pure in-memory registry (no CGo, no server package) ─────────────────────

type memModelHandle struct {
	model interface{}
}

func (h *memModelHandle) Model() interface{} { return h.model }
func (h *memModelHandle) Release()           {}

type memRegistry struct {
	mu     sync.RWMutex
	models map[string]interface{}
}

func newMemRegistry() *memRegistry {
	return &memRegistry{models: make(map[string]interface{})}
}

func (r *memRegistry) register(name string, model interface{}) {
	r.mu.Lock()
	r.models[name] = model
	r.mu.Unlock()
}

func (r *memRegistry) Get(name string) (infergogrpc.ModelHandle, error) {
	r.mu.RLock()
	m, ok := r.models[name]
	r.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("model %q not found", name)
	}
	return &memModelHandle{model: m}, nil
}

func (r *memRegistry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.models))
	for k := range r.models {
		names = append(names, k)
	}
	return names
}

// ─── Test setup helper ────────────────────────────────────────────────────────

func startTestServer(t *testing.T) (pb.InfergoClient, *memRegistry, func()) {
	t.Helper()

	reg := newMemRegistry()
	reg.register("llm-model", &fakeLLM{})
	reg.register("embed-model", &fakeEmbedder{})
	reg.register("detect-model", &fakeDetector{})

	srv := infergogrpc.New(reg)

	// Find a free ephemeral port.
	tmp, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	addr := tmp.Addr().String()
	tmp.Close()

	go func() {
		_ = srv.Serve(addr)
	}()

	// Wait for the gRPC server to start accepting connections.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", addr, 100*time.Millisecond)
		if err == nil {
			conn.Close()
			break
		}
		time.Sleep(20 * time.Millisecond)
	}

	cc, err := realgrpc.NewClient(addr,
		realgrpc.WithTransportCredentials(insecure.NewCredentials()),
		realgrpc.WithDefaultCallOptions(realgrpc.ForceCodec(infergogrpc.JSONCodec{})),
	)
	if err != nil {
		srv.Stop()
		t.Fatalf("dial: %v", err)
	}

	client := pb.NewInfergoClient(cc)
	cleanup := func() {
		cc.Close()
		srv.Stop()
	}
	return client, reg, cleanup
}

// ─── T1: gRPC chat completion ─────────────────────────────────────────────────

func TestGRPC_ChatCompletion(t *testing.T) {
	client, _, cleanup := startTestServer(t)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, err := client.ChatCompletion(ctx, &pb.ChatRequest{
		Model: "llm-model",
		Messages: []*pb.Message{
			{Role: "user", Content: "hello"},
		},
		MaxTokens: 64,
	})
	if err != nil {
		t.Fatalf("ChatCompletion RPC: %v", err)
	}

	var chunks []*pb.ChatChunk
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("stream.Recv: %v", err)
		}
		chunks = append(chunks, chunk)
	}

	if len(chunks) == 0 {
		t.Fatal("expected at least one ChatChunk, got none")
	}

	// Last chunk must have Done=true.
	last := chunks[len(chunks)-1]
	if !last.Done {
		t.Errorf("last chunk Done=false, want true")
	}

	// At least one chunk must carry non-empty token text.
	var hasText bool
	for _, c := range chunks {
		if c.Token != "" {
			hasText = true
			break
		}
	}
	if !hasText {
		t.Error("no chunk carried a non-empty token")
	}
	t.Logf("received %d chunk(s), last: done=%v", len(chunks), last.Done)
}

// ─── T2: gRPC embedding ───────────────────────────────────────────────────────

func TestGRPC_Embed(t *testing.T) {
	client, _, cleanup := startTestServer(t)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := client.Embed(ctx, &pb.EmbedRequest{
		Model: "embed-model",
		Input: "test input",
	})
	if err != nil {
		t.Fatalf("Embed RPC: %v", err)
	}
	if len(resp.Values) == 0 {
		t.Fatal("EmbedResponse.Values is empty")
	}
	t.Logf("embedding dim=%d first=%.4f", len(resp.Values), resp.Values[0])
}

// ─── T3: gRPC detection ───────────────────────────────────────────────────────

func TestGRPC_Detect(t *testing.T) {
	client, _, cleanup := startTestServer(t)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := client.Detect(ctx, &pb.DetectRequest{
		Model:      "detect-model",
		ImageBytes: []byte{0xFF, 0xD8, 0xFF}, // minimal JPEG header bytes
		ConfThresh: 0.25,
		IouThresh:  0.45,
	})
	if err != nil {
		t.Fatalf("Detect RPC: %v", err)
	}
	if len(resp.Detections) == 0 {
		t.Fatal("DetectResponse.Detections is empty")
	}
	d := resp.Detections[0]
	t.Logf("detection: class=%d conf=%.3f box=[%.0f,%.0f,%.0f,%.0f]",
		d.ClassId, d.Confidence, d.X1, d.Y1, d.X2, d.Y2)
}

// ─── T4: HTTP + gRPC co-exist ─────────────────────────────────────────────────
// Start an HTTP server (in-memory fake) and a gRPC server on separate ports;
// verify both serve requests concurrently without interference.

// fakeHTTPHandler is a minimal HTTP handler for testing co-existence.
type fakeHTTPHandler struct{}

func (h *fakeHTTPHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, `{"object":"list","data":[]}`)
}

func TestGRPC_HTTPAndGRPCCoexist(t *testing.T) {
	reg := newMemRegistry()
	reg.register("llm-model", &fakeLLM{})

	// Start HTTP server.
	httpSrv := httptest.NewServer(&fakeHTTPHandler{})
	defer httpSrv.Close()

	// Start gRPC server.
	grpcSrv := infergogrpc.New(reg)
	tmp, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	grpcAddr := tmp.Addr().String()
	tmp.Close()

	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = grpcSrv.Serve(grpcAddr)
	}()
	defer func() {
		grpcSrv.Stop()
		<-done
	}()

	// Wait for gRPC port.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", grpcAddr, 100*time.Millisecond)
		if err == nil {
			conn.Close()
			break
		}
		time.Sleep(20 * time.Millisecond)
	}

	// Verify HTTP still responds.
	resp, err := http.Get(fmt.Sprintf("%s/v1/models", httpSrv.URL))
	if err != nil {
		t.Fatalf("HTTP GET /v1/models: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("HTTP status=%d, want 200", resp.StatusCode)
	}

	// Verify gRPC still responds.
	cc, err := realgrpc.NewClient(grpcAddr,
		realgrpc.WithTransportCredentials(insecure.NewCredentials()),
		realgrpc.WithDefaultCallOptions(realgrpc.ForceCodec(infergogrpc.JSONCodec{})),
	)
	if err != nil {
		t.Fatalf("grpc dial: %v", err)
	}
	defer cc.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	lmResp, err := pb.NewInfergoClient(cc).ListModels(ctx, &pb.ListModelsRequest{})
	if err != nil {
		t.Fatalf("ListModels RPC: %v", err)
	}
	if len(lmResp.Models) == 0 {
		t.Error("ListModelsResponse.Models is empty")
	}
	t.Logf("HTTP OK + gRPC ListModels returned %d model(s)", len(lmResp.Models))
}

// ─── T5: Latency note ─────────────────────────────────────────────────────────
//
// OPT-13-T5 (gRPC P50 ≤ HTTP P50 - 1ms) requires a real LLM model and
// sustained load. This benchmark must be run on gpu_dev with:
//
//	go test -bench=BenchmarkGRPCvsHTTP -benchtime=30s ./go/grpc/
//
// It is not run in CI because it depends on the llama3-8b-q4.gguf model.
