package server_test

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ailakshya/infergo/server"
)

// ─── mock models ─────────────────────────────────────────────────────────────

type mockLLM struct {
	reply string
}

func (m *mockLLM) Close() {}
func (m *mockLLM) Generate(_ context.Context, _ string, _ int, _ float32) (string, int, int, error) {
	return m.reply, 5, 3, nil
}

type mockEmbedder struct {
	vec []float32
}

func (m *mockEmbedder) Close() {}
func (m *mockEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	return m.vec, nil
}

type mockDetector struct {
	objs []server.DetectedObject
}

func (m *mockDetector) Close() {}
func (m *mockDetector) Detect(_ context.Context, _ []byte, _, _ float32) ([]server.DetectedObject, error) {
	return m.objs, nil
}

// ─── helper ──────────────────────────────────────────────────────────────────

func newTestServer(t *testing.T) (*server.Server, *server.Registry) {
	t.Helper()
	reg := server.NewRegistry()
	return server.NewServer(reg), reg
}

func doRequest(t *testing.T, srv http.Handler, method, path string, body any) *httptest.ResponseRecorder {
	t.Helper()
	var buf bytes.Buffer
	if body != nil {
		json.NewEncoder(&buf).Encode(body)
	}
	req := httptest.NewRequest(method, path, &buf)
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, req)
	return rr
}

// ─── GET /v1/models ───────────────────────────────────────────────────────────

func TestRouter_ListModels_Empty(t *testing.T) {
	srv, _ := newTestServer(t)
	rr := doRequest(t, srv, http.MethodGet, "/v1/models", nil)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}
	var resp server.ModelListResponse
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.Object != "list" {
		t.Errorf("expected object=list, got %q", resp.Object)
	}
	if len(resp.Data) != 0 {
		t.Errorf("expected 0 models, got %d", len(resp.Data))
	}
}

func TestRouter_ListModels_WithModels(t *testing.T) {
	srv, reg := newTestServer(t)
	reg.Load("llama3", &mockLLM{})
	reg.Load("bert", &mockEmbedder{})

	rr := doRequest(t, srv, http.MethodGet, "/v1/models", nil)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}
	var resp server.ModelListResponse
	json.NewDecoder(rr.Body).Decode(&resp)
	if len(resp.Data) != 2 {
		t.Errorf("expected 2 models, got %d", len(resp.Data))
	}
}

// ─── POST /v1/chat/completions ────────────────────────────────────────────────

func TestRouter_ChatCompletions_Valid(t *testing.T) {
	srv, reg := newTestServer(t)
	reg.Load("llama3", &mockLLM{reply: "Hello, world!"})

	rr := doRequest(t, srv, http.MethodPost, "/v1/chat/completions", server.ChatCompletionRequest{
		Model:    "llama3",
		Messages: []server.ChatMessage{{Role: "user", Content: "Hi"}},
	})
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var resp server.ChatCompletionResponse
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.Object != "chat.completion" {
		t.Errorf("expected object=chat.completion, got %q", resp.Object)
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(resp.Choices))
	}
	if resp.Choices[0].Message.Content != "Hello, world!" {
		t.Errorf("unexpected content: %q", resp.Choices[0].Message.Content)
	}
	if resp.Choices[0].Message.Role != "assistant" {
		t.Errorf("expected role=assistant, got %q", resp.Choices[0].Message.Role)
	}
	if resp.Choices[0].FinishReason != "stop" {
		t.Errorf("expected finish_reason=stop")
	}
	if resp.Usage.TotalTokens == 0 {
		t.Error("expected non-zero total_tokens")
	}
	if !strings.HasPrefix(resp.ID, "chatcmpl-") {
		t.Errorf("unexpected ID format: %q", resp.ID)
	}
}

func TestRouter_ChatCompletions_ModelNotFound(t *testing.T) {
	srv, _ := newTestServer(t)
	rr := doRequest(t, srv, http.MethodPost, "/v1/chat/completions", server.ChatCompletionRequest{
		Model:    "nope",
		Messages: []server.ChatMessage{{Role: "user", Content: "hi"}},
	})
	if rr.Code != http.StatusNotFound {
		t.Errorf("expected 404, got %d", rr.Code)
	}
}

func TestRouter_ChatCompletions_WrongModelType(t *testing.T) {
	srv, reg := newTestServer(t)
	reg.Load("bert", &mockEmbedder{})
	rr := doRequest(t, srv, http.MethodPost, "/v1/chat/completions", server.ChatCompletionRequest{
		Model:    "bert",
		Messages: []server.ChatMessage{{Role: "user", Content: "hi"}},
	})
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rr.Code)
	}
}

func TestRouter_ChatCompletions_MissingModel(t *testing.T) {
	srv, _ := newTestServer(t)
	rr := doRequest(t, srv, http.MethodPost, "/v1/chat/completions", map[string]any{
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	})
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rr.Code)
	}
}

func TestRouter_ChatCompletions_EmptyMessages(t *testing.T) {
	srv, reg := newTestServer(t)
	reg.Load("llama3", &mockLLM{})
	rr := doRequest(t, srv, http.MethodPost, "/v1/chat/completions", server.ChatCompletionRequest{
		Model:    "llama3",
		Messages: []server.ChatMessage{},
	})
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rr.Code)
	}
}

// ─── POST /v1/completions ─────────────────────────────────────────────────────

func TestRouter_Completions_Valid(t *testing.T) {
	srv, reg := newTestServer(t)
	reg.Load("gpt2", &mockLLM{reply: "the quick brown fox"})

	rr := doRequest(t, srv, http.MethodPost, "/v1/completions", server.CompletionRequest{
		Model:  "gpt2",
		Prompt: "Once upon a time",
	})
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var resp server.CompletionResponse
	json.NewDecoder(rr.Body).Decode(&resp)
	if resp.Object != "text_completion" {
		t.Errorf("expected object=text_completion, got %q", resp.Object)
	}
	if resp.Choices[0].Text != "the quick brown fox" {
		t.Errorf("unexpected text: %q", resp.Choices[0].Text)
	}
}

func TestRouter_Completions_EmptyPrompt(t *testing.T) {
	srv, reg := newTestServer(t)
	reg.Load("gpt2", &mockLLM{})
	rr := doRequest(t, srv, http.MethodPost, "/v1/completions", server.CompletionRequest{
		Model: "gpt2",
	})
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rr.Code)
	}
}

// ─── POST /v1/embeddings ─────────────────────────────────────────────────────

func TestRouter_Embeddings_Valid(t *testing.T) {
	srv, reg := newTestServer(t)
	reg.Load("bert", &mockEmbedder{vec: []float32{0.1, 0.2, 0.3}})

	rr := doRequest(t, srv, http.MethodPost, "/v1/embeddings", server.EmbeddingRequest{
		Model: "bert",
		Input: "hello world",
	})
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var resp server.EmbeddingResponse
	json.NewDecoder(rr.Body).Decode(&resp)
	if resp.Object != "list" {
		t.Errorf("expected object=list, got %q", resp.Object)
	}
	if len(resp.Data) != 1 || len(resp.Data[0].Embedding) != 3 {
		t.Errorf("unexpected embedding: %+v", resp.Data)
	}
}

func TestRouter_Embeddings_WrongModelType(t *testing.T) {
	srv, reg := newTestServer(t)
	reg.Load("llama3", &mockLLM{})
	rr := doRequest(t, srv, http.MethodPost, "/v1/embeddings", server.EmbeddingRequest{
		Model: "llama3",
		Input: "hello",
	})
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rr.Code)
	}
}

// ─── POST /v1/detect ─────────────────────────────────────────────────────────

func TestRouter_Detect_Valid(t *testing.T) {
	srv, reg := newTestServer(t)
	reg.Load("yolo", &mockDetector{objs: []server.DetectedObject{
		{X1: 10, Y1: 20, X2: 50, Y2: 60, ClassID: 0, Confidence: 0.9},
	}})

	imgB64 := base64.StdEncoding.EncodeToString([]byte("fakeimagebytes"))
	rr := doRequest(t, srv, http.MethodPost, "/v1/detect", server.DetectRequest{
		Model:    "yolo",
		ImageB64: imgB64,
	})
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var resp server.DetectResponse
	json.NewDecoder(rr.Body).Decode(&resp)
	if len(resp.Objects) != 1 {
		t.Fatalf("expected 1 object, got %d", len(resp.Objects))
	}
	if resp.Objects[0].Confidence != 0.9 {
		t.Errorf("unexpected confidence: %v", resp.Objects[0].Confidence)
	}
}

func TestRouter_Detect_MissingImage(t *testing.T) {
	srv, reg := newTestServer(t)
	reg.Load("yolo", &mockDetector{})
	rr := doRequest(t, srv, http.MethodPost, "/v1/detect", server.DetectRequest{Model: "yolo"})
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rr.Code)
	}
}

func TestRouter_Detect_InvalidBase64(t *testing.T) {
	srv, reg := newTestServer(t)
	reg.Load("yolo", &mockDetector{})
	rr := doRequest(t, srv, http.MethodPost, "/v1/detect", server.DetectRequest{
		Model:    "yolo",
		ImageB64: "!!!notbase64!!!",
	})
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rr.Code)
	}
}

// ─── POST /v1/detect/binary ──────────────────────────────────────────────────

func doBinaryRequest(t *testing.T, srv http.Handler, path string, body []byte, headers map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(body))
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, req)
	return rr
}

func TestHandleDetectBinary(t *testing.T) {
	srv, reg := newTestServer(t)
	reg.Load("yolo", &mockDetector{objs: []server.DetectedObject{
		{X1: 10, Y1: 20, X2: 50, Y2: 60, ClassID: 0, Confidence: 0.9},
	}})

	rr := doBinaryRequest(t, srv,
		"/v1/detect/binary?model=yolo&conf=0.3&iou=0.5",
		[]byte("fakeimagebytes"),
		map[string]string{"Content-Type": "image/jpeg"},
	)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var resp server.DetectResponse
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.Model != "yolo" {
		t.Errorf("expected model=yolo, got %q", resp.Model)
	}
	if len(resp.Objects) != 1 {
		t.Fatalf("expected 1 object, got %d", len(resp.Objects))
	}
	if resp.Objects[0].Confidence != 0.9 {
		t.Errorf("unexpected confidence: %v", resp.Objects[0].Confidence)
	}
}

func TestHandleDetectBinary_ModelFromHeader(t *testing.T) {
	srv, reg := newTestServer(t)
	reg.Load("yolo", &mockDetector{objs: []server.DetectedObject{
		{X1: 1, Y1: 2, X2: 3, Y2: 4, ClassID: 0, Confidence: 0.8},
	}})

	rr := doBinaryRequest(t, srv,
		"/v1/detect/binary",
		[]byte("fakeimagebytes"),
		map[string]string{"Content-Type": "image/jpeg", "X-Model": "yolo"},
	)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var resp server.DetectResponse
	json.NewDecoder(rr.Body).Decode(&resp)
	if resp.Model != "yolo" {
		t.Errorf("expected model=yolo, got %q", resp.Model)
	}
}

func TestHandleDetectBinaryNoModel(t *testing.T) {
	srv, _ := newTestServer(t)

	rr := doBinaryRequest(t, srv,
		"/v1/detect/binary",
		[]byte("fakeimagebytes"),
		map[string]string{"Content-Type": "image/jpeg"},
	)
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestHandleDetectBinaryEmptyBody(t *testing.T) {
	srv, _ := newTestServer(t)

	rr := doBinaryRequest(t, srv,
		"/v1/detect/binary?model=yolo",
		nil,
		map[string]string{"Content-Type": "image/jpeg"},
	)
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestHandleDetectBinary_ModelNotFound(t *testing.T) {
	srv, _ := newTestServer(t)

	rr := doBinaryRequest(t, srv,
		"/v1/detect/binary?model=nonexistent",
		[]byte("fakeimagebytes"),
		map[string]string{"Content-Type": "image/jpeg"},
	)
	if rr.Code != http.StatusNotFound {
		t.Errorf("expected 404, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestHandleDetectBinary_DefaultThresholds(t *testing.T) {
	srv, reg := newTestServer(t)
	reg.Load("yolo", &mockDetector{objs: []server.DetectedObject{}})

	// No conf/iou params — should use defaults (0.25, 0.45) and not error.
	rr := doBinaryRequest(t, srv,
		"/v1/detect/binary?model=yolo",
		[]byte("fakeimagebytes"),
		map[string]string{"Content-Type": "image/jpeg"},
	)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}
}

func BenchmarkHandleDetect(b *testing.B) {
	reg := server.NewRegistry()
	srv := server.NewServer(reg)
	reg.Load("yolo", &mockDetector{objs: []server.DetectedObject{
		{X1: 10, Y1: 20, X2: 50, Y2: 60, ClassID: 0, Confidence: 0.9},
	}})

	// Create a ~100KB base64 payload to simulate realistic overhead.
	fakeImg := make([]byte, 75*1024) // 75KB raw → ~100KB base64
	imgB64 := base64.StdEncoding.EncodeToString(fakeImg)
	body, _ := json.Marshal(server.DetectRequest{
		Model:    "yolo",
		ImageB64: imgB64,
	})

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest(http.MethodPost, "/v1/detect", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		rr := httptest.NewRecorder()
		srv.ServeHTTP(rr, req)
		if rr.Code != http.StatusOK {
			b.Fatalf("expected 200, got %d", rr.Code)
		}
	}
}

func BenchmarkHandleDetectBinary(b *testing.B) {
	reg := server.NewRegistry()
	srv := server.NewServer(reg)
	reg.Load("yolo", &mockDetector{objs: []server.DetectedObject{
		{X1: 10, Y1: 20, X2: 50, Y2: 60, ClassID: 0, Confidence: 0.9},
	}})

	// 75KB raw image bytes — no base64 overhead.
	fakeImg := make([]byte, 75*1024)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest(http.MethodPost, "/v1/detect/binary?model=yolo&conf=0.25&iou=0.45", bytes.NewReader(fakeImg))
		req.Header.Set("Content-Type", "image/jpeg")
		rr := httptest.NewRecorder()
		srv.ServeHTTP(rr, req)
		if rr.Code != http.StatusOK {
			b.Fatalf("expected 200, got %d", rr.Code)
		}
	}
}
