package client_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/ailakshya/infergo/client"
)

// ─── TestChat ────────────────────────────────────────────────────────────────

func TestChat(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"x","model":"test-model","choices":[{"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`)
	}))
	defer srv.Close()

	c := client.New(srv.URL)
	resp, err := c.Chat(context.Background(), client.ChatRequest{
		Model:    "test-model",
		Messages: []client.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Chat() error: %v", err)
	}
	if resp.Content != "hello" {
		t.Errorf("Content = %q, want %q", resp.Content, "hello")
	}
	if resp.ID != "x" {
		t.Errorf("ID = %q, want %q", resp.ID, "x")
	}
	if resp.Usage.PromptTokens != 5 {
		t.Errorf("PromptTokens = %d, want 5", resp.Usage.PromptTokens)
	}
	if resp.Usage.CompletionTokens != 1 {
		t.Errorf("CompletionTokens = %d, want 1", resp.Usage.CompletionTokens)
	}
}

// ─── TestChatStream ──────────────────────────────────────────────────────────

func TestChatStream(t *testing.T) {
	tokens := []string{"Hello", " ", "world", "!", " How", " are", " you", "?"}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusInternalServerError)
			return
		}
		for _, tok := range tokens {
			chunk := map[string]any{
				"id": "stream-1",
				"choices": []map[string]any{
					{
						"delta":         map[string]string{"content": tok},
						"finish_reason": nil,
					},
				},
			}
			b, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", b)
			flusher.Flush()
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	c := client.New(srv.URL)
	tokenCh, errCh := c.ChatStream(context.Background(), client.ChatRequest{
		Model:    "test-model",
		Messages: []client.Message{{Role: "user", Content: "hi"}},
	})

	var received []string
	for tok := range tokenCh {
		received = append(received, tok)
	}
	if err := <-errCh; err != nil {
		t.Fatalf("ChatStream error: %v", err)
	}
	if len(received) < 5 {
		t.Errorf("received %d tokens, want >= 5", len(received))
	}
	if len(received) != len(tokens) {
		t.Errorf("received %d tokens, want %d", len(received), len(tokens))
	}
	for i, want := range tokens {
		if i < len(received) && received[i] != want {
			t.Errorf("token[%d] = %q, want %q", i, received[i], want)
		}
	}
}

// ─── TestEmbed ───────────────────────────────────────────────────────────────

func TestEmbed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/embeddings" {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"object":"list","model":"embed-model","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]}]}`)
	}))
	defer srv.Close()

	c := client.New(srv.URL)
	vec, err := c.Embed(context.Background(), client.EmbedRequest{
		Model: "embed-model",
		Input: "hello world",
	})
	if err != nil {
		t.Fatalf("Embed() error: %v", err)
	}
	if len(vec) != 3 {
		t.Fatalf("len(vec) = %d, want 3", len(vec))
	}
	want := []float32{0.1, 0.2, 0.3}
	for i, v := range want {
		if vec[i] != v {
			t.Errorf("vec[%d] = %f, want %f", i, vec[i], v)
		}
	}
}

// ─── TestDetect ──────────────────────────────────────────────────────────────

func TestDetect(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/detect" {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"model":"yolo-model","objects":[{"X1":10,"Y1":20,"X2":100,"Y2":200,"ClassID":0,"Confidence":0.9}]}`)
	}))
	defer srv.Close()

	c := client.New(srv.URL)
	resp, err := c.Detect(context.Background(), client.DetectRequest{
		Model:    "yolo-model",
		ImageB64: "aGVsbG8=",
	})
	if err != nil {
		t.Fatalf("Detect() error: %v", err)
	}
	if len(resp.Objects) != 1 {
		t.Fatalf("len(Objects) = %d, want 1", len(resp.Objects))
	}
	obj := resp.Objects[0]
	if obj.X1 != 10 || obj.Y1 != 20 || obj.X2 != 100 || obj.Y2 != 200 {
		t.Errorf("bounding box = (%v,%v,%v,%v), want (10,20,100,200)", obj.X1, obj.Y1, obj.X2, obj.Y2)
	}
	if obj.ClassID != 0 {
		t.Errorf("ClassID = %d, want 0", obj.ClassID)
	}
	if obj.Confidence != 0.9 {
		t.Errorf("Confidence = %f, want 0.9", obj.Confidence)
	}
}

// ─── TestListModels ──────────────────────────────────────────────────────────

func TestListModels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/models" {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"object":"list","data":[{"id":"llama3-8b-q4","object":"model","created":1700000000},{"id":"all-MiniLM-L6-v2","object":"model","created":1700000001}]}`)
	}))
	defer srv.Close()

	c := client.New(srv.URL)
	models, err := c.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels() error: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("len(models) = %d, want 2", len(models))
	}
	if models[0].ID != "llama3-8b-q4" {
		t.Errorf("models[0].ID = %q, want %q", models[0].ID, "llama3-8b-q4")
	}
	if models[1].ID != "all-MiniLM-L6-v2" {
		t.Errorf("models[1].ID = %q, want %q", models[1].ID, "all-MiniLM-L6-v2")
	}
}

// ─── TestWithAPIKey ──────────────────────────────────────────────────────────

func TestWithAPIKey(t *testing.T) {
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"object":"list","data":[]}`)
	}))
	defer srv.Close()

	c := client.New(srv.URL, client.WithAPIKey("my-secret-key"))
	_, err := c.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels() error: %v", err)
	}
	want := "Bearer my-secret-key"
	if gotAuth != want {
		t.Errorf("Authorization header = %q, want %q", gotAuth, want)
	}
}

// ─── TestContextCancel ───────────────────────────────────────────────────────

func TestContextCancel(t *testing.T) {
	// Server that streams slowly so we can cancel mid-stream.
	started := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusInternalServerError)
			return
		}
		close(started)
		// Stream a few tokens, then block waiting for client to disconnect.
		for i := 0; i < 3; i++ {
			chunk := map[string]any{
				"id": "stream-cancel",
				"choices": []map[string]any{
					{"delta": map[string]string{"content": fmt.Sprintf("tok%d", i)}, "finish_reason": nil},
				},
			}
			b, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", b)
			flusher.Flush()
		}
		// Block until client disconnects or context done.
		<-r.Context().Done()
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())

	c := client.New(srv.URL, client.WithTimeout(10*time.Second))
	tokenCh, errCh := c.ChatStream(ctx, client.ChatRequest{
		Model:    "test-model",
		Messages: []client.Message{{Role: "user", Content: "hi"}},
	})

	// Wait for server to start sending.
	<-started

	// Read at least one token then cancel.
	var count int
	cancelDone := false
	for tok := range tokenCh {
		_ = tok
		count++
		if count >= 1 && !cancelDone {
			cancel()
			cancelDone = true
		}
	}

	// Drain the error channel — context cancellation is not an error we require
	// the client to surface, but the channels must both close cleanly.
	<-errCh

	if count == 0 {
		t.Error("expected at least one token before cancel")
	}
}
