package server_test

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ailakshya/infergo/server"
)

// ─── mock streaming model ────────────────────────────────────────────────────

type mockStreamLLM struct {
	tokens []string
}

func (m *mockStreamLLM) Close() {}

func (m *mockStreamLLM) Generate(_ context.Context, _ string, _ int, _ float32) (string, int, int, error) {
	return strings.Join(m.tokens, ""), 3, len(m.tokens), nil
}

func (m *mockStreamLLM) Stream(_ context.Context, _ string, _ int, _ float32) (<-chan string, error) {
	ch := make(chan string, len(m.tokens))
	for _, t := range m.tokens {
		ch <- t
	}
	close(ch)
	return ch, nil
}

// ─── SSE helpers ─────────────────────────────────────────────────────────────

// parseSSE reads all SSE events from body and returns the data payloads.
func parseSSE(body string) []string {
	var events []string
	scanner := bufio.NewScanner(strings.NewReader(body))
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			events = append(events, strings.TrimPrefix(line, "data: "))
		}
	}
	return events
}

func doStreamRequest(t *testing.T, srv http.Handler, body any) *httptest.ResponseRecorder {
	t.Helper()
	var buf bytes.Buffer
	json.NewEncoder(&buf).Encode(body)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", &buf)
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, req)
	return rr
}

// ─── Tests ───────────────────────────────────────────────────────────────────

func TestStreaming_ContentTypeIsEventStream(t *testing.T) {
	reg := server.NewRegistry()
	reg.Load("llama3", &mockStreamLLM{tokens: []string{"hi"}})
	srv := server.NewServer(reg)

	rr := doStreamRequest(t, srv, server.ChatCompletionRequest{
		Model:    "llama3",
		Messages: []server.ChatMessage{{Role: "user", Content: "hi"}},
		Stream:   true,
	})

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}
	ct := rr.Header().Get("Content-Type")
	if !strings.HasPrefix(ct, "text/event-stream") {
		t.Errorf("expected text/event-stream, got %q", ct)
	}
}

func TestStreaming_EndsWith_DONE(t *testing.T) {
	reg := server.NewRegistry()
	reg.Load("llama3", &mockStreamLLM{tokens: []string{"Hello", " world"}})
	srv := server.NewServer(reg)

	rr := doStreamRequest(t, srv, server.ChatCompletionRequest{
		Model:    "llama3",
		Messages: []server.ChatMessage{{Role: "user", Content: "hi"}},
		Stream:   true,
	})

	events := parseSSE(rr.Body.String())
	if len(events) == 0 {
		t.Fatal("no SSE events received")
	}
	last := events[len(events)-1]
	if last != "[DONE]" {
		t.Errorf("expected last event to be [DONE], got %q", last)
	}
}

func TestStreaming_TokensDeliveredInOrder(t *testing.T) {
	tokens := []string{"The", " quick", " brown", " fox"}
	reg := server.NewRegistry()
	reg.Load("llama3", &mockStreamLLM{tokens: tokens})
	srv := server.NewServer(reg)

	rr := doStreamRequest(t, srv, server.ChatCompletionRequest{
		Model:    "llama3",
		Messages: []server.ChatMessage{{Role: "user", Content: "tell me"}},
		Stream:   true,
	})

	events := parseSSE(rr.Body.String())
	// Collect content from non-[DONE] events
	var got []string
	for _, e := range events {
		if e == "[DONE]" {
			continue
		}
		var chunk server.ChatCompletionChunk
		if err := json.Unmarshal([]byte(e), &chunk); err != nil {
			t.Fatalf("unmarshal chunk: %v\nevent: %q", err, e)
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		if c := chunk.Choices[0].Delta.Content; c != "" {
			got = append(got, c)
		}
	}

	if len(got) != len(tokens) {
		t.Fatalf("expected %d content chunks, got %d: %v", len(tokens), len(got), got)
	}
	for i, tok := range tokens {
		if got[i] != tok {
			t.Errorf("chunk[%d]: expected %q, got %q", i, tok, got[i])
		}
	}
}

func TestStreaming_ChunkShape(t *testing.T) {
	reg := server.NewRegistry()
	reg.Load("llama3", &mockStreamLLM{tokens: []string{"hi"}})
	srv := server.NewServer(reg)

	rr := doStreamRequest(t, srv, server.ChatCompletionRequest{
		Model:    "llama3",
		Messages: []server.ChatMessage{{Role: "user", Content: "yo"}},
		Stream:   true,
	})

	events := parseSSE(rr.Body.String())
	// Find a content chunk (not role-only, not final, not DONE)
	for _, e := range events {
		if e == "[DONE]" {
			continue
		}
		var chunk server.ChatCompletionChunk
		if err := json.Unmarshal([]byte(e), &chunk); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if chunk.Object != "chat.completion.chunk" {
			t.Errorf("expected object=chat.completion.chunk, got %q", chunk.Object)
		}
		if chunk.Model != "llama3" {
			t.Errorf("expected model=llama3, got %q", chunk.Model)
		}
		if !strings.HasPrefix(chunk.ID, "chatcmpl-") {
			t.Errorf("unexpected ID: %q", chunk.ID)
		}
	}
}

func TestStreaming_FinishReasonOnLastChunk(t *testing.T) {
	reg := server.NewRegistry()
	reg.Load("llama3", &mockStreamLLM{tokens: []string{"ok"}})
	srv := server.NewServer(reg)

	rr := doStreamRequest(t, srv, server.ChatCompletionRequest{
		Model:    "llama3",
		Messages: []server.ChatMessage{{Role: "user", Content: "?"}},
		Stream:   true,
	})

	events := parseSSE(rr.Body.String())
	// Second-to-last is the stop chunk (last is [DONE])
	for i := len(events) - 2; i >= 0; i-- {
		if events[i] == "[DONE]" {
			continue
		}
		var chunk server.ChatCompletionChunk
		json.Unmarshal([]byte(events[i]), &chunk)
		if chunk.Choices[0].FinishReason != nil && *chunk.Choices[0].FinishReason == "stop" {
			return // found it
		}
	}
	t.Error("no chunk with finish_reason=stop found")
}

func TestStreaming_FallbackToBatch(t *testing.T) {
	// mockLLM only implements LLMModel (not StreamingLLMModel) → batch fallback
	reg := server.NewRegistry()
	reg.Load("llama3", &mockLLM{reply: "batch reply"})
	srv := server.NewServer(reg)

	rr := doStreamRequest(t, srv, server.ChatCompletionRequest{
		Model:    "llama3",
		Messages: []server.ChatMessage{{Role: "user", Content: "test"}},
		Stream:   true,
	})

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	events := parseSSE(rr.Body.String())
	last := events[len(events)-1]
	if last != "[DONE]" {
		t.Errorf("expected [DONE], got %q", last)
	}

	// Find the content chunk
	found := false
	for _, e := range events {
		if e == "[DONE]" {
			continue
		}
		var chunk server.ChatCompletionChunk
		json.Unmarshal([]byte(e), &chunk)
		if chunk.Choices[0].Delta.Content == "batch reply" {
			found = true
		}
	}
	if !found {
		t.Errorf("batch reply not found in SSE stream: %s", rr.Body.String())
	}
}

func TestStreaming_ModelNotFound(t *testing.T) {
	reg := server.NewRegistry()
	srv := server.NewServer(reg)

	rr := doStreamRequest(t, srv, server.ChatCompletionRequest{
		Model:    "nope",
		Messages: []server.ChatMessage{{Role: "user", Content: "hi"}},
		Stream:   true,
	})
	if rr.Code != http.StatusNotFound {
		t.Errorf("expected 404, got %d", rr.Code)
	}
}
