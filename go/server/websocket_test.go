package server_test

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"golang.org/x/net/websocket"

	"github.com/ailakshya/infergo/server"
)

// ─── helpers ─────────────────────────────────────────────────────────────────

// dialWS connects to the given httptest.Server at path and returns the conn.
func dialWS(t *testing.T, ts *httptest.Server, path string) *websocket.Conn {
	t.Helper()
	origin := "http://localhost/"
	url := "ws" + strings.TrimPrefix(ts.URL, "http") + path
	ws, err := websocket.Dial(url, "", origin)
	if err != nil {
		t.Fatalf("websocket.Dial: %v", err)
	}
	return ws
}

// collectChunks reads WSChatChunk frames until "[DONE]" or an error.
// Returns all non-[DONE] token strings.
func collectChunks(t *testing.T, ws *websocket.Conn) []string {
	t.Helper()
	var tokens []string
	for {
		var chunk server.WSChatChunk
		if err := websocket.JSON.Receive(ws, &chunk); err != nil {
			// Connection closed by server — treat as end of stream.
			break
		}
		if chunk.Token == "[DONE]" {
			break
		}
		tokens = append(tokens, chunk.Token)
	}
	return tokens
}

// sendRequest sends a WSChatRequest JSON frame over ws.
func sendRequest(t *testing.T, ws *websocket.Conn, req server.WSChatRequest) {
	t.Helper()
	if err := websocket.JSON.Send(ws, req); err != nil {
		t.Fatalf("websocket.JSON.Send: %v", err)
	}
}

// ─── T1: WebSocket handshake succeeds ────────────────────────────────────────

func TestWebSocket_HandshakeSucceeds(t *testing.T) {
	reg := server.NewRegistry()
	reg.Load("llama3", &mockStreamLLM{tokens: []string{"hi"}})
	srv := server.NewServer(reg)

	ts := httptest.NewServer(srv)
	defer ts.Close()

	ws := dialWS(t, ts, "/v1/ws/chat")
	defer ws.Close()

	// Handshake succeeded if we get here without error.
	// Send a minimal request so the server doesn't hang waiting.
	sendRequest(t, ws, server.WSChatRequest{
		Model:    "llama3",
		Messages: []server.ChatMessage{{Role: "user", Content: "ping"}},
	})
	collectChunks(t, ws)
}

// ─── T2: Tokens stream correctly ─────────────────────────────────────────────

func TestWebSocket_TokensStreamCorrectly(t *testing.T) {
	wantTokens := []string{"hello", " ", "world"}
	reg := server.NewRegistry()
	reg.Load("llama3", &mockStreamLLM{tokens: wantTokens})
	srv := server.NewServer(reg)

	ts := httptest.NewServer(srv)
	defer ts.Close()

	ws := dialWS(t, ts, "/v1/ws/chat")
	defer ws.Close()

	sendRequest(t, ws, server.WSChatRequest{
		Model:    "llama3",
		Messages: []server.ChatMessage{{Role: "user", Content: "say hello"}},
		MaxTokens: 64,
	})

	got := collectChunks(t, ws)

	assembled := strings.Join(got, "")
	want := strings.Join(wantTokens, "")
	if assembled != want {
		t.Errorf("assembled response mismatch: got %q, want %q", assembled, want)
	}
	if len(got) != len(wantTokens) {
		t.Errorf("expected %d token frames, got %d", len(wantTokens), len(got))
	}
	for i, tok := range wantTokens {
		if got[i] != tok {
			t.Errorf("frame[%d]: expected %q, got %q", i, tok, got[i])
		}
	}
}

// TestWebSocket_DoneFramePresent verifies the final frame carries token="[DONE]".
func TestWebSocket_DoneFramePresent(t *testing.T) {
	reg := server.NewRegistry()
	reg.Load("llama3", &mockStreamLLM{tokens: []string{"ok"}})
	srv := server.NewServer(reg)

	ts := httptest.NewServer(srv)
	defer ts.Close()

	ws := dialWS(t, ts, "/v1/ws/chat")
	defer ws.Close()

	sendRequest(t, ws, server.WSChatRequest{
		Model:    "llama3",
		Messages: []server.ChatMessage{{Role: "user", Content: "?"}},
	})

	// Read all raw frames looking for [DONE].
	foundDone := false
	for {
		var msg json.RawMessage
		if err := websocket.JSON.Receive(ws, &msg); err != nil {
			break
		}
		var chunk server.WSChatChunk
		if err := json.Unmarshal(msg, &chunk); err == nil && chunk.Token == "[DONE]" {
			foundDone = true
			break
		}
	}
	if !foundDone {
		t.Error("expected a [DONE] frame but never received one")
	}
}

// ─── T3: Client disconnect handled ───────────────────────────────────────────

// TestWebSocket_ClientDisconnectHandled closes the WebSocket connection early
// and verifies the server does not panic (goroutine leak would cause a race
// or test hang; this is best-effort without goroutine tracking).
func TestWebSocket_ClientDisconnectHandled(t *testing.T) {
	// Use a slow mock that will still be streaming when we close.
	slowTokens := make([]string, 20)
	for i := range slowTokens {
		slowTokens[i] = "word"
	}

	reg := server.NewRegistry()
	reg.Load("llama3", &mockStreamLLM{tokens: slowTokens})
	srv := server.NewServer(reg)

	ts := httptest.NewServer(srv)
	defer ts.Close()

	ws := dialWS(t, ts, "/v1/ws/chat")

	sendRequest(t, ws, server.WSChatRequest{
		Model:    "llama3",
		Messages: []server.ChatMessage{{Role: "user", Content: "stream please"}},
	})

	// Read just one frame then close abruptly.
	var chunk server.WSChatChunk
	websocket.JSON.Receive(ws, &chunk) //nolint:errcheck
	ws.Close()

	// Give the server a moment to notice the disconnect and clean up.
	// If the server panics the test process will crash, which is the
	// observable failure mode we care about.
}

// ─── T4: Batch fallback (non-streaming model) ────────────────────────────────

func TestWebSocket_BatchFallback(t *testing.T) {
	reg := server.NewRegistry()
	reg.Load("llama3", &mockLLM{reply: "batch reply"})
	srv := server.NewServer(reg)

	ts := httptest.NewServer(srv)
	defer ts.Close()

	ws := dialWS(t, ts, "/v1/ws/chat")
	defer ws.Close()

	sendRequest(t, ws, server.WSChatRequest{
		Model:    "llama3",
		Messages: []server.ChatMessage{{Role: "user", Content: "hello"}},
	})

	got := collectChunks(t, ws)
	if len(got) != 1 || got[0] != "batch reply" {
		t.Errorf("expected single batch frame %q, got %v", "batch reply", got)
	}
}

// ─── T5: Model not found ─────────────────────────────────────────────────────

func TestWebSocket_ModelNotFound(t *testing.T) {
	reg := server.NewRegistry()
	srv := server.NewServer(reg)

	ts := httptest.NewServer(srv)
	defer ts.Close()

	ws := dialWS(t, ts, "/v1/ws/chat")
	defer ws.Close()

	sendRequest(t, ws, server.WSChatRequest{
		Model:    "nonexistent",
		Messages: []server.ChatMessage{{Role: "user", Content: "hi"}},
	})

	// Server should send an error frame.
	var resp map[string]string
	if err := websocket.JSON.Receive(ws, &resp); err != nil {
		t.Fatalf("expected an error frame, got read error: %v", err)
	}
	if resp["error"] == "" {
		t.Errorf("expected non-empty error field in response, got %v", resp)
	}
}

// ─── T6: Context cancellation propagates ────────────────────────────────────

func TestWebSocket_ContextCancellation(t *testing.T) {
	// blockingLLM is a StreamingLLMModel that blocks until its context is
	// cancelled, then closes the channel.
	blockingLLM := &blockingStreamLLM{}

	reg := server.NewRegistry()
	reg.Load("llama3", blockingLLM)
	srv := server.NewServer(reg)

	ts := httptest.NewServer(srv)
	defer ts.Close()

	ws := dialWS(t, ts, "/v1/ws/chat")

	sendRequest(t, ws, server.WSChatRequest{
		Model:    "llama3",
		Messages: []server.ChatMessage{{Role: "user", Content: "block"}},
	})

	// Close immediately — server should notice context cancellation.
	ws.Close()
	// No panic = pass.
}

// blockingStreamLLM streams nothing but respects context cancellation.
type blockingStreamLLM struct{}

func (b *blockingStreamLLM) Close() {}

func (b *blockingStreamLLM) Generate(_ context.Context, _ string, _ int, _ float32) (string, int, int, error) {
	return "", 0, 0, nil
}

func (b *blockingStreamLLM) Stream(ctx context.Context, _ string, _ int, _ float32) (<-chan string, error) {
	ch := make(chan string)
	go func() {
		defer close(ch)
		<-ctx.Done()
	}()
	return ch, nil
}
