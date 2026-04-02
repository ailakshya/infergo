package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// StreamingLLMModel extends LLMModel with a token-streaming interface.
// Stream returns a channel that emits tokens one at a time and is closed
// when generation ends or the context is cancelled.
// The channel sends an empty string to signal end-of-generation before closing.
type StreamingLLMModel interface {
	LLMModel
	Stream(ctx context.Context, prompt string, maxTokens int, temp float32) (<-chan string, error)
}

// ─── SSE chunk types ─────────────────────────────────────────────────────────

// ChatCompletionChunk is one SSE event in a streaming chat response.
type ChatCompletionChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []ChunkChoice `json:"choices"`
}

// ChunkChoice carries the incremental delta for one streaming step.
type ChunkChoice struct {
	Index        int     `json:"index"`
	Delta        Delta   `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

// Delta holds the incremental content (or role for the first chunk).
type Delta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// ─── SSE writer ──────────────────────────────────────────────────────────────

// sseWriter wraps an http.ResponseWriter and sends Server-Sent Events.
type sseWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
}

func newSSEWriter(w http.ResponseWriter) (*sseWriter, bool) {
	f, ok := w.(http.Flusher)
	if !ok {
		return nil, false
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	return &sseWriter{w: w, flusher: f}, true
}

func (s *sseWriter) sendEvent(data string) error {
	_, err := fmt.Fprintf(s.w, "data: %s\n\n", data)
	if err != nil {
		return err
	}
	s.flusher.Flush()
	return nil
}

func (s *sseWriter) sendDone() {
	fmt.Fprint(s.w, "data: [DONE]\n\n")
	s.flusher.Flush()
}

// ─── Streaming handler ────────────────────────────────────────────────────────

// streamChatCompletions handles the streaming path of /v1/chat/completions.
// Called from handleChatCompletions when req.Stream == true.
func (s *Server) streamChatCompletions(w http.ResponseWriter, r *http.Request, req ChatCompletionRequest) {
	ref, err := s.registry.Get(req.Model)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	defer ref.Release()

	sllm, ok := ref.Model.(StreamingLLMModel)
	if !ok {
		// Fall back: check if it at least supports non-streaming, then buffer.
		llm, ok2 := ref.Model.(LLMModel)
		if !ok2 {
			writeError(w, http.StatusBadRequest, fmt.Sprintf("model %q does not support chat completions", req.Model))
			return
		}
		s.streamViaBatch(w, r, req, llm)
		return
	}

	prompt := buildPrompt(req.Messages)
	maxToks := req.MaxTokens
	if maxToks <= 0 {
		maxToks = 256
	}

	tokens, err := sllm.Stream(r.Context(), prompt, maxToks, req.Temp)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "stream failed: "+err.Error())
		return
	}

	sse, ok := newSSEWriter(w)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported by transport")
		return
	}

	id := newID("chatcmpl")
	created := time.Now().Unix()

	// First chunk: role only
	s.sendChunk(sse, id, created, req.Model, Delta{Role: "assistant"}, nil)

	// Content chunks
	for tok := range tokens {
		if err := s.sendChunk(sse, id, created, req.Model, Delta{Content: tok}, nil); err != nil {
			return
		}
	}

	// Final chunk: finish_reason=stop, empty delta
	stop := "stop"
	s.sendChunk(sse, id, created, req.Model, Delta{}, &stop)
	sse.sendDone()
}

// streamViaBatch buffers the full response then streams it as a single chunk.
// Used when the model implements LLMModel but not StreamingLLMModel.
func (s *Server) streamViaBatch(w http.ResponseWriter, r *http.Request, req ChatCompletionRequest, llm LLMModel) {
	prompt := buildPrompt(req.Messages)
	maxToks := req.MaxTokens
	if maxToks <= 0 {
		maxToks = 256
	}

	text, _, _, err := llm.Generate(r.Context(), prompt, maxToks, req.Temp)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "generation failed: "+err.Error())
		return
	}

	sse, ok := newSSEWriter(w)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported by transport")
		return
	}

	id := newID("chatcmpl")
	created := time.Now().Unix()
	s.sendChunk(sse, id, created, req.Model, Delta{Role: "assistant"}, nil)
	s.sendChunk(sse, id, created, req.Model, Delta{Content: text}, nil)
	stop := "stop"
	s.sendChunk(sse, id, created, req.Model, Delta{}, &stop)
	sse.sendDone()
}

func (s *Server) sendChunk(sse *sseWriter, id string, created int64, model string, delta Delta, finishReason *string) error {
	chunk := ChatCompletionChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []ChunkChoice{{
			Index:        0,
			Delta:        delta,
			FinishReason: finishReason,
		}},
	}
	b, err := json.Marshal(chunk)
	if err != nil {
		return err
	}
	return sse.sendEvent(string(b))
}
