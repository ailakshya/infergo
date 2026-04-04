package server

import (
	"context"
	"fmt"
	"net/http"

	"golang.org/x/net/websocket"
)

// WSChatRequest is the single JSON message clients send over WebSocket.
type WSChatRequest struct {
	Model     string        `json:"model"`
	Messages  []ChatMessage `json:"messages"`
	MaxTokens int           `json:"max_tokens"`
	Temp      float32       `json:"temperature"`
}

// WSChatChunk is each token frame sent back to the client.
type WSChatChunk struct {
	Token string `json:"token"` // "[DONE]" on final frame
}

// HandleWSChat returns an http.Handler for GET /v1/ws/chat.
func (s *Server) HandleWSChat() http.Handler {
	return websocket.Handler(s.wsChat)
}

// wsChat is the WebSocket handler for /v1/ws/chat.
// Protocol:
//  1. Client sends one JSON message: WSChatRequest
//  2. Server streams token frames: {"token":"word"} … {"token":"[DONE]"}
//  3. Server closes the connection after [DONE]
//  4. Client disconnect mid-stream cancels the context, freeing the sequence.
func (s *Server) wsChat(ws *websocket.Conn) {
	// Derive a cancellable context so that a client disconnect propagates
	// through to any ongoing model.Stream / model.Generate call.
	ctx, cancel := context.WithCancel(ws.Request().Context())
	defer cancel()

	// 1. Read the client's request message.
	var req WSChatRequest
	if err := websocket.JSON.Receive(ws, &req); err != nil {
		// Client disconnected before sending a request — nothing to do.
		return
	}

	// 2. Validate fields.
	if req.Model == "" {
		websocket.JSON.Send(ws, map[string]string{"error": "model field is required"}) //nolint:errcheck
		return
	}
	if len(req.Messages) == 0 {
		websocket.JSON.Send(ws, map[string]string{"error": "messages must not be empty"}) //nolint:errcheck
		return
	}

	// 3. Look up the model in the registry.
	ref, err := s.registry.Get(req.Model)
	if err != nil {
		websocket.JSON.Send(ws, map[string]string{"error": err.Error()}) //nolint:errcheck
		return
	}
	defer ref.Release()

	prompt := buildPrompt(req.Messages)
	maxToks := req.MaxTokens
	if maxToks <= 0 {
		maxToks = 256
	}

	// 4. Prefer the streaming interface; fall back to batch generation.
	if sllm, ok := ref.Model.(StreamingLLMModel); ok {
		s.wsChatStream(ctx, ws, sllm, prompt, maxToks, req.Temp)
		return
	}

	llm, ok := ref.Model.(LLMModel)
	if !ok {
		websocket.JSON.Send(ws, map[string]string{ //nolint:errcheck
			"error": fmt.Sprintf("model %q does not support chat completions", req.Model),
		})
		return
	}
	s.wsChatBatch(ctx, ws, llm, prompt, maxToks, req.Temp)
}

// wsChatStream uses StreamingLLMModel.Stream to send tokens one frame at a time.
func (s *Server) wsChatStream(ctx context.Context, ws *websocket.Conn, sllm StreamingLLMModel, prompt string, maxToks int, temp float32) {
	tokens, err := sllm.Stream(ctx, prompt, maxToks, temp)
	if err != nil {
		websocket.JSON.Send(ws, map[string]string{"error": "stream failed: " + err.Error()}) //nolint:errcheck
		return
	}

	for {
		select {
		case tok, ok := <-tokens:
			if !ok {
				// Channel closed — generation complete, send [DONE].
				websocket.JSON.Send(ws, WSChatChunk{Token: "[DONE]"}) //nolint:errcheck
				return
			}
			if err := websocket.JSON.Send(ws, WSChatChunk{Token: tok}); err != nil {
				// Client disconnected mid-stream; cancel propagates upstream.
				return
			}
		case <-ctx.Done():
			return
		}
	}
}

// wsChatBatch uses LLMModel.Generate (blocking) and sends the result as a
// single content frame followed by [DONE].
func (s *Server) wsChatBatch(ctx context.Context, ws *websocket.Conn, llm LLMModel, prompt string, maxToks int, temp float32) {
	text, _, _, err := llm.Generate(ctx, prompt, maxToks, temp)
	if err != nil {
		websocket.JSON.Send(ws, map[string]string{"error": "generation failed: " + err.Error()}) //nolint:errcheck
		return
	}

	if err := websocket.JSON.Send(ws, WSChatChunk{Token: text}); err != nil {
		return
	}
	websocket.JSON.Send(ws, WSChatChunk{Token: "[DONE]"}) //nolint:errcheck
}
