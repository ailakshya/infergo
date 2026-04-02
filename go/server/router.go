package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// ─── Capability interfaces ────────────────────────────────────────────────────

// LLMModel is a model that can generate chat completions.
// Implementations must also implement Model (Close).
type LLMModel interface {
	Model
	// Generate runs inference given a prompt string and returns the generated
	// text, plus token counts for billing/usage fields.
	Generate(ctx context.Context, prompt string, maxTokens int, temp float32) (text string, promptToks int, genToks int, err error)
}

// EmbeddingModel is a model that can produce embedding vectors.
type EmbeddingModel interface {
	Model
	Embed(ctx context.Context, input string) ([]float32, error)
}

// DetectionModel is a model that can run object detection on raw image bytes.
type DetectionModel interface {
	Model
	Detect(ctx context.Context, imageBytes []byte, confThresh, iouThresh float32) ([]DetectedObject, error)
}

// DetectedObject is a single detection result (used in /v1/detect response).
type DetectedObject struct {
	X1, Y1, X2, Y2 float32
	ClassID         int
	Confidence      float32
}

// ─── OpenAI-compatible request / response types ───────────────────────────────

// ChatMessage is a single turn in a conversation.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatCompletionRequest mirrors the OpenAI /v1/chat/completions body.
type ChatCompletionRequest struct {
	Model     string        `json:"model"`
	Messages  []ChatMessage `json:"messages"`
	MaxTokens int           `json:"max_tokens"`
	Temp      float32       `json:"temperature"`
	Stream    bool          `json:"stream"`
}

// ChatCompletionResponse mirrors the OpenAI chat completion object.
type ChatCompletionResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []ChatChoice   `json:"choices"`
	Usage   UsageInfo      `json:"usage"`
}

type ChatChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type UsageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// EmbeddingRequest mirrors the OpenAI /v1/embeddings body.
type EmbeddingRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

// EmbeddingResponse mirrors the OpenAI embeddings object.
type EmbeddingResponse struct {
	Object string          `json:"object"`
	Model  string          `json:"model"`
	Data   []EmbeddingData `json:"data"`
	Usage  UsageInfo       `json:"usage"`
}

type EmbeddingData struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

// CompletionRequest mirrors the OpenAI /v1/completions body.
type CompletionRequest struct {
	Model     string  `json:"model"`
	Prompt    string  `json:"prompt"`
	MaxTokens int     `json:"max_tokens"`
	Temp      float32 `json:"temperature"`
}

// CompletionResponse mirrors the OpenAI text completion object.
type CompletionResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []CompletionChoice `json:"choices"`
	Usage   UsageInfo          `json:"usage"`
}

type CompletionChoice struct {
	Index        int    `json:"index"`
	Text         string `json:"text"`
	FinishReason string `json:"finish_reason"`
}

// DetectRequest is our non-standard /v1/detect body.
type DetectRequest struct {
	Model      string  `json:"model"`
	ImageB64   string  `json:"image_b64"`   // base64-encoded image bytes
	ConfThresh float32 `json:"conf_thresh"`
	IouThresh  float32 `json:"iou_thresh"`
}

type DetectResponse struct {
	Model   string           `json:"model"`
	Objects []DetectedObject `json:"objects"`
}

// ModelListResponse mirrors GET /v1/models.
type ModelListResponse struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}

type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
}

// errorResponse is returned on any handler error.
type errorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

// ─── Server ───────────────────────────────────────────────────────────────────

// Server holds the HTTP mux and model registry.
type Server struct {
	mux      *http.ServeMux
	registry *Registry
}

// NewServer creates a Server backed by the given Registry and registers all routes.
func NewServer(reg *Registry) *Server {
	s := &Server{
		mux:      http.NewServeMux(),
		registry: reg,
	}
	s.mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	s.mux.HandleFunc("POST /v1/completions", s.handleCompletions)
	s.mux.HandleFunc("POST /v1/embeddings", s.handleEmbeddings)
	s.mux.HandleFunc("POST /v1/detect", s.handleDetect)
	s.mux.HandleFunc("GET /v1/models", s.handleModels)
	return s
}

// ServeHTTP implements http.Handler.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mux.ServeHTTP(w, r)
}

// ─── helpers ─────────────────────────────────────────────────────────────────

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	var resp errorResponse
	resp.Error.Message = msg
	resp.Error.Type = "invalid_request_error"
	writeJSON(w, status, resp)
}

func newID(prefix string) string {
	return fmt.Sprintf("%s-%d", prefix, time.Now().UnixNano())
}

// ─── Handlers ─────────────────────────────────────────────────────────────────

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "model field is required")
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages must not be empty")
		return
	}

	ref, err := s.registry.Get(req.Model)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	defer ref.Release()

	llm, ok := ref.Model.(LLMModel)
	if !ok {
		writeError(w, http.StatusBadRequest, fmt.Sprintf("model %q does not support chat completions", req.Model))
		return
	}

	// Build a simple prompt from messages
	prompt := buildPrompt(req.Messages)
	maxToks := req.MaxTokens
	if maxToks <= 0 {
		maxToks = 256
	}

	text, promptToks, genToks, err := llm.Generate(r.Context(), prompt, maxToks, req.Temp)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "generation failed: "+err.Error())
		return
	}

	resp := ChatCompletionResponse{
		ID:      newID("chatcmpl"),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []ChatChoice{{
			Index:        0,
			Message:      ChatMessage{Role: "assistant", Content: text},
			FinishReason: "stop",
		}},
		Usage: UsageInfo{
			PromptTokens:     promptToks,
			CompletionTokens: genToks,
			TotalTokens:      promptToks + genToks,
		},
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleCompletions(w http.ResponseWriter, r *http.Request) {
	var req CompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "model field is required")
		return
	}
	if req.Prompt == "" {
		writeError(w, http.StatusBadRequest, "prompt must not be empty")
		return
	}

	ref, err := s.registry.Get(req.Model)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	defer ref.Release()

	llm, ok := ref.Model.(LLMModel)
	if !ok {
		writeError(w, http.StatusBadRequest, fmt.Sprintf("model %q does not support completions", req.Model))
		return
	}

	maxToks := req.MaxTokens
	if maxToks <= 0 {
		maxToks = 256
	}

	text, promptToks, genToks, err := llm.Generate(r.Context(), req.Prompt, maxToks, req.Temp)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "generation failed: "+err.Error())
		return
	}

	resp := CompletionResponse{
		ID:      newID("cmpl"),
		Object:  "text_completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []CompletionChoice{{
			Index:        0,
			Text:         text,
			FinishReason: "stop",
		}},
		Usage: UsageInfo{
			PromptTokens:     promptToks,
			CompletionTokens: genToks,
			TotalTokens:      promptToks + genToks,
		},
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	var req EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "model field is required")
		return
	}
	if req.Input == "" {
		writeError(w, http.StatusBadRequest, "input must not be empty")
		return
	}

	ref, err := s.registry.Get(req.Model)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	defer ref.Release()

	emb, ok := ref.Model.(EmbeddingModel)
	if !ok {
		writeError(w, http.StatusBadRequest, fmt.Sprintf("model %q does not support embeddings", req.Model))
		return
	}

	vec, err := emb.Embed(r.Context(), req.Input)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "embedding failed: "+err.Error())
		return
	}

	resp := EmbeddingResponse{
		Object: "list",
		Model:  req.Model,
		Data: []EmbeddingData{{
			Object:    "embedding",
			Index:     0,
			Embedding: vec,
		}},
		Usage: UsageInfo{PromptTokens: len(req.Input), TotalTokens: len(req.Input)},
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleDetect(w http.ResponseWriter, r *http.Request) {
	var req DetectRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "model field is required")
		return
	}
	if req.ImageB64 == "" {
		writeError(w, http.StatusBadRequest, "image_b64 must not be empty")
		return
	}

	ref, err := s.registry.Get(req.Model)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	defer ref.Release()

	det, ok := ref.Model.(DetectionModel)
	if !ok {
		writeError(w, http.StatusBadRequest, fmt.Sprintf("model %q does not support detection", req.Model))
		return
	}

	imgBytes, err := decodeBase64(req.ImageB64)
	if err != nil {
		writeError(w, http.StatusBadRequest, "image_b64: "+err.Error())
		return
	}

	confThresh := req.ConfThresh
	if confThresh == 0 {
		confThresh = 0.25
	}
	iouThresh := req.IouThresh
	if iouThresh == 0 {
		iouThresh = 0.45
	}

	objs, err := det.Detect(r.Context(), imgBytes, confThresh, iouThresh)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "detection failed: "+err.Error())
		return
	}

	writeJSON(w, http.StatusOK, DetectResponse{Model: req.Model, Objects: objs})
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	names := s.registry.Names()
	data := make([]ModelInfo, len(names))
	now := time.Now().Unix()
	for i, n := range names {
		data[i] = ModelInfo{ID: n, Object: "model", Created: now}
	}
	writeJSON(w, http.StatusOK, ModelListResponse{Object: "list", Data: data})
}

// ─── helpers ─────────────────────────────────────────────────────────────────

// buildPrompt flattens a message list into a plain text prompt.
func buildPrompt(messages []ChatMessage) string {
	var out string
	for _, m := range messages {
		switch m.Role {
		case "system":
			out += "[system]: " + m.Content + "\n"
		case "user":
			out += "[user]: " + m.Content + "\n"
		case "assistant":
			out += "[assistant]: " + m.Content + "\n"
		default:
			out += m.Content + "\n"
		}
	}
	return out
}

func decodeBase64(s string) ([]byte, error) {
	return base64.StdEncoding.DecodeString(s)
}
