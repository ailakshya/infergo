package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// maxBinaryBodySize is the maximum allowed body size for the binary detect
// endpoint to prevent OOM from unbounded reads. 10 MB.
const maxBinaryBodySize = 10 * 1024 * 1024

// detectBufPool is a sync.Pool of reusable byte slices for base64 decoding
// in the JSON detect endpoint, avoiding per-request allocation of ~512KB.
var detectBufPool = sync.Pool{
	New: func() interface{} {
		b := make([]byte, 0, 512*1024) // 512KB initial capacity
		return &b
	},
}

// ─── Capability interfaces ────────────────────────────────────────────────────

// LLMModel is a model that can generate chat completions.
// Implementations must also implement Model (Close).
type LLMModel interface {
	Model
	// Generate runs inference given a prompt string and returns the generated
	// text, plus token counts for billing/usage fields.
	Generate(ctx context.Context, prompt string, maxTokens int, temp float32) (text string, promptToks int, genToks int, err error)
}

// KVSerializable is optionally implemented by LLM models that support
// prefill-decode separation. The model tokenizes a prompt, runs prefill,
// serializes the resulting KV cache, and returns the bytes plus the number
// of prompt tokens processed.
//
// DecodeFromKV accepts those bytes, deserializes the KV cache into a fresh
// sequence slot, and runs generation, returning the generated text.
type KVSerializable interface {
	// PrefillPrompt tokenizes prompt, runs the prefill forward pass, serializes
	// the KV cache, and returns (kvBytes, promptTokenCount, error).
	PrefillPrompt(ctx context.Context, prompt string) (kvBytes []byte, nPromptToks int, err error)

	// DecodeFromKV deserializes kvBytes into a sequence slot and runs generation
	// for up to maxTokens steps, returning the generated text.
	DecodeFromKV(ctx context.Context, kvBytes []byte, nPromptToks int, maxTokens int, temp float32) (text string, err error)
}

// EmbeddingModel is a model that can produce embedding vectors.
type EmbeddingModel interface {
	Model
	Embed(ctx context.Context, input string) ([]float32, error)
}

// BatchEmbeddingModel extends EmbeddingModel with batch support.
type BatchEmbeddingModel interface {
	EmbeddingModel
	EmbedBatch(ctx context.Context, inputs []string) ([][]float32, error)
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
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// ResponseFormat specifies the output format for chat completions.
// Supports OpenAI-compatible response_format field.
type ResponseFormat struct {
	// Type: "text" (default), "json_object" (any valid JSON), or "grammar" (custom GBNF).
	Type string `json:"type"`
	// Grammar is a custom GBNF grammar string. Only used when Type == "grammar".
	Grammar string `json:"grammar,omitempty"`
}

// ChatCompletionRequest mirrors the OpenAI /v1/chat/completions body.
type ChatCompletionRequest struct {
	Model          string          `json:"model"`
	Messages       []ChatMessage   `json:"messages"`
	MaxTokens      int             `json:"max_tokens"`
	Temp           float32         `json:"temperature"`
	Stream         bool            `json:"stream"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
	Tools          []Tool          `json:"tools,omitempty"`
	ToolChoice     interface{}     `json:"tool_choice,omitempty"` // "none"|"auto"|object
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
// Input accepts either a string or an array of strings (batch mode).
type EmbeddingRequest struct {
	Model    string          `json:"model"`
	Input    string          `json:"input,omitempty"` // backward-compat: single string
	InputArr []string        `json:"-"`               // batch (set by UnmarshalInput)
	RawInput json.RawMessage `json:"-"`               // internal: raw JSON
}

// UnmarshalJSON handles both string and []string input formats.
func (r *EmbeddingRequest) UnmarshalJSON(data []byte) error {
	// Parse into a temporary struct to get model + raw input
	type alias struct {
		Model string          `json:"model"`
		Input json.RawMessage `json:"input"`
	}
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	r.Model = a.Model
	r.RawInput = a.Input

	if len(a.Input) == 0 {
		return nil
	}
	// Try string first
	var s string
	if json.Unmarshal(a.Input, &s) == nil {
		r.Input = s
		r.InputArr = []string{s}
		return nil
	}
	// Try array of strings
	var arr []string
	if json.Unmarshal(a.Input, &arr) == nil {
		r.InputArr = arr
		if len(arr) == 1 {
			r.Input = arr[0]
		}
		return nil
	}
	return nil
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

// DetectBackendKey is the context key used to pass a per-request backend
// override from the HTTP handler to the adaptive detector.
type DetectBackendKey struct{}

// DetectRequest is our non-standard /v1/detect body.
type DetectRequest struct {
	Model      string  `json:"model"`
	ImageB64   string  `json:"image_b64"`   // base64-encoded image bytes
	ConfThresh float32 `json:"conf_thresh"`
	IouThresh  float32 `json:"iou_thresh"`
	MaxDet     int     `json:"max_det,omitempty"`  // max detections to return (default 300)
	Classes    []int   `json:"classes,omitempty"`   // filter to specific class IDs (empty = all)
	Backend    string  `json:"backend,omitempty"`   // per-request backend override (e.g. "torch-gpu", "onnx-cuda", "cpu")
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
	ID         string           `json:"id"`
	Object     string           `json:"object"`
	Created    int64            `json:"created"`
	Source     string           `json:"source,omitempty"`
	Format     string           `json:"format,omitempty"`
	ImgSize    int              `json:"imgsz,omitempty"`
	Validation *ModelValidation `json:"validation,omitempty"`
}

// ModelValidation is the validation result attached to a model entry via
// the convert/validate workflow.
type ModelValidation struct {
	Samples int     `json:"samples"`
	MaxDiff float64 `json:"max_diff"`
	Passed  bool    `json:"passed"`
}

// PrefillRequest is the body for POST /v1/prefill.
// The server runs prefill (prompt processing) only, serializes the resulting
// KV cache, and returns it so a decode node can continue generation.
type PrefillRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`
}

// PrefillResponse is returned by POST /v1/prefill.
// KVData is base64-encoded KV cache state for the processed sequence.
type PrefillResponse struct {
	Model       string `json:"model"`
	KVData      string `json:"kv_data"`      // base64-encoded KV cache bytes
	SeqPosition int    `json:"seq_position"` // number of prompt tokens processed
}

// DecodeRequest is the body for POST /v1/decode.
// The node deserializes the provided KV cache and runs generation.
type DecodeRequest struct {
	Model     string `json:"model"`
	KVData    string `json:"kv_data"`    // base64-encoded KV cache bytes (from /v1/prefill)
	MaxTokens int    `json:"max_tokens"`
}

// DecodeResponse is returned by POST /v1/decode.
type DecodeResponse struct {
	Model string `json:"model"`
	Text  string `json:"text"`
}

// errorResponse is returned on any handler error.
type errorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

// ─── Server ───────────────────────────────────────────────────────────────────

// ReloadFunc is called by handleAdminReload to swap in a new model weight file.
// It must be safe to call concurrently and should load the new model into the
// registry under name, atomically replacing any existing entry.
type ReloadFunc func(name, path string) error

// Server holds the HTTP mux and model registry.
type Server struct {
	mux               *http.ServeMux
	registry          *Registry
	reload            ReloadFunc // optional; nil = reload not configured
	mode              string     // "combined" (default), "prefill", or "decode"
	ModelRegistryPath string     // optional; path to models/registry.json for convert/validate metadata
	guardrail         *Guardrail      // optional content safety filter
	templates         *TemplateStore   // server-side prompt templates
	respCache         *ResponseCache   // exact-match response cache (OPT-46)
	memory            *ConversationMemory // multi-turn session memory (OPT-44)
	feedback          *FeedbackStore      // user feedback collection (OPT-114)
	pii               *PIIDetector        // PII detection/redaction (OPT-57)
	metrics           *Metrics         // prometheus metrics (optional)
}

// NewServer creates a Server backed by the given Registry and registers all routes.
func NewServer(reg *Registry) *Server {
	s := &Server{
		mux:      http.NewServeMux(),
		registry: reg,
		mode:     "combined",
	}
	s.mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	s.mux.HandleFunc("POST /v1/completions", s.handleCompletions)
	s.mux.HandleFunc("POST /v1/embeddings", s.handleEmbeddings)
	s.mux.HandleFunc("POST /v1/detect", s.handleDetect)
	s.mux.HandleFunc("POST /v1/detect/binary", s.handleDetectBinary)
	s.mux.HandleFunc("GET /v1/models", s.handleModels)
	s.mux.HandleFunc("POST /v1/admin/reload", s.handleAdminReload)
	s.mux.Handle("GET /v1/ws/chat", s.HandleWSChat())
	// Prefill-decode separation endpoints (OPT-26)
	s.mux.HandleFunc("POST /v1/prefill", s.handlePrefill)
	s.mux.HandleFunc("POST /v1/decode", s.handleDecode)
	s.mux.HandleFunc("POST /v1/embeddings/binary", s.handleEmbeddingsBinary)
	s.mux.HandleFunc("POST /v1/search", s.handleSearch)
	s.mux.HandleFunc("POST /v1/rerank", s.handleRerank)
	s.mux.HandleFunc("POST /v1/detect/stream", s.handleDetectStream)
	s.mux.HandleFunc("POST /v1/images/generations", s.handleImageGeneration)
	s.mux.HandleFunc("POST /v1/audio/transcriptions", s.handleTranscription)
	s.mux.HandleFunc("GET /v1/admin/guardrails", s.handleGuardrailConfig)
	s.mux.HandleFunc("POST /v1/admin/guardrails", s.handleGuardrailConfig)
	s.mux.HandleFunc("GET /v1/admin/pii", s.handlePIIConfig)
	s.mux.HandleFunc("POST /v1/admin/pii", s.handlePIIConfig)
	s.mux.HandleFunc("POST /v1/rag", s.handleRAG)
	s.mux.HandleFunc("POST /v1/ingest", s.handleIngest)
	s.mux.HandleFunc("POST /v1/ingest/url", s.handleIngestURL)
	s.mux.HandleFunc("POST /v1/batches", s.handleBatchCreate)
	s.mux.HandleFunc("GET /v1/batches", s.handleBatchStatus)
	s.mux.HandleFunc("GET /v1/admin/templates", s.handleTemplates)
	s.mux.HandleFunc("POST /v1/admin/templates", s.handleTemplates)
	s.mux.HandleFunc("GET /ui", s.handleWebUI)
	s.mux.HandleFunc("GET /v1/openapi.json", s.handleOpenAPISpec)
	s.mux.HandleFunc("GET /ui/docs", s.handleSwaggerUI)
	s.mux.HandleFunc("DELETE /v1/sessions/{id}", s.handleDeleteSession)
	s.mux.HandleFunc("POST /v1/feedback", s.handleFeedback)
	// NLP task endpoints (OPT-76..80)
	s.mux.HandleFunc("POST /v1/ner", s.handleNER)
	s.mux.HandleFunc("POST /v1/sentiment", s.handleSentiment)
	s.mux.HandleFunc("POST /v1/classify", s.handleClassify)
	s.mux.HandleFunc("POST /v1/summarize", s.handleSummarize)
	// Agent framework (OPT-51)
	s.mux.HandleFunc("POST /v1/agents/run", s.handleAgentRun)
	s.mux.HandleFunc("POST /v1/rag/stream", s.handleStreamingRAG)
	// Code execution sandbox (OPT-50)
	s.mux.HandleFunc("POST /v1/code/execute", s.handleCodeExecute)
	return s
}

// SetReloader injects the function that performs the actual model hot-swap.
// Call this once after NewServer, before serving traffic.
func (s *Server) SetReloader(f ReloadFunc) {
	s.reload = f
}

// SetMode sets the server's operating mode: "combined" (default), "prefill", or "decode".
// In prefill mode only /v1/prefill is active for LLM requests.
// In decode mode only /v1/decode is active for LLM requests.
// In combined mode all endpoints are active.
func (s *Server) SetMode(mode string) {
	s.mode = mode
}

// SetCache injects a ResponseCache for exact-match caching of chat completions.
// When set, identical non-streaming requests return cached responses in < 0.1ms.
func (s *Server) SetCache(c *ResponseCache) {
	s.respCache = c
}

// SetPII injects a PII detector for request scanning.
func (s *Server) SetPII(d *PIIDetector) {
	s.pii = d
}

// SetMetrics injects Prometheus metrics for cache hit/miss tracking.
func (s *Server) SetMetrics(m *Metrics) {
	s.metrics = m
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

func writeReloadError(w http.ResponseWriter, status int, msg string) {
	var resp errorResponse
	resp.Error.Message = msg
	resp.Error.Type = "reload_error"
	writeJSON(w, status, resp)
}

func newID(prefix string) string {
	return fmt.Sprintf("%s-%d", prefix, time.Now().UnixNano())
}

// ─── Handlers ─────────────────────────────────────────────────────────────────

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	// Read body + unmarshal (faster than json.Decoder for small bodies)
	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20)) // 1MB max
	if err != nil {
		writeError(w, http.StatusBadRequest, "read body: "+err.Error())
		return
	}
	var req ChatCompletionRequest
	if err := json.Unmarshal(body, &req); err != nil {
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

	// Resolve response_format to a GBNF grammar string (if any).
	ctx := r.Context()
	if req.ResponseFormat != nil {
		switch req.ResponseFormat.Type {
		case "json_object":
			ctx = WithGrammar(ctx, JSONGrammar)
		case "toon":
			ctx = WithGrammar(ctx, TOONGrammar)
		case "grammar":
			if req.ResponseFormat.Grammar == "" {
				writeError(w, http.StatusBadRequest, "response_format type 'grammar' requires a non-empty 'grammar' field")
				return
			}
			if len(req.ResponseFormat.Grammar) > 65536 {
				writeError(w, http.StatusBadRequest, "grammar exceeds maximum size (64KB)")
				return
			}
			ctx = WithGrammar(ctx, req.ResponseFormat.Grammar)
		case "text", "":
			// No grammar — default text output.
		default:
			writeError(w, http.StatusBadRequest, fmt.Sprintf("unsupported response_format type: %q", req.ResponseFormat.Type))
			return
		}
	}

	// ── Function calling: generate grammar from tools ──
	toolsActive := false
	var resolvedChoice ResolvedToolChoice
	if len(req.Tools) > 0 {
		resolvedChoice = ResolveToolChoice(req.ToolChoice)
		if resolvedChoice.Mode != "none" {
			var fcGrammar string
			switch resolvedChoice.Mode {
			case "function":
				fcGrammar = GenerateFunctionCallGrammar(req.Tools, resolvedChoice.FunctionName)
			case "required":
				fcGrammar = GenerateFunctionCallGrammar(req.Tools, "")
			case "auto":
				fcGrammar = GenerateFunctionCallGrammar(req.Tools, "")
			}
			if fcGrammar != "" {
				ctx = WithGrammar(ctx, fcGrammar)
				toolsActive = true
			}
		}
	}
	_ = resolvedChoice // used only for mode check above

	if req.Stream {
		s.streamChatCompletions(w, ctx, req)
		return
	}

	// ── Response cache: check for cached response (skip if X-No-Cache or streaming) ──
	noCache := r.Header.Get("X-No-Cache") == "true"
	var cacheKey uint64
	if s.respCache != nil && !noCache {
		cacheKey = CacheKey(&req)
		if cached, ok := s.respCache.Get(cacheKey); ok {
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("X-Cache", "HIT")
			w.WriteHeader(http.StatusOK)
			w.Write(cached) //nolint:errcheck
			if s.metrics != nil {
				s.metrics.CacheHits.Inc()
			}
			return
		}
		if s.metrics != nil {
			s.metrics.CacheMisses.Inc()
		}
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

	text, promptToks, genToks, err := llm.Generate(ctx, prompt, maxToks, req.Temp)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "generation failed: "+err.Error())
		return
	}

	// ── Function call response parsing ──
	if toolsActive {
		if tc, ok := ParseFunctionCallResponse(text); ok {
			writeChatCompletionToolCall(w, fastID("chatcmpl"), req.Model, []ToolCall{tc}, promptToks, genToks)
			return
		}
		// If parsing failed, fall through to return as normal text.
	}

	// ── Write response and store in cache ──
	if s.respCache != nil && !noCache {
		w.Header().Set("X-Cache", "MISS")
		resp := buildChatCompletionJSON(fastID("chatcmpl"), req.Model, text, promptToks, genToks)
		s.respCache.Put(cacheKey, resp)
		if s.metrics != nil {
			s.metrics.CacheSize.Set(float64(s.respCache.Len()))
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(resp) //nolint:errcheck
	} else {
		writeChatCompletionFast(w, fastID("chatcmpl"), req.Model, text, promptToks, genToks)
	}
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
	if len(req.InputArr) == 0 {
		writeError(w, http.StatusBadRequest, "input must not be empty")
		return
	}
	for _, inp := range req.InputArr {
		if inp == "" {
			writeError(w, http.StatusBadRequest, "input strings must not be empty")
			return
		}
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

	// Batch path: if model supports batch and multiple inputs provided.
	var vecs [][]float32
	if batch, ok2 := emb.(BatchEmbeddingModel); ok2 && len(req.InputArr) > 1 {
		vecs, err = batch.EmbedBatch(r.Context(), req.InputArr)
	} else {
		// Single or sequential fallback.
		vecs = make([][]float32, len(req.InputArr))
		for i, input := range req.InputArr {
			vecs[i], err = emb.Embed(r.Context(), input)
			if err != nil {
				break
			}
		}
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, "embedding failed: "+err.Error())
		return
	}

	// Use fast JSON writer — skips json.Encoder overhead for float arrays
	writeEmbeddingJSONFast(w, req.Model, vecs, req.InputArr)
	return
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

	// Inject per-request backend override into context for adaptive detector.
	ctx := r.Context()
	if req.Backend != "" {
		ctx = context.WithValue(ctx, DetectBackendKey{}, req.Backend)
	}

	objs, err := det.Detect(ctx, imgBytes, confThresh, iouThresh)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "detection failed: "+err.Error())
		return
	}

	objs = FilterDetections(objs, req.Classes, req.MaxDet)
	writeJSON(w, http.StatusOK, DetectResponse{Model: req.Model, Objects: objs})
}

// handleDetectBinary accepts raw image bytes directly in the request body,
// eliminating the base64 encoding overhead (~33% larger payload) and JSON
// parsing of the large image string. This saves ~3ms per request compared
// to the JSON-based /v1/detect endpoint.
//
// Model name is provided via query param (?model=yolo11n) or X-Model header.
// Confidence and IoU thresholds are optional query params with sensible defaults.
func (s *Server) handleDetectBinary(w http.ResponseWriter, r *http.Request) {
	// Model name from query param or header.
	model := r.URL.Query().Get("model")
	if model == "" {
		model = r.Header.Get("X-Model")
	}
	if model == "" {
		writeError(w, http.StatusBadRequest, "model required (query param or X-Model header)")
		return
	}

	// Thresholds and limits from query params (defaults: 0.25, 0.45, 300).
	confThresh := parseFloat32(r.URL.Query().Get("conf"), 0.25)
	iouThresh := parseFloat32(r.URL.Query().Get("iou"), 0.45)
	maxDet := parseInt(r.URL.Query().Get("max_det"), 0)
	classes := parseIntSlice(r.URL.Query().Get("classes"))

	// Read raw image bytes directly — no base64, no JSON.
	// Limit body size to prevent OOM.
	imageBytes, err := io.ReadAll(io.LimitReader(r.Body, maxBinaryBodySize+1))
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request body: "+err.Error())
		return
	}
	if len(imageBytes) == 0 {
		writeError(w, http.StatusBadRequest, "empty or unreadable body")
		return
	}
	if len(imageBytes) > maxBinaryBodySize {
		writeError(w, http.StatusRequestEntityTooLarge, "request body exceeds 10MB limit")
		return
	}

	ref, err := s.registry.Get(model)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	defer ref.Release()

	det, ok := ref.Model.(DetectionModel)
	if !ok {
		writeError(w, http.StatusBadRequest, fmt.Sprintf("model %q does not support detection", model))
		return
	}

	// Inject per-request backend override into context for adaptive detector.
	ctx := r.Context()
	if backendOverride := r.URL.Query().Get("backend"); backendOverride != "" {
		ctx = context.WithValue(ctx, DetectBackendKey{}, backendOverride)
	} else if backendOverride := r.Header.Get("X-Detect-Backend"); backendOverride != "" {
		ctx = context.WithValue(ctx, DetectBackendKey{}, backendOverride)
	}

	objs, err := det.Detect(ctx, imageBytes, confThresh, iouThresh)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "detection failed: "+err.Error())
		return
	}

	objs = FilterDetections(objs, classes, maxDet)
	writeJSON(w, http.StatusOK, DetectResponse{Model: model, Objects: objs})
}

// parseFloat32 parses a string as float32, returning def if the string is
// empty or unparseable.
func parseFloat32(s string, def float32) float32 {
	if s == "" {
		return def
	}
	v, err := strconv.ParseFloat(s, 32)
	if err != nil {
		return def
	}
	return float32(v)
}

// parseInt parses a string as int, returning def if the string is empty or
// unparseable.
func parseInt(s string, def int) int {
	if s == "" {
		return def
	}
	v, err := strconv.Atoi(s)
	if err != nil {
		return def
	}
	return v
}

// parseIntSlice parses a comma-separated string of integers (e.g. "0,2,5")
// into an int slice. Returns nil if the string is empty or all values are
// unparseable.
func parseIntSlice(s string) []int {
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	var result []int
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.Atoi(p)
		if err != nil {
			continue
		}
		result = append(result, v)
	}
	return result
}

// FilterDetections applies class filtering and max-detection limiting to a
// slice of detections. The input is assumed to be sorted by confidence
// descending (as returned by NMS). If classes is non-empty, only detections
// with a ClassID in that set are kept. If maxDet > 0, the result is truncated
// to at most maxDet entries. If maxDet is 0, the default limit of 300 is used.
func FilterDetections(objs []DetectedObject, classes []int, maxDet int) []DetectedObject {
	// Apply class filter if specified.
	if len(classes) > 0 {
		classSet := make(map[int]bool, len(classes))
		for _, c := range classes {
			classSet[c] = true
		}
		filtered := make([]DetectedObject, 0, len(objs))
		for _, o := range objs {
			if classSet[o.ClassID] {
				filtered = append(filtered, o)
			}
		}
		objs = filtered
	}

	// Apply max_det limit (default 300).
	limit := maxDet
	if limit <= 0 {
		limit = 300
	}
	if len(objs) > limit {
		objs = objs[:limit]
	}

	return objs
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	names := s.registry.Names()
	data := make([]ModelInfo, len(names))
	now := time.Now().Unix()

	// Try to load convert/validate registry metadata for enriched responses.
	regEntries := s.loadModelRegistryMetadata()

	for i, n := range names {
		info := ModelInfo{ID: n, Object: "model", Created: now}
		// Enrich with registry metadata if available (match by name).
		if entry, ok := regEntries[n]; ok {
			info.Source = entry.Source
			info.Format = entry.Format
			info.ImgSize = entry.ImgSize
			if entry.Validation != nil {
				info.Validation = &ModelValidation{
					Samples: entry.Validation.Samples,
					MaxDiff: entry.Validation.MaxDiff,
					Passed:  entry.Validation.Passed,
				}
			}
		}
		data[i] = info
	}
	writeJSON(w, http.StatusOK, ModelListResponse{Object: "list", Data: data})
}

// modelRegistryFile is the on-disk format for entries written by "infergo convert".
type modelRegistryFile struct {
	Name       string                   `json:"name"`
	Source     string                   `json:"source"`
	Export     string                   `json:"export"`
	Format     string                   `json:"format"`
	ImgSize    int                      `json:"imgsz"`
	Validation *modelRegistryValidation `json:"validation,omitempty"`
}

type modelRegistryValidation struct {
	Samples int     `json:"samples"`
	MaxDiff float64 `json:"max_diff"`
	Passed  bool    `json:"passed"`
}

// loadModelRegistryMetadata reads the convert/validate registry file and
// returns a map keyed by entry name. Returns nil on any error.
func (s *Server) loadModelRegistryMetadata() map[string]modelRegistryFile {
	if s.ModelRegistryPath == "" {
		return nil
	}
	data, err := os.ReadFile(s.ModelRegistryPath)
	if err != nil {
		return nil
	}
	var entries []modelRegistryFile
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil
	}
	m := make(map[string]modelRegistryFile, len(entries))
	for _, e := range entries {
		m[e.Name] = e
	}
	return m
}

// ─── helpers ─────────────────────────────────────────────────────────────────

// buildPrompt flattens a message list into a ChatML prompt.
// ChatML is the standard format for Qwen, LLaMA 3, Mistral, etc.
// <|im_start|>role\ncontent<|im_end|>\n
func buildPrompt(messages []ChatMessage) string {
	var sb strings.Builder
	sb.Grow(256)
	for _, m := range messages {
		sb.WriteString("<|im_start|>")
		sb.WriteString(m.Role)
		sb.WriteByte('\n')
		sb.WriteString(m.Content)
		sb.WriteString("<|im_end|>\n")
	}
	sb.WriteString("<|im_start|>assistant\n")
	return sb.String()
}

func decodeBase64(s string) ([]byte, error) {
	return base64.StdEncoding.DecodeString(s)
}

// ─── Admin handlers ───────────────────────────────────────────────────────────

// ReloadRequest is the body for POST /v1/admin/reload.
type ReloadRequest struct {
	Model string `json:"model"`
	Path  string `json:"path"`
}

// ReloadResponse is the success body for POST /v1/admin/reload.
type ReloadResponse struct {
	Status string `json:"status"`
	Model  string `json:"model"`
	Path   string `json:"path"`
}

func (s *Server) handleAdminReload(w http.ResponseWriter, r *http.Request) {
	if s.reload == nil {
		writeReloadError(w, http.StatusNotImplemented, "reload not configured")
		return
	}

	var req ReloadRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeReloadError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	if req.Model == "" {
		writeReloadError(w, http.StatusBadRequest, "model field is required")
		return
	}
	if req.Path == "" {
		writeReloadError(w, http.StatusBadRequest, "path field is required")
		return
	}

	// Validate file exists before calling the reloader.
	if _, err := os.Stat(req.Path); err != nil {
		writeReloadError(w, http.StatusBadRequest, "model file not found: "+err.Error())
		return
	}

	if err := s.reload(req.Model, req.Path); err != nil {
		writeReloadError(w, http.StatusInternalServerError, "reload failed: "+err.Error())
		return
	}

	writeJSON(w, http.StatusOK, ReloadResponse{
		Status: "ok",
		Model:  req.Model,
		Path:   req.Path,
	})
}

// ─── Prefill-Decode separation handlers (OPT-26) ─────────────────────────────

// handlePrefill implements POST /v1/prefill.
// It tokenizes the provided messages, runs the prefill forward pass, serializes
// the resulting KV cache, and returns it base64-encoded.
// Only active when --mode combined or --mode prefill.
func (s *Server) handlePrefill(w http.ResponseWriter, r *http.Request) {
	if s.mode == "decode" {
		writeError(w, http.StatusNotFound, "/v1/prefill is not available in decode mode")
		return
	}

	var req PrefillRequest
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

	kvsm, ok := ref.Model.(KVSerializable)
	if !ok {
		writeError(w, http.StatusBadRequest, fmt.Sprintf("model %q does not support prefill-decode separation", req.Model))
		return
	}

	prompt := buildPrompt(req.Messages)
	kvBytes, nPromptToks, err := kvsm.PrefillPrompt(r.Context(), prompt)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "prefill failed: "+err.Error())
		return
	}

	writeJSON(w, http.StatusOK, PrefillResponse{
		Model:       req.Model,
		KVData:      base64.StdEncoding.EncodeToString(kvBytes),
		SeqPosition: nPromptToks,
	})
}

// handleDecode implements POST /v1/decode.
// It accepts base64-encoded KV cache bytes (from /v1/prefill), deserializes them
// into a fresh sequence slot, and runs generation, returning the generated text.
// Only active when --mode combined or --mode decode.
func (s *Server) handleDecode(w http.ResponseWriter, r *http.Request) {
	if s.mode == "prefill" {
		writeError(w, http.StatusNotFound, "/v1/decode is not available in prefill mode")
		return
	}

	var req DecodeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "model field is required")
		return
	}
	if req.KVData == "" {
		writeError(w, http.StatusBadRequest, "kv_data must not be empty")
		return
	}

	kvBytes, err := base64.StdEncoding.DecodeString(req.KVData)
	if err != nil {
		writeError(w, http.StatusBadRequest, "kv_data: invalid base64: "+err.Error())
		return
	}

	ref, err := s.registry.Get(req.Model)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	defer ref.Release()

	kvsm, ok := ref.Model.(KVSerializable)
	if !ok {
		writeError(w, http.StatusBadRequest, fmt.Sprintf("model %q does not support prefill-decode separation", req.Model))
		return
	}

	maxToks := req.MaxTokens
	if maxToks <= 0 {
		maxToks = 256
	}

	text, err := kvsm.DecodeFromKV(r.Context(), kvBytes, 0, maxToks, 0)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "decode failed: "+err.Error())
		return
	}

	writeJSON(w, http.StatusOK, DecodeResponse{
		Model: req.Model,
		Text:  text,
	})
}
