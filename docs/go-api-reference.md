# Go API Reference

Import path: `github.com/ailakshya/infergo`

---

## Package `server`

The `server` package is the OpenAI-compatible HTTP serving layer with 30+ endpoints, caching, PII detection, RBAC, function calling, agents, NLP tasks, and more. Use it to embed infergo inside your own Go application.

### Quick example

```go
import (
    "net/http"
    "github.com/ailakshya/infergo/server"
)

reg := server.NewRegistry()
reg.Load("my-model", myModel) // myModel implements server.Model

srv := server.NewServer(reg)
metrics := server.NewMetrics()
health := server.NewHealthChecker(reg, 1)

// Optional features
srv.SetCache(server.NewResponseCache(1000))
srv.SetPII(server.NewPIIDetector("redact"))
srv.SetSemanticCache(server.NewSemanticCache(embeddingModel, 0.92))
srv.SetTenants(server.NewTenantStore())
srv.SetCanary(server.NewCanaryDeploy())
srv.SetMetrics(metrics)

mux := http.NewServeMux()
mux.Handle("/v1/", metrics.WrapServer(srv))
health.RegisterRoutes(mux)
mux.Handle("/metrics", metrics.Handler())

http.ListenAndServe(":9090", mux)
```

---

### Model interfaces

#### `Model`

```go
type Model interface {
    Close()
}
```

Base interface all registered models must implement. `Close` releases C-side resources (GPU memory, file handles).

#### `LLMModel`

```go
type LLMModel interface {
    Model
    Generate(ctx context.Context, prompt string, maxTokens int, temp float32) (text string, promptToks int, genToks int, err error)
}
```

Implement this for any model that handles `/v1/chat/completions` and `/v1/completions`.

#### `StreamingLLMModel`

```go
type StreamingLLMModel interface {
    LLMModel
    Stream(ctx context.Context, prompt string, maxTokens int, temp float32) (<-chan string, error)
}
```

Optional extension of `LLMModel`. If your model implements `Stream`, the server sends real token-by-token SSE instead of buffering the full response. The channel must be closed when generation is complete.

#### `KVSerializable`

```go
type KVSerializable interface {
    PrefillPrompt(ctx context.Context, prompt string) (kvBytes []byte, nPromptToks int, err error)
    DecodeFromKV(ctx context.Context, kvBytes []byte, nPromptToks int, maxTokens int, temp float32) (text string, err error)
}
```

Enables prefill-decode separation. The model tokenizes a prompt, runs prefill, serializes the KV cache, and returns the bytes. `DecodeFromKV` deserializes the KV cache and runs generation.

#### `EmbeddingModel`

```go
type EmbeddingModel interface {
    Model
    Embed(ctx context.Context, input string) ([]float32, error)
}
```

Handles `/v1/embeddings`. Input is text or a base64-encoded image.

#### `BatchEmbeddingModel`

```go
type BatchEmbeddingModel interface {
    EmbeddingModel
    EmbedBatch(ctx context.Context, inputs []string) ([][]float32, error)
}
```

Extends `EmbeddingModel` with batch support for multiple inputs in one call.

#### `DetectionModel`

```go
type DetectionModel interface {
    Model
    Detect(ctx context.Context, imageBytes []byte, confThresh, iouThresh float32) ([]DetectedObject, error)
}
```

Handles `/v1/detect` and `/v1/detect/binary`.

```go
type DetectedObject struct {
    X1, Y1, X2, Y2 float32
    ClassID         int
    Confidence      float32
}
```

#### `SearchModel`

```go
type SearchModel interface {
    Model
    Search(ctx context.Context, query string, k int) ([]SearchHit, error)
    IndexSize() int
}
```

Handles vector search via `/v1/search`.

#### `HybridSearchModel`

```go
type HybridSearchModel interface {
    SearchModel
    SearchBM25(ctx context.Context, query string, k int) ([]SearchHit, error)
    SearchHybrid(ctx context.Context, query string, k int, alpha float32) ([]SearchHit, error)
}
```

Extends `SearchModel` with BM25 keyword search and hybrid (BM25 + vector) search.

#### `LoRAModel`

```go
type LoRAModel interface {
    Model
    LoadAdapter(name, path string) error
    UnloadAdapter(name string) error
    ListAdapters() []LoRAAdapter
}
```

Supports hot-swapping LoRA adapters per request.

---

### Endpoints

`Server` implements `http.Handler` and registers all routes automatically. The full endpoint table:

#### Core inference

| Method + Path | Description |
|---|---|
| `POST /v1/chat/completions` | LLM chat (streaming or batch), function calling, grammar constraints |
| `POST /v1/completions` | LLM text completion |
| `POST /v1/embeddings` | Embedding vector (single or batch) |
| `POST /v1/embeddings/binary` | Embedding from raw bytes |
| `POST /v1/detect` | Object detection (JSON with base64 image) |
| `POST /v1/detect/binary` | Object detection (raw JPEG/PNG bytes) |
| `POST /v1/detect/stream` | Streaming detection (video frames) |
| `GET /v1/models` | List loaded models |

#### Search and RAG

| Method + Path | Description |
|---|---|
| `POST /v1/search` | Vector, BM25, or hybrid search |
| `POST /v1/rerank` | Rerank documents by query relevance |
| `POST /v1/ingest` | Ingest documents into vector DB |
| `POST /v1/ingest/url` | Ingest from URL |
| `POST /v1/rag` | Full RAG pipeline (retrieve + generate) |
| `POST /v1/rag/stream` | Streaming RAG with SSE output |

#### NLP tasks

| Method + Path | Description |
|---|---|
| `POST /v1/ner` | Named Entity Recognition |
| `POST /v1/sentiment` | Sentiment analysis |
| `POST /v1/classify` | Text classification |
| `POST /v1/summarize` | Text summarization |

#### Agents and function calling

| Method + Path | Description |
|---|---|
| `POST /v1/agents/run` | ReAct-style agent loop with tool use |
| `POST /v1/agents/sql` | SQL query agent |
| `POST /v1/code/execute` | Code execution sandbox |

#### Multimodal

| Method + Path | Description |
|---|---|
| `POST /v1/images/generations` | Image generation (diffusion models) |
| `POST /v1/audio/transcriptions` | Audio transcription |

#### Prefill-decode separation

| Method + Path | Description |
|---|---|
| `POST /v1/prefill` | Run prefill only, return serialized KV cache |
| `POST /v1/decode` | Decode from serialized KV cache |

#### Batch processing

| Method + Path | Description |
|---|---|
| `POST /v1/batches` | Create batch processing job |
| `GET /v1/batches` | Get batch job status |

#### Admin and configuration

| Method + Path | Description |
|---|---|
| `POST /v1/admin/reload` | Hot-reload model weights |
| `GET/POST /v1/admin/guardrails` | Content safety filter config |
| `GET/POST /v1/admin/pii` | PII detection/redaction config |
| `GET/POST /v1/admin/templates` | Server-side prompt templates |
| `POST /v1/admin/tenants` | Create/update tenant |
| `GET /v1/admin/tenants/{id}/usage` | Get tenant usage stats |
| `POST /v1/admin/canary` | Create canary deployment |
| `GET /v1/admin/canary` | Get canary status |
| `DELETE /v1/admin/canary` | Cancel canary deployment |
| `POST /v1/admin/triggers` | Create event trigger |
| `GET /v1/admin/triggers` | List triggers |
| `DELETE /v1/admin/triggers/{name}` | Delete trigger |
| `POST /v1/admin/optimize-prompt` | Optimize a prompt |
| `DELETE /v1/admin/gdpr/{user_id}` | GDPR data deletion |

#### Knowledge graph

| Method + Path | Description |
|---|---|
| `POST /v1/knowledge/extract` | Extract entities and relations |
| `GET /v1/knowledge/query` | Query the knowledge graph |

#### Sessions and feedback

| Method + Path | Description |
|---|---|
| `DELETE /v1/sessions/{id}` | Delete conversation session |
| `POST /v1/feedback` | Submit user feedback |

#### WebSocket

| Method + Path | Description |
|---|---|
| `GET /v1/ws/chat` | WebSocket chat endpoint |

#### UI and documentation

| Method + Path | Description |
|---|---|
| `GET /ui` | Web UI |
| `GET /ui/playground` | Interactive playground |
| `GET /ui/dashboard` | Health and metrics dashboard |
| `GET /ui/docs` | Swagger UI |
| `GET /v1/openapi.json` | OpenAPI specification |

---

### Response caching

```go
func NewResponseCache(maxEntries int) *ResponseCache
func (s *Server) SetCache(c *ResponseCache)
```

Exact-match response cache for chat completions. Identical non-streaming requests return cached responses in under 0.1ms.

### Semantic caching

```go
func NewSemanticCache(embedModel EmbeddingModel, threshold float32) *SemanticCache
func (s *Server) SetSemanticCache(sc *SemanticCache)
```

Embedding-based cache lookups. Queries with cosine similarity above `threshold` return cached responses without re-running inference.

---

### PII detection

```go
func NewPIIDetector(mode string) *PIIDetector
func (s *Server) SetPII(d *PIIDetector)
```

Scans requests for personal identifiable information. Modes:

| Mode | Behavior |
|---|---|
| `block` | Reject requests containing PII with 400 status |
| `redact` | Replace PII with `[REDACTED]` before sending to model |
| `log` | Log PII occurrences but allow the request (default) |

Detects: email addresses, phone numbers, SSNs, credit card numbers, IP addresses.

---

### RBAC (Role-Based Access Control)

```go
func NewRBACConfig(pairs map[string]string) *RBACConfig
func RBACMiddleware(rbac *RBACConfig) func(http.Handler) http.Handler
```

Three roles:

| Role | Permissions |
|---|---|
| `admin` | All endpoints |
| `user` | Inference endpoints only (no `/v1/admin/*`) |
| `readonly` | Read-only: `/v1/models`, health, metrics, UI |

```go
rbac := server.NewRBACConfig(map[string]string{
    "admin-key-123": "admin",
    "user-key-456":  "user",
    "view-key-789":  "readonly",
})
handler := server.RBACMiddleware(rbac)(srv)
```

---

### Function calling

Function calling uses GBNF grammar constraints to force the LLM to output valid JSON matching the tool schema. Pass `tools` and `tool_choice` in the `ChatCompletionRequest`:

```go
type Tool struct {
    Type     string       `json:"type"`     // "function"
    Function ToolFunction `json:"function"`
}

type ToolFunction struct {
    Name        string          `json:"name"`
    Description string          `json:"description"`
    Parameters  json.RawMessage `json:"parameters"` // JSON Schema
}
```

The server generates a grammar from the tools, constraining the LLM output to produce valid function calls.

---

### Agent framework

```go
type AgentRunRequest struct {
    Model         string   `json:"model"`
    Query         string   `json:"query"`
    Tools         []string `json:"tools,omitempty"`
    MaxIterations int      `json:"max_iterations,omitempty"`
}
```

ReAct-style agent loop: the agent plans, executes tools (calculator, web search, code execution), observes results, and repeats until it has an answer. Built-in tools include calculator, and custom tools can be registered.

---

### Multi-tenant management

```go
func NewTenantStore() *TenantStore
func (s *Server) SetTenants(ts *TenantStore)
```

Per-API-key resource limits:

```go
type TenantConfig struct {
    ID            string   `json:"id"`
    APIKey        string   `json:"api_key"`
    AllowedModels []string `json:"allowed_models,omitempty"`
    RateLimit     float64  `json:"rate_limit,omitempty"`     // requests/second
    MaxTokens     int      `json:"max_tokens,omitempty"`     // per request
    MonthlyQuota  int64    `json:"monthly_quota,omitempty"`  // tokens/month
}
```

---

### Canary deployments

```go
func NewCanaryDeploy() *CanaryDeploy
func (s *Server) SetCanary(cd *CanaryDeploy)
```

Route a percentage of traffic to a new model with automatic rollback on high error rates:

```go
type CanaryConfig struct {
    BaseModel        string  `json:"base_model"`
    NewModel         string  `json:"new_model"`
    TrafficPct       float64 `json:"traffic_pct"`         // 0.0-1.0
    MaxErrorRate     float64 `json:"max_error_rate"`       // auto-rollback threshold
    AutoPromoteAfter int64   `json:"auto_promote_after"`   // promote after N successes
}
```

---

### Circuit breaker

```go
func NewCircuitBreaker(threshold, cooldownSec int) *CircuitBreaker
```

Tracks failures per model and auto-disables after threshold. States: closed (normal) -> open (disabled) -> half-open (testing). Defaults: 5 failures, 30s cooldown.

---

### A/B testing

```go
func NewABTest(modelA, modelB string, splitPct float64) *ABTest
```

Routes traffic between two models for comparison. `splitPct` controls the percentage of traffic sent to model B.

---

### Audit logging

```go
func NewAuditLogger(path string, hashOnly bool) (*AuditLogger, error)
```

Writes structured JSONL audit entries. When `hashOnly` is true, stores prompt hashes instead of full text for privacy. Each entry includes timestamp, masked API key, model, prompt hash, token count, latency, and status.

---

### Data retention

```go
func NewRetentionPolicy(days int, paths ...string) *RetentionPolicy
```

Auto-deletes old data files after a configurable period. Call `StartPeriodicCleanup` to run cleanup on a schedule.

---

### Content guardrails

Content safety filtering with configurable rules. Set via `POST /v1/admin/guardrails`.

---

### Conversation memory

Multi-turn session memory for chat conversations. Sessions are automatically managed and can be deleted via `DELETE /v1/sessions/{id}`.

---

### Event triggers

```go
func (s *Server) SetTriggers(te *TriggerEngine)
```

Register webhooks that fire on specific events (model loaded, error threshold, etc.) via `POST /v1/admin/triggers`.

---

### Knowledge graph

```go
func (s *Server) SetKnowledgeGraph(kg *KnowledgeGraph)
```

Extract entities and relations from text (`POST /v1/knowledge/extract`) and query the graph (`GET /v1/knowledge/query`).

---

### Registry

```go
func NewRegistry() *Registry
func (r *Registry) Load(name string, m Model) error
func (r *Registry) Unload(name string) error
func (r *Registry) Get(name string) (*ModelRef, error)
func (r *Registry) Names() []string
```

Model registry with reference-counted handles for safe hot reload. `Load` replaces an existing model atomically; `Unload` defers `Close` until all in-flight requests finish. Always call `ref.Release()` after using a model reference.

---

### BatchScheduler

```go
func New(maxBatch int, maxWait time.Duration, process ProcessFn) (*BatchScheduler, error)
type ProcessFn func(inputs []*tensor.Tensor) ([]*tensor.Tensor, error)
func (s *BatchScheduler) Submit(req *tensor.Tensor) (*tensor.Tensor, error)
func (s *BatchScheduler) Stop()
```

Groups individual `Submit` calls into batches for throughput efficiency. `Submit` blocks until the batch is processed.

---

### Metrics

```go
func NewMetrics() *Metrics
func (m *Metrics) Handler() http.Handler
func (m *Metrics) WrapServer(srv http.Handler) http.Handler
func (m *Metrics) InstrumentHandler(model, endpoint string, next http.Handler) http.Handler
func (m *Metrics) ObserveBatch(size int)
func (m *Metrics) ObserveTokensPerSecond(model string, tokens int, elapsed time.Duration)
func (m *Metrics) SetGPUMemory(deviceID int, bytes int64)
```

Prometheus metrics: request count, latency histograms, batch sizes, tokens/sec, GPU memory.

---

### HealthChecker

```go
func NewHealthChecker(reg *Registry, minModels int) *HealthChecker
func (h *HealthChecker) AddReadyCheck(name string, fn func() error)
func (h *HealthChecker) SetLive(live bool)
func (h *HealthChecker) RegisterRoutes(mux *http.ServeMux)
```

Mounts `/healthz` (liveness) and `/readyz` (readiness). Both return JSON `{"status":"ok"}` or `{"status":"fail","details":{...}}`.

---

## Package `llm`

Low-level wrapper over the llama.cpp C API. Use this for direct control over tokenization, KV cache, or per-sequence sampling.

```go
import "github.com/ailakshya/infergo/llm"
```

### Loading a model

```go
m, err := llm.Load(
    "models/llama3-8b-q4.gguf",
    999,   // n_gpu_layers (set 0 for CPU)
    4096,  // context size
    4,     // max parallel sequences
    512,   // batch size
)
defer m.Close()
```

### Tokenization

```go
tokens, err := m.Tokenize("Hello, world!", true, 512)
piece, err  := m.TokenToPiece(tokens[0])
```

### GenerateC -- full C generation loop

Runs the entire generation loop in C++ with one CGo call, eliminating all per-token CGo overhead:

```go
text, genTokens, err := m.GenerateC(promptTokens, maxTokens, temperature, topP, grammar)
```

Parameters:
- `promptTokens` -- pre-tokenized prompt (including BOS)
- `maxTokens` -- max generation length
- `temperature` -- sampling temperature (0 = greedy)
- `topP` -- nucleus sampling (1.0 = disabled)
- `grammar` -- GBNF grammar string (empty = no constraint)

### GenerateBatch -- continuous batching

Runs N requests simultaneously using continuous batching. All sequences share one `llama_decode` call per step:

```go
requests := []llm.BatchRequest{
    {PromptTokens: tokens1, MaxTokens: 256},
    {PromptTokens: tokens2, MaxTokens: 256},
}
results, err := m.GenerateBatch(requests, temperature, topP, grammar)
for _, r := range results {
    fmt.Println(r.Text, r.GenTokens)
}
```

### LoRA adapters

```go
adapter, err := m.LoadLoRA("path/to/adapter.bin")
defer adapter.Free()

err = m.ApplyLoRA([]llm.LoRAHandle{adapter}, []float32{1.0})
// Generate with adapter applied...
err = m.ApplyLoRA(nil, nil) // Clear all adapters
```

### BM25 full-text search

```go
idx, err := llm.NewBM25Index(1.2, 0.75)
defer idx.Close()

idx.Insert(1, "The quick brown fox")
idx.Insert(2, "The lazy dog")

results, err := idx.Search("quick fox", 10)
for _, r := range results {
    fmt.Printf("ID=%d Score=%.3f\n", r.ID, r.Score)
}
```

### Hybrid search (BM25 + vector fusion)

```go
vecResults := vectorIndex.Search(queryVec, 10)
bm25Results := bm25Index.Search(queryText, 10)

// alpha=0.5 gives equal weight to vector and BM25
hybridResults, err := llm.HybridSearch(vecResults, bm25Results, 0.5, 10)
for _, r := range hybridResults {
    fmt.Printf("ID=%d Score=%.3f\n", r.ID, r.Score)
}
```

### Manual generation loop

For fine-grained control over tokenization, decoding, and sampling:

```go
seq, err := m.NewSequence(tokens)
defer seq.Close()

for !seq.IsDone() && seq.Position() < maxTokens {
    if err := m.BatchDecode([]*llm.Sequence{seq}); err != nil {
        break
    }
    tok, err := seq.SampleToken(0.7, 0.9) // temperature, top-p
    if m.IsEOG(tok) {
        break
    }
    piece, _ := m.TokenToPiece(tok)
    fmt.Print(piece)
    seq.AppendToken(tok)
}
```

### Multi-sequence batching

Pass multiple sequences to `BatchDecode` to amortize the cost of the forward pass:

```go
seqs := make([]*llm.Sequence, batchSize)
for i := range seqs {
    seqs[i], _ = m.NewSequence(promptTokens[i])
}
for anyActive(seqs) {
    m.BatchDecode(seqs)
    for _, s := range seqs {
        tok, _ := s.SampleToken(0.7, 0.9)
        // ...
    }
}
```

---

## Package `client`

Typed Go client for the infergo HTTP API.

```go
import "github.com/ailakshya/infergo/client"
```

### Creating a client

```go
c := client.New("http://localhost:9090")

// With options
c := client.New("http://localhost:9090",
    client.WithAPIKey("my-key"),
    client.WithTimeout(60 * time.Second),
)
```

### Chat

```go
resp, err := c.Chat(ctx, client.ChatRequest{
    Model:     "llama3-8b-q4",
    Messages:  []client.Message{{Role: "user", Content: "Hello!"}},
    MaxTokens: 256,
    Temp:      0.7,
})
fmt.Println(resp.Content)
```

### Chat streaming

```go
tokens, errc := c.ChatStream(ctx, client.ChatRequest{
    Model:    "llama3-8b-q4",
    Messages: []client.Message{{Role: "user", Content: "Count to 5"}},
})
for tok := range tokens {
    fmt.Print(tok)
}
if err := <-errc; err != nil {
    log.Fatal(err)
}
```

### Embed

```go
vec, err := c.Embed(ctx, client.EmbedRequest{
    Model: "embed",
    Input: "hello world",
})
```

### Detect

```go
resp, err := c.Detect(ctx, client.DetectRequest{
    Model:      "yolo11n",
    ImageB64:   base64EncodedImage,
    ConfThresh: 0.25,
    IouThresh:  0.45,
    MaxDet:     300,
    Classes:    []int{0, 2}, // filter to person and car
})
for _, obj := range resp.Objects {
    fmt.Printf("Class %d: %.2f at (%.0f,%.0f)-(%.0f,%.0f)\n",
        obj.ClassID, obj.Confidence, obj.X1, obj.Y1, obj.X2, obj.Y2)
}
```

### Search

Use the HTTP client directly via `doJSON` for search, ingest, and NLP endpoints.

### List models

```go
models, err := c.ListModels(ctx)
for _, m := range models {
    fmt.Println(m.ID)
}
```

---

## Package `tracker`

ByteTrack multi-object tracking implementation.

```go
import "github.com/ailakshya/infergo/tracker"
```

### Configuration

```go
cfg := tracker.DefaultConfig()
// Defaults:
//   TrackHighThresh: 0.20
//   TrackLowThresh:  0.08
//   NewTrackThresh:  0.25
//   TrackBuffer:     60  (max frames to keep lost tracks)
//   MatchThresh:     0.80
//   FuseScore:       true
```

### Usage

```go
bt := tracker.NewByteTracker(cfg)

// For each video frame:
detections := []tracker.Detection{
    {X1: 10, Y1: 20, X2: 100, Y2: 200, ClassID: 0, Confidence: 0.95},
    // ...
}
tracks := bt.Update(detections)

for _, t := range tracks {
    fmt.Printf("Track %d: class=%d pos=(%.0f,%.0f,%.0f,%.0f)\n",
        t.TrackID, t.ClassID, t.X1, t.Y1, t.X2, t.Y2)
}
```

The tracker uses Kalman filtering for motion prediction and a 3-stage association strategy that leverages both high and low confidence detections, following the ByteTrack paper (Zhang et al., ECCV 2022).

---

## Package `tensor`

```go
import "github.com/ailakshya/infergo/tensor"
```

### Creating tensors

```go
// CPU
t, err := tensor.NewTensorCPU([]int{1, 3, 224, 224}, tensor.Float32)
defer t.Free()

// CUDA device 0
t, err := tensor.NewTensorCUDA([]int{1, 3, 224, 224}, tensor.Float32, 0)
```

### Data access

```go
t.Shape()     // []int{1, 3, 224, 224}
t.NElements() // 150528
t.NBytes()    // 602112

ptr := t.DataPtr() // unsafe.Pointer to raw data
t.CopyFrom(srcPtr, nbytes)
```

### Device transfer

```go
t.ToDevice(0) // host -> CUDA:0
t.ToHost()    // CUDA -> host
```

### DType constants

| Constant | C type |
|---|---|
| `tensor.Float32` | float |
| `tensor.Float16` | __fp16 |
| `tensor.BFloat16` | bfloat16 |
| `tensor.Int32` | int32_t |
| `tensor.Int64` | int64_t |
| `tensor.UInt8` | uint8_t |
| `tensor.Bool` | bool |
