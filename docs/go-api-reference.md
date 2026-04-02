# Go API Reference

Import path: `github.com/ailakshya/infergo`

---

## Package `server`

The `server` package is the OpenAI-compatible HTTP serving layer. Use it to embed infergo inside your own Go application.

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

#### `EmbeddingModel`

```go
type EmbeddingModel interface {
    Model
    Embed(ctx context.Context, input string) ([]float32, error)
}
```

Handles `/v1/embeddings`. The `input` is an image encoded as a base64 string (for vision models) or raw text.

#### `DetectionModel`

```go
type DetectionModel interface {
    Model
    Detect(ctx context.Context, imageBytes []byte, confThresh, iouThresh float32) ([]DetectedObject, error)
}
```

Handles `/v1/detect`.

```go
type DetectedObject struct {
    X1, Y1, X2, Y2 float32
    ClassID         int
    Confidence      float32
}
```

---

### Registry

```go
func NewRegistry() *Registry
```

Creates an empty model registry.

```go
func (r *Registry) Load(name string, m Model) error
```

Registers a model under `name`. If a model with that name is already loaded, it is hot-reloaded: new requests go to the new model immediately; the old model is closed once all in-flight requests finish.

```go
func (r *Registry) Unload(name string) error
```

Removes a model. Returns an error if the name is not registered. The model's `Close` is deferred until all in-flight requests finish.

```go
func (r *Registry) Get(name string) (*ModelRef, error)
```

Returns a reference-counted handle. The caller **must** call `ref.Release()` when done to avoid leaking the model during hot reload.

```go
func (r *Registry) Names() []string
```

Returns all currently registered model names.

#### `ModelRef`

```go
type ModelRef struct {
    Model Model
}

func (r *ModelRef) Release()
```

Call `Release` after you are done using the model. Do not use `ref.Model` after calling `Release`.

---

### Server

```go
func NewServer(reg *Registry) *Server
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request)
```

`Server` implements `http.Handler`. Wire it to a `ServeMux` under any prefix — the server uses Go 1.22 pattern routing internally.

**Endpoints handled:**

| Method + Path | Handler |
|---|---|
| `POST /v1/chat/completions` | LLM chat, streaming or batch |
| `POST /v1/completions` | LLM text completion |
| `POST /v1/embeddings` | Embedding vector |
| `POST /v1/detect` | Object detection |
| `GET /v1/models` | List loaded models |

---

### BatchScheduler

Groups individual `Submit` calls into batches for throughput efficiency.

```go
func New(maxBatch int, maxWait time.Duration, process ProcessFn) (*BatchScheduler, error)
```

- `maxBatch` — flush when this many requests are queued
- `maxWait` — flush after this duration even if batch isn't full
- `process` — your batch inference function

```go
type ProcessFn func(inputs []*tensor.Tensor) ([]*tensor.Tensor, error)

func (s *BatchScheduler) Submit(req *tensor.Tensor) (*tensor.Tensor, error)
func (s *BatchScheduler) Stop()
```

`Submit` blocks until the batch is processed. Call `Stop` on shutdown to drain in-flight requests.

---

### Metrics

```go
func NewMetrics() *Metrics
func (m *Metrics) Handler() http.Handler          // mount at /metrics
func (m *Metrics) WrapServer(srv http.Handler) http.Handler
func (m *Metrics) InstrumentHandler(model, endpoint string, next http.Handler) http.Handler
func (m *Metrics) ObserveBatch(size int)
func (m *Metrics) ObserveTokensPerSecond(model string, tokens int, elapsed time.Duration)
func (m *Metrics) SetGPUMemory(deviceID int, bytes int64)
```

`WrapServer` wraps any handler to auto-instrument all requests by model name and endpoint. `InstrumentHandler` is lower-level for manual control.

---

### HealthChecker

```go
func NewHealthChecker(reg *Registry, minModels int) *HealthChecker
```

- `minModels` — readiness requires at least this many models loaded (set to 0 to skip)

```go
func (h *HealthChecker) AddReadyCheck(name string, fn func() error)
func (h *HealthChecker) SetLive(live bool)
func (h *HealthChecker) RegisterRoutes(mux *http.ServeMux)
```

`RegisterRoutes` mounts `/healthz` (liveness) and `/readyz` (readiness). Both return JSON `{"status":"ok"}` on success or `{"status":"fail","details":{...}}` on failure.

---

## Package `llm`

Low-level wrapper over the llama.cpp C API. Use this if you need direct control over tokenization, KV cache, or per-sequence sampling.

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

### Generation loop

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
t.ToDevice(0) // host → CUDA:0
t.ToHost()    // CUDA → host
```

### Crossing CGo boundaries

`UnsafePtr` / `WrapUnsafePtr` let you pass tensors across package boundaries without copying:

```go
raw := t.UnsafePtr()
// pass raw to another package
t2 := tensor.WrapUnsafePtr(raw)
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
