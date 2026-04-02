# Infergo — Full Roadmap to a Complete Go Inference Library

> **Status legend:** `[ ]` pending · `[~]` in progress · `[x]` done  
> **Effort:** S = 1–2 days · M = 3–5 days · L = 1–2 weeks · XL = 2–4 weeks

---

## PHASE A — Performance Optimizations

### OPT-1 — CPU: OpenBLAS for BLAS-accelerated prefill `[ ]` S

**Problem:** `GGML_BLAS=OFF` in current build. Long-prompt prefill (512 tokens) is
slow because GGML uses scalar kernels for the prompt-processing GEMM. Python's
llama-cpp-python links OpenBLAS and runs prefill ~10% faster.

**What changes:**
- `apt install libopenblas-dev` on gpu_dev
- CMake reconfigure: `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS` inside `vendor/llama.cpp`
- Rebuild `infer_api.so` + `infergo` binary

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-1-T1 | Build links OpenBLAS | `ldd build/cpp/api/libinfer_api.so` contains `libopenblas` |
| OPT-1-T2 | Prefill speedup | Single 512-token prompt, 64 output: TTFT ≤ 1500 ms on CPU (was ~25 s) |
| OPT-1-T3 | Long CPU tok/s improves | `bench_full.py --device cpu` long: `tok/s ≥ 11` (matches Python) |
| OPT-1-T4 | Short prompts unaffected | CPU short tok/s within ±5% of 10 tok/s baseline |
| OPT-1-T5 | CUDA results unchanged | CUDA bench passes; BLAS only on CPU path |

---

### OPT-2 — GPU/CPU: Continuous batching scheduler `[ ]` L

**Problem:** `llmAdapter.Generate()` holds a mutex for the full request. P50 under
concurrency=4 is 3× single-client latency. GPU utilization ~25%.

**Architecture (Go layer only — no C++ changes):**

```
HTTP handlers ──► request channel ──► scheduler goroutine ──► BatchDecode([seq1,seq2,...])
                                              │
                                   per-request result channel ──► handler response
```

**What changes:**
- `go/cmd/infergo/scheduler.go` — new file, `schedulerModel` type
- `Submit(ctx, prompt, maxTokens, temp) <-chan TokenEvent` — enqueue, returns token stream
- Scheduler loop: drain queue → assemble batch → `BatchDecode` → sample each seq → route tokens
- HTTP SSE handler consumes `<-chan TokenEvent`, writes `data:` lines
- Non-streaming handler drains channel, assembles full response

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-2-T1 | Single request through scheduler | Returns correct text, no deadlock |
| OPT-2-T2 | 4 concurrent requests complete | All goroutines get non-empty responses, no panic |
| OPT-2-T3 | Race detector clean | `go test -race ./go/...` exits 0 |
| OPT-2-T4 | P50 latency drops | `bench_full.py --device cuda --concurrency 4`: P50 ≤ 600 ms (was 1423 ms) |
| OPT-2-T5 | Throughput does not regress | CUDA short `tok/s ≥ 140` |
| OPT-2-T6 | SSE streaming works | `curl -N` with `"stream":true` emits `data:` lines per token |
| OPT-2-T7 | Graceful shutdown | SIGTERM with 4 in-flight requests: all complete before exit |
| OPT-2-T8 | Client disconnect frees KV slot | Disconnect mid-stream: sequence removed, slot reused on next request |

---

## PHASE B — Core Inference Expansion

### OPT-3 — ONNX Runtime inference engine `[ ]` L

**Scope:** Real ONNX session execution. Prerequisite for OPT-4, OPT-5, OPT-6.
Current `onnxAdapter` only registers the model path — does not run inference.

**What changes:**
- `cpp/onnx/onnx_engine.hpp/.cpp` — wrap `Ort::Session`, `Ort::RunOptions`, `Ort::Value`
- `cpp/api/api.cpp` — `infer_onnx_run(handle, input_data, input_shape, ndim, out)` C function
- `go/onnx/session.go` — `Run(inputs []Tensor) ([]Tensor, error)` via CGo
- `server/server.go` — wire `/v1/embeddings` and `/v1/detect` to ONNX sessions

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-3-T1 | C++ ONNX session creates | `ctest -R onnx_engine_test` passes |
| OPT-3-T2 | Embedding model runs | `all-MiniLM-L6-v2` on "hello world" → output shape `[1, 384]` |
| OPT-3-T3 | Detection model runs | `yolov8n` on 640×640 zeros → output shape `[1, 84, 8400]` |
| OPT-3-T4 | No memory leak | ASan + 1000 ONNX runs: zero leaks, zero errors |
| OPT-3-T5 | Go wrapper works | `go test ./go/onnx/...` exits 0 |
| OPT-3-T6 | CPU + CUDA providers | Model loads with `--provider cpu` and `--provider cuda` |
| OPT-3-T7 | Concurrent ONNX sessions | 4 goroutines call `Run()` simultaneously: all succeed, no crash |

---

### OPT-4 — Embeddings API + benchmark `[ ]` M

**Scope:** `/v1/embeddings` endpoint (OpenAI-compatible) + benchmark vs
`sentence-transformers`.

**Models:**

| Model | Params | Size | Specialty |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 22M | 90 MB | Lightweight baseline |
| `nomic-embed-text-v1.5` | 137M | 274 MB | General English |
| `bge-m3` | 570M | 570 MB | Multilingual, long context |

**Benchmark script:** `benchmarks/vs_python/bench_embedding.py`

**Scenarios:** batch=1/8/32/64, short sentences (15 tok) + long paragraphs (256 tok), CUDA + CPU

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-4-T1 | Endpoint returns correct shape | `POST /v1/embeddings` → `data[0].embedding` length matches model dim |
| OPT-4-T2 | Vector correctness | Cosine sim vs sentence-transformers on same text ≥ 0.999 |
| OPT-4-T3 | Batch throughput ≥ Python | CUDA batch=32: sentences/sec ≥ sentence-transformers |
| OPT-4-T4 | CUDA vs CPU ≥ 5× | CUDA batch=32 at least 5× faster than CPU batch=32 |
| OPT-4-T5 | Concurrent embedding requests | 8 goroutines simultaneously: all return correct vectors |
| OPT-4-T6 | Results documented | `benchmarks/vs_python/results_embedding.md` populated |

---

### OPT-5 — Detection API + benchmark `[ ]` M

**Scope:** `/v1/detect` endpoint + benchmark vs `ultralytics` / ONNX Runtime.

**Models:**

| Model | Params | mAP50 | Latency target |
|---|---|---|---|
| `yolov8n.onnx` | 3.2M | 37.3 | ≤ 5 ms CUDA |
| `yolov8s.onnx` | 11M | 44.9 | ≤ 10 ms CUDA |
| `yolov8m.onnx` | 25M | 50.2 | ≤ 20 ms CUDA |

**Request:** `POST /v1/detect` with `{ "image": "<base64>", "model": "yolov8n" }`

**Response:**
```json
{
  "detections": [
    { "class": "person", "confidence": 0.92, "box": { "x1": 10, "y1": 20, "x2": 300, "y2": 500 } }
  ],
  "inference_ms": 3.2
}
```

**Benchmark script:** `benchmarks/vs_python/bench_detection.py`

**Scenarios:** batch=1/8/32/64 images, 640×640, COCO val2017 sample (100 images), CUDA + CPU

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-5-T1 | Endpoint returns bounding boxes | `POST /v1/detect` with COCO image returns ≥ 1 detection |
| OPT-5-T2 | mAP correctness | mAP50 on COCO val sample within 0.5% of ultralytics reference |
| OPT-5-T3 | yolov8n single-image P50 | CUDA P50 ≤ 5 ms |
| OPT-5-T4 | Batch throughput ≥ Python ONNX Runtime | CUDA batch=32 images/sec ≥ Python |
| OPT-5-T5 | Preprocessing separate from inference | Response includes separate `preprocess_ms` and `inference_ms` fields |
| OPT-5-T6 | Results documented | `benchmarks/vs_python/results_detection.md` populated |

---

### OPT-6 — Image preprocessing pipeline `[ ]` S

**Scope:** Resize, letterbox, normalize for YOLO input. Required for correct
detection results.

**What changes:**
- `cpp/preprocess/image.hpp/.cpp` — letterbox resize to 640×640, normalize [0,1], CHW layout
- `go/preprocess/image.go` — Go wrapper + CGo binding
- Used internally by `/v1/detect` handler

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-6-T1 | Letterbox correctness | 1280×720 image → 640×640 tensor with correct padding pixels |
| OPT-6-T2 | Normalize range | All tensor values ∈ [0.0, 1.0] after normalization |
| OPT-6-T3 | Output layout | Output tensor shape `[1, 3, 640, 640]` (NCHW) |
| OPT-6-T4 | Round-trip accuracy | Preprocess with infergo, compare tensor vs ultralytics preprocess: max diff ≤ 1e-4 |

---

### OPT-7 — BERT/RoBERTa tokenizer for embedding models `[ ]` M

**Scope:** Embedding models need WordPiece / SentencePiece tokenization, not
llama.cpp's BPE. Required for correct embedding output.

**What changes:**
- `go/tokenizer/bert.go` — `BERTTokenizer` wrapping HuggingFace tokenizers via CGo or pure Go
- Handles `[CLS]` / `[SEP]` tokens, padding to max length, attention mask output
- Used by `/v1/embeddings` handler

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-7-T1 | Tokenize "hello world" | Token IDs match HuggingFace tokenizer output exactly |
| OPT-7-T2 | CLS + SEP tokens | Output starts with `[CLS]` ID (101), ends with `[SEP]` ID (102) |
| OPT-7-T3 | Padding to max_length | Batch of 2 sentences padded to same length; attention_mask correct |
| OPT-7-T4 | Truncation at 512 tokens | Input of 600 tokens truncated to 512 with no panic |
| OPT-7-T5 | Go tests pass | `go test ./go/tokenizer/...` exits 0 |

---

## PHASE C — Production Serving

### OPT-8 — Multi-model serving in one process `[ ]` M

**Problem:** Current `infergo serve` loads exactly one model. Production workloads
need multiple models (LLM + embedding + detection) in one server.

**What changes:**
- `--model` flag becomes repeatable: `--model llm:llama3.gguf --model embed:nomic.onnx`
- Registry already supports multiple models; serve.go needs multi-flag parsing
- `/v1/models` lists all loaded models with their type

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-8-T1 | Two models load | `--model llm:llama3.gguf --model embed:nomic.onnx` → `/health/ready` ok |
| OPT-8-T2 | Requests route correctly | Chat req → LLM; embedding req → ONNX; no cross-routing |
| OPT-8-T3 | Models list | `GET /v1/models` returns both models with correct types |
| OPT-8-T4 | Memory isolation | OOM loading one model does not corrupt the other |

---

### OPT-9 — Model hot-reload without restart `[ ]` M

**What changes:**
- `POST /v1/admin/reload` with `{ "model": "llama3", "path": "..." }` — loads new weights, swaps atomically
- Registry uses `sync.RWMutex`; readers continue serving during load
- Old model freed after last in-flight request completes

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-9-T1 | Reload while serving | 10 in-flight requests + reload: all complete, no 500s |
| OPT-9-T2 | Old model freed | RSS drops after reload + GC; no memory leak |
| OPT-9-T3 | Bad path rejected | Reload with nonexistent path returns 400, old model still works |
| OPT-9-T4 | Race detector clean | `go test -race` during concurrent reload: exits 0 |

---

### OPT-10 — Request queue + priority scheduling `[ ]` M

**What changes:**
- `--max-queue` flag (default 100): requests beyond this get 503
- Optional `X-Priority` header: `high` / `normal` / `low`
- Prometheus metric: `infergo_queue_depth` gauge

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-10-T1 | Queue cap enforced | 101st concurrent request gets HTTP 503 |
| OPT-10-T2 | High priority served first | High-priority request submitted after 10 normal ones completes before them |
| OPT-10-T3 | Queue depth metric | `GET /metrics` shows `infergo_queue_depth` during load |
| OPT-10-T4 | Queue drains on shutdown | SIGTERM: all queued requests complete or get 503, no hang |

---

### OPT-11 — API key authentication `[ ]` S

**What changes:**
- `--api-key <key>` flag or `INFERGO_API_KEY` env var
- Middleware checks `Authorization: Bearer <key>` on all `/v1/` routes
- Returns 401 if missing or wrong; `/health/` and `/metrics` exempt

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-11-T1 | Valid key passes | Request with correct Bearer token gets 200 |
| OPT-11-T2 | Missing key blocked | Request without header gets 401 |
| OPT-11-T3 | Wrong key blocked | Request with wrong token gets 401 |
| OPT-11-T4 | Health exempt | `GET /health/live` with no key gets 200 |
| OPT-11-T5 | No key configured = open | Server without `--api-key` flag accepts all requests |

---

### OPT-12 — Rate limiting per API key `[ ]` S

**What changes:**
- `--rate-limit N` flag: max N requests/second per key (token bucket)
- Returns 429 with `Retry-After` header when exceeded
- Prometheus metric: `infergo_rate_limited_total` counter

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-12-T1 | Limit enforced | 20 req/s with limit=10: ~50% get 429 |
| OPT-12-T2 | Retry-After header present | 429 response includes `Retry-After: 1` |
| OPT-12-T3 | Per-key isolation | Key A at limit does not block key B |
| OPT-12-T4 | Metric increments | `infergo_rate_limited_total` counter increases on 429 |

---

### OPT-13 — gRPC API `[ ]` L

**Scope:** Low-latency service-to-service alternative to HTTP+JSON.

**Proto:**
```protobuf
service Infergo {
  rpc ChatCompletion(ChatRequest) returns (stream ChatChunk);
  rpc Embed(EmbedRequest) returns (EmbedResponse);
  rpc Detect(DetectRequest) returns (DetectResponse);
}
```

**What changes:**
- `proto/infergo.proto` — service definition
- `go/grpc/server.go` — gRPC server wrapping existing Registry
- `--grpc-port` flag (default 9091)

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-13-T1 | gRPC chat completion | `grpcurl` streaming call returns tokens |
| OPT-13-T2 | gRPC embedding | Returns float32 vector with correct dimension |
| OPT-13-T3 | gRPC detection | Returns bounding boxes for test image |
| OPT-13-T4 | HTTP + gRPC co-exist | Both ports serve simultaneously without conflict |
| OPT-13-T5 | Latency < HTTP | gRPC P50 ≤ HTTP P50 - 1ms for same model + prompt |

---

### OPT-14 — WebSocket streaming `[ ]` S

**Scope:** Alternative to SSE for clients that prefer WebSocket.

**Protocol:** Connect to `ws://host/v1/ws/chat`, send JSON request, receive token
frames, connection closes on `[DONE]`.

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-14-T1 | WS handshake succeeds | `wscat -c ws://localhost:9090/v1/ws/chat` connects |
| OPT-14-T2 | Tokens stream correctly | Full response assembled from frames matches non-streaming response |
| OPT-14-T3 | Client disconnect handled | Server removes sequence, no goroutine leak |

---

## PHASE D — Ecosystem

### OPT-15 — Go SDK / client library `[ ]` M

**Scope:** `go get github.com/ailakshya/infergo/client` — typed Go client, not
raw HTTP. Mirrors OpenAI Go SDK ergonomics.

```go
c := client.New("http://localhost:9090", client.WithAPIKey("..."))
stream, _ := c.Chat(ctx, &client.ChatRequest{
    Model: "llama3-8b-q4",
    Messages: []client.Message{{Role: "user", Content: "Hello"}},
})
for tok := range stream.Tokens() { fmt.Print(tok) }
```

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-15-T1 | Chat completion (blocking) | `client.ChatBlocking()` returns full text |
| OPT-15-T2 | Chat completion (streaming) | Token channel receives ≥ 5 tokens |
| OPT-15-T3 | Embeddings | `client.Embed()` returns `[]float32` with correct length |
| OPT-15-T4 | Detection | `client.Detect()` returns `[]Detection` with boxes |
| OPT-15-T5 | Context cancellation | Cancel ctx mid-stream: channel closes, no goroutine leak |
| OPT-15-T6 | pkg.go.dev renders | `go doc github.com/ailakshya/infergo/client` shows all exported types |

---

### OPT-16 — HuggingFace model hub download `[ ]` L

**Scope:** `infergo pull <repo/model>` CLI command downloads GGUF/ONNX from HF Hub.

```
infergo pull bartowski/Meta-Llama-3-8B-Instruct-GGUF --quant Q4_K_M
infergo pull sentence-transformers/all-MiniLM-L6-v2 --format onnx
```

**What changes:**
- `go/cmd/infergo/pull.go` — new subcommand
- HuggingFace API calls to resolve file URLs
- Resume-capable download with progress bar
- SHA256 verification after download

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-16-T1 | Pull GGUF model | `infergo pull` downloads `.gguf` to `~/.infergo/models/` |
| OPT-16-T2 | Pull ONNX model | Downloads `.onnx` to correct path |
| OPT-16-T3 | Resume download | Kill mid-download, re-run: completes from offset, no corruption |
| OPT-16-T4 | SHA256 verified | Corrupt file detected and re-downloaded |
| OPT-16-T5 | Private repo with token | `--hf-token` flag downloads private model |

---

### OPT-17 — Multi-model LLM benchmark `[ ]` M

**Scope:** Run full LLM benchmark across 3 models — proves infergo advantage is
not cherry-picked on one checkpoint.

**Models:**

| Model | Params | Size | Quantization |
|---|---|---|---|
| `llama3-8b-q4.gguf` | 8B | 4.6 GB | Q4_K_M |
| `phi-3.5-mini-instruct.Q4_K_M.gguf` | 3.8B | 2.2 GB | Q4_K_M |
| `gemma-2-9b-it.Q4_K_M.gguf` | 9B | 5.5 GB | Q4_K_M |

**Changes to bench_full.py:** `--models name:path,name:path` flag; outer loop per model;
aggregate summary table in markdown.

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-17-T1 | Phi-3.5 mini loads + responds | `/health/ready` ok; sample prompt returns text |
| OPT-17-T2 | Gemma 2 9B loads + responds | `/health/ready` ok; sample prompt returns text |
| OPT-17-T3 | All 3 models benchmarked | `results_multimodel.md` has a row for each model |
| OPT-17-T4 | infergo ≥ Python on all 3 | tok/s advantage ≥ +5% CUDA for each model |
| OPT-17-T5 | No KV leaks across models | 100 requests per model sequential: no positional errors |

---

### OPT-18 — OpenTelemetry distributed tracing `[ ]` M

**What changes:**
- `go.opentelemetry.io/otel` SDK added as dependency
- Each HTTP request gets a trace span: `parse → queue → decode → respond`
- `--otlp-endpoint` flag exports to Jaeger / Tempo

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-18-T1 | Spans emitted | Jaeger UI shows spans for chat completions |
| OPT-18-T2 | Trace propagation | W3C `traceparent` header forwarded from client to server span |
| OPT-18-T3 | Decode duration tracked | Span has `decode_ms` attribute matching actual time |
| OPT-18-T4 | No perf regression | P50 latency with tracing ON within 2% of tracing OFF |

---

### OPT-19 — TensorRT backend `[ ]` XL

**Scope:** `--provider tensorrt` compiles ONNX model to TRT engine on first load,
then runs on TensorRT. 2–5× faster than CUDA ONNX Runtime for fixed batch sizes.

**What changes:**
- `cpp/onnx/onnx_engine.cpp` — add TensorRT execution provider to ONNX Runtime session
- CMake: `find_package(TensorRT)`, wrap in `if(INFER_TENSORRT)`
- Engine cache: compiled `.trt` file saved next to `.onnx`, reused on restart

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-19-T1 | TRT engine compiles | First load of yolov8n with `--provider tensorrt` succeeds in ≤ 60 s |
| OPT-19-T2 | Engine cache used | Second load ≤ 2 s (cache hit) |
| OPT-19-T3 | Correctness preserved | mAP50 within 0.5% of CUDA ONNX run |
| OPT-19-T4 | TRT faster than CUDA | yolov8n batch=32 images/sec ≥ 1.5× CUDA ONNX Runtime |

---

### OPT-20 — CoreML backend (Apple Silicon) `[ ]` XL

**Scope:** `--provider coreml` runs ONNX models via Apple's CoreML on macOS.
Enables Mac deployment without CUDA.

**What changes:**
- ONNX Runtime CoreML execution provider (already in ONNX Runtime macOS builds)
- CMake: `if(APPLE AND INFER_COREML)`
- `llama.cpp` Metal backend already works; this is for ONNX models only

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-20-T1 | CoreML loads on Mac | `--provider coreml` with all-MiniLM reaches `/health/ready` |
| OPT-20-T2 | Faster than CPU | CoreML embedding batch=8: sentences/sec ≥ 2× CPU |
| OPT-20-T3 | Correctness | Cosine sim vs reference ≥ 0.999 |

---

### OPT-21 — KEDA / HPA autoscaling metrics `[ ]` S

**Scope:** Expose Prometheus metrics that KEDA uses to scale infergo pods.

**New metrics:**
- `infergo_queue_depth` — pending requests (already in OPT-10)
- `infergo_active_sequences` — sequences currently decoding
- `infergo_gpu_utilization_percent` — from `nvmlDeviceGetUtilizationRates`

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-21-T1 | All three metrics present | `GET /metrics` contains all three metric names |
| OPT-21-T2 | GPU util reflects load | Metric > 80 during active CUDA generation |
| OPT-21-T3 | KEDA ScaledObject | Sample KEDA manifest in `docs/deployment.md` validated with `kubectl dry-run` |

---

## Execution order

```
OPT-1   CPU BLAS              ← quick win, independent
OPT-2   continuous batching   ← biggest GPU impact, independent
OPT-3   ONNX inference        ← prerequisite for OPT-4, OPT-5, OPT-6, OPT-7
OPT-6   image preprocessing   ← pairs with OPT-3
OPT-7   BERT tokenizer        ← pairs with OPT-3
OPT-4   embeddings API+bench  ← requires OPT-3 + OPT-7
OPT-5   detection API+bench   ← requires OPT-3 + OPT-6
OPT-8   multi-model serving   ← requires OPT-3
OPT-9   hot-reload            ← requires OPT-8
OPT-10  request queue         ← requires OPT-2
OPT-11  API key auth          ← independent
OPT-12  rate limiting         ← requires OPT-11
OPT-13  gRPC API              ← requires OPT-3 + OPT-4 + OPT-5
OPT-14  WebSocket streaming   ← requires OPT-2
OPT-15  Go SDK                ← requires OPT-4 + OPT-5 + OPT-13
OPT-16  HF model hub download ← independent
OPT-17  multi-model LLM bench ← requires OPT-1 + OPT-2
OPT-18  OpenTelemetry tracing ← independent
OPT-19  TensorRT backend      ← requires OPT-3
OPT-20  CoreML backend        ← requires OPT-3
OPT-21  KEDA metrics          ← requires OPT-10
```

---

## Completion targets

| After completing | infergo is comparable to |
|---|---|
| OPT-1 + OPT-2 | vLLM (LLM serving performance) |
| OPT-3 + OPT-4 + OPT-5 | ONNX Runtime server + sentence-transformers |
| OPT-3 through OPT-12 | Triton Inference Server (features) |
| OPT-3 through OPT-16 | Full Python ML serving stack (transformers + ultralytics + vLLM) in Go |
| OPT-3 through OPT-21 | Production-grade, cloud-native Go inference platform |

---

## Expected metric outcomes

| Task | Metric | Before | Target |
|---|---|---|---|
| OPT-1 | CPU long tok/s | 10 | ≥ 12 |
| OPT-1 | CPU long P50 latency | 25936 ms | ≤ 22000 ms |
| OPT-2 | CUDA P50 @ concurrency=4 | 1423 ms | ≤ 600 ms |
| OPT-2 | CUDA GPU utilization | ~25% | ≥ 85% |
| OPT-2 | CUDA tok/s | 142 | ≥ 200 |
| OPT-4 | Embedding CUDA sentences/sec | — | ≥ sentence-transformers |
| OPT-5 | Detection CUDA batch=32 images/sec | — | ≥ ultralytics ONNX Runtime |
| OPT-19 | TRT detection vs CUDA ONNX | — | ≥ 1.5× faster |
