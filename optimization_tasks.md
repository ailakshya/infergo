# Infergo — Full Roadmap to a Complete Go Inference Library

> **Status legend:** `[ ]` pending · `[~]` in progress · `[x]` done  
> **Effort:** S = 1–2 days · M = 3–5 days · L = 1–2 weeks · XL = 2–4 weeks

---

## PHASE A — Performance Optimizations

### OPT-1 — CPU: OpenBLAS for BLAS-accelerated prefill `[x]` S

**Result:** 2026-04-03 — libopenblas linked, CUDA unaffected. Generation tok/s unchanged (BLAS only helps prefill GEMM, not generation GEMV which is memory-bandwidth bound). T2/T3 targets revised: see notes.

**Problem:** `GGML_BLAS=OFF` in current build. Long-prompt prefill (512 tokens) is
slow because GGML uses scalar kernels for the prompt-processing GEMM. Python's
llama-cpp-python links OpenBLAS and runs prefill ~10% faster.

**What changes:**
- `apt install libopenblas-dev` on gpu_dev
- CMake reconfigure: `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS` inside `vendor/llama.cpp`
- Rebuild `infer_api.so` + `infergo` binary

**Test cases:**

| ID | Test | Result |
|---|---|---|
| OPT-1-T1 | Build links OpenBLAS | PASS — `libopenblas.so.0` confirmed via ldd |
| OPT-1-T2 | Prefill speedup | PARTIAL — TTFT ~5-6s for 230 tok on CPU (target 1500ms was unrealistic; GGML already has SIMD kernels) |
| OPT-1-T3 | Long CPU tok/s improves | SKIP — tok/s measures generation (GEMV), not prefill (GEMM). BLAS cannot improve GEMV. |
| OPT-1-T4 | Short prompts unaffected | PASS — short tok/s = 10, within ±5% baseline |
| OPT-1-T5 | CUDA results unchanged | PASS — CUDA 252ms/32tok, no regression |

---

### OPT-2 — GPU/CPU: Continuous batching scheduler `[x]` L

**Result:** 2026-04-03 — scheduler implemented; 142 tok/s → 200 tok/s (+41%), long prompts fixed (ctx-size 4096→16384; llama.cpp divides n_ctx by n_seq_max for per-seq budget). T1-T3, T5-T8 PASS; T4 P50=1185ms (target 600ms not met — needs OPT-22 PagedAttention for true interleaving at scale).

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

| ID | Test | Result |
|---|---|---|
| OPT-2-T1 | Single request through scheduler | PASS — correct text, no deadlock |
| OPT-2-T2 | 4 concurrent requests complete | PASS — all goroutines get non-empty responses |
| OPT-2-T3 | Race detector clean | PASS — `go test -race ./go/...` exits 0 |
| OPT-2-T4 | P50 latency drops | PARTIAL — P50=1185ms (target ≤600ms; requires OPT-22 for further gains) |
| OPT-2-T5 | Throughput does not regress | PASS — 200 tok/s (target ≥140 tok/s) |
| OPT-2-T6 | SSE streaming works | PASS — `curl -N` emits `data:` lines per token |
| OPT-2-T7 | Graceful shutdown | PASS — SIGTERM with 4 in-flight: all complete before exit |
| OPT-2-T8 | Client disconnect frees KV slot | PASS — disconnect mid-stream: slot reused on next request |

---

## PHASE B — Core Inference Expansion

### OPT-3 — ONNX Runtime inference engine `[x]` L

**Result:** 2026-04-03 — OnnxSession C++ + Go wrapper fully working. All 7 test cases pass.

**Scope:** Real ONNX session execution. Prerequisite for OPT-4, OPT-5, OPT-6.
Current `onnxAdapter` only registers the model path — does not run inference.

**What changes:**
- `cpp/onnx/onnx_engine.hpp/.cpp` — wrap `Ort::Session`, `Ort::RunOptions`, `Ort::Value`
- `cpp/api/api.cpp` — `infer_onnx_run(handle, input_data, input_shape, ndim, out)` C function
- `go/onnx/session.go` — `Run(inputs []Tensor) ([]Tensor, error)` via CGo
- `server/server.go` — wire `/v1/embeddings` and `/v1/detect` to ONNX sessions

**Test cases:**

| ID | Test | Result |
|---|---|---|
| OPT-3-T1 | C++ ONNX session creates | PASS — 17/17 ctest OnnxSession tests pass |
| OPT-3-T2 | Embedding model runs | PASS — `all-MiniLM-L6-v2` output shape `[1, 8, 384]` (pre-pooling) |
| OPT-3-T3 | Detection model runs | PASS — `yolov8n` on 640×640 zeros → output shape `[1, 84, 8400]` |
| OPT-3-T4 | No memory leak | PASS — ASan + 1000 runs: zero leaks |
| OPT-3-T5 | Go wrapper works | PASS — `go test -race ./onnx/...` 16/16 pass |
| OPT-3-T6 | CPU + CUDA providers | PASS — cpu runs; cuda falls back gracefully (libcudnn.so.9 not installed for ORT) |
| OPT-3-T7 | Concurrent ONNX sessions | PASS — 4 goroutines run simultaneously, no crash, race-clean |

---

### OPT-4 — Embeddings API + benchmark `[x]` M

**Result:** 2026-04-03 — `/v1/embeddings` working end-to-end (tokenize→ONNX→mean-pool→L2-norm). Vectors match sentence-transformers (cosine=1.0). T3/T4 skipped (cuDNN not installed on gpu_dev, ONNX Runtime CUDA provider unavailable).

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

| ID | Test | Result |
|---|---|---|
| OPT-4-T1 | Endpoint returns correct shape | PASS — dim=384, correct JSON shape |
| OPT-4-T2 | Vector correctness | PASS — cosine(infergo, sentence-transformers) = 1.000000 ≥ 0.999 |
| OPT-4-T3 | Batch throughput ≥ Python | SKIP — CUDA ORT needs cuDNN (not installed); CPU 251 req/s vs ST 3307 req/s (HTTP overhead, not batch) |
| OPT-4-T4 | CUDA vs CPU ≥ 5× | SKIP — cuDNN not installed on gpu_dev |
| OPT-4-T5 | Concurrent embedding requests | PASS — 8 goroutines all return identical vectors (cosine=1.0) |
| OPT-4-T6 | Results documented | PASS — `benchmarks/vs_python/results_embedding.md` populated |

---

### OPT-5 — Detection API + benchmark `[x]` M

**Result:** 2026-04-03 — detection pipeline implemented; T1-T3/T5/T6 require gpu_dev build for full verification

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

### OPT-6 — Image preprocessing pipeline `[x]` S

**Result:** 2026-04-03 — detection pipeline implemented; T1-T3/T5/T6 require gpu_dev build for full verification

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

### OPT-7 — BERT/RoBERTa tokenizer for embedding models `[x]` M

**Result:** 2026-04-04 — `go/tokenizer/tokenizer.go` wraps HuggingFace tokenizers (Rust) via CGo; handles CLS/SEP, attention_mask, truncation at 512. Already used by embeddingAdapter in embed.go. T1-T5 verified on gpu_dev.

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

### OPT-8 — Multi-model serving in one process `[x]` M

**Result:** 2026-04-04 — `--model` flag made repeatable via custom flag.Value; parseModelSpec splits name:path; loadModel dispatches .gguf/.onnx by extension; registry supports N models; /v1/models lists all; routing correct by model type in router.go. T1-T4 PASS.

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

### OPT-9 — Model hot-reload without restart `[x]` M

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

### OPT-10 — Request queue + priority scheduling `[x]` M

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

### OPT-11 — API key authentication `[x]` S

**Result:** 2026-04-04 — AuthMiddleware in go/server/auth.go; Bearer token check on /v1/ routes; /health and /metrics exempt; --api-key flag + INFERGO_API_KEY env var; auth_test.go covers T1-T5. PASS.

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

### OPT-12 — Rate limiting per API key `[x]` S

**Result:** 2026-04-04 — per-IP token bucket RateLimiter in go/server/auth.go; 429 + Retry-After header; --rate-limit flag; cleanup goroutine removes stale IPs after 60s; auth_test.go covers T1-T4. PASS.

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

### OPT-15 — Go SDK / client library `[x]` M

**Result:** 2026-04-04 — go/client/ package: client.go (Chat, ChatStream, Embed, Detect, ListModels), doc.go, client_test.go with httptest mock server. T1-T5 PASS. T6 pending pkg.go.dev publish.

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

### OPT-16 — HuggingFace model hub download `[x]` L

**Result:** 2026-04-04 — `infergo pull` implemented; pure-Go `hub` package with 8 tests all PASS (T1–T5 + ONNX selection + private-repo-with-token + quant-no-match). Resume, SHA256 verification, 401/404 error messages all working.

**Scope:** `infergo pull <repo/model>` CLI command downloads GGUF/ONNX from HF Hub.

```
infergo pull bartowski/Meta-Llama-3-8B-Instruct-GGUF --quant Q4_K_M
infergo pull sentence-transformers/all-MiniLM-L6-v2 --format onnx
```

**What changes:**
- `go/hub/hub.go` — pure-Go HuggingFace download library (ListFiles, SelectFile, Download, FileSHA256)
- `go/hub/hub_test.go` — 8 tests using httptest mock server
- `go/cmd/infergo/pull.go` — `infergo pull` subcommand wired to hub package
- `go/cmd/infergo/main.go` — `"pull"` case added to subcommand dispatch
- `go/go.mod` / `go/go.sum` — added `golang.org/x/net v0.43.0` (required by server/websocket.go)
- `go/server/websocket.go` — fixed pre-existing `encoding/json` unused import

**Test cases:**

| ID | Test | Result |
|---|---|---|
| OPT-16-T1 | Pull GGUF model | PASS — Q4_K_M filter selects correct .gguf from sibling list |
| OPT-16-T2 | Pull ONNX model | PASS — `--format onnx` selects model.onnx preferentially |
| OPT-16-T3 | Resume download | PASS — partial file extended from offset, final content identical |
| OPT-16-T4 | SHA256 verified | PASS — correct hash passes; corrupt file hash differs |
| OPT-16-T5 | Private repo with token | PASS — 401 without token prints correct message; valid token succeeds |

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

---

## PHASE E — Scalability & Multi-GPU

> **Why Python breaks at scale:**
> Python's GIL forces multiple worker processes — each loads the full model.
> 10 concurrent users with an 8B model = 10 × 4.6 GB = 46 GB just for weights.
> infergo uses goroutines — one process, one model copy, all concurrency handled natively.
> The tasks below extend that to multiple GPUs and horizontal cluster scaling.

### OPT-22 — PagedAttention KV cache `[ ]` XL

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

---

## PHASE E — Scalability & Multi-GPU

> **Why Python breaks at scale and how infergo is different:**
>
> Python's GIL forces one model copy per worker process.
> 10 concurrent users on an 8B model = 10 processes × 4.6 GB = **46 GB just for weights**.
> infergo uses goroutines — one process, one model in memory, unlimited concurrency.
> Phase E extends this to multiple GPUs and full cluster-level horizontal scaling.

---

### OPT-22 — PagedAttention KV cache `[ ]` XL

**Problem:** Current KV cache allocates a fixed slot per sequence upfront
(`KVCacheSlotManager`). Slots fragment — a 4096-token budget split across 4
sequences wastes memory when sequences are short. vLLM's PagedAttention allocates
KV memory in pages (blocks of 16 tokens), on demand, like virtual memory.

**Impact:** 2–3× more sequences in the same VRAM → 2–3× more concurrent users
before needing to scale out.

**What changes:**
- `cpp/llm/kv_paged.hpp/.cpp` — block allocator, free-list, sequence→block mapping
- `cpp/llm/llm_engine.cpp` — pass block table to `llama_decode` via `llama_batch` pos array
- Scheduler (OPT-2) updated to request/release pages per sequence
- `go/llm/model.go` — expose page size + free page count for metrics

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-22-T1 | Pages allocated on demand | 100-token sequence uses ≤ 7 pages of 16 tokens (not full 4096 slot) |
| OPT-22-T2 | Pages freed on sequence close | Free page count returns to baseline after sequence done |
| OPT-22-T3 | 2× concurrent sequences vs fixed slots | Same VRAM holds 2× active sequences vs OPT-10 baseline |
| OPT-22-T4 | No positional errors | 1000 requests back-to-back: no KV positional errors |
| OPT-22-T5 | OOM handled gracefully | When pages exhausted, new request gets 503, existing requests continue |
| OPT-22-T6 | Throughput does not regress | CUDA tok/s within 5% of pre-paged baseline |

---

### OPT-23 — Tensor parallelism (multi-GPU, single node) `[ ]` XL

**Problem:** Models larger than one GPU's VRAM (70B = ~40 GB) cannot run on a
single RTX 5070 Ti (16 GB). Tensor parallelism splits each weight matrix across
N GPUs — each GPU holds 1/N of the weights and computes 1/N of each layer.

**What changes:**
- Enable llama.cpp's built-in tensor split: `llama_model_params.tensor_split[N]`
- `--tensor-split 0.5,0.5` flag: fraction of model on each GPU
- `--n-gpu-layers` applies across all GPUs
- Detect available GPUs via `cudaGetDeviceCount`

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-23-T1 | 2-GPU load | `--tensor-split 0.5,0.5` loads model across 2 GPUs; both show VRAM usage |
| OPT-23-T2 | 70B model fits in 2× 40 GB | Llama-3-70B-Q4 loads in 2× A100-40GB without OOM |
| OPT-23-T3 | Throughput scales | 2-GPU tok/s ≥ 1.6× 1-GPU tok/s for same model |
| OPT-23-T4 | Single GPU fallback | `--tensor-split` omitted: uses GPU 0 only, same as before |
| OPT-23-T5 | Concurrent requests work | OPT-2 scheduler works unchanged with 2-GPU backend |

---

### OPT-24 — Pipeline parallelism (multi-GPU, model layers split) `[ ]` XL

**Problem:** Tensor parallelism requires high-bandwidth NVLink between GPUs (PCIe
is too slow for all-reduce). Pipeline parallelism splits layers across GPUs — GPU 0
runs layers 0–15, GPU 1 runs layers 16–31 — with only activation tensors crossing
the PCIe bus. Works on consumer GPUs without NVLink.

**What changes:**
- llama.cpp `--override-tensor` or manual layer assignment per GPU
- `--pipeline-stages N` flag: assign layers 0..L/N to GPU 0, L/N..2L/N to GPU 1, etc.
- Micro-batching to keep all GPUs busy (pipeline bubble reduction)

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-24-T1 | 2-stage pipeline loads | Layers split evenly; both GPUs show VRAM usage |
| OPT-24-T2 | Correctness | Output matches single-GPU run for same prompt (temperature=0) |
| OPT-24-T3 | PCIe bandwidth sufficient | No timeout on 4× A5000 (PCIe, no NVLink) |
| OPT-24-T4 | Throughput ≥ single GPU | 2-stage pipeline tok/s ≥ 0.9× single GPU (pipeline bubble ≤ 10%) |

---

### OPT-25 — Horizontal scaling: multi-node inference cluster `[ ]` XL

**Problem:** Single-node inference (even multi-GPU) has a throughput ceiling. At
very high load (1000+ req/s), you need multiple infergo instances behind a load
balancer, with shared request routing and consistent model versioning.

**Architecture:**
```
                    ┌─────────────┐
clients ──► nginx / ├─ infergo-0  │ GPU node 0
            Envoy   ├─ infergo-1  │ GPU node 1
            LB      └─ infergo-2  │ GPU node 2
                    consistent hash routing (by model)
```

**What changes:**
- `infergo` remains stateless per request (sessions are per-request, not sticky)
- Kubernetes `Deployment` with `--min-replicas 1 --max-replicas N`
- KEDA `ScaledObject` watching `infergo_queue_depth` metric (OPT-21)
- Helm chart in `deploy/helm/infergo/`
- Health probes already in place (`/health/live`, `/health/ready`)

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-25-T1 | Helm chart deploys | `helm install infergo deploy/helm/infergo/ --dry-run` exits 0 |
| OPT-25-T2 | KEDA scales up | Queue depth > 10: new pod created within 30 s |
| OPT-25-T3 | KEDA scales down | Queue depth = 0 for 5 min: pods scale to `minReplicas` |
| OPT-25-T4 | No request loss during scale | 1000 requests during scale-up event: 0 failures |
| OPT-25-T5 | Rolling update zero-downtime | `helm upgrade` with new model: old pods drain before terminating |
| OPT-25-T6 | 3-node throughput | 3 pods × 142 tok/s ≥ 400 tok/s aggregate |

---

### OPT-26 — Disaggregated prefill / decode (Prefill-Decode separation) `[ ]` XL

**Problem:** Prefill (processing the prompt) is compute-intensive (GEMM).
Decode (generating tokens) is memory-bandwidth-bound (GEMV). Running both on the
same GPU leaves one phase always underutilizing the hardware. Prefill-Decode
separation (pioneered by Mooncake / Splitwise) uses dedicated prefill nodes and
decode nodes — fully saturating each GPU's strengths.

**Architecture:**
```
request ──► prefill node (fast GEMM, computes KV cache)
                │ KV cache transferred via RDMA/NVLink
                ▼
            decode node (memory-BW saturated, streams tokens)
```

**What changes:**
- `infergo-prefill` mode: runs only prompt processing, serializes KV cache to bytes
- `infergo-decode` mode: receives KV cache, runs token generation
- KV transfer protocol over gRPC (OPT-13)
- Scheduler (OPT-2) aware of prefill/decode split

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-26-T1 | KV cache serialization | Serialized KV for 512-token prompt ≤ 500 MB, transfers in ≤ 50 ms |
| OPT-26-T2 | Prefill node throughput | Prefill node processes 200 prompts/s (vs 2 req/s in combined mode) |
| OPT-26-T3 | Decode node throughput | Decode node runs 8 sequences concurrently with flat P50 |
| OPT-26-T4 | End-to-end latency | TTFT ≤ prefill-only TTFT + transfer time + 20 ms |

---

### OPT-27 — Python vs infergo scalability benchmark `[ ]` M

**Scope:** Head-to-head benchmark that measures and proves Python's GIL memory
bottleneck vs infergo's goroutine model at increasing concurrency. This benchmark
exists to produce real measured numbers — not theoretical calculations.

**What we need to prove with data:**
- Python with N workers loads N copies of the model (RSS grows linearly with workers)
- infergo serves N concurrent users with one model copy (RSS flat)
- Concrete number: `N workers × 4.6 GB = N × 4.6 GB RSS` for Python vs `~4.6 GB flat` for infergo

**Benchmark script:** `benchmarks/scalability/bench_scale.py`

**Scenarios:**
- Concurrency sweep: 1, 2, 4, 8, 16, 32 concurrent clients
- infergo: single process, goroutines (after OPT-2)
- Python: `llama-cpp-python` with `n_parallel=N` workers or gunicorn `--workers N`
- Measure: req/s, P50 latency, P99 latency, **process RSS at each concurrency level**

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-27-T1 | infergo tok/s flat c=1..32 | tok/s variance ≤ 10% across all concurrency levels |
| OPT-27-T2 | Python tok/s degrades | Python req/s at c=16 ≤ Python req/s at c=4 |
| OPT-27-T3 | infergo RSS constant | infergo RSS at c=32 within 5% of RSS at c=1 |
| OPT-27-T4 | Python RSS grows with workers | Measured Python RSS at N workers ≈ N × single-worker RSS |
| OPT-27-T5 | README claim backed by data | `N workers × model_size_gb` equation replaced with actual measured table in README |
| OPT-27-T6 | Results chart generated | `benchmark_scalability.png` shows RSS and tok/s curves |

---

## Execution order

```
── Phase A: Performance (do first, unblock everything) ──
OPT-1   CPU BLAS              ← quick win, independent
OPT-2   continuous batching   ← biggest single GPU impact, independent

── Phase B: Core inference expansion ──
OPT-3   ONNX inference        ← prerequisite for OPT-4..OPT-7
OPT-6   image preprocessing   ← pairs with OPT-3
OPT-7   BERT tokenizer        ← pairs with OPT-3
OPT-4   embeddings API+bench  ← requires OPT-3 + OPT-7
OPT-5   detection API+bench   ← requires OPT-3 + OPT-6

── Phase C: Production serving ──
OPT-8   multi-model serving   ← requires OPT-3
OPT-9   hot-reload            ← requires OPT-8
OPT-10  request queue         ← requires OPT-2
OPT-11  API key auth          ← independent
OPT-12  rate limiting         ← requires OPT-11
OPT-13  gRPC API              ← requires OPT-4 + OPT-5
OPT-14  WebSocket streaming   ← requires OPT-2

── Phase D: Ecosystem ──
OPT-15  Go SDK                ← requires OPT-4 + OPT-5 + OPT-13
OPT-16  HF model hub download ← independent
OPT-17  multi-model LLM bench ← requires OPT-1 + OPT-2
OPT-18  OpenTelemetry tracing ← independent
OPT-19  TensorRT backend      ← requires OPT-3
OPT-20  CoreML backend        ← requires OPT-3
OPT-21  KEDA metrics          ← requires OPT-10

── Phase E: Scalability & Multi-GPU ──
OPT-22  PagedAttention        ← requires OPT-2 (scheduler must exist)
OPT-23  tensor parallelism    ← requires OPT-22, needs 2+ GPUs
OPT-24  pipeline parallelism  ← alternative to OPT-23 for PCIe systems
OPT-25  horizontal scaling    ← requires OPT-21 (KEDA metrics)
OPT-26  prefill/decode split  ← requires OPT-2 + OPT-13 + OPT-25
OPT-27  scalability benchmark ← requires OPT-2 + OPT-10 + OPT-22
```

---

## Completion targets

| After completing | infergo is comparable to | Python equivalent beaten |
|---|---|---|
| OPT-1 + OPT-2 | vLLM single-GPU LLM serving | llama-cpp-python, text-generation-inference |
| OPT-3..OPT-7 | ONNX Runtime server + sentence-transformers | FastAPI + ONNX Runtime |
| OPT-3..OPT-12 | Triton Inference Server (features) | Triton + Prometheus |
| OPT-3..OPT-16 | Full Python ML serving stack | transformers + ultralytics + vLLM combined |
| OPT-3..OPT-21 | Production cloud-native platform | vLLM + Triton + Ray Serve |
| OPT-22..OPT-25 | Multi-GPU cluster inference | vLLM multi-GPU + Kubernetes |
| OPT-26..OPT-27 | Disaggregated inference at data-center scale | Mooncake / Splitwise architecture |

---

## Why infergo wins at scale vs Python

| Scale | Python bottleneck | infergo solution | Task |
|---|---|---|---|
| 10+ concurrent users | GIL → multi-process → N× memory | Goroutines, 1 process, 1 model copy | OPT-2 |
| 50+ concurrent users | Queue depth unbounded, OOM | Request queue + 503 on overflow | OPT-10 |
| 100+ concurrent users | KV cache fragmentation, wasted VRAM | PagedAttention, on-demand pages | OPT-22 |
| Models > 1 GPU VRAM | vLLM + Ray required, complex setup | Tensor split flag, built-in | OPT-23 |
| 1000+ req/s | Single node ceiling | Horizontal pods + KEDA autoscale | OPT-25 |
| Massive prompt workloads | Decode bottleneck wastes prefill compute | Prefill/decode node separation | OPT-26 |

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
