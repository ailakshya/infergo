# infergo

<p align="center">
  <strong>Production AI inference in Go. Faster than Python. One binary.</strong>
</p>

<p align="center">
  <a href="https://github.com/ailakshya/infergo/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://pkg.go.dev/github.com/ailakshya/infergo"><img src="https://pkg.go.dev/badge/github.com/ailakshya/infergo.svg" alt="Go Reference"></a>
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900.svg" alt="CUDA 12">
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey.svg" alt="Platform">
</p>

<p align="center">
  <a href="docs/getting-started.md">Getting Started</a> · <a href="docs/python.md">Python</a> · <a href="docs/detection.md">Detection</a> · <a href="docs/go-api-reference.md">Go API</a> · <a href="docs/deployment.md">Deployment</a> · <a href="benchmarks/vs_python/results_full.md">Benchmarks</a>
</p>

---

## What is infergo

infergo is a production inference runtime that serves LLMs, embedding models, and object detection from a single Go binary. It wraps llama.cpp and ONNX Runtime behind an OpenAI-compatible HTTP API. No Python required.

```bash
# One command. LLM + embedding + detection on one port.
infergo serve \
  --model llm:models/llama3-8b-q4.gguf \
  --model embed:models/all-MiniLM-L6-v2.onnx \
  --model detect:models/yolov8n.onnx \
  --provider cuda
```

```python
# Works with any OpenAI client. Zero code changes.
from openai import OpenAI
client = OpenAI(base_url="http://localhost:9090/v1", api_key="none")
response = client.chat.completions.create(
    model="llm",
    messages=[{"role": "user", "content": "Hello"}],
    response_format={"type": "json_object"}  # guaranteed valid JSON
)
```

---

## Performance

Measured on RTX 5070 Ti, CUDA 12.8. All numbers are real, reproducible, and published with benchmark scripts.

### Single request latency

| Task | infergo | Python | Speedup |
|---|---|---|---|
| LLM generation (per token) | **1.69 ms** | 13.62 ms | **8.1x** |
| Speculative decoding (8B+1B draft) | **74 ms** | 496 ms | **6.7x** |
| RAG pipeline (embed + search + generate) | **116 ms** | 642 ms | **5.5x** |
| Structured JSON output | **125 ms** | impossible | **100% valid** |
| Single embedding | **0.9 ms** | 1.9 ms | **2.1x** |
| Batch embedding (3 texts) | **1.4 ms** | 2.5 ms | **1.8x** |
| Reranking (3 docs) | **1.2 ms** | 6.2 ms | **5.2x** |
| Object detection (yolo11n) | **2.4 ms** | 2.7 ms | **1.1x** |
| HNSW vector search (k=10) | **0.03 ms** | ~1 ms | **33x** |

### Throughput under concurrent load

| Concurrency | infergo embedding | Python embedding | infergo detection | infergo reranking |
|---|---|---|---|---|
| c=1 | 692 req/s | 437 req/s | 177 req/s | 212 req/s |
| c=4 | 957 req/s | 459 req/s | 110 req/s | 250 req/s |
| c=8 | 951 req/s | **46 req/s** | 124 req/s | 250 req/s |
| c=16 | 927 req/s | **crashed** | 233 req/s | 246 req/s |
| c=32 | **911 req/s** | **crashed** | **239 req/s** | **241 req/s** |

At c=8, Python's P99 latency spikes to 1,036ms. At c=16, Python crashes entirely. infergo serves 927 req/s at c=32 with zero errors.

### RAG pipeline — 5 configurations

Full end-to-end: embed query, search documents, generate answer. Qwen 2.5 Coder 1.5B + all-MiniLM-L6-v2.

| Configuration | P50 | Cold start | VRAM | GPU util. |
|---|---|---|---|---|
| Python CPU | 636 ms | 4,931 ms | — | 0% |
| Python GPU | 642 ms | 4,907 ms | 2,720 MB | 15% |
| **infergo CPU** | **162 ms** | 1,043 ms | — | 0% |
| **infergo GPU** | **116 ms** | 1,037 ms | 5,350 MB | 85% |
| **infergo GPU + JSON** | **125 ms** | 1,037 ms | 5,350 MB | 85% |

### Cloud cost (per 1M RAG requests)

| Cloud tier | Python | infergo | Savings |
|---|---|---|---|
| T4 ($0.35/hr) | $62 | $11 | **82%** |
| A100 ($3/hr) | $534 | $96 | **82%** |
| H100 ($8/hr) | $1,424 | $256 | **82%** |

---

## Why infergo exists

Python inference servers have five production problems:

**1. The GIL blocks concurrency.** Python's Global Interpreter Lock allows only one thread to execute at a time. Ten concurrent users means the tenth waits for the first nine to finish. The standard fix — forking processes — loads a full model copy per process. Llama 3 8B needs 4.6 GB VRAM per copy. Ten users need 46 GB.

**2. Cold start kills autoscaling.** Python + PyTorch + transformers takes 5 seconds to start. Kubernetes pods are useless during that time. infergo starts in 1 second.

**3. No structured output guarantee.** When you ask an LLM for JSON, Python hopes the model complies. infergo enforces it with GBNF grammar sampling — the output is syntactically valid by construction. 100% validity, every request.

**4. Three servers for three model types.** vLLM for LLMs, sentence-transformers for embeddings, ultralytics for detection. Three processes, three configurations, three failure domains. infergo serves all three from one binary on one port.

**5. Container bloat.** Python + PyTorch + CUDA runtime + model framework = 10 GB Docker image. infergo's CPU image is 0.18 GB. The CUDA image is 1.52 GB.

---

## Architecture

### Request flow

```
Client (Python / Go / curl / any language)
  │
  │  POST /v1/chat/completions (JSON)
  ▼
┌──────────────────────────────────────────────────┐
│  Go HTTP layer                                   │
│  Parse JSON → route → 1 CGo call → respond      │
│  Time: 0.3ms. No inference compute.              │
└──────────────────────┬───────────────────────────┘
                       │  1 CGo call (entire request)
┌──────────────────────▼───────────────────────────┐
│  C++ compute engine                              │
│  ┌────────────┬────────────┬──────────┬────────┐ │
│  │ llama.cpp  │ ONNX RT /  │ libtorch │ HNSW   │ │
│  │ LLM decode │ TorchScript│ nvJPEG   │ search │ │
│  │ grammar    │ embedding  │ detect   │ rerank │ │
│  │ speculate  │ batch      │ NMS      │ vector │ │
│  │ cache      │ pool+norm  │ preproc  │ DB     │ │
│  └────────────┴────────────┴──────────┴────────┘ │
└──────────────────────┬───────────────────────────┘
                       │
              NVIDIA CUDA / CPU / Metal
```

### Why Go + C++

Go handles HTTP, routing, authentication, metrics, and concurrency. C++ handles all inference compute. The boundary is one CGo call per request — not one per token.

**Go's role (the receptionist):**
- Parse HTTP request: 0.1ms
- Route to correct model: 0.01ms
- Serialize JSON response: 0.2ms
- Handle 10,000+ concurrent connections with goroutines (8 KB each)
- No GIL. No interpreter. Compiled binary.

**C++'s role (the doctor):**
- Tokenize, prefill, decode, sample, detokenize: 115ms
- Prompt cache, grammar sampling, speculative decoding
- Flash Attention, nvJPEG GPU decode, HNSW search
- Zero copies between languages during generation

### How Go handles concurrency vs Python's GIL

```
Python: 10 users → 10th waits 6,420ms
─────────────────────────────────────
  Request 1: [===GIL LOCKED=== 642ms ===GIL UNLOCKED===]
  Request 2:                                              [=== 642ms ===]
  Request 3:                                                              [...]
  ...
  Request 10: waits 5,778ms before starting

  Fix: fork 10 processes
    Each loads full model → 27,200 MB VRAM
    Each loads Python + PyTorch → 11,200 MB RAM
    Won't fit on any single GPU.


infergo: 10 users → all start immediately
─────────────────────────────────────────
  Goroutine 1 (8 KB): [== 116ms ==]
  Goroutine 2 (8 KB): [== 116ms ==]
  Goroutine 3 (8 KB): [== 116ms ==]   ← continuous batching
  ...                                    groups all into one
  Goroutine 10 (8 KB): [== 116ms ==]    GPU forward pass

  Total memory: 80 KB goroutine stacks + 1 model copy
  Total VRAM: same 5,350 MB regardless of user count
```

### GPU memory breakdown

```
Python (2,720 MB VRAM, 15% GPU utilization):
  Model weights:        1,100 MB
  KV cache:               200 MB
  PyTorch allocator:      920 MB  (pre-allocated, mostly idle)
  Embedding model:        100 MB
  Compute workspace:      400 MB  (small — GPU starved for work)

  Per token: GIL lock (0.5ms) → GPU decode (2ms) → copy logits
  to Python (1ms) → sample in Python (1ms) → GIL release (0.2ms)
  GPU busy: 2ms out of 5ms = 40%. Idle 60%.

infergo (5,350 MB VRAM, 85% GPU utilization):
  Model weights:        1,100 MB
  KV cache:               200 MB
  Compute workspace:    1,200 MB  (3x larger — GPU stays busy)
  Flash Attention:      2,100 MB  (2x faster prefill, O(N) memory)
  Embedding model:        100 MB
  Prompt cache:           100 MB  (skip prefill on repeat prompts)
  nvJPEG + speculative:   550 MB  (GPU JPEG decode + draft model)

  Per token: GPU decode (2ms) → sample in C++ (0.01ms) → next
  GPU busy: 2ms out of 2.01ms = 99%. Zero idle time.
  
  More VRAM spent = more GPU utilized = faster inference.
  The VRAM is already paid for. Not using it is waste.
```

---

## Features

**Inference:** LLM (GGUF), embedding (ONNX/TorchScript), detection (TorchScript/ONNX/TensorRT), vector search (HNSW), reranking, RAG pipeline

**Performance:** Full C generation loop (1 CGo call/request), nvJPEG GPU decode, speculative decoding (6.7x), prompt caching, continuous batching, Flash Attention 2, grammar sampling, zero-copy sampling

**AI capabilities:** Structured output (JSON/GBNF), function calling, batch embeddings, vector database (CRUD + persistence), document ingestion, reranking, guardrails

**Production:** Multi-model serving, hot reload, API key auth, rate limiting, request queue, Prometheus metrics, OpenTelemetry tracing, KEDA autoscaling, preemption

**Deployment:** Docker (CPU 0.18 GB, CUDA 1.52 GB), Helm chart, multi-GPU (tensor split, pipeline stages, auto-shard), 1-second cold start

**Endpoints:** 25 HTTP endpoints including `/v1/chat/completions`, `/v1/embeddings`, `/v1/detect`, `/v1/search`, `/v1/rerank`, `/v1/rag`, `/v1/ingest`, `/ui` (built-in chat)

---

## Quickstart

```bash
# Install
curl -sSL https://github.com/ailakshya/infergo/releases/latest/download/infergo-linux-amd64-cpu.tar.gz \
  | tar xz && sudo mv infergo /usr/local/bin/

# Download a model
infergo pull bartowski/Meta-Llama-3-8B-Instruct-GGUF \
  --filename Meta-Llama-3-8B-Instruct-Q4_K_M.gguf

# Serve
infergo serve --model models/llama3-8b-q4.gguf --provider cuda

# Chat
curl http://localhost:9090/v1/chat/completions \
  -d '{"model":"llama3-8b-q4","messages":[{"role":"user","content":"Hello"}]}'

# Speculative decoding (6.7x faster)
infergo serve --model llm:models/llama3-8b-q4.gguf \
  --draft-model models/llama3.2-1b-q4.gguf --provider cuda

# Multi-model
infergo serve \
  --model llm:models/llama3-8b-q4.gguf \
  --model embed:models/all-MiniLM-L6-v2.onnx \
  --model detect:models/yolov8n.onnx \
  --provider cuda
```

### Docker

```bash
# CPU (0.18 GB image)
docker run --rm -p 9090:9090 -v ./models:/models:ro \
  ghcr.io/ailakshya/infergo:cpu serve --model /models/llama3-8b-q4.gguf

# CUDA (1.52 GB image)
docker run --rm --gpus all -p 9090:9090 -v ./models:/models:ro \
  ghcr.io/ailakshya/infergo:cuda serve --model /models/llama3-8b-q4.gguf --provider cuda
```

---

## API

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat completion (streaming, structured output, function calling) |
| `POST` | `/v1/embeddings` | Dense embeddings (single or batch) |
| `POST` | `/v1/search` | Vector similarity search (HNSW) |
| `POST` | `/v1/rerank` | Rerank documents by query relevance |
| `POST` | `/v1/rag` | Full RAG pipeline (embed + search + generate) |
| `POST` | `/v1/ingest` | Ingest documents into vector DB |
| `POST` | `/v1/detect` | Object detection (JSON + base64) |
| `POST` | `/v1/detect/binary` | Object detection (raw JPEG, faster) |
| `POST` | `/v1/detect/stream` | Streaming detection (SSE) |
| `POST` | `/v1/images/generations` | Image generation |
| `POST` | `/v1/audio/transcriptions` | Speech-to-text |
| `POST` | `/v1/batches` | Async batch inference |
| `GET` | `/v1/models` | List loaded models |
| `POST` | `/v1/admin/reload` | Hot-swap model weights |
| `POST` | `/v1/admin/guardrails` | Configure content safety |
| `POST` | `/v1/admin/templates` | Manage prompt templates |
| `GET` | `/ui` | Built-in chat interface |
| `GET` | `/health/live` | Liveness probe (20,623 req/s) |
| `GET` | `/health/ready` | Readiness probe |
| `GET` | `/metrics` | Prometheus metrics |

### Go SDK

```bash
go get github.com/ailakshya/infergo@v1.1.0
```

```go
c := client.New("http://localhost:9090", client.WithAPIKey("key"))

resp, _ := c.Chat(ctx, client.ChatRequest{
    Model:    "llm",
    Messages: []client.Message{{Role: "user", Content: "Hello"}},
})

vec, _ := c.Embed(ctx, client.EmbedRequest{Model: "embed", Input: "hello"})

dets, _ := c.Detect(ctx, client.DetectRequest{Model: "detect", ImageB64: b64})
```

### CLI

```
infergo serve       Start inference server
infergo pull        Download model from HuggingFace
infergo convert     Export model (ONNX, TorchScript, TensorRT, quantize)
infergo models      List / info / delete local models
infergo benchmark   Load test a running server
```

---

## Build from source

```bash
git clone https://github.com/ailakshya/infergo && cd infergo

# CPU
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target infer_api -j$(nproc)
go build -C go -o ../infergo ./cmd/infergo

# CUDA
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DINFER_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --target infer_api -j$(nproc)
go build -C go -o ../infergo ./cmd/infergo
```

---

## Testing

```bash
# 342 C++ tests (100% pass rate)
ctest --test-dir build --output-on-failure

# 19 Go packages, race detector clean
cd go && go test -race ./...
```

---

## License

[Apache 2.0](LICENSE)
