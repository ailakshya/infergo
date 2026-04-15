# infergo

<p align="center">
  <strong>Production AI platform in Go. Same speed as llama.cpp. One binary for LLM + embedding + detection + RAG.</strong>
</p>

<p align="center">
  <a href="https://github.com/ailakshya/infergo/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://pkg.go.dev/github.com/ailakshya/infergo"><img src="https://pkg.go.dev/badge/github.com/ailakshya/infergo.svg" alt="Go Reference"></a>
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900.svg" alt="CUDA 12">
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey.svg" alt="Platform">
</p>

<p align="center">
  <a href="docs/getting-started.md">Getting Started</a> ·
  <a href="docs/python.md">Python</a> ·
  <a href="docs/detection.md">Detection</a> ·
  <a href="docs/go-api-reference.md">Go API</a> ·
  <a href="docs/deployment.md">Deployment</a> ·
  <a href="benchmarks/vs_python/results_full.md">Benchmarks</a>
</p>

---

## Why infergo

Production AI needs more than just an LLM. You need embeddings, vector search, detection, RAG — and they all need to be fast, concurrent, and deployable as one unit.

The standard approach: 6 Python services, 6 ports, 6 containers, 8 GB RAM, Python GIL killing concurrency.

infergo: **one binary, one port, one container.** Same LLM speed as llama.cpp with zero overhead.

```bash
# One command. LLM + embedding + detection + vector search on one port.
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
    response_format={"type": "json_object"}
)
```

---

## Performance

All numbers measured on RTX 5070 Ti, CUDA 12.8, Qwen 2.5 Coder 1.5B Q4_K_M. Benchmark scripts included.

### LLM inference — same speed as raw llama.cpp

| Engine | Avg latency | ms/tok | Go overhead |
|---|---|---|---|
| **Raw llama.cpp** (C++, no server) | 204ms | 3.46 | — |
| **infergo** (Go + HTTP + llama.cpp) | 205ms | 3.48 | **0.7ms (0.3%)** |
| LM Studio | 176ms | 3.10 | custom CUDA fork |
| Python (llama-cpp-python) | 1,575ms | 29.0 | GIL + ctypes |

infergo runs at **100% of raw llama.cpp speed.** The Go HTTP layer adds 0.7ms — within measurement noise. The 29ms gap vs LM Studio is their proprietary CUDA kernels, not Go overhead.

### Concurrent throughput (infergo vs LM Studio vs Python)

| Concurrency | infergo | LM Studio | Python |
|---|---|---|---|
| c=1 | 206ms / 231 tok/s | 172ms / 248 tok/s | 1,607ms / 33 tok/s |
| c=4 | 756ms / 289 tok/s | 491ms / 444 tok/s | 4,936ms / 41 tok/s |
| c=8 | 1,356ms / 287 tok/s | 873ms / 460 tok/s | 8,146ms / 44 tok/s |

infergo is **7x faster than Python** at every concurrency level. Continuous batching groups concurrent requests into shared GPU decode calls.

### Multi-model capabilities (things llama.cpp can't do)

| Task | infergo | llama-server | Python |
|---|---|---|---|
| LLM generation | 3.5 ms/tok | 3.4 ms/tok | 29 ms/tok |
| Single embedding | 0.9 ms | not supported | 1.9 ms |
| Batch embedding (3) | 1.4 ms | not supported | 2.5 ms |
| Reranking (3 docs) | 1.2 ms | not supported | 6.2 ms |
| Object detection | 2.4 ms | not supported | 2.7 ms |
| Vector search (k=10) | 0.03 ms | not supported | ~1 ms |
| RAG pipeline (end-to-end) | 116 ms | not supported | 642 ms |
| Structured JSON output | 125 ms | partial | unreliable |

### Embedding + detection throughput

| Concurrency | infergo embedding | Python embedding |
|---|---|---|
| c=1 | 692 req/s | 437 req/s |
| c=8 | 951 req/s | **46 req/s** |
| c=16 | 927 req/s | **crashed** |
| c=32 | **911 req/s** | **crashed** |

At c=16, Python crashes. infergo serves 927 req/s at c=32 with zero errors.

---

## Architecture

```
Client (Python / Go / curl / any language)
  |
  |  POST /v1/chat/completions
  v
+--------------------------------------------------+
|  Go HTTP layer (0.7ms overhead)                  |
|  JSON parse -> route -> 1 CGo call -> respond    |
+-------------------------+------------------------+
                          |  1 CGo call
+-------------------------v------------------------+
|  C++ inference engine                            |
|  +------------+------------+----------+--------+ |
|  | llama.cpp  | ONNX RT /  | libtorch | HNSW   | |
|  | LLM decode | TorchScript| nvJPEG   | search | |
|  | grammar    | embedding  | detect   | rerank | |
|  | speculate  | batch      | NMS      | vector | |
|  | cache      | pool+norm  | preproc  | DB     | |
|  +------------+------------+----------+--------+ |
+-------------------------+------------------------+
                          |
                 NVIDIA CUDA / CPU
```

### Why Go + C++

**Go** handles HTTP, routing, metrics, and concurrency. 10,000 concurrent connections cost 80 KB of goroutine stacks. No GIL. Compiled binary.

**C++** handles all inference compute. The entire generation loop runs in C++ — one CGo call per request, not one per token. Result: **0.7ms overhead** vs raw C++.

**Why not just llama-server?** If you only need LLM, use llama-server. It's great. But if you need LLM + embeddings + detection + search + RAG in production, that's 6 Python services. infergo replaces all of them.

---

## Features

| Category | Capabilities |
|---|---|
| **Inference** | LLM (GGUF), embedding (ONNX/TorchScript), detection (TorchScript/ONNX), vector search (HNSW), reranking, RAG pipeline |
| **LLM** | Full C generation loop, prefix caching, continuous batching, Flash Attention, grammar sampling (JSON/GBNF/TOON), speculative decoding, ChatML templates |
| **AI** | Structured output, function calling, batch embeddings, vector DB (CRUD + persistence), document ingestion, reranking, guardrails |
| **Production** | Multi-model serving, hot reload, API key auth, rate limiting, Prometheus metrics, health checks, preemption |
| **Deployment** | Docker (CPU 0.18 GB, CUDA 1.52 GB), Helm chart, multi-GPU support, 1s cold start |
| **API** | 25 OpenAI-compatible HTTP endpoints, gRPC, Go SDK, built-in chat UI |

---

## Quickstart

```bash
# Install
curl -sSL https://github.com/ailakshya/infergo/releases/latest/download/infergo-linux-amd64-cpu.tar.gz \
  | tar xz && sudo mv infergo /usr/local/bin/

# Serve
infergo serve --model models/llama3-8b-q4.gguf --provider cuda

# Chat
curl http://localhost:9090/v1/chat/completions \
  -d '{"model":"llama3-8b-q4","messages":[{"role":"user","content":"Hello"}]}'

# Multi-model (LLM + embedding + detection)
infergo serve \
  --model llm:models/llama3-8b-q4.gguf \
  --model embed:models/all-MiniLM-L6-v2.onnx \
  --model detect:models/yolov8n.onnx \
  --provider cuda

# Speculative decoding (6.7x faster)
infergo serve --model llm:models/llama3-8b-q4.gguf \
  --draft-model models/llama3.2-1b-q4.gguf --provider cuda
```

### Docker

```bash
# CPU (0.18 GB)
docker run --rm -p 9090:9090 -v ./models:/models:ro \
  ghcr.io/ailakshya/infergo:cpu serve --model /models/llama3-8b-q4.gguf

# CUDA (1.52 GB)
docker run --rm --gpus all -p 9090:9090 -v ./models:/models:ro \
  ghcr.io/ailakshya/infergo:cuda serve --model /models/llama3-8b-q4.gguf --provider cuda
```

---

## API

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat completion (streaming, JSON mode, function calling) |
| `POST` | `/v1/completions` | Text completion |
| `POST` | `/v1/embeddings` | Dense embeddings (single or batch) |
| `POST` | `/v1/search` | Vector similarity search (HNSW) |
| `POST` | `/v1/rerank` | Rerank documents by query relevance |
| `POST` | `/v1/rag` | Full RAG pipeline (embed + search + generate) |
| `POST` | `/v1/ingest` | Ingest documents into vector DB |
| `POST` | `/v1/detect` | Object detection (JSON + base64) |
| `POST` | `/v1/detect/binary` | Object detection (raw JPEG) |
| `POST` | `/v1/batches` | Async batch inference |
| `GET` | `/v1/models` | List loaded models |
| `POST` | `/v1/admin/reload` | Hot-swap model weights |
| `GET` | `/ui` | Built-in chat interface |
| `GET` | `/health/live` | Liveness probe |
| `GET` | `/health/ready` | Readiness probe |
| `GET` | `/metrics` | Prometheus metrics |

### Go SDK

```go
c := client.New("http://localhost:9090", client.WithAPIKey("key"))

resp, _ := c.Chat(ctx, client.ChatRequest{
    Model:    "llm",
    Messages: []client.Message{{Role: "user", Content: "Hello"}},
})

vec, _ := c.Embed(ctx, client.EmbedRequest{Model: "embed", Input: "hello"})

dets, _ := c.Detect(ctx, client.DetectRequest{Model: "detect", ImageB64: b64})
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
  -DGGML_CUDA=ON -DGGML_CUDA_FA=ON
cmake --build build --target infer_api -j$(nproc)
go build -C go -o ../infergo ./cmd/infergo
```

---

## License

[Apache 2.0](LICENSE)
