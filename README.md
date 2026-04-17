# infergo

<p align="center">
  <strong>Production AI inference platform in Go.<br>One 90 MB binary for LLM + embedding + detection + RAG + search + agents.</strong>
</p>

<p align="center">
  <a href="https://github.com/ailakshya/infergo/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://pkg.go.dev/github.com/ailakshya/infergo"><img src="https://pkg.go.dev/badge/github.com/ailakshya/infergo.svg" alt="Go Reference"></a>
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900.svg" alt="CUDA 12">
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey.svg" alt="Platform">
  <img src="https://img.shields.io/badge/tests-252%20passing-brightgreen.svg" alt="Tests">
</p>

<p align="center">
  <a href="docs/getting-started.md">Getting Started</a> --
  <a href="docs/sdk.md">Native SDKs (14 languages)</a> --
  <a href="docs/python.md">Python SDK</a> --
  <a href="docs/transport.md">Transport Guide</a> --
  <a href="docs/detection.md">Detection</a> --
  <a href="docs/video-annotation.md">Video Pipeline</a> --
  <a href="docs/go-api-reference.md">Go API</a> --
  <a href="docs/security.md">Security</a> --
  <a href="docs/deployment.md">Deployment</a> --
  <a href="benchmarks/vs_python/results_full.md">Benchmarks</a>
</p>

---

## Why infergo

Production AI needs more than just an LLM. You need embeddings, vector search, detection, RAG, agents, content safety, observability -- and they all need to be fast, concurrent, and deployable as one unit.

The standard approach: 6+ Python services, 6 ports, 6 containers, 8 GB RAM, Python GIL killing concurrency.

infergo: **one binary, one port, one container.** Same LLM speed as llama.cpp with zero overhead. 252 tests passing across 16 Go packages.

```bash
# One command. Everything on one port.
infergo serve \
  --model llm:models/llama3-8b-q4.gguf \
  --model embed:models/all-MiniLM-L6-v2.onnx \
  --model detect:models/yolov8n.torchscript \
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

All numbers measured on RTX 5070 Ti, CUDA 12.8, Qwen 2.5 Coder 1.5B Q4_K_M, with `--batch-timeout-ms 5 --max-batch-size 8`. Benchmark scripts included. Performance scales with GPU — A100 achieves ~2x faster latency.

### LLM inference -- same speed as raw llama.cpp

| Engine | Avg latency | ms/tok | Go overhead |
|---|---|---|---|
| **Raw llama.cpp** (C++, no server) | 204ms | 3.46 | -- |
| **infergo** (Go + HTTP + llama.cpp) | 205ms | 3.48 | **0.7ms (0.3%)** |
| LM Studio | 176ms | 3.10 | custom CUDA fork |
| Python (llama-cpp-python) | 1,575ms | 29.0 | GIL + ctypes |

infergo runs at **100% of raw llama.cpp speed.** The Go HTTP layer adds 0.7ms -- within measurement noise.

### Concurrent throughput

| Concurrency | infergo | LM Studio | Python | infergo vs Python |
|---|---|---|---|---|
| c=1 | 205ms / 231 tok/s | 172ms / 248 tok/s | 1,607ms / 33 tok/s | **7x** |
| c=4 | 756ms / 289 tok/s | 491ms / 444 tok/s | 4,936ms / 41 tok/s | **7x** |
| c=8 | 1,356ms / 434 tok/s | 873ms / 460 tok/s | 8,146ms / 44 tok/s | **12x** |

infergo is **12x faster than Python** at c=8. Continuous batching groups concurrent requests into shared GPU decode calls.

### Multi-model capabilities

| Task | infergo | llama-server | Python |
|---|---|---|---|
| LLM generation | 3.5 ms/tok | 3.4 ms/tok | 29 ms/tok |
| Single embedding | 0.9 ms | not supported | 1.9 ms |
| Batch embedding (3) | 1.4 ms | not supported | 2.5 ms |
| Reranking (3 docs) | 1.2 ms | not supported | 6.2 ms |
| Object detection | 2.4 ms | not supported | 2.7 ms |
| BM25 search (10K docs) | 0.14 ms | not supported | ~1 ms |
| Vector search (k=10) | 0.03 ms | not supported | ~1 ms |
| Response cache hit | <0.1 ms | not supported | not supported |
| RAG pipeline (end-to-end) | 116 ms | not supported | 642 ms |
| Structured JSON output | 125 ms | partial | unreliable |

### Embedding + detection throughput

| Concurrency | infergo embedding | Python embedding |
|---|---|---|
| c=1 | 692 req/s | 437 req/s |
| c=8 | 951 req/s | 46 req/s |
| c=16 | 927 req/s | **crashed** |
| c=32 | 911 req/s | **crashed** |

### Binary size

| Component | Size |
|---|---|
| Go binary (stripped) | 16 MB |
| C++ engine (libinfer_api) | 74 MB |
| **Total** | **90 MB** |

---

## Architecture

```
Client (Python / TypeScript / Go / curl / any language)
  |
  |  POST /v1/chat/completions
  v
+----------------------------------------------------------+
|  Go HTTP layer (0.7ms overhead)                          |
|  JSON parse -> route -> 1 CGo call -> respond            |
|  Auth | Rate limit | Metrics | Cache | Queue | RBAC      |
+----------------------------+-----------------------------+
                             |  1 CGo call
+----------------------------v-----------------------------+
|  C++ inference engine                                    |
|  +------------+------------+----------+---------+------+ |
|  | llama.cpp  | ONNX RT /  | libtorch | HNSW   | BM25 | |
|  | LLM decode | TorchScript| nvJPEG   | search | text | |
|  | grammar    | embedding  | detect   | rerank | rank | |
|  | speculate  | batch      | NMS(CUDA)| vector |      | |
|  | KV cache   | pool+norm  | preproc  | DB     |      | |
|  +------------+------------+----------+---------+------+ |
+----------------------------+-----------------------------+
                             |
                    NVIDIA CUDA / CPU
```

**Go** handles HTTP, routing, metrics, caching, auth, rate limiting, and concurrency. 10,000 concurrent connections cost 80 KB of goroutine stacks. No GIL. Compiled binary.

**C++** handles all inference compute. The entire generation loop runs in C++ -- one CGo call per request, not one per token. Result: **0.7ms overhead** vs raw C++.

---

## Features

### LLM Inference

| Feature | Description |
|---|---|
| Chat completion | OpenAI-compatible `/v1/chat/completions` with streaming |
| Text completion | Raw text completion via `/v1/completions` |
| Streaming (SSE) | Server-sent events for token-by-token delivery |
| JSON mode | Guaranteed valid JSON output via constrained generation |
| GBNF grammar | Arbitrary grammar-constrained output (BNF format) |
| TOON format | Typed object output notation for structured extraction |
| Function calling | Tool-use with automatic argument extraction |
| Speculative decoding | Draft model acceleration (6.7x faster) |
| Prefix caching | KV cache reuse across requests with shared prefixes |
| Continuous batching | Dynamic request grouping for throughput maximization |
| Flash Attention | Memory-efficient attention (CUDA) |
| ChatML templates | Jinja2-compatible chat template rendering |
| LoRA adapter hot-swap | Load/unload LoRA adapters at runtime without restart |

### Embedding

| Feature | Description |
|---|---|
| ONNX models | ONNX Runtime inference for embedding models |
| TorchScript models | LibTorch inference for TorchScript embedding models |
| Batch embedding | Multiple inputs in a single request |
| Mean pooling + L2 norm | Standard pooling and normalization for similarity search |

### Object Detection

| Feature | Description |
|---|---|
| YOLO models | TorchScript and ONNX YOLO model support |
| GPU NMS | CUDA kernel for non-maximum suppression |
| Multi-stream batching | Concurrent detection across multiple video streams |
| Adaptive backend | Automatic selection of optimal runtime (ONNX/Torch) |
| nvJPEG decode | Hardware-accelerated JPEG decoding on GPU |
| Filtering | Configurable confidence, IoU, max detections, class filtering |

### Search and Retrieval

| Feature | Description |
|---|---|
| HNSW vector search | Approximate nearest neighbor search with CRUD and persistence |
| BM25 keyword search | Full-text search with 140us latency on 10K documents |
| Hybrid search | Reciprocal rank fusion combining vector + keyword results |
| Real-time index updates | Add, update, and delete documents without rebuilding indexes |
| Reranking | Cross-encoder reranking of search results |

### RAG Pipeline

| Feature | Description |
|---|---|
| End-to-end RAG | Embed query, search, augment prompt, generate -- one call |
| Streaming RAG | Token-by-token RAG response via SSE |
| Document ingestion | Ingest `.txt`, `.md`, `.csv`, `.html` with automatic chunking |
| Web scraping + crawling | Crawl and ingest web pages by URL |
| URL ingestion | Direct URL-to-vector-DB pipeline |

### AI Tasks

| Feature | Description |
|---|---|
| NER | Named entity recognition extraction |
| Sentiment analysis | Positive/negative/neutral classification with confidence |
| Text classification | Custom label classification with few-shot examples |
| Summarization | Abstractive text summarization |
| SQL agent | Natural language to SQL with database execution |
| Function calling | Tool-use with structured argument parsing |
| Agent framework | ReAct loop with built-in tools: calculator, search, code executor, current_time |

### Code Execution

| Feature | Description |
|---|---|
| Sandboxed execution | Run Python, JavaScript, Go, and Bash in isolation |
| Timeout limits | Configurable per-execution time limits |
| Output limits | Configurable stdout/stderr capture size |

### Production Infrastructure

| Feature | Description |
|---|---|
| Multi-model serving | Load multiple models (LLM, embed, detect) on one port |
| Hot reload | Swap model weights at runtime via admin API |
| API key auth | Bearer token authentication |
| Rate limiting | Per-key and global request rate limits |
| Request queue | Backpressure handling with configurable queue depth |
| RBAC | Role-based access control: admin, user, readonly |
| Multi-tenant isolation | Tenant-scoped data and model access |
| Circuit breaker | Automatic failure detection and recovery |
| IP allowlisting | Restrict access by source IP address |

### Caching

| Feature | Description |
|---|---|
| Response cache | LRU cache with `X-Cache` headers (<0.1ms hit latency) |
| Semantic cache | Embedding-similarity-based cache lookup |
| Prefix KV cache | Shared KV cache across requests with common prefixes |

### Content Safety

| Feature | Description |
|---|---|
| PII detection + redaction | Detect and redact email, phone, SSN, credit card, IP address |
| Content filtering | Classify and block violence, hate speech, self-harm |
| Guardrails | Configurable input/output safety policies |

### Observability

| Feature | Description |
|---|---|
| Prometheus metrics | 92 metrics exported at `/metrics` |
| Health checks | Liveness (`/health/live`) and readiness (`/health/ready`) probes |
| OpenTelemetry tracing | Distributed request tracing |
| Cost tracking | Per-request and per-model inference cost tracking |
| Drift detection | Monitor model output distribution changes |
| Quality monitoring | Automated output quality scoring |
| Confidence scoring | Per-response confidence estimation |
| Hallucination detection | Flag potentially hallucinated outputs |

### Deployment

| Feature | Description |
|---|---|
| Docker (CPU) | Minimal CPU image (0.18 GB) |
| Docker (CUDA) | GPU-accelerated image (1.52 GB) |
| Helm chart | Kubernetes deployment with configurable values |
| Canary deployments | Gradual traffic shifting to new model versions |
| A/B testing | Split traffic between model variants |
| Model registry | Versioned model storage with metadata |
| KEDA autoscaling | Event-driven horizontal pod autoscaling |
| Blue/green deploy | Zero-downtime deployment via hot reload |

### Developer Tools

| Feature | Description |
|---|---|
| OpenAPI 3.0 spec | Machine-readable API spec at `/v1/openapi.json` |
| Swagger UI | Interactive API documentation at `/ui/docs` |
| Playground | Interactive testing interface at `/ui/playground` |
| Health dashboard | System status dashboard at `/ui/dashboard` |
| CLI chat | Interactive chat via `infergo chat` |
| Benchmarking | Model performance testing via `infergo bench` |
| Model conversion | Format conversion via `infergo convert` |
| Model validation | Pre-flight model checks via `infergo validate` |
| Python SDK | `pip install infergo` -- chat, embed, detect, search, NER, sentiment, classify, summarize |
| TypeScript SDK | `npm install @infergo/client` -- streaming, fully typed |
| Native SDKs | 14 languages: C/C++, Python, Rust, Java, Node.js, C#, Swift, Ruby, PHP, Dart, Zig, Elixir, Lua, WASM |
| Transport layers | Shared memory (0.01ms), Unix socket (0.3ms), gRPC (1.5ms), HTTP (4ms) |

### Enterprise

| Feature | Description |
|---|---|
| Audit logging | JSONL + cryptographic hash chain for tamper-evident logs |
| GDPR data deletion | Per-user data purge via `/v1/admin/gdpr/{user_id}` |
| Data retention policies | Configurable automatic data expiry |
| Model cards | Structured model documentation and metadata |
| Webhooks | Event notifications with HMAC signature verification |

### Knowledge and Prompts

| Feature | Description |
|---|---|
| Knowledge graph extraction | Extract entities and relationships from text |
| Prompt versioning | Version control with rollback for prompt templates |
| Prompt optimization | Automated prompt tuning and improvement |
| Prompt library | Reusable prompt template storage |

### Advanced

| Feature | Description |
|---|---|
| MoE routing | Topic-based model selection across loaded models |
| Ensemble inference | Parallel multi-model inference with result aggregation |
| Context extension | YaRN/NTK RoPE scaling for extended context windows |

---

## Quickstart

### Install

```bash
# Download binary
curl -sSL https://github.com/ailakshya/infergo/releases/latest/download/infergo-linux-amd64-cuda.tar.gz \
  | tar xz && sudo mv infergo /usr/local/bin/
```

### Serve

```bash
# Single model
infergo serve --model models/llama3-8b-q4.gguf --provider cuda

# Multi-model (LLM + embedding + detection)
infergo serve \
  --model llm:models/llama3-8b-q4.gguf \
  --model embed:models/all-MiniLM-L6-v2.onnx \
  --model detect:models/yolov8n.torchscript \
  --provider cuda

# Speculative decoding
infergo serve --model llm:models/llama3-8b-q4.gguf \
  --draft-model models/llama3.2-1b-q4.gguf --provider cuda
```

### Chat

```bash
# HTTP
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llm","messages":[{"role":"user","content":"Hello"}]}'

# CLI
infergo chat --model llm
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

## API Reference

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat completion (streaming, JSON mode, function calling) |
| `POST` | `/v1/completions` | Text completion |
| `POST` | `/v1/embeddings` | Dense embeddings (single or batch) |
| `POST` | `/v1/search` | Vector similarity search (HNSW) + BM25 + hybrid |
| `POST` | `/v1/rerank` | Rerank documents by query relevance |
| `POST` | `/v1/rag` | Full RAG pipeline (embed + search + generate) |
| `POST` | `/v1/rag/stream` | Streaming RAG pipeline via SSE |
| `POST` | `/v1/ingest` | Ingest documents into vector DB |
| `POST` | `/v1/ingest/url` | Crawl and ingest web pages by URL |
| `POST` | `/v1/detect` | Object detection (JSON + base64 image) |
| `POST` | `/v1/detect/binary` | Object detection (raw JPEG body) |
| `POST` | `/v1/ner` | Named entity recognition |
| `POST` | `/v1/sentiment` | Sentiment analysis |
| `POST` | `/v1/classify` | Text classification |
| `POST` | `/v1/summarize` | Text summarization |
| `POST` | `/v1/code/execute` | Sandboxed code execution (Python/JS/Go/Bash) |
| `POST` | `/v1/agents/run` | Run ReAct agent with tools |
| `POST` | `/v1/agents/sql` | Natural language to SQL agent |
| `POST` | `/v1/knowledge/extract` | Knowledge graph extraction |
| `GET` | `/v1/knowledge/query` | Query knowledge graph |
| `POST` | `/v1/feedback` | Submit feedback for quality monitoring |
| `POST` | `/v1/batches` | Async batch inference |
| `GET` | `/v1/models` | List loaded models |
| `POST` | `/v1/admin/reload` | Hot-swap model weights |
| `POST` | `/v1/admin/tenants` | Manage tenants |
| `POST` | `/v1/admin/canary` | Configure canary deployments |
| `POST` | `/v1/admin/triggers` | Manage scheduled triggers |
| `POST` | `/v1/admin/optimize-prompt` | Optimize prompt templates |
| `DELETE` | `/v1/admin/gdpr/{user_id}` | Delete all data for a user (GDPR) |
| `GET` | `/v1/openapi.json` | OpenAPI 3.0 specification |
| `GET` | `/ui` | Built-in web interface |
| `GET` | `/ui/docs` | Swagger API documentation |
| `GET` | `/ui/playground` | Interactive API playground |
| `GET` | `/ui/dashboard` | Health and metrics dashboard |
| `GET` | `/health/live` | Liveness probe |
| `GET` | `/health/ready` | Readiness probe |
| `GET` | `/metrics` | Prometheus metrics (92 metrics) |

### Request examples

**Chat completion with JSON mode:**

```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "llm",
    "messages": [{"role": "user", "content": "List 3 capitals as JSON"}],
    "response_format": {"type": "json_object"},
    "stream": true
  }'
```

**Embedding:**

```bash
curl http://localhost:9090/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "embed", "input": ["hello world", "goodbye world"]}'
```

**RAG pipeline:**

```bash
curl http://localhost:9090/v1/rag \
  -H "Content-Type: application/json" \
  -d '{"model": "llm", "query": "What is infergo?", "collection": "docs", "top_k": 5}'
```

**Object detection:**

```bash
curl http://localhost:9090/v1/detect/binary \
  -H "Content-Type: image/jpeg" \
  --data-binary @photo.jpg
```

**Named entity recognition:**

```bash
curl http://localhost:9090/v1/ner \
  -H "Content-Type: application/json" \
  -d '{"model": "llm", "text": "John Smith works at Google in Mountain View."}'
```

**Agent with tools:**

```bash
curl http://localhost:9090/v1/agents/run \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llm",
    "prompt": "What is 42 * 17 and what time is it?",
    "tools": ["calculator", "current_time"],
    "max_steps": 5
  }'
```

**Code execution:**

```bash
curl http://localhost:9090/v1/code/execute \
  -H "Content-Type: application/json" \
  -d '{"language": "python", "code": "print(sum(range(100)))", "timeout": 5}'
```

---

## CLI

| Command | Description |
|---|---|
| `infergo serve` | Start the inference server |
| `infergo chat` | Interactive CLI chat session |
| `infergo bench` | Run model performance benchmarks |
| `infergo convert` | Convert between model formats |
| `infergo validate` | Validate a model file before serving |
| `infergo pull` | Download models from registry |
| `infergo models` | List available models |
| `infergo detect` | Run detection from command line |

```bash
# Interactive chat
infergo chat --model models/llama3-8b-q4.gguf --provider cuda

# Benchmark a model
infergo bench --model models/llama3-8b-q4.gguf --provider cuda \
  --prompt "Explain quantum computing" --concurrency 1,4,8

# Validate before serving
infergo validate --model models/llama3-8b-q4.gguf

# Convert model format
infergo convert --input model.onnx --output model.torchscript
```

---

## SDKs

infergo provides **two types** of SDK:

1. **HTTP SDKs** — connect to a running `infergo serve` instance over the network (any machine)
2. **Native SDKs** — link directly to `libinfer_api.so` for zero-overhead inference (same machine as GPU)

### Native SDKs (14 languages, zero HTTP overhead)

Link directly to the C inference engine. No server, no serialization, no network. 0.01ms overhead.

| Language | Directory | Binding | Install |
|---|---|---|---|
| **C / C++** | `sdk/c/` | Direct link | `gcc app.c -linfer_api` or CMake `find_package(infergo)` |
| **Python** | `sdk/python-native/` | ctypes | `export INFERGO_LIB_DIR=...` |
| **Rust** | `sdk/rust/` | FFI (infergo-sys) | `cargo add infergo` |
| **Java / Kotlin** | `sdk/java/` | JNI | `gradle build` |
| **Node.js** | `sdk/nodejs/` | N-API addon | `npm install` (builds native) |
| **C# / .NET** | `sdk/dotnet/` | P/Invoke | `dotnet build` |
| **Swift** | `sdk/swift/` | C bridging | Swift Package Manager |
| **Ruby** | `sdk/ruby/` | FFI gem | `gem install ffi` |
| **PHP** | `sdk/php/` | FFI extension | `composer install` (PHP 7.4+) |
| **Dart / Flutter** | `sdk/dart/` | dart:ffi | Add to `pubspec.yaml` |
| **Zig** | `sdk/zig/` | @cImport | `zig build` |
| **Elixir / Erlang** | `sdk/elixir/` | NIF | `mix compile` |
| **Lua** | `sdk/lua/` | LuaJIT FFI | `require("infergo")` |
| **WASM / JS** | `sdk/wasm/` | Emscripten | `make` (CPU only) |

Every native SDK wraps the same C API (`libinfer_api.so`) and provides: **LLM generation, embedding, vector search, BM25 search, LoRA adapters, and RAG pipeline.**

```python
# Python — zero HTTP, direct C call
from infergo_native import LLM, VectorDB

with LLM("model.gguf") as llm:
    print(llm.generate("Hello!"))  # 0.01ms overhead vs raw C++
```

```rust
// Rust — safe RAII bindings
let llm = infergo::Llm::new("model.gguf", -1, 4096, 1, 2048)?;
let result = llm.generate("Hello!", 128, 0.7, 0.9, None)?;
```

```cpp
// C++ — header-only RAII
infergo::LLM llm("model.gguf");
std::cout << llm.generate("Hello!") << std::endl;
```

See **[Native SDK Reference](docs/sdk.md)** for full API docs, install guides, and examples for all 14 languages.

### HTTP SDKs (any machine)

Connect to `infergo serve` over the network. Works from any machine.

#### Python

```bash
pip install infergo
```

```python
from infergo import InfergoClient
client = InfergoClient("http://localhost:9090", api_key="YOUR_KEY")
response = client.chat("What is Go?", model="llm")
vectors = client.embed(["hello"], model="embed")
entities = client.ner("John works at Google.", model="llm")
```

#### TypeScript

```bash
npm install @infergo/client
```

```typescript
import { InfergoClient } from '@infergo/client';
const client = new InfergoClient('http://localhost:9090');
const response = await client.chat({ model: 'llm', messages: [{ role: 'user', content: 'Hello' }] });
for await (const chunk of client.chatStream({ model: 'llm', messages: [...] })) {
  process.stdout.write(chunk.content);
}
```

#### Go

```go
import "github.com/ailakshya/infergo/go/client"
c := client.New("http://localhost:9090", client.WithAPIKey("key"))
resp, _ := c.Chat(ctx, client.ChatRequest{Model: "llm", Messages: msgs})
vec, _ := c.Embed(ctx, client.EmbedRequest{Model: "embed", Input: "hello"})
```

### Transport Selection

| Where is your app? | Use | Overhead |
|---|---|---|
| Same process (embedded) | Native SDK (direct link) | 0.001ms |
| Same machine (client/server) | [Shared memory or Unix socket](docs/transport.md) | 0.01–0.3ms |
| Different machine (remote GPU) | HTTP or gRPC | 1.5–4ms |

See **[Transport Guide](docs/transport.md)** for architecture decision flowchart.

---

## Build from source

```bash
git clone https://github.com/ailakshya/infergo && cd infergo

# CPU
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target infer_api -j$(nproc)
go build -C go -o ../infergo ./cmd/infergo

# CUDA (with Flash Attention)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DGGML_CUDA_FA=ON
cmake --build build --target infer_api -j$(nproc)
go build -C go -o ../infergo ./cmd/infergo

# CUDA + TorchScript backend
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DGGML_CUDA_FA=ON \
  -DTorch_DIR=/path/to/libtorch/share/cmake/Torch
cmake --build build --target infer_api -j$(nproc)
go build -C go -o ../infergo ./cmd/infergo
```

### Run tests

```bash
# All Go tests (252 tests across 16 packages)
cd go && go test ./...

# Specific package
cd go && go test ./pkg/search/...
cd go && go test ./pkg/rag/...
```

---

## Deployment

### Kubernetes with Helm

```bash
helm install infergo deploy/helm/infergo \
  --set image.tag=cuda \
  --set model.path=/models/llama3-8b-q4.gguf \
  --set provider=cuda \
  --set replicas=2
```

### KEDA autoscaling

```yaml
# Scale based on request queue depth
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: infergo
spec:
  scaleTargetRef:
    name: infergo
  minReplicaCount: 1
  maxReplicaCount: 10
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus:9090
        metricName: infergo_request_queue_depth
        threshold: "10"
```

### Canary deployment

```bash
# Route 10% of traffic to new model version
curl http://localhost:9090/v1/admin/canary \
  -H "Authorization: Bearer ADMIN_KEY" \
  -d '{"model": "llm", "canary_weight": 0.1, "canary_model": "models/llama3-8b-q4-v2.gguf"}'
```

---

## Configuration

infergo uses command-line flags and environment variables:

| Flag | Env | Default | Description |
|---|---|---|---|
| `--model` | `INFERGO_MODEL` | -- | Model path(s), format: `alias:path` |
| `--provider` | `INFERGO_PROVIDER` | `cpu` | Compute provider: `cpu`, `cuda` |
| `--port` | `INFERGO_PORT` | `9090` | HTTP listen port |
| `--api-key` | `INFERGO_API_KEY` | -- | API key for authentication |
| `--draft-model` | `INFERGO_DRAFT_MODEL` | -- | Draft model for speculative decoding |
| `--backend` | `INFERGO_BACKEND` | `auto` | Backend: `auto`, `onnx`, `tensorrt`, `torch` |
| `--ctx-size` | `INFERGO_CTX_SIZE` | `4096` | Context window size |
| `--batch-size` | `INFERGO_BATCH_SIZE` | `512` | Batch size for prompt processing |
| `--threads` | `INFERGO_THREADS` | auto | CPU thread count |
| `--gpu-layers` | `INFERGO_GPU_LAYERS` | `999` | Layers to offload to GPU |
| `--flash-attn` | `INFERGO_FLASH_ATTN` | `true` | Enable Flash Attention (CUDA) |

---

## License

[Apache 2.0](LICENSE)
