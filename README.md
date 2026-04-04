# infergo

<p align="center">
  <strong>Production-grade AI inference for Go.</strong><br>
  One binary. No Python. No GIL.
</p>

<p align="center">
  <a href="https://github.com/ailakshya/infergo/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  </a>
  <a href="https://pkg.go.dev/github.com/ailakshya/infergo">
    <img src="https://pkg.go.dev/badge/github.com/ailakshya/infergo.svg" alt="Go Reference">
  </a>
  <a href="https://golang.org">
    <img src="https://img.shields.io/badge/Go-1.23+-00ADD8.svg" alt="Go version">
  </a>
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900.svg" alt="CUDA 12">
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey.svg" alt="Platform">
</p>

<p align="center">
  <a href="docs/getting-started.md">Getting Started</a> ·
  <a href="docs/go-api-reference.md">Go API</a> ·
  <a href="docs/deployment.md">Deployment</a> ·
  <a href="benchmarks/vs_python/results_full.md">Benchmarks</a>
</p>

---

infergo is a Go-native inference runtime for LLMs, embedding models, and object detection. It wraps llama.cpp and ONNX Runtime behind a clean Go API and an OpenAI-compatible HTTP server — with no Python dependency at any layer.

Go services that need model inference today must call a Python sidecar, pay for a hosted API, or write their own CGo bindings. infergo eliminates all three options. One `go get`, one binary, one process.

---

## Contents

- [Features](#features)
- [Quickstart](#quickstart)
- [Benchmarks](#benchmarks)
- [API reference](#api-reference)
- [Go library](#go-library)
- [Architecture](#architecture)
- [Supported models](#supported-models)
- [Multi-GPU](#multi-gpu)
- [Kubernetes](#kubernetes)
- [Testing](#testing)
- [Contributing](#contributing)

---

## Features

- **LLM inference** — GGUF models (LLaMA 3, Mistral, Phi-3.5, Gemma 2) with OpenAI-compatible API
- **Continuous batching** — scheduler batches all waiting sequences into every `BatchDecode` call; P50 stays flat under concurrent load, GPU utilization ≥ 85%
- **PagedAttention KV cache** — 2–3× more concurrent sequences per GPU; pages freed per-sequence with no leaks
- **Embedding models** — ONNX Runtime inference for nomic-embed-text, bge-m3, all-MiniLM via `/v1/embeddings`
- **Object detection** — YOLOv8 bounding-box output via `/v1/detect`; letterbox + normalize preprocessing
- **Multi-model serving** — LLM + embedding + detection on one port via repeatable `--model` flags
- **gRPC API** — parallel to HTTP; `--grpc-port` enables on startup
- **WebSocket streaming** — `ws://host/v1/chat/completions/ws` for browser-native token streaming
- **Go SDK** — `client.Chat()`, `client.ChatStream()`, `client.Embed()`, `client.Detect()` — on [pkg.go.dev](https://pkg.go.dev/github.com/ailakshya/infergo)
- **Hot model reload** — `POST /v1/admin/reload` swaps weights without restart
- **Model download** — `infergo pull <hf-repo>` fetches GGUF/ONNX from HuggingFace
- **OpenTelemetry tracing** — `--otlp-endpoint` ships spans to Jaeger/Tempo/Honeycomb
- **API key auth** — `--api-key` or `INFERGO_API_KEY` env var; Bearer token validation
- **Rate limiting** — `--rate-limit` caps requests/second per client IP (token bucket)
- **Request queue** — `--max-queue` / `--max-active` with 503 on overflow; queue depth in Prometheus
- **Prometheus metrics** — `infergo_requests_total`, `infergo_tokens_total`, `infergo_active_sequences`, queue depth
- **KEDA autoscaling** — custom metrics adapter exposes queue depth to Kubernetes KEDA
- **Kubernetes health probes** — `/health/live` and `/health/ready` with configurable readiness threshold
- **GC tuning** — `GOGC=50` + `--gc-interval` keeps RSS flat; drift measured at +0.3% after 1,000 requests
- **Multi-GPU** — tensor split (`--tensor-split 0.5,0.5`) and pipeline stages (`--pipeline-stages 4`)
- **TensorRT backend** — 1.5–2× faster ONNX detection on NVIDIA GPUs
- **CoreML backend** — Apple Silicon ONNX model acceleration
- **Graceful shutdown** — in-flight requests complete before exit; SIGTERM-safe
- **Docker** — CPU image **0.18 GB**, CUDA image ~2.5 GB; weights via volume mount

---

## Quickstart

### Install (pre-built binary)

Download the binary for your platform — no compiler, no CMake, no Python required.

```bash
# Linux x86-64 (CPU)
curl -sSL https://github.com/ailakshya/infergo/releases/latest/download/infergo-linux-amd64-cpu.tar.gz \
  | tar xz && sudo mv infergo /usr/local/bin/

# Linux x86-64 (CUDA)
curl -sSL https://github.com/ailakshya/infergo/releases/latest/download/infergo-linux-amd64-cuda.tar.gz \
  | tar xz && sudo mv infergo /usr/local/bin/

# macOS Apple Silicon
curl -sSL https://github.com/ailakshya/infergo/releases/latest/download/infergo-darwin-arm64.tar.gz \
  | tar xz && sudo mv infergo /usr/local/bin/
```

The only runtime requirement for CUDA builds is an NVIDIA driver (≥ 525) — CUDA Toolkit does **not** need to be installed.

### Download a model

```bash
infergo pull bartowski/Meta-Llama-3-8B-Instruct-GGUF --filename Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

Or directly:

```bash
mkdir -p models
wget -O models/llama3-8b-q4.gguf \
  "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
```

### Serve

```bash
# CPU
./infergo serve --model models/llama3-8b-q4.gguf --port 9090

# CUDA — offload all layers to GPU
./infergo serve \
  --model models/llama3-8b-q4.gguf \
  --provider cuda \
  --gpu-layers 999 \
  --port 9090

# Multi-model (LLM + embedding + detection on one port)
./infergo serve \
  --model llama3:models/llama3-8b-q4.gguf \
  --model embed:models/all-MiniLM-L6-v2.onnx \
  --model detector:models/yolov8n.onnx \
  --provider cuda
```

### Query

```bash
# Chat
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3","messages":[{"role":"user","content":"Explain KV caching."}],"max_tokens":200}'

# Streaming
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3","messages":[{"role":"user","content":"Count to 5"}],"stream":true}'

# Embeddings
curl http://localhost:9090/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"embed","input":"hello world"}'

# Detection
curl http://localhost:9090/v1/detect \
  -H "Content-Type: application/json" \
  -d '{"model":"detector","image_b64":"<base64-encoded-jpeg>"}'
```

### Docker

```bash
# CPU (0.18 GB image)
docker build -f Dockerfile.cpu -t infergo:cpu .
docker run --rm -p 9090:9090 -v ./models:/models:ro infergo:cpu \
  --model /models/llama3-8b-q4.gguf

# CUDA  (requires nvidia-container-toolkit)
docker build -f Dockerfile.cuda -t infergo:cuda .
docker run --rm --gpus all -p 9090:9090 -v ./models:/models:ro infergo:cuda \
  --model /models/llama3-8b-q4.gguf --provider cuda --gpu-layers 999
```

---

## Benchmarks

Measured on RTX 5070 Ti (16 GB VRAM, SM 12.0 Blackwell), LLaMA 3 8B Q4\_K\_M, Ubuntu 22.04, CUDA 12.8.
Full methodology, raw numbers, and plots: [`benchmarks/vs_python/results_full.md`](benchmarks/vs_python/results_full.md)

### CUDA throughput and latency vs llama-cpp-python

| Scenario | infergo | llama-cpp-python | Delta |
|---|---|---|---|
| Short prompts — tok/s | **142** | 133 | **+7%** |
| Short prompts — req/s | **2.7** | 2.1 | **+29%** |
| Long prompts — tok/s | **144** | 131 | **+10%** |
| Long prompts — req/s | **0.6** | 0.5 | **+20%** |
| Cold start (CUDA) | **456 ms** | 494 ms | −8% |

### Concurrent load (continuous batching)

| Concurrency | P50 latency | Throughput |
|---|---|---|
| c=1 | ~300 ms | 2.7 req/s |
| c=4 | **1,063 ms** | 3.8 req/s |
| c=32 | — | **17.4 req/s** |

### Memory stability (1,000 requests)

| Metric | Before (mutex) | After (PagedAttention + GOGC=50) |
|---|---|---|
| RSS drift | +11.9% | **+0.3%** |
| Base RSS | 1,168 MB | 1,168 MB |

### CPU throughput vs llama-cpp-python

| Scenario | infergo | llama-cpp-python | Delta |
|---|---|---|---|
| Short prompts — tok/s | **10** | 5 | **+100%** |
| Cold start (CPU) | **6.45 s** | 9.90 s | −35% |

### Container footprint

| | infergo | Typical Python stack |
|---|---|---|
| Runtime binary | 22 MB | ~10 GB (Python + PyTorch + deps) |
| Docker image (CPU) | **0.18 GB** | ~10 GB |
| Docker image (CUDA) | ~2.5 GB | ~12 GB |

---

## API reference

### CLI

```
infergo serve         Start the inference server
infergo list-models   Query a running server for loaded models
infergo benchmark     Run a concurrent load test against a running server
infergo pull          Download a model from HuggingFace
```

```
serve flags:
  --model            Path/spec for a model; repeatable. Format: [name:]path.{gguf,onnx}
  --provider         cpu | cuda | tensorrt | coreml          (default: cpu)
  --port             HTTP listen port                         (default: 9090)
  --grpc-port        gRPC listen port; 0 = disabled          (default: 9091)
  --gpu-layers       Transformer layers to offload to GPU    (default: 999 = all)
  --ctx-size         Total KV cache tokens                   (default: 16384)
  --threads          CPU inference threads                   (default: NumCPU/2)
  --max-seqs         Max concurrent sequences / KV slots     (default: 16)
  --max-batch-size   Max sequences per BatchDecode call      (default: 0 = unlimited)
  --batch-timeout-ms ms to wait for more reqs before batch fires (default: 0)
  --gc-interval      Call runtime.GC() every N completions   (default: 100)
  --min-models       Models required before /health/ready    (default: 1)
  --api-key          Bearer token for auth (or INFERGO_API_KEY env)
  --rate-limit       Max requests/second per client IP       (default: 0 = unlimited)
  --max-queue        Max in-flight requests; 503 beyond      (default: 100)
  --max-active       Max concurrent request handlers         (default: same as --max-queue)
  --otlp-endpoint    OTLP HTTP endpoint for tracing          (e.g. localhost:4318)
  --tensor-split     Comma-separated GPU fractions for tensor parallelism (e.g. 0.5,0.5)
  --pipeline-stages  GPU pipeline stages (1 = single GPU)    (default: 1)
  --mode             Server role: combined | prefill | decode (default: combined)
```

### HTTP endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat completion — OpenAI-compatible, `"stream": true` supported |
| `POST` | `/v1/completions` | Text completion |
| `POST` | `/v1/embeddings` | Dense embeddings from ONNX embedding models |
| `POST` | `/v1/detect` | Object detection — YOLOv8 bounding boxes |
| `GET` | `/v1/models` | List all loaded models |
| `POST` | `/v1/admin/reload` | Hot-swap a model without restart |
| `GET` | `/health/live` | Liveness probe — 200 if process is running |
| `GET` | `/health/ready` | Readiness probe — 200 when ≥ `--min-models` loaded |
| `GET` | `/metrics` | Prometheus metrics |
| `GET/ws` | `/v1/chat/completions/ws` | WebSocket token streaming |

---

## Go library

The client SDK is pure Go — no CGo, no native libraries, no special setup:

```bash
go get github.com/ailakshya/infergo@v0.1.0
```

```go
import "github.com/ailakshya/infergo/client"

c := client.New("http://localhost:9090",
    client.WithAPIKey("my-key"),
    client.WithTimeout(30*time.Second),
)

// Blocking chat
resp, err := c.Chat(ctx, client.ChatRequest{
    Model:    "llama3",
    Messages: []client.Message{{Role: "user", Content: "Hello"}},
})
fmt.Println(resp.Content)

// Streaming
tokenCh, errCh := c.ChatStream(ctx, client.ChatRequest{
    Model:    "llama3",
    Messages: []client.Message{{Role: "user", Content: "Count to 5"}},
})
for tok := range tokenCh {
    fmt.Print(tok)
}
if err := <-errCh; err != nil {
    log.Fatal(err)
}

// Embeddings
vec, err := c.Embed(ctx, client.EmbedRequest{Model: "embed", Input: "hello world"})

// Detection
result, err := c.Detect(ctx, client.DetectRequest{Model: "detector", ImageB64: b64})
for _, obj := range result.Objects {
    fmt.Printf("class=%d conf=%.2f box=(%v,%v,%v,%v)\n",
        obj.ClassID, obj.Confidence, obj.X1, obj.Y1, obj.X2, obj.Y2)
}
```

For embedding infergo in a Go service (server-side), see [`docs/go-api-reference.md`](docs/go-api-reference.md).

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Clients — OpenAI SDK / curl / LangChain / gRPC         │
└───────────────────────────┬─────────────────────────────┘
                            │  HTTP (OpenAI-compatible) + gRPC + WebSocket
┌───────────────────────────▼─────────────────────────────┐
│  infergo server  (Go)                                   │
│  Continuous batching scheduler · PagedAttention KV      │
│  Prometheus · OTel tracing · auth · rate-limit · queue  │
└───────────────────────────┬─────────────────────────────┘
                            │  CGo
┌───────────────────────────▼─────────────────────────────┐
│  C API boundary  — infer_api.h                          │
│  Only C types cross this boundary                       │
└──────┬──────────────┬──────────────┬────────────────────┘
       │              │              │
┌──────▼──────┐ ┌─────▼──────┐ ┌────▼───────────────────┐
│ libinfer_   │ │ libinfer_  │ │ libinfer_tokenizer      │
│ llm         │ │ onnx       │ │ (HuggingFace / Rust FFI)│
│ llama.cpp   │ │ ONNX       │ └────────────────────────┘
│ KV cache    │ │ Runtime    │ ┌────────────────────────┐
│ sampler     │ │ (CPU/TRT)  │ │ libinfer_preprocess    │
└──────┬──────┘ └─────┬──────┘ │ image · letterbox ·   │
       │              │        │ normalize              │
       └──────┬───────┘        └────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────┐
│  Hardware: NVIDIA CUDA · TensorRT · CPU (x86/ARM) · Metal│
└─────────────────────────────────────────────────────────┘
```

The C API boundary is a hard rule — no C++ types cross it. This keeps CGo correct, lets each layer be tested independently, and means the compute engine can be replaced without touching Go code.

---

## Supported models

| Type | Format | Tested models |
|---|---|---|
| LLM | GGUF | LLaMA 3 8B, Mistral 7B, Phi-3.5 Mini, Gemma 2 9B, Qwen 2 |
| Embedding | ONNX | nomic-embed-text-v1.5, bge-m3, all-MiniLM-L6-v2 |
| Detection | ONNX | YOLOv8n / YOLOv8s / YOLOv8m, YOLOv9, RT-DETR |

Any GGUF model compatible with llama.cpp will work. ONNX models are auto-detected by the presence of `tokenizer.json` (embedding) or the model filename (detection).

---

## Multi-GPU

```bash
# Tensor split — distribute weights across 2 GPUs (e.g. 70B models)
./infergo serve \
  --model models/llama-70b-q4.gguf \
  --provider cuda \
  --tensor-split 0.5,0.5

# Pipeline stages — 4-GPU pipeline parallelism over PCIe
./infergo serve \
  --model models/llama-70b-q4.gguf \
  --provider cuda \
  --pipeline-stages 4

# Prefill/decode separation (prefill node)
./infergo serve --model models/llama3-8b-q4.gguf --mode prefill --port 9090

# Prefill/decode separation (decode node)
./infergo serve --model models/llama3-8b-q4.gguf --mode decode --port 9091
```

---

## Kubernetes

A Helm chart is included in `deploy/helm/infergo/`. KEDA autoscaling uses the `infergo_queue_depth` metric to scale the deployment.

```bash
# Deploy to Kubernetes
helm install infergo deploy/helm/infergo/ \
  --set image.tag=cpu \
  --set model.path=/models/llama3-8b-q4.gguf

# KEDA ScaledObject is deployed automatically when keda.enabled=true
helm install infergo deploy/helm/infergo/ \
  --set keda.enabled=true \
  --set keda.minReplicas=1 \
  --set keda.maxReplicas=10
```

See [`docs/deployment.md`](docs/deployment.md) for full Kubernetes, systemd, and bare-metal deployment guides.

---

## Build from source

Building from source is only needed to contribute to infergo or to add a new backend. Most users should use the [pre-built binary](#install-pre-built-binary).

```bash
# Requirements: CMake 3.20+, GCC 12+, Go 1.23+, Rust (for tokenizer)
git clone https://github.com/ailakshya/infergo && cd infergo

# CPU build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target infer_api -j$(nproc)
go build -C go -o ../infergo ./cmd/infergo

# CUDA build (requires CUDA Toolkit 12.x)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DINFER_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --target infer_api -j$(nproc)
go build -C go -o ../infergo ./cmd/infergo
```

CUDA arch values: `80` = A100, `89` = RTX 4090, `120` = RTX 5000.

---

## Testing

```bash
# C++ unit tests (276 cases: KV cache, sampler, sequence, API, ONNX, tokenizer)
ctest --test-dir build --output-on-failure

# Address sanitizer — zero leaks, zero errors (81 tests, CPU targets)
cmake -S . -B build-asan \
  -DINFER_CUDA=OFF -DCMAKE_BUILD_TYPE=Debug -DASAN=ON
cmake --build build-asan -j$(nproc)
ctest --test-dir build-asan

# Go tests (pure-Go packages)
cd go && go test -race ./...

# Benchmark against Python (requires infergo server running on :9090)
source ~/llama_venv/bin/activate
python benchmarks/vs_python/bench_full.py \
  --model-path models/llama3-8b-q4.gguf \
  --infergo-addr http://localhost:9090 \
  --device cuda
```

CI runs on every push and PR: Go tests (race detector), Helm lint, and C++ unit tests (CPU, no GPU required).

---

## Documentation

| | |
|---|---|
| [Getting started](docs/getting-started.md) | Build, download a model, first request in 5 minutes |
| [Deployment](docs/deployment.md) | Docker, Kubernetes, systemd, bare metal, nginx |
| [Go API reference](docs/go-api-reference.md) | Embed infergo in your Go application |
| [C API reference](docs/c-api-reference.md) | Call the C layer from any language with FFI |
| [Contributing](docs/contributing.md) | Add a new backend, model type, or execution provider |

---

## Contributing

Issues and pull requests are welcome. For significant changes, open an issue first to align on approach.

Before submitting:

```bash
ctest --test-dir build --output-on-failure
cd go && go test -race ./...
```

Please keep commits clean — author only, no AI attribution, no generated footers.

---

## License

[Apache 2.0](LICENSE)
