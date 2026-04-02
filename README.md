# infergo

<p align="center">
  <strong>Production-grade AI inference for Go.</strong><br>
  One binary. No Python. No GIL.
</p>

<p align="center">
  <a href="https://github.com/ailakshya/infergo/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
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
  <a href="optimization_tasks.md">Roadmap</a> ·
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
- [Roadmap](#roadmap)
- [Testing](#testing)
- [Contributing](#contributing)

---

## Features

**Today**

- LLM inference — GGUF models (LLaMA 3, Mistral, Phi-3.5, Gemma 2) with OpenAI-compatible API
- CUDA and CPU — same binary, same API, full GPU offload or CPU-only
- SSE token streaming — `"stream": true` in the chat completions request
- Prometheus metrics — `infergo_requests_total`, `infergo_tokens_total` built in
- Kubernetes health probes — `/health/live` and `/health/ready` with configurable readiness threshold
- Graceful shutdown — in-flight requests complete before the process exits
- Docker — CPU and CUDA multi-stage images; model weights via volume mount
- ~22 MB binary — vs 10+ GB for a comparable Python inference stack

**In progress** — see [Roadmap](#roadmap)

- Continuous batching scheduler (flat latency under concurrent load)
- ONNX Runtime inference (embedding models, YOLO detection)
- Multi-model serving, gRPC API, Go SDK

---

## Quickstart

### Requirements

- CMake 3.20+, GCC 12+ or Clang 15+, Go 1.23+
- For CUDA: CUDA Toolkit 12.x, compatible GPU driver

### Build

```bash
git clone https://github.com/ailakshya/infergo && cd infergo

# CPU build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target infer_api -j$(nproc)
go build -C go -o ../infergo ./cmd/infergo
```

For CUDA, pass `-DINFER_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89` to cmake (adjust the arch for your GPU — e.g. `80` for A100, `89` for RTX 4090, `120` for RTX 5000).

### Download a model

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
```

### Query

```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-8b-q4",
    "messages": [{"role": "user", "content": "Explain KV caching in one paragraph."}],
    "max_tokens": 200
  }'
```

Streaming:

```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3-8b-q4","messages":[{"role":"user","content":"Count to 5"}],"stream":true}'
```

### Docker

```bash
# CPU
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

### CPU throughput vs llama-cpp-python

| Scenario | infergo | llama-cpp-python | Delta |
|---|---|---|---|
| Short prompts — tok/s | **10** | 5 | **+100%** |
| Cold start (CPU) | **6.45 s** | 9.90 s | −35% |

> **Note:** infergo CUDA P50 latency under concurrency=4 is higher than Python because the current implementation serializes requests with a mutex (llama.cpp is not thread-safe). This is the primary target of [OPT-2](#roadmap). At concurrency=1, infergo matches Python P50.

### Container footprint

| | infergo | Typical Python stack |
|---|---|---|
| Runtime binary | 22 MB | ~10 GB (Python + PyTorch + deps) |
| Docker image (CPU) | ~1.8 GB | ~10 GB |
| Docker image (CUDA) | ~2.5 GB | ~12 GB |

---

## API reference

### CLI

```
infergo serve         Start the inference server
infergo list-models   Query a running server for loaded models
infergo benchmark     Run a concurrent load test against a running server
```

```
serve flags:
  --model        Path to model file (.gguf)
  --provider     cpu | cuda | tensorrt | coreml       (default: cpu)
  --port         HTTP listen port                     (default: 9090)
  --gpu-layers   Transformer layers to offload to GPU (default: 999 = all)
  --ctx-size     KV cache token budget per sequence   (default: 4096)
  --threads      CPU inference threads                (default: NumCPU/2)
  --min-models   Models required before /health/ready (default: 1)
```

### HTTP endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat completion — OpenAI-compatible, `"stream": true` supported |
| `POST` | `/v1/completions` | Text completion |
| `GET` | `/v1/models` | List all loaded models |
| `GET` | `/health/live` | Liveness probe — returns 200 if process is running |
| `GET` | `/health/ready` | Readiness probe — returns 200 when ≥ `--min-models` are loaded |
| `GET` | `/metrics` | Prometheus metrics |

---

## Go library

```go
import (
    "github.com/ailakshya/infergo/llm"
    "github.com/ailakshya/infergo/server"
)

// Load a GGUF model
m, err := llm.Load(
    "models/llama3-8b-q4.gguf",
    999,   // GPU layers (999 = all)
    4096,  // KV cache size (tokens)
    8,     // CPU threads
    512,   // max batch size
)
if err != nil {
    log.Fatal(err)
}
defer m.Close()

// Register and serve
reg := server.NewRegistry()
reg.Load("llama3", &myAdapter{m: m})

mux := http.NewServeMux()
apiSrv := server.NewServer(reg)
mux.Handle("/v1/", apiSrv)

server.NewHealthChecker(reg, 1).RegisterRoutes(mux)
mux.Handle("/metrics", server.NewMetrics().Handler())

http.ListenAndServe(":9090", mux)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Clients — OpenAI SDK / curl / LangChain / gRPC         │
└───────────────────────────┬─────────────────────────────┘
                            │  HTTP (OpenAI-compatible)
┌───────────────────────────▼─────────────────────────────┐
│  infergo server  (Go)                                   │
│  net/http · goroutines · Prometheus · health checks     │
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
│ sampler     │ │            │ │ libinfer_preprocess    │
└──────┬──────┘ └─────┬──────┘ │ image · letterbox ·   │
       │              │        │ normalize              │
       └──────┬───────┘        └────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────┐
│  Hardware: NVIDIA CUDA · CPU (x86/ARM) · Apple Metal    │
└─────────────────────────────────────────────────────────┘
```

The C API boundary is a hard rule — no C++ types cross it. This keeps CGo correct, lets each layer be tested independently, and means the compute engine can be replaced without touching Go code.

---

## Supported models

| Type | Format | Tested models |
|---|---|---|
| LLM | GGUF | LLaMA 3 8B, Mistral 7B, Phi-3.5 Mini, Gemma 2 9B, Qwen 2 |
| Embedding | ONNX | nomic-embed-text-v1.5, bge-m3, all-MiniLM-L6-v2 *(OPT-4)* |
| Detection | ONNX | YOLOv8n / YOLOv8s / YOLOv8m, YOLOv9, RT-DETR *(OPT-5)* |

Any GGUF model compatible with llama.cpp will work. ONNX model inference is in active development (OPT-3).

---

## Roadmap

Full task list with test cases and pass criteria: [`optimization_tasks.md`](optimization_tasks.md)
Progress against 10 target problems: [`problemsolve.md`](problemsolve.md)

### Active work

| Task | What | Why it matters |
|---|---|---|
| **OPT-1** | OpenBLAS for CPU prefill | CPU long-prompt tok/s currently below Python due to `GGML_BLAS=OFF` |
| **OPT-2** | Continuous batching scheduler | Removes the request mutex; P50 stays flat under concurrency; GPU util ≥ 85% |

OPT-2 is the most impactful near-term change. llama.cpp is not thread-safe, so today all requests serialize behind a mutex. The fix is a scheduler goroutine that owns the LLM, batches all waiting sequences into each `BatchDecode` call, and routes tokens back to callers via per-request channels. Go-layer only — no C++ changes.

### Phase B — Inference expansion

| Task | Adds |
|---|---|
| OPT-3 | ONNX Runtime session: create, run, free |
| OPT-4 | `/v1/embeddings` — nomic-embed-text, bge-m3, all-MiniLM |
| OPT-5 | `/v1/detect` — YOLOv8 bounding box output |
| OPT-6 | Image preprocessing — letterbox, normalize, CHW layout |
| OPT-7 | BERT / RoBERTa tokenizer for embedding models |

### Phase C — Production serving

| Task | Adds |
|---|---|
| OPT-8 | Multi-model: LLM + embedding + detection on one port |
| OPT-9 | Hot model reload without restart |
| OPT-10 | Request queue, priority scheduling, 503 on overflow |
| OPT-11 | API key authentication |
| OPT-12 | Per-key rate limiting |
| OPT-13 | gRPC API |
| OPT-14 | WebSocket streaming |

### Phase D — Ecosystem

| Task | Adds |
|---|---|
| OPT-15 | Go SDK — `client.Chat()`, `client.Embed()`, `client.Detect()` |
| OPT-16 | `infergo pull <hf-repo>` — HuggingFace model download |
| OPT-18 | OpenTelemetry distributed tracing |
| OPT-19 | TensorRT backend — 1.5–2× faster detection |
| OPT-20 | CoreML backend — Apple Silicon ONNX models |
| OPT-21 | KEDA autoscaling metrics |

### Phase E — Scalability and multi-GPU

| Task | Adds |
|---|---|
| OPT-22 | PagedAttention KV cache — 2–3× more concurrent sequences per GPU |
| OPT-23 | Tensor parallelism — `--tensor-split 0.5,0.5` for 70B+ models, no Ray |
| OPT-24 | Pipeline parallelism — multi-GPU over PCIe without NVLink |
| OPT-25 | Horizontal scaling + Helm chart + KEDA autoscale |
| OPT-26 | Prefill/decode node separation — data-center scale architecture |
| OPT-27 | Scalability benchmark — measured RSS and throughput vs Python at c=1..32 |

### Platform support

| Platform | LLM | ONNX | Status |
|---|---|---|---|
| Linux x86-64 + CUDA | ✅ | ✅ | Primary target, fully tested |
| Linux x86-64 CPU | ✅ | ✅ | Fully supported |
| macOS ARM64 (Apple Silicon) | ✅ Metal | planned (OPT-20) | LLM works via llama.cpp Metal |
| Windows x86-64 | planned | planned | See note below |

**Windows:** Pure Go binaries cross-compile with `GOOS=windows go build`. infergo cannot, because CGo requires the target platform's C++ compiler and pre-built native libraries. Windows support needs a CMake Windows configuration, MSVC or MinGW-w64 toolchain, and `infer_api.dll` instead of `.so`. This is tracked as OPT-28. Linux is the priority because GPU inference production workloads run there; Windows is meaningful for local developer tooling.

---

## Testing

```bash
# C++ unit tests (276 cases, including KV cache, sampler, sequence, API)
ctest --test-dir build --output-on-failure

# Address sanitizer — zero leaks, zero errors (81 tests, CPU targets)
cmake -S . -B build-asan \
  -DINFER_CUDA=OFF -DCMAKE_BUILD_TYPE=Debug -DASAN=ON
cmake --build build-asan -j$(nproc)
ctest --test-dir build-asan

# Go tests
cd go && go test ./...

# Go race detector
cd go && go test -race ./...

# Benchmark against Python (requires infergo server running on :9090)
source ~/llama_venv/bin/activate
python benchmarks/vs_python/bench_full.py \
  --model-path models/llama3-8b-q4.gguf \
  --infergo-addr http://localhost:9090 \
  --device cuda
```

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
