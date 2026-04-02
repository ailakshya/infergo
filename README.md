# infergo

<p align="center">
  <strong>Production-grade AI inference for Go — no Python required.</strong><br/>
  Serve LLMs, embedding models, and object detection from a single Go binary.
</p>

<p align="center">
  <a href="https://github.com/ailakshya/infergo/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://golang.org"><img src="https://img.shields.io/badge/Go-1.23+-00ADD8.svg" alt="Go version"></a>
  <a href="https://github.com/ailakshya/infergo/actions"><img src="https://img.shields.io/badge/build-passing-brightgreen.svg" alt="Build"></a>
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey.svg" alt="Platform">
</p>

---

## The problem

Every serious inference stack is Python-first. You get vLLM, FastAPI, PyTorch, and a 10 GB container image — plus Python's GIL, which forces you to run one model copy per worker process to handle concurrent users.

Go services that need inference today must call a Python sidecar, pay for a hosted API, or write CGo bindings from scratch. None of those are good answers.

infergo is a Go-native inference runtime that eliminates all three. One binary. One model copy in memory. Unlimited concurrent goroutines.

---

# ── WHAT EXISTS TODAY ──────────────────────────────────────────

---

## What infergo does today

- **LLM inference** — serve any GGUF model (LLaMA 3, Mistral, Phi-3, Gemma 2) via OpenAI-compatible API
- **CUDA + CPU** — full GPU offload or CPU-only, same binary, same API
- **OpenAI-compatible** — drop-in for any OpenAI SDK, LangChain, or `curl` command
- **Prometheus metrics** — `infergo_requests_total`, `infergo_tokens_total` out of the box
- **Kubernetes health probes** — `/health/live` and `/health/ready` built in
- **Graceful shutdown** — in-flight requests complete before the process exits
- **CLI** — `infergo serve`, `infergo list-models`, `infergo benchmark`
- **Docker** — CPU and CUDA multi-stage images, model weights via volume mount
- **Small footprint** — ~22 MB Go binary, 1.8 GB Docker image (vs 10+ GB Python stack)

---

## Quickstart

### Build

```bash
git clone https://github.com/ailakshya/infergo
cd infergo

# Build C++ engine (CPU)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target infer_api -j$(nproc)

# Build CLI
go build -C go -o ../infergo ./cmd/infergo
```

For CUDA, add `-DINFER_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89` (adjust arch for your GPU).

### Download a model

```bash
mkdir -p models
wget -O models/llama3-8b-q4.gguf \
  https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

### Serve

```bash
# CPU
./infergo serve --model models/llama3-8b-q4.gguf --port 9090

# GPU — full CUDA offload
./infergo serve --model models/llama3-8b-q4.gguf --provider cuda --gpu-layers 999 --port 9090
```

### Query

```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-8b-q4",
    "messages": [{"role": "user", "content": "What is Go?"}],
    "max_tokens": 128
  }'
```

### Docker

```bash
# CPU
docker build -f Dockerfile.cpu -t infergo:cpu .
docker run --rm -p 9090:9090 -v ./models:/models:ro infergo:cpu \
  --model /models/llama3-8b-q4.gguf

# CUDA (requires nvidia-container-toolkit)
docker build -f Dockerfile.cuda -t infergo:cuda .
docker run --rm --gpus all -p 9090:9090 -v ./models:/models:ro infergo:cuda \
  --model /models/llama3-8b-q4.gguf --provider cuda --gpu-layers 999
```

---

## Benchmarks

Measured on RTX 5070 Ti (16 GB, CUDA SM 12.0), LLaMA 3 8B Q4\_K\_M.
Full methodology and raw data: [`benchmarks/vs_python/results_full.md`](benchmarks/vs_python/results_full.md)

### CUDA — infergo vs llama-cpp-python

| Scenario | infergo | llama-cpp-python | Advantage |
|---|---|---|---|
| Short prompts — tok/s | **142** | 133 | +7% |
| Short prompts — req/s | **2.7** | 2.1 | +29% |
| Long prompts — tok/s | **144** | 131 | +10% |
| Long prompts — req/s | **0.6** | 0.5 | +20% |
| Cold start | **456 ms** | 494 ms | −8% |

### CPU — infergo vs llama-cpp-python

| Scenario | infergo | llama-cpp-python | Advantage |
|---|---|---|---|
| Short prompts — tok/s | **10** | 5 | **+100%** |
| Cold start | **6.45 s** | 9.90 s | −35% |

### Container footprint

| | infergo | Python stack |
|---|---|---|
| Binary / runtime | 22 MB | ~10 GB |
| Docker image (CPU) | ~1.8 GB | ~10 GB |
| Docker image (CUDA) | ~2.5 GB | ~12 GB |

---

## CLI reference

```
infergo serve         Start the inference server
infergo list-models   Query a running server for loaded models
infergo benchmark     Run a load test against a running server
```

```
infergo serve flags:
  --model        path to model file (.gguf)
  --provider     cpu | cuda | tensorrt | coreml  (default: cpu)
  --port         HTTP listen port                (default: 9090)
  --gpu-layers   transformer layers on GPU       (default: 999 = all)
  --ctx-size     KV cache token budget           (default: 4096)
  --threads      CPU inference threads           (default: NumCPU/2)
  --min-models   models required for /health/ready (default: 1)
```

---

## HTTP API (today)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/v1/chat/completions` | LLM chat — OpenAI-compatible, supports `"stream": true` |
| POST | `/v1/completions` | LLM text completion |
| GET | `/v1/models` | List loaded models |
| GET | `/health/live` | Kubernetes liveness probe |
| GET | `/health/ready` | Kubernetes readiness probe |
| GET | `/metrics` | Prometheus metrics |

---

## Use as a Go library

```go
import (
    "github.com/ailakshya/infergo/llm"
    "github.com/ailakshya/infergo/server"
)

m, err := llm.Load("models/llama3-8b-q4.gguf",
    999,   // gpu layers
    4096,  // context size
    8,     // cpu threads
    512,   // batch size
)
if err != nil { log.Fatal(err) }
defer m.Close()

reg := server.NewRegistry()
reg.Load("llama3", adapter)

mux := http.NewServeMux()
mux.Handle("/v1/", server.NewServer(reg))
http.ListenAndServe(":9090", mux)
```

---

## Architecture

```
HTTP clients (OpenAI SDK / curl / LangChain)
        │
        ▼
   infergo server  (Go — net/http, goroutines, Prometheus)
        │
        ▼
   C API boundary  (infer_api.h — only C types cross here)
        │
        ▼
   C++ compute engine
   ├── libinfer_llm         llama.cpp — LLM inference, KV cache, sampler
   ├── libinfer_onnx        ONNX Runtime — embedding + detection models
   ├── libinfer_tokenizer   HuggingFace tokenizers (Rust FFI)
   └── libinfer_preprocess  image decode / resize / normalize
        │
        ▼
   Hardware: CUDA / CPU / CoreML / TensorRT
```

The C API boundary is a strict rule — no C++ types cross it. This keeps CGo safe, makes every layer independently testable, and lets the compute engine be replaced without touching Go code.

---

## Supported models (today)

| Type | Format | Models |
|---|---|---|
| LLM | GGUF | LLaMA 3, Mistral, Phi-3.5, Gemma 2, Qwen 2, any llama.cpp-compatible |

---

## Testing

```bash
# C++ unit tests (276 cases)
ctest --test-dir build --output-on-failure

# Address sanitizer
cmake -S . -B build-asan -DINFER_CUDA=OFF -DCMAKE_BUILD_TYPE=Debug -DASAN=ON
cmake --build build-asan -j$(nproc)
ctest --test-dir build-asan

# Go tests + race detector
cd go && go test -race ./...

# Benchmark vs Python
source ~/llama_venv/bin/activate
python benchmarks/vs_python/bench_full.py \
  --model-path models/llama3-8b-q4.gguf \
  --infergo-addr http://localhost:9090 \
  --device cuda
```

---

# ── WHAT IS BEING BUILT ────────────────────────────────────────

---

## Active optimization work

Full task list with test cases and pass criteria: [`optimization_tasks.md`](optimization_tasks.md)

### Phase A — Performance (in progress)

These two tasks are the most impactful and are being worked on now.

| Task | Problem being solved | Target metric |
|---|---|---|
| **OPT-1** OpenBLAS CPU prefill | CPU long-prompt is slower than Python because BLAS is disabled (`GGML_BLAS=OFF`). Python's llama-cpp-python links OpenBLAS and wins the prefill phase. | CPU long tok/s ≥ Python (11 tok/s) |
| **OPT-2** Continuous batching | Today one mutex serializes all requests. At concurrency=4, P50 latency is 3× single-client. GPU utilization is ~25%. | P50 ≤ 600 ms at c=4; GPU util ≥ 85% |

**OPT-2 in detail:** llama.cpp is not thread-safe — concurrent `llama_decode` calls crash. The fix is a scheduler goroutine that owns the LLM exclusively, batches all waiting sequences into every `BatchDecode` call, and routes tokens back to each handler via a per-request channel. P50 latency stays flat as concurrency grows. No C++ changes needed — Go layer only.

### Phase B — Core inference expansion

| Task | What it adds |
|---|---|
| **OPT-3** ONNX Runtime inference | Run any ONNX model — session create, input tensors, output tensors, memory cleanup |
| **OPT-4** `/v1/embeddings` | Embedding API — nomic-embed-text, bge-m3, all-MiniLM-L6-v2 |
| **OPT-5** `/v1/detect` | Detection API — YOLOv8n/s/m with bounding box JSON output |
| **OPT-6** Image preprocessing | Letterbox resize, normalize, CHW layout for YOLO input |
| **OPT-7** BERT tokenizer | WordPiece tokenizer with `[CLS]`/`[SEP]`/padding for embedding models |

### Phase C — Production serving

| Task | What it adds |
|---|---|
| **OPT-8** Multi-model serving | `--model llm:llama3.gguf --model embed:nomic.onnx` — all types on one port |
| **OPT-9** Hot model reload | `POST /v1/admin/reload` — swap weights without restart |
| **OPT-10** Request queue | Bounded queue + priority (`X-Priority` header) + 503 on overflow |
| **OPT-11** API key auth | `Authorization: Bearer <key>` middleware |
| **OPT-12** Rate limiting | Per-key token bucket, 429 + `Retry-After` |
| **OPT-13** gRPC API | `service Infergo` proto — lower latency than HTTP+JSON for service-to-service |
| **OPT-14** WebSocket streaming | `ws://host/v1/ws/chat` — alternative to SSE |

### Phase D — Ecosystem

| Task | What it adds |
|---|---|
| **OPT-15** Go SDK | `go get .../client` — typed `client.Chat()`, `client.Embed()`, `client.Detect()` |
| **OPT-16** `infergo pull` | Download GGUF/ONNX from HuggingFace Hub with resume + SHA256 |
| **OPT-17** Multi-model LLM bench | Benchmark across Llama 3 + Phi-3.5 + Gemma 2 |
| **OPT-18** OpenTelemetry | Distributed traces: parse → queue → decode → respond |
| **OPT-19** TensorRT backend | `--provider tensorrt` — 1.5–2× faster detection vs CUDA ONNX Runtime |
| **OPT-20** CoreML backend | `--provider coreml` — Apple Silicon ONNX models |
| **OPT-21** KEDA metrics | `infergo_queue_depth`, `infergo_gpu_utilization_percent` for autoscaling |

---

# ── FUTURE TARGETS & HYPOTHESES ───────────────────────────────

---

## Scalability roadmap (Phase E)

These are the targets that take infergo from single-GPU serving to data-center scale.
Each is a significant architectural change; implementation begins after Phase A–D is complete.

### OPT-22 — PagedAttention KV cache

**Hypothesis:** Current fixed-slot KV allocation wastes VRAM when sequences are short.
Block-level allocation (16 tokens per page, on demand) should fit 2–3× more concurrent
sequences into the same VRAM, directly multiplying throughput without new hardware.

**Target:** Same VRAM holds 2–3× active sequences vs fixed-slot baseline.

### OPT-23 — Tensor parallelism (multi-GPU)

**Hypothesis:** llama.cpp's built-in `tensor_split` parameter can shard weight matrices
across N GPUs with a single flag, no Ray, no NCCL tuning. For models larger than one
GPU's VRAM (70B+ at Q4 = ~40 GB), this is the only way to serve them on consumer hardware.

**Target:** `--tensor-split 0.5,0.5` serves Llama-3-70B on 2× 24 GB GPUs.
2-GPU tok/s ≥ 1.6× single GPU.

### OPT-24 — Pipeline parallelism

**Hypothesis:** Tensor parallelism requires all-reduce across GPUs on every layer —
high NVLink bandwidth needed. Pipeline parallelism only passes activations between
stages, making it viable over PCIe. GPU 0 runs layers 0–15, GPU 1 runs 16–31;
infergo micro-batches to keep both busy.

**Target:** Correct output on 2× PCIe GPUs (no NVLink); throughput ≥ 0.9× single GPU.

### OPT-25 — Horizontal scaling + Helm chart

**Hypothesis:** infergo is stateless per-request (no sticky sessions). Multiple pods
behind a standard Kubernetes load balancer should scale linearly. KEDA watching
`infergo_queue_depth` should auto-scale pods within 30 seconds of a traffic spike.

**Target:** 3 pods × 142 tok/s ≥ 400 tok/s aggregate. Zero request loss during scale-up.

### OPT-26 — Prefill/decode node separation

**Hypothesis:** Prefill (processing the prompt) is compute-bound (GEMM); decode
(generating tokens) is memory-bandwidth-bound (GEMV). Running both on the same GPU
means each phase underutilises the hardware. Separating them — dedicated prefill nodes
feed KV cache to dedicated decode nodes — should fully saturate both GPU types.
This is the architecture Mooncake and Splitwise have proven at scale.

**Target:** Prefill node processes 200 prompts/s. Decode node runs 8 concurrent
sequences with flat latency. End-to-end TTFT ≤ prefill-only + transfer + 20 ms.

---

## Long-term vision

| Milestone | infergo is comparable to |
|---|---|
| Phase A + B done | vLLM + ONNX Runtime server + sentence-transformers — in one Go binary |
| Phase C + D done | NVIDIA Triton Inference Server — multi-model, auth, tracing, gRPC |
| Phase E done | Full Python ML serving stack (vLLM + Ray Serve + Triton) — no Python, no GIL |

---

## Platform support

| Platform | Status | Notes |
|---|---|---|
| Linux x86-64 + CUDA | **Supported** | Primary development target |
| Linux x86-64 CPU | **Supported** | Full feature parity with CUDA build |
| macOS ARM64 (Apple Silicon) | **Supported (LLM)** | Metal via llama.cpp; ONNX CoreML — OPT-20 |
| Windows x86-64 | **Planned** | See below |

### Windows support

Pure Go binaries cross-compile trivially with `GOOS=windows go build`. infergo cannot,
because it uses CGo — Go's foreign function interface to C++. CGo requires the *target
platform's* C++ compiler and pre-built native libraries. To build for Windows, you need:

- MSVC or MinGW-w64 toolchain on a Windows machine
- CMake Windows configuration (different flags, different CUDA paths)
- `infer_api.dll` instead of `libinfer_api.so`
- Windows CUDA runtime paths (`%CUDA_PATH%` vs `/usr/local/cuda`)

This is straightforward engineering — llama.cpp builds on Windows today — but requires
explicit work and a Windows CI runner. Tracked as **OPT-28** on the roadmap.
GPU inference production workloads almost exclusively run on Linux, so Linux is the
priority. Windows support is meaningful for local developer tooling and desktop inference.

---

## Progress tracking

| Document | What it contains |
|---|---|
| [`optimization_tasks.md`](optimization_tasks.md) | All 27 optimization tasks with test cases and pass criteria |
| [`problemsolve.md`](problemsolve.md) | 10 target problems, solution map, 52-item progress checklist |

---

## Documentation

| Doc | Description |
|---|---|
| [Getting started](docs/getting-started.md) | Build, download a model, first request in 5 minutes |
| [Deployment](docs/deployment.md) | Docker, Kubernetes, systemd, bare metal, nginx |
| [Go API reference](docs/go-api-reference.md) | Embed infergo in your Go application |
| [C API reference](docs/c-api-reference.md) | Call the C layer from any FFI language |
| [Contributing](docs/contributing.md) | Add a new backend, model type, or execution provider |

---

## Contributing

Pull requests are welcome. For large changes, open an issue first to discuss the approach.

```bash
# Run the full test suite before submitting
ctest --test-dir build --output-on-failure
cd go && go test -race ./...
```

Please keep commits clean — no AI attribution, no `Co-Authored-By` lines.

---

## License

[Apache 2.0](LICENSE)
