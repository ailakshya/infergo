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

Every serious inference stack is Python-first. You get vLLM, FastAPI, PyTorch, and a 10 GB container image — plus Python's GIL, which forces you to run one model copy per worker:

```
10 concurrent users × 4.6 GB model = 46 GB just for weights
```

Go services that need inference today must call a Python sidecar, pay for a hosted API, or write CGo bindings from scratch. **None of those are good answers.**

infergo is a Go-native inference runtime that eliminates all three. One binary. One model copy in memory. Unlimited concurrent goroutines.

---

## What infergo does today

- **LLM inference** — serve any GGUF model (LLaMA 3, Mistral, Phi-3, Gemma 2) with an OpenAI-compatible API
- **CUDA + CPU** — full GPU offload or CPU-only, same API either way
- **OpenAI-compatible** — drop-in for any OpenAI SDK, LangChain, or `curl` command
- **Production-ready** — Prometheus metrics, Kubernetes health probes, graceful shutdown
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

| Scenario | infergo | llama-cpp-python | infergo advantage |
|---|---|---|---|
| Short prompts — tok/s | **142** | 133 | +7% |
| Short prompts — req/s | **2.7** | 2.1 | +29% |
| Long prompts — tok/s | **144** | 131 | +10% |
| Long prompts — req/s | **0.6** | 0.5 | +20% |
| Cold start | **456 ms** | 494 ms | −8% |

### CPU — infergo vs llama-cpp-python

| Scenario | infergo | llama-cpp-python | infergo advantage |
|---|---|---|---|
| Short prompts — tok/s | **10** | 5 | **+100%** |
| Cold start | **6.45 s** | 9.90 s | −35% |

### Container size

| | infergo | Python stack |
|---|---|---|
| Binary / runtime | 22 MB | ~10 GB |
| Docker image (CPU) | ~1.8 GB | ~10 GB |
| Docker image (CUDA) | ~2.5 GB | ~12 GB |

---

## CLI reference

```
infergo serve       Start the inference server
infergo list-models Query a running server for loaded models
infergo benchmark   Run a load test against a running server
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

## HTTP API

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

// Load model
m, err := llm.Load("models/llama3-8b-q4.gguf",
    999,   // gpu layers
    4096,  // context size
    8,     // cpu threads
    512,   // batch size
)
if err != nil { log.Fatal(err) }
defer m.Close()

// Register and serve
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
   C API boundary  (infer_api.h — no C++ types cross here)
        │
        ▼
   C++ compute engine
   ├── libinfer_llm        llama.cpp — LLM inference, KV cache, sampler
   ├── libinfer_onnx       ONNX Runtime — embedding + detection models
   ├── libinfer_tokenizer  HuggingFace tokenizers (Rust FFI)
   └── libinfer_preprocess image decode / resize / normalize
        │
        ▼
   Hardware: CUDA / CPU / CoreML / TensorRT
```

The C API boundary is a hard rule: only C types cross it. This keeps CGo safe, lets
the C++ layer be replaced without touching Go code, and makes every layer independently
testable.

---

## Supported models

| Type | Format | Models |
|---|---|---|
| LLM | GGUF | LLaMA 3, Mistral, Phi-3.5, Gemma 2, Qwen 2, any llama.cpp-compatible model |
| Embedding | ONNX | nomic-embed-text, bge-m3, all-MiniLM-L6-v2 *(coming — OPT-4)* |
| Detection | ONNX | YOLOv8n/s/m, YOLOv9, RT-DETR *(coming — OPT-5)* |

---

## Optimization plan

Active work tracked in [`optimization_tasks.md`](optimization_tasks.md) with test cases for every item.

### Phase A — Performance (next up)

| Task | What | Target |
|---|---|---|
| **OPT-1** | OpenBLAS for CPU prefill | CPU long-prompt tok/s ≥ Python baseline |
| **OPT-2** | Continuous batching scheduler | CUDA P50 ≤ 600 ms at concurrency=4 (was 1423 ms); GPU util ≥ 85% |

**OPT-2** is the single most impactful change. Today the server processes one request at a time behind a mutex because llama.cpp is not thread-safe. A scheduler goroutine will own the LLM exclusively, batch all waiting sequences into every `BatchDecode` call, and route tokens back to each HTTP handler — P50 latency stays flat as concurrency grows.

### Phase B — Core inference expansion

| Task | What | Unlocks |
|---|---|---|
| **OPT-3** | ONNX Runtime inference engine | Embedding + detection models |
| **OPT-4** | `/v1/embeddings` endpoint | RAG, semantic search, vector DBs |
| **OPT-5** | `/v1/detect` endpoint | YOLO object detection |
| **OPT-6** | Image preprocessing pipeline | Correct YOLO input (letterbox, normalize) |
| **OPT-7** | BERT/RoBERTa tokenizer | Correct embedding output |

### Phase C — Production serving

| Task | What |
|---|---|
| **OPT-8** | Multi-model: LLM + embedding + detection on one port |
| **OPT-9** | Hot model reload without restart |
| **OPT-10** | Request queue + priority scheduling + 503 on overflow |
| **OPT-11** | API key authentication |
| **OPT-12** | Per-key rate limiting |
| **OPT-13** | gRPC API alongside HTTP |
| **OPT-14** | WebSocket streaming |

### Phase D — Ecosystem

| Task | What |
|---|---|
| **OPT-15** | Go SDK: `client.Chat()`, `client.Embed()`, `client.Detect()` |
| **OPT-16** | `infergo pull <hf-repo>` — HuggingFace model download |
| **OPT-17** | Multi-model LLM benchmark (Llama 3 + Phi-3.5 + Gemma 2) |
| **OPT-18** | OpenTelemetry distributed tracing |
| **OPT-19** | TensorRT backend (1.5–2× faster detection) |
| **OPT-20** | CoreML backend (Apple Silicon) |
| **OPT-21** | KEDA autoscaling metrics |

### Phase E — Scalability & multi-GPU

| Task | What | Solves |
|---|---|---|
| **OPT-22** | PagedAttention KV cache | 2–3× more concurrent users per GPU |
| **OPT-23** | Tensor parallelism (`--tensor-split`) | 70B+ models across 2+ GPUs, no Ray |
| **OPT-24** | Pipeline parallelism | Multi-GPU over PCIe, no NVLink required |
| **OPT-25** | Horizontal scaling + Helm chart | 1000+ req/s, KEDA auto-scale pods |
| **OPT-26** | Prefill/decode node separation | Data-center scale, max GPU utilization |
| **OPT-27** | Scalability benchmark vs Python | Proves GIL wall at concurrency=1..32 |

---

## Future vision

infergo's goal is to be the Go equivalent of the entire Python inference stack — no Python required at any layer.

**After Phase A+B:** comparable to vLLM + ONNX Runtime server + sentence-transformers, in a single Go binary.

**After Phase C+D:** comparable to NVIDIA Triton Inference Server — multi-model, multi-backend, production auth, tracing, gRPC.

**After Phase E:** data-center-scale inference. Multi-GPU without Ray. Disaggregated prefill/decode. Kubernetes-native autoscaling. The architecture that frontier companies are building in Python today — in Go, with none of the GIL overhead.

**Platform support:**

| Platform | LLM (GGUF) | ONNX | Status |
|---|---|---|---|
| Linux x86-64 + CUDA | ✅ | ✅ | Fully supported |
| Linux x86-64 CPU | ✅ | ✅ | Fully supported |
| macOS ARM64 (Apple Silicon) | ✅ Metal | ✅ CoreML (OPT-20) | LLM working; ONNX coming |
| Windows | planned | planned | Needs CGo + DLL build (OPT-28) |

> **Why not Windows yet?** infergo uses CGo — Go's foreign function interface to C++. Pure Go binaries cross-compile trivially (`GOOS=windows go build`), but CGo requires the *target platform's* C++ compiler and built libraries. Windows support needs a separate CMake configuration, MSVC/MinGW toolchain, and `infer_api.dll` build. It is on the roadmap.

Full progress tracking: [`problemsolve.md`](problemsolve.md) — 10 target problems, 52 checkpoints, current status.

---

## Testing

```bash
# C++ unit tests (276 cases)
ctest --test-dir build --output-on-failure

# Address sanitizer (memory safety)
cmake -S . -B build-asan -DINFER_CUDA=OFF -DCMAKE_BUILD_TYPE=Debug -DASAN=ON
cmake --build build-asan -j$(nproc)
ctest --test-dir build-asan

# Go tests
cd go && go test ./...

# Race detector
cd go && go test -race ./...

# Benchmark vs Python
source ~/llama_venv/bin/activate
python benchmarks/vs_python/bench_full.py \
  --model-path models/llama3-8b-q4.gguf \
  --infergo-addr http://localhost:9090 \
  --device cuda
```

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
