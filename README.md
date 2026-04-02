# infergo

**A production-grade, open source AI inference runtime for Go.**

Train in any framework. Serve in Go. Deploy anywhere.

```bash
infergo serve --model llama3-8b-q4.gguf --provider cuda --port 9090
```

```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3-8b-q4","messages":[{"role":"user","content":"Hello!"}]}'
```

---

## Why infergo?

Every serious inference stack today is Python-first. You get vLLM, FastAPI, and a 500 MB container just to serve a model. infergo is different:

- **Zero Python.** Single Go binary + one `.so`. Ships in a 100 MB Docker image (CPU) or 300 MB (CUDA), model weights not included.
- **OpenAI-compatible API.** Drop-in replacement — use any OpenAI SDK, LangChain, or CLI tool unchanged.
- **Multi-backend.** GGUF models via llama.cpp, ONNX models via ONNXRuntime. CUDA, CPU, CoreML, TensorRT.
- **Production-ready.** Prometheus metrics, Kubernetes health checks, SSE streaming, hot model reload, batch scheduling — all built in.

| | infergo | vLLM | FastAPI + ONNX |
|---|---|---|---|
| Language | Go | Python | Python |
| Binary size | ~12 MB | ~2 GB env | ~500 MB env |
| Cold start | <1s | 10–30s | 3–10s |
| Concurrency | goroutines | asyncio | threads |
| Streaming | SSE native | SSE | manual |
| K8s health probes | built-in | manual | manual |
| Prometheus metrics | built-in | manual | manual |
| No Python required | ✓ | ✗ | ✗ |

---

## Quickstart

### 1. Build

**Requirements:** CMake 3.20+, GCC 12+/Clang 15+, Go 1.23+, [vcpkg](https://github.com/microsoft/vcpkg)

```bash
git clone https://github.com/ailakshya/infergo
cd infergo

# Bootstrap vcpkg (first time only)
git clone https://github.com/microsoft/vcpkg ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh -disableMetrics

# Build C++ library (CPU)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --target infer_api -j$(nproc)

# Build CLI
cd go
CGO_CFLAGS="-I../cpp/include" \
CGO_LDFLAGS="-L../build/cpp/api -linfer_api -Wl,-rpath,$(pwd)/../build/cpp/api" \
go build -o ../infergo ./cmd/infergo
cd ..
```

For CUDA add `-DCMAKE_CUDA_ARCHITECTURES=89` (adjust for your GPU).

### 2. Download a model

```bash
mkdir models
# LLaMA 3 8B Q4_K_M (~4.7 GB)
wget -O models/llama3-8b-q4.gguf \
  https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

### 3. Serve

```bash
# CPU
./infergo serve --model models/llama3-8b-q4.gguf --port 9090

# GPU (CUDA — offload all layers)
./infergo serve --model models/llama3-8b-q4.gguf --provider cuda --gpu-layers 999 --port 9090
```

### 4. Query

```bash
# Chat
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3-8b-q4","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":64}'

# Streaming
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3-8b-q4","messages":[{"role":"user","content":"Count to 5"}],"stream":true}'

# List models
./infergo list-models
```

### 5. Docker

```bash
# CPU
docker build -f Dockerfile.cpu -t infergo:cpu .
docker run --rm -p 9090:9090 -v ./models:/models:ro infergo:cpu

# CUDA (requires nvidia-container-toolkit)
docker build -f Dockerfile.cuda -t infergo:cuda .
docker run --rm --gpus all -p 9090:9090 -v ./models:/models:ro infergo:cuda
```

---

## API

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | LLM chat — supports `"stream": true` for SSE |
| `POST /v1/completions` | LLM text completion |
| `POST /v1/embeddings` | Image / text embedding vectors |
| `POST /v1/detect` | Object detection (YOLO-style) |
| `GET /v1/models` | List loaded models |
| `GET /healthz` | Kubernetes liveness probe |
| `GET /readyz` | Kubernetes readiness probe |
| `GET /metrics` | Prometheus metrics |

---

## Supported models

| Type | Format | Examples |
|---|---|---|
| LLM | GGUF | LLaMA 3, Mistral, Qwen 2, Phi-3, Gemma 2 |
| Embedding | ONNX | ResNet-50, ViT, CLIP |
| Detection | ONNX | YOLOv8, YOLOv9, RT-DETR |

---

## Use as a Go library

```go
import (
    "github.com/ailakshya/infergo/server"
    "github.com/ailakshya/infergo/llm"
)

m, _ := llm.Load("models/llama3-8b-q4.gguf", 999, 4096, 4, 512)
defer m.Close()

reg := server.NewRegistry()
reg.Load("llama3", &myAdapter{m})

srv := server.NewServer(reg)
http.ListenAndServe(":9090", srv)
```

---

## Architecture

```
Go application
     │  import "github.com/ailakshya/infergo/llm"
     ▼
  Go wrapper packages    (CGo — zero-copy tensors)
     │  #include "infer_api.h"
     ▼
  C API boundary         (extern "C" — no C++ types cross here)
     ▼
  C++ compute engine
  ├── libinfer_tensor      tensor memory (CPU + CUDA)
  ├── libinfer_onnx        ONNX Runtime sessions
  ├── libinfer_llm         llama.cpp + KV cache + sampler
  ├── libinfer_tokenizer   HuggingFace tokenizers (Rust FFI)
  ├── libinfer_preprocess  image decode / resize / normalize
  └── libinfer_postprocess NMS / softmax / embedding normalize
```

---

## What's built

| Capability | Status |
|---|---|
| CPU + CUDA tensor allocation and transfer | ✅ |
| ONNX inference (CPU / CUDA / TensorRT / CoreML) | ✅ |
| LLM inference via llama.cpp (continuous batching) | ✅ |
| HuggingFace tokenizer (BPE, WordPiece) | ✅ |
| Image preprocessing (decode, letterbox, normalize) | ✅ |
| NMS postprocessing (YOLO format) | ✅ |
| OpenAI-compatible HTTP API | ✅ |
| SSE token streaming | ✅ |
| Prometheus metrics + Kubernetes health checks | ✅ |
| Hot model reload | ✅ |
| Batch scheduler | ✅ |
| CLI (serve / list-models / benchmark) | ✅ |
| Docker (CPU + CUDA) | ✅ |

---

## Benchmarks

*Run on your hardware with the included scripts:*

```bash
# vs vLLM
python benchmarks/vs_python/bench_llm.py --compare \
  --infergo-addr http://localhost:9090 --infergo-model llama3 \
  --vllm-addr http://localhost:8000 --vllm-model meta-llama/Meta-Llama-3-8B-Instruct

# vs FastAPI + ONNX
python benchmarks/vs_python/bench_onnx.py --compare \
  --infergo-addr http://localhost:9090 --infergo-model resnet50 \
  --fastapi-addr http://localhost:8001 --fastapi-model resnet50
```

---

## Documentation

- [Getting started](docs/getting-started.md) — build, download a model, serve in 5 minutes
- [Deployment](docs/deployment.md) — Docker, Kubernetes, bare metal, systemd, nginx
- [Go API reference](docs/go-api-reference.md) — embed infergo in your Go application
- [C API reference](docs/c-api-reference.md) — call the C layer directly from any FFI
- [Contributing](docs/contributing.md) — add a new execution provider or model type

---

## License

Apache 2.0
