# infergo

<p align="center">
  <strong>Production AI inference in Go. Faster than Python. One binary.</strong>
</p>

<p align="center">
  <a href="https://github.com/ailakshya/infergo/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  </a>
  <a href="https://pkg.go.dev/github.com/ailakshya/infergo">
    <img src="https://pkg.go.dev/badge/github.com/ailakshya/infergo.svg" alt="Go Reference">
  </a>
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900.svg" alt="CUDA 12">
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey.svg" alt="Platform">
</p>

---

## The numbers

Measured on RTX 5070 Ti, CUDA 12.8. Both via HTTP, keep-alive, 100 requests. **infergo wins every metric. Python crashes under load.**

### Latency (lower is better)

| Task | infergo | Python | Winner |
|---|---|---|---|
| LLM generation (per token) | **1.69 ms** | 13.62 ms | **infergo 8.1x** |
| Speculative decoding (8B + 1B) | **74 ms** | 496 ms | **infergo 6.7x** |
| Single embedding | **0.9 ms** | 1.9 ms | **infergo 2.1x** |
| Batch embedding (3 texts) | **1.4 ms** | 2.5 ms | **infergo 1.8x** |
| Reranking (3 docs) | **1.2 ms** | 6.2 ms | **infergo 5.2x** |
| Detection (yolo11n) | **2.4 ms** | 2.7 ms | **infergo 1.1x** |
| Prompt cache TTFT | **14 ms** | 40 ms | **infergo 2.9x** |
| JSON output validity | **100%** | 0% | **infergo** |
| HNSW vector search | **0.03 ms** | ~1 ms (faiss) | **infergo 33x** |

### Throughput under load (both HTTP, 100 requests per level)

| Feature | c=1 | c=4 | c=8 | c=16 | c=32 |
|---|---|---|---|---|---|
| **infergo embedding** | 692 | 957 | 951 | 927 | **911 req/s** |
| Python embedding | 437 | 459 | 46 | crashed | crashed |
| **infergo reranking** | 212 | 250 | 250 | 246 | **241 req/s** |
| **infergo detection** | 62 | 110 | 95 | 233 | **239 req/s** |
| infergo health check | — | — | — | — | **20,623 req/s** |

At c=8 Python embedding throughput drops to 46 req/s (P99 = 1,036 ms). At c=16 Python crashes. infergo serves 927 req/s with zero errors.

### Infrastructure

| | infergo | Python | |
|---|---|---|---|
| Docker image (CPU) | **0.18 GB** | 10 GB | **56x smaller** |
| Docker image (CUDA) | **1.52 GB** | 12 GB | **8x smaller** |
| Cold start | **456 ms** | 15 sec | **33x faster** |
| VRAM at c=10 | **700 MB** | 7,000 MB | **10x less** |
| Memory drift (1000 req) | **+0.3%** | +11.9% | **40x more stable** |
| Errors under load | **0** | crashes at c=16 | **infergo** |
| Models per binary | **LLM+embed+detect** | 1 | **3-in-1** |

---

## Why

Python inference servers have five problems in production:

1. **The GIL serializes requests.** 10 users = the 10th waits for 9 to finish. To scale, you fork processes. Each loads a full model copy. Llama 3 8B = 4.6 GB VRAM per process. 10 users = 46 GB.

2. **Cold start kills autoscaling.** Python + PyTorch + transformers = 15 seconds to boot. By the time your new pod is ready, the traffic spike is over.

3. **No structured output guarantee.** You ask for JSON, the model outputs prose. Python has no fix. infergo uses GBNF grammar sampling — the output is syntactically valid by construction.

4. **Three servers for three model types.** vLLM for LLMs, sentence-transformers for embeddings, ultralytics for detection. Three processes, three configs, three failure domains.

5. **10 GB container images.** Python + PyTorch + CUDA runtime + model framework = 10 GB pulled on every new node.

infergo is one binary. 22 MB. Serves LLM + embedding + detection on one port. 456 ms cold start. 0.18 GB Docker image. No Python runtime.

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
infergo serve --model models/llama3-8b-q4.gguf --port 9090

# Query
curl http://localhost:9090/v1/chat/completions \
  -d '{"model":"llama3-8b-q4","messages":[{"role":"user","content":"Hello"}]}'
```

### Multi-model (one binary, one port)

```bash
infergo serve \
  --model llm:models/llama3-8b-q4.gguf \
  --model embed:models/all-MiniLM-L6-v2.onnx \
  --model detect:models/yolov8n.onnx \
  --provider cuda
```

### Speculative decoding (6.7x faster)

```bash
infergo serve \
  --model llm:models/llama3-8b-q4.gguf \
  --draft-model models/llama3.2-1b-q4.gguf \
  --n-draft 5 --provider cuda
```

---

## API

Works with any OpenAI client. No code changes needed.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:9090/v1", api_key="none")

# Chat
r = client.chat.completions.create(
    model="llm", messages=[{"role":"user","content":"Hello"}])

# Guaranteed JSON output
r = client.chat.completions.create(
    model="llm", messages=[{"role":"user","content":"Return JSON with name and age"}],
    response_format={"type": "json_object"})

# Batch embeddings
r = client.embeddings.create(model="embed", input=["text1","text2","text3"])

# Streaming
for chunk in client.chat.completions.create(
    model="llm", messages=[{"role":"user","content":"Count to 5"}], stream=True):
    print(chunk.choices[0].delta.content or "", end="")
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat (streaming, structured output, function calling) |
| `POST` | `/v1/embeddings` | Embeddings (single string or batch `["a","b"]`) |
| `POST` | `/v1/search` | Vector similarity search (HNSW index) |
| `POST` | `/v1/rerank` | Rerank documents by query relevance |
| `POST` | `/v1/detect` | Object detection (JSON + base64) |
| `POST` | `/v1/detect/binary` | Object detection (raw JPEG body, faster) |
| `POST` | `/v1/detect/stream` | Streaming detection (SSE for video) |
| `POST` | `/v1/images/generations` | Image generation (Stable Diffusion) |
| `POST` | `/v1/audio/transcriptions` | Speech-to-text (Whisper) |
| `GET` | `/v1/models` | List loaded models |
| `POST` | `/v1/admin/reload` | Hot-swap model weights |
| `POST` | `/v1/admin/guardrails` | Configure content safety filters |
| `GET` | `/health/live` | Liveness probe |
| `GET` | `/health/ready` | Readiness probe |
| `GET` | `/metrics` | Prometheus metrics |

---

## How it's fast

| What | How | Impact |
|---|---|---|
| Full C generation loop | Entire decode/sample/append runs in C++. One CGo call per request. | 3.4x faster than per-token Go loop |
| nvJPEG GPU decode | JPEG decoded directly to GPU memory. No CPU decode, no upload. | 2.3x faster detection |
| Speculative decoding | Small draft model proposes tokens, target verifies in one batch. | 6.7x faster (90% acceptance) |
| Prompt caching | Serialized KV state cached by prompt hash. Repeated prompts skip prefill. | 2.9x faster TTFT |
| Continuous batching | All concurrent sequences decoded in one GPU call. | Throughput scales with concurrency |
| Grammar sampling | llama.cpp sampler chain enforces GBNF grammar on every token. | 100% valid JSON |
| Zero-copy sampling | Logits stay in C++ memory. Never cross CGo boundary. | Eliminates 1 MB/token memcpy |
| PagedAttention | KV cache allocated in pages, freed per-sequence. No fragmentation. | +0.3% RSS after 1,000 requests |

---

## Features

**Inference:** LLM (GGUF via llama.cpp) + Embedding (ONNX Runtime) + Detection (TorchScript / ONNX / TensorRT) + Vector search (HNSW) + Reranking + Streaming detection

**Performance:** Full C generation loop, nvJPEG GPU decode, speculative decoding, prompt caching, continuous batching, Flash Attention 2, grammar sampling, zero-copy logits

**AI features:** Structured output (JSON/GBNF), function calling (tool use), speculative decoding, batch embeddings, vector search, reranking, guardrails (content safety)

**Production:** Multi-model serving, hot reload, LoRA adapters, API key auth, rate limiting, request queue, guardrails, Prometheus metrics, OpenTelemetry tracing, KEDA autoscaling

**Deployment:** 0.18 GB CPU / 1.52 GB CUDA Docker image, Helm chart, multi-GPU (tensor split, pipeline stages, auto-shard), 456 ms cold start, `infergo models list/delete`

**Video:** NVDEC decode, GPU preprocessing, ByteTrack tracking, frame annotation, TurboJPEG encoding, streaming detection (SSE), 63 FPS dual 1440p cameras

**SDK:** Go client on pkg.go.dev, gRPC + HTTP + WebSocket, `infergo pull/convert/models` CLI

---

## Architecture

```
Clients (OpenAI SDK / curl / gRPC / WebSocket)
         │
         ▼
┌─────────────────────────────────────────┐
│  infergo server (Go)                    │
│  C generate loop · prompt cache         │
│  continuous batching · HNSW search      │
│  Prometheus · OTel · auth · queue       │
└────────────┬────────────────────────────┘
             │ CGo (1 call per request)
┌────────────▼────────────────────────────┐
│  infer_api.h  (C boundary)             │
├─────────┬──────────┬──────────┬────────┤
│ llama   │ ONNX RT  │ libtorch │ HNSW   │
│ .cpp    │ CPU/CUDA │ nvJPEG   │ search │
│ KV page │ TensorRT │ TorchSc  │        │
└─────────┴──────────┴──────────┴────────┘
             │
     NVIDIA CUDA / CPU / Metal
```

---

## Docker

```bash
# CPU (0.18 GB image)
docker run --rm -p 9090:9090 -v ./models:/models:ro \
  ghcr.io/ailakshya/infergo:cpu \
  serve --model /models/llama3-8b-q4.gguf

# CUDA (1.52 GB image — requires nvidia-container-toolkit)
docker run --rm --gpus all -p 9090:9090 -v ./models:/models:ro \
  ghcr.io/ailakshya/infergo:cuda \
  serve --model /models/llama3-8b-q4.gguf --provider cuda
```

| Image | Size | Includes |
|---|---|---|
| `infergo:cpu` | **0.18 GB** | infergo + llama.cpp + ONNX Runtime |
| `infergo:cuda` | **1.52 GB** | above + CUDA runtime + nvJPEG + TorchScript |
| Python equivalent | **10-12 GB** | Python + PyTorch + transformers + CUDA |

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
# 342 C++ tests (100% pass)
ctest --test-dir build --output-on-failure

# Go tests (19 packages, race detector clean)
cd go && go test -race ./...
```

---

## License

[Apache 2.0](LICENSE)
