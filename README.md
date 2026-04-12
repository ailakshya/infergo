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

### RAG pipeline — 5 modes (Qwen Coder + Embedding + Vector Search)

Full end-to-end: embed query → search documents → generate answer.
Each mode tested on both CPU and GPU. Qwen 2.5 Coder 1.5B + all-MiniLM-L6-v2.

| Mode | P50 | Cold Start | VRAM | GPU% | What it does |
|---|---|---|---|---|---|
| Python CPU | **636 ms** | 4,931 ms | — | 0% | llama-cpp-python + sentence-transformers, all CPU |
| Python GPU | **642 ms** | 4,907 ms | 2,720 MB | 15% | same libs, CUDA — barely faster (Python overhead wastes GPU) |
| **infergo CPU** | **162 ms** | 1,043 ms | — | 0% | one binary, CPU — 3.9x faster than Python |
| **infergo GPU** | **116 ms** | 1,037 ms | 5,350 MB | 85% | one binary, CUDA — 5.5x faster than Python |
| **infergo GPU+JSON** | **597 ms** | 1,037 ms | 5,350 MB | 85% | same + guaranteed valid JSON output |

**Why Python GPU (642ms) is barely faster than Python CPU (636ms):**
Python's per-token overhead is 3ms (GIL lock + logits copy + Python sampling). The GPU saves 2ms per token but Python adds 3ms back. Net GPU gain: ~6ms over 50 tokens. The GPU sits idle 77% of the time waiting for Python. infergo's GPU utilization is 85% because the entire decode loop runs in C++ with zero Python overhead.

**Cost estimate per 1M RAG requests:**

| | Python GPU | infergo GPU | Savings |
|---|---|---|---|
| GPU-hours | 178 hrs | 32 hrs | 146 hrs saved |
| T4 ($0.35/hr) | $62 | $11 | **82% cheaper** |
| A100 ($3/hr) | $534 | $96 | **82% cheaper** |
| H100 ($8/hr) | $1,424 | $256 | **82% cheaper** |

### Infrastructure

| | infergo | Python | |
|---|---|---|---|
| Docker image (CPU) | **0.18 GB** | 10 GB | **56x smaller** |
| Docker image (CUDA) | **1.52 GB** | 12 GB | **8x smaller** |
| Cold start | **1,037 ms** | 4,907 ms | **4.7x faster** |
| VRAM at c=10 | **700 MB** | 7,000 MB | **10x less** |
| Memory drift (1000 req) | **+0.3%** | +11.9% | **40x more stable** |
| Errors under load | **0** | crashes at c=16 | **infergo** |
| Models per binary | **LLM+embed+detect** | 1 | **3-in-1** |
| Pip packages needed | **0** | 12+ | **zero deps** |
| Python HTTP server | **not needed** | crashes with 2 models | **stable** |

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

## How infergo works

```
Your app (Python / Go / curl / any language)
    │
    │  HTTP request (OpenAI-compatible JSON)
    ▼
┌─────────────────────────────────────────────────────┐
│  infergo binary (22 MB)                             │
│                                                     │
│  Go layer (HTTP only — no inference compute here):  │
│    Parse JSON → 1 CGo call → Format response       │
│    ~0.3ms overhead. No GIL. No interpreter.         │
│                                                     │
│  C++ layer (ALL compute happens here):              │
│    ┌─────────────────────────────────────────────┐  │
│    │  LLM: llama.cpp                             │  │
│    │    Full generate loop in C++                 │  │
│    │    Prompt cache (skip repeat prefills)       │  │
│    │    Grammar sampling (guaranteed JSON)        │  │
│    │    Speculative decoding (draft+verify)       │  │
│    │    Flash Attention 2 (auto-enabled)          │  │
│    │    1 CGo call per request, not per token     │  │
│    ├─────────────────────────────────────────────┤  │
│    │  Embedding: ONNX Runtime / TorchScript      │  │
│    │    Tokenize + infer + pool + normalize       │  │
│    │    All in one C++ call                       │  │
│    │    TorchScript auto-selected for CUDA        │  │
│    ├─────────────────────────────────────────────┤  │
│    │  Detection: libtorch + nvJPEG               │  │
│    │    JPEG decoded on GPU (not CPU)             │  │
│    │    Preprocess + infer + NMS all on GPU       │  │
│    ├─────────────────────────────────────────────┤  │
│    │  Search: HNSW index (C++)                   │  │
│    │    0.03ms per query, 20K+ queries/sec       │  │
│    │    Persistent save/load to disk              │  │
│    ├─────────────────────────────────────────────┤  │
│    │  Rerank: embed + cosine in one C++ call     │  │
│    └─────────────────────────────────────────────┘  │
│                                                     │
│  GPU: 85-91% utilization (vs Python's 15-23%)       │
└─────────────────────────────────────────────────────┘
    │
    ▼  NVIDIA CUDA / CPU / Metal
```

**Why it's faster than Python:** Python's per-token overhead is 3ms (GIL lock + logits copy to Python + sampling in Python + GIL release). Over 50 tokens that's 150ms wasted. In infergo, the entire decode loop runs in C++ — the GPU never waits for an interpreter. Python uses 15% of the GPU. infergo uses 85%.

**VRAM footprint (measured, RTX 5070 Ti):**

| Configuration | VRAM | RSS (CPU) |
|---|---|---|
| 1 model (Qwen 1.5B, Q4) | 2,421 MB | ~200 MB |
| 1 model (Llama 3 8B, Q4) | ~4,200 MB | ~1,300 MB |
| 2 models (LLM + embedding) | ~5,000 MB | ~1,450 MB |
| 3 models (LLM + embed + detect) | ~5,350 MB | ~1,500 MB |
| Same 3 models in Python | ~2,720 MB + 2,691 MB RSS | 3 processes |

### Why infergo uses more VRAM but runs 5.5x faster

This seems backwards — Python uses 2,720 MB, infergo uses 5,350 MB. More memory = faster?

**Yes. Here's why:**

```
PYTHON GPU VRAM (2,720 MB) — underusing the GPU
────────────────────────────────────────────────

  Model weights (Q4):         1,100 MB   ← same as infergo
  KV cache:                     200 MB   ← small, only 1 seq
  PyTorch CUDA allocator:       920 MB   ← pre-allocated heap
  MiniLM embedding:             100 MB   ← same
  Compute workspace:            400 MB   ← SMALL
                              ─────────
  TOTAL:                      2,720 MB

  The compute workspace is small because Python doesn't use it
  efficiently. Between every token, Python:
    1. Acquires the GIL (0.5ms)
    2. Copies 594KB of logits GPU→CPU (1ms)
    3. Samples in Python (1ms)
    4. Releases the GIL (0.2ms)
  
  During those 2.7ms the GPU has NOTHING to do.
  It sits idle with allocated but unused memory.
  
  GPU busy: 2ms out of every 5ms = 40% utilization
  GPU idle: 3ms out of every 5ms = 60% WASTED


INFERGO GPU VRAM (5,350 MB) — fully using the GPU
────────────────────────────────────────────────

  Model weights (Q4):         1,100 MB   ← same
  KV cache:                     200 MB   ← same
  Compute workspace:          1,200 MB   ← 3x LARGER
  MiniLM TorchScript:           100 MB   ← same
  Flash Attention buffers:    2,100 MB   ← Python doesn't have this
  nvJPEG decoder:                50 MB   ← Python doesn't have this
  Prompt cache (KV states):     100 MB   ← Python doesn't have this
  Speculative decoder ctx:      500 MB   ← Python doesn't have this
                              ─────────
  TOTAL:                      5,350 MB

  WHY each extra allocation makes it faster:

  Flash Attention (2,100 MB):
    Standard attention: O(N²) memory, slow for long prompts
    Flash Attention: O(N) memory, 2x faster prefill
    Needs workspace buffers pre-allocated on GPU
    Python's llama-cpp-python doesn't allocate these

  Compute workspace (1,200 MB vs 400 MB):
    Larger workspace = GPU can pipeline operations
    While one matmul runs, the next one's data is already loaded
    Python's small workspace forces serial execution

  Prompt cache (100 MB):
    Serialized KV state for repeated prompts
    Second request with same prompt: skip prefill entirely
    Python has no equivalent — every request starts from scratch

  nvJPEG (50 MB):
    JPEG decoded directly on GPU
    Python decodes on CPU, then uploads (2ms wasted per image)

  The key insight:
    Python SAVES memory by NOT using the GPU efficiently.
    infergo SPENDS memory to KEEP the GPU busy.
    
    It's like buying a $10,000 GPU and then:
    - Python: uses 40% of it, saves 2.6 GB of VRAM
    - infergo: uses 85% of it, spends 2.6 GB more VRAM
    
    The VRAM is already paid for. Not using it is waste.
```

**What happens during one token generation:**

```
PYTHON (5ms per token — GPU idle 60% of the time):
  ┌──────┐┌─────────────────┐┌──────┐┌──────────┐
  │ GIL  ││   GPU decode    ││ copy ││ Py sample │
  │ lock ││   (2ms)         ││logits││ (1ms)     │
  │0.5ms ││   ████████      ││ 1ms  ││           │
  └──────┘└─────────────────┘└──────┘└──────────┘
  ▓▓▓▓▓▓▓ ████████████████████ ░░░░░░ ░░░░░░░░░░
  Python   GPU busy            GPU idle (copying + Python)
  
  GPU: ████████░░░░░░░░░░  = 40% busy

INFERGO (2ms per token — GPU busy 95% of the time):
  ┌──────────────────┐┌────┐
  │   GPU decode     ││next│
  │   (2ms)          ││0.1 │
  │   ████████████   ││    │
  └──────────────────┘└────┘
  ████████████████████ ▓
  GPU busy              sample
                        (in C++,
                         zero copy)
  
  GPU: ██████████████████  = 95% busy
```

At c=10 concurrency:
- **infergo:** 1 process, same VRAM, all 10 users share 1 model copy
- **Python:** 10 processes × 2,720 MB = 27,200 MB VRAM (won't fit on any single GPU)

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
