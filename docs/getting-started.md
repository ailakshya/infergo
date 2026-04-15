# Getting Started with infergo

Production AI inference in Go. Serve LLMs, embeddings, object detection, search, RAG, function calling, agents, and NLP tasks from a single binary.

---

## Prerequisites

| Tool | Version |
|---|---|
| Go | 1.23+ |
| CMake | 3.20+ |
| C++ compiler | GCC 12+ or Clang 15+ |
| (optional) CUDA | 12.0+ for GPU acceleration |
| (optional) libtorch | PyTorch C++ runtime for TorchScript detection models |
| (optional) OpenCV | 4.x for image preprocessing |

---

## 1. Clone and build

```bash
git clone https://github.com/ailakshya/infergo
cd infergo

# Build the C++ shared library (CPU)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target infer_api -j$(nproc)

# Build the CLI
cd go
CGO_CFLAGS="-I../cpp/include" \
CGO_LDFLAGS="-L../build/cpp/api -linfer_api -Wl,-rpath,$(pwd)/../build/cpp/api" \
go build -o ../infergo ./cmd/infergo
cd ..
```

For GPU (CUDA + libtorch):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  -DTorch_DIR=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")/Torch

cmake --build build --target infer_api -j$(nproc)
```

---

## 2. Download a model

infergo loads GGUF models (LLaMA, Mistral, Qwen, Phi, and others) via llama.cpp.

```bash
# Example: LLaMA 3 8B Q4_K_M (~4.7 GB)
mkdir -p models
wget -O models/llama3-8b-q4.gguf \
  https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf

# Or use the built-in pull command
./infergo pull bartowski/Meta-Llama-3-8B-Instruct-GGUF --quant Q4_K_M
```

---

## 3. Start the server

```bash
# CPU
./infergo serve --model models/llama3-8b-q4.gguf --port 9090

# GPU (CUDA -- offload all layers)
./infergo serve --model models/llama3-8b-q4.gguf --provider cuda --gpu-layers 999 --port 9090
```

You should see:

```
[infergo] loaded llama3-8b-q4 (provider=cpu, ctx=16384)
[infergo] listening on :9090
```

### Serve flags

| Flag | Default | Description |
|---|---|---|
| `--model` | (required) | Model to load; repeatable. Format: `[name:]path.{gguf,onnx,pt}` |
| `--provider` | `cpu` | Execution provider: `cpu`, `cuda`, `tensorrt`, `coreml` |
| `--backend` | `auto` | Inference backend: `auto`, `onnx`, `tensorrt`, `torch` |
| `--port` | `9090` | HTTP listen port |
| `--grpc-port` | `9091` | gRPC listen port (0 = disabled) |
| `--gpu-layers` | `999` | Transformer layers to offload to GPU |
| `--ctx-size` | `16384` | Total KV cache token budget |
| `--max-seqs` | `16` | Max concurrent sequences / KV cache slots |
| `--threads` | auto | CPU threads (0 = physical cores / 2) |
| `--api-key` | (open) | Bearer auth key (or set `INFERGO_API_KEY` env var) |
| `--rate-limit` | `0` | Max requests/second per client IP (0 = unlimited) |
| `--max-queue` | `100` | Max in-flight requests; 503 beyond this |
| `--max-active` | `0` | Max concurrent handlers (0 = same as max-queue) |
| `--mode` | `combined` | Server role: `combined`, `prefill`, `decode` |
| `--adaptive` | `false` | Adaptive hybrid detection backend routing |
| `--safe-mode` | `false` | Disable batching/adaptive; single-image libtorch only |
| `--cache-size` | `1000` | Max response cache entries (0 = disabled) |
| `--draft-model` | (none) | Draft GGUF model for speculative decoding |
| `--n-draft` | `5` | Tokens to draft per speculative step |
| `--tensor-split` | (none) | GPU fractions for tensor parallelism (e.g. `0.5,0.5`) |
| `--pipeline-stages` | `1` | Pipeline parallelism across N GPUs |
| `--otlp-endpoint` | (none) | OTLP HTTP endpoint for distributed tracing |
| `--batch-timeout-ms` | `0` | Wait time before firing a batch |
| `--gc-interval` | `100` | GC frequency (every N requests) |

---

## 4. CLI commands

### `infergo chat` -- Interactive terminal chat

```bash
./infergo chat --server http://localhost:9090 --model llama3-8b-q4

# With a system prompt
./infergo chat --model llama3-8b-q4 --system "You are a helpful coding assistant."
```

In-chat commands: `/help`, `/clear`, `/system <text>`, `/model <name>`.

### `infergo bench` -- Quick model benchmarking

```bash
# Benchmark a GGUF model directly (no server needed)
./infergo bench models/llama3-8b-q4.gguf --output results.json
```

Reports load time, prompt tokens, prefill speed, and generation tok/s.

### `infergo convert` -- Model format conversion

```bash
# PyTorch to TorchScript
./infergo convert --input model.pt --format torchscript --output model.torchscript.pt

# PyTorch to ONNX
./infergo convert --input model.pt --format onnx --output model.onnx --imgsz 640

# ONNX to TensorRT
./infergo convert --input model.onnx --format tensorrt --output model.trt

# HuggingFace to GGUF
./infergo convert --input model.safetensors --format gguf --quant q4_k_m
```

### `infergo validate` -- Model validation

```bash
# Quick validation (single model)
./infergo validate models/llama3-8b-q4.gguf
./infergo validate models/yolo11n.onnx --type detect

# Comparison validation (source vs exported)
./infergo validate --source original.pt --export exported.onnx --samples 100 --tolerance 1e-4
```

### `infergo pull` -- Download from HuggingFace

```bash
./infergo pull bartowski/Meta-Llama-3-8B-Instruct-GGUF --quant Q4_K_M --dir models/
```

---

## 5. Quick examples

### Chat completion

```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-8b-q4",
    "messages": [{"role": "user", "content": "Hello! What can you do?"}],
    "max_tokens": 256
  }'
```

### Streaming

```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3-8b-q4","messages":[{"role":"user","content":"Count to 5"}],"stream":true}'
```

### Embeddings

```bash
curl http://localhost:9090/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"embed","input":"hello world"}'
```

### Object detection

```bash
# Binary endpoint (fastest -- raw JPEG bytes)
curl -X POST "http://localhost:9090/v1/detect/binary?model=yolo11n" \
  -H "Content-Type: image/jpeg" \
  --data-binary @photo.jpg

# JSON endpoint
curl http://localhost:9090/v1/detect \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"yolo11n\",\"image_b64\":\"$(base64 -w0 photo.jpg)\"}"
```

### Search (vector, BM25, hybrid)

```bash
curl http://localhost:9090/v1/search \
  -H "Content-Type: application/json" \
  -d '{"model":"embed","query":"transformer architecture","k":5,"mode":"hybrid","alpha":0.5}'
```

### RAG (Retrieval-Augmented Generation)

```bash
# Ingest documents first
curl http://localhost:9090/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"model":"embed","texts":["Document 1 text...","Document 2 text..."]}'

# Query with RAG
curl http://localhost:9090/v1/rag \
  -H "Content-Type: application/json" \
  -d '{"model":"llm","embed_model":"embed","query":"What is the main topic?","k":5}'
```

### Function calling

```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-8b-q4",
    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"}
          },
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "auto"
  }'
```

### Agents

```bash
curl http://localhost:9090/v1/agents/run \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-8b-q4",
    "query": "What is 25 * 17 + 3?",
    "tools": ["calculator"],
    "max_iterations": 5
  }'
```

### NLP tasks

```bash
# Named Entity Recognition
curl http://localhost:9090/v1/ner \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3-8b-q4","text":"Apple CEO Tim Cook announced new products in Cupertino."}'

# Sentiment analysis
curl http://localhost:9090/v1/sentiment \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3-8b-q4","text":"This product is absolutely fantastic!"}'

# Classification
curl http://localhost:9090/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3-8b-q4","text":"Stock prices surged today.","labels":["business","sports","politics"]}'

# Summarization
curl http://localhost:9090/v1/summarize \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3-8b-q4","text":"<long article text>","max_length":100}'
```

---

## 6. Python SDK quick start

```bash
pip install infergo
```

```python
from infergo import InfergoClient

client = InfergoClient("http://localhost:9090")

# Chat
response = client.chat([{"role": "user", "content": "Hello!"}])
print(response)

# Streaming
for token in client.chat_stream([{"role": "user", "content": "Count to 5"}]):
    print(token, end="", flush=True)

# Embeddings
vec = client.embed("hello world")

# Detection
import base64
with open("photo.jpg", "rb") as f:
    detections = client.detect(base64.b64encode(f.read()).decode())

# Search
results = client.search("transformer architecture", mode="hybrid")

# NLP tasks
entities = client.ner("Tim Cook is the CEO of Apple.")
sentiment = client.sentiment("This is great!")
label = client.classify("Stock prices rose.", labels=["business", "sports"])
summary = client.summarize("Long article text...")
```

See [python.md](python.md) for the full SDK reference.

---

## 7. TypeScript SDK quick start

```bash
npm install infergo
```

```typescript
import { InfergoClient } from "infergo";

const client = new InfergoClient({ baseUrl: "http://localhost:9090" });

// Chat
const reply = await client.chat([{ role: "user", content: "Hello!" }]);

// Streaming
for await (const token of client.chatStream([{ role: "user", content: "Count to 5" }])) {
  process.stdout.write(token);
}

// Embeddings
const vec = await client.embed("hello world");

// Detection
const objects = await client.detect(imageBase64, "yolo11n");

// Search
const results = await client.search("query text", "embed", 5, "hybrid");
```

---

## 8. OpenAI SDK compatibility

infergo is OpenAI API-compatible. Use the standard OpenAI SDK with any language:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9090/v1", api_key="none")
resp = client.chat.completions.create(
    model="llama3-8b-q4",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

---

## 9. Check what's loaded

```bash
# Via CLI
./infergo list-models --addr http://localhost:9090

# Via API
curl http://localhost:9090/v1/models
```

---

## 10. Health and metrics

```bash
# Kubernetes liveness / readiness
curl http://localhost:9090/healthz
curl http://localhost:9090/readyz

# Prometheus metrics
curl http://localhost:9090/metrics

# OpenAPI spec
curl http://localhost:9090/v1/openapi.json

# Web UI
open http://localhost:9090/ui

# Playground
open http://localhost:9090/ui/playground

# Health dashboard
open http://localhost:9090/ui/dashboard
```

---

## Next steps

- [deployment.md](deployment.md) -- Docker, Kubernetes, multi-tenant, RBAC, canary deployments
- [go-api-reference.md](go-api-reference.md) -- Embed infergo in your Go application
- [c-api-reference.md](c-api-reference.md) -- Call the C API directly
- [detection.md](detection.md) -- Object detection, backends, tracking
- [python.md](python.md) -- Python SDK reference
- [contributing.md](contributing.md) -- Add a new execution provider
