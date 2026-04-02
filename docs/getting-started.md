# Getting Started with infergo

Serve a LLaMA 3 model in Go in under 10 minutes.

---

## Prerequisites

| Tool | Version |
|---|---|
| Go | 1.23+ |
| CMake | 3.20+ |
| C++ compiler | GCC 12+ or Clang 15+ |
| (optional) CUDA | 12.0+ for GPU acceleration |
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

---

## 2. Download a model

infergo loads GGUF models (LLaMA, Mistral, Qwen, Phi, …) via llama.cpp.

```bash
# Example: LLaMA 3 8B Q4_K_M (~4.7 GB)
mkdir -p models
wget -O models/llama3-8b-q4.gguf \
  https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

---

## 3. Start the server

```bash
# CPU
./infergo serve --model models/llama3-8b-q4.gguf --port 9090

# GPU (CUDA — offload all layers)
./infergo serve --model models/llama3-8b-q4.gguf --provider cuda --gpu-layers 999 --port 9090
```

You should see:

```
[infergo] loaded llama3-8b-q4 (provider=cpu, ctx=4096)
[infergo] listening on :9090
```

---

## 4. Send a request

infergo speaks the OpenAI API. Use curl, the Python openai SDK, or any HTTP client.

```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-8b-q4",
    "messages": [{"role": "user", "content": "Hello! What can you do?"}],
    "max_tokens": 256
  }'
```

Streaming:

```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3-8b-q4","messages":[{"role":"user","content":"Count to 5"}],"stream":true}'
```

Python (openai SDK):

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

## 5. Serve an ONNX model (embeddings or detection)

```bash
# Embeddings
./infergo serve --model models/resnet50.onnx --port 9090

curl http://localhost:9090/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"resnet50","input":"<base64-encoded-image>"}'

# Object detection
./infergo serve --model models/yolov8n.onnx --port 9090

curl http://localhost:9090/v1/detect \
  -H "Content-Type: application/json" \
  -d '{"model":"yolov8n","image_b64":"<base64>","conf_thresh":0.25,"iou_thresh":0.45}'
```

---

## 6. Check what's loaded

```bash
# Via CLI
./infergo list-models

# Via API
curl http://localhost:9090/v1/models
```

---

## 7. Health and metrics

```bash
# Kubernetes liveness / readiness
curl http://localhost:9090/healthz
curl http://localhost:9090/readyz

# Prometheus metrics
curl http://localhost:9090/metrics
```

---

## Next steps

- [deployment.md](deployment.md) — Docker, Kubernetes, bare metal
- [go-api-reference.md](go-api-reference.md) — Embed infergo in your Go application
- [c-api-reference.md](c-api-reference.md) — Call the C API directly
- [contributing.md](contributing.md) — Add a new execution provider
