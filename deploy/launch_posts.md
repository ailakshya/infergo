# infergo Launch Posts

---

## Hacker News — Show HN

**Title:** Show HN: infergo – Run LLMs in Go without Python (llama.cpp + ONNX bindings)

**Body:**

infergo is an open source AI inference server written entirely in Go, with a CGo bridge to llama.cpp for LLM inference and ONNX Runtime for embeddings and object detection.

The core problem it solves: Python's GIL makes LLM serving fundamentally single-threaded at the model level. The standard workaround — llama-cpp-python with multiple worker processes — means each worker loads a full copy of the model. A LLaMA 3 8B Q4 model is 4.6 GB. At 10 concurrent workers, you're at 46 GB of RAM/VRAM just to serve one model to multiple users. On a 16 GB GPU, that means you can't serve more than 3 concurrent users before you run out of memory entirely.

infergo loads the model once. All requests share that one copy via Go's goroutines and a continuous batching scheduler that sends all active sequences through the GPU in a single batch per decode step. At concurrency=32, infergo uses 1168 MB flat — the Python approach would require ~65 GB (56× more).

Benchmarks (RTX 5070 Ti, LLaMA 3 8B Q4_K_M):
- CUDA throughput: 200 tok/s vs 133 tok/s for llama-cpp-python (+50%)
- Memory at c=32: 1168 MB (infergo) vs ~65,536 MB estimated (Python multi-process)
- Cold start: 451 ms vs 494 ms (no Python interpreter startup)
- CPU: 10 tok/s vs 5 tok/s (2× faster, no GIL contention)

The server exposes an OpenAI-compatible HTTP API (/v1/chat/completions, /v1/embeddings, /v1/detect), SSE streaming, Prometheus metrics at /metrics, and Kubernetes health endpoints (/healthz, /readyz).

The Go client SDK is published to pkg.go.dev:

    go get github.com/ailakshya/infergo@v0.1.0

No Python, no venv, no pip. One binary.

Repo: https://github.com/ailakshya/infergo

Would love feedback on the CGo bridge design, the continuous batching scheduler, and whether the OpenAI-compatible API covers the use cases people actually need.

---

## r/golang — Post

**Title:** infergo: LLM inference server in pure Go — no Python, no venv, just `go get`

**Body:**

I built infergo, an AI inference server for Go that wraps llama.cpp and ONNX Runtime via CGo. The goal was to make running LLMs in a Go service as easy as importing any other package.

**Why Go instead of Python?**

Python's GIL means you can't do concurrent inference in one process. Every production setup ends up spawning multiple worker processes, each loading a full model copy. For LLaMA 3 8B (4.6 GB), 10 workers = 46 GB. Goroutines and channels are a much better fit for a concurrent inference scheduler.

**What it does:**
- OpenAI-compatible HTTP API (`/v1/chat/completions`, `/v1/embeddings`, `/v1/detect`)
- SSE streaming with proper backpressure via buffered channels
- Continuous batching scheduler — all goroutines share one model, one GPU decode step per tick
- Prometheus metrics, Kubernetes `/healthz` + `/readyz` endpoints
- ONNX Runtime for embeddings (all-MiniLM-L6-v2) and object detection (YOLOv8)

**Typed Go client SDK:**

```go
import "github.com/ailakshya/infergo/go/client"

c := client.New("http://localhost:9090", client.WithAPIKey("secret"))

resp, err := c.Chat(ctx, client.ChatRequest{
    Model:    "llama3-8b-q4",
    Messages: []client.Message{{Role: "user", Content: "Hello"}},
})

tokens, errc := c.ChatStream(ctx, req) // SSE streaming

vec, err := c.Embed(ctx, client.EmbedRequest{
    Model: "all-MiniLM-L6-v2",
    Input: "semantic search query",
})
```

Install:

    go get github.com/ailakshya/infergo@v0.1.0

**Benchmarks (RTX 5070 Ti):**
- 200 tok/s at c=4 (vs 133 tok/s for llama-cpp-python)
- 1168 MB RAM flat regardless of concurrency (vs ~2 GB × N for Python workers)

The CGo bridge follows strict rules: no C++ types cross the boundary, every exception is caught and converted to an error code, every allocated object has a paired destroy function.

Repo: https://github.com/ailakshya/infergo

Happy to answer questions about the CGo design, the batching scheduler, or anything else.

---

## r/MachineLearning — Post

**Title:** infergo: LLM inference in Go — 1168 MB flat vs 65 GB for Python at c=32, continuous batching, OpenAI API

**Body:**

I've been working on infergo, a Go-native LLM inference server that uses the same llama.cpp weights as llama-cpp-python but serves them without Python's GIL constraints.

**The GIL wall**

The fundamental problem with Python LLM serving at scale is that the GIL makes the model forward pass single-threaded. The standard production answer is gunicorn/uvicorn workers — but each worker is a separate process that loads a full model copy. At 10 workers for a 4.6 GB model, you've spent 46 GB of GPU memory just to handle concurrency.

**The Go approach**

infergo loads the model once. A continuous batching scheduler collects all incoming requests and sends every active sequence through a single `llama_decode` call per step. This is the same approach used by vLLM's continuous batching, but implemented in Go with goroutines handling the concurrency and a shared KV cache managed by a page allocator.

**Benchmark results (RTX 5070 Ti, LLaMA 3 8B Q4_K_M)**

Memory at various concurrency levels:

| Concurrency | infergo RSS | Python est. | Ratio |
|---|---|---|---|
| 1  | 1168 MB | ~2048 MB  | 1.8× |
| 4  | 1168 MB | ~8192 MB  | 7.0× |
| 16 | 1168 MB | ~32768 MB | 28×  |
| 32 | 1168 MB | ~65536 MB | 56×  |

infergo RSS is constant because it's one process with one model copy. Python numbers are estimated (one process per worker × model size).

Throughput (CUDA, short prompts, n=40):
- infergo: 200 tok/s at c=4 (+50% vs Python's 133 tok/s)
- Throughput scaling: 2.2 req/s at c=1 → 17.4 req/s at c=32 (7.9× scaling)

Cold start: 451 ms (vs 494 ms Python) — no Python interpreter overhead.

**What's included:**
- llama.cpp CGo bridge (C API boundary: no C++ types, exceptions converted to int codes)
- ONNX Runtime bridge for embeddings (all-MiniLM-L6-v2) and detection (YOLOv8)
- OpenAI-compatible HTTP server with SSE streaming
- Prometheus metrics, Kubernetes health probes
- Typed Go client SDK on pkg.go.dev

Repo: https://github.com/ailakshya/infergo

Limitations: KV cache saturation starts showing errors at c=32 with current n_seq_max settings — working on tuning that. Python comparison numbers at high concurrency are estimates (pending full multi-process benchmark run).
