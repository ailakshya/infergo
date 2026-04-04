# Using infergo from Python

infergo exposes an OpenAI-compatible HTTP server. Any Python code that already uses the OpenAI SDK works with infergo unchanged — just point `base_url` at your infergo server.

---

## Contents

- [The right architecture](#the-right-architecture)
- [Quickstart](#quickstart)
- [OpenAI SDK](#openai-sdk)
- [Zero-dependency (stdlib only)](#zero-dependency-stdlib-only)
- [Async / asyncio](#async--asyncio)
- [Streaming](#streaming)
- [Embeddings](#embeddings)
- [LangChain](#langchain)
- [Production patterns](#production-patterns)
- [Scaling](#scaling)
- [Native Python bindings](#native-python-bindings)
- [Benchmark: all 5 approaches](#benchmark-all-5-approaches)

---

## The right architecture

```
Python app
    │
    │  HTTP (OpenAI-compatible)
    ▼
infergo serve  ←── one binary, one GPU, continuous batching
    │
    │  CGo
    ▼
llama.cpp / CUDA
```

Start the server once. Point all your Python code at it. Multiple processes, threads, and requests all share the same GPU efficiently — no locks, no serialisation.

---

## Quickstart

```bash
# 1. Download a model
wget -O llama3-8b-q4.gguf \
  "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

# 2. Start the server (CPU)
infergo serve --model llama3-8b-q4.gguf --port 9090

# 3. Or with GPU
infergo serve --model llama3-8b-q4.gguf --provider cuda --gpu-layers 999 --port 9090

# 4. Install the OpenAI SDK
pip install openai
```

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

## OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9090/v1",
    api_key="none",          # infergo ignores this unless --api-key is set
)

# Basic chat
resp = client.chat.completions.create(
    model="llama3-8b-q4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Explain KV caching in one paragraph."},
    ],
    max_tokens=200,
    temperature=0.7,
)
print(resp.choices[0].message.content)
print(f"tokens used: {resp.usage.completion_tokens}")
```

**With API key auth** (when infergo is started with `--api-key mytoken`):

```python
client = OpenAI(
    base_url="http://localhost:9090/v1",
    api_key="mytoken",
)
```

---

## Zero-dependency (stdlib only)

No pip install needed. Uses Python's built-in `urllib`:

```python
import urllib.request
import json

def chat(prompt: str, model: str = "llama3-8b-q4", max_tokens: int = 200) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }).encode()

    req = urllib.request.Request(
        "http://localhost:9090/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"]

print(chat("What is a transformer model?"))
```

---

## Async / asyncio

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:9090/v1", api_key="none")

async def ask(question: str) -> str:
    resp = await client.chat.completions.create(
        model="llama3-8b-q4",
        messages=[{"role": "user", "content": question}],
        max_tokens=200,
    )
    return resp.choices[0].message.content

async def main():
    # Fire 4 requests concurrently — infergo batches them on the GPU
    questions = [
        "What is attention?",
        "What is a transformer?",
        "What is CUDA?",
        "What is quantisation?",
    ]
    answers = await asyncio.gather(*[ask(q) for q in questions])
    for q, a in zip(questions, answers):
        print(f"Q: {q}\nA: {a}\n")

asyncio.run(main())
```

Because infergo uses continuous batching, all 4 requests above go through the GPU together in the same forward pass — same latency as 1 request, 4x the throughput.

---

## Streaming

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9090/v1", api_key="none")

with client.chat.completions.stream(
    model="llama3-8b-q4",
    messages=[{"role": "user", "content": "Write a haiku about inference."}],
    max_tokens=64,
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
print()
```

Tokens are streamed via SSE (Server-Sent Events) as they are generated. The WebSocket endpoint (`ws://localhost:9090/v1/chat/completions/ws`) is also available for browser clients.

---

## Embeddings

```python
from openai import OpenAI

# Start server with an embedding model:
# infergo serve --model embed:models/all-MiniLM-L6-v2.onnx --port 9090

client = OpenAI(base_url="http://localhost:9090/v1", api_key="none")

result = client.embeddings.create(
    model="embed",
    input=["hello world", "infergo is fast"],
)

vec1 = result.data[0].embedding  # list of floats
vec2 = result.data[1].embedding

# Cosine similarity
import numpy as np
cos = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
print(f"similarity: {cos:.3f}")
```

---

## LangChain

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# LLM
llm = ChatOpenAI(
    base_url="http://localhost:9090/v1",
    api_key="none",
    model="llama3-8b-q4",
    temperature=0.7,
)
print(llm.invoke("Explain transformers in one sentence.").content)

# Embeddings
embeddings = OpenAIEmbeddings(
    base_url="http://localhost:9090/v1",
    api_key="none",
    model="embed",
)
vecs = embeddings.embed_documents(["hello", "world"])
print(f"embedding dim: {len(vecs[0])}")
```

Works with any LangChain chain, agent, or retrieval pipeline — no custom integration needed.

---

## Production patterns

**FastAPI service**

```python
from fastapi import FastAPI
from openai import AsyncOpenAI
from pydantic import BaseModel

app = FastAPI()
client = AsyncOpenAI(base_url="http://localhost:9090/v1", api_key="none")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    resp = await client.chat.completions.create(
        model="llama3-8b-q4",
        messages=[{"role": "user", "content": req.message}],
        max_tokens=200,
    )
    return {"reply": resp.choices[0].message.content}
```

**Multiple infergo servers behind a load balancer**

```python
import random
from openai import OpenAI

SERVERS = [
    "http://gpu-node-1:9090/v1",
    "http://gpu-node-2:9090/v1",
    "http://gpu-node-3:9090/v1",
]

def get_client() -> OpenAI:
    return OpenAI(base_url=random.choice(SERVERS), api_key="none")
```

Each infergo server saturates one GPU. Add more servers for linear throughput scaling.

---

## Scaling

| Users | Setup | Expected throughput |
|---|---|---|
| 1–10 | Single infergo server | ~3–4 req/s on one GPU |
| 10–100 | Single infergo server | Same req/s, higher latency — queue builds |
| 100+ | Multiple infergo servers + load balancer | Linear: N servers = N × req/s |

**What infergo does for you at scale:**
- All concurrent requests share one GPU via continuous batching
- No Python GIL — Go handles all scheduling
- Prometheus metrics (`/metrics`) show queue depth, active sequences, tok/s
- KEDA autoscaler can scale infergo pods based on queue depth

**What you need to do:**
- Run infergo behind nginx or a Kubernetes service for load balancing
- Set `--max-queue` to control backpressure (default 100)
- Monitor `infergo_queue_depth` in Grafana

---

## Native Python bindings

For cases where you cannot run a separate server process, the `python/infergo` package provides ctypes bindings that load `libinfer_api.so` directly into your Python process.

**Install**

```bash
# From source — requires libinfer_api.so to be built first
PYTHONPATH=/path/to/infergo/python python your_script.py

# Or point to the built library
INFERGO_LIB=/path/to/build/cpp/api/libinfer_api.so python your_script.py
```

**Usage**

```python
import infergo

# Load model
llm = infergo.LLM(
    "models/llama3-8b-q4.gguf",
    gpu_layers=999,    # offload all layers to GPU
    ctx_size=16384,
)

# Chat (applies LLaMA 3 template automatically)
reply = llm.chat("Explain KV caching.", max_tokens=200)
print(reply)

# Raw generation
text = llm.generate("The transformer architecture", max_tokens=100, temperature=0.8)
print(text)

# Streaming
for piece in llm.stream("Count to five:", max_tokens=50):
    print(piece, end="", flush=True)

# Tokenize
token_ids = llm.tokenize("Hello world")
print(token_ids)

# Always close when done
llm.close()

# Or use as context manager
with infergo.LLM("models/llama3-8b-q4.gguf") as llm:
    print(llm.chat("Hello"))
```

**When to use native bindings vs the server**

| Situation | Use |
|---|---|
| Production API serving users | **infergo server + HTTP** |
| Multiple concurrent users | **infergo server + HTTP** |
| Offline batch processing, single thread | native bindings |
| Self-contained script, no server process | native bindings |
| Testing / development | either |

The server is always faster under concurrent load because it batches requests. Native bindings serialize through a Python lock — one request at a time.

---

## Benchmark: all 5 approaches

Measured on RTX 5070 Ti, LLaMA 3 8B Q4\_K\_M, 20 requests per scenario.

| Approach | c=1 P50 | c=1 tok/s | c=4 P50 | c=4 req/s | c=4 tok/s |
|---|---|---|---|---|---|
| infergo native (ctypes) | 545ms | 119 | 2182ms | 1.83 | 119 |
| infergo server + OpenAI SDK | 457ms | 139 | 1055ms | 3.79 | 242 |
| infergo server + urllib | 457ms | 139 | 1045ms | **3.83** | **245** |
| infergo server + Go CLI | 459ms | 139 | 1048ms | 3.80 | 244 |
| llama-cpp-python (baseline) | 456ms | 140 | 1823ms | 2.19 | 140 |

**Key findings:**

- At c=1: all approaches are identical — same GPU, same model, same speed
- At c=4: infergo server is **1.75× faster** than llama-cpp-python (3.83 vs 2.19 req/s)
- llama-cpp-python throughput is flat regardless of concurrency — serialised lock
- Native bindings hit the same wall as llama-cpp-python at c=4
- The OpenAI SDK, urllib, and Go CLI all show the same server performance — the network/SDK layer is not the bottleneck

The benchmark script is at [`python/benchmarks/bench_three_ways.py`](../python/benchmarks/bench_three_ways.py).
