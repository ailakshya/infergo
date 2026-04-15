# Python SDK Reference

The infergo Python SDK provides a typed client for the infergo AI platform. Zero dependencies -- uses only the Python standard library.

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Client configuration](#client-configuration)
- [Chat completions](#chat-completions)
- [Streaming](#streaming)
- [Embeddings](#embeddings)
- [Object detection](#object-detection)
- [Search](#search)
- [Document ingestion](#document-ingestion)
- [NLP tasks](#nlp-tasks)
- [Models and health](#models-and-health)
- [OpenAI SDK compatibility](#openai-sdk-compatibility)
- [Async / asyncio](#async--asyncio)
- [LangChain integration](#langchain-integration)
- [Native Python bindings](#native-python-bindings)
- [Production patterns](#production-patterns)
- [Performance](#performance)

---

## Installation

```bash
pip install infergo
```

Or from source:

```bash
cd sdk/python
pip install .
```

---

## Quick start

```python
from infergo import InfergoClient

client = InfergoClient("http://localhost:9090")

# Chat
response = client.chat([{"role": "user", "content": "Hello!"}])
print(response)

# Stream tokens
for token in client.chat_stream([{"role": "user", "content": "Count to 5"}]):
    print(token, end="", flush=True)
print()
```

---

## Client configuration

```python
from infergo import InfergoClient

# Basic
client = InfergoClient("http://localhost:9090")

# With API key authentication
client = InfergoClient("http://localhost:9090", api_key="my-api-key")
```

The client uses Python's built-in `urllib` with a 120-second timeout. No external dependencies required.

---

## Chat completions

```python
response = client.chat(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain KV caching in one paragraph."},
    ],
    model="llama3-8b-q4",
    max_tokens=200,
    temperature=0.7,
)
print(response)  # string: the assistant's reply
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `messages` | list[dict] | (required) | Conversation messages with `role` and `content` |
| `model` | str | `"llm"` | Model name |
| `max_tokens` | int | `256` | Maximum tokens to generate |
| `temperature` | float | `0.7` | Sampling temperature |
| `**kwargs` | any | -- | Additional fields passed to the API (e.g. `tools`, `response_format`) |

### Return value

Returns a `str` containing the assistant's response text.

---

## Streaming

```python
for token in client.chat_stream(
    messages=[{"role": "user", "content": "Write a haiku about inference."}],
    model="llama3-8b-q4",
    max_tokens=64,
    temperature=0.7,
):
    print(token, end="", flush=True)
print()
```

`chat_stream` returns an iterator of token strings. Tokens are delivered via SSE (Server-Sent Events) as they are generated.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `messages` | list[dict] | (required) | Conversation messages |
| `model` | str | `"llm"` | Model name |
| `max_tokens` | int | `256` | Maximum tokens to generate |
| `temperature` | float | `0.7` | Sampling temperature |

---

## Embeddings

### Single text

```python
vector = client.embed("hello world", model="embed")
print(f"Dimension: {len(vector)}")  # e.g. 384
```

### Batch embedding

```python
vectors = client.embed_batch(
    ["hello world", "infergo is fast", "transformers are powerful"],
    model="embed",
)
print(f"{len(vectors)} vectors, dim={len(vectors[0])}")
```

### Cosine similarity

```python
import math

def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

v1 = client.embed("machine learning")
v2 = client.embed("deep learning")
print(f"Similarity: {cosine_sim(v1, v2):.3f}")
```

---

## Object detection

```python
import base64

with open("photo.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

detections = client.detect(
    image_b64=image_b64,
    model="yolo11n",
    conf=0.25,
    iou=0.45,
)

for obj in detections:
    print(f"Class {obj['ClassID']}: {obj['Confidence']:.2f} "
          f"at ({obj['X1']:.0f},{obj['Y1']:.0f})-({obj['X2']:.0f},{obj['Y2']:.0f})")
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_b64` | str | (required) | Base64-encoded JPEG/PNG image |
| `model` | str | `"detect"` | Detection model name |
| `conf` | float | `0.25` | Confidence threshold |
| `iou` | float | `0.45` | IoU threshold for NMS |

### Return value

Returns a `list[dict]` where each dict has: `X1`, `Y1`, `X2`, `Y2`, `ClassID`, `Confidence`.

---

## Search

```python
results = client.search(
    query="transformer architecture",
    model="embed",
    k=5,
    mode="hybrid",  # "vector", "bm25", or "hybrid"
)

for hit in results:
    print(f"ID={hit['id']} Score={hit['score']:.3f} {hit.get('metadata', '')}")
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | str | (required) | Search query text |
| `model` | str | `"embed"` | Embedding model for vector search |
| `k` | int | `5` | Number of results to return |
| `mode` | str | `"hybrid"` | Search mode: `"vector"`, `"bm25"`, or `"hybrid"` |

---

## Document ingestion

```python
result = client.ingest(
    texts=["Document 1 content...", "Document 2 content..."],
    model="embed",
    metadata=[
        {"source": "file1.txt"},
        {"source": "file2.txt"},
    ],
)
print(result)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `texts` | list[str] | (required) | Documents to ingest |
| `model` | str | `"embed"` | Embedding model for vectorization |
| `metadata` | list[dict] | `None` | Optional metadata per document |

---

## NLP tasks

### Named Entity Recognition

```python
entities = client.ner(
    text="Apple CEO Tim Cook announced new products in Cupertino.",
    model="llama3-8b-q4",
)
for entity in entities:
    print(f"{entity['text']} -> {entity['type']}")
# Apple -> ORG
# Tim Cook -> PERSON
# Cupertino -> LOCATION
```

### Sentiment analysis

```python
result = client.sentiment(
    text="This product is absolutely fantastic!",
    model="llama3-8b-q4",
)
print(f"{result['sentiment']}: {result['score']:.2f}")
# positive: 0.95
```

### Text classification

```python
result = client.classify(
    text="Stock prices surged today on Wall Street.",
    labels=["business", "sports", "politics", "technology"],
    model="llama3-8b-q4",
)
print(f"Label: {result['label']}, Score: {result['score']:.2f}")
# Label: business, Score: 0.92
```

### Summarization

```python
summary = client.summarize(
    text="<long article text...>",
    model="llama3-8b-q4",
    max_length=100,
)
print(summary)
```

---

## Models and health

### List loaded models

```python
models = client.models()
for m in models:
    print(m["id"])
```

### Health check

```python
status = client.health()
print(status)  # {"status": "ok"}
```

---

## OpenAI SDK compatibility

infergo is fully compatible with the OpenAI Python SDK. Use it as a drop-in replacement:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9090/v1", api_key="none")

# Chat
resp = client.chat.completions.create(
    model="llama3-8b-q4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain KV caching in one paragraph."},
    ],
    max_tokens=200,
    temperature=0.7,
)
print(resp.choices[0].message.content)
print(f"Tokens used: {resp.usage.completion_tokens}")
```

**With API key auth** (when infergo is started with `--api-key mytoken`):

```python
client = OpenAI(base_url="http://localhost:9090/v1", api_key="mytoken")
```

### Streaming with OpenAI SDK

```python
with client.chat.completions.stream(
    model="llama3-8b-q4",
    messages=[{"role": "user", "content": "Write a haiku about inference."}],
    max_tokens=64,
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
print()
```

### Embeddings with OpenAI SDK

```python
result = client.embeddings.create(
    model="embed",
    input=["hello world", "infergo is fast"],
)
vec1 = result.data[0].embedding
vec2 = result.data[1].embedding
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

Because infergo uses continuous batching, all 4 requests go through the GPU together in the same forward pass -- same latency as 1 request, 4x the throughput.

---

## LangChain integration

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
print(f"Embedding dim: {len(vecs[0])}")
```

Works with any LangChain chain, agent, or retrieval pipeline.

---

## Native Python bindings

For cases where you cannot run a separate server process, the `python/infergo` package provides ctypes bindings that load `libinfer_api.so` directly into your Python process.

### Install

```bash
# Point to the built library
INFERGO_LIB=/path/to/build/cpp/api/libinfer_api.so python your_script.py

# Or add to PYTHONPATH
PYTHONPATH=/path/to/infergo/python python your_script.py
```

### Usage

```python
import infergo

# Load model
llm = infergo.LLM(
    "models/llama3-8b-q4.gguf",
    gpu_layers=999,
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

### ONNX sessions

```python
session = infergo.Session("cuda", 0)
session.load("model.onnx")

tensor = infergo.Tensor.cpu([1, 3, 224, 224], infergo.FLOAT32)
# ... fill tensor data ...
outputs = session.run([tensor])
```

### TorchScript sessions

```python
torch_session = infergo.TorchSession("cuda", 0)
torch_session.load("model.torchscript.pt")
outputs = torch_session.run([input_tensor])
```

### When to use native bindings vs the server

| Situation | Use |
|---|---|
| Production API serving users | **infergo server + HTTP** |
| Multiple concurrent users | **infergo server + HTTP** |
| Offline batch processing, single thread | native bindings |
| Self-contained script, no server process | native bindings |
| Testing / development | either |

The server is always faster under concurrent load because it batches requests. Native bindings serialize through a Python lock -- one request at a time.

---

## Production patterns

### FastAPI service

```python
from fastapi import FastAPI
from infergo import InfergoClient

app = FastAPI()
client = InfergoClient("http://localhost:9090")

@app.post("/chat")
async def chat(message: str):
    return {"reply": client.chat([{"role": "user", "content": message}])}

@app.post("/detect")
async def detect(image_b64: str):
    return {"objects": client.detect(image_b64)}

@app.post("/search")
async def search(query: str):
    return {"results": client.search(query)}
```

### Multiple infergo servers behind a load balancer

```python
import random
from infergo import InfergoClient

SERVERS = [
    "http://gpu-node-1:9090",
    "http://gpu-node-2:9090",
    "http://gpu-node-3:9090",
]

def get_client() -> InfergoClient:
    return InfergoClient(random.choice(SERVERS))
```

### RAG pipeline

```python
# 1. Ingest documents
client.ingest(
    texts=["Document 1...", "Document 2...", "Document 3..."],
    model="embed",
)

# 2. Search for relevant documents
results = client.search("What is the main topic?", mode="hybrid", k=5)

# 3. Build context and generate
context = "\n".join(r.get("metadata", "") for r in results)
response = client.chat([
    {"role": "system", "content": f"Answer based on this context:\n{context}"},
    {"role": "user", "content": "What is the main topic?"},
])
```

---

## Performance

Measured on RTX 5070 Ti, LLaMA 3 8B Q4\_K\_M, 20 requests per scenario.

| Approach | c=1 P50 | c=1 tok/s | c=4 P50 | c=4 req/s | c=4 tok/s |
|---|---|---|---|---|---|
| infergo native (ctypes) | 545ms | 119 | 2182ms | 1.83 | 119 |
| infergo server + OpenAI SDK | 457ms | 139 | 1055ms | 3.79 | 242 |
| infergo server + infergo SDK | 457ms | 139 | 1045ms | **3.83** | **245** |
| infergo server + Go CLI | 459ms | 139 | 1048ms | 3.80 | 244 |
| llama-cpp-python (baseline) | 456ms | 140 | 1823ms | 2.19 | 140 |

Key findings:

- At c=1: all approaches are identical -- same GPU, same model, same speed
- At c=4: infergo server is **1.75x faster** than llama-cpp-python (3.83 vs 2.19 req/s)
- llama-cpp-python throughput is flat regardless of concurrency -- serialized lock
- Native bindings hit the same wall as llama-cpp-python at c=4
- The OpenAI SDK, infergo SDK, and Go CLI all show the same server performance -- the network/SDK layer is not the bottleneck

---

## Complete API reference

| Method | Signature | Returns | Description |
|---|---|---|---|
| `chat` | `chat(messages, model, max_tokens, temperature, **kwargs)` | `str` | Chat completion |
| `chat_stream` | `chat_stream(messages, model, max_tokens, temperature)` | `Iterator[str]` | Streaming chat |
| `embed` | `embed(text, model)` | `list[float]` | Single text embedding |
| `embed_batch` | `embed_batch(texts, model)` | `list[list[float]]` | Batch embeddings |
| `detect` | `detect(image_b64, model, conf, iou)` | `list[dict]` | Object detection |
| `search` | `search(query, model, k, mode)` | `list[dict]` | Vector/BM25/hybrid search |
| `ingest` | `ingest(texts, model, metadata)` | `dict` | Document ingestion |
| `ner` | `ner(text, model)` | `list[dict]` | Named entity recognition |
| `sentiment` | `sentiment(text, model)` | `dict` | Sentiment analysis |
| `classify` | `classify(text, labels, model)` | `dict` | Text classification |
| `summarize` | `summarize(text, model, max_length)` | `str` | Text summarization |
| `models` | `models()` | `list[dict]` | List loaded models |
| `health` | `health()` | `dict` | Server health check |
