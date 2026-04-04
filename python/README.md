# infergo Python package

Python interface to [infergo](https://github.com/ailakshya/infergo) — production-grade LLM inference.

Two ways to use infergo from Python:

1. **HTTP client (recommended)** — point the OpenAI SDK at an `infergo serve` process
2. **Native bindings** — ctypes wrapper around `libinfer_api.so` for in-process use

---

## The fast path: infergo server + OpenAI SDK

Start the server once, call it from anywhere:

```bash
infergo serve --model models/llama3-8b-q4.gguf --port 9090
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9090/v1", api_key="none")

resp = client.chat.completions.create(
    model="llama3-8b-q4",
    messages=[{"role": "user", "content": "Explain transformers in two sentences."}],
    max_tokens=200,
)
print(resp.choices[0].message.content)
```

This works with any OpenAI-compatible client — no code changes to existing projects.

**Why the server is faster at scale:**

| | c=1 | c=4 req/s | c=4 tok/s |
|---|---|---|---|
| infergo server | 457ms | **3.83** | **245** |
| llama-cpp-python | 456ms | 2.19 | 140 |
| infergo native | 545ms | 1.83 | 119 |

The server batches all concurrent requests into one GPU forward pass. llama-cpp-python and native bindings serialize through a lock.

---

## Native bindings

Install (requires `libinfer_api.so` from a built infergo):

```bash
pip install -e /path/to/infergo/python            # dev install
# or
INFERGO_LIB=/path/to/libinfer_api.so python app.py
```

```python
import infergo

with infergo.LLM("models/llama3-8b-q4.gguf", gpu_layers=999) as llm:
    # Chat — LLaMA 3 template applied automatically
    print(llm.chat("What is a transformer?", max_tokens=200))

    # Stream tokens
    for piece in llm.stream("Count to five:", max_tokens=50):
        print(piece, end="", flush=True)

    # Embeddings (ONNX)
    session = infergo.Session("models/all-MiniLM-L6-v2.onnx")
    vec = session.run({"input": tokens})
```

---

## Full documentation

See [`docs/python.md`](https://github.com/ailakshya/infergo/blob/main/docs/python.md) for:
- Async / asyncio usage
- Streaming
- LangChain integration
- Production patterns (FastAPI, load balancing)
- Scaling guide
- Full native bindings API reference
- Benchmark methodology
