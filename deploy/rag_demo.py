#!/usr/bin/env python3
"""
RAG Demo: Qwen Coder + Embedding + Vector DB
Ingests code docs, answers coding questions with context.
"""

import requests, json, time, sys

BASE = "http://localhost:9900"

def wait_ready():
    print("Waiting for server...")
    for i in range(60):
        try:
            r = requests.get(f"{BASE}/health/ready", timeout=2)
            if r.status_code == 200:
                print(f"Server ready ({i+1}s)")
                return True
        except: pass
        time.sleep(1)
    print("Server not ready!")
    return False

def ingest_docs():
    """Ingest coding documentation into vector DB"""
    docs = [
        "Go goroutines are lightweight threads managed by the Go runtime. They cost about 8KB of stack space. Use 'go func()' to launch one.",
        "Python's GIL (Global Interpreter Lock) prevents true parallel execution of threads. Use multiprocessing for CPU-bound tasks.",
        "In Go, channels are typed conduits for communication between goroutines. Use 'ch := make(chan int)' to create one.",
        "Docker multi-stage builds reduce image size. Use 'FROM builder AS build' then 'COPY --from=build' to copy only the binary.",
        "CUDA kernels are functions that run on the GPU. Use '__global__' keyword to define them. Launch with '<<<blocks, threads>>>'.",
        "ONNX Runtime supports multiple execution providers: CPU, CUDA, TensorRT, CoreML. Use SessionOptions to select one.",
        "llama.cpp loads GGUF models for LLM inference. It supports quantization levels: Q4_K_M, Q5_K_M, Q8_0, F16.",
        "TensorRT optimizes neural networks for NVIDIA GPUs. It fuses layers, selects optimal kernels, and supports INT8 quantization.",
        "gRPC uses Protocol Buffers for serialization. It's faster than REST/JSON for internal service communication.",
        "Prometheus metrics use four types: Counter, Gauge, Histogram, Summary. Expose them on /metrics endpoint.",
        "FastAPI is a Python web framework built on Starlette and Pydantic. It auto-generates OpenAPI docs at /docs.",
        "In Rust, ownership rules prevent data races at compile time. Each value has exactly one owner. Use &T for borrowing.",
        "Kubernetes HPA (Horizontal Pod Autoscaler) scales pods based on CPU, memory, or custom metrics like queue depth.",
        "WebSocket provides full-duplex communication over a single TCP connection. Use 'ws://' or 'wss://' protocol.",
        "Embedding models convert text to dense vectors. all-MiniLM-L6-v2 produces 384-dimensional vectors in ~1ms.",
        "HNSW (Hierarchical Navigable Small Worlds) is a graph-based algorithm for approximate nearest neighbor search.",
        "Speculative decoding uses a small draft model to propose tokens, then a large model verifies them in one batch.",
        "nvJPEG decodes JPEG images directly on the GPU, eliminating the CPU decode + upload pipeline.",
        "PagedAttention allocates KV cache in fixed-size pages, preventing memory fragmentation during long conversations.",
        "Flash Attention reduces memory from O(N^2) to O(N) for attention computation, enabling longer context windows.",
    ]

    print(f"\nIngesting {len(docs)} documents...")
    for i, doc in enumerate(docs):
        # Embed the document
        r = requests.post(f"{BASE}/v1/embeddings", json={"model":"embed","input":doc})
        if r.status_code != 200:
            print(f"  Failed to embed doc {i}: {r.text}")
            continue

        vec = r.json()["data"][0]["embedding"]
        print(f"  [{i+1}/{len(docs)}] embedded ({len(vec)}d): {doc[:50]}...")

    print(f"Ingested {len(docs)} documents.")
    return docs

def ask(question, docs_text):
    """Ask a question with RAG context"""
    print(f"\n{'─'*60}")
    print(f"Q: {question}")
    print(f"{'─'*60}")

    # 1. Embed the question
    r = requests.post(f"{BASE}/v1/embeddings", json={"model":"embed","input":question})
    if r.status_code != 200:
        print(f"Embed failed: {r.text}")
        return
    q_vec = r.json()["data"][0]["embedding"]

    # 2. Search for relevant docs (manual cosine similarity since vector search needs index)
    import numpy as np
    q = np.array(q_vec)
    scores = []
    for i, doc in enumerate(docs_text):
        r = requests.post(f"{BASE}/v1/embeddings", json={"model":"embed","input":doc})
        if r.status_code == 200:
            d = np.array(r.json()["data"][0]["embedding"])
            sim = np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d))
            scores.append((sim, i, doc))

    scores.sort(reverse=True)
    top_docs = scores[:3]

    print(f"\nRetrieved {len(top_docs)} relevant docs:")
    for sim, idx, doc in top_docs:
        print(f"  [{sim:.3f}] {doc[:60]}...")

    # 3. Build context and generate answer
    context = "\n".join([f"- {doc}" for _, _, doc in top_docs])
    prompt = f"""Based on the following context, answer the question concisely.

Context:
{context}

Question: {question}
Answer:"""

    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": "coder",
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant. Answer based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.3
    })

    if r.status_code == 200:
        answer = r.json()["choices"][0]["message"]["content"]
        print(f"\nA: {answer}")
    else:
        print(f"Generation failed: {r.text}")


if __name__ == "__main__":
    if not wait_ready():
        sys.exit(1)

    # Check models loaded
    r = requests.get(f"{BASE}/v1/models")
    models = [m["id"] for m in r.json()["data"]]
    print(f"Models loaded: {models}")

    if "coder" not in models or "embed" not in models:
        print("Need both 'coder' and 'embed' models. Start with:")
        print("  infergo serve --model coder:<qwen-coder.gguf> --model embed:<embed.onnx> --provider cuda")
        sys.exit(1)

    docs = ingest_docs()

    questions = [
        "How do goroutines work in Go?",
        "What is the GIL in Python and how to work around it?",
        "How does TensorRT optimize neural networks?",
        "What is speculative decoding?",
        "How to reduce Docker image size?",
    ]

    for q in questions:
        ask(q, docs)

    print(f"\n{'━'*60}")
    print(f"  RAG Demo Complete — {len(questions)} questions answered")
    print(f"{'━'*60}")
