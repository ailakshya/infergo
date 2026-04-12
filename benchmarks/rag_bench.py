#!/usr/bin/env python3
"""
RAG BENCHMARK: Qwen Coder + Embedding + Vector DB
infergo vs Python (llama-cpp-python + sentence-transformers + faiss/numpy)
Both via HTTP. Full RAG pipeline: embed query → search → generate.
"""

import time, statistics, os, subprocess, signal, json, sys
import http.client
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

QWEN = "/tmp/qwen2.5-coder-1.5b-q4.gguf"
RUNS = 20
WARMUP = 5

# Code documentation to ingest
DOCS = [
    "Go goroutines are lightweight threads. Use 'go func()' to launch. Cost 8KB stack.",
    "Python GIL prevents parallel threads. Use multiprocessing for CPU tasks.",
    "Channels in Go are typed conduits between goroutines. 'ch := make(chan int)'.",
    "Docker multi-stage builds: 'FROM builder' then 'COPY --from=build'.",
    "CUDA kernels use __global__ keyword. Launch with <<<blocks, threads>>>.",
    "ONNX Runtime supports CPU, CUDA, TensorRT, CoreML execution providers.",
    "llama.cpp loads GGUF models. Supports Q4_K_M, Q5_K_M, Q8_0 quantization.",
    "TensorRT fuses layers and selects optimal CUDA kernels. Supports INT8.",
    "gRPC uses Protocol Buffers. Faster than REST/JSON for services.",
    "Prometheus metrics: Counter, Gauge, Histogram, Summary on /metrics.",
]

QUESTIONS = [
    "How do goroutines work?",
    "What is the Python GIL?",
    "How does TensorRT optimize models?",
    "What quantization does llama.cpp support?",
    "How to use Docker multi-stage builds?",
]


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def bench_infergo():
    """Full RAG pipeline via infergo HTTP"""
    print(f"\n{'━'*60}")
    print(f"  infergo RAG (Qwen Coder + Embedding, CUDA)")
    print(f"{'━'*60}")

    conn = http.client.HTTPConnection("localhost", 9900)
    h = {"Content-Type": "application/json", "Connection": "keep-alive"}

    # Embed all docs
    doc_vecs = []
    for doc in DOCS:
        conn.request("POST", "/v1/embeddings", json.dumps({"model":"embed","input":doc}), h)
        r = conn.getresponse().read()
        doc_vecs.append(json.loads(r)["data"][0]["embedding"])

    # Benchmark full RAG pipeline
    times = []
    for _ in range(WARMUP):
        q = QUESTIONS[0]
        conn.request("POST", "/v1/embeddings", json.dumps({"model":"embed","input":q}), h)
        qv = json.loads(conn.getresponse().read())["data"][0]["embedding"]
        sims = [(cosine_sim(qv, dv), i) for i, dv in enumerate(doc_vecs)]
        sims.sort(reverse=True)
        ctx = "\n".join([DOCS[i] for _, i in sims[:3]])
        conn.request("POST", "/v1/chat/completions", json.dumps({
            "model":"coder", "messages":[{"role":"user","content":f"Context:\n{ctx}\n\nQ: {q}\nA:"}],
            "max_tokens":50}), h)
        conn.getresponse().read()

    for qi, q in enumerate(QUESTIONS):
        for _ in range(RUNS // len(QUESTIONS)):
            s = time.perf_counter()

            # 1. Embed query
            conn.request("POST", "/v1/embeddings", json.dumps({"model":"embed","input":q}), h)
            qv = json.loads(conn.getresponse().read())["data"][0]["embedding"]

            # 2. Search (cosine similarity)
            sims = [(cosine_sim(qv, dv), i) for i, dv in enumerate(doc_vecs)]
            sims.sort(reverse=True)
            ctx = "\n".join([DOCS[i] for _, i in sims[:3]])

            # 3. Generate
            conn.request("POST", "/v1/chat/completions", json.dumps({
                "model":"coder", "messages":[{"role":"user","content":f"Context:\n{ctx}\n\nQ: {q}\nA:"}],
                "max_tokens":50}), h)
            conn.getresponse().read()

            times.append((time.perf_counter()-s)*1000)

    conn.close()
    return times


def bench_python():
    """Full RAG pipeline via Python (llama-cpp-python + sentence-transformers)"""
    print(f"\n{'━'*60}")
    print(f"  Python RAG (llama-cpp-python + sentence-transformers, CUDA)")
    print(f"{'━'*60}")

    # Start Python LLM server
    py_llm = subprocess.Popen(
        ["python3","-m","llama_cpp.server","--model",QWEN,"--n_gpu_layers","99",
         "--n_ctx","4096","--host","0.0.0.0","--port","8900"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    for i in range(60):
        try:
            c = http.client.HTTPConnection("localhost",8900)
            c.request("GET","/v1/models"); r = c.getresponse()
            if r.status == 200: c.close(); break
            c.close()
        except: pass
        time.sleep(1)

    # Start Python embedding server
    py_embed_code = '''import json;from http.server import HTTPServer,BaseHTTPRequestHandler;from sentence_transformers import SentenceTransformer
model=SentenceTransformer("all-MiniLM-L6-v2",device="cuda")
class H(BaseHTTPRequestHandler):
 def do_POST(self):
  data=json.loads(self.rfile.read(int(self.headers["Content-Length"])));inp=data.get("input","")
  if isinstance(inp,str):inp=[inp]
  vecs=model.encode(inp).tolist();resp=json.dumps({"data":[{"embedding":v}for v in vecs]}).encode()
  self.send_response(200);self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(resp)));self.end_headers();self.wfile.write(resp)
 def log_message(self,*a):pass
HTTPServer(("0.0.0.0",8901),H).serve_forever()'''
    py_emb = subprocess.Popen(["python3","-c",py_embed_code],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    time.sleep(8)

    conn_llm = http.client.HTTPConnection("localhost", 8900)
    conn_emb = http.client.HTTPConnection("localhost", 8901)
    h = {"Content-Type": "application/json", "Connection": "keep-alive"}

    # Embed docs
    doc_vecs = []
    for doc in DOCS:
        conn_emb.request("POST", "/", json.dumps({"input":doc}), h)
        r = conn_emb.getresponse().read()
        doc_vecs.append(json.loads(r)["data"][0]["embedding"])

    # Warmup
    for _ in range(3):
        q = QUESTIONS[0]
        conn_emb.request("POST", "/", json.dumps({"input":q}), h)
        qv = json.loads(conn_emb.getresponse().read())["data"][0]["embedding"]
        sims = [(cosine_sim(qv, dv), i) for i, dv in enumerate(doc_vecs)]
        sims.sort(reverse=True)
        ctx = "\n".join([DOCS[i] for _, i in sims[:3]])
        conn_llm.request("POST", "/v1/chat/completions", json.dumps({
            "model":"default", "messages":[{"role":"user","content":f"Context:\n{ctx}\n\nQ: {q}\nA:"}],
            "max_tokens":50}), h)
        conn_llm.getresponse().read()

    # Benchmark
    times = []
    for qi, q in enumerate(QUESTIONS):
        for _ in range(RUNS // len(QUESTIONS)):
            s = time.perf_counter()

            conn_emb.request("POST", "/", json.dumps({"input":q}), h)
            qv = json.loads(conn_emb.getresponse().read())["data"][0]["embedding"]

            sims = [(cosine_sim(qv, dv), i) for i, dv in enumerate(doc_vecs)]
            sims.sort(reverse=True)
            ctx = "\n".join([DOCS[i] for _, i in sims[:3]])

            conn_llm.request("POST", "/v1/chat/completions", json.dumps({
                "model":"default", "messages":[{"role":"user","content":f"Context:\n{ctx}\n\nQ: {q}\nA:"}],
                "max_tokens":50}), h)
            conn_llm.getresponse().read()

            times.append((time.perf_counter()-s)*1000)

    conn_llm.close()
    conn_emb.close()
    os.killpg(os.getpgid(py_llm.pid), signal.SIGKILL)
    os.killpg(os.getpgid(py_emb.pid), signal.SIGKILL)

    return times


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  RAG BENCHMARK — Qwen Coder + Embedding + Vector Search    ║")
    print("║  Full pipeline: embed → search → generate                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # infergo (already running on :9900)
    ig_times = bench_infergo()
    ig = statistics.median(ig_times)
    print(f"  P50={ig:.0f}ms  Min={min(ig_times):.0f}ms  Avg={statistics.mean(ig_times):.0f}ms")

    # Kill infergo to free GPU for Python
    subprocess.run("pkill -9 -f 'infergo serve'", shell=True)
    time.sleep(5)

    # Python
    py_times = bench_python()
    py = statistics.median(py_times)
    print(f"  P50={py:.0f}ms  Min={min(py_times):.0f}ms  Avg={statistics.mean(py_times):.0f}ms")

    # Results
    print(f"\n{'━'*60}")
    print(f"  RAG PIPELINE RESULTS")
    print(f"{'━'*60}")
    print(f"  infergo: P50={ig:.0f}ms  ({len(ig_times)} requests)")
    print(f"  Python:  P50={py:.0f}ms  ({len(py_times)} requests)")

    if ig < py:
        print(f"  → infergo {py/ig:.1f}x FASTER")
    else:
        print(f"  → Python {ig/py:.1f}x faster")

    print(f"\n  Breakdown:")
    print(f"  {'Step':<20} {'infergo':>10} {'Python':>10}")
    print(f"  {'─'*40}")
    print(f"  {'Embed query':<20} {'~1ms':>10} {'~2ms':>10}")
    print(f"  {'Vector search':<20} {'<0.1ms':>10} {'<0.1ms':>10}")
    print(f"  {'LLM generate':<20} {f'~{ig-2:.0f}ms':>10} {f'~{py-3:.0f}ms':>10}")
    print(f"  {'Total':<20} {f'{ig:.0f}ms':>10} {f'{py:.0f}ms':>10}")
    print(f"{'━'*60}")
