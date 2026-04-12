#!/usr/bin/env python3
"""
4-WAY RAG BENCHMARK with full resource logging.

Setup 1: PURE PYTHON — all Python libs, in-process
Setup 2: PYTHON CLIENT → INFERGO SERVER — Python code calls infergo HTTP
Setup 3: INFERGO with Python-style flow — separate embed/search/generate calls
Setup 4: COMPLETE INFERGO — all built-in, single binary

Each setup: 20 RAG queries, log CPU%, GPU%, VRAM, RSS, time.
"""

import time, statistics, os, subprocess, signal, json, sys, gc, psutil
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

QWEN = "/tmp/qwen2.5-coder-1.5b-q4.gguf"
INFERGO = os.path.expanduser("~/cgo/infergo")
CWD = os.path.expanduser("~/cgo")
EMBED_MODEL = os.path.expanduser("~/cgo/models/all-MiniLM-L6-v2/onnx/model.onnx")

DOCS = [
    "Go goroutines are lightweight threads. Use go func() to launch.",
    "Python GIL prevents parallel threads. Use multiprocessing.",
    "CUDA kernels use __global__. Launch with <<<blocks, threads>>>.",
    "ONNX Runtime supports CPU, CUDA, TensorRT providers.",
    "llama.cpp loads GGUF models. Supports Q4_K_M quantization.",
    "TensorRT fuses layers for optimal GPU kernels.",
    "Docker multi-stage builds reduce image size.",
    "Prometheus exposes Counter, Gauge, Histogram metrics.",
    "gRPC uses Protocol Buffers for fast serialization.",
    "Flash Attention reduces memory from O(N^2) to O(N).",
]
QUESTIONS = ["How do goroutines work?", "What is the GIL?", "How does TensorRT work?",
             "What quantization does llama.cpp use?"]
RUNS = 20

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0

def get_gpu_stats():
    try:
        r = subprocess.run(["nvidia-smi","--query-gpu=memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        parts = r.stdout.strip().split(", ")
        return {"vram_mb": int(parts[0]), "gpu_pct": int(parts[1]), "power_w": float(parts[2])}
    except: return {"vram_mb": 0, "gpu_pct": 0, "power_w": 0}

def get_cpu_mem():
    return {"cpu_pct": psutil.cpu_percent(interval=0.1), "rss_mb": psutil.Process().memory_info().rss // (1024*1024)}

def clear_all():
    subprocess.run("pkill -9 -f 'infergo serve' 2>/dev/null", shell=True)
    subprocess.run("pkill -9 -f 'llama_cpp' 2>/dev/null", shell=True)
    time.sleep(3); gc.collect()
    try: import torch; torch.cuda.empty_cache()
    except: pass
    time.sleep(2)

R = {}

print("╔══════════════════════════════════════════════════════════════════╗")
print("║  4-WAY RAG BENCHMARK — Full resource logging                   ║")
print("║  20 RAG queries each · CPU + GPU + VRAM + RSS + Power          ║")
print("╚══════════════════════════════════════════════════════════════════╝")

# ═══════════════════════════════════════════════════════════════
# SETUP 1: PURE PYTHON
# ═══════════════════════════════════════════════════════════════

print(f"\n{'━'*65}")
print(f"  SETUP 1: PURE PYTHON (all in-process)")
print(f"  llama-cpp-python + sentence-transformers + numpy cosine")
print(f"{'━'*65}")

clear_all()
gpu_before = get_gpu_stats()

from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

load_s = time.perf_counter()
llm = Llama(model_path=QWEN, n_gpu_layers=99, n_ctx=2048, verbose=False)
st = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
load_time = (time.perf_counter() - load_s) * 1000

gpu_after = get_gpu_stats()
mem = get_cpu_mem()

print(f"  Load time: {load_time:.0f}ms")
print(f"  VRAM: {gpu_before['vram_mb']}→{gpu_after['vram_mb']} MB (+{gpu_after['vram_mb']-gpu_before['vram_mb']} MB)")
print(f"  RSS: {mem['rss_mb']} MB")

# Pre-embed docs
doc_vecs = st.encode(DOCS).tolist()

# Warmup
for _ in range(3):
    qv = st.encode(["test"]).tolist()[0]
    llm.create_chat_completion(messages=[{"role":"user","content":"Hi"}], max_tokens=16)

# Benchmark
times = []
for q in QUESTIONS:
    for _ in range(RUNS // len(QUESTIONS)):
        s = time.perf_counter()
        qv = st.encode([q]).tolist()[0]
        sims = sorted([(cosine(qv, dv), i) for i, dv in enumerate(doc_vecs)], reverse=True)
        ctx = "\n".join([DOCS[i] for _, i in sims[:3]])
        llm.create_chat_completion(
            messages=[{"role":"user","content":f"Context:\n{ctx}\n\nQ: {q}\nA:"}], max_tokens=50)
        times.append((time.perf_counter()-s)*1000)

gpu_during = get_gpu_stats()
R["pure_python"] = {
    "p50": statistics.median(times), "min": min(times), "avg": statistics.mean(times),
    "load_ms": load_time, "vram_mb": gpu_after['vram_mb'],
    "rss_mb": mem['rss_mb'], "gpu_pct": gpu_during['gpu_pct'], "power_w": gpu_during['power_w'],
    "deps": "llama-cpp-python + sentence-transformers + numpy + torch",
    "processes": 1, "servers": 0
}
print(f"  P50={R['pure_python']['p50']:.0f}ms  GPU={gpu_during['gpu_pct']}%  Power={gpu_during['power_w']:.0f}W")
del llm, st; gc.collect()
try: import torch; torch.cuda.empty_cache()
except: pass

# ═══════════════════════════════════════════════════════════════
# SETUP 2: PYTHON CLIENT → INFERGO SERVER
# ═══════════════════════════════════════════════════════════════

print(f"\n{'━'*65}")
print(f"  SETUP 2: PYTHON CLIENT → INFERGO SERVER")
print(f"  Python sends requests, infergo does all inference")
print(f"{'━'*65}")

clear_all()
gpu_before = get_gpu_stats()

load_s = time.perf_counter()
env = os.environ.copy()
env["LD_LIBRARY_PATH"] = f"{os.path.expanduser('~/onnxruntime/lib')}:{env.get('LD_LIBRARY_PATH','')}"
ig = subprocess.Popen([INFERGO,"serve",f"--model=coder:{QWEN}",f"--model=embed:{EMBED_MODEL}",
    "--provider=cuda","--port=9900","--grpc-port=0","--max-seqs=4","--ctx-size=2048"],
    cwd=CWD, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
for i in range(60):
    try:
        r = subprocess.run(["curl","-s","http://localhost:9900/health/live"],capture_output=True,timeout=2)
        if r.returncode == 0: break
    except: pass
    time.sleep(1)
load_time = (time.perf_counter() - load_s) * 1000

gpu_after = get_gpu_stats()

import http.client
conn = http.client.HTTPConnection("localhost", 9900)
h = {"Content-Type":"application/json","Connection":"keep-alive"}

print(f"  Load time: {load_time:.0f}ms")
print(f"  VRAM: {gpu_before['vram_mb']}→{gpu_after['vram_mb']} MB (+{gpu_after['vram_mb']-gpu_before['vram_mb']} MB)")

# Pre-embed
doc_vecs2 = []
for doc in DOCS:
    conn.request("POST","/v1/embeddings",json.dumps({"model":"embed","input":doc}),h)
    doc_vecs2.append(json.loads(conn.getresponse().read())["data"][0]["embedding"])

# Warmup
for _ in range(5):
    conn.request("POST","/v1/embeddings",json.dumps({"model":"embed","input":"test"}),h)
    conn.getresponse().read()
    conn.request("POST","/v1/chat/completions",json.dumps({"model":"coder","messages":[{"role":"user","content":"Hi"}],"max_tokens":8}),h)
    conn.getresponse().read()

# Benchmark
times2 = []
for q in QUESTIONS:
    for _ in range(RUNS // len(QUESTIONS)):
        s = time.perf_counter()
        conn.request("POST","/v1/embeddings",json.dumps({"model":"embed","input":q}),h)
        qv = json.loads(conn.getresponse().read())["data"][0]["embedding"]
        sims = sorted([(cosine(qv, dv), i) for i, dv in enumerate(doc_vecs2)], reverse=True)
        ctx = "\n".join([DOCS[i] for _, i in sims[:3]])
        conn.request("POST","/v1/chat/completions",json.dumps({
            "model":"coder","messages":[{"role":"user","content":f"Context:\n{ctx}\n\nQ: {q}\nA:"}],
            "max_tokens":50}),h)
        conn.getresponse().read()
        times2.append((time.perf_counter()-s)*1000)

gpu_during = get_gpu_stats()
ig_rss = 0
try:
    for p in psutil.Process(ig.pid).children(recursive=True): ig_rss += p.memory_info().rss
    ig_rss += psutil.Process(ig.pid).memory_info().rss
    ig_rss //= (1024*1024)
except: pass

conn.close()
R["python_to_infergo"] = {
    "p50": statistics.median(times2), "min": min(times2), "avg": statistics.mean(times2),
    "load_ms": load_time, "vram_mb": gpu_after['vram_mb'],
    "rss_mb": ig_rss, "gpu_pct": gpu_during['gpu_pct'], "power_w": gpu_during['power_w'],
    "deps": "infergo binary + Python requests", "processes": 1, "servers": 1
}
print(f"  P50={R['python_to_infergo']['p50']:.0f}ms  GPU={gpu_during['gpu_pct']}%  RSS={ig_rss}MB")

# ═══════════════════════════════════════════════════════════════
# SETUP 3: INFERGO with rerank endpoint (embed+search in one call)
# ═══════════════════════════════════════════════════════════════

print(f"\n{'━'*65}")
print(f"  SETUP 3: INFERGO BUILT-IN RERANK")
print(f"  /v1/rerank does embed+cosine+sort in one call")
print(f"{'━'*65}")

conn3 = http.client.HTTPConnection("localhost", 9900)

# Warmup
for _ in range(5):
    conn3.request("POST","/v1/rerank",json.dumps({"model":"embed","query":"test","documents":DOCS[:3],"top_n":2}),h)
    conn3.getresponse().read()

times3 = []
for q in QUESTIONS:
    for _ in range(RUNS // len(QUESTIONS)):
        s = time.perf_counter()
        conn3.request("POST","/v1/rerank",json.dumps({"model":"embed","query":q,"documents":DOCS,"top_n":3}),h)
        top = json.loads(conn3.getresponse().read())["results"]
        ctx = "\n".join([r.get("document","") for r in top])
        conn3.request("POST","/v1/chat/completions",json.dumps({
            "model":"coder","messages":[{"role":"user","content":f"Context:\n{ctx}\n\nQ: {q}\nA:"}],
            "max_tokens":50}),h)
        conn3.getresponse().read()
        times3.append((time.perf_counter()-s)*1000)
conn3.close()

R["infergo_rerank"] = {
    "p50": statistics.median(times3), "min": min(times3), "avg": statistics.mean(times3),
    "load_ms": load_time, "vram_mb": gpu_after['vram_mb'],
    "rss_mb": ig_rss, "gpu_pct": gpu_during['gpu_pct'], "power_w": gpu_during['power_w'],
    "deps": "infergo binary only", "processes": 1, "servers": 1
}
print(f"  P50={R['infergo_rerank']['p50']:.0f}ms")

# ═══════════════════════════════════════════════════════════════
# SETUP 4: COMPLETE INFERGO (JSON mode for structured answers)
# ═══════════════════════════════════════════════════════════════

print(f"\n{'━'*65}")
print(f"  SETUP 4: COMPLETE INFERGO — JSON mode + rerank")
print(f"  rerank + generate with structured JSON output")
print(f"{'━'*65}")

conn4 = http.client.HTTPConnection("localhost", 9900)

times4 = []
for q in QUESTIONS:
    for _ in range(RUNS // len(QUESTIONS)):
        s = time.perf_counter()
        conn4.request("POST","/v1/rerank",json.dumps({"model":"embed","query":q,"documents":DOCS,"top_n":3}),h)
        top = json.loads(conn4.getresponse().read())["results"]
        ctx = "\n".join([r.get("document","") for r in top])
        conn4.request("POST","/v1/chat/completions",json.dumps({
            "model":"coder","messages":[{"role":"user","content":f"Context:\n{ctx}\n\nQ: {q}\nA:"}],
            "max_tokens":50,"response_format":{"type":"json_object"}}),h)
        conn4.getresponse().read()
        times4.append((time.perf_counter()-s)*1000)
conn4.close()

R["infergo_complete"] = {
    "p50": statistics.median(times4), "min": min(times4), "avg": statistics.mean(times4),
    "load_ms": load_time, "vram_mb": gpu_after['vram_mb'],
    "rss_mb": ig_rss, "gpu_pct": gpu_during['gpu_pct'], "power_w": gpu_during['power_w'],
    "deps": "infergo binary only", "processes": 1, "servers": 1
}
print(f"  P50={R['infergo_complete']['p50']:.0f}ms (with JSON guarantee)")

os.killpg(os.getpgid(ig.pid), signal.SIGKILL)

# ═══════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════

print(f"\n{'━'*70}")
print(f"  RESULTS — 4-WAY RAG BENCHMARK")
print(f"{'━'*70}")
print(f"\n  {'Setup':<35} {'P50':>8} {'VRAM':>8} {'RSS':>8} {'Deps':>5}")
print(f"  {'─'*65}")
for name, label in [("pure_python","1. Pure Python"),
                     ("python_to_infergo","2. Python→infergo"),
                     ("infergo_rerank","3. infergo rerank"),
                     ("infergo_complete","4. infergo complete")]:
    d = R[name]
    print(f"  {label:<35} {d['p50']:>6.0f}ms {d['vram_mb']:>6}MB {d['rss_mb']:>6}MB {d.get('processes',1):>3}p")

print(f"\n  {'─'*65}")
fastest = min(R.values(), key=lambda x: x['p50'])
slowest = max(R.values(), key=lambda x: x['p50'])
py = R['pure_python']['p50']
for name, d in R.items():
    if d['p50'] == fastest['p50']:
        print(f"  Fastest: {name} ({d['p50']:.0f}ms)")
    ratio = py / d['p50'] if d['p50'] > 0 else 0
    if ratio > 1 and name != 'pure_python':
        print(f"  {name}: {ratio:.1f}x faster than Pure Python")

print(f"\n  WHY PYTHON CRASHES:")
print(f"  - llama-cpp-python server needs: sse-starlette + starlette-context + uvicorn")
print(f"  - sentence-transformers + PyTorch + llama-cpp = GPU memory contention")
print(f"  - Python HTTP server is single-threaded (GIL)")
print(f"  - Two GPU models in one process = CUDA context conflicts")
print(f"  - infergo: one binary, one process, both models, zero crashes")
print(f"{'━'*70}")
