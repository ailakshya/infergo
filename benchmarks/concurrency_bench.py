#!/usr/bin/env python3
"""
THROUGHPUT & CONCURRENCY BENCHMARK
infergo vs Python at c=1, c=4, c=8, c=16
LLM, Embedding, Detection — measures req/s, tok/s, P50 under load
"""

import time, statistics, os, subprocess, signal, json, io, sys
import concurrent.futures
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

INFERGO = os.path.expanduser("~/cgo/infergo")
CWD = os.path.expanduser("~/cgo")
LLM_MODEL = "/tmp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
EMBED_MODEL = os.path.expanduser("~/cgo/models/all-MiniLM-L6-v2/onnx/model.onnx")
DET_PT = os.path.expanduser("~/cgo/models/yolo11n.torchscript.pt")

REQUESTS_PER_WORKER = 10
WARMUP_REQUESTS = 5

def banner(t):
    print(f"\n{'━'*70}")
    print(f"  {t}")
    print(f"{'━'*70}")

def start_server(model_spec, provider, port, extra=None):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{os.path.expanduser('~/onnxruntime/lib')}:{env.get('LD_LIBRARY_PATH','')}"
    args = [INFERGO, "serve", f"--model={model_spec}", f"--provider={provider}",
            f"--port={port}", "--grpc-port=0"]
    if extra: args.extend(extra)
    proc = subprocess.Popen(args, cwd=CWD, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    for i in range(60):
        try:
            r = subprocess.run(["curl","-s",f"http://localhost:{port}/health/live"],
                             capture_output=True, timeout=2)
            if r.returncode == 0: return proc
        except: pass
        time.sleep(1)
    return proc

def stop(proc):
    try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL); proc.wait(5)
    except: pass
    time.sleep(2)

def http_worker(url, data, n_requests):
    """Single worker: sends n_requests sequentially, returns list of latencies."""
    import urllib.request
    times = []
    for _ in range(n_requests):
        req = urllib.request.Request(url, data=data.encode(),
            headers={"Content-Type": "application/json"})
        s = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read()
        except Exception as e:
            times.append(float('inf'))
            continue
        times.append((time.perf_counter()-s)*1000)
    return times

def http_binary_worker(url, filepath, n_requests):
    """Single worker for binary detection endpoint."""
    import urllib.request
    with open(filepath, "rb") as f:
        img_data = f.read()
    times = []
    for _ in range(n_requests):
        req = urllib.request.Request(url, data=img_data,
            headers={"Content-Type": "application/octet-stream"})
        s = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()
        except:
            times.append(float('inf'))
            continue
        times.append((time.perf_counter()-s)*1000)
    return times

def run_concurrent(url, data, concurrency, n_per_worker=REQUESTS_PER_WORKER, binary_file=None):
    """Run concurrent HTTP requests, return all latencies."""
    # Warmup
    for _ in range(WARMUP_REQUESTS):
        if binary_file:
            http_binary_worker(url, binary_file, 1)
        else:
            http_worker(url, data, 1)

    all_times = []
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        if binary_file:
            futures = [pool.submit(http_binary_worker, url, binary_file, n_per_worker)
                      for _ in range(concurrency)]
        else:
            futures = [pool.submit(http_worker, url, data, n_per_worker)
                      for _ in range(concurrency)]
        for f in concurrent.futures.as_completed(futures):
            all_times.extend(f.result())
    wall = (time.perf_counter()-start)*1000

    valid = [t for t in all_times if t < float('inf')]
    if not valid:
        return {"p50": 0, "rps": 0, "errors": len(all_times)}

    total_reqs = len(valid)
    return {
        "p50": statistics.median(valid),
        "avg": statistics.mean(valid),
        "rps": total_reqs / (wall/1000),
        "total": total_reqs,
        "errors": len(all_times) - len(valid),
        "wall_ms": wall
    }


# ═══════════════════════════════════════════════════════════════
print("╔══════════════════════════════════════════════════════════════╗")
print("║  THROUGHPUT & CONCURRENCY BENCHMARK                         ║")
print("║  infergo vs Python — c=1, c=4, c=8, c=16                   ║")
print("╚══════════════════════════════════════════════════════════════╝")

CONCURRENCIES = [1, 4, 8, 16]
PROMPT = "What is machine learning?"
MAX_TOK = 32

# ═══════════════════════════════════════════════════════════════
banner("1. LLM THROUGHPUT")
# ═══════════════════════════════════════════════════════════════

# infergo
print("\n  ▸ infergo (continuous batching, CUDA)")
proc = start_server(f"llm:{LLM_MODEL}", "cuda", 9500,
    extra=["--max-seqs","32","--ctx-size","8192"])
req = json.dumps({"model":"llm","messages":[{"role":"user","content":PROMPT}],"max_tokens":MAX_TOK})

ig_llm = {}
for c in CONCURRENCIES:
    r = run_concurrent(f"http://localhost:9500/v1/chat/completions", req, c, n_per_worker=8)
    ig_llm[c] = r
    print(f"    c={c:2d}: P50={r['p50']:7.0f}ms  {r['rps']:5.1f} req/s  errors={r['errors']}")
stop(proc)

# Python
print("\n  ▸ Python llama-cpp-python (GIL-locked, CUDA)")
py_srv = subprocess.Popen(
    ["python3","-m","llama_cpp.server","--model",LLM_MODEL,"--n_gpu_layers","99",
     "--n_ctx","8192","--host","0.0.0.0","--port","8500"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
for i in range(60):
    try:
        r = subprocess.run(["curl","-s","http://localhost:8500/v1/models"], capture_output=True, timeout=2)
        if r.returncode == 0 and b"model" in r.stdout: break
    except: pass
    time.sleep(1)

req_py = json.dumps({"model":"default","messages":[{"role":"user","content":PROMPT}],"max_tokens":MAX_TOK})
py_llm = {}
for c in CONCURRENCIES:
    r = run_concurrent(f"http://localhost:8500/v1/chat/completions", req_py, c, n_per_worker=8)
    py_llm[c] = r
    print(f"    c={c:2d}: P50={r['p50']:7.0f}ms  {r['rps']:5.1f} req/s  errors={r['errors']}")
stop(py_srv)

# ═══════════════════════════════════════════════════════════════
banner("2. EMBEDDING THROUGHPUT")
# ═══════════════════════════════════════════════════════════════

TEXTS = ["The quick brown fox.", "Machine learning.", "Go is compiled."]

print("\n  ▸ infergo ONNX CUDA (HTTP)")
if os.path.exists(EMBED_MODEL):
    proc = start_server(f"embed:{EMBED_MODEL}", "cuda", 9501)
    req = json.dumps({"model":"embed","input":TEXTS})
    ig_emb = {}
    for c in CONCURRENCIES:
        r = run_concurrent(f"http://localhost:9501/v1/embeddings", req, c, n_per_worker=20)
        ig_emb[c] = r
        print(f"    c={c:2d}: P50={r['p50']:7.1f}ms  {r['rps']:5.0f} req/s")
    stop(proc)

# ═══════════════════════════════════════════════════════════════
banner("3. DETECTION THROUGHPUT")
# ═══════════════════════════════════════════════════════════════

img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
from PIL import Image
pil = Image.fromarray(img)
buf = io.BytesIO(); pil.save(buf, format="JPEG", quality=85)
with open("/tmp/conc_bench.jpg","wb") as f: f.write(buf.getvalue())

print("\n  ▸ infergo TorchScript (HTTP binary, CUDA)")
if os.path.exists(DET_PT):
    proc = start_server(f"detect:{DET_PT}", "cuda", 9502)
    ig_det = {}
    for c in CONCURRENCIES:
        r = run_concurrent(f"http://localhost:9502/v1/detect/binary?model=detect&conf=0.25",
                          None, c, n_per_worker=20, binary_file="/tmp/conc_bench.jpg")
        ig_det[c] = r
        print(f"    c={c:2d}: P50={r['p50']:7.1f}ms  {r['rps']:5.0f} req/s")
    stop(proc)

# ═══════════════════════════════════════════════════════════════
banner("FINAL — THROUGHPUT COMPARISON")
# ═══════════════════════════════════════════════════════════════

print(f"\n  LLM Throughput (req/s)")
print(f"  {'c':>4}  {'infergo':>10}  {'Python':>10}  {'Speedup':>10}")
print(f"  {'─'*40}")
for c in CONCURRENCIES:
    ig = ig_llm.get(c, {}).get("rps", 0)
    py = py_llm.get(c, {}).get("rps", 0)
    sp = f"{ig/py:.1f}x" if py > 0 else "—"
    print(f"  {c:>4}  {ig:>8.1f}  {py:>8.1f}  {sp:>10}")

print(f"\n  LLM P50 Latency (ms)")
print(f"  {'c':>4}  {'infergo':>10}  {'Python':>10}  {'Winner':>10}")
print(f"  {'─'*40}")
for c in CONCURRENCIES:
    ig = ig_llm.get(c, {}).get("p50", 0)
    py = py_llm.get(c, {}).get("p50", 0)
    if ig > 0 and py > 0:
        w = f"ig {py/ig:.1f}x" if ig < py else f"py {ig/py:.1f}x"
    else: w = "—"
    print(f"  {c:>4}  {ig:>8.0f}  {py:>8.0f}  {w:>10}")

if 'ig_emb' in dir():
    print(f"\n  Embedding Throughput (req/s)")
    print(f"  {'c':>4}  {'infergo':>10}")
    print(f"  {'─'*20}")
    for c in CONCURRENCIES:
        ig = ig_emb.get(c, {}).get("rps", 0)
        print(f"  {c:>4}  {ig:>8.0f}")

if 'ig_det' in dir():
    print(f"\n  Detection Throughput (req/s)")
    print(f"  {'c':>4}  {'infergo':>10}")
    print(f"  {'─'*20}")
    for c in CONCURRENCIES:
        ig = ig_det.get(c, {}).get("rps", 0)
        print(f"  {c:>4}  {ig:>8.0f}")

print(f"\n  KEY INSIGHT:")
ig_c1 = ig_llm.get(1, {}).get("rps", 0)
ig_c16 = ig_llm.get(16, {}).get("rps", 0)
py_c1 = py_llm.get(1, {}).get("rps", 0)
py_c16 = py_llm.get(16, {}).get("rps", 0)
if ig_c1 > 0 and ig_c16 > 0:
    print(f"  infergo scales: {ig_c1:.1f} → {ig_c16:.1f} req/s (c=1→16) = {ig_c16/ig_c1:.1f}x")
if py_c1 > 0 and py_c16 > 0:
    print(f"  Python  scales: {py_c1:.1f} → {py_c16:.1f} req/s (c=1→16) = {py_c16/py_c1:.1f}x")
if ig_c16 > 0 and py_c16 > 0:
    print(f"  At c=16: infergo {ig_c16/py_c16:.1f}x faster than Python")

print(f"\n{'━'*70}")
