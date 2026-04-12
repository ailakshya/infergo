#!/usr/bin/env python3
"""
SCALABILITY BENCHMARK: Every feature at c=1, c=4, c=8, c=16
All GPU, all HTTP, measures req/s and P50 under load.
"""

import time, statistics, os, subprocess, signal, json, io, sys
import concurrent.futures
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

INFERGO = os.path.expanduser("~/cgo/infergo")
CWD = os.path.expanduser("~/cgo")
LLM = "/tmp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
EMBED = os.path.expanduser("~/cgo/models/all-MiniLM-L6-v2/onnx/model.onnx")
DET_PT = os.path.expanduser("~/cgo/models/yolo11n.torchscript.pt")

CONC = [1, 4, 8, 16]
REQ_PER_WORKER = 8

def start(spec, prov, port, extra=None):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{os.path.expanduser('~/onnxruntime/lib')}:{env.get('LD_LIBRARY_PATH','')}"
    args = [INFERGO,"serve",f"--model={spec}",f"--provider={prov}",f"--port={port}","--grpc-port=0"]
    if extra: args.extend(extra)
    p = subprocess.Popen(args, cwd=CWD, env=env, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    for i in range(60):
        try:
            r = subprocess.run(["curl","-s",f"http://localhost:{port}/health/live"],
                capture_output=True, timeout=2)
            if r.returncode == 0: return p
        except: pass
        time.sleep(1)
    return p

def stop(p):
    try: os.killpg(os.getpgid(p.pid), signal.SIGKILL); p.wait(5)
    except: pass
    time.sleep(2)

def worker(url, payload, n):
    import urllib.request
    times = []
    for _ in range(n):
        req = urllib.request.Request(url, data=payload,
            headers={"Content-Type":"application/json"})
        s = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as r: r.read()
            times.append((time.perf_counter()-s)*1000)
        except: times.append(float('inf'))
    return times

def bin_worker(url, fpath, n):
    import urllib.request
    with open(fpath,"rb") as f: data = f.read()
    times = []
    for _ in range(n):
        req = urllib.request.Request(url, data=data,
            headers={"Content-Type":"application/octet-stream"})
        s = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as r: r.read()
            times.append((time.perf_counter()-s)*1000)
        except: times.append(float('inf'))
    return times

def run_conc(url, payload, c, n=REQ_PER_WORKER, binary_file=None):
    # warmup
    for _ in range(3):
        if binary_file: bin_worker(url, binary_file, 1)
        else: worker(url, payload, 1)

    all_t = []
    wall_s = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
        if binary_file:
            futs = [pool.submit(bin_worker, url, binary_file, n) for _ in range(c)]
        else:
            futs = [pool.submit(worker, url, payload, n) for _ in range(c)]
        for f in concurrent.futures.as_completed(futs):
            all_t.extend(f.result())
    wall = (time.perf_counter()-wall_s)

    good = [t for t in all_t if t < float('inf')]
    if not good: return {"p50":0,"rps":0,"err":len(all_t)}
    return {
        "p50": statistics.median(good),
        "rps": len(good)/wall,
        "err": len(all_t)-len(good),
        "total": len(good)
    }


print("╔══════════════════════════════════════════════════════════════════╗")
print("║  SCALABILITY BENCHMARK — ALL FEATURES AT c=1,4,8,16            ║")
print("║  All GPU, HTTP, RTX 5070 Ti                                     ║")
print("╚══════════════════════════════════════════════════════════════════╝")

results = {}

def bench_feature(name, spec, prov, port, url_path, payload, extra=None, binary_file=None):
    print(f"\n  {'━'*60}")
    print(f"  {name}")
    print(f"  {'━'*60}")
    p = start(spec, prov, port, extra)
    data = {}
    print(f"  {'c':>4}  {'P50':>10}  {'req/s':>10}  {'errors':>8}")
    print(f"  {'─'*40}")
    for c in CONC:
        url = f"http://localhost:{port}{url_path}"
        r = run_conc(url, payload, c, binary_file=binary_file)
        data[c] = r
        print(f"  {c:>4}  {r['p50']:>8.0f}ms  {r['rps']:>8.1f}  {r['err']:>8}")
    stop(p)
    results[name] = data
    return data


# ═══════════════════════════════════════════════════════════
# 1. LLM
# ═══════════════════════════════════════════════════════════

llm_payload = json.dumps({
    "model":"llm",
    "messages":[{"role":"user","content":"What is 2+2?"}],
    "max_tokens":16
}).encode()

bench_feature("LLM Generation",
    f"llm:{LLM}", "cuda", 9600, "/v1/chat/completions", llm_payload,
    extra=["--max-seqs","32","--ctx-size","8192"])

# ═══════════════════════════════════════════════════════════
# 2. LLM + JSON mode
# ═══════════════════════════════════════════════════════════

json_payload = json.dumps({
    "model":"llm",
    "messages":[{"role":"user","content":"Return JSON with answer=4"}],
    "max_tokens":32,
    "response_format":{"type":"json_object"}
}).encode()

bench_feature("LLM JSON Mode",
    f"llm:{LLM}", "cuda", 9601, "/v1/chat/completions", json_payload,
    extra=["--max-seqs","32","--ctx-size","8192"])

# ═══════════════════════════════════════════════════════════
# 3. Embeddings (single)
# ═══════════════════════════════════════════════════════════

emb_payload = json.dumps({"model":"embed","input":"hello world"}).encode()

if os.path.exists(EMBED):
    bench_feature("Embedding (single text)",
        f"embed:{EMBED}", "cuda", 9602, "/v1/embeddings", emb_payload)

# ═══════════════════════════════════════════════════════════
# 4. Embeddings (batch)
# ═══════════════════════════════════════════════════════════

batch_payload = json.dumps({
    "model":"embed",
    "input":["quick fox","ML great","Go fast"]
}).encode()

if os.path.exists(EMBED):
    bench_feature("Embedding (batch 3 texts)",
        f"embed:{EMBED}", "cuda", 9603, "/v1/embeddings", batch_payload)

# ═══════════════════════════════════════════════════════════
# 5. Detection (TorchScript)
# ═══════════════════════════════════════════════════════════

from PIL import Image
img = np.random.randint(0,255,(640,640,3),dtype=np.uint8)
pil = Image.fromarray(img)
buf = io.BytesIO(); pil.save(buf,format="JPEG",quality=85)
with open("/tmp/scale_bench.jpg","wb") as f: f.write(buf.getvalue())

if os.path.exists(DET_PT):
    bench_feature("Detection (TorchScript)",
        f"detect:{DET_PT}", "cuda", 9604,
        "/v1/detect/binary?model=detect&conf=0.25", None,
        binary_file="/tmp/scale_bench.jpg")

# ═══════════════════════════════════════════════════════════
# 6. Reranking
# ═══════════════════════════════════════════════════════════

rerank_payload = json.dumps({
    "model":"embed",
    "query":"what is machine learning",
    "documents":["ML is AI","cats are fluffy","deep learning uses nets"],
    "top_n":2
}).encode()

if os.path.exists(EMBED):
    bench_feature("Reranking (3 docs)",
        f"embed:{EMBED}", "cuda", 9605, "/v1/rerank", rerank_payload)

# ═══════════════════════════════════════════════════════════
# 7. Vector Search
# ═══════════════════════════════════════════════════════════

# Vector search doesn't have a server mode yet — skip concurrent bench
# Use the in-process number: 0.17ms per query

# ═══════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════

print(f"\n{'━'*70}")
print(f"  SCALABILITY SUMMARY — req/s at each concurrency level")
print(f"{'━'*70}")

print(f"\n  {'Feature':<28} {'c=1':>8} {'c=4':>8} {'c=8':>8} {'c=16':>8} {'Scale':>8}")
print(f"  {'─'*68}")

for name, data in results.items():
    r1 = data.get(1,{}).get("rps",0)
    r4 = data.get(4,{}).get("rps",0)
    r8 = data.get(8,{}).get("rps",0)
    r16 = data.get(16,{}).get("rps",0)
    scale = f"{r16/r1:.1f}x" if r1 > 0 and r16 > 0 else "—"
    print(f"  {name:<28} {r1:>7.0f} {r4:>7.0f} {r8:>7.0f} {r16:>7.0f} {scale:>8}")

print(f"\n  {'Feature':<28} {'c=1':>8} {'c=4':>8} {'c=8':>8} {'c=16':>8}")
print(f"  {'─'*60}")
print(f"  {'':28} {'P50 latency (ms)':^36}")
print(f"  {'─'*60}")

for name, data in results.items():
    p1 = data.get(1,{}).get("p50",0)
    p4 = data.get(4,{}).get("p50",0)
    p8 = data.get(8,{}).get("p50",0)
    p16 = data.get(16,{}).get("p50",0)
    print(f"  {name:<28} {p1:>7.0f} {p4:>7.0f} {p8:>7.0f} {p16:>7.0f}")

print(f"\n  HNSW vector search: 0.17ms per query (in-process, not HTTP)")
print(f"\n{'━'*70}")
