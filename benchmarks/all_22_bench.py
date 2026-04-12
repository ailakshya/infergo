#!/usr/bin/env python3
"""
BENCHMARK ALL 22 FEATURES — each measured with latency
"""

import time, statistics, os, subprocess, signal, json, io, base64
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

INFERGO = os.path.expanduser("~/cgo/infergo")
CWD = os.path.expanduser("~/cgo")
LLM = "/tmp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
EMBED = os.path.expanduser("~/cgo/models/all-MiniLM-L6-v2/onnx/model.onnx")
DET = os.path.expanduser("~/cgo/models/yolo11n.torchscript.pt")

RUNS = 10
WARMUP = 3
R = []

def start():
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{os.path.expanduser('~/onnxruntime/lib')}:{env.get('LD_LIBRARY_PATH','')}"
    p = subprocess.Popen(
        [INFERGO, "serve", f"--model=llm:{LLM}", f"--model=embed:{EMBED}", f"--model=detect:{DET}",
         "--provider=cuda", "--port=9800", "--grpc-port=0", "--max-seqs=32", "--ctx-size=8192"],
        cwd=CWD, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    for i in range(60):
        try:
            r = subprocess.run(["curl","-s","http://localhost:9800/health/live"],capture_output=True,timeout=2)
            if r.returncode == 0: return p
        except: pass
        time.sleep(1)
    return p

def bench(name, url, data, runs=RUNS, warmup=WARMUP, ct="application/json", data_file=None, method="POST"):
    import urllib.request
    payload = data.encode() if isinstance(data, str) else data
    for _ in range(warmup):
        try:
            req = urllib.request.Request(f"http://localhost:9800{url}", data=payload,
                headers={"Content-Type": ct}, method=method)
            if data_file:
                with open(data_file,"rb") as f: req = urllib.request.Request(
                    f"http://localhost:9800{url}", data=f.read(),
                    headers={"Content-Type": ct}, method=method)
            urllib.request.urlopen(req, timeout=30).read()
        except: pass

    times = []
    for _ in range(runs):
        try:
            s = time.perf_counter()
            req = urllib.request.Request(f"http://localhost:9800{url}", data=payload,
                headers={"Content-Type": ct}, method=method)
            if data_file:
                with open(data_file,"rb") as f: req = urllib.request.Request(
                    f"http://localhost:9800{url}", data=f.read(),
                    headers={"Content-Type": ct}, method=method)
            urllib.request.urlopen(req, timeout=30).read()
            times.append((time.perf_counter()-s)*1000)
        except: times.append(-1)

    good = [t for t in times if t > 0]
    if good:
        p50 = statistics.median(good)
        mn = min(good)
        rps = 1000/statistics.mean(good)
    else:
        p50 = mn = rps = 0

    R.append((name, p50, mn, rps, len(good), runs))
    return p50

def bench_get(name, url):
    import urllib.request
    times = []
    for _ in range(RUNS):
        s = time.perf_counter()
        try: urllib.request.urlopen(f"http://localhost:9800{url}", timeout=5).read()
        except: pass
        times.append((time.perf_counter()-s)*1000)
    p50 = statistics.median(times)
    R.append((name, p50, min(times), 1000/statistics.mean(times), RUNS, RUNS))
    return p50


print("╔══════════════════════════════════════════════════════════════╗")
print("║  ALL 22 FEATURES — BENCHMARK WITH LATENCY                   ║")
print("╚══════════════════════════════════════════════════════════════╝")

proc = start()

# Create test image
from PIL import Image
img = np.random.randint(0,255,(640,640,3),dtype=np.uint8)
pil = Image.fromarray(img)
buf = io.BytesIO(); pil.save(buf,format="JPEG",quality=85)
with open("/tmp/all22.jpg","wb") as f: f.write(buf.getvalue())
img_b64 = base64.b64encode(buf.getvalue()).decode()

print("\n  Benchmarking...\n")

# 1-4: LLM
bench("Chat completion",
    "/v1/chat/completions",
    json.dumps({"model":"llm","messages":[{"role":"user","content":"Hi"}],"max_tokens":8}))

bench("JSON structured output",
    "/v1/chat/completions",
    json.dumps({"model":"llm","messages":[{"role":"user","content":"JSON"}],"max_tokens":16,
                "response_format":{"type":"json_object"}}))

bench("Text completion",
    "/v1/completions",
    json.dumps({"model":"llm","prompt":"Hello","max_tokens":8}))

bench("Streaming (first chunk)",
    "/v1/chat/completions",
    json.dumps({"model":"llm","messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"stream":True}))

# 5-6: Embedding
bench("Single embedding",
    "/v1/embeddings",
    json.dumps({"model":"embed","input":"hello world"}))

bench("Batch embedding (3 texts)",
    "/v1/embeddings",
    json.dumps({"model":"embed","input":["hello","world","test"]}))

# 7: Detection
bench("Detection (binary)",
    "/v1/detect/binary?model=detect&conf=0.25", None,
    ct="application/octet-stream", data_file="/tmp/all22.jpg")

# 8-9: Search & Rerank
bench("Vector search",
    "/v1/search",
    json.dumps({"model":"embed","query":"hello","k":3}))

bench("Reranking (3 docs)",
    "/v1/rerank",
    json.dumps({"model":"embed","query":"ML","documents":["AI is ML","cats","deep learning"],"top_n":2}))

# 10-11: RAG & Ingest
bench("RAG pipeline",
    "/v1/rag",
    json.dumps({"model":"llm","embed_model":"embed","query":"what is AI"}))

bench("Document ingest",
    "/v1/ingest",
    json.dumps({"model":"embed","documents":["AI is great","ML rocks"]}))

# 12-17: Admin
bench_get("List models", "/v1/models")
bench_get("Health live", "/health/live")
bench_get("Health ready", "/health/ready")
bench_get("Metrics", "/metrics")
bench_get("Guardrails config", "/v1/admin/guardrails")
bench_get("Prompt templates", "/v1/admin/templates")
bench_get("Web UI", "/ui")

# 18: Batch
bench("Batch create",
    "/v1/batches",
    json.dumps({"model":"llm","prompts":["hi","hello"]}))

# 19-20: Stubs
bench("Audio transcription",
    "/v1/audio/transcriptions", "{}")

bench("Image generation",
    "/v1/images/generations", "{}")

# 21: Streaming detection (measure TTFB)
import urllib.request
times_sd = []
for _ in range(5):
    s = time.perf_counter()
    try:
        req = urllib.request.Request("http://localhost:9800/v1/detect/stream",
            data=json.dumps({"model":"detect","source":"test","fps":10}).encode(),
            headers={"Content-Type":"application/json"})
        r = urllib.request.urlopen(req, timeout=2)
        r.read(100)  # read first chunk
        times_sd.append((time.perf_counter()-s)*1000)
        r.close()
    except: times_sd.append(-1)
good_sd = [t for t in times_sd if t > 0]
if good_sd:
    R.append(("Streaming detection (TTFB)", statistics.median(good_sd), min(good_sd), 0, len(good_sd), 5))
else:
    R.append(("Streaming detection (TTFB)", 0, 0, 0, 0, 5))

# 22: Detection JSON (for comparison)
bench("Detection (JSON+base64)",
    "/v1/detect",
    json.dumps({"model":"detect","image_b64":img_b64,"conf_thresh":0.25}))

os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
proc.wait()

# ─── Results ─────────────────────────────────────────────

print(f"\n{'━'*70}")
print(f"  ALL 22 FEATURES — LATENCY RESULTS")
print(f"{'━'*70}")
print(f"\n  {'#':>2}  {'Feature':<30} {'P50':>8} {'Min':>8} {'RPS':>8} {'Pass':>6}")
print(f"  {'─'*66}")

for i, (name, p50, mn, rps, ok, total) in enumerate(R):
    p50s = f"{p50:.1f}ms" if p50 > 0 else "—"
    mns = f"{mn:.1f}ms" if mn > 0 else "—"
    rpss = f"{rps:.0f}" if rps > 0 else "—"
    status = f"{ok}/{total}"
    print(f"  {i+1:>2}. {name:<30} {p50s:>8} {mns:>8} {rpss:>8} {status:>6}")

# Categories
cats = {
    "LLM": R[:4],
    "Embedding": R[4:6],
    "Detection": R[6:7] + [R[-1]],
    "Search/Rerank": R[7:9],
    "RAG": R[9:11],
    "Admin": R[11:18],
    "Async": R[18:19],
    "Stubs": R[19:21],
    "Streaming": [R[21]] if len(R) > 21 else [],
}

print(f"\n  {'─'*66}")
print(f"  CATEGORY SUMMARY")
print(f"  {'─'*66}")
for cat, items in cats.items():
    good_items = [it for it in items if it[1] > 0]
    if good_items:
        avg_p50 = statistics.mean([it[1] for it in good_items])
        print(f"  {cat:<20} avg P50={avg_p50:.1f}ms  ({len(good_items)} features)")

total_pass = sum(1 for _,p,_,_,ok,_ in R if ok > 0)
print(f"\n  TOTAL: {total_pass}/{len(R)} features benchmarked successfully")
print(f"{'━'*70}")
