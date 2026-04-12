#!/usr/bin/env python3
"""
ULTIMATE BENCHMARK — Every feature, every concurrency level, long runs.
infergo vs Python counterparts. 100 requests per test, c=1,4,8,16,32.
"""

import time, statistics, os, subprocess, signal, json, io, sys
import http.client
import concurrent.futures
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

INFERGO = os.path.expanduser("~/cgo/infergo")
CWD = os.path.expanduser("~/cgo")
LLM = "/tmp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
EMBED = os.path.expanduser("~/cgo/models/all-MiniLM-L6-v2/onnx/model.onnx")
DET = os.path.expanduser("~/cgo/models/yolo11n.torchscript.pt")

RUNS = 100
WARMUP = 20
CONC = [1, 4, 8, 16, 32]

ALL = []

def banner(n, t):
    print(f"\n{'━'*70}")
    print(f"  {n}. {t}")
    print(f"{'━'*70}")

def start_ig(spec, prov, port, extra=None):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{os.path.expanduser('~/onnxruntime/lib')}:{env.get('LD_LIBRARY_PATH','')}"
    args = [INFERGO,"serve",f"--model={spec}",f"--provider={prov}",f"--port={port}","--grpc-port=0"]
    if extra: args.extend(extra)
    p = subprocess.Popen(args, cwd=CWD, env=env, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    for i in range(60):
        try:
            r = subprocess.run(["curl","-s",f"http://localhost:{port}/health/live"],capture_output=True,timeout=2)
            if r.returncode == 0: return p
        except: pass
        time.sleep(1)
    return p

def start_py(cmd, port, check_path="/v1/models"):
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    for i in range(60):
        try:
            c = http.client.HTTPConnection("localhost", port)
            c.request("GET", check_path)
            if c.getresponse().status == 200: c.close(); return p
            c.close()
        except: pass
        time.sleep(1)
    return p

def stop(p):
    try: os.killpg(os.getpgid(p.pid), signal.SIGKILL); p.wait(5)
    except: pass
    time.sleep(3)

def http_worker(host, port, path, body, n):
    conn = http.client.HTTPConnection(host, port)
    h = {"Content-Type":"application/json","Connection":"keep-alive"}
    times = []
    for _ in range(n):
        try:
            s = time.perf_counter()
            conn.request("POST", path, body, h)
            conn.getresponse().read()
            times.append((time.perf_counter()-s)*1000)
        except:
            times.append(-1)
            try: conn = http.client.HTTPConnection(host, port)
            except: pass
    conn.close()
    return times

def bench_conc(host, port, path, body, concurrency, total=RUNS):
    per_worker = max(total // concurrency, 5)
    # Warmup
    for _ in range(min(WARMUP, 10)):
        try:
            c = http.client.HTTPConnection(host, port)
            c.request("POST", path, body, {"Content-Type":"application/json"})
            c.getresponse().read(); c.close()
        except: pass

    all_t = []
    wall_s = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(http_worker, host, port, path, body, per_worker) for _ in range(concurrency)]
        for f in concurrent.futures.as_completed(futs):
            all_t.extend(f.result())
    wall = time.perf_counter() - wall_s

    good = [t for t in all_t if t > 0]
    if not good: return {"p50":0,"p99":0,"avg":0,"min":0,"max":0,"rps":0,"err":len(all_t),"total":0}
    return {
        "p50": statistics.median(good),
        "p99": sorted(good)[min(int(len(good)*0.99), len(good)-1)],
        "avg": statistics.mean(good),
        "min": min(good),
        "max": max(good),
        "rps": len(good)/wall,
        "err": len(all_t)-len(good),
        "total": len(good)
    }

def row(name, ig, py):
    ALL.append((name, ig, py))

print("╔══════════════════════════════════════════════════════════════════════╗")
print("║  ULTIMATE BENCHMARK — 100 req/test, c=1,4,8,16,32, long runs      ║")
print("║  infergo vs Python — both HTTP — RTX 5070 Ti                       ║")
print("╚══════════════════════════════════════════════════════════════════════╝")

# ═══════════════════════════════════════════════════════════════
banner(1, "LLM CHAT — infergo vs llama-cpp-python (both HTTP)")
# ═══════════════════════════════════════════════════════════════

llm_body = json.dumps({"model":"llm","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":16})
llm_body_py = json.dumps({"model":"default","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":16})

# infergo
ig = start_ig(f"llm:{LLM}", "cuda", 9800, ["--max-seqs","32","--ctx-size","8192"])
ig_results = {}
for c in CONC:
    r = bench_conc("localhost", 9800, "/v1/chat/completions", llm_body, c)
    ig_results[c] = r
    print(f"  infergo c={c:>2}: P50={r['p50']:>7.0f}ms  P99={r['p99']:>7.0f}ms  {r['rps']:>6.1f} req/s  err={r['err']}")
stop(ig)

# Python
py = start_py(["python3","-m","llama_cpp.server","--model",LLM,"--n_gpu_layers","99",
               "--n_ctx","8192","--host","0.0.0.0","--port","8800"], 8800)
py_results = {}
for c in CONC:
    r = bench_conc("localhost", 8800, "/v1/chat/completions", llm_body_py, c, total=50)
    py_results[c] = r
    print(f"  Python  c={c:>2}: P50={r['p50']:>7.0f}ms  P99={r['p99']:>7.0f}ms  {r['rps']:>6.1f} req/s  err={r['err']}")
stop(py)

row("LLM chat", ig_results, py_results)

# ═══════════════════════════════════════════════════════════════
banner(2, "EMBEDDING — infergo vs sentence-transformers (both HTTP)")
# ═══════════════════════════════════════════════════════════════

emb_body = json.dumps({"model":"embed","input":["quick fox","ML great","Go fast"]})
emb_body_py = json.dumps({"input":["quick fox","ML great","Go fast"]})

ig = start_ig(f"embed:{EMBED}", "cuda", 9801)
ig_emb = {}
for c in CONC:
    r = bench_conc("localhost", 9801, "/v1/embeddings", emb_body, c)
    ig_emb[c] = r
    print(f"  infergo c={c:>2}: P50={r['p50']:>7.1f}ms  P99={r['p99']:>7.1f}ms  {r['rps']:>6.0f} req/s  err={r['err']}")
stop(ig)

# Python embedding server
py_embed_code = '''import json,sys;from http.server import HTTPServer,BaseHTTPRequestHandler;from sentence_transformers import SentenceTransformer
model=SentenceTransformer("all-MiniLM-L6-v2",device="cuda")
class H(BaseHTTPRequestHandler):
 def do_POST(self):
  data=json.loads(self.rfile.read(int(self.headers["Content-Length"])));inp=data.get("input",[])
  if isinstance(inp,str):inp=[inp]
  vecs=model.encode(inp).tolist();resp=json.dumps({"data":[{"embedding":v}for v in vecs]}).encode()
  self.send_response(200);self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(resp)));self.end_headers();self.wfile.write(resp)
 def log_message(self,*a):pass
HTTPServer(("0.0.0.0",8801),H).serve_forever()'''
py = subprocess.Popen(["python3","-c",py_embed_code],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,preexec_fn=os.setsid)
time.sleep(8)

py_emb = {}
for c in [1, 4, 8]:  # Python HTTP server can't handle c=16+
    r = bench_conc("localhost", 8801, "/", emb_body_py, c, total=50)
    py_emb[c] = r
    print(f"  Python  c={c:>2}: P50={r['p50']:>7.1f}ms  P99={r['p99']:>7.1f}ms  {r['rps']:>6.0f} req/s  err={r['err']}")
stop(py)

row("Embedding batch", ig_emb, py_emb)

# ═══════════════════════════════════════════════════════════════
banner(3, "DETECTION — infergo vs ultralytics (infergo HTTP, Python in-process)")
# ═══════════════════════════════════════════════════════════════

from PIL import Image
img = np.random.randint(0,255,(640,640,3),dtype=np.uint8)
pil = Image.fromarray(img)
buf = io.BytesIO(); pil.save(buf,format="JPEG",quality=85)
with open("/tmp/ult_bench.jpg","wb") as f: f.write(buf.getvalue())

# infergo detection at all concurrencies
ig = start_ig(f"detect:{DET}", "cuda", 9802)
ig_det = {}
for c in CONC:
    per_w = max(100//c, 5)
    # Use binary endpoint via subprocess (can't do binary with http.client easily)
    all_t = []
    wall_s = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
        def det_worker(n):
            import urllib.request
            with open("/tmp/ult_bench.jpg","rb") as f: data=f.read()
            times=[]
            for _ in range(n):
                try:
                    req=urllib.request.Request("http://localhost:9802/v1/detect/binary?model=detect&conf=0.25",
                        data=data,headers={"Content-Type":"application/octet-stream"})
                    s=time.perf_counter()
                    urllib.request.urlopen(req,timeout=30).read()
                    times.append((time.perf_counter()-s)*1000)
                except: times.append(-1)
            return times
        futs=[pool.submit(det_worker,per_w) for _ in range(c)]
        for f in concurrent.futures.as_completed(futs): all_t.extend(f.result())
    wall = time.perf_counter()-wall_s
    good=[t for t in all_t if t>0]
    if good:
        r={"p50":statistics.median(good),"p99":sorted(good)[min(int(len(good)*0.99),len(good)-1)],
           "rps":len(good)/wall,"err":len(all_t)-len(good)}
    else: r={"p50":0,"p99":0,"rps":0,"err":len(all_t)}
    ig_det[c] = r
    print(f"  infergo c={c:>2}: P50={r['p50']:>7.1f}ms  P99={r['p99']:>7.1f}ms  {r['rps']:>6.0f} req/s  err={r['err']}")
stop(ig)

row("Detection", ig_det, {})

# ═══════════════════════════════════════════════════════════════
banner(4, "RERANKING — infergo HTTP")
# ═══════════════════════════════════════════════════════════════

rerank_body = json.dumps({"model":"embed","query":"ML","documents":["AI is ML","cats","deep learning","robots","data science"],"top_n":3})
ig = start_ig(f"embed:{EMBED}", "cuda", 9803)
ig_rr = {}
for c in CONC:
    r = bench_conc("localhost", 9803, "/v1/rerank", rerank_body, c)
    ig_rr[c] = r
    print(f"  infergo c={c:>2}: P50={r['p50']:>7.1f}ms  P99={r['p99']:>7.1f}ms  {r['rps']:>6.0f} req/s  err={r['err']}")
stop(ig)

row("Reranking", ig_rr, {})

# ═══════════════════════════════════════════════════════════════
banner(5, "ADMIN ENDPOINTS — infergo")
# ═══════════════════════════════════════════════════════════════

ig = start_ig(f"llm:{LLM}", "cuda", 9804, ["--max-seqs","32"])
admin_tests = [
    ("Health", "/health/live", "GET"),
    ("Models", "/v1/models", "GET"),
    ("Metrics", "/metrics", "GET"),
]
ig_admin = {}
for name, path, method in admin_tests:
    times = []
    conn = http.client.HTTPConnection("localhost", 9804)
    for _ in range(200):
        s = time.perf_counter()
        conn.request(method, path)
        conn.getresponse().read()
        times.append((time.perf_counter()-s)*1000)
    conn.close()
    r = {"p50":statistics.median(times),"p99":sorted(times)[int(len(times)*0.99)],"rps":1000/statistics.mean(times)}
    ig_admin[name] = r
    print(f"  {name:<10}: P50={r['p50']:.2f}ms  P99={r['p99']:.2f}ms  {r['rps']:.0f} req/s")
stop(ig)

# ═══════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════

print(f"\n{'━'*70}")
print(f"  ULTIMATE RESULTS — SCALABILITY TABLE")
print(f"{'━'*70}")

for name, ig_data, py_data in ALL:
    print(f"\n  {name}:")
    print(f"  {'':>6} {'c=1':>12} {'c=4':>12} {'c=8':>12} {'c=16':>12} {'c=32':>12}")

    # infergo row
    vals = []
    for c in CONC:
        r = ig_data.get(c, {})
        if r.get("rps",0) > 0: vals.append(f"{r['rps']:.0f}")
        else: vals.append("—")
    print(f"  {'ig':>6} " + " ".join(f"{v:>12}" for v in vals) + "  req/s")

    # Python row
    if py_data:
        vals = []
        for c in CONC:
            r = py_data.get(c, {})
            if r.get("rps",0) > 0: vals.append(f"{r['rps']:.0f}")
            else: vals.append("—")
        print(f"  {'py':>6} " + " ".join(f"{v:>12}" for v in vals) + "  req/s")

    # Winner at each level
    winners = []
    for c in CONC:
        ig_r = ig_data.get(c,{}).get("rps",0)
        py_r = py_data.get(c,{}).get("rps",0) if py_data else 0
        if ig_r > 0 and py_r > 0:
            if ig_r > py_r: winners.append(f"ig {ig_r/py_r:.1f}x")
            else: winners.append(f"py {py_r/ig_r:.1f}x")
        elif ig_r > 0: winners.append("ig")
        else: winners.append("—")
    print(f"  {'win':>6} " + " ".join(f"{w:>12}" for w in winners))

print(f"\n  Admin endpoints (200 requests each):")
for name, r in ig_admin.items():
    print(f"    {name}: {r['p50']:.2f}ms P50, {r['rps']:.0f} req/s")

# Count wins
total_wins = 0; total_losses = 0
for name, ig_data, py_data in ALL:
    if not py_data: continue
    for c in CONC:
        ig_r = ig_data.get(c,{}).get("rps",0)
        py_r = py_data.get(c,{}).get("rps",0)
        if ig_r > 0 and py_r > 0:
            if ig_r > py_r: total_wins += 1
            else: total_losses += 1

print(f"\n  {'━'*60}")
print(f"  SCORE: infergo {total_wins} — Python {total_losses}")
print(f"  (across all concurrency levels where both have data)")
print(f"{'━'*70}")
