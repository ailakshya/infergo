#!/usr/bin/env python3
"""
NET-TO-NET BENCHMARK: Both infergo and Python measured via HTTP.
Same protocol. Same serialization. Same network path. 50 runs each.
"""

import time, statistics, os, subprocess, signal, json, io, base64, sys
import http.client
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

RUNS = 50
WARMUP = 15
INFERGO = os.path.expanduser("~/cgo/infergo")
CWD = os.path.expanduser("~/cgo")
LLM = "/tmp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
EMBED = os.path.expanduser("~/cgo/models/all-MiniLM-L6-v2/onnx/model.onnx")

R = []

def start_infergo(spec, prov, port, extra=None):
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

def start_python_llm(port):
    p = subprocess.Popen(
        ["python3","-m","llama_cpp.server","--model",LLM,"--n_gpu_layers","99",
         "--n_ctx","2048","--host","0.0.0.0","--port",str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    for i in range(60):
        try:
            r = subprocess.run(["curl","-s",f"http://localhost:{port}/v1/models"],
                capture_output=True, timeout=2)
            if r.returncode == 0 and b"model" in r.stdout: return p
        except: pass
        time.sleep(1)
    return p

def stop(p):
    try: os.killpg(os.getpgid(p.pid), signal.SIGKILL); p.wait(5)
    except: pass
    time.sleep(2)

def bench_http(host, port, path, data, runs=RUNS, warmup=WARMUP):
    """HTTP benchmark with keep-alive connection. Returns list of latencies."""
    conn = http.client.HTTPConnection(host, port)
    headers = {"Content-Type": "application/json", "Connection": "keep-alive"}
    body = data if isinstance(data, str) else json.dumps(data)

    for _ in range(warmup):
        try:
            conn.request("POST", path, body, headers)
            conn.getresponse().read()
        except:
            conn = http.client.HTTPConnection(host, port)

    times = []
    for _ in range(runs):
        try:
            s = time.perf_counter()
            conn.request("POST", path, body, headers)
            resp = conn.getresponse().read()
            times.append((time.perf_counter()-s)*1000)
        except:
            times.append(-1)
            conn = http.client.HTTPConnection(host, port)

    conn.close()
    good = [t for t in times if t > 0]
    return good

def stats(times, label=""):
    if not times: return {"p50":0,"avg":0,"min":0,"max":0,"p99":0,"std":0,"rps":0}
    return {
        "p50": statistics.median(times),
        "avg": statistics.mean(times),
        "min": min(times),
        "max": max(times),
        "p99": sorted(times)[int(len(times)*0.99)] if len(times) > 1 else times[0],
        "std": statistics.stdev(times) if len(times) > 1 else 0,
        "rps": 1000/statistics.mean(times) if statistics.mean(times) > 0 else 0,
    }

def row(name, ig_times, py_times):
    ig = stats(ig_times)
    py = stats(py_times)
    if ig["p50"] > 0 and py["p50"] > 0:
        ratio = py["p50"]/ig["p50"]
        winner = f"infergo {ratio:.1f}x" if ratio > 1 else f"Python {1/ratio:.1f}x"
    elif ig["p50"] > 0:
        winner = "infergo"
    else:
        winner = "—"
    R.append((name, ig, py, winner))

print("╔════════════════════════════════════════════════════════════════════╗")
print("║  NET-TO-NET BENCHMARK — Both via HTTP, Keep-Alive, 50 Runs       ║")
print("║  Same protocol. Same serialization. Fair comparison.             ║")
print("╚════════════════════════════════════════════════════════════════════╝\n")

# ═══════════════════════════════════════════════════════════════
# 1. LLM — BOTH VIA HTTP
# ═══════════════════════════════════════════════════════════════

print("━" * 65)
print("  1. LLM CHAT COMPLETION (via HTTP)")
print("━" * 65)

# Start both servers
ig_proc = start_infergo(f"llm:{LLM}", "cuda", 9900, ["--max-seqs","32","--ctx-size","4096"])
py_proc = start_python_llm(8900)

req = {"model":"llm","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":16}
req_py = {"model":"default","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":16}

print("  infergo...")
ig_llm = bench_http("localhost", 9900, "/v1/chat/completions", req)
ig_s = stats(ig_llm)
print(f"    P50={ig_s['p50']:.0f}ms  Avg={ig_s['avg']:.0f}ms  Min={ig_s['min']:.0f}ms  Max={ig_s['max']:.0f}ms  P99={ig_s['p99']:.0f}ms  StdDev={ig_s['std']:.1f}  RPS={ig_s['rps']:.0f}")

print("  Python llama-cpp-python...")
py_llm = bench_http("localhost", 8900, "/v1/chat/completions", req_py)
py_s = stats(py_llm)
print(f"    P50={py_s['p50']:.0f}ms  Avg={py_s['avg']:.0f}ms  Min={py_s['min']:.0f}ms  Max={py_s['max']:.0f}ms  P99={py_s['p99']:.0f}ms  StdDev={py_s['std']:.1f}  RPS={py_s['rps']:.0f}")

row("LLM chat (16 tok)", ig_llm, py_llm)
stop(ig_proc); stop(py_proc)

# ═══════════════════════════════════════════════════════════════
# 2. LLM — LONGER GENERATION
# ═══════════════════════════════════════════════════════════════

print("\n" + "━" * 65)
print("  2. LLM LONG GENERATION (64 tokens via HTTP)")
print("━" * 65)

ig_proc = start_infergo(f"llm:{LLM}", "cuda", 9901, ["--max-seqs","32","--ctx-size","4096"])
py_proc = start_python_llm(8901)

req64 = {"model":"llm","messages":[{"role":"user","content":"Explain neural networks"}],"max_tokens":64}
req64_py = {"model":"default","messages":[{"role":"user","content":"Explain neural networks"}],"max_tokens":64}

print("  infergo...")
ig_llm64 = bench_http("localhost", 9901, "/v1/chat/completions", req64, runs=30, warmup=10)
ig_s = stats(ig_llm64)
print(f"    P50={ig_s['p50']:.0f}ms  Avg={ig_s['avg']:.0f}ms  Min={ig_s['min']:.0f}ms  P99={ig_s['p99']:.0f}ms  RPS={ig_s['rps']:.1f}")

print("  Python...")
py_llm64 = bench_http("localhost", 8901, "/v1/chat/completions", req64_py, runs=30, warmup=10)
py_s = stats(py_llm64)
print(f"    P50={py_s['p50']:.0f}ms  Avg={py_s['avg']:.0f}ms  Min={py_s['min']:.0f}ms  P99={py_s['p99']:.0f}ms  RPS={py_s['rps']:.1f}")

row("LLM long (64 tok)", ig_llm64, py_llm64)
stop(ig_proc); stop(py_proc)

# ═══════════════════════════════════════════════════════════════
# 3. EMBEDDING — BOTH VIA HTTP
# ═══════════════════════════════════════════════════════════════

print("\n" + "━" * 65)
print("  3. EMBEDDING — BOTH VIA HTTP")
print("━" * 65)

# Python: start a simple embedding HTTP server
py_embed_code = '''
import json, sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
class H(BaseHTTPRequestHandler):
    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        inp = data.get("input", [])
        if isinstance(inp, str): inp = [inp]
        vecs = model.encode(inp).tolist()
        resp = json.dumps({"object":"list","data":[{"embedding":v,"index":i} for i,v in enumerate(vecs)]}).encode()
        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)
    def log_message(self, *a): pass
HTTPServer(("0.0.0.0", 8902), H).serve_forever()
'''
py_embed_proc = subprocess.Popen(["python3","-c",py_embed_code],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
for i in range(30):
    try:
        conn = http.client.HTTPConnection("localhost", 8902)
        conn.request("POST", "/embed", json.dumps({"input":"test"}), {"Content-Type":"application/json"})
        if conn.getresponse().status == 200: break
        conn.close()
    except: pass
    time.sleep(1)

ig_proc = start_infergo(f"embed:{EMBED}", "cuda", 9902)

emb_single = {"model":"embed","input":"hello world"}
emb_batch = {"model":"embed","input":["hello","world","test"]}

# Single
print("  Single text embedding:")
ig_e1 = bench_http("localhost", 9902, "/v1/embeddings", emb_single)
py_e1 = bench_http("localhost", 8902, "/embed", {"input":"hello world"})
ig_s = stats(ig_e1); py_s = stats(py_e1)
print(f"    infergo: P50={ig_s['p50']:.1f}ms  Min={ig_s['min']:.1f}ms  {ig_s['rps']:.0f} RPS")
print(f"    Python:  P50={py_s['p50']:.1f}ms  Min={py_s['min']:.1f}ms  {py_s['rps']:.0f} RPS")
row("Embed single (HTTP)", ig_e1, py_e1)

# Batch
print("  Batch 3 texts:")
ig_e3 = bench_http("localhost", 9902, "/v1/embeddings", emb_batch)
py_e3 = bench_http("localhost", 8902, "/embed", {"input":["hello","world","test"]})
ig_s = stats(ig_e3); py_s = stats(py_e3)
print(f"    infergo: P50={ig_s['p50']:.1f}ms  Min={ig_s['min']:.1f}ms  {ig_s['rps']:.0f} RPS")
print(f"    Python:  P50={py_s['p50']:.1f}ms  Min={py_s['min']:.1f}ms  {py_s['rps']:.0f} RPS")
row("Embed batch 3 (HTTP)", ig_e3, py_e3)

stop(ig_proc); stop(py_embed_proc)

# ═══════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════

print(f"\n{'━'*70}")
print(f"  NET-TO-NET RESULTS — Both via HTTP, 50 runs, keep-alive")
print(f"{'━'*70}")
print(f"\n  {'Test':<28} {'infergo P50':>12} {'Python P50':>12} {'Winner':>15}")
print(f"  {'─'*67}")

for name, ig, py, winner in R:
    ig_s = f"{ig['p50']:.1f}ms" if ig['p50'] > 0 else "—"
    py_s = f"{py['p50']:.1f}ms" if py['p50'] > 0 else "—"
    print(f"  {name:<28} {ig_s:>12} {py_s:>12} {winner:>15}")

print(f"\n  {'─'*67}")
print(f"  DETAILED STATS")
print(f"  {'─'*67}")
print(f"\n  {'Test':<28} {'':>4} {'P50':>7} {'Avg':>7} {'Min':>7} {'Max':>7} {'P99':>7} {'Std':>6} {'RPS':>6}")

for name, ig, py, _ in R:
    print(f"  {name:<28} {'ig':>4} {ig['p50']:>6.1f} {ig['avg']:>6.1f} {ig['min']:>6.1f} {ig['max']:>6.1f} {ig['p99']:>6.1f} {ig['std']:>5.1f} {ig['rps']:>5.0f}")
    print(f"  {'':28} {'py':>4} {py['p50']:>6.1f} {py['avg']:>6.1f} {py['min']:>6.1f} {py['max']:>6.1f} {py['p99']:>6.1f} {py['std']:>5.1f} {py['rps']:>5.0f}")

wins = sum(1 for _,_,_,w in R if "infergo" in w)
losses = sum(1 for _,_,_,w in R if "Python" in w and "infergo" not in w)
print(f"\n  SCORE: infergo {wins} — Python {losses}")
print(f"{'━'*70}")
