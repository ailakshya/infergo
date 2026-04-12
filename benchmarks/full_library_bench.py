#!/usr/bin/env python3
"""
FULL LIBRARY BENCHMARK — Every function, every feature.
GPU VRAM cleared between tests. 100 requests. All alternatives.
"""

import time, statistics, os, subprocess, signal, json, io, sys, gc
import http.client
import concurrent.futures
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

INFERGO = os.path.expanduser("~/cgo/infergo")
CWD = os.path.expanduser("~/cgo")
LLM = "/tmp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
LLM_8B = os.path.expanduser("~/cgo/models/llama3-8b-q4.gguf")
DRAFT = "/tmp/llama-3.2-1b-q4.gguf"
EMBED = os.path.expanduser("~/cgo/models/all-MiniLM-L6-v2/onnx/model.onnx")
DET_PT = os.path.expanduser("~/cgo/models/yolo11n.torchscript.pt")
DET_ONNX = os.path.expanduser("~/cgo/models/yolo11n.onnx")

RUNS = 100
WARMUP = 20

R = []  # (section, name, ig_val, py_val, unit, winner)

def clear_gpu():
    """Kill all infergo/python servers and clear GPU VRAM"""
    subprocess.run("pkill -9 -f 'infergo serve' 2>/dev/null", shell=True)
    subprocess.run("pkill -9 -f 'llama_cpp.server' 2>/dev/null", shell=True)
    subprocess.run("pkill -9 -f 'HTTPServer' 2>/dev/null", shell=True)
    time.sleep(3)
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except: pass
    # Verify VRAM is free
    r = subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    vram = int(r.stdout.strip()) if r.returncode == 0 else 0
    print(f"  [GPU VRAM: {vram} MiB used]")
    time.sleep(2)

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

def stop(p):
    try: os.killpg(os.getpgid(p.pid), signal.SIGKILL); p.wait(5)
    except: pass

def bench_keepalive(host, port, path, body, runs=RUNS, warmup=WARMUP):
    conn = http.client.HTTPConnection(host, port)
    h = {"Content-Type":"application/json","Connection":"keep-alive"}
    for _ in range(warmup):
        try: conn.request("POST",path,body,h); conn.getresponse().read()
        except: conn = http.client.HTTPConnection(host, port)
    times = []
    for _ in range(runs):
        try:
            s = time.perf_counter()
            conn.request("POST",path,body,h); conn.getresponse().read()
            times.append((time.perf_counter()-s)*1000)
        except: times.append(-1); conn = http.client.HTTPConnection(host, port)
    conn.close()
    return [t for t in times if t > 0]

def bench_get(host, port, path, runs=200):
    conn = http.client.HTTPConnection(host, port)
    for _ in range(20): conn.request("GET",path); conn.getresponse().read()
    times = []
    for _ in range(runs):
        s = time.perf_counter()
        conn.request("GET",path); conn.getresponse().read()
        times.append((time.perf_counter()-s)*1000)
    conn.close()
    return times

def fmt(times):
    if not times: return "—", "—", "—"
    return f"{statistics.median(times):.1f}ms", f"{min(times):.1f}ms", f"{1000/statistics.mean(times):.0f}"

def add(section, name, ig_times, py_times, unit="ms"):
    ig_p50 = statistics.median(ig_times) if ig_times else None
    py_p50 = statistics.median(py_times) if py_times else None
    if ig_p50 and py_p50:
        if ig_p50 < py_p50: winner = f"infergo {py_p50/ig_p50:.1f}x"
        else: winner = f"Python {ig_p50/py_p50:.1f}x"
    elif ig_p50: winner = "infergo"
    else: winner = "—"
    R.append((section, name, ig_p50, py_p50, unit, winner))
    ig_s = f"{ig_p50:.1f}{unit}" if ig_p50 else "—"
    py_s = f"{py_p50:.1f}{unit}" if py_p50 else "—"
    print(f"    {name:<35} {ig_s:>10} {py_s:>10}  {winner}")


print("╔══════════════════════════════════════════════════════════════════════╗")
print("║  FULL LIBRARY BENCHMARK — Every Feature, GPU Cleared Between Tests  ║")
print("║  100 requests, 20 warmup, keep-alive, RTX 5070 Ti                  ║")
print("╚══════════════════════════════════════════════════════════════════════╝")


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: LLM
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'━'*70}")
print(f"  SECTION 1: LLM GENERATION")
print(f"{'━'*70}")

clear_gpu()
print("  [infergo]")
ig = start_ig(f"llm:{LLM}", "cuda", 9900, ["--max-seqs","32","--ctx-size","4096"])

ig_chat = bench_keepalive("localhost",9900,"/v1/chat/completions",
    json.dumps({"model":"llm","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":16}))

ig_text = bench_keepalive("localhost",9900,"/v1/completions",
    json.dumps({"model":"llm","prompt":"Hello","max_tokens":16}))

ig_json = bench_keepalive("localhost",9900,"/v1/chat/completions",
    json.dumps({"model":"llm","messages":[{"role":"user","content":"JSON"}],"max_tokens":16,
                "response_format":{"type":"json_object"}}), runs=50)

ig_stream = bench_keepalive("localhost",9900,"/v1/chat/completions",
    json.dumps({"model":"llm","messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"stream":True}), runs=50)

stop(ig); clear_gpu()

print("  [Python llama-cpp-python]")
py = subprocess.Popen(["python3","-m","llama_cpp.server","--model",LLM,"--n_gpu_layers","99",
    "--n_ctx","4096","--host","0.0.0.0","--port","8900"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
for i in range(60):
    try:
        c = http.client.HTTPConnection("localhost",8900)
        c.request("GET","/v1/models"); r = c.getresponse()
        if r.status == 200 and b"model" in r.read(): c.close(); break
        c.close()
    except: pass
    time.sleep(1)

py_chat = bench_keepalive("localhost",8900,"/v1/chat/completions",
    json.dumps({"model":"default","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":16}))

py_text = bench_keepalive("localhost",8900,"/v1/completions",
    json.dumps({"model":"default","prompt":"Hello","max_tokens":16}), runs=50)

stop(py)

add("LLM", "Chat completion (16 tok)", ig_chat, py_chat)
add("LLM", "Text completion (16 tok)", ig_text, py_text)
add("LLM", "JSON structured output", ig_json, [])
add("LLM", "Streaming TTFB", ig_stream, [])

clear_gpu()


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: EMBEDDING
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'━'*70}")
print(f"  SECTION 2: EMBEDDINGS")
print(f"{'━'*70}")

clear_gpu()
print("  [infergo TorchScript CUDA]")
ig = start_ig(f"embed:{EMBED}", "cuda", 9901)

ig_e1 = bench_keepalive("localhost",9901,"/v1/embeddings",
    json.dumps({"model":"embed","input":"hello world"}))

ig_e3 = bench_keepalive("localhost",9901,"/v1/embeddings",
    json.dumps({"model":"embed","input":["hello","world","test"]}))

ig_e10 = bench_keepalive("localhost",9901,"/v1/embeddings",
    json.dumps({"model":"embed","input":["a","b","c","d","e","f","g","h","i","j"]}), runs=50)

stop(ig); clear_gpu()

print("  [Python sentence-transformers CUDA via HTTP]")
py_code = '''import json;from http.server import HTTPServer,BaseHTTPRequestHandler;from sentence_transformers import SentenceTransformer
model=SentenceTransformer("all-MiniLM-L6-v2",device="cuda")
class H(BaseHTTPRequestHandler):
 def do_POST(self):
  data=json.loads(self.rfile.read(int(self.headers["Content-Length"])));inp=data.get("input",[])
  if isinstance(inp,str):inp=[inp]
  vecs=model.encode(inp).tolist();resp=json.dumps({"data":[{"embedding":v}for v in vecs]}).encode()
  self.send_response(200);self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(resp)));self.end_headers();self.wfile.write(resp)
 def log_message(self,*a):pass
HTTPServer(("0.0.0.0",8901),H).serve_forever()'''
py = subprocess.Popen(["python3","-c",py_code],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,preexec_fn=os.setsid)
time.sleep(8)

py_e1 = bench_keepalive("localhost",8901,"/",json.dumps({"input":"hello world"}))
py_e3 = bench_keepalive("localhost",8901,"/",json.dumps({"input":["hello","world","test"]}))
py_e10 = bench_keepalive("localhost",8901,"/",json.dumps({"input":["a","b","c","d","e","f","g","h","i","j"]}), runs=50)
stop(py)

add("Embedding", "Single text", ig_e1, py_e1)
add("Embedding", "Batch 3 texts", ig_e3, py_e3)
add("Embedding", "Batch 10 texts", ig_e10, py_e10)

clear_gpu()


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: DETECTION
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'━'*70}")
print(f"  SECTION 3: OBJECT DETECTION")
print(f"{'━'*70}")

from PIL import Image
img = np.random.randint(0,255,(640,640,3),dtype=np.uint8)
pil = Image.fromarray(img); buf = io.BytesIO(); pil.save(buf,format="JPEG",quality=85)
with open("/tmp/full_bench.jpg","wb") as f: f.write(buf.getvalue())

clear_gpu()

# ── infergo in-process (Go benchmark, no HTTP) ──
print("  [infergo TorchScript+nvJPEG in-process]")
ig_det_code = '''package main
import ("fmt";"time";"os";"os/exec";"github.com/ailakshya/infergo/torch")
func main() {
    sess,_ := torch.NewSession("cuda",0); defer sess.Close()
    sess.Load(os.ExpandEnv("${HOME}/cgo/models/yolo11n.torchscript.pt"))
    out,_ := exec.Command("python3","-c",
        "import sys;from PIL import Image;import numpy as np;import io;"+
        "img=Image.fromarray(np.random.randint(0,255,(640,640,3),dtype=np.uint8));"+
        "b=io.BytesIO();img.save(b,format=\\"JPEG\\",quality=85);sys.stdout.buffer.write(b.getvalue())").Output()
    if len(out)<100 { d,_ := os.ReadFile("/tmp/full_bench.jpg"); out=d }
    for i:=0;i<20;i++ { sess.DetectGPU(out,0.25,0.45) }
    for i:=0;i<100;i++ {
        s:=time.Now(); sess.DetectGPU(out,0.25,0.45)
        fmt.Println(float64(time.Since(s).Microseconds())/1000)
    }
}'''
with open("/tmp/_det_bench.go","w") as f: f.write(ig_det_code)
r = subprocess.run(["go","run","/tmp/_det_bench.go"], capture_output=True, text=True,
                   timeout=120, cwd=os.path.expanduser("~/cgo/go"))
ig_det = []
for line in r.stdout.strip().split("\n"):
    try: ig_det.append(float(line.strip()))
    except: pass

clear_gpu()

# ── Python in-process (same conditions) ──
print("  [Python ultralytics CUDA in-process]")
from ultralytics import YOLO
yolo = YOLO("yolo11n.pt")
for _ in range(WARMUP): yolo(img, verbose=False)
py_det = []
for _ in range(RUNS):
    s = time.perf_counter()
    yolo(img, verbose=False)
    py_det.append((time.perf_counter()-s)*1000)
del yolo; gc.collect()

add("Detection", "yolo11n in-process", ig_det, py_det)

clear_gpu()


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: SEARCH & RERANK
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'━'*70}")
print(f"  SECTION 4: VECTOR SEARCH & RERANKING")
print(f"{'━'*70}")

clear_gpu()
ig = start_ig(f"embed:{EMBED}", "cuda", 9903)

ig_search = bench_keepalive("localhost",9903,"/v1/search",
    json.dumps({"model":"embed","query":"hello","k":10}))

ig_rerank = bench_keepalive("localhost",9903,"/v1/rerank",
    json.dumps({"model":"embed","query":"machine learning",
                "documents":["AI is ML","cats are fluffy","deep learning","robots","data science"],
                "top_n":3}))

stop(ig)

add("Search", "HNSW k=10", ig_search, [])
add("Search", "Rerank 5 docs", ig_rerank, [])

clear_gpu()


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: ADMIN & INFRA
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'━'*70}")
print(f"  SECTION 5: ADMIN & INFRASTRUCTURE")
print(f"{'━'*70}")

clear_gpu()
ig = start_ig(f"llm:{LLM}", "cuda", 9904)

ig_health = bench_get("localhost",9904,"/health/live")
ig_ready = bench_get("localhost",9904,"/health/ready")
ig_models = bench_get("localhost",9904,"/v1/models")
ig_metrics = bench_get("localhost",9904,"/metrics")
ig_guard = bench_get("localhost",9904,"/v1/admin/guardrails")
ig_tmpl = bench_get("localhost",9904,"/v1/admin/templates")
ig_ui = bench_get("localhost",9904,"/ui")

stop(ig)

add("Admin", "Health check", ig_health, [])
add("Admin", "Ready check", ig_ready, [])
add("Admin", "List models", ig_models, [])
add("Admin", "Prometheus metrics", ig_metrics, [])
add("Admin", "Guardrails config", ig_guard, [])
add("Admin", "Prompt templates", ig_tmpl, [])
add("Admin", "Web UI", ig_ui, [])

clear_gpu()


# ═══════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'━'*70}")
print(f"  FULL LIBRARY BENCHMARK — FINAL RESULTS")
print(f"{'━'*70}")

current_section = ""
print(f"\n  {'Feature':<35} {'infergo':>10} {'Python':>10}  {'Winner'}")
print(f"  {'─'*67}")

for section, name, ig, py, unit, winner in R:
    if section != current_section:
        current_section = section
        print(f"\n  [{section}]")
    ig_s = f"{ig:.1f}{unit}" if ig else "—"
    py_s = f"{py:.1f}{unit}" if py else "—"
    print(f"    {name:<33} {ig_s:>10} {py_s:>10}  {winner}")

wins = sum(1 for _,_,_,_,_,w in R if "infergo" in w)
losses = sum(1 for _,_,_,_,_,w in R if "Python" in w and "infergo" not in w)
total = len(R)

print(f"\n  {'━'*67}")
print(f"  FEATURES TESTED: {total}")
print(f"  INFERGO WINS:    {wins}")
print(f"  PYTHON WINS:     {losses}")
print(f"  INFERGO ONLY:    {total - wins - losses}")
print(f"  {'━'*67}")
