#!/usr/bin/env python3
"""
OVERNIGHT BENCHMARK — 5 modes, 1000 requests each, full cost analysis.
Run unattended. Results saved to benchmarks/overnight_results.json
"""

import time, statistics, os, subprocess, signal, json, gc, sys
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"

QWEN = "/tmp/qwen2.5-coder-1.5b-q4.gguf"
INFERGO = os.path.expanduser("~/cgo/infergo")
CWD = os.path.expanduser("~/cgo")
EMBED = os.path.expanduser("~/cgo/models/all-MiniLM-L6-v2/onnx/model.onnx")
RESULTS_FILE = os.path.expanduser("~/cgo/benchmarks/overnight_results.json")

DOCS = ["Go goroutines are lightweight threads.","Python GIL blocks threads.",
        "CUDA kernels run on GPU.","ONNX Runtime supports CUDA.",
        "llama.cpp loads GGUF Q4_K_M.","TensorRT fuses layers.",
        "Docker multi-stage builds.","Prometheus metrics.",
        "gRPC uses protobuf.","Flash Attention O(N)."]
QS = ["How do goroutines work?","What is GIL?","How does TensorRT work?","What is GGUF?"]

REQUESTS = 1000

def gpu_stats():
    try:
        r = subprocess.run(["nvidia-smi","--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        p = r.stdout.strip().split(", ")
        return {"vram_mb":int(p[0]),"gpu_pct":int(p[1]),"power_w":float(p[2]),"temp_c":int(p[3])}
    except: return {"vram_mb":0,"gpu_pct":0,"power_w":0,"temp_c":0}

def cosine(a,b):
    a,b=np.array(a),np.array(b)
    d=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/d) if d>0 else 0

def clear():
    subprocess.run("pkill -9 -f 'infergo serve' 2>/dev/null",shell=True)
    subprocess.run("pkill -9 -f 'llama_cpp' 2>/dev/null",shell=True)
    time.sleep(5); gc.collect()
    try: import torch; torch.cuda.empty_cache()
    except: pass
    time.sleep(3)

def start_ig(prov, port):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{os.path.expanduser('~/onnxruntime/lib')}:{env.get('LD_LIBRARY_PATH','')}"
    p = subprocess.Popen([INFERGO,"serve",f"--model=coder:{QWEN}",f"--model=embed:{EMBED}",
        f"--provider={prov}",f"--port={port}","--grpc-port=0","--max-seqs=4","--ctx-size=2048"],
        cwd=CWD,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,preexec_fn=os.setsid)
    for i in range(120):
        try:
            r = subprocess.run(["curl","-s",f"http://localhost:{port}/health/live"],capture_output=True,timeout=2)
            if r.returncode == 0: return p
        except: pass
        time.sleep(1)
    return p

def ig_rag(port, n, use_rerank=True, use_json=False):
    import http.client
    conn = http.client.HTTPConnection("localhost", port)
    h = {"Content-Type":"application/json","Connection":"keep-alive"}

    if use_rerank:
        dvecs = None
    else:
        dvecs = []
        for doc in DOCS:
            conn.request("POST","/v1/embeddings",json.dumps({"model":"embed","input":doc}),h)
            dvecs.append(json.loads(conn.getresponse().read())["data"][0]["embedding"])

    # Warmup
    for _ in range(20):
        if use_rerank:
            conn.request("POST","/v1/rerank",json.dumps({"model":"embed","query":"test","documents":DOCS[:3],"top_n":2}),h)
            conn.getresponse().read()
        conn.request("POST","/v1/chat/completions",json.dumps({"model":"coder","messages":[{"role":"user","content":"hi"}],"max_tokens":4}),h)
        conn.getresponse().read()

    times = []; gpu_samples = []; power_samples = []
    for i in range(n):
        q = QS[i % len(QS)]
        s = time.perf_counter()

        if use_rerank:
            conn.request("POST","/v1/rerank",json.dumps({"model":"embed","query":q,"documents":DOCS,"top_n":3}),h)
            top = json.loads(conn.getresponse().read())["results"]
            ctx = "\n".join([r.get("document","") for r in top])
        else:
            conn.request("POST","/v1/embeddings",json.dumps({"model":"embed","input":q}),h)
            qv = json.loads(conn.getresponse().read())["data"][0]["embedding"]
            sims = sorted([(cosine(qv,dv),i2) for i2,dv in enumerate(dvecs)],reverse=True)
            ctx = "\n".join([DOCS[i2] for _,i2 in sims[:3]])

        body = {"model":"coder","messages":[{"role":"user","content":f"Context:\n{ctx}\n\nQ: {q}\nA:"}],"max_tokens":30}
        if use_json: body["response_format"] = {"type":"json_object"}
        conn.request("POST","/v1/chat/completions",json.dumps(body),h)
        conn.getresponse().read()

        times.append((time.perf_counter()-s)*1000)
        if i % 50 == 0:
            g = gpu_stats()
            gpu_samples.append(g["gpu_pct"])
            power_samples.append(g["power_w"])
            sys.stdout.write(f"\r    {i}/{n} ({times[-1]:.0f}ms)")
            sys.stdout.flush()

    conn.close()
    print()
    return times, gpu_samples, power_samples


def py_rag(mode, n):
    from llama_cpp import Llama
    from sentence_transformers import SentenceTransformer

    device = "cuda" if mode == "gpu" else "cpu"
    gpu_layers = 99 if mode == "gpu" else 0

    llm = Llama(model_path=QWEN, n_gpu_layers=gpu_layers, n_ctx=2048, verbose=False)
    st = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    dvecs = st.encode(DOCS).tolist()

    for _ in range(5):
        st.encode(["hi"]); llm.create_chat_completion(messages=[{"role":"user","content":"hi"}],max_tokens=4)

    times = []; gpu_samples = []; power_samples = []
    for i in range(n):
        q = QS[i % len(QS)]
        s = time.perf_counter()
        qv = st.encode([q]).tolist()[0]
        sims = sorted([(cosine(qv,dv),i2) for i2,dv in enumerate(dvecs)],reverse=True)
        ctx = "\n".join([DOCS[i2] for _,i2 in sims[:3]])
        llm.create_chat_completion(messages=[{"role":"user","content":f"Context:\n{ctx}\n\nQ: {q}\nA:"}],max_tokens=30)
        times.append((time.perf_counter()-s)*1000)
        if i % 50 == 0:
            g = gpu_stats()
            gpu_samples.append(g["gpu_pct"])
            power_samples.append(g["power_w"])
            sys.stdout.write(f"\r    {i}/{n} ({times[-1]:.0f}ms)")
            sys.stdout.flush()

    print()
    del llm, st; gc.collect()
    try: import torch; torch.cuda.empty_cache()
    except: pass
    return times, gpu_samples, power_samples


ALL = {}

print("╔══════════════════════════════════════════════════════════════╗")
print(f"║  OVERNIGHT BENCHMARK — {REQUESTS} requests per mode            ║")
print("║  5 modes · CPU + GPU · Full cost analysis                   ║")
print("╚══════════════════════════════════════════════════════════════╝")
print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

modes = [
    ("1. Python CPU",       "py_cpu"),
    ("2. Python GPU",       "py_gpu"),
    ("3. infergo CPU",      "ig_cpu"),
    ("4. infergo GPU",      "ig_gpu"),
    ("5. infergo GPU+JSON", "ig_gpu_json"),
]

for label, key in modes:
    print(f"\n{'━'*55}")
    print(f"  {label} ({REQUESTS} requests)")
    print(f"{'━'*55}")

    clear()
    gpu_before = gpu_stats()
    load_s = time.perf_counter()

    if key == "py_cpu":
        times, gpu_s, power_s = py_rag("cpu", REQUESTS)
    elif key == "py_gpu":
        times, gpu_s, power_s = py_rag("gpu", REQUESTS)
    elif key == "ig_cpu":
        ig = start_ig("cpu", 9800)
        load_ms = int((time.perf_counter()-load_s)*1000)
        times, gpu_s, power_s = ig_rag(9800, REQUESTS, use_rerank=True)
        os.killpg(os.getpgid(ig.pid), signal.SIGKILL)
    elif key == "ig_gpu":
        ig = start_ig("cuda", 9801)
        load_ms = int((time.perf_counter()-load_s)*1000)
        times, gpu_s, power_s = ig_rag(9801, REQUESTS, use_rerank=True)
        os.killpg(os.getpgid(ig.pid), signal.SIGKILL)
    elif key == "ig_gpu_json":
        ig = start_ig("cuda", 9802)
        load_ms = int((time.perf_counter()-load_s)*1000)
        times, gpu_s, power_s = ig_rag(9802, REQUESTS, use_rerank=True, use_json=True)
        os.killpg(os.getpgid(ig.pid), signal.SIGKILL)

    if key.startswith("py"):
        load_ms = int((time.perf_counter()-load_s)*1000) - int(sum(times))

    gpu_after = gpu_stats()
    total_s = sum(times) / 1000
    avg_power = statistics.mean(power_s) if power_s else 0
    energy_wh = avg_power * (total_s / 3600)

    d = {
        "p50": round(statistics.median(times), 1),
        "p99": round(sorted(times)[int(len(times)*0.99)], 1),
        "avg": round(statistics.mean(times), 1),
        "min": round(min(times), 1),
        "max": round(max(times), 1),
        "std": round(statistics.stdev(times), 1),
        "total_s": round(total_s, 1),
        "rps": round(len(times)/total_s, 2),
        "vram_mb": gpu_after["vram_mb"],
        "avg_gpu_pct": round(statistics.mean(gpu_s), 1) if gpu_s else 0,
        "avg_power_w": round(avg_power, 1),
        "energy_wh": round(energy_wh, 3),
        "requests": len(times),
    }
    ALL[key] = d

    print(f"  P50={d['p50']:.0f}ms  P99={d['p99']:.0f}ms  RPS={d['rps']:.1f}")
    print(f"  GPU={d['avg_gpu_pct']:.0f}%  Power={d['avg_power_w']:.0f}W  Energy={d['energy_wh']:.3f}Wh")
    print(f"  VRAM={d['vram_mb']}MB  Total={d['total_s']:.0f}s")

# Save results
with open(RESULTS_FILE, "w") as f:
    json.dump(ALL, f, indent=2)
print(f"\nResults saved to {RESULTS_FILE}")

# Final report
print(f"\n{'━'*65}")
print(f"  FINAL RESULTS — {REQUESTS} REQUESTS EACH")
print(f"{'━'*65}")
print(f"\n  {'Mode':<25} {'P50':>7} {'P99':>7} {'RPS':>6} {'GPU%':>5} {'Power':>6} {'VRAM':>7} {'Energy':>7}")
print(f"  {'─'*65}")
for key, label in [("py_cpu","Python CPU"),("py_gpu","Python GPU"),
                    ("ig_cpu","infergo CPU"),("ig_gpu","infergo GPU"),("ig_gpu_json","infergo GPU+JSON")]:
    d = ALL[key]
    print(f"  {label:<25} {d['p50']:>5.0f}ms {d['p99']:>5.0f}ms {d['rps']:>5.1f} {d['avg_gpu_pct']:>4.0f}% {d['avg_power_w']:>4.0f}W {d['vram_mb']:>5}MB {d['energy_wh']:>6.3f}Wh")

print(f"\n  {'─'*65}")
print(f"  COST ESTIMATE (per 1M requests)")
print(f"  {'─'*65}")
for key, label in [("py_cpu","Python CPU"),("py_gpu","Python GPU"),
                    ("ig_cpu","infergo CPU"),("ig_gpu","infergo GPU"),("ig_gpu_json","infergo GPU+JSON")]:
    d = ALL[key]
    gpu_hrs = d['total_s'] / 3600 * (1_000_000 / REQUESTS)
    cost_t4 = gpu_hrs * 0.35
    cost_a100 = gpu_hrs * 3.00
    print(f"  {label:<25} {gpu_hrs:>6.0f} GPU-hrs  ${cost_t4:>6.0f} (T4)  ${cost_a100:>7.0f} (A100)")

py_gpu_hrs = ALL['py_gpu']['total_s'] / 3600 * (1_000_000 / REQUESTS)
ig_gpu_hrs = ALL['ig_gpu']['total_s'] / 3600 * (1_000_000 / REQUESTS)
savings = (1 - ig_gpu_hrs/py_gpu_hrs) * 100

print(f"\n  infergo saves {savings:.0f}% GPU cost vs Python")
print(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'━'*65}")
