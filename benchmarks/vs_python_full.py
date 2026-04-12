#!/usr/bin/env python3
"""
Full infergo vs Python benchmark — LLM, embedding, detection
Measures: latency, throughput, memory, cold start
"""

import subprocess, time, json, os, sys, base64, signal, statistics

MODEL_LLM = "/tmp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_ONNX = os.path.expanduser("~/cgo/models/yolo11n.onnx")
INFERGO = os.path.expanduser("~/cgo/infergo")
CWD = os.path.expanduser("~/cgo")
RUNS = 15
WARMUP = 3

def banner(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

# ═══════════════════════════════════════════════════════════════
#  LLM BENCHMARK
# ═══════════════════════════════════════════════════════════════

def bench_infergo_llm():
    """Benchmark infergo LLM via HTTP API"""
    banner("LLM: infergo (C loop, CUDA)")

    # Start server
    env = os.environ.copy()
    proc = subprocess.Popen(
        [INFERGO, "serve", f"--model=llm:{MODEL_LLM}", "--port=9195", "--grpc-port=0", "--provider=cuda"],
        cwd=CWD, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid)

    for i in range(30):
        try:
            r = subprocess.run(["curl", "-s", "http://localhost:9195/health/live"],
                             capture_output=True, timeout=2)
            if r.returncode == 0: break
        except: pass
        time.sleep(1)

    req = json.dumps({
        "model": "llm",
        "messages": [{"role": "user", "content": "Explain transformers in two sentences."}],
        "max_tokens": 64
    })

    # Cold start (first request)
    cold_start = time.perf_counter()
    subprocess.run(["curl", "-s", "-X", "POST", "http://localhost:9195/v1/chat/completions",
                   "-H", "Content-Type: application/json", "-d", req], capture_output=True)
    cold_ms = (time.perf_counter() - cold_start) * 1000

    # Warmup
    for _ in range(WARMUP):
        subprocess.run(["curl", "-s", "-X", "POST", "http://localhost:9195/v1/chat/completions",
                       "-H", "Content-Type: application/json", "-d", req], capture_output=True)

    # Benchmark
    times = []
    tokens = []
    for i in range(RUNS):
        start = time.perf_counter()
        r = subprocess.run(["curl", "-s", "-X", "POST", "http://localhost:9195/v1/chat/completions",
                           "-H", "Content-Type: application/json", "-d", req], capture_output=True, text=True)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        try:
            resp = json.loads(r.stdout)
            tokens.append(resp.get("usage", {}).get("completion_tokens", 0))
        except:
            tokens.append(0)

    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    proc.wait()
    time.sleep(2)

    avg_tok = statistics.mean(tokens) if tokens else 0
    return {
        "cold_start_ms": cold_ms,
        "avg_ms": statistics.mean(times),
        "p50_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "avg_tokens": avg_tok,
        "tok_per_sec": avg_tok / (statistics.mean(times) / 1000) if statistics.mean(times) > 0 else 0
    }

def bench_python_llm():
    """Benchmark llama-cpp-python"""
    banner("LLM: llama-cpp-python (Python, CUDA)")

    try:
        from llama_cpp import Llama
    except ImportError:
        print("  SKIP: llama-cpp-python not installed")
        return None

    llm = Llama(model_path=MODEL_LLM, n_gpu_layers=99, n_ctx=2048, verbose=False)

    prompt = "Explain transformers in two sentences."

    # Cold start
    cold_start = time.perf_counter()
    llm.create_chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=64)
    cold_ms = (time.perf_counter() - cold_start) * 1000

    # Warmup
    for _ in range(WARMUP):
        llm.create_chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=64)

    # Benchmark
    times = []
    tokens = []
    for i in range(RUNS):
        start = time.perf_counter()
        resp = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=64)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        tokens.append(resp.get("usage", {}).get("completion_tokens", 0))

    del llm

    avg_tok = statistics.mean(tokens) if tokens else 0
    return {
        "cold_start_ms": cold_ms,
        "avg_ms": statistics.mean(times),
        "p50_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "avg_tokens": avg_tok,
        "tok_per_sec": avg_tok / (statistics.mean(times) / 1000) if statistics.mean(times) > 0 else 0
    }

# ═══════════════════════════════════════════════════════════════
#  DETECTION BENCHMARK
# ═══════════════════════════════════════════════════════════════

def bench_infergo_detect():
    """Benchmark infergo detection via HTTP"""
    banner("Detection: infergo (ONNX CUDA)")

    if not os.path.exists(MODEL_ONNX):
        print(f"  SKIP: {MODEL_ONNX} not found")
        return None

    # Create test image
    try:
        from PIL import Image
        import numpy as np
        img = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
        img.save("/tmp/bench_detect.jpg", quality=85)
    except:
        print("  SKIP: PIL not available")
        return None

    with open("/tmp/bench_detect.jpg", "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    req_file = "/tmp/bench_detect_req.json"
    with open(req_file, "w") as f:
        json.dump({"model": "detect", "image_b64": img_b64, "conf_thresh": 0.25}, f)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{os.path.expanduser('~/onnxruntime/lib')}:{env.get('LD_LIBRARY_PATH', '')}"
    proc = subprocess.Popen(
        [INFERGO, "serve", f"--model=detect:{MODEL_ONNX}", "--provider=cuda", "--port=9196", "--grpc-port=0"],
        cwd=CWD, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid)

    for i in range(30):
        try:
            r = subprocess.run(["curl", "-s", "http://localhost:9196/health/live"], capture_output=True, timeout=2)
            if r.returncode == 0: break
        except: pass
        time.sleep(1)

    # Warmup
    for _ in range(WARMUP):
        subprocess.run(["curl", "-s", "-X", "POST", "http://localhost:9196/v1/detect",
                       "-d", f"@{req_file}", "-H", "Content-Type: application/json"], capture_output=True)

    times = []
    for i in range(RUNS):
        start = time.perf_counter()
        subprocess.run(["curl", "-s", "-X", "POST", "http://localhost:9196/v1/detect",
                       "-d", f"@{req_file}", "-H", "Content-Type: application/json"], capture_output=True)
        times.append((time.perf_counter() - start) * 1000)

    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    proc.wait()
    time.sleep(2)

    return {
        "avg_ms": statistics.mean(times),
        "p50_ms": statistics.median(times),
        "min_ms": min(times),
        "rps": 1000 / statistics.mean(times)
    }

def bench_python_detect():
    """Benchmark ultralytics YOLO detection"""
    banner("Detection: ultralytics (Python, CUDA)")

    try:
        from ultralytics import YOLO
        import numpy as np
    except ImportError:
        print("  SKIP: ultralytics not installed")
        return None

    model = YOLO("yolo11n.pt")
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(WARMUP):
        model(img, verbose=False)

    times = []
    for i in range(RUNS):
        start = time.perf_counter()
        model(img, verbose=False)
        times.append((time.perf_counter() - start) * 1000)

    return {
        "avg_ms": statistics.mean(times),
        "p50_ms": statistics.median(times),
        "min_ms": min(times),
        "rps": 1000 / statistics.mean(times)
    }

# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("INFERGO vs PYTHON — FULL BENCHMARK")
    print(f"Runs: {RUNS} | Warmup: {WARMUP}")
    print(f"LLM Model: TinyLlama 1.1B Q4_K_M")
    print(f"Detection: yolo11n")

    results = {}

    # LLM benchmarks
    results["infergo_llm"] = bench_infergo_llm()
    results["python_llm"] = bench_python_llm()

    # Detection benchmarks
    results["infergo_detect"] = bench_infergo_detect()
    results["python_detect"] = bench_python_detect()

    # ═══════════════════════════════════════════════════════════
    banner("RESULTS SUMMARY")

    print(f"\n{'─'*60}")
    print(f"  LLM GENERATION (TinyLlama 1.1B, 64 tokens, CUDA)")
    print(f"{'─'*60}")
    print(f"  {'Metric':<25} {'infergo':>12} {'Python':>12} {'Winner':>10}")

    ig = results.get("infergo_llm", {})
    py = results.get("python_llm", {})

    if ig and py:
        def compare(name, ig_val, py_val, lower_better=True):
            if lower_better:
                winner = "infergo" if ig_val < py_val else "Python"
                ratio = py_val / ig_val if ig_val > 0 else 0
            else:
                winner = "infergo" if ig_val > py_val else "Python"
                ratio = ig_val / py_val if py_val > 0 else 0
            mark = f"{ratio:.1f}x" if ratio > 1 else f"{1/ratio:.1f}x"
            print(f"  {name:<25} {ig_val:>10.1f}ms {py_val:>10.1f}ms {winner:>6} {mark}")

        compare("P50 latency", ig.get("p50_ms",0), py.get("p50_ms",0))
        compare("Min latency", ig.get("min_ms",0), py.get("min_ms",0))
        compare("Cold start", ig.get("cold_start_ms",0), py.get("cold_start_ms",0))

        ig_tps = ig.get("tok_per_sec", 0)
        py_tps = py.get("tok_per_sec", 0)
        winner = "infergo" if ig_tps > py_tps else "Python"
        print(f"  {'Tok/s':<25} {ig_tps:>10.0f}    {py_tps:>10.0f}    {winner}")
    elif ig:
        print(f"  infergo: P50={ig.get('p50_ms',0):.1f}ms, {ig.get('tok_per_sec',0):.0f} tok/s")
        print(f"  Python:  SKIPPED")

    print(f"\n{'─'*60}")
    print(f"  DETECTION (yolo11n, 640x640, CUDA)")
    print(f"{'─'*60}")

    ig_d = results.get("infergo_detect", {})
    py_d = results.get("python_detect", {})

    if ig_d and py_d:
        print(f"  {'Metric':<25} {'infergo':>12} {'Python':>12} {'Winner':>10}")

        ig_avg = ig_d.get("avg_ms", 0)
        py_avg = py_d.get("avg_ms", 0)
        winner = "infergo" if ig_avg < py_avg else "Python"
        ratio = py_avg / ig_avg if ig_avg > 0 else 0
        print(f"  {'Avg latency':<25} {ig_avg:>10.1f}ms {py_avg:>10.1f}ms {winner:>6} {ratio:.1f}x")

        ig_rps = ig_d.get("rps", 0)
        py_rps = py_d.get("rps", 0)
        winner = "infergo" if ig_rps > py_rps else "Python"
        print(f"  {'Throughput (RPS)':<25} {ig_rps:>10.0f}    {py_rps:>10.0f}    {winner}")
    elif ig_d:
        print(f"  infergo: {ig_d.get('avg_ms',0):.1f}ms, {ig_d.get('rps',0):.0f} RPS")

    # WHERE WE'RE LOSING
    print(f"\n{'─'*60}")
    print(f"  WHERE INFERGO LOSES / NEEDS IMPROVEMENT")
    print(f"{'─'*60}")

    issues = []
    if ig and py:
        if ig.get("p50_ms",0) > py.get("p50_ms",0) * 1.05:
            issues.append(f"  - LLM P50 latency: {ig['p50_ms']:.0f}ms vs Python {py['p50_ms']:.0f}ms")
        if ig.get("cold_start_ms",0) > py.get("cold_start_ms",0) * 1.1:
            issues.append(f"  - Cold start: {ig['cold_start_ms']:.0f}ms vs Python {py['cold_start_ms']:.0f}ms (HTTP overhead)")
        if ig.get("tok_per_sec",0) < py.get("tok_per_sec",0) * 0.95:
            issues.append(f"  - Tok/s: {ig['tok_per_sec']:.0f} vs Python {py['tok_per_sec']:.0f}")

    if ig_d and py_d:
        if ig_d.get("avg_ms",0) > py_d.get("avg_ms",0) * 1.05:
            issues.append(f"  - Detection latency: {ig_d['avg_ms']:.0f}ms vs Python {py_d['avg_ms']:.0f}ms")

    if not issues:
        print("  None — infergo matches or beats Python on all metrics")
    else:
        for issue in issues:
            print(issue)

    print(f"\n{'='*60}")
