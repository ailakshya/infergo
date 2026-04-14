#!/usr/bin/env python3
"""Full comparison: infergo vs Ollama vs Open WebUI vs LM Studio vs Python
All engines run the same model (Qwen 2.5 Coder 1.5B Q4) on the same GPU."""

import json, os, subprocess, time, statistics, requests, concurrent.futures, sys

MODEL_PATH = "/tmp/qwen2.5-coder-1.5b-q4.gguf"
N = 20
MAX_TOKENS = 32
CONCURRENCY = [1, 4, 8]
PROMPT = "Return a JSON object with name, age, city."
SYS = "Output valid JSON only."

LMS_BIN = os.path.expanduser("~/.lmstudio/bin/lms")
INFERGO_BIN = os.path.expanduser("~/cgo/infergo")
LD = ":".join([os.path.expanduser(p) for p in [
    "~/cgo/build/cpp/api", "~/cgo/build/cpp/onnx", "~/cgo/build/cpp/tokenizer",
    "~/yolo-env/lib/python3.12/site-packages/torch/lib"
]] + ["/usr/local/cuda/lib64"])


def kill_gpu():
    subprocess.run("pkill -9 -f 'port 9191'; pkill -9 -f llmster; pkill -9 -f 'ollama serve'",
                   shell=True, capture_output=True)
    # Kill GPU processes
    try:
        pids = subprocess.check_output(
            "nvidia-smi --query-compute-apps=pid --format=csv,noheader",
            shell=True, text=True).strip().split('\n')
        for pid in pids:
            pid = pid.strip()
            if pid and pid != '2930':  # keep Xorg
                subprocess.run(f"kill -9 {pid}", shell=True, capture_output=True)
    except: pass
    time.sleep(3)


def wait_health(url, timeout=60):
    t = time.time()
    while time.time() - t < timeout:
        try:
            if requests.get(url, timeout=2).status_code == 200: return True
        except: pass
        time.sleep(1)
    return False


def do_req(url, model_id):
    try:
        t0 = time.perf_counter()
        r = requests.post(f"{url}/v1/chat/completions", json={
            "model": model_id,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": PROMPT}],
            "max_tokens": MAX_TOKENS, "temperature": 0.7
        }, timeout=60)
        t1 = time.perf_counter()
        if r.status_code != 200: return None, 0
        d = r.json()
        return (t1-t0)*1000, d.get("usage", {}).get("completion_tokens", 0)
    except: return None, 0


def bench(name, url, model_id, c):
    # Warmup
    for _ in range(3):
        do_req(url, model_id)

    lats, toks, errs = [], [], 0
    t0 = time.perf_counter()
    if c == 1:
        for _ in range(N):
            ms, tok = do_req(url, model_id)
            if ms is None: errs += 1
            else: lats.append(ms); toks.append(tok)
    else:
        with concurrent.futures.ThreadPoolExecutor(c) as pool:
            futs = [pool.submit(do_req, url, model_id) for _ in range(N)]
            for f in concurrent.futures.as_completed(futs):
                ms, tok = f.result()
                if ms is None: errs += 1
                else: lats.append(ms); toks.append(tok)
    wall = time.perf_counter() - t0

    if not lats: return None
    avg = statistics.mean(lats)
    return {
        "name": f"{name} c={c}", "avg": round(avg, 1),
        "p50": round(statistics.median(lats), 1),
        "tok": round(statistics.mean(toks), 1) if toks else 0,
        "mpt": round(avg / statistics.mean(toks), 1) if toks and statistics.mean(toks) > 0 else 0,
        "rps": round(len(lats) / wall, 2),
        "tps": round(sum(toks) / wall, 1) if toks else 0,
        "err": errs, "n": len(lats)
    }


def run_engine(name, start_fn, url, model_id, stop_fn):
    results = []
    for c in CONCURRENCY:
        kill_gpu()
        print(f"  [{name} c={c}] starting...", end=" ", flush=True)
        if not start_fn():
            print("FAIL")
            continue
        r = bench(name, url, model_id, c)
        if r:
            results.append(r)
            print(f"avg={r['avg']:.0f}ms rps={r['rps']:.1f} tok/s={r['tps']:.0f}")
        else:
            print("all requests failed")
        stop_fn()
    return results


def main():
    print("=" * 95)
    print("FULL COMPARISON: infergo vs Ollama vs Open WebUI vs LM Studio vs Python")
    print(f"Model: Qwen 2.5 Coder 1.5B Q4 | N={N} | max_tokens={MAX_TOKENS}")
    print("=" * 95)

    all_results = []

    # 1. infergo
    print("\n[1/5] infergo")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LD + ":" + env.get("LD_LIBRARY_PATH", "")
    infergo_proc = [None]
    def start_infergo():
        infergo_proc[0] = subprocess.Popen([
            INFERGO_BIN, "serve", "--model", f"llm:{MODEL_PATH}",
            "--provider", "cuda", "--port", "9191", "--grpc-port", "0",
            "--max-seqs", "8", "--ctx-size", "8192"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return wait_health("http://localhost:9191/health/live")
    def stop_infergo():
        if infergo_proc[0]:
            infergo_proc[0].kill(); infergo_proc[0].wait()
    all_results += run_engine("infergo", start_infergo, "http://localhost:9191", "llm", stop_infergo)

    # 2. Ollama
    print("\n[2/5] Ollama")
    ollama_proc = [None]
    def start_ollama():
        # Start ollama serve
        subprocess.run("ollama serve &", shell=True, capture_output=True)
        time.sleep(3)
        # Pre-load model
        try:
            requests.post("http://localhost:11434/api/generate",
                          json={"model": "qwen2.5-coder:1.5b", "prompt": "hi", "stream": False},
                          timeout=60)
        except: pass
        return wait_health("http://localhost:11434/api/tags", timeout=30)
    def stop_ollama():
        subprocess.run("pkill -9 -f 'ollama serve'", shell=True, capture_output=True)
    all_results += run_engine("ollama", start_ollama, "http://localhost:11434",
                              "qwen2.5-coder:1.5b", stop_ollama)

    # 3. LM Studio
    print("\n[3/5] LM Studio")
    def start_lms():
        subprocess.run([LMS_BIN, "daemon", "up"], capture_output=True, timeout=30)
        time.sleep(3)
        subprocess.run([LMS_BIN, "server", "start", "--port", "1234"], capture_output=True, timeout=15)
        time.sleep(2)
        subprocess.run([LMS_BIN, "load", "qwen2.5-coder-1.5b", "--gpu", "max", "-y"],
                       capture_output=True, timeout=60)
        time.sleep(3)
        return wait_health("http://localhost:1234/v1/models", timeout=30)
    def stop_lms():
        subprocess.run([LMS_BIN, "unload", "--all", "-y"], capture_output=True)
        subprocess.run([LMS_BIN, "server", "stop"], capture_output=True)
    all_results += run_engine("lm-studio", start_lms, "http://localhost:1234",
                              "qwen2.5-coder-1.5b", stop_lms)

    # 4. Python (llama-cpp-python)
    print("\n[4/5] Python (llama-cpp-python)")
    py_proc = [None]
    def start_python():
        py_proc[0] = subprocess.Popen([
            "python3", "-m", "llama_cpp.server",
            "--model", MODEL_PATH, "--n_gpu_layers", "-1",
            "--port", "8080", "--host", "0.0.0.0"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return wait_health("http://localhost:8080/v1/models")
    def stop_python():
        if py_proc[0]:
            py_proc[0].kill(); py_proc[0].wait()
    all_results += run_engine("python", start_python, "http://localhost:8080",
                              MODEL_PATH, stop_python)

    # Results table
    print("\n" + "=" * 95)
    print(f"{'Engine':<22} {'Avg ms':>8} {'P50 ms':>8} {'Tok':>5} {'ms/tok':>7} {'RPS':>6} {'tok/s':>7} {'Err':>4}")
    print("-" * 95)
    for r in all_results:
        print(f"{r['name']:<22} {r['avg']:>8.1f} {r['p50']:>8.1f} {r['tok']:>5.1f} "
              f"{r['mpt']:>7.1f} {r['rps']:>6.2f} {r['tps']:>7.1f} {r['err']:>4}")

    # Winners per concurrency
    for c in CONCURRENCY:
        cr = [r for r in all_results if f"c={c}" in r["name"]]
        if len(cr) >= 2:
            fastest = min(cr, key=lambda x: x["avg"])
            highest_tps = max(cr, key=lambda x: x["tps"])
            print(f"\n  c={c} fastest latency: {fastest['name']} ({fastest['avg']:.0f}ms)")
            print(f"  c={c} highest tok/s:  {highest_tps['name']} ({highest_tps['tps']:.0f} tok/s)")

    # Save
    with open("benchmarks/full_comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to benchmarks/full_comparison_results.json")


if __name__ == "__main__":
    main()
