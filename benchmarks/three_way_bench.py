#!/usr/bin/env python3
"""3-way benchmark: infergo vs LM Studio vs Python (llama-cpp-python)
Each server runs alone — no GPU contention."""

import json
import os
import signal
import subprocess
import time
import statistics
import requests
import concurrent.futures

MODEL = "/tmp/qwen2.5-coder-1.5b-q4.gguf"
N = 20
MAX_TOKENS = 64
CONCURRENCY = [1, 4, 8]  # concurrent users to test
PROMPT = "Return a JSON object with fields: name, age, city, occupation, hobbies (array of 3)."
SYS = "You are a helpful assistant. Output valid JSON only."

LMS_BIN = os.path.expanduser("~/.lmstudio/bin/lms")
INFERGO_BIN = os.path.expanduser("~/cgo/infergo")

LD_PATH = ":".join([
    os.path.expanduser("~/cgo/build/cpp/api"),
    os.path.expanduser("~/cgo/build/cpp/onnx"),
    os.path.expanduser("~/cgo/build/cpp/tokenizer"),
    os.path.expanduser("~/yolo-env/lib/python3.12/site-packages/torch/lib"),
    "/usr/local/cuda/lib64",
])


def kill_all():
    """Kill all inference servers."""
    for pat in ["port 9191", "llama_cpp.server", "llmster"]:
        subprocess.run(f"pkill -9 -f '{pat}'", shell=True, capture_output=True)
    time.sleep(3)


def wait_for_health(url, timeout=60):
    """Wait for server to respond."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def single_request(url, model_id):
    """Send one request and return (latency_ms, tokens, content)."""
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": PROMPT},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }
    t0 = time.perf_counter()
    r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=60)
    t1 = time.perf_counter()
    if r.status_code != 200:
        return None, 0, ""
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    tok = data.get("usage", {}).get("completion_tokens", 0)
    return (t1 - t0) * 1000, tok, content


def bench(name, url, model_id, concurrency=1):
    """Run N requests at given concurrency and return stats."""
    label = f"{name} (c={concurrency})"
    print(f"\n  Warming up {label}...")
    for _ in range(3):
        try:
            requests.post(f"{url}/v1/chat/completions", json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 4,
            }, timeout=15)
        except Exception:
            pass

    latencies = []
    tokens_list = []
    errors = 0
    sample = ""

    print(f"  Running {N} requests (concurrency={concurrency})...")
    t_wall_start = time.perf_counter()

    if concurrency == 1:
        for i in range(N):
            try:
                ms, tok, content = single_request(url, model_id)
                if ms is None:
                    errors += 1
                    continue
                latencies.append(ms)
                tokens_list.append(tok)
                if not sample:
                    sample = content[:150]
            except Exception:
                errors += 1
            time.sleep(0.05)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(single_request, url, model_id) for _ in range(N)]
            for f in concurrent.futures.as_completed(futures):
                try:
                    ms, tok, content = f.result()
                    if ms is None:
                        errors += 1
                        continue
                    latencies.append(ms)
                    tokens_list.append(tok)
                    if not sample:
                        sample = content[:150]
                except Exception:
                    errors += 1

    t_wall_end = time.perf_counter()
    wall_s = t_wall_end - t_wall_start

    if not latencies:
        return {"name": label, "error": f"all {errors} requests failed"}

    avg_tok = statistics.mean(tokens_list) if tokens_list else 0
    avg_ms = statistics.mean(latencies)
    total_tok = sum(tokens_list)
    return {
        "name": label,
        "concurrency": concurrency,
        "n": len(latencies),
        "errors": errors,
        "avg_ms": round(avg_ms, 1),
        "p50_ms": round(statistics.median(latencies), 1),
        "p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 1),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
        "avg_tokens": round(avg_tok, 1),
        "ms_per_tok": round(avg_ms / avg_tok, 1) if avg_tok > 0 else 0,
        "tok_per_sec": round(avg_tok / (avg_ms / 1000), 1) if avg_ms > 0 else 0,
        "rps": round(len(latencies) / wall_s, 2),
        "total_tok_per_sec": round(total_tok / wall_s, 1),
        "wall_s": round(wall_s, 1),
        "sample": sample,
    }


def main():
    print("=" * 80)
    print("3-WAY BENCHMARK: infergo vs LM Studio vs Python")
    print(f"Model: {MODEL}")
    print(f"N={N}, max_tokens={MAX_TOKENS}")
    print("=" * 80)

    all_results = []

    def run_engine(engine_name, start_fn, bench_fn, stop_fn):
        """Run benchmarks at all concurrency levels for one engine."""
        for c in CONCURRENCY:
            kill_all()
            time.sleep(2)
            if not start_fn():
                all_results.append({"name": f"{engine_name} (c={c})", "error": "server didn't start"})
                continue
            all_results.append(bench_fn(c))
            stop_fn()

    # ── 1. infergo ──
    print("\n[1/3] infergo (Go + C++ + CUDA + Flash Attention)")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LD_PATH + ":" + env.get("LD_LIBRARY_PATH", "")

    def start_infergo():
        proc = subprocess.Popen([
            INFERGO_BIN, "serve",
            "--model", f"llm:{MODEL}",
            "--provider", "cuda",
            "--port", "9191",
            "--grpc-port", "0",
            "--max-seqs", "8",
            "--ctx-size", "8192",
        ], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        start_infergo.proc = proc
        return wait_for_health("http://localhost:9191/health/live")

    def bench_infergo(c):
        return bench("infergo", "http://localhost:9191", "llm", concurrency=c)

    def stop_infergo():
        try:
            start_infergo.proc.kill()
            start_infergo.proc.wait()
        except Exception:
            pass

    run_engine("infergo", start_infergo, bench_infergo, stop_infergo)

    # ── 2. LM Studio ──
    print("\n[2/3] LM Studio (native llama.cpp runtime)")

    def start_lms():
        subprocess.run([LMS_BIN, "daemon", "up"], capture_output=True, timeout=30)
        time.sleep(3)
        subprocess.run([LMS_BIN, "server", "start", "--port", "1234"], capture_output=True, timeout=15)
        time.sleep(2)
        subprocess.run([LMS_BIN, "load", "qwen2.5-coder-1.5b", "--gpu", "max", "-y"],
                       capture_output=True, timeout=60)
        time.sleep(3)
        return wait_for_health("http://localhost:1234/v1/models", timeout=30)

    def bench_lms(c):
        return bench("lm-studio", "http://localhost:1234", "qwen2.5-coder-1.5b", concurrency=c)

    def stop_lms():
        subprocess.run([LMS_BIN, "unload", "--all", "-y"], capture_output=True, timeout=15)
        subprocess.run([LMS_BIN, "server", "stop"], capture_output=True, timeout=15)

    run_engine("lm-studio", start_lms, bench_lms, stop_lms)

    # ── 3. Python (llama-cpp-python) ──
    print("\n[3/3] Python (llama-cpp-python + uvicorn)")

    def start_python():
        proc = subprocess.Popen([
            "python3", "-m", "llama_cpp.server",
            "--model", MODEL,
            "--n_gpu_layers", "-1",
            "--port", "8080",
            "--host", "0.0.0.0",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        start_python.proc = proc
        return wait_for_health("http://localhost:8080/v1/models")

    def bench_python(c):
        return bench("python", "http://localhost:8080", MODEL, concurrency=c)

    def stop_python():
        try:
            start_python.proc.kill()
            start_python.proc.wait()
        except Exception:
            pass

    run_engine("python", start_python, bench_python, stop_python)

    # ── Results ──
    print("\n" + "=" * 115)
    print(f"{'Engine':<22} {'Avg ms':>8} {'P50 ms':>8} {'P99 ms':>8} {'Tokens':>7} {'ms/tok':>7} {'RPS':>6} {'Tot tok/s':>10} {'Wall':>6} {'Err':>4}")
    print("-" * 115)
    for r in all_results:
        if "error" in r:
            print(f"{r['name']:<22} {'ERROR':>8}  {r.get('error','')}")
        else:
            print(f"{r['name']:<22} {r['avg_ms']:>8.1f} {r['p50_ms']:>8.1f} {r['p99_ms']:>8.1f} "
                  f"{r['avg_tokens']:>7.1f} {r['ms_per_tok']:>7.1f} {r['rps']:>6.2f} "
                  f"{r['total_tok_per_sec']:>10.1f} {r['wall_s']:>6.1f} {r['errors']:>4}")

    # Per-concurrency comparison
    for c in CONCURRENCY:
        c_results = [r for r in all_results if "error" not in r and r.get("concurrency") == c]
        if len(c_results) >= 2:
            fastest = min(c_results, key=lambda x: x["avg_ms"])
            highest_rps = max(c_results, key=lambda x: x["rps"])
            print(f"\n  Concurrency={c}: Fastest latency: {fastest['name']} ({fastest['avg_ms']:.0f}ms)")
            print(f"  Concurrency={c}: Highest throughput: {highest_rps['name']} ({highest_rps['rps']:.1f} rps, {highest_rps['total_tok_per_sec']:.0f} tok/s)")

    # Samples
    print("\n--- Sample outputs ---")
    seen = set()
    for r in all_results:
        base = r.get("name", "").split(" (")[0]
        if "sample" in r and base not in seen:
            seen.add(base)
            print(f"[{base}]: {r['sample']}")

    # Save
    with open("benchmarks/three_way_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nResults saved to benchmarks/three_way_results.json")


if __name__ == "__main__":
    main()
