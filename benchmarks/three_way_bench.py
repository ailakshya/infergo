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

MODEL = "/tmp/qwen2.5-coder-1.5b-q4.gguf"
N = 20
MAX_TOKENS = 64
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


def bench(name, url, model_id):
    """Run N requests and return stats."""
    print(f"\n  Warming up {name}...")
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

    print(f"  Running {N} requests...")
    for i in range(N):
        body = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": PROMPT},
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.7,
        }
        try:
            t0 = time.perf_counter()
            r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=30)
            t1 = time.perf_counter()

            if r.status_code != 200:
                errors += 1
                continue

            data = r.json()
            latency_ms = (t1 - t0) * 1000
            latencies.append(latency_ms)

            content = data["choices"][0]["message"]["content"]
            tok = data.get("usage", {}).get("completion_tokens", 0)
            tokens_list.append(tok)
            if not sample:
                sample = content[:150]
        except Exception as e:
            errors += 1
        time.sleep(0.05)

    if not latencies:
        return {"name": name, "error": f"all {errors} requests failed"}

    avg_tok = statistics.mean(tokens_list) if tokens_list else 0
    avg_ms = statistics.mean(latencies)
    return {
        "name": name,
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
        "rps": round(len(latencies) / (sum(latencies) / 1000), 2),
        "sample": sample,
    }


def main():
    print("=" * 80)
    print("3-WAY BENCHMARK: infergo vs LM Studio vs Python")
    print(f"Model: {MODEL}")
    print(f"N={N}, max_tokens={MAX_TOKENS}")
    print("=" * 80)

    results = []

    # ── 1. infergo ──
    print("\n[1/3] infergo (Go + C++ + CUDA)")
    kill_all()
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LD_PATH + ":" + env.get("LD_LIBRARY_PATH", "")
    proc = subprocess.Popen([
        INFERGO_BIN, "serve",
        "--model", f"llm:{MODEL}",
        "--provider", "cuda",
        "--port", "9191",
        "--grpc-port", "0",
        "--max-seqs", "4",
        "--ctx-size", "4096",
    ], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("  Starting infergo...")
    if wait_for_health("http://localhost:9191/health/live"):
        results.append(bench("infergo", "http://localhost:9191", "llm"))
    else:
        print("  FAILED: infergo didn't start")
        results.append({"name": "infergo", "error": "server didn't start"})
    proc.kill()
    proc.wait()

    # ── 2. LM Studio ──
    print("\n[2/3] LM Studio (native llama.cpp runtime)")
    kill_all()
    time.sleep(2)
    # Start daemon
    subprocess.run([LMS_BIN, "daemon", "up"], capture_output=True, timeout=30)
    time.sleep(3)
    # Start server
    subprocess.run([LMS_BIN, "server", "start", "--port", "1234"], capture_output=True, timeout=15)
    time.sleep(2)
    # Load model
    subprocess.run([LMS_BIN, "load", "qwen2.5-coder-1.5b", "--gpu", "max", "-y"],
                   capture_output=True, timeout=60)
    time.sleep(3)

    if wait_for_health("http://localhost:1234/v1/models", timeout=30):
        results.append(bench("lm-studio", "http://localhost:1234", "qwen2.5-coder-1.5b"))
    else:
        print("  FAILED: LM Studio didn't start")
        results.append({"name": "lm-studio", "error": "server didn't start"})
    subprocess.run([LMS_BIN, "server", "stop"], capture_output=True)
    subprocess.run([LMS_BIN, "daemon", "down"], capture_output=True)

    # ── 3. Python (llama-cpp-python) ──
    print("\n[3/3] Python (llama-cpp-python + uvicorn)")
    kill_all()
    time.sleep(2)
    proc = subprocess.Popen([
        "python3", "-m", "llama_cpp.server",
        "--model", MODEL,
        "--n_gpu_layers", "-1",
        "--port", "8080",
        "--host", "0.0.0.0",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("  Starting Python server...")
    if wait_for_health("http://localhost:8080/v1/models"):
        results.append(bench("python", "http://localhost:8080", MODEL))
    else:
        print("  FAILED: Python server didn't start")
        results.append({"name": "python", "error": "server didn't start"})
    proc.kill()
    proc.wait()

    # ── Results ──
    print("\n" + "=" * 100)
    print(f"{'Engine':<14} {'Avg ms':>8} {'P50 ms':>8} {'P99 ms':>8} {'Tokens':>8} {'ms/tok':>8} {'tok/s':>8} {'RPS':>6} {'Err':>4}")
    print("-" * 100)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<14} {'ERROR':>8}  {r.get('error','')}")
        else:
            print(f"{r['name']:<14} {r['avg_ms']:>8.1f} {r['p50_ms']:>8.1f} {r['p99_ms']:>8.1f} "
                  f"{r['avg_tokens']:>8.1f} {r['ms_per_tok']:>8.1f} {r['tok_per_sec']:>8.1f} "
                  f"{r['rps']:>6.2f} {r['errors']:>4}")

    # Comparison
    valid = [r for r in results if "error" not in r]
    if len(valid) >= 2:
        fastest = min(valid, key=lambda x: x["avg_ms"])
        print(f"\nFastest: {fastest['name']} ({fastest['avg_ms']:.0f}ms avg)")
        for r in valid:
            if r != fastest:
                ratio = r["avg_ms"] / fastest["avg_ms"]
                print(f"  vs {r['name']}: {ratio:.2f}x slower")

    # Samples
    print("\n--- Sample outputs ---")
    for r in results:
        if "sample" in r:
            print(f"[{r['name']}]: {r['sample']}")

    # Save
    with open("benchmarks/three_way_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to benchmarks/three_way_results.json")


if __name__ == "__main__":
    main()
