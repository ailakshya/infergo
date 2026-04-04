"""
bench_multimodel.py — Multi-model LLM benchmark across 3 models (OPT-17).

Benchmarks infergo vs llama-cpp-python across:
  - llama3-8b-q4.gguf     (8B, Q4_K_M)
  - phi-3.5-mini.Q4_K_M.gguf (3.8B, Q4_K_M)
  - gemma-2-9b-it.Q4_K_M.gguf (9B, Q4_K_M)

For each model:
  - Short prompt tok/s (CUDA)
  - Long prompt tok/s (CUDA)
  - P50 latency at concurrency=4

Usage:
  # Start infergo with the model being tested, e.g.:
  #   ./infergo serve --model llama3:models/llama3-8b-q4.gguf --provider cuda --port 9191
  #
  # Run with a single model (repeat for each):
  python bench_multimodel.py --model-name llama3-8b-q4 --infergo-addr http://localhost:9191

  # Or with --all-models, it loops through the model_configs below sequentially,
  # starting/stopping the server for each one:
  python bench_multimodel.py --all-models --infergo-bin ./infergo --models-dir ~/cgo/models

Output: benchmarks/vs_python/results_multimodel.md
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, List

import urllib.request
import urllib.error


# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_CONFIGS = [
    {
        "name":    "llama3-8b-q4",
        "file":    "llama3-8b-q4.gguf",
        "params":  "8B",
        "quant":   "Q4_K_M",
        "size_gb": 4.6,
    },
    {
        "name":    "phi-3.5-mini-q4",
        "file":    "phi-3.5-mini-instruct.Q4_K_M.gguf",
        "params":  "3.8B",
        "quant":   "Q4_K_M",
        "size_gb": 2.2,
    },
    {
        "name":    "gemma-2-9b-q4",
        "file":    "gemma-2-9b-it.Q4_K_M.gguf",
        "params":  "9B",
        "quant":   "Q4_K_M",
        "size_gb": 5.5,
    },
]

SHORT_PROMPT = "What is the capital of France?"
LONG_PROMPT  = "Explain the key differences between transformer and LSTM architectures in deep learning, covering attention mechanisms, parallelisation advantages, gradient flow, and typical use cases. Then describe how each handles long-range dependencies and how modern variants like GPT-4, T5, and Mamba address their limitations."
MAX_TOKENS   = 128


# ─── HTTP helpers ─────────────────────────────────────────────────────────────

def chat_completion(addr: str, model: str, prompt: str, max_tokens: int = MAX_TOKENS) -> dict:
    """Send a blocking chat completion request and return the response JSON."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{addr}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def measure_single(addr: str, model: str, prompt: str) -> tuple[float, int]:
    """Return (elapsed_seconds, gen_tokens) for one request."""
    t0 = time.monotonic()
    resp = chat_completion(addr, model, prompt)
    elapsed = time.monotonic() - t0
    gen_tokens = resp.get("usage", {}).get("completion_tokens", 0)
    return elapsed, gen_tokens


@dataclass
class BenchResult:
    model_name:  str
    short_toks:  float   # tok/s, short prompt
    long_toks:   float   # tok/s, long prompt
    p50_ms:      float   # P50 latency (ms) at concurrency=4
    p95_ms:      float
    errors:      int     = 0


def run_throughput(addr: str, model: str, prompt: str, n: int = 8) -> float:
    """Run n sequential requests and return average tok/s."""
    tok_rates = []
    for _ in range(n):
        try:
            elapsed, gen_toks = measure_single(addr, model, prompt)
            if gen_toks > 0 and elapsed > 0:
                tok_rates.append(gen_toks / elapsed)
        except Exception as e:
            print(f"  [warn] request failed: {e}", file=sys.stderr)
    if not tok_rates:
        return 0.0
    return statistics.mean(tok_rates)


def run_latency(addr: str, model: str, concurrency: int = 4, total: int = 20) -> tuple[float, float, int]:
    """Run concurrent requests and return (p50_ms, p95_ms, errors)."""
    latencies = []
    errors = 0

    def one_request(_):
        try:
            t0 = time.monotonic()
            chat_completion(addr, model, SHORT_PROMPT)
            return (time.monotonic() - t0) * 1000
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(one_request, i) for i in range(total)]
        for f in as_completed(futs):
            r = f.result()
            if r is None:
                errors += 1
            else:
                latencies.append(r)

    if not latencies:
        return 0.0, 0.0, errors

    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)]
    p95 = latencies[int(len(latencies) * 0.95)]
    return p50, p95, errors


# ─── Server management ────────────────────────────────────────────────────────

def wait_ready(addr: str, timeout: float = 60.0) -> bool:
    """Poll /health/ready until 200 or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{addr}/health/ready", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def start_server(infergo_bin: str, model_file: str, model_name: str, port: int = 9191) -> subprocess.Popen:
    """Launch infergo serve and return the process."""
    cmd = [
        infergo_bin, "serve",
        "--model", f"{model_name}:{model_file}",
        "--provider", "cuda",
        "--port", str(port),
    ]
    print(f"  Starting: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


# ─── Benchmark a single model ─────────────────────────────────────────────────

def benchmark_model(addr: str, model_name: str) -> BenchResult:
    print(f"\n=== Benchmarking {model_name} at {addr} ===")

    print("  Short prompt throughput ...", end=" ", flush=True)
    short_toks = run_throughput(addr, model_name, SHORT_PROMPT, n=10)
    print(f"{short_toks:.1f} tok/s")

    print("  Long prompt throughput  ...", end=" ", flush=True)
    long_toks = run_throughput(addr, model_name, LONG_PROMPT, n=4)
    print(f"{long_toks:.1f} tok/s")

    print("  P50 latency (c=4)       ...", end=" ", flush=True)
    p50, p95, errors = run_latency(addr, model_name, concurrency=4, total=20)
    print(f"P50={p50:.0f}ms  P95={p95:.0f}ms  errors={errors}")

    return BenchResult(
        model_name=model_name,
        short_toks=short_toks,
        long_toks=long_toks,
        p50_ms=p50,
        p95_ms=p95,
        errors=errors,
    )


# ─── Markdown output ──────────────────────────────────────────────────────────

def write_results(results: List[BenchResult], out_path: str):
    lines = [
        "# Multi-Model Benchmark — infergo CUDA",
        "",
        f"*Date: {time.strftime('%Y-%m-%d')}*  ",
        f"*GPU: RTX 5070 Ti, max_tokens={MAX_TOKENS}*",
        "",
        "## Results",
        "",
        "| Model | Params | Short tok/s | Long tok/s | P50 ms (c=4) | P95 ms | Errors |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        cfg = next((c for c in MODEL_CONFIGS if c["name"] == r.model_name), {})
        params = cfg.get("params", "?")
        lines.append(
            f"| {r.model_name} | {params} | {r.short_toks:.1f} | {r.long_toks:.1f}"
            f" | {r.p50_ms:.0f} | {r.p95_ms:.0f} | {r.errors} |"
        )
    lines += ["", "## Notes", ""]
    lines.append("- Short prompt: ~20 tokens, n=10 requests")
    lines.append("- Long prompt: ~230 tokens, n=4 requests")
    lines.append("- P50/P95: 20 concurrent requests at c=4")
    content = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(content)
    print(f"\nResults written to {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-name", default="llama3-8b-q4",
                    help="Model name as loaded in infergo (must match --model name: in serve)")
    ap.add_argument("--infergo-addr", default="http://localhost:9191",
                    help="infergo server address")
    ap.add_argument("--all-models", action="store_true",
                    help="Iterate through all MODEL_CONFIGS (starts/stops server each time)")
    ap.add_argument("--infergo-bin", default="./infergo",
                    help="Path to infergo binary (--all-models only)")
    ap.add_argument("--models-dir", default="~/cgo/models",
                    help="Directory containing model files (--all-models only)")
    ap.add_argument("--port", type=int, default=9191)
    ap.add_argument("--out", default="benchmarks/vs_python/results_multimodel.md")
    args = ap.parse_args()

    results: List[BenchResult] = []

    if args.all_models:
        models_dir = os.path.expanduser(args.models_dir)
        addr = f"http://localhost:{args.port}"
        for cfg in MODEL_CONFIGS:
            model_file = os.path.join(models_dir, cfg["file"])
            if not os.path.exists(model_file):
                print(f"Skipping {cfg['name']}: {model_file} not found")
                continue
            proc = start_server(args.infergo_bin, model_file, cfg["name"], args.port)
            try:
                if not wait_ready(addr, timeout=120):
                    print(f"  [error] server not ready for {cfg['name']}, skipping")
                    proc.terminate()
                    continue
                results.append(benchmark_model(addr, cfg["name"]))
            finally:
                proc.terminate()
                proc.wait()
                time.sleep(2)  # brief pause for port release
    else:
        # Single-model mode: assume server is already running.
        if not wait_ready(args.infergo_addr, timeout=10):
            sys.exit(f"Error: infergo not ready at {args.infergo_addr}/health/ready")
        results.append(benchmark_model(args.infergo_addr, args.model_name))

    if results:
        write_results(results, args.out)
    else:
        print("No results collected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
