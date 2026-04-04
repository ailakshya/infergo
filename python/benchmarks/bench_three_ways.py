"""
Benchmark: 5 ways to run inference from Python

1. Native bindings    — import infergo; llm = infergo.LLM(...)
2. HTTP / OpenAI SDK  — import openai; client = openai.OpenAI(base_url=...)
3. Core Python HTTP   — import requests; requests.post(...)  [no SDK overhead]
4. infergo CLI        — subprocess: infergo benchmark ...    [pure Go, no Python]
5. llama-cpp-python   — from llama_cpp import Llama          [traditional Python]

Approaches 2, 3, and 4 all hit the same infergo server so their inference
time is identical — the difference shows only Python/SDK overhead.
Approach 1 goes direct to the shared library (no server needed).
Approach 5 is the traditional in-process Python baseline.

Run:
    python benchmarks/bench_three_ways.py \
        --model  /path/to/llama3-8b-q4.gguf \
        --server http://localhost:9090 \
        --concurrency 1 4 16 \
        --n-requests 50

The script measures:
    - P50 / P95 / P99 latency (ms)
    - Throughput (req/s)
    - Tokens/s (generation speed)
    - RSS memory usage (MB)

Results are printed as a table and saved to bench_results.json.
"""

import argparse
import json
import os
import resource
import subprocess
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

PROMPT = "Explain what a transformer model is in exactly two sentences."
MAX_TOKENS = 64


# ─── helpers ──────────────────────────────────────────────────────────────────

def rss_mb() -> float:
    """Current process RSS in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform != "darwin" else 1024)


def percentile(data, p):
    data = sorted(data)
    idx = int(len(data) * p / 100)
    return data[min(idx, len(data) - 1)]


def run_bench(name: str, fn, n_requests: int, concurrency: int) -> dict:
    """
    fn() must return (elapsed_sec, n_tokens_generated).
    Runs n_requests total with `concurrency` workers.
    """
    latencies = []
    token_counts = []
    errors = 0
    rss_before = rss_mb()
    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(fn) for _ in range(n_requests)]
        for f in as_completed(futures):
            try:
                elapsed, n_toks = f.result()
                latencies.append(elapsed * 1000)  # ms
                token_counts.append(n_toks)
            except Exception as e:
                errors += 1

    wall_elapsed = time.perf_counter() - wall_start
    rss_after = rss_mb()

    if not latencies:
        return {"name": name, "concurrency": concurrency, "error": "all requests failed"}

    total_tokens = sum(token_counts)
    return {
        "name": name,
        "concurrency": concurrency,
        "n_requests": n_requests,
        "errors": errors,
        "p50_ms": round(percentile(latencies, 50), 1),
        "p95_ms": round(percentile(latencies, 95), 1),
        "p99_ms": round(percentile(latencies, 99), 1),
        "req_per_s": round(n_requests / wall_elapsed, 2),
        "tok_per_s": round(total_tokens / wall_elapsed, 1),
        "rss_delta_mb": round(rss_after - rss_before, 1),
    }


# ─── approach 1: native bindings ──────────────────────────────────────────────

def bench_native(model_path: str, n_requests: int, concurrency: int) -> dict:
    try:
        import infergo
    except ImportError:
        return {"name": "native", "concurrency": concurrency, "error": "infergo not installed (pip install -e python/)"}

    try:
        llm = infergo.LLM(model_path, gpu_layers=999, max_seqs=max(concurrency, 16))
    except Exception as e:
        return {"name": "native", "concurrency": concurrency, "error": str(e)}

    lock = threading.Lock()  # llm.generate is not thread-safe; serialize at C level via lock

    def one_request():
        t0 = time.perf_counter()
        with lock:
            # tokenize prompt to get n_prompt, then generate
            prompt_ids = llm.tokenize(PROMPT)
            text = llm.generate(PROMPT, max_tokens=MAX_TOKENS)
            n_toks = len(llm.tokenize(text)) if text else 1
        elapsed = time.perf_counter() - t0
        return elapsed, n_toks

    result = run_bench("native_bindings", one_request, n_requests, concurrency)
    llm.close()
    return result


# ─── approach 2: HTTP client (OpenAI-compatible) ──────────────────────────────

def bench_http(server_url: str, n_requests: int, concurrency: int) -> dict:
    try:
        import openai
    except ImportError:
        return {"name": "http_client", "concurrency": concurrency, "error": "openai not installed (pip install openai)"}

    client = openai.OpenAI(base_url=f"{server_url}/v1", api_key="none")

    def one_request():
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model="llama3-8b-q4",
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=MAX_TOKENS,
        )
        elapsed = time.perf_counter() - t0
        n_toks = resp.usage.completion_tokens if resp.usage else MAX_TOKENS // 2
        return elapsed, n_toks

    return run_bench("http_openai_client", one_request, n_requests, concurrency)


# ─── approach 3: llama-cpp-python (traditional) ───────────────────────────────

def bench_llama_cpp(model_path: str, n_requests: int, concurrency: int) -> dict:
    try:
        from llama_cpp import Llama
    except ImportError:
        return {"name": "llama_cpp_python", "concurrency": concurrency, "error": "llama-cpp-python not installed (pip install llama-cpp-python)"}

    try:
        llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=4096, verbose=False)
    except Exception as e:
        return {"name": "llama_cpp_python", "concurrency": concurrency, "error": str(e)}

    lock = threading.Lock()  # llama-cpp-python is not thread-safe

    def one_request():
        t0 = time.perf_counter()
        with lock:
            out = llm(PROMPT, max_tokens=MAX_TOKENS, echo=False)
        elapsed = time.perf_counter() - t0
        text = out["choices"][0]["text"]
        n_toks = out["usage"]["completion_tokens"] if "usage" in out else len(text.split())
        return elapsed, n_toks

    result = run_bench("llama_cpp_python", one_request, n_requests, concurrency)
    del llm
    return result


# ─── approach 4: core Python (raw requests — no OpenAI SDK) ──────────────────

def bench_core_python(server_url: str, n_requests: int, concurrency: int) -> dict:
    """
    Uses the built-in urllib.request (zero external deps) to POST directly
    to /v1/chat/completions.  Shows HTTP overhead without any SDK layer.
    """
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": "llama3-8b-q4",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
    }).encode()
    url = f"{server_url}/v1/chat/completions"

    def one_request():
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read())
        except urllib.error.URLError as e:
            raise RuntimeError(str(e))
        elapsed = time.perf_counter() - t0
        n_toks = body.get("usage", {}).get("completion_tokens", MAX_TOKENS // 2)
        return elapsed, n_toks

    return run_bench("core_python_urllib", one_request, n_requests, concurrency)


# ─── approach 5: infergo benchmark CLI (pure Go, no Python overhead) ──────────

def bench_infergo_cli(server_url: str, n_requests: int, concurrency: int) -> dict:
    """
    Runs `infergo benchmark` as a subprocess.  This is pure Go with no
    Python interpreter in the hot path — it shows the ceiling performance
    achievable from the server.
    """
    binary = os.environ.get("INFERGO_BIN", "infergo")

    # Check binary exists
    try:
        subprocess.run([binary, "--help"], capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {
            "name": "infergo_cli",
            "concurrency": concurrency,
            "error": f"'{binary}' not found. Set INFERGO_BIN or add infergo to PATH.",
        }

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [
                binary, "benchmark",
                "--addr",        server_url,
                "--model",       "llama3-8b-q4",
                "--prompt",      PROMPT,
                "--max-tokens",  str(MAX_TOKENS),
                "--requests",    str(n_requests),
                "--concurrency", str(concurrency),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return {"name": "infergo_cli", "concurrency": concurrency, "error": "timed out"}

    wall_elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        return {
            "name": "infergo_cli",
            "concurrency": concurrency,
            "error": proc.stderr.strip() or f"exit {proc.returncode}",
        }

    # Parse JSON output from `infergo benchmark --json`
    try:
        data = json.loads(proc.stdout)
        return {
            "name": "infergo_cli",
            "concurrency": concurrency,
            "n_requests": n_requests,
            "errors": data.get("errors", 0),
            "p50_ms": data.get("p50_ms", 0),
            "p95_ms": data.get("p95_ms", 0),
            "p99_ms": data.get("p99_ms", 0),
            "req_per_s": data.get("req_per_s", round(n_requests / wall_elapsed, 2)),
            "tok_per_s": data.get("tok_per_s", 0),
            "rss_delta_mb": 0.0,  # measured in Go process, not this Python process
        }
    except (json.JSONDecodeError, KeyError):
        # Parse infergo benchmark plain-text output:
        #   Throughput:   2.1 req/s
        #   Latency P50:  467 ms
        #   Latency P99:  486 ms
        import re
        out = proc.stdout
        def _g(pat, default=0.0):
            m = re.search(pat, out)
            return float(m.group(1)) if m else default
        return {
            "name": "infergo_cli",
            "concurrency": concurrency,
            "n_requests": n_requests,
            "errors": int(_g(r"(\d+) errors", 0)),
            "p50_ms": _g(r"P50:\s+([\d.]+)\s*ms"),
            "p95_ms": _g(r"P95:\s+([\d.]+)\s*ms"),
            "p99_ms": _g(r"P99:\s+([\d.]+)\s*ms"),
            "req_per_s": _g(r"Throughput:\s+([\d.]+)\s*req/s"),
            "tok_per_s": _g(r"tok/s:\s+([\d.]+)"),
            "rss_delta_mb": 0.0,
        }


# ─── main ─────────────────────────────────────────────────────────────────────

def print_table(results: list[dict]):
    header = f"{'Approach':<22} {'c':>3} {'P50':>8} {'P95':>8} {'req/s':>7} {'tok/s':>7} {'ΔRSS MB':>8} {'err':>4}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['name']:<22} {r.get('concurrency',0):>3}  ERROR: {r['error']}")
        else:
            print(
                f"{r['name']:<22} {r['concurrency']:>3} "
                f"{r['p50_ms']:>7.0f}ms {r['p95_ms']:>7.0f}ms "
                f"{r['req_per_s']:>7.2f} {r['tok_per_s']:>7.1f} "
                f"{r['rss_delta_mb']:>8.1f} {r['errors']:>4}"
            )
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",       default="models/llama3-8b-q4.gguf", help="Path to GGUF model")
    ap.add_argument("--server",      default="http://localhost:9090",     help="Running infergo server URL")
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1, 4], help="Concurrency levels to test")
    ap.add_argument("--n-requests",  type=int,  default=30,               help="Total requests per scenario")
    ap.add_argument("--approaches",  nargs="+",
                    choices=["native", "http", "corepython", "infergo", "llamacpp", "all"],
                    default=["all"])
    ap.add_argument("--output",      default="bench_results.json",        help="JSON output file")
    args = ap.parse_args()

    do_all = "all" in args.approaches
    all_results = []

    for c in args.concurrency:
        print(f"\nConcurrency = {c}, {args.n_requests} requests per approach")

        if do_all or "native" in args.approaches:
            print("  [1/5] Native bindings (infergo Python package)...")
            all_results.append(bench_native(args.model, args.n_requests, c))

        if do_all or "http" in args.approaches:
            print("  [2/5] HTTP client (OpenAI SDK → infergo server)...")
            all_results.append(bench_http(args.server, args.n_requests, c))

        if do_all or "corepython" in args.approaches:
            print("  [3/5] Core Python (urllib, zero deps → infergo server)...")
            all_results.append(bench_core_python(args.server, args.n_requests, c))

        if do_all or "infergo" in args.approaches:
            print("  [4/5] infergo CLI benchmark (pure Go, no Python)...")
            all_results.append(bench_infergo_cli(args.server, args.n_requests, c))

        if do_all or "llamacpp" in args.approaches:
            print("  [5/5] llama-cpp-python (traditional Python, in-process)...")
            all_results.append(bench_llama_cpp(args.model, args.n_requests, c))

    print_table(all_results)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
