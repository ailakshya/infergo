"""
Benchmark: 3 ways to use infergo from Python

1. Native bindings  — import infergo; llm = infergo.LLM(...)
2. HTTP client      — import openai; client = openai.OpenAI(base_url=...)
3. llama-cpp-python — from llama_cpp import Llama (traditional Python approach)

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
            text = llm.generate(PROMPT, max_tokens=MAX_TOKENS)
        elapsed = time.perf_counter() - t0
        # Rough token count from whitespace splitting
        n_toks = len(text.split())
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
    ap.add_argument("--approaches",  nargs="+", choices=["native", "http", "llamacpp", "all"], default=["all"])
    ap.add_argument("--output",      default="bench_results.json",        help="JSON output file")
    args = ap.parse_args()

    do_all = "all" in args.approaches
    all_results = []

    for c in args.concurrency:
        print(f"\nConcurrency = {c}, {args.n_requests} requests per approach")

        if do_all or "native" in args.approaches:
            print("  Running native bindings...")
            all_results.append(bench_native(args.model, args.n_requests, c))

        if do_all or "http" in args.approaches:
            print("  Running HTTP client...")
            all_results.append(bench_http(args.server, args.n_requests, c))

        if do_all or "llamacpp" in args.approaches:
            print("  Running llama-cpp-python...")
            all_results.append(bench_llama_cpp(args.model, args.n_requests, c))

    print_table(all_results)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
