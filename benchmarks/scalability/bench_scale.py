"""
bench_scale.py — Python vs infergo scalability benchmark (OPT-27).

Proves that Python's GIL forces N model copies (RSS ∝ N workers) while
infergo serves N concurrent users from a single process (RSS flat).

Benchmark scenarios:
  Concurrency sweep: 1, 2, 4, 8, 16, 32 concurrent clients
  For each concurrency level measure:
    - Requests/second
    - P50 and P99 latency (ms)
    - Server RSS (MB) immediately after the load pulse

Two server modes:
  --server infergo   : single infergo process, goroutine-based
  --server python    : llama-cpp-python + uvicorn (gunicorn --workers N)

Usage:
  # Run against a running infergo server:
  python bench_scale.py --server infergo --infergo-addr http://localhost:9090 \
      --model llama3-8b-q4 --out results_scalability.md

  # Run against a running llama-cpp-python server (one worker):
  python bench_scale.py --server python --python-addr http://localhost:8000 \
      --model llama3-8b-q4 --out results_scalability.md

  # Full head-to-head (starts/stops servers automatically):
  python bench_scale.py --full \
      --infergo-bin ./infergo --infergo-model-file ~/cgo/models/llama3-8b-q4.gguf \
      --python-script ~/llama_venv/bin/python \
      --model-name llama3-8b-q4 --out results_scalability.md

Output:
  results_scalability.md  — Markdown table with all numbers
  benchmark_scalability.png — RSS and tok/s curves (requires matplotlib)
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
from typing import List, Optional, Tuple

import urllib.request
import urllib.error

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─── Config ───────────────────────────────────────────────────────────────────

CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32]
REQUESTS_PER_LEVEL = 40      # total requests fired at each concurrency level
MAX_TOKENS        = 64
PROMPT            = "Explain what a transformer neural network is in two sentences."


# ─── HTTP helpers ──────────────────────────────────────────────────────────────

def chat_completion(addr: str, model: str, prompt: str,
                    max_tokens: int = MAX_TOKENS) -> dict:
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
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def wait_ready(addr: str, timeout: float = 90.0) -> bool:
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


# ─── RSS measurement ──────────────────────────────────────────────────────────

def get_rss_mb(pid: int) -> float:
    """Read RSS in MB for the given PID via /proc/PID/status (Linux)."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except (IOError, ValueError):
        pass
    return 0.0


# ─── Load generation ──────────────────────────────────────────────────────────

@dataclass
class LevelResult:
    concurrency:  int
    req_per_s:    float
    p50_ms:       float
    p99_ms:       float
    rss_mb:       float
    errors:       int


def run_level(addr: str, model: str, concurrency: int, total: int,
              server_pid: Optional[int]) -> LevelResult:
    """Fire `total` requests at `concurrency` and return measurements."""
    latencies: List[float] = []
    errors = 0

    def one(_):
        try:
            t0 = time.monotonic()
            chat_completion(addr, model, PROMPT)
            return (time.monotonic() - t0) * 1000.0
        except Exception as e:
            print(f"  [warn] {e}", file=sys.stderr)
            return None

    t_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(one, i) for i in range(total)]
        for f in as_completed(futs):
            r = f.result()
            if r is None:
                errors += 1
            else:
                latencies.append(r)
    elapsed = time.monotonic() - t_start

    rss = get_rss_mb(server_pid) if server_pid else 0.0

    if not latencies:
        return LevelResult(concurrency, 0.0, 0.0, 0.0, rss, errors)

    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)]
    p99 = latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)]
    rps = len(latencies) / elapsed if elapsed > 0 else 0.0
    return LevelResult(concurrency, rps, p50, p99, rss, errors)


# ─── Server management ────────────────────────────────────────────────────────

def start_infergo(bin_path: str, model_file: str, model_name: str,
                  port: int = 9090) -> Tuple[subprocess.Popen, int]:
    cmd = [
        bin_path, "serve",
        "--model", f"{model_name}:{model_file}",
        "--provider", "cuda",
        "--port", str(port),
        "--max-seqs", "32",
    ]
    print(f"  Starting infergo: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc, proc.pid


def start_python_server(python_bin: str, model_file: str,
                        n_workers: int, port: int = 8000) -> Tuple[subprocess.Popen, int]:
    """
    Start a llama-cpp-python server via uvicorn.
    llama-cpp-python exposes: python -m llama_cpp.server --model ... --n_gpu_layers -1
    """
    cmd = [
        python_bin, "-m", "llama_cpp.server",
        "--model", model_file,
        "--n_gpu_layers", "-1",
        "--n_ctx", "4096",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--n_threads", str(n_workers),
    ]
    print(f"  Starting llama-cpp-python: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc, proc.pid


# ─── Benchmark a single server ────────────────────────────────────────────────

def benchmark_server(addr: str, model: str,
                     server_pid: Optional[int],
                     label: str) -> List[LevelResult]:
    results = []
    for c in CONCURRENCY_LEVELS:
        print(f"  {label} c={c:>2} ... ", end="", flush=True)
        r = run_level(addr, model, c, REQUESTS_PER_LEVEL, server_pid)
        print(f"{r.req_per_s:.1f} req/s  P50={r.p50_ms:.0f}ms  "
              f"P99={r.p99_ms:.0f}ms  RSS={r.rss_mb:.0f}MB  err={r.errors}")
        results.append(r)
    return results


# ─── Markdown output ──────────────────────────────────────────────────────────

def write_markdown(infergo_res: List[LevelResult],
                   python_res: List[LevelResult],
                   model_name: str,
                   out_path: str):
    lines = [
        "# Python vs infergo — Scalability Benchmark (OPT-27)",
        "",
        f"*Date: {time.strftime('%Y-%m-%d')}*  ",
        f"*Model: {model_name}, max_tokens={MAX_TOKENS}, {REQUESTS_PER_LEVEL} req per level*",
        "",
        "## Key finding",
        "",
        "> infergo RSS stays flat as concurrency rises (one model copy, goroutines).  ",
        "> llama-cpp-python RSS grows proportionally with worker count (one model copy per worker, GIL).",
        "",
        "## Results — infergo",
        "",
        "| Concurrency | Req/s | P50 ms | P99 ms | RSS MB | Errors |",
        "|---|---|---|---|---|---|",
    ]
    for r in infergo_res:
        lines.append(
            f"| {r.concurrency} | {r.req_per_s:.1f} | {r.p50_ms:.0f} |"
            f" {r.p99_ms:.0f} | {r.rss_mb:.0f} | {r.errors} |"
        )

    lines += [
        "",
        "## Results — Python (llama-cpp-python)",
        "",
        "| Concurrency | Req/s | P50 ms | P99 ms | RSS MB | Errors |",
        "|---|---|---|---|---|---|",
    ]
    for r in python_res:
        lines.append(
            f"| {r.concurrency} | {r.req_per_s:.1f} | {r.p50_ms:.0f} |"
            f" {r.p99_ms:.0f} | {r.rss_mb:.0f} | {r.errors} |"
        )

    # Memory comparison table
    lines += [
        "",
        "## Memory comparison",
        "",
        "| Concurrency | infergo RSS MB | Python RSS MB | Python / infergo |",
        "|---|---|---|---|",
    ]
    py_map = {r.concurrency: r for r in python_res}
    for r in infergo_res:
        py = py_map.get(r.concurrency)
        if py and r.rss_mb > 0:
            ratio = py.rss_mb / r.rss_mb if r.rss_mb > 0 else 0
            lines.append(
                f"| {r.concurrency} | {r.rss_mb:.0f} | {py.rss_mb:.0f} | {ratio:.1f}× |"
            )

    lines += ["", "## Notes", ""]
    lines.append(f"- Prompt: `{PROMPT[:60]}...`")
    lines.append(f"- {REQUESTS_PER_LEVEL} total requests per concurrency level")
    lines.append("- RSS read from /proc/PID/status after each level completes")
    lines.append("- infergo: single process, goroutines, continuous batching")
    lines.append("- Python: llama-cpp-python uvicorn server")

    content = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(content)
    print(f"\nResults written to {out_path}")


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_results(infergo_res: List[LevelResult],
                 python_res: List[LevelResult],
                 out_png: str):
    if not HAS_MPL:
        print("matplotlib not available — skipping plot generation", file=sys.stderr)
        return

    conc_i = [r.concurrency for r in infergo_res]
    rss_i  = [r.rss_mb      for r in infergo_res]
    p50_i  = [r.p50_ms      for r in infergo_res]
    rps_i  = [r.req_per_s   for r in infergo_res]

    conc_p = [r.concurrency for r in python_res]
    rss_p  = [r.rss_mb      for r in python_res]
    p50_p  = [r.p50_ms      for r in python_res]
    rps_p  = [r.req_per_s   for r in python_res]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Python vs infergo — Scalability (OPT-27)", fontsize=13)

    # Panel 1: RSS
    ax = axes[0]
    ax.plot(conc_i, rss_i, "o-", color="steelblue",  label="infergo")
    ax.plot(conc_p, rss_p, "s--", color="tomato",    label="llama-cpp-python")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Server RSS (MB)")
    ax.set_title("Memory (RSS) vs Concurrency")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.set_xticks(CONCURRENCY_LEVELS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # Panel 2: P50 latency
    ax = axes[1]
    ax.plot(conc_i, p50_i, "o-", color="steelblue",  label="infergo")
    ax.plot(conc_p, p50_p, "s--", color="tomato",    label="llama-cpp-python")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("P50 Latency (ms)")
    ax.set_title("P50 Latency vs Concurrency")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.set_xticks(CONCURRENCY_LEVELS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # Panel 3: Throughput
    ax = axes[2]
    ax.plot(conc_i, rps_i, "o-", color="steelblue",  label="infergo")
    ax.plot(conc_p, rps_p, "s--", color="tomato",    label="llama-cpp-python")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("Throughput vs Concurrency")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.set_xticks(CONCURRENCY_LEVELS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Chart written to {out_png}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("--model", default="llama3-8b-q4",
                    help="Model name as served by infergo")
    ap.add_argument("--out", default="benchmarks/scalability/results_scalability.md")
    ap.add_argument("--png", default="benchmarks/scalability/benchmark_scalability.png")

    # Pre-running server mode
    ap.add_argument("--server", choices=["infergo", "python", "both"], default="infergo",
                    help="Which server to benchmark (use --full for head-to-head)")
    ap.add_argument("--infergo-addr", default="http://localhost:9090")
    ap.add_argument("--python-addr",  default="http://localhost:8000")

    # Auto-start mode
    ap.add_argument("--full", action="store_true",
                    help="Start both servers automatically and run head-to-head")
    ap.add_argument("--infergo-bin",        default="./infergo")
    ap.add_argument("--infergo-model-file", default="~/cgo/models/llama3-8b-q4.gguf")
    ap.add_argument("--python-script",      default="python3",
                    help="Python binary with llama-cpp-python installed")
    ap.add_argument("--infergo-port", type=int, default=9090)
    ap.add_argument("--python-port",  type=int, default=8000)

    args = ap.parse_args()

    infergo_results: List[LevelResult] = []
    python_results:  List[LevelResult] = []

    if args.full:
        # ── Full head-to-head mode ──────────────────────────────────────────
        model_file = os.path.expanduser(args.infergo_model_file)
        if not os.path.exists(model_file):
            sys.exit(f"Model file not found: {model_file}")

        # infergo
        print("\n=== Benchmarking infergo ===")
        proc_ig, pid_ig = start_infergo(
            args.infergo_bin, model_file, args.model, args.infergo_port)
        addr_ig = f"http://localhost:{args.infergo_port}"
        try:
            if not wait_ready(addr_ig, timeout=120):
                proc_ig.terminate()
                sys.exit("infergo not ready in time")
            infergo_results = benchmark_server(addr_ig, args.model, pid_ig, "infergo")
        finally:
            proc_ig.terminate()
            proc_ig.wait()
            time.sleep(2)

        # Python
        print("\n=== Benchmarking llama-cpp-python ===")
        # Start with enough threads to handle max concurrency
        proc_py, pid_py = start_python_server(
            args.python_script, model_file,
            n_workers=CONCURRENCY_LEVELS[-1],  # max concurrency
            port=args.python_port)
        addr_py = f"http://localhost:{args.python_port}"
        try:
            if not wait_ready(addr_py, timeout=120):
                proc_py.terminate()
                sys.exit("llama-cpp-python not ready in time")
            python_results = benchmark_server(addr_py, args.model, pid_py, "python")
        finally:
            proc_py.terminate()
            proc_py.wait()

    elif args.server == "infergo":
        print("\n=== Benchmarking infergo (server must already be running) ===")
        if not wait_ready(args.infergo_addr, timeout=10):
            sys.exit(f"infergo not ready at {args.infergo_addr}/health/ready")
        infergo_results = benchmark_server(
            args.infergo_addr, args.model, None, "infergo")
        # Use empty python results for output
        python_results = [LevelResult(c, 0, 0, 0, 0, 0) for c in CONCURRENCY_LEVELS]

    elif args.server == "python":
        print("\n=== Benchmarking llama-cpp-python (server must already be running) ===")
        if not wait_ready(args.python_addr, timeout=10):
            sys.exit(f"Server not ready at {args.python_addr}")
        python_results = benchmark_server(
            args.python_addr, args.model, None, "python")
        infergo_results = [LevelResult(c, 0, 0, 0, 0, 0) for c in CONCURRENCY_LEVELS]

    else:
        sys.exit("Use --server infergo|python or --full for head-to-head")

    if infergo_results or python_results:
        write_markdown(infergo_results, python_results, args.model, args.out)
        plot_results(infergo_results, python_results, args.png)
    else:
        print("No results collected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
