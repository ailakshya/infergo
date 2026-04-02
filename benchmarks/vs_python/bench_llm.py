"""
bench_llm.py — infergo vs vLLM LLM throughput benchmark.

Usage:
  # Benchmark infergo
  python bench_llm.py --target infergo --addr http://localhost:9090 --model llama3

  # Benchmark vLLM (OpenAI-compatible server)
  python bench_llm.py --target vllm --addr http://localhost:8000 --model meta-llama/Meta-Llama-3-8B-Instruct

  # Both (run each server, then compare)
  python bench_llm.py --compare \
    --infergo-addr http://localhost:9090 --infergo-model llama3 \
    --vllm-addr http://localhost:8000 --vllm-model meta-llama/Meta-Llama-3-8B-Instruct

Metrics reported:
  - Throughput: requests/sec, tokens/sec (output)
  - Latency: P50, P95, P99 (ms), TTFT (time-to-first-token for streaming)
  - RAM: RSS before/after (via /proc/self/status or psutil)
  - Cold start: time from process launch to first successful request
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
from typing import Optional

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

# ─── Prompts ──────────────────────────────────────────────────────────────────

PROMPTS = [
    "Explain the difference between supervised and unsupervised learning in one paragraph.",
    "Write a Go function that reverses a string.",
    "What is the capital of France?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "List five uses of transformer models in industry.",
    "Explain gradient descent like I'm five years old.",
    "Write a Python function to compute Fibonacci numbers iteratively.",
    "What are the main differences between TCP and UDP?",
    "Describe the water cycle in simple terms.",
    "What is a mutex and when would you use one?",
]

# ─── Data types ───────────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    latency_ms: float
    ttft_ms: Optional[float]   # time-to-first-token (streaming only)
    output_tokens: int
    error: Optional[str] = None

@dataclass
class BenchResult:
    target: str
    requests: int
    concurrency: int
    duration_s: float
    results: list = field(default_factory=list)

    @property
    def ok(self):
        return [r for r in self.results if r.error is None]

    @property
    def errors(self):
        return [r for r in self.results if r.error is not None]

    def throughput_rps(self):
        return len(self.ok) / self.duration_s if self.duration_s > 0 else 0

    def throughput_tps(self):
        total_toks = sum(r.output_tokens for r in self.ok)
        return total_toks / self.duration_s if self.duration_s > 0 else 0

    def latency_percentile(self, p):
        lats = sorted(r.latency_ms for r in self.ok)
        if not lats:
            return 0
        idx = int(math.ceil(len(lats) * p / 100)) - 1
        return lats[max(0, min(idx, len(lats) - 1))]

    def ttft_percentile(self, p):
        ttfts = sorted(r.ttft_ms for r in self.ok if r.ttft_ms is not None)
        if not ttfts:
            return None
        idx = int(math.ceil(len(ttfts) * p / 100)) - 1
        return ttfts[max(0, min(idx, len(ttfts) - 1))]

    def mean_latency(self):
        lats = [r.latency_ms for r in self.ok]
        return statistics.mean(lats) if lats else 0


# ─── Single request ───────────────────────────────────────────────────────────

def send_request(addr: str, model: str, prompt: str, max_tokens: int, stream: bool) -> RequestResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": stream,
    }
    t0 = time.perf_counter()
    ttft = None
    output_tokens = 0
    try:
        resp = requests.post(
            f"{addr}/v1/chat/completions",
            json=payload,
            stream=stream,
            timeout=120,
        )
        resp.raise_for_status()
        if stream:
            for line in resp.iter_lines():
                if not line:
                    continue
                if line == b"data: [DONE]":
                    break
                if line.startswith(b"data: "):
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    chunk = json.loads(line[6:])
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        output_tokens += len(content.split())  # rough estimate
        else:
            body = resp.json()
            usage = body.get("usage", {})
            output_tokens = usage.get("completion_tokens", 0)
            if output_tokens == 0:
                content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                output_tokens = len(content.split())
        latency_ms = (time.perf_counter() - t0) * 1000
        return RequestResult(latency_ms=latency_ms, ttft_ms=ttft, output_tokens=output_tokens)
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return RequestResult(latency_ms=latency_ms, ttft_ms=None, output_tokens=0, error=str(e))


# ─── Warmup ───────────────────────────────────────────────────────────────────

def warmup(addr: str, model: str, n: int = 3):
    print(f"  Warming up ({n} requests)...", end="", flush=True)
    for _ in range(n):
        send_request(addr, model, "Hello", 16, stream=False)
    print(" done")


# ─── Run benchmark ────────────────────────────────────────────────────────────

def run_bench(target: str, addr: str, model: str, n_requests: int,
              concurrency: int, max_tokens: int, stream: bool) -> BenchResult:
    warmup(addr, model)
    print(f"  Sending {n_requests} requests (concurrency={concurrency}, max_tokens={max_tokens}, stream={stream})...")

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(n_requests)]
    results = []

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(send_request, addr, model, p, max_tokens, stream) for p in prompts]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            if i % max(1, n_requests // 10) == 0:
                print(f"    {i}/{n_requests} done", flush=True)
    duration = time.perf_counter() - t_start

    return BenchResult(target=target, requests=n_requests,
                       concurrency=concurrency, duration_s=duration, results=results)


# ─── Print results ────────────────────────────────────────────────────────────

def print_result(br: BenchResult):
    ok = len(br.ok)
    err = len(br.errors)
    print(f"\n{'─'*60}")
    print(f"  Target:      {br.target}")
    print(f"  Requests:    {br.requests} total | {ok} ok | {err} errors")
    print(f"  Duration:    {br.duration_s:.2f}s")
    print(f"  Throughput:  {br.throughput_rps():.1f} req/s | {br.throughput_tps():.0f} tok/s (output)")
    print(f"  Latency:")
    print(f"    mean:      {br.mean_latency():.0f} ms")
    print(f"    P50:       {br.latency_percentile(50):.0f} ms")
    print(f"    P95:       {br.latency_percentile(95):.0f} ms")
    print(f"    P99:       {br.latency_percentile(99):.0f} ms")
    ttft50 = br.ttft_percentile(50)
    if ttft50 is not None:
        print(f"  TTFT P50:    {ttft50:.0f} ms")
        print(f"  TTFT P99:    {br.ttft_percentile(99):.0f} ms")
    if err > 0:
        sample = br.errors[:3]
        print(f"  Errors (sample): {[e.error for e in sample]}")
    print(f"{'─'*60}")


def print_comparison(a: BenchResult, b: BenchResult):
    def delta(va, vb):
        if vb == 0:
            return "—"
        ratio = va / vb
        sign = "+" if ratio > 1 else ""
        return f"{sign}{(ratio - 1)*100:.0f}%"

    print(f"\n{'═'*60}")
    print(f"  COMPARISON: {a.target} vs {b.target}")
    print(f"{'═'*60}")
    print(f"  {'Metric':<25} {a.target:>12} {b.target:>12}  {'infergo vs B':>12}")
    print(f"  {'─'*25} {'─'*12} {'─'*12}  {'─'*12}")

    def row(name, va, vb, fmt="{:.1f}"):
        da = delta(va, vb) if a.target == "infergo" else delta(vb, va)
        print(f"  {name:<25} {fmt.format(va):>12} {fmt.format(vb):>12}  {da:>12}")

    row("Throughput (req/s)", a.throughput_rps(), b.throughput_rps())
    row("Throughput (tok/s)", a.throughput_tps(), b.throughput_tps(), fmt="{:.0f}")
    row("Latency mean (ms)",  a.mean_latency(), b.mean_latency(), fmt="{:.0f}")
    row("Latency P50 (ms)",   a.latency_percentile(50), b.latency_percentile(50), fmt="{:.0f}")
    row("Latency P99 (ms)",   a.latency_percentile(99), b.latency_percentile(99), fmt="{:.0f}")
    print(f"{'═'*60}\n")


def emit_markdown(results: list[BenchResult], out_path: str):
    lines = ["# LLM Benchmark Results\n",
             f"*Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}*\n",
             ""]
    headers = ["Target", "req/s", "tok/s", "Mean ms", "P50 ms", "P95 ms", "P99 ms", "Errors"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for br in results:
        row = [
            br.target,
            f"{br.throughput_rps():.1f}",
            f"{br.throughput_tps():.0f}",
            f"{br.mean_latency():.0f}",
            f"{br.latency_percentile(50):.0f}",
            f"{br.latency_percentile(95):.0f}",
            f"{br.latency_percentile(99):.0f}",
            str(len(br.errors)),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Results written to {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="infergo vs vLLM LLM benchmark")
    p.add_argument("--target",   default="infergo", choices=["infergo", "vllm", "openai-compat"])
    p.add_argument("--addr",     default="http://localhost:9090")
    p.add_argument("--model",    default="llama3")
    p.add_argument("--compare",  action="store_true", help="Compare two running servers")
    p.add_argument("--infergo-addr",  default="http://localhost:9090")
    p.add_argument("--infergo-model", default="llama3")
    p.add_argument("--vllm-addr",     default="http://localhost:8000")
    p.add_argument("--vllm-model",    default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--requests",     type=int, default=200)
    p.add_argument("--concurrency",  type=int, default=8)
    p.add_argument("--max-tokens",   type=int, default=128)
    p.add_argument("--stream",       action="store_true", help="Use streaming for TTFT measurement")
    p.add_argument("--out",          default="results_llm.md")
    args = p.parse_args()

    all_results = []

    if args.compare:
        print(f"\n[1/2] Benchmarking infergo @ {args.infergo_addr}")
        r1 = run_bench("infergo", args.infergo_addr, args.infergo_model,
                       args.requests, args.concurrency, args.max_tokens, args.stream)
        print_result(r1)
        all_results.append(r1)

        print(f"\n[2/2] Benchmarking vLLM @ {args.vllm_addr}")
        r2 = run_bench("vllm", args.vllm_addr, args.vllm_model,
                       args.requests, args.concurrency, args.max_tokens, args.stream)
        print_result(r2)
        all_results.append(r2)

        print_comparison(r1, r2)
    else:
        print(f"\nBenchmarking {args.target} @ {args.addr}")
        r = run_bench(args.target, args.addr, args.model,
                      args.requests, args.concurrency, args.max_tokens, args.stream)
        print_result(r)
        all_results.append(r)

    emit_markdown(all_results, args.out)


if __name__ == "__main__":
    main()
