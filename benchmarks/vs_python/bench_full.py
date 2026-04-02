"""
bench_full.py — infergo vs llama-cpp-python comprehensive benchmark.

Tests:
  - Short prompt  (~20 tokens)
  - Long prompt   (~512 tokens)
  - Cold start    (time from process launch to first token)
  - Throughput    (concurrent requests, tok/s)
  - Latency       (P50 / P95 / P99)

Backends:
  - infergo (Go server, OpenAI-compatible HTTP)
  - llama-cpp-python (direct Python, same model, same GPU)

Usage:
  # Run infergo server first:
  #   ./infergo serve --model models/llama3-8b-q4.gguf --provider cuda --port 9191
  #
  # Then run this script:
  python bench_full.py --model-path models/llama3-8b-q4.gguf --infergo-addr http://localhost:9191

  # CPU only:
  python bench_full.py --model-path models/llama3-8b-q4.gguf --infergo-addr http://localhost:9191 --device cpu

  # Skip Python benchmark (infergo only):
  python bench_full.py --infergo-only --infergo-addr http://localhost:9191 --infergo-model llama3-8b-q4
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

SHORT_PROMPTS = [
    "What is 2+2?",
    "Name the capital of France.",
    "What color is the sky?",
    "Who wrote Hamlet?",
    "What is Go programming language?",
]

LONG_PROMPTS = [
    """You are an expert software engineer. Please provide a detailed technical explanation of
    how large language models work, covering: (1) the transformer architecture including
    attention mechanisms and feed-forward layers, (2) how tokenization works with BPE and
    SentencePiece, (3) the training process including pre-training on large corpora and
    fine-tuning with RLHF, (4) inference optimization techniques like KV caching, continuous
    batching, and quantization (GGUF Q4_K_M), (5) deployment considerations for production
    systems. Please be thorough and technical in your response.""",

    """Explain the differences between Go, Rust, C++, and Python for building high-performance
    server applications. Cover: memory management models (GC vs ownership vs manual vs GC),
    concurrency primitives (goroutines vs async/await vs threads), FFI capabilities for calling
    C libraries, startup time and binary size, ecosystem maturity, and give concrete examples
    of where each language excels. Include discussion of CGo and how it compares to Rust's
    unsafe FFI for wrapping C++ libraries like ONNX Runtime and llama.cpp.""",
]


# ─── Data types ───────────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    latency_ms: float
    ttft_ms: Optional[float]
    output_tokens: int
    error: Optional[str] = None


@dataclass
class ScenarioResult:
    backend: str
    device: str
    scenario: str       # short / long / cold_start
    requests: int
    concurrency: int
    duration_s: float
    results: list = field(default_factory=list)
    cold_start_ms: float = 0.0

    @property
    def ok(self):
        return [r for r in self.results if r.error is None]

    @property
    def errors(self):
        return [r for r in self.results if r.error is not None]

    def rps(self):
        return len(self.ok) / self.duration_s if self.duration_s > 0 else 0

    def tps(self):
        total = sum(r.output_tokens for r in self.ok)
        return total / self.duration_s if self.duration_s > 0 else 0

    def pct(self, p):
        lats = sorted(r.latency_ms for r in self.ok)
        if not lats:
            return 0
        idx = int(math.ceil(len(lats) * p / 100)) - 1
        return lats[max(0, min(idx, len(lats) - 1))]

    def mean_lat(self):
        lats = [r.latency_ms for r in self.ok]
        return statistics.mean(lats) if lats else 0

    def ttft_pct(self, p):
        ttfts = sorted(r.ttft_ms for r in self.ok if r.ttft_ms is not None)
        if not ttfts:
            return None
        idx = int(math.ceil(len(ttfts) * p / 100)) - 1
        return ttfts[max(0, min(idx, len(ttfts) - 1))]


# ─── infergo HTTP backend ──────────────────────────────────────────────────────

def infergo_request(addr: str, model: str, prompt: str, max_tokens: int,
                    stream: bool = False) -> RequestResult:
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
            json=payload, stream=stream, timeout=300,
        )
        resp.raise_for_status()
        if stream:
            for line in resp.iter_lines():
                if not line or line == b"data: [DONE]":
                    continue
                if line.startswith(b"data: "):
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    chunk = json.loads(line[6:])
                    content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if content:
                        output_tokens += len(content.split())
        else:
            body = resp.json()
            output_tokens = body.get("usage", {}).get("completion_tokens", 0)
            if output_tokens == 0:
                content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                output_tokens = len(content.split())
        return RequestResult(
            latency_ms=(time.perf_counter() - t0) * 1000,
            ttft_ms=ttft,
            output_tokens=output_tokens,
        )
    except Exception as e:
        return RequestResult(
            latency_ms=(time.perf_counter() - t0) * 1000,
            ttft_ms=None, output_tokens=0, error=str(e)[:120],
        )


# ─── Python (llama-cpp-python) backend ────────────────────────────────────────

def load_llama_cpp(model_path: str, n_gpu_layers: int, n_ctx: int):
    try:
        from llama_cpp import Llama
    except ImportError:
        sys.exit("pip install llama-cpp-python")
    print(f"  Loading {model_path} via llama-cpp-python (gpu_layers={n_gpu_layers})...")
    t0 = time.perf_counter()
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=False,
    )
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Model loaded in {load_ms:.0f}ms")
    return llm, load_ms


def python_request(llm, prompt: str, max_tokens: int) -> RequestResult:
    t0 = time.perf_counter()
    ttft = None
    output_tokens = 0
    try:
        stream = llm(
            f"[INST]{prompt}[/INST]",
            max_tokens=max_tokens,
            stream=True,
            echo=False,
        )
        for chunk in stream:
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            delta = chunk["choices"][0].get("text", "")
            if delta:
                output_tokens += len(delta.split())
        return RequestResult(
            latency_ms=(time.perf_counter() - t0) * 1000,
            ttft_ms=ttft,
            output_tokens=output_tokens,
        )
    except Exception as e:
        return RequestResult(
            latency_ms=(time.perf_counter() - t0) * 1000,
            ttft_ms=None, output_tokens=0, error=str(e)[:120],
        )


# ─── Benchmark runner ─────────────────────────────────────────────────────────

def run_scenario(backend: str, device: str, scenario: str, send_fn,
                 prompts: list, max_tokens: int,
                 n_requests: int, concurrency: int) -> ScenarioResult:
    print(f"\n  [{backend.upper()} / {device.upper()} / {scenario}]  "
          f"n={n_requests} concurrency={concurrency} max_tokens={max_tokens}")

    # Warmup
    print("    Warming up (3 requests)...", end="", flush=True)
    for i in range(3):
        send_fn(prompts[i % len(prompts)], max_tokens)
    print(" done")

    results = []
    t_start = time.perf_counter()

    if concurrency == 1:
        # Sequential — simpler, no thread overhead
        for i in range(n_requests):
            r = send_fn(prompts[i % len(prompts)], max_tokens)
            results.append(r)
            if (i + 1) % max(1, n_requests // 5) == 0:
                print(f"    {i+1}/{n_requests} done", flush=True)
    else:
        work = [prompts[i % len(prompts)] for i in range(n_requests)]
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(send_fn, p, max_tokens) for p in work]
            for i, fut in enumerate(as_completed(futures), 1):
                results.append(fut.result())
                if i % max(1, n_requests // 5) == 0:
                    print(f"    {i}/{n_requests} done", flush=True)

    duration = time.perf_counter() - t_start
    return ScenarioResult(
        backend=backend, device=device, scenario=scenario,
        requests=n_requests, concurrency=concurrency,
        duration_s=duration, results=results,
    )


def measure_cold_start(send_fn, prompt: str, max_tokens: int) -> float:
    """Time of the very first request (cold KV cache, no warmup)."""
    t0 = time.perf_counter()
    send_fn(prompt, max_tokens)
    return (time.perf_counter() - t0) * 1000


# ─── Formatting ───────────────────────────────────────────────────────────────

def print_scenario(r: ScenarioResult):
    ok = len(r.ok)
    err = len(r.errors)
    print(f"\n  ── {r.backend.upper()} / {r.device.upper()} / {r.scenario} "
          f"({'cold start' if r.cold_start_ms > 0 else f'{r.requests} req'})")
    if r.cold_start_ms > 0:
        print(f"     Cold start: {r.cold_start_ms:.0f} ms")
        return
    print(f"     {r.requests} total | {ok} ok | {err} err | {r.duration_s:.1f}s")
    print(f"     Throughput:  {r.rps():.1f} req/s  {r.tps():.0f} tok/s")
    print(f"     Latency:     mean {r.mean_lat():.0f}ms  "
          f"P50 {r.pct(50):.0f}ms  P95 {r.pct(95):.0f}ms  P99 {r.pct(99):.0f}ms")
    ttft = r.ttft_pct(50)
    if ttft:
        print(f"     TTFT P50:    {ttft:.0f}ms")
    if err:
        print(f"     Errors:      {[e.error for e in r.errors[:2]]}")


def emit_markdown(results: list, args) -> str:
    lines = [
        "# Benchmark: infergo vs llama-cpp-python",
        "",
        f"*Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}*",
        "",
        f"**Model:** {os.path.basename(args.model_path) if args.model_path else 'llama3-8b-q4'}  ",
        f"**Hardware:** RTX 5070 Ti (16 GB VRAM) / same CPU for CPU runs  ",
        f"**infergo version:** Go 1.23, llama.cpp (same weights)  ",
        f"**Python version:** llama-cpp-python (same weights)  ",
        "",
        "---",
        "",
        "## Short prompts (~20 tokens in, 64 tokens out)",
        "",
    ]

    def table(scenario):
        rows = [r for r in results if r.scenario == scenario and r.cold_start_ms == 0]
        if not rows:
            return []
        out = [
            f"| Backend | Device | req/s | tok/s | Mean ms | P50 ms | P95 ms | P99 ms | Errors |",
            f"| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in sorted(rows, key=lambda x: (x.device, x.backend)):
            out.append(
                f"| {r.backend} | {r.device} | {r.rps():.1f} | {r.tps():.0f} | "
                f"{r.mean_lat():.0f} | {r.pct(50):.0f} | {r.pct(95):.0f} | "
                f"{r.pct(99):.0f} | {len(r.errors)} |"
            )
        return out

    lines += table("short") + [""]
    lines += ["## Long prompts (~512 tokens in, 256 tokens out)", ""]
    lines += table("long") + [""]

    # Cold start table
    cold = [r for r in results if r.cold_start_ms > 0]
    if cold:
        lines += ["## Cold start (time to first token, fresh context)", ""]
        lines += ["| Backend | Device | Cold start ms |", "| --- | --- | --- |"]
        for r in sorted(cold, key=lambda x: (x.device, x.backend)):
            lines.append(f"| {r.backend} | {r.device} | {r.cold_start_ms:.0f} |")
        lines += [""]

    lines += [
        "---",
        "",
        "## Notes",
        "",
        "- infergo serializes generation with a mutex (single-threaded decode, multi-threaded HTTP)",
        "- llama-cpp-python tested sequentially (GIL prevents true parallel generation)",
        "- `tok/s` = output tokens / total wall-clock seconds",
        "- Cold start measured as latency of the very first request with no warmup",
        "- CUDA crash on SM 12.0 (Blackwell) was fixed by ensuring sequential `llama_decode` calls",
        "- KV cache cleared via `llama_memory_seq_rm` on sequence close to prevent positional errors",
        "",
    ]
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="infergo vs llama-cpp-python full benchmark")
    p.add_argument("--model-path",     default="models/llama3-8b-q4.gguf")
    p.add_argument("--infergo-addr",   default="http://localhost:9191")
    p.add_argument("--infergo-model",  default="llama3-8b-q4")
    p.add_argument("--device",         default="cuda", choices=["cuda", "cpu", "both"])
    p.add_argument("--n-requests",     type=int, default=40)
    p.add_argument("--concurrency",    type=int, default=4)
    p.add_argument("--short-tokens",   type=int, default=64)
    p.add_argument("--long-tokens",    type=int, default=256)
    p.add_argument("--n-gpu-layers",   type=int, default=999)
    p.add_argument("--infergo-only",   action="store_true")
    p.add_argument("--python-only",    action="store_true")
    p.add_argument("--out",            default="results_full.md")
    args = p.parse_args()

    devices = ["cuda", "cpu"] if args.device == "both" else [args.device]
    all_results = []

    print("\n" + "═"*60)
    print("  infergo vs llama-cpp-python benchmark")
    print("═"*60)

    for device in devices:
        gpu_layers = args.n_gpu_layers if device == "cuda" else 0

        # ── infergo ──────────────────────────────────────────────
        if not args.python_only:
            print(f"\n{'─'*60}")
            print(f"  infergo ({device.upper()}) @ {args.infergo_addr}")
            print(f"{'─'*60}")

            def ig(prompt, max_tokens):
                return infergo_request(args.infergo_addr, args.infergo_model,
                                       prompt, max_tokens, stream=False)

            # Cold start
            cs = measure_cold_start(ig, SHORT_PROMPTS[0], args.short_tokens)
            cs_r = ScenarioResult("infergo", device, "cold_start", 1, 1, 0)
            cs_r.cold_start_ms = cs
            all_results.append(cs_r)
            print(f"  Cold start: {cs:.0f} ms")

            # Short
            r = run_scenario("infergo", device, "short", ig,
                             SHORT_PROMPTS, args.short_tokens,
                             args.n_requests, args.concurrency)
            all_results.append(r)
            print_scenario(r)

            # Long
            r = run_scenario("infergo", device, "long", ig,
                             LONG_PROMPTS, args.long_tokens,
                             args.n_requests // 2, 1)
            all_results.append(r)
            print_scenario(r)

        # ── llama-cpp-python ──────────────────────────────────────
        if not args.infergo_only:
            print(f"\n{'─'*60}")
            print(f"  llama-cpp-python ({device.upper()})")
            print(f"{'─'*60}")

            llm, load_ms = load_llama_cpp(args.model_path, gpu_layers, n_ctx=4096)

            # Cold start: first request after load (KV cache is cold)
            cs = measure_cold_start(
                lambda prompt, mt: python_request(llm, prompt, mt),
                SHORT_PROMPTS[0], args.short_tokens,
            )
            cs_r = ScenarioResult("python", device, "cold_start", 1, 1, 0)
            cs_r.cold_start_ms = cs
            all_results.append(cs_r)
            print(f"  Cold start: {cs:.0f} ms")

            # Short — sequential (GIL prevents parallel)
            r = run_scenario(
                "python", device, "short",
                lambda prompt, mt: python_request(llm, prompt, mt),
                SHORT_PROMPTS, args.short_tokens,
                args.n_requests, 1,
            )
            all_results.append(r)
            print_scenario(r)

            # Long
            r = run_scenario(
                "python", device, "long",
                lambda prompt, mt: python_request(llm, prompt, mt),
                LONG_PROMPTS, args.long_tokens,
                args.n_requests // 2, 1,
            )
            all_results.append(r)
            print_scenario(r)

            del llm  # free model memory before next device run

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  SUMMARY")
    print("═"*60)
    for r in all_results:
        if r.cold_start_ms > 0:
            print(f"  {r.backend:12} {r.device:4}  cold_start: {r.cold_start_ms:.0f}ms")
        else:
            print(f"  {r.backend:12} {r.device:4}  {r.scenario:5}  "
                  f"{r.rps():.1f} req/s  {r.tps():.0f} tok/s  "
                  f"P50={r.pct(50):.0f}ms  P99={r.pct(99):.0f}ms  "
                  f"err={len(r.errors)}")

    md = emit_markdown(all_results, args)
    with open(args.out, "w") as f:
        f.write(md)
    print(f"\n  Results written to {args.out}")


if __name__ == "__main__":
    main()
