"""
bench_onnx.py — infergo vs FastAPI+ONNX embedding/detection throughput benchmark.

Usage:
  # Benchmark infergo embedding endpoint
  python bench_onnx.py --task embed --target infergo --addr http://localhost:9090 --model resnet50

  # Benchmark FastAPI+ONNX (standard onnxruntime server)
  python bench_onnx.py --task embed --target fastapi --addr http://localhost:8001 --model resnet50

  # Compare both
  python bench_onnx.py --task embed --compare \
    --infergo-addr http://localhost:9090 --infergo-model resnet50 \
    --fastapi-addr http://localhost:8001 --fastapi-model resnet50

  # Detection task (YOLO-style)
  python bench_onnx.py --task detect --compare \
    --infergo-addr http://localhost:9090 --infergo-model yolov8n \
    --fastapi-addr http://localhost:8001 --fastapi-model yolov8n

Metrics reported:
  - Throughput: requests/sec
  - Latency: mean, P50, P95, P99 (ms)
  - RAM delta: resident memory before/after (requires psutil)
  - Cold start: time from first request to steady-state (first 10 requests)
"""

import argparse
import base64
import io
import json
import math
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ─── Synthetic payload generators ─────────────────────────────────────────────

def make_embed_payload(model: str) -> dict:
    """224×224 RGB image encoded as base64 JPEG."""
    if HAS_NUMPY:
        import struct, zlib
        # Minimal valid PNG: 1×1 white pixel, 224x224 would be large — use small JPEG via raw bytes
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        # Encode as JPEG via cv2 if available, else raw PNG
        try:
            import cv2
            _, buf = cv2.imencode(".jpg", img)
            b64 = base64.b64encode(buf.tobytes()).decode()
        except ImportError:
            # fallback: send random base64 as placeholder
            b64 = base64.b64encode(img.tobytes()[:4096]).decode()
    else:
        # Minimal 1-byte JPEG placeholder (server should return error but measures overhead)
        b64 = base64.b64encode(b"\xff\xd8\xff\xd9").decode()

    return {"model": model, "input": b64}


def make_detect_payload(model: str) -> dict:
    """Same as embed — 224×224 image for detection."""
    return make_embed_payload(model)


# ─── Data types ───────────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    latency_ms: float
    error: Optional[str] = None

@dataclass
class BenchResult:
    target: str
    task: str
    requests: int
    concurrency: int
    duration_s: float
    results: list = field(default_factory=list)
    ram_before_mb: float = 0.0
    ram_after_mb: float = 0.0
    cold_start_ms: float = 0.0  # latency of first request

    @property
    def ok(self):
        return [r for r in self.results if r.error is None]

    @property
    def errors(self):
        return [r for r in self.results if r.error is not None]

    def throughput_rps(self):
        return len(self.ok) / self.duration_s if self.duration_s > 0 else 0

    def latency_percentile(self, p):
        lats = sorted(r.latency_ms for r in self.ok)
        if not lats:
            return 0
        idx = int(math.ceil(len(lats) * p / 100)) - 1
        return lats[max(0, min(idx, len(lats) - 1))]

    def mean_latency(self):
        lats = [r.latency_ms for r in self.ok]
        return statistics.mean(lats) if lats else 0


# ─── Single request ───────────────────────────────────────────────────────────

def send_embed(addr: str, payload: dict) -> RequestResult:
    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{addr}/v1/embeddings", json=payload, timeout=30)
        resp.raise_for_status()
        return RequestResult(latency_ms=(time.perf_counter() - t0) * 1000)
    except Exception as e:
        return RequestResult(latency_ms=(time.perf_counter() - t0) * 1000, error=str(e))


def send_detect(addr: str, payload: dict) -> RequestResult:
    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{addr}/v1/detect", json=payload, timeout=30)
        resp.raise_for_status()
        return RequestResult(latency_ms=(time.perf_counter() - t0) * 1000)
    except Exception as e:
        return RequestResult(latency_ms=(time.perf_counter() - t0) * 1000, error=str(e))


# ─── RAM snapshot ─────────────────────────────────────────────────────────────

def server_ram_mb(addr: str) -> float:
    """Try to read RSS from /metrics (Prometheus) or return 0."""
    try:
        resp = requests.get(f"{addr}/metrics", timeout=5)
        for line in resp.text.splitlines():
            if line.startswith("process_resident_memory_bytes") and not line.startswith("#"):
                return float(line.split()[-1]) / (1024 * 1024)
    except Exception:
        pass
    return 0.0


# ─── Cold start measurement ───────────────────────────────────────────────────

def measure_cold_start(addr: str, payload: dict, task: str) -> float:
    """Send 10 sequential requests and return mean of first 3 (cold) vs last 3 (warm)."""
    latencies = []
    fn = send_embed if task == "embed" else send_detect
    for _ in range(6):
        r = fn(addr, payload)
        latencies.append(r.latency_ms)
    cold = statistics.mean(latencies[:3])
    warm = statistics.mean(latencies[-3:])
    print(f"    Cold-start: first 3 avg {cold:.0f}ms → warm {warm:.0f}ms (delta {cold-warm:.0f}ms)")
    return cold


# ─── Run benchmark ────────────────────────────────────────────────────────────

def run_bench(target: str, task: str, addr: str, model: str,
              n_requests: int, concurrency: int) -> BenchResult:
    payload = make_embed_payload(model) if task == "embed" else make_detect_payload(model)
    fn = send_embed if task == "embed" else send_detect

    ram_before = server_ram_mb(addr)

    print(f"  Measuring cold start...")
    cold_ms = measure_cold_start(addr, payload, task)

    print(f"  Sending {n_requests} requests (concurrency={concurrency})...")
    results = []
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(fn, addr, payload) for _ in range(n_requests)]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % max(1, n_requests // 5) == 0:
                print(f"    {i}/{n_requests} done", flush=True)
    duration = time.perf_counter() - t_start

    ram_after = server_ram_mb(addr)

    return BenchResult(
        target=target, task=task, requests=n_requests,
        concurrency=concurrency, duration_s=duration,
        results=results, ram_before_mb=ram_before,
        ram_after_mb=ram_after, cold_start_ms=cold_ms,
    )


# ─── Print results ────────────────────────────────────────────────────────────

def print_result(br: BenchResult):
    ok = len(br.ok)
    err = len(br.errors)
    print(f"\n{'─'*60}")
    print(f"  Target:      {br.target}  task={br.task}")
    print(f"  Requests:    {br.requests} total | {ok} ok | {err} errors")
    print(f"  Duration:    {br.duration_s:.2f}s")
    print(f"  Throughput:  {br.throughput_rps():.1f} req/s")
    print(f"  Latency:")
    print(f"    mean:      {br.mean_latency():.1f} ms")
    print(f"    P50:       {br.latency_percentile(50):.1f} ms")
    print(f"    P95:       {br.latency_percentile(95):.1f} ms")
    print(f"    P99:       {br.latency_percentile(99):.1f} ms")
    print(f"  Cold start:  {br.cold_start_ms:.0f} ms (first request)")
    if br.ram_after_mb > 0:
        delta = br.ram_after_mb - br.ram_before_mb
        print(f"  RAM:         {br.ram_after_mb:.0f} MB RSS (delta {delta:+.0f} MB)")
    if err > 0:
        print(f"  Errors (sample): {[e.error for e in br.errors[:2]]}")
    print(f"{'─'*60}")


def print_comparison(a: BenchResult, b: BenchResult):
    def pct(va, vb):
        if vb == 0:
            return "—"
        r = va / vb
        sign = "+" if r >= 1 else ""
        return f"{sign}{(r-1)*100:.0f}%"

    print(f"\n{'═'*60}")
    print(f"  COMPARISON: {a.target} vs {b.target}  (task={a.task})")
    print(f"{'═'*60}")
    header = f"  {'Metric':<25} {a.target:>12} {b.target:>12}  {'vs B':>8}"
    print(header)
    print(f"  {'─'*25} {'─'*12} {'─'*12}  {'─'*8}")

    def row(name, va, vb, fmt="{:.1f}"):
        p = pct(va, vb) if a.target == "infergo" else pct(vb, va)
        print(f"  {name:<25} {fmt.format(va):>12} {fmt.format(vb):>12}  {p:>8}")

    row("Throughput (req/s)", a.throughput_rps(), b.throughput_rps())
    row("Latency mean (ms)", a.mean_latency(), b.mean_latency())
    row("Latency P50 (ms)", a.latency_percentile(50), b.latency_percentile(50))
    row("Latency P99 (ms)", a.latency_percentile(99), b.latency_percentile(99))
    row("Cold start (ms)", a.cold_start_ms, b.cold_start_ms, fmt="{:.0f}")
    if a.ram_after_mb > 0 and b.ram_after_mb > 0:
        row("RAM (MB)", a.ram_after_mb, b.ram_after_mb, fmt="{:.0f}")
    print(f"{'═'*60}\n")


def emit_markdown(results: list, out_path: str):
    lines = [
        "# ONNX Benchmark Results\n",
        f"*Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}*\n",
        "",
    ]
    headers = ["Target", "Task", "req/s", "Mean ms", "P50 ms", "P95 ms", "P99 ms", "Cold ms", "RAM MB", "Errors"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for br in results:
        row = [
            br.target, br.task,
            f"{br.throughput_rps():.1f}",
            f"{br.mean_latency():.1f}",
            f"{br.latency_percentile(50):.1f}",
            f"{br.latency_percentile(95):.1f}",
            f"{br.latency_percentile(99):.1f}",
            f"{br.cold_start_ms:.0f}",
            f"{br.ram_after_mb:.0f}" if br.ram_after_mb > 0 else "—",
            str(len(br.errors)),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Results written to {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="infergo vs FastAPI+ONNX benchmark")
    p.add_argument("--task",    default="embed", choices=["embed", "detect"])
    p.add_argument("--target",  default="infergo", choices=["infergo", "fastapi", "onnx-server"])
    p.add_argument("--addr",    default="http://localhost:9090")
    p.add_argument("--model",   default="resnet50")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--infergo-addr",   default="http://localhost:9090")
    p.add_argument("--infergo-model",  default="resnet50")
    p.add_argument("--fastapi-addr",   default="http://localhost:8001")
    p.add_argument("--fastapi-model",  default="resnet50")
    p.add_argument("--requests",    type=int, default=500)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--out",         default="results_onnx.md")
    args = p.parse_args()

    all_results = []

    if args.compare:
        print(f"\n[1/2] Benchmarking infergo @ {args.infergo_addr}  task={args.task}")
        r1 = run_bench("infergo", args.task, args.infergo_addr, args.infergo_model,
                       args.requests, args.concurrency)
        print_result(r1)
        all_results.append(r1)

        print(f"\n[2/2] Benchmarking fastapi @ {args.fastapi_addr}  task={args.task}")
        r2 = run_bench("fastapi", args.task, args.fastapi_addr, args.fastapi_model,
                       args.requests, args.concurrency)
        print_result(r2)
        all_results.append(r2)

        print_comparison(r1, r2)
    else:
        print(f"\nBenchmarking {args.target} @ {args.addr}  task={args.task}")
        r = run_bench(args.target, args.task, args.addr, args.model,
                      args.requests, args.concurrency)
        print_result(r)
        all_results.append(r)

    emit_markdown(all_results, args.out)


if __name__ == "__main__":
    main()
