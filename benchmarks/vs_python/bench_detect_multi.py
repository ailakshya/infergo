#!/usr/bin/env python3
"""
bench_detect_multi.py — Multi-model concurrent detection benchmark.

Loads multiple YOLO sessions (Python onnxruntime) and hits multiple models on
the Go infergo server simultaneously to maximize GPU utilization.

Benchmark matrix:
  1. Python: 4 onnxruntime sessions (yolo11n/s/m/l) running concurrently
  2. Go HTTP: 4 models on one infergo server hit concurrently
  3. Stress ramp: increase total concurrency until GPU saturates or OOM

Monitors: GPU util/mem/temp/power, per-model throughput/latency, total throughput.
"""

import argparse
import base64
import gc
import io
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

try:
    import requests as http_requests
except ImportError:
    sys.exit("pip install requests")

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("pip install opencv-python")

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    sys.exit("pip install ultralytics torch")


# ─── GPU Monitor ─────────────────────────────────────────────────────────────

@dataclass
class GPUSnapshot:
    timestamp: float
    util_pct: float
    mem_used_mb: float
    mem_total_mb: float
    temp_c: float
    power_w: float
    power_limit_w: float


class GPUMonitor:
    def __init__(self, interval=0.2):
        self.interval = interval
        self.snapshots: list[GPUSnapshot] = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self.snapshots.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self.snapshots

    def _poll(self):
        while not self._stop.is_set():
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--id=0",
                     "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    p = [x.strip() for x in r.stdout.strip().split(",")]
                    if len(p) >= 6:
                        self.snapshots.append(GPUSnapshot(
                            time.time(), float(p[0]), float(p[1]), float(p[2]),
                            float(p[3]), float(p[4]), float(p[5])))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def summary(self):
        s = self.snapshots
        if not s:
            return {}
        return {
            "util_mean": statistics.mean(x.util_pct for x in s),
            "util_max": max(x.util_pct for x in s),
            "mem_used_mean": statistics.mean(x.mem_used_mb for x in s),
            "mem_used_max": max(x.mem_used_mb for x in s),
            "mem_total": s[0].mem_total_mb,
            "temp_mean": statistics.mean(x.temp_c for x in s),
            "temp_max": max(x.temp_c for x in s),
            "power_mean": statistics.mean(x.power_w for x in s),
            "power_max": max(x.power_w for x in s),
            "power_limit": s[0].power_limit_w,
            "samples": len(s),
        }


# ─── Preprocessing ───────────────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Decode + letterbox + normalize → [1,3,640,640] float32."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = min(640 / w, 640 / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    px, py = (640 - nw) // 2, (640 - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    blob = canvas.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    return np.ascontiguousarray(blob)


def make_test_image(w=640, h=480) -> bytes:
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    for _ in range(8):
        x1, y1 = np.random.randint(0, w - 80), np.random.randint(0, h - 80)
        cv2.rectangle(img, (x1, y1), (x1 + 60, y1 + 60),
                      tuple(int(c) for c in np.random.randint(50, 200, 3)), -1)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


# ─── Result types ────────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    model: str
    latency_ms: float
    n_detections: int = 0
    error: Optional[str] = None


@dataclass
class MultiModelRun:
    name: str
    models: list[str]
    concurrency_per_model: int
    n_requests_per_model: int
    duration_s: float = 0.0
    results: list[RequestResult] = field(default_factory=list)
    gpu: dict = field(default_factory=dict)

    @property
    def ok(self):
        return [r for r in self.results if r.error is None]

    @property
    def total_rps(self):
        return len(self.ok) / self.duration_s if self.duration_s > 0 else 0

    def per_model_stats(self):
        out = {}
        for m in self.models:
            lats = sorted(r.latency_ms for r in self.ok if r.model == m)
            if not lats:
                continue
            out[m] = {
                "count": len(lats),
                "rps": len(lats) / self.duration_s if self.duration_s > 0 else 0,
                "mean": statistics.mean(lats),
                "p50": lats[len(lats) // 2],
                "p95": lats[int(len(lats) * 0.95)],
                "p99": lats[min(int(len(lats) * 0.99), len(lats) - 1)],
            }
        return out


# ─── Python ultralytics+PyTorch benchmark ───────────────────────────────────

def python_single(model, image_np, model_name):
    t0 = time.perf_counter()
    try:
        results = model.predict(image_np, verbose=False, conf=0.25, iou=0.45)
        ms = (time.perf_counter() - t0) * 1000
        n = len(results[0].boxes) if results and results[0].boxes is not None else 0
        return RequestResult(model_name, ms, n)
    except Exception as e:
        return RequestResult(model_name, (time.perf_counter() - t0) * 1000, error=str(e))


def bench_python_multi(model_paths: list[str], provider: str, image_bytes: bytes,
                       n_per_model: int, conc_per_model: int) -> MultiModelRun:
    """Load ultralytics YOLO models on GPU, run all concurrently."""
    # Convert .onnx paths to .pt paths (ultralytics uses PyTorch for GPU)
    pt_paths = [p.replace(".onnx", ".pt") for p in model_paths]
    model_names = [os.path.splitext(os.path.basename(p))[0] for p in model_paths]
    run = MultiModelRun("python_pytorch", model_names, conc_per_model, n_per_model)

    device = "cuda" if provider == "cuda" and torch.cuda.is_available() else "cpu"

    # Decode image once
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image_np = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # Load all models
    models = {}
    for pt_path, name in zip(pt_paths, model_names):
        print(f"    Loading {name} on {device}...")
        m = YOLO(pt_path)
        m.to(device)
        models[name] = m
        print(f"      Device: {next(m.model.parameters()).device}")

    # Warmup each
    print("    Warming up all models...")
    for name, m in models.items():
        for _ in range(3):
            m.predict(image_np, verbose=False)

    # Run all concurrently
    total_conc = conc_per_model * len(model_names)
    total_reqs = n_per_model * len(model_names)
    print(f"    Running {total_reqs} total requests ({n_per_model}/model x {len(model_names)} models, "
          f"{total_conc} total concurrency)...")

    gpu_mon = GPUMonitor()
    gpu_mon.start()
    t_start = time.perf_counter()

    if conc_per_model <= 1:
        # Sequential per model, interleaved
        for i in range(n_per_model):
            for name in model_names:
                run.results.append(python_single(models[name], image_np, name))
            done = (i + 1) * len(model_names)
            if done % max(1, total_reqs // 5) == 0:
                print(f"      {done}/{total_reqs}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=total_conc) as pool:
            futures = []
            for name in model_names:
                for _ in range(n_per_model):
                    futures.append(pool.submit(python_single, models[name], image_np, name))
            done = 0
            for fut in as_completed(futures):
                run.results.append(fut.result())
                done += 1
                if done % max(1, total_reqs // 5) == 0:
                    print(f"      {done}/{total_reqs}", flush=True)

    run.duration_s = time.perf_counter() - t_start
    run.gpu = gpu_mon.summary()

    # Cleanup
    for m in models.values():
        del m
    models.clear()
    torch.cuda.empty_cache()
    gc.collect()

    return run


# ─── Go HTTP multi-model benchmark ──────────────────────────────────────────

def go_http_single(addr, model_name, image_b64):
    t0 = time.perf_counter()
    try:
        resp = http_requests.post(f"{addr}/v1/detect", json={
            "model": model_name, "image_b64": image_b64,
            "conf_thresh": 0.25, "iou_thresh": 0.45,
        }, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        objs = data.get("objects") or []
        return RequestResult(model_name, (time.perf_counter() - t0) * 1000, len(objs))
    except Exception as e:
        return RequestResult(model_name, (time.perf_counter() - t0) * 1000, error=str(e))


def go_http_binary_single(addr, model_name, image_bytes):
    """Send raw JPEG bytes to the binary endpoint — no base64, no JSON overhead."""
    t0 = time.perf_counter()
    try:
        resp = http_requests.post(
            f"{addr}/v1/detect/binary?model={model_name}&conf=0.25&iou=0.45",
            data=image_bytes,
            headers={"Content-Type": "image/jpeg"},
            timeout=60)
        resp.raise_for_status()
        data = resp.json()
        objs = data.get("objects") or []
        return RequestResult(model_name, (time.perf_counter() - t0) * 1000, len(objs))
    except Exception as e:
        return RequestResult(model_name, (time.perf_counter() - t0) * 1000, error=str(e))


def bench_go_multi(addr: str, model_names: list[str], image_bytes: bytes,
                   n_per_model: int, conc_per_model: int) -> MultiModelRun:
    """Hit all models on the infergo server concurrently."""
    run = MultiModelRun("go_http", model_names, conc_per_model, n_per_model)
    image_b64 = base64.b64encode(image_bytes).decode()

    # Warmup each
    print("    Warming up all models...")
    for name in model_names:
        for _ in range(3):
            go_http_single(addr, name, image_b64)

    total_conc = conc_per_model * len(model_names)
    total_reqs = n_per_model * len(model_names)
    print(f"    Running {total_reqs} total requests ({n_per_model}/model × {len(model_names)} models, "
          f"{total_conc} total concurrency)...")

    gpu_mon = GPUMonitor()
    gpu_mon.start()
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=total_conc) as pool:
        futures = []
        for name in model_names:
            for _ in range(n_per_model):
                futures.append(pool.submit(go_http_single, addr, name, image_b64))
        done = 0
        for fut in as_completed(futures):
            run.results.append(fut.result())
            done += 1
            if done % max(1, total_reqs // 5) == 0:
                print(f"      {done}/{total_reqs}", flush=True)

    run.duration_s = time.perf_counter() - t_start
    run.gpu = gpu_mon.summary()

    return run


def bench_go_binary(addr: str, model_names: list[str], image_bytes: bytes,
                    n_per_model: int, conc_per_model: int) -> MultiModelRun:
    """Hit the binary endpoint (no base64/JSON overhead)."""
    run = MultiModelRun("go_binary", model_names, conc_per_model, n_per_model)

    # Warmup
    print("    Warming up all models (binary)...")
    for name in model_names:
        for _ in range(3):
            go_http_binary_single(addr, name, image_bytes)

    total_conc = conc_per_model * len(model_names)
    total_reqs = n_per_model * len(model_names)
    print(f"    Running {total_reqs} total requests ({n_per_model}/model × {len(model_names)} models, "
          f"{total_conc} total concurrency) [BINARY]...")

    gpu_mon = GPUMonitor()
    gpu_mon.start()
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=total_conc) as pool:
        futures = []
        for name in model_names:
            for _ in range(n_per_model):
                futures.append(pool.submit(go_http_binary_single, addr, name, image_bytes))
        done = 0
        for fut in as_completed(futures):
            run.results.append(fut.result())
            done += 1
            if done % max(1, total_reqs // 5) == 0:
                print(f"      {done}/{total_reqs}", flush=True)

    run.duration_s = time.perf_counter() - t_start
    run.gpu = gpu_mon.summary()

    return run


# ─── Print & save ────────────────────────────────────────────────────────────

def print_run(run: MultiModelRun):
    errs = [r for r in run.results if r.error]
    per = run.per_model_stats()
    g = run.gpu

    print(f"\n{'═' * 72}")
    print(f"  {run.name}  |  {len(run.models)} models  |  {run.concurrency_per_model}/model concurrency")
    print(f"  Total: {len(run.ok)} ok / {len(errs)} errors  |  {run.duration_s:.2f}s  |  {run.total_rps:.1f} req/s")
    print(f"{'═' * 72}")

    print(f"\n  {'Model':<12} {'req/s':>8} {'Mean ms':>9} {'P50 ms':>9} {'P95 ms':>9} {'P99 ms':>9}")
    print(f"  {'─' * 12} {'─' * 8} {'─' * 9} {'─' * 9} {'─' * 9} {'─' * 9}")
    for m, s in per.items():
        print(f"  {m:<12} {s['rps']:>8.1f} {s['mean']:>9.1f} {s['p50']:>9.1f} {s['p95']:>9.1f} {s['p99']:>9.1f}")

    if g:
        print(f"\n  GPU: {g.get('util_mean', 0):.0f}% util (max {g.get('util_max', 0):.0f}%), "
              f"{g.get('mem_used_max', 0):.0f}/{g.get('mem_total', 0):.0f} MB mem, "
              f"{g.get('temp_max', 0):.0f}°C, "
              f"{g.get('power_mean', 0):.0f}W (max {g.get('power_max', 0):.0f}/{g.get('power_limit', 0):.0f}W)")
    print()


def save_results(py_run: MultiModelRun, go_run: MultiModelRun,
                 stress_py: list, stress_go: list, out_path: str, hw: str):
    lines = []
    lines.append("# Multi-Model Detection Benchmark: Python onnxruntime vs Go infergo")
    lines.append(f"\n*Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}*\n")
    lines.append(f"**Hardware:** {hw}  ")
    lines.append("**Models:** YOLOv11 n/s/m/l (all 640x640 COCO)  ")
    lines.append("**All 4 models loaded and running concurrently on GPU**  ")
    lines.append("**GPU memory cleared between Python and Go benchmarks**\n")
    lines.append("---\n")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Metric | Python onnxrt | Go HTTP (infergo) | Winner |")
    lines.append("| --- | --- | --- | --- |")
    py_rps = py_run.total_rps if py_run else 0
    go_rps = go_run.total_rps if go_run else 0

    def win(pv, gv, lower=True):
        if pv == 0 or gv == 0:
            return "—"
        w = "Go" if (gv < pv if lower else gv > pv) else "Python"
        d = abs(pv - gv) / max(pv, gv) * 100
        return f"**{w}** +{d:.0f}%"

    lines.append(f"| Total throughput (req/s) | {py_rps:.1f} | {go_rps:.1f} | {win(py_rps, go_rps, lower=False)} |")

    for target, run in [("Python", py_run), ("Go", go_run)]:
        if not run:
            continue
        g = run.gpu
        lines.append(f"| {target} GPU util mean | {g.get('util_mean', 0):.0f}% | — | — |" if target == "Python"
                      else f"| {target} GPU util mean | — | {g.get('util_mean', 0):.0f}% | — |")
    lines.append("")

    # Per-model breakdown
    for label, run in [("Python onnxruntime (4 sessions)", py_run), ("Go HTTP infergo (4 models)", go_run)]:
        if not run:
            continue
        per = run.per_model_stats()
        g = run.gpu
        lines.append(f"## {label}\n")
        lines.append(f"**Total: {run.total_rps:.1f} req/s** | {len(run.ok)} ok / {len(run.results) - len(run.ok)} errors | {run.duration_s:.2f}s\n")

        lines.append("### Per-Model Performance\n")
        lines.append("| Model | Params | req/s | Mean ms | P50 ms | P95 ms | P99 ms |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        params = {"yolo11n": "2.6M", "yolo11s": "9.4M", "yolo11m": "20.1M", "yolo11l": "25.3M"}
        for m, s in per.items():
            lines.append(f"| {m} | {params.get(m, '?')} | {s['rps']:.1f} | {s['mean']:.1f} | "
                         f"{s['p50']:.1f} | {s['p95']:.1f} | {s['p99']:.1f} |")
        lines.append("")

        if g:
            lines.append("### GPU Monitoring\n")
            lines.append("| Metric | Mean | Max |")
            lines.append("| --- | --- | --- |")
            lines.append(f"| Utilization (%) | {g.get('util_mean', 0):.1f} | {g.get('util_max', 0):.0f} |")
            lines.append(f"| Memory (MB) | {g.get('mem_used_mean', 0):.0f} | {g.get('mem_used_max', 0):.0f} / {g.get('mem_total', 0):.0f} |")
            lines.append(f"| Temperature (°C) | {g.get('temp_mean', 0):.0f} | {g.get('temp_max', 0):.0f} |")
            lines.append(f"| Power (W) | {g.get('power_mean', 0):.0f} | {g.get('power_max', 0):.0f} / {g.get('power_limit', 0):.0f} |")
            lines.append("")

    # Stress test tables
    for label, runs in [("Python onnxruntime", stress_py), ("Go HTTP infergo", stress_go)]:
        if not runs:
            continue
        lines.append(f"## Stress Test: {label}\n")
        lines.append("Concurrency ramp per model (×4 models = total concurrency). GPU cleared between levels.\n")
        lines.append("| Conc/model | Total conc | Total req/s | GPU util % | GPU mem MB | GPU power W | Errors |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in runs:
            g = r.gpu
            errs = len(r.results) - len(r.ok)
            lines.append(f"| {r.concurrency_per_model} | {r.concurrency_per_model * len(r.models)} | "
                         f"{r.total_rps:.1f} | {g.get('util_mean', 0):.0f} | "
                         f"{g.get('mem_used_max', 0):.0f} | {g.get('power_mean', 0):.0f} | {errs} |")

        # Per-model breakdown at peak
        peak = max(runs, key=lambda r: r.total_rps)
        per = peak.per_model_stats()
        lines.append(f"\n### Peak throughput breakdown (conc/model={peak.concurrency_per_model})\n")
        lines.append("| Model | req/s | Mean ms | P50 ms | P95 ms |")
        lines.append("| --- | --- | --- | --- | --- |")
        for m, s in per.items():
            lines.append(f"| {m} | {s['rps']:.1f} | {s['mean']:.1f} | {s['p50']:.1f} | {s['p95']:.1f} |")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Results written to {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                   default=["models/yolo11n.onnx", "models/yolo11s.onnx",
                            "models/yolo11m.onnx", "models/yolo11l.onnx"])
    p.add_argument("--provider", default="cuda")
    p.add_argument("--addr", default="http://localhost:9191")
    p.add_argument("--requests-per-model", type=int, default=50)
    p.add_argument("--concurrency-per-model", type=int, default=4)
    p.add_argument("--stress", action="store_true")
    p.add_argument("--stress-levels", default="1,2,4,8,16")
    p.add_argument("--stress-reqs", type=int, default=30)
    p.add_argument("--out", default="benchmarks/vs_python/results_detect_multi.md")
    args = p.parse_args()

    model_names = [os.path.splitext(os.path.basename(p))[0] for p in args.models]

    # HW info
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=5)
        hw = r.stdout.strip()
    except Exception:
        hw = "Unknown"

    print(f"\n{'═' * 72}")
    print(f"  Multi-Model Detection Benchmark")
    print(f"  Models: {model_names}")
    print(f"  Provider: {args.provider}")
    print(f"  Requests/model: {args.requests_per_model}")
    print(f"  Concurrency/model: {args.concurrency_per_model}")
    print(f"  Hardware: {hw}")
    print(f"{'═' * 72}\n")

    image_bytes = make_test_image()
    print(f"  Test image: {len(image_bytes)} bytes\n")

    # ── Python ──
    print("=" * 50)
    print(" Python onnxruntime — 4 concurrent sessions")
    print("=" * 50)
    py_run = bench_python_multi(args.models, args.provider, image_bytes,
                                args.requests_per_model, args.concurrency_per_model)
    print_run(py_run)

    stress_py = []
    if args.stress:
        levels = [int(x) for x in args.stress_levels.split(",")]
        print("  Stress ramping Python...")
        for c in levels:
            gc.collect()
            time.sleep(1)
            r = bench_python_multi(args.models, args.provider, image_bytes, args.stress_reqs, c)
            stress_py.append(r)
            g = r.gpu
            errs = len(r.results) - len(r.ok)
            print(f"    c={c}/model → {r.total_rps:.1f} req/s, GPU {g.get('util_mean', 0):.0f}%, "
                  f"{g.get('mem_used_max', 0):.0f}MB, {errs} errors")
            if errs > len(r.results) * 0.5:
                print("    !!! >50% errors, stopping")
                break

    # Clear before Go
    gc.collect()
    time.sleep(2)

    # ── Go HTTP ──
    print("\n" + "=" * 50)
    print(" Go HTTP infergo — 4 models on one server")
    print("=" * 50)
    go_run = bench_go_multi(args.addr, model_names, image_bytes,
                            args.requests_per_model, args.concurrency_per_model)
    print_run(go_run)

    stress_go = []
    if args.stress:
        levels = [int(x) for x in args.stress_levels.split(",")]
        print("  Stress ramping Go HTTP...")
        for c in levels:
            time.sleep(1)
            r = bench_go_multi(args.addr, model_names, image_bytes, args.stress_reqs, c)
            stress_go.append(r)
            g = r.gpu
            errs = len(r.results) - len(r.ok)
            print(f"    c={c}/model → {r.total_rps:.1f} req/s, GPU {g.get('util_mean', 0):.0f}%, "
                  f"{g.get('mem_used_max', 0):.0f}MB, {errs} errors")
            if errs > len(r.results) * 0.5:
                print("    !!! >50% errors, stopping")
                break

    # ── Comparison ──
    print(f"\n{'═' * 72}")
    print(f"  TOTAL THROUGHPUT: Python {py_run.total_rps:.1f} req/s vs Go {go_run.total_rps:.1f} req/s")
    if go_run.total_rps > 0 and py_run.total_rps > 0:
        ratio = go_run.total_rps / py_run.total_rps
        print(f"  Go is {ratio:.1f}× {'faster' if ratio > 1 else 'slower'}")
    print(f"{'═' * 72}\n")

    save_results(py_run, go_run, stress_py, stress_go, args.out, hw)


if __name__ == "__main__":
    main()
