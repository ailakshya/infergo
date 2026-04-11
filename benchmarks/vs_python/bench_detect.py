#!/usr/bin/env python3
"""
bench_detect.py — Comprehensive detection benchmark: Python native bindings vs Native Go HTTP.

Monitors every stage of the pipeline with GPU stress testing:
  - Per-stage latency: decode, letterbox, normalize, stack, inference, NMS/postprocess
  - End-to-end latency: mean, P50, P95, P99
  - Throughput: req/s, images/s
  - GPU: utilization %, memory used/free/total, temperature, power draw
  - CPU: usage %, RSS memory
  - Cold start: first request latency
  - Stress test: ramp concurrency 1→2→4→8→16→32→64 until saturation/OOM

Clears GPU memory between each benchmark for clean measurements.

Usage:
  # Python native bindings only (in-process, no server)
  python bench_detect.py --target python --model models/yolov8n.onnx --provider cuda

  # Native Go HTTP only (requires running: infergo serve --model yolov8n.onnx --provider cuda)
  python bench_detect.py --target go --addr http://localhost:9090 --model yolov8n

  # Compare both (will start/stop Go server automatically if --infergo-bin given)
  python bench_detect.py --compare \
    --model-path models/yolov8n.onnx --provider cuda \
    --addr http://localhost:9090 --model-name yolov8n

  # Max GPU stress test
  python bench_detect.py --stress --model-path models/yolov8n.onnx --provider cuda

  # All-in-one: compare + stress + save results
  python bench_detect.py --compare --stress \
    --model-path models/yolov8n.onnx --provider cuda \
    --addr http://localhost:9090 --model-name yolov8n \
    --out benchmarks/vs_python/results_detect.md
"""

import argparse
import base64
import gc
import io
import json
import math
import os
import signal
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

try:
    import numpy as np
except ImportError:
    sys.exit("pip install numpy")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ─── GPU Monitor ─────────────────────────────────────────────────────────────

@dataclass
class GPUSnapshot:
    """Single nvidia-smi reading."""
    timestamp: float
    utilization_pct: float    # GPU core utilization %
    memory_used_mb: float
    memory_total_mb: float
    memory_free_mb: float
    temperature_c: float
    power_draw_w: float
    power_limit_w: float


class GPUMonitor:
    """Polls nvidia-smi in a background thread to collect GPU stats."""

    def __init__(self, interval: float = 0.25, device_id: int = 0):
        self.interval = interval
        self.device_id = device_id
        self.snapshots: list[GPUSnapshot] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self.snapshots.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> list[GPUSnapshot]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self.snapshots

    def _poll(self):
        while not self._stop.is_set():
            snap = self._read_smi()
            if snap:
                self.snapshots.append(snap)
            self._stop.wait(self.interval)

    def _read_smi(self) -> Optional[GPUSnapshot]:
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 f"--id={self.device_id}",
                 "--query-gpu=utilization.gpu,memory.used,memory.total,memory.free,temperature.gpu,power.draw,power.limit",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return None
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) < 7:
                return None
            return GPUSnapshot(
                timestamp=time.time(),
                utilization_pct=float(parts[0]),
                memory_used_mb=float(parts[1]),
                memory_total_mb=float(parts[2]),
                memory_free_mb=float(parts[3]),
                temperature_c=float(parts[4]),
                power_draw_w=float(parts[5]),
                power_limit_w=float(parts[6]),
            )
        except Exception:
            return None

    @staticmethod
    def summary(snapshots: list[GPUSnapshot]) -> dict:
        if not snapshots:
            return {}
        return {
            "gpu_util_mean": statistics.mean(s.utilization_pct for s in snapshots),
            "gpu_util_max": max(s.utilization_pct for s in snapshots),
            "gpu_mem_used_mean_mb": statistics.mean(s.memory_used_mb for s in snapshots),
            "gpu_mem_used_max_mb": max(s.memory_used_mb for s in snapshots),
            "gpu_mem_total_mb": snapshots[0].memory_total_mb,
            "gpu_temp_mean_c": statistics.mean(s.temperature_c for s in snapshots),
            "gpu_temp_max_c": max(s.temperature_c for s in snapshots),
            "gpu_power_mean_w": statistics.mean(s.power_draw_w for s in snapshots),
            "gpu_power_max_w": max(s.power_draw_w for s in snapshots),
            "gpu_power_limit_w": snapshots[0].power_limit_w,
            "n_samples": len(snapshots),
        }


# ─── CPU / Memory Monitor ───────────────────────────────────────────────────

def get_process_rss_mb() -> float:
    """Current process RSS in MB."""
    if HAS_PSUTIL:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    return 0.0


def get_cpu_percent() -> float:
    """System-wide CPU usage %."""
    if HAS_PSUTIL:
        return psutil.cpu_percent(interval=0.1)
    return 0.0


# ─── GPU Memory Clearing ────────────────────────────────────────────────────

def clear_gpu_memory():
    """Best-effort GPU memory clearing between benchmarks."""
    gc.collect()
    # Force Python garbage collection for any leftover C objects
    for _ in range(3):
        gc.collect()
    # Try CUDA cache clear if torch is available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    # Small sleep to let GPU reclaim memory
    time.sleep(1)


def nvidia_smi_reset_gpu(device_id: int = 0):
    """Aggressive GPU memory reset via nvidia-smi (requires root or persistence mode off)."""
    try:
        subprocess.run(
            ["nvidia-smi", f"--id={device_id}", "--gpu-reset"],
            capture_output=True, timeout=10
        )
    except Exception:
        pass


# ─── Synthetic Image Generation ──────────────────────────────────────────────

def make_test_image_bytes(width: int = 640, height: int = 480) -> bytes:
    """Generate a synthetic JPEG image for benchmarking."""
    try:
        import cv2
        img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        # Add some structure so detections are realistic
        for i in range(5):
            x1, y1 = np.random.randint(0, width - 100), np.random.randint(0, height - 100)
            x2, y2 = x1 + np.random.randint(50, 100), y1 + np.random.randint(50, 100)
            color = tuple(int(c) for c in np.random.randint(0, 255, 3))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes()
    except ImportError:
        # Fallback: create a minimal valid JPEG using PIL
        try:
            from PIL import Image
            img = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return buf.getvalue()
        except ImportError:
            sys.exit("pip install opencv-python or Pillow")


# ─── Data Types ──────────────────────────────────────────────────────────────

@dataclass
class StageTimings:
    """Per-stage latency for one detection request (ms)."""
    decode_ms: float = 0.0
    letterbox_ms: float = 0.0
    normalize_ms: float = 0.0
    stack_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0
    n_detections: int = 0
    error: Optional[str] = None


@dataclass
class BenchRun:
    """Results for one benchmark run (one target, one concurrency level)."""
    target: str                # "python_native" or "go_http"
    provider: str              # "cuda" or "cpu"
    concurrency: int
    n_requests: int
    duration_s: float = 0.0
    timings: list[StageTimings] = field(default_factory=list)
    gpu_stats: dict = field(default_factory=dict)
    cpu_pct_mean: float = 0.0
    rss_before_mb: float = 0.0
    rss_after_mb: float = 0.0
    cold_start_ms: float = 0.0

    @property
    def ok_timings(self) -> list[StageTimings]:
        return [t for t in self.timings if t.error is None]

    @property
    def errors(self) -> list[StageTimings]:
        return [t for t in self.timings if t.error is not None]

    def throughput_rps(self) -> float:
        return len(self.ok_timings) / self.duration_s if self.duration_s > 0 else 0

    def _percentile(self, values: list[float], p: int) -> float:
        if not values:
            return 0
        s = sorted(values)
        idx = int(math.ceil(len(s) * p / 100)) - 1
        return s[max(0, min(idx, len(s) - 1))]

    def latency_stats(self) -> dict:
        totals = [t.total_ms for t in self.ok_timings]
        if not totals:
            return {}
        return {
            "mean": statistics.mean(totals),
            "p50": self._percentile(totals, 50),
            "p95": self._percentile(totals, 95),
            "p99": self._percentile(totals, 99),
            "min": min(totals),
            "max": max(totals),
            "stdev": statistics.stdev(totals) if len(totals) > 1 else 0,
        }

    def stage_stats(self) -> dict:
        """Per-stage average latencies."""
        ok = self.ok_timings
        if not ok:
            return {}
        stages = {}
        for name in ["decode_ms", "letterbox_ms", "normalize_ms", "stack_ms", "inference_ms", "postprocess_ms"]:
            vals = [getattr(t, name) for t in ok]
            stages[name.replace("_ms", "")] = {
                "mean": statistics.mean(vals),
                "p50": self._percentile(vals, 50),
                "p95": self._percentile(vals, 95),
            }
        return stages


# ─── Python Native Bindings Benchmark ────────────────────────────────────────

def _numpy_preprocess(image_bytes: bytes):
    """Decode + letterbox + normalize using numpy/PIL (bypasses C decode_image).
    Returns (input_array, decode_ms, letterbox_ms, normalize_ms, stack_ms).
    """
    import cv2

    # Stage 1: Decode
    t = time.perf_counter()
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR [H,W,3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    decode_ms = (time.perf_counter() - t) * 1000

    # Stage 2: Letterbox to 640x640
    t = time.perf_counter()
    h, w = img.shape[:2]
    scale = min(640 / w, 640 / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (640 - new_w) // 2, (640 - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    letterbox_ms = (time.perf_counter() - t) * 1000

    # Stage 3: Normalize HWC→CHW, scale to [0,1]
    t = time.perf_counter()
    blob = canvas.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)  # [3, 640, 640]
    normalize_ms = (time.perf_counter() - t) * 1000

    # Stage 4: Stack to batch [1,3,640,640]
    t = time.perf_counter()
    batch = blob[np.newaxis, ...]  # [1, 3, 640, 640]
    batch = np.ascontiguousarray(batch)
    stack_ms = (time.perf_counter() - t) * 1000

    return batch, decode_ms, letterbox_ms, normalize_ms, stack_ms


def _numpy_nms(output: np.ndarray, conf_thresh: float = 0.25, iou_thresh: float = 0.45):
    """Pure-numpy NMS on YOLOv8/v11 output [1, 84, 8400]."""
    # Transpose to [8400, 84]
    preds = output[0].T  # [8400, 84]
    # Columns: cx, cy, w, h, class_scores[80]
    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    scores = preds[:, 4:]  # [8400, 80]

    class_ids = scores.argmax(axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]

    mask = confidences > conf_thresh
    if not mask.any():
        return 0

    cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # NMS per class
    order = confidences.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area_r = (x2[rest] - x1[rest]) * (y2[rest] - y1[rest])
        iou = inter / (area_i + area_r - inter + 1e-6)
        order = rest[iou <= iou_thresh]

    return len(keep)


def run_python_native_single(session, input_name: str, image_bytes: bytes) -> StageTimings:
    """Run one detection through Python onnxruntime, timing each stage."""
    timings = StageTimings()

    try:
        t0 = time.perf_counter()

        # Stages 1-4: Preprocess (numpy path)
        batch, timings.decode_ms, timings.letterbox_ms, timings.normalize_ms, timings.stack_ms = \
            _numpy_preprocess(image_bytes)

        # Stage 5: ONNX inference via onnxruntime
        t_stage = time.perf_counter()
        outputs = session.run(None, {input_name: batch})
        timings.inference_ms = (time.perf_counter() - t_stage) * 1000

        # Stage 6: NMS postprocessing (numpy path)
        t_stage = time.perf_counter()
        n_dets = _numpy_nms(outputs[0], conf_thresh=0.25, iou_thresh=0.45)
        timings.postprocess_ms = (time.perf_counter() - t_stage) * 1000

        timings.total_ms = (time.perf_counter() - t0) * 1000
        timings.n_detections = n_dets

    except Exception as e:
        timings.error = str(e)
        timings.total_ms = (time.perf_counter() - t0) * 1000

    return timings


def bench_python_native(model_path: str, provider: str, image_bytes: bytes,
                        n_requests: int, concurrency: int) -> BenchRun:
    """Benchmark the Python onnxruntime detection pipeline."""
    import onnxruntime as ort

    run = BenchRun(
        target="python_onnxrt", provider=provider,
        concurrency=concurrency, n_requests=n_requests,
    )

    # Create and load session with matching provider
    providers = []
    if provider == "cuda":
        providers.append(("CUDAExecutionProvider", {"device_id": 0}))
    elif provider == "tensorrt":
        providers.append(("TensorrtExecutionProvider", {"device_id": 0}))
    providers.append("CPUExecutionProvider")

    print(f"    Loading ONNX model: {model_path} (providers={[p if isinstance(p, str) else p[0] for p in providers]})")
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    active_provider = session.get_providers()[0]
    print(f"    Active provider: {active_provider}")
    print(f"    Input: {input_name} {session.get_inputs()[0].shape}")

    # Cold start: first inference
    print("    Measuring cold start...")
    cold = run_python_native_single(session, input_name, image_bytes)
    run.cold_start_ms = cold.total_ms
    print(f"    Cold start: {cold.total_ms:.1f}ms ({cold.n_detections} detections)")

    # Warmup
    for _ in range(3):
        run_python_native_single(session, input_name, image_bytes)

    # Main benchmark
    run.rss_before_mb = get_process_rss_mb()
    gpu_mon = GPUMonitor(interval=0.2)
    gpu_mon.start()
    cpu_samples = []

    print(f"    Running {n_requests} requests (concurrency={concurrency})...")
    t_start = time.perf_counter()

    if concurrency <= 1:
        for i in range(n_requests):
            run.timings.append(run_python_native_single(session, input_name, image_bytes))
            if HAS_PSUTIL:
                cpu_samples.append(psutil.cpu_percent(interval=None))
            if (i + 1) % max(1, n_requests // 5) == 0:
                print(f"      {i + 1}/{n_requests} done", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(run_python_native_single, session, input_name, image_bytes)
                       for _ in range(n_requests)]
            for i, fut in enumerate(as_completed(futures), 1):
                run.timings.append(fut.result())
                if i % max(1, n_requests // 5) == 0:
                    print(f"      {i}/{n_requests} done", flush=True)

    run.duration_s = time.perf_counter() - t_start
    gpu_snapshots = gpu_mon.stop()
    run.gpu_stats = GPUMonitor.summary(gpu_snapshots)
    run.rss_after_mb = get_process_rss_mb()
    if cpu_samples:
        run.cpu_pct_mean = statistics.mean(cpu_samples)

    # Cleanup
    del session
    clear_gpu_memory()

    return run


# ─── Native Go HTTP Benchmark ───────────────────────────────────────────────

def run_go_http_single(addr: str, model_name: str, image_b64: str) -> StageTimings:
    """Send one detection request to the Go infergo server, timing the round-trip."""
    timings = StageTimings()

    try:
        t0 = time.perf_counter()

        # The Go server handles all stages internally
        # We measure the total HTTP round-trip
        payload = {
            "model": model_name,
            "image_b64": image_b64,
            "conf_thresh": 0.25,
            "iou_thresh": 0.45,
        }

        t_stage = time.perf_counter()
        resp = http_requests.post(f"{addr}/v1/detect", json=payload, timeout=30)
        total_http = (time.perf_counter() - t_stage) * 1000
        resp.raise_for_status()

        data = resp.json()
        objs = data.get("objects") or []
        n_objects = len(objs)

        # For Go HTTP, all stages are opaque — measure total only
        timings.total_ms = (time.perf_counter() - t0) * 1000
        timings.inference_ms = total_http  # approximate — server-side is all-in-one
        timings.n_detections = n_objects

    except Exception as e:
        timings.error = str(e)
        timings.total_ms = (time.perf_counter() - t0) * 1000

    return timings


def bench_go_http(addr: str, model_name: str, image_bytes: bytes,
                  n_requests: int, concurrency: int, provider: str = "cuda") -> BenchRun:
    """Benchmark the Native Go HTTP detection endpoint."""
    run = BenchRun(
        target="go_http", provider=provider,
        concurrency=concurrency, n_requests=n_requests,
    )

    image_b64 = base64.b64encode(image_bytes).decode()

    # Verify server is up
    print(f"    Checking server at {addr}...")
    try:
        resp = http_requests.get(f"{addr}/v1/models", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        print(f"    Server models: {[m['id'] for m in models]}")
    except Exception as e:
        print(f"    ERROR: Server not reachable: {e}")
        return run

    # Cold start: first request
    print("    Measuring cold start...")
    cold = run_go_http_single(addr, model_name, image_b64)
    run.cold_start_ms = cold.total_ms
    print(f"    Cold start: {cold.total_ms:.1f}ms ({cold.n_detections} detections)")

    # Warmup
    for _ in range(3):
        run_go_http_single(addr, model_name, image_b64)

    # Main benchmark
    run.rss_before_mb = get_process_rss_mb()
    gpu_mon = GPUMonitor(interval=0.2)
    gpu_mon.start()

    print(f"    Running {n_requests} requests (concurrency={concurrency})...")
    t_start = time.perf_counter()

    if concurrency <= 1:
        for i in range(n_requests):
            run.timings.append(run_go_http_single(addr, model_name, image_b64))
            if (i + 1) % max(1, n_requests // 5) == 0:
                print(f"      {i + 1}/{n_requests} done", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(run_go_http_single, addr, model_name, image_b64)
                       for _ in range(n_requests)]
            for i, fut in enumerate(as_completed(futures), 1):
                run.timings.append(fut.result())
                if i % max(1, n_requests // 5) == 0:
                    print(f"      {i}/{n_requests} done", flush=True)

    run.duration_s = time.perf_counter() - t_start
    gpu_snapshots = gpu_mon.stop()
    run.gpu_stats = GPUMonitor.summary(gpu_snapshots)
    run.rss_after_mb = get_process_rss_mb()

    return run


# ─── Stress Test (Ramp Concurrency) ─────────────────────────────────────────

def stress_test(target: str, model_path: str = "", addr: str = "",
                model_name: str = "", provider: str = "cuda",
                image_bytes: bytes = b"",
                concurrency_levels: list[int] = None,
                requests_per_level: int = 50) -> list[BenchRun]:
    """Ramp concurrency to find saturation point. Clears GPU memory between levels."""
    if concurrency_levels is None:
        concurrency_levels = [1, 2, 4, 8, 16, 32, 64]

    results = []
    image_b64 = base64.b64encode(image_bytes).decode() if image_bytes else ""

    for c in concurrency_levels:
        print(f"\n  === Stress level: concurrency={c} ===")
        clear_gpu_memory()
        time.sleep(0.5)

        if target == "python_onnxrt":
            run = bench_python_native(model_path, provider, image_bytes, requests_per_level, c)
        else:
            run = bench_go_http(addr, model_name, image_bytes, requests_per_level, c, provider)

        results.append(run)

        # Check for failures — if error rate > 50%, stop ramping
        err_rate = len(run.errors) / max(1, len(run.timings))
        if err_rate > 0.5:
            print(f"  !!! Error rate {err_rate:.0%} at concurrency={c}, stopping stress test")
            break

        stats = run.latency_stats()
        print(f"    → {run.throughput_rps():.1f} req/s, P50={stats.get('p50', 0):.1f}ms, "
              f"P99={stats.get('p99', 0):.1f}ms, errors={len(run.errors)}")
        if run.gpu_stats:
            print(f"    → GPU: {run.gpu_stats.get('gpu_util_mean', 0):.0f}% util, "
                  f"{run.gpu_stats.get('gpu_mem_used_max_mb', 0):.0f}MB mem, "
                  f"{run.gpu_stats.get('gpu_power_mean_w', 0):.0f}W power")

    return results


# ─── Pretty Printing ─────────────────────────────────────────────────────────

def print_run(run: BenchRun):
    """Print detailed results for a single benchmark run."""
    ok = len(run.ok_timings)
    err = len(run.errors)
    stats = run.latency_stats()
    stages = run.stage_stats()

    print(f"\n{'─' * 72}")
    print(f"  Target:       {run.target}  (provider={run.provider})")
    print(f"  Concurrency:  {run.concurrency}")
    print(f"  Requests:     {run.n_requests} total | {ok} ok | {err} errors")
    print(f"  Duration:     {run.duration_s:.2f}s")
    print(f"  Throughput:   {run.throughput_rps():.1f} req/s  ({run.throughput_rps() * 1000:.0f} images/1000s)")
    print(f"  Cold start:   {run.cold_start_ms:.1f} ms")

    if stats:
        print(f"\n  End-to-end latency:")
        print(f"    Mean:       {stats['mean']:.1f} ms")
        print(f"    P50:        {stats['p50']:.1f} ms")
        print(f"    P95:        {stats['p95']:.1f} ms")
        print(f"    P99:        {stats['p99']:.1f} ms")
        print(f"    Min:        {stats['min']:.1f} ms")
        print(f"    Max:        {stats['max']:.1f} ms")
        print(f"    Stdev:      {stats['stdev']:.1f} ms")

    if stages:
        print(f"\n  Per-stage latency (mean / P50 / P95 ms):")
        for name, s in stages.items():
            if s["mean"] > 0.01:
                print(f"    {name:>12s}:  {s['mean']:7.2f} / {s['p50']:7.2f} / {s['p95']:7.2f}")

    if run.ok_timings:
        det_counts = [t.n_detections for t in run.ok_timings]
        print(f"\n  Detections per image: mean={statistics.mean(det_counts):.1f}, "
              f"min={min(det_counts)}, max={max(det_counts)}")

    g = run.gpu_stats
    if g:
        print(f"\n  GPU monitoring ({g.get('n_samples', 0)} samples):")
        print(f"    Utilization: {g.get('gpu_util_mean', 0):.1f}% mean / {g.get('gpu_util_max', 0):.0f}% max")
        print(f"    Memory:      {g.get('gpu_mem_used_mean_mb', 0):.0f} MB mean / {g.get('gpu_mem_used_max_mb', 0):.0f} MB max / {g.get('gpu_mem_total_mb', 0):.0f} MB total")
        print(f"    Temperature: {g.get('gpu_temp_mean_c', 0):.0f}°C mean / {g.get('gpu_temp_max_c', 0):.0f}°C max")
        print(f"    Power:       {g.get('gpu_power_mean_w', 0):.0f}W mean / {g.get('gpu_power_max_w', 0):.0f}W max / {g.get('gpu_power_limit_w', 0):.0f}W limit")

    rss_delta = run.rss_after_mb - run.rss_before_mb
    print(f"\n  Memory (client process):")
    print(f"    RSS before:  {run.rss_before_mb:.0f} MB")
    print(f"    RSS after:   {run.rss_after_mb:.0f} MB")
    print(f"    Delta:       {rss_delta:+.0f} MB")

    if err > 0:
        print(f"\n  Errors (first 3): {[e.error for e in run.errors[:3]]}")

    print(f"{'─' * 72}")


def print_comparison(py_run: BenchRun, go_run: BenchRun):
    """Print side-by-side comparison table."""
    py_stats = py_run.latency_stats()
    go_stats = go_run.latency_stats()

    def ratio_str(a, b, lower_is_better=True):
        if b == 0:
            return "—"
        r = a / b if lower_is_better else b / a
        if r > 1:
            return f"+{(r - 1) * 100:.0f}%"
        return f"{(r - 1) * 100:.0f}%"

    print(f"\n{'═' * 72}")
    print(f"  COMPARISON: Python Native vs Native Go HTTP  (concurrency={py_run.concurrency})")
    print(f"{'═' * 72}")
    header = f"  {'Metric':<30} {'Python':>12} {'Go HTTP':>12} {'Advantage':>12}"
    print(header)
    print(f"  {'─' * 30} {'─' * 12} {'─' * 12} {'─' * 12}")

    def row(name, py_val, go_val, fmt="{:.1f}", lower_better=True):
        adv_val = go_val / py_val if py_val > 0 else 0
        if lower_better:
            who = "Go" if go_val < py_val else "Python"
            pct = abs(py_val - go_val) / max(py_val, go_val) * 100 if max(py_val, go_val) > 0 else 0
        else:
            who = "Go" if go_val > py_val else "Python"
            pct = abs(go_val - py_val) / max(py_val, go_val) * 100 if max(py_val, go_val) > 0 else 0
        adv = f"{who} +{pct:.0f}%"
        print(f"  {name:<30} {fmt.format(py_val):>12} {fmt.format(go_val):>12} {adv:>12}")

    row("Throughput (req/s)", py_run.throughput_rps(), go_run.throughput_rps(), lower_better=False)
    if py_stats and go_stats:
        row("Latency mean (ms)", py_stats["mean"], go_stats["mean"])
        row("Latency P50 (ms)", py_stats["p50"], go_stats["p50"])
        row("Latency P95 (ms)", py_stats["p95"], go_stats["p95"])
        row("Latency P99 (ms)", py_stats["p99"], go_stats["p99"])
    row("Cold start (ms)", py_run.cold_start_ms, go_run.cold_start_ms)

    py_gpu = py_run.gpu_stats
    go_gpu = go_run.gpu_stats
    if py_gpu and go_gpu:
        row("GPU util mean (%)", py_gpu.get("gpu_util_mean", 0), go_gpu.get("gpu_util_mean", 0), fmt="{:.0f}", lower_better=False)
        row("GPU mem max (MB)", py_gpu.get("gpu_mem_used_max_mb", 0), go_gpu.get("gpu_mem_used_max_mb", 0), fmt="{:.0f}")
        row("GPU power mean (W)", py_gpu.get("gpu_power_mean_w", 0), go_gpu.get("gpu_power_mean_w", 0), fmt="{:.0f}")

    print(f"{'═' * 72}\n")


# ─── Markdown Output ─────────────────────────────────────────────────────────

def emit_markdown(all_runs: list[BenchRun], stress_results: dict, out_path: str,
                  hw_info: str = ""):
    """Write comprehensive markdown results file."""
    lines = []
    lines.append("# Detection Benchmark: Python Native Bindings vs Native Go HTTP\n")
    lines.append(f"*Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}*\n")
    if hw_info:
        lines.append(f"**Hardware:** {hw_info}  ")
    lines.append("**Model:** YOLOv8n (640×640 input, 80 COCO classes)  ")
    lines.append("**Image:** Synthetic 640×480 JPEG  ")
    lines.append("**GPU memory cleared between every benchmark run**\n")
    lines.append("---\n")

    # Main comparison table
    if len(all_runs) >= 2:
        lines.append("## Head-to-Head Comparison\n")
        headers = ["Metric", "Python Native", "Go HTTP", "Winner"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        py = next((r for r in all_runs if r.target == "python_onnxrt"), None)
        go = next((r for r in all_runs if r.target == "go_http"), None)
        if py and go:
            py_s = py.latency_stats()
            go_s = go.latency_stats()

            def winner(py_v, go_v, lower_better=True):
                if lower_better:
                    w = "Python" if py_v < go_v else "Go"
                else:
                    w = "Python" if py_v > go_v else "Go"
                diff = abs(py_v - go_v) / max(py_v, go_v) * 100 if max(py_v, go_v) > 0 else 0
                return f"**{w}** +{diff:.0f}%"

            rows = [
                ("Throughput (req/s)", f"{py.throughput_rps():.1f}", f"{go.throughput_rps():.1f}",
                 winner(py.throughput_rps(), go.throughput_rps(), lower_better=False)),
                ("Mean latency (ms)", f"{py_s.get('mean', 0):.1f}", f"{go_s.get('mean', 0):.1f}",
                 winner(py_s.get('mean', 0), go_s.get('mean', 0))),
                ("P50 latency (ms)", f"{py_s.get('p50', 0):.1f}", f"{go_s.get('p50', 0):.1f}",
                 winner(py_s.get('p50', 0), go_s.get('p50', 0))),
                ("P95 latency (ms)", f"{py_s.get('p95', 0):.1f}", f"{go_s.get('p95', 0):.1f}",
                 winner(py_s.get('p95', 0), go_s.get('p95', 0))),
                ("P99 latency (ms)", f"{py_s.get('p99', 0):.1f}", f"{go_s.get('p99', 0):.1f}",
                 winner(py_s.get('p99', 0), go_s.get('p99', 0))),
                ("Cold start (ms)", f"{py.cold_start_ms:.1f}", f"{go.cold_start_ms:.1f}",
                 winner(py.cold_start_ms, go.cold_start_ms)),
            ]
            for name, pv, gv, w in rows:
                lines.append(f"| {name} | {pv} | {gv} | {w} |")
        lines.append("")

    # Detailed per-run results
    for run in all_runs:
        stats = run.latency_stats()
        stages = run.stage_stats()
        lines.append(f"## {run.target} (concurrency={run.concurrency}, provider={run.provider})\n")

        lines.append("### Throughput & Latency\n")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| Throughput | **{run.throughput_rps():.1f} req/s** |")
        lines.append(f"| Requests | {len(run.ok_timings)} ok / {len(run.errors)} errors |")
        lines.append(f"| Duration | {run.duration_s:.2f}s |")
        lines.append(f"| Cold start | {run.cold_start_ms:.1f}ms |")
        if stats:
            lines.append(f"| Mean latency | {stats['mean']:.1f}ms |")
            lines.append(f"| P50 | {stats['p50']:.1f}ms |")
            lines.append(f"| P95 | {stats['p95']:.1f}ms |")
            lines.append(f"| P99 | {stats['p99']:.1f}ms |")
            lines.append(f"| Min | {stats['min']:.1f}ms |")
            lines.append(f"| Max | {stats['max']:.1f}ms |")
            lines.append(f"| Stdev | {stats['stdev']:.1f}ms |")
        lines.append("")

        if stages:
            lines.append("### Per-Stage Latency (ms)\n")
            lines.append("| Stage | Mean | P50 | P95 |")
            lines.append("| --- | --- | --- | --- |")
            for name, s in stages.items():
                if s["mean"] > 0.01:
                    lines.append(f"| {name} | {s['mean']:.2f} | {s['p50']:.2f} | {s['p95']:.2f} |")
            lines.append("")

        g = run.gpu_stats
        if g:
            lines.append("### GPU Monitoring\n")
            lines.append("| Metric | Mean | Max |")
            lines.append("| --- | --- | --- |")
            lines.append(f"| Utilization (%) | {g.get('gpu_util_mean', 0):.1f} | {g.get('gpu_util_max', 0):.0f} |")
            lines.append(f"| Memory used (MB) | {g.get('gpu_mem_used_mean_mb', 0):.0f} | {g.get('gpu_mem_used_max_mb', 0):.0f} / {g.get('gpu_mem_total_mb', 0):.0f} |")
            lines.append(f"| Temperature (°C) | {g.get('gpu_temp_mean_c', 0):.0f} | {g.get('gpu_temp_max_c', 0):.0f} |")
            lines.append(f"| Power (W) | {g.get('gpu_power_mean_w', 0):.0f} | {g.get('gpu_power_max_w', 0):.0f} / {g.get('gpu_power_limit_w', 0):.0f} |")
            lines.append(f"| Samples collected | {g.get('n_samples', 0)} | — |")
            lines.append("")

        rss_delta = run.rss_after_mb - run.rss_before_mb
        lines.append("### Memory (Client Process)\n")
        lines.append(f"| RSS before | RSS after | Delta |")
        lines.append(f"| --- | --- | --- |")
        lines.append(f"| {run.rss_before_mb:.0f} MB | {run.rss_after_mb:.0f} MB | {rss_delta:+.0f} MB |")
        lines.append("")

    # Stress test results
    for target, runs in stress_results.items():
        if not runs:
            continue
        lines.append(f"## Stress Test: {target}\n")
        lines.append("Concurrency ramp to find saturation point. GPU memory cleared between each level.\n")
        headers = ["Concurrency", "req/s", "Mean ms", "P50 ms", "P95 ms", "P99 ms",
                    "GPU util %", "GPU mem MB", "GPU power W", "Errors"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for run in runs:
            stats = run.latency_stats()
            g = run.gpu_stats
            row = [
                str(run.concurrency),
                f"{run.throughput_rps():.1f}",
                f"{stats.get('mean', 0):.1f}",
                f"{stats.get('p50', 0):.1f}",
                f"{stats.get('p95', 0):.1f}",
                f"{stats.get('p99', 0):.1f}",
                f"{g.get('gpu_util_mean', 0):.0f}" if g else "—",
                f"{g.get('gpu_mem_used_max_mb', 0):.0f}" if g else "—",
                f"{g.get('gpu_power_mean_w', 0):.0f}" if g else "—",
                str(len(run.errors)),
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Results written to {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Detection benchmark: Python native vs Go HTTP")

    # Target selection
    p.add_argument("--target", default="python", choices=["python", "go", "both"],
                   help="Which target to benchmark (default: python)")
    p.add_argument("--compare", action="store_true",
                   help="Run both targets and compare")

    # Python native settings
    p.add_argument("--model-path", default="models/yolov8n.onnx",
                   help="Path to ONNX model file (for Python native)")
    p.add_argument("--provider", default="cuda", choices=["cpu", "cuda", "tensorrt"],
                   help="ONNX execution provider (default: cuda)")

    # Go HTTP settings
    p.add_argument("--addr", default="http://localhost:9090",
                   help="infergo server address (for Go HTTP)")
    p.add_argument("--model-name", default="yolov8n",
                   help="Model name on infergo server")

    # Benchmark parameters
    p.add_argument("--requests", type=int, default=100,
                   help="Number of requests per benchmark (default: 100)")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Concurrent requests (default: 1)")
    p.add_argument("--image-width", type=int, default=640)
    p.add_argument("--image-height", type=int, default=480)

    # Stress test
    p.add_argument("--stress", action="store_true",
                   help="Run stress test ramping concurrency 1→64")
    p.add_argument("--stress-requests", type=int, default=50,
                   help="Requests per concurrency level in stress test (default: 50)")
    p.add_argument("--stress-levels", default="1,2,4,8,16,32,64",
                   help="Comma-separated concurrency levels for stress test")

    # Output
    p.add_argument("--out", default="benchmarks/vs_python/results_detect.md",
                   help="Markdown output file")
    p.add_argument("--hw-info", default="",
                   help="Hardware description for results header")

    args = p.parse_args()

    if args.compare:
        args.target = "both"

    # Auto-detect hardware info
    hw_info = args.hw_info
    if not hw_info:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                hw_info = result.stdout.strip()
        except Exception:
            hw_info = "Unknown GPU"

    print(f"\n{'═' * 72}")
    print(f"  Detection Benchmark")
    print(f"  Target:      {args.target}")
    print(f"  Provider:    {args.provider}")
    print(f"  Requests:    {args.requests}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Hardware:    {hw_info}")
    print(f"{'═' * 72}\n")

    # Generate test image
    print("Generating synthetic test image...")
    image_bytes = make_test_image_bytes(args.image_width, args.image_height)
    print(f"  Image: {args.image_width}×{args.image_height} JPEG, {len(image_bytes)} bytes\n")

    all_runs = []
    stress_results = {}

    # ── Python Native ──
    if args.target in ("python", "both"):
        print("=" * 40)
        print(" Python Native Bindings")
        print("=" * 40)
        clear_gpu_memory()
        run = bench_python_native(args.model_path, args.provider, image_bytes,
                                  args.requests, args.concurrency)
        print_run(run)
        all_runs.append(run)

        if args.stress:
            levels = [int(x) for x in args.stress_levels.split(",")]
            print(f"\n  Starting stress test: concurrency levels {levels}")
            clear_gpu_memory()
            stress_results["python_onnxrt"] = stress_test(
                "python_onnxrt", model_path=args.model_path,
                provider=args.provider, image_bytes=image_bytes,
                concurrency_levels=levels, requests_per_level=args.stress_requests,
            )

    # ── Clear GPU memory between targets ──
    if args.target == "both":
        print("\n  Clearing GPU memory before Go HTTP benchmark...")
        clear_gpu_memory()
        time.sleep(2)

    # ── Native Go HTTP ──
    if args.target in ("go", "both"):
        print("\n" + "=" * 40)
        print(" Native Go HTTP")
        print("=" * 40)
        run = bench_go_http(args.addr, args.model_name, image_bytes,
                            args.requests, args.concurrency, args.provider)
        print_run(run)
        all_runs.append(run)

        if args.stress:
            levels = [int(x) for x in args.stress_levels.split(",")]
            print(f"\n  Starting stress test: concurrency levels {levels}")
            stress_results["go_http"] = stress_test(
                "go_http", addr=args.addr, model_name=args.model_name,
                provider=args.provider, image_bytes=image_bytes,
                concurrency_levels=levels, requests_per_level=args.stress_requests,
            )

    # ── Comparison ──
    if args.target == "both" and len(all_runs) >= 2:
        py_run = next((r for r in all_runs if r.target == "python_native"), None)
        go_run = next((r for r in all_runs if r.target == "go_http"), None)
        if py_run and go_run:
            print_comparison(py_run, go_run)

    # ── Save results ──
    emit_markdown(all_runs, stress_results, args.out, hw_info)


if __name__ == "__main__":
    main()
