#!/usr/bin/env python3
"""
bench_all_combos.py — Run EVERY detection benchmark combination.

Tests:
  1. Python PyTorch GPU (in-process, ultralytics)
  2. Python PyTorch CPU (in-process, ultralytics)
  3. Go libtorch GPU (HTTP, torch backend)
  4. Go libtorch CPU (HTTP, torch backend)
  5. Go ONNX CUDA EP GPU (HTTP, onnx backend)
  6. Go ONNX TensorRT GPU (HTTP, tensorrt backend)
  7. Go libtorch GPU binary endpoint (HTTP, raw JPEG)
  8. Go adaptive GPU (HTTP, adaptive backend)

Each test: 4 models (yolo11n/s/m/l), sequential + concurrent (c=1,4,8,16)
"""

import argparse, base64, json, os, signal, subprocess, sys, time, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np, cv2

try:
    import requests as http_req
except ImportError:
    sys.exit("pip install requests")


# ─── Test image ──────────────────────────────────────────────────────────────

def make_image(w=640, h=480):
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    for _ in range(5):
        x1, y1 = np.random.randint(0, w-80), np.random.randint(0, h-80)
        cv2.rectangle(img, (x1,y1), (x1+60,y1+60), (100,200,50), -1)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return img, buf.tobytes()


# ─── Result types ────────────────────────────────────────────────────────────

@dataclass
class Result:
    backend: str
    provider: str  # cuda or cpu
    model: str
    concurrency: int
    n_requests: int
    mean_ms: float = 0
    p50_ms: float = 0
    p95_ms: float = 0
    p99_ms: float = 0
    rps: float = 0
    errors: int = 0
    endpoint: str = ""  # json, binary, native


# ─── Python native benchmark ────────────────────────────────────────────────

def bench_python(img_np, provider, models, n_per_model, concurrency_levels):
    from ultralytics import YOLO
    import torch

    results = []
    for name in models:
        m = YOLO(f"models/{name}.pt")
        device = "cuda" if provider == "cuda" and torch.cuda.is_available() else "cpu"
        m.to(device)
        # warmup
        for _ in range(10):
            m.predict(img_np, verbose=False)

        for c in concurrency_levels:
            ts = []
            for _ in range(n_per_model):
                t0 = time.perf_counter()
                m.predict(img_np, verbose=False)
                ts.append((time.perf_counter()-t0)*1000)
            ts.sort()
            n = len(ts)
            results.append(Result(
                backend="python_pytorch", provider=provider, model=name,
                concurrency=c, n_requests=n_per_model,
                mean_ms=statistics.mean(ts), p50_ms=ts[n//2],
                p95_ms=ts[int(n*0.95)], p99_ms=ts[min(int(n*0.99), n-1)],
                rps=1000/statistics.mean(ts), errors=0, endpoint="native",
            ))
        del m
        if provider == "cuda":
            torch.cuda.empty_cache()
    return results


# ─── Go HTTP benchmark ──────────────────────────────────────────────────────

def bench_go_http(addr, models, jpeg_bytes, n_per_model, concurrency_levels, endpoint="json"):
    b64 = base64.b64encode(jpeg_bytes).decode()
    results = []

    for name in models:
        # warmup
        for _ in range(5):
            if endpoint == "binary":
                http_req.post(f"{addr}/v1/detect/binary?model={name}&conf=0.25&iou=0.45",
                              data=jpeg_bytes, headers={"Content-Type":"image/jpeg"}, timeout=30)
            else:
                http_req.post(f"{addr}/v1/detect", json={"model":name,"image_b64":b64,
                              "conf_thresh":0.25,"iou_thresh":0.45}, timeout=30)

        for c in concurrency_levels:
            ts = []
            errs = 0

            def do_request():
                t0 = time.perf_counter()
                try:
                    if endpoint == "binary":
                        r = http_req.post(f"{addr}/v1/detect/binary?model={name}&conf=0.25&iou=0.45",
                                          data=jpeg_bytes, headers={"Content-Type":"image/jpeg"}, timeout=30)
                    else:
                        r = http_req.post(f"{addr}/v1/detect", json={"model":name,"image_b64":b64,
                                          "conf_thresh":0.25,"iou_thresh":0.45}, timeout=30)
                    ms = (time.perf_counter()-t0)*1000
                    if r.status_code != 200:
                        return ms, True
                    return ms, False
                except:
                    return (time.perf_counter()-t0)*1000, True

            if c <= 1:
                for _ in range(n_per_model):
                    ms, err = do_request()
                    ts.append(ms)
                    if err: errs += 1
            else:
                with ThreadPoolExecutor(max_workers=c) as pool:
                    futs = [pool.submit(do_request) for _ in range(n_per_model)]
                    for f in as_completed(futs):
                        ms, err = f.result()
                        ts.append(ms)
                        if err: errs += 1

            ts.sort()
            n = len(ts)
            duration = sum(ts) / 1000
            results.append(Result(
                backend="go_http", provider="cuda", model=name,
                concurrency=c, n_requests=n_per_model,
                mean_ms=statistics.mean(ts), p50_ms=ts[n//2],
                p95_ms=ts[int(n*0.95)], p99_ms=ts[min(int(n*0.99), n-1)],
                rps=len([t for t in ts if t]) / duration if duration > 0 else 0,
                errors=errs, endpoint=endpoint,
            ))
    return results


# ─── Server management ───────────────────────────────────────────────────────

LD = ("build/cpp/api:build/cpp/llm:build/cpp/onnx:build/cpp/tokenizer:build/cpp/tensor:"
      "build/cpp/preprocess:build/cpp/postprocess:build/cpp/torch:"
      "/home/lakshya/onnxruntime/lib:"
      "/home/lakshya/.local/lib/python3.12/site-packages/tensorrt_libs:"
      "/home/lakshya/yolo-env/lib/python3.12/site-packages/torch/lib:"
      "/home/lakshya/yolo-env/lib/python3.12/site-packages/nvidia/cudnn/lib:"
      "/home/lakshya/yolo-env/lib/python3.12/site-packages/nvidia/cublas/lib:"
      "/home/lakshya/yolo-env/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:"
      "/home/lakshya/yolo-env/lib/python3.12/site-packages/nvidia/cufft/lib:"
      "/home/lakshya/yolo-env/lib/python3.12/site-packages/nvidia/curand/lib:"
      "/home/lakshya/vcpkg/installed/x64-linux/lib")

def start_server(backend, provider, port=9192):
    subprocess.run(["pkill", "-f", "infergo serve"], capture_output=True)
    time.sleep(2)

    models = {
        "torch": [f"yolo11{s}:models/yolo11{s}.torchscript.pt" for s in "nsml"],
        "onnx":  [f"yolo11{s}:models/yolo11{s}.onnx" for s in "nsml"],
        "tensorrt": [f"yolo11{s}:models/yolo11{s}.onnx" for s in "nsml"],
        "adaptive": [f"yolo11{s}:models/yolo11{s}.torchscript.pt" for s in "nsml"],
    }

    cmd = ["./infergo", "serve", "--provider", provider, "--backend", backend,
           "--port", str(port), "--grpc-port", "0"]
    for m in models.get(backend, models["torch"]):
        cmd.extend(["--model", m])
    if backend == "adaptive":
        cmd.append("--adaptive")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LD

    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for server
    wait = 120 if backend in ("tensorrt", "adaptive") else 30
    for _ in range(wait):
        time.sleep(1)
        try:
            r = http_req.get(f"http://localhost:{port}/v1/models", timeout=2)
            if r.status_code == 200:
                return proc
        except:
            pass
    print(f"  WARNING: server didn't start for {backend}/{provider}")
    proc.kill()
    return None


def stop_server(proc):
    if proc:
        proc.terminate()
        proc.wait(timeout=10)
    time.sleep(2)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30, help="Requests per model per concurrency level")
    p.add_argument("--out", default="benchmarks/vs_python/results_all_combos.md")
    args = p.parse_args()

    models = ["yolo11n", "yolo11s", "yolo11m", "yolo11l"]
    concurrency = [1, 4, 8, 16]
    img_np, jpeg = make_image()
    all_results = []

    print("=" * 70)
    print("  FULL DETECTION BENCHMARK — EVERY COMBINATION")
    print("=" * 70)

    # ── 1. Python PyTorch GPU ──
    print("\n[1/8] Python PyTorch GPU...")
    all_results.extend(bench_python(img_np, "cuda", models, args.n, [1]))
    print(f"  Done: {all_results[-1].p50_ms:.1f}ms P50 (yolo11n)")

    # ── 2. Python PyTorch CPU ──
    print("\n[2/8] Python PyTorch CPU...")
    all_results.extend(bench_python(img_np, "cpu", models, args.n, [1]))
    print(f"  Done: {all_results[-1].p50_ms:.1f}ms P50 (yolo11n)")

    # ── 3. Go libtorch GPU ──
    print("\n[3/8] Go libtorch GPU (JSON)...")
    proc = start_server("torch", "cuda")
    if proc:
        all_results.extend(bench_go_http("http://localhost:9192", models, jpeg, args.n, concurrency, "json"))
        print(f"  Done: {[r for r in all_results if r.backend=='go_http' and r.model=='yolo11n' and r.concurrency==1][-1].p50_ms:.1f}ms P50")
        stop_server(proc)

    # ── 4. Go libtorch CPU ──
    print("\n[4/8] Go libtorch CPU (JSON)...")
    proc = start_server("torch", "cpu")
    if proc:
        res = bench_go_http("http://localhost:9192", models, jpeg, args.n, [1, 4], "json")
        for r in res: r.provider = "cpu"
        all_results.extend(res)
        print(f"  Done: {res[0].p50_ms:.1f}ms P50 (yolo11n)")
        stop_server(proc)

    # ── 5. Go ONNX CUDA ──
    print("\n[5/8] Go ONNX CUDA (JSON)...")
    proc = start_server("onnx", "cuda")
    if proc:
        res = bench_go_http("http://localhost:9192", models, jpeg, args.n, concurrency, "json")
        for r in res: r.backend = "go_onnx_cuda"
        all_results.extend(res)
        print(f"  Done: {[r for r in res if r.model=='yolo11n' and r.concurrency==1][0].p50_ms:.1f}ms P50")
        stop_server(proc)

    # ── 6. Go TensorRT GPU ──
    print("\n[6/8] Go TensorRT GPU (JSON)...")
    proc = start_server("tensorrt", "cuda")
    if proc:
        res = bench_go_http("http://localhost:9192", models, jpeg, args.n, concurrency, "json")
        for r in res: r.backend = "go_tensorrt"
        all_results.extend(res)
        print(f"  Done: {[r for r in res if r.model=='yolo11n' and r.concurrency==1][0].p50_ms:.1f}ms P50")
        stop_server(proc)

    # ── 7. Go libtorch GPU binary ──
    print("\n[7/8] Go libtorch GPU (Binary endpoint)...")
    proc = start_server("torch", "cuda")
    if proc:
        res = bench_go_http("http://localhost:9192", models, jpeg, args.n, concurrency, "binary")
        for r in res: r.backend = "go_torch_binary"
        all_results.extend(res)
        print(f"  Done: {[r for r in res if r.model=='yolo11n' and r.concurrency==1][0].p50_ms:.1f}ms P50")
        stop_server(proc)

    # ── 8. Go adaptive GPU ──
    print("\n[8/8] Go Adaptive GPU (JSON)...")
    proc = start_server("adaptive", "cuda")
    if proc:
        res = bench_go_http("http://localhost:9192", models, jpeg, args.n, concurrency, "json")
        for r in res: r.backend = "go_adaptive"
        all_results.extend(res)
        print(f"  Done: {[r for r in res if r.model=='yolo11n' and r.concurrency==1][0].p50_ms:.1f}ms P50")
        stop_server(proc)

    # ── Save results ──
    save_markdown(all_results, args.out)
    print(f"\nResults saved to {args.out}")


def save_markdown(results, path):
    lines = []
    lines.append("# Detection Benchmark: All Combinations\n")
    lines.append(f"*Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}*\n")
    lines.append("**Hardware:** NVIDIA GeForce RTX 5070 Ti 16GB, AMD Ryzen 9 9900X 12-Core  ")
    lines.append("**Models:** YOLOv11 n/s/m/l (640x640 COCO)  ")
    lines.append("**Image:** Synthetic 640x480 JPEG\n")
    lines.append("---\n")

    # Summary: best P50 per backend at c=1
    lines.append("## Summary — yolo11n P50 latency (c=1)\n")
    lines.append("| Backend | Provider | Endpoint | P50 ms | req/s | Errors |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    seen = set()
    for r in results:
        if r.model == "yolo11n" and r.concurrency == 1:
            key = (r.backend, r.provider, r.endpoint)
            if key not in seen:
                seen.add(key)
                lines.append(f"| {r.backend} | {r.provider} | {r.endpoint} | {r.p50_ms:.1f} | {r.rps:.0f} | {r.errors} |")
    lines.append("")

    # Per-model at c=1
    lines.append("## Per-Model — Sequential (c=1)\n")
    lines.append("| Backend | Provider | yolo11n | yolo11s | yolo11m | yolo11l |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    backends_seen = set()
    for r in results:
        if r.concurrency == 1:
            key = (r.backend, r.provider, r.endpoint)
            if key not in backends_seen:
                backends_seen.add(key)
                row = [r.backend, r.provider]
                for m in ["yolo11n", "yolo11s", "yolo11m", "yolo11l"]:
                    match = [x for x in results if x.backend==r.backend and x.provider==r.provider
                             and x.endpoint==r.endpoint and x.model==m and x.concurrency==1]
                    if match:
                        row.append(f"{match[0].p50_ms:.1f}ms")
                    else:
                        row.append("—")
                lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Concurrency scaling for each backend (yolo11n only)
    lines.append("## Concurrency Scaling — yolo11n\n")
    lines.append("| Backend | c=1 req/s | c=4 req/s | c=8 req/s | c=16 req/s | Errors |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    backends_seen = set()
    for r in results:
        if r.model == "yolo11n":
            key = (r.backend, r.provider, r.endpoint)
            if key not in backends_seen:
                backends_seen.add(key)
                row = [f"{r.backend} ({r.provider})"]
                total_errs = 0
                for c in [1, 4, 8, 16]:
                    match = [x for x in results if x.backend==r.backend and x.provider==r.provider
                             and x.endpoint==r.endpoint and x.model=="yolo11n" and x.concurrency==c]
                    if match:
                        row.append(f"{match[0].rps:.0f}")
                        total_errs += match[0].errors
                    else:
                        row.append("—")
                row.append(str(total_errs))
                lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Full raw data
    lines.append("## Raw Data\n")
    lines.append("| Backend | Provider | Endpoint | Model | Conc | Mean ms | P50 ms | P95 ms | P99 ms | req/s | Errors |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        lines.append(f"| {r.backend} | {r.provider} | {r.endpoint} | {r.model} | {r.concurrency} | "
                     f"{r.mean_ms:.1f} | {r.p50_ms:.1f} | {r.p95_ms:.1f} | {r.p99_ms:.1f} | "
                     f"{r.rps:.0f} | {r.errors} |")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
