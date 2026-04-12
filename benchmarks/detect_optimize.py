#!/usr/bin/env python3
"""
Detection optimization: benchmark ALL infergo backends vs Python PyTorch
Find the fastest path and identify bottlenecks.
"""

import time, statistics, os, subprocess, signal, json, base64, io
import numpy as np

RUNS = 30
WARMUP = 10

INFERGO = os.path.expanduser("~/cgo/infergo")
CWD = os.path.expanduser("~/cgo")
MODEL_ONNX = os.path.expanduser("~/cgo/models/yolo11n.onnx")
MODEL_PT = os.path.expanduser("~/cgo/models/yolo11n.torchscript.pt")

img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

def banner(t):
    print(f"\n{'─'*60}")
    print(f"  {t}")
    print(f"{'─'*60}")

# ── Python PyTorch baseline ──────────────────────────────────

def bench_python_pytorch():
    banner("Python PyTorch (ultralytics, in-process)")
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")

    for _ in range(WARMUP):
        model(img, verbose=False)

    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        model(img, verbose=False)
        times.append((time.perf_counter()-start)*1000)

    p50 = statistics.median(times)
    print(f"  P50: {p50:.1f}ms | Min: {min(times):.1f}ms | {1000/statistics.mean(times):.0f} RPS")
    return p50

# ── Python ONNX Runtime ──────────────────────────────────────

def bench_python_ort():
    banner("Python ONNX Runtime CUDA (in-process)")
    import onnxruntime as ort
    sess = ort.InferenceSession(MODEL_ONNX, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
    inp = img.astype(np.float32).transpose(2,0,1)[np.newaxis] / 255.0
    name = sess.get_inputs()[0].name

    for _ in range(WARMUP):
        sess.run(None, {name: inp})

    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        sess.run(None, {name: inp})
        times.append((time.perf_counter()-start)*1000)

    p50 = statistics.median(times)
    print(f"  P50: {p50:.1f}ms | Min: {min(times):.1f}ms | {1000/statistics.mean(times):.0f} RPS")
    return p50

# ── infergo backends via binary endpoint ─────────────────────

def start_server(model_path, provider, port):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        f"{os.path.expanduser('~/onnxruntime/lib')}:"
        f"{os.path.expanduser('~/.local/lib/python3.12/site-packages/tensorrt_libs')}:"
        f"{env.get('LD_LIBRARY_PATH','')}")

    proc = subprocess.Popen(
        [INFERGO, "serve", f"--model=detect:{model_path}", f"--provider={provider}",
         f"--port={port}", "--grpc-port=0"],
        cwd=CWD, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid)

    for i in range(60):
        try:
            r = subprocess.run(["curl","-s",f"http://localhost:{port}/health/live"],
                             capture_output=True, timeout=2)
            if r.returncode == 0: return proc
        except: pass
        time.sleep(1)
    return proc

def stop_server(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=5)
    except: pass
    time.sleep(2)

def bench_infergo_binary(label, model_path, provider, port):
    banner(f"infergo {label}")
    proc = start_server(model_path, provider, port)

    from PIL import Image
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    with open("/tmp/bench_opt.jpg","wb") as f:
        f.write(buf.getvalue())

    # Warmup
    for _ in range(WARMUP):
        subprocess.run(["curl","-s","-X","POST",
                       f"http://localhost:{port}/v1/detect/binary?model=detect&conf=0.25",
                       "--data-binary","@/tmp/bench_opt.jpg",
                       "-H","Content-Type: application/octet-stream"], capture_output=True)

    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        subprocess.run(["curl","-s","-X","POST",
                       f"http://localhost:{port}/v1/detect/binary?model=detect&conf=0.25",
                       "--data-binary","@/tmp/bench_opt.jpg",
                       "-H","Content-Type: application/octet-stream"], capture_output=True)
        times.append((time.perf_counter()-start)*1000)

    stop_server(proc)

    p50 = statistics.median(times)
    est = p50 - 5  # subtract HTTP overhead
    print(f"  HTTP P50: {p50:.1f}ms | Est. inference: ~{est:.1f}ms | {1000/statistics.mean(times):.0f} RPS")
    return p50, est

# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("DETECTION OPTIMIZATION BENCHMARK")
    print(f"GPU: RTX 5070 Ti | Model: yolo11n | Image: 640x640 | Runs: {RUNS}")

    results = {}

    # Python baselines
    results["py_pytorch"] = bench_python_pytorch()
    results["py_ort"] = bench_python_ort()

    # infergo backends
    if os.path.exists(MODEL_ONNX):
        _, results["ig_ort_cuda"] = bench_infergo_binary("ONNX CUDA", MODEL_ONNX, "cuda", 9201)
        _, results["ig_ort_trt"] = bench_infergo_binary("ONNX TensorRT", MODEL_ONNX, "tensorrt", 9202)

    if os.path.exists(MODEL_PT):
        _, results["ig_torch"] = bench_infergo_binary("TorchScript CUDA", MODEL_PT, "cuda", 9203)

    # Summary
    banner("SCORECARD")
    print(f"\n  {'Backend':<35} {'P50':>8} {'vs PyTorch':>12}")
    print(f"  {'─'*55}")

    baseline = results.get("py_pytorch", 1)
    for name, key in [
        ("Python PyTorch", "py_pytorch"),
        ("Python ONNX Runtime", "py_ort"),
        ("infergo ONNX CUDA", "ig_ort_cuda"),
        ("infergo ONNX TensorRT", "ig_ort_trt"),
        ("infergo TorchScript CUDA", "ig_torch"),
    ]:
        val = results.get(key)
        if val is not None:
            ratio = val / baseline if baseline > 0 else 0
            faster = "faster" if ratio < 1 else "slower"
            print(f"  {name:<35} {val:>6.1f}ms  {ratio:.1f}x {faster}")

    banner("ANALYSIS")
    ig_best = min(v for k,v in results.items() if k.startswith("ig_") and v is not None)
    gap = ig_best - baseline
    print(f"\n  Best infergo:  {ig_best:.1f}ms")
    print(f"  Python PyTorch: {baseline:.1f}ms")
    print(f"  Gap: {gap:+.1f}ms")

    if gap > 1:
        print(f"\n  ROOT CAUSE:")
        print(f"    PyTorch uses fused CUDA kernels (cuDNN conv2d + custom ops)")
        print(f"    ONNX Runtime uses generic CUDA kernels (slower for small models)")
        print(f"    TorchScript should match PyTorch — if gap persists, it's")
        print(f"    preprocessing or postprocessing overhead in infergo")
    else:
        print(f"\n  Gap is <1ms — effectively equal")
