#!/usr/bin/env python3
"""Detection backend benchmark: TensorRT vs CUDA vs CPU on yolo11n.onnx"""

import subprocess, time, json, os, sys, base64, signal

MODEL = "models/yolo11n.onnx"
IMG = "/tmp/test_detect.jpg"
WARMUP = 3
RUNS = 20
CWD = "/home/lakshya/cgo"

# Build request JSON
with open(IMG, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

req_file = "/tmp/detect_req.json"
with open(req_file, "w") as f:
    json.dump({"model": "detect", "image_b64": img_b64, "conf_thresh": 0.25}, f)

def start_server(provider, port):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/home/lakshya/onnxruntime/lib:/home/lakshya/.local/lib/python3.12/site-packages/tensorrt_libs:" + env.get("LD_LIBRARY_PATH", "")
    proc = subprocess.Popen(
        [f"{CWD}/infergo", "serve", f"--model=detect:{MODEL}",
         f"--provider={provider}", f"--port={port}", "--grpc-port=0"],
        cwd=CWD, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid
    )
    # Wait for ready
    for i in range(120):
        try:
            r = subprocess.run(["curl", "-s", f"http://localhost:{port}/health/live"],
                             capture_output=True, timeout=2)
            if r.returncode == 0:
                return proc
        except:
            pass
        time.sleep(1)
    print(f"  TIMEOUT: {provider} server did not start in 120s")
    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    return None

def stop_server(proc):
    if proc:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)
        except:
            pass

def bench(provider, port):
    print(f"\n=== {provider.upper()} yolo11n ({RUNS} runs) ===")
    proc = start_server(provider, port)
    if not proc:
        return None

    # Warmup
    for _ in range(WARMUP):
        subprocess.run(["curl", "-s", "-X", "POST", f"http://localhost:{port}/v1/detect",
                        "-d", f"@{req_file}", "-H", "Content-Type: application/json"],
                       capture_output=True)

    # Benchmark
    times = []
    for i in range(RUNS):
        start = time.perf_counter()
        r = subprocess.run(["curl", "-s", "-X", "POST", f"http://localhost:{port}/v1/detect",
                           "-d", f"@{req_file}", "-H", "Content-Type: application/json"],
                          capture_output=True, text=True)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg = sum(times) / len(times)
    mn = min(times)
    mx = max(times)
    rps = 1000 / avg if avg > 0 else 0

    for i, t in enumerate(times):
        print(f"  run {i+1:2d}: {t:6.1f}ms")
    print(f"  Avg: {avg:.1f}ms | Min: {mn:.1f}ms | Max: {mx:.1f}ms | RPS: {rps:.0f}")

    stop_server(proc)
    time.sleep(2)
    return {"avg": avg, "min": mn, "max": mx, "rps": rps}

if __name__ == "__main__":
    print(f"Model: {MODEL}")
    print(f"Image: 640x640 random")
    print(f"Warmup: {WARMUP} | Runs: {RUNS}")

    results = {}
    for provider, port in [("tensorrt", 9192), ("cuda", 9193), ("cpu", 9194)]:
        r = bench(provider, port)
        if r:
            results[provider] = r

    print("\n=== SUMMARY ===")
    print(f"{'Backend':<15} {'Avg(ms)':>8} {'Min(ms)':>8} {'Max(ms)':>8} {'RPS':>6}")
    for name, r in results.items():
        print(f"{name:<15} {r['avg']:8.1f} {r['min']:8.1f} {r['max']:8.1f} {r['rps']:6.0f}")

    if "tensorrt" in results and "cuda" in results:
        speedup = results["cuda"]["avg"] / results["tensorrt"]["avg"]
        print(f"\nTensorRT speedup over CUDA: {speedup:.1f}x")
    if "tensorrt" in results and "cpu" in results:
        speedup = results["cpu"]["avg"] / results["tensorrt"]["avg"]
        print(f"TensorRT speedup over CPU: {speedup:.1f}x")
