# Multi-Model Detection Benchmark: Python onnxruntime vs Go infergo

*Generated: 2026-04-09 06:32 UTC*

**Hardware:** NVIDIA GeForce RTX 5070 Ti, 16303 MiB  
**Models:** YOLOv11 n/s/m/l (all 640x640 COCO)  
**All 4 models loaded and running concurrently on GPU**  
**GPU memory cleared between Python and Go benchmarks**

---

## Summary

| Metric | Python onnxrt | Go HTTP (infergo) | Winner |
| --- | --- | --- | --- |
| Total throughput (req/s) | 146.7 | 295.3 | **Go** +50% |
| Python GPU util mean | 23% | — | — |
| Go GPU util mean | — | 0% | — |

## Python onnxruntime (4 sessions)

**Total: 146.7 req/s** | 200 ok / 0 errors | 1.36s

### Per-Model Performance

| Model | Params | req/s | Mean ms | P50 ms | P95 ms | P99 ms |
| --- | --- | --- | --- | --- | --- | --- |
| yolo11n | 2.6M | 36.7 | 127.1 | 99.9 | 280.1 | 298.9 |
| yolo11s | 9.4M | 36.7 | 89.4 | 82.3 | 154.6 | 158.3 |
| yolo11m | 20.1M | 36.7 | 108.3 | 113.4 | 171.2 | 177.8 |
| yolo11l | 25.3M | 36.7 | 93.4 | 109.7 | 117.4 | 118.0 |

### GPU Monitoring

| Metric | Mean | Max |
| --- | --- | --- |
| Utilization (%) | 23.4 | 41 |
| Memory (MB) | 3128 | 3379 / 16303 |
| Temperature (°C) | 49 | 52 |
| Power (W) | 84 | 117 / 300 |

## Go HTTP infergo (4 models)

**Total: 295.3 req/s** | 200 ok / 0 errors | 0.68s

### Per-Model Performance

| Model | Params | req/s | Mean ms | P50 ms | P95 ms | P99 ms |
| --- | --- | --- | --- | --- | --- | --- |
| yolo11n | 2.6M | 73.8 | 37.5 | 38.2 | 50.2 | 50.9 |
| yolo11s | 9.4M | 73.8 | 37.5 | 37.6 | 42.6 | 46.0 |
| yolo11m | 20.1M | 73.8 | 54.7 | 56.0 | 61.7 | 64.4 |
| yolo11l | 25.3M | 73.8 | 72.5 | 73.0 | 77.7 | 78.8 |

### GPU Monitoring

| Metric | Mean | Max |
| --- | --- | --- |
| Utilization (%) | 0.0 | 0 |
| Memory (MB) | 6818 | 6919 / 16303 |
| Temperature (°C) | 51 | 57 |
| Power (W) | 77 | 108 / 300 |

## Stress Test: Python onnxruntime

Concurrency ramp per model (×4 models = total concurrency). GPU cleared between levels.

| Conc/model | Total conc | Total req/s | GPU util % | GPU mem MB | GPU power W | Errors |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 222.7 | 0 | 3355 | 82 | 0 |
| 2 | 8 | 167.4 | 36 | 3351 | 80 | 0 |
| 4 | 16 | 152.5 | 6 | 3379 | 78 | 0 |
| 8 | 32 | 181.8 | 6 | 3933 | 79 | 0 |
| 16 | 64 | 161.3 | 17 | 6629 | 88 | 0 |

### Peak throughput breakdown (conc/model=1)

| Model | req/s | Mean ms | P50 ms | P95 ms |
| --- | --- | --- | --- | --- |
| yolo11n | 55.7 | 3.0 | 3.0 | 3.6 |
| yolo11s | 55.7 | 3.1 | 3.1 | 3.4 |
| yolo11m | 55.7 | 5.0 | 5.0 | 5.2 |
| yolo11l | 55.7 | 6.8 | 6.8 | 6.9 |

## Stress Test: Go HTTP infergo

Concurrency ramp per model (×4 models = total concurrency). GPU cleared between levels.

| Conc/model | Total conc | Total req/s | GPU util % | GPU mem MB | GPU power W | Errors |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 255.9 | 0 | 6919 | 71 | 0 |
| 2 | 8 | 297.5 | 0 | 6919 | 80 | 0 |
| 4 | 16 | 301.9 | 5 | 6919 | 77 | 0 |
| 8 | 32 | 295.1 | 5 | 6919 | 74 | 0 |
| 16 | 64 | 285.4 | 95 | 6919 | 79 | 0 |

### Peak throughput breakdown (conc/model=4)

| Model | req/s | Mean ms | P50 ms | P95 ms |
| --- | --- | --- | --- | --- |
| yolo11n | 75.5 | 34.7 | 35.5 | 46.4 |
| yolo11s | 75.5 | 36.2 | 35.6 | 43.8 |
| yolo11m | 75.5 | 50.9 | 53.9 | 59.5 |
| yolo11l | 75.5 | 66.7 | 70.3 | 72.7 |
