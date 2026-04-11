# Multi-Model Detection Benchmark: Python onnxruntime vs Go infergo

*Generated: 2026-04-09 11:20 UTC*

**Hardware:** NVIDIA GeForce RTX 5070 Ti, 16303 MiB  
**Models:** YOLOv11 n/s/m/l (all 640x640 COCO)  
**All 4 models loaded and running concurrently on GPU**  
**GPU memory cleared between Python and Go benchmarks**

---

## Summary

| Metric | Python onnxrt | Go HTTP (infergo) | Winner |
| --- | --- | --- | --- |
| Total throughput (req/s) | 154.1 | 180.1 | **Go** +14% |
| Python GPU util mean | 22% | — | — |
| Go GPU util mean | — | 1% | — |

## Python onnxruntime (4 sessions)

**Total: 154.1 req/s** | 200 ok / 0 errors | 1.30s

### Per-Model Performance

| Model | Params | req/s | Mean ms | P50 ms | P95 ms | P99 ms |
| --- | --- | --- | --- | --- | --- | --- |
| yolo11n | 2.6M | 38.5 | 120.9 | 92.0 | 272.6 | 289.7 |
| yolo11s | 9.4M | 38.5 | 84.9 | 77.2 | 148.2 | 201.4 |
| yolo11m | 20.1M | 38.5 | 103.8 | 106.4 | 172.9 | 248.6 |
| yolo11l | 25.3M | 38.5 | 88.6 | 101.8 | 110.8 | 110.8 |

### GPU Monitoring

| Metric | Mean | Max |
| --- | --- | --- |
| Utilization (%) | 21.5 | 39 |
| Memory (MB) | 2133 | 2431 / 16303 |
| Temperature (°C) | 45 | 46 |
| Power (W) | 73 | 108 / 300 |

## Go HTTP infergo (4 models)

**Total: 180.1 req/s** | 200 ok / 0 errors | 1.11s

### Per-Model Performance

| Model | Params | req/s | Mean ms | P50 ms | P95 ms | P99 ms |
| --- | --- | --- | --- | --- | --- | --- |
| yolo11n | 2.6M | 45.0 | 64.9 | 63.3 | 106.2 | 114.2 |
| yolo11s | 9.4M | 45.0 | 62.3 | 62.0 | 84.3 | 132.4 |
| yolo11m | 20.1M | 45.0 | 99.2 | 97.0 | 139.5 | 171.7 |
| yolo11l | 25.3M | 45.0 | 116.9 | 120.6 | 161.0 | 202.1 |

### GPU Monitoring

| Metric | Mean | Max |
| --- | --- | --- |
| Utilization (%) | 1.0 | 1 |
| Memory (MB) | 6947 | 7359 / 16303 |
| Temperature (°C) | 45 | 45 |
| Power (W) | 85 | 123 / 300 |

## Stress Test: Python onnxruntime

Concurrency ramp per model (×4 models = total concurrency). GPU cleared between levels.

| Conc/model | Total conc | Total req/s | GPU util % | GPU mem MB | GPU power W | Errors |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 220.9 | 4 | 2395 | 82 | 0 |
| 2 | 8 | 166.1 | 17 | 2439 | 77 | 0 |
| 4 | 16 | 143.3 | 0 | 2439 | 75 | 0 |
| 8 | 32 | 170.4 | 0 | 3587 | 84 | 0 |
| 16 | 64 | 161.8 | 48 | 5575 | 86 | 0 |

### Peak throughput breakdown (conc/model=1)

| Model | req/s | Mean ms | P50 ms | P95 ms |
| --- | --- | --- | --- | --- |
| yolo11n | 55.2 | 3.0 | 2.9 | 3.7 |
| yolo11s | 55.2 | 3.2 | 3.1 | 3.8 |
| yolo11m | 55.2 | 5.1 | 5.1 | 5.2 |
| yolo11l | 55.2 | 6.8 | 6.8 | 7.0 |

## Stress Test: Go HTTP infergo

Concurrency ramp per model (×4 models = total concurrency). GPU cleared between levels.

| Conc/model | Total conc | Total req/s | GPU util % | GPU mem MB | GPU power W | Errors |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 167.6 | 1 | 7405 | 86 | 0 |
| 2 | 8 | 187.9 | 76 | 7405 | 80 | 0 |
| 4 | 16 | 187.3 | 69 | 7455 | 81 | 0 |
| 8 | 32 | 165.5 | 41 | 8221 | 93 | 0 |
| 16 | 64 | 154.2 | 0 | 9333 | 91 | 0 |

### Peak throughput breakdown (conc/model=2)

| Model | req/s | Mean ms | P50 ms | P95 ms |
| --- | --- | --- | --- | --- |
| yolo11n | 47.0 | 29.3 | 29.7 | 38.5 |
| yolo11s | 47.0 | 31.2 | 32.8 | 41.9 |
| yolo11m | 47.0 | 47.9 | 52.8 | 65.0 |
| yolo11l | 47.0 | 57.5 | 62.7 | 73.1 |
