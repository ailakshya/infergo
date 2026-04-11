# Multi-Model Detection Benchmark: Python onnxruntime vs Go infergo

*Generated: 2026-04-09 17:02 UTC*

**Hardware:** NVIDIA GeForce RTX 5070 Ti, 16303 MiB  
**Models:** YOLOv11 n/s/m/l (all 640x640 COCO)  
**All 4 models loaded and running concurrently on GPU**  
**GPU memory cleared between Python and Go benchmarks**

---

## Summary

| Metric | Python onnxrt | Go HTTP (infergo) | Winner |
| --- | --- | --- | --- |
| Total throughput (req/s) | 157.0 | 153.6 | **Python** +2% |
| Python GPU util mean | 24% | — | — |
| Go GPU util mean | — | 27% | — |

## Python onnxruntime (4 sessions)

**Total: 157.0 req/s** | 200 ok / 0 errors | 1.27s

### Per-Model Performance

| Model | Params | req/s | Mean ms | P50 ms | P95 ms | P99 ms |
| --- | --- | --- | --- | --- | --- | --- |
| yolo11n | 2.6M | 39.2 | 119.5 | 92.9 | 272.2 | 290.5 |
| yolo11s | 9.4M | 39.2 | 83.6 | 76.9 | 145.6 | 145.8 |
| yolo11m | 20.1M | 39.2 | 101.5 | 105.0 | 173.9 | 253.8 |
| yolo11l | 25.3M | 39.2 | 86.3 | 99.3 | 108.2 | 108.3 |

### GPU Monitoring

| Metric | Mean | Max |
| --- | --- | --- |
| Utilization (%) | 24.0 | 43 |
| Memory (MB) | 3730 | 4015 / 16303 |
| Temperature (°C) | 50 | 51 |
| Power (W) | 71 | 112 / 300 |

## Go HTTP infergo (4 models)

**Total: 153.6 req/s** | 200 ok / 0 errors | 1.30s

### Per-Model Performance

| Model | Params | req/s | Mean ms | P50 ms | P95 ms | P99 ms |
| --- | --- | --- | --- | --- | --- | --- |
| yolo11n | 2.6M | 38.4 | 39.3 | 34.5 | 56.5 | 114.8 |
| yolo11s | 9.4M | 38.4 | 39.9 | 36.2 | 55.3 | 103.2 |
| yolo11m | 20.1M | 38.4 | 60.1 | 60.4 | 71.1 | 77.0 |
| yolo11l | 25.3M | 38.4 | 86.3 | 69.8 | 83.2 | 880.4 |

### GPU Monitoring

| Metric | Mean | Max |
| --- | --- | --- |
| Utilization (%) | 26.7 | 30 |
| Memory (MB) | 8155 | 8327 / 16303 |
| Temperature (°C) | 55 | 62 |
| Power (W) | 111 | 148 / 300 |

## Stress Test: Python onnxruntime

Concurrency ramp per model (×4 models = total concurrency). GPU cleared between levels.

| Conc/model | Total conc | Total req/s | GPU util % | GPU mem MB | GPU power W | Errors |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 224.6 | 4 | 3991 | 81 | 0 |
| 2 | 8 | 168.0 | 4 | 4015 | 81 | 0 |
| 4 | 16 | 151.0 | 8 | 4015 | 78 | 0 |
| 8 | 32 | 180.8 | 19 | 4599 | 79 | 0 |
| 16 | 64 | 164.1 | 0 | 7307 | 89 | 0 |

### Peak throughput breakdown (conc/model=1)

| Model | req/s | Mean ms | P50 ms | P95 ms |
| --- | --- | --- | --- | --- |
| yolo11n | 56.2 | 3.1 | 3.1 | 3.7 |
| yolo11s | 56.2 | 3.2 | 3.0 | 3.8 |
| yolo11m | 56.2 | 4.9 | 4.9 | 5.1 |
| yolo11l | 56.2 | 6.6 | 6.6 | 6.8 |

## Stress Test: Go HTTP infergo

Concurrency ramp per model (×4 models = total concurrency). GPU cleared between levels.

| Conc/model | Total conc | Total req/s | GPU util % | GPU mem MB | GPU power W | Errors |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 83.1 | 17 | 8481 | 83 | 0 |
| 2 | 8 | 120.7 | 0 | 8675 | 96 | 0 |
| 4 | 16 | 303.6 | 0 | 8691 | 78 | 0 |
| 8 | 32 | 293.4 | 0 | 8763 | 81 | 0 |
| 16 | 64 | 294.6 | 0 | 8847 | 82 | 0 |

### Peak throughput breakdown (conc/model=4)

| Model | req/s | Mean ms | P50 ms | P95 ms |
| --- | --- | --- | --- | --- |
| yolo11n | 75.9 | 34.6 | 33.9 | 46.1 |
| yolo11s | 75.9 | 37.7 | 37.1 | 52.0 |
| yolo11m | 75.9 | 52.6 | 52.0 | 70.1 |
| yolo11l | 75.9 | 62.7 | 68.6 | 73.1 |
