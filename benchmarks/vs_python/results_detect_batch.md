# Multi-Model Detection Benchmark: Python onnxruntime vs Go infergo

*Generated: 2026-04-09 12:40 UTC*

**Hardware:** NVIDIA GeForce RTX 5070 Ti, 16303 MiB  
**Models:** YOLOv11 n/s/m/l (all 640x640 COCO)  
**All 4 models loaded and running concurrently on GPU**  
**GPU memory cleared between Python and Go benchmarks**

---

## Summary

| Metric | Python onnxrt | Go HTTP (infergo) | Winner |
| --- | --- | --- | --- |
| Total throughput (req/s) | 155.7 | 53.0 | **Python** +66% |
| Python GPU util mean | 14% | — | — |
| Go GPU util mean | — | 12% | — |

## Python onnxruntime (4 sessions)

**Total: 155.7 req/s** | 200 ok / 0 errors | 1.28s

### Per-Model Performance

| Model | Params | req/s | Mean ms | P50 ms | P95 ms | P99 ms |
| --- | --- | --- | --- | --- | --- | --- |
| yolo11n | 2.6M | 38.9 | 123.7 | 98.9 | 280.2 | 298.9 |
| yolo11s | 9.4M | 38.9 | 83.7 | 76.8 | 140.6 | 148.6 |
| yolo11m | 20.1M | 38.9 | 100.9 | 105.9 | 161.0 | 172.9 |
| yolo11l | 25.3M | 38.9 | 86.1 | 99.1 | 108.4 | 108.5 |

### GPU Monitoring

| Metric | Mean | Max |
| --- | --- | --- |
| Utilization (%) | 13.5 | 24 |
| Memory (MB) | 2191 | 2497 / 16303 |
| Temperature (°C) | 48 | 50 |
| Power (W) | 66 | 73 / 300 |

## Go HTTP infergo (4 models)

**Total: 53.0 req/s** | 200 ok / 0 errors | 3.77s

### Per-Model Performance

| Model | Params | req/s | Mean ms | P50 ms | P95 ms | P99 ms |
| --- | --- | --- | --- | --- | --- | --- |
| yolo11n | 2.6M | 13.3 | 338.5 | 76.6 | 955.2 | 955.8 |
| yolo11s | 9.4M | 13.3 | 164.4 | 73.0 | 379.5 | 402.2 |
| yolo11m | 20.1M | 13.3 | 183.8 | 116.5 | 360.7 | 364.0 |
| yolo11l | 25.3M | 13.3 | 495.4 | 125.0 | 1351.1 | 1355.8 |

### GPU Monitoring

| Metric | Mean | Max |
| --- | --- | --- |
| Utilization (%) | 12.1 | 31 |
| Memory (MB) | 6923 | 7409 / 16303 |
| Temperature (°C) | 50 | 54 |
| Power (W) | 77 | 106 / 300 |

## Stress Test: Python onnxruntime

Concurrency ramp per model (×4 models = total concurrency). GPU cleared between levels.

| Conc/model | Total conc | Total req/s | GPU util % | GPU mem MB | GPU power W | Errors |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 221.4 | 4 | 2459 | 71 | 0 |
| 2 | 8 | 161.5 | 2 | 2481 | 68 | 0 |
| 4 | 16 | 156.1 | 17 | 2483 | 72 | 0 |
| 8 | 32 | 177.5 | 30 | 3651 | 71 | 0 |
| 16 | 64 | 164.7 | 32 | 5633 | 71 | 0 |

### Peak throughput breakdown (conc/model=1)

| Model | req/s | Mean ms | P50 ms | P95 ms |
| --- | --- | --- | --- | --- |
| yolo11n | 55.3 | 3.3 | 3.2 | 3.8 |
| yolo11s | 55.3 | 3.2 | 3.1 | 4.0 |
| yolo11m | 55.3 | 4.9 | 4.9 | 5.1 |
| yolo11l | 55.3 | 6.6 | 6.6 | 6.8 |

## Stress Test: Go HTTP infergo

Concurrency ramp per model (×4 models = total concurrency). GPU cleared between levels.

| Conc/model | Total conc | Total req/s | GPU util % | GPU mem MB | GPU power W | Errors |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 66.9 | 10 | 7409 | 74 | 0 |
| 2 | 8 | 144.4 | 4 | 7409 | 81 | 0 |
| 4 | 16 | 193.6 | 0 | 7409 | 80 | 0 |
| 8 | 32 | 219.4 | 0 | 7409 | 90 | 0 |
| 16 | 64 | 237.8 | 21 | 7413 | 83 | 0 |

### Peak throughput breakdown (conc/model=16)

| Model | req/s | Mean ms | P50 ms | P95 ms |
| --- | --- | --- | --- | --- |
| yolo11n | 59.5 | 96.5 | 106.6 | 190.4 |
| yolo11s | 59.5 | 103.6 | 87.1 | 170.5 |
| yolo11m | 59.5 | 222.4 | 229.0 | 366.4 |
| yolo11l | 59.5 | 236.5 | 256.5 | 299.0 |
