# Multi-Model Detection Benchmark: Python onnxruntime vs Go infergo

*Generated: 2026-04-09 11:45 UTC*

**Hardware:** NVIDIA GeForce RTX 5070 Ti, 16303 MiB  
**Models:** YOLOv11 n/s/m/l (all 640x640 COCO)  
**All 4 models loaded and running concurrently on GPU**  
**GPU memory cleared between Python and Go benchmarks**

---

## Summary

| Metric | Python onnxrt | Go HTTP (infergo) | Winner |
| --- | --- | --- | --- |
| Total throughput (req/s) | 147.7 | 183.0 | **Go** +19% |
| Python GPU util mean | 43% | — | — |
| Go GPU util mean | — | 18% | — |

## Python onnxruntime (4 sessions)

**Total: 147.7 req/s** | 200 ok / 0 errors | 1.35s

### Per-Model Performance

| Model | Params | req/s | Mean ms | P50 ms | P95 ms | P99 ms |
| --- | --- | --- | --- | --- | --- | --- |
| yolo11n | 2.6M | 36.9 | 121.8 | 92.7 | 274.1 | 292.4 |
| yolo11s | 9.4M | 36.9 | 93.8 | 87.2 | 153.0 | 160.1 |
| yolo11m | 20.1M | 36.9 | 104.6 | 108.9 | 174.6 | 238.1 |
| yolo11l | 25.3M | 36.9 | 95.5 | 110.3 | 115.5 | 117.0 |

### GPU Monitoring

| Metric | Mean | Max |
| --- | --- | --- |
| Utilization (%) | 43.4 | 72 |
| Memory (MB) | 2170 | 2427 / 16303 |
| Temperature (°C) | 45 | 50 |
| Power (W) | 69 | 107 / 300 |

## Go HTTP infergo (4 models)

**Total: 183.0 req/s** | 200 ok / 0 errors | 1.09s

### Per-Model Performance

| Model | Params | req/s | Mean ms | P50 ms | P95 ms | P99 ms |
| --- | --- | --- | --- | --- | --- | --- |
| yolo11n | 2.6M | 45.8 | 51.4 | 40.8 | 100.4 | 112.7 |
| yolo11s | 9.4M | 45.8 | 52.6 | 58.7 | 72.2 | 82.0 |
| yolo11m | 20.1M | 45.8 | 102.8 | 108.8 | 120.1 | 123.6 |
| yolo11l | 25.3M | 45.8 | 126.2 | 131.1 | 149.3 | 154.8 |

### GPU Monitoring

| Metric | Mean | Max |
| --- | --- | --- |
| Utilization (%) | 18.0 | 90 |
| Memory (MB) | 7124 | 7611 / 16303 |
| Temperature (°C) | 48 | 54 |
| Power (W) | 80 | 134 / 300 |

## Stress Test: Python onnxruntime

Concurrency ramp per model (×4 models = total concurrency). GPU cleared between levels.

| Conc/model | Total conc | Total req/s | GPU util % | GPU mem MB | GPU power W | Errors |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 208.1 | 47 | 2395 | 75 | 0 |
| 2 | 8 | 168.0 | 22 | 2417 | 76 | 0 |
| 4 | 16 | 152.9 | 8 | 2445 | 74 | 0 |
| 8 | 32 | 176.5 | 0 | 3545 | 73 | 0 |
| 16 | 64 | 161.1 | 16 | 5575 | 78 | 0 |

### Peak throughput breakdown (conc/model=1)

| Model | req/s | Mean ms | P50 ms | P95 ms |
| --- | --- | --- | --- | --- |
| yolo11n | 52.0 | 3.4 | 3.4 | 3.9 |
| yolo11s | 52.0 | 3.7 | 3.6 | 4.4 |
| yolo11m | 52.0 | 5.2 | 5.2 | 5.6 |
| yolo11l | 52.0 | 6.9 | 6.9 | 7.2 |

## Stress Test: Go HTTP infergo

Concurrency ramp per model (×4 models = total concurrency). GPU cleared between levels.

| Conc/model | Total conc | Total req/s | GPU util % | GPU mem MB | GPU power W | Errors |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 160.7 | 16 | 7611 | 89 | 0 |
| 2 | 8 | 175.5 | 0 | 7611 | 82 | 0 |
| 4 | 16 | 191.1 | 0 | 7679 | 80 | 0 |
| 8 | 32 | 213.2 | 47 | 8111 | 88 | 0 |
| 16 | 64 | 196.5 | 0 | 9357 | 85 | 0 |

### Peak throughput breakdown (conc/model=8)

| Model | req/s | Mean ms | P50 ms | P95 ms |
| --- | --- | --- | --- | --- |
| yolo11n | 53.3 | 64.8 | 52.7 | 197.5 |
| yolo11s | 53.3 | 128.2 | 123.0 | 196.4 |
| yolo11m | 53.3 | 161.3 | 148.0 | 247.3 |
| yolo11l | 53.3 | 194.5 | 218.1 | 289.1 |
