# Detection Benchmark: Python Native Bindings vs Native Go HTTP

*Generated: 2026-04-09 04:31 UTC*

**Hardware:** NVIDIA GeForce RTX 5070 Ti 16GB VRAM, AMD Ryzen 9 9900X 12-Core  
**Model:** YOLOv11n (640×640 input, 80 COCO classes, 2.6M params, 6.5 GFLOPs)  
**Image:** Synthetic 640×480 JPEG  
**GPU memory cleared between every benchmark run**

---

## Head-to-Head Comparison

| Metric | Python Native | Go HTTP | Winner |
| --- | --- | --- | --- |
| Throughput (req/s) | 45.6 | 72.3 | **Go** +37% |
| Mean latency (ms) | 86.3 | 53.9 | **Go** +38% |
| P50 latency (ms) | 83.4 | 53.7 | **Go** +36% |
| P95 latency (ms) | 118.0 | 72.5 | **Go** +39% |
| P99 latency (ms) | 146.3 | 77.2 | **Go** +47% |
| Cold start (ms) | 1605.9 | 29723.6 | **Python** +95% |

## python_onnxrt (concurrency=4, provider=cuda)

### Throughput & Latency

| Metric | Value |
| --- | --- |
| Throughput | **45.6 req/s** |
| Requests | 100 ok / 0 errors |
| Duration | 2.19s |
| Cold start | 1605.9ms |
| Mean latency | 86.3ms |
| P50 | 83.4ms |
| P95 | 118.0ms |
| P99 | 146.3ms |
| Min | 41.8ms |
| Max | 153.0ms |
| Stdev | 20.6ms |

### Per-Stage Latency (ms)

| Stage | Mean | P50 | P95 |
| --- | --- | --- | --- |
| decode | 5.67 | 5.24 | 11.09 |
| letterbox | 0.36 | 0.13 | 1.24 |
| normalize | 1.05 | 0.53 | 3.89 |
| stack | 0.66 | 0.37 | 1.39 |
| inference | 77.11 | 73.83 | 112.29 |
| postprocess | 1.46 | 0.89 | 4.84 |

### GPU Monitoring

| Metric | Mean | Max |
| --- | --- | --- |
| Utilization (%) | 12.4 | 25 |
| Memory used (MB) | 879 | 905 / 16303 |
| Temperature (°C) | 42 | 42 |
| Power (W) | 59 | 63 / 300 |
| Samples collected | 10 | — |

### Memory (Client Process)

| RSS before | RSS after | Delta |
| --- | --- | --- |
| 1069 MB | 1196 MB | +127 MB |

## go_http (concurrency=4, provider=cuda)

### Throughput & Latency

| Metric | Value |
| --- | --- |
| Throughput | **72.3 req/s** |
| Requests | 100 ok / 0 errors |
| Duration | 1.38s |
| Cold start | 29723.6ms |
| Mean latency | 53.9ms |
| P50 | 53.7ms |
| P95 | 72.5ms |
| P99 | 77.2ms |
| Min | 37.0ms |
| Max | 82.2ms |
| Stdev | 9.4ms |

### Per-Stage Latency (ms)

| Stage | Mean | P50 | P95 |
| --- | --- | --- | --- |
| inference | 53.87 | 53.64 | 72.52 |

### GPU Monitoring

| Metric | Mean | Max |
| --- | --- | --- |
| Utilization (%) | 3.4 | 12 |
| Memory used (MB) | 1432 | 1465 / 16303 |
| Temperature (°C) | 45 | 45 |
| Power (W) | 56 | 62 / 300 |
| Samples collected | 7 | — |

### Memory (Client Process)

| RSS before | RSS after | Delta |
| --- | --- | --- |
| 1567 MB | 1571 MB | +4 MB |

## Stress Test: python_onnxrt

Concurrency ramp to find saturation point. GPU memory cleared between each level.

| Concurrency | req/s | Mean ms | P50 ms | P95 ms | P99 ms | GPU util % | GPU mem MB | GPU power W | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 25.9 | 38.4 | 33.8 | 69.7 | 76.3 | 10 | 663 | 56 | 0 |
| 2 | 34.7 | 57.0 | 52.4 | 90.3 | 101.7 | 13 | 743 | 55 | 0 |
| 4 | 48.3 | 79.8 | 76.4 | 113.7 | 119.7 | 5 | 905 | 54 | 0 |
| 8 | 47.7 | 160.2 | 148.8 | 293.5 | 344.0 | 4 | 1297 | 54 | 0 |
| 16 | 38.1 | 389.5 | 365.2 | 627.8 | 829.7 | 10 | 2083 | 56 | 0 |
| 32 | 38.3 | 679.8 | 790.6 | 1042.7 | 1164.3 | 11 | 2631 | 56 | 0 |

## Stress Test: go_http

Concurrency ramp to find saturation point. GPU memory cleared between each level.

| Concurrency | req/s | Mean ms | P50 ms | P95 ms | P99 ms | GPU util % | GPU mem MB | GPU power W | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32.3 | 30.9 | 31.8 | 36.2 | 41.9 | 3 | 1465 | 54 | 0 |
| 2 | 49.8 | 39.5 | 39.2 | 48.7 | 52.0 | 5 | 1465 | 54 | 0 |
| 4 | 74.8 | 52.6 | 50.7 | 70.9 | 80.2 | 3 | 1465 | 54 | 0 |
| 8 | 101.4 | 74.7 | 73.7 | 90.5 | 96.6 | 9 | 1857 | 53 | 0 |
| 16 | 128.6 | 108.3 | 109.3 | 148.5 | 150.7 | 0 | 2525 | 51 | 0 |
| 32 | 135.8 | 175.7 | 193.6 | 266.6 | 320.9 | 0 | 2525 | 53 | 0 |
