# Detection Benchmark: All Combinations

*Generated: 2026-04-09 19:37 UTC*

**Hardware:** NVIDIA GeForce RTX 5070 Ti 16GB, AMD Ryzen 9 9900X 12-Core  
**Models:** YOLOv11 n/s/m/l (640x640 COCO)  
**Image:** Synthetic 640x480 JPEG

---

## Summary — yolo11n P50 latency (c=1)

| Backend | Provider | Endpoint | P50 ms | req/s | Errors |
| --- | --- | --- | --- | --- | --- |
| python_pytorch | cuda | native | 2.5 | 395 | 0 |
| python_pytorch | cpu | native | 41.8 | 24 | 0 |
| go_http | cuda | json | 9.1 | 109 | 0 |
| go_http | cpu | json | 63.8 | 16 | 0 |
| go_onnx_cuda | cuda | json | 14.9 | 67 | 0 |
| go_tensorrt | cuda | json | 14.6 | 70 | 0 |
| go_torch_binary | cuda | binary | 7.7 | 130 | 0 |
| go_adaptive | cuda | json | 6.7 | 149 | 0 |

## Per-Model — Sequential (c=1)

| Backend | Provider | yolo11n | yolo11s | yolo11m | yolo11l |
| --- | --- | --- | --- | --- | --- |
| python_pytorch | cuda | 2.5ms | 2.9ms | 4.9ms | 6.5ms |
| python_pytorch | cpu | 41.8ms | 113.0ms | 336.2ms | 426.7ms |
| go_http | cuda | 9.1ms | 9.9ms | 11.9ms | 14.0ms |
| go_http | cpu | 63.8ms | 129.8ms | 353.4ms | 460.5ms |
| go_onnx_cuda | cuda | 14.9ms | 15.8ms | 15.8ms | 21.3ms |
| go_tensorrt | cuda | 14.6ms | 15.5ms | 19.0ms | 20.9ms |
| go_torch_binary | cuda | 7.7ms | 8.2ms | 10.4ms | 12.3ms |
| go_adaptive | cuda | 6.7ms | 8.1ms | 10.2ms | 11.6ms |

## Concurrency Scaling — yolo11n

| Backend | c=1 req/s | c=4 req/s | c=8 req/s | c=16 req/s | Errors |
| --- | --- | --- | --- | --- | --- |
| python_pytorch (cuda) | 395 | — | — | — | 0 |
| python_pytorch (cpu) | 24 | — | — | — | 0 |
| go_http (cuda) | 109 | 7 | 26 | 18 | 0 |
| go_http (cpu) | 16 | 4 | — | — | 0 |
| go_onnx_cuda (cuda) | 67 | 47 | 39 | 23 | 0 |
| go_tensorrt (cuda) | 70 | 46 | 38 | 21 | 0 |
| go_torch_binary (cuda) | 130 | 17 | 5 | 22 | 0 |
| go_adaptive (cuda) | 149 | 19 | 52 | 33 | 0 |

## Raw Data

| Backend | Provider | Endpoint | Model | Conc | Mean ms | P50 ms | P95 ms | P99 ms | req/s | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| python_pytorch | cuda | native | yolo11n | 1 | 2.5 | 2.5 | 3.1 | 3.1 | 395 | 0 |
| python_pytorch | cuda | native | yolo11s | 1 | 3.0 | 2.9 | 3.4 | 3.4 | 337 | 0 |
| python_pytorch | cuda | native | yolo11m | 1 | 4.9 | 4.9 | 5.0 | 5.0 | 205 | 0 |
| python_pytorch | cuda | native | yolo11l | 1 | 6.5 | 6.5 | 6.6 | 6.7 | 155 | 0 |
| python_pytorch | cpu | native | yolo11n | 1 | 41.4 | 41.8 | 43.9 | 44.1 | 24 | 0 |
| python_pytorch | cpu | native | yolo11s | 1 | 113.7 | 113.0 | 119.3 | 121.8 | 9 | 0 |
| python_pytorch | cpu | native | yolo11m | 1 | 337.4 | 336.2 | 348.8 | 361.4 | 3 | 0 |
| python_pytorch | cpu | native | yolo11l | 1 | 425.9 | 426.7 | 432.4 | 434.6 | 2 | 0 |
| go_http | cuda | json | yolo11n | 1 | 9.2 | 9.1 | 10.0 | 10.3 | 109 | 0 |
| go_http | cuda | json | yolo11n | 4 | 135.5 | 30.0 | 558.3 | 559.2 | 7 | 0 |
| go_http | cuda | json | yolo11n | 8 | 38.0 | 34.7 | 50.5 | 51.4 | 26 | 0 |
| go_http | cuda | json | yolo11n | 16 | 54.9 | 53.0 | 73.7 | 73.9 | 18 | 0 |
| go_http | cuda | json | yolo11s | 1 | 10.1 | 9.9 | 10.6 | 12.2 | 99 | 0 |
| go_http | cuda | json | yolo11s | 4 | 65.7 | 25.3 | 324.0 | 325.6 | 15 | 0 |
| go_http | cuda | json | yolo11s | 8 | 194.3 | 42.3 | 627.6 | 627.9 | 5 | 0 |
| go_http | cuda | json | yolo11s | 16 | 66.1 | 67.6 | 93.8 | 94.9 | 15 | 0 |
| go_http | cuda | json | yolo11m | 1 | 12.0 | 11.9 | 12.8 | 13.0 | 84 | 0 |
| go_http | cuda | json | yolo11m | 4 | 68.1 | 32.8 | 293.6 | 294.1 | 15 | 0 |
| go_http | cuda | json | yolo11m | 8 | 196.7 | 58.2 | 593.9 | 595.4 | 5 | 0 |
| go_http | cuda | json | yolo11m | 16 | 89.2 | 99.9 | 125.2 | 127.8 | 11 | 0 |
| go_http | cuda | json | yolo11l | 1 | 14.0 | 14.0 | 14.5 | 14.7 | 72 | 0 |
| go_http | cuda | json | yolo11l | 4 | 96.5 | 37.1 | 478.9 | 479.4 | 10 | 0 |
| go_http | cuda | json | yolo11l | 8 | 276.0 | 123.7 | 816.2 | 816.8 | 4 | 0 |
| go_http | cuda | json | yolo11l | 16 | 102.5 | 117.6 | 129.8 | 135.1 | 10 | 0 |
| go_http | cpu | json | yolo11n | 1 | 63.5 | 63.8 | 66.5 | 69.0 | 16 | 0 |
| go_http | cpu | json | yolo11n | 4 | 271.4 | 283.1 | 300.7 | 301.0 | 4 | 0 |
| go_http | cpu | json | yolo11s | 1 | 130.2 | 129.8 | 134.4 | 135.5 | 8 | 0 |
| go_http | cpu | json | yolo11s | 4 | 671.5 | 691.2 | 725.5 | 726.1 | 1 | 0 |
| go_http | cpu | json | yolo11m | 1 | 355.1 | 353.4 | 368.4 | 369.5 | 3 | 0 |
| go_http | cpu | json | yolo11m | 4 | 1769.1 | 1820.7 | 1940.8 | 1941.7 | 1 | 0 |
| go_http | cpu | json | yolo11l | 1 | 466.7 | 460.5 | 521.1 | 528.0 | 2 | 0 |
| go_http | cpu | json | yolo11l | 4 | 2148.4 | 2225.6 | 2294.8 | 2295.7 | 0 | 0 |
| go_onnx_cuda | cuda | json | yolo11n | 1 | 14.8 | 14.9 | 15.8 | 16.0 | 67 | 0 |
| go_onnx_cuda | cuda | json | yolo11n | 4 | 21.2 | 18.7 | 40.4 | 47.9 | 47 | 0 |
| go_onnx_cuda | cuda | json | yolo11n | 8 | 25.6 | 26.2 | 32.2 | 32.9 | 39 | 0 |
| go_onnx_cuda | cuda | json | yolo11n | 16 | 42.8 | 39.9 | 66.9 | 71.6 | 23 | 0 |
| go_onnx_cuda | cuda | json | yolo11s | 1 | 15.7 | 15.8 | 16.6 | 17.1 | 64 | 0 |
| go_onnx_cuda | cuda | json | yolo11s | 4 | 22.0 | 20.5 | 36.4 | 39.2 | 45 | 0 |
| go_onnx_cuda | cuda | json | yolo11s | 8 | 28.3 | 27.2 | 42.3 | 48.5 | 35 | 0 |
| go_onnx_cuda | cuda | json | yolo11s | 16 | 52.8 | 52.1 | 74.7 | 76.4 | 19 | 0 |
| go_onnx_cuda | cuda | json | yolo11m | 1 | 16.8 | 15.8 | 19.4 | 19.4 | 60 | 0 |
| go_onnx_cuda | cuda | json | yolo11m | 4 | 18.4 | 17.8 | 33.0 | 39.8 | 54 | 20 |
| go_onnx_cuda | cuda | json | yolo11m | 8 | 20.1 | 20.0 | 29.9 | 33.4 | 50 | 30 |
| go_onnx_cuda | cuda | json | yolo11m | 16 | 28.5 | 29.6 | 41.0 | 42.4 | 35 | 30 |
| go_onnx_cuda | cuda | json | yolo11l | 1 | 20.7 | 21.3 | 23.6 | 24.3 | 48 | 0 |
| go_onnx_cuda | cuda | json | yolo11l | 4 | 18.6 | 15.7 | 36.3 | 41.8 | 54 | 20 |
| go_onnx_cuda | cuda | json | yolo11l | 8 | 21.6 | 20.6 | 34.3 | 41.8 | 46 | 26 |
| go_onnx_cuda | cuda | json | yolo11l | 16 | 28.6 | 27.3 | 40.9 | 52.4 | 35 | 28 |
| go_tensorrt | cuda | json | yolo11n | 1 | 14.2 | 14.6 | 15.5 | 15.8 | 70 | 0 |
| go_tensorrt | cuda | json | yolo11n | 4 | 22.0 | 20.9 | 34.7 | 36.4 | 46 | 0 |
| go_tensorrt | cuda | json | yolo11n | 8 | 26.6 | 26.8 | 40.4 | 41.0 | 38 | 0 |
| go_tensorrt | cuda | json | yolo11n | 16 | 46.5 | 44.0 | 63.3 | 69.5 | 21 | 0 |
| go_tensorrt | cuda | json | yolo11s | 1 | 15.2 | 15.5 | 16.7 | 17.0 | 66 | 0 |
| go_tensorrt | cuda | json | yolo11s | 4 | 21.7 | 19.5 | 39.2 | 40.0 | 46 | 0 |
| go_tensorrt | cuda | json | yolo11s | 8 | 29.4 | 30.4 | 38.1 | 43.5 | 34 | 0 |
| go_tensorrt | cuda | json | yolo11s | 16 | 54.2 | 52.2 | 79.3 | 81.3 | 18 | 0 |
| go_tensorrt | cuda | json | yolo11m | 1 | 17.9 | 19.0 | 20.2 | 20.3 | 56 | 0 |
| go_tensorrt | cuda | json | yolo11m | 4 | 18.1 | 16.8 | 35.0 | 35.3 | 55 | 29 |
| go_tensorrt | cuda | json | yolo11m | 8 | 18.8 | 20.3 | 24.9 | 31.1 | 53 | 30 |
| go_tensorrt | cuda | json | yolo11m | 16 | 27.1 | 25.7 | 42.9 | 44.7 | 37 | 30 |
| go_tensorrt | cuda | json | yolo11l | 1 | 19.9 | 20.9 | 21.9 | 21.9 | 50 | 0 |
| go_tensorrt | cuda | json | yolo11l | 4 | 17.7 | 15.3 | 36.3 | 42.6 | 57 | 21 |
| go_tensorrt | cuda | json | yolo11l | 8 | 17.6 | 16.6 | 27.8 | 30.9 | 57 | 25 |
| go_tensorrt | cuda | json | yolo11l | 16 | 29.4 | 31.3 | 44.3 | 46.5 | 34 | 27 |
| go_torch_binary | cuda | binary | yolo11n | 1 | 7.7 | 7.7 | 8.2 | 9.4 | 130 | 0 |
| go_torch_binary | cuda | binary | yolo11n | 4 | 58.8 | 19.2 | 305.3 | 305.7 | 17 | 0 |
| go_torch_binary | cuda | binary | yolo11n | 8 | 202.2 | 109.1 | 568.3 | 568.4 | 5 | 0 |
| go_torch_binary | cuda | binary | yolo11n | 16 | 46.0 | 50.8 | 66.6 | 66.7 | 22 | 0 |
| go_torch_binary | cuda | binary | yolo11s | 1 | 8.2 | 8.2 | 8.7 | 8.9 | 122 | 0 |
| go_torch_binary | cuda | binary | yolo11s | 4 | 62.4 | 22.7 | 313.1 | 313.6 | 16 | 0 |
| go_torch_binary | cuda | binary | yolo11s | 8 | 190.1 | 125.4 | 524.9 | 525.6 | 5 | 0 |
| go_torch_binary | cuda | binary | yolo11s | 16 | 64.4 | 65.0 | 93.6 | 93.6 | 16 | 0 |
| go_torch_binary | cuda | binary | yolo11m | 1 | 10.4 | 10.4 | 10.7 | 10.8 | 96 | 0 |
| go_torch_binary | cuda | binary | yolo11m | 4 | 65.4 | 30.4 | 284.7 | 286.0 | 15 | 0 |
| go_torch_binary | cuda | binary | yolo11m | 8 | 142.5 | 56.4 | 516.4 | 516.8 | 7 | 0 |
| go_torch_binary | cuda | binary | yolo11m | 16 | 131.2 | 158.2 | 209.9 | 210.8 | 8 | 0 |
| go_torch_binary | cuda | binary | yolo11l | 1 | 12.3 | 12.3 | 12.8 | 12.9 | 81 | 0 |
| go_torch_binary | cuda | binary | yolo11l | 4 | 96.3 | 35.0 | 490.4 | 490.8 | 10 | 0 |
| go_torch_binary | cuda | binary | yolo11l | 8 | 279.6 | 111.2 | 831.2 | 832.2 | 4 | 0 |
| go_torch_binary | cuda | binary | yolo11l | 16 | 106.0 | 119.0 | 165.7 | 166.3 | 9 | 0 |
| go_adaptive | cuda | json | yolo11n | 1 | 6.7 | 6.7 | 7.1 | 7.3 | 149 | 0 |
| go_adaptive | cuda | json | yolo11n | 4 | 52.5 | 12.0 | 402.6 | 407.0 | 19 | 0 |
| go_adaptive | cuda | json | yolo11n | 8 | 19.1 | 18.5 | 25.7 | 28.3 | 52 | 0 |
| go_adaptive | cuda | json | yolo11n | 16 | 29.9 | 28.9 | 41.9 | 43.2 | 33 | 0 |
| go_adaptive | cuda | json | yolo11s | 1 | 8.1 | 8.1 | 8.6 | 8.7 | 123 | 0 |
| go_adaptive | cuda | json | yolo11s | 4 | 55.5 | 10.2 | 435.4 | 440.9 | 18 | 0 |
| go_adaptive | cuda | json | yolo11s | 8 | 47.4 | 16.9 | 320.5 | 321.6 | 21 | 0 |
| go_adaptive | cuda | json | yolo11s | 16 | 30.0 | 30.8 | 41.4 | 49.4 | 33 | 0 |
| go_adaptive | cuda | json | yolo11m | 1 | 10.1 | 10.2 | 10.6 | 10.7 | 99 | 0 |
| go_adaptive | cuda | json | yolo11m | 4 | 59.6 | 12.0 | 464.2 | 464.5 | 17 | 0 |
| go_adaptive | cuda | json | yolo11m | 8 | 28.6 | 27.9 | 39.7 | 40.1 | 35 | 0 |
| go_adaptive | cuda | json | yolo11m | 16 | 47.1 | 51.8 | 60.2 | 62.4 | 21 | 0 |
| go_adaptive | cuda | json | yolo11l | 1 | 11.7 | 11.6 | 12.3 | 12.6 | 86 | 0 |
| go_adaptive | cuda | json | yolo11l | 4 | 102.0 | 12.9 | 874.1 | 874.2 | 10 | 0 |
| go_adaptive | cuda | json | yolo11l | 8 | 34.6 | 36.0 | 41.1 | 45.0 | 29 | 0 |
| go_adaptive | cuda | json | yolo11l | 16 | 56.9 | 65.1 | 73.1 | 77.8 | 18 | 0 |
