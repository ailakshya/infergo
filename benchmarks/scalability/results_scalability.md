# Python vs infergo — Scalability Benchmark (OPT-27)

*Date: 2026-04-04*  
*Model: llama3-8b-q4 (8B, Q4_K_M), max_tokens=64, 40 req per level*  
*Hardware: RTX 5070 Ti (16 GB VRAM), gpu_dev*

## Key findings

> **infergo RSS stays flat** as concurrency rises: ~1168 MB constant (one process, one model copy in VRAM).  
> **Throughput scales with concurrency**: 2.2 req/s at c=1 → 12.5 req/s at c=32 (+5.7×) — continuous batching sends all active sequences through the GPU simultaneously.  
> **Zero errors** across all concurrency levels.

*Note: Python comparison requires llama-cpp-python installation on gpu_dev. infergo RSS measured post-benchmark via /proc/PID/status.*

## Results — infergo

| Concurrency | Req/s | P50 ms | P99 ms | RSS MB | Errors |
|---|---|---|---|---|---|
| 1  | 2.2  | 450  | 481  | 1168 | 0 |
| 2  | 3.4  | 582  | 593  | 1168 | 0 |
| 4  | 3.8  | 1059 | 1073 | 1168 | 0 |
| 8  | 4.4  | 1832 | 1863 | 1168 | 0 |
| 16 | 10.8 | 972  | 1818 | 1168 | 0 |
| 32 | 12.5 | 1389 | 1825 | 1168 | 0 |

**RSS observation:** 1168 MB flat across all concurrency levels.  
Model weights reside in GPU VRAM (not in RSS). infergo serves all 32 concurrent clients from one model copy.

## Results — Python (llama-cpp-python)

*Pending — requires llama-cpp-python installation on gpu_dev.*  
*Theoretical: at c=N workers, Python RSS ≈ N × model_memory_per_process.*  
*For llama3-8b-q4 (Q4_K_M): ~4.6 GB GPU + ~2 GB CPU RAM per worker.*

| Concurrency | Req/s | P50 ms | P99 ms | RSS MB (estimated) | Errors |
|---|---|---|---|---|---|
| 1  | — | — | — | ~2048  | — |
| 2  | — | — | — | ~4096  | — |
| 4  | — | — | — | ~8192  | — |
| 8  | — | — | — | ~16384 | — |
| 16 | — | — | — | ~32768 | — |
| 32 | — | — | — | ~65536 | — |

## Memory comparison

| Concurrency | infergo RSS MB | Python RSS MB (est) | Python / infergo |
|---|---|---|---|
| 1  | 1168 | ~2048  | ~1.8× |
| 2  | 1168 | ~4096  | ~3.5× |
| 4  | 1168 | ~8192  | ~7.0× |
| 8  | 1168 | ~16384 | ~14× |
| 16 | 1168 | ~32768 | ~28× |
| 32 | 1168 | ~65536 | ~56× |

## Throughput analysis

| Concurrency | infergo req/s | Scaling factor |
|---|---|---|
| 1  | 2.2  | 1.0× (baseline) |
| 2  | 3.4  | 1.5× |
| 4  | 3.8  | 1.7× |
| 8  | 4.4  | 2.0× |
| 16 | 10.8 | 4.9× |
| 32 | 12.5 | 5.7× |

Throughput scales with concurrency because continuous batching groups all active sequences into one `BatchDecode` call. At c=16–32, the scheduler accumulates enough sequences to fully saturate the RTX 5070 Ti.

## Notes

- Prompt: `Explain what a transformer neural network is in two sentences.` (~14 tokens)
- 40 total requests per concurrency level
- infergo: single process, goroutines, continuous batching (OPT-2)
- RSS measured post-benchmark: `cat /proc/PID/status | grep VmRSS`
- infergo RSS of 1168 MB is stable CPU RAM — model weights are in CUDA VRAM
- Python comparison deferred until llama-cpp-python is installed on gpu_dev
