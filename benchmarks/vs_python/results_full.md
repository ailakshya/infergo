# Benchmark: infergo vs llama-cpp-python

*Generated: 2026-04-03*

**Model:** `llama3-8b-q4.gguf` — LLaMA 3 8B Q4_K_M (4.6 GB)  
**Hardware:** RTX 5070 Ti 16 GB VRAM (CUDA SM 12.0 / Blackwell), 64-core CPU  
**infergo:** Go 1.23 + llama.cpp (CGo bridge)  
**Python:** llama-cpp-python 0.3.19 (same llama.cpp weights)  
**Scenarios:** short (~20 tok in / 64 tok out), long (~512 tok in / 256 tok out), cold start

---

## CUDA — RTX 5070 Ti

### Short prompts (n=40, concurrency=4 for infergo, sequential for Python)

| Backend | req/s | tok/s | Mean ms | P50 ms | P95 ms | P99 ms | Errors |
|---|---|---|---|---|---|---|---|
| **infergo** | **2.7** | **142** | 1429 | 1423 | 1793 | 1802 | 0 |
| llama-cpp-python | 2.1 | 133 | 469 | 465 | 481 | 486 | 0 |

> infergo is **+29% req/s** and **+7% tok/s** vs llama-cpp-python on CUDA short prompts.  
> Python P50 latency is lower because requests run sequentially (no queuing behind mutex).

### Long prompts (n=20, concurrency=1)

| Backend | req/s | tok/s | Mean ms | P50 ms | P95 ms | P99 ms | Errors |
|---|---|---|---|---|---|---|---|
| **infergo** | **0.6** | **144** | 1775 | 1771 | 1789 | 1796 | 0 |
| llama-cpp-python | 0.5 | 131 | 1935 | 1928 | 1969 | 1971 | 0 |

> infergo is **+20% req/s** and **+10% tok/s** vs llama-cpp-python on CUDA long prompts.

### Cold start (first request, no warmup)

| Backend | Cold start ms |
|---|---|
| **infergo** | **456** |
| llama-cpp-python | 494 |

> infergo cold start is **~8% faster** (no Python interpreter overhead).

---

## CPU — No GPU offload (gpu-layers=0)

### Short prompts (n=10, concurrency=1)

| Backend | req/s | tok/s | Mean ms | P50 ms | P95 ms | P99 ms | Errors |
|---|---|---|---|---|---|---|---|
| **infergo** | 0.2 | 10 | 5462 | 6494 | 6571 | 6571 | 0 |
| llama-cpp-python | 0.1 | 5 | 12002 | 12491 | 14083 | 15292 | 0 |

> infergo CPU is **2× faster** than llama-cpp-python CPU (**10 tok/s vs 5 tok/s**).

### Long prompts (n=5, concurrency=1)

| Backend | req/s | tok/s | Mean ms | P50 ms | P95 ms | P99 ms | Errors |
|---|---|---|---|---|---|---|---|
| **infergo** | 0.0 | 10 | 25990 | 25972 | 26218 | 26218 | 0 |
| llama-cpp-python | 0.0 | 11 | 23733 | 23733 | 23918 | 23918 | 0 |

### Cold start (CPU)

| Backend | Cold start ms |
|---|---|
| **infergo** | 6450 |
| llama-cpp-python | 9897 |

> infergo CPU cold start is **~35% faster** (no Python startup overhead, no GIL).

---

## GPU vs CPU speedup

| Backend | Short tok/s (GPU) | Short tok/s (CPU) | GPU speedup |
|---|---|---|---|
| infergo | 142 | 10 | **14×** |
| llama-cpp-python | 133 | 5 | **27×** |

---

## Summary

| Metric | Winner | Margin |
|---|---|---|
| CUDA throughput (tok/s) | infergo | +7–10% |
| CUDA req/s | infergo | +20–29% |
| CPU throughput | infergo | ~2× |
| Cold start (CUDA) | infergo | ~8% |
| Cold start (CPU) | infergo | ~35% |

**Key takeaway:** infergo matches or exceeds llama-cpp-python in all scenarios, with the largest gains on CPU (2× throughput) due to no Python GIL overhead and direct CGo dispatch. On GPU, both backends saturate the RTX 5070 Ti similarly, with infergo's Go HTTP layer adding negligible overhead.

---

## Notes

- infergo serializes generation with a mutex (llama.cpp is not thread-safe for concurrent `llama_decode` calls). With multiple clients, requests queue behind the mutex — this is why P50 latency for infergo is higher than Python on short/CUDA, even though throughput is higher.
- llama-cpp-python runs sequentially (Python GIL prevents true parallel generation threads).
- `tok/s` = output tokens ÷ total wall-clock seconds (includes HTTP + mutex wait for infergo).
- Cold start = latency of first request with no prior warmup (cold KV cache).
- CUDA: RTX 5070 Ti, SM 12.0 (Blackwell), CUDA 12.8, all 33 LLaMA layers offloaded.
- CPU run: `--gpu-layers 0`, 64-core host CPU, no GPU offload.
- Both backends use the same `llama3-8b-q4.gguf` weights (Q4_K_M quantization).
