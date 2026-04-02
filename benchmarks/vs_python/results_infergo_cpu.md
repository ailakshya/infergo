# Benchmark: infergo vs llama-cpp-python

*Generated: 2026-04-02 18:38 UTC*

**Model:** llama3-8b-q4.gguf  
**Hardware:** RTX 5070 Ti (16 GB VRAM) / same CPU for CPU runs  
**infergo version:** Go 1.23, llama.cpp (same weights)  
**Python version:** llama-cpp-python (same weights)  

---

## Short prompts (~20 tokens in, 64 tokens out)

| Backend | Device | req/s | tok/s | Mean ms | P50 ms | P95 ms | P99 ms | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| infergo | cpu | 0.2 | 10 | 5462 | 6494 | 6571 | 6571 | 0 |

## Long prompts (~512 tokens in, 256 tokens out)

| Backend | Device | req/s | tok/s | Mean ms | P50 ms | P95 ms | P99 ms | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| infergo | cpu | 0.0 | 10 | 25990 | 25972 | 26218 | 26218 | 0 |

## Cold start (time to first token, fresh context)

| Backend | Device | Cold start ms |
| --- | --- | --- |
| infergo | cpu | 6450 |

---

## Notes

- infergo serializes generation with a mutex (single-threaded decode, multi-threaded HTTP)
- llama-cpp-python tested sequentially (GIL prevents true parallel generation)
- `tok/s` = output tokens / total wall-clock seconds
- Cold start measured as latency of the very first request with no warmup
- CUDA crash on SM 12.0 (Blackwell) was fixed by ensuring sequential `llama_decode` calls
- KV cache cleared via `llama_memory_seq_rm` on sequence close to prevent positional errors
