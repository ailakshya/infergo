# Embedding Benchmark Results — OPT-4

**Date:** 2026-04-03  
**Model:** all-MiniLM-L6-v2 (ONNX)  
**Hardware:** NVIDIA GeForce RTX 5070 Ti (15 GB VRAM), AMD Ryzen 9 9900X  
**Note:** ONNX Runtime CUDA provider unavailable (libcudnn.so.9 not installed) — all runs on CPU.

## Throughput

| Backend | Provider | n_requests | concurrency | req/s | dim | errors |
|---------|----------|-----------|-------------|-------|-----|--------|
| infergo HTTP | CPU | 200 | 8 | 251.8 | 384 | 0 |
| sentence-transformers | CPU batch=32 | 200 | in-process | 3307.5 | 384 | 0 |

**Note:** The sentence-transformers benchmark runs batch=32 in-process (no HTTP overhead).
infergo uses individual HTTP requests (concurrency=8). The comparison is not apples-to-apples.
CPU ONNX computation per-request is competitive; the gap is HTTP overhead + no batching.

## Correctness

| Test | Result |
|------|--------|
| Cosine(infergo, sentence-transformers) on "hello world" | 1.000000 |
| Cosine(infergo, sentence-transformers) on "the quick brown fox..." | 1.000000 |
| Cosine(infergo, sentence-transformers) on "inference is fast..." | 1.000000 |
| 8 concurrent requests return identical vectors | cosine=1.000000 |
