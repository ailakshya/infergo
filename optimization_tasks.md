# Infergo — Optimization & Benchmark Tasks

> **Status legend:** `[ ]` pending · `[~]` in progress · `[x]` done

---

## OPT-1 — CPU: OpenBLAS for BLAS-accelerated prefill

**Problem:** `GGML_BLAS=OFF` in current build. Long-prompt prefill (512 tokens) is
slow because GGML uses scalar kernels for the prompt-processing GEMM. Python's
llama-cpp-python build links OpenBLAS and runs the prefill ~10% faster.

**What changes:**
- `apt install libopenblas-dev` on gpu_dev
- CMake reconfigure: add `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS` to
  `add_subdirectory(vendor/llama.cpp)`
- Rebuild `infer_api.so` + `infergo` binary

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-1-T1 | Build succeeds with BLAS on | `cmake --build build` exits 0; `ldd build/cpp/api/libinfer_api.so` shows `libopenblas` |
| OPT-1-T2 | Prefill speedup on long prompt | Single request, 512-token prompt, 64 output tokens: TTFT ≤ 1500 ms (was ~25000 ms first-token on CPU) |
| OPT-1-T3 | Long CPU benchmark | `bench_full.py --device cpu`: infergo CPU long `tok/s ≥ 11` (matches or beats Python 11 tok/s) |
| OPT-1-T4 | Short prompts unaffected | infergo CPU short `tok/s` within ±5% of pre-BLAS run (10 tok/s) |
| OPT-1-T5 | CUDA build still works | `bench_full.py --device cuda` passes with same CUDA results (BLAS only active on CPU path) |

---

## OPT-2 — GPU / CPU: Continuous batching scheduler

**Problem:** `llmAdapter.Generate()` holds a mutex for the full duration of each
request. Under concurrency=4 the P50 latency is 3× higher than single-client
latency. GPU utilization is ~25% because only one sequence is decoded per step.

**What changes (Go layer only — no C++ changes):**

1. New type `schedulerModel` replaces `llmAdapter`
2. `Submit(ctx, prompt, maxTokens, temp) <-chan TokenEvent` — non-blocking enqueue
3. Scheduler goroutine (single owner of the LLM):
   - Reads pending requests from a buffered channel
   - Maintains a map of active `*llm.Sequence` per request
   - Each loop iteration: admit new requests up to `maxBatch`, call
     `m.BatchDecode(activeSeqs)`, sample + route each token to its result channel,
     remove finished sequences
4. HTTP SSE handler consumes `<-chan TokenEvent` and writes `data:` chunks
5. Non-streaming handler drains the channel and assembles the full response

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-2-T1 | Unit: single request through scheduler | `go test ./go/cmd/infergo/...` — scheduler returns correct text for a known prompt |
| OPT-2-T2 | Unit: 4 concurrent requests complete | All 4 goroutines get non-empty responses, no deadlock, no panic |
| OPT-2-T3 | No mutex in handler path | `go vet` + race detector: `go test -race ./go/...` exits 0 |
| OPT-2-T4 | CUDA P50 latency under concurrency=4 | `bench_full.py --device cuda --concurrency 4`: P50 ≤ 600 ms (was 1423 ms) |
| OPT-2-T5 | Throughput does not regress | CUDA short `tok/s ≥ 140` (was 142) |
| OPT-2-T6 | SSE streaming works | `curl -N .../v1/chat/completions -d '{"stream":true,...}'` emits `data:` lines as tokens arrive |
| OPT-2-T7 | Graceful shutdown drains in-flight requests | SIGTERM while 4 requests in flight: all complete before process exits |
| OPT-2-T8 | Context cancellation propagates | Client disconnect mid-stream: scheduler drops sequence, KV slot freed |

---

## OPT-3 — Embedding benchmark (infergo ONNX vs sentence-transformers)

**Scope:** Design + run the benchmark. ONNX inference in infergo must be
implemented first (current `onnxAdapter` only registers the path).

**Models:**
| Model | Size | Use case |
|---|---|---|
| `nomic-embed-text-v1.5` | 274 MB ONNX | General English embedding |
| `bge-m3` | 570 MB ONNX | Multilingual, long context |
| `all-MiniLM-L6-v2` | 90 MB ONNX | Lightweight baseline |

**Python baseline:** `sentence-transformers` library

**Benchmark script:** `benchmarks/vs_python/bench_embedding.py`

**Scenarios:**
- Short sentences (avg 15 tokens), batch sizes: 1, 8, 32, 64
- Long paragraphs (avg 256 tokens), batch sizes: 1, 8, 32
- Device: CUDA + CPU

**Metrics:** sentences/sec, tok/s, mean latency ms, P50 / P99

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-3-T1 | ONNX inference implemented | `POST /v1/embeddings` returns `{"object":"list","data":[{"embedding":[...]}]}` |
| OPT-3-T2 | Embedding vector correctness | Cosine similarity between infergo and sentence-transformers embedding of same text ≥ 0.999 |
| OPT-3-T3 | Batch throughput | infergo CUDA batch=32: sentences/sec ≥ Python sentences/sec |
| OPT-3-T4 | CUDA vs CPU speedup | CUDA batch=32 at least 5× faster than CPU batch=32 |
| OPT-3-T5 | Results written | `results_embedding.md` exists with populated tables |

**Blocker:** `onnxAdapter` needs real ONNX Runtime inference (run session, parse
output tensor) before this benchmark is meaningful.

---

## OPT-4 — Detection benchmark (infergo ONNX vs ultralytics/onnxruntime)

**Scope:** Object detection throughput and latency.

**Models:**
| Model | Size | mAP50 | Use case |
|---|---|---|---|
| `yolov8n.onnx` | 6.3 MB | 37.3 | Edge / real-time |
| `yolov8s.onnx` | 22 MB | 44.9 | Balanced |
| `yolov8m.onnx` | 52 MB | 50.2 | Accuracy |

**Python baseline:** `ultralytics` library + ONNX Runtime

**Benchmark script:** `benchmarks/vs_python/bench_detection.py`

**Scenarios:**
- Single image 640×640 (latency)
- Batch: 1, 8, 32, 64 images (throughput)
- Device: CUDA + CPU
- Input: COCO val2017 sample (100 images)

**Metrics:** images/sec, P50 / P99 latency ms, preprocessing time separate from
inference time

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-4-T1 | ONNX detection runs | `POST /v1/detect` with base64 image returns bounding boxes JSON |
| OPT-4-T2 | Detection correctness | On COCO val sample, mAP50 within 0.5% of ultralytics reference |
| OPT-4-T3 | Single-image latency | infergo CUDA yolov8n: P50 ≤ 5 ms |
| OPT-4-T4 | Batch throughput | infergo CUDA batch=32: images/sec ≥ Python ONNX Runtime images/sec |
| OPT-4-T5 | Results written | `results_detection.md` exists with populated tables |

**Blocker:** Same as OPT-3 — needs real ONNX Runtime inference implementation.

---

## OPT-5 — Multi-model LLM benchmark

**Scope:** Run the full LLM benchmark across 3 models to show infergo's advantage
is model-agnostic, not cherry-picked on one checkpoint.

**Models:**
| Model | Size | Params | Quantization |
|---|---|---|---|
| `llama3-8b-q4.gguf` | 4.6 GB | 8B | Q4_K_M |
| `phi-3.5-mini-instruct.Q4_K_M.gguf` | 2.2 GB | 3.8B | Q4_K_M |
| `gemma-2-9b-it.Q4_K_M.gguf` | 5.5 GB | 9B | Q4_K_M |

**Changes to bench_full.py:**
- Add `--models` flag: comma-separated list of `name:path` pairs
- Loop outer: for each model, start/restart infergo server with that model
- Markdown output: one section per model + aggregate summary table

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-5-T1 | Phi-3.5 mini loads | `infergo serve --model phi-3.5-mini.gguf` reaches `/health/ready` |
| OPT-5-T2 | Gemma 2 9B loads | `infergo serve --model gemma-2-9b.gguf` reaches `/health/ready` |
| OPT-5-T3 | All 3 models benchmarked | `results_multimodel.md` has rows for llama3, phi3.5, gemma2 |
| OPT-5-T4 | infergo wins on all 3 models | tok/s advantage ≥ +5% on CUDA for each model vs llama-cpp-python |
| OPT-5-T5 | No KV cache leaks between model runs | Run 100 requests per model back-to-back: no positional errors, no crashes |

---

## OPT-6 — ONNX Runtime inference implementation

**Scope:** Prerequisite for OPT-3 and OPT-4. Implement real ONNX Runtime inference
in the C++ layer and expose it through the existing C API.

**What changes:**
- `cpp/onnx/onnx_engine.hpp/.cpp` — wrap `Ort::Session`, `Ort::RunOptions`
- `cpp/api/api.cpp` — `infer_onnx_run()` C function: allocate input tensors,
  run session, copy output
- `go/onnx/session.go` — Go wrapper calling `infer_onnx_run` via CGo
- `server/server.go` — add `/v1/embeddings` and `/v1/detect` routes

**Test cases:**

| ID | Test | Pass condition |
|---|---|---|
| OPT-6-T1 | ONNX session creates | `ctest -R onnx_test` passes |
| OPT-6-T2 | Embedding model runs | Run `all-MiniLM-L6-v2` on "hello world", output shape `[1, 384]` |
| OPT-6-T3 | Detection model runs | Run `yolov8n` on 640×640 zeros, output shape `[1, 84, 8400]` |
| OPT-6-T4 | No memory leak | Valgrind / ASan on 1000 ONNX runs: zero leaks |
| OPT-6-T5 | Go tests pass | `go test ./go/onnx/...` exits 0 |

---

## Execution order

```
OPT-1  (CPU BLAS)           ← independent, do first, quick win
OPT-2  (continuous batching) ← independent, biggest GPU impact
OPT-6  (ONNX inference)     ← prerequisite for OPT-3 and OPT-4
OPT-3  (embedding bench)    ← requires OPT-6
OPT-4  (detection bench)    ← requires OPT-6
OPT-5  (multi-model LLM)    ← requires OPT-1 + OPT-2 done first for clean numbers
```

## Expected outcomes

| Optimization | Metric | Before | Target |
|---|---|---|---|
| OPT-1 CPU BLAS | CPU long tok/s | 10 | ≥ 12 |
| OPT-1 CPU BLAS | CPU long P50 latency | 25936 ms | ≤ 22000 ms |
| OPT-2 cont. batching | CUDA short P50 @ concurrency=4 | 1423 ms | ≤ 600 ms |
| OPT-2 cont. batching | CUDA GPU utilization | ~25% | ≥ 85% |
| OPT-2 cont. batching | CUDA tok/s | 142 | ≥ 200 |
| OPT-6 + OPT-3 | Embedding CUDA sentences/sec | — | ≥ Python baseline |
| OPT-6 + OPT-4 | Detection CUDA images/sec | — | ≥ Python ONNX Runtime |
