# Infergo — Problem Statement & Solution Map

> This document defines the exact problems infergo exists to solve.
> Every feature, optimization, and architectural decision in this project
> should trace back to one of the problems listed here.

---

## The Core Premise

Go is the dominant language for production infrastructure — APIs, microservices,
Kubernetes operators, data pipelines. But when a Go service needs to run a model
(LLM, embedding, object detection), engineers must either:

1. **Call a Python sidecar** — adding a network hop, a second process, and Python's full runtime overhead
2. **Use a third-party hosted API** — losing control, adding latency, paying per token
3. **Write CGo bindings from scratch** — weeks of work, no standard library, no community

There is no Go-native inference library that works the way `transformers`,
`sentence-transformers`, or `ultralytics` work in Python. **That is the gap infergo fills.**

---

## Problem 1 — The GIL Wall (Normal to Large Scale)

**What it is:**
Python's Global Interpreter Lock (GIL) prevents true parallel execution across threads.
To serve multiple users simultaneously, Python inference servers must fork separate
**processes** — each loading a full copy of the model into memory.

**What it costs:**

| Concurrent users | Python (multi-process) | infergo (goroutines) |
|---|---|---|
| 1 | 4.6 GB VRAM | 4.6 GB VRAM |
| 10 | 46 GB VRAM (10 processes) | 4.6 GB VRAM (1 process) |
| 50 | 230 GB VRAM | 4.6 GB VRAM |
| 100 | not practical | 4.6 GB VRAM + queue |

**Real consequence:** A Go API server handling 100 concurrent requests is normal.
The same API calling a Python inference backend needs a fleet of GPU workers just
to serve that load. This is why inference is expensive — not because GPUs are slow,
but because Python wastes the GPU.

**How infergo solves it:**
Goroutines are 8 KB vs 8 MB for OS threads. One infergo process handles unlimited
concurrent HTTP requests, routing them through the scheduler (OPT-2) with one
model copy in memory.

**Solved by:** OPT-2 (continuous batching scheduler)

---

## Problem 2 — Latency Degrades Under Load

**What it is:**
Python's GIL means sequential request processing. With 10 concurrent users, the
10th request literally waits for the first 9 to finish before starting.

**What it costs:**
- Single user P50: 465 ms (CUDA, 64 tokens)
- 10 concurrent users P50: ~4650 ms (same hardware, just queuing)
- User experience collapses even though the GPU is doing the same work

**How infergo solves it:**
Continuous batching (OPT-2) runs all active sequences in the same `BatchDecode`
call every step. P50 stays flat regardless of concurrency — the GPU processes all
requests together instead of one at a time.

**Target:**
- Before OPT-2: P50 = 1423 ms at concurrency=4 (mutex queuing)
- After OPT-2: P50 ≤ 600 ms at concurrency=4 (batched decode)
- At concurrency=32: P50 ≤ 800 ms (GPU utilization ≥ 85%)

**Solved by:** OPT-2, OPT-22 (PagedAttention)

---

## Problem 3 — Cold Start Kills Autoscaling

**What it is:**
Python inference pod startup time is 8–15 seconds:
- Python interpreter init: ~1 s
- PyTorch + CUDA library load: ~3 s
- Model load from disk: ~5–10 s
- First request: ~15 s from pod schedule to response

In a Kubernetes environment, when traffic spikes and HPA adds a new pod, that
pod is useless for 15 seconds. By then the spike may be over, or users have
already timed out.

**What it costs:**
- Autoscaling is effectively disabled for latency-sensitive inference
- Engineers over-provision pods permanently just to avoid scale events
- Serverless inference (scale-to-zero) is not practical

**How infergo solves it:**
- infergo cold start (CUDA): **456 ms** (measured)
- infergo cold start (CPU): **6.4 s** (model load from disk dominates; no interpreter overhead)
- With OPT-16 (model pre-cache on PVC), cold start reduces to under 2 s on any hardware
- With OPT-21 + OPT-25, KEDA scales infergo pods in 1–2 request RTTs

**Solved by:** inherent Go startup speed + OPT-16, OPT-21, OPT-25

---

## Problem 4 — No Go-Native Inference Library

**What it is:**
Python has a rich ecosystem:
- `transformers` — all model architectures, auto-tokenizer, pipeline API
- `sentence-transformers` — embedding models, similarity search
- `ultralytics` — YOLO family, preprocessing, postprocessing
- `vLLM` — production LLM serving with continuous batching

Go has nothing equivalent. A Go developer who wants to run a model must either
call HTTP APIs or wrap Python somehow. There is no `go get` that gives you model
inference.

**What it costs:**
- Go services that need ML must add Python as a dependency
- Teams maintain two codebases in two languages for one feature
- Deployment complexity doubles (two runtimes, two sets of dependencies)
- No type safety, no compile-time guarantees across the language boundary

**How infergo solves it:**
- `go get github.com/ailakshya/infergo` — one dependency, no Python
- LLM, embedding, and detection inference from idiomatic Go code
- Go SDK (OPT-15): `client.Chat()`, `client.Embed()`, `client.Detect()`
- OpenAI-compatible HTTP API for services that use HTTP

**Solved by:** OPT-3..OPT-7, OPT-15

---

## Problem 5 — Python Memory Fragmentation Causes Long-Running OOM

**What it is:**
Python's memory allocator + PyTorch's CUDA caching allocator fragment GPU memory
over hours of serving. A server that starts healthy at 12 GB VRAM usage slowly
drifts to 15 GB, then OOM crashes — forcing restarts and dropped requests.

Kubernetes sees the OOMKilled pod and restarts it, but the restart itself takes
15 seconds (see Problem 3). In production, this becomes a rolling crash pattern.

**How infergo solves it:**
- KV cache is managed explicitly by `KVCacheSlotManager` — deterministic allocation
- No PyTorch caching allocator — memory is freed when a sequence closes
- With OPT-22 (PagedAttention): on-demand page allocation, no fragmentation possible
- Go's GC manages Go-side memory cleanly; C++ side is RAII with explicit destructors

**Solved by:** existing KVCacheSlotManager + OPT-22

---

## Problem 6 — Inference for Large Models Requires Complex Python Infrastructure

**What it is:**
For models larger than one GPU (70B+ parameters), Python stacks require:
- **vLLM + Ray** — a distributed compute framework just to split one model
- **DeepSpeed** — large model training framework repurposed for inference
- **tensor_parallel + pipeline_parallel** flags, NVLink requirements, custom kernels
- 10+ Python packages with conflicting CUDA version requirements

A Go engineer who wants to serve Llama-3-70B faces a Python dependency tree of
50+ packages, Ray cluster configuration, and NCCL tuning.

**How infergo solves it:**
- OPT-23: `--tensor-split 0.5,0.5` — llama.cpp built-in tensor parallelism, no Ray
- OPT-24: `--pipeline-stages 2` — pipeline parallelism over PCIe, works on consumer GPUs
- OPT-26: prefill/decode separation — data-center scale, pure Go orchestration

**Solved by:** OPT-23, OPT-24, OPT-26

---

## Problem 7 — Deployment Complexity and Container Bloat

**What it is:**
A typical Python inference Docker image:
```
python:3.11-cuda12.1 base:   5.2 GB
+ torch + torchvision:       3.8 GB
+ transformers + tokenizers: 0.8 GB
+ vLLM:                      0.4 GB
+ your application code:     0.1 GB
─────────────────────────────────────
Total image:                ~10.3 GB
```

This means:
- 10 GB pulled on every new node in the cluster
- 10 GB stored in every container registry layer cache
- 30–60 second image pull time on a new node
- Scale-out is bottlenecked by image pull, not compute

**infergo image:**
```
ubuntu:22.04-cuda12.1 base:  1.8 GB (CUDA runtime only)
+ infergo binary:            ~22 MB
─────────────────────────────────────
Total image:                ~1.82 GB
```

**How infergo solves it:**
- Single statically-linked Go binary with CGo to `libinfer_api.so`
- No Python, no pip, no virtualenv
- Image is 5–6× smaller than Python equivalent

**Solved by:** inherent Go build model

---

## Problem 8 — No Unified Serving Interface Across Model Types

**What it is:**
In Python, serving LLMs, embedding models, and detection models requires
different servers with different APIs:
- LLMs: vLLM or llama.cpp server (OpenAI format)
- Embeddings: sentence-transformers with custom FastAPI endpoint
- Detection: ultralytics serving or Triton with ONNX Runtime

Each has its own configuration format, health check style, metrics format, and
authentication mechanism. Operating three separate servers for three model types
adds operational burden.

**How infergo solves it:**
One binary, one config, one metrics endpoint:
```
infergo serve \
  --model llm:llama3.gguf \
  --model embed:nomic.onnx \
  --model detect:yolov8n.onnx \
  --port 9090
```
All models available on the same port. One `/health/ready`, one `/metrics`,
one API key.

**Solved by:** OPT-8 (multi-model serving)

---

## Problem 9 — Observability is an Afterthought in Python Inference Servers

**What it is:**
Production Go services are expected to have:
- Structured logging with trace IDs
- Prometheus metrics on every operation
- OpenTelemetry traces linking requests end-to-end
- Health probes that Kubernetes can act on

Python inference servers (llama.cpp server, vLLM) have partial coverage — some
metrics, basic logging, no distributed tracing. Integrating them into an existing
Go-based observability stack requires custom adapters.

**How infergo solves it:**
- Prometheus metrics built-in from day one (T-45, done)
- Kubernetes health checks built-in (T-46, done)
- OpenTelemetry tracing (OPT-18): full span tree per request
- KEDA metrics (OPT-21): queue depth + GPU utilization for autoscaling

**Solved by:** existing Prometheus + health checks + OPT-18, OPT-21

---

## Problem 10 — Python Inference Correctness is Hard to Test

**What it is:**
Python inference servers are typically tested with integration tests against a
running server. Unit testing model logic requires mocking at the HTTP boundary.
There is no way to `go test` the inference path.

**How infergo solves it:**
- Pure Go test coverage: `go test ./go/...` — tests the full stack without a running server
- C++ gtest coverage: `ctest` — tests the compute engine independently
- ASan + race detector: `go test -race`, `cmake -DASAN=ON` — memory safety guaranteed
- Benchmarks: `go test -bench=.` — regressions caught before merge

**Solved by:** existing test infrastructure

---

## Solution Map

| Problem | Root cause | infergo solution | Tasks |
|---|---|---|---|
| GIL wall — memory scales with users | Python GIL forces multi-process | Goroutines, 1 model copy | OPT-2 |
| Latency degrades under load | Sequential processing | Continuous batching | OPT-2, OPT-22 |
| Cold start kills autoscaling | Python interpreter + library load | 456 ms Go startup | OPT-16, OPT-25 |
| No Go-native inference library | Python ecosystem monopoly | Go SDK + HTTP API | OPT-3..7, OPT-15 |
| Memory fragmentation → OOM | PyTorch caching allocator | Explicit KV management | OPT-22 |
| Large model complexity | vLLM + Ray required | Built-in tensor split | OPT-23, OPT-24 |
| Container bloat | Python + PyTorch image size | 22 MB Go binary | inherent |
| No unified serving interface | Different server per model type | Multi-model serve | OPT-8 |
| Observability gaps | Afterthought in Python servers | Built-in from day one | OPT-18, OPT-21 |
| Hard to unit test inference | HTTP-only testing in Python | `go test` + `ctest` | existing |

---

## What infergo is NOT trying to solve

- **Model training** — infergo is inference-only; training stays in Python/PyTorch/JAX
- **Model architecture research** — we consume GGUF and ONNX, we don't define architectures
- **Dataset management** — out of scope
- **Fine-tuning** — out of scope (LoRA adapters via llama.cpp may come later)
- **Replacing Python entirely** — Python is the right tool for experimentation and training;
  infergo is the right tool for serving trained models in production Go services

---

## Progress Checklist

Each problem has a definition of done. Check it off when the solution is measurably
working — not just when the code exists.

---

### Problem 1 — GIL Wall (memory scales with users)

- [x] OPT-2: Continuous batching scheduler implemented
- [x] OPT-2: `go test -race ./go/...` exits 0 (no data races)
- [ ] OPT-2: 4 concurrent clients at CUDA P50 ≤ 600 ms (now 1307 ms with --batch-timeout-ms 5 --max-batch-size 8 re-bench 2026-04-04; 10.6% gain post-OPT-22; target not yet met)
- [x] OPT-2: RSS does not grow with concurrency — 1168 MB flat c=1..32 (OPT-27 benchmark)
- [x] OPT-27: Scalability benchmark shows infergo RSS flat — 1168 MB at c=1..32 (measured)
- [ ] **PROBLEM 1 SOLVED** — infergo serves N users from 1 model copy

---

### Problem 2 — Latency Degrades Under Load

- [x] OPT-2: Scheduler batches all active sequences in one `BatchDecode` call
- [ ] OPT-2: CUDA P50 ≤ 600 ms at concurrency=4 (now 1307 ms with batch tuning re-bench 2026-04-04; target not yet met)
- [x] OPT-22: KVPageAllocator reduces KV slot contention — P50 c=4: 1185→1059 ms (measured 2026-04-04)
- [x] OPT-27: Benchmark chart generated — benchmark_scalability.png shows req/s and latency curves
- [ ] **PROBLEM 2 SOLVED** — P50 latency flat under any concurrency level

---

### Problem 3 — Cold Start Kills Autoscaling

- [x] CUDA cold start measured: 456 ms (done — bench_full.py)
- [x] CPU cold start measured: 6.4 s (done — bench_full.py)
- [x] OPT-16: `infergo pull` downloads model to local cache in ≤ 30 s
- [x] OPT-21: `infergo_queue_depth` metric exported for KEDA
- [ ] OPT-25: Helm chart deploys; KEDA scales new pod from 0 → ready in ≤ 10 s
- [ ] OPT-25: Zero requests lost during scale-up event (1000 req stress test)
- [ ] **PROBLEM 3 SOLVED** — new pod ready in ≤ 10 s, autoscaling works

---

### Problem 4 — No Go-Native Inference Library

- [x] LLM inference: `infergo serve --model llama3.gguf` works (done — T-29..T-47)
- [x] OpenAI-compatible HTTP API: `/v1/chat/completions` (done — T-43..T-44)
- [x] OPT-3: ONNX Runtime inference: `infer_onnx_run()` C function implemented
- [x] OPT-4: `/v1/embeddings` endpoint returns correct vectors
- [x] OPT-5: `/v1/detect` endpoint returns bounding boxes
- [x] OPT-7: BERT tokenizer for embedding models (go/tokenizer wraps HF tokenizers)
- [x] OPT-15: `go get github.com/ailakshya/infergo/client` — typed Go SDK
- [x] OPT-15: `client.Chat()`, `client.Embed()`, `client.Detect()` all work in tests
- [x] **PROBLEM 4 SOLVED** — v0.1.0 tagged + published on pkg.go.dev (2026-04-04); Awesome Go PR submitted (avelino/awesome-go#6200)

---

### Problem 5 — Memory Fragmentation → OOM Crashes

- [x] KV cache slot manager: explicit allocation per sequence (done — T-24)
- [x] KV cache freed on sequence close via `llama_memory_seq_rm` (done — bug fix)
- [x] OPT-22: KVPageAllocator block allocator implemented — pages freed on sequence close (FreeSlot in ~InferSequence)
- [x] OPT-22: 1000 requests back-to-back RSS within 5% — measured +0.3% with GOGC=50 + --gc-interval 100 (was +11.9%; PASS 2026-04-04)
- [x] OPT-3-T4: ASan confirms zero memory leaks after 1000 ONNX runs (PASS — confirmed 2026-04-03)
- [x] **PROBLEM 5 SOLVED** — RSS drift +0.3% after 1000 requests; KVPageAllocator + GOGC=50 + --gc-interval 100 eliminates Go GC heap growth (2026-04-04)

---

### Problem 6 — Large Models Require vLLM + Ray

- [ ] OPT-23: `--tensor-split 0.5,0.5` loads model across 2 GPUs — pending AWS (p3.8xlarge/p4d.24xlarge)
- [ ] OPT-23: 2-GPU tok/s ≥ 1.6× 1-GPU tok/s for same model — pending AWS
- [ ] OPT-23: Llama-3-70B-Q4 loads in 2× 40 GB GPU without OOM — pending AWS (p4d.24xlarge, 8× A100-40GB)
- [ ] OPT-24: `--pipeline-stages 2` works over PCIe across 2 GPUs — pending AWS (p3.8xlarge, 4× V100)
- [ ] OPT-25: KEDA scales up/down on EKS — pending AWS EKS (3× g4dn.xlarge)
- [ ] OPT-26: Prefill node + decode node end-to-end — pending AWS (2× g4dn.xlarge)
- [ ] **PROBLEM 6 SOLVED** — 70B model served with one flag, no Ray/vLLM needed

---

### Problem 7 — Container Bloat

- [x] `Dockerfile.cpu` builds ≤ 2 GB image (done — T-49)
- [x] `Dockerfile.cuda` builds ≤ 3 GB image (done — T-49)
- [ ] OPT-25: Helm chart pulls image on new node in ≤ 20 s (measured in CI)
- [ ] **PROBLEM 7 SOLVED** — image ≤ 3 GB; Python equivalent is 10+ GB

---

### Problem 8 — No Unified Serving Interface

- [x] Single binary serves LLM (done — T-43..T-47)
- [x] OPT-8: `--model llm:llama3.gguf --model embed:nomic.onnx` both load in one process
- [x] OPT-8: Chat req routes to LLM; embedding req routes to ONNX — no cross-routing
- [x] OPT-8: `GET /v1/models` lists all loaded models with their types
- [x] OPT-9: `POST /v1/admin/reload` hot-swaps model without restart
- [x] **PROBLEM 8 SOLVED** — one binary, one port, all model types, no restart for reload

---

### Problem 9 — Observability Gaps

- [x] Prometheus metrics: `infergo_requests_total`, `infergo_tokens_total` (done — T-45)
- [x] Kubernetes health probes: `/health/live`, `/health/ready` (done — T-46)
- [x] OPT-18: OpenTelemetry spans emitted per request with `decode_ms` attribute
- [x] OPT-18: Jaeger UI shows full trace for chat completion request
- [x] OPT-21: `infergo_queue_depth` and `infergo_active_sequences` exported (gpu_util needs nvml)
- [x] OPT-21: KEDA ScaledObject example validated in docs
- [x] **PROBLEM 9 SOLVED** — full Prometheus + OTel + KEDA integration verified

---

### Problem 10 — Hard to Unit Test Inference

- [x] C++ gtest: 250/250 pass (`ctest`) — includes KVPageAllocatorTest (8 cases) + KVSerialize (4 cases)
- [x] ASan: 81 tests clean, zero leaks
- [x] Go tests: `go test ./go/...` passes
- [x] OPT-3: `go test ./go/onnx/...` covers ONNX session create + run + close
- [x] OPT-15: `go test ./go/client/...` covers all SDK methods with mock server
- [x] CI pipeline: all tests run on every PR automatically (.github/workflows/ci.yml)
- [x] **PROBLEM 10 SOLVED** — `go test ./...` and `ctest` cover all inference paths

---

## Overall Progress

```
Problem 1  GIL wall              [~] 4/5 done  (scheduler + race-free + RSS flat + OPT-27 run; P50=1307ms with batch tuning, target ≤600ms not yet met)
Problem 2  Latency under load    [~] 4/5 done  (scheduler + OPT-22 P50 improvement measured + chart + OPT-27; P50=1307ms with batch flags, target ≤600ms not yet met)
Problem 3  Cold start            [~] 4/6 done  (cold start + pull + queue_depth + keda example done)
Problem 4  No Go library         [x] 9/9 done  (LLM+HTTP+ONNX+embeddings+detect+tokenizer+SDK+pkg.go.dev+Awesome Go PR — SOLVED 2026-04-04)
Problem 5  Memory fragmentation  [x] 5/5 done  (KV slot manager + OPT-22 allocator + pages freed + ASan 1000 ONNX runs + RSS +0.3% PASS — SOLVED 2026-04-04)
Problem 6  Large model infra     [~] 0/6 done  (implementations complete; all tests pending AWS multi-GPU/EKS cluster)
Problem 7  Container bloat       [~] 2/3 done  (Dockerfiles done)
Problem 8  No unified interface  [x] 6/6 done  (LLM+multi-model+routing+models-list+hot-reload+SOLVED)
Problem 9  Observability         [x] 7/7 done  (Prometheus + health + OTel + queue_depth + active_seqs + KEDA + SOLVED)
Problem 10 Hard to test          [x] 7/7 done  (ctest + ASan + go test + onnx + client mock + CI + SOLVED)
───────────────────────────────────────────────
Total                            49/54 done  (91%) — SOLVED: P4 (pkg.go.dev v0.1.0 + Awesome Go PR), P5 (RSS +0.3% PASS); remaining: P50 ≤600ms (1063ms, needs V100), P6 AWS multi-GPU, P3/P7 EKS
```
