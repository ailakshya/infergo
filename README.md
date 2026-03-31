# infergo

**Train in any framework. Serve in Go. Deploy anywhere.**

infergo is a production-grade, open source AI inference runtime written in Go. It exposes a clean Go API over a high-performance C++ compute engine, bridged via CGo — giving Go developers first-class access to GPU-accelerated inference without Python in the critical path.

---

## Why infergo?

Every serious inference stack today is Python-first. You get vLLM, FastAPI, and a 500MB container just to serve a model. infergo is built on a different premise:

- **Go is the serving layer.** Goroutines, channels, and the Go HTTP stack handle concurrency natively — no async/await, no GIL, no event loop.
- **C++ is the compute layer.** ONNX Runtime, llama.cpp, and CUDA kernels do the heavy lifting behind a thin C API.
- **No Python in production.** The final binary is a single Go executable that links against shared C++ libraries. Cold starts in milliseconds, not seconds.

---

## What it does

| Capability | Status |
|---|---|
| CPU tensor allocation + transfer | ✅ Done |
| CUDA tensor allocation + H↔D transfer | ✅ Done |
| ONNX inference (CPU / CUDA / TensorRT / CoreML) | 🔨 In progress |
| LLM inference via llama.cpp (continuous batching) | 🔨 In progress |
| HuggingFace tokenizer (BPE, WordPiece) | 🔨 In progress |
| Image preprocessing (decode, letterbox, normalize) | 🔨 In progress |
| NMS postprocessing (YOLO format) | 🔨 In progress |
| OpenAI-compatible HTTP API | 🔨 In progress |
| SSE token streaming | 🔨 In progress |
| Prometheus metrics + Kubernetes health checks | 🔨 In progress |

---

## Architecture

```
Go application
     │
     │  import "github.com/ailakshya/infergo/tensor"
     │  import "github.com/ailakshya/infergo/onnx"
     │  import "github.com/ailakshya/infergo/llm"
     ▼
  Go wrapper packages   (CGo — zero-copy for large tensors)
     │
     │  #include "infer_api.h"
     ▼
  C API boundary        (extern "C", no C++ types cross here)
     │
     ▼
  C++ compute engine
  ├── libinfer_tensor      — tensor memory management (CPU + CUDA)
  ├── libinfer_onnx        — ONNX Runtime sessions
  ├── libinfer_llm         — LLM engine (llama.cpp + KV cache + sampler)
  ├── libinfer_tokenizer   — HuggingFace tokenizers FFI
  ├── libinfer_preprocess  — image decode / resize / normalize
  └── libinfer_postprocess — NMS / softmax / embedding normalize
```

The C API (`cpp/include/infer_api.h`) is the only header Go ever sees. No C++ types cross into Go — only `void*`, `int`, `float`, and `char*`.

---

## End goal

Beat vLLM on tokens/second for LLaMA 3.1 8B on the same hardware, use less RAM, and ship the whole inference server as a single Go binary under 50MB (excluding model weights).

Specific targets:

- **Throughput:** higher tokens/sec than vLLM at batch size ≥ 4
- **Latency:** P99 < 200ms for a 512-token prompt on an RTX 4090-class GPU
- **Memory:** ≤ 80% of vLLM's VRAM footprint for the same model
- **Cold start:** server ready in < 2 seconds from process start
- **Binary size:** `infergo serve` binary < 50MB
- **Zero Python:** no Python dependency anywhere in the production path

---

## Build

**Requirements**
- CMake 3.20+
- GCC 13+ or Clang 16+
- CUDA Toolkit 12.4+ (optional — CPU-only builds work without it)
- [vcpkg](https://github.com/microsoft/vcpkg) with `gtest` installed

```sh
# Clone
git clone https://github.com/ailakshya/infergo.git
cd infergo

# Bootstrap vcpkg (first time only)
git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh -disableMetrics
export VCPKG_ROOT=~/vcpkg

# Configure + build
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build

# Run C++ tests
cd build && ctest --output-on-failure
```

**Debug build (Address Sanitizer)**
```sh
cmake -B build_debug -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build_debug
cd build_debug && ASAN_OPTIONS=protect_shadow_gap=0:detect_leaks=1 ctest
```

---

## Project status

Early development — the C++ tensor library (`libinfer_tensor`) is complete and tested. ONNX, LLM, tokenizer, preprocessing, postprocessing, and the Go serving layer are being built task by task.

See [`Infergo plan.md`](Infergo%20plan.md) for the full task list and architecture decisions.

---

## License

Apache 2.0
