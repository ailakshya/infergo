# Contributing to infergo

---

## Project structure

```
infergo/
├── cpp/
│   ├── include/infer_api.h     C API (the only thing Go sees)
│   ├── api/                    C wrappers + infer_api shared library
│   ├── llm/                    llama.cpp integration
│   ├── onnx/                   ONNX Runtime integration
│   ├── preprocess/             OpenCV image preprocessing
│   └── postprocess/            Classify / NMS / embedding normalize
├── go/
│   ├── llm/                    Go wrapper for InferLLM
│   ├── onnx/                   Go wrapper for InferSession
│   ├── tensor/                 Go wrapper for InferTensor
│   ├── preprocess/             Go preprocessing pipeline
│   ├── postprocess/            Go postprocessing pipeline
│   ├── server/                 OpenAI-compatible HTTP server
│   └── cmd/infergo/            CLI (serve, list-models, benchmark)
├── benchmarks/vs_python/       Python benchmark scripts
├── Dockerfile.cpu
├── Dockerfile.cuda
└── CMakeLists.txt
```

---

## How to add a new execution provider

An execution provider is a backend that runs ONNX models (e.g. CPU, CUDA, TensorRT, CoreML, DirectML). Adding one requires changes in three places.

### Step 1 — C++ side (`cpp/onnx/`)

Open `cpp/onnx/session.cpp`. The `infer_session_create` function switches on the `provider` string:

```cpp
// Add a new branch:
} else if (provider == "my_provider") {
#ifdef INFER_MY_PROVIDER
    MyProviderOptions opts;
    opts.device_id = device_id;
    session_options.AppendExecutionProvider_MyProvider(opts);
#else
    infer_set_last_error("infergo not built with MY_PROVIDER support");
    return INFER_ERR_INVALID;
#endif
}
```

Add the CMake option in `cpp/onnx/CMakeLists.txt`:

```cmake
option(INFER_MY_PROVIDER "Enable MyProvider execution provider" OFF)
if(INFER_MY_PROVIDER)
    target_compile_definitions(infer_onnx PRIVATE INFER_MY_PROVIDER=1)
    target_link_libraries(infer_onnx PRIVATE <provider_libs>)
endif()
```

### Step 2 — Go CLI (`go/cmd/infergo/serve.go`)

Add the provider string to the `--provider` flag choices comment and wire it in `loadONNX`:

```go
// In the provider switch:
case "my_provider":
    // Provider name passed directly to infer_session_create via C API
```

The provider string is passed through as-is to the C API — no Go logic needed unless you want provider-specific flags.

### Step 3 — Tests

Add an integration test in `cpp/api/api_test.cpp`:

```cpp
TEST(Session, MyProvider) {
    InferSession s = infer_session_create("my_provider", 0);
    ASSERT_NE(s, nullptr);
    // ...
    infer_session_destroy(s);
}
```

And a Go test in `go/onnx/onnx_test.go` using a small public ONNX model (e.g. MobileNetV2).

---

## Adding a new model type

If your model doesn't fit `LLMModel`, `EmbeddingModel`, or `DetectionModel`, add a new capability interface:

### Step 1 — Define the interface in `go/server/router.go`

```go
// AudioModel transcribes or synthesises audio.
type AudioModel interface {
    Model
    Transcribe(ctx context.Context, audioBytes []byte) (string, error)
}
```

### Step 2 — Add a handler

```go
func (s *Server) handleAudio(w http.ResponseWriter, r *http.Request) {
    // parse model name from request
    ref, err := s.reg.Get(modelName)
    if err != nil { http.Error(w, err.Error(), 404); return }
    defer ref.Release()

    am, ok := ref.Model.(AudioModel)
    if !ok { http.Error(w, "model does not support audio", 400); return }

    // run inference, write JSON response
}
```

### Step 3 — Wire the route

```go
// In NewServer:
mux.HandleFunc("POST /v1/audio/transcriptions", s.handleAudio)
```

---

## Hard rules (from `Infergo plan.md`)

These rules are non-negotiable and protect the CGo boundary:

1. **No C++ types cross into Go.** Only `void*`, `int`, `float`, `double`, `char*`.
2. **Every C++ function in the C API must be `extern "C"`.**
3. **Every C++ exception must be caught inside the C wrapper** and converted to `InferError`. Never let an exception propagate into Go.
4. **`infer_api` is a shared library (`.so`).** All other C++ libraries are static.
5. **`cpp/preprocess` and `cpp/postprocess` must be added via `add_subdirectory` before `cpp/api`** in `CMakeLists.txt` so their targets exist when `cpp/api` processes `if(TARGET …)` guards.

---

## Running tests

```bash
# C++ unit tests (requires a build)
cd build && ctest --output-on-failure

# Go tests
cd go && go test ./...

# Go tests with race detector
cd go && go test -race ./...
```

---

## Commit style

```
feat(server): add audio transcription handler
fix(llm): correct EOS token check for Mistral models
docs: update C API reference for infer_seq_get_logits
test(postprocess): add NMS edge case for empty predictions
```

Format: `type(scope): description` — keep subject under 72 chars.

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `build`, `ci`.
