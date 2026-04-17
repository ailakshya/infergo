# Contributing to infergo

infergo is a Go + C++ inference runtime. Contributions are welcome across Go, C++, CUDA, and the native SDKs.

---

## Quick start

```bash
git clone https://github.com/ailakshya/infergo --recursive
cd infergo

# CPU build (no GPU required)
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF \
  -DINFER_CUDA=OFF \
  -DONNXRUNTIME_ROOT=/opt/onnxruntime
cmake --build build -j$(nproc)

# Go build
cd go && go build ./cmd/infergo
```

See [docs/getting-started.md](docs/getting-started.md) for full setup.

---

## What to work on

The [optimization_tasks.md](optimization_tasks.md) roadmap lists every planned task with full problem descriptions and test cases. Tasks marked `[ ]` are open. Pick one and comment on the issue before starting so we don't duplicate effort.

Good first tasks: anything marked `S` (1–2 days effort) that is `[ ]` pending.

---

## Development workflow

1. Fork and clone with `--recursive` (llama.cpp is a submodule)
2. Create a branch: `git checkout -b feat/my-change`
3. Make your change
4. Run tests (see below)
5. Open a PR against `main`

---

## Running tests

### C++ tests
```bash
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure -j$(nproc)
```

### Go tests (pure-Go, no GPU needed)
```bash
cd go
go test -race ./hub/... ./grpc/... ./client/... ./tracker/... ./analytics/...
go test ./server/...   # CGO_ENABLED=0
```

### All tests
```bash
# C++
ctest --test-dir build --output-on-failure

# Go (full, requires CGo + ONNX Runtime)
cd go && go test -race ./...
```

---

## Code style

### Go
- `gofmt` before committing (CI enforces this via `golangci-lint`)
- `go vet` must be clean
- No unused imports or variables

### C++
- `clang-format` before committing:
  ```bash
  find cpp -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i
  ```
- C++17, no exceptions in hot paths
- RAII for all resource management — no raw `new`/`delete`

### Commits
- Format: `type(scope): description`
  - `feat` — new feature
  - `fix` — bug fix
  - `perf` — performance improvement
  - `bench` — benchmark results
  - `docs` — documentation only
  - `refactor` — no behavior change
  - `test` — test-only change
  - `ci` — CI / tooling
- Keep commits focused — one logical change per commit
- No AI attribution in commit messages

---

## Adding a new backend

New hardware backends (ONNX Runtime execution providers, new quantization formats, etc.) follow this pattern:

1. C++ session class in `cpp/onnx/` or `cpp/torch/` implementing the `InferSession` interface from `cpp/include/infer_api.h`
2. CGo binding in `go/onnx/` or `go/torch/`
3. Registration in `go/server/router.go` under the `--backend` flag
4. Test cases following the pattern in `cpp/onnx/onnx_session_test.cpp`
5. Documentation in `docs/`

---

## Adding a native SDK

New language bindings go in `sdk/<language>/`. Follow the structure of an existing SDK (e.g., `sdk/python/` or `sdk/rust/`):

- Thin wrapper around the C API (`sdk/c/`)
- README with install + quickstart
- At least one example covering chat, embed, and detect
- Publish to the language's package registry if possible

---

## PR checklist

- [ ] C++ changes: `clang-format` applied
- [ ] Go changes: `gofmt` + `go vet` clean
- [ ] Tests pass locally: `ctest` + `go test`
- [ ] New features have test cases
- [ ] Performance changes include benchmark numbers

---

## License

By contributing you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
