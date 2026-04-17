# infergo C/C++ SDK

Thin wrappers around the infergo C API (`libinfer_api`).

- **C examples** (`examples/`) — pure C programs using `infer_api.h` directly
- **C++ header** (`infergo.h`) — single-include RAII wrapper with `infergo::LLM`, `infergo::Embedding`, `infergo::VectorDB`, `infergo::BM25`

## Build examples

```bash
# Option 1: CMake
cmake -B build -DINFERGO_LIB_DIR=/path/to/lib -DINFERGO_INCLUDE_DIR=/path/to/include
cmake --build build

# Option 2: Direct compile (if libinfer_api is installed)
gcc examples/chat.c -linfer_api -o chat
gcc examples/embed.c -linfer_api -o embed
gcc examples/detect.c -linfer_api -o detect
gcc examples/rag.c -linfer_api -o rag
```

## C++ usage (infergo.h)

```cpp
#include "infergo.h"

// LLM
infergo::LLM llm("model.gguf");
std::string reply = llm.generate("What is 2+2?");

// Embedding + VectorDB
infergo::Embedding emb("model.onnx", "tokenizer.json");
auto vec = emb.embed("hello world");

infergo::VectorDB db(vec.size());
db.insert(0, vec, "hello world");
auto results = db.search(vec, 5);

// BM25
infergo::BM25 bm25;
bm25.insert(0, "hello world");
auto hits = bm25.search("hello", 5);

// RAG
std::string answer = infergo::rag_pipeline(llm, emb, db, "What is this?");
```

## pkg-config

```bash
gcc $(pkg-config --cflags --libs infergo) examples/chat.c -o chat
```

## API reference

See [`infer_api.h`](../../cpp/include/infer_api.h) for the complete C API.
See [`infergo.h`](infergo.h) for C++ RAII wrappers.
