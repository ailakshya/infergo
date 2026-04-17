# Native SDK Reference — 14 Languages, Zero HTTP Overhead

infergo provides native SDK bindings for 14 languages. Every SDK links directly to `libinfer_api.so` — the same C library that powers the Go server. No HTTP, no serialization, no network. Just a function call.

## Transport Options

| You are... | Use | Overhead |
|---|---|---|
| Building a C/C++/Rust/Go/Zig app | Direct link (`-linfer_api`) | **0.001ms** |
| Python/Ruby/PHP/Lua/Dart/Elixir on GPU machine | ctypes/FFI to `libinfer_api.so` | **0.01ms** |
| Java/Kotlin/C#/Swift on GPU machine | JNI/P-Invoke/C bridge | **0.01ms** |
| Node.js on GPU machine | N-API addon | **0.01ms** |
| Browser/WASM | Emscripten (CPU only) | **0.1ms** |
| Any language, different machine | HTTP SDK or [Transport Guide](transport.md) | **1-4ms** |

## Prerequisites

All native SDKs require `libinfer_api.so` built from the infergo C++ engine:

```bash
git clone https://github.com/ailakshya/infergo && cd infergo

# CPU build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target infer_api -j$(nproc)

# CUDA build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build --target infer_api -j$(nproc)

# Install system-wide (optional)
sudo cp build/cpp/api/libinfer_api.so /usr/local/lib/
sudo cp cpp/include/infer_api.h /usr/local/include/
sudo ldconfig
```

Set `INFERGO_LIB_DIR` to the directory containing `libinfer_api.so` if not installed system-wide.

---

## 1. C / C++ SDK

**Directory:** `sdk/c/`

### Install

```bash
# Option 1: pkg-config
export PKG_CONFIG_PATH=/path/to/infergo/build

# Option 2: CMake
cmake -DINFERGO_LIB_DIR=/path/to/build/cpp/api \
      -DINFERGO_INCLUDE_DIR=/path/to/cpp/include ..

# Option 3: Direct link
gcc myapp.c -I/path/to/cpp/include -L/path/to/build/cpp/api -linfer_api -o myapp
```

### C Example

```c
#include "infer_api.h"

int main() {
    InferLLM llm = infer_llm_create("model.gguf", -1, 4096, 1, 2048);

    int tokens[512];
    int n = infer_llm_tokenize(llm, "Hello!", 1, tokens, 512);

    char buf[8192];
    int gen = 0;
    infer_llm_generate(llm, tokens, n, 128, 0.7f, 0.9f,
                       NULL, NULL, NULL, buf, sizeof(buf), &gen);
    printf("%s\n", buf);

    infer_llm_destroy(llm);
}
```

### C++ Example (RAII)

```cpp
#include "infergo.h"  // sdk/c/infergo.h — single-include C++ header

int main() {
    infergo::LLM llm("model.gguf");
    std::string response = llm.generate("What is C++?");
    std::cout << response << std::endl;

    infergo::Embedding embed("model.onnx", "tokenizer.json");
    auto vec = embed.embed("hello world");

    infergo::VectorDB db(384);
    db.insert(1, vec, "hello world");
    auto results = db.search(vec, 5);

    infergo::BM25 bm25;
    bm25.insert(1, "hello world");
    auto hits = bm25.search("hello", 10);
}
```

### API

| Class | Methods |
|---|---|
| `infergo::LLM` | `generate(prompt, max_tokens, temp)`, `tokenize(text)`, `vocab_size()` |
| `infergo::Embedding` | `embed(text)`, `embed_batch(texts)`, `rerank(query, docs)` |
| `infergo::VectorDB` | `insert(id, vec, meta)`, `search(query, k)`, `save(path)`, `load(path)` |
| `infergo::BM25` | `insert(id, text)`, `search(query, k)`, `save(path)`, `load(path)` |
| `infergo::Session` | `load(model_path)`, `run(inputs) -> outputs` |
| `infergo::Tokenizer` | `encode(text) -> ids`, `decode(ids) -> text` |

All classes use RAII — resources are freed in the destructor.

---

## 2. Python SDK (ctypes)

**Directory:** `sdk/python-native/`

### Install

```bash
pip install numpy  # required dependency
export INFERGO_LIB_DIR=/path/to/build/cpp/api
```

### Usage

```python
from infergo_native import LLM, Embedding, VectorDB, BM25, LoRA

# LLM generation
with LLM("model.gguf", gpu_layers=-1) as llm:
    text = llm.generate("What is Python?", max_tokens=128, temperature=0.7)
    print(text)

    tokens = llm.tokenize("Hello")  # returns list[int]
    print(f"Vocab size: {llm.vocab_size}")

# Embedding
with Embedding("model.onnx", "tokenizer.json") as embed:
    vec = embed.embed("hello world")          # numpy float32 array
    vecs = embed.embed_batch(["a", "b", "c"]) # numpy 2D array

# Vector search
with VectorDB(dim=384) as db:
    db.insert(1, vec, metadata="hello world")
    results = db.search(vec, k=10)  # [(id, distance), ...]
    db.save("vectors.db")

# BM25 keyword search
with BM25() as bm25:
    bm25.insert(1, "hello world")
    results = bm25.search("hello", k=10)  # [(id, score), ...]

# LoRA adapters
with LLM("model.gguf") as llm:
    lora = LoRA(llm, "adapter.bin")
    lora.apply(scale=1.0)
    text = llm.generate("Hello with LoRA")
    lora.remove()
    lora.close()
```

### API

| Class | Methods |
|---|---|
| `LLM` | `generate(prompt, max_tokens, temperature, top_p, grammar)`, `tokenize(text)`, `vocab_size` |
| `Embedding` | `embed(text) -> np.ndarray`, `embed_batch(texts) -> np.ndarray` |
| `VectorDB` | `insert(id, vec, meta)`, `search(query, k, ef_search, filter)`, `delete(id)`, `save(path)`, `load(path)` |
| `BM25` | `insert(id, text)`, `search(query, k)`, `len(bm25)` |
| `LoRA` | `apply(scale)`, `remove()`, `close()` |

All classes support `with` statement (context managers) and explicit `close()`.

---

## 3. Rust SDK

**Directory:** `sdk/rust/`

### Install

```toml
# Cargo.toml
[dependencies]
infergo = { path = "sdk/rust/infergo" }

# Or from local registry
# infergo = "1.0"
```

```bash
export INFERGO_LIB_DIR=/path/to/build/cpp/api
```

### Usage

```rust
use infergo::{Llm, Embedding, VectorDb, Bm25};

fn main() -> Result<(), infergo::Error> {
    // LLM
    let llm = Llm::new("model.gguf", -1, 4096, 1, 2048)?;
    let result = llm.generate("What is Rust?", 128, 0.7, 0.9, None)?;
    println!("{}", result.text);

    // Embedding
    let embed = Embedding::new("model.onnx", "tokenizer.json", "cpu", 0)?;
    let vec = embed.embed("hello world")?;

    // Vector DB
    let mut db = VectorDb::new(384, 16, 200)?;
    db.insert(1, &vec, "hello world")?;
    let results = db.search(&vec, 10, 50, None)?;

    // BM25
    let mut bm25 = Bm25::new(1.2, 0.75)?;
    bm25.insert(1, "hello world");
    let hits = bm25.search("hello", 10)?;

    Ok(())
}
```

### API

| Struct | Methods |
|---|---|
| `Llm` | `new()`, `generate()`, `tokenize()`, `token_to_piece()`, `vocab_size()`, `bos()`, `eos()` |
| `Embedding` | `new()`, `embed()`, `embed_batch()`, `rerank()` |
| `VectorDb` | `new()`, `insert()`, `delete()`, `update()`, `search()`, `save()`, `load()`, `size()` |
| `Bm25` | `new()`, `insert()`, `remove()`, `search()`, `save()`, `load()`, `size()` |

All types implement `Drop` (RAII). All types are `Send + Sync` (thread-safe). Error handling via `thiserror`.

---

## 4. Java / Kotlin SDK

**Directory:** `sdk/java/`

### Build

```bash
cd sdk/java
gradle build         # builds JNI native lib + Java classes
gradle runChat --args="model.gguf"
```

### Usage

```java
import com.infergo.*;

// LLM
try (LLM llm = new LLM("model.gguf", -1, 4096)) {
    String response = llm.generate("What is Java?", 128, 0.7f);
    System.out.println(response);
    System.out.println("Vocab: " + llm.vocabSize());
}

// Embedding
try (Embedding embed = new Embedding("model.onnx", "tokenizer.json")) {
    float[] vec = embed.embed("hello world");
    float[][] batch = embed.embedBatch(new String[]{"a", "b", "c"});
}

// Vector DB
try (VectorDB db = new VectorDB(384, 16, 200)) {
    db.insert(1, vec, "hello world");
    VectorDB.SearchResult[] results = db.search(vec, 10, 50);
    db.save("vectors.db");
}

// BM25
try (BM25 bm25 = new BM25()) {
    bm25.insert(1, "hello world");
    BM25.SearchResult[] results = bm25.search("hello", 10);
}
```

All classes implement `AutoCloseable` — use with try-with-resources.

---

## 5. Node.js SDK (N-API)

**Directory:** `sdk/nodejs/`

### Install

```bash
cd sdk/nodejs
npm install   # builds native addon via node-gyp
```

### Usage

```javascript
const { LLM, Embedding, VectorDB, BM25 } = require('@infergo/native');

// LLM
const llm = new LLM('model.gguf', { gpuLayers: -1, ctxSize: 4096 });
const result = llm.generate('What is Node.js?', { maxTokens: 128, temperature: 0.7 });
console.log(result.text);
llm.destroy();

// Embedding
const embed = new Embedding('model.onnx', 'tokenizer.json');
const vec = embed.embed('hello world');    // Float32Array
const batch = embed.embedBatch(['a', 'b']); // [Float32Array, ...]
embed.destroy();

// Vector DB
const db = new VectorDB({ dim: 384 });
db.insert(1, vec, 'hello world');
const results = db.search(vec, { k: 10 });
db.destroy();

// BM25
const bm25 = new BM25();
bm25.insert(1, 'hello world');
const hits = bm25.search('hello', { k: 10 });
bm25.destroy();
```

Full TypeScript declarations included (`lib/index.d.ts`).

---

## 6. C# / .NET SDK

**Directory:** `sdk/dotnet/`

### Install

```bash
cd sdk/dotnet/src/Infergo
dotnet build
```

### Usage

```csharp
using Infergo;

// LLM
using var llm = new Llm("model.gguf", gpuLayers: -1, ctxSize: 4096);
string response = llm.Generate("What is C#?", maxTokens: 128, temperature: 0.7f);
Console.WriteLine(response);

// Streaming
llm.Generate(tokens, maxTokens: 128, temperature: 0.7f, callback: (token, piece) => {
    Console.Write(piece);
    return true;  // continue
});

// Embedding
using var embed = new Embedding("model.onnx", "tokenizer.json");
float[] vec = embed.Embed("hello world");

// Vector DB
using var db = new VectorDb(dim: 384);
db.Insert(1, vec, "hello world");
var results = db.Search(vec, k: 10);

// BM25
using var bm25 = new Bm25();
bm25.Insert(1, "hello world");
var hits = bm25.Search("hello", k: 10);
```

All classes implement `IDisposable` — use with `using` statement. Targets .NET 8.0.

---

## 7. Swift SDK

**Directory:** `sdk/swift/`

### Install

```swift
// Package.swift
dependencies: [
    .package(path: "sdk/swift")
]
```

```bash
swift build -Xlinker -L/path/to/build/cpp/api -Xcc -I/path/to/cpp/include
```

### Usage

```swift
import Infergo

// LLM
let llm = try LLM("model.gguf", gpuLayers: -1, ctxSize: 4096)
let response = try llm.generate("What is Swift?", maxTokens: 128, temperature: 0.7)
print(response)

// Streaming
try llm.generate("Tell a story", maxTokens: 256) { token, piece in
    print(piece, terminator: "")
    return true  // continue
}

// Embedding
let embed = try Embedding("model.onnx", tokenizerPath: "tokenizer.json")
let vec = try embed.embed("hello world")  // [Float]

// Vector DB
let db = try VectorDB(dim: 384)
try db.insert(1, vector: vec, metadata: "hello")
let results = try db.search(vec, k: 10)  // [SearchResult]

// BM25
let bm25 = try BM25()
bm25.insert(1, text: "hello world")
let hits = try bm25.search("hello", k: 10)
```

RAII via `deinit`. All operations `throw InfergoError` on failure.

---

## 8. Ruby SDK

**Directory:** `sdk/ruby/`

### Install

```bash
gem install ffi  # dependency
export INFERGO_LIB_DIR=/path/to/build/cpp/api
```

### Usage

```ruby
require 'infergo'

# LLM
llm = Infergo::LLM.new("model.gguf", gpu_layers: -1)
puts llm.generate("What is Ruby?", max_tokens: 128)
puts "Vocab: #{llm.vocab_size}"
llm.close

# Embedding
embed = Infergo::Embedding.new("model.onnx", "tokenizer.json")
vec = embed.embed("hello world")  # Array of Float
embed.close

# Vector DB
db = Infergo::VectorDB.new(dim: 384)
db.insert(1, vec, metadata: "hello")
results = db.search(vec, k: 10)  # [{id: 1, distance: 0.0}]
db.close

# BM25
bm25 = Infergo::BM25.new
bm25.insert(1, "hello world")
results = bm25.search("hello", k: 10)
bm25.close
```

Cleanup via finalizers and explicit `close()`.

---

## 9. PHP SDK

**Directory:** `sdk/php/`

### Install

```bash
# Requires PHP 7.4+ with FFI extension
composer require infergo/infergo
export INFERGO_LIB_DIR=/path/to/build/cpp/api
```

### Usage

```php
<?php
use Infergo\{LLM, Embedding, VectorDB, BM25};

// LLM
$llm = new LLM("model.gguf", gpuLayers: -1);
echo $llm->generate("What is PHP?", maxTokens: 128) . "\n";
echo "Vocab: " . $llm->vocabSize() . "\n";
$llm->close();

// Embedding
$embed = new Embedding("model.onnx", "tokenizer.json");
$vec = $embed->embed("hello world");  // array of float
$embed->close();

// Vector DB
$db = new VectorDB(dim: 384);
$db->insert(1, $vec, "hello");
$results = $db->search($vec, k: 10);
$db->close();

// BM25
$bm25 = new BM25();
$bm25->insert(1, "hello world");
$results = $bm25->search("hello", k: 10);
$bm25->close();
```

Cleanup via `__destruct()` and explicit `close()`.

---

## 10. Dart / Flutter SDK

**Directory:** `sdk/dart/`

### Install

```yaml
# pubspec.yaml
dependencies:
  infergo:
    path: sdk/dart
```

### Usage

```dart
import 'package:infergo/infergo.dart';

// LLM
final llm = LLM('model.gguf', gpuLayers: -1);
print(llm.generate('What is Dart?', maxTokens: 128));
print('Vocab: ${llm.vocabSize}');
llm.close();

// Embedding
final embed = Embedding('model.onnx', 'tokenizer.json');
final vec = embed.embed('hello world');  // Float32List
embed.close();

// Vector DB
final db = VectorDB(dim: 384);
db.insert(1, vec, metadata: 'hello');
final results = db.search(vec, k: 10);
db.close();

// BM25
final bm25 = BM25();
bm25.insert(1, 'hello world');
final hits = bm25.search('hello', k: 10);
bm25.close();
```

---

## 11. Zig SDK

**Directory:** `sdk/zig/`

### Build

```bash
cd sdk/zig
zig build -Doptimize=ReleaseFast
```

### Usage

```zig
const infergo = @import("infergo");

pub fn main() !void {
    var llm = try infergo.LLM.init("model.gguf", -1, 4096, 1, 2048);
    defer llm.deinit();

    var tok_buf: [4096]i32 = undefined;
    const n = try llm.tokenize("What is Zig?", true, &tok_buf);

    var out: [8192]u8 = undefined;
    const text = try llm.generate(tok_buf[0..n], 128, 0.7, 0.9, &out);
    std.debug.print("{s}\n", .{text});

    // Vector DB
    var db = try infergo.VectorDB.init(384, 16, 200);
    defer db.deinit();

    // BM25
    var bm25 = try infergo.BM25.init(1.2, 0.75);
    defer bm25.deinit();
}
```

Uses `@cImport` for zero-cost C interop. `deinit()` for cleanup. Zig error unions for error handling.

---

## 12. Elixir / Erlang SDK

**Directory:** `sdk/elixir/`

### Install

```bash
cd sdk/elixir
mix deps.get && mix compile
```

### Usage

```elixir
# Load model
{:ok, llm} = Infergo.llm_create("model.gguf", gpu_layers: -1)

# Generate
{:ok, text} = Infergo.generate(llm, "What is Elixir?", max_tokens: 128)
IO.puts(text)
```

Uses Erlang NIFs for native interop. Resource types ensure cleanup on garbage collection.

---

## 13. Lua SDK (LuaJIT)

**Directory:** `sdk/lua/`

### Usage

```lua
local infergo = require("infergo")

-- LLM
local llm = infergo.LLM("model.gguf")
print(llm:generate("What is Lua?", 128))
llm:close()

-- Embedding
local embed = infergo.Embedding("model.onnx", "tokenizer.json")
local vec = embed:embed("hello world")
embed:close()

-- Vector DB
local db = infergo.VectorDB(384)
db:insert(1, vec, "hello")
local results = db:search(vec, 10)
db:close()

-- BM25
local bm25 = infergo.BM25()
bm25:insert(1, "hello world")
local hits = bm25:search("hello", 10)
bm25:close()
```

Uses LuaJIT FFI. Metatables with `__gc` for automatic cleanup.

---

## 14. WASM / JavaScript SDK

**Directory:** `sdk/wasm/`

### Build

```bash
cd sdk/wasm
# Requires Emscripten
make
```

### Usage

```javascript
const { Infergo } = require('@infergo/wasm');

const infergo = await Infergo.load('infergo.wasm');

// LLM (CPU only in browser)
const llm = infergo.createLLM('model.gguf', 0, 2048);
const text = llm.generate('What is WASM?', 64, 0.7);
console.log(text);
llm.destroy();

// Vector DB
const db = infergo.createVectorDB(384);
db.insert(1, new Float32Array(384), 'hello');
db.destroy();

// BM25
const bm25 = infergo.createBM25();
bm25.insert(1, 'hello world');
bm25.destroy();
```

CPU-only inference. Suitable for edge/offline use cases where the model fits in memory.

---

## Common API Surface

All SDKs expose the same core capabilities:

| Capability | C Function | Description |
|---|---|---|
| **LLM** | `infer_llm_create` | Load GGUF model |
| | `infer_llm_generate` | Full generation loop (tokenize + decode + sample) |
| | `infer_llm_tokenize` | Text to token IDs |
| | `infer_llm_destroy` | Free model |
| **Embedding** | `infer_embed_pipeline` | Text to normalized vector |
| | `infer_embed_batch_pipeline` | Batch embedding |
| **Vector DB** | `infer_vectordb_create` | Create HNSW index |
| | `infer_vectordb_insert` | Insert vector + metadata |
| | `infer_vectordb_search` | k-NN search with optional filter |
| | `infer_vectordb_save/load` | Persist/restore index |
| **BM25** | `infer_bm25_create` | Create full-text index |
| | `infer_bm25_insert` | Index document |
| | `infer_bm25_search` | Keyword search |
| **RAG** | `infer_rag_pipeline` | End-to-end retrieval-augmented generation |
| **Rerank** | `infer_rerank_pipeline` | Rerank documents by query relevance |
| **LoRA** | `infer_lora_load/apply/free` | Load and hot-swap LoRA adapters |

See [C API Reference](c-api-reference.md) for the full function list.

---

## Thread Safety

All C handles are thread-safe for concurrent read access. Write operations (generate, insert) should be serialized per handle. Create separate handles for parallel write workloads.

## Error Handling

Every SDK maps C error codes to native exceptions/errors:
- C: return codes + `infer_last_error_string()`
- C++/Rust/Swift/Zig: exceptions/Result types
- Python/Ruby/PHP/Lua: language exceptions
- Java/C#: checked exceptions / `InfergoException`
- Node.js: thrown Error
- Elixir: `{:error, reason}` tuples
