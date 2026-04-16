# Transport Guide — Choosing the Right Connection Method

infergo supports 4 transport methods. Choose based on where your app runs relative to the GPU.

## Decision Flowchart

```
Where is your app?
|
+-- Same process as infergo?
|   |
|   YES --> Direct C link (libinfer_api.so)
|           Overhead: 0.001ms
|           Languages: C, C++, Rust, Go (CGo), Python (ctypes)
|
+-- Same machine as GPU?
|   |
|   +-- Need absolute lowest latency?
|   |   |
|   |   YES --> Shared Memory
|   |           Overhead: 0.01ms (50 microseconds)
|   |           Languages: Any with mmap support
|   |
|   +-- Normal low latency is fine?
|       |
|       YES --> Unix Domain Socket
|               Overhead: 0.3ms
|               Languages: Any with socket support
|
+-- Different machine from GPU?
    |
    +-- Internal microservice?
    |   |
    |   YES --> gRPC (protobuf)
    |           Overhead: 1.5ms
    |           Languages: All (gRPC codegen)
    |
    +-- Public API / web app?
        |
        YES --> HTTP (JSON)
                Overhead: 4ms
                Languages: All (curl, fetch, any HTTP client)
```

## Performance Comparison

All measured on same machine (RTX 5070 Ti), same model (Qwen 1.5B Q4), 8-token generation:

| Transport | Overhead | Total latency | Throughput | Use case |
|---|---|---|---|---|
| **Direct C link** | 0.001ms | 28ms | ~35 req/s | Embedded in C/C++/Rust app |
| **Shared Memory** | 0.01ms | 28ms | ~35 req/s | Python/Java on GPU machine |
| **Unix Socket** | 0.3ms | 28ms | ~34 req/s | Any language on GPU machine |
| **gRPC** | 1.5ms | 30ms | ~32 req/s | Microservices on same network |
| **HTTP (JSON)** | 4ms | 32ms | ~30 req/s | Web apps, public APIs |

Note: For LLM generation (28ms+), transport overhead is <15% of total even for HTTP. Transport choice matters more for fast operations (embedding: 0.9ms, search: 0.03ms).

## 1. Direct C Link

Best for: C/C++/Rust/Go apps running on the GPU machine.

```c
#include "infer_api.h"

// Load model
InferLLM llm = infer_llm_create("model.gguf", -1, 4096, 1, 2048);

// Generate
int tokens[256];
int n = infer_llm_tokenize(llm, "Hello", 0, tokens, 256);
char buf[8192];
int gen;
infer_llm_generate(llm, tokens, n, 32, 0.7, 0.9, NULL, NULL, NULL, buf, 8192, &gen);
printf("%s\n", buf);

// Cleanup
infer_llm_destroy(llm);
```

Compile: `gcc myapp.c -linfer_api -o myapp`

### Python (ctypes)

```python
from ctypes import cdll, c_char_p, c_int, c_float, create_string_buffer

lib = cdll.LoadLibrary("libinfer_api.so")
llm = lib.infer_llm_create(b"model.gguf", -1, 4096, 1, 2048)

tokens = (c_int * 256)()
n = lib.infer_llm_tokenize(llm, b"Hello", 0, tokens, 256)

buf = create_string_buffer(8192)
gen = c_int()
lib.infer_llm_generate(llm, tokens, n, 32, c_float(0.7), c_float(0.9),
                        None, None, None, buf, 8192, gen)
print(buf.value.decode())
```

## 2. Shared Memory

Best for: Python/Java/Node apps on the same machine as the GPU server. Lowest possible overhead without linking the C library.

### Server

```bash
infergo serve --model llm:model.gguf --provider cuda --shm infergo_shm
```

### Python Client

```python
from infergo_native.shm_client import SHMClient

with SHMClient("infergo_shm") as client:
    response = client.generate("Hello world", max_tokens=32)
    print(response)
```

### How it works

```
Client process                    Server process
    |                                 |
    |  write request to mmap'd RAM    |
    |  ─────────────────────────>     |
    |  set slot.state = READY         |
    |                                 |  poll slots, find READY
    |                                 |  process on GPU
    |                                 |  write response to mmap'd RAM
    |  <─────────────────────────     |
    |  slot.state == DONE             |  set slot.state = DONE
    |  read response from RAM         |
    |  set slot.state = FREE          |
```

No network. No serialization. No syscalls in the data path. Just atomic state transitions on shared memory pages.

## 3. Unix Domain Socket

Best for: Any language on the same machine. Simpler than shared memory, still no TCP overhead.

### Server

```bash
infergo serve --model llm:model.gguf --provider cuda --uds /tmp/infergo.sock
```

### Python Client

```python
from infergo_native.uds_client import UDSClient

with UDSClient("/tmp/infergo.sock") as client:
    response = client.generate("Hello", max_tokens=32)
    print(response)
```

### Any language (raw socket)

```python
import socket, struct, json

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect("/tmp/infergo.sock")

# Send: [4-byte length][1-byte type=1][JSON payload]
payload = json.dumps({"prompt": "Hello", "max_tokens": 32}).encode()
msg = bytes([1]) + payload  # type 1 = GENERATE
sock.sendall(struct.pack("<I", len(msg)) + msg)

# Receive: [4-byte length][response]
resp_len = struct.unpack("<I", sock.recv(4))[0]
response = sock.recv(resp_len).decode()
print(response)
```

## 4. gRPC (Protobuf)

Best for: Microservices calling the GPU server over the network. Binary protocol, multiplexed connections, streaming.

### Server

```bash
infergo serve --model llm:model.gguf --provider cuda --grpc-port 9091
```

### Python Client

```python
import grpc
from infergo_pb2 import GenerateRequest
from infergo_pb2_grpc import InfergoStub

channel = grpc.insecure_channel("gpu-server:9091")
stub = InfergoStub(channel)

response = stub.Generate(GenerateRequest(
    prompt="Hello",
    max_tokens=32,
    temperature=0.7,
))
print(response.text)
```

## 5. HTTP (JSON)

Best for: Web apps, public APIs, any language on any platform. Universal compatibility.

### Server

```bash
infergo serve --model llm:model.gguf --provider cuda --port 9090
```

### Any language

```bash
curl http://gpu-server:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llm","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

Works with OpenAI Python SDK, LangChain, LlamaIndex — zero configuration.

## When Transport Doesn't Matter

For LLM generation (30ms+ per request), the transport overhead is noise:

| Transport | Overhead | % of 30ms request |
|---|---|---|
| HTTP | 4ms | 13% |
| gRPC | 1.5ms | 5% |
| UDS | 0.3ms | 1% |
| SHM | 0.01ms | 0.03% |

**Transport matters for fast operations:**

| Operation | Latency | HTTP overhead % | SHM overhead % |
|---|---|---|---|
| LLM (32 tok) | 30ms | 13% | 0.03% |
| Embedding | 0.9ms | 444% | 1% |
| Vector search | 0.03ms | 13,333% | 33% |
| BM25 search | 0.14ms | 2,857% | 7% |
| Cache hit | 0.1ms | 4,000% | 10% |

For embedding and search, shared memory makes a **massive** difference.

## Summary

| You are... | Use this | Overhead |
|---|---|---|
| C/C++ developer on GPU machine | Direct link | 0 |
| Python data scientist on GPU machine | Shared memory or ctypes | 0.01ms |
| Microservice on same network | gRPC | 1.5ms |
| Web app anywhere | HTTP | 4ms |
| Browser | HTTP or WebSocket | 4ms |
| Mobile app (on-device model) | Direct link (Swift/Kotlin FFI) | 0 |
