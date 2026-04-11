# Object Detection Guide

infergo serves YOLOv8 and YOLOv11 object detection models over HTTP with GPU-accelerated preprocessing, multiple inference backends, and adaptive routing.

## Quick start

```bash
# Download and convert model
python tools/convert_to_torchscript.py --source yolo11n --output models/yolo11n.torchscript.pt

# Start server
./infergo serve \
  --model yolo11n:models/yolo11n.torchscript.pt \
  --provider cuda --backend torch --port 9090

# Detect objects (binary — fastest)
curl -X POST "http://localhost:9090/v1/detect/binary?model=yolo11n" \
  -H "Content-Type: image/jpeg" \
  --data-binary @photo.jpg

# Detect objects (JSON — compatible with any client)
curl -X POST http://localhost:9090/v1/detect \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"yolo11n\", \"image_b64\": \"$(base64 -w0 photo.jpg)\"}"
```

## Endpoints

### `POST /v1/detect/binary` (recommended)

Send raw JPEG/PNG bytes directly. No base64 encoding, no JSON overhead. ~2ms faster per request.

```
POST /v1/detect/binary?model=yolo11n&conf=0.25&iou=0.45
Content-Type: image/jpeg

<raw JPEG bytes>
```

**Parameters (query string):**
- `model` (required) — model name
- `conf` — confidence threshold (default: 0.25)
- `iou` — IoU threshold for NMS (default: 0.45)

### `POST /v1/detect` (JSON)

Standard JSON endpoint with base64-encoded image. Compatible with any HTTP client.

```json
{
  "model": "yolo11n",
  "image_b64": "<base64 encoded JPEG/PNG>",
  "conf_thresh": 0.25,
  "iou_thresh": 0.45
}
```

### Response format (both endpoints)

```json
{
  "model": "yolo11n",
  "objects": [
    {"X1": 10.5, "Y1": 20.3, "X2": 150.7, "Y2": 200.1, "ClassID": 0, "Confidence": 0.92},
    {"X1": 300.0, "Y1": 100.0, "X2": 450.0, "Y2": 350.0, "ClassID": 2, "Confidence": 0.87}
  ]
}
```

ClassID maps to COCO 80-class labels (0=person, 1=bicycle, 2=car, ...).

## Backends

### Choose your backend

| Backend | Best for | Command |
|---|---|---|
| `torch` | General use, low VRAM | `--backend torch --model name:model.torchscript.pt` |
| `tensorrt` | Max throughput | `--backend tensorrt --model name:model.onnx` |
| `adaptive` | Production (auto-optimizes) | `--backend adaptive --model name:model.torchscript.pt` |
| `onnx` | Compatibility | `--backend onnx --model name:model.onnx` |

### Adaptive backend

The adaptive backend loads multiple inference engines and routes each request to the optimal one:

| Queue depth | Backend used | Why |
|---|---|---|
| 0-1 | libtorch single | Lowest per-image latency |
| 2-8 | libtorch batch | Amortizes Go-C overhead across N images |
| 8+ | TensorRT | Highest raw throughput |

Enable with `--backend adaptive` or `--adaptive`. Requires both `.torchscript.pt` and `.onnx` files for the same model (auto-discovered from sibling files).

### Safe mode

`--safe-mode` disables all batching and adaptive routing. Uses single-image libtorch only. Useful for debugging.

## Multi-model serving

```bash
./infergo serve \
  --model yolo11n:models/yolo11n.torchscript.pt \
  --model yolo11s:models/yolo11s.torchscript.pt \
  --model yolo11m:models/yolo11m.torchscript.pt \
  --model yolo11l:models/yolo11l.torchscript.pt \
  --provider cuda --backend torch
```

All models share one CUDA memory pool (libtorch's caching allocator). Idle VRAM: ~500MB for 4 models.

## Model conversion

### From ultralytics (recommended)

```bash
# Single model
python tools/convert_to_torchscript.py --source yolo11n --output models/yolo11n.torchscript.pt

# All sizes
python tools/convert_to_torchscript.py --batch yolo11n,yolo11s,yolo11m,yolo11l --output-dir models/
```

### From ONNX

```bash
# ONNX models work directly with onnx/tensorrt backends
./infergo serve --model yolo11n:models/yolo11n.onnx --backend tensorrt --provider cuda
```

### Manual TorchScript export

```python
from ultralytics import YOLO
YOLO("yolo11n.pt").export(format="torchscript", imgsz=640)
# Creates yolo11n.torchscript in the same directory
```

## Python client

```python
import requests, base64

# Binary endpoint (fastest)
with open("photo.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:9090/v1/detect/binary?model=yolo11n",
        data=f.read(),
        headers={"Content-Type": "image/jpeg"},
    )

# JSON endpoint (compatible)
with open("photo.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
resp = requests.post("http://localhost:9090/v1/detect", json={
    "model": "yolo11n",
    "image_b64": b64,
})

for obj in resp.json()["objects"]:
    print(f"Class {obj['ClassID']}: {obj['Confidence']:.2f} at ({obj['X1']:.0f},{obj['Y1']:.0f})-({obj['X2']:.0f},{obj['Y2']:.0f})")
```

## Performance

Measured on RTX 5070 Ti (16GB), YOLOv11 n/s/m/l, 640x480 JPEG.

### Throughput (4 models, concurrent load)

| Clients | Python PyTorch | infergo (adaptive) |
|---|---|---|
| 4 | 157 req/s | **304 req/s (1.9x)** |
| 16 | 151 req/s | **304 req/s (2.0x)** |
| 64 | 164 req/s | **295 req/s (1.8x)** |

### Latency (P50, 16 concurrent clients)

| Model | Python | infergo | Speedup |
|---|---|---|---|
| yolo11n | 93ms | **35ms** | 2.7x |
| yolo11s | 77ms | **36ms** | 2.1x |
| yolo11m | 105ms | **60ms** | 1.8x |
| yolo11l | 99ms | **70ms** | 1.4x |

### Raw per-image (no HTTP, single image)

| Model | Python | Go (libtorch) | C++ (raw) |
|---|---|---|---|
| yolo11n | 2.6ms | 4.8ms | 2.6ms |
| yolo11s | 3.0ms | 5.4ms | — |
| yolo11m | 4.9ms | 7.7ms | — |
| yolo11l | 6.6ms | 9.5ms | — |

Go adds ~2ms per image due to Go runtime CGo overhead. Under concurrent load, Go's goroutine concurrency more than compensates.

### Binary vs JSON endpoint

| Endpoint | yolo11n P50 | Payload size |
|---|---|---|
| `/v1/detect` (JSON+base64) | 10.8ms | 481KB |
| `/v1/detect/binary` (raw JPEG) | **8.6ms** | **360KB** |

Binary saves ~2ms per request by eliminating base64 encoding/decoding overhead.

## GPU memory

| Setup | VRAM |
|---|---|
| 4 models idle (libtorch) | **503 MB** |
| 4 models under load (libtorch) | 7.4 GB |
| 4 models under load (TensorRT) | 6.9 GB |
| 4 models under load (adaptive) | 8.5 GB |
| RTX 5070 Ti total | 16 GB |

libtorch shares one CUDA memory pool across all models. TensorRT pre-allocates per-engine.
