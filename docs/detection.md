# Object Detection Guide

infergo serves YOLOv8 and YOLOv11 object detection models over HTTP with GPU-accelerated preprocessing, GPU NMS via CUDA kernels, multiple inference backends, adaptive routing, multi-stream batching, and ByteTrack object tracking.

## Quick start

```bash
# Download and convert model
python tools/convert_to_torchscript.py --source yolo11n --output models/yolo11n.torchscript.pt

# Start server
./infergo serve \
  --model yolo11n:models/yolo11n.torchscript.pt \
  --provider cuda --backend torch --port 9090

# Detect objects (binary -- fastest)
curl -X POST "http://localhost:9090/v1/detect/binary?model=yolo11n" \
  -H "Content-Type: image/jpeg" \
  --data-binary @photo.jpg

# Detect objects (JSON -- compatible with any client)
curl -X POST http://localhost:9090/v1/detect \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"yolo11n\", \"image_b64\": \"$(base64 -w0 photo.jpg)\"}"
```

## Endpoints

### `POST /v1/detect/binary` (recommended)

Send raw JPEG/PNG bytes directly. No base64 encoding, no JSON overhead. ~2ms faster per request.

```
POST /v1/detect/binary?model=yolo11n&conf=0.25&iou=0.45&max_det=300&classes=0,2
Content-Type: image/jpeg

<raw JPEG bytes>
```

**Parameters (query string):**

| Parameter | Default | Description |
|---|---|---|
| `model` | (required) | Model name |
| `conf` | `0.25` | Confidence threshold |
| `iou` | `0.45` | IoU threshold for NMS |
| `max_det` | `300` | Maximum detections to return |
| `classes` | (all) | Comma-separated class IDs to filter (e.g. `0,2,5`) |

### `POST /v1/detect` (JSON)

Standard JSON endpoint with base64-encoded image.

```json
{
  "model": "yolo11n",
  "image_b64": "<base64 encoded JPEG/PNG>",
  "conf_thresh": 0.25,
  "iou_thresh": 0.45,
  "max_det": 300,
  "classes": [0, 2],
  "backend": "torch-gpu"
}
```

| Field | Default | Description |
|---|---|---|
| `model` | (required) | Model name |
| `image_b64` | (required) | Base64-encoded image bytes |
| `conf_thresh` | `0.25` | Minimum confidence threshold |
| `iou_thresh` | `0.45` | IoU threshold for NMS |
| `max_det` | `300` | Maximum detections to return |
| `classes` | (all) | Filter to specific class IDs (empty = all 80 COCO classes) |
| `backend` | (auto) | Per-request backend override: `torch-gpu`, `onnx-cuda`, `cpu` |

### `POST /v1/detect/stream`

Streaming detection for video frames. Accepts a sequence of frames and returns detections per frame.

### Response format (all endpoints)

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

---

## GPU NMS CUDA kernel

infergo includes a custom CUDA kernel (`infer_nms_cuda`) that runs the entire NMS pipeline on GPU:

1. **Confidence filter** -- reject detections below threshold on GPU
2. **Sort** -- sort by confidence descending on GPU
3. **IoU computation** -- compute pairwise IoU on GPU
4. **Greedy suppression** -- class-aware suppression on GPU
5. **Copy to host** -- only the final kept detections cross the PCIe bus

This eliminates the CPU bottleneck where NMS traditionally runs. The GPU NMS kernel processes all detections in a single launch, keeping data on-device until the final results.

### C API

```c
InferError infer_nms_cuda(
    const float* d_boxes,    // device pointer: N * 6 floats [x1,y1,x2,y2,conf,class]
    int n_boxes,
    float conf_thresh,
    float iou_thresh,
    InferBox* out_boxes,     // host buffer for results
    int max_out,
    int* out_count,
    void* stream             // CUDA stream (NULL = default)
);
```

---

## Backends

### Choose your backend

| Backend | Best for | Command |
|---|---|---|
| `torch` | General use, low VRAM | `--backend torch --model name:model.torchscript.pt` |
| `tensorrt` | Max throughput | `--backend tensorrt --model name:model.onnx` |
| `adaptive` | Production (auto-optimizes) | `--backend adaptive --model name:model.torchscript.pt` |
| `onnx` | Compatibility | `--backend onnx --model name:model.onnx` |

### Adaptive backend selection

The adaptive backend loads multiple inference engines and routes each request to the optimal one based on current queue depth:

| Queue depth | Backend used | Why |
|---|---|---|
| 0-1 | libtorch single | Lowest per-image latency |
| 2-8 | libtorch batch | Amortizes Go-C overhead across N images |
| 8+ | TensorRT | Highest raw throughput |

Enable with `--backend adaptive` or `--adaptive`. Requires both `.torchscript.pt` and `.onnx` files for the same model (auto-discovered from sibling files).

Configuration flags:
- `--batch-threshold N` -- queue depth at which batch mode activates (default: 3)
- `--detect-gpu-slots N` -- max concurrent GPU detection slots (default: 8)
- `--warmup-backends` -- warm up all backends with dummy inferences at startup (default: true)

### Safe mode

`--safe-mode` disables all batching and adaptive routing. Uses single-image libtorch only. Useful for debugging.

---

## Multi-stream batching

When multiple detection requests arrive simultaneously, infergo batches them into a single GPU forward pass:

```c
// C API for batch detection
int infer_torch_detect_gpu_batch(
    InferTorchSession s,
    const void** jpeg_data_array, const int* nbytes_array, int batch_size,
    float conf_thresh, float iou_thresh,
    InferBox** out_boxes_array, int* out_counts, int max_boxes_per_image
);
```

Benefits:
- Amortizes CGo/C++ overhead across N images
- Single GPU forward pass for the batch
- Automatic batch assembly based on queue depth

### Raw pixel and YUV detection

For video pipelines, bypass JPEG encoding entirely:

```c
// From raw RGB pixels (no JPEG overhead)
int infer_torch_detect_gpu_raw(s, rgb_data, width, height, ...);

// From NV12/YUV frames (zero CPU color conversion)
int infer_torch_detect_gpu_yuv(s, yuv_data, width, height, linesize, ...);
```

The YUV path performs NV12->RGB conversion on GPU, making it the fastest path for video pipeline integration.

---

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

---

## ByteTrack tracking integration

infergo includes a full ByteTrack implementation for multi-object tracking across video frames.

### Go API

```go
import "github.com/ailakshya/infergo/tracker"

cfg := tracker.DefaultConfig()
bt := tracker.NewByteTracker(cfg)

// Per frame:
detections := []tracker.Detection{
    {X1: 10, Y1: 20, X2: 100, Y2: 200, ClassID: 0, Confidence: 0.95},
}
tracks := bt.Update(detections)

for _, t := range tracks {
    fmt.Printf("Track %d: class=%d at (%.0f,%.0f)-(%.0f,%.0f)\n",
        t.TrackID, t.ClassID, t.X1, t.Y1, t.X2, t.Y2)
}
```

### Configuration

```go
type Config struct {
    TrackHighThresh float64 // High confidence threshold (default 0.20)
    TrackLowThresh  float64 // Low confidence threshold (default 0.08)
    NewTrackThresh  float64 // Min confidence for new tracks (default 0.25)
    TrackBuffer     int     // Max frames to keep lost tracks (default 60)
    MatchThresh     float64 // IoU threshold for matching (default 0.80)
    FuseScore       bool    // Fuse confidence with IoU distance (default true)
}
```

### How ByteTrack works

ByteTrack uses a 3-stage association strategy:

1. **Primary association** -- match high-confidence detections to existing tracks using IoU
2. **Secondary association** -- match low-confidence detections to unmatched tracks (recovers occluded objects)
3. **New track creation** -- remaining high-confidence detections initialize new tracks

Lost tracks are kept alive for `TrackBuffer` frames and can be re-associated if the object reappears.

### Detection control center

For video surveillance with multi-camera support:

```bash
./infergo detect \
  --model yolo11n:models/yolo11n.torchscript.pt \
  --source rtsp://cam1,rtsp://cam2 \
  --zones zones.yaml \
  --webhook http://alert-service:8080
```

---

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
./infergo serve --model yolo11n:models/yolo11n.onnx --backend tensorrt --provider cuda
```

### Manual TorchScript export

```python
from ultralytics import YOLO
YOLO("yolo11n.pt").export(format="torchscript", imgsz=640)
```

### Using the convert command

```bash
./infergo convert --input model.pt --format torchscript --output model.torchscript.pt --imgsz 640
./infergo convert --input model.pt --format onnx --output model.onnx
```

---

## Python client

### Using the infergo SDK

```python
from infergo import InfergoClient
import base64

client = InfergoClient("http://localhost:9090")

# Detection
with open("photo.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
    detections = client.detect(b64, model="yolo11n", conf=0.25)

for obj in detections:
    print(f"Class {obj['ClassID']}: {obj['Confidence']:.2f}")
```

### Using requests directly

```python
import requests, base64

# Binary endpoint (fastest)
with open("photo.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:9090/v1/detect/binary?model=yolo11n",
        data=f.read(),
        headers={"Content-Type": "image/jpeg"},
    )

# JSON endpoint
with open("photo.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
resp = requests.post("http://localhost:9090/v1/detect", json={
    "model": "yolo11n",
    "image_b64": b64,
    "conf_thresh": 0.25,
    "iou_thresh": 0.45,
    "max_det": 300,
    "classes": [0, 2],  # person and car only
})

for obj in resp.json()["objects"]:
    print(f"Class {obj['ClassID']}: {obj['Confidence']:.2f} "
          f"at ({obj['X1']:.0f},{obj['Y1']:.0f})-({obj['X2']:.0f},{obj['Y2']:.0f})")
```

---

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
| yolo11s | 3.0ms | 5.4ms | -- |
| yolo11m | 4.9ms | 7.7ms | -- |
| yolo11l | 6.6ms | 9.5ms | -- |

Go adds ~2ms per image due to Go runtime CGo overhead. Under concurrent load, Go's goroutine concurrency more than compensates.

### Binary vs JSON endpoint

| Endpoint | yolo11n P50 | Payload size |
|---|---|---|
| `/v1/detect` (JSON+base64) | 10.8ms | 481KB |
| `/v1/detect/binary` (raw JPEG) | **8.6ms** | **360KB** |

Binary saves ~2ms per request by eliminating base64 encoding/decoding overhead.

---

## GPU memory

| Setup | VRAM |
|---|---|
| 4 models idle (libtorch) | **503 MB** |
| 4 models under load (libtorch) | 7.4 GB |
| 4 models under load (TensorRT) | 6.9 GB |
| 4 models under load (adaptive) | 8.5 GB |
| RTX 5070 Ti total | 16 GB |

libtorch shares one CUDA memory pool across all models. TensorRT pre-allocates per-engine.
