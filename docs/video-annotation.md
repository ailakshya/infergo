# Video Annotation & Frame Processing

infergo provides C-accelerated frame annotation, JPEG encoding, and video decoding through the `video` package. These functions use TurboJPEG and OpenCV under the hood, achieving 5-10x speedup over pure Go equivalents.

## Quick start

```go
import "github.com/ailakshya/infergo/video"

// Open a video with hardware-accelerated decode (NVDEC)
dec, _ := video.OpenDecoder("camera.mp4", true)
defer dec.Close()

// Read frames resized to 1280x720 in C (4x less memcpy than full res)
rgb, info, _ := dec.NextFrameResized(1280, 720)

// Annotate with boxes + labels + JPEG encode in one C call
objects := []video.AnnotateObject{
    {X1: 100, Y1: 200, X2: 300, Y2: 400, ClassID: 0, Confidence: 0.95, TrackID: 1, Label: "person #1 95%"},
}
jpeg, _ := video.AnnotateFast(rgb, info.Width, info.Height, objects, 75)

// Or batch ALL drawing in a single C call (fastest path)
input := video.AnnotateFullInput{
    Boxes: objects,
    Lines: []video.AnnotateLine{
        {X1: 0, Y1: 500, X2: 1280, Y2: 500, R: 0, G: 255, B: 255, Thickness: 4},
    },
    Texts: []video.AnnotateText{
        {X: 10, Y: 10, Text: "CAM1 - BAGS: 5", R: 0, G: 255, B: 0, Scale: 2},
    },
    Rects: []video.AnnotateRect{
        {X1: 5, Y1: 5, X2: 300, Y2: 50, R: 0, G: 0, B: 0, Alpha: 180},
    },
}
jpeg, _ = video.AnnotateFull(rgb, 1280, 720, input, 640, 360, 75)
```

## Functions

### Video Decode

```go
// Open video — file, RTSP, or V4L2 device. hwAccel enables NVDEC GPU decode.
func OpenDecoder(url string, hwAccel bool) (*Decoder, error)

// Decode next frame at original resolution (returns RGB24 []byte).
func (d *Decoder) NextFrame() ([]byte, FrameInfo, error)

// Decode + resize in C before copying to Go. 4x less memcpy for 2560x1440 -> 1280x720.
func (d *Decoder) NextFrameResized(targetW, targetH int) ([]byte, FrameInfo, error)

func (d *Decoder) Width() int
func (d *Decoder) Height() int
func (d *Decoder) FPS() float64
func (d *Decoder) IsHWAccelerated() bool
func (d *Decoder) Close()
```

### Frame Annotation

```go
// Draw boxes + labels on RGB frame, return JPEG. Uses TurboJPEG (2-3ms at 1080p).
func AnnotateFast(rgb []byte, w, h int, objects []AnnotateObject, quality int) ([]byte, error)

// Batch ALL drawing + resize + JPEG in ONE C call. Fastest path for complex overlays.
// Accepts boxes, lines, polygons, text, and filled rects in a single struct.
func AnnotateFull(rgb []byte, w, h int, input AnnotateFullInput, outW, outH, quality int) ([]byte, error)

// Pure Go fallback (no C dependency). 5x slower but always available.
func Annotate(rgb []byte, w, h int, objects []AnnotateObject) ([]byte, error)
```

### JPEG & Resize

```go
// Resize RGB frame + JPEG encode in C. OpenCV SIMD resize + TurboJPEG.
func ResizeJPEG(rgb []byte, srcW, srcH, dstW, dstH, quality int) ([]byte, error)

// Combine two RGB frames side-by-side with status bar text, JPEG encode.
func CombineJPEG(rgb1 []byte, w1, h1 int, rgb2 []byte, w2, h2 int,
                  statusText string, targetW, targetH, quality int) ([]byte, error)
```

### In-Place Drawing

These modify the `[]byte` RGB buffer directly via C. Zero allocation.

```go
// Draw line with thickness (Bresenham).
func DrawLineFast(rgb []byte, w, h, x1, y1, x2, y2 int, r, g, b uint8, thickness int) error

// Draw polygon outline + alpha-blended fill.
func DrawPolygonFast(rgb []byte, w, h int, points [][2]int, r, g, b, alpha uint8) error

// Draw text using built-in 5x8 bitmap font. scale=1 is 8px, scale=2 is 16px.
func DrawTextFast(rgb []byte, w, h, x, y int, text string, r, g, b uint8, scale int) error
```

### Video Encode

```go
// Open output video — h264_nvenc (GPU) with libx264 fallback.
func OpenEncoder(path string, w, h, fps int, codec string) (*Encoder, error)

func (e *Encoder) WriteFrame(rgb []byte) error
func (e *Encoder) IsHWAccelerated() bool
func (e *Encoder) Close()
```

## Types

```go
type AnnotateObject struct {
    X1, Y1, X2, Y2 float32   // bounding box pixel coords
    ClassID         int       // class index (selects color from 20-color palette)
    Confidence      float32   // 0.0-1.0
    TrackID         int       // persistent track ID
    Label           string    // text drawn above box (auto-generated if empty)
}

type AnnotateFullInput struct {
    Boxes    []AnnotateObject   // bounding boxes with labels
    Lines    []AnnotateLine     // lines (counting lines, etc.)
    Polygons []AnnotatePolygon  // polygons with alpha fill (zones)
    Texts    []AnnotateText     // text overlays (stats, camera name)
    Rects    []AnnotateRect     // filled rectangles with alpha (backgrounds)
}

type AnnotateLine struct {
    X1, Y1, X2, Y2 int
    R, G, B         uint8
    Thickness       int
}

type AnnotatePolygon struct {
    Points         [][2]int
    R, G, B, Alpha uint8
}

type AnnotateText struct {
    X, Y    int
    Text    string
    R, G, B uint8
    Scale   int       // 1=8px, 2=16px, 3=24px, 4=32px
}

type AnnotateRect struct {
    X1, Y1, X2, Y2    int
    R, G, B, Alpha     uint8
}

type FrameInfo struct {
    Width, Height int
    PTS           int64  // presentation timestamp (microseconds)
    FrameNumber   int
}
```

## Performance

Benchmarked on AMD Ryzen 9 9900X + RTX 5070 Ti at 1920x1080 with 5 bounding boxes:

| Operation | Pure Go | C-Accelerated | Speedup |
|---|---|---|---|
| JPEG encode (1080p) | 18.7 ms | **2.0 ms** | 9x |
| Resize 2560->640 | 10 ms | **0.3 ms** | 33x |
| Annotate + JPEG | 18.7 ms | **8.0 ms** | 2.3x |
| AnnotateFull (batch) | 50 ms | **6 ms** | 8x |
| ResizeJPEG | N/A | **1.0 ms** | - |
| Decode + resize (2560->1280) | 11 MB copy | **2.7 MB copy** | 4x less |

### Real-world detection pipeline (dual 2560x1440 cameras)

| Metric | Python (ultralytics) | Go (pure) | Go (C-accel) |
|---|---|---|---|
| FPS | 24.7 | 4.6 | **25.2** |
| CPU | 200%+ | 114% | **159%** |
| RAM | 3-5 GB | 14 GB | **2.2 GB** |
| Startup | 8-15s | <1s | **<1s** |
| Binary | ~5 GB | 12 MB | **12 MB** |

## C API

The Go functions wrap these C API functions from `infer_api.h`:

```c
// Batch annotate + resize + JPEG (single call, fastest path)
InferError infer_frame_annotate_full(
    const uint8_t* rgb, int w, int h,
    const InferAnnotateBox* boxes, int n_boxes,
    const InferLine* lines, int n_lines,
    const InferPolygonOverlay* polygons, int n_polygons,
    const InferTextOverlay* texts, int n_texts,
    const InferFilledRect* rects, int n_rects,
    int out_w, int out_h, int quality,
    uint8_t** out_jpeg, int* out_size);

// Individual operations
InferError infer_frame_annotate_jpeg(rgb, w, h, boxes, n, quality, out_jpeg, out_size);
InferError infer_frame_resize_jpeg(rgb, sw, sh, dw, dh, quality, out_jpeg, out_size);
InferError infer_frame_combine_jpeg(rgb1, w1, h1, rgb2, w2, h2, status, tw, th, quality, out_jpeg, out_size);
InferError infer_frame_draw_line(rgb, w, h, x1, y1, x2, y2, r, g, b, thickness);
InferError infer_frame_draw_polygon(rgb, w, h, pts, n, r, g, b, alpha);
InferError infer_frame_draw_text(rgb, w, h, x, y, text, r, g, b, scale);
void       infer_frame_jpeg_free(buf);

// Decode + resize (avoids copying full-res frame to Go)
int infer_video_decoder_next_frame_resized(dec, target_w, target_h, out_rgb, out_w, out_h, out_pts, out_frame_num);
```

## Dependencies

- **TurboJPEG** (`libturbojpeg0-dev`) — JPEG encoding, 10x faster than Go stdlib
- **OpenCV** (`libopencv-dev`, core + imgproc) — SIMD resize
- **FFmpeg** (`libavcodec-dev`, `libavformat-dev`) — video decode/encode
- **CUDA** (optional) — NVDEC hardware video decode, NVENC hardware encode

All dependencies are detected at CMake configure time. If TurboJPEG or OpenCV are missing, the C functions return errors and Go falls back to pure Go implementations.

## Build

```bash
# Build C++ library
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Build Go binary (links against libinfer_api.so)
go build -C go -o ../infergo ./cmd/infergo/

# Run tests
cmake --build build --target test
go test -C go -v ./video/

# Benchmarks
go test -C go -bench BenchmarkAnnotate -benchmem ./video/
```
