// YOLOv8 object detection demo.
//
// Usage:
//
//	go run examples/yolov8/main.go <image.jpg|png> [model.onnx]
//
// Defaults: model = ~/yolov8n.onnx, conf_threshold = 0.25, iou_threshold = 0.45
//
// Prints detected bounding boxes in the format:
//
//	[class_name] conf=0.87  x1=120 y1=45 x2=380 y2=290
package main

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"sort"
	"unsafe"

	"github.com/ailakshya/infergo/onnx"
	"github.com/ailakshya/infergo/tensor"
)

const (
	modelSize     = 640
	confThreshold = 0.25
	iouThreshold  = 0.45
)

// ─── COCO class names ─────────────────────────────────────────────────────────

var cocoNames = [80]string{
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
	"truck", "boat", "traffic light", "fire hydrant", "stop sign",
	"parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
	"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
	"tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
	"baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
	"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
	"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
	"donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
	"toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
	"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
	"vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}

// ─── Detection box ────────────────────────────────────────────────────────────

type Box struct {
	X1, Y1, X2, Y2 float32
	Conf            float32
	ClassID         int
}

// ─── Main ─────────────────────────────────────────────────────────────────────

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: %s <image> [model.onnx]\n", os.Args[0])
		os.Exit(1)
	}
	imgPath := os.Args[1]

	modelPath := filepath.Join(os.Getenv("HOME"), "yolov8n.onnx")
	if len(os.Args) >= 3 {
		modelPath = os.Args[2]
	}

	// ── Load and preprocess image ─────────────────────────────────────────────
	img, err := loadImage(imgPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load image: %v\n", err)
		os.Exit(1)
	}
	origW := img.Bounds().Dx()
	origH := img.Bounds().Dy()

	pixels, padLeft, padTop, scale := letterbox(img, modelSize)

	in, err := tensor.NewTensorCPU([]int{1, 3, modelSize, modelSize}, tensor.Float32)
	if err != nil {
		fmt.Fprintf(os.Stderr, "alloc tensor: %v\n", err)
		os.Exit(1)
	}
	defer in.Free()

	if err := in.CopyFrom(unsafe.Pointer(&pixels[0]), len(pixels)*4); err != nil {
		fmt.Fprintf(os.Stderr, "copy pixels: %v\n", err)
		os.Exit(1)
	}

	// ── Run inference ─────────────────────────────────────────────────────────
	sess, err := onnx.NewSession("cpu", 0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new session: %v\n", err)
		os.Exit(1)
	}
	defer sess.Close()

	if err := sess.Load(modelPath); err != nil {
		fmt.Fprintf(os.Stderr, "load model %s: %v\n", modelPath, err)
		os.Exit(1)
	}

	outputs, err := sess.Run([]*tensor.Tensor{in})
	if err != nil {
		fmt.Fprintf(os.Stderr, "run: %v\n", err)
		os.Exit(1)
	}
	defer outputs[0].Free()

	// ── Decode output [1, 84, 8400] ───────────────────────────────────────────
	outData := (*[1 * 84 * 8400]float32)(outputs[0].DataPtr())
	boxes := decodeBoxes(outData[:], padLeft, padTop, scale, origW, origH)
	boxes = nms(boxes, iouThreshold)

	if len(boxes) == 0 {
		fmt.Printf("No detections above conf=%.2f\n", confThreshold)
		return
	}
	for _, b := range boxes {
		name := "unknown"
		if b.ClassID >= 0 && b.ClassID < 80 {
			name = cocoNames[b.ClassID]
		}
		fmt.Printf("[%-15s] conf=%.2f  x1=%4.0f y1=%4.0f x2=%4.0f y2=%4.0f\n",
			name, b.Conf, b.X1, b.Y1, b.X2, b.Y2)
	}
}

// ─── Image loading ────────────────────────────────────────────────────────────

func loadImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

// ─── Letterbox resize + normalise → CHW float32 ──────────────────────────────

// letterbox resizes img to fit inside size×size with grey padding.
// Returns CHW float32 pixels, padding offsets, and the scale factor.
func letterbox(img image.Image, size int) (pixels []float32, padLeft, padTop int, scale float64) {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	scale = math.Min(float64(size)/float64(w), float64(size)/float64(h))
	newW := int(math.Round(float64(w) * scale))
	newH := int(math.Round(float64(h) * scale))
	padLeft = (size - newW) / 2
	padTop = (size - newH) / 2

	// Allocate CHW: 3 channels × size × size
	pixels = make([]float32, 3*size*size)
	// Fill with grey (114/255 ≈ 0.447 — YOLOv8 default padding colour)
	for i := range pixels {
		pixels[i] = 114.0 / 255.0
	}

	// Sample scaled image and write into CHW layout
	for py := 0; py < newH; py++ {
		// source y in original image (nearest neighbour)
		srcY := int(float64(py) / scale)
		if srcY >= h {
			srcY = h - 1
		}
		for px := 0; px < newW; px++ {
			srcX := int(float64(px) / scale)
			if srcX >= w {
				srcX = w - 1
			}
			r, g, b, _ := img.At(bounds.Min.X+srcX, bounds.Min.Y+srcY).RGBA()
			dstY := padTop + py
			dstX := padLeft + px
			idx := dstY*size + dstX
			pixels[0*size*size+idx] = float32(r>>8) / 255.0 // R channel
			pixels[1*size*size+idx] = float32(g>>8) / 255.0 // G channel
			pixels[2*size*size+idx] = float32(b>>8) / 255.0 // B channel
		}
	}
	return
}

// ─── Box decoding ─────────────────────────────────────────────────────────────

// decodeBoxes parses YOLOv8 output [84 × 8400].
// Each column: [cx, cy, w, h, class0_score, ..., class79_score]
// Coordinates are in model-input space (0..640); we map back to original image.
func decodeBoxes(data []float32, padLeft, padTop int, scale float64, origW, origH int) []Box {
	const numAnchors = 8400
	const numClasses = 80

	boxes := make([]Box, 0, 32)

	for a := 0; a < numAnchors; a++ {
		cx := data[0*numAnchors+a]
		cy := data[1*numAnchors+a]
		bw := data[2*numAnchors+a]
		bh := data[3*numAnchors+a]

		// Find best class
		bestConf := float32(0)
		bestCls := 0
		for c := 0; c < numClasses; c++ {
			s := data[(4+c)*numAnchors+a]
			if s > bestConf {
				bestConf = s
				bestCls = c
			}
		}
		if bestConf < confThreshold {
			continue
		}

		// Convert from model coords to original image coords
		x1 := (float64(cx-bw/2) - float64(padLeft)) / scale
		y1 := (float64(cy-bh/2) - float64(padTop)) / scale
		x2 := (float64(cx+bw/2) - float64(padLeft)) / scale
		y2 := (float64(cy+bh/2) - float64(padTop)) / scale

		// Clamp to image bounds
		x1 = math.Max(0, math.Min(x1, float64(origW)))
		y1 = math.Max(0, math.Min(y1, float64(origH)))
		x2 = math.Max(0, math.Min(x2, float64(origW)))
		y2 = math.Max(0, math.Min(y2, float64(origH)))

		if x2 <= x1 || y2 <= y1 {
			continue
		}

		boxes = append(boxes, Box{
			X1: float32(x1), Y1: float32(y1),
			X2: float32(x2), Y2: float32(y2),
			Conf: bestConf, ClassID: bestCls,
		})
	}
	return boxes
}

// ─── Non-maximum suppression ──────────────────────────────────────────────────

func nms(boxes []Box, iouThresh float32) []Box {
	// Sort by confidence descending
	sort.Slice(boxes, func(i, j int) bool {
		return boxes[i].Conf > boxes[j].Conf
	})

	kept := boxes[:0]
	suppressed := make([]bool, len(boxes))

	for i := range boxes {
		if suppressed[i] {
			continue
		}
		kept = append(kept, boxes[i])
		for j := i + 1; j < len(boxes); j++ {
			if !suppressed[j] && iou(boxes[i], boxes[j]) > iouThresh {
				suppressed[j] = true
			}
		}
	}
	return kept
}

func iou(a, b Box) float32 {
	ix1 := max32(a.X1, b.X1)
	iy1 := max32(a.Y1, b.Y1)
	ix2 := min32(a.X2, b.X2)
	iy2 := min32(a.Y2, b.Y2)
	if ix2 <= ix1 || iy2 <= iy1 {
		return 0
	}
	inter := (ix2 - ix1) * (iy2 - iy1)
	areaA := (a.X2 - a.X1) * (a.Y2 - a.Y1)
	areaB := (b.X2 - b.X1) * (b.Y2 - b.Y1)
	return inter / (areaA + areaB - inter)
}

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func min32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}
