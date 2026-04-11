package main

import (
	"context"
	"fmt"
	"runtime"
	"sort"
	"time"
	"unsafe"

	"github.com/ailakshya/infergo/preprocess"
	"github.com/ailakshya/infergo/server"
	"github.com/ailakshya/infergo/tensor"
	"github.com/ailakshya/infergo/torch"
)

// ─── Batch detection types ───────────────────────────────────────────────────

// detectRequest is a single detection request submitted to the batch loop.
type detectRequest struct {
	imageBytes []byte
	confThresh float32
	iouThresh  float32
	result     chan detectResult
}

// detectResult is the result returned from the batch loop to a waiting caller.
type detectResult struct {
	objects []server.DetectedObject
	err     error
}

// torchDetectionAdapter wraps a torch session for YOLOv8/v11 object detection.
// It implements server.DetectionModel using the same preprocessing and NMS
// pipeline as the ONNX-based detectionAdapter.
//
// When the batch loop is running (requests != nil), Detect submits to the
// channel and waits for the batch result. Otherwise it falls back to
// single-image DetectGPU.
type torchDetectionAdapter struct {
	sess     *torch.Session
	requests chan *detectRequest
}

// DetectRaw runs detection on raw RGB pixels — no JPEG encode/decode.
// Implements RawRGBDetector for the pipeline's fast path.
func (a *torchDetectionAdapter) DetectRaw(rgb []byte, width, height int, confThresh, iouThresh float32) ([]server.DetectedObject, error) {
	dets, err := a.sess.DetectGPURaw(rgb, width, height, confThresh, iouThresh)
	if err != nil {
		return nil, err
	}
	results := make([]server.DetectedObject, len(dets))
	for i, d := range dets {
		results[i] = server.DetectedObject{
			X1: d.X1, Y1: d.Y1, X2: d.X2, Y2: d.Y2,
			ClassID: d.ClassID, Confidence: d.Confidence,
		}
	}
	return results, nil
}

// Close releases the torch session and shuts down the batch loop.
// Implements server.Model.
func (a *torchDetectionAdapter) Close() {
	if a.requests != nil {
		close(a.requests)
	}
	a.sess.Close()
}

// Detect runs GPU-accelerated detection.
// If the batch loop is running, it submits to the channel and waits.
// Otherwise it falls back to single-image DetectGPU + CPU fallback.
// Implements server.DetectionModel.
func (a *torchDetectionAdapter) Detect(ctx context.Context, imageBytes []byte, confThresh, iouThresh float32) ([]server.DetectedObject, error) {
	// Batch path: submit to the batch loop goroutine.
	if a.requests != nil {
		req := &detectRequest{
			imageBytes: imageBytes,
			confThresh: confThresh,
			iouThresh:  iouThresh,
			result:     make(chan detectResult, 1),
		}
		select {
		case a.requests <- req:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		select {
		case res := <-req.result:
			return res.objects, res.err
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	// Non-batch path: try the single-call GPU pipeline first.
	dets, err := a.sess.DetectGPU(imageBytes, confThresh, iouThresh)
	if err == nil {
		results := make([]server.DetectedObject, len(dets))
		for i, d := range dets {
			results[i] = server.DetectedObject{
				X1: d.X1, Y1: d.Y1, X2: d.X2, Y2: d.Y2,
				ClassID: d.ClassID, Confidence: d.Confidence,
			}
		}
		return results, nil
	}
	// GPU pipeline failed — fall back to the full preprocessing path.
	return a.detectFallback(ctx, imageBytes, confThresh, iouThresh)
}

// batchLoop is the goroutine that accumulates detection requests and
// dispatches them in batches via DetectGPUBatch. It locks the OS thread
// to avoid CGo overhead from thread switching.
func (a *torchDetectionAdapter) batchLoop() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	const maxBatch = 8
	const maxWait = 2 * time.Millisecond

	batch := make([]*detectRequest, 0, maxBatch)
	timer := time.NewTimer(maxWait)
	timer.Stop() // Don't fire until we have a request.

	for {
		if len(batch) == 0 {
			// Wait for the first request (blocking).
			req, ok := <-a.requests
			if !ok {
				// Channel closed — shut down.
				return
			}
			batch = append(batch, req)
			timer.Reset(maxWait)
			continue
		}

		// We have at least one request; accumulate more up to maxBatch or maxWait.
		select {
		case req, ok := <-a.requests:
			if !ok {
				// Channel closed — process remaining batch and exit.
				a.processBatch(batch)
				return
			}
			batch = append(batch, req)
			if len(batch) >= maxBatch {
				timer.Stop()
				a.processBatch(batch)
				batch = batch[:0]
			}
		case <-timer.C:
			a.processBatch(batch)
			batch = batch[:0]
		}
	}
}

// processBatch calls DetectGPUBatch for the accumulated requests and fans
// results back to each waiting caller.
func (a *torchDetectionAdapter) processBatch(batch []*detectRequest) {
	if len(batch) == 0 {
		return
	}

	// All requests in a batch must share the same thresholds.
	// Use the first request's thresholds (server always uses the same values).
	confThresh := batch[0].confThresh
	iouThresh := batch[0].iouThresh

	images := make([][]byte, len(batch))
	for i, req := range batch {
		images[i] = req.imageBytes
	}

	allDets, err := a.sess.DetectGPUBatch(images, confThresh, iouThresh)
	if err != nil {
		// Return error to all callers.
		for _, req := range batch {
			req.result <- detectResult{err: err}
		}
		return
	}

	for i, req := range batch {
		dets := allDets[i]
		objects := make([]server.DetectedObject, len(dets))
		for j, d := range dets {
			objects[j] = server.DetectedObject{
				X1: d.X1, Y1: d.Y1, X2: d.X2, Y2: d.Y2,
				ClassID: d.ClassID, Confidence: d.Confidence,
			}
		}
		req.result <- detectResult{objects: objects}
	}
}

// detectFallback runs the full YOLOv8 detection pipeline on raw image bytes:
//  1. Decode image (JPEG or PNG)
//  2. Letterbox resize to 640x640 preserving aspect ratio
//  3. Build [1,3,640,640] float32 NCHW tensor, normalize to [0,1]
//  4. Run torch session -> [1, 84, 8400] output
//  5. Parse anchors: filter by confThresh, decode boxes
//  6. Rescale boxes to original image coordinates
//  7. Non-maximum suppression with iouThresh
func (a *torchDetectionAdapter) detectFallback(ctx context.Context, imageBytes []byte, confThresh, iouThresh float32) ([]server.DetectedObject, error) {
	const targetW, targetH = 640, 640

	// 1. Decode image via C/OpenCV.
	decoded, err := preprocess.DecodeImage(imageBytes)
	if err != nil {
		return nil, fmt.Errorf("detect: decode image: %w", err)
	}
	defer decoded.Free()

	// 2. Letterbox resize via C/OpenCV.
	lb, err := preprocess.Letterbox(decoded, targetW, targetH)
	if err != nil {
		return nil, fmt.Errorf("detect: letterbox: %w", err)
	}
	defer lb.Free()

	// 3. Normalize HWC->CHW and scale to [0,1] via C.
	norm, err := preprocess.Normalize(lb, 255.0, [3]float32{0, 0, 0}, [3]float32{1, 1, 1})
	if err != nil {
		return nil, fmt.Errorf("detect: normalize: %w", err)
	}
	defer norm.Free()

	// 4. Stack to [1, 3, 640, 640] batch.
	inputTensor, err := preprocess.StackBatch([]*tensor.Tensor{norm})
	if err != nil {
		return nil, fmt.Errorf("detect: stack batch: %w", err)
	}
	defer inputTensor.Free()

	// 5. Run torch session (GPU-optimized: non-blocking H2D, inference on GPU, D2H output).
	outputs, err := a.sess.RunGPU([]*tensor.Tensor{inputTensor})
	if err != nil {
		return nil, fmt.Errorf("detect: torch run: %w", err)
	}
	defer func() {
		for _, o := range outputs {
			o.Free()
		}
	}()
	if len(outputs) == 0 {
		return nil, fmt.Errorf("detect: no outputs from model")
	}

	// 6. Parse raw YOLOv8 output: shape [1, 84, 8400].
	out := outputs[0]
	outShape := out.Shape()
	if len(outShape) != 3 || outShape[0] != 1 || outShape[1] != 84 {
		return nil, fmt.Errorf("detect: unexpected output shape %v (want [1, 84, 8400])", outShape)
	}
	nAnchors := outShape[2]

	rawData := unsafe.Slice((*float32)(out.DataPtr()), out.NElements())

	type candidate struct {
		x1, y1, x2, y2 float32
		classID         int
		confidence      float32
	}

	var candidates []candidate

	for i := 0; i < nAnchors; i++ {
		cx := rawData[0*nAnchors+i]
		cy := rawData[1*nAnchors+i]
		w := rawData[2*nAnchors+i]
		h := rawData[3*nAnchors+i]

		// Find argmax over 80 class scores.
		bestClass := 0
		bestScore := rawData[4*nAnchors+i]
		for c := 1; c < 80; c++ {
			score := rawData[(4+c)*nAnchors+i]
			if score > bestScore {
				bestScore = score
				bestClass = c
			}
		}

		if bestScore < confThresh {
			continue
		}

		x1 := cx - w/2
		y1 := cy - h/2
		x2 := cx + w/2
		y2 := cy + h/2

		candidates = append(candidates, candidate{
			x1:         x1,
			y1:         y1,
			x2:         x2,
			y2:         y2,
			classID:    bestClass,
			confidence: bestScore,
		})
	}

	// 7. Rescale boxes from padded 640x640 space back to original image coordinates.
	decodedShape := decoded.Shape()
	origH, origW := float32(decodedShape[0]), float32(decodedShape[1])
	scale := minF32(float32(targetW)/origW, float32(targetH)/origH)
	padX := (float32(targetW) - origW*scale) / 2.0
	padY := (float32(targetH) - origH*scale) / 2.0

	for idx := range candidates {
		c := &candidates[idx]
		c.x1 = (c.x1 - padX) / scale
		c.y1 = (c.y1 - padY) / scale
		c.x2 = (c.x2 - padX) / scale
		c.y2 = (c.y2 - padY) / scale

		c.x1 = clampF32(c.x1, 0, origW)
		c.y1 = clampF32(c.y1, 0, origH)
		c.x2 = clampF32(c.x2, 0, origW)
		c.y2 = clampF32(c.y2, 0, origH)
	}

	// 8. Non-maximum suppression.
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].confidence > candidates[j].confidence
	})

	suppressed := make([]bool, len(candidates))
	var results []server.DetectedObject

	for i := 0; i < len(candidates); i++ {
		if suppressed[i] {
			continue
		}
		a := candidates[i]
		results = append(results, server.DetectedObject{
			X1:         a.x1,
			Y1:         a.y1,
			X2:         a.x2,
			Y2:         a.y2,
			ClassID:    a.classID,
			Confidence: a.confidence,
		})
		for j := i + 1; j < len(candidates); j++ {
			if suppressed[j] {
				continue
			}
			b := candidates[j]
			if a.classID == b.classID && iou(a.x1, a.y1, a.x2, a.y2, b.x1, b.y1, b.x2, b.y2) > iouThresh {
				suppressed[j] = true
			}
		}
	}

	return results, nil
}
