package main

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"log"
	"sync"
	"time"

	"github.com/ailakshya/infergo/server"
	"github.com/ailakshya/infergo/tracker"
)

// StreamConfig describes a single video stream to be processed.
type StreamConfig struct {
	ID    int    // unique stream identifier
	URL   string // rtsp://, file path, /dev/video0
	Model string // model name in registry
	FPS   int    // target FPS (0 = native)
}

// StreamResult carries one processed frame from a stream worker.
type StreamResult struct {
	StreamID    int
	FrameNumber int
	Timestamp   time.Time
	Detections  []server.DetectedObject
	Tracks      []TrackedObject
	FrameRGB    []byte // raw RGB pixels (for annotation)
	Width       int
	Height      int
}

// TrackedObject extends DetectedObject with a persistent track ID.
type TrackedObject struct {
	server.DetectedObject
	TrackID int
}

// FrameSource abstracts the video decoder so the pipeline can work with
// any frame provider — real decoders, test generators, etc.
type FrameSource interface {
	// NextFrame returns the next decoded frame as raw RGB pixels.
	// Returns io.EOF when the stream ends.
	NextFrame() (rgb []byte, info FrameInfo, err error)
	Close()
}

// FrameInfo carries metadata about a decoded frame.
type FrameInfo struct {
	FrameNumber int
	Width       int
	Height      int
}

// FrameSourceFactory creates a FrameSource for the given URL.
// The pipeline uses this to abstract away the concrete decoder implementation.
type FrameSourceFactory func(url string) (FrameSource, error)

// streamWorker processes frames from a single video stream.
type streamWorker struct {
	config   StreamConfig
	detector server.DetectionModel
	factory  FrameSourceFactory
	results  chan<- StreamResult
	cancel   context.CancelFunc
	done     chan struct{}
}

// run is the main goroutine for a stream worker. It decodes frames,
// runs detection, applies tracking, and pushes results.
func (w *streamWorker) run(ctx context.Context) {
	defer close(w.done)

	var source FrameSource
	if w.factory != nil {
		var err error
		source, err = w.factory(w.config.URL)
		if err != nil {
			log.Printf("pipeline: stream %d: open source %q: %v", w.config.ID, w.config.URL, err)
			return
		}
		defer source.Close()
	}

	bt := tracker.NewByteTracker(tracker.DefaultConfig())

	// FPS throttle
	var ticker *time.Ticker
	if w.config.FPS > 0 {
		ticker = time.NewTicker(time.Second / time.Duration(w.config.FPS))
		defer ticker.Stop()
	}

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		// Throttle if FPS target is set.
		if ticker != nil {
			select {
			case <-ticker.C:
			case <-ctx.Done():
				return
			}
		}

		if source == nil {
			// No frame source — pipeline is in "push" mode or waiting for
			// the video decoder to be wired up. Sleep briefly.
			time.Sleep(10 * time.Millisecond)
			continue
		}

		rgb, info, err := source.NextFrame()
		if err != nil {
			if err != io.EOF {
				log.Printf("pipeline: stream %d: frame %d: %v", w.config.ID, info.FrameNumber, err)
			}
			return
		}

		w.processFrame(ctx, bt, rgb, info)
	}
}

// processFrame runs detection + tracking on a single frame and pushes the result.
// Exported as a method so callers can push raw frames without a FrameSource.
// RawRGBDetector is optionally implemented by detection backends that can
// accept raw RGB pixels directly, skipping the JPEG encode/decode round-trip.
type RawRGBDetector interface {
	DetectRaw(rgb []byte, width, height int, confThresh, iouThresh float32) ([]server.DetectedObject, error)
}

func (w *streamWorker) processFrame(ctx context.Context, bt *tracker.ByteTracker, rgb []byte, info FrameInfo) {
	var dets []server.DetectedObject
	var err error

	// Fast path: if detector supports raw RGB, skip JPEG encoding entirely.
	// This saves ~30ms per frame at 1080p.
	if raw, ok := w.detector.(RawRGBDetector); ok {
		dets, err = raw.DetectRaw(rgb, info.Width, info.Height, 0.25, 0.45)
	} else {
		// Slow path: encode RGB→JPEG for detectors that only accept encoded images.
		jpegBytes, encErr := encodeRGBToJPEG(rgb, info.Width, info.Height)
		if encErr != nil {
			log.Printf("pipeline: stream %d: frame %d: JPEG encode: %v", w.config.ID, info.FrameNumber, encErr)
			return
		}
		dets, err = w.detector.Detect(ctx, jpegBytes, 0.25, 0.45)
	}
	if err != nil {
		log.Printf("pipeline: stream %d: detect frame %d: %v", w.config.ID, info.FrameNumber, err)
		return
	}

	// Convert to tracker format.
	trackDets := make([]tracker.Detection, len(dets))
	for i, d := range dets {
		trackDets[i] = tracker.Detection{
			Box:        [4]float64{float64(d.X1), float64(d.Y1), float64(d.X2), float64(d.Y2)},
			Class:      d.ClassID,
			Confidence: float64(d.Confidence),
		}
	}

	// Update tracker.
	tracks := bt.Update(trackDets)

	// Merge track IDs with detection results.
	tracked := mergeTrackIDs(dets, tracks)

	// Non-blocking send — drop if consumer is slow.
	result := StreamResult{
		StreamID:    w.config.ID,
		FrameNumber: info.FrameNumber,
		Timestamp:   time.Now(),
		Detections:  dets,
		Tracks:      tracked,
		FrameRGB:    rgb,
		Width:       info.Width,
		Height:      info.Height,
	}

	select {
	case w.results <- result:
	default:
		// Drop frame — consumer can't keep up.
	}
}

// mergeTrackIDs associates tracker output with detections. The tracker's
// Track.Box is matched to the closest detection by IoU overlap.
func mergeTrackIDs(dets []server.DetectedObject, tracks []tracker.Track) []TrackedObject {
	result := make([]TrackedObject, 0, len(tracks))

	for _, t := range tracks {
		// Find the detection that best matches this track's box.
		bestIdx := -1
		bestIoU := float64(0)
		for i, d := range dets {
			ov := boxIoU(
				t.Box[0], t.Box[1], t.Box[2], t.Box[3],
				float64(d.X1), float64(d.Y1), float64(d.X2), float64(d.Y2),
			)
			if ov > bestIoU {
				bestIoU = ov
				bestIdx = i
			}
		}

		var obj server.DetectedObject
		if bestIdx >= 0 {
			obj = dets[bestIdx]
		} else {
			// Track box with no matching detection — use track coords.
			obj = server.DetectedObject{
				X1:         float32(t.Box[0]),
				Y1:         float32(t.Box[1]),
				X2:         float32(t.Box[2]),
				Y2:         float32(t.Box[3]),
				ClassID:    t.Class,
				Confidence: float32(t.Confidence),
			}
		}

		result = append(result, TrackedObject{
			DetectedObject: obj,
			TrackID:        t.ID,
		})
	}

	return result
}

// boxIoU computes intersection-over-union between two boxes in x1y1x2y2 format.
func boxIoU(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2 float64) float64 {
	ix1 := max64(ax1, bx1)
	iy1 := max64(ay1, by1)
	ix2 := min64(ax2, bx2)
	iy2 := min64(ay2, by2)

	iw := ix2 - ix1
	ih := iy2 - iy1
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := iw * ih
	aArea := (ax2 - ax1) * (ay2 - ay1)
	bArea := (bx2 - bx1) * (by2 - by1)
	union := aArea + bArea - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

func max64(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func min64(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

// StreamPipeline manages multiple concurrent video stream workers, fanning
// results into a shared channel. It holds a reference to a shared detection
// model (from the registry) that all workers use.
type StreamPipeline struct {
	streams  map[int]*streamWorker
	detector server.DetectionModel
	factory  FrameSourceFactory
	results  chan StreamResult
	mu       sync.RWMutex
	closed   bool
}

// NewStreamPipeline creates a new pipeline backed by the given detection model.
// bufSize controls the capacity of the shared results channel.
func NewStreamPipeline(detector server.DetectionModel, factory FrameSourceFactory, bufSize int) *StreamPipeline {
	if bufSize <= 0 {
		bufSize = 256
	}
	return &StreamPipeline{
		streams:  make(map[int]*streamWorker),
		detector: detector,
		factory:  factory,
		results:  make(chan StreamResult, bufSize),
	}
}

// AddStream starts a new stream worker. Returns an error if the stream ID
// already exists or the pipeline is closed.
func (p *StreamPipeline) AddStream(config StreamConfig) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return errPipelineClosed
	}
	if _, exists := p.streams[config.ID]; exists {
		return errStreamExists
	}

	ctx, cancel := context.WithCancel(context.Background())
	w := &streamWorker{
		config:   config,
		detector: p.detector,
		factory:  p.factory,
		results:  p.results,
		cancel:   cancel,
		done:     make(chan struct{}),
	}

	p.streams[config.ID] = w
	go w.run(ctx)
	return nil
}

// RemoveStream stops and removes the stream with the given ID.
func (p *StreamPipeline) RemoveStream(id int) {
	p.mu.Lock()
	w, ok := p.streams[id]
	if ok {
		delete(p.streams, id)
	}
	p.mu.Unlock()

	if ok {
		w.cancel()
		<-w.done
	}
}

// Results returns the shared channel that receives all stream results.
func (p *StreamPipeline) Results() <-chan StreamResult {
	return p.results
}

// StreamIDs returns the IDs of all currently active streams.
func (p *StreamPipeline) StreamIDs() []int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	ids := make([]int, 0, len(p.streams))
	for id := range p.streams {
		ids = append(ids, id)
	}
	return ids
}

// Close stops all stream workers and closes the results channel.
func (p *StreamPipeline) Close() {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return
	}
	p.closed = true
	workers := make([]*streamWorker, 0, len(p.streams))
	for _, w := range p.streams {
		workers = append(workers, w)
	}
	p.streams = nil
	p.mu.Unlock()

	// Cancel all workers and wait for them to finish.
	for _, w := range workers {
		w.cancel()
	}
	for _, w := range workers {
		<-w.done
	}
	close(p.results)
}

// PushFrame allows external callers to inject a raw frame into a specific
// stream's processing path. This is useful when the video decoder is
// external or for testing without a FrameSource.
func (p *StreamPipeline) PushFrame(streamID int, rgb []byte, info FrameInfo) {
	p.mu.RLock()
	w, ok := p.streams[streamID]
	p.mu.RUnlock()
	if !ok {
		return
	}

	// We need a ByteTracker per push — but since PushFrame is typically
	// called sequentially, we use a lightweight approach: run detection
	// and push result. Tracking requires state, so we use the worker's
	// processFrame if possible. For simplicity here, we do detection only.
	dets, err := p.detector.Detect(context.Background(), rgb, 0.25, 0.45)
	if err != nil {
		return
	}

	tracked := make([]TrackedObject, len(dets))
	for i, d := range dets {
		tracked[i] = TrackedObject{DetectedObject: d, TrackID: -1}
	}

	result := StreamResult{
		StreamID:    streamID,
		FrameNumber: info.FrameNumber,
		Timestamp:   time.Now(),
		Detections:  dets,
		Tracks:      tracked,
		FrameRGB:    rgb,
		Width:       info.Width,
		Height:      info.Height,
	}

	select {
	case w.results <- result:
	default:
	}
}

// Sentinel errors.
var (
	errPipelineClosed = pipelineError("pipeline is closed")
	errStreamExists   = pipelineError("stream already exists")
)

type pipelineError string

func (e pipelineError) Error() string { return string(e) }

// encodeRGBToJPEG encodes raw RGB pixel data to JPEG bytes.
func encodeRGBToJPEG(rgb []byte, width, height int) ([]byte, error) {
	expected := width * height * 3
	if len(rgb) < expected {
		return nil, fmt.Errorf("RGB buffer too small: got %d, need %d", len(rgb), expected)
	}

	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			srcIdx := (y*width + x) * 3
			dstIdx := (y*width + x) * 4
			img.Pix[dstIdx+0] = rgb[srcIdx+0] // R
			img.Pix[dstIdx+1] = rgb[srcIdx+1] // G
			img.Pix[dstIdx+2] = rgb[srcIdx+2] // B
			img.Pix[dstIdx+3] = 255            // A
		}
	}

	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 85}); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
