package main

import (
	"context"
	"io"
	"sync"
	"testing"
	"time"

	"github.com/ailakshya/infergo/server"
	"github.com/ailakshya/infergo/tracker"
)

// ─── mock detector for pipeline tests ────────────────────────────────────────

type mockPipelineDetector struct {
	mu   sync.Mutex
	objs []server.DetectedObject
}

func (m *mockPipelineDetector) Close() {}

func (m *mockPipelineDetector) Detect(_ context.Context, _ []byte, _, _ float32) ([]server.DetectedObject, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.objs, nil
}

// ─── mock frame source ──────────────────────────────────────────────────────

type mockFrameSource struct {
	frames []mockFrame
	idx    int
	mu     sync.Mutex
	closed bool
}

type mockFrame struct {
	rgb  []byte
	info FrameInfo
}

func (s *mockFrameSource) NextFrame() ([]byte, FrameInfo, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.idx >= len(s.frames) {
		return nil, FrameInfo{}, io.EOF
	}
	f := s.frames[s.idx]
	s.idx++
	return f.rgb, f.info, nil
}

func (s *mockFrameSource) Close() {
	s.mu.Lock()
	s.closed = true
	s.mu.Unlock()
}

// makeTestFrames creates n mock frames with the given dimensions.
func makeTestFrames(n, width, height int) []mockFrame {
	frames := make([]mockFrame, n)
	rgb := make([]byte, width*height*3) // shared backing; detector is a mock
	for i := 0; i < n; i++ {
		frames[i] = mockFrame{
			rgb: rgb,
			info: FrameInfo{
				FrameNumber: i + 1,
				Width:       width,
				Height:      height,
			},
		}
	}
	return frames
}

// ─── TestPipelineSingleStream ────────────────────────────────────────────────

func TestPipelineSingleStream(t *testing.T) {
	det := &mockPipelineDetector{
		objs: []server.DetectedObject{
			{X1: 10, Y1: 20, X2: 100, Y2: 200, ClassID: 0, Confidence: 0.9},
		},
	}

	nFrames := 5
	frames := makeTestFrames(nFrames, 640, 480)
	source := &mockFrameSource{frames: frames}

	factory := func(url string) (FrameSource, error) {
		return source, nil
	}

	pipeline := NewStreamPipeline(det, factory, 64)
	defer pipeline.Close()

	err := pipeline.AddStream(StreamConfig{
		ID:    0,
		URL:   "test://video.mp4",
		Model: "yolo",
	})
	if err != nil {
		t.Fatalf("AddStream: %v", err)
	}

	// Collect results.
	var results []StreamResult
	timeout := time.After(2 * time.Second)
	for i := 0; i < nFrames; i++ {
		select {
		case r := <-pipeline.Results():
			results = append(results, r)
		case <-timeout:
			t.Fatalf("timed out waiting for result %d/%d", i+1, nFrames)
		}
	}

	if len(results) != nFrames {
		t.Errorf("expected %d results, got %d", nFrames, len(results))
	}

	hasTrack := false
	for _, r := range results {
		if r.StreamID != 0 {
			t.Errorf("expected stream_id=0, got %d", r.StreamID)
		}
		if len(r.Detections) != 1 {
			t.Errorf("expected 1 detection, got %d", len(r.Detections))
		}
		if len(r.Tracks) > 0 {
			hasTrack = true
		}
	}
	// ByteTrack needs 2+ frames to confirm a track — at least one result should have tracks.
	if !hasTrack {
		t.Error("expected at least one result with tracked objects (ByteTrack needs 2+ frames)")
	}
}

// ─── TestPipelineMultiStream ─────────────────────────────────────────────────

func TestPipelineMultiStream(t *testing.T) {
	det := &mockPipelineDetector{
		objs: []server.DetectedObject{
			{X1: 0, Y1: 0, X2: 50, Y2: 50, ClassID: 2, Confidence: 0.85},
		},
	}

	nFrames := 3
	source0 := &mockFrameSource{frames: makeTestFrames(nFrames, 320, 240)}
	source1 := &mockFrameSource{frames: makeTestFrames(nFrames, 320, 240)}

	sources := map[string]*mockFrameSource{
		"test://stream0.mp4": source0,
		"test://stream1.mp4": source1,
	}

	factory := func(url string) (FrameSource, error) {
		return sources[url], nil
	}

	pipeline := NewStreamPipeline(det, factory, 64)
	defer pipeline.Close()

	pipeline.AddStream(StreamConfig{ID: 0, URL: "test://stream0.mp4", Model: "yolo"})
	pipeline.AddStream(StreamConfig{ID: 1, URL: "test://stream1.mp4", Model: "yolo"})

	// Collect results from both streams.
	seen := map[int]int{} // stream_id -> count
	timeout := time.After(3 * time.Second)
	total := nFrames * 2

	for i := 0; i < total; i++ {
		select {
		case r := <-pipeline.Results():
			seen[r.StreamID]++
		case <-timeout:
			t.Fatalf("timed out after %d/%d results: seen=%v", i, total, seen)
		}
	}

	if seen[0] != nFrames {
		t.Errorf("stream 0: expected %d results, got %d", nFrames, seen[0])
	}
	if seen[1] != nFrames {
		t.Errorf("stream 1: expected %d results, got %d", nFrames, seen[1])
	}
}

// ─── TestPipelineAddRemove ───────────────────────────────────────────────────

func TestPipelineAddRemove(t *testing.T) {
	det := &mockPipelineDetector{
		objs: []server.DetectedObject{
			{X1: 0, Y1: 0, X2: 50, Y2: 50, ClassID: 0, Confidence: 0.9},
		},
	}

	source := &mockFrameSource{frames: makeTestFrames(10, 320, 240)}
	factory := func(url string) (FrameSource, error) {
		return source, nil
	}

	pipeline := NewStreamPipeline(det, factory, 64)
	defer pipeline.Close()

	// Add a stream.
	err := pipeline.AddStream(StreamConfig{ID: 5, URL: "test://vid.mp4", Model: "yolo"})
	if err != nil {
		t.Fatalf("AddStream: %v", err)
	}

	// Verify it's in the stream list.
	ids := pipeline.StreamIDs()
	found := false
	for _, id := range ids {
		if id == 5 {
			found = true
		}
	}
	if !found {
		t.Error("stream 5 not found in StreamIDs after AddStream")
	}

	// Adding duplicate should fail.
	err = pipeline.AddStream(StreamConfig{ID: 5, URL: "test://dup.mp4", Model: "yolo"})
	if err == nil {
		t.Error("expected error when adding duplicate stream ID")
	}

	// Remove the stream.
	pipeline.RemoveStream(5)

	ids = pipeline.StreamIDs()
	for _, id := range ids {
		if id == 5 {
			t.Error("stream 5 still present after RemoveStream")
		}
	}

	// Removing non-existent stream should not panic.
	pipeline.RemoveStream(999)
}

// ─── TestPipelineClose ───────────────────────────────────────────────────────

func TestPipelineClose(t *testing.T) {
	det := &mockPipelineDetector{}

	// Use a source that produces frames indefinitely (until closed).
	infiniteFrames := make([]mockFrame, 1000)
	rgb := make([]byte, 320*240*3)
	for i := range infiniteFrames {
		infiniteFrames[i] = mockFrame{
			rgb:  rgb,
			info: FrameInfo{FrameNumber: i, Width: 320, Height: 240},
		}
	}

	source0 := &mockFrameSource{frames: infiniteFrames}
	source1 := &mockFrameSource{frames: infiniteFrames}

	callNum := 0
	var factoryMu sync.Mutex
	factory := func(url string) (FrameSource, error) {
		factoryMu.Lock()
		defer factoryMu.Unlock()
		callNum++
		if callNum == 1 {
			return source0, nil
		}
		return source1, nil
	}

	pipeline := NewStreamPipeline(det, factory, 64)

	pipeline.AddStream(StreamConfig{ID: 0, URL: "test://a.mp4", Model: "yolo"})
	pipeline.AddStream(StreamConfig{ID: 1, URL: "test://b.mp4", Model: "yolo"})

	// Let workers start.
	time.Sleep(50 * time.Millisecond)

	// Close should stop all workers and close the results channel.
	done := make(chan struct{})
	go func() {
		pipeline.Close()
		close(done)
	}()

	select {
	case <-done:
		// Good — Close returned.
	case <-time.After(3 * time.Second):
		t.Fatal("Close did not return within 3 seconds")
	}

	// Results channel should be closed — drain any buffered results.
	for range pipeline.Results() {
	}

	// Adding to a closed pipeline should fail.
	err := pipeline.AddStream(StreamConfig{ID: 99, URL: "test://x.mp4"})
	if err == nil {
		t.Error("expected error when adding stream to closed pipeline")
	}
}

// ─── TestPipelinePushFrame ───────────────────────────────────────────────────

func TestPipelinePushFrame(t *testing.T) {
	det := &mockPipelineDetector{
		objs: []server.DetectedObject{
			{X1: 0, Y1: 0, X2: 50, Y2: 50, ClassID: 0, Confidence: 0.9},
		},
	}

	// No factory needed — we'll push frames manually.
	pipeline := NewStreamPipeline(det, nil, 64)
	defer pipeline.Close()

	// Add a stream (it won't produce frames on its own since factory is nil).
	pipeline.AddStream(StreamConfig{ID: 0, URL: "manual", Model: "yolo"})

	// Push a frame.
	rgb := make([]byte, 320*240*3)
	pipeline.PushFrame(0, rgb, FrameInfo{FrameNumber: 1, Width: 320, Height: 240})

	// Read result.
	select {
	case r := <-pipeline.Results():
		if r.StreamID != 0 {
			t.Errorf("expected stream_id=0, got %d", r.StreamID)
		}
		if r.FrameNumber != 1 {
			t.Errorf("expected frame_number=1, got %d", r.FrameNumber)
		}
		if len(r.Detections) != 1 {
			t.Errorf("expected 1 detection, got %d", len(r.Detections))
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for push frame result")
	}
}

// ─── TestMergeTrackIDs ──────────────────────────────────────────────────────

func TestMergeTrackIDs(t *testing.T) {
	dets := []server.DetectedObject{
		{X1: 10, Y1: 10, X2: 100, Y2: 100, ClassID: 0, Confidence: 0.9},
		{X1: 200, Y1: 200, X2: 300, Y2: 300, ClassID: 2, Confidence: 0.8},
	}

	tracks := []tracker.Track{
		{ID: 1, Box: [4]float64{10, 10, 100, 100}, Class: 0, Confidence: 0.9},
		{ID: 2, Box: [4]float64{200, 200, 300, 300}, Class: 2, Confidence: 0.8},
	}

	result := mergeTrackIDs(dets, tracks)
	if len(result) != 2 {
		t.Fatalf("expected 2 tracked objects, got %d", len(result))
	}

	if result[0].TrackID != 1 {
		t.Errorf("expected track ID 1, got %d", result[0].TrackID)
	}
	if result[1].TrackID != 2 {
		t.Errorf("expected track ID 2, got %d", result[1].TrackID)
	}

	// Verify detection data is preserved.
	if result[0].ClassID != 0 || result[0].Confidence != 0.9 {
		t.Errorf("detection data not preserved for track 1: %+v", result[0])
	}
}

// ─── TestBoxIoU ─────────────────────────────────────────────────────────────

func TestBoxIoU(t *testing.T) {
	// Perfect overlap.
	iou := boxIoU(0, 0, 100, 100, 0, 0, 100, 100)
	if iou < 0.99 || iou > 1.01 {
		t.Errorf("perfect overlap: expected ~1.0, got %f", iou)
	}

	// No overlap.
	iou = boxIoU(0, 0, 50, 50, 100, 100, 200, 200)
	if iou != 0 {
		t.Errorf("no overlap: expected 0, got %f", iou)
	}

	// Partial overlap.
	iou = boxIoU(0, 0, 100, 100, 50, 50, 150, 150)
	// Intersection: 50x50 = 2500, Union: 10000+10000-2500 = 17500
	expected := 2500.0 / 17500.0
	if iou < expected-0.01 || iou > expected+0.01 {
		t.Errorf("partial overlap: expected ~%f, got %f", expected, iou)
	}
}
