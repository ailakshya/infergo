package recording

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func TestRingBuffer(t *testing.T) {
	rb := NewRingBuffer(50)

	// Add 100 frames; oldest 50 should be evicted.
	for i := 0; i < 100; i++ {
		rb.Push(BufferedFrame{
			RGB:       []byte{byte(i)},
			Width:     640,
			Height:    480,
			Timestamp: time.Now().Add(time.Duration(i) * time.Millisecond),
		})
	}

	if rb.Len() != 50 {
		t.Errorf("expected 50 frames, got %d", rb.Len())
	}

	frames := rb.Frames()
	if len(frames) != 50 {
		t.Fatalf("expected 50 frames from Frames(), got %d", len(frames))
	}

	// Verify oldest evicted: first frame should have data from frame 50+.
	if frames[0].RGB[0] != 50 {
		t.Errorf("expected oldest frame data=50, got %d", frames[0].RGB[0])
	}

	// Verify newest is last.
	if frames[49].RGB[0] != 99 {
		t.Errorf("expected newest frame data=99, got %d", frames[49].RGB[0])
	}
}

func TestRingBufferSmall(t *testing.T) {
	rb := NewRingBuffer(10)

	// Add fewer frames than capacity.
	for i := 0; i < 5; i++ {
		rb.Push(BufferedFrame{
			RGB:       []byte{byte(i)},
			Width:     320,
			Height:    240,
			Timestamp: time.Now().Add(time.Duration(i) * time.Millisecond),
		})
	}

	if rb.Len() != 5 {
		t.Errorf("expected 5 frames, got %d", rb.Len())
	}

	frames := rb.Frames()
	if len(frames) != 5 {
		t.Errorf("expected 5 frames, got %d", len(frames))
	}
}

func TestRingBufferFramesInRange(t *testing.T) {
	rb := NewRingBuffer(100)

	t0 := time.Date(2026, 4, 9, 12, 0, 0, 0, time.UTC)
	for i := 0; i < 30; i++ {
		rb.Push(BufferedFrame{
			RGB:       []byte{byte(i)},
			Width:     640,
			Height:    480,
			Timestamp: t0.Add(time.Duration(i) * time.Second),
		})
	}

	// Get frames from second 10 to second 20.
	from := t0.Add(10 * time.Second)
	to := t0.Add(20 * time.Second)
	frames := rb.FramesInRange(from, to)

	if len(frames) != 11 { // seconds 10, 11, ..., 20 inclusive
		t.Errorf("expected 11 frames in range, got %d", len(frames))
	}
}

func TestSaveClip(t *testing.T) {
	tmpDir := t.TempDir()
	rec := NewEventRecorder(tmpDir, 2, 2, 0)

	t0 := time.Date(2026, 4, 9, 12, 0, 0, 0, time.UTC)

	// Buffer 5 seconds of frames (150 frames at 30fps).
	for i := 0; i < 150; i++ {
		ts := t0.Add(time.Duration(i) * 33 * time.Millisecond)
		frame := make([]byte, 640*480*3)
		frame[0] = byte(i % 256)
		rec.BufferFrame(1, frame, 640, 480, ts)
	}

	// Trigger event at t0 + 2.5s.
	eventTime := t0.Add(2500 * time.Millisecond)
	clipPath, err := rec.SaveClip(1, eventTime, map[string]string{
		"event_name": "intrusion",
		"zone":       "restricted",
	})
	if err != nil {
		t.Fatalf("SaveClip: %v", err)
	}

	// Verify clip directory exists.
	if _, err := os.Stat(clipPath); os.IsNotExist(err) {
		t.Fatalf("clip directory does not exist: %s", clipPath)
	}

	// Verify metadata file.
	metaPath := filepath.Join(clipPath, "clip.json")
	metaData, err := os.ReadFile(metaPath)
	if err != nil {
		t.Fatalf("read clip metadata: %v", err)
	}

	var meta ClipMetadata
	if err := json.Unmarshal(metaData, &meta); err != nil {
		t.Fatalf("unmarshal clip metadata: %v", err)
	}

	if meta.StreamID != 1 {
		t.Errorf("meta.StreamID = %d, want 1", meta.StreamID)
	}
	if meta.Frames == 0 {
		t.Error("meta.Frames should be > 0")
	}
	if meta.Metadata["event_name"] != "intrusion" {
		t.Errorf("meta.Metadata[event_name] = %q, want intrusion", meta.Metadata["event_name"])
	}

	// Verify frame files exist.
	frameFiles, err := filepath.Glob(filepath.Join(clipPath, "frame_*.raw"))
	if err != nil {
		t.Fatal(err)
	}
	if len(frameFiles) == 0 {
		t.Error("expected frame files in clip directory")
	}

	// Verify clips index.
	indexPath := filepath.Join(tmpDir, "clips.json")
	indexData, err := os.ReadFile(indexPath)
	if err != nil {
		t.Fatalf("read clips index: %v", err)
	}
	var clips []ClipMetadata
	if err := json.Unmarshal(indexData, &clips); err != nil {
		t.Fatalf("unmarshal clips index: %v", err)
	}
	if len(clips) != 1 {
		t.Errorf("expected 1 clip in index, got %d", len(clips))
	}
}

func TestSaveClipNoBuffer(t *testing.T) {
	tmpDir := t.TempDir()
	rec := NewEventRecorder(tmpDir, 2, 2, 0)

	_, err := rec.SaveClip(999, time.Now(), nil)
	if err == nil {
		t.Error("expected error when no ring buffer exists for stream")
	}
}

func TestCleanup(t *testing.T) {
	tmpDir := t.TempDir()
	// Set max storage to a very small value.
	rec := NewEventRecorder(tmpDir, 1, 1, 0.000001) // ~1KB limit

	t0 := time.Date(2026, 4, 9, 12, 0, 0, 0, time.UTC)

	// Buffer and save several clips.
	for clipIdx := 0; clipIdx < 5; clipIdx++ {
		streamID := 1
		baseTime := t0.Add(time.Duration(clipIdx) * 10 * time.Second)

		for i := 0; i < 90; i++ { // 3 seconds of frames
			ts := baseTime.Add(time.Duration(i) * 33 * time.Millisecond)
			frame := make([]byte, 1024) // 1KB per frame
			rec.BufferFrame(streamID, frame, 32, 32, ts)
		}

		eventTime := baseTime.Add(1500 * time.Millisecond)
		_, err := rec.SaveClip(streamID, eventTime, map[string]string{
			"event_name": "test",
		})
		if err != nil {
			t.Fatalf("SaveClip %d: %v", clipIdx, err)
		}
	}

	// Run cleanup.
	err := rec.Cleanup()
	if err != nil {
		t.Fatalf("Cleanup: %v", err)
	}

	// After cleanup, some clips should be deleted.
	remaining := rec.GetClips()
	if len(remaining) >= 5 {
		t.Errorf("expected fewer than 5 clips after cleanup, got %d", len(remaining))
	}
}

func TestConcurrentStreams(t *testing.T) {
	tmpDir := t.TempDir()
	rec := NewEventRecorder(tmpDir, 1, 1, 0)

	t0 := time.Now()

	// 4 goroutines buffering frames concurrently.
	var wg sync.WaitGroup
	for streamID := 1; streamID <= 4; streamID++ {
		wg.Add(1)
		go func(sid int) {
			defer wg.Done()
			for i := 0; i < 60; i++ { // 2 seconds at 30fps
				ts := t0.Add(time.Duration(i) * 33 * time.Millisecond)
				frame := make([]byte, 640*480*3)
				rec.BufferFrame(sid, frame, 640, 480, ts)
			}
		}(streamID)
	}
	wg.Wait()

	// Verify all 4 streams have buffers.
	for streamID := 1; streamID <= 4; streamID++ {
		rec.mu.Lock()
		rb, ok := rec.ringBuffers[streamID]
		rec.mu.Unlock()
		if !ok {
			t.Errorf("stream %d: no ring buffer", streamID)
			continue
		}
		if rb.Len() == 0 {
			t.Errorf("stream %d: ring buffer is empty", streamID)
		}
	}

	// Save clips from all 4 streams.
	eventTime := t0.Add(1 * time.Second)
	for streamID := 1; streamID <= 4; streamID++ {
		clipPath, err := rec.SaveClip(streamID, eventTime, map[string]string{
			"event_name": "concurrent_test",
		})
		if err != nil {
			t.Errorf("stream %d SaveClip: %v", streamID, err)
			continue
		}
		if _, err := os.Stat(clipPath); os.IsNotExist(err) {
			t.Errorf("stream %d clip directory missing: %s", streamID, clipPath)
		}
	}

	clips := rec.GetClips()
	if len(clips) != 4 {
		t.Errorf("expected 4 clips, got %d", len(clips))
	}
}

func TestGetClips(t *testing.T) {
	rec := NewEventRecorder(t.TempDir(), 1, 1, 0)

	clips := rec.GetClips()
	if len(clips) != 0 {
		t.Errorf("expected 0 clips initially, got %d", len(clips))
	}
}

func BenchmarkRingBufferPush(b *testing.B) {
	rb := NewRingBuffer(300) // 10 seconds at 30fps

	frame := BufferedFrame{
		RGB:       make([]byte, 640*480*3),
		Width:     640,
		Height:    480,
		Timestamp: time.Now(),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rb.Push(frame)
	}
}

func BenchmarkBufferFrame(b *testing.B) {
	rec := NewEventRecorder(b.TempDir(), 5, 5, 0)
	frame := make([]byte, 640*480*3)
	ts := time.Now()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rec.BufferFrame(1, frame, 640, 480, ts)
		ts = ts.Add(33 * time.Millisecond)
	}
}
