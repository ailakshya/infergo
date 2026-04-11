// Package recording provides event-triggered clip recording with ring buffering.
// It captures frames from video streams into per-stream ring buffers and saves
// clips around event timestamps for later review.
package recording

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// BufferedFrame holds a single video frame with metadata.
type BufferedFrame struct {
	RGB       []byte    `json:"-"`
	Width     int       `json:"width"`
	Height    int       `json:"height"`
	Timestamp time.Time `json:"timestamp"`
}

// RingBuffer is a lock-free (single-writer) ring buffer for video frames.
// It uses atomic operations for the head index to allow concurrent reads
// during clip extraction.
type RingBuffer struct {
	frames []BufferedFrame
	head   atomic.Int64
	size   int
	count  atomic.Int64 // total frames written (may exceed size)
}

// NewRingBuffer creates a ring buffer with the given capacity.
func NewRingBuffer(capacity int) *RingBuffer {
	return &RingBuffer{
		frames: make([]BufferedFrame, capacity),
		size:   capacity,
	}
}

// Push adds a frame to the ring buffer, overwriting the oldest frame if full.
func (rb *RingBuffer) Push(frame BufferedFrame) {
	idx := rb.head.Load() % int64(rb.size)
	rb.frames[idx] = frame
	rb.head.Add(1)
	rb.count.Add(1)
}

// Frames returns all buffered frames in chronological order.
func (rb *RingBuffer) Frames() []BufferedFrame {
	h := rb.head.Load()
	n := int64(rb.size)
	total := rb.count.Load()

	if total == 0 {
		return nil
	}

	var result []BufferedFrame
	numFrames := total
	if numFrames > n {
		numFrames = n
	}

	start := h - numFrames
	if start < 0 {
		start = 0
	}

	for i := start; i < h; i++ {
		idx := i % n
		f := rb.frames[idx]
		if f.Timestamp.IsZero() {
			continue
		}
		result = append(result, f)
	}
	return result
}

// FramesInRange returns frames within the given time range [from, to].
func (rb *RingBuffer) FramesInRange(from, to time.Time) []BufferedFrame {
	all := rb.Frames()
	var result []BufferedFrame
	for _, f := range all {
		if !f.Timestamp.Before(from) && !f.Timestamp.After(to) {
			result = append(result, f)
		}
	}
	return result
}

// Len returns the number of valid frames in the buffer.
func (rb *RingBuffer) Len() int {
	total := rb.count.Load()
	if total > int64(rb.size) {
		return rb.size
	}
	return int(total)
}

// ClipMetadata holds information about a saved clip.
type ClipMetadata struct {
	Path      string            `json:"path"`
	StreamID  int               `json:"stream_id"`
	EventTime time.Time         `json:"event_time"`
	StartTime time.Time         `json:"start_time"`
	EndTime   time.Time         `json:"end_time"`
	Frames    int               `json:"frames"`
	Metadata  map[string]string `json:"metadata,omitempty"`
	SizeBytes int64             `json:"size_bytes"`
	CreatedAt time.Time         `json:"created_at"`
}

// EventRecorder manages per-stream ring buffers and saves event clips.
type EventRecorder struct {
	mu          sync.Mutex
	ringBuffers map[int]*RingBuffer // per-stream frame ring buffer
	outputDir   string
	preSeconds  int
	postSeconds int
	maxStorageGB float64

	clips []ClipMetadata // index of all saved clips
}

// NewEventRecorder creates a new EventRecorder.
//
// Parameters:
//   - outputDir: base directory for saving clips
//   - preSeconds: seconds of video to capture before the event
//   - postSeconds: seconds of video to capture after the event
//   - maxStorageGB: maximum total storage for clips (0 = unlimited)
func NewEventRecorder(outputDir string, preSeconds, postSeconds int, maxStorageGB float64) *EventRecorder {
	return &EventRecorder{
		ringBuffers:  make(map[int]*RingBuffer),
		outputDir:    outputDir,
		preSeconds:   preSeconds,
		postSeconds:  postSeconds,
		maxStorageGB: maxStorageGB,
	}
}

// BufferFrame adds a frame to the ring buffer for the given stream.
// The ring buffer is created on first use with a capacity based on
// preSeconds + postSeconds at an assumed 30 FPS.
func (r *EventRecorder) BufferFrame(streamID int, frame []byte, w, h int, ts time.Time) {
	r.mu.Lock()
	rb, ok := r.ringBuffers[streamID]
	if !ok {
		// Allocate enough frames for pre+post seconds at ~30 FPS.
		capacity := (r.preSeconds + r.postSeconds + 5) * 30
		rb = NewRingBuffer(capacity)
		r.ringBuffers[streamID] = rb
	}
	r.mu.Unlock()

	// Copy the frame data to avoid external mutation.
	frameCopy := make([]byte, len(frame))
	copy(frameCopy, frame)

	rb.Push(BufferedFrame{
		RGB:       frameCopy,
		Width:     w,
		Height:    h,
		Timestamp: ts,
	})
}

// SaveClip extracts frames around the event time from the ring buffer
// and saves them as raw frame data (one file per frame) in a clip directory.
// Returns the path to the clip directory and any error.
//
// In a production system, this would encode to MP4 using the video encoder.
// For now, it saves raw frames and a metadata JSON file.
func (r *EventRecorder) SaveClip(streamID int, eventTime time.Time, metadata map[string]string) (string, error) {
	r.mu.Lock()
	rb, ok := r.ringBuffers[streamID]
	r.mu.Unlock()

	if !ok {
		return "", fmt.Errorf("no ring buffer for stream %d", streamID)
	}

	from := eventTime.Add(-time.Duration(r.preSeconds) * time.Second)
	to := eventTime.Add(time.Duration(r.postSeconds) * time.Second)

	frames := rb.FramesInRange(from, to)
	if len(frames) == 0 {
		return "", fmt.Errorf("no frames in range [%s, %s] for stream %d", from, to, streamID)
	}

	// Create output directory: outputDir/streamID/YYYY-MM-DD_HH-MM-SS_eventName/
	eventName := "event"
	if name, ok := metadata["event_name"]; ok {
		eventName = name
	}
	dirName := fmt.Sprintf("%s_%s",
		eventTime.Format("2006-01-02_15-04-05"),
		eventName,
	)
	clipDir := filepath.Join(r.outputDir, fmt.Sprintf("stream_%d", streamID), dirName)

	if err := os.MkdirAll(clipDir, 0o755); err != nil {
		return "", fmt.Errorf("create clip dir: %w", err)
	}

	// Save each frame as a raw file.
	var totalSize int64
	for i, f := range frames {
		framePath := filepath.Join(clipDir, fmt.Sprintf("frame_%04d.raw", i))
		if err := os.WriteFile(framePath, f.RGB, 0o644); err != nil {
			return "", fmt.Errorf("write frame %d: %w", i, err)
		}
		totalSize += int64(len(f.RGB))
	}

	// Save metadata.
	clipMeta := ClipMetadata{
		Path:      clipDir,
		StreamID:  streamID,
		EventTime: eventTime,
		StartTime: frames[0].Timestamp,
		EndTime:   frames[len(frames)-1].Timestamp,
		Frames:    len(frames),
		Metadata:  metadata,
		SizeBytes: totalSize,
		CreatedAt: time.Now(),
	}

	metaPath := filepath.Join(clipDir, "clip.json")
	metaData, err := json.MarshalIndent(clipMeta, "", "  ")
	if err != nil {
		return "", fmt.Errorf("marshal clip metadata: %w", err)
	}
	if err := os.WriteFile(metaPath, metaData, 0o644); err != nil {
		return "", fmt.Errorf("write clip metadata: %w", err)
	}

	r.mu.Lock()
	r.clips = append(r.clips, clipMeta)
	r.mu.Unlock()

	// Update the clips index.
	if err := r.saveClipsIndex(); err != nil {
		// Non-fatal: clip is saved, index update failed.
		_ = err
	}

	return clipDir, nil
}

// Cleanup deletes the oldest clips when total storage exceeds maxStorageGB.
// Returns the number of clips deleted and any error.
func (r *EventRecorder) Cleanup() error {
	if r.maxStorageGB <= 0 {
		return nil // unlimited storage
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	maxBytes := int64(r.maxStorageGB * 1024 * 1024 * 1024)

	// Calculate total storage.
	var totalBytes int64
	for _, c := range r.clips {
		totalBytes += c.SizeBytes
	}

	if totalBytes <= maxBytes {
		return nil
	}

	// Sort clips by creation time (oldest first).
	sort.Slice(r.clips, func(i, j int) bool {
		return r.clips[i].CreatedAt.Before(r.clips[j].CreatedAt)
	})

	// Delete oldest clips until under limit.
	var remaining []ClipMetadata
	for _, c := range r.clips {
		if totalBytes > maxBytes {
			if err := os.RemoveAll(c.Path); err != nil {
				// Continue cleanup even if one delete fails.
				remaining = append(remaining, c)
				continue
			}
			totalBytes -= c.SizeBytes
		} else {
			remaining = append(remaining, c)
		}
	}

	r.clips = remaining
	return r.saveClipsIndexLocked()
}

// saveClipsIndex writes the clips index JSON file.
func (r *EventRecorder) saveClipsIndex() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.saveClipsIndexLocked()
}

// saveClipsIndexLocked writes the clips index JSON (caller must hold mu).
func (r *EventRecorder) saveClipsIndexLocked() error {
	indexPath := filepath.Join(r.outputDir, "clips.json")
	data, err := json.MarshalIndent(r.clips, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(indexPath, data, 0o644)
}

// GetClips returns a copy of all clip metadata.
func (r *EventRecorder) GetClips() []ClipMetadata {
	r.mu.Lock()
	defer r.mu.Unlock()
	result := make([]ClipMetadata, len(r.clips))
	copy(result, r.clips)
	return result
}
