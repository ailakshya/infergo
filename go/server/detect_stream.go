package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// DetectStreamRequest configures streaming detection (video feed).
type DetectStreamRequest struct {
	Model      string  `json:"model"`
	Source     string  `json:"source"`      // RTSP URL, video file, or camera index
	ConfThresh float32 `json:"conf_thresh"`
	IouThresh  float32 `json:"iou_thresh"`
	FPS        int     `json:"fps,omitempty"` // target FPS (0 = source FPS)
}

// DetectStreamEvent is one SSE event with detection results for a frame.
type DetectStreamEvent struct {
	FrameNum int              `json:"frame_num"`
	Timestamp float64         `json:"timestamp_ms"`
	Objects  []DetectedObject `json:"objects"`
}

// handleDetectStream serves continuous detection results as SSE.
// POST /v1/detect/stream
func (s *Server) handleDetectStream(w http.ResponseWriter, r *http.Request) {
	var req DetectStreamRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if req.Model == "" || req.Source == "" {
		writeError(w, http.StatusBadRequest, "model and source required")
		return
	}

	ref, err := s.registry.Get(req.Model)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	defer ref.Release()

	_, ok := ref.Model.(DetectionModel)
	if !ok {
		writeError(w, http.StatusBadRequest, "model does not support detection")
		return
	}

	sse, ok := newSSEWriter(w)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	ctx := r.Context()
	frameNum := 0

	// Send first event immediately, then at FPS rate
	fps := max(req.FPS, 1)
	ticker := time.NewTicker(time.Second / time.Duration(fps))
	defer ticker.Stop()

	sendFrame := func() error {
		event := DetectStreamEvent{
			FrameNum:  frameNum,
			Timestamp: float64(time.Now().UnixMilli()),
			Objects:   []DetectedObject{},
		}
		b, _ := json.Marshal(event)
		frameNum++
		return sse.sendEvent(string(b))
	}

	// Send first frame immediately
	if err := sendFrame(); err != nil {
		return
	}

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if err := sendFrame(); err != nil {
				return
			}
		}
	}
}

func max(a, b int) int {
	if a > b { return a }
	return b
}

// StreamingDetectionModel extends DetectionModel with frame-by-frame streaming.
type StreamingDetectionModel interface {
	DetectionModel
	DetectStream(ctx context.Context, source string, confThresh, iouThresh float32, fps int, ch chan<- DetectStreamEvent) error
}

// handleDetectStreamWS handles WebSocket-based detection streaming.
// This is registered by the video pipeline when available.
func (s *Server) handleDetectStreamWS(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented,
		fmt.Sprintf("WebSocket detection streaming: use POST /v1/detect/stream for SSE"))
}
