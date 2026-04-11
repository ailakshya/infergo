package server

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"sort"
	"strconv"
	"sync"
	"time"

	"golang.org/x/net/websocket"
)

// ─── WebSocket detection stream types ────────────────────────────────────────

// WSDetectFrame is the JSON message sent to each subscribed WebSocket client
// for every processed frame.
type WSDetectFrame struct {
	StreamID    int              `json:"stream_id"`
	FrameNumber int              `json:"frame_number"`
	Timestamp   string           `json:"timestamp"` // RFC3339
	Objects     []WSDetectObject `json:"objects"`
	FPS         float64          `json:"fps"`
	FrameB64    string           `json:"frame_b64,omitempty"` // optional annotated JPEG
}

// WSDetectObject describes a single detected+tracked object in a frame.
type WSDetectObject struct {
	X1         float32 `json:"x1"`
	Y1         float32 `json:"y1"`
	X2         float32 `json:"x2"`
	Y2         float32 `json:"y2"`
	ClassID    int     `json:"class_id"`
	Confidence float32 `json:"confidence"`
	TrackID    int     `json:"track_id"`
	Label      string  `json:"label"`
}

// ─── StreamHub ───────────────────────────────────────────────────────────────

// StreamHub manages WebSocket subscriptions for detection streams.
// One goroutine per stream publishes frames; clients subscribe via stream ID.
type StreamHub struct {
	mu      sync.RWMutex
	clients map[int]map[*wsDetectClient]struct{} // stream_id -> set of clients
}

// NewStreamHub creates a new StreamHub.
func NewStreamHub() *StreamHub {
	return &StreamHub{
		clients: make(map[int]map[*wsDetectClient]struct{}),
	}
}

// wsDetectClient represents a single WebSocket subscriber.
type wsDetectClient struct {
	ws        *websocket.Conn
	send      chan []byte // buffered outbound messages
	streamID  int
	wantFrame bool // whether to include base64 JPEG frames
	closing   chan struct{}
	done      chan struct{}
}

const wsDetectClientBuf = 16 // backpressure buffer per client

// subscribe adds a client to the hub for the given stream ID.
func (h *StreamHub) subscribe(c *wsDetectClient) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.clients[c.streamID] == nil {
		h.clients[c.streamID] = make(map[*wsDetectClient]struct{})
	}
	h.clients[c.streamID][c] = struct{}{}
}

// unsubscribe removes a client from the hub. After this returns, Publish
// will never send to c.send again, making it safe for the caller to close it.
func (h *StreamHub) unsubscribe(c *wsDetectClient) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if subs, ok := h.clients[c.streamID]; ok {
		delete(subs, c)
		if len(subs) == 0 {
			delete(h.clients, c.streamID)
		}
	}
}

// Publish sends a detection frame to all subscribers of the given stream.
// Non-blocking: if a client's buffer is full the frame is dropped for that client.
//
// The read lock is held for the entire publish cycle, which guarantees that
// unsubscribe (which takes the write lock) cannot complete between the
// subscriber snapshot and the send — preventing send-on-closed-channel panics.
func (h *StreamHub) Publish(streamID int, frame WSDetectFrame, annotatedJPEG []byte) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	subs := h.clients[streamID]
	if len(subs) == 0 {
		return
	}

	// Pre-encode the frame without the JPEG for clients that don't want it.
	noFrameData, _ := json.Marshal(frame)

	// Encode with JPEG only if any client wants it and JPEG is available.
	var withFrameData []byte
	if len(annotatedJPEG) > 0 {
		frameWithB64 := frame
		frameWithB64.FrameB64 = base64.StdEncoding.EncodeToString(annotatedJPEG)
		withFrameData, _ = json.Marshal(frameWithB64)
	}

	for c := range subs {
		var data []byte
		if c.wantFrame && withFrameData != nil {
			data = withFrameData
		} else {
			data = noFrameData
		}

		// Non-blocking send — drop if buffer full.
		select {
		case c.send <- data:
		default:
			// Client can't keep up — drop this frame.
		}
	}
}

// ActiveStreams returns a sorted list of stream IDs that have at least one subscriber.
func (h *StreamHub) ActiveStreams() []int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	ids := make([]int, 0, len(h.clients))
	for id := range h.clients {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	return ids
}

// ─── WebSocket handler ──────────────────────────────────────────────────────

// HandleWSDetect returns an http.Handler for GET /v1/ws/detect.
// Query parameters:
//   - stream: stream ID (int, required)
//   - frame: "true" to include base64 annotated JPEG in each message
func (s *Server) HandleWSDetect(hub *StreamHub) websocket.Handler {
	return func(ws *websocket.Conn) {
		query := ws.Request().URL.Query()

		streamIDStr := query.Get("stream")
		if streamIDStr == "" {
			websocket.JSON.Send(ws, map[string]string{"error": "stream query param is required"}) //nolint:errcheck
			return
		}
		streamID, err := strconv.Atoi(streamIDStr)
		if err != nil {
			websocket.JSON.Send(ws, map[string]string{"error": "stream must be an integer"}) //nolint:errcheck
			return
		}

		wantFrame := query.Get("frame") == "true"

		client := &wsDetectClient{
			ws:        ws,
			send:      make(chan []byte, wsDetectClientBuf),
			streamID:  streamID,
			wantFrame: wantFrame,
			closing:   make(chan struct{}),
			done:      make(chan struct{}),
		}

		hub.subscribe(client)

		// Writer goroutine: sends queued messages to the WebSocket.
		// Exits when the closing channel is closed and the send channel drains,
		// or when a write error occurs.
		go func() {
			defer close(client.done)
			for {
				select {
				case msg, ok := <-client.send:
					if !ok {
						// send was closed after unsubscribe — drain complete.
						return
					}
					if _, err := ws.Write(msg); err != nil {
						return
					}
				case <-client.closing:
					// Drain remaining buffered messages.
					for {
						select {
						case msg := <-client.send:
							ws.Write(msg) //nolint:errcheck
						default:
							return
						}
					}
				}
			}
		}()

		// Reader: reads from the WebSocket to detect disconnect.
		// We don't expect any client messages, but we need to read to
		// detect when the connection is closed.
		buf := make([]byte, 512)
		for {
			_, err := ws.Read(buf)
			if err != nil {
				break
			}
		}

		// Client disconnected — clean up.
		// 1. Unsubscribe so Publish can never target this client again.
		//    This takes the write lock, so it waits for any in-progress Publish
		//    to finish. After this returns, no new sends to client.send will occur.
		hub.unsubscribe(client)

		// 2. Signal writer to drain and stop.
		close(client.closing)

		// 3. Wait for writer to finish.
		<-client.done
	}
}

// ─── Stream info types ──────────────────────────────────────────────────────

// StreamInfo describes an active stream for the GET /v1/streams endpoint.
type StreamInfo struct {
	ID      int     `json:"id"`
	URL     string  `json:"url,omitempty"`
	FPS     float64 `json:"fps,omitempty"`
	Clients int     `json:"clients"`
}

// StreamListResponse is the response body for GET /v1/streams.
type StreamListResponse struct {
	Streams []StreamInfo `json:"streams"`
}

// HandleStreams returns an http.HandlerFunc for GET /v1/streams.
// It lists all active stream IDs that have WebSocket subscribers.
func (s *Server) HandleStreams(hub *StreamHub) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		hub.mu.RLock()
		streams := make([]StreamInfo, 0, len(hub.clients))
		for id, subs := range hub.clients {
			streams = append(streams, StreamInfo{
				ID:      id,
				Clients: len(subs),
			})
		}
		hub.mu.RUnlock()

		// Sort by ID for deterministic output.
		sort.Slice(streams, func(i, j int) bool {
			return streams[i].ID < streams[j].ID
		})

		writeJSON(w, http.StatusOK, StreamListResponse{Streams: streams})
	}
}

// ─── FPS tracker ────────────────────────────────────────────────────────────

// FPSTracker measures frames-per-second over a sliding window.
type FPSTracker struct {
	mu        sync.Mutex
	times     []time.Time
	windowSec float64
}

// NewFPSTracker creates a tracker with the given window duration in seconds.
func NewFPSTracker(windowSec float64) *FPSTracker {
	if windowSec <= 0 {
		windowSec = 2.0
	}
	return &FPSTracker{windowSec: windowSec}
}

// Tick records a frame timestamp and returns the current FPS.
func (f *FPSTracker) Tick() float64 {
	f.mu.Lock()
	defer f.mu.Unlock()

	now := time.Now()
	f.times = append(f.times, now)

	// Prune old entries.
	cutoff := now.Add(-time.Duration(f.windowSec * float64(time.Second)))
	start := 0
	for start < len(f.times) && f.times[start].Before(cutoff) {
		start++
	}
	f.times = f.times[start:]

	if len(f.times) < 2 {
		return 0
	}
	elapsed := f.times[len(f.times)-1].Sub(f.times[0]).Seconds()
	if elapsed <= 0 {
		return 0
	}
	return float64(len(f.times)-1) / elapsed
}
