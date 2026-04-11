package server_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"golang.org/x/net/websocket"

	"github.com/ailakshya/infergo/server"
)

// ─── helpers ─────────────────────────────────────────────────────────────────

// setupWSDetect creates a test server with the /v1/ws/detect endpoint and
// a GET /v1/streams endpoint, returning the server and hub.
func setupWSDetect(t *testing.T) (*httptest.Server, *server.StreamHub) {
	t.Helper()
	reg := server.NewRegistry()
	srv := server.NewServer(reg)
	hub := server.NewStreamHub()

	mux := http.NewServeMux()
	mux.Handle("/v1/", srv)
	mux.Handle("/v1/ws/detect", srv.HandleWSDetect(hub))

	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	return ts, hub
}

// dialWSDetect connects to /v1/ws/detect with the given query params.
func dialWSDetect(t *testing.T, ts *httptest.Server, query string) *websocket.Conn {
	t.Helper()
	origin := "http://localhost/"
	url := "ws" + strings.TrimPrefix(ts.URL, "http") + "/v1/ws/detect?" + query
	ws, err := websocket.Dial(url, "", origin)
	if err != nil {
		t.Fatalf("websocket.Dial: %v", err)
	}
	return ws
}

// readDetectFrame reads one WSDetectFrame from the WebSocket.
func readDetectFrame(t *testing.T, ws *websocket.Conn) server.WSDetectFrame {
	t.Helper()
	var frame server.WSDetectFrame
	if err := websocket.JSON.Receive(ws, &frame); err != nil {
		t.Fatalf("read detect frame: %v", err)
	}
	return frame
}

// ─── T1: Connect and receive initial message ────────────────────────────────

func TestWSDetectConnect(t *testing.T) {
	ts, hub := setupWSDetect(t)

	ws := dialWSDetect(t, ts, "stream=0")
	defer ws.Close()

	// Publish a frame — client should receive it.
	hub.Publish(0, server.WSDetectFrame{
		StreamID:    0,
		FrameNumber: 1,
		Timestamp:   time.Now().UTC().Format(time.RFC3339),
		Objects: []server.WSDetectObject{
			{X1: 10, Y1: 20, X2: 150, Y2: 200, ClassID: 0, Confidence: 0.92, TrackID: 5, Label: "person"},
		},
		FPS: 30.0,
	}, nil)

	frame := readDetectFrame(t, ws)
	if frame.StreamID != 0 {
		t.Errorf("expected stream_id=0, got %d", frame.StreamID)
	}
	if frame.FrameNumber != 1 {
		t.Errorf("expected frame_number=1, got %d", frame.FrameNumber)
	}
	if len(frame.Objects) != 1 {
		t.Fatalf("expected 1 object, got %d", len(frame.Objects))
	}
	if frame.Objects[0].Label != "person" {
		t.Errorf("expected label=person, got %q", frame.Objects[0].Label)
	}
	if frame.FrameB64 != "" {
		t.Error("expected no frame_b64 when frame=true not set")
	}
}

// ─── T2: Multiple clients on the same stream ────────────────────────────────

func TestWSDetectMultiClient(t *testing.T) {
	ts, hub := setupWSDetect(t)

	// Connect 3 clients to the same stream.
	clients := make([]*websocket.Conn, 3)
	for i := range clients {
		clients[i] = dialWSDetect(t, ts, "stream=42")
		defer clients[i].Close()
	}

	// Small delay to let subscriptions register.
	time.Sleep(20 * time.Millisecond)

	// Publish a frame.
	hub.Publish(42, server.WSDetectFrame{
		StreamID:    42,
		FrameNumber: 100,
		Timestamp:   time.Now().UTC().Format(time.RFC3339),
		FPS:         15.0,
	}, nil)

	// All 3 clients should receive the frame.
	for i, ws := range clients {
		frame := readDetectFrame(t, ws)
		if frame.StreamID != 42 {
			t.Errorf("client %d: expected stream_id=42, got %d", i, frame.StreamID)
		}
		if frame.FrameNumber != 100 {
			t.Errorf("client %d: expected frame_number=100, got %d", i, frame.FrameNumber)
		}
	}
}

// ─── T3: Backpressure — slow client doesn't block ───────────────────────────

func TestWSDetectBackpressure(t *testing.T) {
	ts, hub := setupWSDetect(t)

	ws := dialWSDetect(t, ts, "stream=0")
	defer ws.Close()

	// Small delay to let subscription register.
	time.Sleep(20 * time.Millisecond)

	// Flood the hub with more frames than the client buffer can hold.
	// wsDetectClientBuf is 16, so send 32 frames.
	for i := 0; i < 32; i++ {
		hub.Publish(0, server.WSDetectFrame{
			StreamID:    0,
			FrameNumber: i,
			Timestamp:   time.Now().UTC().Format(time.RFC3339),
			FPS:         30.0,
		}, nil)
	}

	// Should not hang. Read what we can within a timeout.
	received := 0
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			var frame server.WSDetectFrame
			ws.SetReadDeadline(time.Now().Add(200 * time.Millisecond))
			if err := websocket.JSON.Receive(ws, &frame); err != nil {
				return
			}
			received++
		}
	}()
	<-done

	// We should have received some frames but possibly not all (dropped by backpressure).
	if received == 0 {
		t.Error("expected at least some frames to be received")
	}
	// The key assertion: the test completed without hanging.
	t.Logf("received %d/32 frames (backpressure dropped %d)", received, 32-received)
}

// ─── T4: Clean disconnect, no goroutine leak ────────────────────────────────

func TestWSDetectDisconnect(t *testing.T) {
	ts, hub := setupWSDetect(t)

	ws := dialWSDetect(t, ts, "stream=0")

	// Small delay to let subscription register.
	time.Sleep(20 * time.Millisecond)

	// Close the connection.
	ws.Close()

	// Give the server a moment to notice the disconnect and clean up.
	time.Sleep(50 * time.Millisecond)

	// Publish after disconnect — should not panic.
	hub.Publish(0, server.WSDetectFrame{
		StreamID:    0,
		FrameNumber: 999,
		Timestamp:   time.Now().UTC().Format(time.RFC3339),
	}, nil)

	// Verify no subscribers remain.
	streams := hub.ActiveStreams()
	for _, id := range streams {
		if id == 0 {
			t.Error("expected stream 0 to have no subscribers after disconnect")
		}
	}
}

// ─── T5: Missing stream query param ─────────────────────────────────────────

func TestWSDetectMissingStream(t *testing.T) {
	ts, _ := setupWSDetect(t)

	ws := dialWSDetect(t, ts, "")
	defer ws.Close()

	// Should receive an error message.
	var resp map[string]string
	if err := websocket.JSON.Receive(ws, &resp); err != nil {
		t.Fatalf("expected error frame, got read error: %v", err)
	}
	if resp["error"] == "" {
		t.Error("expected non-empty error message")
	}
}

// ─── T6: Frame included when frame=true ─────────────────────────────────────

func TestWSDetectWithFrame(t *testing.T) {
	ts, hub := setupWSDetect(t)

	ws := dialWSDetect(t, ts, "stream=0&frame=true")
	defer ws.Close()

	// Small delay.
	time.Sleep(20 * time.Millisecond)

	// Publish with an annotated JPEG.
	fakeJPEG := []byte{0xFF, 0xD8, 0xFF, 0xE0} // JPEG header bytes
	hub.Publish(0, server.WSDetectFrame{
		StreamID:    0,
		FrameNumber: 1,
		Timestamp:   time.Now().UTC().Format(time.RFC3339),
	}, fakeJPEG)

	var raw json.RawMessage
	if err := websocket.JSON.Receive(ws, &raw); err != nil {
		t.Fatalf("read frame: %v", err)
	}

	var frame server.WSDetectFrame
	if err := json.Unmarshal(raw, &frame); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if frame.FrameB64 == "" {
		t.Error("expected frame_b64 to be non-empty when frame=true")
	}
}

// ─── T7: Multiple streams don't cross-talk ──────────────────────────────────

func TestWSDetectStreamIsolation(t *testing.T) {
	ts, hub := setupWSDetect(t)

	ws0 := dialWSDetect(t, ts, "stream=0")
	defer ws0.Close()
	ws1 := dialWSDetect(t, ts, "stream=1")
	defer ws1.Close()

	// Small delay.
	time.Sleep(20 * time.Millisecond)

	// Publish to stream 1 only.
	hub.Publish(1, server.WSDetectFrame{
		StreamID:    1,
		FrameNumber: 42,
		Timestamp:   time.Now().UTC().Format(time.RFC3339),
	}, nil)

	// Stream 1 client should receive it.
	frame := readDetectFrame(t, ws1)
	if frame.StreamID != 1 || frame.FrameNumber != 42 {
		t.Errorf("stream 1: unexpected frame %+v", frame)
	}

	// Stream 0 client should NOT receive it within a short timeout.
	ws0.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
	var shouldFail server.WSDetectFrame
	err := websocket.JSON.Receive(ws0, &shouldFail)
	if err == nil {
		t.Error("stream 0 client received a frame meant for stream 1")
	}
}

// ─── T8: FPSTracker ─────────────────────────────────────────────────────────

func TestFPSTracker(t *testing.T) {
	ft := server.NewFPSTracker(1.0)

	// First tick should return 0 (only one sample).
	fps := ft.Tick()
	if fps != 0 {
		t.Errorf("first tick: expected 0 fps, got %f", fps)
	}

	// Simulate 30 frames at roughly 30fps-equivalent timestamps.
	for i := 0; i < 29; i++ {
		time.Sleep(time.Millisecond)
		ft.Tick()
	}

	// FPS should be positive (exact value depends on timing).
	finalFPS := ft.Tick()
	if finalFPS <= 0 {
		t.Errorf("expected positive FPS after 30 ticks, got %f", finalFPS)
	}
}

// ─── T9: Concurrent Publish safety ──────────────────────────────────────────

func TestWSDetectConcurrentPublish(t *testing.T) {
	var received atomic.Int64

	ts, hub := setupWSDetect(t)
	ws := dialWSDetect(t, ts, "stream=0")
	defer ws.Close()

	time.Sleep(20 * time.Millisecond)

	// Publish concurrently from multiple goroutines.
	var wg sync.WaitGroup
	for g := 0; g < 5; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 10; i++ {
				hub.Publish(0, server.WSDetectFrame{
					StreamID:    0,
					FrameNumber: g*10 + i,
					Timestamp:   time.Now().UTC().Format(time.RFC3339),
				}, nil)
			}
		}(g)
	}
	wg.Wait()

	// Read what we can.
	go func() {
		for {
			var frame server.WSDetectFrame
			ws.SetReadDeadline(time.Now().Add(200 * time.Millisecond))
			if err := websocket.JSON.Receive(ws, &frame); err != nil {
				return
			}
			received.Add(1)
		}
	}()

	time.Sleep(300 * time.Millisecond)
	t.Logf("received %d frames from concurrent publish", received.Load())
	// The main assertion is that we didn't panic.
}
