package server_test

import (
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ailakshya/infergo/server"
)

// gaugeStub is a simple DepthSetter stub for testing.
type gaugeStub struct {
	val atomic.Int64
}

func (g *gaugeStub) Set(v float64) { g.val.Store(int64(v)) }
func (g *gaugeStub) Value() int64  { return g.val.Load() }

// TestQueueCapEnforced verifies the 503 limit (OPT-10-T1).
func TestQueueCapEnforced(t *testing.T) {
	const maxQueue = 5

	// maxActive=1 so requests queue up sequentially.
	q := server.NewQueueMiddleware(1, maxQueue, nil)
	blockCh := make(chan struct{})
	handler := q.Middleware()(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-blockCh
		w.WriteHeader(http.StatusOK)
	}))

	// Launch maxQueue requests — all should enter (1 active, rest waiting).
	var wg sync.WaitGroup
	statuses := make([]int, maxQueue)
	for i := 0; i < maxQueue; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", nil))
			statuses[idx] = rec.Code
		}(i)
	}

	// Wait for all goroutines to enter the queue.
	time.Sleep(60 * time.Millisecond)

	// (maxQueue+1)th request must get 503.
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", nil))
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("expected 503, got %d", rec.Code)
	}

	// Unblock and verify all queued requests complete with 200.
	close(blockCh)
	wg.Wait()
	for i, code := range statuses {
		if code != http.StatusOK {
			t.Errorf("request %d: expected 200, got %d", i, code)
		}
	}
}

// TestHighPriorityFirst verifies priority ordering (OPT-10-T2).
func TestHighPriorityFirst(t *testing.T) {
	// maxActive=1 forces serialization; maxQueue=20 so nothing gets 503.
	q := server.NewQueueMiddleware(1, 20, nil)

	var completionOrder []string
	var mu sync.Mutex
	var started atomic.Bool

	blockCh := make(chan struct{})
	handler := q.Middleware()(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		label := r.Header.Get("X-Label")
		if label == "blocker" {
			started.Store(true)
			<-blockCh
		}
		mu.Lock()
		completionOrder = append(completionOrder, label)
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))

	// Start the blocker to occupy the single active slot.
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
		req.Header.Set("X-Label", "blocker")
		handler.ServeHTTP(httptest.NewRecorder(), req)
	}()

	// Wait until blocker is active.
	for !started.Load() {
		time.Sleep(5 * time.Millisecond)
	}

	// Submit 5 normal-priority requests (they will wait in queue).
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
			req.Header.Set("X-Priority", "normal")
			req.Header.Set("X-Label", "normal")
			handler.ServeHTTP(httptest.NewRecorder(), req)
		}()
	}
	time.Sleep(30 * time.Millisecond) // let normals enter the waiting queue

	// Submit 1 high-priority request — should run before the waiting normals.
	wg.Add(1)
	go func() {
		defer wg.Done()
		req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
		req.Header.Set("X-Priority", "high")
		req.Header.Set("X-Label", "high")
		handler.ServeHTTP(httptest.NewRecorder(), req)
	}()
	time.Sleep(15 * time.Millisecond) // let high enter the queue after normals

	// Release the blocker.
	close(blockCh)
	wg.Wait()

	// "high" must appear before at least some "normal" entries in completion order.
	highIdx := -1
	for i, label := range completionOrder {
		if label == "high" {
			highIdx = i
			break
		}
	}
	if highIdx == -1 {
		t.Fatal("high-priority request never completed")
	}
	normalAfter := 0
	for _, label := range completionOrder[highIdx+1:] {
		if label == "normal" {
			normalAfter++
		}
	}
	if normalAfter == 0 {
		t.Errorf("high priority did not run before normals; order: %v", completionOrder)
	}
}

// TestQueueDepthGauge verifies the depth gauge is updated (OPT-10-T3).
func TestQueueDepthGauge(t *testing.T) {
	gauge := &gaugeStub{}
	q := server.NewQueueMiddleware(1, 10, gauge)

	blockCh := make(chan struct{})
	handler := q.Middleware()(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-blockCh
		w.WriteHeader(http.StatusOK)
	}))

	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			handler.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/v1/chat/completions", nil))
		}()
	}
	time.Sleep(40 * time.Millisecond)

	depth := gauge.Value()
	if depth != 3 {
		t.Errorf("expected depth 3, got %d", depth)
	}

	close(blockCh)
	wg.Wait()

	if gauge.Value() != 0 {
		t.Errorf("expected depth 0 after drain, got %d", gauge.Value())
	}
}

// TestHealthPathExemptsQueue verifies /health bypasses queue (OPT-10-T4 adjacent).
func TestHealthPathExemptsQueue(t *testing.T) {
	// maxQueue=0 → every non-exempt request gets 503.
	q := server.NewQueueMiddleware(0, 0, nil)
	handler := q.Middleware()(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest("GET", "/health/live", nil))
	if rec.Code != http.StatusOK {
		t.Errorf("/health/live: expected 200, got %d", rec.Code)
	}

	rec2 := httptest.NewRecorder()
	handler.ServeHTTP(rec2, httptest.NewRequest("POST", "/v1/chat/completions", nil))
	if rec2.Code != http.StatusServiceUnavailable {
		t.Errorf("/v1/ with full queue: expected 503, got %d", rec2.Code)
	}
}
