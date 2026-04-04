package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"sync"
)

// Priority levels for request queuing.
const (
	PriorityHigh   = 0
	PriorityNormal = 1
	PriorityLow    = 2
)

// DepthSetter is a minimal interface for a gauge that tracks queue depth.
// prometheus.Gauge satisfies this interface; tests can pass a simple stub.
type DepthSetter interface {
	Set(float64)
}

// QueueMiddleware implements a bounded priority queue in front of HTTP handlers.
//
// Requests are dispatched in priority order (high → normal → low). At most
// maxActive requests call next.ServeHTTP() simultaneously. If the total number
// of requests (active + waiting) reaches maxQueue, new requests receive 503.
type QueueMiddleware struct {
	maxActive int
	maxQueue  int

	mu      sync.Mutex
	active  int
	waiting [3][]chan struct{} // index = priority level
	total   int               // active + sum(len(waiting[i]))

	depthGauge DepthSetter
}

// NewQueueMiddleware creates a queue that allows at most maxActive concurrent
// handlers and rejects requests once maxQueue total (active+waiting) is reached.
// If maxActive <= 0 it defaults to maxQueue (unlimited concurrency up to maxQueue).
// depthGauge is updated whenever the queue depth changes (may be nil).
func NewQueueMiddleware(maxActive, maxQueue int, depthGauge DepthSetter) *QueueMiddleware {
	if maxActive <= 0 {
		maxActive = maxQueue
	}
	return &QueueMiddleware{
		maxActive:  maxActive,
		maxQueue:   maxQueue,
		depthGauge: depthGauge,
	}
}

// Middleware returns an http.Handler middleware that enforces priority queuing.
// /health and /metrics paths are always exempt.
func (q *QueueMiddleware) Middleware() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if strings.HasPrefix(r.URL.Path, "/health") || r.URL.Path == "/metrics" {
				next.ServeHTTP(w, r)
				return
			}

			p := parsePriority(r.Header.Get("X-Priority"))

			ready, ok := q.enqueue(p)
			if !ok {
				writeQueueFullError(w)
				return
			}

			// Wait until the dispatcher gives us a slot.
			<-ready

			defer q.release()
			next.ServeHTTP(w, r)
		})
	}
}

// Depth returns the current number of active + waiting requests.
func (q *QueueMiddleware) Depth() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.total
}

// enqueue tries to add a request to the queue and returns a channel that will
// be closed when the request may proceed. Returns (nil, false) if queue is full.
func (q *QueueMiddleware) enqueue(p int) (chan struct{}, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()

	if q.total >= q.maxQueue {
		return nil, false
	}
	q.total++
	q.updateGauge()

	ready := make(chan struct{}, 1)
	q.waiting[p] = append(q.waiting[p], ready)

	// Dispatch immediately if we have an active slot available.
	q.tryDispatch()

	return ready, true
}

// release is called when a request finishes processing.
func (q *QueueMiddleware) release() {
	q.mu.Lock()
	defer q.mu.Unlock()

	q.active--
	q.total--
	q.updateGauge()
	q.tryDispatch()
}

// tryDispatch dispatches waiting requests into active slots. Must be called
// with q.mu held.
func (q *QueueMiddleware) tryDispatch() {
	for q.active < q.maxActive {
		dispatched := false
		for p := PriorityHigh; p <= PriorityLow; p++ {
			if len(q.waiting[p]) > 0 {
				ready := q.waiting[p][0]
				q.waiting[p] = q.waiting[p][1:]
				q.active++
				ready <- struct{}{}
				dispatched = true
				break
			}
		}
		if !dispatched {
			break
		}
	}
}

func (q *QueueMiddleware) updateGauge() {
	if q.depthGauge != nil {
		q.depthGauge.Set(float64(q.total))
	}
}

// parsePriority maps the X-Priority header value to a priority level.
func parsePriority(v string) int {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "high":
		return PriorityHigh
	case "low":
		return PriorityLow
	default:
		return PriorityNormal
	}
}

func writeQueueFullError(w http.ResponseWriter) {
	type errBody struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	var body errBody
	body.Error.Message = "server busy: request queue full"
	body.Error.Type = "queue_full_error"
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Retry-After", "5")
	w.WriteHeader(http.StatusServiceUnavailable)
	json.NewEncoder(w).Encode(body)
}
