package server

import (
	"encoding/json"
	"net/http"
	"sync/atomic"
	"time"
)

// HealthStatus is the JSON body returned by health endpoints.
type HealthStatus struct {
	Status    string            `json:"status"`              // "ok" | "degraded" | "not_ready"
	Timestamp string            `json:"timestamp"`
	Details   map[string]string `json:"details,omitempty"`
}

// HealthChecker tracks liveness and readiness state for the server.
// Liveness:  always true once the process is up.
// Readiness: true once at least one model is loaded (configurable via checks).
type HealthChecker struct {
	registry     *Registry
	minModels    int             // minimum models required for ready
	live         atomic.Bool     // can be set false to force a liveness failure (e.g. OOM)
	extraChecks  []ReadyCheck    // optional additional readiness checks
}

// ReadyCheck is a function that returns an error string if not ready,
// or an empty string if ready.
type ReadyCheck func() string

// NewHealthChecker creates a HealthChecker backed by the given Registry.
// minModels is the minimum number of loaded models required to be "ready".
// Pass 0 to consider the server ready even with no models loaded.
func NewHealthChecker(reg *Registry, minModels int) *HealthChecker {
	h := &HealthChecker{registry: reg, minModels: minModels}
	h.live.Store(true)
	return h
}

// AddReadyCheck registers an additional readiness check.
func (h *HealthChecker) AddReadyCheck(name string, fn func() error) {
	h.extraChecks = append(h.extraChecks, func() string {
		if err := fn(); err != nil {
			return name + ": " + err.Error()
		}
		return ""
	})
}

// SetLive allows forcing the liveness state (e.g. call SetLive(false) on OOM).
func (h *HealthChecker) SetLive(live bool) {
	h.live.Store(live)
}

// LiveHandler handles GET /health/live.
// Returns 200 as long as the process is running and hasn't been marked unhealthy.
func (h *HealthChecker) LiveHandler(w http.ResponseWriter, r *http.Request) {
	if !h.live.Load() {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(HealthStatus{
			Status:    "degraded",
			Timestamp: time.Now().UTC().Format(time.RFC3339),
			Details:   map[string]string{"reason": "liveness check failed"},
		})
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(HealthStatus{
		Status:    "ok",
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	})
}

// ReadyHandler handles GET /health/ready.
// Returns 200 only if the required number of models are loaded and all
// extra readiness checks pass.
func (h *HealthChecker) ReadyHandler(w http.ResponseWriter, r *http.Request) {
	details := map[string]string{}
	ready := true

	// Check model count
	loaded := len(h.registry.Names())
	if loaded < h.minModels {
		ready = false
		details["models"] = "not enough models loaded"
	} else {
		details["models"] = "ok"
	}

	// Run extra checks
	for _, check := range h.extraChecks {
		if msg := check(); msg != "" {
			ready = false
			details["check"] = msg
		}
	}

	w.Header().Set("Content-Type", "application/json")
	if ready {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(HealthStatus{
			Status:    "ok",
			Timestamp: time.Now().UTC().Format(time.RFC3339),
			Details:   details,
		})
	} else {
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(HealthStatus{
			Status:    "not_ready",
			Timestamp: time.Now().UTC().Format(time.RFC3339),
			Details:   details,
		})
	}
}

// RegisterRoutes adds /health/live and /health/ready to the given mux.
func (h *HealthChecker) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /health/live", h.LiveHandler)
	mux.HandleFunc("GET /health/ready", h.ReadyHandler)
}
