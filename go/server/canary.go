package server

import (
	"encoding/json"
	"io"
	"log"
	"math/rand"
	"net/http"
	"sync"
	"sync/atomic"
)

// CanaryConfig defines a canary deployment that routes a percentage of traffic
// to a new model while monitoring error rates for automatic promotion or rollback.
type CanaryConfig struct {
	BaseModel        string  `json:"base_model"`
	NewModel         string  `json:"new_model"`
	TrafficPct       float64 `json:"traffic_pct"`         // 0.0 - 1.0
	MaxErrorRate     float64 `json:"max_error_rate"`       // auto-rollback if exceeded (e.g. 0.05 = 5%)
	AutoPromoteAfter int64   `json:"auto_promote_after"`   // promote after N successful canary requests
}

// CanaryDeploy manages a canary deployment lifecycle.
type CanaryDeploy struct {
	mu            sync.RWMutex
	config        *CanaryConfig
	canaryTotal   atomic.Int64
	canaryErrors  atomic.Int64
	rolledBack    bool
	promoted      bool
}

// NewCanaryDeploy creates a canary deployment manager.
func NewCanaryDeploy() *CanaryDeploy {
	return &CanaryDeploy{}
}

// Configure sets up a new canary deployment. Any existing canary is replaced.
func (cd *CanaryDeploy) Configure(cfg CanaryConfig) {
	cd.mu.Lock()
	defer cd.mu.Unlock()

	cd.config = &cfg
	cd.canaryTotal.Store(0)
	cd.canaryErrors.Store(0)
	cd.rolledBack = false
	cd.promoted = false
}

// Cancel cancels the current canary deployment.
func (cd *CanaryDeploy) Cancel() bool {
	cd.mu.Lock()
	defer cd.mu.Unlock()
	if cd.config == nil {
		return false
	}
	cd.config = nil
	return true
}

// Route decides which model to route a request to. Returns (model, isCanary).
// If no canary is active, returns the original model name unchanged.
func (cd *CanaryDeploy) Route(requestModel string) (string, bool) {
	cd.mu.RLock()
	cfg := cd.config
	rolledBack := cd.rolledBack
	promoted := cd.promoted
	cd.mu.RUnlock()

	if cfg == nil || rolledBack {
		return requestModel, false
	}

	// If promoted, all traffic goes to new model.
	if promoted {
		if requestModel == cfg.BaseModel {
			return cfg.NewModel, false
		}
		return requestModel, false
	}

	// Only intercept requests for the base model.
	if requestModel != cfg.BaseModel {
		return requestModel, false
	}

	// Route traffic_pct to the canary.
	if rand.Float64() < cfg.TrafficPct {
		return cfg.NewModel, true
	}
	return cfg.BaseModel, false
}

// RecordSuccess records a successful canary request and checks for auto-promotion.
func (cd *CanaryDeploy) RecordSuccess() {
	total := cd.canaryTotal.Add(1)

	cd.mu.RLock()
	cfg := cd.config
	promoted := cd.promoted
	rolledBack := cd.rolledBack
	cd.mu.RUnlock()

	if cfg == nil || promoted || rolledBack {
		return
	}

	// Check auto-promote condition.
	if cfg.AutoPromoteAfter > 0 && total >= cfg.AutoPromoteAfter {
		errors := cd.canaryErrors.Load()
		errorRate := float64(errors) / float64(total)
		if errorRate <= cfg.MaxErrorRate {
			cd.mu.Lock()
			if !cd.promoted && !cd.rolledBack {
				cd.promoted = true
				log.Printf("[canary] auto-promoted: %s -> %s (total=%d, errors=%d, rate=%.4f)",
					cfg.BaseModel, cfg.NewModel, total, errors, errorRate)
			}
			cd.mu.Unlock()
		}
	}
}

// RecordError records a failed canary request and checks for auto-rollback.
func (cd *CanaryDeploy) RecordError() {
	cd.canaryErrors.Add(1)
	total := cd.canaryTotal.Add(1)
	errors := cd.canaryErrors.Load()

	cd.mu.RLock()
	cfg := cd.config
	rolledBack := cd.rolledBack
	cd.mu.RUnlock()

	if cfg == nil || rolledBack {
		return
	}

	// Need at least 10 requests before checking error rate.
	if total < 10 {
		return
	}

	errorRate := float64(errors) / float64(total)
	if errorRate > cfg.MaxErrorRate {
		cd.mu.Lock()
		if !cd.rolledBack {
			cd.rolledBack = true
			log.Printf("[canary] auto-rollback: %s (error_rate=%.4f > max=%.4f, total=%d)",
				cfg.NewModel, errorRate, cfg.MaxErrorRate, total)
		}
		cd.mu.Unlock()
	}
}

// Status returns the current canary status.
func (cd *CanaryDeploy) Status() *CanaryStatus {
	cd.mu.RLock()
	defer cd.mu.RUnlock()

	if cd.config == nil {
		return nil
	}

	total := cd.canaryTotal.Load()
	errors := cd.canaryErrors.Load()
	var errorRate float64
	if total > 0 {
		errorRate = float64(errors) / float64(total)
	}

	state := "active"
	if cd.promoted {
		state = "promoted"
	} else if cd.rolledBack {
		state = "rolled_back"
	}

	return &CanaryStatus{
		Config:      *cd.config,
		State:       state,
		TotalReqs:   total,
		ErrorCount:  errors,
		ErrorRate:   errorRate,
	}
}

// CanaryStatus is the JSON-serializable status of a canary deployment.
type CanaryStatus struct {
	Config     CanaryConfig `json:"config"`
	State      string       `json:"state"` // "active", "promoted", "rolled_back"
	TotalReqs  int64        `json:"total_requests"`
	ErrorCount int64        `json:"error_count"`
	ErrorRate  float64      `json:"error_rate"`
}

// ─── HTTP handlers ──────────────────────────────────────────────────────────

// handleCanaryCreate handles POST /v1/admin/canary — configure a canary deployment.
func (s *Server) handleCanaryCreate(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<16))
	if err != nil {
		writeError(w, http.StatusBadRequest, "read body: "+err.Error())
		return
	}

	var cfg CanaryConfig
	if err := json.Unmarshal(body, &cfg); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request: "+err.Error())
		return
	}
	if cfg.BaseModel == "" || cfg.NewModel == "" {
		writeError(w, http.StatusBadRequest, "base_model and new_model are required")
		return
	}
	if cfg.TrafficPct <= 0 || cfg.TrafficPct > 1 {
		writeError(w, http.StatusBadRequest, "traffic_pct must be between 0 and 1")
		return
	}
	if cfg.MaxErrorRate <= 0 {
		cfg.MaxErrorRate = 0.05 // default 5%
	}

	if s.canary == nil {
		writeError(w, http.StatusInternalServerError, "canary deploy not configured")
		return
	}

	s.canary.Configure(cfg)

	status := s.canary.Status()
	writeJSON(w, http.StatusOK, status)
}

// handleCanaryDelete handles DELETE /v1/admin/canary — cancel a canary deployment.
func (s *Server) handleCanaryDelete(w http.ResponseWriter, r *http.Request) {
	if s.canary == nil {
		writeError(w, http.StatusInternalServerError, "canary deploy not configured")
		return
	}

	if !s.canary.Cancel() {
		writeError(w, http.StatusNotFound, "no active canary deployment")
		return
	}

	writeJSON(w, http.StatusOK, map[string]string{"status": "cancelled"})
}

// handleCanaryStatus handles GET /v1/admin/canary — get canary status.
func (s *Server) handleCanaryStatus(w http.ResponseWriter, r *http.Request) {
	if s.canary == nil {
		writeError(w, http.StatusInternalServerError, "canary deploy not configured")
		return
	}

	status := s.canary.Status()
	if status == nil {
		writeError(w, http.StatusNotFound, "no active canary deployment")
		return
	}

	writeJSON(w, http.StatusOK, status)
}
