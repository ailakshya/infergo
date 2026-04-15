package server

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"sync"
	"time"
)

// Trigger defines a rule that fires an action when detection conditions are met.
type Trigger struct {
	Name                string  `json:"name"`
	ClassName           string  `json:"class,omitempty"`              // class name to match (empty = any)
	ClassID             int     `json:"class_id,omitempty"`           // class ID to match (-1 = any)
	ConfidenceThreshold float32 `json:"confidence_threshold"`         // min confidence
	CountThreshold      int     `json:"count_threshold"`              // min count of matching detections
	WebhookURL          string  `json:"webhook_url"`                  // URL to POST when triggered
	Cooldown            string  `json:"cooldown,omitempty"`           // duration string (e.g. "30s", "5m")
	cooldownDuration    time.Duration
}

// TriggerEvent is the payload sent to the webhook.
type TriggerEvent struct {
	TriggerName string           `json:"trigger_name"`
	FiredAt     string           `json:"fired_at"`
	MatchCount  int              `json:"match_count"`
	Detections  []DetectedObject `json:"detections"`
}

// TriggerEngine evaluates detection results against a set of triggers.
type TriggerEngine struct {
	mu       sync.RWMutex
	triggers map[string]*triggerState
	client   *http.Client
}

type triggerState struct {
	Trigger  Trigger
	LastFire time.Time
}

// NewTriggerEngine creates a new trigger engine.
func NewTriggerEngine() *TriggerEngine {
	return &TriggerEngine{
		triggers: make(map[string]*triggerState),
		client:   &http.Client{Timeout: 10 * time.Second},
	}
}

// Add registers a new trigger or updates an existing one.
func (te *TriggerEngine) Add(t Trigger) error {
	// Parse cooldown duration.
	var cd time.Duration
	if t.Cooldown != "" {
		var err error
		cd, err = time.ParseDuration(t.Cooldown)
		if err != nil {
			return err
		}
	}
	t.cooldownDuration = cd

	te.mu.Lock()
	defer te.mu.Unlock()
	te.triggers[t.Name] = &triggerState{Trigger: t}
	return nil
}

// Remove deletes a trigger by name. Returns true if found.
func (te *TriggerEngine) Remove(name string) bool {
	te.mu.Lock()
	defer te.mu.Unlock()
	_, ok := te.triggers[name]
	if ok {
		delete(te.triggers, name)
	}
	return ok
}

// List returns all configured triggers.
func (te *TriggerEngine) List() []Trigger {
	te.mu.RLock()
	defer te.mu.RUnlock()
	out := make([]Trigger, 0, len(te.triggers))
	for _, ts := range te.triggers {
		out = append(out, ts.Trigger)
	}
	return out
}

// Evaluate checks all triggers against the given detections and fires webhooks
// for any that match. classNames maps class IDs to class name strings (optional).
// Returns the number of triggers fired.
func (te *TriggerEngine) Evaluate(detections []DetectedObject, classNames map[int]string) int {
	if len(detections) == 0 {
		return 0
	}

	te.mu.Lock()
	defer te.mu.Unlock()

	now := time.Now()
	fired := 0

	for _, ts := range te.triggers {
		// Check cooldown.
		if ts.Trigger.cooldownDuration > 0 && !ts.LastFire.IsZero() {
			if now.Sub(ts.LastFire) < ts.Trigger.cooldownDuration {
				continue
			}
		}

		// Count matching detections.
		var matches []DetectedObject
		for _, d := range detections {
			if d.Confidence < ts.Trigger.ConfidenceThreshold {
				continue
			}
			// Check class match.
			if ts.Trigger.ClassID > 0 && d.ClassID != ts.Trigger.ClassID {
				continue
			}
			if ts.Trigger.ClassName != "" && classNames != nil {
				name, ok := classNames[d.ClassID]
				if !ok || name != ts.Trigger.ClassName {
					continue
				}
			}
			matches = append(matches, d)
		}

		threshold := ts.Trigger.CountThreshold
		if threshold <= 0 {
			threshold = 1
		}
		if len(matches) < threshold {
			continue
		}

		// Fire webhook.
		ts.LastFire = now
		fired++

		event := TriggerEvent{
			TriggerName: ts.Trigger.Name,
			FiredAt:     now.Format(time.RFC3339),
			MatchCount:  len(matches),
			Detections:  matches,
		}

		go te.fireWebhook(ts.Trigger.WebhookURL, event)
	}

	return fired
}

// fireWebhook sends the event payload to the webhook URL.
func (te *TriggerEngine) fireWebhook(url string, event TriggerEvent) {
	body, err := json.Marshal(event)
	if err != nil {
		log.Printf("[trigger] marshal error for %s: %v", event.TriggerName, err)
		return
	}

	resp, err := te.client.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		log.Printf("[trigger] webhook %s failed: %v", event.TriggerName, err)
		return
	}
	resp.Body.Close()

	if resp.StatusCode >= 300 {
		log.Printf("[trigger] webhook %s returned %d", event.TriggerName, resp.StatusCode)
	}
}

// ─── HTTP handlers ──────────────────────────────────────────────────────────

// handleTriggersCreate handles POST /v1/admin/triggers — create or update a trigger.
func (s *Server) handleTriggersCreate(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<16))
	if err != nil {
		writeError(w, http.StatusBadRequest, "read body: "+err.Error())
		return
	}

	var t Trigger
	if err := json.Unmarshal(body, &t); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request: "+err.Error())
		return
	}
	if t.Name == "" {
		writeError(w, http.StatusBadRequest, "name field is required")
		return
	}
	if t.WebhookURL == "" {
		writeError(w, http.StatusBadRequest, "webhook_url field is required")
		return
	}

	if s.triggers == nil {
		writeError(w, http.StatusInternalServerError, "trigger engine not configured")
		return
	}

	if err := s.triggers.Add(t); err != nil {
		writeError(w, http.StatusBadRequest, "invalid trigger: "+err.Error())
		return
	}

	writeJSON(w, http.StatusOK, map[string]string{
		"status": "ok",
		"name":   t.Name,
	})
}

// handleTriggersList handles GET /v1/admin/triggers — list all triggers.
func (s *Server) handleTriggersList(w http.ResponseWriter, r *http.Request) {
	if s.triggers == nil {
		writeJSON(w, http.StatusOK, []Trigger{})
		return
	}
	writeJSON(w, http.StatusOK, s.triggers.List())
}

// handleTriggersDelete handles DELETE /v1/admin/triggers/{name} — delete a trigger.
func (s *Server) handleTriggersDelete(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if name == "" {
		writeError(w, http.StatusBadRequest, "trigger name is required")
		return
	}

	if s.triggers == nil {
		writeError(w, http.StatusInternalServerError, "trigger engine not configured")
		return
	}

	if !s.triggers.Remove(name) {
		writeError(w, http.StatusNotFound, "trigger not found")
		return
	}

	writeJSON(w, http.StatusOK, map[string]string{"status": "deleted"})
}
