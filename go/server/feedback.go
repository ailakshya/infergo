package server

import (
	"encoding/json"
	"net/http"
	"sync"
	"time"
)

// FeedbackStore collects user feedback on responses for quality tracking.
type FeedbackStore struct {
	mu       sync.Mutex
	entries  []FeedbackEntry
	maxSize  int
}

// FeedbackEntry records one feedback event.
type FeedbackEntry struct {
	RequestID string    `json:"request_id"`
	Rating    string    `json:"rating"`    // "positive" or "negative"
	Comment   string    `json:"comment,omitempty"`
	Timestamp time.Time `json:"timestamp"`
}

// NewFeedbackStore creates a feedback store with max entries.
func NewFeedbackStore(maxSize int) *FeedbackStore {
	if maxSize <= 0 {
		maxSize = 10000
	}
	return &FeedbackStore{
		entries: make([]FeedbackEntry, 0, 256),
		maxSize: maxSize,
	}
}

// Add records a feedback entry.
func (f *FeedbackStore) Add(entry FeedbackEntry) {
	f.mu.Lock()
	defer f.mu.Unlock()
	entry.Timestamp = time.Now()
	f.entries = append(f.entries, entry)
	if len(f.entries) > f.maxSize {
		f.entries = f.entries[len(f.entries)-f.maxSize:]
	}
}

// Stats returns positive/negative counts.
func (f *FeedbackStore) Stats() (positive, negative, total int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, e := range f.entries {
		if e.Rating == "positive" {
			positive++
		} else {
			negative++
		}
	}
	return positive, negative, len(f.entries)
}

func (s *Server) handleFeedback(w http.ResponseWriter, r *http.Request) {
	var req struct {
		RequestID string `json:"request_id"`
		Rating    string `json:"rating"`
		Comment   string `json:"comment"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid body: "+err.Error())
		return
	}
	if req.Rating != "positive" && req.Rating != "negative" {
		writeError(w, http.StatusBadRequest, "rating must be 'positive' or 'negative'")
		return
	}

	if s.feedback != nil {
		s.feedback.Add(FeedbackEntry{
			RequestID: req.RequestID,
			Rating:    req.Rating,
			Comment:   req.Comment,
		})
	}

	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleDeleteSession(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if id == "" {
		writeError(w, http.StatusBadRequest, "session id required")
		return
	}

	if s.memory != nil {
		if s.memory.Delete(id) {
			writeJSON(w, http.StatusOK, map[string]string{"status": "deleted", "session_id": id})
		} else {
			writeError(w, http.StatusNotFound, "session not found: "+id)
		}
	} else {
		writeError(w, http.StatusNotFound, "conversation memory not enabled")
	}
}
