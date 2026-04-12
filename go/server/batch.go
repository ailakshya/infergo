package server

import (
	"encoding/json"
	"net/http"
	"sync"
	"time"
)

// AsyncBatchRequest submits multiple prompts for async processing.
type AsyncBatchRequest struct {
	Model     string   `json:"model"`
	Prompts   []string `json:"prompts"`
	MaxTokens int      `json:"max_tokens,omitempty"`
}

// AsyncBatchResponse returns a batch job ID.
type AsyncBatchResponse struct {
	BatchID string `json:"batch_id"`
	Status  string `json:"status"`
	Total   int    `json:"total"`
}

// BatchResult is the result of a completed batch job.
type BatchResult struct {
	BatchID   string   `json:"batch_id"`
	Status    string   `json:"status"`
	Results   []string `json:"results,omitempty"`
	Completed int      `json:"completed"`
	Total     int      `json:"total"`
}

// BatchStore holds pending and completed batch jobs.
type BatchStore struct {
	mu   sync.RWMutex
	jobs map[string]*batchJob
}

type batchJob struct {
	id        string
	prompts   []string
	results   []string
	status    string
	createdAt time.Time
}

func NewBatchStore() *BatchStore {
	return &BatchStore{jobs: make(map[string]*batchJob)}
}

func (s *Server) handleBatchCreate(w http.ResponseWriter, r *http.Request) {
	var req AsyncBatchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if req.Model == "" || len(req.Prompts) == 0 {
		writeError(w, http.StatusBadRequest, "model and prompts required")
		return
	}

	id := newID("batch")
	writeJSON(w, http.StatusAccepted, AsyncBatchResponse{
		BatchID: id,
		Status:  "queued",
		Total:   len(req.Prompts),
	})
}

func (s *Server) handleBatchStatus(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	if id == "" {
		writeError(w, http.StatusBadRequest, "id required")
		return
	}
	writeJSON(w, http.StatusOK, BatchResult{
		BatchID: id,
		Status:  "processing",
	})
}
