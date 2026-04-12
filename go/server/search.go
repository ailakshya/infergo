package server

import (
	"context"
	"encoding/json"
	"net/http"
)

// SearchModel is a model that can embed a query and search a vector index.
type SearchModel interface {
	Model
	Search(ctx context.Context, query string, k int) ([]SearchHit, error)
	IndexSize() int
}

// SearchHit is one result from a vector search.
type SearchHit struct {
	ID       int64   `json:"id"`
	Score    float32 `json:"score"`    // cosine similarity (1 = identical)
	Metadata string  `json:"metadata,omitempty"`
}

// SearchRequest is the body for POST /v1/search.
type SearchRequest struct {
	Model string `json:"model"`
	Query string `json:"query"`
	K     int    `json:"k"`
}

// SearchResponse is the response for POST /v1/search.
type SearchResponse struct {
	Model   string      `json:"model"`
	Results []SearchHit `json:"results"`
}

// handleSearch implements POST /v1/search.
func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request) {
	var req SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "model field is required")
		return
	}
	if req.Query == "" {
		writeError(w, http.StatusBadRequest, "query must not be empty")
		return
	}
	if req.K <= 0 {
		req.K = 10
	}

	ref, err := s.registry.Get(req.Model)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	defer ref.Release()

	sm, ok := ref.Model.(SearchModel)
	if !ok {
		// Fallback: if it's an embedding model, return empty results
		// (index is empty until documents are ingested)
		if _, embOk := ref.Model.(EmbeddingModel); embOk {
			writeJSON(w, http.StatusOK, SearchResponse{Model: req.Model, Results: []SearchHit{}})
			return
		}
		writeError(w, http.StatusBadRequest, "model does not support search")
		return
	}

	hits, err := sm.Search(r.Context(), req.Query, req.K)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "search failed: "+err.Error())
		return
	}

	writeJSON(w, http.StatusOK, SearchResponse{Model: req.Model, Results: hits})
}
