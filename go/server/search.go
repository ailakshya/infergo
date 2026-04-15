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

// HybridSearchModel extends SearchModel with BM25 and hybrid search.
type HybridSearchModel interface {
	SearchModel
	// SearchBM25 performs keyword-only BM25 search.
	SearchBM25(ctx context.Context, query string, k int) ([]SearchHit, error)
	// SearchHybrid performs combined BM25 + vector search.
	// alpha: 1.0 = pure vector, 0.0 = pure BM25.
	SearchHybrid(ctx context.Context, query string, k int, alpha float32) ([]SearchHit, error)
}

// SearchHit is one result from a search.
type SearchHit struct {
	ID       int64   `json:"id"`
	Score    float32 `json:"score"`    // similarity score (higher = better)
	Metadata string  `json:"metadata,omitempty"`
}

// SearchRequest is the body for POST /v1/search.
type SearchRequest struct {
	Model string  `json:"model"`
	Query string  `json:"query"`
	K     int     `json:"k"`
	Mode  string  `json:"mode,omitempty"`  // "vector" (default), "bm25", "hybrid"
	Alpha float32 `json:"alpha,omitempty"` // hybrid weight: 1.0=pure vector, 0.0=pure BM25 (default 0.5)
}

// SearchResponse is the response for POST /v1/search.
type SearchResponse struct {
	Model   string      `json:"model"`
	Mode    string      `json:"mode"`
	Results []SearchHit `json:"results"`
}

// handleSearch implements POST /v1/search.
// Supports mode: "vector" (default), "bm25", "hybrid".
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
	if req.Mode == "" {
		req.Mode = "vector"
	}
	if req.Mode == "hybrid" && req.Alpha == 0 {
		req.Alpha = 0.5 // default hybrid weight
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
		if _, embOk := ref.Model.(EmbeddingModel); embOk {
			writeJSON(w, http.StatusOK, SearchResponse{Model: req.Model, Mode: req.Mode, Results: []SearchHit{}})
			return
		}
		writeError(w, http.StatusBadRequest, "model does not support search")
		return
	}

	var hits []SearchHit

	switch req.Mode {
	case "vector":
		hits, err = sm.Search(r.Context(), req.Query, req.K)

	case "bm25":
		hsm, hybridOk := ref.Model.(HybridSearchModel)
		if !hybridOk {
			writeError(w, http.StatusBadRequest, "model does not support BM25 search")
			return
		}
		hits, err = hsm.SearchBM25(r.Context(), req.Query, req.K)

	case "hybrid":
		hsm, hybridOk := ref.Model.(HybridSearchModel)
		if !hybridOk {
			writeError(w, http.StatusBadRequest, "model does not support hybrid search")
			return
		}
		hits, err = hsm.SearchHybrid(r.Context(), req.Query, req.K, req.Alpha)

	default:
		writeError(w, http.StatusBadRequest, "invalid mode: must be 'vector', 'bm25', or 'hybrid'")
		return
	}

	if err != nil {
		writeError(w, http.StatusInternalServerError, "search failed: "+err.Error())
		return
	}

	writeJSON(w, http.StatusOK, SearchResponse{Model: req.Model, Mode: req.Mode, Results: hits})
}
