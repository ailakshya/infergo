package server

import (
	"context"
	"encoding/json"
	"net/http"
	"sort"
)

// RerankModel scores query-document relevance for RAG pipelines.
type RerankModel interface {
	Model
	Rerank(ctx context.Context, query string, documents []string) ([]RerankResult, error)
}

// RerankRequest is the body for POST /v1/rerank.
type RerankRequest struct {
	Model     string   `json:"model"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	TopN      int      `json:"top_n,omitempty"`
}

// RerankResult is one scored document.
type RerankResult struct {
	Index    int     `json:"index"`
	Score    float32 `json:"relevance_score"`
	Document string  `json:"document,omitempty"`
}

// RerankResponse is the response for POST /v1/rerank.
type RerankResponse struct {
	Model   string         `json:"model"`
	Results []RerankResult `json:"results"`
}

func (s *Server) handleRerank(w http.ResponseWriter, r *http.Request) {
	var req RerankRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if req.Model == "" || req.Query == "" || len(req.Documents) == 0 {
		writeError(w, http.StatusBadRequest, "model, query, and documents required")
		return
	}

	ref, err := s.registry.Get(req.Model)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	defer ref.Release()

	// Try dedicated reranker first, fall back to embedding-based scoring
	if rm, ok := ref.Model.(RerankModel); ok {
		results, err := rm.Rerank(r.Context(), req.Query, req.Documents)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		if req.TopN > 0 && req.TopN < len(results) {
			results = results[:req.TopN]
		}
		writeJSON(w, http.StatusOK, RerankResponse{Model: req.Model, Results: results})
		return
	}

	// Fallback: use embedding model for cosine-similarity reranking
	emb, ok := ref.Model.(EmbeddingModel)
	if !ok {
		writeError(w, http.StatusBadRequest, "model does not support reranking or embeddings")
		return
	}

	queryVec, err := emb.Embed(r.Context(), req.Query)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	results := make([]RerankResult, len(req.Documents))
	for i, doc := range req.Documents {
		docVec, err := emb.Embed(r.Context(), doc)
		if err != nil {
			continue
		}
		results[i] = RerankResult{
			Index:    i,
			Score:    cosineSimilarity(queryVec, docVec),
			Document: doc,
		}
	}

	sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	if req.TopN > 0 && req.TopN < len(results) {
		results = results[:req.TopN]
	}

	writeJSON(w, http.StatusOK, RerankResponse{Model: req.Model, Results: results})
}

func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float32
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	denom := sqrt32(na) * sqrt32(nb)
	if denom < 1e-7 {
		return 0
	}
	return dot / denom
}

func sqrt32(x float32) float32 {
	if x <= 0 {
		return 0
	}
	// Newton's method
	r := x
	for i := 0; i < 10; i++ {
		r = (r + x/r) / 2
	}
	return r
}
