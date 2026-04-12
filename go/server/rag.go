package server

import (
	"encoding/json"
	"net/http"
)

// RAGRequest is the body for POST /v1/rag.
// One call: question in → cited answer out.
type RAGRequest struct {
	Model      string `json:"model"`       // LLM model name
	EmbedModel string `json:"embed_model"` // embedding model name
	Query      string `json:"query"`
	K          int    `json:"k,omitempty"`          // number of context docs (default 5)
	MaxTokens  int    `json:"max_tokens,omitempty"` // generation length
}

// RAGResponse contains the generated answer with sources.
type RAGResponse struct {
	Model   string      `json:"model"`
	Answer  string      `json:"answer"`
	Sources []RAGSource `json:"sources,omitempty"`
}

// RAGSource is one retrieved document used as context.
type RAGSource struct {
	ID       int64   `json:"id"`
	Score    float32 `json:"score"`
	Content  string  `json:"content"`
}

func (s *Server) handleRAG(w http.ResponseWriter, r *http.Request) {
	var req RAGRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if req.Query == "" || req.Model == "" {
		writeError(w, http.StatusBadRequest, "model and query required")
		return
	}

	// This endpoint requires both LLM and embedding models + a vector DB.
	// The full pipeline runs in C++ via infer_rag_pipeline when wired up.
	writeError(w, http.StatusNotImplemented,
		"RAG pipeline requires: --model llm:model.gguf --model embed:model.onnx --vectordb path")
}
