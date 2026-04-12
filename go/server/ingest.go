package server

import (
	"encoding/json"
	"net/http"
)

// IngestRequest uploads documents to the vector DB.
type IngestRequest struct {
	Model     string   `json:"model"`      // embedding model
	Documents []string `json:"documents"`  // text chunks to embed and index
	IDs       []int64  `json:"ids,omitempty"` // optional IDs (auto-generated if empty)
	Metadata  []string `json:"metadata,omitempty"` // optional metadata per doc
}

// IngestResponse confirms how many documents were indexed.
type IngestResponse struct {
	Indexed int   `json:"indexed"`
	IDs     []int64 `json:"ids"`
}

func (s *Server) handleIngest(w http.ResponseWriter, r *http.Request) {
	var req IngestRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if req.Model == "" || len(req.Documents) == 0 {
		writeError(w, http.StatusBadRequest, "model and documents required")
		return
	}
	writeError(w, http.StatusNotImplemented,
		"document ingestion requires: --model embed:model.onnx --vectordb path")
}
