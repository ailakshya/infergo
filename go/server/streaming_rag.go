package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// StreamingRAGRequest is the body for POST /v1/rag/stream.
type StreamingRAGRequest struct {
	Model      string `json:"model"`       // LLM model
	EmbedModel string `json:"embed_model"` // Embedding model
	Query      string `json:"query"`
	K          int    `json:"k,omitempty"` // number of docs to retrieve (default 5)
}

// handleStreamingRAG implements OPT-45: retrieve context then stream LLM response.
// The retrieval happens first (fast), then LLM streams tokens as they generate.
func (s *Server) handleStreamingRAG(w http.ResponseWriter, r *http.Request) {
	var req StreamingRAGRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	if req.K <= 0 {
		req.K = 5
	}

	// Step 1: Retrieve relevant documents (fast — embedding + search)
	// Get embedding model
	embedRef, err := s.registry.Get(req.EmbedModel)
	if err != nil {
		writeError(w, http.StatusNotFound, "embed model not found: "+err.Error())
		return
	}
	defer embedRef.Release()

	// Get LLM model
	llmRef, err := s.registry.Get(req.Model)
	if err != nil {
		writeError(w, http.StatusNotFound, "llm model not found: "+err.Error())
		return
	}
	defer llmRef.Release()

	llm, ok := llmRef.Model.(StreamingLLMModel)
	if !ok {
		writeError(w, http.StatusBadRequest, "model does not support streaming")
		return
	}

	// Search for relevant context
	searchModel, ok := embedRef.Model.(SearchModel)
	var contextText string
	if ok {
		results, err := searchModel.Search(r.Context(), req.Query, req.K)
		if err == nil && len(results) > 0 {
			var sb strings.Builder
			for i, result := range results {
				sb.WriteString(fmt.Sprintf("Document %d:\n%s\n\n", i+1, result.Metadata))
			}
			contextText = sb.String()
		}
	}

	// Step 2: Build prompt with context
	prompt := fmt.Sprintf(`<|im_start|>system
Answer the question using the provided context. If the context doesn't contain the answer, say so.

Context:
%s<|im_end|>
<|im_start|>user
%s<|im_end|>
<|im_start|>assistant
`, contextText, req.Query)

	// Step 3: Stream LLM response
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	tokenCh, err := llm.Stream(context.Background(), prompt, 256, 0.7)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	for token := range tokenCh {
		chunk := map[string]interface{}{
			"choices": []map[string]interface{}{
				{"delta": map[string]string{"content": token}},
			},
		}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}
