package server

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
)

// handleEmbeddingsBinary serves embeddings as raw float32 bytes.
// POST /v1/embeddings/binary — request is JSON, response is binary float32.
// Response format: 4-byte dim (little-endian int32) + N*dim float32 values.
// ~10x less data than JSON, zero serialization overhead.
func (s *Server) handleEmbeddingsBinary(w http.ResponseWriter, r *http.Request) {
	var req EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if req.Model == "" || len(req.InputArr) == 0 {
		writeError(w, http.StatusBadRequest, "model and input required")
		return
	}

	ref, err := s.registry.Get(req.Model)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	defer ref.Release()

	emb, ok := ref.Model.(EmbeddingModel)
	if !ok {
		writeError(w, http.StatusBadRequest, "model does not support embeddings")
		return
	}

	var vecs [][]float32
	if batch, ok2 := emb.(BatchEmbeddingModel); ok2 && len(req.InputArr) > 1 {
		vecs, err = batch.EmbedBatch(r.Context(), req.InputArr)
	} else {
		vecs = make([][]float32, len(req.InputArr))
		for i, input := range req.InputArr {
			vecs[i], err = emb.Embed(r.Context(), input)
			if err != nil {
				break
			}
		}
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	// Write binary response: [dim:int32] [vec0:float32*dim] [vec1:float32*dim] ...
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("X-Embedding-Dim", fmt.Sprintf("%d", len(vecs[0])))
	w.Header().Set("X-Embedding-Count", fmt.Sprintf("%d", len(vecs)))

	dim := int32(len(vecs[0]))
	binary.Write(w, binary.LittleEndian, dim)
	for _, vec := range vecs {
		for _, v := range vec {
			binary.Write(w, binary.LittleEndian, math.Float32bits(v))
		}
	}
}

// writeJSONFast writes embedding response with pre-formatted float arrays.
// ~3x faster than json.Encoder for float32 arrays.
func writeEmbeddingJSONFast(w http.ResponseWriter, model string, vecs [][]float32, inputs []string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	io.WriteString(w, `{"object":"list","model":"`)
	io.WriteString(w, model)
	io.WriteString(w, `","data":[`)

	for i, vec := range vecs {
		if i > 0 {
			io.WriteString(w, ",")
		}
		io.WriteString(w, `{"object":"embedding","index":`)
		fmt.Fprintf(w, "%d", i)
		io.WriteString(w, `,"embedding":[`)
		for j, v := range vec {
			if j > 0 {
				io.WriteString(w, ",")
			}
			fmt.Fprintf(w, "%.6g", v)
		}
		io.WriteString(w, "]}")
	}

	totalTokens := 0
	for _, inp := range inputs {
		totalTokens += len(strings.Fields(inp))
	}

	io.WriteString(w, `],"usage":{"prompt_tokens":`)
	fmt.Fprintf(w, "%d", totalTokens)
	io.WriteString(w, `,"total_tokens":`)
	fmt.Fprintf(w, "%d", totalTokens)
	io.WriteString(w, "}}")
}
