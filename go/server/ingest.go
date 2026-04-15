package server

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"strings"
	"time"
)

// ─── Document parsing ────────────────────────────────────────────────────────

// Chunk is a single piece of a parsed document.
type Chunk struct {
	Text       string            `json:"text"`
	Index      int               `json:"index"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

// ChunkerConfig controls how text is split into chunks.
type ChunkerConfig struct {
	ChunkSize    int    // target chars per chunk (default 500)
	ChunkOverlap int    // overlap between chunks (default 50)
	Separator    string // split on (default "\n\n")
}

func (c ChunkerConfig) withDefaults() ChunkerConfig {
	if c.ChunkSize <= 0 {
		c.ChunkSize = 500
	}
	if c.ChunkOverlap < 0 {
		c.ChunkOverlap = 0
	}
	if c.ChunkOverlap >= c.ChunkSize {
		c.ChunkOverlap = c.ChunkSize / 5
	}
	if c.Separator == "" {
		c.Separator = "\n\n"
	}
	return c
}

// ChunkText splits text into overlapping chunks of approximately ChunkSize characters.
// It first splits on Separator, then groups segments to fill chunks up to ChunkSize,
// respecting ChunkOverlap between consecutive chunks.
func ChunkText(text string, config ChunkerConfig) []Chunk {
	config = config.withDefaults()

	if len(text) == 0 {
		return nil
	}

	// If text fits in one chunk, return it directly.
	if len(text) <= config.ChunkSize {
		return []Chunk{{Text: text, Index: 0}}
	}

	// Split text on separator.
	segments := strings.Split(text, config.Separator)

	var chunks []Chunk
	var current strings.Builder
	idx := 0

	for _, seg := range segments {
		seg = strings.TrimSpace(seg)
		if seg == "" {
			continue
		}

		// If adding this segment would exceed chunk size, flush current chunk.
		if current.Len() > 0 && current.Len()+len(config.Separator)+len(seg) > config.ChunkSize {
			chunks = append(chunks, Chunk{Text: current.String(), Index: idx})
			idx++

			// Apply overlap: keep the tail of the current chunk.
			if config.ChunkOverlap > 0 {
				tail := current.String()
				if len(tail) > config.ChunkOverlap {
					tail = tail[len(tail)-config.ChunkOverlap:]
				}
				current.Reset()
				current.WriteString(tail)
			} else {
				current.Reset()
			}
		}

		// If a single segment exceeds chunk size, split it by character boundary.
		if len(seg) > config.ChunkSize {
			// Flush any partial content first.
			if current.Len() > 0 {
				chunks = append(chunks, Chunk{Text: current.String(), Index: idx})
				idx++
				current.Reset()
			}
			// Split oversized segment into fixed-size sub-chunks.
			for start := 0; start < len(seg); {
				end := start + config.ChunkSize
				if end > len(seg) {
					end = len(seg)
				}
				chunks = append(chunks, Chunk{Text: seg[start:end], Index: idx})
				idx++
				if config.ChunkOverlap > 0 && end < len(seg) {
					start = end - config.ChunkOverlap
				} else {
					start = end
				}
			}
			continue
		}

		if current.Len() > 0 {
			current.WriteString(config.Separator)
		}
		current.WriteString(seg)
	}

	// Flush remaining.
	if current.Len() > 0 {
		chunks = append(chunks, Chunk{Text: current.String(), Index: idx})
	}

	return chunks
}

// ─── Format-specific parsers ─────────────────────────────────────────────────

// ParseDocument parses a document by extension and returns text chunks.
// Supported: .txt, .md, .csv, .html/.htm
// Unsupported formats return an error.
func ParseDocument(filename string, content string, config ChunkerConfig) ([]Chunk, error) {
	ext := strings.ToLower(filepath.Ext(filename))
	switch ext {
	case ".txt", ".text":
		return ChunkText(content, config), nil
	case ".md", ".markdown":
		return ParseMarkdown(content, config), nil
	case ".csv":
		return ParseCSV(content, filename)
	case ".html", ".htm":
		stripped := StripHTML(content)
		return ChunkText(stripped, config), nil
	case ".pdf":
		return nil, fmt.Errorf("PDF format not supported, convert to text first")
	case ".docx", ".doc":
		return nil, fmt.Errorf("DOCX format not supported, convert to text first")
	default:
		// Treat unknown as plain text.
		return ChunkText(content, config), nil
	}
}

// ParseMarkdown splits markdown text on header boundaries (## / ### / # etc.)
// and returns one chunk per section, with the header as metadata.
func ParseMarkdown(content string, config ChunkerConfig) []Chunk {
	if len(content) == 0 {
		return nil
	}

	lines := strings.Split(content, "\n")
	var chunks []Chunk
	var currentHeader string
	var currentBody strings.Builder
	idx := 0

	flush := func() {
		body := strings.TrimSpace(currentBody.String())
		if body == "" {
			return
		}
		// If the section body is larger than chunk size, sub-chunk it.
		subChunks := ChunkText(body, config)
		for _, sc := range subChunks {
			meta := map[string]string{}
			if currentHeader != "" {
				meta["header"] = currentHeader
			}
			chunks = append(chunks, Chunk{
				Text:     sc.Text,
				Index:    idx,
				Metadata: meta,
			})
			idx++
		}
	}

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "#") {
			// Found a header — flush the previous section.
			flush()
			currentBody.Reset()
			currentHeader = strings.TrimSpace(strings.TrimLeft(trimmed, "#"))
			continue
		}
		if currentBody.Len() > 0 {
			currentBody.WriteString("\n")
		}
		currentBody.WriteString(line)
	}
	// Flush last section.
	flush()

	return chunks
}

// ParseCSV reads CSV content and returns one chunk per row (excluding header).
// Each chunk's metadata includes the column headers as keys.
func ParseCSV(content string, filename string) ([]Chunk, error) {
	reader := csv.NewReader(strings.NewReader(content))
	reader.FieldsPerRecord = -1 // allow variable fields
	reader.TrimLeadingSpace = true

	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("csv parse error: %w", err)
	}
	if len(records) < 2 {
		// Need at least header + one data row.
		if len(records) == 1 {
			// Single row = treat as one chunk.
			return []Chunk{{Text: strings.Join(records[0], ", "), Index: 0}}, nil
		}
		return nil, nil
	}

	headers := records[0]
	chunks := make([]Chunk, 0, len(records)-1)

	for i, row := range records[1:] {
		// Build text: "col1: val1, col2: val2, ..."
		var parts []string
		for j, val := range row {
			if j < len(headers) {
				parts = append(parts, headers[j]+": "+val)
			} else {
				parts = append(parts, val)
			}
		}
		text := strings.Join(parts, ", ")
		meta := map[string]string{"source": filename, "row": fmt.Sprintf("%d", i+1)}
		chunks = append(chunks, Chunk{Text: text, Index: i, Metadata: meta})
	}

	return chunks, nil
}

// StripHTML removes HTML tags and returns plain text.
// Handles common entities and collapses whitespace.
func StripHTML(html string) string {
	var result strings.Builder
	inTag := false

	for i := 0; i < len(html); i++ {
		ch := html[i]
		switch {
		case ch == '<':
			inTag = true
			// Insert space for block-level tags to avoid words merging.
			if i+1 < len(html) {
				next := strings.ToLower(string(html[i+1:min(i+5, len(html))]))
				if strings.HasPrefix(next, "/p") || strings.HasPrefix(next, "/div") ||
					strings.HasPrefix(next, "/h") || strings.HasPrefix(next, "/li") ||
					strings.HasPrefix(next, "br") || strings.HasPrefix(next, "p") ||
					strings.HasPrefix(next, "div") || strings.HasPrefix(next, "h") ||
					strings.HasPrefix(next, "li") {
					result.WriteByte(' ')
				}
			}
		case ch == '>':
			inTag = false
		case !inTag:
			result.WriteByte(ch)
		}
	}

	// Decode common HTML entities.
	out := result.String()
	out = strings.ReplaceAll(out, "&amp;", "&")
	out = strings.ReplaceAll(out, "&lt;", "<")
	out = strings.ReplaceAll(out, "&gt;", ">")
	out = strings.ReplaceAll(out, "&quot;", "\"")
	out = strings.ReplaceAll(out, "&#39;", "'")
	out = strings.ReplaceAll(out, "&nbsp;", " ")

	// Collapse whitespace.
	fields := strings.Fields(out)
	return strings.Join(fields, " ")
}

// ─── Ingestable model interface ──────────────────────────────────────────────

// IngestableModel is a model that supports inserting vectors into its index.
// Embedding models that also have an HNSW index implement this.
type IngestableModel interface {
	EmbeddingModel
	// InsertVector inserts a pre-computed embedding vector with an ID and metadata.
	InsertVector(id int64, vec []float32, metadata string) error
	// InsertBM25 inserts text into the BM25 index with a matching ID.
	InsertBM25(id int64, text string) error
	// NextID returns the next available ID for insertion.
	NextID() int64
}

// ─── Request / response types ────────────────────────────────────────────────

// IngestRequest is the body for POST /v1/ingest.
type IngestRequest struct {
	Model    string              `json:"model"`             // embedding model name
	Texts    []string            `json:"texts,omitempty"`   // raw texts to chunk, embed, index
	Metadata []map[string]string `json:"metadata,omitempty"` // optional per-text metadata

	// Legacy fields (backward compat).
	Documents []string `json:"documents,omitempty"` // alias for Texts
	IDs       []int64  `json:"ids,omitempty"`        // optional IDs

	// Chunker config (optional).
	ChunkSize    int    `json:"chunk_size,omitempty"`
	ChunkOverlap int    `json:"chunk_overlap,omitempty"`
	Separator    string `json:"separator,omitempty"`
}

// IngestResponse is the response for POST /v1/ingest.
type IngestResponse struct {
	Ingested int    `json:"ingested"` // number of input documents processed
	Chunks   int    `json:"chunks"`   // total chunks created
	Status   string `json:"status"`   // "ok" or error description
}

// IngestURLRequest is the body for POST /v1/ingest/url.
type IngestURLRequest struct {
	Model        string `json:"model"`
	URL          string `json:"url"`
	ChunkSize    int    `json:"chunk_size,omitempty"`
	ChunkOverlap int    `json:"chunk_overlap,omitempty"`
}

// ─── Handlers ────────────────────────────────────────────────────────────────

func (s *Server) handleIngest(w http.ResponseWriter, r *http.Request) {
	var req IngestRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	// Support legacy "documents" field as alias for "texts".
	if len(req.Texts) == 0 && len(req.Documents) > 0 {
		req.Texts = req.Documents
	}

	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "model field is required")
		return
	}
	if len(req.Texts) == 0 {
		writeError(w, http.StatusBadRequest, "texts (or documents) must not be empty")
		return
	}

	// Build chunker config.
	config := ChunkerConfig{
		ChunkSize:    req.ChunkSize,
		ChunkOverlap: req.ChunkOverlap,
		Separator:    req.Separator,
	}

	// Chunk all input texts.
	var allChunks []Chunk
	for i, text := range req.Texts {
		if text == "" {
			continue
		}
		chunks := ChunkText(text, config)
		// Attach metadata from request if provided.
		for j := range chunks {
			if chunks[j].Metadata == nil {
				chunks[j].Metadata = map[string]string{}
			}
			chunks[j].Metadata["doc_index"] = fmt.Sprintf("%d", i)
			if i < len(req.Metadata) {
				for k, v := range req.Metadata[i] {
					chunks[j].Metadata[k] = v
				}
			}
		}
		allChunks = append(allChunks, chunks...)
	}

	if len(allChunks) == 0 {
		writeJSON(w, http.StatusOK, IngestResponse{
			Ingested: len(req.Texts),
			Chunks:   0,
			Status:   "ok",
		})
		return
	}

	// Get the embedding model from registry.
	ref, err := s.registry.Get(req.Model)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	defer ref.Release()

	emb, ok := ref.Model.(EmbeddingModel)
	if !ok {
		writeError(w, http.StatusBadRequest,
			fmt.Sprintf("model %q does not support embeddings", req.Model))
		return
	}

	// Check if the model supports direct vector insertion.
	ingestable, hasIngest := ref.Model.(IngestableModel)

	// Embed all chunks.
	chunkTexts := make([]string, len(allChunks))
	for i, c := range allChunks {
		chunkTexts[i] = c.Text
	}

	var vecs [][]float32
	if batch, ok2 := emb.(BatchEmbeddingModel); ok2 {
		vecs, err = batch.EmbedBatch(r.Context(), chunkTexts)
	} else {
		vecs = make([][]float32, len(chunkTexts))
		for i, text := range chunkTexts {
			vecs[i], err = emb.Embed(r.Context(), text)
			if err != nil {
				break
			}
		}
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, "embedding failed: "+err.Error())
		return
	}

	// Insert into indexes if the model supports it.
	if hasIngest {
		for i, vec := range vecs {
			id := ingestable.NextID()
			metaJSON, _ := json.Marshal(allChunks[i].Metadata)
			if insertErr := ingestable.InsertVector(id, vec, string(metaJSON)); insertErr != nil {
				writeError(w, http.StatusInternalServerError,
					fmt.Sprintf("vector insert failed at chunk %d: %v", i, insertErr))
				return
			}
			if bm25Err := ingestable.InsertBM25(id, chunkTexts[i]); bm25Err != nil {
				// BM25 insert failure is non-fatal; log but continue.
				_ = bm25Err
			}
		}
	}

	writeJSON(w, http.StatusOK, IngestResponse{
		Ingested: len(req.Texts),
		Chunks:   len(allChunks),
		Status:   "ok",
	})
}

func (s *Server) handleIngestURL(w http.ResponseWriter, r *http.Request) {
	var req IngestURLRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "model field is required")
		return
	}
	if req.URL == "" {
		writeError(w, http.StatusBadRequest, "url field is required")
		return
	}

	// Fetch the URL.
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(ctx, "GET", req.URL, nil)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid url: "+err.Error())
		return
	}
	httpReq.Header.Set("User-Agent", "infergo/1.0")

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		writeError(w, http.StatusBadGateway, "failed to fetch url: "+err.Error())
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		writeError(w, http.StatusBadGateway,
			fmt.Sprintf("url returned status %d", resp.StatusCode))
		return
	}

	// Read body (limit to 10MB).
	body, err := io.ReadAll(io.LimitReader(resp.Body, 10*1024*1024))
	if err != nil {
		writeError(w, http.StatusBadGateway, "failed to read url body: "+err.Error())
		return
	}

	// Parse HTML content.
	content := StripHTML(string(body))
	if strings.TrimSpace(content) == "" {
		writeError(w, http.StatusBadRequest, "no text content extracted from url")
		return
	}

	// Forward to the main ingest handler logic via internal request.
	ingestReq := IngestRequest{
		Model:        req.Model,
		Texts:        []string{content},
		Metadata:     []map[string]string{{"source": req.URL}},
		ChunkSize:    req.ChunkSize,
		ChunkOverlap: req.ChunkOverlap,
	}

	ingestBody, _ := json.Marshal(ingestReq)
	internalReq, _ := http.NewRequestWithContext(r.Context(), "POST", "/v1/ingest",
		strings.NewReader(string(ingestBody)))
	internalReq.Header.Set("Content-Type", "application/json")

	s.handleIngest(w, internalReq)
}
