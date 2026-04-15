package server

import (
	"encoding/json"
	"net/http"
	"os"
	"regexp"
	"strings"
	"sync"
)

// Triple is a (subject, predicate, object) fact in the knowledge graph.
type Triple struct {
	Subject   string `json:"subject"`
	Predicate string `json:"predicate"`
	Object    string `json:"object"`
}

// KnowledgeGraph stores triples in memory with thread-safe access.
type KnowledgeGraph struct {
	mu       sync.RWMutex
	triples  []Triple
	bySubj   map[string][]int // subject -> indices into triples
	filePath string           // optional path for persistence
}

// NewKnowledgeGraph creates an empty knowledge graph.
// If filePath is non-empty, Save/Load will use that file.
func NewKnowledgeGraph(filePath string) *KnowledgeGraph {
	return &KnowledgeGraph{
		triples:  []Triple{},
		bySubj:   make(map[string][]int),
		filePath: filePath,
	}
}

// AddTriple inserts a (subject, predicate, object) triple.
func (kg *KnowledgeGraph) AddTriple(s, p, o string) {
	kg.mu.Lock()
	defer kg.mu.Unlock()
	idx := len(kg.triples)
	kg.triples = append(kg.triples, Triple{Subject: s, Predicate: p, Object: o})
	key := strings.ToLower(s)
	kg.bySubj[key] = append(kg.bySubj[key], idx)
}

// QueryResult is a (predicate, object) pair returned by Query.
type QueryResult struct {
	Predicate string `json:"predicate"`
	Object    string `json:"object"`
}

// Query returns all (predicate, object) pairs for the given subject.
func (kg *KnowledgeGraph) Query(subject string) []QueryResult {
	kg.mu.RLock()
	defer kg.mu.RUnlock()
	key := strings.ToLower(subject)
	indices := kg.bySubj[key]
	results := make([]QueryResult, 0, len(indices))
	for _, i := range indices {
		t := kg.triples[i]
		results = append(results, QueryResult{Predicate: t.Predicate, Object: t.Object})
	}
	return results
}

// ─── Entity extraction patterns ──────────────────────────────────────────────

var entityPatterns = []*regexp.Regexp{
	// "X works at Y" / "X worked at Y"
	regexp.MustCompile(`(?i)(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+works?\s+at\s+([A-Z][a-zA-Z0-9\s&]+)`),
	// "X lives in Y" / "X lived in Y"
	regexp.MustCompile(`(?i)(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+lives?\s+in\s+([A-Z][a-zA-Z\s]+)`),
	// "X is a Y" / "X is an Y"
	regexp.MustCompile(`(?i)(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+an?\s+([a-zA-Z\s]+)`),
}

var predicateMap = map[int]string{
	0: "works_at",
	1: "lives_in",
	2: "is_a",
}

// ExtractEntities extracts triples from natural language text using regex.
func ExtractEntities(text string) []Triple {
	var triples []Triple
	for i, pat := range entityPatterns {
		matches := pat.FindAllStringSubmatch(text, -1)
		for _, m := range matches {
			if len(m) >= 3 {
				subj := strings.TrimSpace(m[1])
				obj := strings.TrimSpace(m[2])
				// Trim trailing punctuation from object
				obj = strings.TrimRight(obj, ".,;:!?")
				if subj != "" && obj != "" {
					triples = append(triples, Triple{
						Subject:   subj,
						Predicate: predicateMap[i],
						Object:    obj,
					})
				}
			}
		}
	}
	return triples
}

// ─── Persistence ─────────────────────────────────────────────────────────────

// Save writes all triples to the configured JSON file.
func (kg *KnowledgeGraph) Save() error {
	if kg.filePath == "" {
		return nil
	}
	kg.mu.RLock()
	data, err := json.MarshalIndent(kg.triples, "", "  ")
	kg.mu.RUnlock()
	if err != nil {
		return err
	}
	return os.WriteFile(kg.filePath, data, 0644)
}

// Load reads triples from the configured JSON file, replacing current data.
func (kg *KnowledgeGraph) Load() error {
	if kg.filePath == "" {
		return nil
	}
	data, err := os.ReadFile(kg.filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // no file yet — start empty
		}
		return err
	}
	var triples []Triple
	if err := json.Unmarshal(data, &triples); err != nil {
		return err
	}
	kg.mu.Lock()
	defer kg.mu.Unlock()
	kg.triples = triples
	kg.bySubj = make(map[string][]int)
	for i, t := range triples {
		key := strings.ToLower(t.Subject)
		kg.bySubj[key] = append(kg.bySubj[key], i)
	}
	return nil
}

// ─── HTTP Handlers ───────────────────────────────────────────────────────────

// handleKnowledgeExtract handles POST /v1/knowledge/extract.
// Accepts {"text": "..."}, extracts entities, adds triples, returns them.
func (s *Server) handleKnowledgeExtract(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Text string `json:"text"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	if req.Text == "" {
		writeError(w, http.StatusBadRequest, "text field is required")
		return
	}

	triples := ExtractEntities(req.Text)

	if s.knowledgeGraph != nil {
		for _, t := range triples {
			s.knowledgeGraph.AddTriple(t.Subject, t.Predicate, t.Object)
		}
		_ = s.knowledgeGraph.Save()
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"triples": triples,
		"count":   len(triples),
	})
}

// handleKnowledgeQuery handles GET /v1/knowledge/query?subject=X.
// Returns all (predicate, object) pairs for the given subject.
func (s *Server) handleKnowledgeQuery(w http.ResponseWriter, r *http.Request) {
	subject := r.URL.Query().Get("subject")
	if subject == "" {
		writeError(w, http.StatusBadRequest, "subject query parameter is required")
		return
	}

	if s.knowledgeGraph == nil {
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"subject": subject,
			"results": []QueryResult{},
		})
		return
	}

	results := s.knowledgeGraph.Query(subject)
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"subject": subject,
		"results": results,
	})
}
