package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestKnowledgeGraphAddQuery(t *testing.T) {
	kg := NewKnowledgeGraph("")
	kg.AddTriple("Alice", "works_at", "Google")
	kg.AddTriple("Alice", "lives_in", "San Francisco")
	kg.AddTriple("Bob", "works_at", "Meta")

	results := kg.Query("Alice")
	if len(results) != 2 {
		t.Fatalf("expected 2 results for Alice, got %d", len(results))
	}
	if results[0].Predicate != "works_at" || results[0].Object != "Google" {
		t.Errorf("unexpected first result: %+v", results[0])
	}

	results = kg.Query("Bob")
	if len(results) != 1 {
		t.Fatalf("expected 1 result for Bob, got %d", len(results))
	}

	results = kg.Query("Charlie")
	if len(results) != 0 {
		t.Errorf("expected 0 results for Charlie, got %d", len(results))
	}
}

func TestKnowledgeGraphCaseInsensitiveQuery(t *testing.T) {
	kg := NewKnowledgeGraph("")
	kg.AddTriple("Alice", "works_at", "Google")

	results := kg.Query("alice")
	if len(results) != 1 {
		t.Fatalf("expected case-insensitive match, got %d results", len(results))
	}
}

func TestExtractEntities(t *testing.T) {
	tests := []struct {
		text     string
		expected []Triple
	}{
		{
			text: "Alice works at Google.",
			expected: []Triple{
				{Subject: "Alice", Predicate: "works_at", Object: "Google"},
			},
		},
		{
			text: "Bob lives in New York.",
			expected: []Triple{
				{Subject: "Bob", Predicate: "lives_in", Object: "New York"},
			},
		},
		{
			text: "Charlie is a software engineer.",
			expected: []Triple{
				{Subject: "Charlie", Predicate: "is_a", Object: "software engineer"},
			},
		},
		{
			text: "Alice works at Google. Bob lives in London. Carol is a doctor.",
			expected: []Triple{
				{Subject: "Alice", Predicate: "works_at", Object: "Google"},
				{Subject: "Bob", Predicate: "lives_in", Object: "London"},
				{Subject: "Carol", Predicate: "is_a", Object: "doctor"},
			},
		},
		{
			text: "nothing interesting here",
			expected: nil,
		},
	}

	for _, tc := range tests {
		triples := ExtractEntities(tc.text)
		if len(triples) != len(tc.expected) {
			t.Errorf("text=%q: expected %d triples, got %d: %+v", tc.text, len(tc.expected), len(triples), triples)
			continue
		}
		for i, tri := range triples {
			exp := tc.expected[i]
			if tri.Subject != exp.Subject || tri.Predicate != exp.Predicate || tri.Object != exp.Object {
				t.Errorf("text=%q triple[%d]: got %+v, want %+v", tc.text, i, tri, exp)
			}
		}
	}
}

func TestKnowledgeGraphSaveLoad(t *testing.T) {
	dir := t.TempDir()
	fp := filepath.Join(dir, "kg.json")

	// Create and populate
	kg := NewKnowledgeGraph(fp)
	kg.AddTriple("Alice", "works_at", "Google")
	kg.AddTriple("Bob", "lives_in", "London")
	if err := kg.Save(); err != nil {
		t.Fatal(err)
	}

	// Verify file exists
	data, err := os.ReadFile(fp)
	if err != nil {
		t.Fatal(err)
	}
	var triples []Triple
	if err := json.Unmarshal(data, &triples); err != nil {
		t.Fatal(err)
	}
	if len(triples) != 2 {
		t.Fatalf("expected 2 triples in JSON, got %d", len(triples))
	}

	// Load into a new graph
	kg2 := NewKnowledgeGraph(fp)
	if err := kg2.Load(); err != nil {
		t.Fatal(err)
	}
	results := kg2.Query("Alice")
	if len(results) != 1 || results[0].Object != "Google" {
		t.Errorf("loaded graph query failed: %+v", results)
	}
}

func TestKnowledgeGraphLoadMissing(t *testing.T) {
	kg := NewKnowledgeGraph("/nonexistent/path/kg.json")
	if err := kg.Load(); err != nil {
		t.Errorf("Load on missing file should return nil, got: %v", err)
	}
}

func TestHandleKnowledgeExtract(t *testing.T) {
	reg := NewRegistry()
	srv := NewServer(reg)
	kg := NewKnowledgeGraph("")
	srv.SetKnowledgeGraph(kg)

	body := `{"text":"Alice works at Google. Bob lives in London."}`
	req := httptest.NewRequest(http.MethodPost, "/v1/knowledge/extract", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var resp struct {
		Triples []Triple `json:"triples"`
		Count   int      `json:"count"`
	}
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp.Count != 2 {
		t.Errorf("expected 2 triples, got %d", resp.Count)
	}

	// Verify they were added to the graph
	results := kg.Query("Alice")
	if len(results) != 1 {
		t.Errorf("expected Alice in graph, got %d results", len(results))
	}
}

func TestHandleKnowledgeQuery(t *testing.T) {
	reg := NewRegistry()
	srv := NewServer(reg)
	kg := NewKnowledgeGraph("")
	kg.AddTriple("Alice", "works_at", "Google")
	srv.SetKnowledgeGraph(kg)

	req := httptest.NewRequest(http.MethodGet, "/v1/knowledge/query?subject=Alice", nil)
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	var resp struct {
		Subject string        `json:"subject"`
		Results []QueryResult `json:"results"`
	}
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Results) != 1 || resp.Results[0].Object != "Google" {
		t.Errorf("unexpected query results: %+v", resp.Results)
	}
}

func TestHandleKnowledgeQueryMissingSubject(t *testing.T) {
	reg := NewRegistry()
	srv := NewServer(reg)

	req := httptest.NewRequest(http.MethodGet, "/v1/knowledge/query", nil)
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rec.Code)
	}
}
