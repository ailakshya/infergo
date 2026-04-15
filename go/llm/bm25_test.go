package llm

import (
	"testing"
)

// ─── BM25 Tests ─────────────────────────────────────────────────────────────

func TestBM25Create(t *testing.T) {
	idx, err := NewBM25Index(1.2, 0.75)
	if err != nil {
		t.Fatalf("NewBM25Index: %v", err)
	}
	defer idx.Close()

	if idx.Size() != 0 {
		t.Errorf("new index size = %d, want 0", idx.Size())
	}
}

func TestBM25InsertAndSize(t *testing.T) {
	idx, err := NewBM25Index(1.2, 0.75)
	if err != nil {
		t.Fatalf("NewBM25Index: %v", err)
	}
	defer idx.Close()

	idx.Insert(1, "the quick brown fox jumps over the lazy dog")
	idx.Insert(2, "machine learning and deep learning for NLP")
	idx.Insert(3, "golang is a systems programming language")

	if idx.Size() != 3 {
		t.Errorf("size = %d, want 3", idx.Size())
	}
}

func TestBM25ExactKeywordMatch(t *testing.T) {
	idx, err := NewBM25Index(1.2, 0.75)
	if err != nil {
		t.Fatalf("NewBM25Index: %v", err)
	}
	defer idx.Close()

	idx.Insert(1, "golang programming language for backend development")
	idx.Insert(2, "python machine learning framework")
	idx.Insert(3, "javascript frontend web development")

	results, err := idx.Search("golang programming", 3)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected at least 1 result for 'golang programming'")
	}
	if results[0].ID != 1 {
		t.Errorf("top result ID = %d, want 1 (golang doc)", results[0].ID)
	}
}

func TestBM25Ranking(t *testing.T) {
	idx, err := NewBM25Index(1.2, 0.75)
	if err != nil {
		t.Fatalf("NewBM25Index: %v", err)
	}
	defer idx.Close()

	// Doc 1: very relevant to "machine learning"
	idx.Insert(1, "machine learning is a subset of artificial intelligence that uses statistical learning")
	// Doc 2: somewhat relevant
	idx.Insert(2, "deep learning neural networks for computer vision")
	// Doc 3: highly relevant (multiple mentions)
	idx.Insert(3, "machine learning machine learning algorithms for data science and machine learning applications")
	// Doc 4: irrelevant
	idx.Insert(4, "cooking recipes for italian pasta dishes")

	results, err := idx.Search("machine learning", 4)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}

	// Doc 3 should rank highest (most mentions of "machine learning")
	if len(results) < 2 {
		t.Fatalf("expected at least 2 results, got %d", len(results))
	}
	if results[0].ID != 3 {
		t.Errorf("top result ID = %d, want 3 (most mentions)", results[0].ID)
	}
	// Doc 1 should rank second (one mention)
	if results[1].ID != 1 {
		t.Errorf("second result ID = %d, want 1", results[1].ID)
	}
	// Scores should be descending
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("scores not descending: results[%d].Score=%f > results[%d].Score=%f",
				i, results[i].Score, i-1, results[i-1].Score)
		}
	}
}

func TestBM25Remove(t *testing.T) {
	idx, err := NewBM25Index(1.2, 0.75)
	if err != nil {
		t.Fatalf("NewBM25Index: %v", err)
	}
	defer idx.Close()

	idx.Insert(1, "alpha beta gamma")
	idx.Insert(2, "delta epsilon zeta")
	if idx.Size() != 2 {
		t.Errorf("size = %d, want 2", idx.Size())
	}

	idx.Remove(1)
	if idx.Size() != 1 {
		t.Errorf("size after remove = %d, want 1", idx.Size())
	}

	results, err := idx.Search("alpha", 5)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results after removing doc 1, got %d", len(results))
	}
}

func TestBM25EmptyQuery(t *testing.T) {
	idx, err := NewBM25Index(1.2, 0.75)
	if err != nil {
		t.Fatalf("NewBM25Index: %v", err)
	}
	defer idx.Close()

	idx.Insert(1, "some document text")
	results, err := idx.Search("", 5)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results for empty query, got %d", len(results))
	}
}

func TestBM25Stemming(t *testing.T) {
	idx, err := NewBM25Index(1.2, 0.75)
	if err != nil {
		t.Fatalf("NewBM25Index: %v", err)
	}
	defer idx.Close()

	idx.Insert(1, "running quickly through the searching algorithm")
	idx.Insert(2, "the runner searched for optimal solutions")

	// "running" stems to "runn", "run" stems to "run" — these are different stems.
	// But "searching" -> "search" and "searched" -> "search" should match.
	results, err := idx.Search("searched", 5)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) < 2 {
		t.Fatalf("expected 2 results (stemmed match), got %d", len(results))
	}
}

func TestBM25ClosedIndex(t *testing.T) {
	idx, err := NewBM25Index(1.2, 0.75)
	if err != nil {
		t.Fatalf("NewBM25Index: %v", err)
	}
	idx.Close()

	err = idx.Insert(1, "test")
	if err == nil {
		t.Error("expected error on Insert after Close")
	}

	_, err = idx.Search("test", 5)
	if err == nil {
		t.Error("expected error on Search after Close")
	}
}

// ─── Hybrid Search Tests ────────────────────────────────────────────────────

func TestHybridSearchCombinesBoth(t *testing.T) {
	// Simulate: doc 1 is good in vector, doc 2 is good in BM25, doc 3 is good in both
	vecResults := []SearchResult{
		{ID: 1, Distance: 0.1},  // cosine dist 0.1 -> sim 0.9 (very good)
		{ID: 3, Distance: 0.2},  // cosine dist 0.2 -> sim 0.8 (good)
		{ID: 4, Distance: 0.5},  // cosine dist 0.5 -> sim 0.5 (moderate)
	}
	bm25Results := []BM25Result{
		{ID: 2, Score: 5.0},  // high BM25 score
		{ID: 3, Score: 4.0},  // good BM25 score
		{ID: 5, Score: 1.0},  // low BM25 score
	}

	results, err := HybridSearch(vecResults, bm25Results, 0.5, 5)
	if err != nil {
		t.Fatalf("HybridSearch: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected results from hybrid search")
	}

	// Doc 3 appears in BOTH lists, so it should rank highest with alpha=0.5
	if results[0].ID != 3 {
		t.Errorf("top result ID = %d, want 3 (appears in both vector and BM25)", results[0].ID)
	}
}

func TestHybridAlphaPureVector(t *testing.T) {
	vecResults := []SearchResult{
		{ID: 1, Distance: 0.1},  // best vector match
		{ID: 2, Distance: 0.5},
	}
	bm25Results := []BM25Result{
		{ID: 3, Score: 10.0},  // best BM25 match
		{ID: 2, Score: 5.0},
	}

	// alpha=1.0 means pure vector — BM25 results get 0 weight
	results, err := HybridSearch(vecResults, bm25Results, 1.0, 5)
	if err != nil {
		t.Fatalf("HybridSearch: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected results")
	}
	// Doc 1 should be top (best vector match) since alpha=1.0 ignores BM25
	if results[0].ID != 1 {
		t.Errorf("alpha=1.0: top result ID = %d, want 1 (best vector)", results[0].ID)
	}
}

func TestHybridAlphaPureBM25(t *testing.T) {
	vecResults := []SearchResult{
		{ID: 1, Distance: 0.1},  // best vector match
		{ID: 2, Distance: 0.5},
	}
	bm25Results := []BM25Result{
		{ID: 3, Score: 10.0},  // best BM25 match
		{ID: 2, Score: 5.0},
	}

	// alpha=0.0 means pure BM25 — vector results get 0 weight
	results, err := HybridSearch(vecResults, bm25Results, 0.0, 5)
	if err != nil {
		t.Fatalf("HybridSearch: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected results")
	}
	// Doc 3 should be top (best BM25 match) since alpha=0.0 ignores vector
	if results[0].ID != 3 {
		t.Errorf("alpha=0.0: top result ID = %d, want 3 (best BM25)", results[0].ID)
	}
}

func TestHybridEmptyInputs(t *testing.T) {
	// Both empty
	results, err := HybridSearch(nil, nil, 0.5, 5)
	if err != nil {
		t.Fatalf("HybridSearch with empty inputs: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results for empty inputs, got %d", len(results))
	}

	// Only vector results
	vecResults := []SearchResult{{ID: 1, Distance: 0.1}}
	results, err = HybridSearch(vecResults, nil, 0.5, 5)
	if err != nil {
		t.Fatalf("HybridSearch with only vector: %v", err)
	}
	if len(results) != 1 {
		t.Errorf("expected 1 result for vector-only, got %d", len(results))
	}

	// Only BM25 results
	bm25Results := []BM25Result{{ID: 2, Score: 5.0}}
	results, err = HybridSearch(nil, bm25Results, 0.5, 5)
	if err != nil {
		t.Fatalf("HybridSearch with only BM25: %v", err)
	}
	if len(results) != 1 {
		t.Errorf("expected 1 result for BM25-only, got %d", len(results))
	}
}

func TestHybridScoresInRange(t *testing.T) {
	vecResults := []SearchResult{
		{ID: 1, Distance: 0.1},
		{ID: 2, Distance: 0.3},
		{ID: 3, Distance: 0.7},
	}
	bm25Results := []BM25Result{
		{ID: 1, Score: 8.0},
		{ID: 4, Score: 3.0},
		{ID: 2, Score: 1.0},
	}

	results, err := HybridSearch(vecResults, bm25Results, 0.5, 10)
	if err != nil {
		t.Fatalf("HybridSearch: %v", err)
	}

	for _, r := range results {
		if r.Score < 0 || r.Score > 1.0 {
			t.Errorf("score %f out of [0,1] range for doc %d", r.Score, r.ID)
		}
	}

	// Check descending order
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("scores not descending: [%d]=%f > [%d]=%f",
				i, results[i].Score, i-1, results[i-1].Score)
		}
	}
}

// ─── BM25 Benchmark ─────────────────────────────────────────────────────────

func BenchmarkBM25Search10K(b *testing.B) {
	idx, err := NewBM25Index(1.2, 0.75)
	if err != nil {
		b.Fatalf("NewBM25Index: %v", err)
	}
	defer idx.Close()

	// Insert 10K documents with varying content
	docs := []string{
		"machine learning algorithms for data science applications",
		"deep neural networks for computer vision tasks",
		"natural language processing with transformer models",
		"reinforcement learning for robotics control",
		"graph neural networks for social network analysis",
		"generative adversarial networks for image synthesis",
		"federated learning for privacy preserving machine learning",
		"transfer learning techniques for domain adaptation",
		"attention mechanisms in sequence to sequence models",
		"bayesian optimization for hyperparameter tuning",
	}

	for i := 0; i < 10000; i++ {
		idx.Insert(int64(i), docs[i%len(docs)])
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search("machine learning algorithms", 10)
	}
}
