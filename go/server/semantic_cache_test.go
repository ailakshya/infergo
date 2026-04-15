package server

import (
	"context"
	"testing"
)

func TestSemanticCacheExactHit(t *testing.T) {
	sc := NewSemanticCache(100, 0.95)

	// Store a vector and its response.
	emb := []float32{1.0, 0.0, 0.0}
	sc.Store(emb, []byte(`{"result":"cached"}`))

	// Lookup with the exact same vector should hit.
	resp, hit := sc.Lookup(emb)
	if !hit {
		t.Fatal("expected cache hit for exact same embedding")
	}
	if string(resp) != `{"result":"cached"}` {
		t.Errorf("unexpected response: %s", string(resp))
	}
}

func TestSemanticCacheSimilarHit(t *testing.T) {
	sc := NewSemanticCache(100, 0.95)

	// Store a vector.
	emb := []float32{1.0, 0.0, 0.0}
	sc.Store(emb, []byte(`{"result":"similar"}`))

	// Query with a very similar vector (cosine > 0.95).
	// [0.99, 0.1, 0.0] normalized: ~[0.995, 0.1, 0]
	// cosine with [1,0,0] = 0.995 / (1 * sqrt(0.99^2+0.1^2)) = 0.99/0.995 ≈ 0.995
	similar := []float32{0.99, 0.1, 0.0}
	resp, hit := sc.Lookup(similar)
	if !hit {
		t.Fatal("expected cache hit for similar embedding (cosine > 0.95)")
	}
	if string(resp) != `{"result":"similar"}` {
		t.Errorf("unexpected response: %s", string(resp))
	}
}

func TestSemanticCacheMiss(t *testing.T) {
	sc := NewSemanticCache(100, 0.95)

	// Store a vector.
	emb := []float32{1.0, 0.0, 0.0}
	sc.Store(emb, []byte(`{"result":"cached"}`))

	// Query with a very different vector — should miss.
	different := []float32{0.0, 1.0, 0.0}
	_, hit := sc.Lookup(different)
	if hit {
		t.Fatal("expected cache miss for orthogonal embedding")
	}
}

func TestSemanticCacheEviction(t *testing.T) {
	sc := NewSemanticCache(2, 0.95)

	sc.Store([]float32{1, 0, 0}, []byte("first"))
	sc.Store([]float32{0, 1, 0}, []byte("second"))
	sc.Store([]float32{0, 0, 1}, []byte("third"))

	if sc.Len() != 2 {
		t.Fatalf("expected 2 entries after eviction, got %d", sc.Len())
	}

	// First entry should have been evicted — lookup with [1,0,0] should miss.
	_, hit := sc.Lookup([]float32{1, 0, 0})
	if hit {
		t.Error("expected evicted entry to miss")
	}

	// Third entry should still be present.
	resp, hit := sc.Lookup([]float32{0, 0, 1})
	if !hit {
		t.Fatal("expected hit for third entry")
	}
	if string(resp) != "third" {
		t.Errorf("unexpected response: %s", string(resp))
	}
}

func TestSemanticCacheNilSafe(t *testing.T) {
	var sc *SemanticCache

	// All operations on nil cache should be safe.
	_, hit := sc.Lookup([]float32{1, 0, 0})
	if hit {
		t.Error("nil cache should not hit")
	}

	sc.Store([]float32{1, 0, 0}, []byte("data"))
	if sc.Len() != 0 {
		t.Error("nil cache len should be 0")
	}
}

// mockEmbedderInternal implements EmbeddingModel for tests within this package.
type mockEmbedderInternal struct {
	vec []float32
}

func (m *mockEmbedderInternal) Close() {}
func (m *mockEmbedderInternal) Embed(_ context.Context, _ string) ([]float32, error) {
	return m.vec, nil
}

func TestSemanticCacheEmbedAndLookup(t *testing.T) {
	sc := NewSemanticCache(100, 0.95)
	embedder := &mockEmbedderInternal{vec: []float32{1.0, 0.0, 0.0}}

	// Store via EmbedAndStore.
	err := sc.EmbedAndStore(context.Background(), embedder, "hello world", []byte(`{"answer":"42"}`))
	if err != nil {
		t.Fatalf("EmbedAndStore error: %v", err)
	}

	// Lookup via EmbedAndLookup — same embedder returns same vector, so it should hit.
	resp, hit, err := sc.EmbedAndLookup(context.Background(), embedder, "hello world")
	if err != nil {
		t.Fatalf("EmbedAndLookup error: %v", err)
	}
	if !hit {
		t.Fatal("expected hit")
	}
	if string(resp) != `{"answer":"42"}` {
		t.Errorf("unexpected response: %s", string(resp))
	}
}
