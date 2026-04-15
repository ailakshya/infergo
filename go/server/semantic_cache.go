package server

import (
	"context"
	"math"
	"sync"
	"time"
)

// SemanticCache caches chat completion responses keyed by embedding similarity.
// When a new query arrives, it is embedded and compared against cached embeddings
// using cosine similarity. If any cached entry exceeds the threshold, the cached
// response is returned instead of running inference.
type SemanticCache struct {
	mu        sync.RWMutex
	entries   []semanticCacheEntry
	threshold float64 // cosine similarity threshold (default 0.95)
	maxSize   int     // maximum number of cached entries
}

type semanticCacheEntry struct {
	embedding []float32 // normalized embedding vector
	response  []byte    // pre-serialized JSON response
	created   time.Time
}

// NewSemanticCache creates a semantic cache with the given max entry count
// and similarity threshold. A threshold of 0.95 means queries must be 95%
// similar to return a cached result.
func NewSemanticCache(maxSize int, threshold float64) *SemanticCache {
	if threshold <= 0 || threshold > 1 {
		threshold = 0.95
	}
	if maxSize <= 0 {
		maxSize = 1000
	}
	return &SemanticCache{
		entries:   make([]semanticCacheEntry, 0, maxSize),
		threshold: threshold,
		maxSize:   maxSize,
	}
}

// Lookup searches for a cached response whose embedding has cosine similarity
// >= threshold with the query embedding. Returns the cached response bytes and
// true on hit, or nil and false on miss.
func (sc *SemanticCache) Lookup(queryEmb []float32) ([]byte, bool) {
	if sc == nil || len(queryEmb) == 0 {
		return nil, false
	}

	// Normalize query embedding.
	normQuery := normalizeVec(queryEmb)

	sc.mu.RLock()
	defer sc.mu.RUnlock()

	bestSim := -1.0
	bestIdx := -1
	for i, entry := range sc.entries {
		sim := cosineSimilarityF64(normQuery, entry.embedding)
		if sim > bestSim {
			bestSim = sim
			bestIdx = i
		}
	}

	if bestIdx >= 0 && bestSim >= sc.threshold {
		return sc.entries[bestIdx].response, true
	}
	return nil, false
}

// Store adds a query embedding and its corresponding response to the cache.
// If the cache is full, the oldest entry is evicted.
func (sc *SemanticCache) Store(queryEmb []float32, response []byte) {
	if sc == nil || len(queryEmb) == 0 {
		return
	}

	normEmb := normalizeVec(queryEmb)

	sc.mu.Lock()
	defer sc.mu.Unlock()

	// Evict oldest if at capacity.
	if len(sc.entries) >= sc.maxSize {
		sc.entries = sc.entries[1:]
	}

	sc.entries = append(sc.entries, semanticCacheEntry{
		embedding: normEmb,
		response:  response,
		created:   time.Now(),
	})
}

// Len returns the number of entries in the cache.
func (sc *SemanticCache) Len() int {
	if sc == nil {
		return 0
	}
	sc.mu.RLock()
	defer sc.mu.RUnlock()
	return len(sc.entries)
}

// EmbedAndLookup is a convenience method that embeds the query text using the
// provided embedding model, then does a cache lookup. Returns (response, hit, error).
func (sc *SemanticCache) EmbedAndLookup(ctx context.Context, embedder EmbeddingModel, text string) ([]byte, bool, error) {
	if sc == nil {
		return nil, false, nil
	}
	emb, err := embedder.Embed(ctx, text)
	if err != nil {
		return nil, false, err
	}
	resp, hit := sc.Lookup(emb)
	return resp, hit, nil
}

// EmbedAndStore embeds the query text and stores the embedding with the response.
func (sc *SemanticCache) EmbedAndStore(ctx context.Context, embedder EmbeddingModel, text string, response []byte) error {
	if sc == nil {
		return nil
	}
	emb, err := embedder.Embed(ctx, text)
	if err != nil {
		return err
	}
	sc.Store(emb, response)
	return nil
}

// ─── vector math ─────────────────────────────────────────────────────────────

// cosineSimilarityF64 computes the cosine similarity between two vectors,
// returning float64. Both vectors should be pre-normalized for best
// performance, but this function normalizes on the fly as a safety measure.
func cosineSimilarityF64(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

// normalizeVec returns a unit-length copy of v.
func normalizeVec(v []float32) []float32 {
	var norm float64
	for _, val := range v {
		norm += float64(val) * float64(val)
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		out := make([]float32, len(v))
		copy(out, v)
		return out
	}
	out := make([]float32, len(v))
	for i, val := range v {
		out[i] = float32(float64(val) / norm)
	}
	return out
}
