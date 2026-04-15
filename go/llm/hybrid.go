package llm

/*
#include "infer_api.h"
*/
import "C"

import (
	"errors"
)

// HybridResult is one result from hybrid (BM25 + vector) search.
type HybridResult struct {
	ID    int64
	Score float32 // combined score in [0, 1]
}

// HybridSearch combines vector search results and BM25 results using
// weighted score fusion.
//
// alpha controls the balance:
//   - alpha=1.0: pure vector search
//   - alpha=0.0: pure BM25 (keyword) search
//   - alpha=0.5: equal weight to both
//
// vecResults are from VectorIndex.Search (cosine distance, lower = better).
// bm25Results are from BM25Index.Search (BM25 score, higher = better).
func HybridSearch(vecResults []SearchResult, bm25Results []BM25Result, alpha float32, k int) ([]HybridResult, error) {
	if k <= 0 {
		k = 10
	}

	// Prepare vector arrays
	nVec := len(vecResults)
	nBM25 := len(bm25Results)

	if nVec == 0 && nBM25 == 0 {
		return nil, nil
	}

	// Build C arrays for vector results
	vecIDs := make([]C.int64_t, max(nVec, 1))
	vecDists := make([]C.float, max(nVec, 1))
	for i, r := range vecResults {
		vecIDs[i] = C.int64_t(r.ID)
		vecDists[i] = C.float(r.Distance)
	}

	// Build C arrays for BM25 results
	bm25IDs := make([]C.int64_t, max(nBM25, 1))
	bm25Scores := make([]C.float, max(nBM25, 1))
	for i, r := range bm25Results {
		bm25IDs[i] = C.int64_t(r.ID)
		bm25Scores[i] = C.float(r.Score)
	}

	// Output arrays
	outIDs := make([]C.int64_t, k)
	outScores := make([]C.float, k)

	n := C.infer_hybrid_search(
		&vecIDs[0], &vecDists[0], C.int(nVec),
		&bm25IDs[0], &bm25Scores[0], C.int(nBM25),
		C.float(alpha), C.int(k),
		&outIDs[0], &outScores[0], C.int(k))
	if n < 0 {
		return nil, errors.New("llm: hybrid search failed")
	}

	results := make([]HybridResult, int(n))
	for i := 0; i < int(n); i++ {
		results[i] = HybridResult{
			ID:    int64(outIDs[i]),
			Score: float32(outScores[i]),
		}
	}
	return results, nil
}

