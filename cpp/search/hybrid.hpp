#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace infergo {

/// Hybrid search result combining vector and BM25 scores.
struct HybridResult {
    int64_t id;
    float   score;      // combined score in [0, 1]
    float   vec_score;  // normalized vector score (1 = best match)
    float   bm25_score; // normalized BM25 score (1 = best match)
};

/// Combine vector search results and BM25 results using score fusion.
/// alpha: weight for vector scores (1.0 = pure vector, 0.0 = pure BM25).
/// Both input score arrays are normalized to [0,1] before combining.
///
/// vec_ids/vec_distances: HNSW results (cosine distance, lower = better)
/// bm25_ids/bm25_scores:  BM25 results (BM25 score, higher = better)
///
/// Returns fused results sorted by combined score descending.
std::vector<HybridResult> HybridSearch(
    const int64_t* vec_ids,   const float* vec_distances,  int n_vec,
    const int64_t* bm25_ids,  const float* bm25_scores,    int n_bm25,
    float alpha,  // vector weight
    int k);       // max results to return

} // namespace infergo
