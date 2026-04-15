#include "hybrid.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace infergo {

std::vector<HybridResult> HybridSearch(
    const int64_t* vec_ids,   const float* vec_distances,  int n_vec,
    const int64_t* bm25_ids,  const float* bm25_scores,    int n_bm25,
    float alpha,
    int k)
{
    // Clamp alpha to [0, 1]
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;

    // --- Normalize vector scores to [0, 1] ---
    // HNSW returns cosine distance (0 = identical, 1 = orthogonal).
    // Convert to similarity: sim = 1 - distance, then normalize to [0, 1].
    float vec_min_sim = 1.0f, vec_max_sim = 0.0f;
    std::vector<float> vec_sims(n_vec);
    for (int i = 0; i < n_vec; ++i) {
        vec_sims[i] = 1.0f - vec_distances[i];  // cosine similarity
        if (vec_sims[i] < vec_min_sim) vec_min_sim = vec_sims[i];
        if (vec_sims[i] > vec_max_sim) vec_max_sim = vec_sims[i];
    }
    float vec_range = vec_max_sim - vec_min_sim;
    if (vec_range < 1e-7f) vec_range = 1.0f;  // avoid div-by-zero

    // --- Normalize BM25 scores to [0, 1] ---
    float bm25_min = 0.0f, bm25_max = 0.0f;
    if (n_bm25 > 0) {
        bm25_min = bm25_scores[0];
        bm25_max = bm25_scores[0];
        for (int i = 1; i < n_bm25; ++i) {
            if (bm25_scores[i] < bm25_min) bm25_min = bm25_scores[i];
            if (bm25_scores[i] > bm25_max) bm25_max = bm25_scores[i];
        }
    }
    float bm25_range = bm25_max - bm25_min;
    if (bm25_range < 1e-7f) bm25_range = 1.0f;

    // --- Build combined score map ---
    struct DocScores {
        float vec_norm  = 0.0f;
        float bm25_norm = 0.0f;
        bool  has_vec   = false;
        bool  has_bm25  = false;
    };
    std::unordered_map<int64_t, DocScores> combined;

    for (int i = 0; i < n_vec; ++i) {
        float norm = (vec_sims[i] - vec_min_sim) / vec_range;
        auto& ds = combined[vec_ids[i]];
        ds.vec_norm = norm;
        ds.has_vec = true;
    }

    for (int i = 0; i < n_bm25; ++i) {
        float norm = (bm25_scores[i] - bm25_min) / bm25_range;
        auto& ds = combined[bm25_ids[i]];
        ds.bm25_norm = norm;
        ds.has_bm25 = true;
    }

    // --- Compute weighted combined score ---
    std::vector<HybridResult> results;
    results.reserve(combined.size());

    for (const auto& [id, ds] : combined) {
        HybridResult hr;
        hr.id = id;
        hr.vec_score  = ds.has_vec  ? ds.vec_norm  : 0.0f;
        hr.bm25_score = ds.has_bm25 ? ds.bm25_norm : 0.0f;
        hr.score = alpha * hr.vec_score + (1.0f - alpha) * hr.bm25_score;
        results.push_back(hr);
    }

    // Sort by combined score descending
    if (static_cast<int>(results.size()) > k) {
        std::partial_sort(results.begin(), results.begin() + k, results.end(),
            [](const HybridResult& a, const HybridResult& b) {
                return a.score > b.score;
            });
        results.resize(k);
    } else {
        std::sort(results.begin(), results.end(),
            [](const HybridResult& a, const HybridResult& b) {
                return a.score > b.score;
            });
    }

    return results;
}

} // namespace infergo
