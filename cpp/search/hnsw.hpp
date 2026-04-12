#pragma once

#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <vector>
#include <unordered_map>

namespace infergo {

/// Simple HNSW (Hierarchical Navigable Small Worlds) index for vector similarity search.
/// Thread-safe for concurrent search; insert is serialized.
class HNSWIndex {
public:
    /// dim: vector dimension, M: max connections per node, ef_construction: search width during build
    HNSWIndex(int dim, int M = 16, int ef_construction = 200);
    ~HNSWIndex() = default;

    /// Add a vector with an ID and optional metadata string.
    /// Returns true on success.
    bool Insert(int64_t id, const float* vec, const std::string& metadata = "");

    /// Search for the k nearest neighbors to the query vector.
    /// Returns IDs and distances (cosine distance = 1 - similarity).
    void Search(const float* query, int k, int ef_search,
                std::vector<int64_t>& out_ids,
                std::vector<float>& out_distances,
                std::vector<std::string>& out_metadata) const;

    /// Number of vectors in the index.
    int Size() const;

    /// Vector dimension.
    int Dim() const { return dim_; }

private:
    struct Node {
        int64_t     id;
        std::string metadata;
        std::vector<float> vec;
        std::vector<std::vector<int>> neighbors;  // per-layer neighbor lists
        int max_layer;
    };

    float CosineDistance(const float* a, const float* b) const;
    int   RandomLevel();
    void  SearchLayer(const float* query, int entry, int ef, int layer,
                      std::vector<std::pair<float, int>>& result) const;

    int dim_;
    int M_;
    int ef_construction_;
    int max_level_;
    int entry_point_;
    float level_mult_;

    mutable std::mutex mu_;
    std::vector<Node> nodes_;
    std::mt19937 rng_;
};

} // namespace infergo
