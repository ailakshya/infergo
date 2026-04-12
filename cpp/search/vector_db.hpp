#pragma once

#include "hnsw.hpp"
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace infergo {

/// Full vector database: HNSW index + metadata store + persistence.
/// Thread-safe. Supports insert, delete, update, search with metadata filtering.
class VectorDB {
public:
    VectorDB(int dim, int M = 16, int ef_construction = 200);

    // ─── CRUD ───────────────────────────────────────────────────────
    bool Insert(int64_t id, const float* vec, const std::string& metadata);
    bool Delete(int64_t id);
    bool Update(int64_t id, const float* vec, const std::string& metadata);
    bool Get(int64_t id, float* out_vec, std::string& out_metadata) const;

    // ─── Search ─────────────────────────────────────────────────────
    struct SearchResult {
        int64_t     id;
        float       distance;
        std::string metadata;
    };

    std::vector<SearchResult> Search(const float* query, int k, int ef_search,
                                      const std::string& metadata_filter = "") const;

    // ─── Persistence ────────────────────────────────────────────────
    bool Save(const std::string& path) const;
    bool Load(const std::string& path);

    // ─── Info ───────────────────────────────────────────────────────
    int Size() const;
    int Dim() const { return dim_; }

private:
    int dim_;
    HNSWIndex index_;
    mutable std::mutex mu_;

    // Metadata store (keyed by ID)
    struct Entry {
        std::vector<float> vec;
        std::string metadata;
    };
    std::unordered_map<int64_t, Entry> store_;
};

} // namespace infergo
