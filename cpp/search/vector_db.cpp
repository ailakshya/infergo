#include "vector_db.hpp"
#include <algorithm>
#include <cstring>
#include <sstream>

namespace infergo {

VectorDB::VectorDB(int dim, int M, int ef_construction)
    : dim_(dim), index_(dim, M, ef_construction) {}

bool VectorDB::Insert(int64_t id, const float* vec, const std::string& metadata) {
    std::lock_guard<std::mutex> lock(mu_);
    if (store_.count(id)) return false;  // already exists

    Entry e;
    e.vec.assign(vec, vec + dim_);
    e.metadata = metadata;
    store_[id] = std::move(e);

    return index_.Insert(id, vec, metadata);
}

bool VectorDB::Delete(int64_t id) {
    std::lock_guard<std::mutex> lock(mu_);
    return store_.erase(id) > 0;
    // Note: HNSW doesn't support true deletion — the vector stays in the index
    // but won't be returned by search since metadata lookup will fail.
}

bool VectorDB::Update(int64_t id, const float* vec, const std::string& metadata) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = store_.find(id);
    if (it == store_.end()) return false;

    it->second.vec.assign(vec, vec + dim_);
    it->second.metadata = metadata;
    // Re-insert into index (HNSW doesn't support in-place update)
    index_.Insert(id, vec, metadata);
    return true;
}

bool VectorDB::Get(int64_t id, float* out_vec, std::string& out_metadata) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = store_.find(id);
    if (it == store_.end()) return false;

    if (out_vec) {
        std::memcpy(out_vec, it->second.vec.data(),
                     static_cast<size_t>(dim_) * sizeof(float));
    }
    out_metadata = it->second.metadata;
    return true;
}

std::vector<VectorDB::SearchResult> VectorDB::Search(
    const float* query, int k, int ef_search,
    const std::string& metadata_filter) const
{
    std::lock_guard<std::mutex> lock(mu_);

    std::vector<int64_t> ids;
    std::vector<float> dists;
    std::vector<std::string> metas;

    // Search more than k to account for deleted/filtered entries
    int search_k = metadata_filter.empty() ? k : k * 3;
    index_.Search(query, search_k, ef_search, ids, dists, metas);

    std::vector<SearchResult> results;
    for (size_t i = 0; i < ids.size() && static_cast<int>(results.size()) < k; ++i) {
        // Check if entry still exists (not deleted)
        auto it = store_.find(ids[i]);
        if (it == store_.end()) continue;

        // Apply metadata filter (simple substring match)
        if (!metadata_filter.empty() &&
            it->second.metadata.find(metadata_filter) == std::string::npos) {
            continue;
        }

        results.push_back({ids[i], dists[i], it->second.metadata});
    }
    return results;
}

bool VectorDB::Save(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mu_);
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    // Header
    int32_t magic = 0x56444249;  // "VDBI"
    int32_t version = 1;
    int32_t dim = dim_;
    int32_t count = static_cast<int32_t>(store_.size());
    f.write(reinterpret_cast<const char*>(&magic), 4);
    f.write(reinterpret_cast<const char*>(&version), 4);
    f.write(reinterpret_cast<const char*>(&dim), 4);
    f.write(reinterpret_cast<const char*>(&count), 4);

    // Entries
    for (const auto& [id, entry] : store_) {
        int64_t eid = id;
        f.write(reinterpret_cast<const char*>(&eid), 8);
        f.write(reinterpret_cast<const char*>(entry.vec.data()),
                static_cast<std::streamsize>(dim_ * sizeof(float)));
        int32_t mlen = static_cast<int32_t>(entry.metadata.size());
        f.write(reinterpret_cast<const char*>(&mlen), 4);
        f.write(entry.metadata.data(), mlen);
    }

    return f.good();
}

bool VectorDB::Load(const std::string& path) {
    std::lock_guard<std::mutex> lock(mu_);
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    int32_t magic, version, dim, count;
    f.read(reinterpret_cast<char*>(&magic), 4);
    f.read(reinterpret_cast<char*>(&version), 4);
    f.read(reinterpret_cast<char*>(&dim), 4);
    f.read(reinterpret_cast<char*>(&count), 4);

    if (magic != 0x56444249 || version != 1 || dim != dim_) return false;

    store_.clear();
    // Reconstruct index in-place (can't copy-assign because of mutex)
    index_.~HNSWIndex();
    new (&index_) HNSWIndex(dim_, 16, 200);

    for (int32_t i = 0; i < count; ++i) {
        int64_t id;
        f.read(reinterpret_cast<char*>(&id), 8);

        std::vector<float> vec(dim_);
        f.read(reinterpret_cast<char*>(vec.data()),
               static_cast<std::streamsize>(dim_ * sizeof(float)));

        int32_t mlen;
        f.read(reinterpret_cast<char*>(&mlen), 4);
        std::string meta(mlen, '\0');
        f.read(&meta[0], mlen);

        Entry e;
        e.vec = vec;
        e.metadata = meta;
        store_[id] = std::move(e);
        index_.Insert(id, vec.data(), meta);
    }

    return f.good();
}

int VectorDB::Size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(store_.size());
}

} // namespace infergo
