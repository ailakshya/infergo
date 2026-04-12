#include "hnsw.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <queue>

namespace infergo {

HNSWIndex::HNSWIndex(int dim, int M, int ef_construction)
    : dim_(dim), M_(M), ef_construction_(ef_construction),
      max_level_(0), entry_point_(-1),
      level_mult_(1.0f / std::log(static_cast<float>(M))),
      rng_(42) {}

float HNSWIndex::CosineDistance(const float* a, const float* b) const {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < dim_; ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float denom = std::sqrt(na) * std::sqrt(nb);
    if (denom < 1e-12f) return 1.0f;
    return 1.0f - dot / denom;
}

int HNSWIndex::RandomLevel() {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    int level = 0;
    while (dist(rng_) < std::exp(-static_cast<float>(level + 1) / level_mult_) && level < 16) {
        level++;
    }
    return level;
}

void HNSWIndex::SearchLayer(const float* query, int entry, int ef, int layer,
                             std::vector<std::pair<float, int>>& result) const {
    // Min-heap: closest nodes first
    using Pair = std::pair<float, int>;
    auto cmp_max = [](const Pair& a, const Pair& b) { return a.first < b.first; };
    auto cmp_min = [](const Pair& a, const Pair& b) { return a.first > b.first; };

    std::priority_queue<Pair, std::vector<Pair>, decltype(cmp_min)> candidates(cmp_min);
    std::priority_queue<Pair, std::vector<Pair>, decltype(cmp_max)> best(cmp_max);

    float d = CosineDistance(query, nodes_[entry].vec.data());
    candidates.push({d, entry});
    best.push({d, entry});

    std::unordered_map<int, bool> visited;
    visited[entry] = true;

    while (!candidates.empty()) {
        auto [cd, cid] = candidates.top();
        candidates.pop();

        if (cd > best.top().first) break;

        const auto& node = nodes_[cid];
        if (layer < static_cast<int>(node.neighbors.size())) {
            for (int neighbor : node.neighbors[layer]) {
                if (visited.count(neighbor)) continue;
                visited[neighbor] = true;

                float nd = CosineDistance(query, nodes_[neighbor].vec.data());
                if (nd < best.top().first || static_cast<int>(best.size()) < ef) {
                    candidates.push({nd, neighbor});
                    best.push({nd, neighbor});
                    if (static_cast<int>(best.size()) > ef) best.pop();
                }
            }
        }
    }

    result.clear();
    while (!best.empty()) {
        result.push_back(best.top());
        best.pop();
    }
    std::sort(result.begin(), result.end());
}

bool HNSWIndex::Insert(int64_t id, const float* vec, const std::string& metadata) {
    std::lock_guard<std::mutex> lock(mu_);

    int node_idx = static_cast<int>(nodes_.size());
    int level = RandomLevel();

    Node node;
    node.id = id;
    node.metadata = metadata;
    node.vec.assign(vec, vec + dim_);
    node.max_layer = level;
    node.neighbors.resize(static_cast<size_t>(level + 1));
    nodes_.push_back(std::move(node));

    if (entry_point_ < 0) {
        entry_point_ = node_idx;
        max_level_ = level;
        return true;
    }

    int cur = entry_point_;

    // Navigate from top layers down to the node's level
    for (int l = max_level_; l > level; --l) {
        std::vector<std::pair<float, int>> nearest;
        SearchLayer(vec, cur, 1, l, nearest);
        if (!nearest.empty()) cur = nearest[0].second;
    }

    // Insert into each layer from level down to 0
    for (int l = std::min(level, max_level_); l >= 0; --l) {
        std::vector<std::pair<float, int>> nearest;
        SearchLayer(vec, cur, ef_construction_, l, nearest);

        // Connect to M nearest neighbors
        int connections = std::min(M_, static_cast<int>(nearest.size()));
        for (int i = 0; i < connections; ++i) {
            int neighbor = nearest[i].second;
            nodes_[node_idx].neighbors[l].push_back(neighbor);

            // Add reverse connection (if neighbor has this layer)
            if (l < static_cast<int>(nodes_[neighbor].neighbors.size())) {
                nodes_[neighbor].neighbors[l].push_back(node_idx);
                // Prune if too many connections
                if (static_cast<int>(nodes_[neighbor].neighbors[l].size()) > M_ * 2) {
                    // Keep only M closest
                    auto& nbs = nodes_[neighbor].neighbors[l];
                    std::vector<std::pair<float, int>> scored;
                    for (int nb : nbs) {
                        scored.push_back({CosineDistance(nodes_[neighbor].vec.data(),
                                                         nodes_[nb].vec.data()), nb});
                    }
                    std::sort(scored.begin(), scored.end());
                    nbs.clear();
                    for (int j = 0; j < M_ && j < static_cast<int>(scored.size()); ++j) {
                        nbs.push_back(scored[j].second);
                    }
                }
            }
        }

        if (!nearest.empty()) cur = nearest[0].second;
    }

    if (level > max_level_) {
        max_level_ = level;
        entry_point_ = node_idx;
    }

    return true;
}

void HNSWIndex::Search(const float* query, int k, int ef_search,
                        std::vector<int64_t>& out_ids,
                        std::vector<float>& out_distances,
                        std::vector<std::string>& out_metadata) const {
    std::lock_guard<std::mutex> lock(mu_);

    out_ids.clear();
    out_distances.clear();
    out_metadata.clear();

    if (entry_point_ < 0 || nodes_.empty()) return;

    int cur = entry_point_;

    // Navigate from top to layer 1
    for (int l = max_level_; l > 0; --l) {
        std::vector<std::pair<float, int>> nearest;
        SearchLayer(query, cur, 1, l, nearest);
        if (!nearest.empty()) cur = nearest[0].second;
    }

    // Search layer 0 with ef_search width
    std::vector<std::pair<float, int>> result;
    SearchLayer(query, cur, std::max(ef_search, k), 0, result);

    int n = std::min(k, static_cast<int>(result.size()));
    for (int i = 0; i < n; ++i) {
        out_ids.push_back(nodes_[result[i].second].id);
        out_distances.push_back(result[i].first);
        out_metadata.push_back(nodes_[result[i].second].metadata);
    }
}

int HNSWIndex::Size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(nodes_.size());
}

} // namespace infergo
