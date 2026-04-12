#include "prompt_cache.hpp"
#include <cstring>

namespace infergo {

PromptCache::PromptCache(int max_entries)
    : max_entries_(max_entries > 0 ? max_entries : 0) {}

uint64_t PromptCache::HashTokens(const int32_t* tokens, int n) {
    // FNV-1a 64-bit
    uint64_t h = 14695981039346656037ULL;
    const auto* bytes = reinterpret_cast<const uint8_t*>(tokens);
    const size_t nbytes = static_cast<size_t>(n) * sizeof(int32_t);
    for (size_t i = 0; i < nbytes; ++i) {
        h ^= bytes[i];
        h *= 1099511628211ULL;
    }
    return h;
}

bool PromptCache::Get(const int32_t* tokens, int n_tokens,
                       std::vector<uint8_t>& out_kv, int& out_n_tokens) {
    if (max_entries_ <= 0 || tokens == nullptr || n_tokens <= 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mu_);

    uint64_t h = HashTokens(tokens, n_tokens);
    auto it = index_.find(h);
    if (it == index_.end()) {
        return false;
    }

    auto& entry = *it->second;
    if (entry.n_tokens != n_tokens) {
        return false;  // hash collision: different length
    }
    // Verify actual tokens match (not just hash)
    if (std::memcmp(entry.tokens.data(), tokens,
                     static_cast<size_t>(n_tokens) * sizeof(int32_t)) != 0) {
        return false;  // hash collision: different tokens
    }

    out_kv = entry.kv_data;
    out_n_tokens = entry.n_tokens;
    entries_.splice(entries_.begin(), entries_, it->second);
    return true;
}

void PromptCache::Put(const int32_t* tokens, int n_tokens,
                       const uint8_t* kv_data, size_t kv_size) {
    if (max_entries_ <= 0 || tokens == nullptr || n_tokens <= 0 ||
        kv_data == nullptr || kv_size == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(mu_);

    uint64_t h = HashTokens(tokens, n_tokens);

    // Update existing entry (verify tokens match to handle collisions)
    auto it = index_.find(h);
    if (it != index_.end()) {
        auto& entry = *it->second;
        if (entry.n_tokens == n_tokens &&
            std::memcmp(entry.tokens.data(), tokens,
                         static_cast<size_t>(n_tokens) * sizeof(int32_t)) == 0) {
            entry.kv_data.assign(kv_data, kv_data + kv_size);
            entries_.splice(entries_.begin(), entries_, it->second);
            return;
        }
        // Hash collision with different tokens — evict old, insert new below
        index_.erase(it);
        entries_.erase(it->second);
    }

    // Evict LRU if full
    while (static_cast<int>(entries_.size()) >= max_entries_) {
        auto& back = entries_.back();
        index_.erase(back.hash);
        entries_.pop_back();
    }

    // Insert new entry at front (store tokens for collision verification)
    Entry e;
    e.hash = h;
    e.n_tokens = n_tokens;
    e.tokens.assign(tokens, tokens + n_tokens);
    e.kv_data.assign(kv_data, kv_data + kv_size);
    entries_.push_front(std::move(e));
    index_[h] = entries_.begin();
}

int PromptCache::Size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(entries_.size());
}

void PromptCache::Clear() {
    std::lock_guard<std::mutex> lock(mu_);
    entries_.clear();
    index_.clear();
}

} // namespace infergo
