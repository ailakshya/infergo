#pragma once

#include <cstdint>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace infergo {

/// LRU prompt cache: stores serialized KV state keyed by prompt token hash.
/// Thread-safe (locked internally). Used by infer_llm_generate to skip prefill
/// when the same prompt prefix has been seen before.
class PromptCache {
public:
    /// max_entries: max cached prompts (LRU eviction). 0 = disabled.
    explicit PromptCache(int max_entries = 16);

    /// Look up cached KV bytes for a prompt token sequence.
    /// Returns true if found (writes to out_kv and out_n_tokens).
    bool Get(const int32_t* tokens, int n_tokens,
             std::vector<uint8_t>& out_kv, int& out_n_tokens);

    /// Store KV bytes for a prompt token sequence.
    void Put(const int32_t* tokens, int n_tokens,
             const uint8_t* kv_data, size_t kv_size);

    /// Number of entries currently cached.
    int Size() const;

    /// Clear all entries.
    void Clear();

private:
    struct Entry {
        uint64_t              hash;
        int                   n_tokens;
        std::vector<uint8_t>  kv_data;
    };

    static uint64_t HashTokens(const int32_t* tokens, int n);

    int max_entries_;
    mutable std::mutex mu_;
    std::list<Entry> entries_;  // front = most recently used
    std::unordered_map<uint64_t, std::list<Entry>::iterator> index_;
};

} // namespace infergo
