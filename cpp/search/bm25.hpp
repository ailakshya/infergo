#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>

namespace infergo {

/// BM25 full-text search index.
/// Thread-safe: Insert/Remove/Search are all serialized via mutex.
/// Designed for sub-millisecond search on 10K+ documents.
class BM25Index {
public:
    BM25Index(float k1 = 1.2f, float b = 0.75f);
    ~BM25Index() = default;

    /// Add a document to the index.
    void Insert(int64_t id, const char* text);

    /// Remove a document from the index.
    void Remove(int64_t id);

    /// Search: returns (doc_id, score) pairs sorted by BM25 score descending.
    void Search(const char* query, int k,
                std::vector<int64_t>& out_ids,
                std::vector<float>& out_scores);

    /// Persistence
    bool Save(const char* path);
    bool Load(const char* path);

    int Size() const;

private:
    /// Tokenize text into terms (lowercase, basic stemming).
    static std::vector<std::string> Tokenize(const char* text);

    /// Basic English stemmer: strip -ing, -tion, -ly, -ed, -er, -es, -s suffixes.
    static std::string Stem(const std::string& word);

    /// Recompute average document length.
    void UpdateAvgDocLength();

    // Inverted index: term -> list of (doc_id, term_frequency)
    std::unordered_map<std::string, std::vector<std::pair<int64_t, int>>> posting_lists_;

    // Document lengths (in tokens)
    std::unordered_map<int64_t, int> doc_lengths_;

    // Document texts (stored for re-indexing on Load)
    std::unordered_map<int64_t, std::string> doc_texts_;

    float avg_doc_length_ = 0;
    int n_docs_ = 0;

    // BM25 parameters
    float k1_;
    float b_;

    mutable std::mutex mu_;
};

} // namespace infergo
