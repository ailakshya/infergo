#include "bm25.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <utility>

namespace infergo {

BM25Index::BM25Index(float k1, float b)
    : k1_(k1), b_(b) {}

std::string BM25Index::Stem(const std::string& word) {
    // Minimal English suffix stripping for search quality.
    // Order matters: try longest suffixes first.
    if (word.size() <= 3) return word;

    // -tion -> t  (e.g., "optimization" -> "optimizat")
    if (word.size() > 4 && word.compare(word.size() - 4, 4, "tion") == 0) {
        return word.substr(0, word.size() - 3);  // keep the 't'
    }
    // -ing  (e.g., "running" -> "runn", "searching" -> "search")
    if (word.size() > 4 && word.compare(word.size() - 3, 3, "ing") == 0) {
        return word.substr(0, word.size() - 3);
    }
    // -ly   (e.g., "quickly" -> "quick")
    if (word.size() > 3 && word.compare(word.size() - 2, 2, "ly") == 0) {
        return word.substr(0, word.size() - 2);
    }
    // -ed   (e.g., "searched" -> "search")
    if (word.size() > 3 && word.compare(word.size() - 2, 2, "ed") == 0) {
        return word.substr(0, word.size() - 2);
    }
    // -er   (e.g., "faster" -> "fast")
    if (word.size() > 3 && word.compare(word.size() - 2, 2, "er") == 0) {
        return word.substr(0, word.size() - 2);
    }
    // -es   (e.g., "boxes" -> "box")
    if (word.size() > 3 && word.compare(word.size() - 2, 2, "es") == 0) {
        return word.substr(0, word.size() - 2);
    }
    // -s    (e.g., "models" -> "model")
    if (word.size() > 3 && word.back() == 's') {
        return word.substr(0, word.size() - 1);
    }

    return word;
}

std::vector<std::string> BM25Index::Tokenize(const char* text) {
    std::vector<std::string> tokens;
    if (!text) return tokens;

    const char* p = text;
    std::string token;
    token.reserve(32);

    while (*p) {
        if (std::isalnum(static_cast<unsigned char>(*p))) {
            token += static_cast<char>(
                std::tolower(static_cast<unsigned char>(*p)));
        } else {
            if (!token.empty()) {
                // Skip very short tokens (1-char noise)
                if (token.size() >= 2) {
                    tokens.push_back(Stem(token));
                }
                token.clear();
            }
        }
        ++p;
    }
    // Flush last token
    if (token.size() >= 2) {
        tokens.push_back(Stem(token));
    }

    return tokens;
}

void BM25Index::UpdateAvgDocLength() {
    if (n_docs_ == 0) {
        avg_doc_length_ = 0;
        return;
    }
    int64_t total = 0;
    for (const auto& [id, len] : doc_lengths_) {
        total += len;
    }
    avg_doc_length_ = static_cast<float>(total) / static_cast<float>(n_docs_);
}

void BM25Index::Insert(int64_t id, const char* text) {
    std::lock_guard<std::mutex> lock(mu_);

    // If already exists, remove first
    if (doc_texts_.count(id)) {
        // Remove old postings
        auto old_tokens = Tokenize(doc_texts_[id].c_str());
        for (const auto& term : old_tokens) {
            auto it = posting_lists_.find(term);
            if (it != posting_lists_.end()) {
                auto& list = it->second;
                list.erase(
                    std::remove_if(list.begin(), list.end(),
                        [id](const std::pair<int64_t, int>& p) { return p.first == id; }),
                    list.end());
                if (list.empty()) posting_lists_.erase(it);
            }
        }
        doc_lengths_.erase(id);
        n_docs_--;
    }

    doc_texts_[id] = text ? text : "";
    auto tokens = Tokenize(text);
    doc_lengths_[id] = static_cast<int>(tokens.size());
    n_docs_++;

    // Count term frequencies for this document
    std::unordered_map<std::string, int> tf;
    for (const auto& t : tokens) {
        tf[t]++;
    }

    // Add to posting lists
    for (const auto& [term, freq] : tf) {
        posting_lists_[term].push_back({id, freq});
    }

    UpdateAvgDocLength();
}

void BM25Index::Remove(int64_t id) {
    std::lock_guard<std::mutex> lock(mu_);

    auto text_it = doc_texts_.find(id);
    if (text_it == doc_texts_.end()) return;

    auto tokens = Tokenize(text_it->second.c_str());
    for (const auto& term : tokens) {
        auto it = posting_lists_.find(term);
        if (it != posting_lists_.end()) {
            auto& list = it->second;
            list.erase(
                std::remove_if(list.begin(), list.end(),
                    [id](const std::pair<int64_t, int>& p) { return p.first == id; }),
                list.end());
            if (list.empty()) posting_lists_.erase(it);
        }
    }

    doc_texts_.erase(text_it);
    doc_lengths_.erase(id);
    n_docs_--;
    UpdateAvgDocLength();
}

void BM25Index::Search(const char* query, int k,
                       std::vector<int64_t>& out_ids,
                       std::vector<float>& out_scores) {
    std::lock_guard<std::mutex> lock(mu_);

    out_ids.clear();
    out_scores.clear();

    if (n_docs_ == 0 || !query || k <= 0) return;

    auto query_tokens = Tokenize(query);
    if (query_tokens.empty()) return;

    // Accumulate BM25 scores per document
    std::unordered_map<int64_t, float> scores;

    for (const auto& term : query_tokens) {
        auto it = posting_lists_.find(term);
        if (it == posting_lists_.end()) continue;

        const auto& postings = it->second;
        int df = static_cast<int>(postings.size());

        // IDF component: log((N - df + 0.5) / (df + 0.5) + 1)
        float idf = std::log(
            (static_cast<float>(n_docs_) - static_cast<float>(df) + 0.5f) /
            (static_cast<float>(df) + 0.5f) + 1.0f);

        for (const auto& [doc_id, tf] : postings) {
            float dl = static_cast<float>(doc_lengths_.at(doc_id));
            // BM25 TF component: (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
            float tf_norm = (static_cast<float>(tf) * (k1_ + 1.0f)) /
                (static_cast<float>(tf) + k1_ * (1.0f - b_ + b_ * dl / avg_doc_length_));
            scores[doc_id] += idf * tf_norm;
        }
    }

    // Collect and sort by score descending
    std::vector<std::pair<float, int64_t>> scored;
    scored.reserve(scores.size());
    for (const auto& [doc_id, score] : scores) {
        scored.push_back({score, doc_id});
    }

    // Partial sort for top-k (faster than full sort for large result sets)
    if (static_cast<int>(scored.size()) > k) {
        std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        scored.resize(k);
    } else {
        std::sort(scored.begin(), scored.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
    }

    out_ids.reserve(scored.size());
    out_scores.reserve(scored.size());
    for (const auto& [score, doc_id] : scored) {
        out_ids.push_back(doc_id);
        out_scores.push_back(score);
    }
}

bool BM25Index::Save(const char* path) {
    std::lock_guard<std::mutex> lock(mu_);
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    // Magic + version
    int32_t magic = 0x424D3235;  // "BM25"
    int32_t version = 1;
    f.write(reinterpret_cast<const char*>(&magic), 4);
    f.write(reinterpret_cast<const char*>(&version), 4);

    // BM25 params
    f.write(reinterpret_cast<const char*>(&k1_), sizeof(float));
    f.write(reinterpret_cast<const char*>(&b_), sizeof(float));

    // Document count
    int32_t count = static_cast<int32_t>(doc_texts_.size());
    f.write(reinterpret_cast<const char*>(&count), 4);

    // Document texts (we rebuild the index from these on Load)
    for (const auto& [id, text] : doc_texts_) {
        int64_t eid = id;
        f.write(reinterpret_cast<const char*>(&eid), 8);
        int32_t tlen = static_cast<int32_t>(text.size());
        f.write(reinterpret_cast<const char*>(&tlen), 4);
        f.write(text.data(), tlen);
    }

    return f.good();
}

bool BM25Index::Load(const char* path) {
    std::lock_guard<std::mutex> lock(mu_);
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    int32_t magic, version;
    f.read(reinterpret_cast<char*>(&magic), 4);
    f.read(reinterpret_cast<char*>(&version), 4);
    if (magic != 0x424D3235 || version != 1) return false;

    f.read(reinterpret_cast<char*>(&k1_), sizeof(float));
    f.read(reinterpret_cast<char*>(&b_), sizeof(float));

    int32_t count;
    f.read(reinterpret_cast<char*>(&count), 4);

    // Clear existing state
    posting_lists_.clear();
    doc_lengths_.clear();
    doc_texts_.clear();
    n_docs_ = 0;
    avg_doc_length_ = 0;

    // Read documents and rebuild index (without lock since we already hold it)
    for (int32_t i = 0; i < count; ++i) {
        int64_t id;
        f.read(reinterpret_cast<char*>(&id), 8);

        int32_t tlen;
        f.read(reinterpret_cast<char*>(&tlen), 4);
        std::string text(tlen, '\0');
        f.read(&text[0], tlen);

        doc_texts_[id] = text;
        auto tokens = Tokenize(text.c_str());
        doc_lengths_[id] = static_cast<int>(tokens.size());
        n_docs_++;

        std::unordered_map<std::string, int> tf;
        for (const auto& t : tokens) {
            tf[t]++;
        }
        for (const auto& [term, freq] : tf) {
            posting_lists_[term].push_back({id, freq});
        }
    }

    UpdateAvgDocLength();
    return f.good();
}

int BM25Index::Size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return n_docs_;
}

} // namespace infergo
