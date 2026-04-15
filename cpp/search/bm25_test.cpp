// Standalone BM25 + Hybrid search test.
// Build: g++ -std=c++17 -O2 -o bm25_test bm25_test.cpp bm25.cpp hybrid.cpp -lpthread
// Run:   ./bm25_test

#include "bm25.hpp"
#include "hybrid.hpp"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    static void test_##name(); \
    static struct TestRunner_##name { \
        TestRunner_##name() { \
            printf("  TEST %-40s ", #name); \
            try { test_##name(); printf("PASS\n"); tests_passed++; } \
            catch (...) { printf("FAIL\n"); tests_failed++; } \
        } \
    } runner_##name; \
    static void test_##name()

#define ASSERT(cond) do { if (!(cond)) { printf("\n    ASSERT FAILED: %s (line %d)\n", #cond, __LINE__); throw 1; } } while(0)

// ─── BM25 Tests ─────────────────────────────────────────────────────────────

TEST(bm25_create_empty) {
    infergo::BM25Index idx;
    ASSERT(idx.Size() == 0);
}

TEST(bm25_insert_and_size) {
    infergo::BM25Index idx;
    idx.Insert(1, "the quick brown fox");
    idx.Insert(2, "machine learning model");
    idx.Insert(3, "golang programming");
    ASSERT(idx.Size() == 3);
}

TEST(bm25_exact_keyword_match) {
    infergo::BM25Index idx;
    idx.Insert(1, "golang programming language for backend");
    idx.Insert(2, "python machine learning framework");
    idx.Insert(3, "javascript frontend web development");

    std::vector<int64_t> ids;
    std::vector<float> scores;
    idx.Search("golang programming", 3, ids, scores);

    ASSERT(ids.size() >= 1);
    ASSERT(ids[0] == 1);  // golang doc should be top
}

TEST(bm25_ranking_by_relevance) {
    infergo::BM25Index idx;
    idx.Insert(1, "machine learning is a field of AI");
    idx.Insert(2, "deep learning neural networks");
    idx.Insert(3, "machine learning machine learning machine learning algorithms");
    idx.Insert(4, "cooking recipes for pasta");

    std::vector<int64_t> ids;
    std::vector<float> scores;
    idx.Search("machine learning", 4, ids, scores);

    ASSERT(ids.size() >= 2);
    ASSERT(ids[0] == 3);  // most mentions
    ASSERT(ids[1] == 1);  // one mention

    // Scores descending
    for (size_t i = 1; i < scores.size(); ++i) {
        ASSERT(scores[i] <= scores[i-1]);
    }
}

TEST(bm25_remove) {
    infergo::BM25Index idx;
    idx.Insert(1, "alpha beta gamma");
    idx.Insert(2, "delta epsilon zeta");
    ASSERT(idx.Size() == 2);

    idx.Remove(1);
    ASSERT(idx.Size() == 1);

    std::vector<int64_t> ids;
    std::vector<float> scores;
    idx.Search("alpha", 5, ids, scores);
    ASSERT(ids.size() == 0);
}

TEST(bm25_stemming) {
    infergo::BM25Index idx;
    idx.Insert(1, "running quickly through the searching algorithm");
    idx.Insert(2, "the runner searched for optimal solutions");

    std::vector<int64_t> ids;
    std::vector<float> scores;
    // "searched" -> "search", should match "searching" -> "search"
    idx.Search("searched", 5, ids, scores);
    ASSERT(ids.size() >= 2);
}

TEST(bm25_empty_query) {
    infergo::BM25Index idx;
    idx.Insert(1, "some document text");

    std::vector<int64_t> ids;
    std::vector<float> scores;
    idx.Search("", 5, ids, scores);
    ASSERT(ids.size() == 0);
}

TEST(bm25_case_insensitive) {
    infergo::BM25Index idx;
    idx.Insert(1, "Machine Learning Models");

    std::vector<int64_t> ids;
    std::vector<float> scores;
    idx.Search("machine learning", 5, ids, scores);
    ASSERT(ids.size() == 1);
    ASSERT(ids[0] == 1);
}

TEST(bm25_save_load) {
    const char* path = "/tmp/bm25_test.bin";

    {
        infergo::BM25Index idx;
        idx.Insert(1, "alpha beta gamma");
        idx.Insert(2, "delta epsilon zeta");
        ASSERT(idx.Save(path));
    }

    {
        infergo::BM25Index idx;
        ASSERT(idx.Load(path));
        ASSERT(idx.Size() == 2);

        std::vector<int64_t> ids;
        std::vector<float> scores;
        idx.Search("alpha", 5, ids, scores);
        ASSERT(ids.size() == 1);
        ASSERT(ids[0] == 1);
    }
}

TEST(bm25_update_document) {
    infergo::BM25Index idx;
    idx.Insert(1, "old content about cats");

    std::vector<int64_t> ids;
    std::vector<float> scores;
    idx.Search("cats", 5, ids, scores);
    ASSERT(ids.size() == 1);

    // Re-insert with new content (should replace)
    idx.Insert(1, "new content about dogs");
    ASSERT(idx.Size() == 1);

    ids.clear(); scores.clear();
    idx.Search("cats", 5, ids, scores);
    ASSERT(ids.size() == 0);

    ids.clear(); scores.clear();
    idx.Search("dogs", 5, ids, scores);
    ASSERT(ids.size() == 1);
    ASSERT(ids[0] == 1);
}

// ─── Hybrid Search Tests ────────────────────────────────────────────────────

TEST(hybrid_combines_both) {
    // Doc 3 appears in both lists
    int64_t vec_ids[] = {1, 3, 4};
    float vec_dists[] = {0.1f, 0.2f, 0.5f};

    int64_t bm25_ids[] = {2, 3, 5};
    float bm25_scores[] = {5.0f, 4.0f, 1.0f};

    auto results = infergo::HybridSearch(
        vec_ids, vec_dists, 3,
        bm25_ids, bm25_scores, 3,
        0.5f, 5);

    ASSERT(!results.empty());
    ASSERT(results[0].id == 3);  // appears in both
}

TEST(hybrid_alpha_pure_vector) {
    int64_t vec_ids[] = {1, 2};
    float vec_dists[] = {0.1f, 0.5f};

    int64_t bm25_ids[] = {3, 2};
    float bm25_scores[] = {10.0f, 5.0f};

    auto results = infergo::HybridSearch(
        vec_ids, vec_dists, 2,
        bm25_ids, bm25_scores, 2,
        1.0f, 5);

    ASSERT(!results.empty());
    ASSERT(results[0].id == 1);  // best vector match
}

TEST(hybrid_alpha_pure_bm25) {
    int64_t vec_ids[] = {1, 2};
    float vec_dists[] = {0.1f, 0.5f};

    int64_t bm25_ids[] = {3, 2};
    float bm25_scores[] = {10.0f, 5.0f};

    auto results = infergo::HybridSearch(
        vec_ids, vec_dists, 2,
        bm25_ids, bm25_scores, 2,
        0.0f, 5);

    ASSERT(!results.empty());
    ASSERT(results[0].id == 3);  // best BM25 match
}

TEST(hybrid_scores_in_range) {
    int64_t vec_ids[] = {1, 2, 3};
    float vec_dists[] = {0.1f, 0.3f, 0.7f};

    int64_t bm25_ids[] = {1, 4, 2};
    float bm25_scores[] = {8.0f, 3.0f, 1.0f};

    auto results = infergo::HybridSearch(
        vec_ids, vec_dists, 3,
        bm25_ids, bm25_scores, 3,
        0.5f, 10);

    for (const auto& r : results) {
        ASSERT(r.score >= 0.0f && r.score <= 1.0f);
    }

    // Descending order
    for (size_t i = 1; i < results.size(); ++i) {
        ASSERT(results[i].score <= results[i-1].score);
    }
}

// ─── Performance Benchmark ──────────────────────────────────────────────────

TEST(bm25_10k_under_1ms) {
    infergo::BM25Index idx;

    const char* docs[] = {
        "machine learning algorithms for data science applications",
        "deep neural networks for computer vision tasks",
        "natural language processing with transformer models",
        "reinforcement learning for robotics control",
        "graph neural networks for social network analysis",
        "generative adversarial networks for image synthesis",
        "federated learning for privacy preserving machine learning",
        "transfer learning techniques for domain adaptation",
        "attention mechanisms in sequence to sequence models",
        "bayesian optimization for hyperparameter tuning",
    };

    for (int i = 0; i < 10000; ++i) {
        idx.Insert(static_cast<int64_t>(i), docs[i % 10]);
    }
    ASSERT(idx.Size() == 10000);

    // Warm up
    std::vector<int64_t> ids;
    std::vector<float> scores;
    idx.Search("machine learning algorithms", 10, ids, scores);

    // Benchmark: average over 100 queries
    auto start = std::chrono::high_resolution_clock::now();
    const int N = 100;
    for (int i = 0; i < N; ++i) {
        ids.clear(); scores.clear();
        idx.Search("machine learning algorithms", 10, ids, scores);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double avg_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / static_cast<double>(N);

    printf("[%.0f us/query] ", avg_us);
    ASSERT(avg_us < 1000.0);  // must be under 1ms
}

int main() {
    printf("\n=== BM25 + Hybrid Search Tests ===\n\n");
    printf("\nResults: %d passed, %d failed\n\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
