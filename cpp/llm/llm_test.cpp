#include "llm_engine.hpp"

#include <algorithm>
#include <gtest/gtest.h>

#ifndef TEST_MODEL_PATH
#define TEST_MODEL_PATH "/tmp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
#endif

using namespace infergo;

// ─── Helpers ──────────────────────────────────────────────────────────────────

static bool model_available() {
    FILE* f = fopen(TEST_MODEL_PATH, "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

#define SKIP_IF_NO_MODEL() \
    if (!model_available()) { \
        GTEST_SKIP() << "Model not found at " TEST_MODEL_PATH; \
    }

// ─── Construction ─────────────────────────────────────────────────────────────

TEST(LLMEngine, ConstructAndDestroy) {
    EXPECT_NO_THROW(LLMEngine e);
}

// ─── LoadModel ────────────────────────────────────────────────────────────────

TEST(LLMEngine, LoadBadPathThrows) {
    LLMEngine e;
    EXPECT_THROW(e.LoadModel("/no/such/model.gguf"), std::runtime_error);
}

TEST(LLMEngine, LoadValid) {
    SKIP_IF_NO_MODEL();
    LLMEngine e;
    EXPECT_NO_THROW(e.LoadModel(TEST_MODEL_PATH, /*n_gpu_layers=*/99,
                                /*ctx_size=*/512, /*n_seq_max=*/4, /*n_batch=*/256));
}

TEST(LLMEngine, VocabSizePositive) {
    SKIP_IF_NO_MODEL();
    LLMEngine e;
    e.LoadModel(TEST_MODEL_PATH, 99, 512, 4, 256);
    EXPECT_GT(e.VocabSize(), 0);
}

TEST(LLMEngine, BOSEOSValid) {
    SKIP_IF_NO_MODEL();
    LLMEngine e;
    e.LoadModel(TEST_MODEL_PATH, 99, 512, 4, 256);
    EXPECT_GE(e.BOS(), 0);
    EXPECT_GE(e.EOS(), 0);
}

// ─── BatchDecode (single sequence) ───────────────────────────────────────────

TEST(LLMEngine, SingleSequenceGetsLogits) {
    SKIP_IF_NO_MODEL();
    LLMEngine e;
    e.LoadModel(TEST_MODEL_PATH, 99, 512, 4, 256);

    // BOS token as a 1-token prompt
    SequenceInput seq;
    seq.seq_id      = 0;
    seq.tokens      = { e.BOS() };
    seq.pos         = 0;
    seq.want_logits = true;

    auto results = e.BatchDecode({ seq });
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].seq_id, 0);
    EXPECT_EQ(static_cast<int>(results[0].logits.size()), e.VocabSize());

    // Logits must not all be zero
    const auto& logits = results[0].logits;
    bool all_zero = std::all_of(logits.begin(), logits.end(),
                                [](float v){ return v == 0.0f; });
    EXPECT_FALSE(all_zero);
}

// ─── BatchDecode (two sequences simultaneously) ───────────────────────────────

TEST(LLMEngine, TwoSequencesSimultaneously) {
    SKIP_IF_NO_MODEL();
    LLMEngine e;
    e.LoadModel(TEST_MODEL_PATH, 99, 512, 4, 256);

    int32_t bos = e.BOS();

    SequenceInput seq0;
    seq0.seq_id      = 0;
    seq0.tokens      = { bos, bos };  // 2-token prompt, seq 0
    seq0.pos         = 0;
    seq0.want_logits = true;

    SequenceInput seq1;
    seq1.seq_id      = 1;
    seq1.tokens      = { bos };       // 1-token prompt, seq 1
    seq1.pos         = 0;
    seq1.want_logits = true;

    std::vector<SequenceLogits> results;
    ASSERT_NO_THROW(results = e.BatchDecode({ seq0, seq1 }));

    ASSERT_EQ(results.size(), 2u);

    for (const auto& r : results) {
        EXPECT_EQ(static_cast<int>(r.logits.size()), e.VocabSize());
        bool all_zero = std::all_of(r.logits.begin(), r.logits.end(),
                                    [](float v){ return v == 0.0f; });
        EXPECT_FALSE(all_zero) << "seq_id=" << r.seq_id << " has all-zero logits";
    }

    // The two sequences should produce different logit distributions
    // (seq0 has 2 tokens processed, seq1 has 1)
    EXPECT_NE(results[0].logits[0], results[1].logits[0]);
}

// ─── BatchDecode (no logits requested) ───────────────────────────────────────

TEST(LLMEngine, PrefillWithoutLogits) {
    SKIP_IF_NO_MODEL();
    LLMEngine e;
    e.LoadModel(TEST_MODEL_PATH, 99, 512, 4, 256);

    SequenceInput seq;
    seq.seq_id      = 0;
    seq.tokens      = { e.BOS() };
    seq.pos         = 0;
    seq.want_logits = false;  // prefill only, no logits needed

    std::vector<SequenceLogits> results;
    ASSERT_NO_THROW(results = e.BatchDecode({ seq }));
    EXPECT_EQ(results.size(), 0u);
}

// ─── IsEOG ────────────────────────────────────────────────────────────────────

TEST(LLMEngine, EOSIsEOG) {
    SKIP_IF_NO_MODEL();
    LLMEngine e;
    e.LoadModel(TEST_MODEL_PATH, 99, 512, 4, 256);
    EXPECT_TRUE(e.IsEOG(e.EOS()));
}

TEST(LLMEngine, BOSIsNotEOG) {
    SKIP_IF_NO_MODEL();
    LLMEngine e;
    e.LoadModel(TEST_MODEL_PATH, 99, 512, 4, 256);
    EXPECT_FALSE(e.IsEOG(e.BOS()));
}
