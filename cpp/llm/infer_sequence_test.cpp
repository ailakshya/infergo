#include "infer_sequence.hpp"
#include "llm_engine.hpp"
#include "sampler.hpp"

#include <gtest/gtest.h>

#ifndef TEST_MODEL_PATH
#define TEST_MODEL_PATH "/tmp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
#endif

using namespace infergo;

static bool model_available() {
    FILE* f = fopen(TEST_MODEL_PATH, "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

#define SKIP_IF_NO_MODEL() \
    if (!model_available()) { GTEST_SKIP() << "Model not found at " TEST_MODEL_PATH; }

// ─── Construction ─────────────────────────────────────────────────────────────

TEST(InferSequence, ConstructValid) {
    KVCacheSlotManager mgr(4);
    EXPECT_NO_THROW(InferSequence seq(mgr, {1, 2, 3}, /*eos=*/2));
    EXPECT_EQ(mgr.FreeCount(), 4);  // destructor freed the slot
}

TEST(InferSequence, EmptyPromptThrows) {
    KVCacheSlotManager mgr(4);
    EXPECT_THROW(InferSequence seq(mgr, {}, 2), std::invalid_argument);
}

TEST(InferSequence, FullPoolThrows) {
    KVCacheSlotManager mgr(1);
    InferSequence seq(mgr, {1}, 2);  // takes the only slot
    EXPECT_THROW(InferSequence seq2(mgr, {1}, 2), std::runtime_error);
}

// ─── SlotID / Position / Tokens ───────────────────────────────────────────────

TEST(InferSequence, SlotIDInRange) {
    KVCacheSlotManager mgr(8);
    InferSequence seq(mgr, {1, 2, 3}, 2);
    EXPECT_GE(seq.SlotID(), 0);
    EXPECT_LT(seq.SlotID(), 8);
}

TEST(InferSequence, InitialPositionZero) {
    KVCacheSlotManager mgr(4);
    InferSequence seq(mgr, {1, 2, 3}, 2);
    EXPECT_EQ(seq.Position(), 0);
    EXPECT_EQ(seq.GeneratedCount(), 0);
}

TEST(InferSequence, TokensMatchPrompt) {
    KVCacheSlotManager mgr(4);
    InferSequence seq(mgr, {10, 20, 30}, 2);
    ASSERT_EQ(seq.Tokens().size(), 3u);
    EXPECT_EQ(seq.Tokens()[0], 10);
    EXPECT_EQ(seq.Tokens()[2], 30);
    EXPECT_EQ(seq.LastToken(), 30);
}

// ─── NextTokens ───────────────────────────────────────────────────────────────

TEST(InferSequence, NextTokensBeforeFirstDecodeIsFullPrompt) {
    KVCacheSlotManager mgr(4);
    InferSequence seq(mgr, {5, 6, 7}, 2);
    auto next = seq.NextTokens();
    ASSERT_EQ(next.size(), 3u);
    EXPECT_EQ(next[0], 5);
    EXPECT_EQ(next[2], 7);
}

TEST(InferSequence, NextTokensAfterAppendIsSingleToken) {
    KVCacheSlotManager mgr(4);
    InferSequence seq(mgr, {5, 6, 7}, 2);
    seq.AppendToken(99);  // first append = end of prefill
    auto next = seq.NextTokens();
    ASSERT_EQ(next.size(), 1u);
    EXPECT_EQ(next[0], 99);
}

// ─── AppendToken ──────────────────────────────────────────────────────────────

TEST(InferSequence, AppendTokenAdvancesPosition) {
    KVCacheSlotManager mgr(4);
    InferSequence seq(mgr, {1, 2, 3}, /*eos=*/99);

    seq.AppendToken(10);   // end of prefill: kv_pos = 3
    EXPECT_EQ(seq.Position(), 3);
    EXPECT_EQ(seq.GeneratedCount(), 1);

    seq.AppendToken(11);   // kv_pos = 4
    EXPECT_EQ(seq.Position(), 4);
    EXPECT_EQ(seq.GeneratedCount(), 2);
}

TEST(InferSequence, AppendEOSSetsIsDone) {
    KVCacheSlotManager mgr(4);
    InferSequence seq(mgr, {1, 2, 3}, /*eos=*/2);
    EXPECT_FALSE(seq.IsDone());

    seq.AppendToken(42);   // non-EOS
    EXPECT_FALSE(seq.IsDone());

    seq.AppendToken(2);    // EOS
    EXPECT_TRUE(seq.IsDone());
}

TEST(InferSequence, AppendAfterDoneIsNoop) {
    KVCacheSlotManager mgr(4);
    InferSequence seq(mgr, {1}, /*eos=*/2);
    seq.AppendToken(2);   // EOS
    EXPECT_TRUE(seq.IsDone());

    const int pos_before = seq.Position();
    const size_t len_before = seq.Tokens().size();
    seq.AppendToken(99);  // must be ignored
    EXPECT_EQ(seq.Position(), pos_before);
    EXPECT_EQ(seq.Tokens().size(), len_before);
}

// ─── Slot released on destruction ─────────────────────────────────────────────

TEST(InferSequence, DestructorFreesSlot) {
    KVCacheSlotManager mgr(2);
    {
        InferSequence seq(mgr, {1}, 2);
        EXPECT_EQ(mgr.FreeCount(), 1);
    }
    EXPECT_EQ(mgr.FreeCount(), 2);
}

// ─── Two sequences share a slot manager ──────────────────────────────────────

TEST(InferSequence, TwoSequencesDistinctSlots) {
    KVCacheSlotManager mgr(4);
    InferSequence a(mgr, {1, 2}, 2);
    InferSequence b(mgr, {3, 4}, 2);
    EXPECT_NE(a.SlotID(), b.SlotID());
    EXPECT_EQ(mgr.FreeCount(), 2);
}

// ─── Full prefill→decode→EOS loop (integration with LLMEngine) ───────────────

TEST(InferSequence, PrefillDecodeLoopUntilEOS) {
    SKIP_IF_NO_MODEL();

    LLMEngine engine;
    engine.LoadModel(TEST_MODEL_PATH, /*n_gpu_layers=*/99,
                     /*ctx_size=*/256, /*n_seq_max=*/4, /*n_batch=*/256);

    KVCacheSlotManager mgr(4);
    InferSequence seq(mgr, { engine.BOS() }, engine.EOS());

    SamplerParams sparams;
    sparams.temperature = 0.0f;  // greedy — deterministic

    const int max_tokens = 20;
    int steps = 0;

    while (!seq.IsDone() && steps < max_tokens) {
        // Build input for this step
        SequenceInput inp;
        inp.seq_id      = seq.SlotID();
        inp.tokens      = seq.NextTokens();
        inp.pos         = seq.Position();
        inp.want_logits = true;

        auto results = engine.BatchDecode({ inp });
        ASSERT_EQ(results.size(), 1u);

        // Sample next token
        int32_t next = SampleToken(
            results[0].logits.data(),
            engine.VocabSize(),
            seq.Tokens(),
            sparams
        );

        seq.AppendToken(next);
        steps++;
    }

    // Must have produced at least one token beyond the prompt
    EXPECT_GT(seq.GeneratedCount(), 0);
    // Position must have advanced
    EXPECT_GT(seq.Position(), 1);
}
