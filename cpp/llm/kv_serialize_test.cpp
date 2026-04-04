// kv_serialize_test.cpp
// Unit tests for LLMEngine::SerializeKV / DeserializeKV (OPT-26).

#include "llm_engine.hpp"

#include <gtest/gtest.h>
#include <string>
#include <vector>

#ifndef TEST_MODEL_PATH
#define TEST_MODEL_PATH "/tmp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
#endif

using namespace infergo;

// ─── Helpers ─────────────────────────────────────────────────────────────────

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

// ─── T1: SerializeKV returns non-empty bytes for a valid seq after prefill ───

TEST(KVSerialize, T1_SerializeValidSeqNonEmpty) {
    SKIP_IF_NO_MODEL();

    LLMEngine e;
    e.LoadModel(TEST_MODEL_PATH, 0, 512, 4, 512);

    // Tokenize a small prompt and run one decode step (prefill).
    const auto tokens = e.Tokenize("Hello world", /*add_bos=*/true);
    ASSERT_FALSE(tokens.empty());

    // seq_id 0: add the prompt tokens at position 0.
    SequenceInput inp;
    inp.seq_id = 0;
    inp.tokens = tokens;
    inp.pos = 0;
    inp.want_logits = true;
    ASSERT_NO_THROW(e.BatchDecode({inp}));

    // Serialize the KV cache for seq 0.
    const auto kv = e.SerializeKV(0);
    EXPECT_GT(kv.size(), 0u) << "SerializeKV should return non-empty bytes after prefill";
}

// ─── T2: DeserializeKV restores state (byte count matches) ──────────────────

TEST(KVSerialize, T2_DeserializeRestoresState) {
    SKIP_IF_NO_MODEL();

    LLMEngine e;
    e.LoadModel(TEST_MODEL_PATH, 0, 512, 4, 512);

    const auto tokens = e.Tokenize("Hello world", /*add_bos=*/true);
    ASSERT_FALSE(tokens.empty());

    SequenceInput inp;
    inp.seq_id = 0;
    inp.tokens = tokens;
    inp.pos = 0;
    inp.want_logits = true;
    ASSERT_NO_THROW(e.BatchDecode({inp}));

    const auto kv_original = e.SerializeKV(0);
    ASSERT_GT(kv_original.size(), 0u);

    // Deserialize into seq slot 1.
    const bool ok = e.DeserializeKV(1, kv_original.data(), kv_original.size());
    EXPECT_TRUE(ok) << "DeserializeKV should succeed with valid bytes";

    // Re-serialize from slot 1 and check the byte count matches.
    const auto kv_restored = e.SerializeKV(1);
    EXPECT_EQ(kv_original.size(), kv_restored.size())
        << "Byte count should be identical after round-trip";
}

// ─── T3: SerializeKV on an empty/unused seq returns empty ────────────────────

TEST(KVSerialize, T3_SerializeEmptySeqReturnsEmpty) {
    SKIP_IF_NO_MODEL();

    LLMEngine e;
    e.LoadModel(TEST_MODEL_PATH, 0, 512, 4, 512);

    // Seq 0 has never been used — no KV data should exist.
    const auto kv = e.SerializeKV(0);
    EXPECT_EQ(kv.size(), 0u)
        << "SerializeKV on an unused sequence should return empty bytes";
}

// ─── T4: SerializeKV / DeserializeKV on unloaded engine ─────────────────────

TEST(KVSerialize, T4_UnloadedEngineReturnsEmpty) {
    LLMEngine e; // no LoadModel call

    const auto kv = e.SerializeKV(0);
    EXPECT_TRUE(kv.empty()) << "SerializeKV without a loaded model should return empty";

    const uint8_t dummy[4] = {0, 1, 2, 3};
    const bool ok = e.DeserializeKV(0, dummy, sizeof(dummy));
    EXPECT_FALSE(ok) << "DeserializeKV without a loaded model should return false";
}
