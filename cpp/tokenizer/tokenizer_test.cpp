#include "tokenizer.hpp"

#include <gtest/gtest.h>

#ifndef TEST_TOKENIZER_PATH
#define TEST_TOKENIZER_PATH "/tmp/bert_tokenizer/tokenizer.json"
#endif

using namespace infergo;

// ─── Construction ─────────────────────────────────────────────────────────────

TEST(TokenizerWrapper, ConstructValid) {
    EXPECT_NO_THROW(TokenizerWrapper t(TEST_TOKENIZER_PATH));
}

TEST(TokenizerWrapper, ConstructBadPathThrows) {
    EXPECT_THROW(TokenizerWrapper t("/no/such/tokenizer.json"), std::runtime_error);
}

TEST(TokenizerWrapper, MoveConstruct) {
    TokenizerWrapper a(TEST_TOKENIZER_PATH);
    EXPECT_NO_THROW(TokenizerWrapper b(std::move(a)));
}

TEST(TokenizerWrapper, MoveAssign) {
    TokenizerWrapper a(TEST_TOKENIZER_PATH);
    TokenizerWrapper b(TEST_TOKENIZER_PATH);
    EXPECT_NO_THROW(b = std::move(a));
}

// ─── vocab_size ───────────────────────────────────────────────────────────────

TEST(TokenizerWrapper, VocabSizePositive) {
    TokenizerWrapper t(TEST_TOKENIZER_PATH);
    EXPECT_GT(t.vocab_size(), 0);
}

// ─── encode ───────────────────────────────────────────────────────────────────

TEST(TokenizerWrapper, EncodeHelloWorld) {
    TokenizerWrapper t(TEST_TOKENIZER_PATH);
    auto enc = t.encode("Hello world", /*add_special_tokens=*/false);
    ASSERT_GT(static_cast<int>(enc.ids.size()), 0);
    EXPECT_EQ(enc.ids.size(), enc.attention_mask.size());
    for (int m : enc.attention_mask) EXPECT_EQ(m, 1);
}

TEST(TokenizerWrapper, EncodeWithSpecialTokens) {
    TokenizerWrapper t(TEST_TOKENIZER_PATH);
    auto without = t.encode("Hello", false);
    auto with    = t.encode("Hello", true);
    // With special tokens should be longer
    EXPECT_GE(with.ids.size(), without.ids.size());
}

TEST(TokenizerWrapper, EncodeTruncatesAtMaxTokens) {
    TokenizerWrapper t(TEST_TOKENIZER_PATH);
    auto full  = t.encode("the quick brown fox jumps over the lazy dog", false, 512);
    auto trunc = t.encode("the quick brown fox jumps over the lazy dog", false, 3);
    EXPECT_EQ(static_cast<int>(trunc.ids.size()), 3);
    for (int i = 0; i < 3; ++i) EXPECT_EQ(trunc.ids[i], full.ids[i]);
}

TEST(TokenizerWrapper, EncodeEmptyString) {
    TokenizerWrapper t(TEST_TOKENIZER_PATH);
    EXPECT_NO_THROW(auto enc = t.encode("", false));
}

// ─── decode ───────────────────────────────────────────────────────────────────

TEST(TokenizerWrapper, DecodeRoundTrip) {
    TokenizerWrapper t(TEST_TOKENIZER_PATH);
    // The key acceptance criterion for T-19
    auto enc = t.encode("hello world", false);
    ASSERT_GT(static_cast<int>(enc.ids.size()), 0);
    std::string decoded = t.decode(enc.ids, /*skip_special_tokens=*/true);
    EXPECT_EQ(decoded, "hello world");
}

TEST(TokenizerWrapper, DecodeEmptyIds) {
    TokenizerWrapper t(TEST_TOKENIZER_PATH);
    std::string result;
    EXPECT_NO_THROW(result = t.decode({}, true));
    EXPECT_TRUE(result.empty());
}

TEST(TokenizerWrapper, DecodeMultipleSentences) {
    TokenizerWrapper t(TEST_TOKENIZER_PATH);
    const std::vector<std::string> sentences = {
        "hello world",
        "the quick brown fox",
        "inference at the speed of go",
    };
    for (const auto& s : sentences) {
        auto enc = t.encode(s, false);
        ASSERT_GT(static_cast<int>(enc.ids.size()), 0) << "sentence: " << s;
        std::string decoded = t.decode(enc.ids, true);
        EXPECT_EQ(decoded, s) << "round-trip failed for: " << s;
    }
}

// ─── decode_token ─────────────────────────────────────────────────────────────

TEST(TokenizerWrapper, DecodeTokenReturnsNonEmpty) {
    TokenizerWrapper t(TEST_TOKENIZER_PATH);
    // Encode "hello" and decode each token individually
    auto enc = t.encode("hello", false);
    ASSERT_GT(static_cast<int>(enc.ids.size()), 0);
    for (int32_t id : enc.ids) {
        std::string piece;
        EXPECT_NO_THROW(piece = t.decode_token(id));
        // Each piece is a non-empty string for normal tokens
        EXPECT_FALSE(piece.empty()) << "empty piece for id=" << id;
    }
}
