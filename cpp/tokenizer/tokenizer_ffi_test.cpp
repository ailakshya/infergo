#include "tokenizers_ffi.h"

#include <cstring>
#include <string>
#include <gtest/gtest.h>

// Path to a tokenizer.json — bert-base-uncased downloaded during T-18 setup
#ifndef TEST_TOKENIZER_PATH
#define TEST_TOKENIZER_PATH "/tmp/bert_tokenizer/tokenizer.json"
#endif

// ─── tokenizer_load / tokenizer_free ─────────────────────────────────────────

TEST(TokenizerFFI, LoadValidFile) {
    char err[256] = {};
    TokenizerHandle* h = tokenizer_load(TEST_TOKENIZER_PATH, err, sizeof(err));
    ASSERT_NE(h, nullptr) << "load failed: " << err;
    tokenizer_free(h);
}

TEST(TokenizerFFI, LoadBadPathReturnsNull) {
    char err[256] = {};
    TokenizerHandle* h = tokenizer_load("/no/such/tokenizer.json", err, sizeof(err));
    EXPECT_EQ(h, nullptr);
    EXPECT_NE(err[0], '\0'); // error message populated
}

TEST(TokenizerFFI, FreeNullIsNoop) {
    tokenizer_free(nullptr); // must not crash
}

// ─── tokenizer_vocab_size ─────────────────────────────────────────────────────

TEST(TokenizerFFI, VocabSize) {
    char err[256] = {};
    TokenizerHandle* h = tokenizer_load(TEST_TOKENIZER_PATH, err, sizeof(err));
    ASSERT_NE(h, nullptr);
    int vsz = tokenizer_vocab_size(h);
    EXPECT_GT(vsz, 0);
    tokenizer_free(h);
}

TEST(TokenizerFFI, VocabSizeNullReturnsZero) {
    EXPECT_EQ(tokenizer_vocab_size(nullptr), 0);
}

// ─── tokenizer_encode ─────────────────────────────────────────────────────────

TEST(TokenizerFFI, EncodeHelloWorld) {
    char err[256] = {};
    TokenizerHandle* h = tokenizer_load(TEST_TOKENIZER_PATH, err, sizeof(err));
    ASSERT_NE(h, nullptr);

    int ids[64]  = {};
    int mask[64] = {};
    int n = tokenizer_encode(h, "Hello world", 1, ids, mask, 64, err, sizeof(err));
    ASSERT_GT(n, 0) << "encode failed: " << err;

    // All returned IDs must be positive
    for (int i = 0; i < n; ++i) {
        EXPECT_GT(ids[i], 0) << "ids[" << i << "] = " << ids[i];
        EXPECT_EQ(mask[i], 1);
    }
    tokenizer_free(h);
}

TEST(TokenizerFFI, EncodeNullHandleReturnsMinus1) {
    int ids[8] = {};
    EXPECT_EQ(tokenizer_encode(nullptr, "hello", 0, ids, nullptr, 8, nullptr, 0), -1);
}

TEST(TokenizerFFI, EncodeTruncatesToMaxTokens) {
    char err[256] = {};
    TokenizerHandle* h = tokenizer_load(TEST_TOKENIZER_PATH, err, sizeof(err));
    ASSERT_NE(h, nullptr);

    const char* long_text = "the quick brown fox jumps over the lazy dog";
    int ids_full[64] = {};
    int n_full = tokenizer_encode(h, long_text, 0, ids_full, nullptr, 64, err, sizeof(err));
    ASSERT_GT(n_full, 3);

    int ids_trunc[3] = {};
    int n_trunc = tokenizer_encode(h, long_text, 0, ids_trunc, nullptr, 3, err, sizeof(err));
    EXPECT_EQ(n_trunc, 3);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(ids_trunc[i], ids_full[i]);
    }
    tokenizer_free(h);
}

// ─── tokenizer_decode ─────────────────────────────────────────────────────────

TEST(TokenizerFFI, DecodeRoundTrip) {
    char err[256] = {};
    TokenizerHandle* h = tokenizer_load(TEST_TOKENIZER_PATH, err, sizeof(err));
    ASSERT_NE(h, nullptr);

    int ids[64]  = {};
    int n = tokenizer_encode(h, "hello world", 0, ids, nullptr, 64, err, sizeof(err));
    ASSERT_GT(n, 0);

    char out[256] = {};
    int rc = tokenizer_decode(h, ids, n, 1, out, sizeof(out), err, sizeof(err));
    ASSERT_EQ(rc, 0) << "decode failed: " << err;
    EXPECT_STREQ(out, "hello world");

    tokenizer_free(h);
}

TEST(TokenizerFFI, DecodeNullHandleReturnsMinus1) {
    int ids[] = {1, 2, 3};
    char out[64] = {};
    EXPECT_EQ(tokenizer_decode(nullptr, ids, 3, 1, out, sizeof(out), nullptr, 0), -1);
}
