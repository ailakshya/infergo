#include "infer_api.h"

#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>

#ifndef TEST_TOKENIZER_PATH
#define TEST_TOKENIZER_PATH "/tmp/bert_tokenizer/tokenizer.json"
#endif

// ─── infer_last_error_string ──────────────────────────────────────────────────

TEST(ApiErrorString, ReturnsCharPointer) {
    const char* err = infer_last_error_string();
    ASSERT_NE(err, nullptr);
}

// ─── infer_tensor_alloc_cpu ───────────────────────────────────────────────────

TEST(ApiAllocCpu, Float32_234) {
    const int shape[] = {2, 3, 4};
    InferTensor t = infer_tensor_alloc_cpu(shape, 3, INFER_DTYPE_FLOAT32);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(infer_tensor_dtype(t), INFER_DTYPE_FLOAT32);
    EXPECT_EQ(infer_tensor_nbytes(t), 96);   // 2*3*4*4
    EXPECT_EQ(infer_tensor_nelements(t), 24);
    infer_tensor_free(t);
}

TEST(ApiAllocCpu, AllDtypes) {
    struct Case { int dtype; int elem_size; };
    const Case cases[] = {
        {INFER_DTYPE_FLOAT32,  4},
        {INFER_DTYPE_FLOAT16,  2},
        {INFER_DTYPE_BFLOAT16, 2},
        {INFER_DTYPE_INT32,    4},
        {INFER_DTYPE_INT64,    8},
        {INFER_DTYPE_UINT8,    1},
        {INFER_DTYPE_BOOL,     1},
    };
    const int shape[] = {4};
    for (const auto& c : cases) {
        InferTensor t = infer_tensor_alloc_cpu(shape, 1, c.dtype);
        ASSERT_NE(t, nullptr) << "dtype=" << c.dtype;
        EXPECT_EQ(infer_tensor_nbytes(t), 4 * c.elem_size) << "dtype=" << c.dtype;
        infer_tensor_free(t);
    }
}

TEST(ApiAllocCpu, NullShapeReturnsNull) {
    InferTensor t = infer_tensor_alloc_cpu(nullptr, 1, INFER_DTYPE_FLOAT32);
    EXPECT_EQ(t, nullptr);
    EXPECT_NE(infer_last_error_string()[0], '\0');
}

TEST(ApiAllocCpu, ZeroNdimReturnsNull) {
    const int shape[] = {4};
    InferTensor t = infer_tensor_alloc_cpu(shape, 0, INFER_DTYPE_FLOAT32);
    EXPECT_EQ(t, nullptr);
}

TEST(ApiAllocCpu, UnknownDtypeReturnsNull) {
    const int shape[] = {4};
    InferTensor t = infer_tensor_alloc_cpu(shape, 1, 999);
    EXPECT_EQ(t, nullptr);
}

// ─── infer_tensor_free ────────────────────────────────────────────────────────

TEST(ApiTensorFree, NullIsNoop) {
    infer_tensor_free(nullptr);  // must not crash
}

// ─── infer_tensor_data_ptr ────────────────────────────────────────────────────

TEST(ApiDataPtr, CpuTensorNotNull) {
    const int shape[] = {8};
    InferTensor t = infer_tensor_alloc_cpu(shape, 1, INFER_DTYPE_FLOAT32);
    ASSERT_NE(t, nullptr);
    EXPECT_NE(infer_tensor_data_ptr(t), nullptr);
    infer_tensor_free(t);
}

TEST(ApiDataPtr, NullTensorReturnsNull) {
    EXPECT_EQ(infer_tensor_data_ptr(nullptr), nullptr);
}

// ─── infer_tensor_shape ───────────────────────────────────────────────────────

TEST(ApiShape, CorrectDims) {
    const int shape[] = {2, 3, 4};
    InferTensor t = infer_tensor_alloc_cpu(shape, 3, INFER_DTYPE_FLOAT32);
    ASSERT_NE(t, nullptr);
    int out[8] = {};
    int ndim = infer_tensor_shape(t, out, 8);
    EXPECT_EQ(ndim, 3);
    EXPECT_EQ(out[0], 2);
    EXPECT_EQ(out[1], 3);
    EXPECT_EQ(out[2], 4);
    infer_tensor_free(t);
}

TEST(ApiShape, NullTensorReturnsZero) {
    int out[4] = {};
    EXPECT_EQ(infer_tensor_shape(nullptr, out, 4), 0);
}

TEST(ApiShape, NullOutBufReturnsZero) {
    const int shape[] = {4};
    InferTensor t = infer_tensor_alloc_cpu(shape, 1, INFER_DTYPE_FLOAT32);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(infer_tensor_shape(t, nullptr, 4), 0);
    infer_tensor_free(t);
}

// ─── infer_tensor_copy_from ───────────────────────────────────────────────────

TEST(ApiCopyFrom, RoundTrip) {
    const int shape[] = {4};
    InferTensor t = infer_tensor_alloc_cpu(shape, 1, INFER_DTYPE_FLOAT32);
    ASSERT_NE(t, nullptr);

    const float src[] = {1.0f, 2.0f, 3.0f, 4.0f};
    InferError err = infer_tensor_copy_from(t, src, static_cast<int>(sizeof(src)));
    EXPECT_EQ(err, INFER_OK);

    const float* data = static_cast<const float*>(infer_tensor_data_ptr(t));
    ASSERT_NE(data, nullptr);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(data[i], src[i]);
    }
    infer_tensor_free(t);
}

TEST(ApiCopyFrom, NullTensorReturnsErrNull) {
    const float src[] = {1.0f};
    InferError err = infer_tensor_copy_from(nullptr, src, 4);
    EXPECT_EQ(err, INFER_ERR_NULL);
}

TEST(ApiCopyFrom, NullSrcReturnsErrNull) {
    const int shape[] = {4};
    InferTensor t = infer_tensor_alloc_cpu(shape, 1, INFER_DTYPE_FLOAT32);
    ASSERT_NE(t, nullptr);
    InferError err = infer_tensor_copy_from(t, nullptr, 16);
    EXPECT_EQ(err, INFER_ERR_NULL);
    infer_tensor_free(t);
}

TEST(ApiCopyFrom, WrongNbytesReturnsErrInvalid) {
    const int shape[] = {4};
    InferTensor t = infer_tensor_alloc_cpu(shape, 1, INFER_DTYPE_FLOAT32);
    ASSERT_NE(t, nullptr);
    const float src[] = {1.0f, 2.0f};
    InferError err = infer_tensor_copy_from(t, src, 8);  // tensor needs 16 bytes
    EXPECT_EQ(err, INFER_ERR_INVALID);
    infer_tensor_free(t);
}

// ─── CUDA tests (compiled only when CUDA is available) ───────────────────────

#ifdef INFER_CUDA_AVAILABLE

TEST(ApiAllocCuda, Float32_234) {
    const int shape[] = {2, 3, 4};
    InferTensor t = infer_tensor_alloc_cuda(shape, 3, INFER_DTYPE_FLOAT32, 0);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(infer_tensor_dtype(t), INFER_DTYPE_FLOAT32);
    EXPECT_EQ(infer_tensor_nbytes(t), 96);
    EXPECT_EQ(infer_tensor_nelements(t), 24);
    infer_tensor_free(t);
}

TEST(ApiCudaTransfer, RoundTrip) {
    const int shape[] = {4};
    InferTensor t = infer_tensor_alloc_cpu(shape, 1, INFER_DTYPE_FLOAT32);
    ASSERT_NE(t, nullptr);

    const float src[] = {10.0f, 20.0f, 30.0f, 40.0f};
    ASSERT_EQ(infer_tensor_copy_from(t, src, 16), INFER_OK);

    EXPECT_EQ(infer_tensor_to_device(t, 0), INFER_OK);
    EXPECT_EQ(infer_tensor_to_host(t), INFER_OK);

    const float* data = static_cast<const float*>(infer_tensor_data_ptr(t));
    ASSERT_NE(data, nullptr);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(data[i], src[i]);
    }
    infer_tensor_free(t);
}

TEST(ApiCudaTransfer, NullToDeviceReturnsErrNull) {
    EXPECT_EQ(infer_tensor_to_device(nullptr, 0), INFER_ERR_NULL);
}

TEST(ApiCudaTransfer, NullToHostReturnsErrNull) {
    EXPECT_EQ(infer_tensor_to_host(nullptr), INFER_ERR_NULL);
}

#endif // INFER_CUDA_AVAILABLE

// ─── Session API (T-15) ───────────────────────────────────────────────────────

#ifndef TEST_MODEL_PATH
#define TEST_MODEL_PATH "/tmp/scale.onnx"
#endif

TEST(ApiSession, CreateCpuNotNull) {
    InferSession s = infer_session_create("cpu", 0);
    ASSERT_NE(s, nullptr);
    infer_session_destroy(s);
}

TEST(ApiSession, CreateNullProviderDefaultsCpu) {
    InferSession s = infer_session_create(nullptr, 0);
    ASSERT_NE(s, nullptr);
    infer_session_destroy(s);
}

TEST(ApiSession, DestroyNullIsNoop) {
    infer_session_destroy(nullptr); // must not crash
}

TEST(ApiSession, LoadModel) {
    InferSession s = infer_session_create("cpu", 0);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(infer_session_load(s, TEST_MODEL_PATH), INFER_OK);
    EXPECT_EQ(infer_session_num_inputs(s),  1);
    EXPECT_EQ(infer_session_num_outputs(s), 1);
    infer_session_destroy(s);
}

TEST(ApiSession, LoadBadPathReturnsErrLoad) {
    InferSession s = infer_session_create("cpu", 0);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(infer_session_load(s, "/no/such/model.onnx"), INFER_ERR_LOAD);
    EXPECT_NE(infer_last_error_string()[0], '\0');
    infer_session_destroy(s);
}

TEST(ApiSession, LoadNullSessionReturnsErrNull) {
    EXPECT_EQ(infer_session_load(nullptr, TEST_MODEL_PATH), INFER_ERR_NULL);
}

TEST(ApiSession, InputOutputNames) {
    InferSession s = infer_session_create("cpu", 0);
    ASSERT_NE(s, nullptr);
    ASSERT_EQ(infer_session_load(s, TEST_MODEL_PATH), INFER_OK);

    char buf[64];
    EXPECT_EQ(infer_session_input_name(s, 0, buf, sizeof(buf)), INFER_OK);
    EXPECT_STREQ(buf, "x");

    EXPECT_EQ(infer_session_output_name(s, 0, buf, sizeof(buf)), INFER_OK);
    EXPECT_STREQ(buf, "y");

    infer_session_destroy(s);
}

TEST(ApiSession, RunScaleModel) {
    InferSession s = infer_session_create("cpu", 0);
    ASSERT_NE(s, nullptr);
    ASSERT_EQ(infer_session_load(s, TEST_MODEL_PATH), INFER_OK);

    // Input [1,4] float32 = {1,2,3,4}
    const int shape[] = {1, 4};
    InferTensor in = infer_tensor_alloc_cpu(shape, 2, INFER_DTYPE_FLOAT32);
    ASSERT_NE(in, nullptr);
    const float src[] = {1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT_EQ(infer_tensor_copy_from(in, src, sizeof(src)), INFER_OK);

    InferTensor out = nullptr;
    EXPECT_EQ(infer_session_run(s, &in, 1, &out, 1), INFER_OK);
    ASSERT_NE(out, nullptr);

    // y = x * 2: expect [2,4,6,8]
    EXPECT_EQ(infer_tensor_nbytes(out), 16);
    const float* data = static_cast<const float*>(infer_tensor_data_ptr(out));
    EXPECT_FLOAT_EQ(data[0], 2.0f);
    EXPECT_FLOAT_EQ(data[1], 4.0f);
    EXPECT_FLOAT_EQ(data[2], 6.0f);
    EXPECT_FLOAT_EQ(data[3], 8.0f);

    infer_tensor_free(out);
    infer_tensor_free(in);
    infer_session_destroy(s);
}

TEST(ApiSession, RunNullSessionReturnsErrNull) {
    InferTensor out = nullptr;
    EXPECT_EQ(infer_session_run(nullptr, nullptr, 0, &out, 1), INFER_ERR_NULL);
}

TEST(ApiSession, RunWrongInputCountReturnsErr) {
    InferSession s = infer_session_create("cpu", 0);
    ASSERT_NE(s, nullptr);
    ASSERT_EQ(infer_session_load(s, TEST_MODEL_PATH), INFER_OK);

    // Pass 0 inputs instead of 1
    InferTensor out = nullptr;
    EXPECT_NE(infer_session_run(s, nullptr, 0, &out, 1), INFER_OK);
    infer_session_destroy(s);
}

// ─── Tokenizer API ────────────────────────────────────────────────────────────

TEST(ApiTokenizer, LoadValid) {
    InferTokenizer tok = infer_tokenizer_load(TEST_TOKENIZER_PATH);
    ASSERT_NE(tok, nullptr);
    infer_tokenizer_destroy(tok);
}

TEST(ApiTokenizer, LoadNullPathReturnsNull) {
    EXPECT_EQ(infer_tokenizer_load(nullptr), nullptr);
}

TEST(ApiTokenizer, LoadBadPathReturnsNull) {
    EXPECT_EQ(infer_tokenizer_load("/no/such/tokenizer.json"), nullptr);
}

TEST(ApiTokenizer, VocabSizePositive) {
    InferTokenizer tok = infer_tokenizer_load(TEST_TOKENIZER_PATH);
    ASSERT_NE(tok, nullptr);
    EXPECT_GT(infer_tokenizer_vocab_size(tok), 0);
    infer_tokenizer_destroy(tok);
}

TEST(ApiTokenizer, VocabSizeNullReturnsZero) {
    EXPECT_EQ(infer_tokenizer_vocab_size(nullptr), 0);
}

TEST(ApiTokenizer, EncodeHelloWorld) {
    InferTokenizer tok = infer_tokenizer_load(TEST_TOKENIZER_PATH);
    ASSERT_NE(tok, nullptr);

    int ids[512] = {};
    int mask[512] = {};
    const int n = infer_tokenizer_encode(tok, "Hello world", 0, ids, mask, 512);
    EXPECT_GT(n, 0);
    for (int i = 0; i < n; ++i) EXPECT_EQ(mask[i], 1);

    infer_tokenizer_destroy(tok);
}

TEST(ApiTokenizer, EncodeNullArgsReturnsMinus1) {
    EXPECT_EQ(infer_tokenizer_encode(nullptr, "text", 0, nullptr, nullptr, 512), -1);
}

TEST(ApiTokenizer, DecodeRoundTrip) {
    InferTokenizer tok = infer_tokenizer_load(TEST_TOKENIZER_PATH);
    ASSERT_NE(tok, nullptr);

    int ids[512] = {};
    int mask[512] = {};
    const int n = infer_tokenizer_encode(tok, "hello world", 0, ids, mask, 512);
    ASSERT_GT(n, 0);

    char buf[1024] = {};
    EXPECT_EQ(infer_tokenizer_decode(tok, ids, n, 1, buf, static_cast<int>(sizeof(buf))), 0);
    EXPECT_STREQ(buf, "hello world");

    infer_tokenizer_destroy(tok);
}

TEST(ApiTokenizer, DecodeEmptyIdsReturnsEmpty) {
    InferTokenizer tok = infer_tokenizer_load(TEST_TOKENIZER_PATH);
    ASSERT_NE(tok, nullptr);

    char buf[64] = {};
    EXPECT_EQ(infer_tokenizer_decode(tok, nullptr, 0, 1, buf, static_cast<int>(sizeof(buf))), 0);
    EXPECT_EQ(buf[0], '\0');

    infer_tokenizer_destroy(tok);
}

TEST(ApiTokenizer, DecodeToken) {
    InferTokenizer tok = infer_tokenizer_load(TEST_TOKENIZER_PATH);
    ASSERT_NE(tok, nullptr);

    int ids[8] = {};
    int mask[8] = {};
    const int n = infer_tokenizer_encode(tok, "hello", 0, ids, mask, 8);
    ASSERT_GT(n, 0);

    char buf[64] = {};
    EXPECT_EQ(infer_tokenizer_decode_token(tok, ids[0], buf, static_cast<int>(sizeof(buf))), 0);
    EXPECT_GT(static_cast<int>(strlen(buf)), 0);

    infer_tokenizer_destroy(tok);
}

TEST(ApiTokenizer, DestroyNullIsNoop) {
    EXPECT_NO_FATAL_FAILURE(infer_tokenizer_destroy(nullptr));
}

// ─── Preprocess API — decode_image (T-32) ────────────────────────────────────

#ifdef INFER_PREPROCESS_AVAILABLE

// Minimal 1×1 white JPEG (same bytes as preprocess unit tests)
static const uint8_t kWhiteJpeg[] = {
    0xFF,0xD8,0xFF,0xE0,0x00,0x10,0x4A,0x46,0x49,0x46,0x00,0x01,0x01,0x00,0x00,0x01,
    0x00,0x01,0x00,0x00,0xFF,0xDB,0x00,0x43,0x00,0x08,0x06,0x06,0x07,0x06,0x05,0x08,
    0x07,0x07,0x07,0x09,0x09,0x08,0x0A,0x0C,0x14,0x0D,0x0C,0x0B,0x0B,0x0C,0x19,0x12,
    0x13,0x0F,0x14,0x1D,0x1A,0x1F,0x1E,0x1D,0x1A,0x1C,0x1C,0x20,0x24,0x2E,0x27,0x20,
    0x22,0x2C,0x23,0x1C,0x1C,0x28,0x37,0x29,0x2C,0x30,0x31,0x34,0x34,0x34,0x1F,0x27,
    0x39,0x3D,0x38,0x32,0x3C,0x2E,0x33,0x34,0x32,0xFF,0xC0,0x00,0x0B,0x08,0x00,0x01,
    0x00,0x01,0x01,0x01,0x11,0x00,0xFF,0xC4,0x00,0x1F,0x00,0x00,0x01,0x05,0x01,0x01,
    0x01,0x01,0x01,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x02,0x03,0x04,
    0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0xFF,0xC4,0x00,0xB5,0x10,0x00,0x02,0x01,0x03,
    0x03,0x02,0x04,0x03,0x05,0x05,0x04,0x04,0x00,0x00,0x01,0x7D,0x01,0x02,0x03,0x00,
    0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,0x51,0x61,0x07,0x22,0x71,0x14,0x32,
    0x81,0x91,0xA1,0x08,0x23,0x42,0xB1,0xC1,0x15,0x52,0xD1,0xF0,0x24,0x33,0x62,0x72,
    0x82,0x09,0x0A,0x16,0x17,0x18,0x19,0x1A,0x25,0x26,0x27,0x28,0x29,0x2A,0x34,0x35,
    0x36,0x37,0x38,0x39,0x3A,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4A,0x53,0x54,0x55,
    0x56,0x57,0x58,0x59,0x5A,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6A,0x73,0x74,0x75,
    0x76,0x77,0x78,0x79,0x7A,0x83,0x84,0x85,0x86,0x87,0x88,0x89,0x8A,0x92,0x93,0x94,
    0x95,0x96,0x97,0x98,0x99,0x9A,0xA2,0xA3,0xA4,0xA5,0xA6,0xA7,0xA8,0xA9,0xAA,0xB2,
    0xB3,0xB4,0xB5,0xB6,0xB7,0xB8,0xB9,0xBA,0xC2,0xC3,0xC4,0xC5,0xC6,0xC7,0xC8,0xC9,
    0xCA,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7,0xD8,0xD9,0xDA,0xE1,0xE2,0xE3,0xE4,0xE5,0xE6,
    0xE7,0xE8,0xE9,0xEA,0xF1,0xF2,0xF3,0xF4,0xF5,0xF6,0xF7,0xF8,0xF9,0xFA,0xFF,0xDA,
    0x00,0x08,0x01,0x01,0x00,0x00,0x3F,0x00,0xFB,0xD5,0xFF,0xD9
};

TEST(ApiPreprocessDecodeImage, NullDataReturnsNull) {
    InferTensor t = infer_preprocess_decode_image(nullptr, 100);
    EXPECT_EQ(t, nullptr);
    EXPECT_NE(infer_last_error_string()[0], '\0');
}

TEST(ApiPreprocessDecodeImage, ZeroBytesReturnsNull) {
    const uint8_t dummy = 0;
    InferTensor t = infer_preprocess_decode_image(&dummy, 0);
    EXPECT_EQ(t, nullptr);
}

TEST(ApiPreprocessDecodeImage, InvalidDataReturnsNull) {
    const uint8_t bad[] = {0x00, 0x01, 0x02, 0x03};
    InferTensor t = infer_preprocess_decode_image(bad, static_cast<int>(sizeof(bad)));
    EXPECT_EQ(t, nullptr);
}

TEST(ApiPreprocessDecodeImage, ValidJpegShape) {
    InferTensor t = infer_preprocess_decode_image(
        kWhiteJpeg, static_cast<int>(sizeof(kWhiteJpeg)));
    ASSERT_NE(t, nullptr);

    // Shape must be [H, W, 3]
    int shape[8] = {};
    const int ndim = infer_tensor_shape(t, shape, 8);
    EXPECT_EQ(ndim, 3);
    EXPECT_GT(shape[0], 0);   // H
    EXPECT_GT(shape[1], 0);   // W
    EXPECT_EQ(shape[2], 3);   // channels

    EXPECT_EQ(infer_tensor_dtype(t), INFER_DTYPE_FLOAT32);
    EXPECT_GT(infer_tensor_nbytes(t), 0);
    EXPECT_NE(infer_tensor_data_ptr(t), nullptr);

    infer_tensor_free(t);
}

TEST(ApiPreprocessDecodeImage, FreeNullIsNoop) {
    infer_tensor_free(nullptr);  // must not crash
}

#endif // INFER_PREPROCESS_AVAILABLE

// ─── LLM API (T-28) ──────────────────────────────────────────────────────────

#ifdef INFER_LLM_AVAILABLE

#ifndef TEST_LLM_MODEL_PATH
#define TEST_LLM_MODEL_PATH "/tmp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
#endif

static bool llm_model_available() {
    FILE* f = fopen(TEST_LLM_MODEL_PATH, "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

#define SKIP_IF_NO_LLM() \
    if (!llm_model_available()) { GTEST_SKIP() << "LLM model not found at " TEST_LLM_MODEL_PATH; }

TEST(ApiLLM, CreateDestroyNull) {
    infer_llm_destroy(nullptr);  // must not crash
}

TEST(ApiLLM, CreateValid) {
    SKIP_IF_NO_LLM();
    InferLLM llm = infer_llm_create(TEST_LLM_MODEL_PATH, 99, 512, 4, 256);
    ASSERT_NE(llm, nullptr);
    EXPECT_GT(infer_llm_vocab_size(llm), 0);
    EXPECT_GE(infer_llm_bos(llm), 0);
    EXPECT_GE(infer_llm_eos(llm), 0);
    infer_llm_destroy(llm);
}

TEST(ApiLLM, CreateBadPathReturnsNull) {
    InferLLM llm = infer_llm_create("/no/such/model.gguf", 0, 512, 4, 256);
    EXPECT_EQ(llm, nullptr);
    EXPECT_NE(infer_last_error_string()[0], '\0');
}

TEST(ApiLLM, SeqCreateDestroyNull) {
    infer_seq_destroy(nullptr);  // must not crash
}

TEST(ApiLLM, SeqCreateNullLLMReturnsNull) {
    const int tokens[] = {1, 2, 3};
    InferSeq seq = infer_seq_create(nullptr, tokens, 3);
    EXPECT_EQ(seq, nullptr);
}

// Core acceptance criterion: 4 sequences batched together, all get valid logits
TEST(ApiLLM, FourSequencesBatchedGetValidLogits) {
    SKIP_IF_NO_LLM();

    InferLLM llm = infer_llm_create(TEST_LLM_MODEL_PATH, 99, 1024, 4, 512);
    ASSERT_NE(llm, nullptr);

    const int vocab_size = infer_llm_vocab_size(llm);
    ASSERT_GT(vocab_size, 0);

    const int bos = infer_llm_bos(llm);

    // Create 4 sequences with different prompt lengths
    const int prompt0[] = {bos};
    const int prompt1[] = {bos, 100};
    const int prompt2[] = {bos, 200, 300};
    const int prompt3[] = {bos, 400, 500, 600};

    InferSeq seqs[4];
    seqs[0] = infer_seq_create(llm, prompt0, 1);
    seqs[1] = infer_seq_create(llm, prompt1, 2);
    seqs[2] = infer_seq_create(llm, prompt2, 3);
    seqs[3] = infer_seq_create(llm, prompt3, 4);

    for (int i = 0; i < 4; ++i) {
        ASSERT_NE(seqs[i], nullptr) << "seq[" << i << "] is null";
    }

    // Batch decode: all 4 sequences in one call
    InferError err = infer_llm_batch_decode(llm, seqs, 4);
    EXPECT_EQ(err, INFER_OK);

    // All 4 sequences must produce valid logits
    std::vector<float> logits(vocab_size);
    for (int i = 0; i < 4; ++i) {
        InferError lerr = infer_seq_get_logits(seqs[i], logits.data(), vocab_size);
        EXPECT_EQ(lerr, INFER_OK) << "seq[" << i << "] get_logits failed";

        // At least one logit must be non-zero (model produced real output)
        float sum = 0.0f;
        for (int j = 0; j < vocab_size; ++j) sum += std::abs(logits[j]);
        EXPECT_GT(sum, 0.0f) << "seq[" << i << "] logits are all zero";
    }

    // Slot IDs must be distinct
    EXPECT_NE(infer_seq_slot_id(seqs[0]), infer_seq_slot_id(seqs[1]));
    EXPECT_NE(infer_seq_slot_id(seqs[0]), infer_seq_slot_id(seqs[2]));
    EXPECT_NE(infer_seq_slot_id(seqs[1]), infer_seq_slot_id(seqs[3]));

    for (int i = 0; i < 4; ++i) infer_seq_destroy(seqs[i]);
    infer_llm_destroy(llm);
}

TEST(ApiLLM, SeqIsDoneAfterEOS) {
    SKIP_IF_NO_LLM();

    InferLLM llm = infer_llm_create(TEST_LLM_MODEL_PATH, 99, 512, 4, 256);
    ASSERT_NE(llm, nullptr);

    const int bos = infer_llm_bos(llm);
    const int eos = infer_llm_eos(llm);
    const int tokens[] = {bos};
    InferSeq seq = infer_seq_create(llm, tokens, 1);
    ASSERT_NE(seq, nullptr);

    EXPECT_EQ(infer_seq_is_done(seq), 0);
    infer_seq_append_token(seq, eos);
    EXPECT_EQ(infer_seq_is_done(seq), 1);

    infer_seq_destroy(seq);
    infer_llm_destroy(llm);
}

TEST(ApiLLM, IsEOGWorks) {
    SKIP_IF_NO_LLM();

    InferLLM llm = infer_llm_create(TEST_LLM_MODEL_PATH, 99, 512, 4, 256);
    ASSERT_NE(llm, nullptr);

    const int eos = infer_llm_eos(llm);
    EXPECT_EQ(infer_llm_is_eog(llm, eos), 1);
    EXPECT_EQ(infer_llm_is_eog(llm, 0), 0);

    infer_llm_destroy(llm);
}

#endif // INFER_LLM_AVAILABLE
