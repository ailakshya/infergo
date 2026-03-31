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
