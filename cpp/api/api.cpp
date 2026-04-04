#include "infer_api.h"
#include "../tensor/tensor.hpp"
#include "../onnx/onnx_session.hpp"
#include "../tokenizer/tokenizer.hpp"
#include "../llm/kv_cache.hpp"
#include "../llm/kv_paged.hpp"
#include "../llm/llm_engine.hpp"
#include "../llm/infer_sequence.hpp"
#include "../../vendor/llama.cpp/include/llama.h"
#ifdef INFER_PREPROCESS_AVAILABLE
#include "../preprocess/preprocess.hpp"
#endif
#include "../postprocess/postprocess.hpp"

#include <cstring>
#include <exception>
#include <vector>

// ─── Error string ─────────────────────────────────────────────────────────────

const char* infer_last_error_string(void) {
    return infergo::get_last_error();
}

// ─── Tensor API ───────────────────────────────────────────────────────────────

InferTensor infer_tensor_alloc_cpu(const int* shape, int ndim, int dtype) {
    try {
        return static_cast<InferTensor>(
            infergo::tensor_alloc_cpu(shape, ndim, dtype)
        );
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return nullptr;
    } catch (...) {
        infergo::set_last_error("infer_tensor_alloc_cpu: unknown exception");
        return nullptr;
    }
}

InferTensor infer_tensor_alloc_cuda(const int* shape, int ndim, int dtype, int device_id) {
#ifdef INFER_CUDA_AVAILABLE
    try {
        return static_cast<InferTensor>(
            infergo::tensor_alloc_cuda(shape, ndim, dtype, device_id)
        );
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return nullptr;
    } catch (...) {
        infergo::set_last_error("infer_tensor_alloc_cuda: unknown exception");
        return nullptr;
    }
#else
    (void)shape; (void)ndim; (void)dtype; (void)device_id;
    infergo::set_last_error("infer_tensor_alloc_cuda: CUDA not available in this build");
    return nullptr;
#endif
}

void infer_tensor_free(InferTensor t) {
    try {
        infergo::tensor_free(static_cast<infergo::Tensor*>(t));
    } catch (...) {
        // noexcept by contract — swallow silently
    }
}

void* infer_tensor_data_ptr(InferTensor t) {
    if (t == nullptr) { return nullptr; }
    return static_cast<infergo::Tensor*>(t)->data;
}

int infer_tensor_nbytes(InferTensor t) {
    try {
        return infergo::tensor_get_nbytes(static_cast<const infergo::Tensor*>(t));
    } catch (...) {
        return 0;
    }
}

int infer_tensor_nelements(InferTensor t) {
    try {
        return infergo::tensor_get_nelements(static_cast<const infergo::Tensor*>(t));
    } catch (...) {
        return 0;
    }
}

int infer_tensor_shape(InferTensor t, int* out_shape, int max_dims) {
    try {
        return infergo::tensor_get_shape(
            static_cast<const infergo::Tensor*>(t), out_shape, max_dims
        );
    } catch (...) {
        return 0;
    }
}

int infer_tensor_dtype(InferTensor t) {
    try {
        return infergo::tensor_get_dtype(static_cast<const infergo::Tensor*>(t));
    } catch (...) {
        return -1;
    }
}

InferError infer_tensor_to_device(InferTensor t, int device_id) {
#ifdef INFER_CUDA_AVAILABLE
    try {
        if (t == nullptr) {
            infergo::set_last_error("infer_tensor_to_device: null tensor");
            return INFER_ERR_NULL;
        }
        const bool ok = infergo::tensor_to_device(
            static_cast<infergo::Tensor*>(t), device_id
        );
        return ok ? INFER_OK : INFER_ERR_CUDA;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_RUNTIME;
    } catch (...) {
        return INFER_ERR_UNKNOWN;
    }
#else
    (void)t; (void)device_id;
    infergo::set_last_error("infer_tensor_to_device: CUDA not available in this build");
    return INFER_ERR_CUDA;
#endif
}

InferError infer_tensor_to_host(InferTensor t) {
#ifdef INFER_CUDA_AVAILABLE
    try {
        if (t == nullptr) {
            infergo::set_last_error("infer_tensor_to_host: null tensor");
            return INFER_ERR_NULL;
        }
        const bool ok = infergo::tensor_to_host(static_cast<infergo::Tensor*>(t));
        return ok ? INFER_OK : INFER_ERR_CUDA;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_RUNTIME;
    } catch (...) {
        return INFER_ERR_UNKNOWN;
    }
#else
    (void)t;
    infergo::set_last_error("infer_tensor_to_host: CUDA not available in this build");
    return INFER_ERR_CUDA;
#endif
}

InferError infer_tensor_copy_from(InferTensor t, const void* src, int nbytes) {
    try {
        if (t == nullptr) {
            infergo::set_last_error("infer_tensor_copy_from: null tensor");
            return INFER_ERR_NULL;
        }
        if (src == nullptr) {
            infergo::set_last_error("infer_tensor_copy_from: null src");
            return INFER_ERR_NULL;
        }
        if (nbytes <= 0) {
            infergo::set_last_error("infer_tensor_copy_from: nbytes must be > 0");
            return INFER_ERR_INVALID;
        }
        const bool ok = infergo::tensor_copy_from(
            static_cast<infergo::Tensor*>(t), src, nbytes
        );
        if (!ok) {
            // error string already set by tensor_copy_from
            return INFER_ERR_INVALID;
        }
        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_RUNTIME;
    } catch (...) {
        return INFER_ERR_UNKNOWN;
    }
}

// ─── Session API ──────────────────────────────────────────────────────────────

InferSession infer_session_create(const char* provider, int device_id) {
    try {
        const std::string p = (provider != nullptr) ? provider : "cpu";
        auto* s = new infergo::OnnxSession(p, device_id);
        return static_cast<InferSession>(s);
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return nullptr;
    } catch (...) {
        infergo::set_last_error("infer_session_create: unknown exception");
        return nullptr;
    }
}

InferError infer_session_load(InferSession s, const char* model_path) {
    try {
        if (s == nullptr) {
            infergo::set_last_error("infer_session_load: null session");
            return INFER_ERR_NULL;
        }
        if (model_path == nullptr) {
            infergo::set_last_error("infer_session_load: null model_path");
            return INFER_ERR_NULL;
        }
        static_cast<infergo::OnnxSession*>(s)->load_model(model_path);
        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_LOAD;
    } catch (...) {
        return INFER_ERR_UNKNOWN;
    }
}

int infer_session_num_inputs(InferSession s) {
    if (s == nullptr) return 0;
    return static_cast<infergo::OnnxSession*>(s)->num_inputs();
}

int infer_session_num_outputs(InferSession s) {
    if (s == nullptr) return 0;
    return static_cast<infergo::OnnxSession*>(s)->num_outputs();
}

InferError infer_session_input_name(InferSession s, int idx, char* out_buf, int buf_size) {
    try {
        if (s == nullptr || out_buf == nullptr || buf_size <= 0) {
            infergo::set_last_error("infer_session_input_name: invalid argument");
            return INFER_ERR_NULL;
        }
        const std::string& name =
            static_cast<infergo::OnnxSession*>(s)->input_name(idx);
        std::strncpy(out_buf, name.c_str(), static_cast<size_t>(buf_size) - 1);
        out_buf[buf_size - 1] = '\0';
        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_INVALID;
    } catch (...) {
        return INFER_ERR_UNKNOWN;
    }
}

InferError infer_session_output_name(InferSession s, int idx, char* out_buf, int buf_size) {
    try {
        if (s == nullptr || out_buf == nullptr || buf_size <= 0) {
            infergo::set_last_error("infer_session_output_name: invalid argument");
            return INFER_ERR_NULL;
        }
        const std::string& name =
            static_cast<infergo::OnnxSession*>(s)->output_name(idx);
        std::strncpy(out_buf, name.c_str(), static_cast<size_t>(buf_size) - 1);
        out_buf[buf_size - 1] = '\0';
        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_INVALID;
    } catch (...) {
        return INFER_ERR_UNKNOWN;
    }
}

InferError infer_session_run(
    InferSession  s,
    InferTensor*  inputs,  int n_inputs,
    InferTensor*  outputs, int n_outputs)
{
    try {
        if (s == nullptr) {
            infergo::set_last_error("infer_session_run: null session");
            return INFER_ERR_NULL;
        }
        if ((n_inputs > 0 && inputs == nullptr) || (n_outputs > 0 && outputs == nullptr)) {
            infergo::set_last_error("infer_session_run: null inputs/outputs array");
            return INFER_ERR_NULL;
        }

        std::vector<infergo::Tensor*> in_tensors(static_cast<size_t>(n_inputs));
        for (int i = 0; i < n_inputs; ++i) {
            in_tensors[i] = static_cast<infergo::Tensor*>(inputs[i]);
        }

        std::vector<infergo::Tensor*> out_tensors =
            static_cast<infergo::OnnxSession*>(s)->run(in_tensors);

        const int actual = static_cast<int>(out_tensors.size());
        const int copy_n = (actual < n_outputs) ? actual : n_outputs;
        for (int i = 0; i < copy_n; ++i) {
            outputs[i] = static_cast<InferTensor>(out_tensors[i]);
        }
        // Free any extra outputs not fitting in the caller's array
        for (int i = copy_n; i < actual; ++i) {
            infergo::tensor_free(out_tensors[i]);
        }

        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_RUNTIME;
    } catch (...) {
        return INFER_ERR_UNKNOWN;
    }
}

void infer_session_destroy(InferSession s) {
    if (s == nullptr) return;
    try {
        delete static_cast<infergo::OnnxSession*>(s);
    } catch (...) {
        // destructor must not throw
    }
}

// ─── Tokenizer API ────────────────────────────────────────────────────────────

InferTokenizer infer_tokenizer_load(const char* path) {
    try {
        if (path == nullptr) {
            infergo::set_last_error("infer_tokenizer_load: null path");
            return nullptr;
        }
        return static_cast<InferTokenizer>(
            new infergo::TokenizerWrapper(path)
        );
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return nullptr;
    } catch (...) {
        infergo::set_last_error("infer_tokenizer_load: unknown exception");
        return nullptr;
    }
}

int infer_tokenizer_encode(
    InferTokenizer  tok,
    const char*     text,
    int             add_special_tokens,
    int*            out_ids,
    int*            out_mask,
    int             max_tokens)
{
    try {
        if (tok == nullptr || text == nullptr || out_ids == nullptr || out_mask == nullptr) {
            infergo::set_last_error("infer_tokenizer_encode: null argument");
            return -1;
        }
        auto& t = *static_cast<infergo::TokenizerWrapper*>(tok);
        const infergo::Encoding enc = t.encode(
            text, add_special_tokens != 0, max_tokens
        );
        const int n = static_cast<int>(enc.ids.size());
        std::memcpy(out_ids,  enc.ids.data(),            static_cast<size_t>(n) * sizeof(int));
        std::memcpy(out_mask, enc.attention_mask.data(), static_cast<size_t>(n) * sizeof(int));
        return n;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return -1;
    } catch (...) {
        infergo::set_last_error("infer_tokenizer_encode: unknown exception");
        return -1;
    }
}

int infer_tokenizer_decode(
    InferTokenizer  tok,
    const int*      ids,
    int             n_ids,
    int             skip_special_tokens,
    char*           out_buf,
    int             buf_size)
{
    try {
        if (tok == nullptr || out_buf == nullptr || buf_size <= 0) {
            infergo::set_last_error("infer_tokenizer_decode: null argument");
            return -1;
        }
        std::vector<int32_t> id_vec;
        if (n_ids > 0 && ids != nullptr) {
            id_vec.assign(ids, ids + n_ids);
        }
        auto& t = *static_cast<infergo::TokenizerWrapper*>(tok);
        const std::string text = t.decode(id_vec, skip_special_tokens != 0);
        std::strncpy(out_buf, text.c_str(), static_cast<size_t>(buf_size) - 1);
        out_buf[buf_size - 1] = '\0';
        return 0;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return -1;
    } catch (...) {
        infergo::set_last_error("infer_tokenizer_decode: unknown exception");
        return -1;
    }
}

int infer_tokenizer_decode_token(
    InferTokenizer  tok,
    int             id,
    char*           out_buf,
    int             buf_size)
{
    try {
        if (tok == nullptr || out_buf == nullptr || buf_size <= 0) {
            infergo::set_last_error("infer_tokenizer_decode_token: null argument");
            return -1;
        }
        auto& t = *static_cast<infergo::TokenizerWrapper*>(tok);
        const std::string piece = t.decode_token(static_cast<int32_t>(id));
        std::strncpy(out_buf, piece.c_str(), static_cast<size_t>(buf_size) - 1);
        out_buf[buf_size - 1] = '\0';
        return 0;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return -1;
    } catch (...) {
        infergo::set_last_error("infer_tokenizer_decode_token: unknown exception");
        return -1;
    }
}

int infer_tokenizer_vocab_size(InferTokenizer tok) {
    if (tok == nullptr) return 0;
    try {
        return static_cast<infergo::TokenizerWrapper*>(tok)->vocab_size();
    } catch (...) {
        return 0;
    }
}

void infer_tokenizer_destroy(InferTokenizer tok) {
    if (tok == nullptr) return;
    try {
        delete static_cast<infergo::TokenizerWrapper*>(tok);
    } catch (...) {
        // destructor must not throw
    }
}

// ─── LLM Engine API ───────────────────────────────────────────────────────────

// Internal handle: owns the engine and the paged KV allocator together.
struct LLMHandle {
    infergo::LLMEngine       engine;
    infergo::KVPageAllocator pages;  // replaces KVCacheSlotManager
    int n_ctx = 0;

    LLMHandle(int n_seq_max, int ctx_size)
        : pages(infergo::KVPageAllocator::kDefaultPageSize,
                n_seq_max,
                ctx_size / infergo::KVPageAllocator::kDefaultPageSize)
        , n_ctx(ctx_size)
    {}
};

// Internal handle: owns one InferSequence + its last decoded logits.
struct SeqHandle {
    infergo::InferSequence  seq;
    std::vector<float>      logits;  // updated by infer_llm_batch_decode
    llama_context*          ctx;     // needed to clear KV cache on destroy

    SeqHandle(infergo::KVPageAllocator& alloc,
              std::vector<int32_t> tokens,
              int32_t eos,
              llama_context* ctx_)
        : seq(alloc, std::move(tokens), eos), ctx(ctx_) {}

    ~SeqHandle() {
        // Remove this sequence's KV cache entries so the slot can be reused
        // by future requests without stale positional data.
        if (ctx != nullptr) {
            llama_memory_seq_rm(llama_get_memory(ctx),
                                static_cast<llama_seq_id>(seq.SlotID()),
                                -1, -1);
        }
    }
};

InferLLM infer_llm_create(const char* path,
                           int         n_gpu_layers,
                           int         ctx_size,
                           int         n_seq_max,
                           int         n_batch)
{
    try {
        if (path == nullptr) {
            infergo::set_last_error("infer_llm_create: null path");
            return nullptr;
        }
        if (n_seq_max <= 0) {
            infergo::set_last_error("infer_llm_create: n_seq_max must be > 0");
            return nullptr;
        }
        auto* h = new LLMHandle(n_seq_max, ctx_size);
        h->engine.LoadModel(path, n_gpu_layers, ctx_size, n_seq_max, n_batch);
        return static_cast<InferLLM>(h);
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return nullptr;
    } catch (...) {
        infergo::set_last_error("infer_llm_create: unknown exception");
        return nullptr;
    }
}

void infer_llm_destroy(InferLLM llm) {
    if (llm == nullptr) return;
    try { delete static_cast<LLMHandle*>(llm); } catch (...) {}
}

int infer_llm_vocab_size(InferLLM llm) {
    if (llm == nullptr) return 0;
    return static_cast<LLMHandle*>(llm)->engine.VocabSize();
}

int infer_llm_bos(InferLLM llm) {
    if (llm == nullptr) return -1;
    return static_cast<int>(static_cast<LLMHandle*>(llm)->engine.BOS());
}

int infer_llm_eos(InferLLM llm) {
    if (llm == nullptr) return -1;
    return static_cast<int>(static_cast<LLMHandle*>(llm)->engine.EOS());
}

int infer_llm_is_eog(InferLLM llm, int token) {
    if (llm == nullptr) return 0;
    return static_cast<LLMHandle*>(llm)->engine.IsEOG(static_cast<int32_t>(token)) ? 1 : 0;
}

int infer_llm_kv_pages_free(InferLLM llm) {
    if (llm == nullptr) return 0;
    return static_cast<LLMHandle*>(llm)->pages.FreePages();
}

int infer_llm_kv_pages_total(InferLLM llm) {
    if (llm == nullptr) return 0;
    return static_cast<LLMHandle*>(llm)->pages.TotalPages();
}

int infer_llm_kv_page_size(InferLLM llm) {
    if (llm == nullptr) return 0;
    return static_cast<LLMHandle*>(llm)->pages.PageSize();
}

int infer_llm_tokenize(InferLLM llm, const char* text, int add_bos,
                        int* out_ids, int max_tokens) {
    try {
        if (llm == nullptr || text == nullptr || out_ids == nullptr || max_tokens <= 0) {
            infergo::set_last_error("infer_llm_tokenize: invalid argument");
            return -1;
        }
        auto* h = static_cast<LLMHandle*>(llm);
        const auto tokens = h->engine.Tokenize(text, add_bos != 0);
        const int n = std::min(static_cast<int>(tokens.size()), max_tokens);
        for (int i = 0; i < n; ++i) out_ids[i] = static_cast<int>(tokens[i]);
        return n;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return -1;
    } catch (...) {
        return -1;
    }
}

int infer_llm_token_to_piece(InferLLM llm, int token, char* out_buf, int buf_size) {
    try {
        if (llm == nullptr || out_buf == nullptr || buf_size <= 0) {
            infergo::set_last_error("infer_llm_token_to_piece: invalid argument");
            return -1;
        }
        auto* h = static_cast<LLMHandle*>(llm);
        const std::string piece = h->engine.TokenToPiece(static_cast<int32_t>(token));
        const int n = std::min(static_cast<int>(piece.size()), buf_size - 1);
        std::memcpy(out_buf, piece.data(), static_cast<size_t>(n));
        out_buf[n] = '\0';
        return 0;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return -1;
    } catch (...) {
        return -1;
    }
}

InferSeq infer_seq_create(InferLLM llm, const int* tokens, int n_tokens) {
    try {
        if (llm == nullptr || tokens == nullptr || n_tokens <= 0) {
            infergo::set_last_error("infer_seq_create: invalid argument");
            return nullptr;
        }
        auto* h = static_cast<LLMHandle*>(llm);
        std::vector<int32_t> tok_vec(tokens, tokens + n_tokens);
        auto* s = new SeqHandle(h->pages, std::move(tok_vec),
                                static_cast<int32_t>(h->engine.EOS()),
                                h->engine.Context());
        return static_cast<InferSeq>(s);
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return nullptr;
    } catch (...) {
        infergo::set_last_error("infer_seq_create: unknown exception");
        return nullptr;
    }
}

void infer_seq_destroy(InferSeq seq) {
    if (seq == nullptr) return;
    try { delete static_cast<SeqHandle*>(seq); } catch (...) {}
}

int infer_seq_is_done(InferSeq seq) {
    if (seq == nullptr) return 1;
    return static_cast<SeqHandle*>(seq)->seq.IsDone() ? 1 : 0;
}

int infer_seq_position(InferSeq seq) {
    if (seq == nullptr) return 0;
    return static_cast<SeqHandle*>(seq)->seq.Position();
}

int infer_seq_slot_id(InferSeq seq) {
    if (seq == nullptr) return -1;
    return static_cast<SeqHandle*>(seq)->seq.SlotID();
}

void infer_seq_append_token(InferSeq seq, int token) {
    if (seq == nullptr) return;
    static_cast<SeqHandle*>(seq)->seq.AppendToken(static_cast<int32_t>(token));
}

int infer_seq_next_tokens(InferSeq seq, int* out_ids, int max_tokens) {
    try {
        if (seq == nullptr || out_ids == nullptr || max_tokens <= 0) {
            infergo::set_last_error("infer_seq_next_tokens: invalid argument");
            return -1;
        }
        const auto tokens = static_cast<SeqHandle*>(seq)->seq.NextTokens();
        const int n = std::min(static_cast<int>(tokens.size()), max_tokens);
        for (int i = 0; i < n; ++i) out_ids[i] = static_cast<int>(tokens[i]);
        return n;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return -1;
    } catch (...) {
        return -1;
    }
}

InferError infer_llm_batch_decode(InferLLM llm, InferSeq* seqs, int n_seqs) {
    try {
        if (llm == nullptr) {
            infergo::set_last_error("infer_llm_batch_decode: null llm");
            return INFER_ERR_NULL;
        }
        if (seqs == nullptr || n_seqs <= 0) {
            infergo::set_last_error("infer_llm_batch_decode: empty sequence list");
            return INFER_ERR_INVALID;
        }
        auto* h = static_cast<LLMHandle*>(llm);

        // Build SequenceInputs
        std::vector<infergo::SequenceInput> inputs;
        inputs.reserve(static_cast<size_t>(n_seqs));
        for (int i = 0; i < n_seqs; ++i) {
            if (seqs[i] == nullptr) continue;
            auto* sh = static_cast<SeqHandle*>(seqs[i]);
            if (sh->seq.IsDone()) continue;

            infergo::SequenceInput inp;
            inp.seq_id      = sh->seq.SlotID();
            inp.tokens      = sh->seq.NextTokens();
            inp.pos         = sh->seq.Position();
            inp.want_logits = true;
            inputs.push_back(std::move(inp));
        }
        if (inputs.empty()) return INFER_OK;

        // Run batch decode
        auto results = h->engine.BatchDecode(inputs);

        // Write logits back into each SeqHandle by matching seq_id
        for (const auto& r : results) {
            for (int i = 0; i < n_seqs; ++i) {
                if (seqs[i] == nullptr) continue;
                auto* sh = static_cast<SeqHandle*>(seqs[i]);
                if (sh->seq.SlotID() == r.seq_id) {
                    sh->logits = r.logits;
                    break;
                }
            }
        }
        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_RUNTIME;
    } catch (...) {
        return INFER_ERR_UNKNOWN;
    }
}

InferError infer_seq_get_logits(InferSeq seq, float* out_logits, int vocab_size) {
    try {
        if (seq == nullptr || out_logits == nullptr || vocab_size <= 0) {
            infergo::set_last_error("infer_seq_get_logits: invalid argument");
            return INFER_ERR_NULL;
        }
        auto* sh = static_cast<SeqHandle*>(seq);
        if (sh->logits.empty()) {
            infergo::set_last_error("infer_seq_get_logits: no logits available (call batch_decode first)");
            return INFER_ERR_INVALID;
        }
        const int n = std::min(vocab_size, static_cast<int>(sh->logits.size()));
        std::memcpy(out_logits, sh->logits.data(), static_cast<size_t>(n) * sizeof(float));
        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_RUNTIME;
    } catch (...) {
        return INFER_ERR_UNKNOWN;
    }
}

// ─── Preprocessing API ────────────────────────────────────────────────────────

#ifdef INFER_PREPROCESS_AVAILABLE

InferTensor infer_preprocess_decode_image(const void* data, int nbytes) {
    try {
        if (data == nullptr || nbytes <= 0) {
            infergo::set_last_error("infer_preprocess_decode_image: null or empty input");
            return nullptr;
        }
        return static_cast<InferTensor>(
            infergo::decode_image(static_cast<const uint8_t*>(data), nbytes)
        );
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return nullptr;
    } catch (...) {
        infergo::set_last_error("infer_preprocess_decode_image: unknown exception");
        return nullptr;
    }
}

InferTensor infer_preprocess_letterbox(InferTensor src, int target_w, int target_h) {
    try {
        if (src == nullptr) {
            infergo::set_last_error("infer_preprocess_letterbox: null tensor");
            return nullptr;
        }
        if (target_w <= 0 || target_h <= 0) {
            infergo::set_last_error("infer_preprocess_letterbox: target dimensions must be positive");
            return nullptr;
        }
        return static_cast<InferTensor>(
            infergo::letterbox(static_cast<infergo::Tensor*>(src), target_w, target_h)
        );
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return nullptr;
    } catch (...) {
        infergo::set_last_error("infer_preprocess_letterbox: unknown exception");
        return nullptr;
    }
}

InferTensor infer_preprocess_normalize(InferTensor src, float scale,
                                        const float* mean, const float* std) {
    try {
        if (src == nullptr) {
            infergo::set_last_error("infer_preprocess_normalize: null tensor");
            return nullptr;
        }
        if (mean == nullptr || std == nullptr) {
            infergo::set_last_error("infer_preprocess_normalize: null mean or std");
            return nullptr;
        }
        if (scale <= 0.0f) {
            infergo::set_last_error("infer_preprocess_normalize: scale must be positive");
            return nullptr;
        }
        return static_cast<InferTensor>(
            infergo::normalize(static_cast<infergo::Tensor*>(src), scale, mean, std)
        );
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return nullptr;
    } catch (...) {
        infergo::set_last_error("infer_preprocess_normalize: unknown exception");
        return nullptr;
    }
}

InferTensor infer_preprocess_stack_batch(const InferTensor* tensors, int n) {
    try {
        if (tensors == nullptr || n <= 0) {
            infergo::set_last_error("infer_preprocess_stack_batch: null or empty tensor array");
            return nullptr;
        }
        for (int i = 0; i < n; ++i) {
            if (tensors[i] == nullptr) {
                infergo::set_last_error("infer_preprocess_stack_batch: null tensor in array");
                return nullptr;
            }
        }
        std::vector<const infergo::Tensor*> v;
        v.reserve(n);
        for (int i = 0; i < n; ++i)
            v.push_back(static_cast<const infergo::Tensor*>(tensors[i]));
        return static_cast<InferTensor>(infergo::stack_batch(v));
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return nullptr;
    } catch (...) {
        infergo::set_last_error("infer_preprocess_stack_batch: unknown exception");
        return nullptr;
    }
}

#else

InferTensor infer_preprocess_decode_image(const void* /*data*/, int /*nbytes*/) {
    infergo::set_last_error("infer_preprocess_decode_image: OpenCV not available in this build");
    return nullptr;
}

InferTensor infer_preprocess_letterbox(InferTensor /*src*/, int /*target_w*/, int /*target_h*/) {
    infergo::set_last_error("infer_preprocess_letterbox: OpenCV not available in this build");
    return nullptr;
}

InferTensor infer_preprocess_normalize(InferTensor /*src*/, float /*scale*/,
                                        const float* /*mean*/, const float* /*std*/) {
    infergo::set_last_error("infer_preprocess_normalize: OpenCV not available in this build");
    return nullptr;
}

InferTensor infer_preprocess_stack_batch(const InferTensor* /*tensors*/, int /*n*/) {
    infergo::set_last_error("infer_preprocess_stack_batch: OpenCV not available in this build");
    return nullptr;
}

#endif // INFER_PREPROCESS_AVAILABLE

// ─── Postprocessing API ───────────────────────────────────────────────────────

int infer_postprocess_classify(InferTensor logits, int top_k,
                               InferClassResult* out_results) {
    try {
        if (logits == nullptr) {
            infergo::set_last_error("infer_postprocess_classify: null logits tensor");
            return -1;
        }
        if (top_k <= 0) {
            infergo::set_last_error("infer_postprocess_classify: top_k must be positive");
            return -1;
        }
        if (out_results == nullptr) {
            infergo::set_last_error("infer_postprocess_classify: null out_results");
            return -1;
        }
        auto results = infergo::classify(
            static_cast<infergo::Tensor*>(logits), top_k
        );
        for (int i = 0; i < static_cast<int>(results.size()); ++i) {
            out_results[i].label_idx  = results[i].label_idx;
            out_results[i].confidence = results[i].confidence;
        }
        return static_cast<int>(results.size());
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return -1;
    } catch (...) {
        infergo::set_last_error("infer_postprocess_classify: unknown exception");
        return -1;
    }
}

int infer_postprocess_nms(InferTensor predictions,
                          float conf_thresh, float iou_thresh,
                          InferBox* out_boxes, int max_boxes) {
    try {
        if (predictions == nullptr) {
            infergo::set_last_error("infer_postprocess_nms: null predictions tensor");
            return -1;
        }
        if (out_boxes == nullptr || max_boxes <= 0) {
            infergo::set_last_error("infer_postprocess_nms: null or zero out_boxes/max_boxes");
            return -1;
        }
        auto boxes = infergo::nms(
            static_cast<infergo::Tensor*>(predictions),
            conf_thresh, iou_thresh
        );
        const int n = std::min(static_cast<int>(boxes.size()), max_boxes);
        for (int i = 0; i < n; ++i) {
            out_boxes[i].x1         = boxes[i].x1;
            out_boxes[i].y1         = boxes[i].y1;
            out_boxes[i].x2         = boxes[i].x2;
            out_boxes[i].y2         = boxes[i].y2;
            out_boxes[i].class_idx  = boxes[i].class_idx;
            out_boxes[i].confidence = boxes[i].confidence;
        }
        return n;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return -1;
    } catch (...) {
        infergo::set_last_error("infer_postprocess_nms: unknown exception");
        return -1;
    }
}

InferError infer_postprocess_normalize_embedding(InferTensor t) {
    try {
        if (t == nullptr) {
            infergo::set_last_error("infer_postprocess_normalize_embedding: null tensor");
            return INFER_ERR_NULL;
        }
        infergo::normalize_embedding(static_cast<infergo::Tensor*>(t));
        return INFER_OK;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return INFER_ERR_INVALID;
    } catch (...) {
        infergo::set_last_error("infer_postprocess_normalize_embedding: unknown exception");
        return INFER_ERR_UNKNOWN;
    }
}
