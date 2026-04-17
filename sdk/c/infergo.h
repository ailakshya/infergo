// infergo.h — Single-include C++ wrapper for the infergo C API
// Provides RAII classes, std::string returns, and exception-safe resource management.
//
// Usage:
//   #include "infergo.h"
//   infergo::LLM llm("model.gguf");
//   std::string reply = llm.generate("Hello, world!");
//
// Link with: -linfer_api
// Requires: C++17 or later

#pragma once

#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

// The C API header — adjust path as needed for your build system.
// CMake sets the include path automatically via find_package(infergo).
#include "infer_api.h"

namespace infergo {

// ---------------------------------------------------------------------------
// Exception type
// ---------------------------------------------------------------------------

class Error : public std::runtime_error {
public:
    explicit Error(const std::string& msg) : std::runtime_error(msg) {}

    /// Construct from the thread-local error string set by the C API.
    static Error from_last() {
        const char* s = infer_last_error_string();
        return Error(s ? s : "unknown infergo error");
    }
};

// ---------------------------------------------------------------------------
// Tensor (RAII wrapper)
// ---------------------------------------------------------------------------

class Tensor {
public:
    Tensor() : h_(nullptr) {}
    explicit Tensor(InferTensor h) : h_(h) {}

    /// Allocate a CPU tensor.
    static Tensor cpu(const std::vector<int>& shape, int dtype) {
        InferTensor t = infer_tensor_alloc_cpu(shape.data(), static_cast<int>(shape.size()), dtype);
        if (!t) throw Error::from_last();
        return Tensor(t);
    }

    /// Allocate a CUDA tensor.
    static Tensor cuda(const std::vector<int>& shape, int dtype, int device_id = 0) {
        InferTensor t = infer_tensor_alloc_cuda(shape.data(), static_cast<int>(shape.size()), dtype, device_id);
        if (!t) throw Error::from_last();
        return Tensor(t);
    }

    ~Tensor() { if (h_) infer_tensor_free(h_); }

    Tensor(Tensor&& o) noexcept : h_(o.h_) { o.h_ = nullptr; }
    Tensor& operator=(Tensor&& o) noexcept {
        if (this != &o) { if (h_) infer_tensor_free(h_); h_ = o.h_; o.h_ = nullptr; }
        return *this;
    }
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    InferTensor handle() const { return h_; }
    InferTensor release() { InferTensor t = h_; h_ = nullptr; return t; }

    void* data()          const { return infer_tensor_data_ptr(h_); }
    int   nbytes()        const { return infer_tensor_nbytes(h_); }
    int   nelements()     const { return infer_tensor_nelements(h_); }
    int   dtype()         const { return infer_tensor_dtype(h_); }

    std::vector<int> shape() const {
        int buf[8];
        int ndim = infer_tensor_shape(h_, buf, 8);
        return {buf, buf + ndim};
    }

    void to_device(int device_id = 0) {
        InferError e = infer_tensor_to_device(h_, device_id);
        if (e != INFER_OK) throw Error::from_last();
    }

    void to_host() {
        InferError e = infer_tensor_to_host(h_);
        if (e != INFER_OK) throw Error::from_last();
    }

    void copy_from(const void* src, int nbytes) {
        InferError e = infer_tensor_copy_from(h_, src, nbytes);
        if (e != INFER_OK) throw Error::from_last();
    }

    explicit operator bool() const { return h_ != nullptr; }

private:
    InferTensor h_;
};

// ---------------------------------------------------------------------------
// ONNX Session (RAII wrapper)
// ---------------------------------------------------------------------------

class Session {
public:
    Session() : h_(nullptr) {}

    /// Create a session for the given provider ("cpu", "cuda", "tensorrt", ...).
    explicit Session(const char* provider, int device_id = 0) {
        h_ = infer_session_create(provider, device_id);
        if (!h_) throw Error::from_last();
    }

    ~Session() { if (h_) infer_session_destroy(h_); }

    Session(Session&& o) noexcept : h_(o.h_) { o.h_ = nullptr; }
    Session& operator=(Session&& o) noexcept {
        if (this != &o) { if (h_) infer_session_destroy(h_); h_ = o.h_; o.h_ = nullptr; }
        return *this;
    }
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    InferSession handle() const { return h_; }

    void load(const char* model_path) {
        InferError e = infer_session_load(h_, model_path);
        if (e != INFER_OK) throw Error::from_last();
    }

    int num_inputs()  const { return infer_session_num_inputs(h_); }
    int num_outputs() const { return infer_session_num_outputs(h_); }

    std::string input_name(int idx) const {
        char buf[256];
        InferError e = infer_session_input_name(h_, idx, buf, sizeof(buf));
        if (e != INFER_OK) throw Error::from_last();
        return buf;
    }

    std::string output_name(int idx) const {
        char buf[256];
        InferError e = infer_session_output_name(h_, idx, buf, sizeof(buf));
        if (e != INFER_OK) throw Error::from_last();
        return buf;
    }

    /// Run inference. Returns output tensors (caller owns them).
    std::vector<Tensor> run(std::vector<Tensor>& inputs) {
        int n_in  = static_cast<int>(inputs.size());
        int n_out = num_outputs();

        std::vector<InferTensor> in_handles(n_in);
        for (int i = 0; i < n_in; ++i) in_handles[i] = inputs[i].handle();

        std::vector<InferTensor> out_handles(n_out, nullptr);
        InferError e = infer_session_run(h_, in_handles.data(), n_in,
                                         out_handles.data(), n_out);
        if (e != INFER_OK) throw Error::from_last();

        std::vector<Tensor> outputs;
        outputs.reserve(n_out);
        for (int i = 0; i < n_out; ++i) outputs.emplace_back(out_handles[i]);
        return outputs;
    }

private:
    InferSession h_;
};

// ---------------------------------------------------------------------------
// Tokenizer (RAII wrapper)
// ---------------------------------------------------------------------------

class Tokenizer {
public:
    Tokenizer() : h_(nullptr) {}

    explicit Tokenizer(const char* path) {
        h_ = infer_tokenizer_load(path);
        if (!h_) throw Error::from_last();
    }

    ~Tokenizer() { if (h_) infer_tokenizer_destroy(h_); }

    Tokenizer(Tokenizer&& o) noexcept : h_(o.h_) { o.h_ = nullptr; }
    Tokenizer& operator=(Tokenizer&& o) noexcept {
        if (this != &o) { if (h_) infer_tokenizer_destroy(h_); h_ = o.h_; o.h_ = nullptr; }
        return *this;
    }
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    InferTokenizer handle() const { return h_; }
    int vocab_size() const { return infer_tokenizer_vocab_size(h_); }

    /// Encode text to token IDs. Returns (ids, attention_mask).
    std::pair<std::vector<int>, std::vector<int>>
    encode(const std::string& text, bool add_special = true, int max_tokens = 2048) const {
        std::vector<int> ids(max_tokens), mask(max_tokens);
        int n = infer_tokenizer_encode(h_, text.c_str(), add_special ? 1 : 0,
                                       ids.data(), mask.data(), max_tokens);
        if (n < 0) throw Error::from_last();
        ids.resize(n);
        mask.resize(n);
        return {ids, mask};
    }

    /// Decode token IDs to text.
    std::string decode(const std::vector<int>& ids, bool skip_special = true) const {
        std::string buf(ids.size() * 16 + 256, '\0');
        int rc = infer_tokenizer_decode(h_, ids.data(), static_cast<int>(ids.size()),
                                        skip_special ? 1 : 0,
                                        buf.data(), static_cast<int>(buf.size()));
        if (rc < 0) throw Error::from_last();
        buf.resize(std::strlen(buf.c_str()));
        return buf;
    }

    /// Decode a single token ID.
    std::string decode_token(int id) const {
        char buf[256];
        int rc = infer_tokenizer_decode_token(h_, id, buf, sizeof(buf));
        if (rc < 0) throw Error::from_last();
        return buf;
    }

private:
    InferTokenizer h_;
};

// ---------------------------------------------------------------------------
// LLM Engine (RAII wrapper)
// ---------------------------------------------------------------------------

class LLM {
public:
    /// Construct and load a GGUF model.
    /// @param path        Path to .gguf model file.
    /// @param n_gpu       Number of transformer layers on GPU (999 = all).
    /// @param ctx_size    KV cache budget in tokens (default 4096).
    /// @param n_seq_max   Max concurrent sequences (default 1).
    /// @param n_batch     Max tokens per decode call (default 512).
    explicit LLM(const char* path,
                 int n_gpu = 999, int ctx_size = 4096,
                 int n_seq_max = 1, int n_batch = 512) {
        h_ = infer_llm_create(path, n_gpu, ctx_size, n_seq_max, n_batch);
        if (!h_) throw Error::from_last();
    }

    ~LLM() { if (h_) infer_llm_destroy(h_); }

    LLM(LLM&& o) noexcept : h_(o.h_) { o.h_ = nullptr; }
    LLM& operator=(LLM&& o) noexcept {
        if (this != &o) { if (h_) infer_llm_destroy(h_); h_ = o.h_; o.h_ = nullptr; }
        return *this;
    }
    LLM(const LLM&) = delete;
    LLM& operator=(const LLM&) = delete;

    InferLLM handle() const { return h_; }

    int vocab_size() const { return infer_llm_vocab_size(h_); }
    int bos()        const { return infer_llm_bos(h_); }
    int eos()        const { return infer_llm_eos(h_); }

    bool is_eog(int token) const { return infer_llm_is_eog(h_, token) != 0; }

    /// Tokenize text using the model's vocabulary.
    std::vector<int> tokenize(const std::string& text, bool add_bos = true,
                              int max_tokens = 2048) const {
        std::vector<int> ids(max_tokens);
        int n = infer_llm_tokenize(h_, text.c_str(), add_bos ? 1 : 0,
                                   ids.data(), max_tokens);
        if (n < 0) throw Error::from_last();
        ids.resize(n);
        return ids;
    }

    /// Convert a single token ID to its string piece.
    std::string token_to_piece(int token) const {
        char buf[256];
        int rc = infer_llm_token_to_piece(h_, token, buf, sizeof(buf));
        if (rc < 0) throw Error::from_last();
        return buf;
    }

    /// Generate text from a prompt string.
    /// @param prompt      Input text.
    /// @param max_tokens  Maximum tokens to generate.
    /// @param temperature Sampling temperature (0 = greedy).
    /// @param top_p       Nucleus sampling (1.0 = disabled).
    /// @param grammar     GBNF grammar string (nullptr = unconstrained).
    /// @param callback    Per-token callback; return false to stop. Can be nullptr.
    /// @return Generated text.
    std::string generate(const std::string& prompt,
                         int max_tokens = 512,
                         float temperature = 0.7f,
                         float top_p = 0.9f,
                         const char* grammar = nullptr,
                         std::function<bool(int, const char*)> callback = nullptr) {
        // Tokenize the prompt
        std::vector<int> tokens = tokenize(prompt, true);

        // Prepare output buffer
        std::string out_text(max_tokens * 16 + 256, '\0');
        int gen_tokens = 0;

        // Bridge for the C callback
        struct CBData {
            std::function<bool(int, const char*)>* fn;
        };
        CBData cbdata{callback ? &callback : nullptr};

        InferTokenCallback c_cb = nullptr;
        if (callback) {
            c_cb = [](int token, const char* piece, void* ud) -> int {
                auto* d = static_cast<CBData*>(ud);
                return (*d->fn)(token, piece) ? 1 : 0;
            };
        }

        int rc = infer_llm_generate(h_,
                                    tokens.data(), static_cast<int>(tokens.size()),
                                    max_tokens, temperature, top_p, grammar,
                                    c_cb, &cbdata,
                                    out_text.data(), static_cast<int>(out_text.size()),
                                    &gen_tokens);
        if (rc != 0) throw Error::from_last();
        out_text.resize(std::strlen(out_text.c_str()));
        return out_text;
    }

    /// Generate from pre-tokenized prompt.
    std::string generate_tokens(const std::vector<int>& prompt_tokens,
                                int max_tokens = 512,
                                float temperature = 0.7f,
                                float top_p = 0.9f,
                                const char* grammar = nullptr) {
        std::string out_text(max_tokens * 16 + 256, '\0');
        int gen_tokens = 0;
        int rc = infer_llm_generate(h_,
                                    prompt_tokens.data(),
                                    static_cast<int>(prompt_tokens.size()),
                                    max_tokens, temperature, top_p, grammar,
                                    nullptr, nullptr,
                                    out_text.data(), static_cast<int>(out_text.size()),
                                    &gen_tokens);
        if (rc != 0) throw Error::from_last();
        out_text.resize(std::strlen(out_text.c_str()));
        return out_text;
    }

    /// KV cache page metrics.
    int kv_pages_free()  const { return infer_llm_kv_pages_free(h_); }
    int kv_pages_total() const { return infer_llm_kv_pages_total(h_); }
    int kv_page_size()   const { return infer_llm_kv_page_size(h_); }

private:
    InferLLM h_;
};

// ---------------------------------------------------------------------------
// Embedding helper (Session + Tokenizer combo)
// ---------------------------------------------------------------------------

class Embedding {
public:
    /// Load an ONNX embedding model and its HuggingFace tokenizer.
    /// @param model_path      Path to ONNX model file.
    /// @param tokenizer_path  Path to tokenizer.json.
    /// @param provider        "cpu" or "cuda" (default "cpu").
    /// @param device_id       GPU device index (default 0).
    Embedding(const char* model_path,
              const char* tokenizer_path,
              const char* provider = "cpu",
              int device_id = 0)
        : session_(provider, device_id), tokenizer_(tokenizer_path) {
        session_.load(model_path);
    }

    /// Embed a single text. Returns a normalized float vector.
    std::vector<float> embed(const std::string& text, int max_dim = 1024) {
        std::vector<float> vec(max_dim);
        int dim = infer_embed_pipeline(session_.handle(), tokenizer_.handle(),
                                       text.c_str(), vec.data(), max_dim);
        if (dim < 0) throw Error::from_last();
        vec.resize(dim);
        return vec;
    }

    /// Batch-embed multiple texts. Returns a flat vector of (n_texts * dim) floats.
    /// Use dim() after the first call to know the embedding dimension.
    std::vector<float> embed_batch(const std::vector<std::string>& texts,
                                   int max_dim = 1024) {
        int n = static_cast<int>(texts.size());
        std::vector<const char*> ptrs(n);
        for (int i = 0; i < n; ++i) ptrs[i] = texts[i].c_str();

        std::vector<float> vecs(n * max_dim);
        int dim = infer_embed_batch_pipeline(session_.handle(), tokenizer_.handle(),
                                             ptrs.data(), n, vecs.data(), max_dim);
        if (dim < 0) throw Error::from_last();
        dim_ = dim;
        vecs.resize(n * dim);
        return vecs;
    }

    /// Rerank documents by relevance to a query.
    /// Returns (scores, indices) sorted by descending relevance.
    std::pair<std::vector<float>, std::vector<int>>
    rerank(const std::string& query,
           const std::vector<std::string>& documents,
           int max_results = 0) {
        int n = static_cast<int>(documents.size());
        if (max_results <= 0) max_results = n;

        std::vector<const char*> ptrs(n);
        for (int i = 0; i < n; ++i) ptrs[i] = documents[i].c_str();

        std::vector<float> scores(max_results);
        std::vector<int>   indices(max_results);
        int rc = infer_rerank_pipeline(session_.handle(), tokenizer_.handle(),
                                       query.c_str(), ptrs.data(), n,
                                       scores.data(), indices.data(), max_results);
        if (rc < 0) throw Error::from_last();
        scores.resize(rc);
        indices.resize(rc);
        return {scores, indices};
    }

    /// Last-known embedding dimension (valid after embed or embed_batch).
    int dim() const { return dim_; }

    Session&   session()   { return session_; }
    Tokenizer& tokenizer() { return tokenizer_; }

private:
    Session   session_;
    Tokenizer tokenizer_;
    int       dim_ = 0;
};

// ---------------------------------------------------------------------------
// VectorDB (RAII wrapper)
// ---------------------------------------------------------------------------

class VectorDB {
public:
    /// Create a new vector database.
    /// @param dim              Vector dimension (must match your embedding model).
    /// @param M                HNSW max connections per node (default 16).
    /// @param ef_construction  HNSW search width during build (default 200).
    explicit VectorDB(int dim, int M = 16, int ef_construction = 200) {
        h_ = infer_vectordb_create(dim, M, ef_construction);
        if (!h_) throw Error("failed to create VectorDB");
    }

    ~VectorDB() { if (h_) infer_vectordb_free(h_); }

    VectorDB(VectorDB&& o) noexcept : h_(o.h_) { o.h_ = nullptr; }
    VectorDB& operator=(VectorDB&& o) noexcept {
        if (this != &o) { if (h_) infer_vectordb_free(h_); h_ = o.h_; o.h_ = nullptr; }
        return *this;
    }
    VectorDB(const VectorDB&) = delete;
    VectorDB& operator=(const VectorDB&) = delete;

    InferVectorDB handle() const { return h_; }

    void insert(int64_t id, const float* vec, const char* metadata = nullptr) {
        int rc = infer_vectordb_insert(h_, id, vec, metadata);
        if (rc != 0) throw Error::from_last();
    }

    void insert(int64_t id, const std::vector<float>& vec,
                const std::string& metadata = "") {
        insert(id, vec.data(), metadata.empty() ? nullptr : metadata.c_str());
    }

    void remove(int64_t id) {
        int rc = infer_vectordb_delete(h_, id);
        if (rc != 0) throw Error::from_last();
    }

    void update(int64_t id, const float* vec, const char* metadata = nullptr) {
        int rc = infer_vectordb_update(h_, id, vec, metadata);
        if (rc != 0) throw Error::from_last();
    }

    /// Search for k nearest neighbors.
    /// Returns (ids, distances).
    struct SearchResult {
        std::vector<int64_t> ids;
        std::vector<float>   distances;
    };

    SearchResult search(const float* query, int k, int ef_search = 100,
                        const char* metadata_filter = nullptr) const {
        SearchResult r;
        r.ids.resize(k);
        r.distances.resize(k);
        int n = infer_vectordb_search(h_, query, k, ef_search, metadata_filter,
                                      r.ids.data(), r.distances.data(), k);
        if (n < 0) throw Error::from_last();
        r.ids.resize(n);
        r.distances.resize(n);
        return r;
    }

    SearchResult search(const std::vector<float>& query, int k,
                        int ef_search = 100,
                        const char* metadata_filter = nullptr) const {
        return search(query.data(), k, ef_search, metadata_filter);
    }

    void save(const char* path) const {
        int rc = infer_vectordb_save(h_, path);
        if (rc != 0) throw Error::from_last();
    }

    void load(const char* path) {
        int rc = infer_vectordb_load(h_, path);
        if (rc != 0) throw Error::from_last();
    }

    int size() const { return infer_vectordb_size(h_); }

private:
    InferVectorDB h_;
};

// ---------------------------------------------------------------------------
// BM25 full-text search (RAII wrapper)
// ---------------------------------------------------------------------------

class BM25 {
public:
    /// Create a BM25 index.
    /// @param k1  Term frequency saturation (default 1.2).
    /// @param b   Document length normalization (default 0.75).
    explicit BM25(float k1 = 1.2f, float b = 0.75f) {
        h_ = infer_bm25_create(k1, b);
        if (!h_) throw Error("failed to create BM25 index");
    }

    ~BM25() { if (h_) infer_bm25_free(h_); }

    BM25(BM25&& o) noexcept : h_(o.h_) { o.h_ = nullptr; }
    BM25& operator=(BM25&& o) noexcept {
        if (this != &o) { if (h_) infer_bm25_free(h_); h_ = o.h_; o.h_ = nullptr; }
        return *this;
    }
    BM25(const BM25&) = delete;
    BM25& operator=(const BM25&) = delete;

    InferBM25 handle() const { return h_; }

    void insert(int64_t id, const std::string& text) {
        infer_bm25_insert(h_, id, text.c_str());
    }

    void remove(int64_t id) {
        infer_bm25_remove(h_, id);
    }

    /// Search the BM25 index.
    /// Returns (ids, scores) sorted by descending BM25 score.
    struct SearchResult {
        std::vector<int64_t> ids;
        std::vector<float>   scores;
    };

    SearchResult search(const std::string& query, int k) const {
        SearchResult r;
        r.ids.resize(k);
        r.scores.resize(k);
        int n = infer_bm25_search(h_, query.c_str(), k,
                                  r.ids.data(), r.scores.data(), k);
        if (n < 0) throw Error::from_last();
        r.ids.resize(n);
        r.scores.resize(n);
        return r;
    }

    void save(const char* path) const {
        int rc = infer_bm25_save(h_, path);
        if (rc != 0) throw Error::from_last();
    }

    void load(const char* path) {
        int rc = infer_bm25_load(h_, path);
        if (rc != 0) throw Error::from_last();
    }

    int size() const { return infer_bm25_size(h_); }

private:
    InferBM25 h_;
};

// ---------------------------------------------------------------------------
// RAG pipeline (convenience function using existing RAII objects)
// ---------------------------------------------------------------------------

/// Run the full RAG pipeline: embed query, search vector DB, build context,
/// generate with LLM.
inline std::string rag_pipeline(LLM& llm, Embedding& emb, VectorDB& db,
                                const std::string& query,
                                int k = 3, int max_tokens = 512,
                                float temperature = 0.7f) {
    std::string out(max_tokens * 16 + 256, '\0');
    int len = infer_rag_pipeline(llm.handle(),
                                 emb.session().handle(),
                                 emb.tokenizer().handle(),
                                 db.handle(),
                                 query.c_str(), k, max_tokens, temperature,
                                 out.data(), static_cast<int>(out.size()));
    if (len < 0) throw Error::from_last();
    out.resize(len);
    return out;
}

// ---------------------------------------------------------------------------
// VRAM monitoring (free functions)
// ---------------------------------------------------------------------------

inline size_t cuda_vram_free()      { return infer_cuda_vram_free(); }
inline size_t cuda_vram_total()     { return infer_cuda_vram_total(); }
inline int    cuda_vram_used_pct()  { return infer_cuda_vram_used_pct(); }

} // namespace infergo
