// VectorDB + RAG pipeline C API implementation

#include "infer_api.h"
#include "../search/vector_db.hpp"
#include "../onnx/onnx_session.hpp"
#include "../tokenizer/tokenizer.hpp"
#include "../tensor/tensor.hpp"

#include <cstring>
#include <string>
#include <vector>

namespace infergo {
    void set_last_error(const char* msg) noexcept;
}

// ─── VectorDB API ──────────────────────────────────────────────────────────

InferVectorDB infer_vectordb_create(int dim, int M, int ef_construction) {
    try {
        return static_cast<InferVectorDB>(
            new infergo::VectorDB(dim, M > 0 ? M : 16, ef_construction > 0 ? ef_construction : 200));
    } catch (...) { return nullptr; }
}

int infer_vectordb_insert(InferVectorDB db, int64_t id, const float* vec, const char* metadata) {
    if (!db || !vec) return -1;
    return static_cast<infergo::VectorDB*>(db)->Insert(id, vec, metadata ? metadata : "") ? 0 : -1;
}

int infer_vectordb_delete(InferVectorDB db, int64_t id) {
    if (!db) return -1;
    return static_cast<infergo::VectorDB*>(db)->Delete(id) ? 0 : -1;
}

int infer_vectordb_update(InferVectorDB db, int64_t id, const float* vec, const char* metadata) {
    if (!db || !vec) return -1;
    return static_cast<infergo::VectorDB*>(db)->Update(id, vec, metadata ? metadata : "") ? 0 : -1;
}

int infer_vectordb_get(InferVectorDB db, int64_t id, float* out_vec, char* out_meta, int meta_buf_size) {
    if (!db) return -1;
    std::string meta;
    bool ok = static_cast<infergo::VectorDB*>(db)->Get(id, out_vec, meta);
    if (!ok) return -1;
    if (out_meta && meta_buf_size > 0) {
        int n = std::min(static_cast<int>(meta.size()), meta_buf_size - 1);
        std::memcpy(out_meta, meta.data(), static_cast<size_t>(n));
        out_meta[n] = '\0';
    }
    return 0;
}

int infer_vectordb_search(InferVectorDB db, const float* query, int k, int ef_search,
                           const char* metadata_filter,
                           int64_t* out_ids, float* out_distances, int max_results) {
    if (!db || !query || !out_ids) return -1;
    try {
        auto results = static_cast<infergo::VectorDB*>(db)->Search(
            query, std::min(k, max_results), ef_search > 0 ? ef_search : 50,
            metadata_filter ? metadata_filter : "");
        int n = static_cast<int>(results.size());
        for (int i = 0; i < n; ++i) {
            out_ids[i] = results[i].id;
            if (out_distances) out_distances[i] = results[i].distance;
        }
        return n;
    } catch (...) { return -1; }
}

int infer_vectordb_save(InferVectorDB db, const char* path) {
    if (!db || !path) return -1;
    return static_cast<infergo::VectorDB*>(db)->Save(path) ? 0 : -1;
}

int infer_vectordb_load(InferVectorDB db, const char* path) {
    if (!db || !path) return -1;
    return static_cast<infergo::VectorDB*>(db)->Load(path) ? 0 : -1;
}

int infer_vectordb_size(InferVectorDB db) {
    if (!db) return 0;
    return static_cast<infergo::VectorDB*>(db)->Size();
}

void infer_vectordb_free(InferVectorDB db) {
    if (!db) return;
    try { delete static_cast<infergo::VectorDB*>(db); } catch (...) {}
}

// ─── RAG Pipeline ──────────────────────────────────────────────────────────

int infer_rag_pipeline(InferLLM       llm,
                        InferSession   embed_session,
                        InferTokenizer embed_tokenizer,
                        InferVectorDB  vector_db,
                        const char*    query,
                        int            k,
                        int            max_tokens,
                        float          temperature,
                        char*          out_text,
                        int            max_text_len) {
    try {
        if (!llm || !embed_session || !embed_tokenizer || !vector_db || !query) {
            infergo::set_last_error("infer_rag_pipeline: invalid argument");
            return -1;
        }

        // 1. Embed the query
        constexpr int max_dim = 1024;
        float query_vec[max_dim];
        int dim = infer_embed_pipeline(embed_session, embed_tokenizer, query, query_vec, max_dim);
        if (dim <= 0) {
            infergo::set_last_error("infer_rag_pipeline: query embedding failed");
            return -1;
        }

        // 2. Search vector DB
        if (k <= 0) k = 5;
        int64_t ids[32];
        float dists[32];
        int n = infer_vectordb_search(vector_db, query_vec, std::min(k, 32), 50,
                                       nullptr, ids, dists, 32);

        // 3. Build context from search results
        std::string context;
        auto* db = static_cast<infergo::VectorDB*>(vector_db);
        for (int i = 0; i < n; ++i) {
            std::string meta;
            if (db->Get(ids[i], nullptr, meta)) {
                context += "Context " + std::to_string(i + 1) + ": " + meta + "\n";
            }
        }

        // 4. Build prompt: context + query
        std::string prompt = "Based on the following context, answer the question.\n\n" +
                             context + "\nQuestion: " + query + "\nAnswer:";

        // 5. Tokenize and generate (uses infer_llm_generate)
        // We need to access the LLM engine to tokenize
        // For now, pass the prompt as tokens via the LLM's tokenizer
        // Tokenize using the LLM's built-in tokenizer
        int prompt_ids[4096];
        int n_prompt = infer_llm_tokenize(llm, prompt.c_str(), 1, prompt_ids, 4096);
        if (n_prompt <= 0) {
            infergo::set_last_error("infer_rag_pipeline: prompt tokenization failed");
            return -1;
        }

        // 6. Generate
        int gen_tokens = 0;
        int rc = infer_llm_generate(llm, prompt_ids, n_prompt,
                                     max_tokens > 0 ? max_tokens : 256,
                                     temperature, 0.9f, nullptr,
                                     nullptr, nullptr,
                                     out_text, max_text_len, &gen_tokens);
        if (rc != 0) return -1;

        return gen_tokens;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return -1;
    } catch (...) {
        return -1;
    }
}
