// Full C embedding pipeline: tokenize → ONNX → pool → L2 normalize.
// Zero Go compute in the hot path.

#include "infer_api.h"
#include "../onnx/onnx_session.hpp"
#include "../tokenizer/tokenizer.hpp"
#include "../tensor/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

// Use the declarations from tensor.hpp
#include "../tensor/tensor.hpp"

// ─── Single text embedding ──────────────────────────────────────────────────

int infer_embed_pipeline(InferSession   session,
                          InferTokenizer tokenizer,
                          const char*    text,
                          float*         out_vec,
                          int            max_dim) {
    try {
        if (!session || !tokenizer || !text || !out_vec || max_dim <= 0) {
            infergo::set_last_error("infer_embed_pipeline: invalid argument");
            return -1;
        }

        auto* sess = static_cast<infergo::OnnxSession*>(session);
        auto* tok  = static_cast<infergo::TokenizerWrapper*>(tokenizer);

        // 1. Tokenize (in C++ via Rust FFI)
        auto enc = tok->encode(text, true, 512);
        int n = static_cast<int>(enc.ids.size());
        if (n <= 0) {
            infergo::set_last_error("infer_embed_pipeline: tokenization failed");
            return -1;
        }
        const auto& ids = enc.ids;
        const auto& mask = enc.attention_mask;

        // 2. Allocate tensors: [1, seq_len] int64
        const int shape[2] = {1, n};
        auto* t_ids  = infergo::tensor_alloc_cpu(shape, 2, INFER_DTYPE_INT64);
        auto* t_mask = infergo::tensor_alloc_cpu(shape, 2, INFER_DTYPE_INT64);
        auto* t_type = infergo::tensor_alloc_cpu(shape, 2, INFER_DTYPE_INT64);
        if (!t_ids || !t_mask || !t_type) {
            infergo::set_last_error("infer_embed_pipeline: tensor alloc failed");
            if (t_ids)  infergo::tensor_free(t_ids);
            if (t_mask) infergo::tensor_free(t_mask);
            if (t_type) infergo::tensor_free(t_type);
            return -1;
        }

        // Fill tensors
        auto* id_data   = static_cast<int64_t*>(t_ids->data);
        auto* mask_data = static_cast<int64_t*>(t_mask->data);
        auto* type_data = static_cast<int64_t*>(t_type->data);
        for (int i = 0; i < n; ++i) {
            id_data[i]   = static_cast<int64_t>(ids[i]);
            mask_data[i] = static_cast<int64_t>(mask[i]);
            type_data[i] = 0;  // token_type_ids = 0 for single-segment
        }

        // 3. Run ONNX inference
        std::vector<infergo::Tensor*> inputs = {t_ids, t_mask, t_type};
        auto outputs = sess->run(inputs);

        infergo::tensor_free(t_ids);
        infergo::tensor_free(t_mask);
        infergo::tensor_free(t_type);

        if (outputs.empty()) {
            infergo::set_last_error("infer_embed_pipeline: ONNX run returned no outputs");
            return -1;
        }

        // 4. Mean pooling with attention mask
        // Output shape: [1, seq_len, hidden_dim]
        auto* out_tensor = outputs[0];
        if (out_tensor->ndim < 2) {
            for (auto* o : outputs) infergo::tensor_free(o);
            infergo::set_last_error("infer_embed_pipeline: unexpected output shape");
            return -1;
        }

        const int seq_len    = (out_tensor->ndim == 3) ? out_tensor->shape[1] : 1;
        const int hidden_dim = out_tensor->shape[out_tensor->ndim - 1];
        const int dim = std::min(hidden_dim, max_dim);
        const float* out_data = static_cast<const float*>(out_tensor->data);

        // Mean pool: sum masked positions, divide by mask count
        std::memset(out_vec, 0, static_cast<size_t>(dim) * sizeof(float));
        float mask_sum = 0;
        for (int s = 0; s < seq_len && s < n; ++s) {
            float m = static_cast<float>(mask[s]);
            mask_sum += m;
            for (int d = 0; d < dim; ++d) {
                out_vec[d] += out_data[s * hidden_dim + d] * m;
            }
        }
        if (mask_sum > 0) {
            for (int d = 0; d < dim; ++d) out_vec[d] /= mask_sum;
        }

        // 5. L2 normalize
        float norm = 0;
        for (int d = 0; d < dim; ++d) norm += out_vec[d] * out_vec[d];
        norm = std::sqrt(norm);
        if (norm > 1e-12f) {
            for (int d = 0; d < dim; ++d) out_vec[d] /= norm;
        }

        for (auto* o : outputs) infergo::tensor_free(o);
        return dim;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return -1;
    } catch (...) {
        infergo::set_last_error("infer_embed_pipeline: unknown exception");
        return -1;
    }
}

// ─── Batch embedding ────────────────────────────────────────────────────────

int infer_embed_batch_pipeline(InferSession   session,
                                InferTokenizer tokenizer,
                                const char**   texts,
                                int            n_texts,
                                float*         out_vecs,
                                int            max_dim) {
    try {
        if (!session || !tokenizer || !texts || n_texts <= 0 || !out_vecs || max_dim <= 0) {
            infergo::set_last_error("infer_embed_batch_pipeline: invalid argument");
            return -1;
        }

        auto* sess = static_cast<infergo::OnnxSession*>(session);
        auto* tok  = static_cast<infergo::TokenizerWrapper*>(tokenizer);

        // Tokenize all texts, find max length
        std::vector<std::vector<int32_t>> all_ids(n_texts), all_mask(n_texts);
        int max_len = 0;

        for (int i = 0; i < n_texts; ++i) {
            auto enc = tok->encode(texts[i], true, 512);
            all_ids[i] = enc.ids;
            all_mask[i] = enc.attention_mask;
            int n = static_cast<int>(enc.ids.size());
            if (n > max_len) max_len = n;
        }
        if (max_len <= 0) max_len = 1;

        // Allocate padded tensors: [n_texts, max_len] int64
        const int shape[2] = {n_texts, max_len};
        auto* t_ids  = infergo::tensor_alloc_cpu(shape, 2, INFER_DTYPE_INT64);
        auto* t_mask = infergo::tensor_alloc_cpu(shape, 2, INFER_DTYPE_INT64);
        auto* t_type = infergo::tensor_alloc_cpu(shape, 2, INFER_DTYPE_INT64);

        auto* id_data   = static_cast<int64_t*>(t_ids->data);
        auto* mask_data = static_cast<int64_t*>(t_mask->data);
        std::memset(id_data,   0, static_cast<size_t>(n_texts * max_len) * sizeof(int64_t));
        std::memset(mask_data, 0, static_cast<size_t>(n_texts * max_len) * sizeof(int64_t));
        std::memset(t_type->data, 0, static_cast<size_t>(n_texts * max_len) * sizeof(int64_t));

        for (int i = 0; i < n_texts; ++i) {
            for (int j = 0; j < static_cast<int>(all_ids[i].size()); ++j) {
                id_data[i * max_len + j]   = static_cast<int64_t>(all_ids[i][j]);
                mask_data[i * max_len + j] = static_cast<int64_t>(all_mask[i][j]);
            }
        }

        // Run ONNX
        std::vector<infergo::Tensor*> inputs = {t_ids, t_mask, t_type};
        auto outputs = sess->run(inputs);
        infergo::tensor_free(t_ids);
        infergo::tensor_free(t_mask);
        infergo::tensor_free(t_type);

        if (outputs.empty()) {
            infergo::set_last_error("infer_embed_batch_pipeline: no outputs");
            return -1;
        }

        auto* out_tensor = outputs[0];
        const int hidden_dim = out_tensor->shape[out_tensor->ndim - 1];
        const int dim = std::min(hidden_dim, max_dim);
        const float* out_data = static_cast<const float*>(out_tensor->data);

        // Mean pool + L2 normalize each text
        for (int i = 0; i < n_texts; ++i) {
            float* vec = out_vecs + i * max_dim;
            std::memset(vec, 0, static_cast<size_t>(dim) * sizeof(float));

            float mask_sum = 0;
            int seq_len = static_cast<int>(all_ids[i].size());
            for (int s = 0; s < seq_len; ++s) {
                float m = static_cast<float>(all_mask[i][s]);
                mask_sum += m;
                for (int d = 0; d < dim; ++d) {
                    vec[d] += out_data[(i * max_len + s) * hidden_dim + d] * m;
                }
            }
            if (mask_sum > 0) {
                for (int d = 0; d < dim; ++d) vec[d] /= mask_sum;
            }

            float norm = 0;
            for (int d = 0; d < dim; ++d) norm += vec[d] * vec[d];
            norm = std::sqrt(norm);
            if (norm > 1e-12f) {
                for (int d = 0; d < dim; ++d) vec[d] /= norm;
            }
        }

        for (auto* o : outputs) infergo::tensor_free(o);
        return dim;
    } catch (const std::exception& e) {
        infergo::set_last_error(e.what());
        return -1;
    } catch (...) {
        return -1;
    }
}

// ─── Rerank pipeline ────────────────────────────────────────────────────────

int infer_rerank_pipeline(InferSession   session,
                           InferTokenizer tokenizer,
                           const char*    query,
                           const char**   documents,
                           int            n_docs,
                           float*         out_scores,
                           int*           out_indices,
                           int            max_results) {
    try {
        if (!session || !tokenizer || !query || !documents || n_docs <= 0 ||
            !out_scores || !out_indices || max_results <= 0) {
            return -1;
        }

        // Embed query + all documents in one batch
        const int total = 1 + n_docs;
        std::vector<const char*> all_texts(total);
        all_texts[0] = query;
        for (int i = 0; i < n_docs; ++i) all_texts[i + 1] = documents[i];

        constexpr int max_dim = 1024;
        std::vector<float> vecs(total * max_dim, 0.0f);

        int dim = infer_embed_batch_pipeline(session, tokenizer,
            all_texts.data(), total, vecs.data(), max_dim);
        if (dim <= 0) return -1;

        // Cosine similarity: query (index 0) vs each document (index 1..n)
        const float* q_vec = vecs.data();
        std::vector<std::pair<float, int>> scored(n_docs);

        for (int i = 0; i < n_docs; ++i) {
            const float* d_vec = vecs.data() + (i + 1) * max_dim;
            float dot = 0;
            for (int d = 0; d < dim; ++d) dot += q_vec[d] * d_vec[d];
            // Vectors are L2-normalized, so dot product = cosine similarity
            scored[i] = {dot, i};
        }

        // Sort descending by score
        std::sort(scored.begin(), scored.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

        int n = std::min(n_docs, max_results);
        for (int i = 0; i < n; ++i) {
            out_scores[i]  = scored[i].first;
            out_indices[i] = scored[i].second;
        }

        return n;
    } catch (...) {
        return -1;
    }
}
