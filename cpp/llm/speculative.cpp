#include "speculative.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

namespace infergo {

SpeculativeDecoder::SpeculativeDecoder() = default;

SpeculativeDecoder::~SpeculativeDecoder() {
    if (target_ctx_) {
        llama_free(target_ctx_);
        target_ctx_ = nullptr;
    }
    if (draft_ctx_) {
        llama_free(draft_ctx_);
        draft_ctx_ = nullptr;
    }
    if (draft_model_) {
        llama_model_free(draft_model_);
        draft_model_ = nullptr;
    }
}

bool SpeculativeDecoder::Init(const llama_model* target_model,
                               int target_ctx_size,
                               const std::string& draft_path,
                               int n_gpu_layers,
                               int n_draft) {
    target_model_ = target_model;
    n_draft_      = n_draft;

    // Load draft model
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    draft_model_ = llama_model_load_from_file(draft_path.c_str(), mparams);
    if (!draft_model_) return false;

    // Check vocab compatibility
    const llama_vocab* vocab_tgt = llama_model_get_vocab(target_model_);
    const llama_vocab* vocab_dft = llama_model_get_vocab(draft_model_);
    int n_tgt = llama_vocab_n_tokens(vocab_tgt);
    int n_dft = llama_vocab_n_tokens(vocab_dft);

    if (std::abs(n_tgt - n_dft) > 128) {
        llama_model_free(draft_model_);
        draft_model_ = nullptr;
        return false;
    }

    vocab_size_ = std::min(n_tgt, n_dft);
    int ctx_size = target_ctx_size > 0 ? target_ctx_size : 4096;

    // Create dedicated target context
    {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx      = static_cast<uint32_t>(ctx_size);
        cparams.n_batch    = 512;
        cparams.n_ubatch   = 512;
        cparams.n_seq_max  = 1;
        cparams.offload_kqv = true;
        cparams.no_perf     = true;

        target_ctx_ = llama_init_from_model(
            const_cast<llama_model*>(target_model_), cparams);
        if (!target_ctx_) {
            llama_model_free(draft_model_);
            draft_model_ = nullptr;
            return false;
        }
    }

    // Create draft context
    {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx      = static_cast<uint32_t>(ctx_size);
        cparams.n_batch    = 512;
        cparams.n_ubatch   = 512;
        cparams.n_seq_max  = 1;
        cparams.offload_kqv = true;
        cparams.no_perf     = true;

        draft_ctx_ = llama_init_from_model(draft_model_, cparams);
        if (!draft_ctx_) {
            llama_free(target_ctx_);
            target_ctx_ = nullptr;
            llama_model_free(draft_model_);
            draft_model_ = nullptr;
            return false;
        }
    }

    return true;
}

int32_t SpeculativeDecoder::SampleFromCtx(llama_context* ctx, int batch_idx,
                                           float temperature) {
    const float* logits = llama_get_logits_ith(ctx, batch_idx);
    if (!logits) return -1;

    if (temperature <= 0.0f) {
        int32_t best = 0;
        for (int i = 1; i < vocab_size_; ++i) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }

    thread_local std::mt19937 rng(std::random_device{}());
    float max_logit = *std::max_element(logits, logits + vocab_size_);
    std::vector<float> probs(static_cast<size_t>(vocab_size_));
    float sum = 0.0f;
    for (int i = 0; i < vocab_size_; ++i) {
        probs[i] = std::exp((logits[i] - max_logit) / temperature);
        sum += probs[i];
    }
    for (auto& p : probs) p /= sum;

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return static_cast<int32_t>(dist(rng));
}

// Helper: clear a batch for reuse (like common_batch_clear)
static void batch_clear(llama_batch& batch) {
    batch.n_tokens = 0;
}

// Helper: add one token to a batch (like common_batch_add)
static void batch_add(llama_batch& batch, llama_token token, llama_pos pos,
                       bool logits) {
    const int i = batch.n_tokens;
    batch.token[i]      = token;
    batch.pos[i]        = pos;
    batch.n_seq_id[i]   = 1;
    batch.seq_id[i][0]  = 0;
    batch.logits[i]     = logits ? 1 : 0;
    batch.n_tokens++;
}

std::string SpeculativeDecoder::Generate(
    const std::vector<int32_t>& prompt_tokens,
    int max_tokens,
    float temperature,
    const std::string& grammar_str [[maybe_unused]],
    TokenCallback callback,
    int* out_n_predict,
    int* out_n_drafted,
    int* out_n_accepted)
{
    if (!draft_model_ || !target_ctx_) {
        throw std::runtime_error("SpeculativeDecoder not initialized");
    }

    const llama_vocab* vocab = llama_model_get_vocab(target_model_);
    const int n_prompt = static_cast<int>(prompt_tokens.size());

    // Pre-allocate batches (reused throughout generation)
    llama_batch batch_tgt = llama_batch_init(512, 0, 1);
    llama_batch batch_dft = llama_batch_init(512, 0, 1);

    // ── Prefill: process prompt through both models ──
    batch_clear(batch_tgt);
    for (int i = 0; i < n_prompt; ++i) {
        batch_add(batch_tgt, prompt_tokens[i], i, i == n_prompt - 1);
    }
    if (llama_decode(target_ctx_, batch_tgt) != 0) {
        llama_batch_free(batch_tgt);
        llama_batch_free(batch_dft);
        throw std::runtime_error("speculative: target prefill failed");
    }

    batch_clear(batch_dft);
    for (int i = 0; i < n_prompt; ++i) {
        batch_add(batch_dft, prompt_tokens[i], i, i == n_prompt - 1);
    }
    if (llama_decode(draft_ctx_, batch_dft) != 0) {
        llama_batch_free(batch_tgt);
        llama_batch_free(batch_dft);
        throw std::runtime_error("speculative: draft prefill failed");
    }

    // Sample first token from target (logits at the last batch slot)
    int32_t id_last = SampleFromCtx(target_ctx_, batch_tgt.n_tokens - 1, temperature);

    std::string result;
    int n_past_tgt = n_prompt;
    int n_past_dft = n_prompt;
    int n_predict = 0;
    int n_drafted = 0;
    int n_accepted = 0;
    bool stopped = false;

    if (id_last < 0 || llama_vocab_is_eog(vocab, id_last)) {
        stopped = true;
    }

    // Output first token
    if (!stopped) {
        char buf[256];
        int n = llama_token_to_piece(vocab, id_last, buf, sizeof(buf), 0, false);
        if (n > 0) {
            buf[n] = '\0';
            result.append(buf, static_cast<size_t>(n));
            if (callback && !callback(id_last, buf)) stopped = true;
        }
        n_predict++;
    }

    // ── Main speculative loop ──
    while (!stopped && n_predict < max_tokens) {

        // ── Draft N tokens ──
        std::vector<int32_t> draft_tokens;
        draft_tokens.reserve(static_cast<size_t>(n_draft_));

        // Feed id_last to draft model
        batch_clear(batch_dft);
        batch_add(batch_dft, id_last, n_past_dft, true);
        if (llama_decode(draft_ctx_, batch_dft) != 0) break;
        n_past_dft++;

        int32_t draft_tok = SampleFromCtx(draft_ctx_, 0, temperature);
        if (draft_tok >= 0 && !llama_vocab_is_eog(vocab, draft_tok)) {
            draft_tokens.push_back(draft_tok);

            for (int d = 1; d < n_draft_; ++d) {
                batch_clear(batch_dft);
                batch_add(batch_dft, draft_tok, n_past_dft, true);
                if (llama_decode(draft_ctx_, batch_dft) != 0) break;
                n_past_dft++;

                draft_tok = SampleFromCtx(draft_ctx_, 0, temperature);
                if (draft_tok < 0 || llama_vocab_is_eog(vocab, draft_tok)) break;
                draft_tokens.push_back(draft_tok);
            }
        }

        n_drafted += static_cast<int>(draft_tokens.size());

        // ── Verify: feed [id_last, d0..dN-1] to target in ONE batch ──
        batch_clear(batch_tgt);
        batch_add(batch_tgt, id_last, n_past_tgt, true);
        for (int i = 0; i < static_cast<int>(draft_tokens.size()); ++i) {
            batch_add(batch_tgt, draft_tokens[i], n_past_tgt + 1 + i, true);
        }
        if (llama_decode(target_ctx_, batch_tgt) != 0) break;

        // ── Accept/Reject ──
        int accepted_this_step = 0;

        for (int i = 0; i < static_cast<int>(draft_tokens.size()); ++i) {
            int32_t target_tok = SampleFromCtx(target_ctx_, i, temperature);
            if (target_tok < 0) { stopped = true; break; }

            if (target_tok == draft_tokens[i]) {
                // Accept
                char buf[256];
                int n = llama_token_to_piece(vocab, target_tok, buf, sizeof(buf), 0, false);
                if (n > 0) {
                    buf[n] = '\0';
                    result.append(buf, static_cast<size_t>(n));
                    if (callback && !callback(target_tok, buf)) { stopped = true; break; }
                }
                n_predict++;
                accepted_this_step++;

                if (llama_vocab_is_eog(vocab, target_tok) || n_predict >= max_tokens) {
                    stopped = true;
                    break;
                }
            } else {
                // Reject: use target's token
                char buf[256];
                int n = llama_token_to_piece(vocab, target_tok, buf, sizeof(buf), 0, false);
                if (n > 0) {
                    buf[n] = '\0';
                    result.append(buf, static_cast<size_t>(n));
                    if (callback) callback(target_tok, buf);
                }
                id_last = target_tok;
                n_predict++;
                if (llama_vocab_is_eog(vocab, target_tok) || n_predict >= max_tokens) {
                    stopped = true;
                }
                break;
            }
        }

        // If all drafts accepted, sample bonus token from last target logit
        if (!stopped && accepted_this_step == static_cast<int>(draft_tokens.size())) {
            int32_t bonus = SampleFromCtx(target_ctx_,
                static_cast<int>(draft_tokens.size()), temperature);
            if (bonus >= 0 && !llama_vocab_is_eog(vocab, bonus)) {
                char buf[256];
                int n = llama_token_to_piece(vocab, bonus, buf, sizeof(buf), 0, false);
                if (n > 0) {
                    buf[n] = '\0';
                    result.append(buf, static_cast<size_t>(n));
                    if (callback) callback(bonus, buf);
                }
                id_last = bonus;
                n_predict++;
                accepted_this_step++;
            } else {
                stopped = true;
            }
        }

        n_accepted += accepted_this_step;

        // Update positions and clear rejected KV entries
        n_past_tgt += accepted_this_step;
        n_past_dft = n_past_tgt;  // re-sync draft

        llama_memory_seq_rm(llama_get_memory(target_ctx_), 0, n_past_tgt, -1);
        llama_memory_seq_rm(llama_get_memory(draft_ctx_), 0, n_past_dft, -1);
    }

    llama_batch_free(batch_tgt);
    llama_batch_free(batch_dft);

    if (out_n_predict)  *out_n_predict  = n_predict;
    if (out_n_drafted)  *out_n_drafted  = n_drafted;
    if (out_n_accepted) *out_n_accepted = n_accepted;
    return result;
}

} // namespace infergo
