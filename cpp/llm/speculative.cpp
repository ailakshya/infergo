#include "speculative.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

namespace infergo {

SpeculativeDecoder::SpeculativeDecoder() = default;

SpeculativeDecoder::~SpeculativeDecoder() {
    if (draft_ctx_) {
        llama_free(draft_ctx_);
        draft_ctx_ = nullptr;
    }
    if (draft_model_) {
        llama_model_free(draft_model_);
        draft_model_ = nullptr;
    }
}

bool SpeculativeDecoder::Init(llama_model* target_model,
                               llama_context* target_ctx,
                               const std::string& draft_path,
                               int n_gpu_layers,
                               int n_draft) {
    target_model_ = target_model;
    target_ctx_   = target_ctx;
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

    // Create draft context — small, single-sequence
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = llama_n_ctx(target_ctx_);
    cparams.n_batch    = static_cast<uint32_t>(n_draft + 1);
    cparams.n_ubatch   = static_cast<uint32_t>(n_draft + 1);
    cparams.n_seq_max  = 1;
    cparams.offload_kqv = true;
    cparams.no_perf     = true;

    draft_ctx_ = llama_init_from_model(draft_model_, cparams);
    if (!draft_ctx_) {
        llama_model_free(draft_model_);
        draft_model_ = nullptr;
        return false;
    }

    return true;
}

int32_t SpeculativeDecoder::SampleFromCtx(llama_context* ctx, int batch_idx,
                                           float temperature) {
    const float* logits = llama_get_logits_ith(ctx, batch_idx);
    if (!logits) return -1;

    if (temperature <= 0.0f) {
        // Greedy argmax
        int32_t best = 0;
        for (int i = 1; i < vocab_size_; ++i) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }

    // Temperature + multinomial
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

std::string SpeculativeDecoder::Generate(
    const std::vector<int32_t>& prompt_tokens,
    int max_tokens,
    float temperature,
    const std::string& grammar_str,
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

    // ── Prefill: process prompt through both models ──

    // Target prefill (all prompt tokens except last)
    {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; ++i) {
            batch.token[i]    = prompt_tokens[i];
            batch.pos[i]      = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]   = (i == n_prompt - 1) ? 1 : 0;
            batch.n_tokens++;
        }
        if (llama_decode(target_ctx_, batch) != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("speculative: target prefill failed");
        }
        llama_batch_free(batch);
    }

    // Draft prefill (same prompt)
    {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; ++i) {
            batch.token[i]    = prompt_tokens[i];
            batch.pos[i]      = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]   = (i == n_prompt - 1) ? 1 : 0;
            batch.n_tokens++;
        }
        if (llama_decode(draft_ctx_, batch) != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("speculative: draft prefill failed");
        }
        llama_batch_free(batch);
    }

    // Sample first token from target model
    int32_t id_last = SampleFromCtx(target_ctx_, n_prompt - 1, temperature);
    if (id_last < 0 || llama_vocab_is_eog(vocab, id_last)) {
        if (out_n_predict) *out_n_predict = 0;
        if (out_n_drafted)  *out_n_drafted = 0;
        if (out_n_accepted) *out_n_accepted = 0;
        return "";
    }

    std::string result;
    int n_past_tgt = n_prompt;
    int n_past_dft = n_prompt;
    int n_predict = 0;
    int n_drafted = 0;
    int n_accepted = 0;

    // Output first token
    {
        char buf[256];
        int n = llama_token_to_piece(vocab, id_last, buf, sizeof(buf), 0, false);
        if (n > 0) {
            result.append(buf, static_cast<size_t>(n));
            if (callback && !callback(id_last, buf)) {
                goto done;
            }
        }
        n_predict++;
    }

    // ── Main speculative loop ──
    while (n_predict < max_tokens) {
        // ── Step 1: Draft N tokens using draft model ──
        std::vector<int32_t> draft_tokens;
        draft_tokens.reserve(static_cast<size_t>(n_draft_));

        {
            // First, decode id_last in draft model
            llama_batch batch = llama_batch_init(1, 0, 1);
            batch.token[0]    = id_last;
            batch.pos[0]      = n_past_dft;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0]   = 1;
            batch.n_tokens    = 1;

            if (llama_decode(draft_ctx_, batch) != 0) {
                llama_batch_free(batch);
                break;
            }
            llama_batch_free(batch);
            n_past_dft++;

            // Sample from draft model
            int32_t draft_tok = SampleFromCtx(draft_ctx_, 0, temperature);
            if (draft_tok < 0 || llama_vocab_is_eog(vocab, draft_tok)) {
                // Draft model hit EOS — just verify last token
            } else {
                draft_tokens.push_back(draft_tok);

                // Continue drafting
                for (int d = 1; d < n_draft_; ++d) {
                    llama_batch b1 = llama_batch_init(1, 0, 1);
                    b1.token[0]    = draft_tok;
                    b1.pos[0]      = n_past_dft;
                    b1.n_seq_id[0] = 1;
                    b1.seq_id[0][0] = 0;
                    b1.logits[0]   = 1;
                    b1.n_tokens    = 1;

                    if (llama_decode(draft_ctx_, b1) != 0) {
                        llama_batch_free(b1);
                        break;
                    }
                    llama_batch_free(b1);
                    n_past_dft++;

                    draft_tok = SampleFromCtx(draft_ctx_, 0, temperature);
                    if (draft_tok < 0 || llama_vocab_is_eog(vocab, draft_tok)) break;
                    draft_tokens.push_back(draft_tok);
                }
            }
        }

        n_drafted += static_cast<int>(draft_tokens.size());

        // ── Step 2: Verify with target model ──
        // Feed [id_last, d0, d1, ..., dN-1] to target in ONE batch
        {
            const int verify_len = 1 + static_cast<int>(draft_tokens.size());
            llama_batch batch = llama_batch_init(verify_len, 0, 1);

            batch.token[0]    = id_last;
            batch.pos[0]      = n_past_tgt;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0]   = 1;
            batch.n_tokens    = 1;

            for (int i = 0; i < static_cast<int>(draft_tokens.size()); ++i) {
                int idx = i + 1;
                batch.token[idx]    = draft_tokens[i];
                batch.pos[idx]      = n_past_tgt + idx;
                batch.n_seq_id[idx] = 1;
                batch.seq_id[idx][0] = 0;
                batch.logits[idx]   = 1;
                batch.n_tokens++;
            }

            if (llama_decode(target_ctx_, batch) != 0) {
                llama_batch_free(batch);
                break;
            }
            llama_batch_free(batch);
        }

        // ── Step 3: Accept/Reject ──
        // For each draft token, check if target model agrees
        int accepted_this_step = 0;
        bool stopped = false;

        for (int i = 0; i < static_cast<int>(draft_tokens.size()); ++i) {
            int32_t target_tok = SampleFromCtx(target_ctx_, i, temperature);
            if (target_tok < 0) break;

            if (target_tok == draft_tokens[i]) {
                // Accept: output this token
                char buf[256];
                int n = llama_token_to_piece(vocab, target_tok, buf, sizeof(buf), 0, false);
                if (n > 0) {
                    result.append(buf, static_cast<size_t>(n));
                    if (callback && !callback(target_tok, buf)) {
                        stopped = true;
                        break;
                    }
                }
                n_predict++;
                accepted_this_step++;

                if (llama_vocab_is_eog(vocab, target_tok) || n_predict >= max_tokens) {
                    stopped = true;
                    break;
                }
            } else {
                // Reject: use target's token instead, discard rest
                char buf[256];
                int n = llama_token_to_piece(vocab, target_tok, buf, sizeof(buf), 0, false);
                if (n > 0) {
                    result.append(buf, static_cast<size_t>(n));
                    if (callback && !callback(target_tok, buf)) {
                        stopped = true;
                        break;
                    }
                }
                id_last = target_tok;
                n_predict++;

                if (llama_vocab_is_eog(vocab, target_tok) || n_predict >= max_tokens) {
                    stopped = true;
                }
                break;  // Stop accepting from this draft
            }
        }

        // If all drafts accepted, sample one more from the last target logit
        if (!stopped && accepted_this_step == static_cast<int>(draft_tokens.size())) {
            int32_t bonus_tok = SampleFromCtx(target_ctx_,
                static_cast<int>(draft_tokens.size()), temperature);
            if (bonus_tok >= 0 && !llama_vocab_is_eog(vocab, bonus_tok)) {
                char buf[256];
                int n = llama_token_to_piece(vocab, bonus_tok, buf, sizeof(buf), 0, false);
                if (n > 0) {
                    result.append(buf, static_cast<size_t>(n));
                    if (callback) callback(bonus_tok, buf);
                }
                id_last = bonus_tok;
                n_predict++;
                accepted_this_step++;
            } else {
                stopped = true;
            }
        }

        n_accepted += accepted_this_step;

        if (stopped) break;

        // If we rejected, id_last is already set to the target's token
        // If all accepted, id_last is the bonus token
        // Update n_past for both models
        n_past_tgt += accepted_this_step;
        n_past_dft = n_past_tgt;  // Re-sync draft to target position

        // Clear rejected KV entries from both models
        llama_memory_seq_rm(llama_get_memory(target_ctx_), 0, n_past_tgt, -1);
        llama_memory_seq_rm(llama_get_memory(draft_ctx_), 0, n_past_dft, -1);
    }

done:
    if (out_n_predict)  *out_n_predict  = n_predict;
    if (out_n_drafted)  *out_n_drafted  = n_drafted;
    if (out_n_accepted) *out_n_accepted = n_accepted;
    return result;
}

} // namespace infergo
