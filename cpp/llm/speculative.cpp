#include "speculative.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

namespace infergo {

SpeculativeDecoder::SpeculativeDecoder() = default;

SpeculativeDecoder::~SpeculativeDecoder() {
    if (target_ctx_) { llama_free(target_ctx_); target_ctx_ = nullptr; }
    if (draft_ctx_)  { llama_free(draft_ctx_);  draft_ctx_ = nullptr; }
    if (draft_model_){ llama_model_free(draft_model_); draft_model_ = nullptr; }
}

bool SpeculativeDecoder::Init(const llama_model* target_model, int target_ctx_size,
                               const std::string& draft_path, int n_gpu_layers, int n_draft) {
    target_model_ = target_model;
    n_draft_      = n_draft;

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    draft_model_ = llama_model_load_from_file(draft_path.c_str(), mparams);
    if (!draft_model_) return false;

    const llama_vocab* vocab_tgt = llama_model_get_vocab(target_model_);
    const llama_vocab* vocab_dft = llama_model_get_vocab(draft_model_);
    if (std::abs(llama_vocab_n_tokens(vocab_tgt) - llama_vocab_n_tokens(vocab_dft)) > 128) {
        llama_model_free(draft_model_); draft_model_ = nullptr; return false;
    }
    vocab_size_ = std::min(llama_vocab_n_tokens(vocab_tgt), llama_vocab_n_tokens(vocab_dft));
    int ctx_size = target_ctx_size > 0 ? target_ctx_size : 4096;

    auto make_ctx = [&](const llama_model* m, int batch_sz) -> llama_context* {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = static_cast<uint32_t>(ctx_size); cp.n_batch = static_cast<uint32_t>(batch_sz);
        cp.n_ubatch = cp.n_batch; cp.n_seq_max = 1; cp.offload_kqv = true; cp.no_perf = true;
        return llama_init_from_model(const_cast<llama_model*>(m), cp);
    };

    target_ctx_ = make_ctx(target_model_, 512);
    if (!target_ctx_) { llama_model_free(draft_model_); draft_model_ = nullptr; return false; }
    draft_ctx_ = make_ctx(draft_model_, 512);
    if (!draft_ctx_) { llama_free(target_ctx_); target_ctx_ = nullptr;
                       llama_model_free(draft_model_); draft_model_ = nullptr; return false; }
    return true;
}

int32_t SpeculativeDecoder::SampleFromCtx(llama_context* ctx, int batch_idx, float temperature) {
    const float* logits = llama_get_logits_ith(ctx, batch_idx);
    if (!logits) return -1;
    if (temperature <= 0.0f) {
        int32_t best = 0;
        for (int i = 1; i < vocab_size_; ++i) if (logits[i] > logits[best]) best = i;
        return best;
    }
    thread_local std::mt19937 rng(std::random_device{}());
    float mx = *std::max_element(logits, logits + vocab_size_);
    std::vector<float> p(static_cast<size_t>(vocab_size_));
    float s = 0; for (int i = 0; i < vocab_size_; ++i) { p[i] = std::exp((logits[i]-mx)/temperature); s += p[i]; }
    for (auto& v : p) v /= s;
    return static_cast<int32_t>(std::discrete_distribution<int>(p.begin(), p.end())(rng));
}

static void batch_clear(llama_batch& b) { b.n_tokens = 0; }
static void batch_add(llama_batch& b, llama_token tok, llama_pos pos, llama_seq_id seq, bool logits) {
    int i = b.n_tokens++; b.token[i]=tok; b.pos[i]=pos; b.n_seq_id[i]=1; b.seq_id[i][0]=seq; b.logits[i]=logits?1:0;
}

std::string SpeculativeDecoder::Generate(
    const std::vector<int32_t>& prompt_tokens, int max_tokens, float temperature,
    const std::string& /*grammar_str*/, TokenCallback callback,
    int* out_n_predict, int* out_n_drafted, int* out_n_accepted)
{
    if (!draft_model_ || !target_ctx_) throw std::runtime_error("SpeculativeDecoder not initialized");
    const llama_vocab* vocab = llama_model_get_vocab(target_model_);
    const int n_prompt = static_cast<int>(prompt_tokens.size());

    // Clear KV caches from any previous generation.
    // llama_kv_self_clear removes ALL KV data from the context.
    llama_memory_clear(llama_get_memory(target_ctx_), true);
    llama_memory_clear(llama_get_memory(draft_ctx_), true);

    llama_batch bt = llama_batch_init(512, 0, 1), bd = llama_batch_init(512, 0, 1);

    // Prefill both models
    batch_clear(bt); for (int i = 0; i < n_prompt; ++i) batch_add(bt, prompt_tokens[i], i, 0, i==n_prompt-1);
    if (llama_decode(target_ctx_, bt) != 0) { llama_batch_free(bt); llama_batch_free(bd); throw std::runtime_error("prefill fail"); }
    batch_clear(bd); for (int i = 0; i < n_prompt; ++i) batch_add(bd, prompt_tokens[i], i, 0, i==n_prompt-1);
    if (llama_decode(draft_ctx_, bd) != 0) { llama_batch_free(bt); llama_batch_free(bd); throw std::runtime_error("draft prefill fail"); }

    int32_t id_last = SampleFromCtx(target_ctx_, bt.n_tokens - 1, temperature);
    std::string result; int n_past=n_prompt, n_past_d=n_prompt, n_pred=0, n_dft=0, n_acc=0; bool stop=false;
    if (id_last < 0 || llama_vocab_is_eog(vocab, id_last)) {
        stop = true;
    }
    if (!stop) {
        char b[256];
        int n = llama_token_to_piece(vocab, id_last, b, 256, 0, false);
        if (n > 0) {
            b[n] = 0;
            result.append(b, static_cast<size_t>(n));
            if (callback && !callback(id_last, b)) {
                stop = true;
            }
        }
        n_pred++;
    }

    while (!stop && n_pred < max_tokens) {
        std::vector<int32_t> draft;
        batch_clear(bd); batch_add(bd, id_last, n_past_d, 0, true);
        if (llama_decode(draft_ctx_, bd) != 0) break;
        n_past_d++;
        int32_t dt = SampleFromCtx(draft_ctx_, 0, temperature);
        if (dt >= 0 && !llama_vocab_is_eog(vocab, dt)) {
            draft.push_back(dt);
            for (int d = 1; d < n_draft_; ++d) {
                batch_clear(bd);
                batch_add(bd, dt, n_past_d, 0, true);
                if (llama_decode(draft_ctx_, bd) != 0) break;
                n_past_d++;
                dt = SampleFromCtx(draft_ctx_, 0, temperature);
                if (dt < 0 || llama_vocab_is_eog(vocab, dt)) break;
                draft.push_back(dt);
            }
        }
        n_dft += static_cast<int>(draft.size());

        batch_clear(bt); batch_add(bt, id_last, n_past, 0, true);
        for (int i=0; i<(int)draft.size(); ++i) batch_add(bt, draft[i], n_past+1+i, 0, true);
        if (llama_decode(target_ctx_, bt)!=0) break;

        int acc = 0;
        for (int i=0; i<(int)draft.size(); ++i) {
            int32_t tt = SampleFromCtx(target_ctx_, i, temperature);
            if (tt<0) { stop=true; break; }
            if (tt==draft[i]) {
                char b[256]; int n=llama_token_to_piece(vocab,tt,b,256,0,false);
                if(n>0){b[n]=0; result.append(b,n); if(callback&&!callback(tt,b)){stop=true;break;}}
                n_pred++; acc++;
                if (llama_vocab_is_eog(vocab,tt)||n_pred>=max_tokens) { stop=true; break; }
            } else {
                char b[256]; int n=llama_token_to_piece(vocab,tt,b,256,0,false);
                if(n>0){b[n]=0; result.append(b,n); if(callback)callback(tt,b);}
                id_last=tt; n_pred++;
                if (llama_vocab_is_eog(vocab,tt)||n_pred>=max_tokens) stop=true;
                break;
            }
        }
        if (!stop && acc==(int)draft.size()) {
            int32_t bonus=SampleFromCtx(target_ctx_,(int)draft.size(),temperature);
            if (bonus>=0&&!llama_vocab_is_eog(vocab,bonus)) {
                char b[256]; int n=llama_token_to_piece(vocab,bonus,b,256,0,false);
                if(n>0){b[n]=0; result.append(b,n); if(callback)callback(bonus,b);}
                id_last=bonus; n_pred++; acc++;
            } else stop=true;
        }
        n_acc+=acc; n_past+=acc; n_past_d=n_past;
        llama_memory_seq_rm(llama_get_memory(target_ctx_),0,n_past,-1);
        llama_memory_seq_rm(llama_get_memory(draft_ctx_),0,n_past_d,-1);
    }
    llama_batch_free(bt);
    llama_batch_free(bd);
    if (out_n_predict)  *out_n_predict  = n_pred;
    if (out_n_drafted)  *out_n_drafted  = n_dft;
    if (out_n_accepted) *out_n_accepted = n_acc;
    return result;
}

} // namespace infergo
