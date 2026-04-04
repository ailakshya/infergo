#include "llm_engine.hpp"

#include <cstring>
#include <stdexcept>

namespace infergo {

// ─── Constructor / Destructor ─────────────────────────────────────────────────

LLMEngine::LLMEngine() = default;

LLMEngine::~LLMEngine() {
    if (batch_initialized_) {
        llama_batch_free(batch_);
        batch_initialized_ = false;
    }
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
}

// ─── LoadModel ────────────────────────────────────────────────────────────────

void LLMEngine::LoadModel(const std::string& path,
                           int  n_gpu_layers,
                           int  ctx_size,
                           int  n_seq_max,
                           int  n_batch)
{
    // Clean up any previously loaded model
    if (batch_initialized_) { llama_batch_free(batch_); batch_initialized_ = false; }
    if (ctx_)   { llama_free(ctx_);           ctx_   = nullptr; }
    if (model_) { llama_model_free(model_);   model_ = nullptr; }

    // Model params
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    model_ = llama_model_load_from_file(path.c_str(), mparams);
    if (model_ == nullptr) {
        throw std::runtime_error(
            "LLMEngine::LoadModel: failed to load model from '" + path + "'");
    }

    // Context params
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = static_cast<uint32_t>(ctx_size);
    cparams.n_batch    = static_cast<uint32_t>(n_batch);
    cparams.n_ubatch   = static_cast<uint32_t>(n_batch);
    cparams.n_seq_max  = static_cast<uint32_t>(n_seq_max);
    cparams.offload_kqv = true;   // keep KV cache on GPU
    cparams.no_perf     = true;   // skip perf counters

    ctx_ = llama_init_from_model(model_, cparams);
    if (ctx_ == nullptr) {
        llama_model_free(model_);
        model_ = nullptr;
        throw std::runtime_error("LLMEngine::LoadModel: failed to create llama_context");
    }

    // Pre-allocate batch
    n_batch_ = n_batch;
    batch_ = llama_batch_init(n_batch, /*embd=*/0, /*n_seq_max=*/1);
    batch_initialized_ = true;

    // Cache vocab size
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    vocab_size_ = llama_vocab_n_tokens(vocab);
}

// ─── LoadModelSplit ───────────────────────────────────────────────────────────

void LLMEngine::LoadModelSplit(const std::string& path,
                                int          n_gpu_layers,
                                int          ctx_size,
                                int          n_seq_max,
                                int          n_batch,
                                const float* tensor_split,
                                int          n_split)
{
    // Clean up any previously loaded model
    if (batch_initialized_) { llama_batch_free(batch_); batch_initialized_ = false; }
    if (ctx_)   { llama_free(ctx_);           ctx_   = nullptr; }
    if (model_) { llama_model_free(model_);   model_ = nullptr; }

    // Model params
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    // Apply tensor split when provided
    if (tensor_split != nullptr && n_split > 0) {
        mparams.tensor_split = tensor_split;
        // Use row-split so every GPU participates in every layer
        mparams.split_mode = LLAMA_SPLIT_MODE_ROW;
    }

    model_ = llama_model_load_from_file(path.c_str(), mparams);
    if (model_ == nullptr) {
        throw std::runtime_error(
            "LLMEngine::LoadModelSplit: failed to load model from '" + path + "'");
    }

    // Context params
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = static_cast<uint32_t>(ctx_size);
    cparams.n_batch    = static_cast<uint32_t>(n_batch);
    cparams.n_ubatch   = static_cast<uint32_t>(n_batch);
    cparams.n_seq_max  = static_cast<uint32_t>(n_seq_max);
    cparams.offload_kqv = true;
    cparams.no_perf     = true;

    ctx_ = llama_init_from_model(model_, cparams);
    if (ctx_ == nullptr) {
        llama_model_free(model_);
        model_ = nullptr;
        throw std::runtime_error("LLMEngine::LoadModelSplit: failed to create llama_context");
    }

    // Pre-allocate batch
    n_batch_ = n_batch;
    batch_ = llama_batch_init(n_batch, /*embd=*/0, /*n_seq_max=*/1);
    batch_initialized_ = true;

    // Cache vocab size
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    vocab_size_ = llama_vocab_n_tokens(vocab);
}

// ─── BatchDecode ─────────────────────────────────────────────────────────────

std::vector<SequenceLogits> LLMEngine::BatchDecode(
    const std::vector<SequenceInput>& inputs)
{
    if (model_ == nullptr || ctx_ == nullptr) {
        throw std::runtime_error("LLMEngine::BatchDecode: model not loaded");
    }
    if (inputs.empty()) {
        return {};
    }

    // Count total tokens across all inputs
    int total_tokens = 0;
    for (const auto& seq : inputs) {
        total_tokens += static_cast<int>(seq.tokens.size());
    }
    if (total_tokens > n_batch_) {
        throw std::runtime_error(
            "LLMEngine::BatchDecode: total tokens (" + std::to_string(total_tokens) +
            ") exceeds n_batch (" + std::to_string(n_batch_) + ")");
    }

    // Fill batch
    batch_.n_tokens = 0;
    for (const auto& seq : inputs) {
        const int n = static_cast<int>(seq.tokens.size());
        for (int i = 0; i < n; ++i) {
            const int idx = batch_.n_tokens;
            batch_.token   [idx]    = static_cast<llama_token>(seq.tokens[i]);
            batch_.pos     [idx]    = static_cast<llama_pos>(seq.pos + i);
            batch_.n_seq_id[idx]    = 1;
            batch_.seq_id  [idx][0] = static_cast<llama_seq_id>(seq.seq_id);
            // Only request logits for the last token of each sequence
            batch_.logits  [idx]    = (seq.want_logits && i == n - 1) ? 1 : 0;
            batch_.n_tokens++;
        }
    }

    // Run decode
    const int rc = llama_decode(ctx_, batch_);
    if (rc != 0) {
        throw std::runtime_error(
            "LLMEngine::BatchDecode: llama_decode returned " + std::to_string(rc));
    }

    // Collect logits.
    // llama_get_logits_ith(ctx, i) returns logits for batch slot i.
    // Slots where logits=0 return null — we must use the actual slot index
    // of each sequence's last token, not a separate output counter.
    std::vector<SequenceLogits> results;
    int entry = 0;
    for (const auto& seq : inputs) {
        const int n = static_cast<int>(seq.tokens.size());
        if (seq.want_logits) {
            const int last_slot = entry + n - 1;  // batch slot of the last token
            const float* raw = llama_get_logits_ith(ctx_, last_slot);
            if (raw == nullptr) {
                throw std::runtime_error(
                    "LLMEngine::BatchDecode: null logits for seq_id=" +
                    std::to_string(seq.seq_id));
            }
            SequenceLogits sl;
            sl.seq_id = seq.seq_id;
            sl.logits.assign(raw, raw + vocab_size_);
            results.push_back(std::move(sl));
        }
        entry += n;
    }

    return results;
}

// ─── Accessors ────────────────────────────────────────────────────────────────

int LLMEngine::VocabSize() const noexcept {
    return vocab_size_;
}

int32_t LLMEngine::BOS() const noexcept {
    if (model_ == nullptr) return -1;
    return static_cast<int32_t>(
        llama_vocab_bos(llama_model_get_vocab(model_)));
}

int32_t LLMEngine::EOS() const noexcept {
    if (model_ == nullptr) return -1;
    return static_cast<int32_t>(
        llama_vocab_eos(llama_model_get_vocab(model_)));
}

bool LLMEngine::IsEOG(int32_t token) const noexcept {
    if (model_ == nullptr) return false;
    return llama_vocab_is_eog(
        llama_model_get_vocab(model_),
        static_cast<llama_token>(token));
}

// ─── Tokenize ─────────────────────────────────────────────────────────────────

std::vector<int32_t> LLMEngine::Tokenize(const std::string& text, bool add_bos) const {
    if (model_ == nullptr) {
        throw std::runtime_error("LLMEngine::Tokenize: model not loaded");
    }
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    // First call with null buffer: returns -(number of tokens needed)
    int32_t n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                                nullptr, 0, add_bos, /*parse_special=*/true);
    if (n == 0) return {};
    if (n > 0) {
        // Unexpectedly fit in zero buffer — shouldn't happen, but handle it
        n = -n;
    }
    std::vector<llama_token> tokens(static_cast<size_t>(-n));
    const int32_t rc = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                                       tokens.data(), -n, add_bos, /*parse_special=*/true);
    if (rc < 0) {
        throw std::runtime_error("LLMEngine::Tokenize: tokenization failed");
    }
    return std::vector<int32_t>(tokens.begin(), tokens.begin() + rc);
}

// ─── TokenToPiece ─────────────────────────────────────────────────────────────

std::string LLMEngine::TokenToPiece(int32_t token) const {
    if (model_ == nullptr) {
        throw std::runtime_error("LLMEngine::TokenToPiece: model not loaded");
    }
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    char buf[256];
    const int32_t n = llama_token_to_piece(vocab, static_cast<llama_token>(token),
                                            buf, static_cast<int32_t>(sizeof(buf)),
                                            /*lstrip=*/0, /*special=*/false);
    if (n < 0) {
        throw std::runtime_error("LLMEngine::TokenToPiece: buffer too small");
    }
    return std::string(buf, static_cast<size_t>(n));
}

} // namespace infergo
