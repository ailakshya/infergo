// infergo Custom CUDA Engine — experimental
// Loads GGUF Q4_K models and runs inference with fused CUDA kernels.
// Goal: beat llama.cpp on autoregressive decode throughput.

#include "engine.cuh"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

namespace infergo {
namespace cuda {

// ─── GGUF Parser (minimal, Q4_K only) ────────────────────────────────────────

struct GGUFHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
};

struct GGUFTensorInfo {
    std::string name;
    int n_dims;
    int64_t dims[4];
    int type;       // GGML type enum
    uint64_t offset;
};

static const int GGML_TYPE_F32 = 0;
static const int GGML_TYPE_F16 = 1;
static const int GGML_TYPE_Q4_K = 12;

// ─── CUDAEngine Implementation ──────────────────────────────────────────────

CUDAEngine::CUDAEngine() {
    cublasCreate(&cublas_);
    cudaStreamCreate(&stream_);
    memset(&weights_, 0, sizeof(weights_));
    memset(&kv_cache_, 0, sizeof(kv_cache_));
    buf_hidden_ = nullptr;
    buf_residual_ = nullptr;
    buf_qkv_ = nullptr;
    buf_attn_out_ = nullptr;
    buf_ffn_ = nullptr;
    buf_logits_ = nullptr;
}

CUDAEngine::~CUDAEngine() {
    // Free workspace
    cudaFree(buf_hidden_);
    cudaFree(buf_residual_);
    cudaFree(buf_qkv_);
    cudaFree(buf_attn_out_);
    cudaFree(buf_ffn_);
    cudaFree(buf_logits_);

    // Free KV cache
    cudaFree(kv_cache_.k);
    cudaFree(kv_cache_.v);

    // Free weights
    if (weights_.layers) {
        for (int i = 0; i < weights_.n_layer; i++) {
            auto& l = weights_.layers[i];
            cudaFree(l.wq); cudaFree(l.wk); cudaFree(l.wv); cudaFree(l.wo);
            cudaFree(l.bq); cudaFree(l.bk); cudaFree(l.bv);
            cudaFree(l.w_gate); cudaFree(l.w_up); cudaFree(l.w_down);
            cudaFree(l.attn_norm); cudaFree(l.ffn_norm);
        }
        delete[] weights_.layers;
    }
    cudaFree(weights_.tok_embd);
    cudaFree(weights_.output_norm);
    cudaFree(weights_.output);

    cublasDestroy(cublas_);
    cudaStreamDestroy(stream_);
}

bool CUDAEngine::LoadModel(const char* path, const ModelConfig& config) {
    config_ = config;

    // Allocate workspace buffers
    int n = config_.n_embd;
    int n_kv = config_.n_kv_head * config_.head_dim;
    cudaMalloc(&buf_hidden_, n * sizeof(half));
    cudaMalloc(&buf_residual_, n * sizeof(half));
    cudaMalloc(&buf_qkv_, (n + 2 * n_kv) * sizeof(half));
    cudaMalloc(&buf_attn_out_, n * sizeof(half));
    cudaMalloc(&buf_ffn_, config_.n_ff * 2 * sizeof(half));
    cudaMalloc(&buf_logits_, config_.n_vocab * sizeof(float));

    // Allocate KV cache
    kv_cache_.max_seq = config_.n_ctx;
    kv_cache_.n_layer = config_.n_layer;
    kv_cache_.n_kv_head = config_.n_kv_head;
    kv_cache_.head_dim = config_.head_dim;
    size_t kv_size = (size_t)config_.n_layer * config_.n_kv_head *
                     config_.n_ctx * config_.head_dim * sizeof(half);
    cudaMalloc(&kv_cache_.k, kv_size);
    cudaMalloc(&kv_cache_.v, kv_size);
    cudaMemset(kv_cache_.k, 0, kv_size);
    cudaMemset(kv_cache_.v, 0, kv_size);

    // Allocate layer weights
    weights_.n_layer = config_.n_layer;
    weights_.layers = new LayerWeights[config_.n_layer];
    memset(weights_.layers, 0, sizeof(LayerWeights) * config_.n_layer);

    printf("[cuda_engine] Workspace + KV cache allocated (%.1f MB)\n",
           (float)(kv_size * 2 + n * 6 * sizeof(half) + config_.n_vocab * sizeof(float)) / 1e6);

    // TODO: Parse GGUF and load tensor data to GPU
    // For now, print what we'd need to load
    printf("[cuda_engine] Model config: n_embd=%d n_head=%d n_kv_head=%d n_layer=%d n_ff=%d\n",
           config_.n_embd, config_.n_head, config_.n_kv_head, config_.n_layer, config_.n_ff);
    printf("[cuda_engine] GGUF loading: %s (TODO — placeholder)\n", path);

    return true;
}

// Forward one token through the transformer
void CUDAEngine::ForwardToken(int token, int pos) {
    // 1. Token embedding lookup
    // Copy embedding for this token to buf_hidden_
    half* embd_row = weights_.tok_embd + token * config_.n_embd;
    cudaMemcpyAsync(buf_hidden_, embd_row, config_.n_embd * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream_);
    cudaMemcpyAsync(buf_residual_, buf_hidden_, config_.n_embd * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream_);

    for (int layer = 0; layer < config_.n_layer; layer++) {
        auto& lw = weights_.layers[layer];

        // 2. Fused RMSNorm + Q/K/V projections
        //    3 kernel launches instead of 6 (norm + 3 matmuls + 2 bias adds)
        int n_kv = config_.n_kv_head * config_.head_dim;

        // Q projection: [n_embd] → [n_embd]
        half* q_out = buf_qkv_;
        fused_rmsnorm_q4k_gemv(q_out, buf_hidden_, lw.attn_norm,
                                lw.wq, lw.bq,
                                config_.n_embd, config_.n_embd, RMS_EPS, stream_);

        // K projection: [n_embd] → [n_kv_head * head_dim]
        half* k_out = buf_qkv_ + config_.n_embd;
        fused_rmsnorm_q4k_gemv(k_out, buf_hidden_, lw.attn_norm,
                                lw.wk, lw.bk,
                                config_.n_embd, n_kv, RMS_EPS, stream_);

        // V projection: [n_embd] → [n_kv_head * head_dim]
        half* v_out = k_out + n_kv;
        fused_rmsnorm_q4k_gemv(v_out, buf_hidden_, lw.attn_norm,
                                lw.wv, lw.bv,
                                config_.n_embd, n_kv, RMS_EPS, stream_);

        // 3. Fused RoPE + GQA Attention (1 kernel launch)
        fused_gqa_attention(buf_attn_out_, q_out, k_out, v_out,
                           &kv_cache_, layer, pos,
                           config_.n_head, config_.n_kv_head, config_.head_dim,
                           config_.rope_base, stream_);

        // 4. Output projection: attn_out → hidden (1 fused kernel)
        fused_rmsnorm_q4k_gemv(buf_hidden_, buf_attn_out_, nullptr,
                                lw.wo, nullptr,
                                config_.n_embd, config_.n_embd, 1e30f, stream_);

        // 5. Residual connection
        // buf_hidden_ += buf_residual_
        // (TODO: fuse this into the output projection kernel)

        // 6. Fused SwiGLU FFN (2 kernel launches)
        //    Reads hidden once, computes gate+up, then down projection
        fused_swiglu_ffn(buf_ffn_, buf_hidden_, lw.ffn_norm,
                        lw.w_gate, lw.w_up, lw.w_down,
                        config_.n_embd, config_.n_ff, RMS_EPS, stream_);

        // 7. Residual
        // buf_hidden_ = buf_ffn_ + buf_residual_ (TODO: fuse)
    }

    // 8. Final RMSNorm + output projection → logits
    // fused_rmsnorm_q4k_gemv(buf_logits_, buf_hidden_, weights_.output_norm,
    //                        weights_.output, nullptr,
    //                        config_.n_embd, config_.n_vocab, RMS_EPS, stream_);
}

int CUDAEngine::Generate(const int* prompt_tokens, int n_prompt,
                          int max_tokens, float temperature,
                          int* out_tokens, int max_out) {
    // Prefill: process all prompt tokens
    for (int i = 0; i < n_prompt; i++) {
        ForwardToken(prompt_tokens[i], i);
    }

    // Autoregressive decode
    int gen = 0;
    int pos = n_prompt;
    for (int t = 0; t < max_tokens && gen < max_out; t++) {
        int tok = SampleToken(temperature);
        if (tok < 0 || tok == 151645) break;  // EOS = <|im_end|>
        out_tokens[gen++] = tok;
        ForwardToken(tok, pos++);
    }
    return gen;
}

int CUDAEngine::SampleToken(float temperature) {
    // TODO: Copy logits from GPU, apply temperature, sample
    // For now, return -1 (placeholder)
    return -1;
}

} // namespace cuda
} // namespace infergo
