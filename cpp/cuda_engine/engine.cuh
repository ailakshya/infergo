#pragma once

#include <cstdint>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

namespace infergo {
namespace cuda {

// ─── Model config (Qwen 1.5B architecture) ──────────────────────────────────

constexpr float RMS_EPS = 1e-6f;

struct ModelConfig {
    int n_embd      = 1536;    // embedding dimension
    int n_head      = 12;      // attention heads
    int n_kv_head   = 2;       // grouped-query attention KV heads
    int n_layer     = 28;      // transformer layers
    int n_ff        = 8960;    // feed-forward hidden dim (SwiGLU: 2 * n_ff)
    int head_dim    = 128;     // n_embd / n_head
    int n_vocab     = 151936;  // vocabulary size
    int n_ctx       = 4096;    // max context length
    float rope_base = 1000000.0f; // RoPE frequency base
};

// ─── Quantized weight types ─────────────────────────────────────────────────

// Q4_K_M block: 256 values per block, 4-bit quantized with k-quant scales
struct Q4KBlock {
    half  d;           // super-block scale
    half  dmin;        // super-block min
    uint8_t scales[12]; // sub-block scales and mins
    uint8_t qs[128];   // 256 x 4-bit quantized values
};

// ─── KV Cache ────────────────────────────────────────────────────────────────

struct KVCache {
    half* k;     // [n_layer, n_kv_head, max_seq, head_dim] — contiguous
    half* v;     // [n_layer, n_kv_head, max_seq, head_dim]
    int max_seq; // allocated sequence length
    int n_layer;
    int n_kv_head;
    int head_dim;
};

// ─── Layer weights ───────────────────────────────────────────────────────────

struct LayerWeights {
    // Attention
    void* wq;        // Q4_K [n_embd, n_embd]
    void* wk;        // Q4_K [n_embd, n_kv_head * head_dim]
    void* wv;        // Q4_K [n_embd, n_kv_head * head_dim]
    void* wo;        // Q4_K [n_embd, n_embd]
    half* bq;        // bias [n_embd] (Qwen has attention biases)
    half* bk;        // bias [n_kv_head * head_dim]
    half* bv;        // bias [n_kv_head * head_dim]

    // FFN (SwiGLU)
    void* w_gate;    // Q4_K [n_embd, n_ff]
    void* w_up;      // Q4_K [n_embd, n_ff]
    void* w_down;    // Q4_K [n_ff, n_embd]

    // Norms
    half* attn_norm; // RMSNorm weights [n_embd]
    half* ffn_norm;  // RMSNorm weights [n_embd]
};

struct ModelWeights {
    half* tok_embd;      // [n_vocab, n_embd]
    LayerWeights* layers; // [n_layer]
    half* output_norm;   // RMSNorm [n_embd]
    void* output;        // Q4_K or F32 [n_embd, n_vocab]
    int n_layer;
};

// ─── Fused CUDA Kernels ─────────────────────────────────────────────────────

// Fused RMSNorm + Q4_K dequantize + GEMV for single-token decode.
// Combines: normalize → dequant weights → matrix-vector multiply
// in one kernel launch (vs 3 separate launches in llama.cpp).
void fused_rmsnorm_q4k_gemv(
    half* out,           // [out_dim]
    const half* input,   // [in_dim]
    const half* norm_w,  // [in_dim] RMSNorm weights
    const void* weight,  // Q4_K quantized [out_dim, in_dim]
    const half* bias,    // [out_dim] or nullptr
    int in_dim,
    int out_dim,
    float eps,
    cudaStream_t stream);

// Fused RoPE + Grouped-Query Attention for single token.
// Computes Q*K^T attention scores with RoPE positions, then softmax, then V.
// KV cache read + write in one kernel.
void fused_gqa_attention(
    half* out,           // [n_embd]
    const half* q,       // [n_head * head_dim] (already projected)
    const half* k_new,   // [n_kv_head * head_dim] (new token's K)
    const half* v_new,   // [n_kv_head * head_dim] (new token's V)
    KVCache* kv_cache,
    int layer_idx,
    int pos,             // position of new token
    int n_head,
    int n_kv_head,
    int head_dim,
    float rope_base,
    cudaStream_t stream);

// Fused SwiGLU: gate = silu(x * W_gate) * (x * W_up), out = gate * W_down
void fused_swiglu_ffn(
    half* out,           // [n_embd]
    const half* input,   // [n_embd]
    const half* norm_w,  // [n_embd] FFN norm weights
    const void* w_gate,  // Q4_K [n_embd, n_ff]
    const void* w_up,    // Q4_K [n_embd, n_ff]
    const void* w_down,  // Q4_K [n_ff, n_embd]
    int n_embd,
    int n_ff,
    float eps,
    cudaStream_t stream);

// ─── Engine API ──────────────────────────────────────────────────────────────

class CUDAEngine {
public:
    CUDAEngine();
    ~CUDAEngine();

    // Load GGUF model weights to GPU
    bool LoadModel(const char* gguf_path, const ModelConfig& config);

    // Generate tokens (single sequence, autoregressive)
    int Generate(const int* prompt_tokens, int n_prompt,
                 int max_tokens, float temperature,
                 int* out_tokens, int max_out);

    // Get generated text (call after Generate)
    const char* GetText() const { return text_.c_str(); }

private:
    ModelConfig config_;
    ModelWeights weights_;
    KVCache kv_cache_;
    cublasHandle_t cublas_;
    cudaStream_t stream_;
    std::string text_;

    // Workspace buffers (pre-allocated, reused across tokens)
    half* buf_hidden_;   // [n_embd] main hidden state
    half* buf_residual_; // [n_embd] residual connection
    half* buf_qkv_;      // [n_embd + 2 * n_kv_head * head_dim]
    half* buf_attn_out_; // [n_embd]
    half* buf_ffn_;      // [n_ff * 2] gate + up
    float* buf_logits_;  // [n_vocab] output logits

    void ForwardToken(int token, int pos);
    int SampleToken(float temperature);
};

} // namespace cuda
} // namespace infergo
