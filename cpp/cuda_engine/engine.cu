// infergo Custom CUDA Engine — forward pass + sampling
#include "engine.cuh"
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

namespace infergo {
namespace cuda {

// ─── Helper kernels ──────────────────────────────────────────────────────────

// Residual add: out[i] += residual[i]
__global__ void kernel_residual_add(half* out, const half* residual, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2half(__half2float(out[i]) + __half2float(residual[i]));
    }
}

// Copy hidden state to residual buffer
__global__ void kernel_copy(half* dst, const half* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// Embedding lookup: copy row from embedding table
__global__ void kernel_embed_lookup(half* out, const half* table, int token, int n_embd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_embd) {
        out[i] = table[token * n_embd + i];
    }
}

// RMSNorm standalone (for final norm before output)
__global__ void kernel_rmsnorm(half* out, const half* input, const half* weight, int n, float eps) {
    extern __shared__ float smem[];

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = __half2float(input[i]);
        sum_sq += v * v;
    }

    // Block reduce
    smem[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    float rms_inv = rsqrtf(smem[0] / (float)n + eps);

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = __float2half(__half2float(input[i]) * __half2float(weight[i]) * rms_inv);
    }
}

// Simple Q4K GEMV without fused norm (for output projection)
__global__ void kernel_q4k_gemv_no_norm(
    float* __restrict__ out,
    const half* __restrict__ input,
    const Q4KBlock* __restrict__ weight,
    int in_dim, int out_dim)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    if (row >= out_dim) return;

    int blocks_per_row = in_dim / 256;
    const Q4KBlock* row_blocks = weight + row * blocks_per_row;

    float sum = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
        float d = __half2float(row_blocks[b].d);
        float dmin = __half2float(row_blocks[b].dmin);

        for (int j = lane; j < 256; j += 32) {
            int sub = j / 32;
            uint8_t sc, m;
            if (sub < 4) {
                sc = row_blocks[b].scales[sub] & 0x3F;
                m = row_blocks[b].scales[sub + 4] & 0x3F;
            } else {
                sc = ((row_blocks[b].scales[sub + 4] & 0xF) | ((row_blocks[b].scales[sub - 4] >> 6) << 4));
                m = ((row_blocks[b].scales[sub + 4] >> 4) | ((row_blocks[b].scales[sub] >> 6) << 4));
            }
            float scale = d * sc;
            float min_val = dmin * m;

            int byte_idx = (j / 2);
            if (sub >= 4) byte_idx = 64 + (sub - 4) * 16 + (j % 32) / 2;
            else byte_idx = sub * 16 + (j % 32) / 2;

            uint8_t byte = row_blocks[b].qs[byte_idx];
            int nibble = (j & 1) ? (byte >> 4) : (byte & 0xF);
            float w = scale * nibble - min_val;

            int idx = b * 256 + j;
            if (idx < in_dim) {
                sum += w * __half2float(input[idx]);
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (lane == 0) out[row] = sum;
}

// ─── CUDAEngine ──────────────────────────────────────────────────────────────

CUDAEngine::CUDAEngine() {
    cublasCreate(&cublas_);
    cudaStreamCreate(&stream_);
    set_cublas_handle(cublas_);
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
    cudaFree(buf_hidden_);
    cudaFree(buf_residual_);
    cudaFree(buf_qkv_);
    cudaFree(buf_attn_out_);
    cudaFree(buf_ffn_);
    cudaFree(buf_logits_);
    cudaFree(kv_cache_.k);
    cudaFree(kv_cache_.v);
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

void CUDAEngine::ForwardToken(int token, int pos) {
    int n = config_.n_embd;
    int n_kv = config_.n_kv_head * config_.head_dim;
    int threads = 256;

    // 1. Embedding lookup
    kernel_embed_lookup<<<(n + threads - 1) / threads, threads, 0, stream_>>>(
        buf_hidden_, weights_.tok_embd, token, n);

    for (int layer = 0; layer < config_.n_layer; layer++) {
        auto& lw = weights_.layers[layer];

        // Save residual
        kernel_copy<<<(n + threads - 1) / threads, threads, 0, stream_>>>(
            buf_residual_, buf_hidden_, n);

        // 2. QKV projections with type dispatch
        // Macro to call the right GEMV based on weight type
        #define GEMV_DISPATCH(out, in, norm, w, bias, is_f16, in_d, out_d, ep) \
            if (is_f16) \
                f16_gemv(out, in, norm, reinterpret_cast<const half*>(w), bias, in_d, out_d, ep, stream_); \
            else \
                fused_rmsnorm_q4k_gemv(out, in, norm, w, bias, in_d, out_d, ep, stream_);

        half* q_out = buf_qkv_;
        GEMV_DISPATCH(q_out, buf_hidden_, lw.attn_norm, lw.wq, lw.bq, lw.wq_f16, n, n, RMS_EPS);

        half* k_out = buf_qkv_ + n;
        GEMV_DISPATCH(k_out, buf_hidden_, lw.attn_norm, lw.wk, lw.bk, lw.wk_f16, n, n_kv, RMS_EPS);

        half* v_out = k_out + n_kv;
        GEMV_DISPATCH(v_out, buf_hidden_, lw.attn_norm, lw.wv, lw.bv, lw.wv_f16, n, n_kv, RMS_EPS);

        // 3. Fused GQA Attention
        fused_gqa_attention(buf_attn_out_, q_out, k_out, v_out,
                           &kv_cache_, layer, pos,
                           config_.n_head, config_.n_kv_head, config_.head_dim,
                           config_.rope_base, stream_);

        // 4. Output projection (no norm)
        GEMV_DISPATCH(buf_hidden_, buf_attn_out_, nullptr, lw.wo, nullptr, lw.wo_f16, n, n, 1e30f);

        // 5. Residual add
        kernel_residual_add<<<(n + threads - 1) / threads, threads, 0, stream_>>>(
            buf_hidden_, buf_residual_, n);

        // Save residual for FFN
        kernel_copy<<<(n + threads - 1) / threads, threads, 0, stream_>>>(
            buf_residual_, buf_hidden_, n);

        // 6. Fused SwiGLU FFN with type dispatch
        half* ffn_out = buf_ffn_;
        fused_swiglu_ffn(ffn_out, buf_hidden_, lw.ffn_norm,
                        lw.w_gate, lw.w_up, lw.w_down,
                        lw.w_gate_f16, lw.w_up_f16, lw.w_down_f16,
                        n, config_.n_ff, RMS_EPS, stream_);

        // 7. Copy FFN output to hidden + residual
        cudaMemcpyAsync(buf_hidden_, ffn_out, n * sizeof(half),
                        cudaMemcpyDeviceToDevice, stream_);
        kernel_residual_add<<<(n + threads - 1) / threads, threads, 0, stream_>>>(
            buf_hidden_, buf_residual_, n);

        if (layer == 0 && pos == 0) {
            cudaStreamSynchronize(stream_);
            half dbg[8];
            cudaMemcpy(dbg, buf_hidden_, 8 * sizeof(half), cudaMemcpyDeviceToHost);
            printf("[debug] After layer 0: ");
            for (int i = 0; i < 8; i++) printf("%.4f ", __half2float(dbg[i]));
            printf("\n");
            cudaMemcpy(dbg, buf_qkv_, 8 * sizeof(half), cudaMemcpyDeviceToHost);
            printf("[debug] Q proj[0:8]: ");
            for (int i = 0; i < 8; i++) printf("%.4f ", __half2float(dbg[i]));
            printf("\n");
        }
    }

    // 8. Final RMSNorm
    kernel_rmsnorm<<<1, 256, 256 * sizeof(float), stream_>>>(
        buf_hidden_, buf_hidden_, weights_.output_norm, n, RMS_EPS);

    // 9. Output logits with type dispatch
    half* logits_half = reinterpret_cast<half*>(buf_logits_);
    if (weights_.output_f16) {
        f16_gemv(logits_half, buf_hidden_, nullptr,
                 reinterpret_cast<const half*>(weights_.output), nullptr,
                 n, config_.n_vocab, 1e30f, stream_);
    } else {
        fused_rmsnorm_q4k_gemv(logits_half, buf_hidden_, nullptr,
                               weights_.output, nullptr,
                               n, config_.n_vocab, 1e30f, stream_);
    }

    cudaStreamSynchronize(stream_);
}

int CUDAEngine::SampleToken(float temperature) {
    // Copy F16 logits from buf_ffn_ to CPU and convert to float
    std::vector<half> h_logits(config_.n_vocab);
    cudaMemcpy(h_logits.data(), buf_logits_,
               config_.n_vocab * sizeof(half), cudaMemcpyDeviceToHost);
    std::vector<float> logits(config_.n_vocab);
    for (int i = 0; i < config_.n_vocab; i++)
        logits[i] = __half2float(h_logits[i]);

    if (temperature <= 0.0f) {
        // Greedy: argmax
        return static_cast<int>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
    }

    // Temperature + softmax + top-p sampling
    float max_l = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (auto& l : logits) {
        l = expf((l - max_l) / temperature);
        sum += l;
    }
    for (auto& l : logits) l /= sum;

    // Random sample
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    for (int i = 0; i < (int)logits.size(); i++) {
        cumsum += logits[i];
        if (cumsum >= r) return i;
    }
    return config_.n_vocab - 1;
}

int CUDAEngine::Generate(const int* prompt_tokens, int n_prompt,
                          int max_tokens, float temperature,
                          int* out_tokens, int max_out) {
    // Reset KV cache
    size_t kv_size = (size_t)config_.n_layer * config_.n_kv_head *
                     config_.n_ctx * config_.head_dim * sizeof(half);
    cudaMemsetAsync(kv_cache_.k, 0, kv_size, stream_);
    cudaMemsetAsync(kv_cache_.v, 0, kv_size, stream_);

    // Prefill
    for (int i = 0; i < n_prompt; i++) {
        ForwardToken(prompt_tokens[i], i);
    }

    // Autoregressive decode
    int gen = 0;
    int pos = n_prompt;
    int eos_token = 151645;  // <|im_end|> for Qwen

    for (int t = 0; t < max_tokens && gen < max_out; t++) {
        int tok = SampleToken(temperature);
        if (tok < 0 || tok == eos_token) break;
        out_tokens[gen++] = tok;
        ForwardToken(tok, pos++);
    }

    return gen;
}

void CUDAEngine::DebugWeights() {
    printf("\n[debug] Weight check:\n");

    // Token embedding
    if (weights_.tok_embd) {
        half buf[16];
        cudaMemcpy(buf, weights_.tok_embd, 16 * sizeof(half), cudaMemcpyDeviceToHost);
        printf("  token_embd[0:8]: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", __half2float(buf[i]));

        // Check row 1 (token 1)
        cudaMemcpy(buf, weights_.tok_embd + config_.n_embd, 8 * sizeof(half), cudaMemcpyDeviceToHost);
        printf("\n  token_embd[row1]: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", __half2float(buf[i]));

        // Count nonzero in row 0
        std::vector<half> row(config_.n_embd);
        cudaMemcpy(row.data(), weights_.tok_embd, config_.n_embd * sizeof(half), cudaMemcpyDeviceToHost);
        int nz = 0;
        for (int i = 0; i < config_.n_embd; i++) if (__half2float(row[i]) != 0.0f) nz++;
        printf("\n  row 0: %d/%d nonzero\n", nz, config_.n_embd);
    } else printf("  token_embd: NULL!\n");

    // Attn norm (should be F32 → uploaded as raw bytes, need to interpret correctly)
    if (weights_.layers[0].attn_norm) {
        // The norm weights are F32 in GGUF but we uploaded raw bytes.
        // Our kernel expects half*. This is a BUG — we need F32→F16 conversion!
        float f32buf[8];
        cudaMemcpy(f32buf, weights_.layers[0].attn_norm, 8 * sizeof(float), cudaMemcpyDeviceToHost);
        printf("  blk.0.attn_norm (as F32): ");
        for (int i = 0; i < 8; i++) printf("%.4f ", f32buf[i]);
        printf("\n");
    } else printf("  blk.0.attn_norm: NULL!\n");

    // Q4_K weight block
    if (weights_.layers[0].wq) {
        unsigned char raw[144]; // sizeof Q4_K block
        cudaMemcpy(raw, weights_.layers[0].wq, 144, cudaMemcpyDeviceToHost);
        printf("  blk.0.wq first Q4K block raw[0:16]: ");
        for (int i = 0; i < 16; i++) printf("%02x ", raw[i]);
        printf("\n");
    } else printf("  blk.0.wq: NULL!\n");
}

} // namespace cuda
} // namespace infergo
