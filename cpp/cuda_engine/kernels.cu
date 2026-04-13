// Custom CUDA kernels for infergo — fused operations for minimal launch overhead.
// Target: RTX 5070 Ti (SM 12.0, 896 GB/s bandwidth, 16 GB VRAM)
//
// Philosophy: For small models (1-3B), autoregressive decode is memory-bandwidth
// bound. Each token reads the full model weights (~1 GB). The GPU has 896 GB/s
// bandwidth = 862 tok/s theoretical max. llama.cpp achieves ~37% utilization.
//
// Our approach: FUSE operations to reduce kernel launch overhead and maximize
// memory coalescing. One kernel launch reads weights, dequantizes, normalizes,
// and computes the matmul — vs 3+ separate launches in llama.cpp.

#include "engine.cuh"
#include <cmath>

namespace infergo {
namespace cuda {

// ─── Constants ───────────────────────────────────────────────────────────────

constexpr int WARP_SIZE = 32;
constexpr int Q4K_BLOCK_SIZE = 256;  // values per Q4_K block

// ─── Q4_K Dequantization ─────────────────────────────────────────────────────

// Dequantize one Q4_K block element inline
__device__ __forceinline__ float dequant_q4k(const Q4KBlock* block, int idx) {
    // Q4_K_M: 256 values split into 8 sub-blocks of 32
    // Each sub-block has its own scale and min
    int sub_block = idx / 32;
    int sub_idx = idx % 32;

    // Extract scale and min for this sub-block
    float d = __half2float(block->d);
    float dmin = __half2float(block->dmin);

    uint8_t sc_packed;
    uint8_t m_packed;
    if (sub_block < 4) {
        sc_packed = block->scales[sub_block] & 0x3F;
        m_packed = block->scales[sub_block + 4] & 0x3F;
    } else {
        sc_packed = ((block->scales[sub_block + 4] & 0xF) | ((block->scales[sub_block - 4] >> 6) << 4));
        m_packed = ((block->scales[sub_block + 4] >> 4) | ((block->scales[sub_block] >> 6) << 4));
    }

    float scale = d * sc_packed;
    float min_val = dmin * m_packed;

    // Extract 4-bit quantized value
    int byte_idx = sub_block * 16 + sub_idx / 2;
    uint8_t byte = block->qs[byte_idx];
    int nibble = (sub_idx & 1) ? (byte >> 4) : (byte & 0xF);

    return scale * nibble - min_val;
}

// ─── Fused RMSNorm + Q4K GEMV ───────────────────────────────────────────────
//
// One kernel does:
//   1. Compute RMSNorm of input
//   2. For each output row: read Q4K weights, dequantize, dot product with normalized input
//   3. Add bias (optional)
//
// Each warp handles one output dimension.
// Block handles 4 output dimensions (4 warps per block).

__global__ void kernel_fused_rmsnorm_q4k_gemv(
    half* __restrict__ out,
    const half* __restrict__ input,
    const half* __restrict__ norm_w,
    const Q4KBlock* __restrict__ weight,  // Q4_K blocks [out_dim * in_dim / 256]
    const half* __restrict__ bias,
    int in_dim,
    int out_dim,
    float eps)
{
    const int row = blockIdx.x * blockDim.y + threadIdx.y;  // output dimension
    const int lane = threadIdx.x;  // within warp

    if (row >= out_dim) return;

    // Step 1: Compute RMSNorm (shared across all warps in block via first warp)
    // Actually, since input is shared, compute per-warp partial sums
    __shared__ float s_rms_inv;

    if (threadIdx.y == 0) {
        float sum_sq = 0.0f;
        for (int i = lane; i < in_dim; i += WARP_SIZE) {
            float v = __half2float(input[i]);
            sum_sq += v * v;
        }
        // Warp reduce
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
        }
        if (lane == 0) {
            s_rms_inv = rsqrtf(sum_sq / (float)in_dim + eps);
        }
    }
    __syncthreads();
    float rms_inv = s_rms_inv;

    // Step 2: Q4K dequant + dot product
    // Weight layout: each row of out_dim has in_dim/256 Q4K blocks
    int blocks_per_row = in_dim / Q4K_BLOCK_SIZE;
    const Q4KBlock* row_blocks = weight + row * blocks_per_row;

    float sum = 0.0f;
    // Each lane processes a stride of the input
    for (int b = 0; b < blocks_per_row; b++) {
        const Q4KBlock* block = &row_blocks[b];
        int base = b * Q4K_BLOCK_SIZE;

        // Each lane handles 256/32 = 8 elements per block
        for (int j = lane; j < Q4K_BLOCK_SIZE; j += WARP_SIZE) {
            int idx = base + j;
            if (idx < in_dim) {
                float w = dequant_q4k(block, j);
                float x = __half2float(input[idx]) * __half2float(norm_w[idx]) * rms_inv;
                sum += w * x;
            }
        }
    }

    // Warp reduce
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (lane == 0) {
        if (bias) {
            sum += __half2float(bias[row]);
        }
        out[row] = __float2half(sum);
    }
}

void fused_rmsnorm_q4k_gemv(
    half* out, const half* input, const half* norm_w,
    const void* weight, const half* bias,
    int in_dim, int out_dim, float eps, cudaStream_t stream)
{
    // 4 warps per block, each warp handles one output row
    const int warps_per_block = 4;
    dim3 block(WARP_SIZE, warps_per_block);
    dim3 grid((out_dim + warps_per_block - 1) / warps_per_block);

    kernel_fused_rmsnorm_q4k_gemv<<<grid, block, 0, stream>>>(
        out, input, norm_w,
        reinterpret_cast<const Q4KBlock*>(weight),
        bias, in_dim, out_dim, eps);
}

// ─── Fused RoPE + GQA Attention ──────────────────────────────────────────────
//
// For single-token autoregressive: compute attention for one query position.
// 1. Apply RoPE to Q and K_new
// 2. Write K_new, V_new to KV cache
// 3. Compute attention scores: Q * K^T for all cached positions
// 4. Softmax
// 5. Weighted sum of V

__device__ __forceinline__ void apply_rope(
    float& x0, float& x1, int dim_idx, int pos, float base)
{
    float freq = 1.0f / powf(base, (float)(dim_idx * 2) / 128.0f);
    float theta = pos * freq;
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);
    float new_x0 = x0 * cos_t - x1 * sin_t;
    float new_x1 = x0 * sin_t + x1 * cos_t;
    x0 = new_x0;
    x1 = new_x1;
}

__global__ void kernel_fused_gqa_attention(
    half* __restrict__ out,           // [n_head * head_dim]
    const half* __restrict__ q,       // [n_head * head_dim]
    const half* __restrict__ k_new,   // [n_kv_head * head_dim]
    const half* __restrict__ v_new,   // [n_kv_head * head_dim]
    half* __restrict__ k_cache,       // [n_kv_head, max_seq, head_dim]
    half* __restrict__ v_cache,       // [n_kv_head, max_seq, head_dim]
    int layer_idx, int pos,
    int n_head, int n_kv_head, int head_dim, int max_seq,
    float rope_base, int n_layer)
{
    const int head = blockIdx.x;     // which query head
    const int lane = threadIdx.x;    // within warp
    const int kv_head = head / (n_head / n_kv_head);  // GQA mapping

    // Layer offset into KV cache
    const int kv_layer_stride = n_kv_head * max_seq * head_dim;
    half* k_base = k_cache + layer_idx * kv_layer_stride + kv_head * max_seq * head_dim;
    half* v_base = v_cache + layer_idx * kv_layer_stride + kv_head * max_seq * head_dim;

    // Step 1: Load and RoPE-rotate Q for this head
    extern __shared__ float smem[];
    float* s_q = smem;                        // [head_dim]
    float* s_scores = smem + head_dim;        // [pos + 1]

    // Load Q with RoPE
    for (int d = lane; d < head_dim / 2; d += WARP_SIZE) {
        float q0 = __half2float(q[head * head_dim + d * 2]);
        float q1 = __half2float(q[head * head_dim + d * 2 + 1]);
        apply_rope(q0, q1, d, pos, rope_base);
        s_q[d * 2] = q0;
        s_q[d * 2 + 1] = q1;
    }
    __syncwarp();

    // Step 2: Write new K to cache (with RoPE) and new V to cache
    if (head == kv_head * (n_head / n_kv_head)) {
        // Only one head per KV group writes to cache
        for (int d = lane; d < head_dim / 2; d += WARP_SIZE) {
            float k0 = __half2float(k_new[kv_head * head_dim + d * 2]);
            float k1 = __half2float(k_new[kv_head * head_dim + d * 2 + 1]);
            apply_rope(k0, k1, d, pos, rope_base);
            k_base[pos * head_dim + d * 2] = __float2half(k0);
            k_base[pos * head_dim + d * 2 + 1] = __float2half(k1);
        }
        for (int d = lane; d < head_dim; d += WARP_SIZE) {
            v_base[pos * head_dim + d] = v_new[kv_head * head_dim + d];
        }
    }
    __syncwarp();

    // Step 3: Compute attention scores Q * K^T for all positions [0, pos]
    float scale = 1.0f / sqrtf((float)head_dim);
    int seq_len = pos + 1;

    float max_score = -1e30f;
    for (int p = lane; p < seq_len; p += WARP_SIZE) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += s_q[d] * __half2float(k_base[p * head_dim + d]);
        }
        score *= scale;
        s_scores[p] = score;
        max_score = fmaxf(max_score, score);
    }

    // Warp reduce max
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        max_score = fmaxf(max_score, __shfl_down_sync(0xFFFFFFFF, max_score, offset));
    }
    max_score = __shfl_sync(0xFFFFFFFF, max_score, 0);

    // Step 4: Softmax
    float sum_exp = 0.0f;
    for (int p = lane; p < seq_len; p += WARP_SIZE) {
        float e = expf(s_scores[p] - max_score);
        s_scores[p] = e;
        sum_exp += e;
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
    }
    sum_exp = __shfl_sync(0xFFFFFFFF, sum_exp, 0);

    float inv_sum = 1.0f / sum_exp;

    // Step 5: Weighted sum of V
    for (int d = lane; d < head_dim; d += WARP_SIZE) {
        float val = 0.0f;
        for (int p = 0; p < seq_len; p++) {
            val += s_scores[p] * inv_sum * __half2float(v_base[p * head_dim + d]);
        }
        out[head * head_dim + d] = __float2half(val);
    }
}

void fused_gqa_attention(
    half* out, const half* q, const half* k_new, const half* v_new,
    KVCache* kv_cache, int layer_idx, int pos,
    int n_head, int n_kv_head, int head_dim, float rope_base, cudaStream_t stream)
{
    // One block per attention head, one warp per block
    int smem_size = (head_dim + pos + 1) * sizeof(float);
    kernel_fused_gqa_attention<<<n_head, WARP_SIZE, smem_size, stream>>>(
        out, q, k_new, v_new,
        kv_cache->k, kv_cache->v,
        layer_idx, pos, n_head, n_kv_head, head_dim, kv_cache->max_seq,
        rope_base, kv_cache->n_layer);
}

// ─── Fused SwiGLU FFN ────────────────────────────────────────────────────────
//
// SwiGLU: out = (silu(x * W_gate) * (x * W_up)) * W_down
// Fused: one kernel for gate+up (saves reading input twice), one for down.

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void kernel_fused_swiglu_gate_up(
    half* __restrict__ gate_up_out,   // [n_ff] interleaved gate*up result
    const half* __restrict__ input,   // [n_embd]
    const half* __restrict__ norm_w,  // [n_embd]
    const Q4KBlock* __restrict__ w_gate,
    const Q4KBlock* __restrict__ w_up,
    int n_embd, int n_ff, float eps)
{
    const int ff_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;

    if (ff_idx >= n_ff) return;

    // Compute RMSNorm (shared across block)
    __shared__ float s_rms_inv;
    if (threadIdx.y == 0) {
        float sum_sq = 0.0f;
        for (int i = lane; i < n_embd; i += WARP_SIZE) {
            float v = __half2float(input[i]);
            sum_sq += v * v;
        }
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
        if (lane == 0)
            s_rms_inv = rsqrtf(sum_sq / (float)n_embd + eps);
    }
    __syncthreads();

    float rms_inv = s_rms_inv;
    int blocks_per_row = n_embd / Q4K_BLOCK_SIZE;

    // Compute gate and up projections simultaneously
    float gate_sum = 0.0f, up_sum = 0.0f;
    const Q4KBlock* gate_row = w_gate + ff_idx * blocks_per_row;
    const Q4KBlock* up_row = w_up + ff_idx * blocks_per_row;

    for (int b = 0; b < blocks_per_row; b++) {
        for (int j = lane; j < Q4K_BLOCK_SIZE; j += WARP_SIZE) {
            int idx = b * Q4K_BLOCK_SIZE + j;
            if (idx < n_embd) {
                float x = __half2float(input[idx]) * __half2float(norm_w[idx]) * rms_inv;
                gate_sum += dequant_q4k(&gate_row[b], j) * x;
                up_sum += dequant_q4k(&up_row[b], j) * x;
            }
        }
    }

    // Warp reduce both
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, offset);
        up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, offset);
    }

    if (lane == 0) {
        gate_up_out[ff_idx] = __float2half(silu(gate_sum) * up_sum);
    }
}

void fused_swiglu_ffn(
    half* out, const half* input, const half* norm_w,
    const void* w_gate, const void* w_up, const void* w_down,
    int n_embd, int n_ff, float eps, cudaStream_t stream)
{
    // Phase 1: Fused gate + up (output is n_ff intermediate)
    half* intermediate;
    cudaMallocAsync(&intermediate, n_ff * sizeof(half), stream);

    const int warps_per_block = 4;
    dim3 block(WARP_SIZE, warps_per_block);
    dim3 grid((n_ff + warps_per_block - 1) / warps_per_block);

    kernel_fused_swiglu_gate_up<<<grid, block, 0, stream>>>(
        intermediate, input, norm_w,
        reinterpret_cast<const Q4KBlock*>(w_gate),
        reinterpret_cast<const Q4KBlock*>(w_up),
        n_embd, n_ff, eps);

    // Phase 2: Down projection (intermediate → output)
    // Reuse the fused norm+gemv kernel but without normalization
    fused_rmsnorm_q4k_gemv(out, intermediate, nullptr, w_down, nullptr,
                           n_ff, n_embd, 1e30f, stream);  // huge eps = skip norm

    cudaFreeAsync(intermediate, stream);
}

} // namespace cuda
} // namespace infergo
