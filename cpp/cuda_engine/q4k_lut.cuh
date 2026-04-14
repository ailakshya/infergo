// Q4_K GEMV with Lookup Table Dequantization
//
// Key insight: Each Q4_K sub-block has 32 values sharing one (scale, min) pair.
// Only 16 possible outputs exist (nibble values 0-15). Pre-compute all 16 values
// in shared memory, then use the 4-bit index as a direct LUT lookup.
//
// Replaces per-element: d * sc * nibble - dmin * m (3 mul + 1 sub)
// With: shared_lut[nibble] (1 load from 48KB L1-speed shared memory)
//
// Combined with input quantization to Q8_1 + __dp4a, this should match
// or beat llama.cpp's mmvq kernel.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace infergo {
namespace cuda {

// Q8_1 block for pre-quantized input
struct Q8Block {
    half2 ds;      // x=scale, y=d*sum
    int8_t qs[32];
};

// ─── Fused RMSNorm + Q8_1 Quantize ──────────────────────────────────────────
// One kernel: normalize input and quantize to INT8 in one pass.
// Eliminates separate RMSNorm kernel launch.

static __global__ void kernel_rmsnorm_quantize_q8(
    Q8Block* __restrict__ out,
    const half* __restrict__ input,
    const half* __restrict__ norm_w,  // nullable
    int n, float eps)
{
    // Shared memory for RMS reduction
    __shared__ float s_rms_inv;

    // Phase 1: compute RMS (all threads cooperate)
    if (norm_w) {
        float sum_sq = 0.0f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            float v = __half2float(input[i]);
            sum_sq += v * v;
        }
        // Block reduce
        __shared__ float smem_reduce[256];
        smem_reduce[threadIdx.x] = sum_sq;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) smem_reduce[threadIdx.x] += smem_reduce[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) s_rms_inv = rsqrtf(smem_reduce[0] / (float)n + eps);
        __syncthreads();
    }

    // Phase 2: quantize 32-element blocks
    // Each warp handles one Q8_1 block
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int n_blocks = n / 32;

    for (int b = warp_id; b < n_blocks; b += blockDim.x / 32) {
        int idx = b * 32 + lane;
        float val = __half2float(input[idx]);
        if (norm_w) val *= __half2float(norm_w[idx]) * s_rms_inv;

        // Warp-level abs-max and sum
        float amax = fabsf(val);
        float sum = val;
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off));
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, off);
        }

        float d = amax / 127.0f;
        int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(val / d);

        out[b].qs[lane] = q;
        if (lane == 0) {
            out[b].ds = make_half2(__float2half(d), __float2half(d * sum));
        }
    }
}

// ─── LUT-based Q4_K × Q8_1 GEMV ─────────────────────────────────────────────
// Each thread block handles one output row.
// Uses shared memory LUT for fast dequantization.

static __global__ void kernel_q4k_lut_gemv(
    half* __restrict__ out,
    const void* __restrict__ weight,   // Q4_K blocks, row-major
    const Q8Block* __restrict__ q8,    // pre-quantized input
    const half* __restrict__ bias,
    int in_dim, int out_dim)
{
    const int row = blockIdx.x;
    if (row >= out_dim) return;

    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int n_warps = blockDim.y;

    const int blocks_per_row = in_dim / 256;
    const char* row_base = (const char*)weight + (size_t)row * blocks_per_row * 144;

    // LUT in shared memory: 8 sub-blocks × 16 values = 128 floats per Q4K block
    // We reuse the LUT for each Q4K block as we iterate
    __shared__ float s_lut[8][16];  // 512 bytes — fits easily
    __shared__ float s_warp_sums[8]; // inter-warp reduction

    float my_sum = 0.0f;

    // Each warp processes different Q4K blocks, all warps share the LUT
    for (int kbx = 0; kbx < blocks_per_row; kbx++) {
        const char* blk = row_base + kbx * 144;

        // Load dm
        half2 dm;
        if (tid < 2) memcpy(((char*)&dm) + tid * 2, blk + tid * 2, 2);
        __syncthreads();

        float d = __half2float(*(const half*)blk);
        float dmin = __half2float(*(const half*)(blk + 2));
        const uint8_t* scales = (const uint8_t*)(blk + 4);

        // Build LUT: thread tid < 128 builds lut entries
        // 8 sub-blocks × 16 entries = 128 entries, one per thread
        if (tid < 128) {
            int sub = tid / 16;
            int nibble = tid % 16;

            uint8_t sc, m;
            if (sub < 4) {
                sc = scales[sub] & 0x3F;
                m = scales[sub + 4] & 0x3F;
            } else {
                sc = ((scales[sub + 4] & 0xF) | ((scales[sub - 4] >> 6) << 4));
                m = ((scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4));
            }
            s_lut[sub][nibble] = d * sc * nibble - dmin * m;
        }
        __syncthreads();

        // Now each thread uses the LUT for fast dequantization
        // Each warp processes a segment of the 256 values using dp4a
        const uint8_t* qs = (const uint8_t*)(blk + 16);
        const int kby = kbx * 8;  // Q8_1 block offset

        // Each warp handles 256/n_warps values
        int vals_per_warp = 256 / n_warps;
        int start = warp_id * vals_per_warp;
        int end = start + vals_per_warp;

        for (int j = start + lane; j < end; j += 32) {
            int sub = j / 32;
            int byte_idx;
            if (sub < 4) byte_idx = sub * 16 + (j % 32) / 2;
            else byte_idx = 64 + (sub - 4) * 16 + (j % 32) / 2;

            uint8_t byte_val = qs[byte_idx];
            int nibble = (j & 1) ? (byte_val >> 4) : (byte_val & 0xF);

            // LUT lookup — shared memory is as fast as L1 cache
            float w = s_lut[sub][nibble];

            // Get corresponding Q8_1 activation
            int q8_block = kby + j / 32;
            float q8_scale = __half2float(__low2half(q8[q8_block].ds));
            int8_t q8_val = q8[q8_block].qs[j % 32];

            my_sum += w * q8_scale * (float)q8_val;
        }
        __syncthreads();
    }

    // Warp reduce
    for (int off = 16; off > 0; off >>= 1)
        my_sum += __shfl_xor_sync(0xFFFFFFFF, my_sum, off);

    // Inter-warp reduce via shared memory
    if (lane == 0) s_warp_sums[warp_id] = my_sum;
    __syncthreads();

    if (warp_id == 0 && lane < n_warps) {
        float total = s_warp_sums[lane];
        for (int off = n_warps / 2; off > 0; off >>= 1)
            total += __shfl_xor_sync((1u << n_warps) - 1, total, off);

        if (lane == 0) {
            if (bias) total += __half2float(bias[row]);
            out[row] = __float2half(total);
        }
    }
}

// ─── Launch helper ───────────────────────────────────────────────────────────

inline void lut_q4k_gemv(
    half* out, const half* input, const half* norm_w,
    const void* weight, const half* bias,
    int in_dim, int out_dim, float eps,
    Q8Block* q8_buf,
    cudaStream_t stream)
{
    // Step 1: Fused RMSNorm + Q8_1 quantize (1 kernel launch)
    kernel_rmsnorm_quantize_q8<<<1, 256, 0, stream>>>(
        q8_buf, input, norm_w, in_dim, eps);

    // Step 2: LUT-based Q4K GEMV (1 kernel launch)
    // 8 warps per block for good occupancy
    dim3 block(32, 8);
    kernel_q4k_lut_gemv<<<out_dim, block, 0, stream>>>(
        out, weight, q8_buf, bias, in_dim, out_dim);
}

} // namespace cuda
} // namespace infergo
