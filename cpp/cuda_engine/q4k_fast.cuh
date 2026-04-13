// Fast Q4_K GEMV kernel — replicates llama.cpp's dp4a strategy
// with fused RMSNorm for reduced kernel launches.
//
// Strategy:
//   1. Pre-quantize input FP16 → Q8_1 (INT8 + scale)
//   2. Use __dp4a for 4×INT8 dot products (hardware instruction)
//   3. 4 warps (128 threads) per output row
//   4. Vectorized int32 loads from Q4_K weight blocks
//   5. Fused RMSNorm: normalize input before quantization
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace infergo {
namespace cuda {

// Q8_1 block: 32 int8 values + scale + sum
struct BlockQ8_1 {
    half2 ds;      // ds.x = scale, ds.y = d * sum(qs)
    int8_t qs[32]; // quantized values
};

// Pre-quantize FP16 input to Q8_1 format (required before dp4a GEMV)
// rms_inv_ptr is a DEVICE pointer to the pre-computed RMS inverse
static __global__ void kernel_quantize_q8_1(
    BlockQ8_1* __restrict__ out,
    const half* __restrict__ input,
    const half* __restrict__ norm_w,  // nullable: apply RMSNorm before quant
    const float* __restrict__ rms_inv_ptr, // device pointer, nullable
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int block_idx = i / 32;
    const int lane = i % 32;

    if (i >= n) return;

    // Load and optionally normalize
    float val = __half2float(input[i]);
    if (norm_w && rms_inv_ptr) {
        val *= __half2float(norm_w[i]) * (*rms_inv_ptr);
    }

    // Warp-level reduce for abs-max and sum
    float amax = fabsf(val);
    float sum = val;

    // Warp reduce (32 threads = 1 Q8_1 block)
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    float d = amax / 127.0f;
    int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(val / d);

    out[block_idx].qs[lane] = q;
    if (lane == 0) {
        out[block_idx].ds = make_half2(__float2half(d), __float2half(d * sum));
    }
}

// Compute RMS for norm (separate kernel, result shared)
static __global__ void kernel_compute_rms(
    float* __restrict__ out_rms_inv,
    const half* __restrict__ input,
    int n, float eps)
{
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = __half2float(input[i]);
        sum_sq += v * v;
    }

    // Block reduce
    __shared__ float smem[256];
    smem[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *out_rms_inv = rsqrtf(smem[0] / (float)n + eps);
    }
}

// dp4a: dot product of 4 int8 pairs, accumulated into int32
__device__ __forceinline__ int dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    // Fallback
    const int8_t* a8 = reinterpret_cast<const int8_t*>(&a);
    const int8_t* b8 = reinterpret_cast<const int8_t*>(&b);
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
}

// Fast Q4_K × Q8_1 GEMV kernel
// 4 warps (128 threads) per output row
// Each thread handles 2 int32s (8 weight nibbles × 8 activation int8s)
static __global__ void kernel_q4k_dp4a_gemv(
    half* __restrict__ out,           // [out_dim]
    const void* __restrict__ weight,  // Q4_K blocks [out_dim, in_dim/256]
    const BlockQ8_1* __restrict__ q8, // pre-quantized input [in_dim/32]
    const half* __restrict__ bias,    // nullable
    int in_dim, int out_dim)
{
    const int row = blockIdx.x;  // one block per output row
    if (row >= out_dim) return;

    const int tid = threadIdx.y * 32 + threadIdx.x;  // 0..127
    const int blocks_per_row = in_dim / 256;

    // Q4_K block pointer for this row
    const char* row_base = (const char*)weight + (size_t)row * blocks_per_row * 144;

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    // 16 threads per Q4_K block, 128 threads total → 8 blocks per iteration
    const int blocks_per_iter = 128 / 16;  // = 8
    const int block_lane = tid % 16;       // position within Q4_K block
    const int bq8_offset = 2 * (block_lane / 4);  // which Q8_1 sub-pair (0,2,4,6)

    for (int kbx = tid / 16; kbx < blocks_per_row; kbx += blocks_per_iter) {
        const char* blk_ptr = row_base + kbx * 144;

        // Load dm (super-block scale and min) — per-block!
        half2 dm;
        memcpy(&dm, blk_ptr, 4);
        const float2 dm_f = __half22float2(dm);

        // Load scales
        const uint8_t* scales = (const uint8_t*)(blk_ptr + 4);

        // Extract 6-bit scale and min for this sub-block pair
        const int j = bq8_offset / 2;
        uint8_t sc[2], mn[2];
        if (j < 2) {
            sc[0] = scales[j] & 0x3F;
            mn[0] = scales[j + 4] & 0x3F;
            sc[1] = scales[j + 1] & 0x3F;
            mn[1] = scales[j + 5] & 0x3F;
        } else {
            sc[0] = ((scales[j + 4] & 0xF) | ((scales[j - 2] >> 6) << 4));
            mn[0] = ((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
            sc[1] = sc[0]; mn[1] = mn[0]; // same for second half
        }

        // Load Q4_K quants (vectorized int32 loads)
        const int* q4 = (const int*)(blk_ptr + 16 + 16 * bq8_offset + 4 * (block_lane % 4));
        int v0 = q4[0];
        int v1 = q4[4];

        // Process 2 Q8_1 sub-blocks per Q4_K block segment
        const int kby = kbx * 8 + bq8_offset;
        float block_sumf_d = 0.0f, block_sumf_m = 0.0f;

        for (int i = 0; i < 2; i++) {
            const BlockQ8_1* bq8 = &q8[kby + i];
            float d8 = __half2float(__low2half(bq8->ds));

            const int* u = (const int*)bq8->qs + (block_lane % 4);
            int u0 = u[0];
            int u1 = u[4];

            // Extract 4-bit nibbles
            int v0i = (v0 >> (4 * i)) & 0x0F0F0F0F;
            int v1i = (v1 >> (4 * i)) & 0x0F0F0F0F;

            // dp4a: hardware 4×INT8 dot product
            int dot1 = dp4a(v1i, u1, dp4a(v0i, u0, 0));
            int dot2 = dp4a(0x01010101, u1, dp4a(0x01010101, u0, 0));

            block_sumf_d += d8 * (float)(dot1 * sc[i]);
            block_sumf_m += d8 * (float)(dot2 * mn[i]);
        }

        // Apply per-block super-block scale and min
        sumf_d += dm_f.x * block_sumf_d;
        sumf_m += dm_f.y * block_sumf_m;
    }

    float result = sumf_d - sumf_m;

    // Warp reduction: all 128 threads sum
    // First: shared memory for inter-warp reduction
    __shared__ float warp_sums[4];  // one per warp

    // Intra-warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        result += __shfl_xor_sync(0xFFFFFFFF, result, offset);
    }

    // Write warp result to shared memory
    if (threadIdx.x == 0) {
        warp_sums[threadIdx.y] = result;
    }
    __syncthreads();

    // Warp 0 combines all warp results
    if (threadIdx.y == 0) {
        float total = warp_sums[0];
        if (threadIdx.x == 1) total = warp_sums[1];
        if (threadIdx.x == 2) total = warp_sums[2];
        if (threadIdx.x == 3) total = warp_sums[3];

        for (int offset = 2; offset > 0; offset >>= 1) {
            total += __shfl_xor_sync(0x0F, total, offset);
        }

        if (threadIdx.x == 0) {
            if (bias) total += __half2float(bias[row]);
            out[row] = __float2half(total);
        }
    }
}

// Launch the fast Q4K GEMV with fused RMSNorm
inline void fast_q4k_gemv(
    half* out,
    const half* input,
    const half* norm_w,  // nullable
    const void* weight,  // Q4_K blocks
    const half* bias,    // nullable
    int in_dim, int out_dim,
    float eps,
    // Pre-allocated buffers:
    BlockQ8_1* q8_buf,   // [in_dim/32] Q8_1 blocks
    float* rms_buf,      // [1] scalar for RMS inverse
    cudaStream_t stream)
{
    // Step 1: Compute RMS on GPU (if norm needed) — NO CPU sync!
    if (norm_w) {
        kernel_compute_rms<<<1, 256, 0, stream>>>(rms_buf, input, in_dim, eps);
    }

    // Step 2: Quantize input to Q8_1 (reads rms_buf from device, no CPU roundtrip)
    kernel_quantize_q8_1<<<(in_dim + 255) / 256, 256, 0, stream>>>(
        q8_buf, input, norm_w, norm_w ? rms_buf : nullptr, in_dim);

    // Step 3: Q4K × Q8_1 GEMV with dp4a
    dim3 block(32, 4);  // 128 threads, 4 warps
    kernel_q4k_dp4a_gemv<<<out_dim, block, 0, stream>>>(
        out, weight, q8_buf, bias, in_dim, out_dim);
}

} // namespace cuda
} // namespace infergo
