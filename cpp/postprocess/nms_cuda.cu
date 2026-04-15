// cpp/postprocess/nms_cuda.cu
// GPU-side NMS using raw CUDA kernels.
//
// Pipeline:
//   1. Filter by confidence threshold (parallel)
//   2. Sort by confidence descending (thrust::sort)
//   3. Compute IoU matrix in parallel (N*N thread grid)
//   4. Greedy suppression on GPU (sequential scan with parallel IoU reads)
//   5. Compact and copy kept detections to host
//
// Each input detection is 6 floats: [x1, y1, x2, y2, confidence, class_id].
// Output: InferBox structs on host memory.

#include "nms_cuda.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#include <cstdio>
#include <algorithm>

// ─── Constants ──────────────────────────────────────────────────────────────

// Detection layout: 6 floats per detection
static constexpr int DET_STRIDE = 6;
static constexpr int DET_X1     = 0;
static constexpr int DET_Y1     = 1;
static constexpr int DET_X2     = 2;
static constexpr int DET_Y2     = 3;
static constexpr int DET_CONF   = 4;
static constexpr int DET_CLASS  = 5;

// Max detections we handle (after confidence filtering).
// If more pass the threshold, we take the top MAX_FILTERED by confidence.
static constexpr int MAX_FILTERED = 4096;

// Block size for IoU kernel — each block handles one row of the IoU matrix.
// 256 threads per block is a good balance for occupancy.
static constexpr int IOU_BLOCK_SIZE = 256;

// ─── Error checking macro ───────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            return INFER_ERR_CUDA;                                            \
        }                                                                     \
    } while (0)

#define CUDA_CHECK_CPP(call)                                                  \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            throw std::runtime_error(cudaGetErrorString(err));                \
        }                                                                     \
    } while (0)

// ─── Kernel 1: Confidence threshold filter ──────────────────────────────────
// Each thread checks one detection. If conf >= threshold, atomically
// reserves a slot in the output array.

__global__ void filter_by_confidence_kernel(
    const float* __restrict__ d_boxes,    // [N, 6]
    int n_boxes,
    float conf_thresh,
    float* __restrict__ d_filtered,       // [MAX_FILTERED, 6]
    int* __restrict__ d_filtered_count)   // atomic counter
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_boxes) return;

    const float conf = d_boxes[idx * DET_STRIDE + DET_CONF];
    if (conf < conf_thresh) return;

    const int slot = atomicAdd(d_filtered_count, 1);
    if (slot >= MAX_FILTERED) return;  // overflow protection

    // Copy the 6-float detection to the filtered array
    const float* src = d_boxes + idx * DET_STRIDE;
    float* dst = d_filtered + slot * DET_STRIDE;
    dst[DET_X1]    = src[DET_X1];
    dst[DET_Y1]    = src[DET_Y1];
    dst[DET_X2]    = src[DET_X2];
    dst[DET_Y2]    = src[DET_Y2];
    dst[DET_CONF]  = src[DET_CONF];
    dst[DET_CLASS] = src[DET_CLASS];
}

// ─── Sort comparator for thrust ─────────────────────────────────────────────
// Sort indices by confidence descending. The actual data stays in place;
// we sort an index array.

struct ConfidenceDescending {
    const float* d_filtered;

    __host__ __device__
    bool operator()(int a, int b) const {
        return d_filtered[a * DET_STRIDE + DET_CONF] >
               d_filtered[b * DET_STRIDE + DET_CONF];
    }
};

// ─── Kernel 2: Compute IoU matrix ───────────────────────────────────────────
// For N filtered detections, compute the upper-triangle of the NxN IoU matrix.
// We pack the IoU values into a flat array: iou_matrix[i * N + j] for i < j.
// We only need the upper triangle (i < j) since IoU is symmetric.
//
// Launch: N blocks of min(N, IOU_BLOCK_SIZE) threads each.
// Block i computes IoU(i, j) for j = i+1..N-1.

__global__ void compute_iou_matrix_kernel(
    const float* __restrict__ d_sorted,  // [N, 6] sorted by confidence
    const int* __restrict__ d_indices,   // sorted index array
    int N,
    float* __restrict__ d_iou_matrix)    // [N * N] flat (only upper triangle used)
{
    const int i = blockIdx.x;
    if (i >= N) return;

    const int idx_i = d_indices[i];
    const float ax1 = d_sorted[idx_i * DET_STRIDE + DET_X1];
    const float ay1 = d_sorted[idx_i * DET_STRIDE + DET_Y1];
    const float ax2 = d_sorted[idx_i * DET_STRIDE + DET_X2];
    const float ay2 = d_sorted[idx_i * DET_STRIDE + DET_Y2];
    const float area_a = (ax2 - ax1) * (ay2 - ay1);

    // Each thread in this block handles a different j
    for (int j = i + 1 + threadIdx.x; j < N; j += blockDim.x) {
        const int idx_j = d_indices[j];
        const float bx1 = d_sorted[idx_j * DET_STRIDE + DET_X1];
        const float by1 = d_sorted[idx_j * DET_STRIDE + DET_Y1];
        const float bx2 = d_sorted[idx_j * DET_STRIDE + DET_X2];
        const float by2 = d_sorted[idx_j * DET_STRIDE + DET_Y2];

        const float ix1 = fmaxf(ax1, bx1);
        const float iy1 = fmaxf(ay1, by1);
        const float ix2 = fminf(ax2, bx2);
        const float iy2 = fminf(ay2, by2);

        const float iw = fmaxf(0.0f, ix2 - ix1);
        const float ih = fmaxf(0.0f, iy2 - iy1);
        const float inter = iw * ih;

        const float area_b = (bx2 - bx1) * (by2 - by1);
        const float union_area = area_a + area_b - inter;

        float iou_val = 0.0f;
        if (union_area > 0.0f) {
            iou_val = inter / union_area;
        }

        d_iou_matrix[i * N + j] = iou_val;
    }
}

// ─── Kernel 3: Greedy suppression ───────────────────────────────────────────
// Sequential scan through sorted detections. For each non-suppressed
// detection i, suppress all j > i where IoU(i,j) > threshold AND
// same class. This kernel runs on a single thread (the sequential
// nature of greedy NMS prevents parallelization of the outer loop).
// The IoU lookups are fast because the matrix is in GPU global memory.

__global__ void greedy_suppress_kernel(
    const float* __restrict__ d_sorted,    // [N, 6]
    const int* __restrict__ d_indices,     // sorted index array
    const float* __restrict__ d_iou_matrix, // [N * N]
    int N,
    float iou_thresh,
    int* __restrict__ d_suppressed,        // [N] — 0 = kept, 1 = suppressed
    int* __restrict__ d_kept_indices,      // indices of kept detections
    int* __restrict__ d_kept_count)        // number of kept detections
{
    // Single-thread kernel
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int count = 0;
    for (int i = 0; i < N; ++i) {
        if (d_suppressed[i]) continue;

        d_kept_indices[count] = i;
        ++count;

        const int idx_i = d_indices[i];
        const int cls_i = __float2int_rn(d_sorted[idx_i * DET_STRIDE + DET_CLASS]);

        for (int j = i + 1; j < N; ++j) {
            if (d_suppressed[j]) continue;

            const int idx_j = d_indices[j];
            const int cls_j = __float2int_rn(d_sorted[idx_j * DET_STRIDE + DET_CLASS]);

            // Class-aware NMS: only suppress same class
            if (cls_i != cls_j) continue;

            if (d_iou_matrix[i * N + j] > iou_thresh) {
                d_suppressed[j] = 1;
            }
        }
    }

    *d_kept_count = count;
}

// ─── Kernel 4: Gather kept detections ───────────────────────────────────────
// Copy kept detections from the filtered/sorted array into a compact output.
// Each thread handles one kept detection.

__global__ void gather_kept_kernel(
    const float* __restrict__ d_sorted,       // [N, 6] filtered detections
    const int* __restrict__ d_indices,        // sorted index array
    const int* __restrict__ d_kept_indices,   // which sorted positions are kept
    int n_kept,
    float* __restrict__ d_output)             // [n_kept, 6] compact output
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_kept) return;

    const int sorted_pos = d_kept_indices[tid];
    const int data_idx = d_indices[sorted_pos];
    const float* src = d_sorted + data_idx * DET_STRIDE;
    float* dst = d_output + tid * DET_STRIDE;

    dst[DET_X1]    = src[DET_X1];
    dst[DET_Y1]    = src[DET_Y1];
    dst[DET_X2]    = src[DET_X2];
    dst[DET_Y2]    = src[DET_Y2];
    dst[DET_CONF]  = src[DET_CONF];
    dst[DET_CLASS] = src[DET_CLASS];
}

// ─── C API implementation ───────────────────────────────────────────────────
// Uses goto-cleanup to ensure all GPU allocations are freed on error paths.

extern "C"
InferError infer_nms_cuda(const float* d_boxes, int n_boxes,
                          float conf_thresh, float iou_thresh,
                          InferBox* out_boxes, int max_out,
                          int* out_count, void* stream)
{
    if (!out_count) return INFER_ERR_NULL;
    *out_count = 0;

    if (!d_boxes || !out_boxes) return INFER_ERR_NULL;
    if (n_boxes <= 0 || max_out <= 0) return INFER_OK;

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    InferError result = INFER_OK;

    // All GPU allocations tracked for cleanup
    float* d_filtered      = nullptr;
    int*   d_filtered_count = nullptr;
    int*   d_indices       = nullptr;
    float* d_iou_matrix    = nullptr;
    int*   d_suppressed    = nullptr;
    int*   d_kept_indices  = nullptr;
    int*   d_kept_count    = nullptr;
    float* d_output        = nullptr;
    float* h_output        = nullptr;

    // Macro: on CUDA error, set result and jump to cleanup
    #define NMS_CUDA_CHECK(call)                                              \
        do {                                                                  \
            cudaError_t err = (call);                                         \
            if (err != cudaSuccess) {                                         \
                result = INFER_ERR_CUDA;                                      \
                goto cleanup;                                                 \
            }                                                                 \
        } while (0)

    // ── Step 1: Allocate filtered detections buffer ──────────────────────────

    NMS_CUDA_CHECK(cudaMalloc(&d_filtered, MAX_FILTERED * DET_STRIDE * sizeof(float)));
    NMS_CUDA_CHECK(cudaMalloc(&d_filtered_count, sizeof(int)));
    NMS_CUDA_CHECK(cudaMemsetAsync(d_filtered_count, 0, sizeof(int), cuda_stream));

    // ── Step 2: Filter by confidence threshold ───────────────────────────────
    {
        const int block = 256;
        const int grid = (n_boxes + block - 1) / block;
        filter_by_confidence_kernel<<<grid, block, 0, cuda_stream>>>(
            d_boxes, n_boxes, conf_thresh, d_filtered, d_filtered_count);
        NMS_CUDA_CHECK(cudaGetLastError());
    }

    {
        int h_filtered_count = 0;
        NMS_CUDA_CHECK(cudaMemcpyAsync(&h_filtered_count, d_filtered_count, sizeof(int),
                                        cudaMemcpyDeviceToHost, cuda_stream));
        NMS_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));

        if (h_filtered_count <= 0) goto cleanup;  // result already INFER_OK

        // Clamp to MAX_FILTERED
        h_filtered_count = std::min(h_filtered_count, MAX_FILTERED);
        const int N = h_filtered_count;

        // ── Step 3: Sort by confidence descending ────────────────────────────
        NMS_CUDA_CHECK(cudaMalloc(&d_indices, N * sizeof(int)));

        {
            thrust::device_ptr<int> d_idx_ptr(d_indices);
            thrust::sequence(thrust::device, d_idx_ptr, d_idx_ptr + N);
            thrust::sort(thrust::device, d_idx_ptr, d_idx_ptr + N,
                         ConfidenceDescending{d_filtered});
        }

        // ── Step 4: Compute IoU matrix ───────────────────────────────────────
        NMS_CUDA_CHECK(cudaMalloc(&d_iou_matrix, (size_t)N * N * sizeof(float)));
        NMS_CUDA_CHECK(cudaMemsetAsync(d_iou_matrix, 0,
                                        (size_t)N * N * sizeof(float), cuda_stream));

        {
            const int threads = std::min(N, IOU_BLOCK_SIZE);
            compute_iou_matrix_kernel<<<N, threads, 0, cuda_stream>>>(
                d_filtered, d_indices, N, d_iou_matrix);
            NMS_CUDA_CHECK(cudaGetLastError());
        }

        // ── Step 5: Greedy suppression ───────────────────────────────────────
        NMS_CUDA_CHECK(cudaMalloc(&d_suppressed, N * sizeof(int)));
        NMS_CUDA_CHECK(cudaMemsetAsync(d_suppressed, 0, N * sizeof(int), cuda_stream));
        NMS_CUDA_CHECK(cudaMalloc(&d_kept_indices, N * sizeof(int)));
        NMS_CUDA_CHECK(cudaMalloc(&d_kept_count, sizeof(int)));
        NMS_CUDA_CHECK(cudaMemsetAsync(d_kept_count, 0, sizeof(int), cuda_stream));

        greedy_suppress_kernel<<<1, 1, 0, cuda_stream>>>(
            d_filtered, d_indices, d_iou_matrix, N, iou_thresh,
            d_suppressed, d_kept_indices, d_kept_count);
        NMS_CUDA_CHECK(cudaGetLastError());

        int h_kept_count = 0;
        NMS_CUDA_CHECK(cudaMemcpyAsync(&h_kept_count, d_kept_count, sizeof(int),
                                        cudaMemcpyDeviceToHost, cuda_stream));
        NMS_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));

        if (h_kept_count <= 0) goto cleanup;  // result already INFER_OK

        const int n_output = std::min(h_kept_count, max_out);

        // ── Step 6: Gather kept detections ───────────────────────────────────
        NMS_CUDA_CHECK(cudaMalloc(&d_output, n_output * DET_STRIDE * sizeof(float)));

        {
            const int block = 256;
            const int grid = (n_output + block - 1) / block;
            gather_kept_kernel<<<grid, block, 0, cuda_stream>>>(
                d_filtered, d_indices, d_kept_indices, n_output, d_output);
            NMS_CUDA_CHECK(cudaGetLastError());
        }

        // ── Step 7: Copy result to host and convert to InferBox ──────────────
        h_output = new float[n_output * DET_STRIDE];
        NMS_CUDA_CHECK(cudaMemcpyAsync(h_output, d_output,
                                        n_output * DET_STRIDE * sizeof(float),
                                        cudaMemcpyDeviceToHost, cuda_stream));
        NMS_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));

        for (int i = 0; i < n_output; ++i) {
            const float* src = h_output + i * DET_STRIDE;
            out_boxes[i].x1         = src[DET_X1];
            out_boxes[i].y1         = src[DET_Y1];
            out_boxes[i].x2         = src[DET_X2];
            out_boxes[i].y2         = src[DET_Y2];
            out_boxes[i].confidence = src[DET_CONF];
            out_boxes[i].class_idx  = static_cast<int>(src[DET_CLASS] + 0.5f);
        }

        *out_count = n_output;
    }

cleanup:
    delete[] h_output;
    cudaFree(d_output);
    cudaFree(d_kept_count);
    cudaFree(d_kept_indices);
    cudaFree(d_suppressed);
    cudaFree(d_iou_matrix);
    cudaFree(d_indices);
    cudaFree(d_filtered_count);
    cudaFree(d_filtered);

    #undef NMS_CUDA_CHECK
    return result;
}

// ─── C++ wrapper ─────────────────────────────────────────────────────────────

namespace infergo {

NmsCudaResult nms_cuda(const float* d_boxes, int n_boxes,
                       float conf_thresh, float iou_thresh,
                       int max_out)
{
    NmsCudaResult result{nullptr, 0};
    if (n_boxes <= 0 || max_out <= 0) return result;

    result.boxes = new InferBox[max_out];
    InferError err = infer_nms_cuda(d_boxes, n_boxes,
                                    conf_thresh, iou_thresh,
                                    result.boxes, max_out,
                                    &result.count, nullptr);
    if (err != INFER_OK) {
        delete[] result.boxes;
        result.boxes = nullptr;
        result.count = 0;
    }
    return result;
}

} // namespace infergo
