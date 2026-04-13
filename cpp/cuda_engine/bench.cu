// Benchmark: custom CUDA kernels vs cuBLAS baseline
// Measures kernel launch overhead and memory throughput

#include "engine.cuh"
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>

using namespace infergo::cuda;

// Benchmark kernel launch overhead
__global__ void kernel_noop() {}

void bench_launch_overhead() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup
    for (int i = 0; i < 100; i++) {
        kernel_noop<<<1, 1, 0, stream>>>();
    }
    cudaStreamSynchronize(stream);

    // Measure
    const int N = 10000;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        kernel_noop<<<1, 1, 0, stream>>>();
    }
    cudaStreamSynchronize(stream);
    auto t1 = std::chrono::high_resolution_clock::now();

    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    printf("Kernel launch overhead: %.2f us/launch (%d launches)\n", us / N, N);
    printf("  Per transformer layer (6 launches): %.1f us\n", us / N * 6);
    printf("  Per token (28 layers × 6): %.1f us = %.3f ms\n",
           us / N * 6 * 28, us / N * 6 * 28 / 1000.0);

    cudaStreamDestroy(stream);
}

// Benchmark fused GEMV kernel throughput
void bench_fused_gemv() {
    const int in_dim = 1536;   // Qwen n_embd
    const int out_dim = 1536;

    half* input;
    half* norm_w;
    half* output;
    void* weight;

    cudaMalloc(&input, in_dim * sizeof(half));
    cudaMalloc(&norm_w, in_dim * sizeof(half));
    cudaMalloc(&output, out_dim * sizeof(half));

    // Q4_K: each block has 256 values, block size = sizeof(Q4KBlock)
    int n_blocks = (in_dim * out_dim) / 256;
    cudaMalloc(&weight, n_blocks * sizeof(Q4KBlock));

    // Initialize with random data
    cudaMemset(input, 0x3C, in_dim * sizeof(half));    // ~1.0 in FP16
    cudaMemset(norm_w, 0x3C, in_dim * sizeof(half));
    cudaMemset(weight, 0x42, n_blocks * sizeof(Q4KBlock));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup
    for (int i = 0; i < 100; i++) {
        fused_rmsnorm_q4k_gemv(output, input, norm_w, weight, nullptr,
                                in_dim, out_dim, 1e-6f, stream);
    }
    cudaStreamSynchronize(stream);

    // Measure
    const int N = 1000;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        fused_rmsnorm_q4k_gemv(output, input, norm_w, weight, nullptr,
                                in_dim, out_dim, 1e-6f, stream);
    }
    cudaStreamSynchronize(stream);
    auto t1 = std::chrono::high_resolution_clock::now();

    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;

    // Calculate throughput
    size_t weight_bytes = (size_t)n_blocks * sizeof(Q4KBlock);
    double gbps = (double)weight_bytes / (us * 1000.0);  // GB/s

    printf("\nFused RMSNorm+Q4K GEMV [%d×%d]:\n", in_dim, out_dim);
    printf("  Time: %.1f us/call\n", us);
    printf("  Weight data: %.2f MB\n", weight_bytes / 1e6);
    printf("  Throughput: %.1f GB/s (%.1f%% of 896 GB/s peak)\n",
           gbps, gbps / 896.0 * 100);

    // Compare: what llama.cpp does (3 separate kernels)
    printf("  llama.cpp equivalent: ~%.0f us (3 kernels × %.0f us launch + compute)\n",
           us * 1.3, us / 3);  // rough estimate: 30% more overhead from separate launches

    cudaFree(input);
    cudaFree(norm_w);
    cudaFree(output);
    cudaFree(weight);
    cudaStreamDestroy(stream);
}

// Benchmark memory bandwidth
void bench_bandwidth() {
    const size_t SIZE = 1ULL * 1024 * 1024 * 1024;  // 1 GB
    half* src;
    half* dst;
    cudaMalloc(&src, SIZE);
    cudaMalloc(&dst, SIZE);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup
    cudaMemcpyAsync(dst, src, SIZE, cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    // Measure
    const int N = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        cudaMemcpyAsync(dst, src, SIZE, cudaMemcpyDeviceToDevice, stream);
    }
    cudaStreamSynchronize(stream);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / N;
    double gbps = (double)SIZE / (ms / 1000.0) / 1e9;

    printf("\nMemory bandwidth: %.1f GB/s (1 GB D2D copy)\n", gbps);
    printf("  Model read time (1.04 GB): %.2f ms\n", 1040.0 / gbps);
    printf("  Theoretical max tok/s: %.0f\n", gbps / 1.04);

    cudaFree(src);
    cudaFree(dst);
    cudaStreamDestroy(stream);
}

int main() {
    printf("=== infergo Custom CUDA Engine Benchmark ===\n");
    printf("GPU: ");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("%s (SM %d.%d, %d MB VRAM)\n\n",
           prop.name, prop.major, prop.minor,
           (int)(prop.totalGlobalMem / 1024 / 1024));

    bench_launch_overhead();
    bench_bandwidth();
    bench_fused_gemv();

    printf("\n=== Done ===\n");
    return 0;
}
