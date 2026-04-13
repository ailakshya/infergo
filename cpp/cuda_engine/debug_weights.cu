#include "engine.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

using namespace infergo::cuda;

int main() {
    CUDAEngine engine;
    ModelConfig config;
    config.n_embd = 1536; config.n_head = 12; config.n_kv_head = 2;
    config.n_layer = 28; config.n_ff = 8960; config.head_dim = 128;
    config.n_vocab = 151936; config.n_ctx = 256;

    if (!engine.LoadModel("/tmp/qwen2.5-coder-1.5b-q4.gguf", config)) {
        printf("FAIL\n"); return 1;
    }

    // Check embedding weights - read first few values
    half embd_cpu[16];
    cudaMemcpy(embd_cpu, engine.weights_.tok_embd, 16 * sizeof(half), cudaMemcpyDeviceToHost);
    printf("token_embd first 8 values: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", __half2float(embd_cpu[i]));
    printf("\n");

    // Check if all zeros
    int nonzero = 0;
    half big_buf[1536];
    cudaMemcpy(big_buf, engine.weights_.tok_embd, 1536 * sizeof(half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 1536; i++) if (__half2float(big_buf[i]) != 0.0f) nonzero++;
    printf("token_embd row 0: %d/1536 nonzero\n", nonzero);

    // Check attn_norm (should be F32 weights, ~1.0)
    if (engine.weights_.layers[0].attn_norm) {
        half norm_cpu[16];
        cudaMemcpy(norm_cpu, engine.weights_.layers[0].attn_norm, 16 * sizeof(half), cudaMemcpyDeviceToHost);
        printf("blk.0.attn_norm first 8: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", __half2float(norm_cpu[i]));
        printf("\n");
    } else {
        printf("blk.0.attn_norm: NULL\n");
    }

    // Check Q4_K weight data (wq)
    if (engine.weights_.layers[0].wq) {
        Q4KBlock blk_cpu;
        cudaMemcpy(&blk_cpu, engine.weights_.layers[0].wq, sizeof(Q4KBlock), cudaMemcpyDeviceToHost);
        printf("blk.0.attn_q first Q4K block: d=%.6f dmin=%.6f\n",
               __half2float(blk_cpu.d), __half2float(blk_cpu.dmin));
        printf("  scales: ");
        for (int i = 0; i < 12; i++) printf("%d ", blk_cpu.scales[i]);
        printf("\n  qs[0..7]: ");
        for (int i = 0; i < 8; i++) printf("%02x ", blk_cpu.qs[i]);
        printf("\n");
    } else {
        printf("blk.0.attn_q.weight: NULL\n");
    }

    return 0;
}
