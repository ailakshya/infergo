// Full benchmark: Custom CUDA Engine vs llama.cpp (via infer_api)
#include "engine.cuh"
#include "infer_api.h"
#include <chrono>
#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace infergo::cuda;

void bench_custom_engine(const char* model_path) {
    printf("\n=== Custom CUDA Engine ===\n");

    CUDAEngine engine;
    ModelConfig config;
    // Qwen 1.5B
    config.n_embd = 1536;
    config.n_head = 12;
    config.n_kv_head = 2;
    config.n_layer = 28;
    config.n_ff = 8960;
    config.head_dim = 128;
    config.n_vocab = 151936;
    config.n_ctx = 4096;

    if (!engine.LoadModel(model_path, config)) {
        printf("FAIL: model load\n");
        return;
    }

    // Simple prompt tokens (BOS + "Hi")
    // We can't tokenize without llama.cpp, so use raw token IDs
    int prompt[] = {151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 13048, 151645, 198, 151644, 77091, 198};
    int n_prompt = 20;
    int max_tokens = 16;

    // Warmup
    printf("  Warming up...\n");
    int out_tokens[64];
    engine.Generate(prompt, n_prompt, 4, 0.0f, out_tokens, 64);

    // Benchmark
    printf("  Benchmarking (%d tokens decode)...\n", max_tokens);
    std::vector<double> lats;
    for (int i = 0; i < 10; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        int gen = engine.Generate(prompt, n_prompt, max_tokens, 0.0f, out_tokens, 64);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        lats.push_back(ms);
        if (i == 0) printf("  Generated %d tokens\n", gen);
    }

    std::sort(lats.begin(), lats.end());
    double avg = std::accumulate(lats.begin(), lats.end(), 0.0) / lats.size();
    printf("  Avg: %.1fms  P50: %.1fms  Min: %.1fms\n", avg, lats[5], lats[0]);
    printf("  ms/tok: %.2f  tok/s: %.1f\n",
           avg / max_tokens, max_tokens / (avg / 1000.0));
}

void bench_llamacpp(const char* model_path) {
    printf("\n=== llama.cpp (via infer_api) ===\n");

    InferLLM llm = infer_llm_create(model_path, -1, 4096, 1, 2048);
    if (!llm) {
        printf("FAIL: %s\n", infer_last_error_string());
        return;
    }

    const char* prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
    int tokens[512];
    int n_tok = infer_llm_tokenize(llm, prompt, 0, tokens, 512);
    int max_tokens = 16;

    // Warmup
    printf("  Warming up...\n");
    char buf[4096]; int gen;
    infer_llm_generate(llm, tokens, n_tok, 4, 0.0f, 0.9f, nullptr, nullptr, nullptr, buf, 4096, &gen);

    // Benchmark
    printf("  Benchmarking (%d tokens decode)...\n", max_tokens);
    std::vector<double> lats;
    for (int i = 0; i < 10; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        infer_llm_generate(llm, tokens, n_tok, max_tokens, 0.0f, 0.9f, nullptr, nullptr, nullptr, buf, 4096, &gen);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        lats.push_back(ms);
        if (i == 0) printf("  Generated %d tokens: %.30s...\n", gen, buf);
    }

    std::sort(lats.begin(), lats.end());
    double avg = std::accumulate(lats.begin(), lats.end(), 0.0) / lats.size();
    printf("  Avg: %.1fms  P50: %.1fms  Min: %.1fms\n", avg, lats[5], lats[0]);
    printf("  ms/tok: %.2f  tok/s: %.1f\n",
           avg / max_tokens, max_tokens / (avg / 1000.0));

    infer_llm_destroy(llm);
}

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "/tmp/qwen2.5-coder-1.5b-q4.gguf";

    printf("============================================================\n");
    printf("  infergo CUDA Engine vs llama.cpp — Head to Head\n");
    printf("  Model: %s\n", model_path);
    printf("============================================================\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d, %d MB)\n", prop.name, prop.major, prop.minor,
           (int)(prop.totalGlobalMem / 1024 / 1024));

    bench_llamacpp(model_path);
    bench_custom_engine(model_path);

    printf("\n============================================================\n");
    return 0;
}
