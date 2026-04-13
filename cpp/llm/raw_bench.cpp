// Raw C++ benchmark — bypass Go, measure pure llama.cpp perf
#include "infer_api.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#include <numeric>
#include <algorithm>

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "/tmp/qwen2.5-coder-1.5b-q4.gguf";
    const int N = 20;
    const int max_tokens = 64;

    printf("Loading model: %s\n", model_path);
    auto t0 = std::chrono::high_resolution_clock::now();

    InferLLM llm = infer_llm_create(model_path, -1, 4096, 1, 2048);
    if (!llm) {
        printf("Failed to load: %s\n", infer_last_error_string());
        return 1;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("Model loaded in %.0fms\n", load_ms);

    // Tokenize prompt (ChatML format)
    const char* prompt = "<|im_start|>system\nYou are a helpful assistant. Output valid JSON only.<|im_end|>\n<|im_start|>user\nReturn a JSON object with fields: name, age, city, occupation, hobbies (array of 3).<|im_end|>\n<|im_start|>assistant\n";

    int tokens[512];
    int n_tokens = infer_llm_tokenize(llm, prompt, 0, tokens, 512);
    printf("Prompt tokens: %d\n", n_tokens);

    // Benchmark: raw C++ generation, no Go, no HTTP
    std::vector<double> latencies;
    std::vector<int> tok_counts;
    char buf[4096];
    int gen_tokens;

    printf("Running %d raw C++ requests (max_tokens=%d)...\n", N, max_tokens);
    for (int i = 0; i < N; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        int rc = infer_llm_generate(llm, tokens, n_tokens, max_tokens,
                                     0.7f, 0.9f, nullptr, nullptr, nullptr,
                                     buf, sizeof(buf), &gen_tokens);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        if (rc != 0) {
            printf("  req %d: FAILED (%s)\n", i, infer_last_error_string());
            continue;
        }
        latencies.push_back(ms);
        tok_counts.push_back(gen_tokens);
        if (i == 0) printf("  Sample (%d tok): %.80s...\n", gen_tokens, buf);
    }

    if (latencies.empty()) {
        printf("All requests failed!\n");
        infer_llm_destroy(llm);
        return 1;
    }

    double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / (double)latencies.size();
    double avg_tok = std::accumulate(tok_counts.begin(), tok_counts.end(), 0.0) / (double)tok_counts.size();
    std::sort(latencies.begin(), latencies.end());
    double p50 = latencies[latencies.size() / 2];
    double mn = latencies.front();

    printf("\n=== RAW C++ (no Go, no HTTP) ===\n");
    printf("  Avg: %.1fms  P50: %.1fms  Min: %.1fms\n", avg, p50, mn);
    printf("  Avg tokens: %.1f\n", avg_tok);
    printf("  ms/tok: %.2f  tok/s: %.1f\n", avg / avg_tok, avg_tok / (avg / 1000.0));

    infer_llm_destroy(llm);
    return 0;
}
