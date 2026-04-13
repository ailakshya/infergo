// Standalone test for infer_llm_generate_batch
#include "infer_api.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    printf("=== Batch Generation Test ===\n");

    InferLLM llm = infer_llm_create("/tmp/qwen2.5-coder-1.5b-q4.gguf", -1, 8192, 8, 2048);
    if (!llm) {
        printf("FAIL: %s\n", infer_last_error_string());
        return 1;
    }
    printf("Model loaded\n");

    // Tokenize prompts
    const char* prompts[] = {
        "<|im_start|>user\nSay hello<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nSay goodbye<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nCount to 3<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is 2+2<|im_end|>\n<|im_start|>assistant\n",
    };

    const int N = 4;
    std::vector<int> all_tokens;
    int offsets[N + 1];

    for (int i = 0; i < N; i++) {
        int toks[256];
        int n = infer_llm_tokenize(llm, prompts[i], 0, toks, 256);
        if (n <= 0) {
            printf("FAIL: tokenize prompt %d failed\n", i);
            return 1;
        }
        offsets[i] = static_cast<int>(all_tokens.size());
        for (int j = 0; j < n; j++) {
            all_tokens.push_back(toks[j]);
        }
        printf("Prompt %d: %d tokens\n", i, n);
    }
    offsets[N] = static_cast<int>(all_tokens.size());
    printf("Total tokens: %d\n", static_cast<int>(all_tokens.size()));

    // Test 1: Single request via batch API
    printf("\n--- Test 1: Single request via batch API ---\n");
    {
        char buf[4096];
        char* out_ptrs[] = {buf};
        int gen_toks[1];
        int single_offsets[] = {offsets[0], offsets[1]};

        int rc = infer_llm_generate_batch(llm, 1,
            all_tokens.data(), single_offsets,
            8, 0.7f, 0.9f, nullptr,
            out_ptrs, 4096, gen_toks);

        if (rc != 0) {
            printf("FAIL: %s\n", infer_last_error_string());
        } else {
            printf("OK: %d tokens: %s\n", gen_toks[0], buf);
        }
    }

    // Test 2: Two requests batched
    printf("\n--- Test 2: Two requests batched ---\n");
    {
        char buf0[4096], buf1[4096];
        char* out_ptrs[] = {buf0, buf1};
        int gen_toks[2];
        int two_offsets[] = {offsets[0], offsets[1], offsets[2]};

        int rc = infer_llm_generate_batch(llm, 2,
            all_tokens.data(), two_offsets,
            8, 0.7f, 0.9f, nullptr,
            out_ptrs, 4096, gen_toks);

        if (rc != 0) {
            printf("FAIL: %s\n", infer_last_error_string());
        } else {
            printf("OK[0]: %d tokens: %s\n", gen_toks[0], buf0);
            printf("OK[1]: %d tokens: %s\n", gen_toks[1], buf1);
        }
    }

    // Test 3: Four requests batched + timing
    printf("\n--- Test 3: Four requests batched ---\n");
    {
        char buf0[4096], buf1[4096], buf2[4096], buf3[4096];
        char* out_ptrs[] = {buf0, buf1, buf2, buf3};
        int gen_toks[4];

        auto t0 = std::chrono::high_resolution_clock::now();
        int rc = infer_llm_generate_batch(llm, 4,
            all_tokens.data(), offsets,
            16, 0.7f, 0.9f, nullptr,
            out_ptrs, 4096, gen_toks);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (rc != 0) {
            printf("FAIL: %s\n", infer_last_error_string());
        } else {
            int total_tok = 0;
            for (int i = 0; i < 4; i++) {
                printf("OK[%d]: %d tokens: %s\n", i, gen_toks[i], out_ptrs[i]);
                total_tok += gen_toks[i];
            }
            printf("Time: %.1fms for 4 requests (%d total tokens)\n", ms, total_tok);
            printf("Throughput: %.1f tok/s, %.1f rps\n",
                total_tok / (ms / 1000.0), 4.0 / (ms / 1000.0));
        }
    }

    // Test 4: Compare batch vs sequential
    printf("\n--- Test 4: Sequential 4x for comparison ---\n");
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        int total_tok = 0;
        for (int i = 0; i < 4; i++) {
            char buf[4096];
            int gen;
            int prompt_off = offsets[i];
            int prompt_len = offsets[i+1] - offsets[i];
            int rc = infer_llm_generate(llm,
                all_tokens.data() + prompt_off, prompt_len,
                16, 0.7f, 0.9f, nullptr, nullptr, nullptr,
                buf, 4096, &gen);
            if (rc != 0) {
                printf("FAIL[%d]: %s\n", i, infer_last_error_string());
            } else {
                total_tok += gen;
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("Time: %.1fms sequential (%d total tokens)\n", ms, total_tok);
        printf("Throughput: %.1f tok/s, %.1f rps\n",
            total_tok / (ms / 1000.0), 4.0 / (ms / 1000.0));
    }

    infer_llm_destroy(llm);
    printf("\n=== Done ===\n");
    return 0;
}
