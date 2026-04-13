#include "engine.cuh"
#include "infer_api.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace infergo::cuda;

int main() {
    const char* model_path = "/tmp/qwen2.5-coder-1.5b-q4.gguf";
    InferLLM llm = infer_llm_create(model_path, -1, 4096, 1, 2048);
    if (!llm) { printf("FAIL\n"); return 1; }

    const char* prompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
    int tokens[256];
    int n_tok = infer_llm_tokenize(llm, prompt, 0, tokens, 256);

    printf("=== llama.cpp ===\n");
    char buf[4096]; int gen;
    infer_llm_generate(llm, tokens, n_tok, 8, 0.0f, 0.9f, nullptr, nullptr, nullptr, buf, 4096, &gen);
    printf("  %d tok: %s\n", gen, buf);

    printf("\n=== Custom Engine ===\n");
    CUDAEngine engine;
    ModelConfig config;
    config.n_embd = 1536; config.n_head = 12; config.n_kv_head = 2;
    config.n_layer = 28; config.n_ff = 8960; config.head_dim = 128;
    config.n_vocab = 151936; config.n_ctx = 256;

    if (!engine.LoadModel(model_path, config)) { printf("FAIL\n"); return 1; }

    // Run one forward pass manually and check intermediate values
    int out_tokens[64];
    int n_gen = engine.Generate(tokens, n_tok, 8, 0.0f, out_tokens, 64);
    printf("  %d tok, IDs: ", n_gen);
    for (int i = 0; i < n_gen; i++) {
        char piece[256];
        int pn = infer_llm_token_to_piece(llm, out_tokens[i], piece, 256);
        if (pn > 0) { piece[pn] = 0; printf("[%d=%s] ", out_tokens[i], piece); }
        else printf("[%d] ", out_tokens[i]);
    }
    printf("\n");

    infer_llm_destroy(llm);
    return 0;
}

// Check CUDA error after the fact
void check_cuda() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
    }
}
