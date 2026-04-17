// infergo C SDK — Chat example
// Build: gcc chat.c -linfer_api -o chat
#include "infer_api.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [prompt]\n", argv[0]);
        return 1;
    }
    const char* prompt = argc > 2 ? argv[2] : "Hello, how are you?";

    InferLLM llm = infer_llm_create(argv[1], -1, 4096, 1, 2048);
    if (!llm) { fprintf(stderr, "Load failed: %s\n", infer_last_error_string()); return 1; }

    int tokens[512];
    int n = infer_llm_tokenize(llm, prompt, 1, tokens, 512);
    if (n < 0) { fprintf(stderr, "Tokenize failed\n"); infer_llm_destroy(llm); return 1; }

    char output[8192];
    int gen = 0;
    int rc = infer_llm_generate(llm, tokens, n, 128, 0.7f, 0.9f, NULL, NULL, NULL, output, sizeof(output), &gen);
    if (rc == 0) printf("%s\n", output);
    else fprintf(stderr, "Generate failed: %s\n", infer_last_error_string());

    infer_llm_destroy(llm);
    return 0;
}
