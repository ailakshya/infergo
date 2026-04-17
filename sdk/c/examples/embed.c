// infergo C SDK — Embedding example
#include "infer_api.h"
#include <stdio.h>
#include <math.h>

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s <model.onnx> <tokenizer.json> [text]\n", argv[0]); return 1; }
    InferSession s = infer_session_create("cpu", 0);
    infer_session_load(s, argv[1]);
    InferTokenizer tok = infer_tokenizer_load(argv[2]);
    const char* text = argc > 3 ? argv[3] : "Hello world";
    float vec[1024];
    int dim = infer_embed_pipeline(s, tok, text, vec, 1024);
    if (dim > 0) {
        printf("dim=%d first=[%.4f, %.4f, %.4f]\n", dim, vec[0], vec[1], vec[2]);
    }
    infer_tokenizer_destroy(tok);
    infer_session_destroy(s);
    return 0;
}
