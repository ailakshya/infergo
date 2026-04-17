// infergo C SDK — RAG Pipeline example
// Build: gcc rag.c -linfer_api -o rag
#include "infer_api.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <llm.gguf> <embed.onnx> <tokenizer.json>\n", argv[0]);
        return 1;
    }

    // Load LLM
    InferLLM llm = infer_llm_create(argv[1], -1, 4096, 1, 2048);
    if (!llm) {
        fprintf(stderr, "LLM load failed: %s\n", infer_last_error_string());
        return 1;
    }

    // Load embedding model
    InferSession embed = infer_session_create("cpu", 0);
    infer_session_load(embed, argv[2]);
    InferTokenizer tok = infer_tokenizer_load(argv[3]);

    // Create vector DB
    InferVectorDB db = infer_vectordb_create(384, 16, 200);

    // Ingest documents
    const char* docs[] = {
        "infergo is a production AI inference runtime written in Go and C++.",
        "It supports LLM, embedding, detection, and RAG pipelines.",
        "infergo achieves zero overhead compared to raw llama.cpp performance.",
        "Vector search uses HNSW with BM25 hybrid fusion.",
        "The server exposes an OpenAI-compatible REST API on port 9090.",
    };
    int n_docs = 5;

    float vecs[5 * 384];
    infer_embed_batch_pipeline(embed, tok, docs, n_docs, vecs, 384);
    for (int i = 0; i < n_docs; i++) {
        infer_vectordb_insert(db, i, &vecs[i * 384], docs[i]);
    }
    printf("Ingested %d documents\n", n_docs);

    // RAG query
    char answer[4096];
    int len = infer_rag_pipeline(llm, embed, tok, db, "What is infergo?",
                                  3, 256, 0.7f, answer, sizeof(answer));
    if (len > 0) {
        printf("\nQuery: What is infergo?\nAnswer: %s\n", answer);
    }

    infer_vectordb_free(db);
    infer_tokenizer_destroy(tok);
    infer_session_destroy(embed);
    infer_llm_destroy(llm);
    return 0;
}
