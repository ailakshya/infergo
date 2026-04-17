// infergo WASM wrapper — Emscripten bindings
// CPU-only inference (no GPU in browser)
#include <emscripten.h>
#include "infer_api.h"
#include <string.h>
#include <stdlib.h>

// ── Error ───────────────────────────────────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
const char* wasm_last_error(void) {
    return infer_last_error_string();
}

// ── Memory helpers for JS ───────────────────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
void* wasm_malloc(int size) { return malloc(size); }

EMSCRIPTEN_KEEPALIVE
void wasm_free(void* ptr) { free(ptr); }

// ── LLM ─────────────────────────────────────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
void* wasm_llm_create(const char* path, int gpu_layers, int ctx_size,
                       int n_seq_max, int n_batch) {
    return infer_llm_create(path, 0, ctx_size,
                             n_seq_max > 0 ? n_seq_max : 1,
                             n_batch > 0 ? n_batch : 512);
}

EMSCRIPTEN_KEEPALIVE
void wasm_llm_destroy(void* llm) {
    infer_llm_destroy(llm);
}

EMSCRIPTEN_KEEPALIVE
int wasm_llm_vocab_size(void* llm) {
    return infer_llm_vocab_size(llm);
}

EMSCRIPTEN_KEEPALIVE
int wasm_llm_bos(void* llm) { return infer_llm_bos(llm); }

EMSCRIPTEN_KEEPALIVE
int wasm_llm_eos(void* llm) { return infer_llm_eos(llm); }

EMSCRIPTEN_KEEPALIVE
int wasm_llm_is_eog(void* llm, int token) { return infer_llm_is_eog(llm, token); }

EMSCRIPTEN_KEEPALIVE
int wasm_llm_tokenize(void* llm, const char* text, int add_bos,
                       int* out_ids, int max_tokens) {
    return infer_llm_tokenize(llm, text, add_bos, out_ids, max_tokens);
}

EMSCRIPTEN_KEEPALIVE
int wasm_llm_token_to_piece(void* llm, int token, char* out, int buf_size) {
    return infer_llm_token_to_piece(llm, token, out, buf_size);
}

EMSCRIPTEN_KEEPALIVE
int wasm_llm_generate(void* llm, const int* tokens, int n_prompt,
                       int max_tokens, float temperature, float top_p,
                       const char* grammar, char* out_text, int max_text_len,
                       int* out_gen_tokens) {
    return infer_llm_generate(llm, tokens, n_prompt, max_tokens,
                               temperature, top_p, grammar,
                               NULL, NULL, out_text, max_text_len, out_gen_tokens);
}

// Convenience: tokenize + generate in one call (simpler JS interop)
EMSCRIPTEN_KEEPALIVE
char* wasm_llm_generate_text(void* llm, const char* prompt,
                              int max_tokens, float temperature) {
    int tokens[4096];
    int n = infer_llm_tokenize(llm, prompt, 1, tokens, 4096);
    if (n < 0) return NULL;

    static char buf[32768];
    int gen = 0;
    int rc = infer_llm_generate(llm, tokens, n, max_tokens, temperature, 0.9f,
                                 NULL, NULL, NULL, buf, sizeof(buf), &gen);
    if (rc < 0) return NULL;
    return buf;
}

// ── Tokenizer ───────────────────────────────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
void* wasm_tokenizer_load(const char* path) {
    return infer_tokenizer_load(path);
}

EMSCRIPTEN_KEEPALIVE
void wasm_tokenizer_destroy(void* tok) {
    infer_tokenizer_destroy(tok);
}

EMSCRIPTEN_KEEPALIVE
int wasm_tokenizer_encode(void* tok, const char* text, int add_special,
                           int* out_ids, int* out_mask, int max_tokens) {
    return infer_tokenizer_encode(tok, text, add_special, out_ids, out_mask, max_tokens);
}

EMSCRIPTEN_KEEPALIVE
int wasm_tokenizer_decode(void* tok, const int* ids, int n_ids,
                           int skip_special, char* out_buf, int buf_size) {
    return infer_tokenizer_decode(tok, ids, n_ids, skip_special, out_buf, buf_size);
}

EMSCRIPTEN_KEEPALIVE
int wasm_tokenizer_vocab_size(void* tok) {
    return infer_tokenizer_vocab_size(tok);
}

// ── ONNX Session (CPU only) ────────────────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
void* wasm_session_create(void) {
    return infer_session_create("cpu", 0);
}

EMSCRIPTEN_KEEPALIVE
int wasm_session_load(void* s, const char* path) {
    return infer_session_load(s, path);
}

EMSCRIPTEN_KEEPALIVE
void wasm_session_destroy(void* s) {
    infer_session_destroy(s);
}

EMSCRIPTEN_KEEPALIVE
int wasm_session_num_inputs(void* s) { return infer_session_num_inputs(s); }

EMSCRIPTEN_KEEPALIVE
int wasm_session_num_outputs(void* s) { return infer_session_num_outputs(s); }

// ── Embedding Pipeline ─────────────────────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
int wasm_embed_pipeline(void* session, void* tokenizer, const char* text,
                         float* out_vec, int max_dim) {
    return infer_embed_pipeline(session, tokenizer, text, out_vec, max_dim);
}

// ── Vector Database ─────────────────────────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
void* wasm_vectordb_create(int dim, int M, int ef) {
    return infer_vectordb_create(dim, M, ef);
}

EMSCRIPTEN_KEEPALIVE
int wasm_vectordb_insert(void* db, int id, const float* vec, const char* meta) {
    return infer_vectordb_insert(db, (int64_t)id, vec, meta);
}

EMSCRIPTEN_KEEPALIVE
int wasm_vectordb_delete(void* db, int id) {
    return infer_vectordb_delete(db, (int64_t)id);
}

EMSCRIPTEN_KEEPALIVE
int wasm_vectordb_search(void* db, const float* query, int k, int ef_search,
                          const char* filter,
                          int64_t* out_ids, float* out_dists, int max_results) {
    return infer_vectordb_search(db, query, k, ef_search, filter,
                                  out_ids, out_dists, max_results);
}

EMSCRIPTEN_KEEPALIVE
int wasm_vectordb_size(void* db) { return infer_vectordb_size(db); }

EMSCRIPTEN_KEEPALIVE
int wasm_vectordb_save(void* db, const char* path) {
    return infer_vectordb_save(db, path);
}

EMSCRIPTEN_KEEPALIVE
int wasm_vectordb_load(void* db, const char* path) {
    return infer_vectordb_load(db, path);
}

EMSCRIPTEN_KEEPALIVE
void wasm_vectordb_free(void* db) {
    infer_vectordb_free(db);
}

// ── BM25 ────────────────────────────────────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
void* wasm_bm25_create(float k1, float b) {
    return infer_bm25_create(k1, b);
}

EMSCRIPTEN_KEEPALIVE
void wasm_bm25_insert(void* idx, int id, const char* text) {
    infer_bm25_insert(idx, (int64_t)id, text);
}

EMSCRIPTEN_KEEPALIVE
void wasm_bm25_remove(void* idx, int id) {
    infer_bm25_remove(idx, (int64_t)id);
}

EMSCRIPTEN_KEEPALIVE
int wasm_bm25_search(void* idx, const char* query, int k,
                      int64_t* out_ids, float* out_scores, int max_results) {
    return infer_bm25_search(idx, query, k, out_ids, out_scores, max_results);
}

EMSCRIPTEN_KEEPALIVE
int wasm_bm25_size(void* idx) { return infer_bm25_size(idx); }

EMSCRIPTEN_KEEPALIVE
void wasm_bm25_free(void* idx) {
    infer_bm25_free(idx);
}

// ── TOON ────────────────────────────────────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
const char* wasm_toon_grammar(void) { return infer_toon_grammar(); }

EMSCRIPTEN_KEEPALIVE
int wasm_toon_to_json(const char* toon, int len, char* out, int max) {
    return infer_toon_to_json(toon, len, out, max);
}

EMSCRIPTEN_KEEPALIVE
int wasm_json_to_toon(const char* json, int len, char* out, int max) {
    return infer_json_to_toon(json, len, out, max);
}
