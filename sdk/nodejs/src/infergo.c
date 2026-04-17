/* sdk/nodejs/src/infergo.c — N-API native addon wrapping libinfer_api */

#include <node_api.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Forward declarations from infer_api.h — we use void* handles */
typedef void* InferLLM;
typedef void* InferSession;
typedef void* InferTokenizer;
typedef void* InferVectorDB;
typedef void* InferBM25;

/* Token callback signature */
typedef int (*InferTokenCallback)(int token, const char* piece, void* user_data);

/* C API imports */
extern const char* infer_last_error_string(void);

extern InferLLM infer_llm_create(const char* path, int n_gpu_layers, int ctx_size,
                                  int n_seq_max, int n_batch);
extern void infer_llm_destroy(InferLLM llm);
extern int  infer_llm_tokenize(InferLLM llm, const char* text, int add_bos,
                                int* out_ids, int max_tokens);
extern int  infer_llm_generate(InferLLM llm, const int* prompt_tokens, int n_prompt,
                                int max_tokens, float temperature, float top_p,
                                const char* grammar, InferTokenCallback callback,
                                void* user_data, char* out_text, int max_text_len,
                                int* out_gen_tokens);
extern int  infer_llm_vocab_size(InferLLM llm);
extern int  infer_llm_bos(InferLLM llm);
extern int  infer_llm_eos(InferLLM llm);
extern int  infer_llm_token_to_piece(InferLLM llm, int token, char* out_buf, int buf_size);

extern InferSession infer_session_create(const char* provider, int device_id);
extern int  infer_session_load(InferSession s, const char* model_path);
extern void infer_session_destroy(InferSession s);

extern InferTokenizer infer_tokenizer_load(const char* path);
extern void infer_tokenizer_destroy(InferTokenizer tok);
extern int  infer_tokenizer_encode(InferTokenizer tok, const char* text,
                                    int add_special_tokens, int* out_ids,
                                    int* out_mask, int max_tokens);
extern int  infer_tokenizer_decode(InferTokenizer tok, const int* ids, int n_ids,
                                    int skip_special_tokens, char* out_buf, int buf_size);
extern int  infer_tokenizer_vocab_size(InferTokenizer tok);

extern int  infer_embed_pipeline(InferSession session, InferTokenizer tokenizer,
                                  const char* text, float* out_vec, int max_dim);
extern int  infer_embed_batch_pipeline(InferSession session, InferTokenizer tokenizer,
                                        const char** texts, int n_texts,
                                        float* out_vecs, int max_dim);

extern InferVectorDB infer_vectordb_create(int dim, int M, int ef_construction);
extern int  infer_vectordb_insert(InferVectorDB db, int64_t id, const float* vec,
                                   const char* metadata);
extern int  infer_vectordb_delete(InferVectorDB db, int64_t id);
extern int  infer_vectordb_search(InferVectorDB db, const float* query, int k,
                                   int ef_search, const char* metadata_filter,
                                   int64_t* out_ids, float* out_distances, int max_results);
extern int  infer_vectordb_size(InferVectorDB db);
extern int  infer_vectordb_save(InferVectorDB db, const char* path);
extern int  infer_vectordb_load(InferVectorDB db, const char* path);
extern void infer_vectordb_free(InferVectorDB db);

extern InferBM25 infer_bm25_create(float k1, float b);
extern void infer_bm25_insert(InferBM25 idx, int64_t id, const char* text);
extern void infer_bm25_remove(InferBM25 idx, int64_t id);
extern int  infer_bm25_search(InferBM25 idx, const char* query, int k,
                               int64_t* out_ids, float* out_scores, int max_results);
extern int  infer_bm25_size(InferBM25 idx);
extern int  infer_bm25_save(InferBM25 idx, const char* path);
extern int  infer_bm25_load(InferBM25 idx, const char* path);
extern void infer_bm25_free(InferBM25 idx);

/* ─────────────────────────────────────────────────────────────────────────────
 * Helpers
 * ───────────────────────────────────────────────────────────────────────────── */

#define NAPI_CALL(env, call)                                       \
    do {                                                           \
        napi_status _s = (call);                                   \
        if (_s != napi_ok) {                                       \
            const napi_extended_error_info* _ei;                   \
            napi_get_last_error_info((env), &_ei);                 \
            napi_throw_error((env), NULL,                           \
                _ei->error_message ? _ei->error_message : "N-API error"); \
            return NULL;                                           \
        }                                                          \
    } while (0)

static napi_value throw_infer_error(napi_env env) {
    const char* msg = infer_last_error_string();
    napi_throw_error(env, NULL, msg ? msg : "unknown infergo error");
    return NULL;
}

/* Handle wrapper to prevent double-free between destroy() and GC destructor */
typedef struct { void* ptr; } handle_box_t;

static handle_box_t* alloc_box(void* ptr) {
    handle_box_t* box = (handle_box_t*)malloc(sizeof(handle_box_t));
    if (box) box->ptr = ptr;
    return box;
}

/* Extract a void* handle from a napi_external (via handle_box_t) */
static void* unwrap_handle(napi_env env, napi_value val) {
    void* data = NULL;
    napi_get_value_external(env, val, &data);
    if (!data) return NULL;
    return ((handle_box_t*)data)->ptr;
}

/* Clear handle in box (prevents double-free from GC destructor) */
static void clear_handle(napi_env env, napi_value val) {
    void* data = NULL;
    napi_get_value_external(env, val, &data);
    if (data) ((handle_box_t*)data)->ptr = NULL;
}

/* Get a C string from a JS string argument. Caller must free() the result. */
static char* get_string_arg(napi_env env, napi_value val) {
    size_t len = 0;
    napi_get_value_string_utf8(env, val, NULL, 0, &len);
    char* buf = (char*)malloc(len + 1);
    if (!buf) return NULL;
    napi_get_value_string_utf8(env, val, buf, len + 1, &len);
    return buf;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * LLM bindings
 * ───────────────────────────────────────────────────────────────────────────── */

static void _llm_destructor(napi_env env, void* data, void* hint) {
    (void)env; (void)hint;
    handle_box_t* box = (handle_box_t*)data;
    if (box) { if (box->ptr) infer_llm_destroy((InferLLM)box->ptr); free(box); }
}

/* llmCreate(path, gpuLayers, ctxSize, seqMax, batch) → external */
static napi_value napi_llm_create(napi_env env, napi_callback_info info) {
    size_t argc = 5;
    napi_value argv[5];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));

    char* path = get_string_arg(env, argv[0]);
    int32_t gpu_layers, ctx_size, seq_max, batch;
    napi_get_value_int32(env, argv[1], &gpu_layers);
    napi_get_value_int32(env, argv[2], &ctx_size);
    napi_get_value_int32(env, argv[3], &seq_max);
    napi_get_value_int32(env, argv[4], &batch);

    InferLLM llm = infer_llm_create(path, gpu_layers, ctx_size, seq_max, batch);
    free(path);

    if (!llm) return throw_infer_error(env);

    handle_box_t* box = alloc_box(llm);
    napi_value ext;
    NAPI_CALL(env, napi_create_external(env, box, _llm_destructor, NULL, &ext));
    return ext;
}

/* llmDestroy(handle) */
static napi_value napi_llm_destroy(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
    void* ptr = unwrap_handle(env, argv[0]);
    if (ptr) infer_llm_destroy((InferLLM)ptr);
    clear_handle(env, argv[0]);  /* prevent double-free from GC destructor */
    return NULL;
}

/* llmTokenize(handle, text, addBos) → Int32Array */
static napi_value napi_llm_tokenize(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value argv[3];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));

    InferLLM llm = (InferLLM)unwrap_handle(env, argv[0]);
    char* text = get_string_arg(env, argv[1]);
    int32_t add_bos;
    napi_get_value_int32(env, argv[2], &add_bos);

    int ids[8192];
    int n = infer_llm_tokenize(llm, text, add_bos, ids, 8192);
    free(text);

    if (n < 0) return throw_infer_error(env);

    napi_value result;
    NAPI_CALL(env, napi_create_array_with_length(env, (size_t)n, &result));
    for (int i = 0; i < n; i++) {
        napi_value v;
        napi_create_int32(env, ids[i], &v);
        napi_set_element(env, result, (uint32_t)i, v);
    }
    return result;
}

/* llmGenerate(handle, tokens, maxTokens, temperature, topP, grammar) → { text, numTokens } */
static napi_value napi_llm_generate(napi_env env, napi_callback_info info) {
    size_t argc = 6;
    napi_value argv[6];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));

    InferLLM llm = (InferLLM)unwrap_handle(env, argv[0]);

    /* argv[1] = Int32Array of token IDs */
    uint32_t n_tokens;
    napi_get_array_length(env, argv[1], &n_tokens);
    int* tokens = (int*)malloc(sizeof(int) * n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        napi_value el;
        napi_get_element(env, argv[1], i, &el);
        napi_get_value_int32(env, el, &tokens[i]);
    }

    int32_t max_tokens;
    napi_get_value_int32(env, argv[2], &max_tokens);
    double temperature, top_p;
    napi_get_value_double(env, argv[3], &temperature);
    napi_get_value_double(env, argv[4], &top_p);

    /* grammar — string or null */
    char* grammar = NULL;
    napi_valuetype vt;
    napi_typeof(env, argv[5], &vt);
    if (vt == napi_string) {
        grammar = get_string_arg(env, argv[5]);
    }

    char out_text[65536];
    int out_gen = 0;
    int rc = infer_llm_generate(llm, tokens, (int)n_tokens, max_tokens,
                                 (float)temperature, (float)top_p,
                                 grammar, NULL, NULL,
                                 out_text, (int)sizeof(out_text), &out_gen);
    free(tokens);
    free(grammar);

    if (rc != 0) return throw_infer_error(env);

    napi_value result, txt_val, ntok_val;
    NAPI_CALL(env, napi_create_object(env, &result));
    NAPI_CALL(env, napi_create_string_utf8(env, out_text, NAPI_AUTO_LENGTH, &txt_val));
    NAPI_CALL(env, napi_create_int32(env, out_gen, &ntok_val));

    napi_set_named_property(env, result, "text", txt_val);
    napi_set_named_property(env, result, "numTokens", ntok_val);
    return result;
}

/* llmVocabSize(handle) → number */
static napi_value napi_llm_vocab_size(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
    InferLLM llm = (InferLLM)unwrap_handle(env, argv[0]);
    napi_value result;
    NAPI_CALL(env, napi_create_int32(env, infer_llm_vocab_size(llm), &result));
    return result;
}

/* llmBos(handle) → number */
static napi_value napi_llm_bos(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
    InferLLM llm = (InferLLM)unwrap_handle(env, argv[0]);
    napi_value result;
    NAPI_CALL(env, napi_create_int32(env, infer_llm_bos(llm), &result));
    return result;
}

/* llmEos(handle) → number */
static napi_value napi_llm_eos(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
    InferLLM llm = (InferLLM)unwrap_handle(env, argv[0]);
    napi_value result;
    NAPI_CALL(env, napi_create_int32(env, infer_llm_eos(llm), &result));
    return result;
}

/* llmTokenToPiece(handle, token) → string */
static napi_value napi_llm_token_to_piece(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value argv[2];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
    InferLLM llm = (InferLLM)unwrap_handle(env, argv[0]);
    int32_t token;
    napi_get_value_int32(env, argv[1], &token);
    char buf[256];
    int rc = infer_llm_token_to_piece(llm, token, buf, (int)sizeof(buf));
    if (rc != 0) return throw_infer_error(env);
    napi_value result;
    NAPI_CALL(env, napi_create_string_utf8(env, buf, NAPI_AUTO_LENGTH, &result));
    return result;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Session (ONNX) bindings
 * ───────────────────────────────────────────────────────────────────────────── */

static void _session_destructor(napi_env env, void* data, void* hint) {
    (void)env; (void)hint;
    handle_box_t* box = (handle_box_t*)data;
    if (box) { if (box->ptr) infer_session_destroy((InferSession)box->ptr); free(box); }
}

/* sessionCreate(provider, deviceId) → external */
static napi_value napi_session_create(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value argv[2];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));

    char* provider = get_string_arg(env, argv[0]);
    int32_t device_id;
    napi_get_value_int32(env, argv[1], &device_id);

    InferSession s = infer_session_create(provider, device_id);
    free(provider);
    if (!s) return throw_infer_error(env);

    napi_value ext;
    handle_box_t* box = alloc_box(s);
    NAPI_CALL(env, napi_create_external(env, box, _session_destructor, NULL, &ext));
    return ext;
}

/* sessionLoad(handle, path) → void */
static napi_value napi_session_load(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value argv[2];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));

    InferSession s = (InferSession)unwrap_handle(env, argv[0]);
    char* path = get_string_arg(env, argv[1]);
    int rc = infer_session_load(s, path);
    free(path);
    if (rc != 0) return throw_infer_error(env);
    return NULL;
}

/* sessionDestroy(handle) */
static napi_value napi_session_destroy(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
    void* ptr = unwrap_handle(env, argv[0]);
    if (ptr) infer_session_destroy((InferSession)ptr);
    clear_handle(env, argv[0]);
    return NULL;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Tokenizer bindings
 * ───────────────────────────────────────────────────────────────────────────── */

static void _tokenizer_destructor(napi_env env, void* data, void* hint) {
    (void)env; (void)hint;
    handle_box_t* box = (handle_box_t*)data;
    if (box) { if (box->ptr) infer_tokenizer_destroy((InferTokenizer)box->ptr); free(box); }
}

/* tokenizerLoad(path) → external */
static napi_value napi_tokenizer_load(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));

    char* path = get_string_arg(env, argv[0]);
    InferTokenizer tok = infer_tokenizer_load(path);
    free(path);
    if (!tok) return throw_infer_error(env);

    napi_value ext;
    handle_box_t* box = alloc_box(tok);
    NAPI_CALL(env, napi_create_external(env, box, _tokenizer_destructor, NULL, &ext));
    return ext;
}

/* tokenizerEncode(handle, text, addSpecial) → number[] */
static napi_value napi_tokenizer_encode(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value argv[3];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));

    InferTokenizer tok = (InferTokenizer)unwrap_handle(env, argv[0]);
    char* text = get_string_arg(env, argv[1]);
    int32_t add_special;
    napi_get_value_int32(env, argv[2], &add_special);

    int ids[8192], mask[8192];
    int n = infer_tokenizer_encode(tok, text, add_special, ids, mask, 8192);
    free(text);
    if (n < 0) return throw_infer_error(env);

    napi_value result;
    NAPI_CALL(env, napi_create_array_with_length(env, (size_t)n, &result));
    for (int i = 0; i < n; i++) {
        napi_value v;
        napi_create_int32(env, ids[i], &v);
        napi_set_element(env, result, (uint32_t)i, v);
    }
    return result;
}

/* tokenizerDecode(handle, ids, skipSpecial) → string */
static napi_value napi_tokenizer_decode(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value argv[3];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));

    InferTokenizer tok = (InferTokenizer)unwrap_handle(env, argv[0]);

    uint32_t n_ids;
    napi_get_array_length(env, argv[1], &n_ids);
    int* ids = (int*)malloc(sizeof(int) * n_ids);
    for (uint32_t i = 0; i < n_ids; i++) {
        napi_value el;
        napi_get_element(env, argv[1], i, &el);
        napi_get_value_int32(env, el, &ids[i]);
    }

    int32_t skip_special;
    napi_get_value_int32(env, argv[2], &skip_special);

    char buf[65536];
    int rc = infer_tokenizer_decode(tok, ids, (int)n_ids, skip_special, buf, (int)sizeof(buf));
    free(ids);
    if (rc != 0) return throw_infer_error(env);

    napi_value result;
    NAPI_CALL(env, napi_create_string_utf8(env, buf, NAPI_AUTO_LENGTH, &result));
    return result;
}

/* tokenizerVocabSize(handle) → number */
static napi_value napi_tokenizer_vocab_size(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
    InferTokenizer tok = (InferTokenizer)unwrap_handle(env, argv[0]);
    napi_value result;
    NAPI_CALL(env, napi_create_int32(env, infer_tokenizer_vocab_size(tok), &result));
    return result;
}

/* tokenizerDestroy(handle) */
static napi_value napi_tokenizer_destroy(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
    void* ptr = unwrap_handle(env, argv[0]);
    if (ptr) infer_tokenizer_destroy((InferTokenizer)ptr);
    clear_handle(env, argv[0]);
    return NULL;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Embedding pipeline bindings
 * ───────────────────────────────────────────────────────────────────────────── */

/* embedPipeline(sessionHandle, tokenizerHandle, text, maxDim) → Float64Array */
static napi_value napi_embed_pipeline(napi_env env, napi_callback_info info) {
    size_t argc = 4;
    napi_value argv[4];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));

    InferSession session = (InferSession)unwrap_handle(env, argv[0]);
    InferTokenizer tokenizer = (InferTokenizer)unwrap_handle(env, argv[1]);
    char* text = get_string_arg(env, argv[2]);
    int32_t max_dim;
    napi_get_value_int32(env, argv[3], &max_dim);

    float* vec = (float*)malloc(sizeof(float) * max_dim);
    int dim = infer_embed_pipeline(session, tokenizer, text, vec, max_dim);
    free(text);

    if (dim < 0) {
        free(vec);
        return throw_infer_error(env);
    }

    napi_value result;
    NAPI_CALL(env, napi_create_array_with_length(env, (size_t)dim, &result));
    for (int i = 0; i < dim; i++) {
        napi_value v;
        napi_create_double(env, (double)vec[i], &v);
        napi_set_element(env, result, (uint32_t)i, v);
    }
    free(vec);
    return result;
}

/* embedBatchPipeline(sessionHandle, tokenizerHandle, texts, maxDim) → number[][] */
static napi_value napi_embed_batch_pipeline(napi_env env, napi_callback_info info) {
    size_t argc = 4;
    napi_value argv[4];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));

    InferSession session = (InferSession)unwrap_handle(env, argv[0]);
    InferTokenizer tokenizer = (InferTokenizer)unwrap_handle(env, argv[1]);

    uint32_t n_texts;
    napi_get_array_length(env, argv[2], &n_texts);
    char** texts = (char**)malloc(sizeof(char*) * n_texts);
    for (uint32_t i = 0; i < n_texts; i++) {
        napi_value el;
        napi_get_element(env, argv[2], i, &el);
        texts[i] = get_string_arg(env, el);
    }

    int32_t max_dim;
    napi_get_value_int32(env, argv[3], &max_dim);

    float* vecs = (float*)malloc(sizeof(float) * n_texts * max_dim);
    int dim = infer_embed_batch_pipeline(session, tokenizer,
                                          (const char**)texts, (int)n_texts,
                                          vecs, max_dim);
    for (uint32_t i = 0; i < n_texts; i++) free(texts[i]);
    free(texts);

    if (dim < 0) {
        free(vecs);
        return throw_infer_error(env);
    }

    napi_value result;
    NAPI_CALL(env, napi_create_array_with_length(env, (size_t)n_texts, &result));
    for (uint32_t i = 0; i < n_texts; i++) {
        napi_value row;
        napi_create_array_with_length(env, (size_t)dim, &row);
        for (int j = 0; j < dim; j++) {
            napi_value v;
            napi_create_double(env, (double)vecs[i * max_dim + j], &v);
            napi_set_element(env, row, (uint32_t)j, v);
        }
        napi_set_element(env, result, i, row);
    }
    free(vecs);
    return result;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * VectorDB bindings
 * ───────────────────────────────────────────────────────────────────────────── */

static void _vectordb_destructor(napi_env env, void* data, void* hint) {
    (void)env; (void)hint;
    handle_box_t* box = (handle_box_t*)data;
    if (box) { if (box->ptr) infer_vectordb_free((InferVectorDB)box->ptr); free(box); }
}

/* vectordbCreate(dim, M, efConstruction) → external */
static napi_value napi_vectordb_create(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value argv[3];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));

    int32_t dim, M, ef;
    napi_get_value_int32(env, argv[0], &dim);
    napi_get_value_int32(env, argv[1], &M);
    napi_get_value_int32(env, argv[2], &ef);

    InferVectorDB db = infer_vectordb_create(dim, M, ef);
    if (!db) return throw_infer_error(env);

    napi_value ext;
    handle_box_t* box = alloc_box(db);
    NAPI_CALL(env, napi_create_external(env, box, _vectordb_destructor, NULL, &ext));
    return ext;
}

/* vectordbInsert(handle, id, vec, metadata) → void */
static napi_value napi_vectordb_insert(napi_env env, napi_callback_info info) {
    size_t argc = 4;
    napi_value argv[4];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));

    InferVectorDB db = (InferVectorDB)unwrap_handle(env, argv[0]);

    int64_t id;
    napi_get_value_int64(env, argv[1], &id);

    /* vec: number[] */
    uint32_t vlen;
    napi_get_array_length(env, argv[2], &vlen);
    float* vec = (float*)malloc(sizeof(float) * vlen);
    for (uint32_t i = 0; i < vlen; i++) {
        napi_value el;
        double d;
        napi_get_element(env, argv[2], i, &el);
        napi_get_value_double(env, el, &d);
        vec[i] = (float)d;
    }

    /* metadata: string or null */
    char* meta = NULL;
    napi_valuetype vt;
    napi_typeof(env, argv[3], &vt);
    if (vt == napi_string) {
        meta = get_string_arg(env, argv[3]);
    }

    int rc = infer_vectordb_insert(db, id, vec, meta);
    free(vec);
    free(meta);
    if (rc != 0) return throw_infer_error(env);
    return NULL;
}

/* vectordbDelete(handle, id) → void */
static napi_value napi_vectordb_delete(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value argv[2];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));

    InferVectorDB db = (InferVectorDB)unwrap_handle(env, argv[0]);
    int64_t id;
    napi_get_value_int64(env, argv[1], &id);

    int rc = infer_vectordb_delete(db, id);
    if (rc != 0) return throw_infer_error(env);
    return NULL;
}

/* vectordbSearch(handle, query, k, efSearch, filter, maxResults) → { ids, distances } */
static napi_value napi_vectordb_search(napi_env env, napi_callback_info info) {
    size_t argc = 6;
    napi_value argv[6];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));

    InferVectorDB db = (InferVectorDB)unwrap_handle(env, argv[0]);

    /* query vec */
    uint32_t qlen;
    napi_get_array_length(env, argv[1], &qlen);
    float* query = (float*)malloc(sizeof(float) * qlen);
    for (uint32_t i = 0; i < qlen; i++) {
        napi_value el;
        double d;
        napi_get_element(env, argv[1], i, &el);
        napi_get_value_double(env, el, &d);
        query[i] = (float)d;
    }

    int32_t k, ef_search, max_results;
    napi_get_value_int32(env, argv[2], &k);
    napi_get_value_int32(env, argv[3], &ef_search);

    /* filter: string or null */
    char* filter = NULL;
    napi_valuetype vt;
    napi_typeof(env, argv[4], &vt);
    if (vt == napi_string) {
        filter = get_string_arg(env, argv[4]);
    }

    napi_get_value_int32(env, argv[5], &max_results);

    int64_t* out_ids = (int64_t*)malloc(sizeof(int64_t) * max_results);
    float*   out_dist = (float*)malloc(sizeof(float) * max_results);

    int n = infer_vectordb_search(db, query, k, ef_search, filter,
                                   out_ids, out_dist, max_results);
    free(query);
    free(filter);

    if (n < 0) {
        free(out_ids);
        free(out_dist);
        return throw_infer_error(env);
    }

    napi_value result, ids_arr, dist_arr;
    NAPI_CALL(env, napi_create_object(env, &result));
    NAPI_CALL(env, napi_create_array_with_length(env, (size_t)n, &ids_arr));
    NAPI_CALL(env, napi_create_array_with_length(env, (size_t)n, &dist_arr));

    for (int i = 0; i < n; i++) {
        napi_value id_val, d_val;
        napi_create_int64(env, out_ids[i], &id_val);
        napi_create_double(env, (double)out_dist[i], &d_val);
        napi_set_element(env, ids_arr, (uint32_t)i, id_val);
        napi_set_element(env, dist_arr, (uint32_t)i, d_val);
    }
    napi_set_named_property(env, result, "ids", ids_arr);
    napi_set_named_property(env, result, "distances", dist_arr);

    free(out_ids);
    free(out_dist);
    return result;
}

/* vectordbSize(handle) → number */
static napi_value napi_vectordb_size(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
    InferVectorDB db = (InferVectorDB)unwrap_handle(env, argv[0]);
    napi_value result;
    NAPI_CALL(env, napi_create_int32(env, infer_vectordb_size(db), &result));
    return result;
}

/* vectordbSave(handle, path) → void */
static napi_value napi_vectordb_save(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value argv[2];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
    InferVectorDB db = (InferVectorDB)unwrap_handle(env, argv[0]);
    char* path = get_string_arg(env, argv[1]);
    int rc = infer_vectordb_save(db, path);
    free(path);
    if (rc != 0) return throw_infer_error(env);
    return NULL;
}

/* vectordbLoad(handle, path) → void */
static napi_value napi_vectordb_load(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value argv[2];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
    InferVectorDB db = (InferVectorDB)unwrap_handle(env, argv[0]);
    char* path = get_string_arg(env, argv[1]);
    int rc = infer_vectordb_load(db, path);
    free(path);
    if (rc != 0) return throw_infer_error(env);
    return NULL;
}

/* vectordbFree(handle) */
static napi_value napi_vectordb_free(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
    void* ptr = unwrap_handle(env, argv[0]);
    if (ptr) infer_vectordb_free((InferVectorDB)ptr);
    clear_handle(env, argv[0]);
    return NULL;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * BM25 bindings
 * ───────────────────────────────────────────────────────────────────────────── */

static void _bm25_destructor(napi_env env, void* data, void* hint) {
    (void)env; (void)hint;
    handle_box_t* box = (handle_box_t*)data;
    if (box) { if (box->ptr) infer_bm25_free((InferBM25)box->ptr); free(box); }
}

/* bm25Create(k1, b) → external */
static napi_value napi_bm25_create(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value argv[2];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));

    double k1, b;
    napi_get_value_double(env, argv[0], &k1);
    napi_get_value_double(env, argv[1], &b);

    InferBM25 idx = infer_bm25_create((float)k1, (float)b);
    if (!idx) return throw_infer_error(env);

    napi_value ext;
    handle_box_t* box = alloc_box(idx);
    NAPI_CALL(env, napi_create_external(env, box, _bm25_destructor, NULL, &ext));
    return ext;
}

/* bm25Insert(handle, id, text) → void */
static napi_value napi_bm25_insert(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value argv[3];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));

    InferBM25 idx = (InferBM25)unwrap_handle(env, argv[0]);
    int64_t id;
    napi_get_value_int64(env, argv[1], &id);
    char* text = get_string_arg(env, argv[2]);

    infer_bm25_insert(idx, id, text);
    free(text);
    return NULL;
}

/* bm25Remove(handle, id) → void */
static napi_value napi_bm25_remove(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value argv[2];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));

    InferBM25 idx = (InferBM25)unwrap_handle(env, argv[0]);
    int64_t id;
    napi_get_value_int64(env, argv[1], &id);

    infer_bm25_remove(idx, id);
    return NULL;
}

/* bm25Search(handle, query, k, maxResults) → { ids, scores } */
static napi_value napi_bm25_search(napi_env env, napi_callback_info info) {
    size_t argc = 4;
    napi_value argv[4];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));

    InferBM25 idx = (InferBM25)unwrap_handle(env, argv[0]);
    char* query = get_string_arg(env, argv[1]);
    int32_t k, max_results;
    napi_get_value_int32(env, argv[2], &k);
    napi_get_value_int32(env, argv[3], &max_results);

    int64_t* out_ids = (int64_t*)malloc(sizeof(int64_t) * max_results);
    float*   out_scores = (float*)malloc(sizeof(float) * max_results);

    int n = infer_bm25_search(idx, query, k, out_ids, out_scores, max_results);
    free(query);

    if (n < 0) {
        free(out_ids);
        free(out_scores);
        return throw_infer_error(env);
    }

    napi_value result, ids_arr, scores_arr;
    NAPI_CALL(env, napi_create_object(env, &result));
    NAPI_CALL(env, napi_create_array_with_length(env, (size_t)n, &ids_arr));
    NAPI_CALL(env, napi_create_array_with_length(env, (size_t)n, &scores_arr));

    for (int i = 0; i < n; i++) {
        napi_value id_val, s_val;
        napi_create_int64(env, out_ids[i], &id_val);
        napi_create_double(env, (double)out_scores[i], &s_val);
        napi_set_element(env, ids_arr, (uint32_t)i, id_val);
        napi_set_element(env, scores_arr, (uint32_t)i, s_val);
    }
    napi_set_named_property(env, result, "ids", ids_arr);
    napi_set_named_property(env, result, "scores", scores_arr);

    free(out_ids);
    free(out_scores);
    return result;
}

/* bm25Size(handle) → number */
static napi_value napi_bm25_size(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
    InferBM25 idx = (InferBM25)unwrap_handle(env, argv[0]);
    napi_value result;
    NAPI_CALL(env, napi_create_int32(env, infer_bm25_size(idx), &result));
    return result;
}

/* bm25Save(handle, path) → void */
static napi_value napi_bm25_save(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value argv[2];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
    InferBM25 idx = (InferBM25)unwrap_handle(env, argv[0]);
    char* path = get_string_arg(env, argv[1]);
    int rc = infer_bm25_save(idx, path);
    free(path);
    if (rc != 0) return throw_infer_error(env);
    return NULL;
}

/* bm25Load(handle, path) → void */
static napi_value napi_bm25_load(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value argv[2];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
    InferBM25 idx = (InferBM25)unwrap_handle(env, argv[0]);
    char* path = get_string_arg(env, argv[1]);
    int rc = infer_bm25_load(idx, path);
    free(path);
    if (rc != 0) return throw_infer_error(env);
    return NULL;
}

/* bm25Free(handle) */
static napi_value napi_bm25_free(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
    void* ptr = unwrap_handle(env, argv[0]);
    if (ptr) infer_bm25_free((InferBM25)ptr);
    clear_handle(env, argv[0]);
    return NULL;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * lastErrorString()
 * ───────────────────────────────────────────────────────────────────────────── */

static napi_value napi_last_error_string(napi_env env, napi_callback_info info) {
    (void)info;
    const char* msg = infer_last_error_string();
    napi_value result;
    if (msg) {
        NAPI_CALL(env, napi_create_string_utf8(env, msg, NAPI_AUTO_LENGTH, &result));
    } else {
        NAPI_CALL(env, napi_get_null(env, &result));
    }
    return result;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Module init — register all functions
 * ───────────────────────────────────────────────────────────────────────────── */

#define EXPORT_FN(name, fn)                                       \
    do {                                                          \
        napi_value _fn;                                           \
        napi_create_function(env, name, NAPI_AUTO_LENGTH, fn, NULL, &_fn); \
        napi_set_named_property(env, exports, name, _fn);         \
    } while (0)

static napi_value Init(napi_env env, napi_value exports) {
    /* LLM */
    EXPORT_FN("llmCreate",        napi_llm_create);
    EXPORT_FN("llmDestroy",       napi_llm_destroy);
    EXPORT_FN("llmTokenize",      napi_llm_tokenize);
    EXPORT_FN("llmGenerate",      napi_llm_generate);
    EXPORT_FN("llmVocabSize",     napi_llm_vocab_size);
    EXPORT_FN("llmBos",           napi_llm_bos);
    EXPORT_FN("llmEos",           napi_llm_eos);
    EXPORT_FN("llmTokenToPiece",  napi_llm_token_to_piece);

    /* Session (ONNX) */
    EXPORT_FN("sessionCreate",    napi_session_create);
    EXPORT_FN("sessionLoad",      napi_session_load);
    EXPORT_FN("sessionDestroy",   napi_session_destroy);

    /* Tokenizer */
    EXPORT_FN("tokenizerLoad",      napi_tokenizer_load);
    EXPORT_FN("tokenizerEncode",    napi_tokenizer_encode);
    EXPORT_FN("tokenizerDecode",    napi_tokenizer_decode);
    EXPORT_FN("tokenizerVocabSize", napi_tokenizer_vocab_size);
    EXPORT_FN("tokenizerDestroy",   napi_tokenizer_destroy);

    /* Embedding */
    EXPORT_FN("embedPipeline",      napi_embed_pipeline);
    EXPORT_FN("embedBatchPipeline", napi_embed_batch_pipeline);

    /* VectorDB */
    EXPORT_FN("vectordbCreate",  napi_vectordb_create);
    EXPORT_FN("vectordbInsert",  napi_vectordb_insert);
    EXPORT_FN("vectordbDelete",  napi_vectordb_delete);
    EXPORT_FN("vectordbSearch",  napi_vectordb_search);
    EXPORT_FN("vectordbSize",    napi_vectordb_size);
    EXPORT_FN("vectordbSave",    napi_vectordb_save);
    EXPORT_FN("vectordbLoad",    napi_vectordb_load);
    EXPORT_FN("vectordbFree",    napi_vectordb_free);

    /* BM25 */
    EXPORT_FN("bm25Create",  napi_bm25_create);
    EXPORT_FN("bm25Insert",  napi_bm25_insert);
    EXPORT_FN("bm25Remove",  napi_bm25_remove);
    EXPORT_FN("bm25Search",  napi_bm25_search);
    EXPORT_FN("bm25Size",    napi_bm25_size);
    EXPORT_FN("bm25Save",    napi_bm25_save);
    EXPORT_FN("bm25Load",    napi_bm25_load);
    EXPORT_FN("bm25Free",    napi_bm25_free);

    /* Utility */
    EXPORT_FN("lastErrorString", napi_last_error_string);

    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
