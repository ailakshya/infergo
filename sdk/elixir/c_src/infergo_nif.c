// infergo Elixir NIF — wraps libinfer_api.so
#include <erl_nif.h>
#include "infer_api.h"
#include <string.h>

// Resource types for handle pointers
static ErlNifResourceType* LLM_RES;
static ErlNifResourceType* VDB_RES;
static ErlNifResourceType* BM25_RES;

typedef struct { InferLLM handle; } llm_res_t;
typedef struct { InferVectorDB handle; } vdb_res_t;
typedef struct { InferBM25 handle; } bm25_res_t;

static void llm_dtor(ErlNifEnv* env, void* obj) { llm_res_t* r = obj; if (r->handle) infer_llm_destroy(r->handle); }
static void vdb_dtor(ErlNifEnv* env, void* obj) { vdb_res_t* r = obj; if (r->handle) infer_vectordb_free(r->handle); }
static void bm25_dtor(ErlNifEnv* env, void* obj) { bm25_res_t* r = obj; if (r->handle) infer_bm25_free(r->handle); }

static int load(ErlNifEnv* env, void** priv, ERL_NIF_TERM info) {
    LLM_RES = enif_open_resource_type(env, NULL, "llm", llm_dtor, ERL_NIF_RT_CREATE, NULL);
    VDB_RES = enif_open_resource_type(env, NULL, "vdb", vdb_dtor, ERL_NIF_RT_CREATE, NULL);
    BM25_RES = enif_open_resource_type(env, NULL, "bm25", bm25_dtor, ERL_NIF_RT_CREATE, NULL);
    return 0;
}

static ERL_NIF_TERM nif_llm_create(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    char path[1024];
    int gpu, ctx, seq, batch;
    if (!enif_get_string(env, argv[0], path, sizeof(path), ERL_NIF_LATIN1)) return enif_make_badarg(env);
    enif_get_int(env, argv[1], &gpu);
    enif_get_int(env, argv[2], &ctx);
    enif_get_int(env, argv[3], &seq);
    enif_get_int(env, argv[4], &batch);

    llm_res_t* res = enif_alloc_resource(LLM_RES, sizeof(llm_res_t));
    res->handle = infer_llm_create(path, gpu, ctx, seq, batch);
    if (!res->handle) {
        enif_release_resource(res);
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_string(env, infer_last_error_string(), ERL_NIF_LATIN1));
    }
    ERL_NIF_TERM term = enif_make_resource(env, res);
    enif_release_resource(res);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), term);
}

static ERL_NIF_TERM nif_llm_generate(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    llm_res_t* res;
    char prompt[4096];
    int max_tokens;
    double temp;
    if (!enif_get_resource(env, argv[0], LLM_RES, (void**)&res)) return enif_make_badarg(env);
    if (!enif_get_string(env, argv[1], prompt, sizeof(prompt), ERL_NIF_LATIN1)) return enif_make_badarg(env);
    enif_get_int(env, argv[2], &max_tokens);
    enif_get_double(env, argv[3], &temp);

    int tokens[4096];
    int n = infer_llm_tokenize(res->handle, prompt, 1, tokens, 4096);
    if (n < 0) return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "tokenize"));

    char out[32768];
    int gen = 0;
    int rc = infer_llm_generate(res->handle, tokens, n, max_tokens, (float)temp, 0.9f, NULL, NULL, NULL, out, sizeof(out), &gen);
    if (rc < 0) return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "generate"));

    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_string(env, out, ERL_NIF_LATIN1));
}

static ErlNifFunc nif_funcs[] = {
    {"llm_create_nif", 5, nif_llm_create},
    {"llm_generate_nif", 4, nif_llm_generate},
};

ERL_NIF_INIT(Elixir.Infergo, nif_funcs, load, NULL, NULL, NULL)
