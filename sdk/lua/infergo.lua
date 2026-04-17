-- infergo Lua SDK — LuaJIT FFI bindings to libinfer_api.so
local ffi = require("ffi")

ffi.cdef[[
    typedef void* InferLLM;
    typedef void* InferSession;
    typedef void* InferTokenizer;
    typedef void* InferVectorDB;
    typedef void* InferBM25;

    const char* infer_last_error_string(void);

    InferLLM infer_llm_create(const char* path, int gpu_layers, int ctx, int seq_max, int batch);
    void infer_llm_destroy(InferLLM llm);
    int infer_llm_vocab_size(InferLLM llm);
    int infer_llm_tokenize(InferLLM llm, const char* text, int bos, int* out, int max);
    int infer_llm_generate(InferLLM llm, const int* tokens, int n, int max, float temp, float top_p,
                           const char* grammar, void* cb, void* ud, char* buf, int len, int* gen);

    InferSession infer_session_create(const char* provider, int device);
    int infer_session_load(InferSession s, const char* path);
    void infer_session_destroy(InferSession s);
    InferTokenizer infer_tokenizer_load(const char* path);
    void infer_tokenizer_destroy(InferTokenizer tok);
    int infer_embed_pipeline(InferSession s, InferTokenizer tok, const char* text, float* out, int dim);

    InferVectorDB infer_vectordb_create(int dim, int M, int ef);
    int infer_vectordb_insert(InferVectorDB db, int64_t id, const float* vec, const char* meta);
    int infer_vectordb_search(InferVectorDB db, const float* q, int k, int ef, const char* filt,
                              int64_t* ids, float* dists, int max);
    int infer_vectordb_size(InferVectorDB db);
    void infer_vectordb_free(InferVectorDB db);

    InferBM25 infer_bm25_create(float k1, float b);
    void infer_bm25_insert(InferBM25 idx, int64_t id, const char* text);
    int infer_bm25_search(InferBM25 idx, const char* q, int k, int64_t* ids, float* scores, int max);
    int infer_bm25_size(InferBM25 idx);
    void infer_bm25_free(InferBM25 idx);
]]

local lib_path = os.getenv("INFERGO_LIB_DIR") and (os.getenv("INFERGO_LIB_DIR") .. "/libinfer_api.so") or "libinfer_api.so"
local C = ffi.load(lib_path)

local infergo = {}

-- LLM
local LLM = {}
LLM.__index = LLM

function infergo.LLM(path, gpu_layers, ctx_size, n_seq_max, n_batch)
    local handle = C.infer_llm_create(path, gpu_layers or -1, ctx_size or 4096, n_seq_max or 1, n_batch or 2048)
    if handle == nil then error("Failed to load: " .. ffi.string(C.infer_last_error_string())) end
    local self = setmetatable({ _handle = handle }, LLM)
    ffi.gc(handle, C.infer_llm_destroy)
    return self
end

function LLM:vocab_size()
    return C.infer_llm_vocab_size(self._handle)
end

function LLM:tokenize(text, add_bos)
    local out = ffi.new("int[4096]")
    local n = C.infer_llm_tokenize(self._handle, text, add_bos and 1 or 0, out, 4096)
    if n < 0 then error("Tokenize failed") end
    local tokens = {}
    for i = 0, n-1 do tokens[i+1] = out[i] end
    return tokens, n
end

function LLM:generate(prompt, max_tokens, temperature, top_p)
    local tokens, n = self:tokenize(prompt, true)
    local tok_arr = ffi.new("int[?]", n)
    for i = 0, n-1 do tok_arr[i] = tokens[i+1] end

    local buf = ffi.new("char[32768]")
    local gen = ffi.new("int[1]")
    local rc = C.infer_llm_generate(self._handle, tok_arr, n, max_tokens or 128,
                                     temperature or 0.7, top_p or 0.9, nil, nil, nil, buf, 32768, gen)
    if rc < 0 then error("Generate failed: " .. ffi.string(C.infer_last_error_string())) end
    return ffi.string(buf)
end

function LLM:close()
    if self._handle ~= nil then
        ffi.gc(self._handle, nil)
        C.infer_llm_destroy(self._handle)
        self._handle = nil
    end
end

-- Embedding
local Embedding = {}
Embedding.__index = Embedding

function infergo.Embedding(model_path, tok_path, provider, device)
    local session = C.infer_session_create(provider or "cpu", device or 0)
    C.infer_session_load(session, model_path)
    local tokenizer = C.infer_tokenizer_load(tok_path)
    local self = setmetatable({ _session = session, _tokenizer = tokenizer }, Embedding)
    return self
end

function Embedding:embed(text)
    local out = ffi.new("float[2048]")
    local dim = C.infer_embed_pipeline(self._session, self._tokenizer, text, out, 2048)
    if dim < 0 then error("Embed failed") end
    local vec = {}
    for i = 0, dim-1 do vec[i+1] = out[i] end
    return vec, dim
end

function Embedding:close()
    if self._tokenizer then C.infer_tokenizer_destroy(self._tokenizer); self._tokenizer = nil end
    if self._session then C.infer_session_destroy(self._session); self._session = nil end
end

-- VectorDB
local VectorDB = {}
VectorDB.__index = VectorDB

function infergo.VectorDB(dim, M, ef)
    local handle = C.infer_vectordb_create(dim or 384, M or 16, ef or 200)
    return setmetatable({ _handle = handle, _dim = dim or 384 }, VectorDB)
end

function VectorDB:insert(id, vec, meta)
    local v = ffi.new("float[?]", #vec)
    for i, val in ipairs(vec) do v[i-1] = val end
    C.infer_vectordb_insert(self._handle, id, v, meta or "")
end

function VectorDB:search(query, k, ef)
    k = k or 10; ef = ef or 50
    local q = ffi.new("float[?]", #query)
    for i, val in ipairs(query) do q[i-1] = val end
    local ids = ffi.new("int64_t[?]", k)
    local dists = ffi.new("float[?]", k)
    local n = C.infer_vectordb_search(self._handle, q, k, ef, nil, ids, dists, k)
    local results = {}
    for i = 0, n-1 do results[i+1] = { id = tonumber(ids[i]), distance = dists[i] } end
    return results
end

function VectorDB:size() return C.infer_vectordb_size(self._handle) end
function VectorDB:close() if self._handle then C.infer_vectordb_free(self._handle); self._handle = nil end end

-- BM25
local BM25 = {}
BM25.__index = BM25

function infergo.BM25(k1, b)
    local handle = C.infer_bm25_create(k1 or 1.2, b or 0.75)
    return setmetatable({ _handle = handle }, BM25)
end

function BM25:insert(id, text) C.infer_bm25_insert(self._handle, id, text) end

function BM25:search(query, k)
    k = k or 10
    local ids = ffi.new("int64_t[?]", k)
    local scores = ffi.new("float[?]", k)
    local n = C.infer_bm25_search(self._handle, query, k, ids, scores, k)
    local results = {}
    for i = 0, n-1 do results[i+1] = { id = tonumber(ids[i]), score = scores[i] } end
    return results
end

function BM25:size() return C.infer_bm25_size(self._handle) end
function BM25:close() if self._handle then C.infer_bm25_free(self._handle); self._handle = nil end end

return infergo
