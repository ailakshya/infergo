//! infergo Zig SDK — C interop bindings to libinfer_api
const std = @import("std");

const c = @cImport({
    @cInclude("infer_api.h");
});

pub const InferError = error{
    LoadFailed,
    TokenizeFailed,
    GenerateFailed,
    EmbedFailed,
    InsertFailed,
    SearchFailed,
    NullHandle,
};

fn lastError() []const u8 {
    const msg = c.infer_last_error_string();
    if (msg == null) return "unknown error";
    return std.mem.span(msg);
}

pub const LLM = struct {
    handle: *anyopaque,

    pub fn init(path: [:0]const u8, gpu_layers: i32, ctx_size: i32, n_seq_max: i32, n_batch: i32) !LLM {
        const h = c.infer_llm_create(path.ptr, gpu_layers, ctx_size, n_seq_max, n_batch);
        if (h == null) return InferError.LoadFailed;
        return LLM{ .handle = h.? };
    }

    pub fn deinit(self: *LLM) void {
        c.infer_llm_destroy(self.handle);
    }

    pub fn vocabSize(self: *const LLM) i32 {
        return c.infer_llm_vocab_size(self.handle);
    }

    pub fn tokenize(self: *const LLM, text: [:0]const u8, add_bos: bool, out: []i32) !usize {
        const n = c.infer_llm_tokenize(self.handle, text.ptr, if (add_bos) 1 else 0, out.ptr, @intCast(out.len));
        if (n < 0) return InferError.TokenizeFailed;
        return @intCast(n);
    }

    pub fn generate(self: *const LLM, tokens: []const i32, max_tokens: i32, temp: f32, top_p: f32, out: []u8) ![]u8 {
        var gen: i32 = 0;
        const rc = c.infer_llm_generate(self.handle, tokens.ptr, @intCast(tokens.len), max_tokens, temp, top_p, null, null, null, out.ptr, @intCast(out.len), &gen);
        if (rc < 0) return InferError.GenerateFailed;
        const len = std.mem.indexOfScalar(u8, out, 0) orelse out.len;
        return out[0..len];
    }
};

pub const Embedding = struct {
    session: *anyopaque,
    tokenizer: *anyopaque,

    pub fn init(model_path: [:0]const u8, tok_path: [:0]const u8, provider: [:0]const u8, device: i32) !Embedding {
        const s = c.infer_session_create(provider.ptr, device) orelse return InferError.LoadFailed;
        if (c.infer_session_load(s, model_path.ptr) != 0) return InferError.LoadFailed;
        const t = c.infer_tokenizer_load(tok_path.ptr) orelse return InferError.LoadFailed;
        return Embedding{ .session = s, .tokenizer = t };
    }

    pub fn deinit(self: *Embedding) void {
        c.infer_tokenizer_destroy(self.tokenizer);
        c.infer_session_destroy(self.session);
    }

    pub fn embed(self: *const Embedding, text: [:0]const u8, out: []f32) !usize {
        const dim = c.infer_embed_pipeline(self.session, self.tokenizer, text.ptr, out.ptr, @intCast(out.len));
        if (dim < 0) return InferError.EmbedFailed;
        return @intCast(dim);
    }
};

pub const VectorDB = struct {
    handle: *anyopaque,

    pub fn init(dim: i32, M: i32, ef: i32) !VectorDB {
        const h = c.infer_vectordb_create(dim, M, ef) orelse return InferError.NullHandle;
        return VectorDB{ .handle = h };
    }

    pub fn deinit(self: *VectorDB) void {
        c.infer_vectordb_free(self.handle);
    }

    pub fn insert(self: *const VectorDB, id: i64, vec: []const f32, meta: [:0]const u8) !void {
        if (c.infer_vectordb_insert(self.handle, id, vec.ptr, meta.ptr) < 0) return InferError.InsertFailed;
    }

    pub fn search(self: *const VectorDB, query: []const f32, k: i32, ef: i32, out_ids: []i64, out_dists: []f32) !usize {
        const n = c.infer_vectordb_search(self.handle, query.ptr, k, ef, null, out_ids.ptr, out_dists.ptr, @intCast(out_ids.len));
        if (n < 0) return InferError.SearchFailed;
        return @intCast(n);
    }
};

pub const BM25 = struct {
    handle: *anyopaque,

    pub fn init(k1: f32, b: f32) !BM25 {
        const h = c.infer_bm25_create(k1, b) orelse return InferError.NullHandle;
        return BM25{ .handle = h };
    }

    pub fn deinit(self: *BM25) void {
        c.infer_bm25_free(self.handle);
    }

    pub fn insert(self: *const BM25, id: i64, text: [:0]const u8) void {
        c.infer_bm25_insert(self.handle, id, text.ptr);
    }

    pub fn search(self: *const BM25, query: [:0]const u8, k: i32, out_ids: []i64, out_scores: []f32) !usize {
        const n = c.infer_bm25_search(self.handle, query.ptr, k, out_ids.ptr, out_scores.ptr, @intCast(out_ids.len));
        if (n < 0) return InferError.SearchFailed;
        return @intCast(n);
    }
};
