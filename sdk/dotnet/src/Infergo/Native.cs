using System.Runtime.InteropServices;

namespace Infergo;

/// <summary>
/// Raw P/Invoke declarations for libinfer_api.
/// All handles are IntPtr. Callers should use the managed wrappers instead.
/// </summary>
internal static class Native
{
    private const string Lib = "infer_api";

    // ── Error ────────────────────────────────────────────────────────────────

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr infer_last_error_string();

    /// <summary>
    /// Get the thread-local error string from the last failed C API call.
    /// </summary>
    internal static string? GetLastError()
    {
        IntPtr ptr = infer_last_error_string();
        if (ptr == IntPtr.Zero) return null;
        return Marshal.PtrToStringAnsi(ptr);
    }

    // ── LLM ──────────────────────────────────────────────────────────────────

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr infer_llm_create(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path,
        int n_gpu_layers,
        int ctx_size,
        int n_seq_max,
        int n_batch);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void infer_llm_destroy(IntPtr llm);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int infer_llm_generate(
        IntPtr llm,
        [MarshalAs(UnmanagedType.LPArray)] int[] prompt_tokens,
        int n_prompt,
        int max_tokens,
        float temperature,
        float top_p,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? grammar,
        IntPtr callback,
        IntPtr user_data,
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 10)] byte[] out_text,
        int max_text_len,
        out int out_gen_tokens);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int infer_llm_tokenize(
        IntPtr llm,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string text,
        int add_bos,
        [MarshalAs(UnmanagedType.LPArray)] int[] out_ids,
        int max_tokens);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int infer_llm_vocab_size(IntPtr llm);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int infer_llm_bos(IntPtr llm);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int infer_llm_eos(IntPtr llm);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int infer_llm_is_eog(IntPtr llm, int token);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int infer_llm_token_to_piece(
        IntPtr llm,
        int token,
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 3)] byte[] out_buf,
        int buf_size);

    // ── ONNX Session ─────────────────────────────────────────────────────────

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr infer_session_create(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string provider,
        int device_id);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int infer_session_load(
        IntPtr session,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string model_path);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void infer_session_destroy(IntPtr session);

    // ── Tokenizer ────────────────────────────────────────────────────────────

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr infer_tokenizer_load(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void infer_tokenizer_destroy(IntPtr tok);

    // ── Embedding Pipeline ───────────────────────────────────────────────────

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int infer_embed_pipeline(
        IntPtr session,
        IntPtr tokenizer,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string text,
        [MarshalAs(UnmanagedType.LPArray)] float[] out_vec,
        int max_dim);

    // ── VectorDB ─────────────────────────────────────────────────────────────

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr infer_vectordb_create(int dim, int M, int ef_construction);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int infer_vectordb_insert(
        IntPtr db,
        long id,
        [MarshalAs(UnmanagedType.LPArray)] float[] vec,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? metadata);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int infer_vectordb_delete(IntPtr db, long id);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int infer_vectordb_update(
        IntPtr db,
        long id,
        [MarshalAs(UnmanagedType.LPArray)] float[] vec,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? metadata);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int infer_vectordb_search(
        IntPtr db,
        [MarshalAs(UnmanagedType.LPArray)] float[] query,
        int k,
        int ef_search,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? metadata_filter,
        [MarshalAs(UnmanagedType.LPArray)] long[] out_ids,
        [MarshalAs(UnmanagedType.LPArray)] float[] out_distances,
        int max_results);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int infer_vectordb_save(
        IntPtr db,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int infer_vectordb_load(
        IntPtr db,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int infer_vectordb_size(IntPtr db);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void infer_vectordb_free(IntPtr db);

    // ── BM25 ─────────────────────────────────────────────────────────────────

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr infer_bm25_create(float k1, float b);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void infer_bm25_insert(
        IntPtr idx,
        long id,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string text);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void infer_bm25_remove(IntPtr idx, long id);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int infer_bm25_search(
        IntPtr idx,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string query,
        int k,
        [MarshalAs(UnmanagedType.LPArray)] long[] out_ids,
        [MarshalAs(UnmanagedType.LPArray)] float[] out_scores,
        int max_results);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int infer_bm25_save(
        IntPtr idx,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int infer_bm25_load(
        IntPtr idx,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int infer_bm25_size(IntPtr idx);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void infer_bm25_free(IntPtr idx);
}
