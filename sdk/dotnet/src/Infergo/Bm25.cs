namespace Infergo;

/// <summary>
/// Search result from the BM25 full-text index.
/// </summary>
public readonly record struct Bm25SearchResult(long Id, float Score);

/// <summary>
/// Managed wrapper for the infergo BM25 full-text search index.
/// </summary>
public sealed class Bm25 : IDisposable
{
    private IntPtr _handle;
    private bool _disposed;

    /// <summary>
    /// Create a new BM25 full-text search index.
    /// </summary>
    /// <param name="k1">Term frequency saturation parameter (default 1.2).</param>
    /// <param name="b">Document length normalization parameter (default 0.75).</param>
    public Bm25(float k1 = 1.2f, float b = 0.75f)
    {
        _handle = Native.infer_bm25_create(k1, b);
        if (_handle == IntPtr.Zero)
            throw InfergoException.FromNative("Failed to create BM25 index");
    }

    /// <summary>
    /// Number of documents in the index.
    /// </summary>
    public int Count
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return Native.infer_bm25_size(_handle);
        }
    }

    /// <summary>
    /// Insert a document into the index.
    /// </summary>
    /// <param name="id">Unique document ID.</param>
    /// <param name="text">Document text.</param>
    public void Insert(long id, string text)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(text);

        Native.infer_bm25_insert(_handle, id, text);
    }

    /// <summary>
    /// Remove a document from the index.
    /// </summary>
    public void Remove(long id)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Native.infer_bm25_remove(_handle, id);
    }

    /// <summary>
    /// Search the index for documents matching a query.
    /// </summary>
    /// <param name="query">Search query string.</param>
    /// <param name="k">Maximum number of results to return.</param>
    /// <returns>Array of results sorted by BM25 score (descending).</returns>
    public Bm25SearchResult[] Search(string query, int k = 10)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(query);
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));

        long[] ids = new long[k];
        float[] scores = new float[k];

        int found = Native.infer_bm25_search(_handle, query, k, ids, scores, k);
        if (found < 0)
            throw InfergoException.FromNative("BM25 search failed");

        var results = new Bm25SearchResult[found];
        for (int i = 0; i < found; i++)
            results[i] = new Bm25SearchResult(ids[i], scores[i]);
        return results;
    }

    /// <summary>
    /// Save the index to a file.
    /// </summary>
    public void Save(string path)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(path);

        int rc = Native.infer_bm25_save(_handle, path);
        if (rc != 0)
            throw InfergoException.FromNative($"BM25 save failed: {path}", rc);
    }

    /// <summary>
    /// Load the index from a file.
    /// </summary>
    public void Load(string path)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(path);

        int rc = Native.infer_bm25_load(_handle, path);
        if (rc != 0)
            throw InfergoException.FromNative($"BM25 load failed: {path}", rc);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_handle != IntPtr.Zero)
        {
            Native.infer_bm25_free(_handle);
            _handle = IntPtr.Zero;
        }
    }
}
