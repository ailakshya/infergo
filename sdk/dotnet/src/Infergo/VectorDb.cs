namespace Infergo;

/// <summary>
/// Search result from the vector database.
/// </summary>
public readonly record struct VectorSearchResult(long Id, float Distance);

/// <summary>
/// Managed wrapper for the infergo HNSW vector database.
/// </summary>
public sealed class VectorDb : IDisposable
{
    private IntPtr _handle;
    private bool _disposed;
    private readonly int _dim;

    /// <summary>
    /// Create a new in-memory vector database.
    /// </summary>
    /// <param name="dim">Vector dimension (must match embedding model output).</param>
    /// <param name="m">HNSW M parameter — max connections per node (16 is a good default).</param>
    /// <param name="efConstruction">HNSW ef_construction — search width during build (200 is a good default).</param>
    public VectorDb(int dim, int m = 16, int efConstruction = 200)
    {
        if (dim <= 0) throw new ArgumentOutOfRangeException(nameof(dim));

        _dim = dim;
        _handle = Native.infer_vectordb_create(dim, m, efConstruction);
        if (_handle == IntPtr.Zero)
            throw InfergoException.FromNative("Failed to create vector database");
    }

    /// <summary>
    /// Vector dimension.
    /// </summary>
    public int Dimension => _dim;

    /// <summary>
    /// Number of vectors in the database.
    /// </summary>
    public int Count
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return Native.infer_vectordb_size(_handle);
        }
    }

    /// <summary>
    /// Insert a vector with an ID and optional metadata.
    /// </summary>
    /// <param name="id">Unique document ID.</param>
    /// <param name="vector">Float array of length <see cref="Dimension"/>.</param>
    /// <param name="metadata">Optional JSON metadata string.</param>
    public void Insert(long id, float[] vector, string? metadata = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(vector);
        if (vector.Length != _dim)
            throw new ArgumentException($"Vector length {vector.Length} does not match dimension {_dim}.", nameof(vector));

        int rc = Native.infer_vectordb_insert(_handle, id, vector, metadata);
        if (rc != 0)
            throw InfergoException.FromNative($"VectorDB insert failed for id={id}", rc);
    }

    /// <summary>
    /// Delete a vector by ID.
    /// </summary>
    public void Delete(long id)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        int rc = Native.infer_vectordb_delete(_handle, id);
        if (rc != 0)
            throw InfergoException.FromNative($"VectorDB delete failed for id={id}", rc);
    }

    /// <summary>
    /// Update a vector and its metadata.
    /// </summary>
    public void Update(long id, float[] vector, string? metadata = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(vector);
        if (vector.Length != _dim)
            throw new ArgumentException($"Vector length {vector.Length} does not match dimension {_dim}.", nameof(vector));

        int rc = Native.infer_vectordb_update(_handle, id, vector, metadata);
        if (rc != 0)
            throw InfergoException.FromNative($"VectorDB update failed for id={id}", rc);
    }

    /// <summary>
    /// Search for the k nearest neighbors to a query vector.
    /// </summary>
    /// <param name="query">Query vector of length <see cref="Dimension"/>.</param>
    /// <param name="k">Number of results to return.</param>
    /// <param name="efSearch">Search width (higher = more accurate but slower; 50-200 typical).</param>
    /// <param name="metadataFilter">Optional metadata filter expression.</param>
    /// <returns>Array of search results sorted by distance (ascending).</returns>
    public VectorSearchResult[] Search(float[] query, int k = 10, int efSearch = 100, string? metadataFilter = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(query);
        if (query.Length != _dim)
            throw new ArgumentException($"Query length {query.Length} does not match dimension {_dim}.", nameof(query));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));

        long[] ids = new long[k];
        float[] distances = new float[k];

        int found = Native.infer_vectordb_search(_handle, query, k, efSearch, metadataFilter, ids, distances, k);
        if (found < 0)
            throw InfergoException.FromNative("VectorDB search failed");

        var results = new VectorSearchResult[found];
        for (int i = 0; i < found; i++)
            results[i] = new VectorSearchResult(ids[i], distances[i]);
        return results;
    }

    /// <summary>
    /// Save the database to a file.
    /// </summary>
    public void Save(string path)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(path);

        int rc = Native.infer_vectordb_save(_handle, path);
        if (rc != 0)
            throw InfergoException.FromNative($"VectorDB save failed: {path}", rc);
    }

    /// <summary>
    /// Load the database from a file.
    /// </summary>
    public void Load(string path)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(path);

        int rc = Native.infer_vectordb_load(_handle, path);
        if (rc != 0)
            throw InfergoException.FromNative($"VectorDB load failed: {path}", rc);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_handle != IntPtr.Zero)
        {
            Native.infer_vectordb_free(_handle);
            _handle = IntPtr.Zero;
        }
    }
}
