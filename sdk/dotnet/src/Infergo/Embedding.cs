namespace Infergo;

/// <summary>
/// Managed wrapper for the infergo embedding pipeline (ONNX session + tokenizer).
/// </summary>
public sealed class Embedding : IDisposable
{
    private IntPtr _session;
    private IntPtr _tokenizer;
    private bool _disposed;
    private readonly int _dim;

    /// <summary>
    /// Create an embedding pipeline.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX embedding model.</param>
    /// <param name="tokenizerPath">Path to the HuggingFace tokenizer.json file.</param>
    /// <param name="dim">Embedding dimension (e.g. 384, 768, 1024).</param>
    /// <param name="provider">Execution provider: "cpu", "cuda", "tensorrt", "coreml".</param>
    /// <param name="deviceId">GPU device index (0 for first GPU).</param>
    public Embedding(string modelPath, string tokenizerPath, int dim, string provider = "cpu", int deviceId = 0)
    {
        ArgumentNullException.ThrowIfNull(modelPath);
        ArgumentNullException.ThrowIfNull(tokenizerPath);
        if (dim <= 0)
            throw new ArgumentOutOfRangeException(nameof(dim), "Dimension must be positive.");

        _dim = dim;

        _session = Native.infer_session_create(provider, deviceId);
        if (_session == IntPtr.Zero)
            throw InfergoException.FromNative($"Failed to create session with provider '{provider}'");

        int rc = Native.infer_session_load(_session, modelPath);
        if (rc != 0)
        {
            Native.infer_session_destroy(_session);
            _session = IntPtr.Zero;
            throw InfergoException.FromNative($"Failed to load embedding model: {modelPath}", rc);
        }

        _tokenizer = Native.infer_tokenizer_load(tokenizerPath);
        if (_tokenizer == IntPtr.Zero)
        {
            Native.infer_session_destroy(_session);
            _session = IntPtr.Zero;
            throw InfergoException.FromNative($"Failed to load tokenizer: {tokenizerPath}");
        }
    }

    /// <summary>
    /// Embedding dimension.
    /// </summary>
    public int Dimension => _dim;

    /// <summary>
    /// Compute the embedding vector for a text string.
    /// The full pipeline (tokenize, run ONNX, pool, normalize) executes in a single native call.
    /// </summary>
    /// <param name="text">Input text.</param>
    /// <returns>Normalized embedding vector of length <see cref="Dimension"/>.</returns>
    public float[] Embed(string text)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(text);

        float[] vec = new float[_dim];
        int written = Native.infer_embed_pipeline(_session, _tokenizer, text, vec, _dim);
        if (written < 0)
            throw InfergoException.FromNative("Embedding pipeline failed");

        if (written != _dim)
        {
            // Truncate or expand to match requested dimension.
            float[] result = new float[written];
            Array.Copy(vec, result, written);
            return result;
        }

        return vec;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_tokenizer != IntPtr.Zero)
        {
            Native.infer_tokenizer_destroy(_tokenizer);
            _tokenizer = IntPtr.Zero;
        }

        if (_session != IntPtr.Zero)
        {
            Native.infer_session_destroy(_session);
            _session = IntPtr.Zero;
        }
    }
}
