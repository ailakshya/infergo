using System.Runtime.InteropServices;
using System.Text;

namespace Infergo;

/// <summary>
/// Managed wrapper around the infergo LLM engine (GGUF models via llama.cpp).
/// </summary>
public sealed class Llm : IDisposable
{
    private IntPtr _handle;
    private bool _disposed;

    /// <summary>
    /// Load a GGUF model and create an LLM engine.
    /// </summary>
    /// <param name="modelPath">Path to the .gguf model file.</param>
    /// <param name="gpuLayers">Number of transformer layers to offload to GPU. Use a large value (e.g. 999) for all.</param>
    /// <param name="ctxSize">Total KV cache token budget across all sequences.</param>
    /// <param name="seqMax">Maximum number of concurrent sequences.</param>
    /// <param name="batchSize">Maximum tokens per decode call.</param>
    public Llm(string modelPath, int gpuLayers = 999, int ctxSize = 4096, int seqMax = 1, int batchSize = 512)
    {
        ArgumentNullException.ThrowIfNull(modelPath);

        _handle = Native.infer_llm_create(modelPath, gpuLayers, ctxSize, seqMax, batchSize);
        if (_handle == IntPtr.Zero)
            throw InfergoException.FromNative($"Failed to load LLM model: {modelPath}");
    }

    /// <summary>
    /// The native handle. Throws if disposed.
    /// </summary>
    internal IntPtr Handle
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _handle;
        }
    }

    /// <summary>
    /// Vocabulary size of the loaded model.
    /// </summary>
    public int VocabSize
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return Native.infer_llm_vocab_size(_handle);
        }
    }

    /// <summary>
    /// BOS token ID.
    /// </summary>
    public int BosToken
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return Native.infer_llm_bos(_handle);
        }
    }

    /// <summary>
    /// EOS token ID.
    /// </summary>
    public int EosToken
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return Native.infer_llm_eos(_handle);
        }
    }

    /// <summary>
    /// Check if a token is an end-of-generation token (EOS/EOT).
    /// </summary>
    public bool IsEog(int token)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return Native.infer_llm_is_eog(_handle, token) != 0;
    }

    /// <summary>
    /// Tokenize text using the model's built-in vocabulary.
    /// </summary>
    /// <param name="text">Input text.</param>
    /// <param name="addBos">Whether to prepend the BOS token.</param>
    /// <returns>Array of token IDs.</returns>
    public int[] Tokenize(string text, bool addBos = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(text);

        int[] buf = new int[text.Length + 128];
        int n = Native.infer_llm_tokenize(_handle, text, addBos ? 1 : 0, buf, buf.Length);
        if (n < 0)
            throw InfergoException.FromNative("Tokenization failed");

        int[] result = new int[n];
        Array.Copy(buf, result, n);
        return result;
    }

    /// <summary>
    /// Convert a single token ID to its string piece.
    /// </summary>
    public string TokenToPiece(int token)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        byte[] buf = new byte[256];
        int rc = Native.infer_llm_token_to_piece(_handle, token, buf, buf.Length);
        if (rc < 0)
            throw InfergoException.FromNative($"token_to_piece failed for token {token}");

        // buf is null-terminated
        int len = Array.IndexOf(buf, (byte)0);
        if (len < 0) len = buf.Length;
        return Encoding.UTF8.GetString(buf, 0, len);
    }

    /// <summary>
    /// Callback invoked for each generated token during streaming.
    /// Return true to continue, false to stop.
    /// </summary>
    public delegate bool TokenCallback(int token, string piece);

    // Native callback signature matching InferTokenCallback:
    // int (*)(int token, const char* piece, void* user_data)
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int NativeTokenCallback(int token, IntPtr piece, IntPtr userData);

    /// <summary>
    /// Run the full generation loop: prefill + decode + sample, all in native code.
    /// </summary>
    /// <param name="promptTokens">Pre-tokenized prompt (including BOS).</param>
    /// <param name="maxTokens">Maximum number of tokens to generate.</param>
    /// <param name="temperature">Sampling temperature (0 = greedy).</param>
    /// <param name="topP">Nucleus sampling threshold (1.0 = disabled).</param>
    /// <param name="grammar">Optional GBNF grammar string for constrained generation.</param>
    /// <param name="callback">Optional streaming callback for each token.</param>
    /// <returns>The generated text and token count.</returns>
    public (string Text, int TokenCount) Generate(
        int[] promptTokens,
        int maxTokens = 256,
        float temperature = 0.7f,
        float topP = 1.0f,
        string? grammar = null,
        TokenCallback? callback = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(promptTokens);

        byte[] textBuf = new byte[maxTokens * 16]; // generous buffer
        IntPtr cbPtr = IntPtr.Zero;
        GCHandle? gcHandle = null;

        if (callback != null)
        {
            NativeTokenCallback nativeCb = (token, piecePtr, _) =>
            {
                string piece = piecePtr != IntPtr.Zero
                    ? Marshal.PtrToStringUTF8(piecePtr) ?? ""
                    : "";
                return callback(token, piece) ? 1 : 0;
            };
            gcHandle = GCHandle.Alloc(nativeCb);
            cbPtr = Marshal.GetFunctionPointerForDelegate(nativeCb);
        }

        try
        {
            int rc = Native.infer_llm_generate(
                _handle,
                promptTokens,
                promptTokens.Length,
                maxTokens,
                temperature,
                topP,
                grammar,
                cbPtr,
                IntPtr.Zero,
                textBuf,
                textBuf.Length,
                out int genTokens);

            if (rc != 0)
                throw InfergoException.FromNative("Generation failed", rc);

            int len = Array.IndexOf(textBuf, (byte)0);
            if (len < 0) len = textBuf.Length;
            string text = Encoding.UTF8.GetString(textBuf, 0, len);
            return (text, genTokens);
        }
        finally
        {
            gcHandle?.Free();
        }
    }

    /// <summary>
    /// Convenience: tokenize a prompt string and generate.
    /// </summary>
    public (string Text, int TokenCount) Generate(
        string prompt,
        int maxTokens = 256,
        float temperature = 0.7f,
        float topP = 1.0f,
        string? grammar = null,
        TokenCallback? callback = null)
    {
        int[] tokens = Tokenize(prompt, addBos: true);
        return Generate(tokens, maxTokens, temperature, topP, grammar, callback);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_handle != IntPtr.Zero)
        {
            Native.infer_llm_destroy(_handle);
            _handle = IntPtr.Zero;
        }
    }
}
