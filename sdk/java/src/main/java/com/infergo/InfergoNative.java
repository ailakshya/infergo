package com.infergo;

/**
 * JNI native method declarations for the infergo C API.
 *
 * All opaque C handles (void*) are represented as {@code long} (jlong)
 * holding the native pointer value. A zero value means NULL / invalid.
 */
public final class InfergoNative {

    static {
        System.loadLibrary("infergo_jni");
    }

    private InfergoNative() {}

    // ── Error ────────────────────────────────────────────────────────────────

    /** Returns the last error string from the calling thread, or null. */
    public static native String lastErrorString();

    // ── LLM ──────────────────────────────────────────────────────────────────

    /**
     * Load a GGUF model and create an LLM engine.
     *
     * @param path         path to the GGUF model file
     * @param nGpuLayers   transformer layers to offload to GPU
     * @param ctxSize      total KV cache token budget
     * @param nSeqMax      max concurrent sequences
     * @param nBatch       max tokens per decode call
     * @return native handle, or 0 on failure
     */
    public static native long llmCreate(String path, int nGpuLayers,
                                        int ctxSize, int nSeqMax, int nBatch);

    /** Destroy an LLM engine. Safe to call with 0. */
    public static native void llmDestroy(long llm);

    /** Returns the vocabulary size. */
    public static native int llmVocabSize(long llm);

    /** Returns the BOS token ID. */
    public static native int llmBos(long llm);

    /** Returns the EOS token ID. */
    public static native int llmEos(long llm);

    /**
     * Tokenize text using the model's vocabulary.
     *
     * @param llm       native LLM handle
     * @param text      input text
     * @param addBos    whether to prepend BOS token
     * @param maxTokens maximum tokens to return
     * @return array of token IDs, or null on error
     */
    public static native int[] llmTokenize(long llm, String text,
                                           boolean addBos, int maxTokens);

    /**
     * Run the full generation loop in C.
     *
     * @param llm          native LLM handle
     * @param promptTokens pre-tokenized prompt (including BOS)
     * @param maxTokens    max generation length
     * @param temperature  sampling temperature (0 = greedy)
     * @param topP         nucleus sampling (1.0 = disabled)
     * @param grammar      GBNF grammar string, or null
     * @param maxTextLen   capacity for the output text buffer
     * @return generated text, or null on error
     */
    public static native String llmGenerate(long llm, int[] promptTokens,
                                            int maxTokens, float temperature,
                                            float topP, String grammar,
                                            int maxTextLen);

    // ── Session (ONNX) ───────────────────────────────────────────────────────

    /**
     * Create an inference session.
     *
     * @param provider "cpu", "cuda", "tensorrt", "coreml", "openvino"
     * @param deviceId GPU device index
     * @return native handle, or 0 on failure
     */
    public static native long sessionCreate(String provider, int deviceId);

    /** Load an ONNX model into the session. Returns 0 on success. */
    public static native int sessionLoad(long session, String path);

    /** Destroy session. Safe to call with 0. */
    public static native void sessionDestroy(long session);

    // ── Tokenizer ────────────────────────────────────────────────────────────

    /**
     * Load a HuggingFace tokenizer from a tokenizer.json file.
     *
     * @param path path to tokenizer.json
     * @return native handle, or 0 on failure
     */
    public static native long tokenizerLoad(String path);

    /** Destroy tokenizer. Safe to call with 0. */
    public static native void tokenizerDestroy(long tokenizer);

    /** Returns the vocabulary size. */
    public static native int tokenizerVocabSize(long tokenizer);

    // ── Embedding ────────────────────────────────────────────────────────────

    /**
     * Run the complete embedding pipeline: tokenize, ONNX infer, pool, normalize.
     *
     * @param session   native ONNX session handle
     * @param tokenizer native tokenizer handle
     * @param text      input text
     * @param maxDim    max embedding dimension
     * @return float array of the embedding vector, or null on error
     */
    public static native float[] embedPipeline(long session, long tokenizer,
                                               String text, int maxDim);

    /**
     * Batch embedding: N texts to N vectors in one call.
     *
     * @param session   native ONNX session handle
     * @param tokenizer native tokenizer handle
     * @param texts     array of input texts
     * @param maxDim    max embedding dimension
     * @return flat float array of shape [n_texts * dim], or null on error
     */
    public static native float[] embedBatchPipeline(long session, long tokenizer,
                                                    String[] texts, int maxDim);

    // ── VectorDB ─────────────────────────────────────────────────────────────

    /**
     * Create a persistent vector database.
     *
     * @param dim            vector dimension
     * @param M              max connections per node (16 = good default)
     * @param efConstruction search width during build (200 = good default)
     * @return native handle, or 0 on failure
     */
    public static native long vectordbCreate(int dim, int M, int efConstruction);

    /**
     * Insert a vector with an ID and optional metadata.
     *
     * @param db       native VectorDB handle
     * @param id       unique vector ID
     * @param vec      float array of length dim
     * @param metadata JSON metadata string, or null
     * @return 0 on success
     */
    public static native int vectordbInsert(long db, long id, float[] vec,
                                            String metadata);

    /**
     * Delete a vector by ID.
     *
     * @return 0 on success
     */
    public static native int vectordbDelete(long db, long id);

    /**
     * Search for k nearest neighbors.
     *
     * @param db             native VectorDB handle
     * @param query          query vector
     * @param k              number of results
     * @param efSearch       search width
     * @param metadataFilter metadata filter expression, or null
     * @param maxResults     capacity of output arrays
     * @return search results as [ids..., distances...] interleaved in a
     *         {@link VectorDBSearchResult}, or null on error
     */
    public static native long[] vectordbSearchIds(long db, float[] query,
                                                  int k, int efSearch,
                                                  String metadataFilter,
                                                  int maxResults);

    public static native float[] vectordbSearchDistances(long db, float[] query,
                                                         int k, int efSearch,
                                                         String metadataFilter,
                                                         int maxResults);

    /** Save vector database to file. Returns 0 on success. */
    public static native int vectordbSave(long db, String path);

    /** Load vector database from file. Returns 0 on success. */
    public static native int vectordbLoad(long db, String path);

    /** Number of vectors in the database. */
    public static native int vectordbSize(long db);

    /** Free the vector database. Safe to call with 0. */
    public static native void vectordbFree(long db);

    // ── BM25 ─────────────────────────────────────────────────────────────────

    /**
     * Create a BM25 full-text search index.
     *
     * @param k1 term frequency saturation (default 1.2)
     * @param b  document length normalization (default 0.75)
     * @return native handle, or 0 on failure
     */
    public static native long bm25Create(float k1, float b);

    /** Insert a document. */
    public static native void bm25Insert(long idx, long id, String text);

    /** Remove a document. */
    public static native void bm25Remove(long idx, long id);

    /**
     * Search the BM25 index.
     *
     * @param idx        native BM25 handle
     * @param query      query string
     * @param k          max results
     * @param maxResults capacity of output arrays
     * @return array of result IDs, or null on error
     */
    public static native long[] bm25SearchIds(long idx, String query,
                                              int k, int maxResults);

    public static native float[] bm25SearchScores(long idx, String query,
                                                  int k, int maxResults);

    /** Number of documents in the index. */
    public static native int bm25Size(long idx);

    /** Save BM25 index to file. Returns 0 on success. */
    public static native int bm25Save(long idx, String path);

    /** Load BM25 index from file. Returns 0 on success. */
    public static native int bm25Load(long idx, String path);

    /** Free the BM25 index. Safe to call with 0. */
    public static native void bm25Free(long idx);
}
