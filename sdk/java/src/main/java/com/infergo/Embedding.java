package com.infergo;

/**
 * Embedding model backed by ONNX session + HuggingFace tokenizer.
 *
 * <pre>{@code
 * try (Embedding embed = new Embedding("model.onnx", "tokenizer.json", "cpu", 0)) {
 *     float[] vec = embed.embed("Hello, world!");
 *     System.out.println("Dimension: " + vec.length);
 * }
 * }</pre>
 */
public class Embedding implements AutoCloseable {

    private long sessionHandle;
    private long tokenizerHandle;
    private final int maxDim;

    /**
     * Create an embedding model.
     *
     * @param modelPath     path to the ONNX embedding model
     * @param tokenizerPath path to tokenizer.json
     * @param provider      execution provider ("cpu", "cuda", etc.)
     * @param deviceId      GPU device index
     * @throws InfergoException if model or tokenizer cannot be loaded
     */
    public Embedding(String modelPath, String tokenizerPath,
                     String provider, int deviceId) {
        this(modelPath, tokenizerPath, provider, deviceId, 2048);
    }

    /**
     * Create an embedding model with explicit max dimension.
     *
     * @param modelPath     path to the ONNX embedding model
     * @param tokenizerPath path to tokenizer.json
     * @param provider      execution provider ("cpu", "cuda", etc.)
     * @param deviceId      GPU device index
     * @param maxDim        maximum embedding dimension
     * @throws InfergoException if model or tokenizer cannot be loaded
     */
    public Embedding(String modelPath, String tokenizerPath,
                     String provider, int deviceId, int maxDim) {
        this.maxDim = maxDim;

        this.sessionHandle = InfergoNative.sessionCreate(provider, deviceId);
        if (this.sessionHandle == 0) {
            String err = InfergoNative.lastErrorString();
            throw new InfergoException("Failed to create session: " + (err != null ? err : "unknown error"));
        }

        int rc = InfergoNative.sessionLoad(this.sessionHandle, modelPath);
        if (rc != 0) {
            String err = InfergoNative.lastErrorString();
            InfergoNative.sessionDestroy(this.sessionHandle);
            this.sessionHandle = 0;
            throw new InfergoException(rc, "Failed to load model: " + (err != null ? err : "unknown error"));
        }

        this.tokenizerHandle = InfergoNative.tokenizerLoad(tokenizerPath);
        if (this.tokenizerHandle == 0) {
            String err = InfergoNative.lastErrorString();
            InfergoNative.sessionDestroy(this.sessionHandle);
            this.sessionHandle = 0;
            throw new InfergoException("Failed to load tokenizer: " + (err != null ? err : "unknown error"));
        }
    }

    /**
     * Compute the embedding vector for a single text.
     *
     * @param text input text
     * @return float array of the embedding vector
     * @throws InfergoException on failure
     */
    public float[] embed(String text) {
        checkOpen();
        float[] result = InfergoNative.embedPipeline(sessionHandle, tokenizerHandle,
                                                     text, maxDim);
        if (result == null) {
            String err = InfergoNative.lastErrorString();
            throw new InfergoException("Embedding failed: " + (err != null ? err : "unknown error"));
        }
        return result;
    }

    /**
     * Compute embedding vectors for multiple texts in one call.
     *
     * @param texts array of input texts
     * @return array of embedding vectors (one per text)
     * @throws InfergoException on failure
     */
    public float[][] embedBatch(String[] texts) {
        checkOpen();
        float[] flat = InfergoNative.embedBatchPipeline(sessionHandle, tokenizerHandle,
                                                        texts, maxDim);
        if (flat == null) {
            String err = InfergoNative.lastErrorString();
            throw new InfergoException("Batch embedding failed: " + (err != null ? err : "unknown error"));
        }

        int dim = flat.length / texts.length;
        float[][] result = new float[texts.length][dim];
        for (int i = 0; i < texts.length; i++) {
            System.arraycopy(flat, i * dim, result[i], 0, dim);
        }
        return result;
    }

    /**
     * Returns the native session handle. For advanced use only.
     */
    public long getSessionHandle() {
        checkOpen();
        return sessionHandle;
    }

    /**
     * Returns the native tokenizer handle. For advanced use only.
     */
    public long getTokenizerHandle() {
        checkOpen();
        return tokenizerHandle;
    }

    @Override
    public void close() {
        if (tokenizerHandle != 0) {
            InfergoNative.tokenizerDestroy(tokenizerHandle);
            tokenizerHandle = 0;
        }
        if (sessionHandle != 0) {
            InfergoNative.sessionDestroy(sessionHandle);
            sessionHandle = 0;
        }
    }

    private void checkOpen() {
        if (sessionHandle == 0 || tokenizerHandle == 0) {
            throw new IllegalStateException("Embedding is closed");
        }
    }
}
