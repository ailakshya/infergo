package com.infergo;

/**
 * LLM engine backed by a GGUF model.
 *
 * <pre>{@code
 * try (LLM llm = new LLM("model.gguf", 99, 4096, 1, 512)) {
 *     String response = llm.generate("Hello, world!", 256, 0.7f, 1.0f);
 *     System.out.println(response);
 * }
 * }</pre>
 */
public class LLM implements AutoCloseable {

    private long handle;

    /**
     * Load a GGUF model and create an LLM engine.
     *
     * @param path       path to the GGUF model file
     * @param nGpuLayers transformer layers to offload to GPU (use large value for all)
     * @param ctxSize    total KV cache token budget across all sequences
     * @param nSeqMax    max number of concurrent sequences
     * @param nBatch     max tokens per decode call
     * @throws InfergoException if the model cannot be loaded
     */
    public LLM(String path, int nGpuLayers, int ctxSize, int nSeqMax, int nBatch) {
        this.handle = InfergoNative.llmCreate(path, nGpuLayers, ctxSize, nSeqMax, nBatch);
        if (this.handle == 0) {
            String err = InfergoNative.lastErrorString();
            throw new InfergoException("Failed to create LLM: " + (err != null ? err : "unknown error"));
        }
    }

    /**
     * Returns the vocabulary size.
     */
    public int vocabSize() {
        checkOpen();
        return InfergoNative.llmVocabSize(handle);
    }

    /**
     * Returns the BOS token ID.
     */
    public int bosToken() {
        checkOpen();
        return InfergoNative.llmBos(handle);
    }

    /**
     * Returns the EOS token ID.
     */
    public int eosToken() {
        checkOpen();
        return InfergoNative.llmEos(handle);
    }

    /**
     * Tokenize text using the model's vocabulary.
     *
     * @param text   input text
     * @param addBos whether to prepend BOS token
     * @return array of token IDs
     * @throws InfergoException on tokenization failure
     */
    public int[] tokenize(String text, boolean addBos) {
        return tokenize(text, addBos, 8192);
    }

    /**
     * Tokenize text with explicit max token count.
     *
     * @param text      input text
     * @param addBos    whether to prepend BOS token
     * @param maxTokens maximum tokens to produce
     * @return array of token IDs
     * @throws InfergoException on tokenization failure
     */
    public int[] tokenize(String text, boolean addBos, int maxTokens) {
        checkOpen();
        int[] tokens = InfergoNative.llmTokenize(handle, text, addBos, maxTokens);
        if (tokens == null) {
            String err = InfergoNative.lastErrorString();
            throw new InfergoException("Tokenization failed: " + (err != null ? err : "unknown error"));
        }
        return tokens;
    }

    /**
     * Generate text from a string prompt.
     *
     * @param prompt      input text (will be tokenized with BOS)
     * @param maxTokens   max generation length
     * @param temperature sampling temperature (0 = greedy)
     * @param topP        nucleus sampling (1.0 = disabled)
     * @return generated text
     * @throws InfergoException on generation failure
     */
    public String generate(String prompt, int maxTokens, float temperature, float topP) {
        return generate(prompt, maxTokens, temperature, topP, null);
    }

    /**
     * Generate text from a string prompt with grammar constraint.
     *
     * @param prompt      input text (will be tokenized with BOS)
     * @param maxTokens   max generation length
     * @param temperature sampling temperature (0 = greedy)
     * @param topP        nucleus sampling (1.0 = disabled)
     * @param grammar     GBNF grammar string, or null for unconstrained
     * @return generated text
     * @throws InfergoException on generation failure
     */
    public String generate(String prompt, int maxTokens, float temperature,
                           float topP, String grammar) {
        checkOpen();
        int[] tokens = tokenize(prompt, true);
        return generateFromTokens(tokens, maxTokens, temperature, topP, grammar);
    }

    /**
     * Generate text from pre-tokenized prompt.
     *
     * @param promptTokens pre-tokenized prompt (including BOS)
     * @param maxTokens    max generation length
     * @param temperature  sampling temperature (0 = greedy)
     * @param topP         nucleus sampling (1.0 = disabled)
     * @param grammar      GBNF grammar string, or null for unconstrained
     * @return generated text
     * @throws InfergoException on generation failure
     */
    public String generateFromTokens(int[] promptTokens, int maxTokens,
                                     float temperature, float topP, String grammar) {
        checkOpen();
        int maxTextLen = maxTokens * 8; // conservative estimate for UTF-8
        String result = InfergoNative.llmGenerate(handle, promptTokens, maxTokens,
                                                  temperature, topP, grammar, maxTextLen);
        if (result == null) {
            String err = InfergoNative.lastErrorString();
            throw new InfergoException("Generation failed: " + (err != null ? err : "unknown error"));
        }
        return result;
    }

    /**
     * Returns the native handle. For advanced use only.
     */
    public long getHandle() {
        checkOpen();
        return handle;
    }

    @Override
    public void close() {
        if (handle != 0) {
            InfergoNative.llmDestroy(handle);
            handle = 0;
        }
    }

    private void checkOpen() {
        if (handle == 0) {
            throw new IllegalStateException("LLM is closed");
        }
    }
}
