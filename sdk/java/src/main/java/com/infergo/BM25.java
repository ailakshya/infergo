package com.infergo;

/**
 * BM25 full-text search index.
 *
 * <pre>{@code
 * try (BM25 index = new BM25(1.2f, 0.75f)) {
 *     index.insert(1, "The quick brown fox");
 *     index.insert(2, "Jumped over the lazy dog");
 *     BM25.SearchResult[] results = index.search("quick fox", 5);
 *     for (BM25.SearchResult r : results) {
 *         System.out.printf("id=%d score=%.4f%n", r.id, r.score);
 *     }
 * }
 * }</pre>
 */
public class BM25 implements AutoCloseable {

    private long handle;

    /**
     * A single BM25 search result: ID + score.
     */
    public static class SearchResult {
        public final long id;
        public final float score;

        public SearchResult(long id, float score) {
            this.id = id;
            this.score = score;
        }

        @Override
        public String toString() {
            return "SearchResult{id=" + id + ", score=" + score + "}";
        }
    }

    /**
     * Create a BM25 full-text search index.
     *
     * @param k1 term frequency saturation (default 1.2)
     * @param b  document length normalization (default 0.75)
     * @throws InfergoException if creation fails
     */
    public BM25(float k1, float b) {
        this.handle = InfergoNative.bm25Create(k1, b);
        if (this.handle == 0) {
            String err = InfergoNative.lastErrorString();
            throw new InfergoException("Failed to create BM25: " + (err != null ? err : "unknown error"));
        }
    }

    /**
     * Create a BM25 index with default parameters (k1=1.2, b=0.75).
     */
    public BM25() {
        this(1.2f, 0.75f);
    }

    /**
     * Insert a document into the index.
     *
     * @param id   unique document ID
     * @param text document text
     */
    public void insert(long id, String text) {
        checkOpen();
        InfergoNative.bm25Insert(handle, id, text);
    }

    /**
     * Remove a document from the index.
     *
     * @param id document ID to remove
     */
    public void remove(long id) {
        checkOpen();
        InfergoNative.bm25Remove(handle, id);
    }

    /**
     * Search the index.
     *
     * @param query query string
     * @param k     max number of results
     * @return array of search results sorted by score (descending)
     * @throws InfergoException on failure
     */
    public SearchResult[] search(String query, int k) {
        checkOpen();
        long[] ids = InfergoNative.bm25SearchIds(handle, query, k, k);
        float[] scores = InfergoNative.bm25SearchScores(handle, query, k, k);
        if (ids == null || scores == null) {
            String err = InfergoNative.lastErrorString();
            throw new InfergoException("BM25 search failed: " + (err != null ? err : "unknown error"));
        }

        int n = Math.min(ids.length, scores.length);
        SearchResult[] results = new SearchResult[n];
        for (int i = 0; i < n; i++) {
            results[i] = new SearchResult(ids[i], scores[i]);
        }
        return results;
    }

    /**
     * Returns the number of documents in the index.
     */
    public int size() {
        checkOpen();
        return InfergoNative.bm25Size(handle);
    }

    /**
     * Save the index to a file.
     *
     * @param path file path
     * @throws InfergoException on failure
     */
    public void save(String path) {
        checkOpen();
        int rc = InfergoNative.bm25Save(handle, path);
        if (rc != 0) {
            String err = InfergoNative.lastErrorString();
            throw new InfergoException(rc, "BM25 save failed: " + (err != null ? err : "unknown error"));
        }
    }

    /**
     * Load the index from a file.
     *
     * @param path file path
     * @throws InfergoException on failure
     */
    public void load(String path) {
        checkOpen();
        int rc = InfergoNative.bm25Load(handle, path);
        if (rc != 0) {
            String err = InfergoNative.lastErrorString();
            throw new InfergoException(rc, "BM25 load failed: " + (err != null ? err : "unknown error"));
        }
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
            InfergoNative.bm25Free(handle);
            handle = 0;
        }
    }

    private void checkOpen() {
        if (handle == 0) {
            throw new IllegalStateException("BM25 is closed");
        }
    }
}
