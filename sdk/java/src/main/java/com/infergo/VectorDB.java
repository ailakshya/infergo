package com.infergo;

/**
 * Persistent HNSW vector database.
 *
 * <pre>{@code
 * try (VectorDB db = new VectorDB(384, 16, 200)) {
 *     db.insert(1, embedding, "{\"title\": \"doc1\"}");
 *     VectorDB.SearchResult[] results = db.search(queryVec, 5, 100, null);
 *     for (VectorDB.SearchResult r : results) {
 *         System.out.printf("id=%d dist=%.4f%n", r.id, r.distance);
 *     }
 * }
 * }</pre>
 */
public class VectorDB implements AutoCloseable {

    private long handle;

    /**
     * A single search result: ID + distance.
     */
    public static class SearchResult {
        public final long id;
        public final float distance;

        public SearchResult(long id, float distance) {
            this.id = id;
            this.distance = distance;
        }

        @Override
        public String toString() {
            return "SearchResult{id=" + id + ", distance=" + distance + "}";
        }
    }

    /**
     * Create a persistent vector database.
     *
     * @param dim            vector dimension (must match embedding model output)
     * @param M              max connections per HNSW node (16 = good default)
     * @param efConstruction search width during build (200 = good default)
     * @throws InfergoException if creation fails
     */
    public VectorDB(int dim, int M, int efConstruction) {
        this.handle = InfergoNative.vectordbCreate(dim, M, efConstruction);
        if (this.handle == 0) {
            String err = InfergoNative.lastErrorString();
            throw new InfergoException("Failed to create VectorDB: " + (err != null ? err : "unknown error"));
        }
    }

    /**
     * Insert a vector with metadata.
     *
     * @param id       unique vector ID
     * @param vec      float array of length dim
     * @param metadata JSON metadata string, or null
     * @throws InfergoException on failure
     */
    public void insert(long id, float[] vec, String metadata) {
        checkOpen();
        int rc = InfergoNative.vectordbInsert(handle, id, vec, metadata);
        if (rc != 0) {
            String err = InfergoNative.lastErrorString();
            throw new InfergoException(rc, "VectorDB insert failed: " + (err != null ? err : "unknown error"));
        }
    }

    /**
     * Insert a vector without metadata.
     */
    public void insert(long id, float[] vec) {
        insert(id, vec, null);
    }

    /**
     * Delete a vector by ID.
     *
     * @param id vector ID to delete
     * @throws InfergoException on failure
     */
    public void delete(long id) {
        checkOpen();
        int rc = InfergoNative.vectordbDelete(handle, id);
        if (rc != 0) {
            String err = InfergoNative.lastErrorString();
            throw new InfergoException(rc, "VectorDB delete failed: " + (err != null ? err : "unknown error"));
        }
    }

    /**
     * Search for k nearest neighbors.
     *
     * @param query          query vector
     * @param k              number of results
     * @param efSearch       search width (higher = more accurate, slower)
     * @param metadataFilter metadata filter expression, or null
     * @return array of search results sorted by distance
     * @throws InfergoException on failure
     */
    public SearchResult[] search(float[] query, int k, int efSearch, String metadataFilter) {
        checkOpen();
        long[] ids = InfergoNative.vectordbSearchIds(handle, query, k, efSearch,
                                                     metadataFilter, k);
        float[] distances = InfergoNative.vectordbSearchDistances(handle, query, k, efSearch,
                                                                  metadataFilter, k);
        if (ids == null || distances == null) {
            String err = InfergoNative.lastErrorString();
            throw new InfergoException("VectorDB search failed: " + (err != null ? err : "unknown error"));
        }

        int n = Math.min(ids.length, distances.length);
        SearchResult[] results = new SearchResult[n];
        for (int i = 0; i < n; i++) {
            results[i] = new SearchResult(ids[i], distances[i]);
        }
        return results;
    }

    /**
     * Search without metadata filter.
     */
    public SearchResult[] search(float[] query, int k, int efSearch) {
        return search(query, k, efSearch, null);
    }

    /**
     * Save the database to a file.
     *
     * @param path file path
     * @throws InfergoException on failure
     */
    public void save(String path) {
        checkOpen();
        int rc = InfergoNative.vectordbSave(handle, path);
        if (rc != 0) {
            String err = InfergoNative.lastErrorString();
            throw new InfergoException(rc, "VectorDB save failed: " + (err != null ? err : "unknown error"));
        }
    }

    /**
     * Load the database from a file.
     *
     * @param path file path
     * @throws InfergoException on failure
     */
    public void load(String path) {
        checkOpen();
        int rc = InfergoNative.vectordbLoad(handle, path);
        if (rc != 0) {
            String err = InfergoNative.lastErrorString();
            throw new InfergoException(rc, "VectorDB load failed: " + (err != null ? err : "unknown error"));
        }
    }

    /**
     * Returns the number of vectors in the database.
     */
    public int size() {
        checkOpen();
        return InfergoNative.vectordbSize(handle);
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
            InfergoNative.vectordbFree(handle);
            handle = 0;
        }
    }

    private void checkOpen() {
        if (handle == 0) {
            throw new IllegalStateException("VectorDB is closed");
        }
    }
}
