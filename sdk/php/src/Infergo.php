<?php
/**
 * infergo PHP SDK — FFI bindings to libinfer_api.so
 * Requires PHP 7.4+ with FFI extension enabled.
 */
namespace Infergo;

class Native {
    private static ?\FFI $ffi = null;

    public static function lib(): \FFI {
        if (self::$ffi === null) {
            $path = getenv('INFERGO_LIB_DIR') ?: '/usr/local/lib';
            self::$ffi = \FFI::cdef("
                typedef void* InferLLM;
                typedef void* InferSession;
                typedef void* InferTokenizer;
                typedef void* InferVectorDB;
                typedef void* InferBM25;

                const char* infer_last_error_string(void);

                InferLLM infer_llm_create(const char* path, int gpu_layers, int ctx, int seq_max, int batch);
                void infer_llm_destroy(InferLLM llm);
                int infer_llm_vocab_size(InferLLM llm);
                int infer_llm_tokenize(InferLLM llm, const char* text, int bos, int* out, int max);
                int infer_llm_generate(InferLLM llm, const int* tokens, int n, int max, float temp, float top_p,
                                       const char* grammar, void* cb, void* ud, char* buf, int len, int* gen);

                InferSession infer_session_create(const char* provider, int device);
                int infer_session_load(InferSession s, const char* path);
                void infer_session_destroy(InferSession s);
                InferTokenizer infer_tokenizer_load(const char* path);
                void infer_tokenizer_destroy(InferTokenizer tok);
                int infer_embed_pipeline(InferSession s, InferTokenizer tok, const char* text, float* out, int dim);

                InferVectorDB infer_vectordb_create(int dim, int M, int ef);
                int infer_vectordb_insert(InferVectorDB db, int64_t id, const float* vec, const char* meta);
                int infer_vectordb_search(InferVectorDB db, const float* q, int k, int ef, const char* filt,
                                          int64_t* ids, float* dists, int max);
                int infer_vectordb_size(InferVectorDB db);
                void infer_vectordb_free(InferVectorDB db);

                InferBM25 infer_bm25_create(float k1, float b);
                void infer_bm25_insert(InferBM25 idx, int64_t id, const char* text);
                int infer_bm25_search(InferBM25 idx, const char* q, int k, int64_t* ids, float* scores, int max);
                int infer_bm25_size(InferBM25 idx);
                void infer_bm25_free(InferBM25 idx);
            ", "$path/libinfer_api.so");
        }
        return self::$ffi;
    }
}

class LLM {
    private $handle;

    public function __construct(string $path, int $gpuLayers = -1, int $ctxSize = 4096) {
        $this->handle = Native::lib()->infer_llm_create($path, $gpuLayers, $ctxSize, 1, 2048);
        if (\FFI::isNull($this->handle)) {
            throw new \RuntimeException("Load failed: " . Native::lib()->infer_last_error_string());
        }
    }

    public function generate(string $prompt, int $maxTokens = 128, float $temp = 0.7): string {
        $ffi = Native::lib();
        $tokBuf = $ffi->new("int[4096]");
        $n = $ffi->infer_llm_tokenize($this->handle, $prompt, 1, $tokBuf, 4096);
        if ($n < 0) throw new \RuntimeException("Tokenize failed");

        $out = $ffi->new("char[32768]");
        $gen = $ffi->new("int[1]");
        $rc = $ffi->infer_llm_generate($this->handle, $tokBuf, $n, $maxTokens,
                                         $temp, 0.9, null, null, null, $out, 32768, $gen);
        if ($rc < 0) throw new \RuntimeException("Generate failed");
        return \FFI::string($out);
    }

    public function vocabSize(): int {
        return Native::lib()->infer_llm_vocab_size($this->handle);
    }

    public function close(): void {
        if ($this->handle !== null) {
            Native::lib()->infer_llm_destroy($this->handle);
            $this->handle = null;
        }
    }

    public function __destruct() { $this->close(); }
}

class Embedding {
    private $session;
    private $tokenizer;

    public function __construct(string $model, string $tokenizerPath, string $provider = "cpu") {
        $ffi = Native::lib();
        $this->session = $ffi->infer_session_create($provider, 0);
        $ffi->infer_session_load($this->session, $model);
        $this->tokenizer = $ffi->infer_tokenizer_load($tokenizerPath);
    }

    public function embed(string $text): array {
        $ffi = Native::lib();
        $out = $ffi->new("float[2048]");
        $dim = $ffi->infer_embed_pipeline($this->session, $this->tokenizer, $text, $out, 2048);
        if ($dim < 0) throw new \RuntimeException("Embed failed");
        $result = [];
        for ($i = 0; $i < $dim; $i++) $result[] = $out[$i];
        return $result;
    }

    public function close(): void {
        $ffi = Native::lib();
        if ($this->tokenizer) { $ffi->infer_tokenizer_destroy($this->tokenizer); $this->tokenizer = null; }
        if ($this->session) { $ffi->infer_session_destroy($this->session); $this->session = null; }
    }

    public function __destruct() { $this->close(); }
}

class VectorDB {
    private $handle;

    public function __construct(int $dim = 384, int $M = 16, int $ef = 200) {
        $this->handle = Native::lib()->infer_vectordb_create($dim, $M, $ef);
    }

    public function insert(int $id, array $vec, string $meta = ""): void {
        $ffi = Native::lib();
        $v = $ffi->new("float[" . count($vec) . "]");
        foreach ($vec as $i => $val) $v[$i] = $val;
        $ffi->infer_vectordb_insert($this->handle, $id, $v, $meta);
    }

    public function search(array $query, int $k = 10, int $ef = 50): array {
        $ffi = Native::lib();
        $q = $ffi->new("float[" . count($query) . "]");
        foreach ($query as $i => $val) $q[$i] = $val;
        $ids = $ffi->new("int64_t[$k]");
        $dists = $ffi->new("float[$k]");
        $n = $ffi->infer_vectordb_search($this->handle, $q, $k, $ef, null, $ids, $dists, $k);
        $results = [];
        for ($i = 0; $i < $n; $i++) $results[] = ['id' => $ids[$i], 'distance' => $dists[$i]];
        return $results;
    }

    public function size(): int { return Native::lib()->infer_vectordb_size($this->handle); }
    public function close(): void { if ($this->handle) { Native::lib()->infer_vectordb_free($this->handle); $this->handle = null; } }
    public function __destruct() { $this->close(); }
}

class BM25 {
    private $handle;

    public function __construct(float $k1 = 1.2, float $b = 0.75) {
        $this->handle = Native::lib()->infer_bm25_create($k1, $b);
    }

    public function insert(int $id, string $text): void { Native::lib()->infer_bm25_insert($this->handle, $id, $text); }

    public function search(string $query, int $k = 10): array {
        $ffi = Native::lib();
        $ids = $ffi->new("int64_t[$k]");
        $scores = $ffi->new("float[$k]");
        $n = $ffi->infer_bm25_search($this->handle, $query, $k, $ids, $scores, $k);
        $results = [];
        for ($i = 0; $i < $n; $i++) $results[] = ['id' => $ids[$i], 'score' => $scores[$i]];
        return $results;
    }

    public function size(): int { return Native::lib()->infer_bm25_size($this->handle); }
    public function close(): void { if ($this->handle) { Native::lib()->infer_bm25_free($this->handle); $this->handle = null; } }
    public function __destruct() { $this->close(); }
}
