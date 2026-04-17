//! Safe Rust bindings for infergo.
//!
//! Provides RAII wrappers around the infergo C API for LLM inference,
//! embeddings, vector search (HNSW), BM25 full-text search, and RAG.
//!
//! # Example
//!
//! ```no_run
//! use infergo::Llm;
//!
//! let llm = Llm::new("model.gguf", 99, 4096, 1, 512)?;
//! let tokens = llm.tokenize("Hello, world!", true)?;
//! let response = llm.generate(&tokens, 256, 0.7, 0.9, None, None)?;
//! println!("{}", response.text);
//! # Ok::<(), infergo::Error>(())
//! ```

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

pub use infergo_sys;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors returned by infergo operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("infergo: {0}")]
    Api(String),

    #[error("infergo: null handle returned — {0}")]
    NullHandle(String),

    #[error("infergo: invalid argument — {0}")]
    InvalidArg(String),

    #[error("infergo: string contains interior NUL byte")]
    Nul(#[from] std::ffi::NulError),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Fetch the last error message from the C API (thread-local).
fn last_error() -> String {
    unsafe {
        let ptr = infergo_sys::infer_last_error_string();
        if ptr.is_null() {
            "unknown error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

// ---------------------------------------------------------------------------
// GenerateResult
// ---------------------------------------------------------------------------

/// Result of an LLM generation call.
#[derive(Debug, Clone)]
pub struct GenerateResult {
    /// The generated text.
    pub text: String,
    /// Number of tokens generated.
    pub n_tokens: i32,
}

// ---------------------------------------------------------------------------
// Token callback support
// ---------------------------------------------------------------------------

/// User-facing token callback: receives the token id and string piece.
/// Return `true` to continue generation, `false` to stop.
pub type TokenCallback = Box<dyn FnMut(i32, &str) -> bool + Send>;

struct CallbackCtx {
    cb: TokenCallback,
}

unsafe extern "C" fn token_callback_trampoline(
    token: c_int,
    piece: *const c_char,
    user_data: *mut c_void,
) -> c_int {
    let ctx = &mut *(user_data as *mut CallbackCtx);
    let s = if piece.is_null() {
        ""
    } else {
        CStr::from_ptr(piece).to_str().unwrap_or("")
    };
    if (ctx.cb)(token, s) {
        1
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Llm
// ---------------------------------------------------------------------------

/// RAII wrapper around an infergo LLM engine (GGUF model).
///
/// Implements `Drop` to release the underlying C handle.
pub struct Llm {
    handle: infergo_sys::InferLLM,
}

// The C handles are thread-safe (internal mutex).
unsafe impl Send for Llm {}
unsafe impl Sync for Llm {}

impl Llm {
    /// Load a GGUF model and create an LLM engine.
    ///
    /// - `path`: path to the `.gguf` file
    /// - `n_gpu_layers`: number of transformer layers to offload to GPU (use a
    ///   large value like 99 to offload everything)
    /// - `ctx_size`: total KV cache token budget across all sequences
    /// - `n_seq_max`: maximum number of concurrent sequences
    /// - `n_batch`: maximum tokens per decode call
    pub fn new(
        path: &str,
        n_gpu_layers: i32,
        ctx_size: i32,
        n_seq_max: i32,
        n_batch: i32,
    ) -> Result<Self> {
        let c_path = CString::new(path)?;
        let handle = unsafe {
            infergo_sys::infer_llm_create(
                c_path.as_ptr(),
                n_gpu_layers,
                ctx_size,
                n_seq_max,
                n_batch,
            )
        };
        if handle.is_null() {
            return Err(Error::NullHandle(last_error()));
        }
        Ok(Self { handle })
    }

    /// Returns the vocabulary size.
    pub fn vocab_size(&self) -> i32 {
        unsafe { infergo_sys::infer_llm_vocab_size(self.handle) }
    }

    /// Returns the BOS token ID.
    pub fn bos(&self) -> i32 {
        unsafe { infergo_sys::infer_llm_bos(self.handle) }
    }

    /// Returns the EOS token ID.
    pub fn eos(&self) -> i32 {
        unsafe { infergo_sys::infer_llm_eos(self.handle) }
    }

    /// Returns `true` if the token is an end-of-generation token.
    pub fn is_eog(&self, token: i32) -> bool {
        unsafe { infergo_sys::infer_llm_is_eog(self.handle, token) != 0 }
    }

    /// Tokenize text using the model's built-in vocabulary.
    ///
    /// Returns the token IDs. Set `add_bos` to prepend the BOS token.
    pub fn tokenize(&self, text: &str, add_bos: bool) -> Result<Vec<i32>> {
        let c_text = CString::new(text)?;
        let max_tokens = text.len() as i32 + 64; // generous upper bound
        let mut ids = vec![0i32; max_tokens as usize];
        let n = unsafe {
            infergo_sys::infer_llm_tokenize(
                self.handle,
                c_text.as_ptr(),
                if add_bos { 1 } else { 0 },
                ids.as_mut_ptr(),
                max_tokens,
            )
        };
        if n < 0 {
            return Err(Error::Api(last_error()));
        }
        ids.truncate(n as usize);
        Ok(ids)
    }

    /// Convert a single token ID to its string piece.
    pub fn token_to_piece(&self, token: i32) -> Result<String> {
        let mut buf = vec![0u8; 256];
        let ret = unsafe {
            infergo_sys::infer_llm_token_to_piece(
                self.handle,
                token,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as c_int,
            )
        };
        if ret != 0 {
            return Err(Error::Api(last_error()));
        }
        let s = unsafe { CStr::from_ptr(buf.as_ptr() as *const c_char) };
        Ok(s.to_string_lossy().into_owned())
    }

    /// Run the full generation loop.
    ///
    /// - `prompt_tokens`: pre-tokenized prompt (including BOS)
    /// - `max_tokens`: maximum generation length
    /// - `temperature`: sampling temperature (0 = greedy)
    /// - `top_p`: nucleus sampling (1.0 = disabled)
    /// - `grammar`: optional GBNF grammar string
    /// - `callback`: optional streaming callback (receives each token)
    pub fn generate(
        &self,
        prompt_tokens: &[i32],
        max_tokens: i32,
        temperature: f32,
        top_p: f32,
        grammar: Option<&str>,
        callback: Option<TokenCallback>,
    ) -> Result<GenerateResult> {
        let c_grammar = match grammar {
            Some(g) => Some(CString::new(g)?),
            None => None,
        };
        let grammar_ptr = c_grammar
            .as_ref()
            .map_or(ptr::null(), |g| g.as_ptr());

        let max_text_len: c_int = max_tokens * 16; // generous buffer
        let mut out_text = vec![0u8; max_text_len as usize];
        let mut out_gen_tokens: c_int = 0;

        let (cb_fn, cb_data): (infergo_sys::InferTokenCallback, *mut c_void) = match callback {
            Some(cb) => {
                let ctx = Box::new(CallbackCtx { cb });
                // Leak into raw pointer — reclaimed after generate returns
                let leaked = Box::into_raw(ctx);
                (
                    Some(
                        token_callback_trampoline
                            as unsafe extern "C" fn(c_int, *const c_char, *mut c_void) -> c_int,
                    ),
                    leaked as *mut c_void,
                )
            }
            None => (None, ptr::null_mut()),
        };

        let ret = unsafe {
            infergo_sys::infer_llm_generate(
                self.handle,
                prompt_tokens.as_ptr(),
                prompt_tokens.len() as c_int,
                max_tokens,
                temperature,
                top_p,
                grammar_ptr,
                cb_fn,
                cb_data,
                out_text.as_mut_ptr() as *mut c_char,
                max_text_len,
                &mut out_gen_tokens,
            )
        };

        // Reclaim the callback context
        if !cb_data.is_null() {
            unsafe {
                let _ = Box::from_raw(cb_data as *mut CallbackCtx);
            }
        }

        if ret != 0 {
            return Err(Error::Api(last_error()));
        }

        let text = unsafe { CStr::from_ptr(out_text.as_ptr() as *const c_char) };
        Ok(GenerateResult {
            text: text.to_string_lossy().into_owned(),
            n_tokens: out_gen_tokens,
        })
    }

    /// Get the raw C handle (for advanced use with `infergo_sys`).
    pub fn as_raw(&self) -> infergo_sys::InferLLM {
        self.handle
    }
}

impl Drop for Llm {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { infergo_sys::infer_llm_destroy(self.handle) }
        }
    }
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

/// RAII wrapper for the infergo embedding pipeline (ONNX session + tokenizer).
pub struct Embedding {
    session: infergo_sys::InferSession,
    tokenizer: infergo_sys::InferTokenizer,
    max_dim: i32,
}

unsafe impl Send for Embedding {}
unsafe impl Sync for Embedding {}

impl Embedding {
    /// Create an embedding pipeline.
    ///
    /// - `model_path`: path to the ONNX embedding model
    /// - `tokenizer_path`: path to `tokenizer.json` (HuggingFace format)
    /// - `provider`: execution provider (`"cpu"`, `"cuda"`, etc.)
    /// - `device_id`: GPU device index (0 for first GPU)
    /// - `max_dim`: maximum embedding dimension (e.g. 384, 768, 1024)
    pub fn new(
        model_path: &str,
        tokenizer_path: &str,
        provider: &str,
        device_id: i32,
        max_dim: i32,
    ) -> Result<Self> {
        let c_provider = CString::new(provider)?;
        let session = unsafe {
            infergo_sys::infer_session_create(c_provider.as_ptr(), device_id)
        };
        if session.is_null() {
            return Err(Error::NullHandle(last_error()));
        }

        let c_model = CString::new(model_path)?;
        let err = unsafe { infergo_sys::infer_session_load(session, c_model.as_ptr()) };
        if err != infergo_sys::INFER_OK {
            unsafe { infergo_sys::infer_session_destroy(session) };
            return Err(Error::Api(last_error()));
        }

        let c_tok_path = CString::new(tokenizer_path)?;
        let tokenizer = unsafe { infergo_sys::infer_tokenizer_load(c_tok_path.as_ptr()) };
        if tokenizer.is_null() {
            unsafe { infergo_sys::infer_session_destroy(session) };
            return Err(Error::NullHandle(last_error()));
        }

        Ok(Self {
            session,
            tokenizer,
            max_dim,
        })
    }

    /// Embed a single text, returning a normalized float vector.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let c_text = CString::new(text)?;
        let mut out_vec = vec![0.0f32; self.max_dim as usize];
        let dim = unsafe {
            infergo_sys::infer_embed_pipeline(
                self.session,
                self.tokenizer,
                c_text.as_ptr(),
                out_vec.as_mut_ptr(),
                self.max_dim,
            )
        };
        if dim < 0 {
            return Err(Error::Api(last_error()));
        }
        out_vec.truncate(dim as usize);
        Ok(out_vec)
    }

    /// Embed a batch of texts in one C call.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let c_strings: Vec<CString> = texts
            .iter()
            .map(|t| CString::new(*t))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let c_ptrs: Vec<*const c_char> = c_strings.iter().map(|s| s.as_ptr()).collect();

        let n_texts = texts.len() as c_int;
        let mut out_vecs = vec![0.0f32; texts.len() * self.max_dim as usize];

        let dim = unsafe {
            infergo_sys::infer_embed_batch_pipeline(
                self.session,
                self.tokenizer,
                c_ptrs.as_ptr(),
                n_texts,
                out_vecs.as_mut_ptr(),
                self.max_dim,
            )
        };
        if dim < 0 {
            return Err(Error::Api(last_error()));
        }
        let dim = dim as usize;
        let result = out_vecs
            .chunks_exact(self.max_dim as usize)
            .map(|chunk| chunk[..dim].to_vec())
            .collect();
        Ok(result)
    }

    /// Rerank documents by relevance to a query.
    ///
    /// Returns `(scores, indices)` sorted by descending relevance.
    pub fn rerank(
        &self,
        query: &str,
        documents: &[&str],
        max_results: i32,
    ) -> Result<(Vec<f32>, Vec<i32>)> {
        let c_query = CString::new(query)?;
        let c_docs: Vec<CString> = documents
            .iter()
            .map(|d| CString::new(*d))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let c_ptrs: Vec<*const c_char> = c_docs.iter().map(|s| s.as_ptr()).collect();

        let n_docs = documents.len() as c_int;
        let mut out_scores = vec![0.0f32; max_results as usize];
        let mut out_indices = vec![0i32; max_results as usize];

        let n = unsafe {
            infergo_sys::infer_rerank_pipeline(
                self.session,
                self.tokenizer,
                c_query.as_ptr(),
                c_ptrs.as_ptr(),
                n_docs,
                out_scores.as_mut_ptr(),
                out_indices.as_mut_ptr(),
                max_results,
            )
        };
        if n < 0 {
            return Err(Error::Api(last_error()));
        }
        out_scores.truncate(n as usize);
        out_indices.truncate(n as usize);
        Ok((out_scores, out_indices))
    }

    /// Get the raw session handle (for advanced use).
    pub fn session_handle(&self) -> infergo_sys::InferSession {
        self.session
    }

    /// Get the raw tokenizer handle (for advanced use).
    pub fn tokenizer_handle(&self) -> infergo_sys::InferTokenizer {
        self.tokenizer
    }
}

impl Drop for Embedding {
    fn drop(&mut self) {
        if !self.tokenizer.is_null() {
            unsafe { infergo_sys::infer_tokenizer_destroy(self.tokenizer) }
        }
        if !self.session.is_null() {
            unsafe { infergo_sys::infer_session_destroy(self.session) }
        }
    }
}

// ---------------------------------------------------------------------------
// VectorDB
// ---------------------------------------------------------------------------

/// Search result from a vector database query.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: i64,
    pub distance: f32,
}

/// RAII wrapper around the infergo persistent vector database (HNSW).
pub struct VectorDb {
    handle: infergo_sys::InferVectorDB,
    dim: i32,
}

unsafe impl Send for VectorDb {}
unsafe impl Sync for VectorDb {}

impl VectorDb {
    /// Create a new vector database.
    ///
    /// - `dim`: vector dimension (must match your embedding model output)
    /// - `m`: max connections per HNSW node (16 is a good default)
    /// - `ef_construction`: search width during index build (200 is typical)
    pub fn new(dim: i32, m: i32, ef_construction: i32) -> Result<Self> {
        let handle =
            unsafe { infergo_sys::infer_vectordb_create(dim, m, ef_construction) };
        if handle.is_null() {
            return Err(Error::NullHandle(last_error()));
        }
        Ok(Self { handle, dim })
    }

    /// Insert a vector with metadata.
    pub fn insert(&self, id: i64, vec: &[f32], metadata: Option<&str>) -> Result<()> {
        if vec.len() != self.dim as usize {
            return Err(Error::InvalidArg(format!(
                "vector length {} does not match db dimension {}",
                vec.len(),
                self.dim
            )));
        }
        let c_meta = match metadata {
            Some(m) => Some(CString::new(m)?),
            None => None,
        };
        let meta_ptr = c_meta.as_ref().map_or(ptr::null(), |m| m.as_ptr());
        let ret = unsafe {
            infergo_sys::infer_vectordb_insert(self.handle, id, vec.as_ptr(), meta_ptr)
        };
        if ret != 0 {
            return Err(Error::Api(last_error()));
        }
        Ok(())
    }

    /// Delete a vector by ID.
    pub fn delete(&self, id: i64) -> Result<()> {
        let ret = unsafe { infergo_sys::infer_vectordb_delete(self.handle, id) };
        if ret != 0 {
            return Err(Error::Api(last_error()));
        }
        Ok(())
    }

    /// Update a vector's data and metadata.
    pub fn update(&self, id: i64, vec: &[f32], metadata: Option<&str>) -> Result<()> {
        if vec.len() != self.dim as usize {
            return Err(Error::InvalidArg(format!(
                "vector length {} does not match db dimension {}",
                vec.len(),
                self.dim
            )));
        }
        let c_meta = match metadata {
            Some(m) => Some(CString::new(m)?),
            None => None,
        };
        let meta_ptr = c_meta.as_ref().map_or(ptr::null(), |m| m.as_ptr());
        let ret = unsafe {
            infergo_sys::infer_vectordb_update(self.handle, id, vec.as_ptr(), meta_ptr)
        };
        if ret != 0 {
            return Err(Error::Api(last_error()));
        }
        Ok(())
    }

    /// Get a vector and its metadata by ID.
    pub fn get(&self, id: i64) -> Result<(Vec<f32>, String)> {
        let mut out_vec = vec![0.0f32; self.dim as usize];
        let meta_buf_size: c_int = 4096;
        let mut out_meta = vec![0u8; meta_buf_size as usize];
        let ret = unsafe {
            infergo_sys::infer_vectordb_get(
                self.handle,
                id,
                out_vec.as_mut_ptr(),
                out_meta.as_mut_ptr() as *mut c_char,
                meta_buf_size,
            )
        };
        if ret != 0 {
            return Err(Error::Api(last_error()));
        }
        let meta = unsafe { CStr::from_ptr(out_meta.as_ptr() as *const c_char) };
        Ok((out_vec, meta.to_string_lossy().into_owned()))
    }

    /// Search for the k nearest neighbors.
    ///
    /// - `query`: query vector (must be `dim` floats)
    /// - `k`: number of results to return
    /// - `ef_search`: search width (higher = more accurate but slower)
    /// - `metadata_filter`: optional metadata filter expression
    pub fn search(
        &self,
        query: &[f32],
        k: i32,
        ef_search: i32,
        metadata_filter: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.dim as usize {
            return Err(Error::InvalidArg(format!(
                "query length {} does not match db dimension {}",
                query.len(),
                self.dim
            )));
        }
        let c_filter = match metadata_filter {
            Some(f) => Some(CString::new(f)?),
            None => None,
        };
        let filter_ptr = c_filter.as_ref().map_or(ptr::null(), |f| f.as_ptr());

        let mut out_ids = vec![0i64; k as usize];
        let mut out_distances = vec![0.0f32; k as usize];

        let n = unsafe {
            infergo_sys::infer_vectordb_search(
                self.handle,
                query.as_ptr(),
                k,
                ef_search,
                filter_ptr,
                out_ids.as_mut_ptr(),
                out_distances.as_mut_ptr(),
                k,
            )
        };
        if n < 0 {
            return Err(Error::Api(last_error()));
        }
        let results = (0..n as usize)
            .map(|i| SearchResult {
                id: out_ids[i],
                distance: out_distances[i],
            })
            .collect();
        Ok(results)
    }

    /// Save the database to disk.
    pub fn save(&self, path: &str) -> Result<()> {
        let c_path = CString::new(path)?;
        let ret = unsafe { infergo_sys::infer_vectordb_save(self.handle, c_path.as_ptr()) };
        if ret != 0 {
            return Err(Error::Api(last_error()));
        }
        Ok(())
    }

    /// Load the database from disk.
    pub fn load(&self, path: &str) -> Result<()> {
        let c_path = CString::new(path)?;
        let ret = unsafe { infergo_sys::infer_vectordb_load(self.handle, c_path.as_ptr()) };
        if ret != 0 {
            return Err(Error::Api(last_error()));
        }
        Ok(())
    }

    /// Number of vectors in the database.
    pub fn size(&self) -> i32 {
        unsafe { infergo_sys::infer_vectordb_size(self.handle) }
    }

    /// Get the raw C handle.
    pub fn as_raw(&self) -> infergo_sys::InferVectorDB {
        self.handle
    }
}

impl Drop for VectorDb {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { infergo_sys::infer_vectordb_free(self.handle) }
        }
    }
}

// ---------------------------------------------------------------------------
// Bm25
// ---------------------------------------------------------------------------

/// BM25 search result.
#[derive(Debug, Clone)]
pub struct Bm25Result {
    pub id: i64,
    pub score: f32,
}

/// RAII wrapper around the infergo BM25 full-text search index.
pub struct Bm25 {
    handle: infergo_sys::InferBM25,
}

unsafe impl Send for Bm25 {}
unsafe impl Sync for Bm25 {}

impl Bm25 {
    /// Create a new BM25 index.
    ///
    /// - `k1`: term frequency saturation (default 1.2)
    /// - `b`: document length normalization (default 0.75)
    pub fn new(k1: f32, b: f32) -> Result<Self> {
        let handle = unsafe { infergo_sys::infer_bm25_create(k1, b) };
        if handle.is_null() {
            return Err(Error::NullHandle(last_error()));
        }
        Ok(Self { handle })
    }

    /// Insert a document into the index.
    pub fn insert(&self, id: i64, text: &str) -> Result<()> {
        let c_text = CString::new(text)?;
        unsafe { infergo_sys::infer_bm25_insert(self.handle, id, c_text.as_ptr()) };
        Ok(())
    }

    /// Remove a document from the index by ID.
    pub fn remove(&self, id: i64) {
        unsafe { infergo_sys::infer_bm25_remove(self.handle, id) };
    }

    /// Search for documents matching a query.
    ///
    /// Returns up to `k` results sorted by BM25 score (descending).
    pub fn search(&self, query: &str, k: i32) -> Result<Vec<Bm25Result>> {
        let c_query = CString::new(query)?;
        let mut out_ids = vec![0i64; k as usize];
        let mut out_scores = vec![0.0f32; k as usize];
        let n = unsafe {
            infergo_sys::infer_bm25_search(
                self.handle,
                c_query.as_ptr(),
                k,
                out_ids.as_mut_ptr(),
                out_scores.as_mut_ptr(),
                k,
            )
        };
        if n < 0 {
            return Err(Error::Api(last_error()));
        }
        let results = (0..n as usize)
            .map(|i| Bm25Result {
                id: out_ids[i],
                score: out_scores[i],
            })
            .collect();
        Ok(results)
    }

    /// Number of documents in the index.
    pub fn size(&self) -> i32 {
        unsafe { infergo_sys::infer_bm25_size(self.handle) }
    }

    /// Save the index to disk.
    pub fn save(&self, path: &str) -> Result<()> {
        let c_path = CString::new(path)?;
        let ret = unsafe { infergo_sys::infer_bm25_save(self.handle, c_path.as_ptr()) };
        if ret != 0 {
            return Err(Error::Api(last_error()));
        }
        Ok(())
    }

    /// Load the index from disk.
    pub fn load(&self, path: &str) -> Result<()> {
        let c_path = CString::new(path)?;
        let ret = unsafe { infergo_sys::infer_bm25_load(self.handle, c_path.as_ptr()) };
        if ret != 0 {
            return Err(Error::Api(last_error()));
        }
        Ok(())
    }

    /// Get the raw C handle.
    pub fn as_raw(&self) -> infergo_sys::InferBM25 {
        self.handle
    }
}

impl Drop for Bm25 {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { infergo_sys::infer_bm25_free(self.handle) }
        }
    }
}

// ---------------------------------------------------------------------------
// Hybrid search (free function)
// ---------------------------------------------------------------------------

/// Fused hybrid search result.
#[derive(Debug, Clone)]
pub struct HybridResult {
    pub id: i64,
    pub score: f32,
}

/// Combine vector and BM25 search results using weighted score fusion.
///
/// - `alpha`: weight for vector scores (1.0 = pure vector, 0.0 = pure BM25)
/// - `k`: number of results to return
pub fn hybrid_search(
    vec_results: &[SearchResult],
    bm25_results: &[Bm25Result],
    alpha: f32,
    k: i32,
) -> Result<Vec<HybridResult>> {
    let vec_ids: Vec<i64> = vec_results.iter().map(|r| r.id).collect();
    let vec_distances: Vec<f32> = vec_results.iter().map(|r| r.distance).collect();
    let bm25_ids: Vec<i64> = bm25_results.iter().map(|r| r.id).collect();
    let bm25_scores: Vec<f32> = bm25_results.iter().map(|r| r.score).collect();

    let mut out_ids = vec![0i64; k as usize];
    let mut out_scores = vec![0.0f32; k as usize];

    let n = unsafe {
        infergo_sys::infer_hybrid_search(
            vec_ids.as_ptr(),
            vec_distances.as_ptr(),
            vec_results.len() as c_int,
            bm25_ids.as_ptr(),
            bm25_scores.as_ptr(),
            bm25_results.len() as c_int,
            alpha,
            k,
            out_ids.as_mut_ptr(),
            out_scores.as_mut_ptr(),
            k,
        )
    };
    if n < 0 {
        return Err(Error::Api(last_error()));
    }
    let results = (0..n as usize)
        .map(|i| HybridResult {
            id: out_ids[i],
            score: out_scores[i],
        })
        .collect();
    Ok(results)
}
