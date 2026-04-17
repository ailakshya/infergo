//! Raw FFI bindings to libinfer_api (infergo C API).
//!
//! This crate provides unsafe `extern "C"` declarations that map directly to
//! the functions declared in `infer_api.h`.  Users should prefer the safe
//! wrappers in the `infergo` crate.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_float, c_int, c_void};

// ---------------------------------------------------------------------------
// Opaque handle types
// ---------------------------------------------------------------------------

pub type InferLLM = *mut c_void;
pub type InferSession = *mut c_void;
pub type InferTokenizer = *mut c_void;
pub type InferVectorDB = *mut c_void;
pub type InferBM25 = *mut c_void;
pub type InferLoRA = *mut c_void;
pub type InferSeq = *mut c_void;
pub type InferTensor = *mut c_void;
pub type InferIndex = *mut c_void;
pub type InferSampler = *mut c_void;
pub type InferSpeculative = *mut c_void;

// ---------------------------------------------------------------------------
// Error codes
// ---------------------------------------------------------------------------

pub type InferError = c_int;

pub const INFER_OK: InferError = 0;
pub const INFER_ERR_NULL: InferError = 1;
pub const INFER_ERR_INVALID: InferError = 2;
pub const INFER_ERR_OOM: InferError = 3;
pub const INFER_ERR_CUDA: InferError = 4;
pub const INFER_ERR_LOAD: InferError = 5;
pub const INFER_ERR_RUNTIME: InferError = 6;
pub const INFER_ERR_SHAPE: InferError = 7;
pub const INFER_ERR_DTYPE: InferError = 8;
pub const INFER_ERR_CANCELLED: InferError = 9;
pub const INFER_ERR_UNKNOWN: InferError = 99;

// ---------------------------------------------------------------------------
// Dtype constants
// ---------------------------------------------------------------------------

pub const INFER_DTYPE_FLOAT32: c_int = 0;
pub const INFER_DTYPE_FLOAT16: c_int = 1;
pub const INFER_DTYPE_BFLOAT16: c_int = 2;
pub const INFER_DTYPE_INT32: c_int = 3;
pub const INFER_DTYPE_INT64: c_int = 4;
pub const INFER_DTYPE_UINT8: c_int = 5;
pub const INFER_DTYPE_BOOL: c_int = 6;

// ---------------------------------------------------------------------------
// Callback type
// ---------------------------------------------------------------------------

/// Token callback for streaming generation.
/// Return 1 to continue, 0 to stop.
pub type InferTokenCallback =
    Option<unsafe extern "C" fn(token: c_int, piece: *const c_char, user_data: *mut c_void) -> c_int>;

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct InferClassResult {
    pub label_idx: c_int,
    pub confidence: c_float,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct InferBox {
    pub x1: c_float,
    pub y1: c_float,
    pub x2: c_float,
    pub y2: c_float,
    pub class_idx: c_int,
    pub confidence: c_float,
}

// ---------------------------------------------------------------------------
// FFI declarations
// ---------------------------------------------------------------------------

extern "C" {
    // -- Error ---------------------------------------------------------------

    pub fn infer_last_error_string() -> *const c_char;

    // -- LLM -----------------------------------------------------------------

    pub fn infer_llm_create(
        path: *const c_char,
        n_gpu_layers: c_int,
        ctx_size: c_int,
        n_seq_max: c_int,
        n_batch: c_int,
    ) -> InferLLM;

    pub fn infer_llm_create_split(
        path: *const c_char,
        n_gpu_layers: c_int,
        ctx_size: c_int,
        n_seq_max: c_int,
        n_batch: c_int,
        tensor_split: *const c_float,
        n_split: c_int,
    ) -> InferLLM;

    pub fn infer_llm_create_pipeline(
        path: *const c_char,
        n_gpu_layers: c_int,
        ctx_size: c_int,
        n_seq_max: c_int,
        n_batch: c_int,
        n_stages: c_int,
    ) -> InferLLM;

    pub fn infer_llm_destroy(llm: InferLLM);

    pub fn infer_llm_vocab_size(llm: InferLLM) -> c_int;
    pub fn infer_llm_bos(llm: InferLLM) -> c_int;
    pub fn infer_llm_eos(llm: InferLLM) -> c_int;
    pub fn infer_llm_is_eog(llm: InferLLM, token: c_int) -> c_int;

    pub fn infer_llm_tokenize(
        llm: InferLLM,
        text: *const c_char,
        add_bos: c_int,
        out_ids: *mut c_int,
        max_tokens: c_int,
    ) -> c_int;

    pub fn infer_llm_token_to_piece(
        llm: InferLLM,
        token: c_int,
        out_buf: *mut c_char,
        buf_size: c_int,
    ) -> c_int;

    pub fn infer_llm_generate(
        llm: InferLLM,
        prompt_tokens: *const c_int,
        n_prompt: c_int,
        max_tokens: c_int,
        temperature: c_float,
        top_p: c_float,
        grammar: *const c_char,
        callback: InferTokenCallback,
        user_data: *mut c_void,
        out_text: *mut c_char,
        max_text_len: c_int,
        out_gen_tokens: *mut c_int,
    ) -> c_int;

    pub fn infer_llm_generate_batch(
        llm: InferLLM,
        n_requests: c_int,
        all_tokens: *const c_int,
        token_offsets: *const c_int,
        max_tokens: c_int,
        temperature: c_float,
        top_p: c_float,
        grammar: *const c_char,
        out_texts: *mut *mut c_char,
        max_text_len: c_int,
        out_gen_tokens: *mut c_int,
    ) -> c_int;

    pub fn infer_llm_batch_decode(
        llm: InferLLM,
        seqs: *mut InferSeq,
        n_seqs: c_int,
    ) -> InferError;

    // -- LoRA ----------------------------------------------------------------

    pub fn infer_lora_load(llm: InferLLM, lora_path: *const c_char) -> InferLoRA;
    pub fn infer_lora_apply(
        llm: InferLLM,
        adapters: *mut InferLoRA,
        scales: *mut c_float,
        n_adapters: c_int,
    ) -> c_int;
    pub fn infer_lora_free(lora: InferLoRA);

    // -- Sequence -------------------------------------------------------------

    pub fn infer_seq_create(
        llm: InferLLM,
        tokens: *const c_int,
        n_tokens: c_int,
    ) -> InferSeq;
    pub fn infer_seq_destroy(seq: InferSeq);
    pub fn infer_seq_is_done(seq: InferSeq) -> c_int;
    pub fn infer_seq_position(seq: InferSeq) -> c_int;
    pub fn infer_seq_slot_id(seq: InferSeq) -> c_int;
    pub fn infer_seq_append_token(seq: InferSeq, token: c_int);
    pub fn infer_seq_next_tokens(
        seq: InferSeq,
        out_ids: *mut c_int,
        max_tokens: c_int,
    ) -> c_int;
    pub fn infer_seq_get_logits(
        seq: InferSeq,
        out_logits: *mut c_float,
        vocab_size: c_int,
    ) -> InferError;

    // -- Sampler --------------------------------------------------------------

    pub fn infer_sampler_create(
        llm: InferLLM,
        grammar_str: *const c_char,
        grammar_root: *const c_char,
        temperature: c_float,
        top_p: c_float,
        top_k: c_int,
        seed: u32,
    ) -> InferSampler;
    pub fn infer_sampler_sample(
        smpl: InferSampler,
        logits: *const c_float,
        vocab_size: c_int,
    ) -> c_int;
    pub fn infer_sampler_sample_seq(smpl: InferSampler, seq: InferSeq) -> c_int;
    pub fn infer_sampler_free(smpl: InferSampler);

    // -- Speculative decoding -------------------------------------------------

    pub fn infer_speculative_create(
        target: InferLLM,
        draft_path: *const c_char,
        n_gpu_layers: c_int,
        n_draft: c_int,
    ) -> InferSpeculative;

    pub fn infer_speculative_generate(
        spec: InferSpeculative,
        prompt_tokens: *const c_int,
        n_prompt: c_int,
        max_tokens: c_int,
        temperature: c_float,
        callback: InferTokenCallback,
        user_data: *mut c_void,
        out_text: *mut c_char,
        max_text_len: c_int,
        out_n_predict: *mut c_int,
        out_n_drafted: *mut c_int,
        out_n_accepted: *mut c_int,
    ) -> c_int;

    pub fn infer_speculative_free(spec: InferSpeculative);

    // -- ONNX Session --------------------------------------------------------

    pub fn infer_session_create(
        provider: *const c_char,
        device_id: c_int,
    ) -> InferSession;
    pub fn infer_session_load(s: InferSession, model_path: *const c_char) -> InferError;
    pub fn infer_session_num_inputs(s: InferSession) -> c_int;
    pub fn infer_session_num_outputs(s: InferSession) -> c_int;
    pub fn infer_session_input_name(
        s: InferSession,
        idx: c_int,
        out_buf: *mut c_char,
        buf_size: c_int,
    ) -> InferError;
    pub fn infer_session_output_name(
        s: InferSession,
        idx: c_int,
        out_buf: *mut c_char,
        buf_size: c_int,
    ) -> InferError;
    pub fn infer_session_run(
        s: InferSession,
        inputs: *mut InferTensor,
        n_inputs: c_int,
        outputs: *mut InferTensor,
        n_outputs: c_int,
    ) -> InferError;
    pub fn infer_session_destroy(s: InferSession);

    // -- Tokenizer -----------------------------------------------------------

    pub fn infer_tokenizer_load(path: *const c_char) -> InferTokenizer;
    pub fn infer_tokenizer_encode(
        tok: InferTokenizer,
        text: *const c_char,
        add_special_tokens: c_int,
        out_ids: *mut c_int,
        out_mask: *mut c_int,
        max_tokens: c_int,
    ) -> c_int;
    pub fn infer_tokenizer_decode(
        tok: InferTokenizer,
        ids: *const c_int,
        n_ids: c_int,
        skip_special_tokens: c_int,
        out_buf: *mut c_char,
        buf_size: c_int,
    ) -> c_int;
    pub fn infer_tokenizer_decode_token(
        tok: InferTokenizer,
        id: c_int,
        out_buf: *mut c_char,
        buf_size: c_int,
    ) -> c_int;
    pub fn infer_tokenizer_vocab_size(tok: InferTokenizer) -> c_int;
    pub fn infer_tokenizer_destroy(tok: InferTokenizer);

    // -- Tensor ---------------------------------------------------------------

    pub fn infer_tensor_alloc_cpu(
        shape: *const c_int,
        ndim: c_int,
        dtype: c_int,
    ) -> InferTensor;
    pub fn infer_tensor_alloc_cuda(
        shape: *const c_int,
        ndim: c_int,
        dtype: c_int,
        device_id: c_int,
    ) -> InferTensor;
    pub fn infer_tensor_free(t: InferTensor);
    pub fn infer_tensor_data_ptr(t: InferTensor) -> *mut c_void;
    pub fn infer_tensor_nbytes(t: InferTensor) -> c_int;
    pub fn infer_tensor_nelements(t: InferTensor) -> c_int;
    pub fn infer_tensor_shape(
        t: InferTensor,
        out_shape: *mut c_int,
        max_dims: c_int,
    ) -> c_int;
    pub fn infer_tensor_dtype(t: InferTensor) -> c_int;
    pub fn infer_tensor_to_device(t: InferTensor, device_id: c_int) -> InferError;
    pub fn infer_tensor_to_host(t: InferTensor) -> InferError;
    pub fn infer_tensor_copy_from(
        t: InferTensor,
        src: *const c_void,
        nbytes: c_int,
    ) -> InferError;

    // -- Embedding pipeline ---------------------------------------------------

    pub fn infer_embed_pipeline(
        session: InferSession,
        tokenizer: InferTokenizer,
        text: *const c_char,
        out_vec: *mut c_float,
        max_dim: c_int,
    ) -> c_int;

    pub fn infer_embed_batch_pipeline(
        session: InferSession,
        tokenizer: InferTokenizer,
        texts: *const *const c_char,
        n_texts: c_int,
        out_vecs: *mut c_float,
        max_dim: c_int,
    ) -> c_int;

    // -- Rerank pipeline ------------------------------------------------------

    pub fn infer_rerank_pipeline(
        session: InferSession,
        tokenizer: InferTokenizer,
        query: *const c_char,
        documents: *const *const c_char,
        n_docs: c_int,
        out_scores: *mut c_float,
        out_indices: *mut c_int,
        max_results: c_int,
    ) -> c_int;

    // -- VectorDB -------------------------------------------------------------

    pub fn infer_vectordb_create(
        dim: c_int,
        m: c_int,
        ef_construction: c_int,
    ) -> InferVectorDB;
    pub fn infer_vectordb_insert(
        db: InferVectorDB,
        id: i64,
        vec: *const c_float,
        metadata: *const c_char,
    ) -> c_int;
    pub fn infer_vectordb_delete(db: InferVectorDB, id: i64) -> c_int;
    pub fn infer_vectordb_update(
        db: InferVectorDB,
        id: i64,
        vec: *const c_float,
        metadata: *const c_char,
    ) -> c_int;
    pub fn infer_vectordb_get(
        db: InferVectorDB,
        id: i64,
        out_vec: *mut c_float,
        out_meta: *mut c_char,
        meta_buf_size: c_int,
    ) -> c_int;
    pub fn infer_vectordb_search(
        db: InferVectorDB,
        query: *const c_float,
        k: c_int,
        ef_search: c_int,
        metadata_filter: *const c_char,
        out_ids: *mut i64,
        out_distances: *mut c_float,
        max_results: c_int,
    ) -> c_int;
    pub fn infer_vectordb_save(db: InferVectorDB, path: *const c_char) -> c_int;
    pub fn infer_vectordb_load(db: InferVectorDB, path: *const c_char) -> c_int;
    pub fn infer_vectordb_size(db: InferVectorDB) -> c_int;
    pub fn infer_vectordb_free(db: InferVectorDB);

    // -- BM25 -----------------------------------------------------------------

    pub fn infer_bm25_create(k1: c_float, b: c_float) -> InferBM25;
    pub fn infer_bm25_insert(idx: InferBM25, id: i64, text: *const c_char);
    pub fn infer_bm25_remove(idx: InferBM25, id: i64);
    pub fn infer_bm25_search(
        idx: InferBM25,
        query: *const c_char,
        k: c_int,
        out_ids: *mut i64,
        out_scores: *mut c_float,
        max_results: c_int,
    ) -> c_int;
    pub fn infer_bm25_size(idx: InferBM25) -> c_int;
    pub fn infer_bm25_save(idx: InferBM25, path: *const c_char) -> c_int;
    pub fn infer_bm25_load(idx: InferBM25, path: *const c_char) -> c_int;
    pub fn infer_bm25_free(idx: InferBM25);

    // -- Hybrid search --------------------------------------------------------

    pub fn infer_hybrid_search(
        vec_ids: *const i64,
        vec_distances: *const c_float,
        n_vec: c_int,
        bm25_ids: *const i64,
        bm25_scores: *const c_float,
        n_bm25: c_int,
        alpha: c_float,
        k: c_int,
        out_ids: *mut i64,
        out_scores: *mut c_float,
        max_results: c_int,
    ) -> c_int;

    // -- RAG pipeline ---------------------------------------------------------

    pub fn infer_rag_pipeline(
        llm: InferLLM,
        embed_session: InferSession,
        embed_tokenizer: InferTokenizer,
        vector_db: InferVectorDB,
        query: *const c_char,
        k: c_int,
        max_tokens: c_int,
        temperature: c_float,
        out_text: *mut c_char,
        max_text_len: c_int,
    ) -> c_int;

    // -- TOON -----------------------------------------------------------------

    pub fn infer_toon_grammar() -> *const c_char;
    pub fn infer_toon_to_json(
        toon: *const c_char,
        toon_len: c_int,
        out_json: *mut c_char,
        max_json_len: c_int,
    ) -> c_int;
    pub fn infer_json_to_toon(
        json_str: *const c_char,
        json_len: c_int,
        out_toon: *mut c_char,
        max_toon_len: c_int,
    ) -> c_int;
}
