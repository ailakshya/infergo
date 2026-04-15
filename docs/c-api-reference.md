# C API Reference

Header: `cpp/include/infer_api.h`

All functions are `extern "C"` -- safe to call from C, Go (via CGo), or any language with a C FFI.

---

## Error handling

Every function that can fail returns `InferError` (an `int`). On success the value is `INFER_OK` (0). On failure, call `infer_last_error_string()` for a thread-local human-readable description.

```c
InferError err = infer_tensor_to_device(t, 0);
if (err != INFER_OK) {
    fprintf(stderr, "error: %s\n", infer_last_error_string());
}
```

### Error codes

| Constant | Value | Meaning |
|---|---|---|
| `INFER_OK` | 0 | Success |
| `INFER_ERR_NULL` | 1 | Null pointer argument |
| `INFER_ERR_INVALID` | 2 | Invalid argument value |
| `INFER_ERR_OOM` | 3 | Out of memory |
| `INFER_ERR_CUDA` | 4 | CUDA error (check last error string) |
| `INFER_ERR_LOAD` | 5 | Failed to load model or file |
| `INFER_ERR_RUNTIME` | 6 | Runtime inference error |
| `INFER_ERR_SHAPE` | 7 | Shape mismatch |
| `INFER_ERR_DTYPE` | 8 | Data type mismatch |
| `INFER_ERR_CANCELLED` | 9 | Operation cancelled |
| `INFER_ERR_UNKNOWN` | 99 | Unknown error |

```c
const char* infer_last_error_string(void);
```

Returns a pointer to a thread-local string. Valid until the next call on the same thread.

---

## Opaque handle types

```c
typedef void* InferTensor;
typedef void* InferSession;
typedef void* InferTorchSession;
typedef void* InferTokenizer;
typedef void* InferLLM;
typedef void* InferSeq;
typedef void* InferSampler;
typedef void* InferSpeculative;
typedef void* InferIndex;
typedef void* InferVectorDB;
typedef void* InferBM25;
typedef void* InferLoRA;
```

All handles are heap-allocated. Ownership rules:
- Functions named `*_create` / `*_alloc` / `*_load` allocate; you must call the matching `*_destroy` / `*_free`.
- Functions that return a handle without `create`/`alloc`/`load` in their name do **not** transfer ownership.

---

## Tensor API

### Allocation

```c
InferTensor infer_tensor_alloc_cpu(const int* shape, int ndim, int dtype);
InferTensor infer_tensor_alloc_cuda(const int* shape, int ndim, int dtype, int device_id);
```

Returns `NULL` on failure. No `InferError` -- check for NULL.

```c
void infer_tensor_free(InferTensor t);
```

Safe to call with NULL.

### Inspection

```c
void*  infer_tensor_data_ptr(InferTensor t);
int    infer_tensor_nbytes(InferTensor t);
int    infer_tensor_nelements(InferTensor t);
int    infer_tensor_shape(InferTensor t, int* out_shape, int max_dims);  // returns ndim
int    infer_tensor_dtype(InferTensor t);                                // returns DType
```

### Data transfer

```c
InferError infer_tensor_copy_from(InferTensor t, const void* src, int nbytes);
InferError infer_tensor_to_device(InferTensor t, int device_id);
InferError infer_tensor_to_host(InferTensor t);
```

### DType values

| Value | Constant | Type |
|---|---|---|
| 0 | `INFER_DTYPE_FLOAT32` | float32 |
| 1 | `INFER_DTYPE_FLOAT16` | float16 |
| 2 | `INFER_DTYPE_BFLOAT16` | bfloat16 |
| 3 | `INFER_DTYPE_INT32` | int32 |
| 4 | `INFER_DTYPE_INT64` | int64 |
| 5 | `INFER_DTYPE_UINT8` | uint8 |
| 6 | `INFER_DTYPE_BOOL` | bool |

---

## ONNX Session API

```c
InferSession infer_session_create(const char* provider, int device_id);
InferError infer_session_load(InferSession s, const char* model_path);
void infer_session_destroy(InferSession s);
```

Provider values: `"cpu"`, `"cuda"`, `"tensorrt"`, `"coreml"`, `"openvino"`. Falls back to CPU if the requested provider is unavailable.

### Input / output metadata

```c
int infer_session_num_inputs(InferSession s);
int infer_session_num_outputs(InferSession s);
InferError infer_session_input_name(InferSession s, int idx, char* out_buf, int buf_size);
InferError infer_session_output_name(InferSession s, int idx, char* out_buf, int buf_size);
```

### Running inference

```c
InferError infer_session_run(
    InferSession s,
    InferTensor* inputs,   int n_inputs,
    InferTensor* outputs,  int n_outputs
);
```

The callee allocates each output tensor. The caller must call `infer_tensor_free` on each.

---

## Torch (libtorch) Session API

Mirrors the ONNX session API but uses TorchScript (.pt) models powered by libtorch.

```c
InferTorchSession infer_torch_session_create(const char* provider, int device_id);
InferError infer_torch_session_load(InferTorchSession s, const char* model_path);
int infer_torch_session_num_inputs(InferTorchSession s);
int infer_torch_session_num_outputs(InferTorchSession s);
void infer_torch_session_destroy(InferTorchSession s);
```

### Standard run

```c
InferError infer_torch_session_run(
    InferTorchSession s,
    InferTensor* inputs,   int n_inputs,
    InferTensor* outputs,  int n_outputs
);
```

### GPU-optimized run

Non-blocking H2D upload, inference on GPU, D2H output copy:

```c
InferError infer_torch_session_run_gpu(
    InferTorchSession s,
    InferTensor* inputs,   int n_inputs,
    InferTensor* outputs,  int n_outputs
);
```

### GPU detection (JPEG in, boxes out)

Everything runs on GPU after JPEG decode. Only ~921KB uploaded, ~300B downloaded per image:

```c
int infer_torch_detect_gpu(
    InferTorchSession s,
    const void* jpeg_data, int nbytes,
    float conf_thresh, float iou_thresh,
    InferBox* out_boxes, int max_boxes
);
```

Returns number of detections, or -1 on error.

### Batch GPU detection

Process N JPEG images in one forward pass, amortizing CGo/C++ overhead:

```c
int infer_torch_detect_gpu_batch(
    InferTorchSession s,
    const void** jpeg_data_array, const int* nbytes_array, int batch_size,
    float conf_thresh, float iou_thresh,
    InferBox** out_boxes_array, int* out_counts, int max_boxes_per_image
);
```

Returns 0 on success, -1 on error.

### Raw RGB detection

Detect from raw RGB pixels without JPEG encode/decode overhead:

```c
int infer_torch_detect_gpu_raw(
    InferTorchSession s,
    const void* rgb_data, int width, int height,
    float conf_thresh, float iou_thresh,
    InferBox* out_boxes, int max_boxes
);
```

### YUV/NV12 detection

Detect from NV12 video frames with zero CPU color conversion (NV12->RGB on GPU):

```c
int infer_torch_detect_gpu_yuv(
    InferTorchSession s,
    const void* yuv_data, int width, int height, int linesize,
    float conf_thresh, float iou_thresh,
    InferBox* out_boxes, int max_boxes
);
```

---

## Tokenizer API

```c
InferTokenizer infer_tokenizer_load(const char* path);
void infer_tokenizer_destroy(InferTokenizer tok);
```

### Encoding

```c
int infer_tokenizer_encode(
    InferTokenizer tok, const char* text,
    int add_special_tokens,
    int* out_ids, int* out_attention_mask,
    int max_tokens
);
// Returns number of tokens, or -1 on error.
```

### Decoding

```c
int infer_tokenizer_decode(
    InferTokenizer tok, const int* ids, int n_ids,
    int skip_special_tokens, char* out_buf, int buf_size
);
// Returns 0 on success, or -1 on error.

int infer_tokenizer_decode_token(
    InferTokenizer tok, int id, char* out_buf, int buf_size
);

int infer_tokenizer_vocab_size(InferTokenizer tok);
```

---

## LLM Engine API

### Loading

```c
InferLLM infer_llm_create(
    const char* path, int n_gpu_layers, int ctx_size, int n_seq_max, int n_batch
);

// Multi-GPU tensor split
InferLLM infer_llm_create_split(
    const char* path, int n_gpu_layers, int ctx_size, int n_seq_max, int n_batch,
    const float* tensor_split, int n_split
);

// Pipeline parallelism
InferLLM infer_llm_create_pipeline(
    const char* path, int n_gpu_layers, int ctx_size, int n_seq_max, int n_batch,
    int n_stages
);

void infer_llm_destroy(InferLLM llm);
```

### Vocabulary

```c
int infer_llm_vocab_size(InferLLM llm);
int infer_llm_bos(InferLLM llm);
int infer_llm_eos(InferLLM llm);
int infer_llm_is_eog(InferLLM llm, int token);
```

### Tokenization

```c
int infer_llm_tokenize(InferLLM llm, const char* text, int add_bos,
                        int* out_ids, int max_tokens);
int infer_llm_token_to_piece(InferLLM llm, int token, char* out_buf, int buf_size);
```

### LoRA adapter management

```c
InferLoRA infer_lora_load(InferLLM llm, const char* lora_path);
int infer_lora_apply(InferLLM llm, InferLoRA* adapters, float* scales, int n_adapters);
void infer_lora_free(InferLoRA lora);
```

Load a LoRA adapter, apply one or more adapters with scaling factors, or pass NULL/0 to clear all adapters.

### Sequences

```c
InferSeq infer_seq_create(InferLLM llm, const int* tokens, int n_tokens);
void infer_seq_destroy(InferSeq seq);
int  infer_seq_is_done(InferSeq seq);
int  infer_seq_position(InferSeq seq);
int  infer_seq_slot_id(InferSeq seq);
void infer_seq_append_token(InferSeq seq, int token);
int  infer_seq_next_tokens(InferSeq seq, int* out_ids, int max_tokens);
```

### Batch decode

```c
InferError infer_llm_batch_decode(InferLLM llm, InferSeq* seqs, int n_seqs);
InferError infer_seq_get_logits(InferSeq seq, float* out_logits, int vocab_size);
```

---

## Full C generation loop (`infer_llm_generate`)

Run the complete generation loop in C++ -- one CGo call for the entire request. Prefill + decode + sample all happen natively with no per-token CGo overhead.

```c
typedef int (*InferTokenCallback)(int token, const char* piece, void* user_data);

int infer_llm_generate(
    InferLLM      llm,
    const int*    prompt_tokens, int n_prompt,
    int           max_tokens,
    float         temperature,
    float         top_p,
    const char*   grammar,       // GBNF grammar string (NULL = no constraint)
    InferTokenCallback callback, // called per token (NULL = silent)
    void*         user_data,
    char*         out_text,      // output buffer
    int           max_text_len,
    int*          out_gen_tokens
);
```

Returns 0 on success, -1 on error.

### Example

```c
InferLLM llm = infer_llm_create("model.gguf", 999, 4096, 1, 512);

int tokens[512];
int n = infer_llm_tokenize(llm, "Hello!", 1, tokens, 512);

char output[8192];
int gen_tokens;
infer_llm_generate(llm, tokens, n, 256, 0.7, 0.9, NULL, NULL, NULL,
                   output, sizeof(output), &gen_tokens);
printf("Generated: %s (%d tokens)\n", output, gen_tokens);

infer_llm_destroy(llm);
```

---

## Batch generation (`infer_llm_generate_batch`)

Generate text for N requests simultaneously using continuous batching. All sequences share one `llama_decode` call per step.

```c
int infer_llm_generate_batch(
    InferLLM      llm,
    int           n_requests,
    const int*    all_tokens,      // concatenated prompt tokens
    const int*    token_offsets,   // start offset per request (n_requests+1 entries)
    int           max_tokens,
    float         temperature,
    float         top_p,
    const char*   grammar,
    char**        out_texts,       // array of output buffers
    int           max_text_len,
    int*          out_gen_tokens   // array of token counts
);
```

Returns 0 on success, -1 on error.

---

## Grammar-constrained sampler API

```c
InferSampler infer_sampler_create(
    InferLLM llm, const char* grammar_str, const char* grammar_root,
    float temperature, float top_p, int top_k, uint32_t seed
);

int infer_sampler_sample(InferSampler smpl, const float* logits, int vocab_size);
int infer_sampler_sample_seq(InferSampler smpl, InferSeq seq);  // zero-copy fast path

void infer_sampler_free(InferSampler smpl);
```

Every token sampled through the sampler is forced to comply with the GBNF grammar. `infer_sampler_sample_seq` reads logits directly from the sequence's internal buffer without any data crossing the CGo boundary.

---

## TOON (Token-Oriented Object Notation)

A compact structured output format that uses fewer tokens than JSON.

```c
const char* infer_toon_grammar(void);
int infer_toon_to_json(const char* toon, int toon_len, char* out_json, int max_json_len);
int infer_json_to_toon(const char* json_str, int json_len, char* out_toon, int max_toon_len);
```

- `infer_toon_grammar()` returns the GBNF grammar for constraining LLM output to TOON format.
- `infer_toon_to_json()` converts TOON to standard JSON.
- `infer_json_to_toon()` converts JSON to TOON.

---

## Full C embedding pipeline

Run the complete embedding pipeline in C++: tokenize -> ONNX -> pool -> normalize. One CGo call replaces Go tokenize + tensor alloc + ONNX run + pool + L2 norm.

```c
int infer_embed_pipeline(
    InferSession session, InferTokenizer tokenizer,
    const char* text, float* out_vec, int max_dim
);

int infer_embed_batch_pipeline(
    InferSession session, InferTokenizer tokenizer,
    const char** texts, int n_texts,
    float* out_vecs, int max_dim
);
```

Returns the embedding dimension, or -1 on error.

---

## Full C rerank pipeline

Rerank documents by query relevance -- all in C++. Embeds query + all documents, computes cosine similarity, sorts by score.

```c
int infer_rerank_pipeline(
    InferSession session, InferTokenizer tokenizer,
    const char* query, const char** documents, int n_docs,
    float* out_scores, int* out_indices, int max_results
);
```

---

## BM25 full-text search API

```c
InferBM25 infer_bm25_create(float k1, float b);
void infer_bm25_insert(InferBM25 idx, int64_t id, const char* text);
void infer_bm25_remove(InferBM25 idx, int64_t id);
int  infer_bm25_search(InferBM25 idx, const char* query, int k,
                        int64_t* out_ids, float* out_scores, int max_results);
int  infer_bm25_size(InferBM25 idx);
int  infer_bm25_save(InferBM25 idx, const char* path);
int  infer_bm25_load(InferBM25 idx, const char* path);
void infer_bm25_free(InferBM25 idx);
```

Standard BM25 with configurable k1 (term frequency saturation) and b (document length normalization). Supports persistence via save/load.

---

## Hybrid search API

Combine vector search results and BM25 results using weighted score fusion.

```c
int infer_hybrid_search(
    const int64_t* vec_ids,   const float* vec_distances,  int n_vec,
    const int64_t* bm25_ids,  const float* bm25_scores,    int n_bm25,
    float alpha, int k,
    int64_t* out_ids, float* out_scores, int max_results
);
```

`alpha` controls the balance: 1.0 = pure vector, 0.0 = pure BM25, 0.5 = equal weight. Scores are normalized to [0,1] before combining.

---

## Vector database API (HNSW + persistence + CRUD + filtering)

```c
InferVectorDB infer_vectordb_create(int dim, int M, int ef_construction);

// CRUD operations
int infer_vectordb_insert(InferVectorDB db, int64_t id, const float* vec, const char* metadata);
int infer_vectordb_delete(InferVectorDB db, int64_t id);
int infer_vectordb_update(InferVectorDB db, int64_t id, const float* vec, const char* metadata);
int infer_vectordb_get(InferVectorDB db, int64_t id, float* out_vec, char* out_meta, int meta_buf_size);

// Search with optional metadata filter
int infer_vectordb_search(InferVectorDB db, const float* query, int k, int ef_search,
                           const char* metadata_filter,
                           int64_t* out_ids, float* out_distances, int max_results);

// Persistence
int infer_vectordb_save(InferVectorDB db, const char* path);
int infer_vectordb_load(InferVectorDB db, const char* path);

int  infer_vectordb_size(InferVectorDB db);
void infer_vectordb_free(InferVectorDB db);
```

---

## RAG pipeline (embed + search + generate)

Full RAG in C++: embed query -> search vector DB -> build context -> LLM generate. One CGo call for the entire pipeline.

```c
int infer_rag_pipeline(
    InferLLM llm, InferSession embed_session, InferTokenizer embed_tokenizer,
    InferVectorDB vector_db,
    const char* query, int k, int max_tokens, float temperature,
    char* out_text, int max_text_len
);
```

---

## Legacy vector search API (HNSW, no persistence)

```c
InferIndex infer_index_create(int dim, int M, int ef_construction);
int  infer_index_insert(InferIndex idx, int64_t id, const float* vec, const char* metadata);
int  infer_index_search(InferIndex idx, const float* query, int k, int ef_search,
                         int64_t* out_ids, float* out_distances, int max_results);
int  infer_index_size(InferIndex idx);
void infer_index_free(InferIndex idx);
```

---

## Speculative decoding API

```c
InferSpeculative infer_speculative_create(
    InferLLM target, const char* draft_path, int n_gpu_layers, int n_draft
);

int infer_speculative_generate(
    InferSpeculative spec,
    const int* prompt_tokens, int n_prompt,
    int max_tokens, float temperature,
    InferTokenCallback callback, void* user_data,
    char* out_text, int max_text_len,
    int* out_n_predict, int* out_n_drafted, int* out_n_accepted
);

void infer_speculative_free(InferSpeculative spec);
```

The draft model must share the same vocabulary as the target LLM. Drafts, verifies, accepts/rejects -- everything runs natively in one CGo call.

---

## GPU NMS (CUDA kernel)

```c
InferError infer_nms_cuda(
    const float* d_boxes, int n_boxes,
    float conf_thresh, float iou_thresh,
    InferBox* out_boxes, int max_out,
    int* out_count, void* stream
);
```

Runs the entire NMS pipeline on GPU: confidence filter, sort, IoU computation, and greedy suppression. Only the final kept detections are copied to host.

Parameters:
- `d_boxes` -- device pointer to N detections, each 6 floats: `[x1, y1, x2, y2, confidence, class_id]`
- `stream` -- CUDA stream handle (NULL for default stream)

Only available when built with CUDA support.

---

## KV cache serialization API

```c
// Two-call protocol: call with out_buf=NULL to query size, then with buffer
int infer_llm_kv_serialize(InferLLM llm, int seq_id, uint8_t* out_buf, int out_buf_size);
int infer_llm_kv_deserialize(InferLLM llm, int seq_id, const uint8_t* data, int nbytes);
```

### KV page metrics

```c
int infer_llm_kv_pages_free(InferLLM llm);
int infer_llm_kv_pages_total(InferLLM llm);
int infer_llm_kv_page_size(InferLLM llm);
```

---

## Preprocessing API

Requires OpenCV at build time.

```c
InferTensor infer_preprocess_decode_image(const void* data, int nbytes);
InferTensor infer_preprocess_letterbox(InferTensor src, int target_w, int target_h);
InferTensor infer_preprocess_normalize(InferTensor src, float scale,
                                        const float* mean, const float* std);
InferTensor infer_preprocess_stack_batch(const InferTensor* tensors, int n);
```

All returned tensors are newly allocated; caller must `infer_tensor_free` them.

---

## Postprocessing API

### Classification

```c
typedef struct {
    int   label_idx;
    float confidence;
} InferClassResult;

int infer_postprocess_classify(InferTensor logits, int top_k,
                               InferClassResult* out_results);
```

### Object detection (NMS)

```c
typedef struct InferBox {
    float x1, y1, x2, y2;
    int   class_idx;
    float confidence;
} InferBox;

int infer_postprocess_nms(InferTensor predictions,
                          float conf_thresh, float iou_thresh,
                          InferBox* out_boxes, int max_boxes);
```

### Embedding normalization

```c
InferError infer_postprocess_normalize_embedding(InferTensor t);
```

L2-normalizes the tensor in-place.
