// cpp/include/infer_api.h
// THE ONLY HEADER THAT GO EVER INCLUDES
// All functions are C-compatible — no C++ types cross this boundary

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// ─────────────────────────────────────────────────────────────────────────────
// ERROR CODES
// ─────────────────────────────────────────────────────────────────────────────

typedef int InferError;

#define INFER_OK               0
#define INFER_ERR_NULL         1   // null pointer passed
#define INFER_ERR_INVALID      2   // invalid argument
#define INFER_ERR_OOM          3   // out of memory
#define INFER_ERR_CUDA         4   // CUDA error
#define INFER_ERR_LOAD         5   // model/file load failure
#define INFER_ERR_RUNTIME      6   // inference runtime error
#define INFER_ERR_SHAPE        7   // tensor shape mismatch
#define INFER_ERR_DTYPE        8   // unsupported dtype
#define INFER_ERR_CANCELLED    9   // operation cancelled
#define INFER_ERR_UNKNOWN      99  // unknown error

// Get human-readable error string for last error on this thread.
// Returns a pointer to a thread-local buffer — valid until the next API call
// on this thread. Never free this pointer.
const char* infer_last_error_string(void);

// ─────────────────────────────────────────────────────────────────────────────
// DTYPE CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────

#define INFER_DTYPE_FLOAT32   0
#define INFER_DTYPE_FLOAT16   1
#define INFER_DTYPE_BFLOAT16  2
#define INFER_DTYPE_INT32     3
#define INFER_DTYPE_INT64     4
#define INFER_DTYPE_UINT8     5
#define INFER_DTYPE_BOOL      6

// ─────────────────────────────────────────────────────────────────────────────
// TENSOR API
// ─────────────────────────────────────────────────────────────────────────────

typedef void* InferTensor;

// Allocate tensor on CPU heap.
// Returns NULL on failure; call infer_last_error_string() for details.
InferTensor infer_tensor_alloc_cpu(const int* shape, int ndim, int dtype);

// Allocate tensor on CUDA device (device_id = 0 for first GPU).
// Returns NULL on failure; call infer_last_error_string() for details.
InferTensor infer_tensor_alloc_cuda(const int* shape, int ndim, int dtype, int device_id);

// Free tensor memory (works for both CPU and CUDA tensors).
// Safe to call with NULL.
void infer_tensor_free(InferTensor t);

// Get raw data pointer.
// CPU tensors: pointer to heap memory — safe to read/write from Go.
// CUDA tensors: device pointer — do NOT dereference from Go.
// Returns NULL if t is NULL.
void* infer_tensor_data_ptr(InferTensor t);

// Get tensor size in bytes. Returns 0 if t is NULL.
int infer_tensor_nbytes(InferTensor t);

// Get number of elements. Returns 0 if t is NULL.
int infer_tensor_nelements(InferTensor t);

// Get shape (writes up to max_dims integers into out_shape, returns ndim).
// Returns 0 if t or out_shape is NULL, or max_dims <= 0.
int infer_tensor_shape(InferTensor t, int* out_shape, int max_dims);

// Get dtype constant (INFER_DTYPE_*). Returns -1 if t is NULL.
int infer_tensor_dtype(InferTensor t);

// Copy CPU tensor data to CUDA device.
// No-op if already on the requested device.
InferError infer_tensor_to_device(InferTensor t, int device_id);

// Copy CUDA tensor data back to CPU.
// No-op if already on host.
InferError infer_tensor_to_host(InferTensor t);

// Copy nbytes from src (host pointer) into a CPU tensor's data buffer.
// src must point to at least nbytes bytes. nbytes must equal the tensor's nbytes.
InferError infer_tensor_copy_from(InferTensor t, const void* src, int nbytes);

// ─────────────────────────────────────────────────────────────────────────────
// ONNX INFERENCE SESSION API
// ─────────────────────────────────────────────────────────────────────────────

typedef void* InferSession;

// Create a session for the given execution provider.
// provider: "cpu" | "cuda" | "tensorrt" | "coreml" | "openvino"
// Falls back to CPU if the requested provider is unavailable.
// Returns NULL on failure; call infer_last_error_string() for details.
InferSession infer_session_create(const char* provider, int device_id);

// Load an ONNX model file into the session.
InferError infer_session_load(InferSession s, const char* model_path);

// Get number of model inputs (valid after infer_session_load).
int infer_session_num_inputs(InferSession s);

// Get number of model outputs (valid after infer_session_load).
int infer_session_num_outputs(InferSession s);

// Write the input name at index idx into out_buf (max buf_size bytes, null-terminated).
InferError infer_session_input_name(InferSession s, int idx, char* out_buf, int buf_size);

// Write the output name at index idx into out_buf (max buf_size bytes, null-terminated).
InferError infer_session_output_name(InferSession s, int idx, char* out_buf, int buf_size);

// Run inference.
// inputs:   array of n_inputs InferTensor values (CPU tensors only).
// outputs:  array of n_outputs InferTensor pointers to write results into.
//           Each output tensor is heap-allocated; caller must call infer_tensor_free().
InferError infer_session_run(
    InferSession   s,
    InferTensor*   inputs,   int n_inputs,
    InferTensor*   outputs,  int n_outputs
);

// Destroy session and free all resources. Safe to call with NULL.
void infer_session_destroy(InferSession s);

// ─────────────────────────────────────────────────────────────────────────────
// TOKENIZER API
// ─────────────────────────────────────────────────────────────────────────────

typedef void* InferTokenizer;

// Load a HuggingFace tokenizer from a tokenizer.json file.
// Returns NULL on failure; call infer_last_error_string() for details.
InferTokenizer infer_tokenizer_load(const char* path);

// Encode text into token IDs and attention mask.
// add_special_tokens: 1 = prepend/append BOS/EOS, 0 = raw tokens only.
// max_tokens: hard cap on output length.
// out_ids / out_mask: caller-allocated arrays of length max_tokens.
// Returns number of tokens written, or -1 on error.
int infer_tokenizer_encode(
    InferTokenizer  tok,
    const char*     text,
    int             add_special_tokens,
    int*            out_ids,
    int*            out_mask,
    int             max_tokens
);

// Decode token IDs back to text.
// skip_special_tokens: 1 = omit BOS/EOS/PAD in output.
// out_buf: caller-allocated buffer of buf_size bytes (null-terminated on success).
// Returns 0 on success, -1 on error.
int infer_tokenizer_decode(
    InferTokenizer  tok,
    const int*      ids,
    int             n_ids,
    int             skip_special_tokens,
    char*           out_buf,
    int             buf_size
);

// Decode a single token ID to its string piece (null-terminated in out_buf).
// Returns 0 on success, -1 on error.
int infer_tokenizer_decode_token(
    InferTokenizer  tok,
    int             id,
    char*           out_buf,
    int             buf_size
);

// Returns the vocabulary size, or 0 on error.
int infer_tokenizer_vocab_size(InferTokenizer tok);

// Destroy tokenizer and free all resources. Safe to call with NULL.
void infer_tokenizer_destroy(InferTokenizer tok);

// ─────────────────────────────────────────────────────────────────────────────
// LLM ENGINE API
// ─────────────────────────────────────────────────────────────────────────────

typedef void* InferLLM;
typedef void* InferSeq;

// Load a GGUF model and create an LLM engine.
// n_gpu_layers: transformer layers to offload to GPU (large value = all).
// ctx_size:     total KV cache token budget across all sequences.
// n_seq_max:    max number of concurrent sequences.
// n_batch:      max tokens per decode call.
// Returns NULL on failure; call infer_last_error_string() for details.
InferLLM infer_llm_create(const char* path,
                           int         n_gpu_layers,
                           int         ctx_size,
                           int         n_seq_max,
                           int         n_batch);

// Create LLM with tensor split across multiple GPUs.
// tensor_split: array of n_split floats summing to 1.0; NULL means single GPU.
// n_split: number of GPUs (length of tensor_split array); 0 means single GPU.
InferLLM infer_llm_create_split(const char* path,
                                 int         n_gpu_layers,
                                 int         ctx_size,
                                 int         n_seq_max,
                                 int         n_batch,
                                 const float* tensor_split,
                                 int          n_split);

// Create LLM with pipeline parallelism across n_stages GPUs.
// Layers are distributed evenly using LLAMA_SPLIT_MODE_LAYER.
// n_stages=1 is identical to infer_llm_create (single GPU).
InferLLM infer_llm_create_pipeline(const char* path,
                                    int         n_gpu_layers,
                                    int         ctx_size,
                                    int         n_seq_max,
                                    int         n_batch,
                                    int         n_stages);

// Destroy LLM engine. Safe to call with NULL.
void infer_llm_destroy(InferLLM llm);

// Returns the vocabulary size (valid after infer_llm_create).
int infer_llm_vocab_size(InferLLM llm);

// Returns BOS token ID.
int infer_llm_bos(InferLLM llm);

// Returns EOS token ID.
int infer_llm_eos(InferLLM llm);

// Returns 1 if token is an end-of-generation token (EOS/EOT), 0 otherwise.
int infer_llm_is_eog(InferLLM llm, int token);

// Tokenize text using the model's built-in vocabulary.
// add_bos: 1 = prepend BOS token.
// out_ids: caller-allocated array of max_tokens ints.
// Returns number of tokens written, or -1 on error.
int infer_llm_tokenize(InferLLM llm, const char* text, int add_bos,
                        int* out_ids, int max_tokens);

// Convert a single token ID to its string piece (null-terminated in out_buf).
// Returns 0 on success, -1 on error.
int infer_llm_token_to_piece(InferLLM llm, int token, char* out_buf, int buf_size);

// Create a new sequence with the given prompt tokens.
// Allocates a KV cache slot; returns NULL if pool is full or llm is NULL.
InferSeq infer_seq_create(InferLLM llm, const int* tokens, int n_tokens);

// Destroy a sequence and release its KV cache slot. Safe to call with NULL.
void infer_seq_destroy(InferSeq seq);

// Returns 1 if the sequence has generated an end-of-generation token, 0 otherwise.
int infer_seq_is_done(InferSeq seq);

// Returns the current KV cache position (number of tokens already decoded).
int infer_seq_position(InferSeq seq);

// Returns the KV slot ID (= llama_seq_id for this sequence).
int infer_seq_slot_id(InferSeq seq);

// Append a sampled token to the sequence history and advance the KV position.
void infer_seq_append_token(InferSeq seq, int token);

// Write the tokens that need to be decoded in the next batch into out_ids.
// Returns the number of tokens written, or -1 on error.
int infer_seq_next_tokens(InferSeq seq, int* out_ids, int max_tokens);

// Batch decode: run all n_seqs sequences through one llama_decode call.
// Logits for each sequence are stored internally; retrieve with infer_seq_get_logits.
// Returns INFER_OK on success.
InferError infer_llm_batch_decode(InferLLM llm, InferSeq* seqs, int n_seqs);

// Copy the logits from the last batch_decode into out_logits[0..vocab_size).
// out_logits must be caller-allocated with at least vocab_size floats.
// Returns INFER_OK on success, or INFER_ERR_INVALID if no logits are available.
InferError infer_seq_get_logits(InferSeq seq, float* out_logits, int vocab_size);

// ─────────────────────────────────────────────────────────────────────────────
// PREPROCESSING API
// ─────────────────────────────────────────────────────────────────────────────

// Decode raw image bytes (JPEG / PNG / WebP / BMP) into a CPU tensor.
// Output shape: [H, W, 3], dtype float32, pixel values in [0, 255].
// Returns NULL on failure; call infer_last_error_string() for details.
// Caller owns the returned tensor — call infer_tensor_free() when done.
InferTensor infer_preprocess_decode_image(const void* data, int nbytes);

// Letterbox-resize a [H, W, 3] float32 tensor to exactly [target_h, target_w, 3].
// Scales uniformly (preserving aspect ratio) then pads remaining space with 114.0.
// Returns NULL on failure; call infer_last_error_string() for details.
// Caller owns the returned tensor — call infer_tensor_free() when done.
InferTensor infer_preprocess_letterbox(InferTensor src, int target_w, int target_h);

// Normalize a [H, W, 3] float32 tensor to CHW layout.
// Each channel c: out[c,h,w] = (in[h,w,c] / scale - mean[c]) / std[c]
// mean and std must each point to exactly 3 floats.
// Returns NULL on failure; call infer_last_error_string() for details.
// Caller owns the returned tensor — call infer_tensor_free() when done.
InferTensor infer_preprocess_normalize(InferTensor src, float scale,
                                        const float* mean, const float* std);

// Stack n [C, H, W] float32 tensors into a single [N, C, H, W] batch tensor.
// All tensors must have identical shape and dtype.
// tensors: array of n InferTensor values.
// Returns NULL on failure; call infer_last_error_string() for details.
// Caller owns the returned tensor — call infer_tensor_free() when done.
InferTensor infer_preprocess_stack_batch(const InferTensor* tensors, int n);

// ─────────────────────────────────────────────────────────────────────────────
// POSTPROCESSING API
// ─────────────────────────────────────────────────────────────────────────────

// Classification result: one (label index, confidence) pair.
typedef struct {
    int   label_idx;
    float confidence;
} InferClassResult;

// Bounding box (absolute pixel coordinates, top-left / bottom-right).
typedef struct {
    float x1, y1;   // top-left corner
    float x2, y2;   // bottom-right corner
    int   class_idx;
    float confidence;
} InferBox;

// Compute softmax over a 1-D float32 logits tensor and return the top_k entries
// sorted by confidence descending into out_results[0..top_k).
// out_results must be caller-allocated with at least top_k elements.
// Returns the number of results written (min(top_k, n_classes)), or -1 on error.
int infer_postprocess_classify(InferTensor logits, int top_k,
                               InferClassResult* out_results);

// Run NMS on a YOLO output tensor of shape [1, num_detections, 4+num_classes].
// Each detection row: [cx, cy, w, h, class0_score, class1_score, ...].
// conf_thresh: minimum class score to keep a detection.
// iou_thresh:  IoU threshold above which a box is suppressed.
// out_boxes:   caller-allocated array of at least max_boxes elements.
// Returns the number of boxes written, or -1 on error.
int infer_postprocess_nms(InferTensor predictions,
                          float conf_thresh, float iou_thresh,
                          InferBox* out_boxes, int max_boxes);

// L2-normalize a float32 tensor in-place: divide each element by sqrt(sum of squares).
// No-op if the L2 norm is zero. Returns INFER_OK on success, or -1 on error.
InferError infer_postprocess_normalize_embedding(InferTensor t);

// ─────────────────────────────────────────────────────────────────────────────
// KV CACHE SERIALIZATION API
// ─────────────────────────────────────────────────────────────────────────────

// Serialize the KV cache for a sequence slot.
//
// Two-call protocol:
//   1. Call with out_buf=NULL to query the required buffer size.
//      Returns the required byte count, or -1 on error (llm is NULL or ctx unloaded).
//   2. Call with a caller-allocated out_buf of at least out_buf_size bytes.
//      Returns the number of bytes actually written, or -1 on error.
//
// seq_id: the sequence slot ID (from infer_seq_slot_id()).
int infer_llm_kv_serialize(InferLLM llm, int seq_id,
                            uint8_t* out_buf, int out_buf_size);

// Deserialize KV cache bytes into a sequence slot.
// seq_id:  the destination sequence slot.
// data:    pointer to the serialized bytes (from a previous infer_llm_kv_serialize call).
// nbytes:  length of data in bytes.
// Returns 0 on success, -1 on error.
int infer_llm_kv_deserialize(InferLLM llm, int seq_id,
                              const uint8_t* data, int nbytes);

// ─────────────────────────────────────────────────────────────────────────────
// KV PAGE METRICS
// ─────────────────────────────────────────────────────────────────────────────

// Returns number of KV cache pages currently free.
int infer_llm_kv_pages_free(InferLLM llm);

// Returns total KV cache pages available (= ctx_size / page_size).
int infer_llm_kv_pages_total(InferLLM llm);

// Returns the page size in tokens (always 16 in this build).
int infer_llm_kv_page_size(InferLLM llm);

#ifdef __cplusplus
} // extern "C"
#endif
