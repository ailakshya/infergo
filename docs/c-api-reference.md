# C API Reference

Header: `cpp/include/infer_api.h`

All functions are `extern "C"` — safe to call from C, Go (via CGo), or any language with a C FFI.

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
| `INFER_ERR_UNKNOWN` | 10 | Unknown error |

```c
const char* infer_last_error_string(void);
```

Returns a pointer to a thread-local string. Valid until the next call on the same thread.

---

## Opaque handle types

```c
typedef void* InferTensor;
typedef void* InferSession;
typedef void* InferTokenizer;
typedef void* InferLLM;
typedef void* InferSeq;
```

All handles are heap-allocated. Ownership rules:
- Functions named `*_create` / `*_alloc` / `*_load` allocate; you must call the matching `*_destroy` / `*_free`.
- Functions that return a handle without `create`/`alloc`/`load` in their name do **not** transfer ownership.

---

## Tensor API

### Allocation

```c
InferTensor infer_tensor_alloc_cpu(
    const int* shape,   // pointer to array of dimensions
    int        ndim,    // number of dimensions
    int        dtype    // DType constant (see below)
);

InferTensor infer_tensor_alloc_cuda(
    const int* shape,
    int        ndim,
    int        dtype,
    int        device_id  // CUDA device index
);
```

Returns `NULL` on failure. No `InferError` — check for NULL.

```c
void infer_tensor_free(InferTensor t);
```

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
// Host → device or host → host depending on where t was allocated.

InferError infer_tensor_to_device(InferTensor t, int device_id);
// Move host tensor to CUDA device. No-op if already on that device.

InferError infer_tensor_to_host(InferTensor t);
// Move CUDA tensor to host. No-op if already on host.
```

### DType values

| Value | Type |
|---|---|
| 0 | float32 |
| 1 | float16 |
| 2 | bfloat16 |
| 3 | int32 |
| 4 | int64 |
| 5 | uint8 |
| 6 | bool |

---

## ONNX Session API

```c
InferSession infer_session_create(
    const char* provider,  // "cpu", "cuda", "tensorrt", "coreml"
    int         device_id
);

InferError infer_session_load(InferSession s, const char* model_path);

void infer_session_destroy(InferSession s);
```

### Input / output metadata

```c
int infer_session_num_inputs(InferSession s);
int infer_session_num_outputs(InferSession s);

InferError infer_session_input_name(
    InferSession s, int idx, char* out_buf, int buf_size);

InferError infer_session_output_name(
    InferSession s, int idx, char* out_buf, int buf_size);
```

### Running inference

```c
InferError infer_session_run(
    InferSession s,
    InferTensor* inputs,    // array of input tensors
    int          n_inputs,
    InferTensor* outputs,   // caller-allocated array; tensors created by callee
    int          n_outputs
);
```

The callee allocates each output tensor. The caller must call `infer_tensor_free` on each.

### Example

```c
InferSession s = infer_session_create("cuda", 0);
infer_session_load(s, "yolov8n.onnx");

InferTensor input  = infer_tensor_alloc_cuda(...);
InferTensor output = NULL;

infer_session_run(s, &input, 1, &output, 1);
// use output...
infer_tensor_free(output);
infer_tensor_free(input);
infer_session_destroy(s);
```

---

## Tokenizer API

```c
InferTokenizer infer_tokenizer_load(const char* path);
// path: directory containing tokenizer.json (HuggingFace format)

void infer_tokenizer_destroy(InferTokenizer tok);
```

### Encoding

```c
int infer_tokenizer_encode(
    InferTokenizer tok,
    const char*    text,
    int            add_special_tokens,   // 1 = add BOS/EOS
    int*           out_ids,              // caller-allocated
    int*           out_attention_mask,   // caller-allocated, may be NULL
    int            max_tokens
);
// Returns number of tokens, or -1 on error.
```

### Decoding

```c
int infer_tokenizer_decode(
    InferTokenizer tok,
    const int*     ids,
    int            n_ids,
    int            skip_special_tokens,
    char*          out_buf,
    int            buf_size
);
// Returns number of bytes written (excluding null terminator), or -1 on error.

int infer_tokenizer_decode_token(
    InferTokenizer tok,
    int            id,
    char*          out_buf,
    int            buf_size
);
// Decode a single token to its string piece.

int infer_tokenizer_vocab_size(InferTokenizer tok);
```

---

## LLM Engine API

### Loading

```c
InferLLM infer_llm_create(
    const char* path,         // .gguf model file
    int         n_gpu_layers, // layers to offload to GPU (0 = CPU only)
    int         ctx_size,     // KV cache context window
    int         n_seq_max,    // max parallel sequences
    int         n_batch       // max tokens per batch
);
// Returns NULL on failure.

void infer_llm_destroy(InferLLM llm);
```

### Vocabulary

```c
int infer_llm_vocab_size(InferLLM llm);
int infer_llm_bos(InferLLM llm);    // beginning-of-sequence token ID
int infer_llm_eos(InferLLM llm);    // end-of-sequence token ID
int infer_llm_is_eog(InferLLM llm, int token);  // 1 if token is end-of-generation
```

### Tokenization

```c
int infer_llm_tokenize(
    InferLLM    llm,
    const char* text,
    int         add_bos,     // prepend BOS token
    int*        out_ids,
    int         max_tokens
);
// Returns token count, or negative on error.

int infer_llm_token_to_piece(
    InferLLM llm, int token, char* out_buf, int buf_size);
```

### Sequences

Each sequence corresponds to one independent generation stream with its own KV cache slot.

```c
InferSeq infer_seq_create(
    InferLLM   llm,
    const int* tokens,   // prompt token IDs
    int        n_tokens
);

void infer_seq_destroy(InferSeq seq);

int  infer_seq_is_done(InferSeq seq);     // 1 when EOS reached
int  infer_seq_position(InferSeq seq);    // current generation position
int  infer_seq_slot_id(InferSeq seq);     // KV cache slot index
void infer_seq_append_token(InferSeq seq, int token);
```

### Batch decode

```c
InferError infer_llm_batch_decode(
    InferLLM  llm,
    InferSeq* seqs,
    int       n_seqs
);
// Runs one forward pass over all sequences. After this, call
// infer_seq_get_logits or infer_seq_sample_token on each sequence.
```

### Logits and sampling

```c
InferError infer_seq_get_logits(
    InferSeq seq,
    float*   out_logits,   // caller-allocated, size = vocab_size
    int      vocab_size
);

int infer_seq_next_tokens(InferSeq seq, int* out_ids, int max_tokens);
// Reserved for speculative decoding.
```

> **Note:** Greedy / top-p sampling is implemented in Go (`seq.SampleToken`). For pure C usage, read logits via `infer_seq_get_logits` and sample yourself.

---

## Preprocessing API

Requires OpenCV at build time (`-DINFER_PREPROCESS_AVAILABLE=1`).

```c
InferTensor infer_preprocess_decode_image(
    const unsigned char* data,
    int                  nbytes
);
// Decodes JPEG / PNG / WebP / BMP → float32 [H, W, 3] tensor on CPU.
// Returns NULL on failure.

InferTensor infer_preprocess_letterbox(
    InferTensor src,
    int         target_w,
    int         target_h
);
// Uniform resize with grey padding → float32 [target_h, target_w, 3].

InferTensor infer_preprocess_normalize(
    InferTensor   src,
    float         scale,        // pixel scale factor (e.g. 1/255.0)
    const float*  mean,         // [3] per-channel mean
    const float*  std           // [3] per-channel std
);
// Normalizes and converts HWC → CHW layout. Returns [3, H, W].

InferTensor infer_preprocess_stack_batch(
    InferTensor* tensors,
    int          n
);
// Stacks n [C, H, W] tensors → [N, C, H, W].
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

int infer_postprocess_classify(
    InferTensor      logits,      // [num_classes] float32
    int              top_k,
    InferClassResult* out_results // caller-allocated, size >= top_k
);
// Applies softmax, returns top-k results sorted by confidence descending.
// Returns number of results written (≤ top_k), or -1 on error.
```

### Object detection (NMS)

```c
typedef struct {
    float x1, y1, x2, y2;  // absolute pixel coordinates
    int   class_idx;
    float confidence;
} InferBox;

int infer_postprocess_nms(
    InferTensor predictions,  // YOLO output [1, D, 4+num_classes] float32
    float       conf_thresh,
    float       iou_thresh,
    InferBox*   out_boxes,    // caller-allocated
    int         max_boxes
);
// Returns number of boxes after NMS, or -1 on error.
```

### Embedding normalization

```c
InferError infer_postprocess_normalize_embedding(InferTensor t);
// L2-normalizes t in-place. No-op if norm is zero.
```
