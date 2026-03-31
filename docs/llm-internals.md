# LLM Internals — llama.cpp Integration Design

> Reference document for implementing `cpp/llm/` (T-24 through T-28).
> Covers the llama.cpp C API, key structs, and the multi-sequence batch strategy
> infergo will use to serve concurrent requests.

---

## 1. Object Hierarchy

```
llama_model          — weights, vocab, hparams (shared across contexts)
    └── llama_context    — KV cache, compute graph, per-inference state
            └── llama_batch  — one decode call's worth of tokens
```

All three are opaque C structs. The API accesses them via pointer.

---

## 2. Key Structs

### `llama_model` (opaque)

Loaded from a `.gguf` file. Holds all weight tensors and vocabulary.
One model instance can serve **many concurrent contexts** — weights are read-only after load.

**Lifecycle:**
```c
struct llama_model_params mparams = llama_model_default_params();
mparams.n_gpu_layers = 99;          // offload all layers to GPU
struct llama_model *model = llama_model_load_from_file(path, mparams);
// ... use ...
llama_model_free(model);
```

**Key accessors:**
| Function | Returns |
|---|---|
| `llama_model_n_ctx_train(model)` | training context length |
| `llama_model_n_embd(model)` | embedding dimension |
| `llama_model_n_layer(model)` | number of transformer layers |
| `llama_model_n_params(model)` | total parameter count |
| `llama_model_size(model)` | weight bytes on device |
| `llama_model_get_vocab(model)` | `llama_vocab*` for token ops |
| `llama_model_chat_template(model, NULL)` | Jinja chat template string |

---

### `llama_context` (opaque)

Wraps the KV cache + compute graph for a model. Each context has its own
KV cache. The key parameter for multi-sequence serving is `n_seq_max`.

**Lifecycle:**
```c
struct llama_context_params cparams = llama_context_default_params();
cparams.n_ctx     = 4096;   // total KV cache tokens across all sequences
cparams.n_batch   = 512;    // max logical batch size per llama_decode call
cparams.n_ubatch  = 512;    // max physical batch (GPU kernel granularity)
cparams.n_seq_max = 16;     // max concurrent sequences sharing this KV cache
cparams.offload_kqv = true; // keep KV cache on GPU (critical for performance)

struct llama_context *ctx = llama_init_from_model(model, cparams);
// ... use ...
llama_free(ctx);
```

**Key accessors post-creation (actual values may differ from requested):**
```c
uint32_t llama_n_ctx(ctx);       // actual KV cache size
uint32_t llama_n_seq_max(ctx);   // actual max sequences
uint32_t llama_n_batch(ctx);     // actual max batch size
```

**KV cache memory management** (via `llama_memory_t`):
```c
llama_memory_t mem = llama_get_memory(ctx);

// Remove all tokens for seq 3 at positions [pos, ∞)
llama_memory_seq_rm(mem, /*seq_id=*/3, /*p0=*/pos, /*p1=*/-1);

// Copy seq 0's KV cache into seq 1 (for beam search / forking)
llama_memory_seq_cp(mem, /*src=*/0, /*dst=*/1, /*p0=*/0, /*p1=*/-1);

// Clear everything
llama_memory_clear(mem, /*data=*/true);
```

---

### `llama_batch`

The **central data structure** for feeding tokens into the engine.
One batch can contain tokens from **multiple sequences simultaneously** —
this is the mechanism for concurrent request serving.

**Definition (from llama.h):**
```c
typedef struct llama_batch {
    int32_t       n_tokens;   // number of entries in this batch
    llama_token * token;      // [n_tokens] token IDs (or NULL if using embeddings)
    float       * embd;       // [n_tokens * n_embd] float embeddings (or NULL)
    llama_pos   * pos;        // [n_tokens] position in sequence
    int32_t     * n_seq_id;   // [n_tokens] how many seq_ids each token belongs to
    llama_seq_id** seq_id;    // [n_tokens] array of seq_id arrays
    int8_t      * logits;     // [n_tokens] 1 = compute logits for this token
} llama_batch;
```

**Allocation:**
```c
// n_tokens_max: max tokens in a single decode call
// embd=0: token mode (use token IDs, not embeddings)
// n_seq_max=1: each token belongs to at most 1 sequence
struct llama_batch batch = llama_batch_init(/*n_tokens_max=*/512, /*embd=*/0, /*n_seq_max=*/1);
// ... fill and use ...
llama_batch_free(batch);
```

---

### `llama_token_data_array`

Holds the per-token logit/probability distribution output after `llama_decode`.

```c
typedef struct llama_token_data {
    llama_token id;   // token id
    float logit;      // log-odds (pre-softmax)
    float p;          // probability (post-softmax, if sampled)
} llama_token_data;

typedef struct llama_token_data_array {
    llama_token_data * data;    // pointer to candidate array
    size_t             size;    // number of candidates (vocab size)
    int64_t            selected; // index of the chosen token (-1 if not yet sampled)
    bool               sorted;  // true if sorted by logit descending
} llama_token_data_array;
```

Used by the sampler chain to select the next token from logits.

---

## 3. Multi-Sequence Batch Decode Strategy

This is how infergo will serve N concurrent requests from one `llama_context`.

### Concept

A single `llama_decode()` call processes a batch containing tokens from
multiple sequences. Each token in the batch has a `seq_id` tag and a `pos`
(position in its sequence's KV cache). llama.cpp routes attention correctly
for each sequence independently.

### Step-by-step: Prefill + Decode Loop

```
┌─────────────────────────────────────────────────────────────────┐
│  llama_context  (n_ctx=4096, n_seq_max=16)                      │
│                                                                 │
│  KV cache slots:                                                │
│  [seq_id=0, pos=0..63]  ← request A, prompt tokens             │
│  [seq_id=1, pos=0..31]  ← request B, prompt tokens             │
│  [seq_id=2, pos=0..127] ← request C, prompt tokens             │
└─────────────────────────────────────────────────────────────────┘
```

**Phase 1 — Prefill (prompt ingestion):**

Each new request's prompt is added to the batch as a contiguous run with its `seq_id`:
```
batch entry 0: token=hello,  seq_id=0, pos=0, logits=0
batch entry 1: token=world,  seq_id=0, pos=1, logits=0
batch entry 2: token=how,    seq_id=1, pos=0, logits=0
batch entry 3: token=are,    seq_id=1, pos=1, logits=0
batch entry 4: token=you,    seq_id=1, pos=2, logits=1  ← want logits for last token only
...
```

Call `llama_decode(ctx, batch)` once to process all prompts together.
Only the last token of each sequence has `logits=1`.

**Phase 2 — Decode (token-by-token generation):**

Each decode step submits one token per active sequence:
```
batch entry 0: token=next_A, seq_id=0, pos=64, logits=1
batch entry 1: token=next_B, seq_id=1, pos=32, logits=1
batch entry 2: token=next_C, seq_id=2, pos=128, logits=1
```

After `llama_decode`, read logits:
```c
float *logits_A = llama_get_logits_ith(ctx, 0);  // [vocab_size]
float *logits_B = llama_get_logits_ith(ctx, 1);
float *logits_C = llama_get_logits_ith(ctx, 2);
```

Sample next token for each sequence independently, append to batch next step.

**Phase 3 — Sequence completion:**

When a sequence generates EOS (`llama_vocab_is_eog`), evict its KV slots:
```c
llama_memory_seq_rm(mem, seq_id, -1, -1);  // remove all positions
```
The freed KV slots are immediately available for a new sequence.

---

## 4. KV Cache Sizing Formula

```
total_kv_tokens = sum(max_prompt_len[i] + max_gen_len[i]) for i in max_concurrent_seqs

Example: 16 sequences × (1024 prompt + 512 gen) = 24576 tokens → n_ctx = 24576
```

Set `n_ctx` generously; KV memory is pre-allocated at context creation.

---

## 5. Helper: `llama_batch_add` Macro Pattern

llama.cpp's `common/` provides a helper that infergo will replicate internally:

```cpp
// Add one token to a batch (inline helper)
static void batch_add(llama_batch& batch, llama_token id, llama_pos pos,
                      llama_seq_id seq_id, bool logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = 1;
    batch.seq_id  [batch.n_tokens][0] = seq_id;
    batch.logits  [batch.n_tokens] = logits ? 1 : 0;
    batch.n_tokens++;
}
```

---

## 6. Special Tokens (vocab API)

```c
const struct llama_vocab *vocab = llama_model_get_vocab(model);

llama_token bos = llama_vocab_bos(vocab);  // beginning of sequence
llama_token eos = llama_vocab_eos(vocab);  // end of sequence
llama_token eot = llama_vocab_eot(vocab);  // end of turn (LLaMA 3 instruct)

// Check if a generated token signals stop
if (llama_vocab_is_eog(vocab, token)) { /* stop this sequence */ }
```

---

## 7. Sampler Chain

llama.cpp uses a composable sampler pipeline. For infergo's use case (greedy + temperature):

```c
struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
struct llama_sampler *smpl = llama_sampler_chain_init(sparams);

llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

// After decode:
llama_token next_token = llama_sampler_sample(smpl, ctx, /*batch_idx=*/-1);
llama_sampler_accept(smpl, next_token);  // update sampler state (repetition penalty etc.)

// Cleanup
llama_sampler_free(smpl);
```

---

## 8. Initialization / Teardown Order

```
llama_backend_init()
    → llama_model_load_from_file()
        → llama_init_from_model()        // creates context + KV cache
            → llama_batch_init()
                → [prefill + decode loop]
            → llama_batch_free()
        → llama_free()                   // free context
    → llama_model_free()
llama_backend_free()
```

`llama_backend_init()` / `llama_backend_free()` are process-global — call once each.

---

## 9. infergo Integration Plan (T-24 onwards)

| Task | What | llama.cpp hook |
|---|---|---|
| T-24 | KV slot manager | tracks which `seq_id` slots are free/in-use |
| T-25 | `LLMEngine` C++ class | owns `llama_model` + `llama_context`, runs batch loop |
| T-26 | Sampler | wraps sampler chain per sequence |
| T-27 | `InferSequence` lifecycle | prefill → decode → EOS → evict |
| T-28 | `infer_llm_batch_decode` C API | bridges Go → C++ batch loop |
| T-29 | Go wrapper | `go/llm/model.go`, `go/llm/sequence.go` |
