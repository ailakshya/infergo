// infergo WASM/JavaScript SDK
// CPU-only inference in browser/Node.js via WebAssembly

class Infergo {
  constructor(wasmModule) {
    this._module = wasmModule;
  }

  static async load(wasmPath = 'infergo.wasm') {
    const Module = await import(wasmPath.replace('.wasm', '.js'));
    const instance = await Module.default();
    return new Infergo(instance);
  }

  lastError() {
    const ptr = this._module._wasm_last_error();
    return ptr ? this._module.UTF8ToString(ptr) : '';
  }

  createLLM(modelPath, { ctxSize = 2048, nSeqMax = 1, nBatch = 512 } = {}) {
    const pathPtr = this._module.allocateUTF8(modelPath);
    const handle = this._module._wasm_llm_create(pathPtr, 0, ctxSize, nSeqMax, nBatch);
    this._module._free(pathPtr);
    if (!handle) throw new Error(`LLM load failed: ${this.lastError()}`);
    return new LLM(this._module, handle);
  }

  createTokenizer(path) {
    const pathPtr = this._module.allocateUTF8(path);
    const handle = this._module._wasm_tokenizer_load(pathPtr);
    this._module._free(pathPtr);
    if (!handle) throw new Error(`Tokenizer load failed: ${this.lastError()}`);
    return new Tokenizer(this._module, handle);
  }

  createSession() {
    const handle = this._module._wasm_session_create();
    if (!handle) throw new Error(`Session create failed: ${this.lastError()}`);
    return new Session(this._module, handle);
  }

  createVectorDB(dim = 384, M = 16, ef = 200) {
    const handle = this._module._wasm_vectordb_create(dim, M, ef);
    if (!handle) throw new Error('Failed to create VectorDB');
    return new VectorDB(this._module, handle, dim);
  }

  createBM25(k1 = 1.2, b = 0.75) {
    const handle = this._module._wasm_bm25_create(k1, b);
    if (!handle) throw new Error('Failed to create BM25');
    return new BM25(this._module, handle);
  }

  // Upload file into Emscripten virtual filesystem
  uploadFile(virtualPath, data) {
    if (data instanceof ArrayBuffer) data = new Uint8Array(data);
    this._module.FS.writeFile(virtualPath, data);
  }

  async fetchFile(url, virtualPath) {
    const resp = await fetch(url);
    const buf = await resp.arrayBuffer();
    this.uploadFile(virtualPath, buf);
  }
}

// ── LLM ─────────────────────────────────────────────────────────────────────

class LLM {
  constructor(module, handle) {
    this._module = module;
    this._handle = handle;
  }

  get vocabSize() { return this._module._wasm_llm_vocab_size(this._handle); }
  get bos() { return this._module._wasm_llm_bos(this._handle); }
  get eos() { return this._module._wasm_llm_eos(this._handle); }
  isEog(token) { return this._module._wasm_llm_is_eog(this._handle, token) !== 0; }

  tokenize(text, addBos = true) {
    const textPtr = this._module.allocateUTF8(text);
    const outPtr = this._module._wasm_malloc(4096 * 4);
    const n = this._module._wasm_llm_tokenize(this._handle, textPtr,
      addBos ? 1 : 0, outPtr, 4096);
    this._module._free(textPtr);
    if (n < 0) { this._module._wasm_free(outPtr); throw new Error('Tokenize failed'); }
    const tokens = new Int32Array(this._module.HEAP32.buffer, outPtr, n).slice();
    this._module._wasm_free(outPtr);
    return tokens;
  }

  generateFromTokens(tokens, { maxTokens = 256, temperature = 0.7, topP = 0.9, grammar = null } = {}) {
    const nPrompt = tokens.length;
    const tokPtr = this._module._wasm_malloc(nPrompt * 4);
    new Int32Array(this._module.HEAP32.buffer, tokPtr, nPrompt).set(tokens);

    const outPtr = this._module._wasm_malloc(32768);
    const genPtr = this._module._wasm_malloc(4);

    let gramPtr = 0;
    if (grammar) gramPtr = this._module.allocateUTF8(grammar);

    const rc = this._module._wasm_llm_generate(
      this._handle, tokPtr, nPrompt, maxTokens, temperature, topP,
      gramPtr, outPtr, 32768, genPtr);

    if (gramPtr) this._module._free(gramPtr);
    this._module._wasm_free(tokPtr);

    if (rc < 0) {
      this._module._wasm_free(outPtr);
      this._module._wasm_free(genPtr);
      throw new Error('Generate failed');
    }

    const text = this._module.UTF8ToString(outPtr);
    const tokensGenerated = this._module.HEAP32[genPtr >> 2];
    this._module._wasm_free(outPtr);
    this._module._wasm_free(genPtr);
    return { text, tokensGenerated };
  }

  // Convenience: prompt string -> text
  generate(prompt, opts = {}) {
    const tokens = this.tokenize(prompt, true);
    return this.generateFromTokens(tokens, opts);
  }

  // Simple one-shot (uses static buffer in C)
  generateSimple(prompt, maxTokens = 64, temperature = 0.7) {
    const promptPtr = this._module.allocateUTF8(prompt);
    const resultPtr = this._module._wasm_llm_generate_text(
      this._handle, promptPtr, maxTokens, temperature);
    this._module._free(promptPtr);
    if (!resultPtr) throw new Error('Generation failed');
    return this._module.UTF8ToString(resultPtr);
  }

  destroy() {
    if (this._handle) {
      this._module._wasm_llm_destroy(this._handle);
      this._handle = null;
    }
  }
}

// ── Tokenizer ───────────────────────────────────────────────────────────────

class Tokenizer {
  constructor(module, handle) {
    this._module = module;
    this._handle = handle;
  }

  encode(text, addSpecial = true) {
    const textPtr = this._module.allocateUTF8(text);
    const idsPtr = this._module._wasm_malloc(4096 * 4);
    const maskPtr = this._module._wasm_malloc(4096 * 4);
    const n = this._module._wasm_tokenizer_encode(
      this._handle, textPtr, addSpecial ? 1 : 0, idsPtr, maskPtr, 4096);
    this._module._free(textPtr);
    if (n < 0) {
      this._module._wasm_free(idsPtr);
      this._module._wasm_free(maskPtr);
      throw new Error('Encode failed');
    }
    const ids = new Int32Array(this._module.HEAP32.buffer, idsPtr, n).slice();
    const mask = new Int32Array(this._module.HEAP32.buffer, maskPtr, n).slice();
    this._module._wasm_free(idsPtr);
    this._module._wasm_free(maskPtr);
    return { ids, mask };
  }

  decode(ids, skipSpecial = true) {
    const n = ids.length;
    const idsPtr = this._module._wasm_malloc(n * 4);
    new Int32Array(this._module.HEAP32.buffer, idsPtr, n).set(ids);
    const bufPtr = this._module._wasm_malloc(16384);
    const rc = this._module._wasm_tokenizer_decode(
      this._handle, idsPtr, n, skipSpecial ? 1 : 0, bufPtr, 16384);
    this._module._wasm_free(idsPtr);
    if (rc < 0) { this._module._wasm_free(bufPtr); throw new Error('Decode failed'); }
    const text = this._module.UTF8ToString(bufPtr);
    this._module._wasm_free(bufPtr);
    return text;
  }

  get vocabSize() { return this._module._wasm_tokenizer_vocab_size(this._handle); }

  destroy() {
    if (this._handle) {
      this._module._wasm_tokenizer_destroy(this._handle);
      this._handle = null;
    }
  }
}

// ── ONNX Session ────────────────────────────────────────────────────────────

class Session {
  constructor(module, handle) {
    this._module = module;
    this._handle = handle;
  }

  load(modelPath) {
    const pathPtr = this._module.allocateUTF8(modelPath);
    const rc = this._module._wasm_session_load(this._handle, pathPtr);
    this._module._free(pathPtr);
    if (rc !== 0) throw new Error('Session load failed');
  }

  get numInputs() { return this._module._wasm_session_num_inputs(this._handle); }
  get numOutputs() { return this._module._wasm_session_num_outputs(this._handle); }

  destroy() {
    if (this._handle) {
      this._module._wasm_session_destroy(this._handle);
      this._handle = null;
    }
  }
}

// ── Embedding helper ────────────────────────────────────────────────────────

function embedPipeline(module, session, tokenizer, text, maxDim = 4096) {
  const textPtr = module.allocateUTF8(text);
  const outPtr = module._wasm_malloc(maxDim * 4);
  const dim = module._wasm_embed_pipeline(
    session._handle, tokenizer._handle, textPtr, outPtr, maxDim);
  module._free(textPtr);
  if (dim < 0) { module._wasm_free(outPtr); throw new Error('Embed failed'); }
  const vec = new Float32Array(module.HEAPF32.buffer, outPtr, dim).slice();
  module._wasm_free(outPtr);
  return vec;
}

// ── VectorDB ────────────────────────────────────────────────────────────────

class VectorDB {
  constructor(module, handle, dim) {
    this._module = module;
    this._handle = handle;
    this._dim = dim;
  }

  insert(id, vector, metadata = '') {
    const vecPtr = this._module._malloc(vector.length * 4);
    new Float32Array(this._module.HEAPF32.buffer, vecPtr, vector.length).set(vector);
    const metaPtr = this._module.allocateUTF8(metadata);
    this._module._wasm_vectordb_insert(this._handle, id, vecPtr, metaPtr);
    this._module._free(vecPtr);
    this._module._free(metaPtr);
  }

  delete(id) {
    this._module._wasm_vectordb_delete(this._handle, id);
  }

  search(query, k = 10, efSearch = 50, filter = null) {
    const qPtr = this._module._malloc(query.length * 4);
    new Float32Array(this._module.HEAPF32.buffer, qPtr, query.length).set(query);
    const max = Math.min(k, 256);
    const idsPtr = this._module._malloc(max * 8);  // int64_t
    const distsPtr = this._module._malloc(max * 4);
    let filterPtr = 0;
    if (filter) filterPtr = this._module.allocateUTF8(filter);

    const n = this._module._wasm_vectordb_search(
      this._handle, qPtr, k, efSearch, filterPtr, idsPtr, distsPtr, max);
    this._module._free(qPtr);
    if (filterPtr) this._module._free(filterPtr);

    if (n < 0) {
      this._module._free(idsPtr); this._module._free(distsPtr);
      throw new Error('VectorDB search failed');
    }

    const results = [];
    for (let i = 0; i < n; i++) {
      // Read int64_t as two int32s (little-endian)
      const lo = this._module.HEAP32[(idsPtr >> 2) + i * 2];
      const hi = this._module.HEAP32[(idsPtr >> 2) + i * 2 + 1];
      const id = (hi * 0x100000000) + (lo >>> 0);
      results.push({ id, distance: this._module.HEAPF32[(distsPtr >> 2) + i] });
    }
    this._module._free(idsPtr); this._module._free(distsPtr);
    return results;
  }

  get size() { return this._module._wasm_vectordb_size(this._handle); }

  save(path) {
    const p = this._module.allocateUTF8(path);
    const rc = this._module._wasm_vectordb_save(this._handle, p);
    this._module._free(p);
    if (rc < 0) throw new Error('VectorDB save failed');
  }

  load(path) {
    const p = this._module.allocateUTF8(path);
    const rc = this._module._wasm_vectordb_load(this._handle, p);
    this._module._free(p);
    if (rc < 0) throw new Error('VectorDB load failed');
  }

  destroy() {
    if (this._handle) {
      this._module._wasm_vectordb_free(this._handle);
      this._handle = null;
    }
  }
}

// ── BM25 ────────────────────────────────────────────────────────────────────

class BM25 {
  constructor(module, handle) {
    this._module = module;
    this._handle = handle;
  }

  insert(id, text) {
    const textPtr = this._module.allocateUTF8(text);
    this._module._wasm_bm25_insert(this._handle, id, textPtr);
    this._module._free(textPtr);
  }

  remove(id) {
    this._module._wasm_bm25_remove(this._handle, id);
  }

  search(query, k = 10) {
    const qPtr = this._module.allocateUTF8(query);
    const max = Math.min(k, 256);
    const idsPtr = this._module._malloc(max * 8);
    const scoresPtr = this._module._malloc(max * 4);
    const n = this._module._wasm_bm25_search(this._handle, qPtr, k,
      idsPtr, scoresPtr, max);
    this._module._free(qPtr);
    if (n < 0) {
      this._module._free(idsPtr); this._module._free(scoresPtr);
      throw new Error('BM25 search failed');
    }
    const results = [];
    for (let i = 0; i < n; i++) {
      const lo = this._module.HEAP32[(idsPtr >> 2) + i * 2];
      const hi = this._module.HEAP32[(idsPtr >> 2) + i * 2 + 1];
      results.push({
        id: (hi * 0x100000000) + (lo >>> 0),
        score: this._module.HEAPF32[(scoresPtr >> 2) + i]
      });
    }
    this._module._free(idsPtr); this._module._free(scoresPtr);
    return results;
  }

  get size() { return this._module._wasm_bm25_size(this._handle); }

  destroy() {
    if (this._handle) {
      this._module._wasm_bm25_free(this._handle);
      this._handle = null;
    }
  }
}

if (typeof module !== 'undefined') module.exports = { Infergo, LLM, Tokenizer, Session, VectorDB, BM25, embedPipeline };
