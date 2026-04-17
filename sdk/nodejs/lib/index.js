/* sdk/nodejs/lib/index.js — JavaScript wrapper classes for infergo N-API addon */

'use strict';

const path = require('path');
const native = require('bindings')('infergo');

/* ─────────────────────────────────────────────────────────────────────────────
 * LLM — GGUF model inference with full C generation loop
 * ───────────────────────────────────────────────────────────────────────────── */

class LLM {
  /**
   * Load a GGUF model.
   * @param {string} modelPath  Path to the .gguf file.
   * @param {object} [opts]
   * @param {number} [opts.gpuLayers=99]   Transformer layers to offload to GPU.
   * @param {number} [opts.ctxSize=4096]   Total KV cache token budget.
   * @param {number} [opts.seqMax=1]       Max concurrent sequences.
   * @param {number} [opts.batchSize=512]  Max tokens per decode call.
   */
  constructor(modelPath, opts = {}) {
    const gpuLayers = opts.gpuLayers ?? 99;
    const ctxSize   = opts.ctxSize ?? 4096;
    const seqMax    = opts.seqMax ?? 1;
    const batchSize = opts.batchSize ?? 512;
    this._handle = native.llmCreate(modelPath, gpuLayers, ctxSize, seqMax, batchSize);
    this._destroyed = false;
  }

  /**
   * Tokenize text using the model's built-in vocabulary.
   * @param {string}  text
   * @param {boolean} [addBos=true]  Prepend BOS token.
   * @returns {number[]} Token IDs.
   */
  tokenize(text, addBos = true) {
    this._checkAlive();
    return native.llmTokenize(this._handle, text, addBos ? 1 : 0);
  }

  /**
   * Generate text from a prompt.
   * @param {string} prompt          Text prompt (will be tokenized internally).
   * @param {object} [opts]
   * @param {number} [opts.maxTokens=256]     Max tokens to generate.
   * @param {number} [opts.temperature=0.7]   Sampling temperature (0 = greedy).
   * @param {number} [opts.topP=0.9]          Nucleus sampling threshold.
   * @param {string} [opts.grammar=null]      GBNF grammar constraint.
   * @returns {{ text: string, numTokens: number }}
   */
  generate(prompt, opts = {}) {
    this._checkAlive();
    const maxTokens   = opts.maxTokens ?? 256;
    const temperature = opts.temperature ?? 0.7;
    const topP        = opts.topP ?? 0.9;
    const grammar     = opts.grammar ?? null;

    const tokens = this.tokenize(prompt, true);
    return native.llmGenerate(this._handle, tokens, maxTokens, temperature, topP, grammar);
  }

  /**
   * Generate text from pre-tokenized token IDs.
   * @param {number[]} tokens         Pre-tokenized prompt tokens.
   * @param {object}   [opts]         Same options as generate().
   * @returns {{ text: string, numTokens: number }}
   */
  generateFromTokens(tokens, opts = {}) {
    this._checkAlive();
    const maxTokens   = opts.maxTokens ?? 256;
    const temperature = opts.temperature ?? 0.7;
    const topP        = opts.topP ?? 0.9;
    const grammar     = opts.grammar ?? null;

    return native.llmGenerate(this._handle, tokens, maxTokens, temperature, topP, grammar);
  }

  /**
   * Chat with the model using a message array.
   * Formats messages as "<|role|>\ncontent" and generates a response.
   * @param {Array<{role: string, content: string}>} messages
   * @param {object} [opts]  Same options as generate().
   * @returns {string} The assistant's reply.
   */
  chat(messages, opts = {}) {
    this._checkAlive();
    const prompt = messages
      .map(m => `<|${m.role}|>\n${m.content}`)
      .join('\n') + '\n<|assistant|>\n';
    const result = this.generate(prompt, opts);
    return result.text;
  }

  /** @returns {number} Vocabulary size. */
  get vocabSize() {
    this._checkAlive();
    return native.llmVocabSize(this._handle);
  }

  /** @returns {number} BOS token ID. */
  get bosToken() {
    this._checkAlive();
    return native.llmBos(this._handle);
  }

  /** @returns {number} EOS token ID. */
  get eosToken() {
    this._checkAlive();
    return native.llmEos(this._handle);
  }

  /**
   * Convert a token ID to its string piece.
   * @param {number} tokenId
   * @returns {string}
   */
  tokenToPiece(tokenId) {
    this._checkAlive();
    return native.llmTokenToPiece(this._handle, tokenId);
  }

  /** Release the native LLM handle. */
  destroy() {
    if (!this._destroyed) {
      native.llmDestroy(this._handle);
      this._destroyed = true;
    }
  }

  _checkAlive() {
    if (this._destroyed) throw new Error('LLM handle has been destroyed');
  }
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Embedding — ONNX embedding model + HuggingFace tokenizer
 * ───────────────────────────────────────────────────────────────────────────── */

class Embedding {
  /**
   * Load an embedding model.
   * @param {string} modelPath      Path to the ONNX model file.
   * @param {string} tokenizerPath  Path to tokenizer.json.
   * @param {object} [opts]
   * @param {string} [opts.provider='cpu']  Execution provider ('cpu'|'cuda').
   * @param {number} [opts.deviceId=0]      GPU device ID.
   * @param {number} [opts.maxDim=1024]     Max embedding dimension.
   */
  constructor(modelPath, tokenizerPath, opts = {}) {
    const provider = opts.provider ?? 'cpu';
    const deviceId = opts.deviceId ?? 0;
    this._maxDim   = opts.maxDim ?? 1024;

    this._session   = native.sessionCreate(provider, deviceId);
    native.sessionLoad(this._session, modelPath);
    this._tokenizer = native.tokenizerLoad(tokenizerPath);
    this._destroyed = false;
  }

  /**
   * Embed a single text.
   * @param {string} text
   * @returns {number[]} Embedding vector (L2-normalized).
   */
  embed(text) {
    this._checkAlive();
    return native.embedPipeline(this._session, this._tokenizer, text, this._maxDim);
  }

  /**
   * Embed multiple texts in a single batch.
   * @param {string[]} texts
   * @returns {number[][]} Array of embedding vectors.
   */
  embedBatch(texts) {
    this._checkAlive();
    return native.embedBatchPipeline(this._session, this._tokenizer, texts, this._maxDim);
  }

  /**
   * Tokenize text using the HuggingFace tokenizer.
   * @param {string}  text
   * @param {boolean} [addSpecial=true]
   * @returns {number[]} Token IDs.
   */
  tokenize(text, addSpecial = true) {
    this._checkAlive();
    return native.tokenizerEncode(this._tokenizer, text, addSpecial ? 1 : 0);
  }

  /**
   * Decode token IDs back to text.
   * @param {number[]} ids
   * @param {boolean}  [skipSpecial=true]
   * @returns {string}
   */
  decode(ids, skipSpecial = true) {
    this._checkAlive();
    return native.tokenizerDecode(this._tokenizer, ids, skipSpecial ? 1 : 0);
  }

  /** @returns {number} Tokenizer vocabulary size. */
  get vocabSize() {
    this._checkAlive();
    return native.tokenizerVocabSize(this._tokenizer);
  }

  /** Release native handles. */
  destroy() {
    if (!this._destroyed) {
      native.tokenizerDestroy(this._tokenizer);
      native.sessionDestroy(this._session);
      this._destroyed = true;
    }
  }

  _checkAlive() {
    if (this._destroyed) throw new Error('Embedding handle has been destroyed');
  }
}

/* ─────────────────────────────────────────────────────────────────────────────
 * VectorDB — HNSW vector database with persistence and filtering
 * ───────────────────────────────────────────────────────────────────────────── */

class VectorDB {
  /**
   * Create a vector database.
   * @param {number} dim              Vector dimension.
   * @param {object} [opts]
   * @param {number} [opts.M=16]              HNSW max connections per node.
   * @param {number} [opts.efConstruction=200] Search width during build.
   */
  constructor(dim, opts = {}) {
    const M  = opts.M ?? 16;
    const ef = opts.efConstruction ?? 200;
    this._handle = native.vectordbCreate(dim, M, ef);
    this._dim = dim;
    this._destroyed = false;
  }

  /**
   * Insert a vector.
   * @param {number}      id        Unique integer ID.
   * @param {number[]}    vector    Float vector of length dim.
   * @param {string|null} [metadata=null] JSON metadata string.
   */
  insert(id, vector, metadata = null) {
    this._checkAlive();
    native.vectordbInsert(this._handle, id, vector, metadata);
  }

  /**
   * Delete a vector by ID.
   * @param {number} id
   */
  delete(id) {
    this._checkAlive();
    native.vectordbDelete(this._handle, id);
  }

  /**
   * Search for nearest neighbors.
   * @param {number[]} query       Query vector.
   * @param {number}   [k=5]      Number of results.
   * @param {object}   [opts]
   * @param {number}   [opts.efSearch=50]       Search width.
   * @param {string}   [opts.filter=null]       Metadata filter expression.
   * @param {number}   [opts.maxResults]        Max results (defaults to k).
   * @returns {{ ids: number[], distances: number[] }}
   */
  search(query, k = 5, opts = {}) {
    this._checkAlive();
    const efSearch   = opts.efSearch ?? 50;
    const filter     = opts.filter ?? null;
    const maxResults = opts.maxResults ?? k;
    return native.vectordbSearch(this._handle, query, k, efSearch, filter, maxResults);
  }

  /** @returns {number} Number of vectors in the database. */
  get size() {
    this._checkAlive();
    return native.vectordbSize(this._handle);
  }

  /**
   * Save the database to disk.
   * @param {string} filePath
   */
  save(filePath) {
    this._checkAlive();
    native.vectordbSave(this._handle, filePath);
  }

  /**
   * Load the database from disk.
   * @param {string} filePath
   */
  load(filePath) {
    this._checkAlive();
    native.vectordbLoad(this._handle, filePath);
  }

  /** Release native handle. */
  destroy() {
    if (!this._destroyed) {
      native.vectordbFree(this._handle);
      this._destroyed = true;
    }
  }

  _checkAlive() {
    if (this._destroyed) throw new Error('VectorDB handle has been destroyed');
  }
}

/* ─────────────────────────────────────────────────────────────────────────────
 * BM25 — full-text search index
 * ───────────────────────────────────────────────────────────────────────────── */

class BM25 {
  /**
   * Create a BM25 full-text search index.
   * @param {object} [opts]
   * @param {number} [opts.k1=1.2]  Term frequency saturation.
   * @param {number} [opts.b=0.75]  Document length normalization.
   */
  constructor(opts = {}) {
    const k1 = opts.k1 ?? 1.2;
    const b  = opts.b ?? 0.75;
    this._handle = native.bm25Create(k1, b);
    this._destroyed = false;
  }

  /**
   * Insert a document.
   * @param {number} id    Unique document ID.
   * @param {string} text  Document text.
   */
  insert(id, text) {
    this._checkAlive();
    native.bm25Insert(this._handle, id, text);
  }

  /**
   * Remove a document by ID.
   * @param {number} id
   */
  remove(id) {
    this._checkAlive();
    native.bm25Remove(this._handle, id);
  }

  /**
   * Search the index.
   * @param {string} query        Query string.
   * @param {number} [k=5]       Number of results.
   * @param {number} [maxResults] Max results (defaults to k).
   * @returns {{ ids: number[], scores: number[] }}
   */
  search(query, k = 5, maxResults) {
    this._checkAlive();
    const max = maxResults ?? k;
    return native.bm25Search(this._handle, query, k, max);
  }

  /** @returns {number} Number of documents in the index. */
  get size() {
    this._checkAlive();
    return native.bm25Size(this._handle);
  }

  /**
   * Save the index to disk.
   * @param {string} filePath
   */
  save(filePath) {
    this._checkAlive();
    native.bm25Save(this._handle, filePath);
  }

  /**
   * Load the index from disk.
   * @param {string} filePath
   */
  load(filePath) {
    this._checkAlive();
    native.bm25Load(this._handle, filePath);
  }

  /** Release native handle. */
  destroy() {
    if (!this._destroyed) {
      native.bm25Free(this._handle);
      this._destroyed = true;
    }
  }

  _checkAlive() {
    if (this._destroyed) throw new Error('BM25 handle has been destroyed');
  }
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Utility
 * ───────────────────────────────────────────────────────────────────────────── */

/**
 * Get the last error message from the C library.
 * @returns {string|null}
 */
function lastError() {
  return native.lastErrorString();
}

module.exports = { LLM, Embedding, VectorDB, BM25, lastError };
