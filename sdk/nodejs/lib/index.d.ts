/* sdk/nodejs/lib/index.d.ts — TypeScript declarations for infergo Node.js SDK */

export interface GenerateOptions {
  /** Max tokens to generate (default: 256). */
  maxTokens?: number;
  /** Sampling temperature, 0 = greedy (default: 0.7). */
  temperature?: number;
  /** Nucleus sampling threshold (default: 0.9). */
  topP?: number;
  /** GBNF grammar constraint string (default: null). */
  grammar?: string | null;
}

export interface GenerateResult {
  /** Generated text. */
  text: string;
  /** Number of tokens generated. */
  numTokens: number;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface LLMOptions {
  /** Transformer layers to offload to GPU (default: 99). */
  gpuLayers?: number;
  /** Total KV cache token budget (default: 4096). */
  ctxSize?: number;
  /** Max concurrent sequences (default: 1). */
  seqMax?: number;
  /** Max tokens per decode call (default: 512). */
  batchSize?: number;
}

/** GGUF model inference with full C generation loop. */
export class LLM {
  /**
   * Load a GGUF model.
   * @param modelPath Path to the .gguf file.
   * @param opts      LLM configuration options.
   */
  constructor(modelPath: string, opts?: LLMOptions);

  /**
   * Tokenize text using the model's built-in vocabulary.
   * @param text   Text to tokenize.
   * @param addBos Prepend BOS token (default: true).
   * @returns Array of token IDs.
   */
  tokenize(text: string, addBos?: boolean): number[];

  /**
   * Generate text from a prompt string.
   * @param prompt Text prompt (tokenized internally).
   * @param opts   Generation options.
   */
  generate(prompt: string, opts?: GenerateOptions): GenerateResult;

  /**
   * Generate text from pre-tokenized token IDs.
   * @param tokens Pre-tokenized prompt.
   * @param opts   Generation options.
   */
  generateFromTokens(tokens: number[], opts?: GenerateOptions): GenerateResult;

  /**
   * Chat with the model using a message array.
   * @param messages Array of chat messages.
   * @param opts     Generation options.
   * @returns The assistant's reply text.
   */
  chat(messages: ChatMessage[], opts?: GenerateOptions): string;

  /** Vocabulary size. */
  readonly vocabSize: number;

  /** BOS token ID. */
  readonly bosToken: number;

  /** EOS token ID. */
  readonly eosToken: number;

  /**
   * Convert a token ID to its string piece.
   * @param tokenId Token ID.
   */
  tokenToPiece(tokenId: number): string;

  /** Release the native LLM handle. */
  destroy(): void;
}

export interface EmbeddingOptions {
  /** Execution provider: 'cpu' | 'cuda' (default: 'cpu'). */
  provider?: string;
  /** GPU device ID (default: 0). */
  deviceId?: number;
  /** Max embedding dimension (default: 1024). */
  maxDim?: number;
}

/** ONNX embedding model + HuggingFace tokenizer. */
export class Embedding {
  /**
   * Load an embedding model.
   * @param modelPath     Path to the ONNX model file.
   * @param tokenizerPath Path to tokenizer.json.
   * @param opts          Configuration options.
   */
  constructor(modelPath: string, tokenizerPath: string, opts?: EmbeddingOptions);

  /**
   * Embed a single text.
   * @param text Input text.
   * @returns L2-normalized embedding vector.
   */
  embed(text: string): number[];

  /**
   * Embed multiple texts in a single batch.
   * @param texts Array of input texts.
   * @returns Array of embedding vectors.
   */
  embedBatch(texts: string[]): number[][];

  /**
   * Tokenize text using the HuggingFace tokenizer.
   * @param text       Input text.
   * @param addSpecial Add special tokens (default: true).
   * @returns Array of token IDs.
   */
  tokenize(text: string, addSpecial?: boolean): number[];

  /**
   * Decode token IDs back to text.
   * @param ids         Array of token IDs.
   * @param skipSpecial Skip special tokens (default: true).
   */
  decode(ids: number[], skipSpecial?: boolean): string;

  /** Tokenizer vocabulary size. */
  readonly vocabSize: number;

  /** Release native handles. */
  destroy(): void;
}

export interface VectorDBOptions {
  /** HNSW max connections per node (default: 16). */
  M?: number;
  /** Search width during build (default: 200). */
  efConstruction?: number;
}

export interface VectorSearchOptions {
  /** Search width (default: 50). */
  efSearch?: number;
  /** Metadata filter expression (default: null). */
  filter?: string | null;
  /** Max results to return (default: k). */
  maxResults?: number;
}

export interface VectorSearchResult {
  /** Matched vector IDs. */
  ids: number[];
  /** Cosine distances (lower = more similar). */
  distances: number[];
}

/** HNSW vector database with persistence, CRUD, and filtering. */
export class VectorDB {
  /**
   * Create a vector database.
   * @param dim  Vector dimension.
   * @param opts Configuration options.
   */
  constructor(dim: number, opts?: VectorDBOptions);

  /**
   * Insert a vector.
   * @param id       Unique integer ID.
   * @param vector   Float vector of length dim.
   * @param metadata JSON metadata string (optional).
   */
  insert(id: number, vector: number[], metadata?: string | null): void;

  /**
   * Delete a vector by ID.
   * @param id Vector ID.
   */
  delete(id: number): void;

  /**
   * Search for nearest neighbors.
   * @param query Query vector.
   * @param k     Number of results (default: 5).
   * @param opts  Search options.
   */
  search(query: number[], k?: number, opts?: VectorSearchOptions): VectorSearchResult;

  /** Number of vectors in the database. */
  readonly size: number;

  /**
   * Save the database to disk.
   * @param filePath Output file path.
   */
  save(filePath: string): void;

  /**
   * Load the database from disk.
   * @param filePath Input file path.
   */
  load(filePath: string): void;

  /** Release native handle. */
  destroy(): void;
}

export interface BM25Options {
  /** Term frequency saturation (default: 1.2). */
  k1?: number;
  /** Document length normalization (default: 0.75). */
  b?: number;
}

export interface BM25SearchResult {
  /** Matched document IDs. */
  ids: number[];
  /** BM25 relevance scores (higher = more relevant). */
  scores: number[];
}

/** BM25 full-text search index. */
export class BM25 {
  /**
   * Create a BM25 index.
   * @param opts Configuration options.
   */
  constructor(opts?: BM25Options);

  /**
   * Insert a document.
   * @param id   Unique document ID.
   * @param text Document text.
   */
  insert(id: number, text: string): void;

  /**
   * Remove a document by ID.
   * @param id Document ID.
   */
  remove(id: number): void;

  /**
   * Search the index.
   * @param query      Query string.
   * @param k          Number of results (default: 5).
   * @param maxResults Max results (default: k).
   */
  search(query: string, k?: number, maxResults?: number): BM25SearchResult;

  /** Number of documents in the index. */
  readonly size: number;

  /**
   * Save the index to disk.
   * @param filePath Output file path.
   */
  save(filePath: string): void;

  /**
   * Load the index from disk.
   * @param filePath Input file path.
   */
  load(filePath: string): void;

  /** Release native handle. */
  destroy(): void;
}

/**
 * Get the last error message from the C library.
 * @returns Error string or null.
 */
export function lastError(): string | null;
