// Infergo Swift SDK
// High-level Swift wrapper around the infergo C API (infer_api.h).
// Uses OpaquePointer for void* handles and automatic cleanup via deinit.

import CInfergo
import Foundation

// MARK: - Error Handling

/// Errors returned by infergo operations.
public enum InfergoError: Error, CustomStringConvertible {
    case null
    case invalid
    case outOfMemory
    case cuda
    case load(String)
    case runtime(String)
    case shapeMismatch
    case unsupportedDtype
    case cancelled
    case unknown(String)

    /// Build an ``InfergoError`` from an `InferError` return code.
    internal static func from(code: Int32) -> InfergoError {
        let detail = lastErrorString()
        switch code {
        case 1:  return .null
        case 2:  return .invalid
        case 3:  return .outOfMemory
        case 4:  return .cuda
        case 5:  return .load(detail)
        case 6:  return .runtime(detail)
        case 7:  return .shapeMismatch
        case 8:  return .unsupportedDtype
        case 9:  return .cancelled
        default: return .unknown(detail)
        }
    }

    public var description: String {
        switch self {
        case .null:                return "InfergoError: null pointer"
        case .invalid:             return "InfergoError: invalid argument"
        case .outOfMemory:        return "InfergoError: out of memory"
        case .cuda:                return "InfergoError: CUDA error"
        case .load(let s):         return "InfergoError: load failure — \(s)"
        case .runtime(let s):      return "InfergoError: runtime — \(s)"
        case .shapeMismatch:      return "InfergoError: shape mismatch"
        case .unsupportedDtype:   return "InfergoError: unsupported dtype"
        case .cancelled:           return "InfergoError: cancelled"
        case .unknown(let s):      return "InfergoError: unknown — \(s)"
        }
    }
}

/// Returns the last error string from the C API (thread-local).
public func lastErrorString() -> String {
    guard let cStr = infer_last_error_string() else { return "" }
    return String(cString: cStr)
}

/// Throw if a C API call returns a non-zero error code.
@inline(__always)
internal func check(_ code: Int32) throws {
    if code != 0 {
        throw InfergoError.from(code: code)
    }
}

// MARK: - LLM

/// A large-language-model engine backed by a GGUF file.
///
///     let llm = try LLM(path: "model.gguf", gpuLayers: 99)
///     let reply = try llm.generate(prompt: "Hello!")
///     print(reply.text)
///
public final class LLM {
    internal let handle: OpaquePointer

    /// Load a GGUF model.
    ///
    /// - Parameters:
    ///   - path:        Path to the `.gguf` file.
    ///   - gpuLayers:   Number of transformer layers to offload to GPU.
    ///   - contextSize: Total KV-cache token budget.
    ///   - maxSeqs:     Maximum concurrent sequences.
    ///   - batchSize:   Max tokens per decode call.
    public init(path: String,
                gpuLayers: Int32 = 99,
                contextSize: Int32 = 4096,
                maxSeqs: Int32 = 1,
                batchSize: Int32 = 512) throws {
        guard let h = path.withCString({ cPath in
            infer_llm_create(cPath, gpuLayers, contextSize, maxSeqs, batchSize)
        }) else {
            throw InfergoError.load(lastErrorString())
        }
        self.handle = h
    }

    deinit {
        infer_llm_destroy(handle)
    }

    /// Vocabulary size of the loaded model.
    public var vocabSize: Int32 {
        infer_llm_vocab_size(handle)
    }

    /// Beginning-of-sequence token ID.
    public var bosToken: Int32 {
        infer_llm_bos(handle)
    }

    /// End-of-sequence token ID.
    public var eosToken: Int32 {
        infer_llm_eos(handle)
    }

    /// Tokenize a string using the model's built-in vocabulary.
    ///
    /// - Parameters:
    ///   - text:      Input text.
    ///   - addBOS:    Whether to prepend the BOS token.
    ///   - maxTokens: Upper bound on the number of tokens returned.
    /// - Returns: Array of token IDs.
    public func tokenize(_ text: String, addBOS: Bool = true, maxTokens: Int32 = 2048) throws -> [Int32] {
        var buf = [Int32](repeating: 0, count: Int(maxTokens))
        let n = text.withCString { cText in
            infer_llm_tokenize(handle, cText, addBOS ? 1 : 0, &buf, maxTokens)
        }
        guard n >= 0 else {
            throw InfergoError.runtime(lastErrorString())
        }
        return Array(buf.prefix(Int(n)))
    }

    /// Convert a single token ID to its string piece.
    public func tokenToPiece(_ token: Int32) throws -> String {
        var buf = [CChar](repeating: 0, count: 256)
        let rc = infer_llm_token_to_piece(handle, token, &buf, 256)
        guard rc == 0 else {
            throw InfergoError.runtime(lastErrorString())
        }
        return String(cString: buf)
    }

    /// Result of a ``generate`` call.
    public struct GenerateResult {
        /// The generated text.
        public let text: String
        /// Number of tokens generated.
        public let tokenCount: Int32
    }

    /// Token callback signature for streaming generation.
    /// Return `true` to continue, `false` to stop.
    public typealias TokenCallback = (_ token: Int32, _ piece: String) -> Bool

    /// Run the full generation loop in C.
    ///
    /// - Parameters:
    ///   - prompt:      Text prompt (will be tokenized internally).
    ///   - maxTokens:   Maximum tokens to generate.
    ///   - temperature: Sampling temperature (0 = greedy).
    ///   - topP:        Nucleus sampling threshold.
    ///   - grammar:     Optional GBNF grammar string.
    ///   - callback:    Optional streaming callback.
    /// - Returns: Generated text and token count.
    public func generate(prompt: String,
                         maxTokens: Int32 = 512,
                         temperature: Float = 0.7,
                         topP: Float = 0.9,
                         grammar: String? = nil,
                         callback: TokenCallback? = nil) throws -> GenerateResult {
        // Tokenize the prompt.
        let tokens = try tokenize(prompt, addBOS: true)

        return try generateFromTokens(
            tokens,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            grammar: grammar,
            callback: callback
        )
    }

    /// Generate from pre-tokenized prompt tokens.
    public func generateFromTokens(_ tokens: [Int32],
                                    maxTokens: Int32 = 512,
                                    temperature: Float = 0.7,
                                    topP: Float = 0.9,
                                    grammar: String? = nil,
                                    callback: TokenCallback? = nil) throws -> GenerateResult {
        let maxTextLen: Int32 = maxTokens * 32  // generous buffer
        var outText = [CChar](repeating: 0, count: Int(maxTextLen))
        var genTokens: Int32 = 0

        // Set up the C callback trampoline if needed.
        var callbackBox = callback  // mutable copy so we can take its address
        let cCallback: InferTokenCallback?
        let userData: UnsafeMutableRawPointer?

        if callback != nil {
            cCallback = { (token: Int32, piece: UnsafePointer<CChar>?, ud: UnsafeMutableRawPointer?) -> Int32 in
                guard let ud = ud else { return 0 }
                let cb = ud.assumingMemoryBound(to: LLM.TokenCallback.self).pointee
                let str = piece.map { String(cString: $0) } ?? ""
                return cb(token, str) ? 1 : 0
            }
            userData = withUnsafeMutablePointer(to: &callbackBox) {
                UnsafeMutableRawPointer($0)
            }
        } else {
            cCallback = nil
            userData = nil
        }

        let rc: Int32 = tokens.withUnsafeBufferPointer { tokenBuf in
            if let grammarStr = grammar {
                return grammarStr.withCString { cGrammar in
                    infer_llm_generate(
                        handle,
                        tokenBuf.baseAddress, Int32(tokenBuf.count),
                        maxTokens, temperature, topP,
                        cGrammar,
                        cCallback, userData,
                        &outText, maxTextLen,
                        &genTokens
                    )
                }
            } else {
                return infer_llm_generate(
                    handle,
                    tokenBuf.baseAddress, Int32(tokenBuf.count),
                    maxTokens, temperature, topP,
                    nil,
                    cCallback, userData,
                    &outText, maxTextLen,
                    &genTokens
                )
            }
        }

        guard rc == 0 else {
            throw InfergoError.runtime(lastErrorString())
        }

        let text = String(cString: outText)
        return GenerateResult(text: text, tokenCount: genTokens)
    }

    /// Number of free KV cache pages.
    public var kvPagesFree: Int32 {
        infer_llm_kv_pages_free(handle)
    }

    /// Total KV cache pages.
    public var kvPagesTotal: Int32 {
        infer_llm_kv_pages_total(handle)
    }
}

// MARK: - Tokenizer

/// A HuggingFace tokenizer loaded from a `tokenizer.json` file.
public final class Tokenizer {
    internal let handle: OpaquePointer

    /// Load a tokenizer from a `tokenizer.json` file.
    public init(path: String) throws {
        guard let h = path.withCString({ infer_tokenizer_load($0) }) else {
            throw InfergoError.load(lastErrorString())
        }
        self.handle = h
    }

    deinit {
        infer_tokenizer_destroy(handle)
    }

    /// Vocabulary size.
    public var vocabSize: Int32 {
        infer_tokenizer_vocab_size(handle)
    }

    /// Encode text into token IDs.
    ///
    /// - Parameters:
    ///   - text:             Input text.
    ///   - addSpecialTokens: Prepend/append BOS/EOS.
    ///   - maxTokens:        Upper bound on output length.
    /// - Returns: Tuple of (token IDs, attention mask).
    public func encode(_ text: String,
                       addSpecialTokens: Bool = true,
                       maxTokens: Int32 = 512) throws -> (ids: [Int32], mask: [Int32]) {
        var ids = [Int32](repeating: 0, count: Int(maxTokens))
        var mask = [Int32](repeating: 0, count: Int(maxTokens))
        let n = text.withCString { cText in
            infer_tokenizer_encode(handle, cText, addSpecialTokens ? 1 : 0,
                                   &ids, &mask, maxTokens)
        }
        guard n >= 0 else {
            throw InfergoError.runtime(lastErrorString())
        }
        return (Array(ids.prefix(Int(n))), Array(mask.prefix(Int(n))))
    }

    /// Decode token IDs back to text.
    public func decode(_ ids: [Int32], skipSpecialTokens: Bool = true) throws -> String {
        let bufSize: Int32 = 8192
        var buf = [CChar](repeating: 0, count: Int(bufSize))
        let rc = ids.withUnsafeBufferPointer { idBuf in
            infer_tokenizer_decode(handle, idBuf.baseAddress, Int32(idBuf.count),
                                   skipSpecialTokens ? 1 : 0, &buf, bufSize)
        }
        guard rc == 0 else {
            throw InfergoError.runtime(lastErrorString())
        }
        return String(cString: buf)
    }

    /// Decode a single token ID to its string piece.
    public func decodeToken(_ id: Int32) throws -> String {
        var buf = [CChar](repeating: 0, count: 256)
        let rc = infer_tokenizer_decode_token(handle, id, &buf, 256)
        guard rc == 0 else {
            throw InfergoError.runtime(lastErrorString())
        }
        return String(cString: buf)
    }
}

// MARK: - Session (ONNX)

/// An ONNX inference session.
public final class Session {
    internal let handle: OpaquePointer

    /// Create a session for the given execution provider.
    ///
    /// - Parameters:
    ///   - provider: `"cpu"`, `"cuda"`, `"tensorrt"`, `"coreml"`, or `"openvino"`.
    ///   - deviceId: GPU device index.
    public init(provider: String = "cpu", deviceId: Int32 = 0) throws {
        guard let h = provider.withCString({ infer_session_create($0, deviceId) }) else {
            throw InfergoError.runtime(lastErrorString())
        }
        self.handle = h
    }

    deinit {
        infer_session_destroy(handle)
    }

    /// Load an ONNX model file.
    public func load(path: String) throws {
        let rc = path.withCString { infer_session_load(handle, $0) }
        try check(rc)
    }

    /// Number of model inputs.
    public var numInputs: Int32 {
        infer_session_num_inputs(handle)
    }

    /// Number of model outputs.
    public var numOutputs: Int32 {
        infer_session_num_outputs(handle)
    }
}

// MARK: - Embedding

/// High-level embedding helper that wraps a ``Session`` and ``Tokenizer``.
///
///     let emb = try Embedding(modelPath: "model.onnx", tokenizerPath: "tokenizer.json")
///     let vec = try emb.embed("Hello world")
///
public final class Embedding {
    private let session: Session
    private let tokenizer: Tokenizer
    private let maxDim: Int32

    /// Create an embedding pipeline.
    ///
    /// - Parameters:
    ///   - modelPath:     Path to an ONNX embedding model.
    ///   - tokenizerPath: Path to a HuggingFace `tokenizer.json`.
    ///   - provider:      Execution provider.
    ///   - deviceId:      GPU device index.
    ///   - maxDim:        Maximum embedding dimension (buffer size).
    public init(modelPath: String,
                tokenizerPath: String,
                provider: String = "cpu",
                deviceId: Int32 = 0,
                maxDim: Int32 = 1024) throws {
        self.session = try Session(provider: provider, deviceId: deviceId)
        try self.session.load(path: modelPath)
        self.tokenizer = try Tokenizer(path: tokenizerPath)
        self.maxDim = maxDim
    }

    /// Embed a single text string.
    ///
    /// - Returns: A float array of length equal to the model's embedding dimension.
    public func embed(_ text: String) throws -> [Float] {
        var vec = [Float](repeating: 0, count: Int(maxDim))
        let dim = text.withCString { cText in
            infer_embed_pipeline(session.handle, tokenizer.handle, cText, &vec, maxDim)
        }
        guard dim > 0 else {
            throw InfergoError.runtime(lastErrorString())
        }
        return Array(vec.prefix(Int(dim)))
    }

    /// Embed a batch of texts in one call.
    ///
    /// - Returns: Array of float arrays (one per text).
    public func embedBatch(_ texts: [String]) throws -> [[Float]] {
        let n = texts.count
        var flat = [Float](repeating: 0, count: n * Int(maxDim))

        // Build a C array of const char* pointers.
        let cStrings = texts.map { strdup($0)! }
        defer { cStrings.forEach { free($0) } }

        let dim: Int32 = cStrings.withUnsafeBufferPointer { csBuf in
            // We need a mutable copy for the C API which expects const char**.
            var ptrs = csBuf.map { UnsafePointer($0) as UnsafePointer<CChar>? }
            return ptrs.withUnsafeMutableBufferPointer { ptrBuf in
                infer_embed_batch_pipeline(
                    session.handle, tokenizer.handle,
                    ptrBuf.baseAddress,
                    Int32(n),
                    &flat,
                    maxDim
                )
            }
        }
        guard dim > 0 else {
            throw InfergoError.runtime(lastErrorString())
        }
        let d = Int(dim)
        return (0..<n).map { i in
            Array(flat[(i * d)..<((i + 1) * d)])
        }
    }
}

// MARK: - VectorDB

/// A persistent HNSW vector database.
///
///     let db = try VectorDB(dimension: 384)
///     try db.insert(id: 1, vector: embedding, metadata: "hello")
///     let results = try db.search(query: embedding, k: 5)
///
public final class VectorDB {
    internal let handle: OpaquePointer

    /// Create a new vector database.
    ///
    /// - Parameters:
    ///   - dimension:      Vector dimension (must match embedding model).
    ///   - m:              Max connections per HNSW node (default 16).
    ///   - efConstruction: Search width during index build (default 200).
    public init(dimension: Int32, m: Int32 = 16, efConstruction: Int32 = 200) throws {
        guard let h = infer_vectordb_create(dimension, m, efConstruction) else {
            throw InfergoError.runtime(lastErrorString())
        }
        self.handle = h
    }

    deinit {
        infer_vectordb_free(handle)
    }

    /// Insert a vector with metadata.
    public func insert(id: Int64, vector: [Float], metadata: String? = nil) throws {
        let rc: Int32 = vector.withUnsafeBufferPointer { vecBuf in
            if let meta = metadata {
                return meta.withCString { cMeta in
                    infer_vectordb_insert(handle, id, vecBuf.baseAddress, cMeta)
                }
            } else {
                return infer_vectordb_insert(handle, id, vecBuf.baseAddress, nil)
            }
        }
        guard rc == 0 else {
            throw InfergoError.runtime(lastErrorString())
        }
    }

    /// Delete a vector by ID.
    public func delete(id: Int64) throws {
        let rc = infer_vectordb_delete(handle, id)
        guard rc == 0 else {
            throw InfergoError.runtime(lastErrorString())
        }
    }

    /// Update a vector and its metadata.
    public func update(id: Int64, vector: [Float], metadata: String? = nil) throws {
        let rc: Int32 = vector.withUnsafeBufferPointer { vecBuf in
            if let meta = metadata {
                return meta.withCString { cMeta in
                    infer_vectordb_update(handle, id, vecBuf.baseAddress, cMeta)
                }
            } else {
                return infer_vectordb_update(handle, id, vecBuf.baseAddress, nil)
            }
        }
        guard rc == 0 else {
            throw InfergoError.runtime(lastErrorString())
        }
    }

    /// A single search result.
    public struct SearchResult {
        public let id: Int64
        public let distance: Float
    }

    /// Search for the k nearest neighbors.
    ///
    /// - Parameters:
    ///   - query:          Query vector.
    ///   - k:              Number of results.
    ///   - efSearch:       Search width (higher = more accurate, slower).
    ///   - metadataFilter: Optional metadata filter string.
    /// - Returns: Array of search results sorted by distance.
    public func search(query: [Float],
                       k: Int32 = 10,
                       efSearch: Int32 = 100,
                       metadataFilter: String? = nil) throws -> [SearchResult] {
        var ids = [Int64](repeating: 0, count: Int(k))
        var distances = [Float](repeating: 0, count: Int(k))

        let n: Int32 = query.withUnsafeBufferPointer { qBuf in
            if let filter = metadataFilter {
                return filter.withCString { cFilter in
                    infer_vectordb_search(handle, qBuf.baseAddress, k, efSearch,
                                          cFilter, &ids, &distances, k)
                }
            } else {
                return infer_vectordb_search(handle, qBuf.baseAddress, k, efSearch,
                                             nil, &ids, &distances, k)
            }
        }
        guard n >= 0 else {
            throw InfergoError.runtime(lastErrorString())
        }
        return (0..<Int(n)).map { i in
            SearchResult(id: ids[i], distance: distances[i])
        }
    }

    /// Number of vectors in the database.
    public var count: Int32 {
        infer_vectordb_size(handle)
    }

    /// Save the database to a file.
    public func save(path: String) throws {
        let rc = path.withCString { infer_vectordb_save(handle, $0) }
        guard rc == 0 else {
            throw InfergoError.runtime(lastErrorString())
        }
    }

    /// Load a database from a file.
    public func load(path: String) throws {
        let rc = path.withCString { infer_vectordb_load(handle, $0) }
        guard rc == 0 else {
            throw InfergoError.runtime(lastErrorString())
        }
    }
}

// MARK: - BM25

/// A BM25 full-text search index.
///
///     let bm25 = try BM25()
///     bm25.insert(id: 1, text: "Swift is great")
///     let results = try bm25.search(query: "Swift", k: 5)
///
public final class BM25 {
    internal let handle: OpaquePointer

    /// Create a BM25 index.
    ///
    /// - Parameters:
    ///   - k1: Term frequency saturation (default 1.2).
    ///   - b:  Document length normalization (default 0.75).
    public init(k1: Float = 1.2, b: Float = 0.75) throws {
        guard let h = infer_bm25_create(k1, b) else {
            throw InfergoError.runtime(lastErrorString())
        }
        self.handle = h
    }

    deinit {
        infer_bm25_free(handle)
    }

    /// Insert a document.
    public func insert(id: Int64, text: String) {
        text.withCString { cText in
            infer_bm25_insert(handle, id, cText)
        }
    }

    /// Remove a document by ID.
    public func remove(id: Int64) {
        infer_bm25_remove(handle, id)
    }

    /// A single BM25 search result.
    public struct SearchResult {
        public let id: Int64
        public let score: Float
    }

    /// Search the index.
    ///
    /// - Parameters:
    ///   - query: Query string.
    ///   - k:     Maximum number of results.
    /// - Returns: Results sorted by BM25 score descending.
    public func search(query: String, k: Int32 = 10) throws -> [SearchResult] {
        var ids = [Int64](repeating: 0, count: Int(k))
        var scores = [Float](repeating: 0, count: Int(k))

        let n = query.withCString { cQuery in
            infer_bm25_search(handle, cQuery, k, &ids, &scores, k)
        }
        guard n >= 0 else {
            throw InfergoError.runtime(lastErrorString())
        }
        return (0..<Int(n)).map { i in
            SearchResult(id: ids[i], score: scores[i])
        }
    }

    /// Number of documents in the index.
    public var count: Int32 {
        infer_bm25_size(handle)
    }

    /// Save the index to a file.
    public func save(path: String) throws {
        let rc = path.withCString { infer_bm25_save(handle, $0) }
        guard rc == 0 else {
            throw InfergoError.runtime(lastErrorString())
        }
    }

    /// Load the index from a file.
    public func load(path: String) throws {
        let rc = path.withCString { infer_bm25_load(handle, $0) }
        guard rc == 0 else {
            throw InfergoError.runtime(lastErrorString())
        }
    }
}
