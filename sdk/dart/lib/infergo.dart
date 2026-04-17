/// infergo Dart SDK — dart:ffi bindings to libinfer_api.so
library infergo;

import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

DynamicLibrary _openLib() {
  final env = Platform.environment['INFERGO_LIB_DIR'];
  if (env != null) return DynamicLibrary.open('$env/libinfer_api.so');
  try { return DynamicLibrary.open('libinfer_api.so'); } catch (_) {}
  return DynamicLibrary.open('/usr/local/lib/libinfer_api.so');
}

final _lib = _openLib();

// FFI typedefs
typedef _LlmCreateC = Pointer<Void> Function(Pointer<Utf8>, Int32, Int32, Int32, Int32);
typedef _LlmCreateDart = Pointer<Void> Function(Pointer<Utf8>, int, int, int, int);
typedef _LlmDestroyC = Void Function(Pointer<Void>);
typedef _LlmDestroyDart = void Function(Pointer<Void>);
typedef _LlmVocabC = Int32 Function(Pointer<Void>);
typedef _LlmVocabDart = int Function(Pointer<Void>);
typedef _TokenizeC = Int32 Function(Pointer<Void>, Pointer<Utf8>, Int32, Pointer<Int32>, Int32);
typedef _TokenizeDart = int Function(Pointer<Void>, Pointer<Utf8>, int, Pointer<Int32>, int);
typedef _GenerateC = Int32 Function(Pointer<Void>, Pointer<Int32>, Int32, Int32, Float, Float, Pointer<Utf8>, Pointer<Void>, Pointer<Void>, Pointer<Utf8>, Int32, Pointer<Int32>);
typedef _GenerateDart = int Function(Pointer<Void>, Pointer<Int32>, int, int, double, double, Pointer<Utf8>, Pointer<Void>, Pointer<Void>, Pointer<Utf8>, int, Pointer<Int32>);
typedef _LastErrorC = Pointer<Utf8> Function();
typedef _LastErrorDart = Pointer<Utf8> Function();
typedef _SessionCreateC = Pointer<Void> Function(Pointer<Utf8>, Int32);
typedef _SessionCreateDart = Pointer<Void> Function(Pointer<Utf8>, int);
typedef _SessionLoadC = Int32 Function(Pointer<Void>, Pointer<Utf8>);
typedef _SessionLoadDart = int Function(Pointer<Void>, Pointer<Utf8>);
typedef _SessionDestroyC = Void Function(Pointer<Void>);
typedef _SessionDestroyDart = void Function(Pointer<Void>);
typedef _TokLoadC = Pointer<Void> Function(Pointer<Utf8>);
typedef _TokLoadDart = Pointer<Void> Function(Pointer<Utf8>);
typedef _TokDestroyC = Void Function(Pointer<Void>);
typedef _TokDestroyDart = void Function(Pointer<Void>);
typedef _EmbedC = Int32 Function(Pointer<Void>, Pointer<Void>, Pointer<Utf8>, Pointer<Float>, Int32);
typedef _EmbedDart = int Function(Pointer<Void>, Pointer<Void>, Pointer<Utf8>, Pointer<Float>, int);
typedef _VdbCreateC = Pointer<Void> Function(Int32, Int32, Int32);
typedef _VdbCreateDart = Pointer<Void> Function(int, int, int);
typedef _VdbInsertC = Int32 Function(Pointer<Void>, Int64, Pointer<Float>, Pointer<Utf8>);
typedef _VdbInsertDart = int Function(Pointer<Void>, int, Pointer<Float>, Pointer<Utf8>);
typedef _VdbSearchC = Int32 Function(Pointer<Void>, Pointer<Float>, Int32, Int32, Pointer<Utf8>, Pointer<Int64>, Pointer<Float>, Int32);
typedef _VdbSearchDart = int Function(Pointer<Void>, Pointer<Float>, int, int, Pointer<Utf8>, Pointer<Int64>, Pointer<Float>, int);
typedef _VdbFreeC = Void Function(Pointer<Void>);
typedef _VdbFreeDart = void Function(Pointer<Void>);
typedef _Bm25CreateC = Pointer<Void> Function(Float, Float);
typedef _Bm25CreateDart = Pointer<Void> Function(double, double);
typedef _Bm25InsertC = Void Function(Pointer<Void>, Int64, Pointer<Utf8>);
typedef _Bm25InsertDart = void Function(Pointer<Void>, int, Pointer<Utf8>);
typedef _Bm25SearchC = Int32 Function(Pointer<Void>, Pointer<Utf8>, Int32, Pointer<Int64>, Pointer<Float>, Int32);
typedef _Bm25SearchDart = int Function(Pointer<Void>, Pointer<Utf8>, int, Pointer<Int64>, Pointer<Float>, int);
typedef _Bm25FreeC = Void Function(Pointer<Void>);
typedef _Bm25FreeDart = void Function(Pointer<Void>);

final _lastError = _lib.lookupFunction<_LastErrorC, _LastErrorDart>('infer_last_error_string');
final _llmCreate = _lib.lookupFunction<_LlmCreateC, _LlmCreateDart>('infer_llm_create');
final _llmDestroy = _lib.lookupFunction<_LlmDestroyC, _LlmDestroyDart>('infer_llm_destroy');
final _llmVocab = _lib.lookupFunction<_LlmVocabC, _LlmVocabDart>('infer_llm_vocab_size');
final _tokenize = _lib.lookupFunction<_TokenizeC, _TokenizeDart>('infer_llm_tokenize');
final _generate = _lib.lookupFunction<_GenerateC, _GenerateDart>('infer_llm_generate');
final _sessionCreate = _lib.lookupFunction<_SessionCreateC, _SessionCreateDart>('infer_session_create');
final _sessionLoad = _lib.lookupFunction<_SessionLoadC, _SessionLoadDart>('infer_session_load');
final _sessionDestroy = _lib.lookupFunction<_SessionDestroyC, _SessionDestroyDart>('infer_session_destroy');
final _tokLoad = _lib.lookupFunction<_TokLoadC, _TokLoadDart>('infer_tokenizer_load');
final _tokDestroy = _lib.lookupFunction<_TokDestroyC, _TokDestroyDart>('infer_tokenizer_destroy');
final _embedPipeline = _lib.lookupFunction<_EmbedC, _EmbedDart>('infer_embed_pipeline');
final _vdbCreate = _lib.lookupFunction<_VdbCreateC, _VdbCreateDart>('infer_vectordb_create');
final _vdbInsert = _lib.lookupFunction<_VdbInsertC, _VdbInsertDart>('infer_vectordb_insert');
final _vdbSearch = _lib.lookupFunction<_VdbSearchC, _VdbSearchDart>('infer_vectordb_search');
final _vdbFree = _lib.lookupFunction<_VdbFreeC, _VdbFreeDart>('infer_vectordb_free');
final _bm25Create = _lib.lookupFunction<_Bm25CreateC, _Bm25CreateDart>('infer_bm25_create');
final _bm25Insert = _lib.lookupFunction<_Bm25InsertC, _Bm25InsertDart>('infer_bm25_insert');
final _bm25Search = _lib.lookupFunction<_Bm25SearchC, _Bm25SearchDart>('infer_bm25_search');
final _bm25Free = _lib.lookupFunction<_Bm25FreeC, _Bm25FreeDart>('infer_bm25_free');

class InfergoException implements Exception {
  final String message;
  InfergoException(this.message);
  factory InfergoException.last([String prefix = '']) {
    final err = _lastError();
    return InfergoException('$prefix${err.toDartString()}');
  }
  @override String toString() => 'InfergoException: $message';
}

class LLM {
  Pointer<Void>? _handle;

  LLM(String modelPath, {int gpuLayers = -1, int ctxSize = 4096, int nSeqMax = 1, int nBatch = 2048}) {
    final p = modelPath.toNativeUtf8();
    _handle = _llmCreate(p, gpuLayers, ctxSize, nSeqMax, nBatch);
    calloc.free(p);
    if (_handle == nullptr) throw InfergoException.last('LLM load: ');
  }

  int get vocabSize => _llmVocab(_handle!);

  List<int> tokenize(String text, {bool addBos = true}) {
    final t = text.toNativeUtf8();
    final out = calloc<Int32>(4096);
    final n = _tokenize(_handle!, t, addBos ? 1 : 0, out, 4096);
    calloc.free(t);
    if (n < 0) { calloc.free(out); throw InfergoException('tokenize failed'); }
    final result = List<int>.generate(n, (i) => out[i]);
    calloc.free(out);
    return result;
  }

  String generate(String prompt, {int maxTokens = 128, double temperature = 0.7, double topP = 0.9}) {
    final tokens = tokenize(prompt);
    final tokPtr = calloc<Int32>(tokens.length);
    for (var i = 0; i < tokens.length; i++) tokPtr[i] = tokens[i];
    final buf = calloc<Utf8>(32768);
    final gen = calloc<Int32>(1);
    final rc = _generate(_handle!, tokPtr, tokens.length, maxTokens, temperature, topP, nullptr, nullptr, nullptr, buf, 32768, gen);
    calloc.free(tokPtr);
    if (rc < 0) { calloc.free(buf); calloc.free(gen); throw InfergoException.last('generate: '); }
    final result = buf.toDartString();
    calloc.free(buf); calloc.free(gen);
    return result;
  }

  void close() {
    if (_handle != null) { _llmDestroy(_handle!); _handle = null; }
  }
}

class Embedding {
  Pointer<Void>? _session;
  Pointer<Void>? _tokenizer;

  Embedding(String modelPath, String tokenizerPath, {String provider = 'cpu', int deviceId = 0}) {
    final prov = provider.toNativeUtf8();
    _session = _sessionCreate(prov, deviceId);
    calloc.free(prov);
    final mp = modelPath.toNativeUtf8();
    _sessionLoad(_session!, mp);
    calloc.free(mp);
    final tp = tokenizerPath.toNativeUtf8();
    _tokenizer = _tokLoad(tp);
    calloc.free(tp);
  }

  Float32List embed(String text) {
    final t = text.toNativeUtf8();
    final out = calloc<Float>(2048);
    final dim = _embedPipeline(_session!, _tokenizer!, t, out, 2048);
    calloc.free(t);
    if (dim < 0) { calloc.free(out); throw InfergoException.last('embed: '); }
    final result = Float32List(dim);
    for (var i = 0; i < dim; i++) result[i] = out[i];
    calloc.free(out);
    return result;
  }

  void close() {
    if (_tokenizer != null) { _tokDestroy(_tokenizer!); _tokenizer = null; }
    if (_session != null) { _sessionDestroy(_session!); _session = null; }
  }
}

class VectorDB {
  Pointer<Void>? _handle;
  final int dim;

  VectorDB({this.dim = 384, int m = 16, int efConstruction = 200}) {
    _handle = _vdbCreate(dim, m, efConstruction);
  }

  void insert(int id, Float32List vector, {String metadata = ''}) {
    final v = calloc<Float>(vector.length);
    for (var i = 0; i < vector.length; i++) v[i] = vector[i];
    final m = metadata.toNativeUtf8();
    _vdbInsert(_handle!, id, v, m);
    calloc.free(v); calloc.free(m);
  }

  List<Map<String, dynamic>> search(Float32List query, {int k = 10, int efSearch = 50}) {
    final q = calloc<Float>(query.length);
    for (var i = 0; i < query.length; i++) q[i] = query[i];
    final ids = calloc<Int64>(k);
    final dists = calloc<Float>(k);
    final n = _vdbSearch(_handle!, q, k, efSearch, nullptr, ids, dists, k);
    calloc.free(q);
    final results = List.generate(n, (i) => {'id': ids[i], 'distance': dists[i]});
    calloc.free(ids); calloc.free(dists);
    return results;
  }

  void close() { if (_handle != null) { _vdbFree(_handle!); _handle = null; } }
}

class BM25 {
  Pointer<Void>? _handle;

  BM25({double k1 = 1.2, double b = 0.75}) {
    _handle = _bm25Create(k1, b);
  }

  void insert(int id, String text) {
    final t = text.toNativeUtf8();
    _bm25Insert(_handle!, id, t);
    calloc.free(t);
  }

  List<Map<String, dynamic>> search(String query, {int k = 10}) {
    final q = query.toNativeUtf8();
    final ids = calloc<Int64>(k);
    final scores = calloc<Float>(k);
    final n = _bm25Search(_handle!, q, k, ids, scores, k);
    calloc.free(q);
    final results = List.generate(n, (i) => {'id': ids[i], 'score': scores[i]});
    calloc.free(ids); calloc.free(scores);
    return results;
  }

  void close() { if (_handle != null) { _bm25Free(_handle!); _handle = null; } }
}
