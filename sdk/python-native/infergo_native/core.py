"""infergo native Python SDK — direct ctypes bindings to libinfer_api.so"""
import ctypes
import ctypes.util
import os
import numpy as np
from pathlib import Path

def _find_lib():
    """Find libinfer_api.so"""
    env = os.environ.get("INFERGO_LIB_DIR")
    if env:
        p = os.path.join(env, "libinfer_api.so")
        if os.path.exists(p):
            return p
    for d in [
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "build", "cpp", "api"),
        "/usr/local/lib",
        "/usr/lib",
    ]:
        p = os.path.join(d, "libinfer_api.so")
        if os.path.exists(p):
            return p
    found = ctypes.util.find_library("infer_api")
    if found:
        return found
    raise OSError("Cannot find libinfer_api.so. Set INFERGO_LIB_DIR.")

_lib = ctypes.cdll.LoadLibrary(_find_lib())

# Error
_lib.infer_last_error_string.restype = ctypes.c_char_p
_lib.infer_last_error_string.argtypes = []

def _check(rc, msg=""):
    if rc < 0:
        err = _lib.infer_last_error_string()
        raise RuntimeError(f"infergo: {msg}: {err.decode() if err else 'unknown error'}")
    return rc

# --- LLM ---
_lib.infer_llm_create.restype = ctypes.c_void_p
_lib.infer_llm_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.infer_llm_destroy.argtypes = [ctypes.c_void_p]
_lib.infer_llm_vocab_size.restype = ctypes.c_int
_lib.infer_llm_vocab_size.argtypes = [ctypes.c_void_p]
_lib.infer_llm_tokenize.restype = ctypes.c_int
_lib.infer_llm_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
_lib.infer_llm_generate.restype = ctypes.c_int
_lib.infer_llm_generate.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_float, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)
]

# --- Session / Tokenizer ---
_lib.infer_session_create.restype = ctypes.c_void_p
_lib.infer_session_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
_lib.infer_session_load.restype = ctypes.c_int
_lib.infer_session_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.infer_session_destroy.argtypes = [ctypes.c_void_p]
_lib.infer_tokenizer_load.restype = ctypes.c_void_p
_lib.infer_tokenizer_load.argtypes = [ctypes.c_char_p]
_lib.infer_tokenizer_destroy.argtypes = [ctypes.c_void_p]

# --- Embedding ---
_lib.infer_embed_pipeline.restype = ctypes.c_int
_lib.infer_embed_pipeline.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.infer_embed_batch_pipeline.restype = ctypes.c_int
_lib.infer_embed_batch_pipeline.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]

# --- VectorDB ---
_lib.infer_vectordb_create.restype = ctypes.c_void_p
_lib.infer_vectordb_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.infer_vectordb_insert.restype = ctypes.c_int
_lib.infer_vectordb_insert.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_float), ctypes.c_char_p]
_lib.infer_vectordb_search.restype = ctypes.c_int
_lib.infer_vectordb_search.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.infer_vectordb_delete.restype = ctypes.c_int
_lib.infer_vectordb_delete.argtypes = [ctypes.c_void_p, ctypes.c_int64]
_lib.infer_vectordb_save.restype = ctypes.c_int
_lib.infer_vectordb_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.infer_vectordb_load.restype = ctypes.c_int
_lib.infer_vectordb_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.infer_vectordb_size.restype = ctypes.c_int
_lib.infer_vectordb_size.argtypes = [ctypes.c_void_p]
_lib.infer_vectordb_free.argtypes = [ctypes.c_void_p]

# --- BM25 ---
_lib.infer_bm25_create.restype = ctypes.c_void_p
_lib.infer_bm25_create.argtypes = [ctypes.c_float, ctypes.c_float]
_lib.infer_bm25_insert.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_char_p]
_lib.infer_bm25_search.restype = ctypes.c_int
_lib.infer_bm25_search.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.infer_bm25_size.restype = ctypes.c_int
_lib.infer_bm25_size.argtypes = [ctypes.c_void_p]
_lib.infer_bm25_free.argtypes = [ctypes.c_void_p]

# --- LoRA ---
_lib.infer_lora_load.restype = ctypes.c_void_p
_lib.infer_lora_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.infer_lora_apply.restype = ctypes.c_int
_lib.infer_lora_apply.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.infer_lora_free.argtypes = [ctypes.c_void_p]


class LLM:
    """LLM inference via direct C binding. Zero HTTP overhead."""

    def __init__(self, model_path, gpu_layers=-1, ctx_size=4096, n_seq_max=1, n_batch=2048):
        path = model_path.encode() if isinstance(model_path, str) else model_path
        self._handle = _lib.infer_llm_create(path, gpu_layers, ctx_size, n_seq_max, n_batch)
        if not self._handle:
            raise RuntimeError(f"Failed to load LLM: {_lib.infer_last_error_string().decode()}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._handle:
            _lib.infer_llm_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    @property
    def vocab_size(self):
        return _lib.infer_llm_vocab_size(self._handle)

    def tokenize(self, text, add_bos=True):
        t = text.encode() if isinstance(text, str) else text
        out = (ctypes.c_int * 4096)()
        n = _check(_lib.infer_llm_tokenize(self._handle, t, int(add_bos), out, 4096), "tokenize")
        return list(out[:n])

    def generate(self, prompt, max_tokens=128, temperature=0.7, top_p=0.9, grammar=None):
        tokens = self.tokenize(prompt)
        tok_arr = (ctypes.c_int * len(tokens))(*tokens)
        buf = ctypes.create_string_buffer(32768)
        gen = ctypes.c_int(0)
        g = grammar.encode() if grammar else None
        _check(_lib.infer_llm_generate(
            self._handle, tok_arr, len(tokens), max_tokens,
            temperature, top_p, g, None, None, buf, 32768, ctypes.byref(gen)
        ), "generate")
        return buf.value.decode("utf-8", errors="replace")


class Embedding:
    """Embedding model via direct C binding."""

    def __init__(self, model_path, tokenizer_path, provider="cpu", device_id=0):
        self._session = _lib.infer_session_create(provider.encode(), device_id)
        if not self._session:
            raise RuntimeError("Failed to create session")
        _check(_lib.infer_session_load(self._session, model_path.encode()), "load model")
        self._tokenizer = _lib.infer_tokenizer_load(tokenizer_path.encode())
        if not self._tokenizer:
            raise RuntimeError("Failed to load tokenizer")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._tokenizer:
            _lib.infer_tokenizer_destroy(self._tokenizer)
            self._tokenizer = None
        if self._session:
            _lib.infer_session_destroy(self._session)
            self._session = None

    def __del__(self):
        self.close()

    def embed(self, text):
        out = (ctypes.c_float * 2048)()
        dim = _check(_lib.infer_embed_pipeline(self._session, self._tokenizer, text.encode(), out, 2048), "embed")
        return np.array(out[:dim], dtype=np.float32)

    def embed_batch(self, texts):
        n = len(texts)
        c_texts = (ctypes.c_char_p * n)(*[t.encode() for t in texts])
        out = (ctypes.c_float * (n * 2048))()
        dim = _check(_lib.infer_embed_batch_pipeline(self._session, self._tokenizer, c_texts, n, out, 2048), "embed_batch")
        return np.array(out[:n * dim], dtype=np.float32).reshape(n, dim)


class VectorDB:
    """HNSW vector database via direct C binding."""

    def __init__(self, dim=384, M=16, ef_construction=200):
        self._handle = _lib.infer_vectordb_create(dim, M, ef_construction)
        if not self._handle:
            raise RuntimeError("Failed to create VectorDB")
        self._dim = dim

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._handle:
            _lib.infer_vectordb_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __len__(self):
        return _lib.infer_vectordb_size(self._handle)

    def insert(self, id, vector, metadata=""):
        vec = np.asarray(vector, dtype=np.float32)
        _check(_lib.infer_vectordb_insert(self._handle, id, vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                           metadata.encode()), "insert")

    def search(self, query, k=10, ef_search=50, metadata_filter=None):
        q = np.asarray(query, dtype=np.float32)
        ids = (ctypes.c_int64 * k)()
        dists = (ctypes.c_float * k)()
        filt = metadata_filter.encode() if metadata_filter else None
        n = _check(_lib.infer_vectordb_search(self._handle, q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                               k, ef_search, filt, ids, dists, k), "search")
        return [(int(ids[i]), float(dists[i])) for i in range(n)]

    def delete(self, id):
        _check(_lib.infer_vectordb_delete(self._handle, id), "delete")

    def save(self, path):
        _check(_lib.infer_vectordb_save(self._handle, path.encode()), "save")

    def load(self, path):
        _check(_lib.infer_vectordb_load(self._handle, path.encode()), "load")


class BM25:
    """BM25 full-text search via direct C binding."""

    def __init__(self, k1=1.2, b=0.75):
        self._handle = _lib.infer_bm25_create(k1, b)
        if not self._handle:
            raise RuntimeError("Failed to create BM25 index")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._handle:
            _lib.infer_bm25_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __len__(self):
        return _lib.infer_bm25_size(self._handle)

    def insert(self, id, text):
        _lib.infer_bm25_insert(self._handle, id, text.encode())

    def search(self, query, k=10):
        ids = (ctypes.c_int64 * k)()
        scores = (ctypes.c_float * k)()
        n = _check(_lib.infer_bm25_search(self._handle, query.encode(), k, ids, scores, k), "search")
        return [(int(ids[i]), float(scores[i])) for i in range(n)]


class LoRA:
    """LoRA adapter management."""

    def __init__(self, llm, lora_path):
        self._llm = llm._handle
        self._handle = _lib.infer_lora_load(self._llm, lora_path.encode())
        if not self._handle:
            raise RuntimeError(f"Failed to load LoRA: {_lib.infer_last_error_string().decode()}")

    def apply(self, scale=1.0):
        adapter = ctypes.c_void_p(self._handle)
        s = ctypes.c_float(scale)
        _check(_lib.infer_lora_apply(self._llm, ctypes.byref(adapter), ctypes.byref(s), 1), "apply")

    def remove(self):
        _check(_lib.infer_lora_apply(self._llm, None, None, 0), "remove")

    def close(self):
        if self._handle:
            _lib.infer_lora_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()
