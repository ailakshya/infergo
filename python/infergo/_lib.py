"""
_lib.py — locate and load libinfer_api.so / libinfer_api.dylib via ctypes,
then declare all function prototypes from infer_api.h so callers get type safety.

Single export: ``lib``  (a ctypes.CDLL instance)
"""

import ctypes
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Locate the shared library
# ---------------------------------------------------------------------------

_LIB_NAME_LINUX = "libinfer_api.so"
_LIB_NAME_MACOS = "libinfer_api.dylib"
_LIB_NAME = _LIB_NAME_MACOS if sys.platform == "darwin" else _LIB_NAME_LINUX

_SEARCH_PATHS = [
    # 1. Explicit override via environment variable
    os.environ.get("INFERGO_LIB", ""),
    # 2. Same directory as this package file
    str(Path(__file__).parent / _LIB_NAME),
    # 3. Build output relative to the package root (dev builds)
    str(Path(__file__).parent / ".." / ".." / "build" / "cpp" / "api" / _LIB_NAME),
    # 4. System-wide install prefix
    str(Path("/usr/local/lib/infergo") / _LIB_NAME),
    # 5. Generic system library paths — let ctypes resolve by name only
    _LIB_NAME,
]


def _load_library() -> ctypes.CDLL:
    for candidate in _SEARCH_PATHS:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    raise OSError(
        f"Cannot find '{_LIB_NAME}'. "
        "Set the INFERGO_LIB environment variable to its full path, "
        "place it next to the infergo Python package, "
        "build it to build/cpp/api/, or install it to /usr/local/lib/infergo/."
    )


lib: ctypes.CDLL = _load_library()

# ---------------------------------------------------------------------------
# 2. Declare function prototypes
# ---------------------------------------------------------------------------
# Shorthand aliases
_vp  = ctypes.c_void_p
_pvp = ctypes.POINTER(ctypes.c_void_p)
_cp  = ctypes.c_char_p
_ci  = ctypes.c_int
_pi  = ctypes.POINTER(ctypes.c_int)
_cf  = ctypes.c_float
_pf  = ctypes.POINTER(ctypes.c_float)
_pcv = ctypes.c_void_p   # const void* (opaque read-only pointer — same underlying type)

# ------------------------------------------------------------------
# Error
# ------------------------------------------------------------------
lib.infer_last_error_string.argtypes = []
lib.infer_last_error_string.restype  = _cp

# ------------------------------------------------------------------
# Tensor
# ------------------------------------------------------------------
lib.infer_tensor_alloc_cpu.argtypes = [_pi, _ci, _ci]
lib.infer_tensor_alloc_cpu.restype  = _vp

lib.infer_tensor_alloc_cuda.argtypes = [_pi, _ci, _ci, _ci]
lib.infer_tensor_alloc_cuda.restype  = _vp

lib.infer_tensor_free.argtypes = [_vp]
lib.infer_tensor_free.restype  = None

lib.infer_tensor_data_ptr.argtypes = [_vp]
lib.infer_tensor_data_ptr.restype  = _vp

lib.infer_tensor_nbytes.argtypes = [_vp]
lib.infer_tensor_nbytes.restype  = _ci

lib.infer_tensor_nelements.argtypes = [_vp]
lib.infer_tensor_nelements.restype  = _ci

lib.infer_tensor_shape.argtypes = [_vp, _pi, _ci]
lib.infer_tensor_shape.restype  = _ci

lib.infer_tensor_dtype.argtypes = [_vp]
lib.infer_tensor_dtype.restype  = _ci

lib.infer_tensor_to_device.argtypes = [_vp, _ci]
lib.infer_tensor_to_device.restype  = _ci

lib.infer_tensor_to_host.argtypes = [_vp]
lib.infer_tensor_to_host.restype  = _ci

lib.infer_tensor_copy_from.argtypes = [_vp, _pcv, _ci]
lib.infer_tensor_copy_from.restype  = _ci

# ------------------------------------------------------------------
# ONNX session
# ------------------------------------------------------------------
lib.infer_session_create.argtypes = [_cp, _ci]
lib.infer_session_create.restype  = _vp

lib.infer_session_load.argtypes = [_vp, _cp]
lib.infer_session_load.restype  = _ci

lib.infer_session_num_inputs.argtypes = [_vp]
lib.infer_session_num_inputs.restype  = _ci

lib.infer_session_num_outputs.argtypes = [_vp]
lib.infer_session_num_outputs.restype  = _ci

lib.infer_session_input_name.argtypes = [_vp, _ci, _cp, _ci]
lib.infer_session_input_name.restype  = _ci

lib.infer_session_output_name.argtypes = [_vp, _ci, _cp, _ci]
lib.infer_session_output_name.restype  = _ci

lib.infer_session_run.argtypes = [_vp, _pvp, _ci, _pvp, _ci]
lib.infer_session_run.restype  = _ci

lib.infer_session_destroy.argtypes = [_vp]
lib.infer_session_destroy.restype  = None

# ------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------
lib.infer_tokenizer_load.argtypes = [_cp]
lib.infer_tokenizer_load.restype  = _vp

lib.infer_tokenizer_encode.argtypes = [_vp, _cp, _ci, _pi, _pi, _ci]
lib.infer_tokenizer_encode.restype  = _ci

lib.infer_tokenizer_decode.argtypes = [_vp, _pi, _ci, _ci, _cp, _ci]
lib.infer_tokenizer_decode.restype  = _ci

lib.infer_tokenizer_decode_token.argtypes = [_vp, _ci, _cp, _ci]
lib.infer_tokenizer_decode_token.restype  = _ci

lib.infer_tokenizer_vocab_size.argtypes = [_vp]
lib.infer_tokenizer_vocab_size.restype  = _ci

lib.infer_tokenizer_destroy.argtypes = [_vp]
lib.infer_tokenizer_destroy.restype  = None

# ------------------------------------------------------------------
# LLM
# ------------------------------------------------------------------
lib.infer_llm_create.argtypes = [_cp, _ci, _ci, _ci, _ci]
lib.infer_llm_create.restype  = _vp

lib.infer_llm_create_split.argtypes = [_cp, _ci, _ci, _ci, _ci, _pf, _ci]
lib.infer_llm_create_split.restype  = _vp

lib.infer_llm_create_pipeline.argtypes = [_cp, _ci, _ci, _ci, _ci, _ci]
lib.infer_llm_create_pipeline.restype  = _vp

lib.infer_llm_destroy.argtypes = [_vp]
lib.infer_llm_destroy.restype  = None

lib.infer_llm_vocab_size.argtypes = [_vp]
lib.infer_llm_vocab_size.restype  = _ci

lib.infer_llm_bos.argtypes = [_vp]
lib.infer_llm_bos.restype  = _ci

lib.infer_llm_eos.argtypes = [_vp]
lib.infer_llm_eos.restype  = _ci

lib.infer_llm_is_eog.argtypes = [_vp, _ci]
lib.infer_llm_is_eog.restype  = _ci

lib.infer_llm_tokenize.argtypes = [_vp, _cp, _ci, _pi, _ci]
lib.infer_llm_tokenize.restype  = _ci

lib.infer_llm_token_to_piece.argtypes = [_vp, _ci, _cp, _ci]
lib.infer_llm_token_to_piece.restype  = _ci

# ------------------------------------------------------------------
# Sequence
# ------------------------------------------------------------------
lib.infer_seq_create.argtypes = [_vp, _pi, _ci]
lib.infer_seq_create.restype  = _vp

lib.infer_seq_destroy.argtypes = [_vp]
lib.infer_seq_destroy.restype  = None

lib.infer_seq_is_done.argtypes = [_vp]
lib.infer_seq_is_done.restype  = _ci

lib.infer_seq_position.argtypes = [_vp]
lib.infer_seq_position.restype  = _ci

lib.infer_seq_slot_id.argtypes = [_vp]
lib.infer_seq_slot_id.restype  = _ci

lib.infer_seq_append_token.argtypes = [_vp, _ci]
lib.infer_seq_append_token.restype  = None

lib.infer_seq_next_tokens.argtypes = [_vp, _pi, _ci]
lib.infer_seq_next_tokens.restype  = _ci

lib.infer_llm_batch_decode.argtypes = [_vp, _pvp, _ci]
lib.infer_llm_batch_decode.restype  = _ci

lib.infer_seq_get_logits.argtypes = [_vp, _pf, _ci]
lib.infer_seq_get_logits.restype   = _ci

# ------------------------------------------------------------------
# Preprocess
# ------------------------------------------------------------------
lib.infer_preprocess_decode_image.argtypes = [_pcv, _ci]
lib.infer_preprocess_decode_image.restype  = _vp

lib.infer_preprocess_letterbox.argtypes = [_vp, _ci, _ci]
lib.infer_preprocess_letterbox.restype  = _vp

lib.infer_preprocess_normalize.argtypes = [_vp, _cf, _pf, _pf]
lib.infer_preprocess_normalize.restype  = _vp

lib.infer_preprocess_stack_batch.argtypes = [ctypes.POINTER(_pcv), _ci]
lib.infer_preprocess_stack_batch.restype  = _vp

# ------------------------------------------------------------------
# Postprocess
# ------------------------------------------------------------------
lib.infer_postprocess_classify.argtypes = [_vp, _ci, _vp]
lib.infer_postprocess_classify.restype  = _ci

lib.infer_postprocess_nms.argtypes = [_vp, _cf, _cf, _vp, _ci]
lib.infer_postprocess_nms.restype  = _ci

lib.infer_postprocess_normalize_embedding.argtypes = [_vp]
lib.infer_postprocess_normalize_embedding.restype  = _ci

# ------------------------------------------------------------------
# KV cache
# ------------------------------------------------------------------
lib.infer_llm_kv_serialize.argtypes = [_vp, _ci, _vp, _ci]
lib.infer_llm_kv_serialize.restype  = _ci

lib.infer_llm_kv_deserialize.argtypes = [_vp, _ci, _pcv, _ci]
lib.infer_llm_kv_deserialize.restype  = _ci

lib.infer_llm_kv_pages_free.argtypes = [_vp]
lib.infer_llm_kv_pages_free.restype  = _ci

lib.infer_llm_kv_pages_total.argtypes = [_vp]
lib.infer_llm_kv_pages_total.restype  = _ci

lib.infer_llm_kv_page_size.argtypes = [_vp]
lib.infer_llm_kv_page_size.restype  = _ci
