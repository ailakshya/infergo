"""
infergo.tokenizer — Python wrapper around the infergo tokenizer C API.
"""

import ctypes

from ._lib import lib
from ._types import INFER_OK, _check

# ---------------------------------------------------------------------------
# C function signatures
# ---------------------------------------------------------------------------

# void* infer_tokenizer_load(const char* path)
lib.infer_tokenizer_load.restype = ctypes.c_void_p
lib.infer_tokenizer_load.argtypes = [ctypes.c_char_p]

# int infer_tokenizer_encode(tok, text, add_special, out_ids, out_mask, max_tokens)
lib.infer_tokenizer_encode.restype = ctypes.c_int
lib.infer_tokenizer_encode.argtypes = [
    ctypes.c_void_p,   # tok
    ctypes.c_char_p,   # text
    ctypes.c_int,      # add_special
    ctypes.POINTER(ctypes.c_int),  # out_ids
    ctypes.POINTER(ctypes.c_int),  # out_mask
    ctypes.c_int,      # max_tokens
]

# int infer_tokenizer_decode(tok, ids, n_ids, skip_special, out_buf, buf_size)
lib.infer_tokenizer_decode.restype = ctypes.c_int
lib.infer_tokenizer_decode.argtypes = [
    ctypes.c_void_p,              # tok
    ctypes.POINTER(ctypes.c_int), # ids
    ctypes.c_int,                 # n_ids
    ctypes.c_int,                 # skip_special
    ctypes.c_char_p,              # out_buf
    ctypes.c_int,                 # buf_size
]

# int infer_tokenizer_decode_token(tok, id, out_buf, buf_size)
lib.infer_tokenizer_decode_token.restype = ctypes.c_int
lib.infer_tokenizer_decode_token.argtypes = [
    ctypes.c_void_p,  # tok
    ctypes.c_int,     # id
    ctypes.c_char_p,  # out_buf
    ctypes.c_int,     # buf_size
]

# int infer_tokenizer_vocab_size(tok)
lib.infer_tokenizer_vocab_size.restype = ctypes.c_int
lib.infer_tokenizer_vocab_size.argtypes = [ctypes.c_void_p]

# void infer_tokenizer_destroy(tok)
lib.infer_tokenizer_destroy.restype = None
lib.infer_tokenizer_destroy.argtypes = [ctypes.c_void_p]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DECODE_BUF_SIZE = 65536  # 64 KiB — enough for any reasonable decoded text
_TOKEN_BUF_SIZE = 256     # per-token decode buffer


class Tokenizer:
    """Wraps an infergo tokenizer loaded from a tokenizer.json file."""

    def __init__(self, tokenizer_json_path: str) -> None:
        handle = lib.infer_tokenizer_load(tokenizer_json_path.encode())
        if handle is None:
            raise RuntimeError(
                f"infer_tokenizer_load failed: could not load '{tokenizer_json_path}'"
            )
        self._handle = ctypes.c_void_p(handle)
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_tokens: int = 512,
    ) -> tuple[list[int], list[int]]:
        """Encode *text* into token IDs and an attention mask.

        Returns:
            (token_ids, attention_mask) — both lists have length <= max_tokens.
        """
        self._require_open()
        out_ids = (ctypes.c_int * max_tokens)()
        out_mask = (ctypes.c_int * max_tokens)()
        n = lib.infer_tokenizer_encode(
            self._handle,
            text.encode("utf-8"),
            ctypes.c_int(1 if add_special_tokens else 0),
            out_ids,
            out_mask,
            ctypes.c_int(max_tokens),
        )
        if n < 0:
            raise RuntimeError("infer_tokenizer_encode failed")
        ids = list(out_ids[:n])
        mask = list(out_mask[:n])
        return ids, mask

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode a list of token IDs back to a string."""
        self._require_open()
        n = len(ids)
        arr = (ctypes.c_int * n)(*ids)
        buf = ctypes.create_string_buffer(_DECODE_BUF_SIZE)
        rc = lib.infer_tokenizer_decode(
            self._handle,
            arr,
            ctypes.c_int(n),
            ctypes.c_int(1 if skip_special_tokens else 0),
            buf,
            ctypes.c_int(_DECODE_BUF_SIZE),
        )
        _check(rc, "infer_tokenizer_decode")
        return buf.value.decode("utf-8", errors="replace")

    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID to its string piece."""
        self._require_open()
        buf = ctypes.create_string_buffer(_TOKEN_BUF_SIZE)
        rc = lib.infer_tokenizer_decode_token(
            self._handle,
            ctypes.c_int(token_id),
            buf,
            ctypes.c_int(_TOKEN_BUF_SIZE),
        )
        _check(rc, "infer_tokenizer_decode_token")
        return buf.value.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size of the loaded tokenizer."""
        self._require_open()
        return lib.infer_tokenizer_vocab_size(self._handle)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the native tokenizer handle. Safe to call multiple times."""
        if not self._closed and self._handle is not None:
            lib.infer_tokenizer_destroy(self._handle)
            self._closed = True
            self._handle = None

    def __enter__(self) -> "Tokenizer":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_open(self) -> None:
        if self._closed or self._handle is None:
            raise RuntimeError("Tokenizer is already closed")
