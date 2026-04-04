"""
infergo.llm — High-level LLM class wrapping the infergo C API via ctypes.
"""

from __future__ import annotations

import ctypes
from typing import Iterator

from ._lib import lib
from ._types import _check

__all__ = ["LLM"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _encode(text: str) -> bytes:
    return text.encode("utf-8")


def _sample_token(
    logits: ctypes.Array,
    vocab_size: int,
    temperature: float,
    top_p: float,
) -> int:
    """
    Sample a token from logits using numpy (C-backed, vectorised).

    If temperature == 0, returns the argmax (greedy).
    Otherwise applies temperature scaling + softmax + top-p nucleus sampling.
    All heavy operations run in numpy C loops — no Python iteration over vocab.
    """
    import numpy as np

    # Zero-copy view directly into the ctypes buffer — no data is copied.
    arr = np.frombuffer(logits, dtype=np.float32)

    if temperature == 0.0:
        return int(np.argmax(arr))

    # Temperature scaling + numerically-stable softmax (all in C via numpy).
    arr = arr / temperature
    arr -= arr.max()          # subtract max for numerical stability
    np.exp(arr, out=arr)      # in-place exp
    arr /= arr.sum()          # normalise to probabilities

    # Top-p nucleus sampling: sort descending, cumsum, cut at top_p.
    sorted_idx = np.argsort(arr)[::-1]          # indices sorted by prob desc
    sorted_probs = arr[sorted_idx]
    cumsum = np.cumsum(sorted_probs)
    # Keep tokens up to and including the one that pushes cumsum past top_p.
    cutoff = int(np.searchsorted(cumsum, top_p)) + 1
    nucleus_idx = sorted_idx[:cutoff]
    nucleus_probs = sorted_probs[:cutoff]
    nucleus_probs = nucleus_probs / nucleus_probs.sum()  # re-normalise

    return int(np.random.choice(nucleus_idx, p=nucleus_probs))


# ---------------------------------------------------------------------------
# LLM class
# ---------------------------------------------------------------------------

class LLM:
    """
    High-level interface to an infergo LLM.

    Parameters
    ----------
    model_path:
        Path to the GGUF model file.
    gpu_layers:
        Number of layers to offload to GPU (default 999 = all).
    ctx_size:
        KV-cache context size in tokens.
    max_seqs:
        Maximum concurrent sequences.
    n_batch:
        Logical batch size for prompt processing.
    tensor_split:
        Optional list of floats for multi-GPU tensor splitting.  The length
        determines the number of GPUs used.
    pipeline_stages:
        Number of pipeline-parallel stages (>1 activates pipeline mode).
    """

    def __init__(
        self,
        model_path: str,
        *,
        gpu_layers: int = 999,
        ctx_size: int = 16384,
        max_seqs: int = 16,
        n_batch: int = 2048,
        tensor_split: list[float] | None = None,
        pipeline_stages: int = 1,
    ) -> None:
        path_bytes = _encode(model_path)

        if tensor_split is not None and len(tensor_split) > 0:
            n_split = len(tensor_split)
            split_arr = (ctypes.c_float * n_split)(*tensor_split)
            split_ptr = ctypes.cast(split_arr, ctypes.POINTER(ctypes.c_float))
            llm = lib.infer_llm_create_split(
                path_bytes,
                ctypes.c_int(gpu_layers),
                ctypes.c_int(ctx_size),
                ctypes.c_int(max_seqs),
                ctypes.c_int(n_batch),
                split_ptr,
                ctypes.c_int(n_split),
            )
        elif pipeline_stages > 1:
            llm = lib.infer_llm_create_pipeline(
                path_bytes,
                ctypes.c_int(gpu_layers),
                ctypes.c_int(ctx_size),
                ctypes.c_int(max_seqs),
                ctypes.c_int(n_batch),
                ctypes.c_int(pipeline_stages),
            )
        else:
            llm = lib.infer_llm_create(
                path_bytes,
                ctypes.c_int(gpu_layers),
                ctypes.c_int(ctx_size),
                ctypes.c_int(max_seqs),
                ctypes.c_int(n_batch),
            )

        if not llm:
            err = lib.infer_last_error_string()
            msg = err.decode() if err else "unknown error"
            raise RuntimeError(f"infer_llm_create failed: {msg}")

        self._llm = llm
        # Cache vocab size — used frequently in sampling loops.
        self._vocab_size: int = lib.infer_llm_vocab_size(self._llm)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the model vocabulary."""
        return self._vocab_size

    @property
    def bos_token(self) -> int:
        """Beginning-of-sequence token id."""
        return lib.infer_llm_bos(self._llm)

    @property
    def eos_token(self) -> int:
        """End-of-sequence token id."""
        return lib.infer_llm_eos(self._llm)

    # ------------------------------------------------------------------
    # Token utilities
    # ------------------------------------------------------------------

    def is_eog(self, token: int) -> bool:
        """Return True if *token* is an end-of-generation token."""
        return bool(lib.infer_llm_is_eog(self._llm, ctypes.c_int(token)))

    def tokenize(self, text: str, add_bos: bool = True) -> list[int]:
        """
        Tokenize *text* and return a list of integer token ids.

        Raises RuntimeError on failure.
        """
        text_bytes = _encode(text)
        # Allocate a buffer large enough for the worst case.
        max_tokens = len(text_bytes) + 8
        ids_arr = (ctypes.c_int * max_tokens)()
        n = lib.infer_llm_tokenize(
            self._llm,
            text_bytes,
            ctypes.c_int(1 if add_bos else 0),
            ids_arr,
            ctypes.c_int(max_tokens),
        )
        if n < 0:
            err = lib.infer_last_error_string()
            msg = err.decode() if err else "unknown error"
            raise RuntimeError(f"infer_llm_tokenize failed: {msg}")
        return list(ids_arr[:n])

    def token_to_piece(self, token_id: int) -> str:
        """
        Convert a single token id to its string piece (UTF-8).

        Raises RuntimeError on failure.
        """
        buf_size = 256
        buf = ctypes.create_string_buffer(buf_size)
        n = lib.infer_llm_token_to_piece(
            self._llm,
            ctypes.c_int(token_id),
            buf,
            ctypes.c_int(buf_size),
        )
        if n < 0:
            err = lib.infer_last_error_string()
            msg = err.decode() if err else "unknown error"
            raise RuntimeError(f"infer_llm_token_to_piece failed: {msg}")
        return buf.raw[:n].decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # KV cache info
    # ------------------------------------------------------------------

    def kv_pages_free(self) -> int:
        """Number of free KV-cache pages."""
        return lib.infer_llm_kv_pages_free(self._llm)

    def kv_pages_total(self) -> int:
        """Total number of KV-cache pages."""
        return lib.infer_llm_kv_pages_total(self._llm)

    # ------------------------------------------------------------------
    # Generation internals
    # ------------------------------------------------------------------

    def _make_token_array(self, tokens: list[int]) -> ctypes.Array:
        arr = (ctypes.c_int * len(tokens))(*tokens)
        return arr

    def _resolve_prompt(self, prompt: str | list[int]) -> list[int]:
        """Convert a prompt (string or token list) to a token list."""
        if isinstance(prompt, str):
            return self.tokenize(prompt, add_bos=True)
        return list(prompt)

    def _run_sequence(
        self,
        tokens: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Iterator[int]:
        """
        Core generation loop.

        Creates a sequence, decodes it step-by-step, and yields generated
        token ids until an EOG token is produced or *max_tokens* is reached.
        """
        token_arr = self._make_token_array(tokens)
        n_tokens = len(tokens)

        seq = lib.infer_seq_create(
            self._llm,
            token_arr,
            ctypes.c_int(n_tokens),
        )
        if not seq:
            err = lib.infer_last_error_string()
            msg = err.decode() if err else "unknown error"
            raise RuntimeError(f"infer_seq_create failed: {msg}")

        try:
            vocab_size = self._vocab_size
            logits_arr = (ctypes.c_float * vocab_size)()

            # Build the seqs array for batch_decode (single sequence).
            SeqPtrArr = ctypes.c_void_p * 1
            seqs_arr = SeqPtrArr(seq)

            for _ in range(max_tokens):
                # Run a decode step.
                rc = lib.infer_llm_batch_decode(
                    self._llm,
                    seqs_arr,
                    ctypes.c_int(1),
                )
                _check(rc, "infer_llm_batch_decode failed")

                # Check if the sequence signalled completion internally.
                if lib.infer_seq_is_done(seq):
                    break

                # Retrieve logits.
                rc = lib.infer_seq_get_logits(seq, logits_arr, ctypes.c_int(vocab_size))
                _check(rc, "infer_seq_get_logits failed")

                # Sample the next token.
                next_token = _sample_token(logits_arr, vocab_size, temperature, top_p)

                # Yield before appending so the caller sees it immediately.
                yield next_token

                # Stop if this is an end-of-generation token.
                if self.is_eog(next_token):
                    break

                # Append the sampled token to the sequence for the next step.
                lib.infer_seq_append_token(seq, ctypes.c_int(next_token))
        finally:
            lib.infer_seq_destroy(seq)

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str | list[int],
        *,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> str:
        """
        Generate text from *prompt* and return the complete output string.

        Parameters
        ----------
        prompt:
            A string (will be tokenized) or a pre-tokenized list of ints.
        max_tokens:
            Maximum number of tokens to generate.
        temperature:
            Sampling temperature.  Pass 0 for greedy (argmax) decoding.
        top_p:
            Nucleus sampling cumulative probability threshold.
        """
        tokens = self._resolve_prompt(prompt)
        pieces: list[str] = []
        for tok in self._run_sequence(tokens, max_tokens, temperature, top_p):
            if not self.is_eog(tok):
                pieces.append(self.token_to_piece(tok))
        return "".join(pieces)

    def stream(
        self,
        prompt: str | list[int],
        *,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> Iterator[str]:
        """
        Streaming text generator.

        Yields decoded string pieces as tokens are produced, allowing callers
        to display output incrementally.

        Parameters
        ----------
        prompt:
            A string (will be tokenized) or a pre-tokenized list of ints.
        max_tokens:
            Maximum number of tokens to generate.
        temperature:
            Sampling temperature.  Pass 0 for greedy (argmax) decoding.
        top_p:
            Nucleus sampling cumulative probability threshold.
        """
        tokens = self._resolve_prompt(prompt)
        for tok in self._run_sequence(tokens, max_tokens, temperature, top_p):
            if not self.is_eog(tok):
                yield self.token_to_piece(tok)

    def chat(
        self,
        message: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.8,
    ) -> str:
        """
        Single-turn chat helper.

        Applies a simple ChatML-style system/user/assistant template and
        returns the model's response as a string.

        Uses the LLaMA 3 Instruct chat template (ChatML header format).

        Template (with system)::

            <|begin_of_text|><|start_header_id|>system<|end_header_id|>

            {system}<|eot_id|><|start_header_id|>user<|end_header_id|>

            {message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>


        Template (without system)::

            <|begin_of_text|><|start_header_id|>user<|end_header_id|>

            {message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>


        Parameters
        ----------
        message:
            The user turn text.
        system:
            Optional system prompt.  Omitted from the template when None.
        max_tokens:
            Maximum number of tokens to generate.
        temperature:
            Sampling temperature.
        """
        if system is not None:
            prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{system}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{message}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        else:
            prompt = (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{message}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )

        # The template already starts with <|begin_of_text|> (the BOS token).
        # Tokenize with add_bos=False to avoid a double-BOS which causes the
        # model to output nothing.
        tokens = self.tokenize(prompt, add_bos=False)
        return self.generate(
            tokens,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Release the native LLM handle.

        Idempotent — safe to call multiple times.
        """
        if self._llm is not None:
            lib.infer_llm_destroy(self._llm)
            self._llm = None

    def __enter__(self) -> "LLM":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
