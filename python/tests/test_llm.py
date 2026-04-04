"""
Unit tests for infergo.llm.LLM.

All tests use unittest.mock to patch `infergo.llm.lib` so no real shared
library or model file is required.
"""

import ctypes
import sys
import types
import importlib
from unittest.mock import patch, MagicMock, call
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_lib(llm_ptr=0x1234, seq_ptr=0x5678):
    """Return a MagicMock configured to look like a working infergo lib."""
    mock = MagicMock()

    mock.infer_llm_create.return_value = ctypes.c_void_p(llm_ptr)
    mock.infer_llm_create_split.return_value = ctypes.c_void_p(llm_ptr)
    mock.infer_llm_create_pipeline.return_value = ctypes.c_void_p(llm_ptr)
    mock.infer_llm_destroy.return_value = None
    mock.infer_llm_vocab_size.return_value = 32000
    mock.infer_llm_bos.return_value = 1
    mock.infer_llm_eos.return_value = 2
    mock.infer_llm_is_eog.return_value = 0
    mock.infer_llm_tokenize.return_value = 3  # 3 tokens written
    mock.infer_llm_token_to_piece.return_value = 1  # 1 byte written

    mock.infer_seq_create.return_value = ctypes.c_void_p(seq_ptr)
    mock.infer_seq_destroy.return_value = None
    mock.infer_seq_is_done.return_value = 1  # done after first step → short-circuit
    mock.infer_llm_batch_decode.return_value = 0  # INFER_OK
    mock.infer_seq_get_logits.return_value = 0   # INFER_OK
    mock.infer_seq_slot_id.return_value = 0
    mock.infer_seq_position.return_value = 0

    mock.infer_last_error_string.return_value = b"mock error"

    return mock


# ---------------------------------------------------------------------------
# Test: __init__ raises when infer_llm_create returns NULL
# ---------------------------------------------------------------------------

class TestLLMInit:
    def test_create_null_raises(self):
        """LLM.__init__ must raise RuntimeError when infer_llm_create returns NULL."""
        mock_lib = _make_mock_lib()
        mock_lib.infer_llm_create.return_value = ctypes.c_void_p(None)

        with patch("infergo.llm.lib", mock_lib):
            from infergo.llm import LLM
            with pytest.raises(RuntimeError, match="infer_llm_create failed"):
                LLM("model.gguf")

    def test_create_success(self):
        """LLM.__init__ succeeds when infer_llm_create returns a non-NULL pointer."""
        mock_lib = _make_mock_lib()

        with patch("infergo.llm.lib", mock_lib):
            from infergo.llm import LLM
            m = LLM("model.gguf")
            assert m._llm is not None
            mock_lib.infer_llm_create.assert_called_once()
            m.close()


# ---------------------------------------------------------------------------
# Test: tokenize returns list[int]
# ---------------------------------------------------------------------------

class TestLLMTokenize:
    def test_tokenize_returns_list_of_ints(self):
        """tokenize() must return a list of integers."""
        mock_lib = _make_mock_lib()

        # infer_llm_tokenize writes 3 tokens; we set up the side_effect to
        # populate the output buffer.
        TOKENS = [101, 202, 303]

        def _fake_tokenize(llm, text, add_bos, out_arr, max_tokens):
            for i, tok in enumerate(TOKENS):
                out_arr[i] = tok
            return len(TOKENS)

        mock_lib.infer_llm_tokenize.side_effect = _fake_tokenize

        with patch("infergo.llm.lib", mock_lib):
            from infergo.llm import LLM
            m = LLM("model.gguf")
            ids = m.tokenize("hello world")
            assert isinstance(ids, list)
            assert all(isinstance(x, int) for x in ids)
            assert ids == TOKENS
            m.close()


# ---------------------------------------------------------------------------
# Test: chat calls the expected C functions
# ---------------------------------------------------------------------------

class TestLLMChat:
    def test_chat_calls_c_api(self):
        """chat() must drive infer_llm_create, infer_seq_create, infer_llm_batch_decode,
        and infer_seq_get_logits through the generation loop."""
        mock_lib = _make_mock_lib()

        # Make tokenize return a small token list.
        def _fake_tokenize(llm, text, add_bos, out_arr, max_tokens):
            out_arr[0] = 1
            return 1

        mock_lib.infer_llm_tokenize.side_effect = _fake_tokenize

        # infer_seq_is_done returns 1 immediately so the loop exits after the
        # first batch_decode → no get_logits call in this path (done checked before logits).
        # That is the correct control flow in llm.py (_run_sequence checks is_done
        # *after* batch_decode and *before* get_logits).
        mock_lib.infer_seq_is_done.return_value = 1

        with patch("infergo.llm.lib", mock_lib):
            from infergo.llm import LLM
            m = LLM("model.gguf")
            result = m.chat("hi")
            assert isinstance(result, str)

            mock_lib.infer_llm_create.assert_called_once()
            mock_lib.infer_seq_create.assert_called_once()
            mock_lib.infer_llm_batch_decode.assert_called()
            mock_lib.infer_seq_destroy.assert_called_once()
            m.close()

    def test_chat_get_logits_called_when_not_done(self):
        """When is_done returns 0, batch_decode and get_logits must both be called."""
        mock_lib = _make_mock_lib()

        call_count = {"n": 0}

        def _is_done(seq):
            # Return not-done for the first call, done for the second.
            call_count["n"] += 1
            return 1 if call_count["n"] > 1 else 0

        mock_lib.infer_seq_is_done.side_effect = _is_done

        def _fake_tokenize(llm, text, add_bos, out_arr, max_tokens):
            out_arr[0] = 1
            return 1

        mock_lib.infer_llm_tokenize.side_effect = _fake_tokenize

        # get_logits fills the buffer with zeros → greedy picks token 0.
        mock_lib.infer_seq_get_logits.return_value = 0
        # is_eog returns 0 for token 0 so loop continues until is_done.
        mock_lib.infer_llm_is_eog.return_value = 0
        # token_to_piece returns "x" for the sampled token.
        def _fake_token_to_piece(llm, token, buf, buf_size):
            buf[0] = ord("x")
            return 1
        mock_lib.infer_llm_token_to_piece.side_effect = _fake_token_to_piece

        with patch("infergo.llm.lib", mock_lib):
            from infergo.llm import LLM
            m = LLM("model.gguf")
            result = m.chat("hello")
            assert isinstance(result, str)
            mock_lib.infer_seq_get_logits.assert_called()
            m.close()


# ---------------------------------------------------------------------------
# Test: close() is idempotent
# ---------------------------------------------------------------------------

class TestLLMClose:
    def test_close_idempotent(self):
        """close() must not raise when called multiple times."""
        mock_lib = _make_mock_lib()

        with patch("infergo.llm.lib", mock_lib):
            from infergo.llm import LLM
            m = LLM("model.gguf")
            m.close()
            m.close()  # second call must be a no-op
            # infer_llm_destroy must have been called exactly once.
            mock_lib.infer_llm_destroy.assert_called_once()

    def test_close_sets_handle_none(self):
        """After close(), _llm must be None."""
        mock_lib = _make_mock_lib()

        with patch("infergo.llm.lib", mock_lib):
            from infergo.llm import LLM
            m = LLM("model.gguf")
            m.close()
            assert m._llm is None


# ---------------------------------------------------------------------------
# Test: context manager calls close() on exit
# ---------------------------------------------------------------------------

class TestLLMContextManager:
    def test_context_manager_calls_close(self):
        """'with LLM(...) as m:' must call close() (i.e. infer_llm_destroy) on exit."""
        mock_lib = _make_mock_lib()

        with patch("infergo.llm.lib", mock_lib):
            from infergo.llm import LLM
            with LLM("model.gguf") as m:
                assert m._llm is not None

            # After the with block, the handle must be released.
            mock_lib.infer_llm_destroy.assert_called_once()
            assert m._llm is None

    def test_context_manager_calls_close_on_exception(self):
        """close() must still be called even if the body raises."""
        mock_lib = _make_mock_lib()

        with patch("infergo.llm.lib", mock_lib):
            from infergo.llm import LLM
            with pytest.raises(ValueError):
                with LLM("model.gguf"):
                    raise ValueError("boom")

            mock_lib.infer_llm_destroy.assert_called_once()
