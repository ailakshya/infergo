"""
Unit tests for infergo.tokenizer.Tokenizer.

All tests patch `infergo.tokenizer.lib` so no shared library is required.
"""

import ctypes
from unittest.mock import patch, MagicMock
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_lib(handle=0xABCD):
    """Return a MagicMock configured like a working infergo lib for tokenizer tests."""
    mock = MagicMock()

    mock.infer_tokenizer_load.return_value = handle
    mock.infer_tokenizer_destroy.return_value = None
    mock.infer_tokenizer_vocab_size.return_value = 32000
    mock.infer_tokenizer_encode.return_value = 0   # overridden per-test
    mock.infer_tokenizer_decode.return_value = 0   # INFER_OK
    mock.infer_tokenizer_decode_token.return_value = 0
    mock.infer_last_error_string.return_value = b"mock error"

    return mock


# ---------------------------------------------------------------------------
# Test: __init__ raises on NULL handle
# ---------------------------------------------------------------------------

class TestTokenizerInit:
    def test_null_handle_raises(self):
        """Tokenizer.__init__ must raise RuntimeError when load returns NULL (0 / None)."""
        mock_lib = _make_mock_lib(handle=None)

        with patch("infergo.tokenizer.lib", mock_lib):
            from infergo.tokenizer import Tokenizer
            with pytest.raises(RuntimeError):
                Tokenizer("/nonexistent/tokenizer.json")

    def test_valid_handle_succeeds(self):
        """Tokenizer.__init__ must succeed when load returns a non-NULL pointer."""
        mock_lib = _make_mock_lib()

        with patch("infergo.tokenizer.lib", mock_lib):
            from infergo.tokenizer import Tokenizer
            tok = Tokenizer("tokenizer.json")
            assert tok._handle is not None
            tok.close()


# ---------------------------------------------------------------------------
# Test: encode returns (list[int], list[int])
# ---------------------------------------------------------------------------

class TestTokenizerEncode:
    def test_encode_returns_tuple_of_lists(self):
        """encode() must return a (list[int], list[int]) tuple."""
        mock_lib = _make_mock_lib()

        TOKEN_IDS = [101, 7592, 102]
        MASK      = [1,   1,    1  ]

        def _fake_encode(handle, text, add_special, out_ids, out_mask, max_tokens):
            for i, (tid, m) in enumerate(zip(TOKEN_IDS, MASK)):
                out_ids[i] = tid
                out_mask[i] = m
            return len(TOKEN_IDS)

        mock_lib.infer_tokenizer_encode.side_effect = _fake_encode

        with patch("infergo.tokenizer.lib", mock_lib):
            from infergo.tokenizer import Tokenizer
            tok = Tokenizer("tokenizer.json")
            ids, mask = tok.encode("hello world")

            assert isinstance(ids, list)
            assert isinstance(mask, list)
            assert all(isinstance(x, int) for x in ids)
            assert all(isinstance(x, int) for x in mask)
            assert ids == TOKEN_IDS
            assert mask == MASK
            tok.close()

    def test_encode_failure_raises(self):
        """encode() must raise RuntimeError when the C function returns a negative value."""
        mock_lib = _make_mock_lib()
        mock_lib.infer_tokenizer_encode.return_value = -1

        with patch("infergo.tokenizer.lib", mock_lib):
            from infergo.tokenizer import Tokenizer
            tok = Tokenizer("tokenizer.json")
            with pytest.raises(RuntimeError):
                tok.encode("hello")
            tok.close()


# ---------------------------------------------------------------------------
# Test: decode returns a string
# ---------------------------------------------------------------------------

class TestTokenizerDecode:
    def test_decode_returns_string(self):
        """decode() must return a str."""
        mock_lib = _make_mock_lib()

        DECODED = b"hello world"

        def _fake_decode(handle, ids, n_ids, skip_special, out_buf, buf_size):
            ctypes.memmove(out_buf, DECODED, len(DECODED))
            # The real function writes a null-terminated string; ctypes c_char_p
            # reads up to the null byte via buf.value, so we set the null too.
            ctypes.cast(out_buf, ctypes.POINTER(ctypes.c_char))[len(DECODED)] = b"\x00"
            return 0  # INFER_OK

        mock_lib.infer_tokenizer_decode.side_effect = _fake_decode

        with patch("infergo.tokenizer.lib", mock_lib):
            from infergo.tokenizer import Tokenizer
            tok = Tokenizer("tokenizer.json")
            result = tok.decode([101, 7592, 102])
            assert isinstance(result, str)
            tok.close()

    def test_decode_uses_skip_special_flag(self):
        """decode() must pass skip_special_tokens as 1/0 to the C function."""
        mock_lib = _make_mock_lib()
        mock_lib.infer_tokenizer_decode.return_value = 0

        calls_seen = []

        def _fake_decode(handle, ids, n_ids, skip_special, out_buf, buf_size):
            calls_seen.append(int(skip_special))
            return 0

        mock_lib.infer_tokenizer_decode.side_effect = _fake_decode

        with patch("infergo.tokenizer.lib", mock_lib):
            from infergo.tokenizer import Tokenizer
            tok = Tokenizer("tokenizer.json")
            tok.decode([1], skip_special_tokens=True)
            tok.decode([1], skip_special_tokens=False)
            assert calls_seen == [1, 0]
            tok.close()


# ---------------------------------------------------------------------------
# Test: vocab_size returns int
# ---------------------------------------------------------------------------

class TestTokenizerVocabSize:
    def test_vocab_size_returns_int(self):
        """vocab_size must return an int matching what the C function returns."""
        mock_lib = _make_mock_lib()
        mock_lib.infer_tokenizer_vocab_size.return_value = 50257

        with patch("infergo.tokenizer.lib", mock_lib):
            from infergo.tokenizer import Tokenizer
            tok = Tokenizer("tokenizer.json")
            v = tok.vocab_size
            assert isinstance(v, int)
            assert v == 50257
            tok.close()


# ---------------------------------------------------------------------------
# Test: context manager works
# ---------------------------------------------------------------------------

class TestTokenizerContextManager:
    def test_context_manager_calls_destroy(self):
        """'with Tokenizer(...) as tok:' must call destroy on exit."""
        mock_lib = _make_mock_lib()

        with patch("infergo.tokenizer.lib", mock_lib):
            from infergo.tokenizer import Tokenizer
            with Tokenizer("tokenizer.json") as tok:
                assert not tok._closed

            mock_lib.infer_tokenizer_destroy.assert_called_once()
            assert tok._closed

    def test_context_manager_calls_destroy_on_exception(self):
        """destroy must still be called if the body raises."""
        mock_lib = _make_mock_lib()

        with patch("infergo.tokenizer.lib", mock_lib):
            from infergo.tokenizer import Tokenizer
            with pytest.raises(ValueError):
                with Tokenizer("tokenizer.json"):
                    raise ValueError("boom")

            mock_lib.infer_tokenizer_destroy.assert_called_once()

    def test_close_idempotent(self):
        """close() must not call destroy a second time if already closed."""
        mock_lib = _make_mock_lib()

        with patch("infergo.tokenizer.lib", mock_lib):
            from infergo.tokenizer import Tokenizer
            tok = Tokenizer("tokenizer.json")
            tok.close()
            tok.close()
            mock_lib.infer_tokenizer_destroy.assert_called_once()

    def test_operations_after_close_raise(self):
        """Calling encode/decode/vocab_size on a closed Tokenizer must raise."""
        mock_lib = _make_mock_lib()

        with patch("infergo.tokenizer.lib", mock_lib):
            from infergo.tokenizer import Tokenizer
            tok = Tokenizer("tokenizer.json")
            tok.close()

            with pytest.raises(RuntimeError, match="closed"):
                tok.encode("hello")

            with pytest.raises(RuntimeError, match="closed"):
                tok.decode([1, 2])

            with pytest.raises(RuntimeError, match="closed"):
                _ = tok.vocab_size
