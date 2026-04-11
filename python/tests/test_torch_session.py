"""
Unit tests for infergo.torch_session.TorchSession.

All tests use unittest.mock to patch the C library so no real shared
library or model file is required.
"""

import ctypes
from unittest.mock import patch, MagicMock
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_lib(session_ptr=0xABCD):
    """Return a MagicMock configured to look like a working infergo lib.

    Note: when ctypes CDLL has restype=c_void_p, the actual Python return
    value is a plain int (or None for NULL).  The mock must match this.
    """
    mock = MagicMock()

    # Torch session
    mock.infer_torch_session_create.return_value = session_ptr  # raw int
    mock.infer_torch_session_load.return_value = 0  # INFER_OK
    mock.infer_torch_session_num_inputs.return_value = 1
    mock.infer_torch_session_num_outputs.return_value = 1
    mock.infer_torch_session_run.return_value = 0  # INFER_OK
    mock.infer_torch_session_destroy.return_value = None

    # Tensor helpers (used by run())
    mock.infer_tensor_alloc_cpu.return_value = 0x1111  # raw int
    mock.infer_tensor_copy_from.return_value = 0  # INFER_OK
    mock.infer_tensor_free.return_value = None
    mock.infer_tensor_dtype.return_value = 0  # FLOAT32
    mock.infer_tensor_nbytes.return_value = 4
    mock.infer_tensor_data_ptr.return_value = None  # will be overridden per test
    mock.infer_tensor_nelements.return_value = 1

    def _fake_shape(handle, buf, max_dims):
        buf[0] = 1
        return 1  # ndim = 1

    mock.infer_tensor_shape.side_effect = _fake_shape

    mock.infer_last_error_string.return_value = b"mock error"

    return mock


# ---------------------------------------------------------------------------
# Test: __init__
# ---------------------------------------------------------------------------

class TestTorchSessionInit:
    def test_create_null_raises(self):
        """TorchSession.__init__ must raise when create returns NULL."""
        mock_lib = _make_mock_lib()
        mock_lib.infer_torch_session_create.return_value = None

        with patch("infergo.torch_session.lib", mock_lib):
            from infergo.torch_session import TorchSession
            with pytest.raises(RuntimeError, match="infer_torch_session_create failed"):
                TorchSession()

    def test_create_success(self):
        """TorchSession.__init__ succeeds with a non-NULL pointer."""
        mock_lib = _make_mock_lib()

        with patch("infergo.torch_session.lib", mock_lib):
            from infergo.torch_session import TorchSession
            s = TorchSession(provider="cpu", device_id=0)
            assert s._handle is not None
            assert not s._closed
            mock_lib.infer_torch_session_create.assert_called_once()
            s.close()

    def test_create_with_cuda_provider(self):
        """TorchSession can be created with cuda provider."""
        mock_lib = _make_mock_lib()

        with patch("infergo.torch_session.lib", mock_lib):
            from infergo.torch_session import TorchSession
            s = TorchSession(provider="cuda", device_id=1)
            assert s._handle is not None
            call_args = mock_lib.infer_torch_session_create.call_args
            assert call_args[0][0] == b"cuda"
            s.close()


# ---------------------------------------------------------------------------
# Test: load
# ---------------------------------------------------------------------------

class TestTorchSessionLoad:
    def test_load_success(self):
        """load() returns self for chaining."""
        mock_lib = _make_mock_lib()

        with patch("infergo.torch_session.lib", mock_lib):
            from infergo.torch_session import TorchSession
            s = TorchSession()
            result = s.load("model.pt")
            assert result is s
            mock_lib.infer_torch_session_load.assert_called_once()
            s.close()

    def test_load_failure_raises(self):
        """load() must raise RuntimeError when the C call returns an error."""
        mock_lib = _make_mock_lib()
        mock_lib.infer_torch_session_load.return_value = 5  # INFER_ERR_LOAD

        with patch("infergo.torch_session.lib", mock_lib):
            from infergo.torch_session import TorchSession
            s = TorchSession()
            with pytest.raises(RuntimeError):
                s.load("bad_model.pt")
            s.close()

    def test_load_on_closed_session_raises(self):
        """load() must raise when the session is already closed."""
        mock_lib = _make_mock_lib()

        with patch("infergo.torch_session.lib", mock_lib):
            from infergo.torch_session import TorchSession
            s = TorchSession()
            s.close()
            with pytest.raises(RuntimeError, match="already closed"):
                s.load("model.pt")


# ---------------------------------------------------------------------------
# Test: num_inputs / num_outputs
# ---------------------------------------------------------------------------

class TestTorchSessionMetadata:
    def test_num_inputs(self):
        mock_lib = _make_mock_lib()
        mock_lib.infer_torch_session_num_inputs.return_value = 3

        with patch("infergo.torch_session.lib", mock_lib):
            from infergo.torch_session import TorchSession
            s = TorchSession()
            assert s.num_inputs == 3
            s.close()

    def test_num_outputs(self):
        mock_lib = _make_mock_lib()
        mock_lib.infer_torch_session_num_outputs.return_value = 2

        with patch("infergo.torch_session.lib", mock_lib):
            from infergo.torch_session import TorchSession
            s = TorchSession()
            assert s.num_outputs == 2
            s.close()

    def test_num_inputs_on_closed_raises(self):
        mock_lib = _make_mock_lib()

        with patch("infergo.torch_session.lib", mock_lib):
            from infergo.torch_session import TorchSession
            s = TorchSession()
            s.close()
            with pytest.raises(RuntimeError, match="already closed"):
                _ = s.num_inputs

    def test_num_outputs_on_closed_raises(self):
        mock_lib = _make_mock_lib()

        with patch("infergo.torch_session.lib", mock_lib):
            from infergo.torch_session import TorchSession
            s = TorchSession()
            s.close()
            with pytest.raises(RuntimeError, match="already closed"):
                _ = s.num_outputs


# ---------------------------------------------------------------------------
# Test: run
# ---------------------------------------------------------------------------

class TestTorchSessionRun:
    def test_run_on_closed_raises(self):
        """run() must raise when the session is already closed."""
        mock_lib = _make_mock_lib()

        with patch("infergo.torch_session.lib", mock_lib):
            from infergo.torch_session import TorchSession
            s = TorchSession()
            s.close()
            with pytest.raises(RuntimeError, match="already closed"):
                s.run([])

    def test_run_invalid_input_type_raises(self):
        """run() must raise TypeError for unsupported input types."""
        mock_lib = _make_mock_lib()

        with patch("infergo.torch_session.lib", mock_lib):
            from infergo.torch_session import TorchSession
            s = TorchSession()
            with pytest.raises(TypeError, match="numpy array"):
                s.run(["not_a_tensor"])
            s.close()

    def test_run_calls_c_api(self):
        """run() must call the torch session run C function."""
        mock_lib = _make_mock_lib()
        mock_lib.infer_torch_session_num_outputs.return_value = 0

        with patch("infergo.torch_session.lib", mock_lib):
            from infergo.torch_session import TorchSession
            s = TorchSession()
            # With 0 outputs, run should succeed and return empty list
            result = s.run([])
            assert result == []
            mock_lib.infer_torch_session_run.assert_called_once()
            s.close()

    def test_run_with_tuple_input(self):
        """run() accepts (bytes, shape, dtype) tuple inputs."""
        import struct
        mock_lib = _make_mock_lib()
        mock_lib.infer_torch_session_num_outputs.return_value = 0

        with patch("infergo.torch_session.lib", mock_lib):
            from infergo.torch_session import TorchSession
            s = TorchSession()
            data = struct.pack("f", 1.0)
            result = s.run([(data, [1], 0)])
            assert result == []
            mock_lib.infer_tensor_alloc_cpu.assert_called()
            mock_lib.infer_tensor_copy_from.assert_called()
            s.close()

    def test_run_failure_raises(self):
        """run() must raise RuntimeError when session_run returns error."""
        import struct
        mock_lib = _make_mock_lib()
        mock_lib.infer_torch_session_num_outputs.return_value = 1
        mock_lib.infer_torch_session_run.return_value = 6  # INFER_ERR_RUNTIME

        with patch("infergo.torch_session.lib", mock_lib):
            from infergo.torch_session import TorchSession
            s = TorchSession()
            data = struct.pack("f", 1.0)
            with pytest.raises(RuntimeError):
                s.run([(data, [1], 0)])
            s.close()


# ---------------------------------------------------------------------------
# Test: close
# ---------------------------------------------------------------------------

class TestTorchSessionClose:
    def test_close_idempotent(self):
        """close() must not raise when called multiple times."""
        mock_lib = _make_mock_lib()

        with patch("infergo.torch_session.lib", mock_lib):
            from infergo.torch_session import TorchSession
            s = TorchSession()
            s.close()
            s.close()
            mock_lib.infer_torch_session_destroy.assert_called_once()

    def test_close_sets_handle_none(self):
        """After close(), _handle must be None."""
        mock_lib = _make_mock_lib()

        with patch("infergo.torch_session.lib", mock_lib):
            from infergo.torch_session import TorchSession
            s = TorchSession()
            s.close()
            assert s._handle is None
            assert s._closed is True


# ---------------------------------------------------------------------------
# Test: context manager
# ---------------------------------------------------------------------------

class TestTorchSessionContextManager:
    def test_context_manager_calls_close(self):
        """'with TorchSession() as s:' must call close on exit."""
        mock_lib = _make_mock_lib()

        with patch("infergo.torch_session.lib", mock_lib):
            from infergo.torch_session import TorchSession
            with TorchSession() as s:
                assert s._handle is not None

            mock_lib.infer_torch_session_destroy.assert_called_once()
            assert s._handle is None

    def test_context_manager_calls_close_on_exception(self):
        """close() must still be called even if the body raises."""
        mock_lib = _make_mock_lib()

        with patch("infergo.torch_session.lib", mock_lib):
            from infergo.torch_session import TorchSession
            with pytest.raises(ValueError):
                with TorchSession():
                    raise ValueError("boom")

            mock_lib.infer_torch_session_destroy.assert_called_once()
