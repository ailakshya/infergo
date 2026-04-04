"""
infergo.session — Python wrapper around the infergo ONNX session and tensor C APIs.
"""

import ctypes

from ._lib import lib
from ._types import INFER_OK, _check

# ---------------------------------------------------------------------------
# Optional numpy import
# ---------------------------------------------------------------------------

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NUMPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# dtype mapping: numpy dtype -> infergo dtype int
# ---------------------------------------------------------------------------

# infergo dtype constants (mirrors C enum)
DTYPE_FLOAT32 = 0
DTYPE_FLOAT16 = 1
DTYPE_INT32    = 3
DTYPE_INT64    = 4
DTYPE_UINT8    = 5

_NP_DTYPE_TO_INFER: dict = {}
if _NUMPY_AVAILABLE:
    _NP_DTYPE_TO_INFER = {
        np.dtype("float32"): DTYPE_FLOAT32,
        np.dtype("float16"): DTYPE_FLOAT16,
        np.dtype("int32"):   DTYPE_INT32,
        np.dtype("int64"):   DTYPE_INT64,
        np.dtype("uint8"):   DTYPE_UINT8,
    }

_INFER_DTYPE_ITEMSIZE: dict[int, int] = {
    DTYPE_FLOAT32: 4,
    DTYPE_FLOAT16: 2,
    DTYPE_INT32:   4,
    DTYPE_INT64:   8,
    DTYPE_UINT8:   1,
}

_INFER_DTYPE_TO_NP: dict[int, "np.dtype"] = {}
if _NUMPY_AVAILABLE:
    _INFER_DTYPE_TO_NP = {
        DTYPE_FLOAT32: np.dtype("float32"),
        DTYPE_FLOAT16: np.dtype("float16"),
        DTYPE_INT32:   np.dtype("int32"),
        DTYPE_INT64:   np.dtype("int64"),
        DTYPE_UINT8:   np.dtype("uint8"),
    }

# ---------------------------------------------------------------------------
# C function signatures — session
# ---------------------------------------------------------------------------

# void* infer_session_create(provider, device_id)
lib.infer_session_create.restype = ctypes.c_void_p
lib.infer_session_create.argtypes = [ctypes.c_char_p, ctypes.c_int]

# int infer_session_load(s, model_path)
lib.infer_session_load.restype = ctypes.c_int
lib.infer_session_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

# int infer_session_num_inputs(s)
lib.infer_session_num_inputs.restype = ctypes.c_int
lib.infer_session_num_inputs.argtypes = [ctypes.c_void_p]

# int infer_session_num_outputs(s)
lib.infer_session_num_outputs.restype = ctypes.c_int
lib.infer_session_num_outputs.argtypes = [ctypes.c_void_p]

# int infer_session_input_name(s, idx, out_buf, buf_size)
lib.infer_session_input_name.restype = ctypes.c_int
lib.infer_session_input_name.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int
]

# int infer_session_output_name(s, idx, out_buf, buf_size)
lib.infer_session_output_name.restype = ctypes.c_int
lib.infer_session_output_name.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int
]

# int infer_session_run(s, inputs_ptr, n_inputs, outputs_ptr, n_outputs)
lib.infer_session_run.restype = ctypes.c_int
lib.infer_session_run.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),  # inputs_ptr (array of tensor handles)
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_void_p),  # outputs_ptr (array of tensor handles)
    ctypes.c_int,
]

# void infer_session_destroy(s)
lib.infer_session_destroy.restype = None
lib.infer_session_destroy.argtypes = [ctypes.c_void_p]

# ---------------------------------------------------------------------------
# C function signatures — tensor
# ---------------------------------------------------------------------------

# void* infer_tensor_alloc_cpu(shape_ptr, ndim, dtype)
lib.infer_tensor_alloc_cpu.restype = ctypes.c_void_p
lib.infer_tensor_alloc_cpu.argtypes = [
    ctypes.POINTER(ctypes.c_int64),  # shape (array of int64)
    ctypes.c_int,                    # ndim
    ctypes.c_int,                    # dtype
]

# void infer_tensor_free(t)
lib.infer_tensor_free.restype = None
lib.infer_tensor_free.argtypes = [ctypes.c_void_p]

# void* infer_tensor_data_ptr(t)
lib.infer_tensor_data_ptr.restype = ctypes.c_void_p
lib.infer_tensor_data_ptr.argtypes = [ctypes.c_void_p]

# int infer_tensor_nbytes(t)
lib.infer_tensor_nbytes.restype = ctypes.c_int
lib.infer_tensor_nbytes.argtypes = [ctypes.c_void_p]

# int infer_tensor_nelements(t)
lib.infer_tensor_nelements.restype = ctypes.c_int
lib.infer_tensor_nelements.argtypes = [ctypes.c_void_p]

# int infer_tensor_shape(t, out_shape, max_dims)
lib.infer_tensor_shape.restype = ctypes.c_int
lib.infer_tensor_shape.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_int,
]

# int infer_tensor_dtype(t)
lib.infer_tensor_dtype.restype = ctypes.c_int
lib.infer_tensor_dtype.argtypes = [ctypes.c_void_p]

# int infer_tensor_copy_from(t, src, nbytes)
lib.infer_tensor_copy_from.restype = ctypes.c_int
lib.infer_tensor_copy_from.argtypes = [
    ctypes.c_void_p,   # t
    ctypes.c_void_p,   # src
    ctypes.c_int,      # nbytes
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_NAME_BUF_SIZE = 1024
_MAX_DIMS = 16


def _make_shape_array(shape: list[int]):
    """Build a ctypes int64 array from a Python list of ints."""
    arr = (ctypes.c_int64 * len(shape))(*shape)
    return arr


def _alloc_tensor_from_numpy(arr) -> ctypes.c_void_p:
    """Allocate a C tensor and copy data from a numpy array."""
    dtype_int = _NP_DTYPE_TO_INFER.get(arr.dtype)
    if dtype_int is None:
        raise ValueError(f"Unsupported numpy dtype: {arr.dtype}")

    shape = list(arr.shape)
    ndim = len(shape)
    shape_arr = _make_shape_array(shape)

    handle = lib.infer_tensor_alloc_cpu(shape_arr, ctypes.c_int(ndim), ctypes.c_int(dtype_int))
    if handle is None:
        raise RuntimeError("infer_tensor_alloc_cpu returned NULL")

    # Ensure the array is contiguous (C order) before getting its pointer.
    if _NUMPY_AVAILABLE:
        arr_c = np.ascontiguousarray(arr)
    else:
        arr_c = arr

    src_ptr = arr_c.ctypes.data_as(ctypes.c_void_p)
    nbytes = arr_c.nbytes
    rc = lib.infer_tensor_copy_from(ctypes.c_void_p(handle), src_ptr, ctypes.c_int(nbytes))
    if rc != INFER_OK:
        lib.infer_tensor_free(ctypes.c_void_p(handle))
        raise RuntimeError(f"infer_tensor_copy_from failed (rc={rc})")

    return ctypes.c_void_p(handle)


def _alloc_tensor_from_tuple(data: bytes, shape: list[int], dtype: int) -> ctypes.c_void_p:
    """Allocate a C tensor and copy data from a raw-bytes tuple."""
    ndim = len(shape)
    shape_arr = _make_shape_array(shape)

    handle = lib.infer_tensor_alloc_cpu(shape_arr, ctypes.c_int(ndim), ctypes.c_int(dtype))
    if handle is None:
        raise RuntimeError("infer_tensor_alloc_cpu returned NULL")

    src = (ctypes.c_char * len(data)).from_buffer_copy(data)
    src_ptr = ctypes.cast(src, ctypes.c_void_p)
    rc = lib.infer_tensor_copy_from(ctypes.c_void_p(handle), src_ptr, ctypes.c_int(len(data)))
    if rc != INFER_OK:
        lib.infer_tensor_free(ctypes.c_void_p(handle))
        raise RuntimeError(f"infer_tensor_copy_from failed (rc={rc})")

    return ctypes.c_void_p(handle)


def _read_tensor_to_numpy(handle: ctypes.c_void_p):
    """Read a C tensor back into a numpy array."""
    dtype_int = lib.infer_tensor_dtype(handle)
    np_dtype = _INFER_DTYPE_TO_NP.get(dtype_int)
    if np_dtype is None:
        raise RuntimeError(f"Unknown tensor dtype from C layer: {dtype_int}")

    ndim_or_ndims = _MAX_DIMS
    shape_buf = (ctypes.c_int64 * ndim_or_ndims)()
    ndim = lib.infer_tensor_shape(handle, shape_buf, ctypes.c_int(ndim_or_ndims))
    if ndim < 0:
        raise RuntimeError("infer_tensor_shape failed")
    shape = [int(shape_buf[i]) for i in range(ndim)]

    nbytes = lib.infer_tensor_nbytes(handle)
    data_ptr = lib.infer_tensor_data_ptr(handle)
    if data_ptr is None:
        raise RuntimeError("infer_tensor_data_ptr returned NULL")

    raw = (ctypes.c_char * nbytes).from_address(data_ptr)
    arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape).copy()
    return arr


def _read_tensor_to_tuple(handle: ctypes.c_void_p) -> tuple[bytes, list[int], int]:
    """Read a C tensor back into a (bytes, shape, dtype) tuple."""
    dtype_int = lib.infer_tensor_dtype(handle)

    shape_buf = (ctypes.c_int64 * _MAX_DIMS)()
    ndim = lib.infer_tensor_shape(handle, shape_buf, ctypes.c_int(_MAX_DIMS))
    if ndim < 0:
        raise RuntimeError("infer_tensor_shape failed")
    shape = [int(shape_buf[i]) for i in range(ndim)]

    nbytes = lib.infer_tensor_nbytes(handle)
    data_ptr = lib.infer_tensor_data_ptr(handle)
    if data_ptr is None:
        raise RuntimeError("infer_tensor_data_ptr returned NULL")

    raw = bytes((ctypes.c_char * nbytes).from_address(data_ptr))
    return raw, shape, dtype_int


# ---------------------------------------------------------------------------
# Session class
# ---------------------------------------------------------------------------

class Session:
    """Wraps an infergo ONNX inference session."""

    def __init__(self, provider: str = "cpu", device_id: int = 0) -> None:
        handle = lib.infer_session_create(
            provider.encode(), ctypes.c_int(device_id)
        )
        if handle is None:
            raise RuntimeError(
                f"infer_session_create failed: provider='{provider}' device_id={device_id}"
            )
        self._handle = ctypes.c_void_p(handle)
        self._closed = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self, model_path: str) -> "Session":
        """Load an ONNX model from *model_path*. Returns self for chaining."""
        self._require_open()
        rc = lib.infer_session_load(self._handle, model_path.encode())
        _check(rc, f"infer_session_load('{model_path}')")
        return self

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def num_inputs(self) -> int:
        self._require_open()
        return lib.infer_session_num_inputs(self._handle)

    @property
    def num_outputs(self) -> int:
        self._require_open()
        return lib.infer_session_num_outputs(self._handle)

    def input_name(self, idx: int) -> str:
        self._require_open()
        buf = ctypes.create_string_buffer(_NAME_BUF_SIZE)
        rc = lib.infer_session_input_name(
            self._handle, ctypes.c_int(idx), buf, ctypes.c_int(_NAME_BUF_SIZE)
        )
        _check(rc, f"infer_session_input_name(idx={idx})")
        return buf.value.decode("utf-8")

    def output_name(self, idx: int) -> str:
        self._require_open()
        buf = ctypes.create_string_buffer(_NAME_BUF_SIZE)
        rc = lib.infer_session_output_name(
            self._handle, ctypes.c_int(idx), buf, ctypes.c_int(_NAME_BUF_SIZE)
        )
        _check(rc, f"infer_session_output_name(idx={idx})")
        return buf.value.decode("utf-8")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run(self, inputs: list) -> list:
        """Run inference.

        inputs:
            - If numpy is available: list of numpy arrays.
            - Otherwise: list of (data_bytes, shape: list[int], dtype: int) tuples.

        Returns:
            - List of numpy arrays if numpy is available.
            - List of (bytes, shape, dtype) tuples otherwise.
        """
        self._require_open()

        n_in = len(inputs)
        n_out = self.num_outputs

        # Allocate input tensors and track handles for cleanup.
        in_handles: list[ctypes.c_void_p] = []
        out_handles: list[ctypes.c_void_p] = []

        try:
            for inp in inputs:
                if _NUMPY_AVAILABLE and isinstance(inp, np.ndarray):
                    h = _alloc_tensor_from_numpy(inp)
                elif isinstance(inp, (list, tuple)) and len(inp) == 3:
                    data, shape, dtype = inp
                    if not isinstance(data, (bytes, bytearray)):
                        raise TypeError(
                            "Input tuple must be (bytes, list[int], int); "
                            f"got data type {type(data)}"
                        )
                    h = _alloc_tensor_from_tuple(bytes(data), list(shape), int(dtype))
                else:
                    raise TypeError(
                        "Each input must be a numpy array or a "
                        "(bytes, shape, dtype) tuple"
                    )
                in_handles.append(h)

            # Allocate output tensor handles (C layer populates them during run).
            # We pass NULL-initialised pointers; the C API is expected to fill them.
            out_ptr_arr = (ctypes.c_void_p * n_out)(*([None] * n_out))

            # Build input pointer array.
            in_ptr_arr = (ctypes.c_void_p * n_in)(
                *[h.value for h in in_handles]
            )

            rc = lib.infer_session_run(
                self._handle,
                in_ptr_arr,
                ctypes.c_int(n_in),
                out_ptr_arr,
                ctypes.c_int(n_out),
            )
            _check(rc, "infer_session_run")

            # Collect output handles for later cleanup.
            for i in range(n_out):
                if out_ptr_arr[i] is not None:
                    out_handles.append(ctypes.c_void_p(out_ptr_arr[i]))

            # Deserialize outputs.
            results: list = []
            for h in out_handles:
                if _NUMPY_AVAILABLE:
                    results.append(_read_tensor_to_numpy(h))
                else:
                    results.append(_read_tensor_to_tuple(h))

            return results

        finally:
            for h in in_handles:
                lib.infer_tensor_free(h)
            for h in out_handles:
                lib.infer_tensor_free(h)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Destroy the native session handle. Safe to call multiple times."""
        if not self._closed and self._handle is not None:
            lib.infer_session_destroy(self._handle)
            self._closed = True
            self._handle = None

    def __enter__(self) -> "Session":
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
            raise RuntimeError("Session is already closed")
