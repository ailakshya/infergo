"""
infergo.tensor — Thin Python wrapper around an InferTensor C pointer.

Owns the pointer and frees it via infer_tensor_free() on close().
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

from ._lib import lib
from ._types import FLOAT32, _check

if TYPE_CHECKING:
    pass  # numpy imported lazily

__all__ = ["Tensor"]

# ---------------------------------------------------------------------------
# dtype -> numpy dtype string mapping (populated lazily)
# ---------------------------------------------------------------------------
_DTYPE_TO_NP = {
    0: "float32",   # FLOAT32
    1: "float16",   # FLOAT16
    2: "bfloat16",  # BFLOAT16  (numpy >= 1.25 has it; fall back to float16 if unavailable)
    3: "int32",     # INT32
    4: "int64",     # INT64
    5: "uint8",     # UINT8
    6: "bool",      # BOOL
}

_MAX_DIMS = 8


class Tensor:
    """Wraps an InferTensor C pointer. Owns the pointer and frees it on close()."""

    def __init__(self, ptr) -> None:
        """
        Parameters
        ----------
        ptr:
            A ctypes c_void_p value (int) or None.  Must be non-NULL.
        """
        if ptr is None or ptr == 0:
            raise RuntimeError("Tensor: received NULL pointer from C API")
        # Normalise to a plain Python int so attribute access is uniform.
        self._ptr: int | None = int(ptr)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_numpy(cls, arr) -> "Tensor":
        """Create a CPU tensor from a numpy ndarray.

        The array data is copied into a freshly allocated C tensor, so the
        caller is free to modify or discard *arr* afterwards.

        Parameters
        ----------
        arr:
            A numpy ndarray.  Supported dtypes: float32, float16, int32,
            int64, uint8, bool.

        Returns
        -------
        Tensor
        """
        import numpy as np  # noqa: PLC0415
        arr = np.ascontiguousarray(arr)

        # Map numpy dtype -> infergo dtype constant.
        _NP_TO_DTYPE = {
            "float32":  0,
            "float16":  1,
            "bfloat16": 2,
            "int32":    3,
            "int64":    4,
            "uint8":    5,
            "bool":     6,
        }
        dtype_str = arr.dtype.name
        if dtype_str not in _NP_TO_DTYPE:
            raise TypeError(f"Tensor.from_numpy: unsupported dtype '{dtype_str}'")
        dtype_int = _NP_TO_DTYPE[dtype_str]

        shape = list(arr.shape)
        ndim = len(shape)
        shape_arr = (ctypes.c_int * ndim)(*shape)
        ptr = lib.infer_tensor_alloc_cpu(shape_arr, ctypes.c_int(ndim), ctypes.c_int(dtype_int))
        if not ptr:
            raise RuntimeError("infer_tensor_alloc_cpu returned NULL")

        t = cls(ptr)

        # Copy array bytes into the tensor.
        nbytes = arr.nbytes
        src_ptr = arr.ctypes.data_as(ctypes.c_void_p)
        rc = lib.infer_tensor_copy_from(ctypes.c_void_p(t._ptr), src_ptr, ctypes.c_int(nbytes))
        _check(rc, "infer_tensor_copy_from")

        return t

    @classmethod
    def alloc_cpu(cls, shape: list[int], dtype: int = FLOAT32) -> "Tensor":
        """Allocate a zero-initialised CPU tensor.

        Parameters
        ----------
        shape:
            Dimension sizes, e.g. ``[2, 3, 4]``.
        dtype:
            One of the FLOAT32 / FLOAT16 / … constants from ``_types``.

        Returns
        -------
        Tensor
        """
        ndim = len(shape)
        if ndim == 0:
            raise ValueError("Tensor.alloc_cpu: shape must be non-empty")
        shape_arr = (ctypes.c_int * ndim)(*shape)
        ptr = lib.infer_tensor_alloc_cpu(shape_arr, ctypes.c_int(ndim), ctypes.c_int(dtype))
        if not ptr:
            raise RuntimeError("infer_tensor_alloc_cpu returned NULL")
        return cls(ptr)

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def to_numpy(self):
        """Copy the tensor data into a numpy array.

        If the tensor lives on a device (GPU), ``infer_tensor_to_host()`` is
        called first to bring the data to CPU memory.

        Returns
        -------
        numpy.ndarray
        """
        import numpy as np  # noqa: PLC0415
        self._require_open()

        # Bring to host memory (no-op if already on CPU).
        rc = lib.infer_tensor_to_host(ctypes.c_void_p(self._ptr))
        _check(rc, "infer_tensor_to_host")

        nbytes = lib.infer_tensor_nbytes(ctypes.c_void_p(self._ptr))
        data_ptr = lib.infer_tensor_data_ptr(ctypes.c_void_p(self._ptr))
        dtype_int = lib.infer_tensor_dtype(ctypes.c_void_p(self._ptr))

        dtype_str = _DTYPE_TO_NP.get(dtype_int, "float32")

        # Copy raw bytes from C memory into a numpy array.
        buf = (ctypes.c_uint8 * nbytes).from_address(data_ptr)
        flat = np.frombuffer(bytes(buf), dtype=dtype_str).copy()

        shape = self.shape
        return flat.reshape(shape)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> list[int]:
        """Return the tensor shape as a list of ints."""
        self._require_open()
        out = (ctypes.c_int * _MAX_DIMS)()
        ndim = lib.infer_tensor_shape(ctypes.c_void_p(self._ptr), out, ctypes.c_int(_MAX_DIMS))
        if ndim < 0:
            raise RuntimeError("infer_tensor_shape failed")
        return list(out[:ndim])

    @property
    def dtype(self) -> int:
        """Return the tensor dtype as an integer constant (e.g. FLOAT32 == 0)."""
        self._require_open()
        return lib.infer_tensor_dtype(ctypes.c_void_p(self._ptr))

    @property
    def nbytes(self) -> int:
        """Return total byte count of the tensor data."""
        self._require_open()
        return lib.infer_tensor_nbytes(ctypes.c_void_p(self._ptr))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Free the underlying C tensor. Safe to call multiple times."""
        if self._ptr is not None:
            lib.infer_tensor_free(ctypes.c_void_p(self._ptr))
            self._ptr = None

    def __enter__(self) -> "Tensor":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_open(self) -> None:
        if self._ptr is None:
            raise RuntimeError("Tensor has already been closed")
