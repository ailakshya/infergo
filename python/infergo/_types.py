"""
_types.py — ctypes Structure definitions, integer constants, and the _check helper.
"""

import ctypes
from ._lib import lib

# ---------------------------------------------------------------------------
# C Structures
# ---------------------------------------------------------------------------

class InferClassResult(ctypes.Structure):
    """Maps to: typedef struct { int label_idx; float confidence; } InferClassResult;"""
    _fields_ = [
        ("label_idx",   ctypes.c_int),
        ("confidence",  ctypes.c_float),
    ]


class InferBox(ctypes.Structure):
    """Maps to: typedef struct { float x1, y1, x2, y2; int class_idx; float confidence; } InferBox;"""
    _fields_ = [
        ("x1",          ctypes.c_float),
        ("y1",          ctypes.c_float),
        ("x2",          ctypes.c_float),
        ("y2",          ctypes.c_float),
        ("class_idx",   ctypes.c_int),
        ("confidence",  ctypes.c_float),
    ]

# ---------------------------------------------------------------------------
# Error codes
# ---------------------------------------------------------------------------
INFER_OK          = 0
INFER_ERR_NULL    = 1
INFER_ERR_INVALID = 2
INFER_ERR_OOM     = 3
INFER_ERR_CUDA    = 4
INFER_ERR_LOAD    = 5
INFER_ERR_RUNTIME = 6
INFER_ERR_SHAPE   = 7
INFER_ERR_DTYPE   = 8
INFER_ERR_CANCELLED = 9

# ---------------------------------------------------------------------------
# Dtype constants
# ---------------------------------------------------------------------------
FLOAT32  = 0
FLOAT16  = 1
BFLOAT16 = 2
INT32    = 3
INT64    = 4
UINT8    = 5
BOOL     = 6

# ---------------------------------------------------------------------------
# Error-check helper
# ---------------------------------------------------------------------------

def _check(rc: int, msg: str = "") -> None:
    """Raise RuntimeError if *rc* is not INFER_OK.

    The error detail is pulled from the C library via infer_last_error_string().

    Args:
        rc:  Return code from a C API function.
        msg: Optional human-readable context (e.g. the call site name).

    Raises:
        RuntimeError: when rc != INFER_OK, with the C-side error string appended.
    """
    if rc != INFER_OK:
        c_detail = lib.infer_last_error_string()
        detail = c_detail.decode("utf-8", errors="replace") if c_detail else "(no detail)"
        prefix = f"{msg}: " if msg else ""
        raise RuntimeError(f"{prefix}{detail}")
