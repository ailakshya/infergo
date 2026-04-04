"""
infergo.preprocess — Image pre-processing helpers backed by the infergo C API.

All functions accept and return :class:`~infergo.tensor.Tensor` instances so
the results compose naturally with :mod:`infergo.session` and
:mod:`infergo.postprocess`.
"""

from __future__ import annotations

import ctypes

from ._lib import lib
from ._types import _check
from .tensor import Tensor

__all__ = ["decode_image", "letterbox", "normalize", "stack_batch"]


def decode_image(data: bytes) -> Tensor:
    """Decode JPEG / PNG / WebP bytes into a ``[H, W, 3]`` float32 CPU Tensor.

    Parameters
    ----------
    data:
        Raw image bytes (JPEG, PNG, or WebP).

    Returns
    -------
    Tensor
        A ``[H, W, 3]`` float32 CPU tensor.

    Raises
    ------
    RuntimeError
        If the C library returns NULL (unsupported format, corrupt data, …).
    """
    n = len(data)
    # Use a ctypes char array so the pointer is stable across the call.
    buf = (ctypes.c_uint8 * n)(*data)
    ptr = lib.infer_preprocess_decode_image(buf, ctypes.c_int(n))
    if not ptr:
        raise RuntimeError("infer_preprocess_decode_image returned NULL")
    return Tensor(ptr)


def letterbox(src: Tensor, width: int, height: int) -> Tensor:
    """Letterbox-resize *src* to ``[height, width, 3]``.

    The image is resized while preserving its aspect ratio; the uncovered
    regions are padded with grey (128, 128, 128).

    Parameters
    ----------
    src:
        Source ``[H, W, 3]`` tensor (float32, CPU).
    width:
        Target width in pixels.
    height:
        Target height in pixels.

    Returns
    -------
    Tensor
        A ``[height, width, 3]`` float32 CPU tensor.

    Raises
    ------
    RuntimeError
        If the C library returns NULL.
    """
    src._require_open()
    ptr = lib.infer_preprocess_letterbox(
        ctypes.c_void_p(src._ptr),
        ctypes.c_int(width),
        ctypes.c_int(height),
    )
    if not ptr:
        raise RuntimeError("infer_preprocess_letterbox returned NULL")
    return Tensor(ptr)


def normalize(
    src: Tensor,
    scale: float,
    mean: list[float],
    std: list[float],
) -> Tensor:
    """Normalize a ``[H, W, 3]`` tensor to ``[C, H, W]`` layout.

    Applies the transformation::

        output[c, h, w] = (src[h, w, c] * scale - mean[c]) / std[c]

    Parameters
    ----------
    src:
        Source ``[H, W, 3]`` float32 CPU tensor.
    scale:
        Pixel value scaling factor (e.g. ``1/255.0``).
    mean:
        Per-channel mean, length 3.
    std:
        Per-channel standard deviation, length 3.

    Returns
    -------
    Tensor
        A ``[3, H, W]`` float32 CPU tensor.

    Raises
    ------
    RuntimeError
        If the C library returns NULL.
    ValueError
        If *mean* or *std* do not have exactly 3 elements.
    """
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("normalize: mean and std must each have exactly 3 elements")

    src._require_open()

    mean_arr = (ctypes.c_float * 3)(*mean)
    std_arr = (ctypes.c_float * 3)(*std)

    ptr = lib.infer_preprocess_normalize(
        ctypes.c_void_p(src._ptr),
        ctypes.c_float(scale),
        mean_arr,
        std_arr,
    )
    if not ptr:
        raise RuntimeError("infer_preprocess_normalize returned NULL")
    return Tensor(ptr)


def stack_batch(tensors: list[Tensor]) -> Tensor:
    """Stack N ``[C, H, W]`` tensors into a single ``[N, C, H, W]`` tensor.

    Parameters
    ----------
    tensors:
        A non-empty list of ``[C, H, W]`` float32 CPU tensors.  All tensors
        must have the same shape.

    Returns
    -------
    Tensor
        A ``[N, C, H, W]`` float32 CPU tensor.

    Raises
    ------
    RuntimeError
        If the C library returns NULL.
    ValueError
        If *tensors* is empty.
    """
    if not tensors:
        raise ValueError("stack_batch: tensors list must not be empty")

    n = len(tensors)
    for t in tensors:
        t._require_open()

    # Build a C array of void* (one per tensor pointer).
    PtrArray = ctypes.c_void_p * n
    ptrs = PtrArray(*(ctypes.c_void_p(t._ptr) for t in tensors))

    ptr = lib.infer_preprocess_stack_batch(ptrs, ctypes.c_int(n))
    if not ptr:
        raise RuntimeError("infer_preprocess_stack_batch returned NULL")
    return Tensor(ptr)
