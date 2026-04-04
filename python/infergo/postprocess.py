"""
infergo.postprocess — Post-processing helpers backed by the infergo C API.

Functions accept :class:`~infergo.tensor.Tensor` instances produced by
:mod:`infergo.session` or :mod:`infergo.preprocess`.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

from ._lib import lib
from ._types import InferClassResult, InferBox, INFER_OK, _check
from .tensor import Tensor

__all__ = ["ClassResult", "Box", "classify", "nms", "normalize_embedding"]

# ---------------------------------------------------------------------------
# Result dataclasses (thin Python view over C structs)
# ---------------------------------------------------------------------------


@dataclass
class ClassResult:
    """Top-k classification result for a single class."""

    label_idx: int
    confidence: float


@dataclass
class Box:
    """Bounding box produced by NMS."""

    x1: float
    y1: float
    x2: float
    y2: float
    class_idx: int
    confidence: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify(logits: Tensor, top_k: int = 5) -> list[ClassResult]:
    """Softmax + top-k classification over a logits tensor.

    Parameters
    ----------
    logits:
        1-D float32 tensor of raw class logits (shape ``[num_classes]``).
    top_k:
        Number of top results to return.

    Returns
    -------
    list[ClassResult]
        Up to *top_k* results, ordered from most to least confident.

    Raises
    ------
    RuntimeError
        If the C call fails.
    """
    logits._require_open()

    # Allocate an array of InferClassResult structs for the C function to fill.
    ResultArray = InferClassResult * top_k
    out = ResultArray()

    rc = lib.infer_postprocess_classify(
        ctypes.c_void_p(logits._ptr),
        ctypes.c_int(top_k),
        out,
    )
    _check(rc, "infer_postprocess_classify")

    # rc is the number of results written when non-negative.
    # _check would have raised if rc < 0, so use the requested top_k capped
    # by the actual number written (rc >= 0 guaranteed here).
    count = min(rc, top_k)
    return [
        ClassResult(label_idx=out[i].label_idx, confidence=out[i].confidence)
        for i in range(count)
    ]


def nms(
    predictions: Tensor,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_boxes: int = 300,
) -> list[Box]:
    """Non-maximum suppression on a YOLO output tensor.

    Parameters
    ----------
    predictions:
        YOLO predictions tensor (shape ``[num_predictions, 5 + num_classes]``
        in the standard YOLO format).
    conf_thresh:
        Minimum confidence threshold; boxes below this are discarded.
    iou_thresh:
        IoU threshold for suppression.
    max_boxes:
        Maximum number of boxes to return.

    Returns
    -------
    list[Box]
        Surviving bounding boxes after NMS, up to *max_boxes* entries.

    Raises
    ------
    RuntimeError
        If the C call fails.
    """
    predictions._require_open()

    BoxArray = InferBox * max_boxes
    out = BoxArray()

    rc = lib.infer_postprocess_nms(
        ctypes.c_void_p(predictions._ptr),
        ctypes.c_float(conf_thresh),
        ctypes.c_float(iou_thresh),
        out,
        ctypes.c_int(max_boxes),
    )
    _check(rc, "infer_postprocess_nms")

    count = min(rc, max_boxes)
    return [
        Box(
            x1=out[i].x1,
            y1=out[i].y1,
            x2=out[i].x2,
            y2=out[i].y2,
            class_idx=out[i].class_idx,
            confidence=out[i].confidence,
        )
        for i in range(count)
    ]


def normalize_embedding(t: Tensor) -> None:
    """L2-normalize an embedding tensor in-place.

    Parameters
    ----------
    t:
        Embedding tensor to normalize.  Modified in place.

    Raises
    ------
    RuntimeError
        If the C call fails.
    """
    t._require_open()
    rc = lib.infer_postprocess_normalize_embedding(ctypes.c_void_p(t._ptr))
    _check(rc, "infer_postprocess_normalize_embedding")
