"""AN-17: numeric match with a relative and an absolute tolerance. This file imports only the standard library and
`gatedtrain.canonical`, so its hash is the scorer's code hash (`scorers.manifest_document`)."""

from __future__ import annotations

import math
from typing import Any

from ...canonical import canonical_json


def _finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def label_ok(label: Any) -> bool:
    """A finite int or float (not a bool)."""
    return _finite_number(label)


def numeric_match(prediction: Any, label: Any, *, rel_tol: float = 0.0, abs_tol: float = 0.0) -> float:
    """1.0 when `abs(p - label) <= max(rel_tol * abs(label), abs_tol)`, else 0.0. The prediction is an int or float
    (not a bool) or a string `float()` parses; a value that is not finite, or a prediction that is not canonical
    JSON, scores 0.0."""
    try:
        canonical_json(prediction)
    except ValueError:
        return 0.0
    if not label_ok(label):
        return 0.0
    if isinstance(prediction, str):
        try:
            p = float(prediction)
        except ValueError:
            return 0.0
    elif _finite_number(prediction):
        p = float(prediction)
    else:
        return 0.0
    if not math.isfinite(p):
        return 0.0
    return 1.0 if abs(p - label) <= max(rel_tol * abs(label), abs_tol) else 0.0
