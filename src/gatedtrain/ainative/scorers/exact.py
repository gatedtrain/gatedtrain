"""AN-17: exact match. This file imports only the standard library and `gatedtrain.canonical`, so its hash is the
scorer's code hash (`scorers.manifest_document`)."""

from __future__ import annotations

from typing import Any

from ...canonical import canonical_json


def exact_match(prediction: Any, label: Any) -> float:
    """1.0 when the prediction equals the label as canonical JSON values (key order and whitespace do not matter),
    else 0.0. A prediction that is not canonical JSON (for example NaN) scores 0.0."""
    try:
        return 1.0 if canonical_json(prediction) == canonical_json(label) else 0.0
    except ValueError:
        return 0.0


def label_ok(label: Any) -> bool:
    """Any JSON value canonical JSON can encode."""
    try:
        canonical_json(label)
    except ValueError:
        return False
    return True
