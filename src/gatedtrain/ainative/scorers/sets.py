"""AN-17: set F1 and field-level F1. This file imports only the standard library and `gatedtrain.canonical`, so its
hash is the scorers' code hash (`scorers.manifest_document`)."""

from __future__ import annotations

from typing import Any

from ...canonical import canonical_json


def _f1(pred: set, gold: set) -> float:
    """2|P & L| / (|P| + |L|); 1.0 when both are empty."""
    if not pred and not gold:
        return 1.0
    return 2 * len(pred & gold) / (len(pred) + len(gold))


def set_label_ok(label: Any) -> bool:
    """A list."""
    return isinstance(label, list)


def field_label_ok(label: Any) -> bool:
    """An object."""
    return isinstance(label, dict)


def set_f1(prediction: Any, label: Any) -> float:
    """F1 between the prediction and the label as sets: elements compared by canonical JSON, duplicates collapse. A
    prediction that is not a list, or not canonical JSON, scores 0.0."""
    if not (isinstance(prediction, list) and isinstance(label, list)):
        return 0.0
    try:
        pred = {canonical_json(x) for x in prediction}
        gold = {canonical_json(x) for x in label}
    except ValueError:
        return 0.0
    return _f1(pred, gold)


def field_f1(prediction: Any, label: Any) -> float:
    """F1 over the `(key, canonical JSON of the value)` pairs of the top-level fields. A prediction that is not an
    object, or not canonical JSON, scores 0.0."""
    if not (isinstance(prediction, dict) and isinstance(label, dict)):
        return 0.0
    try:
        canonical_json(prediction)
        pred = {(k, canonical_json(v)) for k, v in prediction.items()}
        gold = {(k, canonical_json(v)) for k, v in label.items()}
    except ValueError:
        return 0.0
    return _f1(pred, gold)
