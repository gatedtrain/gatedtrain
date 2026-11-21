"""AN-17: normalised string match (Unicode NFKC, case fold, whitespace collapse). This file imports only the
standard library and `gatedtrain.canonical`, so its hash is the scorer's code hash (`scorers.manifest_document`)."""

from __future__ import annotations

import unicodedata
from typing import Any

from ...canonical import canonical_json


def _norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def label_ok(label: Any) -> bool:
    """A string."""
    return isinstance(label, str)


def normalised_match(prediction: Any, label: Any) -> float:
    """1.0 when both are strings and equal after NFKC, case folding and collapsing whitespace, else 0.0. A
    prediction that is not canonical JSON scores 0.0."""
    try:
        canonical_json(prediction)
    except ValueError:
        return 0.0
    if not (isinstance(prediction, str) and isinstance(label, str)):
        return 0.0
    return 1.0 if _norm(prediction) == _norm(label) else 0.0
