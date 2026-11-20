"""AN-7: the scorer library of the AI-native layer (step 1: exact match).

`exact_match` is the deciding scorer `evaluate_sealed` uses; it is the only function that scores sealed items in step
1. The evaluator id `EVALUATOR_ID` is registered with a manifest hash computed from this file's bytes, so changing the
scorer changes the registered identity: `registration.resolved_evaluator_hash` resolves the id to `manifest_hash()`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..canonical import canonical_json, file_sha256, sha256_bytes

EVALUATOR_ID = "exact_match.ainative@0.1"
DIRECTION = "higher_is_better"


def exact_match(prediction: Any, label: Any) -> float:
    """1.0 when the prediction equals the label as canonical JSON values (key order and whitespace do not matter),
    else 0.0. A prediction that is not canonical JSON (for example NaN) scores 0.0."""
    try:
        return 1.0 if canonical_json(prediction) == canonical_json(label) else 0.0
    except ValueError:
        return 0.0


def manifest_document() -> dict:
    return {"evaluator": EVALUATOR_ID, "scorer": "exact_match", "direction": DIRECTION, "version": 1,
            "code": {Path(__file__).name: file_sha256(Path(__file__))}}


def manifest_hash() -> str:
    """The evaluator identity a registration carries as `manifest_hash`."""
    return sha256_bytes(canonical_json(manifest_document()))
