"""Per-item evaluation records (05 section 3.8 "Per-item record (package A)"; design revision 11, R11-9).

A record is one evaluation of one item under one repeat:

    {item, repeat, seed, score, error, sub_metrics, output_ref}

plus `numerator` and `denominator` for a clause with `estimator: ratio_of_sums` (05 section 3.1, R11-21).
`item` is an opaque key from the registered item set (never a row position; MULTIMODAL-SPEC MM-1),
`repeat` is in range(K), `seed` is `eval_seeds[repeat]`, exactly one of `score` (natural units) and `error`
(`{class, message}`) is null, and `sub_metrics` and `output_ref` never decide a result. Each arm records the
evaluated checkpoint's `model_manifest_hash` once, beside its records (`gates.evaluate_items`).

`from_nemo_evaluator` converts a NeMo Evaluator per-sample record (engine/eval_loop.py at ea35ba3) into this
shape (tests/README.md section 11.16, R11A-5). Module name: the maintainers' choice.

`EVALUATORS` is the prototype's stand-in for the evaluator manifest (05 section 3.1 direction, 05 section 6.1
score range; design choice 5 in tests/README.md section 11.16): the direction, score range and estimators
each prototype evaluator declares.
"""

from __future__ import annotations

import hashlib
from typing import Any

RECORD_FIELDS = ("item", "repeat", "seed", "score", "error", "sub_metrics", "output_ref")
RATIO_FIELDS = ("numerator", "denominator")
DIRECTIONS = ("higher_is_better", "lower_is_better")
ESTIMATORS = ("mean", "ratio_of_sums")

# Evaluator id -> what its manifest would declare (prototype table; R11A-1). Neither prototype evaluator
# writes a numerator and denominator, so neither declares `ratio_of_sums`.
EVALUATORS: dict[str, dict[str, Any]] = {
    "gsm8k.nemo_evaluator@0.1": {"direction": "higher_is_better", "score_range": (0.0, 1.0), "estimators": ("mean",)},
    "addition_em.cpu_stub@0.1": {"direction": "higher_is_better", "score_range": (0.0, 1.0), "estimators": ("mean",)},
}


def _external_evaluators() -> dict[str, dict[str, Any]]:
    """The external-run evaluators of the adapter registry (MULTIMODAL-SPEC MM-13, MM-37; tests/README.md
    section 11.18 EXT-7, EXT-31): direction and score range as the registry row declares, estimator mean."""
    from .ext.registry import EVALUATOR_ROWS

    return {eid: {"direction": row["direction"], "score_range": tuple(row["score_range"]), "estimators": ("mean",)}
            for eid, row in EVALUATOR_ROWS.items()}


EVALUATORS.update(_external_evaluators())


def score_range(evaluator_id: str | None) -> tuple[float | None, float | None]:
    """The declared score range of an evaluator (05 section 6.1, R11-11); (None, None) when undeclared."""
    decl = EVALUATORS.get(evaluator_id or "")
    return tuple(decl["score_range"]) if decl else (None, None)


def _error(cls: str, message: str) -> dict[str, str]:
    return {"class": cls, "message": message}


def from_nemo_evaluator(record: dict, *, scorer_key: str | None = None) -> dict[str, Any]:
    """One NeMo Evaluator per-sample record as a per-item record (05 section 3.8; R11A-5).

    `item = problem_idx`, `repeat` (default 0, as load_paired_records), `seed = metadata.eval_seed` (null when
    absent: the gate counts the item missing, reason `no_seed`), `score = reward`, or
    `scorer_rewards[scorer_key]` with a scorer (interim rule G-21). A scorer that raised leaves
    `scoring_details[scorer_key] = {error, reward: 0.0}` (engine/eval_loop.py:719-721 at ea35ba3): that record
    becomes score null and an error, never 0. `output_ref` names the SHA-256 of `model_response` (design
    choice 9); null when the record holds no response text.
    """
    meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    seed = meta.get("eval_seed")
    score: Any = None
    error = None
    if scorer_key is not None:
        details = record.get("scoring_details") if isinstance(record.get("scoring_details"), dict) else {}
        detail = details.get(scorer_key)
        rewards = record.get("scorer_rewards") if isinstance(record.get("scorer_rewards"), dict) else {}
        if isinstance(detail, dict) and "error" in detail:
            error = _error("scorer_error", str(detail["error"]))
        elif scorer_key not in rewards or rewards[scorer_key] is None:
            error = _error("missing_score", f"no scorer_rewards[{scorer_key!r}] in the record")
        else:
            score = rewards[scorer_key]
    elif record.get("reward") is None:
        error = _error("missing_score", "no `reward` in the record")
    else:
        score = record["reward"]
    response = record.get("model_response")
    output_ref = ({"field": "model_response",
                   "sha256": "sha256:" + hashlib.sha256(response.encode("utf-8")).hexdigest()}
                  if isinstance(response, str) else None)
    return {"item": record.get("problem_idx"), "repeat": record.get("repeat", 0), "seed": seed, "score": score,
            "error": error, "sub_metrics": None, "output_ref": output_ref}
