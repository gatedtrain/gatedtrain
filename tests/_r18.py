"""Design revision 18 test helpers (R18-1 to R18-6; tests/README.md section 11.30): a second evaluator for the
synthetic adapter, so one gate can block on two clauses, each decided by its own evaluator.

`synthetic.length@0.1` scores the same items as `synthetic.items@0.1` with a lower-is-better count (the response
length in tokens). It is registered only by this module: in a CLI subprocess run as `python -m _r18 <gatedtrain
arguments>` (tests directory on PYTHONPATH), and in process with `install()` inside `mock.patch.dict` of the
registry tables (`patched()`), so the synthetic adapter's own rows and the other tests stay as they are.
"""

from __future__ import annotations

import contextlib
import sys
from typing import Any
from unittest import mock

from gatedtrain.testing.synthetic import EVALUATOR_ID, SyntheticAdapter

LENGTH_ID = "synthetic.length@0.1"
LENGTH_METRIC = "response_tokens"
LENGTH_SCORER = {"name": "token_count", "version": "fixture-1", "weights_sha256": None,
                 "code_sha256": "sha256:" + "5e" * 32}
LENGTH_ROW: dict[str, Any] = {
    "metric": LENGTH_METRIC,
    "direction": "lower_is_better",
    "score_range": (0, None),
    "scorer": LENGTH_SCORER,
    "tier": "T3",
    "binary": False,
    "error_classes": ("scorer_error", "missing_score"),
    "score_outcomes": (),
    "payload_text_field": "question",
    "item_payload": ("question", "answer"),
    "normalizer": False,
    "env_version": False,
    "framework": "fixture",
    "nel_benchmark": None,
    "samples": "30, 2",
}
ADAPTER_CLASS = "_r18:TwoEvaluatorAdapter"


class TwoEvaluatorAdapter(SyntheticAdapter):
    """The synthetic adapter, gating on both evaluators."""

    evaluator_ids = (EVALUATOR_ID, LENGTH_ID)

    def metric(self, evaluator_id: str) -> dict[str, Any]:
        from gatedtrain.ext.registry import EVALUATOR_ROWS

        if evaluator_id not in self.evaluator_ids:
            raise KeyError(evaluator_id)
        row = EVALUATOR_ROWS[evaluator_id]
        return {"metric": row["metric"], "direction": row["direction"], "score_range": tuple(row["score_range"]),
                "error_classes": tuple(row["error_classes"]), "score_outcomes": tuple(row["score_outcomes"]),
                "scorer": dict(row["scorer"]), "binary": row["binary"], "tier": row["tier"]}


def install() -> None:
    """Register the synthetic adapter with both evaluators in this process (the tables must be patched or
    disposable: `patched()` restores them)."""
    from gatedtrain import items, spec
    from gatedtrain.ext import registry
    from gatedtrain.testing import synthetic

    synthetic.install()
    registry.EVALUATOR_ROWS[LENGTH_ID] = LENGTH_ROW
    registry.ADAPTERS[synthetic.NAME] = ADAPTER_CLASS
    spec._external_ids()
    items.EVALUATORS.update(items._external_evaluators())


@contextlib.contextmanager
def patched():
    """`install()` inside patched registry, item and spec tables, restored on exit."""
    from gatedtrain import items, spec
    from gatedtrain.ext import registry

    with mock.patch.dict(registry.EVALUATOR_ROWS), mock.patch.dict(registry.ADAPTERS), \
            mock.patch.dict(registry.TRAINER_ROWS), mock.patch.dict(items.EVALUATORS), \
            mock.patch.dict(spec.EVALUATOR_BACKENDS), mock.patch.dict(spec.TRAINER_BACKENDS):
        install()
        yield


def length_identity_hash() -> str:
    """EXT-6: the identity hash `ext.registry.evaluator_identity_hash` resolves for synthetic.length@0.1."""
    from gatedtrain.canonical import canonical_json, sha256_bytes

    return sha256_bytes(canonical_json({"evaluator": LENGTH_ID, "metric": LENGTH_METRIC,
                                        "direction": "lower_is_better", "scorer": dict(LENGTH_SCORER)}))


def main(argv: list[str] | None = None) -> int:
    install()
    from gatedtrain.cli import main as cli_main

    return cli_main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    import _r18                       # register the importable module's class, not __main__'s

    sys.exit(_r18.main())
