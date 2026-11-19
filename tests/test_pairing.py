"""Pairing by (problem_idx, repeat) as NeMo Evaluator's `load_paired_records`
(src/nemo_evaluator/engine/comparison.py:360 at ea35ba3), MVP-SPEC section 2.7.

`nemo_evaluator` is not an allowed dependency, so the skeleton reimplements
the function. This test loads the upstream function's source from the
pinned clone (read only, without importing the nemo_evaluator package) and
checks that both agree on an edge-case fixture.
"""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path
from typing import Any

from _support import FIXTURES, NEMO_EVAL_COMPARISON

PAIRING = FIXTURES / "pairing"


def upstream_load_paired_records():
    if not NEMO_EVAL_COMPARISON.is_file():
        raise unittest.SkipTest(f"pinned clone not available: {NEMO_EVAL_COMPARISON}")
    tree = ast.parse(NEMO_EVAL_COMPARISON.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "load_paired_records":
            assert node.lineno == 360, f"pinned line moved: {node.lineno}"
            module = ast.Module(body=[node], type_ignores=[])
            ns: dict[str, Any] = {"json": json, "Path": Path, "Any": Any}
            exec(compile(module, str(NEMO_EVAL_COMPARISON), "exec"), ns)
            return ns["load_paired_records"]
    raise AssertionError("load_paired_records not found in the pinned comparison.py")


class TestLoadPairedRecords(unittest.TestCase):
    def test_upstream_reference_on_fixture(self):
        """Pins the fixture's expected keys, so the reference itself is checked."""
        ref = upstream_load_paired_records()
        base = ref(PAIRING / "baseline")
        self.assertEqual(sorted(base), [(0, 0), (0, 1), (1, 0), (2, 0), (3, 0), (4, 1)])
        self.assertEqual(base[(3, 0)]["reward"], 1.0, "duplicate key: last line wins")
        self.assertEqual(ref(PAIRING / "missing"), {}, "missing results.jsonl -> {}")

    def test_skeleton_agrees_with_upstream(self):
        from gatedtrain.gates import load_paired_records

        ref = upstream_load_paired_records()
        for arm in ("baseline", "candidate", "missing"):
            with self.subTest(arm=arm):
                self.assertEqual(load_paired_records(PAIRING / arm), ref(PAIRING / arm))

    def test_evaluate_uses_the_paired_keys(self):
        """Pairs are the key intersection: (0,0), (0,1), (1,0), (2,0), (3,0), (4,1).
        Per-prompt mean differences (K averaged per prompt, 05:98):
        p0: mean(1,1) - mean(1,0) = 0.5; p1: 1 - 0 = 1; p2: 1 - 1 = 0;
        p3: 0 - 1 = -1; p4: 1 - 0.5 = 0.5 -> Delta-hat = 1.0 / 5 = 0.2."""
        from gatedtrain.gates import evaluate, load_paired_records

        clause = {
            "type": "superiority", "metric": "exact_match", "margin": 0.0, "alpha": 0.025, "power": 0.8,
            "planning_effect": 0.10, "sigma_plan": "fixed", "unit": "prompt", "samples_per_prompt": 2,
            "eval_seeds": [11, 12], "interval": "cluster_bootstrap", "blocking": True,
        }
        res = evaluate(clause, load_paired_records(PAIRING / "baseline"), load_paired_records(PAIRING / "candidate"),
                       seed=3, n_boot=1000)
        self.assertEqual(res["n_prompts"], 5)
        self.assertAlmostEqual(res["delta_hat"], 0.2, places=12)


if __name__ == "__main__":
    unittest.main()
