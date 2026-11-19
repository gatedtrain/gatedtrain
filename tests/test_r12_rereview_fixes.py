"""Strict evaluation merges, Tier B evaluator selection and ratio bounds.

See R11A-28 and R12A-26 to R12A-29 in tests/README.md.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from _r11 import evaluate_items
from _support import EXIT_REFUSED, SKEL_ROOT, describe, parse_errors, run_cli
from _tierb import TIERB
from test_r11_items import ratio_arms, ratio_clause, ratio_reference
from test_r11_registration import Case
from test_tierb_merge import MergeCase

TIERB_EVAL = "gsm8k.nemo_evaluator@0.1"
DECOY = "decoy.metric@0.1"
STUB = "addition_em.cpu_stub@0.1"
DUMMY_HASH = "sha256:" + "5" * 64


def problems_of(exc) -> set:
    return {(p.rule, p.field) for p in exc.problems}


# ---------------------------------------------------------------------------
# M-1 (R11A-28 extended): merge-eval reads each run strictly
# ---------------------------------------------------------------------------

class TestMergeEvalStrictReader(MergeCase):
    """Each case corrupts runs[0]'s results.jsonl; the refusal names runs[0] and nothing is written."""

    def corrupt(self, edit):
        runs = self.runs()
        path = runs[0] / "gsm8k" / "results.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(edit(lines)) + "\n", encoding="utf-8")
        return runs

    def refused(self, edit, needle: str):
        out, proc = self.merge(self.corrupt(edit))
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = [e for e in parse_errors(proc.stderr) if e["rule"] == "MERGE-EVAL" and e["field"] == "runs[0]"]
        self.assertTrue(errs, describe(proc))
        self.assertTrue(any(needle in e["message"] for e in errs), (needle, errs))
        self.assertTrue(all(e.get("fix") for e in errs), errs)
        self.assertFalse(out.exists(), "nothing is written on a refusal")
        self.assertNotIn("Traceback", proc.stderr)

    def test_unparsable_line(self):
        self.refused(lambda ls: ls[:1] + ["{not json"] + ls[1:], "line 2")

    def test_non_object_line(self):
        self.refused(lambda ls: ls + ["[1, 2]"], f"line {len(self.rows) + 1}")

    def test_line_without_problem_idx(self):
        def edit(ls):
            rec = json.loads(ls[0])
            del rec["problem_idx"]
            return ls + [json.dumps(rec)]

        self.refused(edit, "problem_idx")

    def test_non_integer_problem_idx(self):
        """Choice: nothing is coerced; `"3"` would be item 3 to NeMo Evaluator's reader and an unknown item to
        the gate's."""
        def edit(ls):
            rec = json.loads(ls[3])
            rec["problem_idx"] = str(rec["problem_idx"])
            return ls[:3] + [json.dumps(rec)] + ls[4:]

        self.refused(edit, "line 4")

    def test_duplicate_problem_repeat(self):
        def edit(ls):
            dup = json.loads(ls[0])
            dup["reward"] = 1.0 - float(dup["reward"])
            return ls + [json.dumps(dup)]

        self.refused(edit, "duplicate")

    def test_malformed_line_before_duplicate_is_reported(self):
        def edit(ls):
            dup = json.loads(ls[0])
            dup["reward"] = 1.0 - float(dup["reward"])
            return ls[:1] + ["{not json"] + ls[1:] + [json.dumps(dup)]

        self.refused(edit, "line 2")

    def test_blank_lines_merge(self):
        """Control: blank lines are skipped, as in read_arm_records; the merged file reads clean."""
        from gatedtrain import runner

        runs = self.corrupt(lambda ls: ls[:2] + ["", "   "] + ls[2:])
        out, proc = self.merge(runs)
        self.assertEqual(proc.returncode, 0, describe(proc))
        recs = runner.read_arm_records(out)
        self.assertEqual(len(recs), len(self.rows) * 2)
        self.assertFalse(any(runner.MALFORMED_LINE in r for r in recs))


# ---------------------------------------------------------------------------
# L-1, L-4 (R12A-27): Tier B commands gate on the Tier B evaluator
# ---------------------------------------------------------------------------

class TierbRegCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-rereview-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        text = (TIERB / "registration.template.yaml").read_text(encoding="utf-8")
        self.base = yaml.safe_load(text.replace("{prompt_set_hash}", "sha256:" + "0" * 64))
        self.real = self.base["evaluators"][0]
        self.n = 0

    def write(self, evaluators, **clause_over) -> Path:
        self.n += 1
        reg = json.loads(json.dumps(self.base))
        reg["evaluators"] = evaluators
        reg["clause"].update(clause_over)
        path = self.tmp / f"reg{self.n}.yaml"
        path.write_text(yaml.safe_dump(reg, sort_keys=False), encoding="utf-8")
        return path

    def load(self, path):
        from gatedtrain.tierb import load_registration

        return load_registration(path)

    def refused(self, path) -> set:
        from gatedtrain.errors import Refused

        with self.assertRaises(Refused) as cm:
            self.load(path)
        return problems_of(cm.exception)


class TestTierbDecidingEvaluator(TierbRegCase):
    """Tier B commands select the Tier B evaluator even when a decoy is listed first."""

    def test_decoy_first_direction_checked_against_tierb(self):
        path = self.write([{"id": DECOY, "manifest_hash": self.real["manifest_hash"]}, self.real],
                          direction="lower_is_better")
        self.assertIn(("REG-FIELD", "registration.clause.direction"), self.refused(path))

    def test_stub_first_direction_checked_against_tierb(self):
        path = self.write([{"id": STUB, "manifest_hash": DUMMY_HASH}, self.real], estimator="ratio_of_sums")
        self.assertIn(("REG-FIELD", "registration.clause.estimator"), self.refused(path))

    def test_decoy_first_consistent_loads(self):
        """Control, and the scorer comes from the Tier B entry."""
        from gatedtrain.tierb import EVALUATOR_ID, registered_scorer

        path = self.write([{"id": DECOY, "manifest_hash": DUMMY_HASH}, {**self.real, "scorer": "numeric_match"}])
        reg = self.load(path)
        self.assertEqual(registered_scorer(reg, EVALUATOR_ID), "numeric_match")


class TestTierbEvaluatorListed(TierbRegCase):
    """L-4: no index-0 fallback; the Tier B evaluator must be listed."""

    FIELD = ("REG-FIELD", "registration.evaluators")

    def test_only_decoy_refused(self):
        path = self.write([{"id": DECOY, "manifest_hash": DUMMY_HASH}])
        self.assertIn(self.FIELD, self.refused(path))

    def test_only_decoy_lower_is_better_refused(self):
        path = self.write([{"id": DECOY, "manifest_hash": DUMMY_HASH}], direction="lower_is_better")
        self.assertIn(self.FIELD, self.refused(path))

    def test_only_stub_refused(self):
        path = self.write([{"id": STUB, "manifest_hash": DUMMY_HASH}])
        self.assertIn(self.FIELD, self.refused(path))

    def test_cli_merge_eval_refused(self):
        path = self.write([{"id": DECOY, "manifest_hash": DUMMY_HASH}])
        out = self.tmp / "merged"
        proc = run_cli(["merge-eval", "--registration", str(path), "--out", str(out),
                        str(self.tmp / "r0"), str(self.tmp / "r1")], cwd=SKEL_ROOT, timeout=60)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn(self.FIELD, {(e["rule"], e["field"]) for e in parse_errors(proc.stderr)}, describe(proc))
        self.assertFalse(out.exists())

    def test_cli_nel_config_refused(self):
        path = self.write([{"id": DECOY, "manifest_hash": DUMMY_HASH}])
        out = self.tmp / "configs"
        proc = run_cli(["nel-config", "--registration", str(path), "--model-url", "http://localhost:8000/v1",
                        "--model-id", "m", "--eval-root", "/tmp/eval", "--out", str(out)], cwd=SKEL_ROOT, timeout=60)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn(self.FIELD, {(e["rule"], e["field"]) for e in parse_errors(proc.stderr)}, describe(proc))
        self.assertFalse(out.exists())


# ---------------------------------------------------------------------------
# L-3 (R12A-26): duplicate evaluator ids
# ---------------------------------------------------------------------------

class TestDuplicateEvaluatorIds(Case):

    def test_validate_refuses_duplicate_id(self):
        def reg(d):
            d["evaluators"] = [dict(d["evaluators"][0]), {**d["evaluators"][0], "manifest_hash": "sha256:" + "6" * 64}]

        self.assert_refused(self.tiera(reg=reg), "REG-FIELD", "registration.evaluators[1].id")

    def test_check_contents_without_gate(self):
        from gatedtrain.registration import check_contents

        reg = yaml.safe_load((Path(__file__).parent.parent / "demo" / "registration.yaml").read_text(encoding="utf-8"))
        reg["evaluators"][0]["manifest_hash"] = DUMMY_HASH
        reg["evaluators"] = [reg["evaluators"][0], {"id": DECOY, "manifest_hash": DUMMY_HASH},
                             dict(reg["evaluators"][0])]
        self.assertIn(("REG-FIELD", "registration.evaluators[2].id"), {(p.rule, p.field) for p in check_contents(reg)})

    def test_tierb_duplicate_with_different_scorer(self):
        """Duplicate evaluator IDs are rejected even when the entries name different scorers."""
        case = TierbRegCase()
        case.setUp()
        try:
            path = case.write([dict(case.real), {**case.real, "scorer": "numeric_match"}])
            self.assertIn(("REG-FIELD", "registration.evaluators[1].id"), case.refused(path))
        finally:
            case.doCleanups()


# ---------------------------------------------------------------------------
# L-2 (R12A-18): evaluate_constraint refuses a repeated item ID
# ---------------------------------------------------------------------------

class TestConstraintDuplicateItems(unittest.TestCase):

    def test_duplicate_items_raise(self):
        from gatedtrain.constraints import evaluate_constraint
        from test_r12_constraints import arm, constraint_clause, constraint_set, qualification, rows

        items = [f"i{i:04d}" for i in range(368)]
        with self.assertRaises(ValueError):
            evaluate_constraint(constraint_clause(), constraint_set(), qualification(), arm(rows(368)),
                                items=items + items[:1])

    def test_control_without_duplicate(self):
        from gatedtrain.constraints import evaluate_constraint
        from test_r12_constraints import arm, constraint_clause, constraint_set, qualification, rows

        res = evaluate_constraint(constraint_clause(), constraint_set(), qualification(), arm(rows(368)),
                                  items=[f"i{i:04d}" for i in range(368)])
        self.assertIn(res["status"], ("PASS", "FAIL", "INCONCLUSIVE", "MISSING"))


# ---------------------------------------------------------------------------
# L-5 (R12A-28): extreme ratio parts
# ---------------------------------------------------------------------------

class TestRatioExtremeParts(unittest.TestCase):

    ONE = (("num_huge_int", "numerator", 10 ** 400), ("den_huge_int", "denominator", 10 ** 400),
           ("num_neg_huge_int", "numerator", -(10 ** 400)), ("num_1e308", "numerator", 1e308),
           ("den_1e308", "denominator", 1e308), ("num_above_limit", "numerator", 1e13),
           ("den_above_limit", "denominator", 1e13), ("den_below_limit", "denominator", 1e-13),
           ("den_subnormal", "denominator", 1e-320))

    def check_finite(self, res):
        for k in ("delta_hat", "ci_lower", "ci_upper", "sd_diff", "achieved_power", "baseline_score",
                  "candidate_score"):
            self.assertTrue(res[k] is not None and math.isfinite(res[k]), (k, res[k]))

    def test_one_extreme_part_is_out_of_range(self):
        ref = ratio_reference(keep=[i for i in range(60) if i != 3])
        for label, field, value in self.ONE:
            with self.subTest(label):
                items, b, c, _ = ratio_arms()
                c["records"][3][field] = value
                res = evaluate_items(ratio_clause(), b, c, items=items, seed=5, n_boot=500, max_unpaired=1)
                self.assertEqual(res["missing_reasons"]["candidate"], {"out_of_range": 1}, res)
                self.assertEqual(res["n_items"], 59)
                self.assertAlmostEqual(res["delta_hat"], ref["delta"], places=12)
                self.check_finite(res)

    def test_above_bound_is_missing_not_exception(self):
        for label, field, value in self.ONE:
            with self.subTest(label):
                items, b, c, _ = ratio_arms()
                c["records"][3][field] = value
                res = evaluate_items(ratio_clause(), b, c, items=items, seed=5, n_boot=500)
                self.assertEqual(res["status"], "MISSING", res)
                self.assertIn("missing_above_bound", res["status_reasons"])

    def test_sums_stay_finite(self):
        """Two parts of 1e308 in one arm would overflow the arm sum: both are out of range."""
        items, b, c, _ = ratio_arms()
        c["records"][3]["numerator"] = 1e308
        c["records"][4]["numerator"] = 1e308
        res = evaluate_items(ratio_clause(), b, c, items=items, seed=5, n_boot=500, max_unpaired=2)
        self.assertEqual(res["missing_reasons"]["candidate"], {"out_of_range": 2}, res)
        self.check_finite(res)

    def test_at_limit_is_usable(self):
        """The limits themselves are usable (|numerator| = 1e12, denominator = 1e-12 or 1e12)."""
        for label, field, value in (("num_at", "numerator", 1e12), ("den_at_hi", "denominator", 1e12),
                                    ("den_at_lo", "denominator", 1e-12)):
            with self.subTest(label):
                items, b, c, _ = ratio_arms()
                c["records"][3][field] = value
                res = evaluate_items(ratio_clause(), b, c, items=items, seed=5, n_boot=500)
                self.assertEqual(res["missing_reasons"]["candidate"], {}, res)
                self.check_finite(res)


if __name__ == "__main__":
    unittest.main()
