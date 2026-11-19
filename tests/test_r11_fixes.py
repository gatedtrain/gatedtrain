"""Registration and gate regressions for evaluator identity, missing records and ratio statistics.

See R11A-23 to R11A-31 in tests/README.md.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml
from scipy.stats import norm

from _r11 import arm, clause, evaluate_items, ids, matrix_arms, rec
from _support import (EXIT_REFUSED, WORKSPACE, describe, file_sha256, parse_errors, run_cli, slow)
from test_r11_items import ratio_arms, ratio_clause, ratio_reference
from test_r11_registration import Case

DUMMY_HASH = "sha256:" + "5" * 64
OTHER_HASH = "sha256:" + "6" * 64
DECOY = "decoy.metric@0.1"
STUB = "addition_em.cpu_stub@0.1"
TIERB_EVAL = "gsm8k.nemo_evaluator@0.1"
# A test-only evaluator declaration patched into items.EVALUATORS: it differs from both prototype evaluators in
# direction, estimators and range, so a check that reads evaluators[0] instead of the deciding one is visible.
LOWMETRIC = "lowmetric.stub@0.1"
LOWMETRIC_DECL = {"direction": "lower_is_better", "score_range": (0.0, 100.0), "estimators": ("mean", "ratio_of_sums")}


def errors_of(proc):
    return {(e["rule"], e["field"]) for e in parse_errors(proc.stderr)}


def load_spec_problems(root: Path, spec: str = "spec.yaml"):
    """In-process validation: [] when load_spec accepts, else the (rule, field) set."""
    from gatedtrain.errors import Refused
    from gatedtrain.spec import load_spec

    try:
        load_spec(root / spec)
    except Refused as exc:
        return {(p.rule, p.field) for p in exc.problems}
    return set()


def patched_evaluators(**extra):
    from gatedtrain import items as items_mod

    return mock.patch.dict(items_mod.EVALUATORS, {LOWMETRIC: dict(LOWMETRIC_DECL, **extra)})


# ---------------------------------------------------------------------------
# R11A-23: the deciding evaluator at registration (HIGH-1)
# ---------------------------------------------------------------------------

class TestDecidingEvaluatorRegistration(Case):
    """Interface: `registration.deciding_evaluator(reg, evaluator_id) -> dict | None` (the entry whose id equals
    the spec gate's `evaluator`); validation reads direction, estimator, score range and scorer from it."""

    def decoy_first(self, d, **clause_over):
        d["evaluators"] = [{"id": DECOY, "manifest_hash": DUMMY_HASH}, {"id": STUB, "manifest_hash": OTHER_HASH}]
        d["clause"].update(clause_over)

    def test_helper(self):
        from gatedtrain.registration import deciding_evaluator

        reg = {"evaluators": [{"id": DECOY, "manifest_hash": DUMMY_HASH}, {"id": STUB, "manifest_hash": OTHER_HASH}]}
        self.assertEqual(deciding_evaluator(reg, STUB), reg["evaluators"][1])
        self.assertIsNone(deciding_evaluator(reg, "absent.metric@0.1"))

    def test_gate_evaluator_not_registered(self):
        """Regression (interim rule G-4): gate.evaluator must be one of the registration evaluator ids."""
        root = self.tiera(reg=lambda d: d["evaluators"].__setitem__(0, {"id": DECOY, "manifest_hash": DUMMY_HASH}))
        proc = self.validate(root)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn(("REG-FIELD", "registration.evaluators"), errors_of(proc), describe(proc))

    def test_decoy_first_consistent_validates(self):
        self.assert_ok(self.tiera(reg=lambda d: self.decoy_first(d)))

    def test_decoy_first_direction_checked_against_deciding(self):
        """Deciding evaluator at index 1 declares higher_is_better: lower_is_better is refused even though the
        unknown decoy at index 0 declares nothing (the review's bypass)."""
        root = self.tiera(reg=lambda d: self.decoy_first(d, direction="lower_is_better"))
        self.assert_refused(root, "REG-FIELD", "registration.clause.direction")

    def test_decoy_first_estimator_checked_against_deciding(self):
        root = self.tiera(reg=lambda d: self.decoy_first(d, estimator="ratio_of_sums"))
        self.assert_refused(root, "REG-FIELD", "registration.clause.estimator")

    def test_unknown_deciding_evaluator_refused(self):
        """Choice: the prototype has only the built-in table, so a deciding evaluator it does not declare (its
        direction cannot be checked) is refused: REG-FIELD registration.evaluators[<i>].id."""
        def reg(d):
            d["evaluators"] = [{"id": STUB, "manifest_hash": DUMMY_HASH}, {"id": DECOY, "manifest_hash": OTHER_HASH}]

        def spec(s):
            s["stages"][1]["evaluator"] = DECOY

        proc = self.validate(self.tiera(reg=reg, spec=spec))
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn(("REG-FIELD", "registration.evaluators[1].id"), errors_of(proc), describe(proc))
        self.assertNotIn("Traceback", proc.stderr)

    def _patched_root(self, **clause_over):
        def reg(d):
            d["evaluators"] = [{"id": STUB, "manifest_hash": DUMMY_HASH}, {"id": LOWMETRIC, "manifest_hash": OTHER_HASH}]
            d["clause"].update(clause_over)

        def spec(s):
            s["stages"][1]["evaluator"] = LOWMETRIC

        return self.tiera(reg=reg, spec=spec)

    def test_declaration_comes_from_deciding_not_index0(self):
        """In-process with a patched table: index 0 (the stub) declares higher_is_better and mean only; the
        deciding LOWMETRIC declares lower_is_better and ratio_of_sums. The deciding declaration rules."""
        with patched_evaluators():
            self.assertEqual(load_spec_problems(self._patched_root(direction="lower_is_better",
                                                                   estimator="ratio_of_sums")), set())
            self.assertEqual(load_spec_problems(self._patched_root(direction="higher_is_better")),
                             {("REG-FIELD", "registration.clause.direction")})

    def test_scorer_of_deciding_evaluator_checked(self):
        """G-21 on the deciding evaluator: Tier B evaluator at index 1 with a scorer off the allowlist."""
        def reg(d):
            real = d["evaluators"][0]
            real["scorer"] = "bogus_scorer"
            d["evaluators"] = [{"id": DECOY, "manifest_hash": DUMMY_HASH}, real]

        root = self.tierb(reg=reg)
        proc = self.validate(root)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn(("REG-SCORER", "registration.evaluators[1].scorer"), errors_of(proc), describe(proc))

    def test_registered_scorer_reads_deciding(self):
        """Interface: `tierb.registered_scorer(reg, evaluator_id)`."""
        from gatedtrain.tierb import registered_scorer

        reg = {"evaluators": [{"id": DECOY, "manifest_hash": DUMMY_HASH},
                              {"id": TIERB_EVAL, "manifest_hash": OTHER_HASH, "scorer": "numeric_match"}]}
        self.assertEqual(registered_scorer(reg, TIERB_EVAL), "numeric_match")


# ---------------------------------------------------------------------------
# R11A-25: every evaluator manifest hash resolved (MEDIUM-2)
# ---------------------------------------------------------------------------

class TestEveryEvaluatorHashResolved(Case):

    def test_second_evaluator_hash(self):
        for label, entry in (("null", {"id": DECOY, "manifest_hash": None}), ("absent", {"id": DECOY}),
                             ("tag", {"id": DECOY, "manifest_hash": "latest"}),
                             ("short", {"id": DECOY, "manifest_hash": "sha256:" + "a" * 63}),
                             ("upper", {"id": DECOY, "manifest_hash": "sha256:" + "A" * 64})):
            with self.subTest(label):
                root = self.tiera(reg=lambda d: d["evaluators"].append(dict(entry)))
                self.assert_refused(root, "REG-FIELD", "registration.evaluators[1].manifest_hash")


# ---------------------------------------------------------------------------
# R11A-24: the deciding evaluator at run time (HIGH-1)
# ---------------------------------------------------------------------------

N_ITEMS = 40


def _write_arm(d: Path, lines: list[str]) -> None:
    d.mkdir(parents=True)
    d.joinpath("results.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def nemo_lines(seeds, p, *, n=N_ITEMS):
    """NeMo Evaluator-shaped records, deterministic rewards with success share about p per item."""
    return [json.dumps({"problem_idx": i, "repeat": r, "reward": float((i * 7 + r * 3) % 10 < p * 10),
                        "metadata": {"eval_seed": s}})
            for i in range(n) for r, s in enumerate(seeds)]


def gate_run(tmp: Path, reg: dict, *, base_lines=None, cand_lines=None, evaluator_hash=None, evaluator=STUB):
    """`runner.gate_and_report` on a hand-built run directory (the production gate path, no training)."""
    from gatedtrain.gates import plan
    from gatedtrain.registration import normalize_clause
    from gatedtrain.runner import gate_and_report, stub_evaluator_hash

    reg = json.loads(json.dumps(reg))
    reg["bootstrap"]["replicates"] = 1000
    seeds = reg["clause"]["eval_seeds"]
    run_dir = Path(tempfile.mkdtemp(dir=tmp, prefix="run-"))
    (run_dir / "manifests").mkdir()
    _write_arm(run_dir / "eval" / "baseline", base_lines if base_lines is not None else nemo_lines(seeds, 0.3))
    _write_arm(run_dir / "eval" / "candidate", cand_lines if cand_lines is not None else nemo_lines(seeds, 0.7))
    c = normalize_clause(reg["clause"])
    wf = types.SimpleNamespace(clause=c, registration=reg, gate={"id": "quality_gate", "evaluator": evaluator},
                               spec={"workflow": "w", "artifacts": {}}, registration_sha256="sha256:" + "0" * 64)
    report, _ = gate_and_report(
        wf, run_dir, the_plan=plan(c, reg["sigma_pilot"], n_available=N_ITEMS),
        hashes={"baseline_checkpoint": "sha256:" + "1" * 64, "trained_checkpoint": "sha256:" + "2" * 64},
        lock_bytes=b"{}", item_set_hash="sha256:" + "3" * 64,
        evaluator_hash=evaluator_hash or stub_evaluator_hash(), stubs=[], items=list(range(N_ITEMS)))
    return report


def demo_registration() -> dict:
    from _support import DEMO_DIR

    return yaml.safe_load((DEMO_DIR / "registration.yaml").read_text(encoding="utf-8"))


class GateCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-r11fix-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)


class TestDecidingEvaluatorRunTime(GateCase):

    def test_control_passes(self):
        rep = gate_run(self.tmp, demo_registration())
        self.assertEqual(rep["status"], "PASS", rep["status_reasons"])

    def test_hash_compared_for_deciding_evaluator(self):
        """Decoy at index 0 carries the hash the run resolves; the deciding stub at index 1 does not: MISSING."""
        from gatedtrain.runner import stub_evaluator_hash

        reg = demo_registration()
        reg["evaluators"] = [{"id": DECOY, "manifest_hash": stub_evaluator_hash()},
                             {"id": STUB, "manifest_hash": OTHER_HASH}]
        rep = gate_run(self.tmp, reg)
        self.assertEqual(rep["status"], "MISSING", rep)
        self.assertIn("evaluator_hash_mismatch", rep["status_reasons"])
        self.assertEqual(rep["evaluator_hash_registered"], OTHER_HASH)

    def test_deciding_hash_matches_at_index1(self):
        from gatedtrain.runner import stub_evaluator_hash

        reg = demo_registration()
        reg["evaluators"] = [{"id": DECOY, "manifest_hash": OTHER_HASH},
                             {"id": STUB, "manifest_hash": stub_evaluator_hash()}]
        rep = gate_run(self.tmp, reg)
        self.assertNotIn("evaluator_hash_mismatch", rep["status_reasons"])
        self.assertEqual(rep["status"], "PASS", rep["status_reasons"])
        self.assertEqual(rep["evaluator_hash_registered"], stub_evaluator_hash())


# ---------------------------------------------------------------------------
# R11A-23 end to end: the review's repro (decoy at index 0, real evaluator at index 1 with a null hash,
# direction lower_is_better, bad-train negative control) is refused and writes nothing.
# ---------------------------------------------------------------------------

def evil_demo(dst: Path, src: Path) -> Path:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("runs*", "__pycache__", ".DS_Store"))
    reg = yaml.safe_load((dst / "registration.yaml").read_text(encoding="utf-8"))
    reg["workflow"] = "skeleton-addition-demo-bad-train"
    reg["clause"]["direction"] = "lower_is_better"
    real = reg["evaluators"][0]
    reg["evaluators"] = [{"id": DECOY, "manifest_hash": real["manifest_hash"]}, {"id": STUB, "manifest_hash": None}]
    evil = dst / "registration-evil.yaml"
    evil.write_text(yaml.safe_dump(reg, sort_keys=False), encoding="utf-8")
    spec = dst / "spec-bad-train.yaml"
    text, n = re.subn(r"file:registration\.yaml@sha256:[0-9a-f]{64}",
                      f"file:registration-evil.yaml@{file_sha256(evil)}", spec.read_text(encoding="utf-8"))
    assert n == 1
    (dst / "spec-evil.yaml").write_text(text, encoding="utf-8")
    return dst


EVIL_ERRORS = {("REG-FIELD", "registration.clause.direction"), ("REG-FIELD", "registration.evaluators[1].manifest_hash")}


class TestReviewReproValidate(GateCase):

    def test_validate_refuses(self):
        from _support import DEMO_DIR

        root = evil_demo(self.tmp / "demo", DEMO_DIR)
        proc = run_cli(["validate", "spec-evil.yaml"], cwd=root, timeout=60)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue(EVIL_ERRORS <= errors_of(proc), describe(proc))


@slow
class TestReviewReproRun(GateCase):

    def test_run_refuses_and_writes_nothing(self):
        WORKSPACE.ensure_baseline()
        root = evil_demo(self.tmp / "demo", WORKSPACE.demo)
        runs = self.tmp / "runs"
        proc = run_cli(["run", "spec-evil.yaml", "--runs-root", str(runs)], cwd=root)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue(EVIL_ERRORS <= errors_of(proc), describe(proc))
        self.assertNotIn("run_dir:", proc.stdout)
        self.assertFalse(runs.exists() and any(runs.iterdir()), f"run written under {runs}")


# ---------------------------------------------------------------------------
# R11A-26: non-finite ratio parts are out_of_range (MEDIUM-3)
# ---------------------------------------------------------------------------

class TestRatioOutOfRange(unittest.TestCase):

    BAD = (("num_nan", "numerator", float("nan")), ("num_inf", "numerator", float("inf")),
           ("num_neginf", "numerator", float("-inf")), ("den_nan", "denominator", float("nan")),
           ("den_inf", "denominator", float("inf")), ("den_zero", "denominator", 0.0),
           ("den_negative", "denominator", -5.0))

    def test_missing_item_within_bound(self):
        ref = ratio_reference(keep=[i for i in range(60) if i != 3])
        for label, field, value in self.BAD:
            with self.subTest(label):
                items, b, c, _ = ratio_arms()
                c["records"][3][field] = value
                res = evaluate_items(ratio_clause(), b, c, items=items, seed=5, n_boot=1000, max_unpaired=1)
                self.assertEqual(res["missing_reasons"]["candidate"], {"out_of_range": 1}, res)
                self.assertEqual(res["n_items"], 59)
                self.assertNotIn("malformed_record", res["status_reasons"])
                self.assertAlmostEqual(res["delta_hat"], ref["delta"], places=12)
                for k in ("delta_hat", "ci_lower", "ci_upper", "sd_diff", "achieved_power"):
                    self.assertTrue(math.isfinite(res[k]), (k, res[k]))

    def test_missing_above_bound_never_pass_or_fail(self):
        for label, field, value in self.BAD:
            with self.subTest(label):
                items, b, c, _ = ratio_arms()
                c["records"][3][field] = value
                res = evaluate_items(ratio_clause(), b, c, items=items, seed=5, n_boot=500)
                self.assertEqual(res["status"], "MISSING", res)
                self.assertIn("missing_above_bound", res["status_reasons"])


# ---------------------------------------------------------------------------
# R11A-27: duplicate item IDs (LOW-4)
# ---------------------------------------------------------------------------

class TestDuplicateItemIds(unittest.TestCase):

    def test_duplicate_items_raise(self):
        """Choice: a programming error of the caller (the registered item set is a set): ValueError."""
        items, b, c = matrix_arms([[0.2]] * 20, [[0.6]] * 20, [11])
        with self.assertRaises(ValueError):
            evaluate_items(clause(), b, c, items=items + items[:1], seed=1, n_boot=200)


# ---------------------------------------------------------------------------
# R11A-28: the production reader keeps malformed and duplicate lines (LOW-5)
# ---------------------------------------------------------------------------

class TestProductionReader(GateCase):
    """`gates.load_paired_records` stays NeMo Evaluator's reader (test_pairing). The gate path
    (`runner.gate_and_report`, used by `run` and `tierb-collect`) must not use its skip and last-wins rules."""

    def seeds(self):
        return demo_registration()["clause"]["eval_seeds"]

    def check(self, cand_lines):
        rep = gate_run(self.tmp, demo_registration(), cand_lines=cand_lines)
        self.assertEqual(rep["status"], "MISSING", rep["status_reasons"])
        self.assertIn("malformed_record", rep["status_reasons"])

    def test_blank_lines_are_fine(self):
        lines = nemo_lines(self.seeds(), 0.7)
        rep = gate_run(self.tmp, demo_registration(), cand_lines=lines[:5] + ["", "   "] + lines[5:])
        self.assertEqual(rep["status"], "PASS", rep["status_reasons"])

    def test_unparsable_line(self):
        self.check(nemo_lines(self.seeds(), 0.7) + ["{not json"])

    def test_non_object_line(self):
        self.check(nemo_lines(self.seeds(), 0.7) + ["[1, 2]"])

    def test_line_without_problem_idx(self):
        self.check(nemo_lines(self.seeds(), 0.7) + [json.dumps({"repeat": 0, "reward": 1.0,
                                                                  "metadata": {"eval_seed": self.seeds()[0]}})])

    def test_duplicate_item_repeat(self):
        lines = nemo_lines(self.seeds(), 0.7)
        dup = json.loads(lines[0])
        dup["reward"] = 1.0 - dup["reward"]
        self.check(lines + [json.dumps(dup)])


# ---------------------------------------------------------------------------
# R11A-29: pilot baseline mean at the floor or ceiling; created_at with amends (LOW-6)
# ---------------------------------------------------------------------------

class TestBaselinePilotMean(Case):
    """R13-9, 05 section 6.1: top-level `baseline_pilot_mean` (beside `sigma_pilot`). Choice: optional (absent =
    not checked, so legacy registrations stay valid); when given it must be a number inside the deciding
    evaluator's declared range and not equal to a closed bound."""

    FIELD = "registration.baseline_pilot_mean"

    def test_interior_ok(self):
        self.assert_ok(self.tiera(reg=lambda d: d.__setitem__("baseline_pilot_mean", 0.42)))

    def test_floor_ceiling_outside_and_type(self):
        for value in (0.0, 1.0, 0, 1, -0.1, 1.5, "0.5", True, None, float("nan")):
            with self.subTest(value=value):
                root = self.tiera(reg=lambda d: d.__setitem__("baseline_pilot_mean", value))
                e = self.assert_refused(root, "REG-FIELD", self.FIELD)
                self.assertNotIn("unknown", e["message"], "a known field, refused for its value")

    def test_range_of_deciding_evaluator(self):
        """Index 0 is the stub (range [0, 1]); the deciding LOWMETRIC declares [0, 100]: 1.0 is interior, 100.0
        is its ceiling. With an open upper bound only the floor is refused."""
        def root(value, **_):
            def reg(d):
                d["evaluators"] = [{"id": STUB, "manifest_hash": DUMMY_HASH},
                                   {"id": LOWMETRIC, "manifest_hash": OTHER_HASH}]
                d["clause"]["direction"] = "lower_is_better"
                d["baseline_pilot_mean"] = value

            return self.tiera(reg=reg, spec=lambda s: s["stages"][1].__setitem__("evaluator", LOWMETRIC))

        with patched_evaluators():
            self.assertEqual(load_spec_problems(root(1.0)), set())
            self.assertEqual(load_spec_problems(root(100.0)), {("REG-FIELD", self.FIELD)})
        with patched_evaluators(score_range=(0.0, None)):
            self.assertEqual(load_spec_problems(root(1000.0)), set())
            self.assertEqual(load_spec_problems(root(0.0)), {("REG-FIELD", self.FIELD)})


class TestCreatedAtWithAmends(Case):

    AMENDS = "sha256:" + "9a" * 32

    def test_amends_without_created_at(self):
        root = self.tiera(reg=lambda d: (d.pop("created_at", None), d.__setitem__("amends", self.AMENDS)))
        self.assert_refused(root, "REG-FIELD", "registration.created_at")

    def test_amends_with_created_at(self):
        self.assert_ok(self.tiera(reg=lambda d: d.update(amends=self.AMENDS, created_at="2026-11-11T09:00:00Z")))

    def test_no_amends_created_at_optional(self):
        self.assert_ok(self.tiera(reg=lambda d: d.pop("created_at", None)))


# ---------------------------------------------------------------------------
# R11A-30: boundaries and R13-8 pinned (LOW-7)
# ---------------------------------------------------------------------------

class TestBoundaries(unittest.TestCase):

    def test_no_seed_within_bound(self):
        """R13-6: a seedless record is a missing item (`no_seed`), never `seed_mismatch`."""
        items, b, c = matrix_arms([[0.2]] * 20, [[0.2 + (0.4 if i % 2 else 0.2)] for i in range(20)], [11])
        b["records"][4]["seed"] = None
        res = evaluate_items(clause(), b, c, items=items, seed=1, n_boot=500, max_unpaired=1)
        self.assertEqual(res["missing_reasons"]["baseline"], {"no_seed": 1})
        self.assertNotIn("seed_mismatch", res["status_reasons"])
        self.assertNotEqual(res["status"], "MISSING", res["status_reasons"])
        self.assertEqual(res["n_items"], 19)

    def _fraction(self, n_missing):
        n = 16
        base = [[0.2] for _ in range(n)]
        cand = [[0.2 + (0.4 if i % 2 else 0.2)] if i >= n_missing else [None] for i in range(n)]
        items, b, c = matrix_arms(base, cand, [11])
        return evaluate_items(clause(), b, c, items=items, seed=1, n_boot=300, max_unpaired=0.125)

    def test_fraction_bound_equal_is_allowed(self):
        """05 section 3.4: MISSING when the missing items "exceed" the bound; 0.125 x 2 x 16 = 4 exactly."""
        self.assertNotIn("missing_above_bound", self._fraction(4)["status_reasons"])
        self.assertIn("missing_above_bound", self._fraction(5)["status_reasons"])

    def test_count_bound_equal_is_allowed(self):
        items, b, c = matrix_arms([[0.2]] * 16, [[0.4]] * 14 + [[None]] * 2, [11])
        res = evaluate_items(clause(), b, c, items=items, seed=1, n_boot=300, max_unpaired=2)
        self.assertNotIn("missing_above_bound", res["status_reasons"])

    def test_post_hoc_equal_timestamp_not_observed(self):
        from gatedtrain.gates import is_post_hoc

        judged = "sha256:" + "1" * 64
        obs = [{"checkpoint": judged, "arm": "candidate", "created_at": "2026-11-11T09:00:00Z"}]
        self.assertIs(is_post_hoc(amends="sha256:" + "9a" * 32, amended_at="2026-11-11T09:00:00Z",
                                  judged_checkpoint=judged, observed=obs), False)
        self.assertIs(is_post_hoc(amends="sha256:" + "9a" * 32, amended_at="2026-11-11T09:00:00.000001Z",
                                  judged_checkpoint=judged, observed=obs), True)


class TestRatioPinned(unittest.TestCase):
    """R13-8 with unequal repeat counts and unequal denominators across arms. Expected values come from an
    independent BCa of the ratio of sums, written out in `setUpClass`: items resampled with
    `numpy.random.default_rng(seed).integers(0, n, size=(n_boot, n))` over complete pairs in registered item
    order (the same index stream as the mean path), z0 = Phi^-1 of the mid-rank share below theta, a from the
    item jackknife, num_i and den_i summed over the repeats usable in both arms, d_i per arm with that arm's
    mean(den)."""

    N, K, B, SEED = 60, 3, 4000, 99
    SEEDS = [5, 6, 7]
    DROP_B = {(3, 2), (7, 0), (7, 1), (10, 1), (15, 0)}
    DROP_C = {(3, 1), (12, 2), (20, 0), (15, 1), (15, 2)}

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        words_b = rng.integers(3, 40, size=(cls.N, cls.K)).astype(float)
        words_c = words_b + rng.integers(0, 15, size=(cls.N, cls.K))          # unequal denominators
        eb = rng.binomial(words_b.astype(int), 0.25).astype(float)
        ec = rng.binomial(words_c.astype(int), 0.20).astype(float)
        cls.items = [f"utt{i}" for i in range(cls.N)]

        def build(e, w, drop):
            out = []
            for i in range(cls.N):
                for r in range(cls.K):
                    if (i, r) not in drop:
                        out.append(rec(cls.items[i], r, float(e[i, r] / w[i, r]), cls.SEEDS[r],
                                       numerator=float(e[i, r]), denominator=float(w[i, r])))
            return arm(out[::-1])

        cls.b, cls.c = build(eb, words_b, cls.DROP_B), build(ec, words_c, cls.DROP_C)
        cls.clause = clause("non_inferiority", metric="wer", direction="lower_is_better", estimator="ratio_of_sums",
                            margin=0.01, planning_effect=0.0, samples_per_item=cls.K, eval_seeds=cls.SEEDS)
        cls.res = evaluate_items(cls.clause, cls.b, cls.c, items=cls.items, seed=cls.SEED, n_boot=cls.B,
                                 max_unpaired=20)
        # Independent reference.
        use = {i: [r for r in range(cls.K) if (i, r) not in cls.DROP_B and (i, r) not in cls.DROP_C]
               for i in range(cls.N)}
        pi = [i for i in range(cls.N) if use[i]]
        nb = np.array([eb[i, use[i]].sum() for i in pi]); db = np.array([words_b[i, use[i]].sum() for i in pi])
        nc = np.array([ec[i, use[i]].sum() for i in pi]); dc = np.array([words_c[i, use[i]].sum() for i in pi])
        n = len(pi)

        def T(nb, db, nc, dc):                     # oriented: lower is better -> baseline - candidate
            return nb.sum(-1) / db.sum(-1) - nc.sum(-1) / dc.sum(-1)

        th = T(nb, db, nc, dc)
        idx = np.random.default_rng(cls.SEED).integers(0, n, size=(cls.B, n))
        bt = T(nb[idx], db[idx], nc[idx], dc[idx])
        z0 = norm.ppf((np.sum(bt < th) + 0.5 * np.sum(bt == th)) / cls.B)
        jk = np.array([T(np.delete(nb, j), np.delete(db, j), np.delete(nc, j), np.delete(dc, j)) for j in range(n)])
        u = jk.mean() - jk
        a = (u ** 3).sum() / (6 * ((u ** 2).sum()) ** 1.5)

        def q(p):
            z = norm.ppf(p)
            return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

        rb, rc = nb.sum() / db.sum(), nc.sum() / dc.sum()
        d = (nb - rb * db) / db.mean() - (nc - rc * dc) / dc.mean()
        sd = d.std(ddof=1)
        cls.ref = {"n": n, "delta": th, "lo": float(np.quantile(bt, q(0.025))), "hi": float(np.quantile(bt, q(0.975))),
                   "sd": sd, "power": norm.cdf(0.01 * math.sqrt(n) / sd - norm.ppf(0.975)), "rb": rb, "rc": rc,
                   "z0": z0, "a": a}

    def test_fixture_guard(self):
        """The data exercise the mutants: unequal repeat counts, unequal arm denominators, non-zero z0 and a."""
        self.assertEqual(self.ref["n"], self.N - 1)                      # item 15: no common repeat
        self.assertGreater(abs(self.ref["z0"]), 0.01)
        self.assertGreater(abs(self.ref["a"]), 1e-4)

    def test_point_and_scores(self):
        self.assertEqual(self.res["n_items"], self.ref["n"])
        self.assertAlmostEqual(self.res["delta_hat"], self.ref["delta"], places=12)
        self.assertAlmostEqual(self.res["baseline_score"], self.ref["rb"], places=12)
        self.assertAlmostEqual(self.res["candidate_score"], self.ref["rc"], places=12)

    def test_interval(self):
        self.assertAlmostEqual(self.res["ci_lower"], self.ref["lo"], places=10)
        self.assertAlmostEqual(self.res["ci_upper"], self.ref["hi"], places=10)

    def test_sd_and_power(self):
        self.assertAlmostEqual(self.res["sd_diff"], self.ref["sd"], places=10)
        self.assertAlmostEqual(self.res["achieved_power"], self.ref["power"], places=10)


if __name__ == "__main__":
    unittest.main()
