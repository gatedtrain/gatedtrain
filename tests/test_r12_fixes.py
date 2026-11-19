"""Constraint-clause validation for item coverage, model identity, batches and egress approval.

See R12A-18 to R12A-24 in tests/README.md and MM-C1 to MM-C4 in MULTIMODAL-SPEC.md.
"""

from __future__ import annotations

import copy
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from _support import DEMO_DIR
from test_r12_constraints import (ALPHA, CS_ID, EGRESS, TRANSFER_HASH, NoNetwork, arm, constraint_clause,
                                  constraint_set, paired_arms, problems, qualification, registration, rows)

MODEL_C = "sha256:" + "c1" * 32
MODEL_B = "sha256:" + "b1" * 32
OTHER = "sha256:" + "ee" * 32


def items_of(n: int) -> list[str]:
    return [f"i{i:04d}" for i in range(n)]


def ev(clause, cs, q, candidate, baseline=None, **kw):
    from gatedtrain.constraints import evaluate_constraint

    return evaluate_constraint(clause, cs, q, candidate, baseline, **kw)


# ---------------------------------------------------------------------------
# R12A-18: the expected set is the registered item set
# ---------------------------------------------------------------------------

def dropped_arm(n=420, violating=range(10)):
    """The review's repro: the violating items' records and batch entries are deleted."""
    a = arm(rows(n, violating=violating))
    gone = {f"i{i:04d}" for i in violating}
    a["records"] = [r for r in a["records"] if r["item"] not in gone]
    for b in a["batches"]:
        b["records"] = [x for x in b["records"] if x[0] not in gone]
    return a


class TestItemsRequired(unittest.TestCase):

    def test_items_is_required(self):
        with self.assertRaises(TypeError):
            ev(constraint_clause(), constraint_set(), qualification(), arm(rows(368)))

    def test_items_is_keyword_only(self):
        from gatedtrain.constraints import evaluate_constraint

        with self.assertRaises(TypeError):
            evaluate_constraint(constraint_clause(), constraint_set(), qualification(), arm(rows(368)), None,
                                items_of(368))

    def test_deleted_violating_items_are_missing(self):
        res = ev(constraint_clause(), constraint_set(), qualification(), dropped_arm(), items=items_of(420))
        self.assertEqual(res["status"], "MISSING", res["status_reasons"])
        self.assertIn("missing_above_bound", res["status_reasons"])
        self.assertEqual(res["missing_reasons"]["candidate"], {"absent": 10})

    def test_deleted_items_never_pass(self):
        for mu in (0, 5, 0.01):
            with self.subTest(max_unpaired=mu):
                res = ev(constraint_clause(), constraint_set(), qualification(), dropped_arm(), items=items_of(420),
                         max_unpaired=mu)
                self.assertNotEqual(res["status"], "PASS", res["status_reasons"])

    def test_noninferiority_expected_set(self):
        cand, base = paired_arms(400)
        base["records"] = [r for r in base["records"] if r["item"] != "i0007"]
        for b in base["batches"]:
            b["records"] = [x for x in b["records"] if x[0] != "i0007"]
        res = ev(constraint_clause("non_inferiority"), constraint_set(), qualification(), cand, base,
                 items=items_of(400))
        self.assertEqual(res["status"], "MISSING")
        self.assertEqual(res["missing_reasons"]["baseline"], {"absent": 1})


# ---------------------------------------------------------------------------
# R12A-19: evaluated model identity
# ---------------------------------------------------------------------------

class TestModelIdentity(unittest.TestCase):
    """05 sections 3.3, 3.8 (R11-14): as `gates.evaluate_items(expected_models=...)`: with `expected_models` given,
    an arm whose `model_manifest_hash` differs or is null is MISSING `model_hash_mismatch`; None means no check."""

    def thr(self, got, expected):
        a = arm(rows(368))
        if got is not ...:
            a["model_manifest_hash"] = got
        return ev(constraint_clause(), constraint_set(), qualification(), a, items=items_of(368),
                  expected_models=expected)

    def test_match(self):
        res = self.thr(MODEL_C, {"candidate": MODEL_C})
        self.assertEqual(res["status"], "PASS", res["status_reasons"])

    def test_mismatch_null_absent(self):
        for got in (OTHER, None, ...):
            with self.subTest(got=got):
                res = self.thr(got, {"candidate": MODEL_C})
                self.assertEqual(res["status"], "MISSING")
                self.assertIn("model_hash_mismatch", res["status_reasons"])

    def test_no_expected_models_no_check(self):
        self.assertEqual(self.thr(OTHER, None)["status"], "PASS")

    def test_noninferiority_baseline_checked(self):
        cand, base = paired_arms(500)
        cand["model_manifest_hash"], base["model_manifest_hash"] = MODEL_C, OTHER
        clause = constraint_clause("non_inferiority")
        res = ev(clause, constraint_set(), qualification(), cand, base, items=items_of(500),
                 expected_models={"candidate": MODEL_C, "baseline": MODEL_B})
        self.assertEqual(res["status"], "MISSING")
        self.assertIn("model_hash_mismatch", res["status_reasons"])
        base["model_manifest_hash"] = MODEL_B
        res = ev(clause, constraint_set(), qualification(), cand, base, items=items_of(500),
                 expected_models={"candidate": MODEL_C, "baseline": MODEL_B})
        self.assertNotIn("model_hash_mismatch", res["status_reasons"])


# ---------------------------------------------------------------------------
# R12A-20: constraint clauses stay out of the quality path
# ---------------------------------------------------------------------------

class TestQualityPathRefusesConstraintClause(unittest.TestCase):

    def test_evaluate_items_raises(self):
        from gatedtrain.gates import evaluate_items

        cand, base = paired_arms(400, cand_only=range(30))
        with self.assertRaises(ValueError):
            evaluate_items(constraint_clause("non_inferiority"), base, cand, items=items_of(400), seed=0, n_boot=200,
                           score_range=(0, 1))


class TestCommandsRefuseConstraintClause(unittest.TestCase):
    """The REG-CONSTRAINT branches of `spec.load_spec` (`gatedtrain run` / `validate`) and
    `tierb.load_registration` (Tier B commands): constraint clauses are library-level only (section 11.17
    scope). Kills review mutants M31 and M32."""

    FIELD = ("REG-CONSTRAINT", "registration.clause.constraint_set")

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-r12fix-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = self.tmp / "demo"
        shutil.copytree(DEMO_DIR, self.root, ignore=shutil.ignore_patterns("tierb", "baseline", "runs*", "__pycache__"))
        (self.root / "baseline").mkdir()
        reg = registration("non_inferiority")
        reg["gate"] = "quality_gate"
        data = yaml.safe_dump(reg, sort_keys=False).encode()
        (self.root / "registration.yaml").write_bytes(data)
        spec = yaml.safe_load((self.root / "spec.yaml").read_text(encoding="utf-8"))
        uri = spec["artifacts"]["registration"]["uri"]
        spec["artifacts"]["registration"]["uri"] = uri.split("@")[0] + "@sha256:" + hashlib.sha256(data).hexdigest()
        gate = spec["stages"][1]
        gate["uses"] = "gate.noninferiority@1.0"
        gate["evaluator"] = f"constraints.{CS_ID}@0.1"
        gate["test"] = {"alpha": ALPHA}
        (self.root / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    def test_load_spec_refuses(self):
        from gatedtrain.errors import Refused
        from gatedtrain.spec import load_spec

        with self.assertRaises(Refused) as cm:
            load_spec(self.root / "spec.yaml")
        self.assertIn(self.FIELD, {(p.rule, p.field) for p in cm.exception.problems})

    def test_tierb_load_registration_refuses(self):
        from gatedtrain.errors import Refused
        from gatedtrain.tierb import load_registration

        with self.assertRaises(Refused) as cm:
            load_registration(self.root / "registration.yaml")
        self.assertIn(self.FIELD, {(p.rule, p.field) for p in cm.exception.problems})


# ---------------------------------------------------------------------------
# R12A-21: canary and batch checks
# ---------------------------------------------------------------------------

def two_judge_set() -> dict:
    """j1 checks c2, j2 checks c3; canary k1 violates both."""
    cs = constraint_set()
    cs["judges"]["j2"] = copy.deepcopy(cs["judges"]["j1"])
    cs["constraints"].append({"id": "c3", "text": "The image must not depict a weapon.",
                              "checker": {"kind": "judge", "judge": "j2",
                                          "no_tool_reason": "no programmatic detector covers weapons"}})
    cs["canaries"][0]["violates"] = ["c2", "c3"]
    return cs


def two_judge_q() -> dict:
    q = qualification()
    q["judges"]["j2"] = copy.deepcopy(q["judges"]["j1"])
    return q


def both_judges_arm(n=368, *, j2=True):
    cs = two_judge_set()
    a = arm(rows(n), cs=cs)
    if j2:
        extra = copy.deepcopy(a["batches"])
        for b in extra:
            b["judge"], b["batch"] = "j2", "j2-" + b["batch"]
        a["batches"] += extra
    return cs, a


class TestCanaryCoverage(unittest.TestCase):

    def run_(self, a, cs=None, q=None, n=368):
        return ev(constraint_clause(), cs or constraint_set(), q or qualification(), a, items=items_of(n))

    def test_two_blocking_judges_both_batched_pass(self):
        cs, a = both_judges_arm()
        res = self.run_(a, cs, two_judge_q())
        self.assertEqual(res["status"], "PASS", res["status_reasons"])

    def test_blocking_judge_without_batches_missing(self):
        """Every blocking judge's records are covered by its own batches holding canaries."""
        cs, a = both_judges_arm(j2=False)
        res = self.run_(a, cs, two_judge_q())
        self.assertEqual(res["status"], "MISSING")
        self.assertIn("canary_absent", res["status_reasons"])

    def test_batch_of_unregistered_judge(self):
        res = self.run_(arm(rows(368), judge="ghost"))
        self.assertEqual(res["status"], "MISSING", res["status_reasons"])
        self.assertIn("malformed_batch", res["status_reasons"])

    def test_record_in_two_batches(self):
        a = arm(rows(368))
        a["batches"][1]["records"].append(list(a["batches"][0]["records"][0]))
        res = self.run_(a)
        self.assertEqual(res["status"], "MISSING", res["status_reasons"])
        self.assertIn("malformed_batch", res["status_reasons"])

    def test_flagged_key_absent_is_not_flagged(self):
        """Pins review mutant M13: a canary without `flagged` was not flagged."""
        a = arm(rows(368))
        del a["batches"][2]["canaries"][0]["flagged"]
        res = self.run_(a)
        self.assertEqual(res["status"], "MISSING")
        self.assertIn("canary_missed", res["status_reasons"])
        self.assertEqual(res["canaries"]["missed"], [{"batch": "b002", "canary": "k1"}])

    def test_flagged_must_be_true(self):
        for value in ("yes", 1, None):
            with self.subTest(flagged=value):
                a = arm(rows(368))
                a["batches"][2]["canaries"][0]["flagged"] = value
                self.assertIn("canary_missed", self.run_(a)["status_reasons"])


class TestCanaryRegistration(unittest.TestCase):

    def test_canary_must_violate_a_judge_checked_constraint(self):
        cs = constraint_set()
        cs["canaries"][0]["violates"] = ["c1"]          # c1 is tool-checked: the judge is never tested
        self.assertIn(("REG-JUDGE", "registration.constraint_set.canaries[0]"), problems(registration(cs=cs)))

    def test_canary_on_judge_constraint_ok(self):
        self.assertEqual(problems(registration()), set())


class TestEgressDestination(unittest.TestCase):
    """Pins review mutant M17: an approval is for destination `upload` only (06 section 9, B20)."""

    APPROVAL = {**EGRESS, "destination": "download"}

    def test_registration_refused(self):
        reg = registration(cs=constraint_set(endpoint="external"),
                           q=qualification(judge_over={"egress_approval": self.APPROVAL}))
        self.assertIn(("REG-EGRESS", "registration.judge_qualification.judges.j1.egress_approval"), problems(reg))

    def test_run_time_missing(self):
        cs = constraint_set(endpoint="external")
        a = arm(rows(368), cs=cs)
        a["egress_approval_id"] = EGRESS["approval_id"]
        with NoNetwork():
            res = ev(constraint_clause(), cs, qualification(judge_over={"egress_approval": self.APPROVAL}), a,
                     items=items_of(368))
        self.assertEqual(res["status"], "MISSING")
        self.assertIn("egress_unapproved", res["status_reasons"])


# ---------------------------------------------------------------------------
# R12A-22: model version deny list and family normalization
# ---------------------------------------------------------------------------

class TestJudgeStrings(unittest.TestCase):

    MV = ("registration.constraint_set.judges.j1.model_version")
    FAM = ("registration.constraint_set.judges.j1.family")

    def test_floating_versions_refused(self):
        for v in ("", "   ", "latest", " Latest ", "CURRENT", "stable", "Default", "newest\n", "\tlatest"):
            with self.subTest(version=v):
                reg = registration(cs=constraint_set(judge_over={"model_version": v}))
                self.assertIn(("REG-JUDGE", self.MV), problems(reg))

    def test_pinned_versions_ok(self):
        for v in ("2025-01-28", "v1.2.3", "latest-2025-01-28"):
            with self.subTest(version=v):
                reg = registration(cs=constraint_set(judge_over={"model_version": v}))
                self.assertNotIn(("REG-JUDGE", self.MV), problems(reg))

    def test_family_normalized_against_policy(self):
        for policy in ("Qwen2.5-VL", " qwen2.5-vl ", "QWEN2.5-VL\t"):
            with self.subTest(policy=policy):
                self.assertIn(("REG-JUDGE", self.FAM), problems(registration(q=qualification(policy_family=policy))))

    def test_family_normalized_against_rewards(self):
        q = qualification(training_reward_families=["clip-vit-l", " Qwen2.5-VL "])
        self.assertIn(("REG-JUDGE", self.FAM), problems(registration(q=q)))

    def test_run_time_judge_not_qualified(self):
        cs = constraint_set(judge_over={"model_version": "Stable"})
        res = ev(constraint_clause(), cs, qualification(), arm(rows(368), cs=cs), items=items_of(368))
        self.assertEqual(res["status"], "MISSING")
        self.assertIn("judge_not_qualified", res["status_reasons"])


# ---------------------------------------------------------------------------
# R12A-23: report transfer-test reference; arm egress approval id
# ---------------------------------------------------------------------------

class TestReportAndEgressId(unittest.TestCase):

    def test_judge_report_has_transfer_test(self):
        from gatedtrain.report import constraint_report

        res = ev(constraint_clause(), constraint_set(), qualification(), arm(rows(368)), items=items_of(368))
        self.assertEqual(constraint_report(res)["judges"]["j1"]["transfer_test"], TRANSFER_HASH)

    def ext(self, arm_id, *, baseline_id=...):
        cs = constraint_set(endpoint="external")
        q = qualification(judge_over={"egress_approval": EGRESS})
        if baseline_id is ...:
            a = arm(rows(368), cs=cs)
            if arm_id is not ...:
                a["egress_approval_id"] = arm_id
            with NoNetwork():
                return ev(constraint_clause(), cs, q, a, items=items_of(368))
        cand, base = arm(rows(500), cs=cs), arm(rows(500), cs=cs)
        cand["egress_approval_id"], base["egress_approval_id"] = arm_id, baseline_id
        with NoNetwork():
            return ev(constraint_clause("non_inferiority"), cs, q, cand, base, items=items_of(500))

    def test_matching_id_pass(self):
        res = self.ext(EGRESS["approval_id"])
        self.assertEqual(res["status"], "PASS", res["status_reasons"])

    def test_other_absent_null_id_missing(self):
        for got in ("egress-approval-9999", None, ...):
            with self.subTest(got=got):
                res = self.ext(got)
                self.assertEqual(res["status"], "MISSING")
                self.assertIn("egress_unapproved", res["status_reasons"])

    def test_baseline_arm_id_checked(self):
        res = self.ext(EGRESS["approval_id"], baseline_id="egress-approval-9999")
        self.assertIn("egress_unapproved", res["status_reasons"])
        res = self.ext(EGRESS["approval_id"], baseline_id=EGRESS["approval_id"])
        self.assertNotIn("egress_unapproved", res["status_reasons"])

    def test_local_judge_needs_no_id(self):
        res = ev(constraint_clause(), constraint_set(), qualification(), arm(rows(368)), items=items_of(368))
        self.assertEqual(res["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
