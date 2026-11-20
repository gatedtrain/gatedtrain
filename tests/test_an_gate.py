"""AN-8, AN-9 and AN-10: the trust stages `evaluate_sealed`, `gate` and `report` (tests/README.md section 12)."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

import yaml

import test_an_intake as ti
from gatedtrain.ainative import scorers, trust_stages
from gatedtrain.ainative.machine import run_stage
from gatedtrain.ainative.project import Project, create_project
from gatedtrain.ainative.stage import Proposal, stage_registry
from gatedtrain.ainative.trust import UNSEALED_GUARANTEE, FixedAlphaLedger, InProcessTrust
from gatedtrain.ainative.trust_stages import (EvaluateSealed, Gate, Report, SCHEMA_MODEL_REF, register_generator,
                                               render_markdown)
from gatedtrain.errors import Refused

GENERATOR = "t7-generator"


def unique_rows(n: int) -> list[dict]:
    return [{"input": f"question-{i}-marker-{i * 7919}", "label": f"label-{i}-xq{i * 104729}"} for i in range(n)]


def _share(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % 100


def generate(model_ref: dict, item: dict, *, sampler: dict, seed: int):
    """Test-local generator: answers right for `rate` of the items (it is given the label only through the table)."""
    GENERATE_CALLS.append((dict(sampler), seed, sorted(item)))
    label = LABELS[item["input"]]
    return label if _share(item["input"]) < model_ref["rate"] * 100 else "wrong"


GENERATE_CALLS: list = []
LABELS: dict = {}


def model_stage(stage_name: str, rate: float, trains: bool) -> type:
    class _Model:
        name = stage_name
        version = "1"
        inputs: dict = {}
        outputs = {"model": SCHEMA_MODEL_REF}
        needs: frozenset = frozenset()
        params: frozenset = frozenset()

        def plan(self, ctx):
            return Proposal(self.name, "model", {}, {})

        def run(self, ctx, decision):
            rec = {"schema": SCHEMA_MODEL_REF, "schema_version": 1, "generator": GENERATOR, "rate": rate,
                   "label": stage_name}
            return {"model": ctx.project.write_artifact(self.name, "model", rec, SCHEMA_MODEL_REF)}

        def check(self, ctx, outputs):
            return []

    _Model.trains = trains
    return _Model


class GateBase(ti.Base):
    base_rate = 0.5
    cand_rate = 1.0

    def setUp(self) -> None:
        super().setUp()
        register_generator(GENERATOR, generate)
        self.addCleanup(trust_stages.GENERATORS.pop, GENERATOR, None)
        GENERATE_CALLS.clear()
        LABELS.clear()

    def make(self, data: bytes | None = None, *, name: str = "proj", n: int = 400) -> Project:
        rows = unique_rows(n)
        LABELS.update({r["input"]: r["label"] for r in rows})
        d = self.tmp / f"{name}.jsonl"
        d.write_bytes(ti.jsonl(rows))
        spec = {"name": "t", "training_stages": ["train_x"], "stages": [
            {"name": "intake_seal", "stage": "intake_seal"},
            {"name": "choose_x", "stage": "choose_x"},
            {"name": "preregister", "stage": "preregister",
             "inputs": {"sealed_summary": "intake_seal.sealed_summary", "deployment": "choose_x.deployment"}},
            {"name": "base_x", "stage": "base_x"},
            {"name": "train_x", "stage": "train_x"},
            {"name": "evaluate_sealed", "stage": "evaluate_sealed",
             "inputs": {"registration": "preregister.registration", "baseline_model": "base_x.model",
                        "candidate_model": "train_x.model"}},
            {"name": "gate", "stage": "gate",
             "inputs": {"registration": "preregister.registration",
                        "baseline_records": "evaluate_sealed.baseline_records",
                        "candidate_records": "evaluate_sealed.candidate_records"}},
            {"name": "report", "stage": "report",
             "inputs": {"registration": "preregister.registration", "power": "intake_seal.power",
                        "gate_result": "gate.gate_result"}}]}
        (self.tmp / f"{name}.yaml").write_text(yaml.safe_dump(spec))
        return create_project(self.tmp / name, goal="g", data=d, pipeline=str(self.tmp / f"{name}.yaml"),
                              budget={"gpu_hours": 1, "llm_usd": 1})

    def registry(self) -> dict:
        return {**stage_registry(), "train_x": model_stage("train_x", self.cand_rate, True),
                "base_x": model_stage("base_x", self.base_rate, False), "choose_x": ti.ChooseDeployment}

    def upto_registration(self, proj: Project) -> None:
        self.run_(proj, "intake_seal")
        self.run_(proj, "choose_x")
        self.run_(proj, "preregister", {"metric_confirmed": True})
        self.run_(proj, "base_x")
        self.run_(proj, "train_x")

    def through(self, proj: Project, last: str = "report") -> dict:
        self.upto_registration(proj)
        out = {}
        for ref in ("evaluate_sealed", "gate", "report"):
            out[ref] = self.run_(proj, ref).outputs
            if ref == last:
                break
        return out

    def reg_ref(self, proj: Project):
        return self.run_(proj, "preregister", {"metric_confirmed": True}).outputs["registration"]

    def outside_trust(self, proj: Project) -> bytes:
        blob = b""
        for path in sorted(proj.root.rglob("*")):
            if path.is_file() and "trust" not in path.relative_to(proj.root).parts:
                blob += path.read_bytes()
        return blob


class RegistryTest(unittest.TestCase):
    def test_real_classes_replace_the_last_three_placeholders(self) -> None:
        reg = stage_registry()
        for name, cls in (("evaluate_sealed", EvaluateSealed), ("gate", Gate), ("report", Report)):
            self.assertIs(reg[name], cls)
            self.assertEqual(cls.params, frozenset())
            self.assertFalse(cls.trains)
            self.assertEqual(cls.needs, frozenset())
            self.assertEqual(cls.name, name)


class EvaluateSealedTest(GateBase):
    def test_records_hold_item_id_seed_and_score_only_and_no_sealed_text(self) -> None:
        proj = self.make()
        self.upto_registration(proj)
        out = self.run_(proj, "evaluate_sealed").outputs
        summary = proj.read_artifact(self.run_(proj, "intake_seal").outputs["sealed_summary"])
        sealed = self.sealed_items(proj, summary["manifest_sha256"])
        self.assertEqual(len(sealed), 155)
        for name in ("baseline_records", "candidate_records"):
            rec = proj.read_artifact(out[name])
            self.assertEqual(len(rec["records"]), 155 * 2)               # two registered eval seeds
            for r in rec["records"]:
                self.assertEqual(set(r), {"item_id", "seed", "score"})
                self.assertIn(r["score"], (0.0, 1.0))
                self.assertIsInstance(r["item_id"], int)
            self.assertEqual({r["seed"] for r in rec["records"]}, {11, 12})
        blob = self.outside_trust(proj)
        for it in sealed:                                               # AN-8: nothing sealed leaves the backend
            self.assertNotIn(it["input"].encode(), blob)
            self.assertNotIn(it["label"].encode(), blob)
        self.assertNotIn(b"wrong", blob)                                 # no prediction text either
        self.assertEqual(InProcessTrust(proj).sealed_guarantee, UNSEALED_GUARANTEE)

    def test_uses_exactly_the_registered_sampler_and_seeds_and_no_label_reaches_the_generator(self) -> None:
        proj = self.make()
        self.upto_registration(proj)
        self.run_(proj, "evaluate_sealed")
        self.assertTrue(GENERATE_CALLS)
        for sampler, seed, keys in GENERATE_CALLS:
            self.assertEqual(sampler, ti.DEPLOY["sampler"])
            self.assertIn(seed, ti.DEPLOY["eval_seeds"])
            self.assertEqual(keys, ["input"])                            # the label stays with the scorer
        self.assertEqual(len(GENERATE_CALLS), 155 * 2 * 2)               # items x seeds x two models

    def test_scores_come_from_exact_match(self) -> None:
        proj = self.make()
        self.upto_registration(proj)
        out = self.run_(proj, "evaluate_sealed").outputs
        cand = proj.read_artifact(out["candidate_records"])
        self.assertTrue(all(r["score"] == 1.0 for r in cand["records"]))      # rate 1.0 answers every item right
        base = proj.read_artifact(out["baseline_records"])
        self.assertTrue(0.3 < sum(r["score"] for r in base["records"]) / len(base["records"]) < 0.7)
        self.assertEqual(scorers.exact_match("a", "a"), 1.0)

    def test_refuses_when_the_registered_evaluator_hash_is_not_the_scorers(self) -> None:
        proj = self.make()
        self.upto_registration(proj)
        real = scorers.manifest_hash
        scorers.manifest_hash = lambda: "sha256:" + "0" * 64
        self.addCleanup(setattr, scorers, "manifest_hash", real)
        p = self.refused(proj, "evaluate_sealed", rule="AN-8")
        self.assertIn("evaluator", p.message)
        self.assertFalse((proj.root / "stages" / "evaluate_sealed").exists())

    def test_unknown_generator_is_refused(self) -> None:
        proj = self.make()
        self.upto_registration(proj)
        trust_stages.GENERATORS.pop(GENERATOR)
        p = self.refused(proj, "evaluate_sealed", rule="AN-8")
        self.assertIn("generator", p.message)

    def test_generator_failure_is_refused_without_item_text(self) -> None:
        proj = self.make()
        self.upto_registration(proj)

        def boom(model_ref, item, *, sampler, seed):
            raise RuntimeError("secret " + item["input"])

        register_generator(GENERATOR, boom)
        p = self.refused(proj, "evaluate_sealed", rule="AN-8")
        self.assertNotIn("marker", p.message)
        self.assertNotIn("secret", p.message)


class GateTest(GateBase):
    def gate_result(self, proj: Project, out: dict) -> dict:
        return proj.read_artifact(out["gate"]["gate_result"])

    def test_pass_when_the_candidate_is_better(self) -> None:
        proj = self.make()
        res = self.gate_result(proj, self.through(proj, "gate"))
        self.assertEqual(res["status"], "PASS")
        self.assertGreater(res["delta_hat_natural"], 0.3)
        self.assertGreater(res["ci_lower_natural"], 0.0)
        self.assertEqual(res["n_items"], 155)

    def test_fail_when_the_candidate_is_worse(self) -> None:
        self.base_rate, self.cand_rate = 1.0, 0.5
        proj = self.make()
        res = self.gate_result(proj, self.through(proj, "gate"))
        self.assertEqual(res["status"], "FAIL")
        self.assertLess(res["delta_hat_natural"], 0.0)

    def test_inconclusive_when_the_candidate_is_equal(self) -> None:
        self.base_rate = self.cand_rate = 0.5
        proj = self.make()
        res = self.gate_result(proj, self.through(proj, "gate"))
        self.assertEqual(res["status"], "INCONCLUSIVE")
        self.assertEqual(res["delta_hat_natural"], 0.0)

    def test_alpha_comes_from_the_ledger_and_the_outcome_is_recorded(self) -> None:
        class Ledger(FixedAlphaLedger):
            def __init__(self, project):
                super().__init__(project, alpha=0.05)
                self.spent = []

            def spend(self, attempt_id):
                self.spent.append(attempt_id)
                return super().spend(attempt_id)

        proj = self.make()
        self.upto_registration(proj)
        self.run_(proj, "evaluate_sealed")
        ledger = Ledger(proj)
        out = run_stage(proj, "gate", {}, registry=self.registry(), ledger=ledger).outputs
        res = proj.read_artifact(out["gate_result"])
        self.assertEqual(len(ledger.spent), 1)
        self.assertEqual(res["attempt_id"], ledger.spent[0])
        self.assertEqual(res["alpha"], 0.05)
        self.assertEqual(res["registration_hash"], proj.read_artifact(
            self.run_(proj, "preregister", {"metric_confirmed": True}).outputs["registration"])["hash"])
        self.assertFalse(res["post_hoc"])
        entries = ledger.entries()
        self.assertEqual(entries, [{"attempt_id": res["attempt_id"], "alpha": 0.05, "passed": True}])

    def test_alpha_is_not_the_registered_alpha_when_the_ledger_differs(self) -> None:
        proj = self.make()
        self.upto_registration(proj)
        self.run_(proj, "evaluate_sealed")
        out = run_stage(proj, "gate", {}, registry=self.registry(), ledger=FixedAlphaLedger(proj, alpha=0.2)).outputs
        res = proj.read_artifact(out["gate_result"])
        self.assertEqual(res["alpha"], 0.2)
        self.assertAlmostEqual(res["ci_level"], 0.6)                    # 1 - 2 * 0.2

    def test_refuses_when_no_alpha_is_available(self) -> None:
        proj = self.make()
        self.upto_registration(proj)
        self.run_(proj, "evaluate_sealed")
        ledger = FixedAlphaLedger(proj)
        ledger.spend("someone-else")
        with self.assertRaises(Refused) as cm:
            run_stage(proj, "gate", {}, registry=self.registry(), ledger=ledger)
        self.assertEqual({p.rule for p in cm.exception.problems}, {"AN-9"})
        self.assertEqual(cm.exception.problems[0].field, "gate")
        self.assertEqual(len(ledger.entries()), 1)                        # nothing new spent

    def test_available_zero_refuses_before_any_spend(self) -> None:
        class Empty(FixedAlphaLedger):
            def available(self):
                return 0.0

        proj = self.make()
        self.upto_registration(proj)
        self.run_(proj, "evaluate_sealed")
        ledger = Empty(proj)
        with self.assertRaises(Refused) as cm:
            run_stage(proj, "gate", {}, registry=self.registry(), ledger=ledger)
        self.assertEqual(cm.exception.problems[0].rule, "AN-9")
        self.assertEqual(FixedAlphaLedger(proj).entries(), [])           # spend was never called

    def test_a_second_gate_run_is_refused(self) -> None:
        proj = self.make()
        self.upto_registration(proj)
        self.run_(proj, "evaluate_sealed")
        self.run_(proj, "gate")
        with self.assertRaises(Refused) as cm:
            self.run_(proj, "gate", {"rerun": True, "again": 1})
        self.assertEqual(cm.exception.problems[0].rule, "AN-9")
        self.assertEqual(len(FixedAlphaLedger(proj).entries()), 1)

    def test_same_gate_run_again_returns_the_recorded_result(self) -> None:
        proj = self.make()
        self.upto_registration(proj)
        self.run_(proj, "evaluate_sealed")
        first = self.run_(proj, "gate")
        again = self.run_(proj, "gate")
        self.assertTrue(again.reused)
        self.assertEqual(first.outputs["gate_result"].sha256, again.outputs["gate_result"].sha256)

    def test_item_set_hash_mismatch_is_refused_before_spending(self) -> None:
        proj = self.make()
        self.upto_registration(proj)
        out = self.run_(proj, "evaluate_sealed").outputs
        ref = out["candidate_records"]
        rec = proj.read_artifact(ref)
        rec["item_set_hash"] = "sha256:" + "1" * 64
        (proj.root / ref.path).write_bytes(__import__("gatedtrain.canonical", fromlist=["x"]).canonical_json(rec))
        # The record changed after it was written: the machine refuses it by hash before the gate runs.
        with self.assertRaises(Refused):
            self.run_(proj, "gate")
        self.assertEqual(FixedAlphaLedger(proj).entries(), [])
        self.assertTrue(FixedAlphaLedger(proj).available() > 0)

    def test_records_for_another_item_set_are_refused_without_spending(self) -> None:
        from gatedtrain.ainative.stage import StageContext
        proj = self.make()
        self.upto_registration(proj)
        out = self.run_(proj, "evaluate_sealed").outputs
        rec = proj.read_artifact(out["candidate_records"])
        rec["item_set_hash"] = "sha256:" + "1" * 64
        forged = proj.write_artifact("evaluate_sealed", "candidate_records", rec, out["candidate_records"].schema)
        proj.append_attempt({"stage": "gate", "attempt": 1, "key": "k1", "event": "start"})
        ctx = StageContext(proj, {"registration": self.reg_ref(proj), "baseline_records": out["baseline_records"],
                                  "candidate_records": forged}, {}, InProcessTrust(proj), FixedAlphaLedger(proj))
        with self.assertRaises(Refused) as cm:
            Gate().run(ctx, {})
        self.assertEqual(cm.exception.problems[0].rule, "AN-9")
        self.assertIn("item set", cm.exception.problems[0].message)
        self.assertEqual(FixedAlphaLedger(proj).entries(), [])
        self.assertGreater(FixedAlphaLedger(proj).available(), 0)

    def test_post_hoc_registration_cannot_pass(self) -> None:
        proj = self.make()
        self.upto_registration(proj)
        # Amend the registration after training: it becomes post hoc.
        self.run_(proj, "preregister", {"metric_confirmed": True, "margin": 0.05, "rerun": True})
        out = {}
        for ref in ("evaluate_sealed", "gate"):
            out[ref] = self.run_(proj, ref).outputs
        res = self.gate_result(proj, out)
        self.assertTrue(res["post_hoc"])
        self.assertNotEqual(res["status"], "PASS")
        self.assertIn("post_hoc", res["status_reasons"])
        self.assertEqual(res["margin"], 0.05)


NUM = re.compile(r"-?\d+(?:\.\d+)?")


class ReportTest(GateBase):
    def reports(self, proj: Project) -> tuple[dict, str]:
        out = self.through(proj)["report"]
        rep = proj.read_artifact(out["report"])
        md = (proj.root / "stages" / "report" / "report.md").read_text(encoding="utf-8")
        return rep, md

    def test_report_states_status_effect_interval_n_attempts_and_the_guarantee(self) -> None:
        proj = self.make()
        rep, md = self.reports(proj)
        self.assertEqual(rep["status"], "PASS")
        self.assertIn("PASS", md.splitlines()[0])
        self.assertIn(rep["sealed_guarantee"], md)
        self.assertEqual(rep["sealed_guarantee"], InProcessTrust(proj).sealed_guarantee)
        self.assertEqual(rep["n_sealed"], 155)
        self.assertIn("155", md)
        for key in ("effect", "ci_lower", "ci_upper"):
            self.assertIn(str(rep["display"][key]), md)
        self.assertEqual(len(rep["ledger"]), 1)
        self.assertIn(rep["ledger"][0]["attempt_id"], md)
        self.assertEqual(rep["registration"]["scorer"], "exact_match")
        self.assertIn("exact_match", md)
        self.assertIn("share of items", md)                               # the metric unit

    def test_every_number_in_report_md_is_in_report_json(self) -> None:
        proj = self.make()
        rep, md = self.reports(proj)
        json_numbers = set(NUM.findall(json.dumps(rep, sort_keys=True)))
        md_numbers = set(NUM.findall(md))
        self.assertTrue(md_numbers)
        self.assertEqual(md_numbers - json_numbers, set())

    def test_report_md_is_rendered_from_report_json_only(self) -> None:
        proj = self.make()
        rep, md = self.reports(proj)
        self.assertEqual(md, render_markdown(rep))
        changed = json.loads(json.dumps(rep))
        changed["display"]["effect"] = 0.1234
        self.assertIn("0.1234", render_markdown(changed))
        self.assertNotIn("0.1234", md)

    def test_post_hoc_never_gets_a_pass_headline(self) -> None:
        proj = self.make()
        rep, _ = self.reports(proj)
        forged = json.loads(json.dumps(rep))
        forged["registration"]["post_hoc"] = True
        forged["gate_result"]["post_hoc"] = True
        forged["status"] = "PASS"
        md = render_markdown(forged)
        head = md.splitlines()[0]
        self.assertNotIn("PASS", head)
        self.assertIn("post hoc, not a confirmed gain", head)

    def test_post_hoc_pipeline_report_headline(self) -> None:
        proj = self.make()
        self.upto_registration(proj)
        self.run_(proj, "preregister", {"metric_confirmed": True, "margin": 0.05, "rerun": True})
        for ref in ("evaluate_sealed", "gate", "report"):
            out = self.run_(proj, ref).outputs
        rep = proj.read_artifact(out["report"])
        md = (proj.root / "stages" / "report" / "report.md").read_text(encoding="utf-8")
        self.assertTrue(rep["registration"]["post_hoc"])
        self.assertIn("post hoc, not a confirmed gain", md.splitlines()[0])
        self.assertNotIn("PASS", md.splitlines()[0])

    def test_report_json_holds_only_records(self) -> None:
        proj = self.make()
        rep, _ = self.reports(proj)
        self.assertEqual(set(rep["registration"]), {"hash", "scorer", "direction", "margin", "alpha",
                                                    "item_set_hash", "post_hoc", "amends"})
        self.assertEqual(rep["power"]["n_sealed"], 155)
        blob = self.outside_trust(proj)
        summary = proj.read_artifact(self.run_(proj, "intake_seal").outputs["sealed_summary"])
        for it in self.sealed_items(proj, summary["manifest_sha256"]):
            self.assertNotIn(it["input"].encode(), blob)
            self.assertNotIn(it["label"].encode(), blob)

    def test_inconclusive_and_fail_headlines(self) -> None:
        self.base_rate = self.cand_rate = 0.5
        rep, md = self.reports(self.make(name="eq"))
        self.assertEqual(rep["status"], "INCONCLUSIVE")
        self.assertIn("INCONCLUSIVE", md.splitlines()[0])
        self.base_rate, self.cand_rate = 1.0, 0.5
        rep, md = self.reports(self.make(name="worse"))
        self.assertEqual(rep["status"], "FAIL")
        self.assertIn("FAIL", md.splitlines()[0])

    def test_check_refuses_a_report_md_that_differs_from_report_json(self) -> None:
        proj = self.make()
        outs = self.through(proj)
        out = outs["report"]
        md = proj.root / "stages" / "report" / "report.md"
        md.write_text(md.read_text(encoding="utf-8") + "\nA new number: 0.9999\n", encoding="utf-8")
        ctx = __import__("gatedtrain.ainative.stage", fromlist=["x"]).StageContext(
            proj, {"registration": self.reg_ref(proj), "power": self.run_(proj, "intake_seal").outputs["power"],
                   "gate_result": outs["gate"]["gate_result"]}, {}, InProcessTrust(proj), FixedAlphaLedger(proj))
        problems = Report().check(ctx, out)
        self.assertTrue(problems)
        self.assertEqual({p.rule for p in problems}, {"AN-10"})


if __name__ == "__main__":
    unittest.main()
