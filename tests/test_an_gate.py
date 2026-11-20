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
from gatedtrain.ainative.trust_stages import (EvaluateSealed, Gate, Report, SCHEMA_MODEL_REF, render_markdown)
from gatedtrain.errors import Refused

GENERATOR = "t7-generator"
TEST_GENERATORS: dict = {}          # what the test-only scenario entry point exposes


class _TestScenario:
    """A test-only `gatedtrain.scenarios` entry point: the loader is patched in setUp, so no stage can reach it."""
    name = "t7"
    value = "test:scenario"

    def load(self):
        return self

    @property
    def generators(self):
        return TEST_GENERATORS


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


def model_stage(stage_name: str, rate: float, trains: bool, generator: str = GENERATOR) -> type:
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
            rec = {"schema": SCHEMA_MODEL_REF, "schema_version": 1, "generator": generator, "rate": rate,
                   "label": stage_name}
            return {"model": ctx.project.write_artifact(self.name, "model", rec, SCHEMA_MODEL_REF)}

        def check(self, ctx, outputs):
            return []

    _Model.trains = trains
    return _Model


def choose_stage(base_rate: float) -> type:
    class _Choose(ti.ChooseDeployment):
        def run(self, ctx, decision):
            out = super().run(ctx, decision)
            base = {"schema": SCHEMA_MODEL_REF, "schema_version": 1, "generator": GENERATOR, "rate": base_rate,
                    "label": "base"}
            out["base"] = ctx.project.write_artifact(self.name, "base", base, SCHEMA_MODEL_REF)
            return out

    return _Choose


class GateBase(ti.Base):
    base_rate = 0.5
    cand_rate = 1.0
    cand_generator = GENERATOR

    def setUp(self) -> None:
        super().setUp()
        TEST_GENERATORS.clear()
        TEST_GENERATORS[GENERATOR] = generate
        self.addCleanup(TEST_GENERATORS.clear)
        real = trust_stages._scenario_entry_points
        trust_stages._scenario_entry_points = lambda: [_TestScenario()]
        self.addCleanup(setattr, trust_stages, "_scenario_entry_points", real)
        trust_stages._reset_scenario_snapshot()
        self.addCleanup(trust_stages._reset_scenario_snapshot)
        GENERATE_CALLS.clear()
        LABELS.clear()

    baseline_src = "choose_x.base"
    candidate_src = "train_x.model"
    extra_stage = False

    def make(self, data: bytes | None = None, *, name: str = "proj", n: int = ti.N_DATA) -> Project:
        rows = unique_rows(n)
        LABELS.update({r["input"]: r["label"] for r in rows})
        d = self.tmp / f"{name}.jsonl"
        d.write_bytes(ti.jsonl(rows))
        spec = {"name": "t", "training_stages": ["train_x"], "stages": [
            {"name": "intake_seal", "stage": "intake_seal"},
            {"name": "choose_x", "stage": "choose_x"},
            {"name": "preregister", "stage": "preregister",
             "inputs": {"sealed_summary": "intake_seal.sealed_summary", "deployment": "choose_x.deployment",
                        "base_model": "choose_x.base"}},
            {"name": "train_x", "stage": "train_x"},
            *([{"name": "other_x", "stage": "other_x"}] if self.extra_stage else []),
            {"name": "evaluate_sealed", "stage": "evaluate_sealed",
             "inputs": {"registration": "preregister.registration", "baseline_model": self.baseline_src,
                        "candidate_model": self.candidate_src}},
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
        return {**stage_registry(), "train_x": model_stage("train_x", self.cand_rate, True, self.cand_generator),
                "other_x": model_stage("other_x", self.cand_rate, False), "choose_x": choose_stage(self.base_rate)}

    def upto_registration(self, proj: Project) -> None:
        self.run_(proj, "intake_seal")
        self.run_(proj, "choose_x")
        self.run_(proj, "preregister", {"metric_confirmed": True})
        self.run_(proj, "train_x")
        if self.extra_stage:
            self.run_(proj, "other_x")

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


SCHEMA_RECORDS = trust_stages.SCHEMA_SEALED_RECORDS


def plugin(name: str, outputs: dict) -> type:
    return type(name, (), {"name": name, "version": "1", "inputs": {}, "outputs": outputs, "needs": frozenset(),
                           "params": frozenset(), "trains": False,
                           "plan": lambda self, ctx: Proposal(name, "x", {}, {}),
                           "run": lambda self, ctx, decision: {}, "check": lambda self, ctx, outputs: []})


class RegistryTest(unittest.TestCase):
    def test_real_classes_replace_the_last_three_placeholders(self) -> None:
        reg = stage_registry()
        for name, cls in (("evaluate_sealed", EvaluateSealed), ("gate", Gate), ("report", Report)):
            self.assertIs(reg[name], cls)
            self.assertEqual(cls.params, frozenset())
            self.assertFalse(cls.trains)
            self.assertEqual(cls.needs, frozenset())
            self.assertEqual(cls.name, name)


class RefusalKeepsStateTest(GateBase):
    """A refusal in `run` is audited as `refused` and never changes the derived state (a refused second gate or
    amendment once left the gate `failed` and the report stale, hiding a recorded verdict)."""

    base_rate = cand_rate = 0.5                  # INCONCLUSIVE

    def test_an_inconclusive_verdict_survives_a_refused_second_gate_and_amendment(self) -> None:
        from gatedtrain.ainative.machine import done_output, status

        proj = self.make()
        self.through(proj)
        reg = self.registry()
        for stage, decision in (("gate", {"rerun": True, "again": 1}),
                                ("preregister", {"metric_confirmed": True, "margin": 0.05, "rerun": True})):
            with self.assertRaises(Refused):
                run_stage(proj, stage, decision, registry=reg)
        states = {k: v["status"] for k, v in status(proj, registry=reg).items()}
        self.assertEqual(set(states.values()), {"done"}, states)
        self.assertEqual(done_output(proj, "gate", "gate_result", registry=reg)["status"], "INCONCLUSIVE")
        self.assertEqual(done_output(proj, "report", "report", registry=reg)["status"], "INCONCLUSIVE")
        refused = [r for r in proj.read_attempts() if r["event"] == "refused"]
        self.assertEqual([(r["stage"], r["rules"]) for r in refused], [("gate", ["AN-9"]), ("preregister", ["AN-7"])])
        self.assertEqual(len(FixedAlphaLedger(proj).entries()), 1)

    def test_a_refusal_after_the_outputs_were_written_still_fails_the_stage(self) -> None:
        from gatedtrain.ainative.machine import status
        from gatedtrain.errors import Problem

        class Bad(model_stage("train_x", 1.0, True)):
            def check(self, ctx, outputs):
                return [Problem("T-1", "model", "the output check refuses", "fix")]

        proj = self.make()
        reg = {**self.registry(), "train_x": Bad}
        for stage in ("intake_seal", "choose_x", "preregister"):
            run_stage(proj, stage, {"metric_confirmed": True} if stage == "preregister" else {}, registry=reg)
        with self.assertRaises(Refused):
            run_stage(proj, "train_x", {}, registry=reg)
        self.assertEqual(status(proj, registry=reg)["train_x"]["status"], "failed")
        self.assertEqual(proj.read_attempts()[-1]["event"], "failed")


    def test_a_refusal_after_overwriting_a_record_fails_the_stage_and_a_plain_rerun_recovers(self) -> None:
        from gatedtrain.ainative.machine import status
        from gatedtrain.errors import Problem

        class WriteThenRefuse(model_stage("train_x", 1.0, True)):
            def run(self, ctx, decision):
                if decision.get("boom"):
                    ctx.project.write_artifact(self.name, "model", {"junk": 1}, SCHEMA_MODEL_REF)
                    raise Refused([Problem("X-1", "model", "refused after writing", "fix")])
                return super().run(ctx, decision)

        proj = self.make()
        reg = {**self.registry(), "train_x": WriteThenRefuse}
        for stage in ("intake_seal", "choose_x", "preregister", "train_x"):
            run_stage(proj, stage, {"metric_confirmed": True} if stage == "preregister" else {}, registry=reg)
        with self.assertRaises(Refused):
            run_stage(proj, "train_x", {"boom": True, "rerun": True}, registry=reg)
        self.assertEqual(status(proj, registry=reg)["train_x"]["status"], "failed")
        last = proj.read_attempts()[-1]
        self.assertEqual((last["event"], last["rules"]), ("failed", ["X-1"]))
        run_stage(proj, "train_x", {}, registry=reg)                       # a plain rerun recovers
        self.assertEqual(status(proj, registry=reg)["train_x"]["status"], "done")

    def test_a_refusal_after_spending_alpha_fails_the_stage(self) -> None:
        from gatedtrain.ainative.machine import status

        proj = self.make()
        self.through(proj, "evaluate_sealed")
        real = FixedAlphaLedger.record

        def boom(self_, attempt_id, passed):
            raise Refused([trust_stages.Problem("X-2", "gate", "refused after the spend", "fix")])

        FixedAlphaLedger.record = boom
        try:
            with self.assertRaises(Refused):
                self.run_(proj, "gate")
        finally:
            FixedAlphaLedger.record = real
        self.assertEqual(len(FixedAlphaLedger(proj).entries()), 1)
        self.assertEqual(status(proj, registry=self.registry())["gate"]["status"], "failed")


class EvaluateSealedTest(GateBase):
    def test_records_hold_item_id_seed_and_score_only_and_no_sealed_text(self) -> None:
        proj = self.make()
        self.upto_registration(proj)
        out = self.run_(proj, "evaluate_sealed").outputs
        summary = proj.read_artifact(self.run_(proj, "intake_seal").outputs["sealed_summary"])
        sealed = self.sealed_items(proj, summary["manifest_sha256"])
        self.assertEqual(len(sealed), ti.N_NEEDED)
        for name in ("baseline_records", "candidate_records"):
            rec = proj.read_artifact(out[name])
            self.assertEqual(len(rec["records"]), ti.N_NEEDED * 2)               # two registered eval seeds
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
        self.assertEqual(len(GENERATE_CALLS), ti.N_NEEDED * 2 * 2)               # items x seeds x two models

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
        TEST_GENERATORS.pop(GENERATOR)
        trust_stages._reset_scenario_snapshot()
        p = self.refused(proj, "evaluate_sealed", rule="AN-8")
        self.assertIn("generator", p.message)

    def test_generator_failure_is_refused_without_item_text(self) -> None:
        proj = self.make()
        self.upto_registration(proj)

        def boom(model_ref, item, *, sampler, seed):
            raise RuntimeError("secret " + item["input"])

        TEST_GENERATORS[GENERATOR] = boom
        trust_stages._reset_scenario_snapshot()
        p = self.refused(proj, "evaluate_sealed", rule="AN-8")
        self.assertNotIn("marker", p.message)
        self.assertNotIn("secret", p.message)


class GeneratorBindingTest(GateBase):
    def test_the_registration_records_the_pack_and_the_generator(self) -> None:
        proj = self.make()
        self.upto_registration(proj)
        full = InProcessTrust(proj).registration(proj.read_artifact(self.reg_ref(proj))["hash"])
        self.assertEqual(full["base_model"]["scenario_pack"], "t7")
        self.assertEqual(full["base_model"]["generator"], GENERATOR)
        self.assertEqual(trust_stages.registered_generator(full), ("t7", GENERATOR))

    def test_mutating_the_pack_after_the_first_load_changes_nothing(self) -> None:
        proj = self.make()
        self.upto_registration(proj)                       # preregister loaded the snapshot
        TEST_GENERATORS[GENERATOR] = lambda model_ref, item, *, sampler, seed: "wrong"
        TEST_GENERATORS["other"] = generate
        out = self.run_(proj, "evaluate_sealed").outputs
        cand = proj.read_artifact(out["candidate_records"])
        self.assertTrue(all(r["score"] == 1.0 for r in cand["records"]))
        self.assertNotIn("other", trust_stages.scenario_generators())
        with self.assertRaises(TypeError):
            trust_stages.scenario_generators()["x"] = generate

    def test_a_pack_without_generators_is_not_called_a_load_failure(self) -> None:
        class NoGens:
            name = "nogens"

            def load(self):
                return object()

        trust_stages._scenario_entry_points = lambda: [NoGens()]
        trust_stages._reset_scenario_snapshot()
        with self.assertRaises(Refused) as cm:
            trust_stages.scenario_generators()
        self.assertIn("no `generators`", cm.exception.problems[0].message)
        self.assertNotIn("failed to load", cm.exception.problems[0].message)


class OtherGeneratorTest(GateBase):
    cand_generator = "other-generator"

    def test_a_candidate_with_another_generator_is_refused(self) -> None:
        TEST_GENERATORS["other-generator"] = generate
        proj = self.make()
        self.upto_registration(proj)
        p = self.refused(proj, "evaluate_sealed", rule="AN-8")
        self.assertIn("candidate", p.message)
        self.assertIn("generator", p.message)
        self.assertFalse((proj.root / "stages" / "evaluate_sealed").exists())


class GateTest(GateBase):
    def gate_result(self, proj: Project, out: dict) -> dict:
        return proj.read_artifact(out["gate"]["gate_result"])

    def test_pass_when_the_candidate_is_better(self) -> None:
        proj = self.make()
        res = self.gate_result(proj, self.through(proj, "gate"))
        self.assertEqual(res["status"], "PASS")
        self.assertGreater(res["delta_hat_natural"], 0.3)
        self.assertGreater(res["ci_lower_natural"], 0.0)
        self.assertEqual(res["n_items"], ti.N_NEEDED)

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

    def test_spent_alpha_fix_texts_ask_for_new_sealed_data_not_a_new_project(self) -> None:
        """Final review I1: a new project on the same data would get a fresh alpha and the same sealed split."""
        from gatedtrain.ainative.trust import NEW_TEST_FIX

        self.assertEqual(NEW_TEST_FIX, "this project's sealed test is used; a new test needs new sealed data")
        proj = self.make()
        self.through(proj, "gate")
        fixes = []
        for stage, decision in (("gate", {"rerun": True, "again": 1}),
                                ("preregister", {"metric_confirmed": True, "margin": 0.05, "rerun": True})):
            with self.assertRaises(Refused) as cm:
                self.run_(proj, stage, decision)
            fixes.append(cm.exception.problems[0].fix)
        with self.assertRaises(Refused) as cm:
            FixedAlphaLedger(proj).spend("again")
        fixes.append(cm.exception.problems[0].fix)
        self.assertEqual(fixes, [NEW_TEST_FIX] * 3)

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
        self.assertEqual(rep["n_sealed"], ti.N_NEEDED)
        self.assertIn(str(ti.N_NEEDED), md)
        for key in ("effect", "ci_lower", "ci_upper"):
            self.assertIn(str(rep["display"][key]), md)
        self.assertEqual(len(rep["ledger"]), 1)
        self.assertIn(rep["ledger"][0]["attempt_id"], md)
        self.assertEqual(rep["registration"]["scorer"], "exact_match")
        self.assertIn("exact_match", md)
        self.assertIn("share of items", md)                               # the metric unit

    def test_known_limits_is_a_list_built_from_constants_and_records(self) -> None:
        proj = self.make()
        rep, md = self.reports(proj)
        limits = rep["known_limits"]
        self.assertEqual([x["id"] for x in limits],
                         ["sealed_guarantee", "alpha_per_project", "metric_confirmation", "model_reference",
                          "sealed_evaluations"])
        by_id = {x["id"]: x for x in limits}
        self.assertEqual(by_id["sealed_guarantee"]["text"], UNSEALED_GUARANTEE)
        self.assertEqual(by_id["sealed_guarantee"]["text"], rep["sealed_guarantee"])
        for words in ("read the sealed items", "rewrite", "trust records", "attempt log"):
            self.assertIn(words, UNSEALED_GUARANTEE)
        self.assertIn("per project", by_id["alpha_per_project"]["text"])
        self.assertIn("same sealed split", by_id["alpha_per_project"]["text"])
        self.assertIn("metric_confirmed", by_id["metric_confirmation"]["text"])
        self.assertIn("content-address", by_id["model_reference"]["text"])
        self.assertEqual(by_id["sealed_evaluations"]["attempts"], 1)
        self.assertIn("1 evaluate_sealed attempt", by_id["sealed_evaluations"]["text"])
        section = md.split("## Known limits\n", 1)[1]
        for x in limits:
            self.assertIn(f"- {x['text']}\n", section)

    def test_the_readme_quotes_the_guarantee_exactly(self) -> None:
        readme = " ".join((Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8").split())
        self.assertTrue(f'"{UNSEALED_GUARANTEE}"' in readme, "README.md must quote sealed_guarantee exactly")
        self.assertTrue("alpha is per project" in readme, "README.md must say the alpha is per project")

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
        self.assertEqual(rep["power"]["n_sealed"], ti.N_NEEDED)
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


class TrustWiringTest(GateBase):
    """Review round 1, critical 1: a plugin cannot stand in for a trust stage's records."""

    def problems(self, mutate) -> list:
        from gatedtrain.ainative.pipeline import load_pipeline, pipeline_problems
        proj = self.make()
        raw = yaml.safe_load(Path(proj.pipeline).read_text())
        mutate(raw)
        path = self.tmp / "mutated.yaml"
        path.write_text(yaml.safe_dump(raw))
        reg = {**self.registry(), "forge_x": plugin("forge_x", {"baseline_records": SCHEMA_RECORDS,
                                                                "registration": trust_stages.SCHEMA_REGISTRATION})}
        return pipeline_problems(load_pipeline(path), reg)

    @staticmethod
    def _insert_forger(raw: dict) -> None:
        names = [x["name"] for x in raw["stages"]]
        raw["stages"].insert(names.index("evaluate_sealed"), {"name": "forge_x", "stage": "forge_x"})

    def test_a_plugin_that_declares_a_trust_schema_is_refused(self) -> None:
        got = self.problems(self._insert_forger)
        refused = {(p.rule, p.field) for p in got}
        self.assertIn(("AN-3", "stages[4].outputs.baseline_records"), refused)
        self.assertIn(("AN-3", "stages[4].outputs.registration"), refused)

    def test_gate_inputs_wired_from_a_plugin_are_refused(self) -> None:
        def mutate(raw):
            self._insert_forger(raw)
            gate = next(x for x in raw["stages"] if x["name"] == "gate")
            gate["inputs"]["baseline_records"] = "forge_x.baseline_records"
            gate["inputs"]["registration"] = "forge_x.registration"

        refused = {(p.rule, p.field) for p in self.problems(mutate)}
        self.assertIn(("AN-4", "stages[6].inputs.baseline_records"), refused)
        self.assertIn(("AN-4", "stages[6].inputs.registration"), refused)

    def test_a_renamed_trust_stage_is_refused(self) -> None:
        def mutate(raw):
            gate = next(x for x in raw["stages"] if x["name"] == "gate")
            gate["name"] = "gate2"
            rep = next(x for x in raw["stages"] if x["name"] == "report")
            rep["inputs"]["gate_result"] = "gate2.gate_result"

        got = self.problems(mutate)
        self.assertTrue(any(p.rule == "AN-4" and "must be named 'gate'" in p.message for p in got), got)

    def test_a_plugin_entry_point_with_a_trust_schema_is_refused(self) -> None:
        from gatedtrain.ainative import stage as stage_mod

        class EP:
            name = "forge_x"
            value = "x:y"

            def load(self):
                return plugin("forge_x", {"gate_result": trust_stages.SCHEMA_GATE_RESULT})

        real = stage_mod._plugin_entry_points
        stage_mod._plugin_entry_points = lambda: [EP()]
        self.addCleanup(setattr, stage_mod, "_plugin_entry_points", real)
        with self.assertRaises(Refused) as cm:
            stage_registry()
        self.assertEqual(cm.exception.problems[0].rule, "AN-3")
        self.assertIn("reserved trust schema", cm.exception.problems[0].message)


class ForgeBase(GateBase):
    """Direct `Gate.run` on records the test writes itself (what a hostile record writer could produce)."""

    def forged_gate(self, proj: Project, *, mutate_cand=None, mutate_base=None, reg_ref=None,
                    record_done: bool = True) -> dict:
        from gatedtrain.ainative.stage import StageContext
        if not any(r["stage"] == "evaluate_sealed" and r["event"] == "done" for r in proj.read_attempts()):
            self.run_(proj, "evaluate_sealed")
        from gatedtrain.ainative.project import ArtifactRef
        done0 = [r for r in proj.read_attempts() if r["stage"] == "evaluate_sealed" and r["event"] == "done"][-1]
        outs = {n: ArtifactRef(**o) for n, o in done0["outputs"].items()}
        refs = {}
        for arm, mut in (("baseline_records", mutate_base), ("candidate_records", mutate_cand)):
            rec = proj.read_artifact(outs[arm])
            if mut:
                mut(rec)
            refs[arm] = proj.write_artifact("evaluate_sealed", arm, rec, SCHEMA_RECORDS)
        if record_done:
            done = [r for r in proj.read_attempts() if r["stage"] == "evaluate_sealed" and r["event"] == "done"][-1]
            row = dict(done, outputs={n: {"stage": r.stage, "name": r.name, "schema": r.schema, "sha256": r.sha256,
                                          "path": r.path} for n, r in refs.items()})
            proj.append_attempt({k: v for k, v in row.items() if k not in ("schema", "schema_version")})
        proj.append_attempt({"stage": "gate", "attempt": 1, "key": "k1", "event": "start"})
        ctx = StageContext(proj, {"registration": reg_ref or self.reg_ref(proj), **refs}, {}, InProcessTrust(proj),
                           FixedAlphaLedger(proj))
        out = Gate().run(ctx, {})
        return proj.read_artifact(out["gate_result"])

    def refused_gate(self, proj: Project, **kw):
        with self.assertRaises(Refused) as cm:
            self.forged_gate(proj, **kw)
        self.assertEqual(cm.exception.problems[0].rule, "AN-9")
        self.assertEqual(FixedAlphaLedger(proj).entries(), [])           # nothing was spent
        return cm.exception.problems[0]

    def test_records_that_are_not_the_ones_evaluate_sealed_recorded_are_refused(self) -> None:
        proj = self.make()
        self.upto_registration(proj)

        def all_ones(rec):
            for r in rec["records"]:
                r["score"] = 1.0

        p = self.refused_gate(proj, mutate_base=all_ones, record_done=False)
        self.assertIn("recorded", p.message)

    def test_an_edited_registration_record_cannot_clear_post_hoc(self) -> None:
        proj = self.make()
        self.upto_registration(proj)
        first = self.reg_ref(proj)
        second = self.run_(proj, "preregister", {"metric_confirmed": True, "margin": 0.05, "rerun": True}).outputs[
            "registration"]
        rec = proj.read_artifact(second)
        self.assertTrue(rec["post_hoc"])
        self.run_(proj, "evaluate_sealed")
        forged = proj.write_artifact("preregister", "registration", dict(rec, post_hoc=False), second.schema)
        p = self.refused_gate(proj, reg_ref=forged)
        self.assertIn("post_hoc", p.field)
        # The older registration (hash not the latest) is refused as well.
        proj2 = self.make(name="p2")
        self.upto_registration(proj2)
        old = self.reg_ref(proj2)
        old_rec = proj2.read_artifact(old)
        self.run_(proj2, "preregister", {"metric_confirmed": True, "margin": 0.05, "rerun": True})
        self.run_(proj2, "evaluate_sealed")
        old = proj2.write_artifact("preregister", "registration", old_rec, old.schema)
        p = self.refused_gate(proj2, reg_ref=old)
        self.assertIn("latest", p.message)

    def test_report_takes_post_hoc_from_the_trust_store(self) -> None:
        from gatedtrain.ainative.stage import StageContext
        proj = self.make()
        self.upto_registration(proj)
        second = self.run_(proj, "preregister", {"metric_confirmed": True, "margin": 0.05, "rerun": True}).outputs[
            "registration"]
        for ref in ("evaluate_sealed", "gate"):
            out = self.run_(proj, ref).outputs
        forged = proj.write_artifact("preregister", "registration",
                                     dict(proj.read_artifact(second), post_hoc=False), second.schema)
        ctx = StageContext(proj, {"registration": forged, "power": self.run_(proj, "intake_seal").outputs["power"],
                                  "gate_result": out["gate_result"]}, {}, InProcessTrust(proj),
                           FixedAlphaLedger(proj))
        with self.assertRaises(Refused) as cm:
            Report().run(ctx, {})
        self.assertEqual(cm.exception.problems[0].rule, "AN-10")

    def test_records_of_the_wrong_model_make_the_gate_missing(self) -> None:
        proj = self.make()
        self.upto_registration(proj)

        def other_model(rec):
            rec["model_ref_sha256"] = "sha256:" + "2" * 64

        res = self.forged_gate(proj, mutate_base=other_model)
        self.assertEqual(res["status"], "MISSING")
        self.assertIn("model_hash_mismatch", res["status_reasons"])

    def test_n_items_must_match_the_registered_sealed_size(self) -> None:
        proj = self.make()
        self.upto_registration(proj)

        def short(rec):
            rec["records"] = rec["records"][:20]
            rec["n_items"] = 10

        p = self.refused_gate(proj, mutate_cand=short)
        self.assertIn("sealed set size", p.message)

    def test_a_malformed_row_counts_as_missing_instead_of_being_dropped(self) -> None:
        proj = self.make()
        self.upto_registration(proj)

        def bad_row(rec):
            rec["records"][0]["note"] = "x"          # a field outside {item_id, seed, score}

        res = self.forged_gate(proj, mutate_cand=bad_row)
        self.assertEqual(res["status"], "MISSING")
        self.assertIn("malformed_record", res["status_reasons"])


class ModelBindingTest(GateBase):
    """Review round 1, important 1: the arms are the registered base and a trained candidate."""

    def refused_eval(self, **attrs) -> str:
        for k, v in attrs.items():
            setattr(self, k, v)
        proj = self.make()
        self.upto_registration(proj)
        p = self.refused(proj, "evaluate_sealed", rule="AN-8")
        self.assertFalse((proj.root / "stages" / "evaluate_sealed").exists())
        return p.message

    def test_baseline_must_be_the_registered_base_model(self) -> None:
        self.assertIn("registered", self.refused_eval(baseline_src="train_x.model", candidate_src="choose_x.base"))

    def test_candidate_must_differ_from_the_baseline(self) -> None:
        self.assertIn("same model", self.refused_eval(candidate_src="choose_x.base"))

    def test_candidate_must_come_from_a_training_stage(self) -> None:
        self.assertIn("training", self.refused_eval(candidate_src="other_x.model", extra_stage=True))

    def test_check_contents_accepts_and_validates_the_base_model_field(self) -> None:
        from gatedtrain.registration import check_contents
        proj = self.make()
        self.upto_registration(proj)
        full = InProcessTrust(proj).registration(proj.read_artifact(self.reg_ref(proj))["hash"])
        self.assertIn("base_model", full)
        self.assertEqual(check_contents(full), [])
        without = {k: v for k, v in full.items() if k != "base_model"}
        self.assertEqual(check_contents(without), [])                   # registrations without it are unchanged
        self.assertEqual(full["registration_id"][:8], "ainative")
        self.assertNotIn("base", full["registration_id"].split("-", 1)[1])
        for bad in ({"model_ref_sha256": "sha256:xyz"}, {"model_ref_sha256": full["base_model"]["model_ref_sha256"],
                                                          "x": 1}, "sha256:" + "0" * 64, {}):
            probs = check_contents({**full, "base_model": bad})
            self.assertTrue(any(p.field.endswith("base_model") for p in probs), bad)

    def test_the_base_sha_is_read_from_the_field(self) -> None:
        full = {"base_model": {"model_ref_sha256": "sha256:" + "a" * 64}, "registration_id": "x-base-" + "b" * 64}
        self.assertEqual(trust_stages.registered_base(full), "sha256:" + "a" * 64)
        self.assertIsNone(trust_stages.registered_base({"registration_id": "x-base-" + "b" * 64}))

    def test_registration_binds_the_base_model(self) -> None:
        proj = self.make()
        self.upto_registration(proj)
        out = proj.read_artifact(self.reg_ref(proj))
        full = InProcessTrust(proj).registration(out["hash"])
        base = self.run_(proj, "choose_x").outputs["base"]
        self.assertEqual(trust_stages.registered_base(full), base.sha256)


def tunable_train(stage_name: str = "train_x") -> type:
    """A training stage whose answer rate comes from the decision (`rate`), so a rerun yields a new candidate."""

    class _Train(model_stage(stage_name, 1.0, True)):
        def run(self, ctx, decision):
            rec = {"schema": SCHEMA_MODEL_REF, "schema_version": 1, "generator": GENERATOR,
                   "rate": decision.get("rate", 1.0), "label": stage_name}
            return {"model": ctx.project.write_artifact(self.name, "model", rec, SCHEMA_MODEL_REF)}

    return _Train


class SealedOnceTest(GateBase):
    """Final review C1: one sealed evaluation per project. Scoring several candidates on the sealed set and gating
    the best one would select on the test data without spending alpha (design 05 §18.5: no repeated testing)."""

    def registry(self) -> dict:
        return {**super().registry(), "train_x": tunable_train()}

    def evaluate_twice(self, proj: Project, rates: tuple[float, float]):
        self.upto_registration(proj)
        self.run_(proj, "train_x", {"rate": rates[0], "rerun": True})
        first = self.run_(proj, "evaluate_sealed").outputs
        self.run_(proj, "train_x", {"rate": rates[1], "rerun": True})
        return first

    def test_a_second_candidate_is_refused(self) -> None:
        proj = self.make()
        first = self.evaluate_twice(proj, (0.6, 0.9))
        before = (proj.root / first["candidate_records"].path).read_bytes()
        with self.assertRaises(Refused) as cm:
            self.run_(proj, "evaluate_sealed")
        p = cm.exception.problems[0]
        self.assertEqual((p.rule, p.field), ("AN-8", "inputs.candidate_model"))
        self.assertIn("this project's sealed test is used", p.fix)
        self.assertIn("new sealed data", p.fix)
        self.assertNotIn("new project", p.fix)
        self.assertEqual((proj.root / first["candidate_records"].path).read_bytes(), before)   # nothing rescored
        self.assertEqual(proj.read_attempts()[-1]["event"], "refused")
        with self.assertRaises(Refused):                                   # and the gate cannot use stale records
            self.run_(proj, "gate")
        self.assertEqual(FixedAlphaLedger(proj).entries(), [])

    def test_the_claim_is_a_trust_record_bound_to_the_candidate_and_the_registration(self) -> None:
        proj = self.make()
        self.upto_registration(proj)
        self.run_(proj, "evaluate_sealed")
        claim = InProcessTrust(proj).sealed_evaluation()
        cand = self.run_(proj, "train_x").outputs["model"]
        self.assertEqual(claim["candidate_model_sha256"], cand.sha256)
        self.assertEqual(claim["registration_hash"], proj.read_artifact(self.reg_ref(proj))["hash"])
        self.assertTrue((proj.root / InProcessTrust.SEALED_EVALUATION).is_file())

    def test_a_second_scoring_of_the_same_candidate_is_refused_and_the_first_still_gates(self) -> None:
        """Re-review of C1: a model reference need not change when its weights do, so once the sealed test is
        claimed no new scoring runs, for the same reference either; the recorded scoring is still reused."""
        proj = self.make()
        self.upto_registration(proj)
        first = self.run_(proj, "evaluate_sealed").outputs
        with self.assertRaises(Refused) as cm:
            self.run_(proj, "evaluate_sealed", {"rerun": True, "again": 1})
        p = cm.exception.problems[0]
        self.assertEqual((p.rule, p.field), ("AN-8", "inputs.candidate_model"))
        self.assertIn("new sealed data", p.fix)
        self.assertEqual(proj.read_attempts()[-1]["event"], "refused")
        again = self.run_(proj, "evaluate_sealed")                          # the recorded scoring, not run again
        self.assertTrue(again.reused)
        self.assertEqual(again.outputs["candidate_records"].sha256, first["candidate_records"].sha256)
        self.assertEqual(self.gate(proj), "PASS")

    def gate(self, proj: Project) -> str:
        return proj.read_artifact(self.run_(proj, "gate").outputs["gate_result"])["status"]

    def test_back_to_the_evaluated_candidate_the_gate_runs_and_the_report_lists_every_attempt(self) -> None:
        proj = self.make()
        self.evaluate_twice(proj, (1.0, 0.9))
        with self.assertRaises(Refused):
            self.run_(proj, "evaluate_sealed")
        self.run_(proj, "train_x", {"rate": 1.0, "rerun": True})             # the evaluated candidate again
        self.run_(proj, "evaluate_sealed")                                  # done and current: a reuse
        self.assertEqual(self.gate(proj), "PASS")
        rep = proj.read_artifact(self.run_(proj, "report").outputs["report"])
        evals = [(a["event"]) for a in rep["attempts"] if a["stage"] == "evaluate_sealed"]
        self.assertEqual(evals, ["start", "done", "start", "refused"])
        md = (proj.root / "stages" / "report" / "report.md").read_text(encoding="utf-8")
        self.assertIn("evaluate_sealed attempt 2 refused", md)

    def test_the_claim_is_a_side_effect_so_a_later_refusal_fails_the_attempt(self) -> None:
        from gatedtrain.ainative.machine import status

        def boom(model_ref, item, *, sampler, seed):
            raise RuntimeError("pack bug")

        proj = self.make()
        self.upto_registration(proj)
        TEST_GENERATORS[GENERATOR] = boom
        trust_stages._reset_scenario_snapshot()
        with self.assertRaises(Refused):
            self.run_(proj, "evaluate_sealed")
        self.assertEqual(proj.read_attempts()[-1]["event"], "failed")       # the claim was made before scoring
        self.assertEqual(status(proj, registry=self.registry())["evaluate_sealed"]["status"], "failed")


MUTABLE_WEIGHTS: dict = {}


def mutable_ref_train() -> type:
    """A training stage whose model reference names a mutable checkpoint path: retraining overwrites the weights
    and leaves the reference, and so its sha256, unchanged (re-review probe test_probe_mutable_ref.py)."""

    class _Train(model_stage("train_x", 1.0, True)):
        def run(self, ctx, decision):
            MUTABLE_WEIGHTS["ckpt/latest"] = decision.get("rate", 1.0)
            rec = {"schema": SCHEMA_MODEL_REF, "schema_version": 1, "generator": GENERATOR, "path": "ckpt/latest"}
            return {"model": ctx.project.write_artifact(self.name, "model", rec, SCHEMA_MODEL_REF)}

    return _Train


class MutableReferenceTest(GateBase):
    def registry(self) -> dict:
        return {**super().registry(), "train_x": mutable_ref_train()}

    def setUp(self) -> None:
        super().setUp()
        MUTABLE_WEIGHTS.clear()

        def gen(model_ref, item, *, sampler, seed):
            if "path" in model_ref:
                rate = MUTABLE_WEIGHTS[model_ref["path"]]
                return LABELS[item["input"]] if _share(item["input"]) < rate * 100 else "wrong"
            return generate(model_ref, item, sampler=sampler, seed=seed)

        TEST_GENERATORS[GENERATOR] = gen
        trust_stages._reset_scenario_snapshot()

    def mean(self, proj: Project, out: dict) -> float:
        rec = proj.read_artifact(out["candidate_records"])
        return sum(r["score"] for r in rec["records"]) / len(rec["records"])

    def test_best_of_n_through_an_unchanged_reference_is_refused(self) -> None:
        proj = self.make()
        self.upto_registration(proj)
        self.run_(proj, "train_x", {"rate": 0.55, "rerun": True})
        first = self.run_(proj, "evaluate_sealed", {"rerun": True, "try": 0.55}).outputs
        self.run_(proj, "train_x", {"rate": 0.9, "rerun": True})            # new weights, same reference sha
        with self.assertRaises(Refused) as cm:
            self.run_(proj, "evaluate_sealed", {"rerun": True, "try": 0.9})
        self.assertEqual(cm.exception.problems[0].rule, "AN-8")
        res = proj.read_artifact(self.run_(proj, "gate").outputs["gate_result"])
        self.assertAlmostEqual(res["candidate_score"], self.mean(proj, first))   # the first scoring decides
        starts = [a for a in proj.read_attempts() if a["stage"] == "evaluate_sealed" and a["event"] == "start"]
        self.assertEqual(len(starts), 2)
        rep = proj.read_artifact(self.run_(proj, "report").outputs["report"])
        lim = next(x for x in rep["known_limits"] if x["id"] == "sealed_evaluations")
        self.assertEqual(lim["attempts"], 2)
        self.assertIn("one scoring", lim["text"])


if __name__ == "__main__":
    unittest.main()

