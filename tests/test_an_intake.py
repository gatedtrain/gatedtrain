"""AN-6 and AN-7: the trust stages `intake_seal` and `preregister` (tests/README.md section 12)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from gatedtrain import gates
from gatedtrain.ainative.machine import plan_stage, run_stage
from gatedtrain.ainative.pipeline import load_pipeline, pipeline_problems
from gatedtrain.ainative.project import Project, create_project
from gatedtrain.ainative.stage import TRUST_STAGES, Proposal, stage_registry
from gatedtrain.ainative.trust import InProcessTrust, SealedRef
from gatedtrain.ainative import scorers
from gatedtrain.ainative.trust_stages import SCHEMA_DEPLOYMENT, IntakeSeal, Preregister
from gatedtrain.errors import Refused
from gatedtrain.registration import check_contents, resolved_evaluator_hash

MiB = 1024 * 1024
SCHEMA_MODEL = "gatedtrain.ainative.model_ref"


class FakeTrain:
    name = "train_x"
    version = "1"
    inputs: dict = {}
    outputs = {"model": SCHEMA_MODEL}
    needs: frozenset = frozenset()
    params: frozenset = frozenset()
    trains = True

    def plan(self, ctx):
        return Proposal(self.name, "train", {}, {})

    def run(self, ctx, decision):
        return {"model": ctx.project.write_artifact(self.name, "model", {"w": 1}, SCHEMA_MODEL)}

    def check(self, ctx, outputs):
        return []


DEPLOY = {"schema": SCHEMA_DEPLOYMENT, "schema_version": 1,
          "sampler": {"name": "registered-sampler", "temperature": 0.9, "max_new_tokens": 32}, "eval_seeds": [11, 12]}


class ChooseDeployment:
    """Stands in for the pluggable stage (choose_task_and_base) that produces the deployment record."""
    name = "choose_x"
    version = "1"
    inputs: dict = {}
    outputs = {"deployment": SCHEMA_DEPLOYMENT}
    needs: frozenset = frozenset()
    params: frozenset = frozenset()
    trains = False
    record = DEPLOY

    def plan(self, ctx):
        return Proposal(self.name, "choose", {}, {})

    def run(self, ctx, decision):
        return {"deployment": ctx.project.write_artifact(self.name, "deployment", self.record, SCHEMA_DEPLOYMENT)}

    def check(self, ctx, outputs):
        return []


def rows(n: int, *, dups: int = 0, labels=("yes", "no")) -> list[dict]:
    out = [{"input": f"question-{i}-marker-{i * 7919}", "label": labels[i % len(labels)]} for i in range(n)]
    return out + [dict(out[i % len(out)]) for i in range(dups)]


def jsonl(items: list[dict]) -> bytes:
    return ("\n".join(json.dumps(x) for x in items) + "\n").encode("utf-8")


def plain_clause() -> dict:
    return {"type": "superiority", "margin": 0.0, "planning_effect": 0.10, "alpha": 0.05, "power": 0.8,
            "sigma_plan": "fixed"}


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def make(self, data: bytes, *, name: str = "proj") -> Project:
        d = self.tmp / f"{name}.jsonl"
        d.write_bytes(data)
        spec = {"name": "t", "training_stages": ["train_x"], "stages": [
            {"name": "intake_seal", "stage": "intake_seal"},
            {"name": "choose_x", "stage": "choose_x"},
            {"name": "preregister", "stage": "preregister",
             "inputs": {"sealed_summary": "intake_seal.sealed_summary", "deployment": "choose_x.deployment"}},
            {"name": "train_x", "stage": "train_x"},
            {"name": "evaluate_sealed", "stage": "evaluate_sealed",
             "inputs": {"registration": "preregister.registration", "baseline_model": "train_x.model",
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

    @staticmethod
    def registry() -> dict:
        return {**stage_registry(), "train_x": FakeTrain, "choose_x": ChooseDeployment}

    def run_(self, project: Project, ref: str, decision: dict | None = None):
        return run_stage(project, ref, decision or {}, registry=self.registry())

    def plan_(self, project: Project, ref: str) -> Proposal:
        return plan_stage(project, ref, registry=self.registry())

    @staticmethod
    def sealed_items(project: Project, manifest: str) -> list[dict]:
        return InProcessTrust(project).run_sealed(SealedRef(manifest), lambda item: item)

    def refused(self, project: Project, ref: str, decision: dict | None = None, rule: str = "AN-6"):
        with self.assertRaises(Refused) as cm:
            self.run_(project, ref, decision)
        self.assertEqual({p.rule for p in cm.exception.problems}, {rule})
        return cm.exception.problems[0]

    def nothing_written(self, project: Project) -> None:
        self.assertFalse((project.root / "stages").exists())
        self.assertFalse((project.root / "trust").exists())


class RegistryTest(unittest.TestCase):
    def test_real_classes_replace_the_placeholders_for_two_stages(self) -> None:
        reg = stage_registry()
        self.assertIs(reg["intake_seal"], IntakeSeal)
        self.assertIs(reg["preregister"], Preregister)
        for cls, needs in ((IntakeSeal, frozenset()), (Preregister, frozenset({"user"}))):
            self.assertEqual(cls.params, frozenset())
            self.assertFalse(cls.trains)
            self.assertEqual(cls.needs, needs)
        for name in ("evaluate_sealed", "gate", "report"):
            self.assertEqual(reg[name].name, name)             # real classes: tests/test_an_gate.py


class IntakeTest(Base):
    def test_dedup_happens_before_the_split(self) -> None:
        proj = self.make(jsonl(rows(400, dups=60)))
        out = self.run_(proj, "intake_seal").outputs
        train = proj.read_artifact(out["train_part"])
        summary = proj.read_artifact(out["sealed_summary"])
        sealed = self.sealed_items(proj, summary["manifest_sha256"])
        train_inputs = [r["input"] for r in train["rows"]]
        sealed_inputs = [r["input"] for r in sealed]
        self.assertEqual(len(train_inputs) + len(sealed_inputs), 400)     # every duplicate counted once
        self.assertEqual(len(set(train_inputs) | set(sealed_inputs)), 400)
        self.assertFalse(set(train_inputs) & set(sealed_inputs))          # Review Focus 1
        self.assertEqual(train["n_duplicates_removed"], 60)

    def test_duplicate_input_with_a_different_label_is_refused(self) -> None:
        data = rows(50) + [{"input": "question-3-marker-%d" % (3 * 7919), "label": "other"}]
        proj = self.make(jsonl(data))
        p = self.refused(proj, "intake_seal")
        self.assertIn("label", p.message)
        self.assertNotIn("marker", p.message)               # no item content in a refusal
        self.nothing_written(proj)

    def test_split_is_deterministic_and_follows_the_power_plan(self) -> None:
        data = jsonl(rows(400))
        a, b = self.make(data, name="a"), self.make(data, name="b")
        ra, rb = self.run_(a, "intake_seal").outputs, self.run_(b, "intake_seal").outputs
        self.assertEqual(ra["train_part"].sha256, rb["train_part"].sha256)
        self.assertEqual(ra["sealed_summary"].sha256, rb["sealed_summary"].sha256)
        power = a.read_artifact(ra["power"])
        want = gates.plan(plain_clause(), {"sigma": 0.5}, n_available=200)["n_required"]
        self.assertEqual(want, 155)
        self.assertEqual(power["n_needed"], want)
        self.assertEqual(power["n_sealed"], want)
        self.assertTrue(power["conclusive_expected"])
        self.assertAlmostEqual(power["mde"], gates.plan(plain_clause(), {"sigma": 0.5}, n_available=want)["mde"])
        self.assertEqual(a.read_artifact(ra["train_part"])["n"], 400 - want)
        other = self.make(jsonl(rows(401)), name="c")
        rc = self.run_(other, "intake_seal").outputs
        self.assertNotEqual(rc["sealed_summary"].sha256, ra["sealed_summary"].sha256)

    def test_sealed_part_is_capped_at_half_the_data(self) -> None:
        proj = self.make(jsonl(rows(200)))
        power = proj.read_artifact(self.run_(proj, "intake_seal", {"accept_inconclusive": True}).outputs["power"])
        self.assertEqual(power["n_sealed"], 100)
        self.assertEqual(power["n_needed"], 155)
        self.assertFalse(power["conclusive_expected"])

    def test_boundary_between_conclusive_and_not(self) -> None:
        ok = self.make(jsonl(rows(310)), name="ok")
        self.assertIsNone(self.plan_(ok, "intake_seal").needs_user)
        small = self.make(jsonl(rows(308)), name="small")
        self.assertEqual(self.plan_(small, "intake_seal").needs_user, "data_too_small")

    def test_refusals_each_have_their_own_message_and_write_nothing(self) -> None:
        cases = {
            "empty": b"",
            "no_label": jsonl(rows(5)[:4]) + b'{"input": "x"}\n',
            "bad_utf8": jsonl(rows(5)) + b'{"input": "\xff", "label": "a"}\n',
            "too_big": b"x" * (64 * MiB + 1),
            "not_json": jsonl(rows(5)) + b"not json\n",
            "extra_key": jsonl(rows(5)) + b'{"input": "x", "label": "a", "note": "n"}\n',
        }
        messages = {}
        for name, data in cases.items():
            proj = self.make(data, name=name)
            for call in (lambda: self.run_(proj, "intake_seal"), lambda: self.plan_(proj, "intake_seal")):
                with self.assertRaises(Refused) as cm:
                    call()
                self.assertEqual(cm.exception.problems[0].rule, "AN-6")
            messages[name] = cm.exception.problems[0].message
            self.nothing_written(proj)
        self.assertEqual(len(set(messages.values())), len(cases), messages)
        proj = self.make(jsonl(rows(4)) + b'{"input": "x", "label": null}\n', name="null_label")
        self.refused(proj, "intake_seal")                    # null reads as no label
        self.assertIn("empty", messages["empty"])
        self.assertIn("label", messages["no_label"])
        self.assertIn("UTF-8", messages["bad_utf8"])
        self.assertIn("64 MiB", messages["too_big"])

    def test_one_distinct_row_cannot_be_split(self) -> None:
        proj = self.make(jsonl(rows(1, dups=3)))
        self.refused(proj, "intake_seal")
        self.nothing_written(proj)

    def test_changed_data_file_is_refused(self) -> None:
        proj = self.make(jsonl(rows(40)))
        Path(proj.data["path"]).write_bytes(jsonl(rows(41)))
        p = self.refused(proj, "intake_seal")
        self.assertIn("changed", p.message)

    def test_too_few_rows_ask_the_user_and_refuse_without_acceptance(self) -> None:
        proj = self.make(jsonl(rows(40)))
        prop = self.plan_(proj, "intake_seal")
        self.assertEqual(prop.needs_user, "data_too_small")
        self.assertEqual(prop.estimate["n_needed"], 155)
        self.assertIn("INCONCLUSIVE", prop.summary)
        self.assertIn("270", prop.summary)                 # 2 * 155 - 40 more labelled examples
        self.nothing_written(proj)                         # plan has no side effects
        p = self.refused(proj, "intake_seal")
        self.assertIn("accept_inconclusive", p.field)
        self.refused(proj, "intake_seal", {"accept_inconclusive": "yes"})   # only true counts
        self.nothing_written(proj)
        out = self.run_(proj, "intake_seal", {"accept_inconclusive": True}).outputs
        power = proj.read_artifact(out["power"])
        self.assertFalse(power["conclusive_expected"])
        self.assertEqual(power["n_sealed"], 20)

    def test_no_stage_output_holds_sealed_item_text(self) -> None:
        proj = self.make(jsonl(rows(400, labels=("yes", "no", "maybe"))))
        out = self.run_(proj, "intake_seal").outputs
        summary = proj.read_artifact(out["sealed_summary"])
        sealed = self.sealed_items(proj, summary["manifest_sha256"])
        self.assertEqual(len(sealed), 155)
        blob = b""
        for path in sorted(proj.root.rglob("*")):
            if path.is_file() and "trust" not in path.relative_to(proj.root).parts:
                blob += path.read_bytes()
        self.assertIn(b"question-", blob)                  # the training part is there
        for item in sealed:
            self.assertNotIn(item["input"].encode(), blob)
        self.assertEqual(summary["n"], 155)
        self.assertEqual(sum(summary["label_counts"].values()) + summary["label_counts_other"], 155)

    def test_label_text_seen_only_in_the_sealed_part_stays_out_of_the_outputs(self) -> None:
        data = rows(400)
        for i, r in enumerate(data):
            r["label"] = f"secret-label-{i}"
        proj = self.make(jsonl(data))
        out = self.run_(proj, "intake_seal").outputs
        summary = proj.read_artifact(out["sealed_summary"])
        train_labels = {r["label"] for r in proj.read_artifact(out["train_part"])["rows"]}
        sealed = self.sealed_items(proj, summary["manifest_sha256"])
        raw = (proj.root / out["sealed_summary"].path).read_text() + (proj.root / out["power"].path).read_text()
        for item in sealed:
            self.assertIn(item["label"], {x["label"] for x in sealed})
            self.assertNotIn(item["label"], raw)
        self.assertFalse({k for k in summary["label_counts"]} - train_labels)
        self.assertEqual(summary["label_counts_other"], 155)

    def test_records_carry_schema_and_version(self) -> None:
        proj = self.make(jsonl(rows(400)))
        for name, ref in self.run_(proj, "intake_seal").outputs.items():
            obj = proj.read_artifact(ref)
            self.assertEqual(obj["schema"], ref.schema)
            self.assertEqual(obj["schema_version"], 1)
            self.assertTrue(ref.schema.startswith("gatedtrain.ainative."))

    def test_pipeline_params_are_refused(self) -> None:
        proj = self.make(jsonl(rows(10)))
        spec = load_pipeline(Path(proj.pipeline))
        s0 = spec.stages[0]
        bad = type(spec)(spec.name, [type(s0)(s0.name, s0.stage, {"mde": 0.5}, {}), *spec.stages[1:]],
                         spec.training_stages)
        problems = pipeline_problems(bad, self.registry())
        self.assertTrue(any(p.rule == "AN-4" and "mde" in p.field for p in problems))


class PreregisterTest(Base):
    def intake(self, n: int = 400, **kw) -> Project:
        proj = self.make(jsonl(rows(n)), **kw)
        self.run_(proj, "intake_seal")
        self.run_(proj, "choose_x")
        return proj

    def test_plan_asks_for_the_metric_in_one_plain_sentence(self) -> None:
        proj = self.intake()
        prop = self.plan_(proj, "preregister")
        self.assertEqual(prop.needs_user, "confirm_metric")
        self.assertEqual(prop.choices["scorer"], "exact_match")
        self.assertEqual(prop.choices["direction"], "higher_is_better")
        s = prop.summary
        self.assertNotIn("\n", s)
        self.assertEqual(s.count(". "), 0)
        self.assertTrue(s.endswith("."))
        for word in ("exact", "beat", "points"):
            self.assertIn(word, s)

    def test_run_refuses_until_the_metric_is_confirmed(self) -> None:
        proj = self.intake()
        for decision in ({}, {"metric_confirmed": False}, {"metric_confirmed": "yes"}, {"metric_confirmed": 1}):
            p = self.refused(proj, "preregister", decision, rule="AN-7")
            self.assertIn("metric_confirmed", p.field)
        self.assertFalse((proj.root / "trust" / "registrations").exists())
        self.assertFalse((proj.root / "stages" / "preregister").exists())

    def test_registration_passes_check_contents_and_has_the_fields(self) -> None:
        proj = self.intake()
        out = self.run_(proj, "preregister", {"metric_confirmed": True}).outputs
        reg = proj.read_artifact(out["registration"])
        for key in ("hash", "scorer", "direction", "margin", "alpha", "item_set_hash", "post_hoc"):
            self.assertIn(key, reg)
        self.assertEqual((reg["scorer"], reg["direction"], reg["margin"], reg["alpha"], reg["post_hoc"]),
                         ("exact_match", "higher_is_better", 0.0, 0.05, False))
        summary = proj.read_artifact(self.run_(proj, "intake_seal").outputs["sealed_summary"])
        self.assertEqual(reg["item_set_hash"], summary["manifest_sha256"])
        full = InProcessTrust(proj).registration(reg["hash"])
        self.assertEqual(check_contents(full), [])
        self.assertEqual(full["eval_distribution"]["item_set_hash"], reg["item_set_hash"])
        self.assertEqual(full["clause"]["margin"], 0.0)
        self.assertEqual(full["clause"]["alpha"], 0.05)
        self.assertEqual(full["sigma_pilot"]["sigma"], 0.5)
        self.assertNotIn("post_hoc", full)
        self.assertEqual(full["sampler"], DEPLOY["sampler"])             # the registered sampler is the deployment's
        self.assertEqual(full["clause"]["eval_seeds"], [11, 12])
        self.assertEqual(full["clause"]["samples_per_item"], 2)
        ev = full["evaluators"][0]
        self.assertEqual(ev["id"], scorers.EVALUATOR_ID)
        self.assertEqual(ev["manifest_hash"], resolved_evaluator_hash({"id": scorers.EVALUATOR_ID}))
        self.assertEqual(ev["manifest_hash"], scorers.manifest_hash())
        self.assertEqual(check_contents(full, evaluator_id=scorers.EVALUATOR_ID), [])

    def test_user_may_change_the_margin_and_the_registration_still_passes(self) -> None:
        proj = self.intake()
        out = self.run_(proj, "preregister", {"metric_confirmed": True, "margin": 0.05}).outputs
        reg = proj.read_artifact(out["registration"])
        full = InProcessTrust(proj).registration(reg["hash"])
        self.assertEqual(check_contents(full), [])
        self.assertEqual(reg["margin"], 0.05)
        self.assertAlmostEqual(full["clause"]["planning_effect"], 0.15)

    def test_unknown_scorer_or_bad_margin_is_refused(self) -> None:
        proj = self.intake()
        for extra in ({"scorer": "vibes"}, {"margin": -0.1}, {"margin": "big"}, {"margin": 1.5},
                      {"direction": "lower_is_better"}):
            p = self.refused(proj, "preregister", {"metric_confirmed": True, **extra}, rule="AN-7")
            self.assertTrue(p.fix)

    def test_a_changed_registration_before_training_is_not_post_hoc(self) -> None:
        proj = self.intake()
        first = proj.read_artifact(self.run_(proj, "preregister", {"metric_confirmed": True}).outputs["registration"])
        second = proj.read_artifact(self.run_(proj, "preregister", {"metric_confirmed": True, "margin": 0.02,
                                                                    "rerun": True}).outputs["registration"])
        self.assertNotEqual(first["hash"], second["hash"])
        self.assertFalse(second["post_hoc"])

    def test_different_values_after_training_mark_the_project_post_hoc(self) -> None:
        proj = self.intake()
        first = proj.read_artifact(self.run_(proj, "preregister", {"metric_confirmed": True}).outputs["registration"])
        self.run_(proj, "train_x")
        second = proj.read_artifact(self.run_(proj, "preregister", {"metric_confirmed": True, "margin": 0.02,
                                                                    "rerun": True}).outputs["registration"])
        self.assertTrue(second["post_hoc"])
        self.assertEqual(second["amends"], first["hash"])
        full = InProcessTrust(proj).registration(second["hash"])
        self.assertTrue(full["post_hoc"])
        self.assertEqual(full["amends"], first["hash"])
        self.assertEqual(check_contents(full), [])

    def test_same_values_after_training_stay_clean_and_post_hoc_sticks(self) -> None:
        proj = self.intake()
        self.run_(proj, "preregister", {"metric_confirmed": True})
        self.run_(proj, "train_x")
        same = proj.read_artifact(self.run_(proj, "preregister", {"metric_confirmed": True, "note": "again",
                                                                  "rerun": True}).outputs["registration"])
        self.assertFalse(same["post_hoc"])
        changed = proj.read_artifact(self.run_(proj, "preregister", {"metric_confirmed": True, "margin": 0.03,
                                                                     "rerun": True}).outputs["registration"])
        self.assertTrue(changed["post_hoc"])
        back = proj.read_artifact(self.run_(proj, "preregister", {"metric_confirmed": True, "margin": 0.03, "note": "back",
                                                                  "rerun": True}).outputs["registration"])
        self.assertTrue(back["post_hoc"])                   # going back to the old values does not undo it

    def test_registration_follows_the_deployment_record(self) -> None:
        proj = self.make(jsonl(rows(400)), name="other")
        self.run_(proj, "intake_seal")
        other = {**DEPLOY, "sampler": {"name": "s2", "temperature": 0.5, "max_new_tokens": 8}, "eval_seeds": [3]}
        reg = {**self.registry(), "choose_x": type("C", (ChooseDeployment,), {"record": other})}
        run_stage(proj, "choose_x", {}, registry=reg)
        out = run_stage(proj, "preregister", {"metric_confirmed": True}, registry=reg).outputs
        full = InProcessTrust(proj).registration(proj.read_artifact(out["registration"])["hash"])
        self.assertEqual(full["sampler"], other["sampler"])
        self.assertEqual(full["clause"]["eval_seeds"], [3])

    def test_bad_deployment_is_refused(self) -> None:
        bads = [{**DEPLOY, "sampler": {"name": "s", "temperature": 0, "max_new_tokens": 8}},
                {**DEPLOY, "eval_seeds": []}, {**DEPLOY, "eval_seeds": [1, 1]},
                {k: v for k, v in DEPLOY.items() if k != "sampler"}]
        for i, rec in enumerate(bads):
            proj = self.make(jsonl(rows(400)), name=f"bad{i}")
            self.run_(proj, "intake_seal")
            reg = {**self.registry(), "choose_x": type("C", (ChooseDeployment,), {"record": rec})}
            run_stage(proj, "choose_x", {}, registry=reg)
            with self.assertRaises(Refused):
                run_stage(proj, "preregister", {"metric_confirmed": True}, registry=reg)
            self.assertFalse((proj.root / "trust" / "registrations").exists())

    def test_deployment_input_is_required(self) -> None:
        self.assertEqual(Preregister.inputs, {"sealed_summary": "gatedtrain.ainative.sealed_summary",
                                              "deployment": SCHEMA_DEPLOYMENT})

    def test_training_start_is_logged_and_read_from_the_log_not_the_spec(self) -> None:
        proj = self.intake()
        self.run_(proj, "preregister", {"metric_confirmed": True})
        self.run_(proj, "train_x")
        starts = {r["stage"]: r["trains"] for r in proj.read_attempts() if r["event"] == "start"}
        self.assertEqual(starts, {"intake_seal": False, "choose_x": False, "preregister": False, "train_x": True})
        # Renaming the training stage in the pipeline spec must not hide that it ran.
        path = Path(proj.pipeline)
        spec = yaml.safe_load(path.read_text())
        spec["training_stages"] = ["train_y"]
        for st in spec["stages"]:
            if st["name"] == "train_x":
                st["name"] = "train_y"
            st["inputs"] = {k: v.replace("train_x.", "train_y.") for k, v in st.get("inputs", {}).items()}
        path.write_text(yaml.safe_dump(spec))
        out = self.run_(proj, "preregister", {"metric_confirmed": True, "margin": 0.3, "rerun": True}).outputs
        self.assertTrue(proj.read_artifact(out["registration"])["post_hoc"])

    def test_post_hoc_stickiness_reads_the_trust_store(self) -> None:
        proj = self.intake()
        self.run_(proj, "preregister", {"metric_confirmed": True})
        self.run_(proj, "train_x")
        second = proj.read_artifact(self.run_(proj, "preregister", {"metric_confirmed": True, "margin": 0.03,
                                                                    "rerun": True}).outputs["registration"])
        # An edited output record cannot clear the mark.
        edited = dict(second, post_hoc=False)
        proj.write_artifact("preregister", "registration", edited, "gatedtrain.ainative.registration")
        with self.assertRaises(Refused):                    # the hash chain refuses the edited record
            self.run_(proj, "preregister", {"metric_confirmed": True, "margin": 0.03, "note": "x", "rerun": True})

    def test_check_flags_a_registration_that_no_longer_matches(self) -> None:
        proj = self.intake()
        out = self.run_(proj, "preregister", {"metric_confirmed": True}).outputs
        from gatedtrain.ainative.stage import StageContext
        from gatedtrain.ainative.trust import FixedAlphaLedger
        inputs = {"sealed_summary": self.run_(proj, "intake_seal").outputs["sealed_summary"],
                  "deployment": self.run_(proj, "choose_x").outputs["deployment"]}
        ctx = StageContext(proj, inputs, {}, InProcessTrust(proj), FixedAlphaLedger(proj))
        self.assertEqual(Preregister().check(ctx, out), [])
        reg = proj.read_artifact(out["registration"])
        bad = dict(reg, item_set_hash="sha256:" + "0" * 64)
        ref = proj.write_artifact("preregister", "registration", bad, out["registration"].schema)
        self.assertTrue(Preregister().check(ctx, {"registration": ref}))

    def test_check_flags_an_unresolvable_evaluator_hash(self) -> None:
        proj = self.intake()
        out = self.run_(proj, "preregister", {"metric_confirmed": True}).outputs
        from gatedtrain.ainative.stage import StageContext
        from gatedtrain.ainative.trust import FixedAlphaLedger
        trust = InProcessTrust(proj)
        full = trust.registration(proj.read_artifact(out["registration"])["hash"])
        full["evaluators"][0]["manifest_hash"] = "sha256:" + "1" * 64
        digest = trust.register(full)
        out_rec = dict(proj.read_artifact(out["registration"]), hash=digest)
        ref = proj.write_artifact("preregister", "registration", out_rec, "gatedtrain.ainative.registration")
        ins = {"sealed_summary": self.run_(proj, "intake_seal").outputs["sealed_summary"],
               "deployment": self.run_(proj, "choose_x").outputs["deployment"]}
        ctx = StageContext(proj, ins, {}, trust, FixedAlphaLedger(proj))
        self.assertTrue(any("evaluator" in p.field for p in Preregister().check(ctx, {"registration": ref})))


class ScorerTest(unittest.TestCase):
    def test_exact_match(self) -> None:
        em = scorers.exact_match
        self.assertEqual(em("a", "a"), 1.0)
        self.assertEqual(em("a", "b"), 0.0)
        self.assertEqual(em({"x": 1, "y": [2]}, {"y": [2], "x": 1}), 1.0)
        self.assertEqual(em(1, "1"), 0.0)
        self.assertEqual(em(float("nan"), 1), 0.0)

    def test_evaluator_id_resolves_to_a_hash_bound_to_the_scorer_source(self) -> None:
        h = resolved_evaluator_hash({"id": scorers.EVALUATOR_ID})
        self.assertEqual(h, scorers.manifest_hash())
        self.assertRegex(h, r"^sha256:[0-9a-f]{64}$")
        from gatedtrain.canonical import file_sha256
        self.assertIn(file_sha256(Path(scorers.__file__)), scorers.manifest_document()["code"].values())


if __name__ == "__main__":
    unittest.main()
