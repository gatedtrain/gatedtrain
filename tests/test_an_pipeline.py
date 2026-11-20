"""AN-3 and AN-4: the stage registry and the pipeline spec validator (tests/README.md section 12).

The pipelines and stage classes here are test-local; the five trust stages are reserved names that a plugin cannot
take, and a pipeline spec is refused when a trust stage is missing, duplicated, replaced or out of order.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from gatedtrain.ainative import stage as stage_mod
from gatedtrain.ainative.pipeline import PipelineSpec, StageRef, load_pipeline, pipeline_problems
from gatedtrain.ainative.stage import TRUST_STAGES, Proposal, StageContext, stage_registry
from gatedtrain.errors import Problem, Refused

S = "gatedtrain.ainative."


def _stub(name, inputs=None, outputs=None, needs=(), params=(), trains=False):
    return type(name, (), {
        "name": name, "version": "1", "inputs": dict(inputs or {}), "outputs": dict(outputs or {}),
        "needs": frozenset(needs), "params": frozenset(params), "trains": trains})


REGISTRY = {
    "intake_seal": _stub("intake_seal", outputs={"views": S + "views"}),
    "preregister": _stub("preregister", {"views": S + "views"}, {"prereg": S + "prereg"}),
    "train": _stub("train", {"prereg": S + "prereg"}, {"model": S + "model"}, needs={"gpu"}, params={"steps"}),
    "tune": _stub("tune", {"prereg": S + "prereg"}, {"model": S + "model"}),   # trains only by name
    "trainer": _stub("trainer", {"prereg": S + "prereg"}, {"model": S + "model"}, trains=True),  # no gpu need
    "prep": _stub("prep", {"views": S + "views"}, {"prepared": S + "views"}),
    "note": _stub("note", outputs={"note": S + "note"}),
    "evaluate_sealed": _stub("evaluate_sealed", {"model": S + "model", "prereg": S + "prereg"},
                             {"records": S + "records"}),
    "gate": _stub("gate", {"records": S + "records", "prereg": S + "prereg"}, {"verdict": S + "verdict"}),
    "report": _stub("report", {"verdict": S + "verdict"}, {"report": S + "report"}),
}

INTAKE = {"name": "intake_seal", "stage": "intake_seal"}
PREREG = {"name": "preregister", "stage": "preregister", "inputs": {"views": "intake_seal.views"}}
TRAIN = {"name": "train", "stage": "train", "params": {"steps": 5}, "inputs": {"prereg": "preregister.prereg"}}
EVAL = {"name": "evaluate_sealed", "stage": "evaluate_sealed",
        "inputs": {"model": "train.model", "prereg": "preregister.prereg"}}
GATE = {"name": "gate", "stage": "gate",
        "inputs": {"records": "evaluate_sealed.records", "prereg": "preregister.prereg"}}
REPORT = {"name": "report", "stage": "report", "inputs": {"verdict": "gate.verdict"}}
GOOD = [INTAKE, PREREG, TRAIN, EVAL, GATE, REPORT]


class PipelineBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def spec(self, stages, **extra) -> PipelineSpec:
        path = self.tmp / "pipeline.yaml"
        path.write_text(yaml.safe_dump({"name": "t", "stages": stages, **extra}))
        return load_pipeline(path)

    def problems(self, stages, **extra) -> list[Problem]:
        return pipeline_problems(self.spec(stages, **extra), REGISTRY)

    def assertRefused(self, problems, field, text="") -> Problem:
        hits = [p for p in problems if p.field == field and text in p.message]
        self.assertTrue(hits, f"no problem for {field!r} containing {text!r}; got {[p.line() for p in problems]}")
        self.assertEqual(hits[0].rule, "AN-4")
        self.assertTrue(hits[0].fix)
        return hits[0]


class PipelineValidatorTest(PipelineBase):
    def test_valid_pipeline(self):
        spec = self.spec(GOOD)
        self.assertEqual([s.name for s in spec.stages], [s["name"] for s in GOOD])
        self.assertEqual(pipeline_problems(spec, REGISTRY), [])

    def test_valid_pipeline_with_pluggable_stages(self):
        prep = {"name": "prep", "stage": "prep", "inputs": {"views": "intake_seal.views"}}
        self.assertEqual(self.problems([INTAKE, PREREG, prep, TRAIN, EVAL, GATE, REPORT]), [])

    def test_missing_trust_stage_refused(self):
        for t in TRUST_STAGES:
            # drop dangling wiring (on copies) so only the missing stage is under test
            stages = [dict(s, inputs={k: v for k, v in s.get("inputs", {}).items() if v.split(".")[0] != t})
                      for s in GOOD if s["stage"] != t]
            with self.subTest(dropped=t):
                self.assertRefused(self.problems(stages), "stages", f"trust stage {t!r} is missing")

    def test_intake_seal_not_first(self):
        prep = {"name": "note", "stage": "note"}
        p = self.problems([prep, INTAKE, PREREG, TRAIN, EVAL, GATE, REPORT])
        self.assertRefused(p, "stages[1].stage", "intake_seal must be the first")

    def test_preregister_after_training_stage(self):
        p = self.problems([INTAKE, TRAIN, PREREG, EVAL, GATE, REPORT])
        self.assertRefused(p, "stages[2].stage", "preregister must come before every training stage")

    def test_preregister_after_name_listed_training_stage(self):
        tune = {"name": "tune", "stage": "tune", "inputs": {"prereg": "preregister.prereg"}}
        p = self.problems([INTAKE, tune, PREREG, EVAL, GATE, REPORT], training_stages=["tune"])
        self.assertRefused(p, "stages[2].stage", "preregister must come before every training stage")

    def test_trains_flag_stage_before_preregister(self):
        trainer = {"name": "trainer", "stage": "trainer", "inputs": {"prereg": "preregister.prereg"}}
        p = self.problems([INTAKE, trainer, PREREG, EVAL, GATE, REPORT])
        self.assertRefused(p, "stages[2].stage", "preregister must come before every training stage")

    def test_training_stages_entries_validated(self):
        p = self.problems(GOOD, training_stages=["ghost"])
        self.assertRefused(p, "training_stages[0]", "matches no stage")
        p = self.problems(GOOD, training_stages=["gate"])
        self.assertRefused(p, "training_stages[0]", "trust stage")
        self.assertEqual(self.problems(GOOD, training_stages=["train"]), [])

    def test_evaluate_sealed_before_last_training_stage(self):
        p = self.problems([INTAKE, PREREG, EVAL, TRAIN, GATE, REPORT])
        self.assertRefused(p, "stages[2].stage", "evaluate_sealed must come after the last training stage")

    def test_gate_before_last_training_stage(self):
        p = self.problems([INTAKE, PREREG, GATE, EVAL, TRAIN, REPORT])
        self.assertRefused(p, "stages[2].stage", "gate must come after the last training stage")

    def test_gate_before_evaluate_sealed(self):
        p = self.problems([INTAKE, PREREG, TRAIN, GATE, EVAL, REPORT])
        self.assertRefused(p, "stages[3].stage", "gate must come after evaluate_sealed")

    def test_report_not_last(self):
        p = self.problems([INTAKE, PREREG, TRAIN, EVAL, REPORT, GATE])
        self.assertRefused(p, "stages[4].stage", "report must be the last stage")

    def test_stage_between_evaluate_sealed_and_report(self):
        note = {"name": "note", "stage": "note"}
        p = self.problems([INTAKE, PREREG, TRAIN, EVAL, note, GATE, REPORT])
        self.assertRefused(p, "stages[4].stage", "runs after evaluate_sealed")
        # a trust stage (gate) between them is fine
        self.assertEqual(self.problems(GOOD), [])

    def test_duplicate_trust_stage(self):
        again = dict(INTAKE, name="intake_again")
        p = self.problems([INTAKE, again, PREREG, TRAIN, EVAL, GATE, REPORT])
        self.assertRefused(p, "stages[1].stage", "appears more than once")

    def test_trust_name_with_other_stage_type(self):
        fake = {"name": "gate", "stage": "note"}
        p = self.problems([INTAKE, PREREG, TRAIN, EVAL, fake, REPORT])
        self.assertRefused(p, "stages[4].name", "reserved for the trust stage")

    def test_input_from_later_stage(self):
        train = dict(TRAIN, inputs={"prereg": "evaluate_sealed.records"})
        p = self.problems([INTAKE, PREREG, train, EVAL, GATE, REPORT])
        self.assertRefused(p, "stages[2].inputs.prereg", "does not run before")

    def test_input_from_missing_stage_or_output(self):
        p = self.problems([INTAKE, dict(PREREG, inputs={"views": "ghost.views"}), TRAIN, EVAL, GATE, REPORT])
        self.assertRefused(p, "stages[1].inputs.views", "not a stage of this pipeline")
        p = self.problems([INTAKE, dict(PREREG, inputs={"views": "intake_seal.nope"}), TRAIN, EVAL, GATE, REPORT])
        self.assertRefused(p, "stages[1].inputs.views", "has no output 'nope'")

    def test_input_schema_mismatch(self):
        train = dict(TRAIN, inputs={"prereg": "intake_seal.views"})
        p = self.problems([INTAKE, PREREG, train, EVAL, GATE, REPORT])
        self.assertRefused(p, "stages[2].inputs.prereg", "expects schema")

    def test_unwired_and_undeclared_input(self):
        p = self.problems([INTAKE, PREREG, dict(TRAIN, inputs={}), EVAL, GATE, REPORT])
        self.assertRefused(p, "stages[2].inputs.prereg", "is not wired")
        p = self.problems([INTAKE, PREREG, dict(TRAIN, inputs={"prereg": "preregister.prereg",
                                                                "x": "intake_seal.views"}), EVAL, GATE, REPORT])
        self.assertRefused(p, "stages[2].inputs.x", "declares no input")

    def test_unknown_stage_and_unknown_param(self):
        p = self.problems([INTAKE, PREREG, dict(TRAIN, stage="mystery"), EVAL, GATE, REPORT])
        self.assertRefused(p, "stages[2].stage", "unknown stage type 'mystery'")
        p = self.problems([INTAKE, PREREG, dict(TRAIN, params={"steps": 1, "lr": 2}), EVAL, GATE, REPORT])
        self.assertRefused(p, "stages[2].params.lr", "does not accept the param 'lr'")
        self.assertFalse([x for x in p if x.field == "stages[2].params.steps"])

    def test_duplicate_stage_name(self):
        p = self.problems([INTAKE, PREREG, TRAIN, dict(TRAIN), EVAL, GATE, REPORT])
        self.assertRefused(p, "stages[3].name", "used twice")

    def test_malformed_spec_refused(self):
        path = self.tmp / "bad.yaml"
        path.write_text("name: x\nstages: nope\n")
        with self.assertRaises(Refused) as cm:
            load_pipeline(path)
        self.assertEqual(cm.exception.problems[0].field, "stages")
        with self.assertRaises(Refused):
            load_pipeline(self.tmp / "missing.yaml")


class StageRegistryTest(unittest.TestCase):
    def test_trust_stages_are_reserved_builtins(self):
        with mock.patch.object(stage_mod, "_plugin_entry_points", return_value=[]):
            reg = stage_registry()
        self.assertEqual(set(TRUST_STAGES) - set(reg), set())
        for name in TRUST_STAGES:
            self.assertEqual(reg[name].name, name)

    def test_plugin_registered(self):
        ep = SimpleNamespace(name="mine", value="x:Y", load=lambda: REGISTRY["note"])
        with mock.patch.object(stage_mod, "_plugin_entry_points", return_value=[ep]):
            self.assertIs(stage_registry()["mine"], REGISTRY["note"])

    def test_plugin_named_like_trust_stage(self):
        for name in TRUST_STAGES:
            ep = SimpleNamespace(name=name, value="evil:Gate", load=lambda: self.fail("must not load"))
            with self.subTest(name=name), mock.patch.object(stage_mod, "_plugin_entry_points", return_value=[ep]):
                with self.assertRaises(Refused) as cm:
                    stage_registry()
                p = cm.exception.problems[0]
                self.assertEqual((p.rule, p.field), ("AN-3", f"entry point {name}"))
                self.assertTrue(p.fix)

    def _refused_plugin(self, cls):
        ep = SimpleNamespace(name="mine", value="x:Y", load=lambda: cls)
        with mock.patch.object(stage_mod, "_plugin_entry_points", return_value=[ep]):
            with self.assertRaises(Refused) as cm:
                stage_registry()
        p = cm.exception.problems[0]
        self.assertEqual((p.rule, p.field), ("AN-3", "entry point mine"))
        self.assertTrue(p.fix)
        return p

    def test_plugin_attribute_validation(self):
        def without(attr):
            return type("P", (), {k: v for k, v in vars(REGISTRY["note"]).items()
                                  if k != attr and not k.startswith("__")})
        for attr in ("name", "version", "inputs", "outputs", "needs", "params", "trains"):
            with self.subTest(missing=attr):
                self.assertIn(attr, self._refused_plugin(without(attr)).message)
        bad = {"needs": frozenset({"GPU"}), "trains": "yes", "name": 3, "version": 1, "inputs": {"a": 1},
               "outputs": [], "params": {"a"}}
        for attr, value in bad.items():
            with self.subTest(bad=attr):
                cls = type("P", (REGISTRY["note"],), {attr: value})
                self.assertIn(attr, self._refused_plugin(cls).message)
        cls = type("P", (REGISTRY["note"],), {"needs": frozenset({"cuda"})})
        self.assertIn("cuda", self._refused_plugin(cls).message)
        cls = type("P", (REGISTRY["note"],), {"needs": ["gpu"]})
        self.assertIn("needs", self._refused_plugin(cls).message)

    def test_plugin_load_failure(self):
        def boom():
            raise ImportError("nope")
        ep = SimpleNamespace(name="mine", value="x:Y", load=boom)
        with mock.patch.object(stage_mod, "_plugin_entry_points", return_value=[ep]):
            with self.assertRaises(Refused) as cm:
                stage_registry()
        self.assertEqual((cm.exception.problems[0].rule, cm.exception.problems[0].field), ("AN-3", "entry point mine"))
        self.assertIn("nope", cm.exception.problems[0].message)

    def test_builtins_do_not_train(self):
        with mock.patch.object(stage_mod, "_plugin_entry_points", return_value=[]):
            reg = stage_registry()
        self.assertTrue(all(reg[n].trains is False for n in TRUST_STAGES))

    def test_proposal_and_context_shapes(self):
        prop = Proposal("train", "summary", {"a": 1}, {"gpu_h": 0}, None)
        self.assertIsNone(prop.needs_user)
        ctx = StageContext(project=None, inputs={}, params={}, trust=None, ledger=None)
        self.assertEqual(ctx.params, {})


if __name__ == "__main__":
    unittest.main()
