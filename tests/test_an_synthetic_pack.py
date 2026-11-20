"""Task 8: the synthetic scenario pack runs the AI-native stage machine end to end on the CPU."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from gatedtrain.ainative import scorers, stage as stage_mod, trust_stages
from gatedtrain.ainative.machine import plan_stage, run_stage
from gatedtrain.ainative.pipeline import load_pipeline, pipeline_problems
from gatedtrain.ainative.project import Project, create_project
from gatedtrain.ainative.stage import StageContext, stage_registry
from gatedtrain.ainative.trust import FixedAlphaLedger, InProcessTrust
from gatedtrain.errors import Refused
from gatedtrain.testing import synthetic_pack as pack

N_ROWS = 200
STAGES = ("profile_data", "choose_task_and_base", "build_train_set", "write_recipe", "smoke", "train")
DECISIONS = {"intake_seal": {"accept_inconclusive": True}, "preregister": {"metric_confirmed": True}}


def fixture_rows(n: int = N_ROWS) -> list[dict]:
    """`<token>-<i>` inputs (20 tokens, 5 labels); the label follows the token except for one occurrence in four,
    which carries the label "odd", so a model that learned the task still misses some items."""
    return [{"input": f"tok{i % 20}-{i}", "label": "odd" if (i // 20) % 4 == 3 else f"L{(i % 20) % 5}"}
            for i in range(n)]


def jsonl(rows: list[dict]) -> bytes:
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode()


class _StageEP:
    def __init__(self, name: str, cls: type) -> None:
        self.name, self.value, self._cls = name, f"gatedtrain.testing.synthetic_pack:{cls.__name__}", cls

    def load(self) -> type:
        return self._cls


class _ScenarioEP:
    name = "synthetic"
    value = "gatedtrain.testing.synthetic_pack:SCENARIO"

    def load(self):
        return pack.SCENARIO


def inject_pack(case: unittest.TestCase) -> None:
    """Make the pack visible to the registry and to evaluate_sealed without any installed entry point."""
    eps = [_StageEP(c.name, c) for c in (pack.ProfileData, pack.ChooseTaskAndBase, pack.BuildTrainSet,
                                         pack.WriteRecipe, pack.Smoke, pack.Train)]
    for mod, attr, value in ((stage_mod, "_plugin_entry_points", lambda: list(eps)),
                             (trust_stages, "_scenario_entry_points", lambda: [_ScenarioEP()])):
        real = getattr(mod, attr)
        setattr(mod, attr, value)
        case.addCleanup(setattr, mod, attr, real)
    trust_stages._reset_scenario_snapshot()
    case.addCleanup(trust_stages._reset_scenario_snapshot)


class PackBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        inject_pack(self)

    def make(self, *, noise: float = 0.0, rows: list[dict] | None = None, name: str = "proj") -> Project:
        data = self.tmp / f"{name}.jsonl"
        data.write_bytes(jsonl(rows if rows is not None else fixture_rows()))
        raw = yaml.safe_load(pack.SCENARIO.pipeline.read_text())
        next(s for s in raw["stages"] if s["name"] == "write_recipe")["params"]["noise"] = noise
        (self.tmp / f"{name}.yaml").write_text(yaml.safe_dump(raw))
        return create_project(self.tmp / name, goal="lookup", data=data, pipeline=str(self.tmp / f"{name}.yaml"),
                              budget={"gpu_hours": 1, "llm_usd": 1})

    def run_(self, proj: Project, ref: str):
        return run_stage(proj, ref, dict(DECISIONS.get(ref, {})), registry=stage_registry())

    def run_all(self, proj: Project, upto: str = "report") -> dict:
        out = {}
        for s in yaml.safe_load(Path(proj.pipeline).read_text())["stages"]:
            out[s["name"]] = self.run_(proj, s["name"]).outputs
            if s["name"] == upto:
                break
        return out


class RegistrationTest(PackBase):
    def test_the_pack_is_found_through_the_injected_entry_points(self) -> None:
        reg = stage_registry()
        for name in STAGES:
            self.assertIn(name, reg)
        self.assertIs(reg["train"], pack.Train)
        self.assertEqual(trust_stages.scenario_pack_of("synthetic.lookup"), "synthetic")
        self.assertIs(trust_stages.scenario_generators()["synthetic.lookup"], pack.generate)

    def test_the_scenario_is_immutable(self) -> None:
        with self.assertRaises(TypeError):
            pack.SCENARIO.generators["synthetic.lookup"] = lambda *a, **k: None
        with self.assertRaises(Exception):
            pack.SCENARIO.generators = {}

    def test_train_declares_gpu_and_trains(self) -> None:
        self.assertEqual(pack.Train.needs, frozenset({"gpu"}))
        self.assertTrue(pack.Train.trains)
        self.assertEqual(pack.Smoke.needs, frozenset())

    def test_the_registration_binds_the_pack_and_generator(self) -> None:
        proj = self.make()
        out = self.run_all(proj, "preregister")
        full = InProcessTrust(proj).registration(proj.read_artifact(out["preregister"]["registration"])["hash"])
        self.assertEqual(trust_stages.registered_generator(full), ("synthetic", "synthetic.lookup"))

    def test_mutating_the_pack_does_not_change_the_evaluation(self) -> None:
        proj = self.make()
        self.run_all(proj, "train")
        self.assertIn("synthetic.lookup", trust_stages.scenario_generators())     # snapshot is taken
        with self.assertRaises(TypeError):
            pack.SCENARIO.generators["synthetic.lookup"] = lambda *a, **k: None
        out = self.run_(proj, "evaluate_sealed").outputs
        self.assertEqual(proj.read_artifact(self.run_(proj, "gate").outputs["gate_result"])["status"], "PASS")
        self.assertEqual(len(proj.read_artifact(out["candidate_records"])["records"]) % 2, 0)

    def test_the_pipeline_passes_the_validator(self) -> None:
        spec = load_pipeline(pack.SCENARIO.pipeline)
        self.assertEqual(pipeline_problems(spec, stage_registry()), [])
        self.assertEqual([s.stage for s in spec.stages if s.stage in stage_mod.TRUST_STAGES],
                         list(stage_mod.TRUST_STAGES))

    def test_pluggable_stages_produce_deployment_and_base_model_before_preregister(self) -> None:
        names = [s.name for s in load_pipeline(pack.SCENARIO.pipeline).stages]
        self.assertLess(names.index("choose_task_and_base"), names.index("preregister"))
        self.assertEqual(pack.ChooseTaskAndBase.outputs["deployment"], trust_stages.SCHEMA_DEPLOYMENT)
        self.assertEqual(pack.ChooseTaskAndBase.outputs["base"], trust_stages.SCHEMA_MODEL_REF)


class StageTest(PackBase):
    def test_plan_and_run_produce_the_declared_schemas(self) -> None:
        proj = self.make()
        reg = stage_registry()
        for name in ("intake_seal", "profile_data", "choose_task_and_base", "preregister", "build_train_set",
                     "write_recipe", "smoke", "train"):
            if name != "intake_seal":
                plan = plan_stage(proj, name, registry=reg)
                self.assertEqual(plan.stage, name)
            res = self.run_(proj, name)
            cls = reg[name]
            self.assertEqual({k: r.schema for k, r in res.outputs.items()}, dict(cls.outputs), name)
        self.assertEqual(proj.read_artifact(self.run_(proj, "profile_data").outputs["profile"])["n_tokens"], 20)

    def test_the_base_model_is_the_empty_table_and_train_part_shaped_output_is_emitted(self) -> None:
        proj = self.make()
        out = self.run_all(proj, "build_train_set")
        self.assertEqual(proj.read_artifact(out["choose_task_and_base"]["base"])["table"], {})
        ts = proj.read_artifact(out["build_train_set"]["train_set"])
        self.assertEqual(ts["schema"], trust_stages.SCHEMA_TRAIN_PART)
        self.assertEqual(ts["n"], len(ts["rows"]))
        self.assertGreater(ts["n"], 0)

    def test_the_whole_pipeline_runs_and_the_trained_model_passes_the_gate(self) -> None:
        proj = self.make()
        out = self.run_all(proj)
        self.assertEqual(proj.read_artifact(out["gate"]["gate_result"])["status"], "PASS")
        self.assertIn("report", out)

    def test_train_runs_on_the_cpu_and_is_recorded_as_training(self) -> None:
        proj = self.make()
        self.run_all(proj, "train")
        start = [r for r in proj.read_attempts() if r["stage"] == "train" and r["event"] == "start"][-1]
        self.assertTrue(start["trains"])


class RefNameTest(PackBase):
    """Final review I4: a stage writes under its pipeline ref name (`ctx.stage`), not its class name, so two refs of
    one stage type keep apart and the machine tracks the files it hashes."""

    def make_two_refs(self) -> Project:
        proj = self.make()
        raw = yaml.safe_load(Path(proj.pipeline).read_text())
        names = [s["name"] for s in raw["stages"]]
        raw["stages"].insert(names.index("write_recipe") + 1,
                             {"name": "write_recipe_b", "stage": "write_recipe", "params": {"noise": 0.5},
                              "inputs": {"train_set": "build_train_set.train_set"}})
        path = self.tmp / "two.yaml"
        path.write_text(yaml.safe_dump(raw))
        return create_project(self.tmp / "two", goal="lookup", data=Path(proj.data["path"]), pipeline=str(path),
                              budget={"gpu_hours": 1, "llm_usd": 1})

    def test_two_refs_of_one_stage_type_write_apart_and_both_stay_done(self) -> None:
        from gatedtrain.ainative.machine import status

        proj = self.make_two_refs()
        out = self.run_all(proj, "write_recipe_b")
        self.assertEqual(out["write_recipe"]["recipe"].path, "stages/write_recipe/recipe.json")
        self.assertEqual(out["write_recipe_b"]["recipe"].path, "stages/write_recipe_b/recipe.json")
        self.assertEqual(proj.read_artifact(out["write_recipe_b"]["recipe"])["noise"], 0.5)
        self.assertEqual(proj.read_artifact(out["write_recipe"]["recipe"])["noise"], 0.0)
        self.run_(proj, "smoke")                                   # reads write_recipe's record, still intact
        states = {k: v["status"] for k, v in status(proj, registry=stage_registry()).items()}
        self.assertEqual((states["write_recipe"], states["write_recipe_b"], states["smoke"]), ("done",) * 3)

    def test_an_output_under_another_stage_name_is_refused(self) -> None:
        from gatedtrain.ainative.machine import status

        class ClassNamed(pack.WriteRecipe):
            def run(self, ctx, decision):
                ref = super().run(ctx, decision)["recipe"]
                rec = ctx.project.read_artifact(ref)
                return {"recipe": ctx.project.write_artifact(self.name, "recipe", rec, ref.schema)}

        proj = self.make_two_refs()
        reg = {**stage_registry(), "write_recipe": ClassNamed}
        for name in ("intake_seal", "profile_data", "choose_task_and_base", "preregister", "build_train_set",
                     "write_recipe"):
            run_stage(proj, name, dict(DECISIONS.get(name, {})), registry=reg)
        with self.assertRaises(Refused) as cm:
            run_stage(proj, "write_recipe_b", {}, registry=reg)
        p = cm.exception.problems[0]
        self.assertEqual(p.rule, "AN-5")
        self.assertIn("write_recipe_b", p.message)
        self.assertEqual(status(proj, registry=reg)["write_recipe_b"]["status"], "failed")

    def test_ctx_stage_is_the_ref_name(self) -> None:
        seen = []

        class Spy(pack.WriteRecipe):
            def run(self, ctx, decision):
                seen.append(ctx.stage)
                return super().run(ctx, decision)

        proj = self.make_two_refs()
        reg = {**stage_registry(), "write_recipe": Spy}
        for name in ("intake_seal", "profile_data", "choose_task_and_base", "preregister", "build_train_set",
                     "write_recipe", "write_recipe_b"):
            run_stage(proj, name, dict(DECISIONS.get(name, {})), registry=reg)
        self.assertEqual(seen, ["write_recipe", "write_recipe_b"])


class LearningTest(unittest.TestCase):
    def accuracy(self, model: dict, rows: list[dict]) -> float:
        return sum(scorers.exact_match(pack.generate(model, {"input": r["input"]}), r["label"])
                   for r in rows) / len(rows)

    def test_noise_zero_beats_the_base_and_noise_one_does_not(self) -> None:
        rows = fixture_rows()
        train, held = rows[:100], rows[100:]
        base = {"table": {}}
        self.assertEqual(self.accuracy(base, held), 0.0)
        self.assertGreater(self.accuracy({"table": pack.fit(train, 0.0)}, held), 0.7)
        self.assertLessEqual(self.accuracy({"table": pack.fit(train, 1.0)}, held), self.accuracy(base, held))

    def test_generate_returns_the_label_for_the_input_only(self) -> None:
        model = {"table": {"tok1": "L1"}}
        self.assertEqual(pack.generate(model, {"input": "tok1-77"}, sampler={}, seed=3), "L1")
        self.assertIsNone(pack.generate(model, {"input": "other-1"}))


class SmokeTest(PackBase):
    def test_smoke_refuses_a_degenerate_baseline(self) -> None:
        proj = self.make(noise=1.0)
        with self.assertRaises(Refused) as cm:
            self.run_all(proj)
        p = cm.exception.problems[0]
        self.assertEqual(p.rule, "MM-38")
        self.assertIn("degenerate_mean", p.message)
        self.assertFalse((proj.root / "stages" / "train").exists())

    def test_smoke_refuses_when_everything_is_learned(self) -> None:
        proj = self.make(rows=[{"input": f"tok0-{i}", "label": "L0"} for i in range(60)])
        with self.assertRaises(Refused) as cm:
            self.run_all(proj)
        self.assertEqual(cm.exception.problems[0].rule, "MM-38")

    def test_smoke_passes_for_a_clean_recipe(self) -> None:
        proj = self.make()
        out = self.run_all(proj, "smoke")
        rec = proj.read_artifact(out["smoke"]["smoke"])
        self.assertEqual(rec["verdict"], "PASS")
        self.assertTrue(0 < rec["mean"] < 1)

    def test_a_bad_noise_param_is_refused(self) -> None:
        proj = self.make(noise=2.0)
        with self.assertRaises(Refused):
            self.run_all(proj)


class ReservationTest(unittest.TestCase):
    def test_a_pluggable_stage_may_emit_train_part_but_not_a_trust_record(self) -> None:
        def plugin(schema: str) -> type:
            class _P:
                name = "p_x"
                version = "1"
                inputs: dict = {}
                outputs = {"out": schema}
                needs = frozenset()
                params = frozenset()
                trains = False

            return _P

        class EP:
            name = "p_x"
            value = "x:y"

            def __init__(self, schema: str) -> None:
                self.schema = schema

            def load(self):
                return plugin(self.schema)

        real = stage_mod._plugin_entry_points
        self.addCleanup(setattr, stage_mod, "_plugin_entry_points", real)
        stage_mod._plugin_entry_points = lambda: [EP(trust_stages.SCHEMA_TRAIN_PART)]
        self.assertIn("p_x", stage_registry())
        stage_mod._plugin_entry_points = lambda: [EP(trust_stages.SCHEMA_REPORT)]
        self.assertIn("p_x", stage_registry())
        for schema in (trust_stages.SCHEMA_GATE_RESULT, trust_stages.SCHEMA_REGISTRATION,
                       trust_stages.SCHEMA_SEALED_RECORDS, trust_stages.SCHEMA_SEALED_SUMMARY,
                       trust_stages.SCHEMA_POWER):
            stage_mod._plugin_entry_points = lambda s=schema: [EP(s)]
            with self.subTest(schema=schema), self.assertRaises(Refused) as cm:
                stage_registry()
            self.assertIn("reserved trust schema", cm.exception.problems[0].message)


class GeneratorLookupTest(unittest.TestCase):
    def test_no_runtime_registration_exists(self) -> None:
        self.assertFalse(hasattr(trust_stages, "register_generator"))
        self.assertFalse(hasattr(trust_stages, "GENERATORS"))

    def patched(self, eps: list) -> None:
        real = trust_stages._scenario_entry_points
        self.addCleanup(setattr, trust_stages, "_scenario_entry_points", real)
        trust_stages._scenario_entry_points = lambda: eps

    def test_duplicate_generator_names_are_refused(self) -> None:
        class EP:
            name = "dup"

            def load(self):
                return pack.SCENARIO

        self.patched([EP(), EP()])
        with self.assertRaises(Refused) as cm:
            trust_stages.scenario_generators()
        self.assertEqual(cm.exception.problems[0].rule, "AN-8")

    def test_a_broken_pack_is_refused_without_its_message(self) -> None:
        class EP:
            name = "broken"

            def load(self):
                raise RuntimeError("secret detail")

        self.patched([EP()])
        with self.assertRaises(Refused) as cm:
            trust_stages.scenario_generators()
        self.assertNotIn("secret", cm.exception.problems[0].message)


if __name__ == "__main__":
    unittest.main()
