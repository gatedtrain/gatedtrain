"""AN-17 and AN-6 (extended): the scorer library and pack scorers by reference (tests/README.md section 13)."""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from unittest import mock

import test_an_gate as tg
import test_an_intake as ti
from gatedtrain.ainative import scorers, trust_stages
from gatedtrain.ainative.scorers import (PLANNING_SD_BOUND, ScorerSpec, bind, evaluator_entry, library,
                                         manifest_document, manifest_hash, resolve, spec_for_evaluator)
from gatedtrain.ainative.trust import FixedAlphaLedger, InProcessTrust, SealedRef
from gatedtrain.canonical import file_sha256
from gatedtrain.errors import Refused
from gatedtrain.registration import check_contents, resolved_evaluator_hash
from gatedtrain.testing.synthetic_pack import Scenario

lib = library()
PACK = "t17"
TEST_SCORERS: dict = {}


def pack_iou(prediction, label) -> float:
    """A test-only pack scorer: 1.0 when equal, else 0.0."""
    return 1.0 if prediction == label else 0.0


def pack_decl(**over) -> dict:
    decl = {"id": "pack_iou.t17@0.1", "fn": pack_iou, "code_sha256": file_sha256(Path(__file__)),
            "planning_sd": 0.5, "direction": "higher_is_better", "range": [0.0, 1.0],
            "what": "the share of items whose answer equals yours", "unit": "share of equal answers (0 to 1)"}
    decl.update(over)
    return decl


class _PackScenario:
    """A test-only `gatedtrain.scenarios` entry point with the generators of test_an_gate and test scorers."""
    name = PACK
    value = "test:pack"

    def load(self):
        return self

    @property
    def generators(self):
        return tg.TEST_GENERATORS

    @property
    def scorers(self):
        return TEST_SCORERS


def inject_pack(case: unittest.TestCase) -> None:
    real = trust_stages._scenario_entry_points
    trust_stages._scenario_entry_points = lambda: [_PackScenario()]
    case.addCleanup(setattr, trust_stages, "_scenario_entry_points", real)
    trust_stages._reset_scenario_snapshot()
    case.addCleanup(trust_stages._reset_scenario_snapshot)
    TEST_SCORERS.clear()
    case.addCleanup(TEST_SCORERS.clear)


class LibraryTest(unittest.TestCase):
    def test_library_ids_and_planning_sd(self) -> None:
        self.assertEqual(sorted(library()), ["exact_match", "field_f1", "normalised_match", "numeric_match", "set_f1"])
        for name, s in library().items():
            self.assertIsInstance(s, ScorerSpec)
            self.assertEqual((s.planning_sd, s.score_range), (math.sqrt(0.5), (0.0, 1.0)))
            self.assertEqual((s.name, s.evaluator_id, s.direction, s.source),
                             (name, f"{name}.ainative@0.1", "higher_is_better", "library"))
        self.assertEqual(PLANNING_SD_BOUND, math.sqrt(0.5))
        self.assertEqual(lib["numeric_match"].params, {"rel_tol": 0.0, "abs_tol": 0.0})
        self.assertEqual((scorers.EVALUATOR_ID, scorers.DIRECTION), ("exact_match.ainative@0.1", "higher_is_better"))

    def test_each_family_file_imports_only_the_standard_library_and_canonical(self) -> None:
        pkg = Path(scorers.__file__).parent
        for name in ("exact.py", "numeric.py", "normalised.py", "sets.py"):
            text = (pkg / name).read_text(encoding="utf-8")
            imports = [ln.strip() for ln in text.splitlines() if ln.startswith(("import ", "from "))]
            for ln in imports:
                ok = ln.startswith(("from __future__", "import math", "import unicodedata", "from typing"))
                ok = ok or ln.startswith(("from ...canonical import", "from gatedtrain.canonical import"))
                self.assertTrue(ok, f"{name}: {ln}")

    def test_exact(self) -> None:
        f = bind(lib["exact_match"], {})
        self.assertEqual((f({"x": 1, "y": [2]}, {"y": [2], "x": 1}), f(1, "1"), f(float("nan"), 1)), (1.0, 0.0, 0.0))
        self.assertIs(scorers.exact_match, lib["exact_match"].fn)

    def test_numeric_tolerance(self) -> None:
        f = bind(lib["numeric_match"], {"rel_tol": 0.01, "abs_tol": 0.0})
        self.assertEqual((f("100.5", 100), f(102, 100), f(True, 1), f("nan", 1)), (1.0, 0.0, 0.0, 0.0))
        g = bind(lib["numeric_match"], {})
        self.assertEqual((g(3, 3), g("3", 3.0), g(3.0000001, 3), g("inf", 1), g(float("inf"), 1), g([3], 3)),
                         (1.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(bind(lib["numeric_match"], {"abs_tol": 0.5})(2.4, 2), 1.0)

    def test_normalised(self) -> None:
        f = bind(lib["normalised_match"], {})
        self.assertEqual(f("Ｆｏｏ  BAR\n", "foo bar"), 1.0)
        self.assertEqual((f("foo", "bar"), f(1, "1"), f("Straße", "STRASSE")), (0.0, 0.0, 1.0))

    def test_set_f1(self) -> None:
        f = bind(lib["set_f1"], {})
        self.assertAlmostEqual(f(["a", "b", "b"], ["b", "c"]), 0.5)
        self.assertEqual((f([], []), f("b", ["b"]), f([{"k": 1}], [{"k": 1}]), f([1], ["1"])), (1.0, 0.0, 1.0, 0.0))
        self.assertEqual(f([float("nan")], ["a"]), 0.0)                 # not canonical JSON

    def test_field_f1(self) -> None:
        f = bind(lib["field_f1"], {})
        self.assertAlmostEqual(f({"a": 1, "b": 2}, {"a": 1, "b": 3}), 0.5)
        self.assertEqual((f({}, {}), f(["a"], {"a": 1}), f({"a": [1, 2]}, {"a": [1, 2]})), (1.0, 0.0, 1.0))
        self.assertAlmostEqual(f({"a": 1}, {"a": 1, "b": 2}), 2 / 3)

    def test_label_ok(self) -> None:
        self.assertFalse(lib["set_f1"].label_ok("x"))
        self.assertTrue(lib["set_f1"].label_ok(["x"]))
        self.assertFalse(lib["numeric_match"].label_ok("3"))
        self.assertFalse(lib["numeric_match"].label_ok(True))
        self.assertTrue(lib["numeric_match"].label_ok(3.5))
        self.assertFalse(lib["field_f1"].label_ok([]))
        self.assertTrue(lib["field_f1"].label_ok({"a": 1}))
        self.assertFalse(lib["normalised_match"].label_ok(3))
        self.assertTrue(lib["exact_match"].label_ok([1, {"a": "b"}]))

    def test_bad_params_are_refused(self) -> None:
        for params in ({"nope": 1.0}, {"rel_tol": float("nan")}, {"rel_tol": float("inf")}, {"abs_tol": -0.1},
                       {"rel_tol": True}, {"rel_tol": "0.1"}):
            with self.assertRaises(Refused) as cm:
                bind(lib["numeric_match"], params)
            self.assertEqual({p.rule for p in cm.exception.problems}, {"AN-17"})
        with self.assertRaises(Refused):
            bind(lib["exact_match"], {"rel_tol": 0.1})
        with self.assertRaises(Refused):
            bind(lib["exact_match"], [])                                   # not an object

    def test_manifest_hash_binds_params(self) -> None:
        num = lib["numeric_match"]
        a, b = manifest_hash(num, {"rel_tol": 0.01}), manifest_hash(num, {"rel_tol": 0.02})
        self.assertNotEqual(a, b)
        self.assertEqual(manifest_hash(num, {}), manifest_hash(num, {"rel_tol": 0.0, "abs_tol": 0.0}))
        entry = evaluator_entry(num, {"rel_tol": 0.01})
        self.assertEqual(entry, {"id": "numeric_match.ainative@0.1", "manifest_hash": a,
                                 "params": {"rel_tol": 0.01, "abs_tol": 0.0}})
        self.assertEqual(resolved_evaluator_hash(entry), a)
        self.assertEqual(spec_for_evaluator(entry), (num, {"rel_tol": 0.01, "abs_tol": 0.0}))
        self.assertEqual(evaluator_entry(lib["set_f1"], {}), {"id": "set_f1.ainative@0.1",
                                                              "manifest_hash": manifest_hash(lib["set_f1"], {})})
        doc = manifest_document(num, {"rel_tol": 0.01})
        self.assertEqual(set(doc), {"evaluator", "scorer", "direction", "version", "params", "code"})
        self.assertEqual(doc["code"], {"numeric.py": file_sha256(Path(scorers.__file__).with_name("numeric.py"))})
        self.assertEqual(manifest_hash(), manifest_hash(lib["exact_match"], {}))        # the step 1 call
        self.assertIsNone(spec_for_evaluator({"id": "nope@0.1"}))

    def test_scenario_dataclass_has_scorers(self) -> None:
        self.assertEqual(dict(Scenario("x").scorers), {})
        s = Scenario("x", scorers={"iou": pack_decl()})
        with self.assertRaises(TypeError):
            s.scorers["other"] = {}                                         # frozen like generators


class PackScorerTest(unittest.TestCase):
    def setUp(self) -> None:
        inject_pack(self)

    def refused(self, name: str) -> str:
        with self.assertRaises(Refused) as cm:
            resolve(name)
        self.assertEqual({p.rule for p in cm.exception.problems}, {"AN-17"})
        return cm.exception.problems[0].message

    def test_pack_scorer_by_reference(self) -> None:
        TEST_SCORERS["pack_iou"] = pack_decl()
        trust_stages._reset_scenario_snapshot()
        spec = resolve("pack_iou")
        self.assertEqual((spec.name, spec.evaluator_id, spec.source, spec.planning_sd, spec.score_range),
                         ("pack_iou", "pack_iou.t17@0.1", PACK, 0.5, (0.0, 1.0)))
        self.assertEqual(spec.code_files, (Path(__file__).resolve(),))
        self.assertTrue(spec.label_ok({"any": ["json"]}))                 # default: any JSON value
        self.assertFalse(spec.label_ok(float("nan")))
        self.assertEqual(bind(spec, {})("a", "a"), 1.0)
        entry = evaluator_entry(spec, {})
        self.assertEqual(set(entry), {"id", "manifest_hash"})
        self.assertEqual(resolved_evaluator_hash(entry), entry["manifest_hash"])
        self.assertEqual(spec_for_evaluator(entry), (spec, {}))
        self.assertIn("pack_iou", trust_stages.scenario_scorers())
        with self.assertRaises(TypeError):
            trust_stages.scenario_scorers()["x"] = {}
        with self.assertRaises(Refused):
            bind(spec, {"rel_tol": 0.1})

    def test_declared_label_ok_is_used(self) -> None:
        TEST_SCORERS["pack_iou"] = pack_decl(label_ok=lambda label: isinstance(label, str))
        self.assertFalse(resolve("pack_iou").label_ok(3))
        self.assertTrue(resolve("pack_iou").label_ok("3"))

    def test_wrong_code_hash_is_refused(self) -> None:
        TEST_SCORERS["pack_iou"] = pack_decl(code_sha256="sha256:" + "0" * 64)
        self.assertIn("code", self.refused("pack_iou"))
        self.assertIsNone(resolved_evaluator_hash({"id": "pack_iou.t17@0.1"}))

    def test_name_clash_with_the_library_is_refused(self) -> None:
        TEST_SCORERS["exact_match"] = pack_decl()
        self.assertIn("library", self.refused("exact_match"))
        TEST_SCORERS.clear()
        TEST_SCORERS["other"] = pack_decl(id="exact_match.ainative@0.1")
        trust_stages._reset_scenario_snapshot()
        self.assertIn("library", self.refused("other"))

    def test_pack_scorer_id_of_an_ext_row_refused(self) -> None:
        from gatedtrain.ext import registry
        row_id = "pack_iou.t17@0.1"
        registry.EVALUATOR_ROWS[row_id] = {}
        self.addCleanup(registry.EVALUATOR_ROWS.pop, row_id, None)
        TEST_SCORERS["pack_iou"] = pack_decl(id=row_id)
        trust_stages._reset_scenario_snapshot()
        message = self.refused("pack_iou")
        self.assertIn("ext evaluator row", message)
        with self.assertRaises(Refused) as cm:
            resolve("pack_iou")
        self.assertIn("its own id", cm.exception.problems[0].fix)

    def test_unknown_name_and_bad_declarations_are_refused(self) -> None:
        self.refused("vibes")
        for bad in ({"planning_sd": 0}, {"planning_sd": float("nan")}, {"direction": "up"}, {"range": [1, 0]},
                    {"fn": "not callable"}, {"what": ""}, {"label_ok": 3}, {"extra": 1}):
            TEST_SCORERS.clear()
            TEST_SCORERS["pack_iou"] = pack_decl(**bad)
            trust_stages._reset_scenario_snapshot()
            self.refused("pack_iou")


# Labels for each library scorer: the test generator answers with the label itself for the share `rate` of items
# and "wrong" otherwise, so every scorer gives 1.0 or 0.0 per item and the base arm is not all zeros.
LABELS_FOR = {
    "exact_match": lambda i: f"label-{i}",
    "numeric_match": lambda i: i * 1.5 + 0.25,
    "normalised_match": lambda i: f"Label {i}",
    "set_f1": lambda i: [f"a{i}", f"b{i}"],
    "field_f1": lambda i: {"k": i, "v": f"x{i}"},
}


def scorer_rows(name: str, n: int) -> list[dict]:
    return [{"input": f"question-{i}-marker-{i * 7919}", "label": LABELS_FOR[name](i)} for i in range(n)]


class PreregisterScorerTest(tg.GateBase):
    def setUp(self) -> None:
        super().setUp()
        inject_pack(self)

    def register(self, name: str, decision: dict | None = None, *, project: str | None = None):
        with mock.patch.object(tg, "unique_rows", lambda n: scorer_rows(name, n)):
            proj = self.make(name=project or name)
        self.run_(proj, "intake_seal")
        self.run_(proj, "choose_x")
        out = self.run_(proj, "preregister", {"metric_confirmed": True, **(decision or {"scorer": name})})
        return proj, out.outputs["registration"]

    def test_preregister_with_each_library_scorer(self) -> None:
        for name, spec in library().items():
            with self.subTest(scorer=name):
                tg.LABELS.clear()
                proj, ref = self.register(name)
                rec = proj.read_artifact(ref)
                self.assertEqual(rec["scorer"], name)
                full = InProcessTrust(proj).registration(rec["hash"])
                self.assertEqual(check_contents(full), [])
                self.assertEqual(full["evaluators"][0], evaluator_entry(spec, {}))
                self.assertEqual(full["clause"]["metric"], name)
                sealed = InProcessTrust(proj).run_sealed(SealedRef(rec["item_set_hash"]), lambda item: item)
                self.assertTrue(all(spec.label_ok(it["label"]) for it in sealed))
                if name == "set_f1":
                    self.assertTrue(all(isinstance(it["label"], list) for it in sealed))   # list labels sealed
                self.assertIn(spec.what, trust_stages._sentence({**rec, "scorer": name}, 10))
                self.run_(proj, "train_x")
                out = self.run_(proj, "evaluate_sealed").outputs
                cand = proj.read_artifact(out["candidate_records"])["records"]
                base = proj.read_artifact(out["baseline_records"])["records"]
                self.assertTrue(all(r["score"] == 1.0 for r in cand))
                self.assertTrue(0.3 < sum(r["score"] for r in base) / len(base) < 0.7)
                gate = proj.read_artifact(self.run_(proj, "gate").outputs["gate_result"])
                self.assertEqual(gate["status"], "PASS")
                rep = proj.read_artifact(self.run_(proj, "report").outputs["report"])
                self.assertEqual(rep["metric"], {"scorer": name, "unit": spec.unit})

    def test_scorer_params_are_registered(self) -> None:
        proj, ref = self.register("numeric_match", {"scorer": "numeric_match", "scorer_params": {"abs_tol": 0.5}})
        full = InProcessTrust(proj).registration(proj.read_artifact(ref)["hash"])
        self.assertEqual(full["evaluators"][0],
                         evaluator_entry(lib["numeric_match"], {"abs_tol": 0.5}))
        self.assertEqual(full["evaluators"][0]["params"], {"rel_tol": 0.0, "abs_tol": 0.5})

    def test_bad_scorer_params_are_refused(self) -> None:
        for i, params in enumerate(({"nope": 1}, {"abs_tol": -1}, [1])):
            with self.subTest(params=params):
                tg.LABELS.clear()
                with self.assertRaises(Refused) as cm:
                    self.register("numeric_match", {"scorer": "numeric_match", "scorer_params": params},
                                  project=f"bad{i}")
                self.assertEqual({p.rule for p in cm.exception.problems}, {"AN-17"})
                self.assertIn("scorer_params", cm.exception.problems[0].field)

    def test_the_sentence_names_what_the_scorer_counts(self) -> None:
        with mock.patch.object(tg, "unique_rows", lambda n: scorer_rows("set_f1", n)):
            proj = self.make(name="sentence")
        self.run_(proj, "intake_seal")
        self.run_(proj, "choose_x")
        out = self.run_(proj, "preregister", {"metric_confirmed": True, "scorer": "set_f1"}).outputs
        self.assertEqual(proj.read_artifact(out["registration"])["scorer"], "set_f1")
        prop = self.plan_(proj, "preregister")
        self.assertIn(lib["exact_match"].what, prop.summary)           # plan() proposes the default scorer
        self.assertIn("set_f1", prop.choices["available_scorers"])

    def test_preregister_with_a_pack_scorer(self) -> None:
        TEST_SCORERS["pack_iou"] = pack_decl()
        trust_stages._reset_scenario_snapshot()
        proj, ref = self.register("exact_match", {"scorer": "pack_iou"})
        full = InProcessTrust(proj).registration(proj.read_artifact(ref)["hash"])
        self.assertEqual(full["evaluators"][0]["id"], "pack_iou.t17@0.1")
        self.assertEqual(check_contents(full), [])

    def test_preregister_refuses_larger_planning_sd(self) -> None:
        TEST_SCORERS["wide"] = pack_decl(id="wide.t17@0.1", planning_sd=0.9)
        trust_stages._reset_scenario_snapshot()
        with self.assertRaises(Refused) as cm:
            self.register("exact_match", {"scorer": "wide"})
        p = cm.exception.problems[0]
        self.assertEqual(p.rule, "AN-17")
        self.assertEqual(p.fix, f"the sealed set was sized for a planning SD of {math.sqrt(0.5)}; pick a library "
                                "scorer or run intake again with new data")


def raising_scorer(prediction, label) -> float:
    """A test-only pack scorer whose exception message quotes the sealed label."""
    raise ValueError(f"cannot score label {label!r}")


def out_of_range_scorer(prediction, label) -> float:
    return 2.0


def nan_scorer(prediction, label) -> float:
    return float("nan")


def ten_point_scorer(prediction, label) -> float:
    """A test-only pack scorer on [0, 10]: 10.0 when equal, else 0.0."""
    return 10.0 if prediction == label else 0.0


class PackScorerEvaluationTest(tg.GateBase):
    """evaluate_sealed and gate with a pack scorer: the no-leak refusals and the score range (AN-8, AN-9, AN-17)."""

    def setUp(self) -> None:
        super().setUp()
        inject_pack(self)

    def upto_scoring(self, fn, *, rng=(0.0, 1.0), project: str = "pack"):
        TEST_SCORERS["pk"] = pack_decl(id="pk.t17@0.1", fn=fn, range=list(rng))
        trust_stages._reset_scenario_snapshot()
        with mock.patch.object(tg, "unique_rows", lambda n: scorer_rows("exact_match", n)):
            proj = self.make(name=project)
        self.run_(proj, "intake_seal")
        self.run_(proj, "choose_x")
        self.run_(proj, "preregister", {"metric_confirmed": True, "scorer": "pk"})
        self.run_(proj, "train_x")
        return proj

    def test_a_raising_pack_scorer_is_refused_without_the_label(self) -> None:
        proj = self.upto_scoring(raising_scorer)
        with self.assertRaises(Refused) as cm:
            self.run_(proj, "evaluate_sealed")
        problems = cm.exception.problems
        self.assertEqual({p.rule for p in problems}, {"AN-8"})
        self.assertIn("ValueError", problems[0].message)
        for p in problems:
            for text in (p.field, p.message, p.fix):
                self.assertNotIn("label-", text)                   # no sealed label echoed
                self.assertNotIn("cannot score", text)             # nor the scorer's own message

    def test_out_of_range_and_non_finite_scores_are_refused(self) -> None:
        for i, fn in enumerate((out_of_range_scorer, nan_scorer)):
            with self.subTest(fn=fn.__name__):
                tg.LABELS.clear()
                trust_stages._reset_scenario_snapshot()
                proj = self.upto_scoring(fn, project=f"range{i}")
                with self.assertRaises(Refused) as cm:
                    self.run_(proj, "evaluate_sealed")
                p = cm.exception.problems[0]
                self.assertEqual(p.rule, "AN-8")
                self.assertIn("range", p.message)

    def test_gate_uses_the_pack_scorers_score_range(self) -> None:
        proj = self.upto_scoring(ten_point_scorer, rng=(0.0, 10.0))
        out = self.run_(proj, "evaluate_sealed").outputs
        cand = proj.read_artifact(out["candidate_records"])["records"]
        self.assertTrue(all(r["score"] == 10.0 for r in cand))
        gate = proj.read_artifact(self.run_(proj, "gate").outputs["gate_result"])
        self.assertEqual(gate["status"], "PASS")                   # (0, 1) would count every 10.0 as missing
        self.assertEqual(gate["candidate_score"], 10.0)
        self.assertTrue(3.0 < gate["baseline_score"] < 7.0)
        rep = proj.read_artifact(self.run_(proj, "report").outputs["report"])
        self.assertEqual(rep["metric"]["unit"], pack_decl()["unit"])

    def test_gate_refuses_before_the_spend_when_the_scorer_is_gone(self) -> None:
        proj = self.upto_scoring(pack_iou)
        self.run_(proj, "evaluate_sealed")
        before = FixedAlphaLedger(proj).entries()
        TEST_SCORERS.clear()
        trust_stages._reset_scenario_snapshot()
        with self.assertRaises(Refused) as cm:
            self.run_(proj, "gate")
        self.assertEqual({p.rule for p in cm.exception.problems}, {"AN-9"})
        self.assertEqual(FixedAlphaLedger(proj).entries(), before)
        self.assertEqual(FixedAlphaLedger(proj).available(), 0.05)  # no alpha spent


class ListLabelIntakeTest(ti.Base):
    def test_a_list_label_is_accepted(self) -> None:
        proj = self.make(ti.jsonl(scorer_rows("set_f1", ti.N_DATA)))
        out = self.run_(proj, "intake_seal").outputs
        summary = proj.read_artifact(out["sealed_summary"])
        sealed = self.sealed_items(proj, summary["manifest_sha256"])
        self.assertTrue(sealed and all(isinstance(it["label"], list) for it in sealed))


if __name__ == "__main__":
    unittest.main()
