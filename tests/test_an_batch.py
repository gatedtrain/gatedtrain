"""AN-25, AN-9 (revision 25): the batch generator protocol, the synthetic pack's batch generator and the batch form of
`sealed_eval.score_arm`. No container runs here: a stub `BatchRunner` runs the generator script on the host with
`sys.executable` and the container paths mapped to a temporary directory."""

from __future__ import annotations

import ast
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gatedtrain.ainative import batch, scorers, sealed_eval, trust_stages
from gatedtrain.ainative.batch import BatchGenerator
from gatedtrain.ainative.trust import FixedAlphaLedger, InProcessTrust
from gatedtrain.canonical import canonical_json
from gatedtrain.errors import Refused
from gatedtrain.testing import synthetic_pack as pack

from test_an_synthetic_pack import PackBase, fixture_rows

SEEDS = [11, 12]
FULL = {"sampler": dict(pack.SAMPLER), "clause": {"eval_seeds": list(SEEDS)}}


def toy_items() -> list[dict]:
    """Sealed-shaped `{input, label}` items; the `zz-*` inputs have a token the table does not know."""
    return [{"input": r["input"], "label": r["label"]} for r in fixture_rows(40)] + \
        [{"input": "zz-1", "label": "L0"}, {"input": "zz-2", "label": "odd"}]


def toy_model() -> dict:
    return {"generator": pack.BATCH_GENERATOR, "table": pack.fit(fixture_rows(), 0.0)}


def run_script(script: Path, tmp: Path, items: list[dict], sampler: dict, seeds: list, table: dict) -> Path:
    """Lay out the container's inputs under `tmp` and run the script on the host; returns the output directory."""
    for d in ("in", "model", "out"):
        (tmp / d).mkdir()
    batch.write_items(tmp / "in" / "items.jsonl", items)
    (tmp / "in" / "sampler.json").write_bytes(canonical_json(batch.sampler_doc(sampler, seeds)))
    (tmp / "model" / "table.json").write_bytes(canonical_json(table))
    gen = pack.BATCH
    paths = {gen.interpreter: sys.executable, f"{batch.CONTAINER_SCRIPT_DIR}/{gen.script.name}": str(script),
             batch.CONTAINER_ITEMS: str(tmp / "in" / "items.jsonl"),
             batch.CONTAINER_SAMPLER: str(tmp / "in" / "sampler.json"),
             batch.CONTAINER_MODEL: str(tmp / "model"), batch.CONTAINER_OUT: str(tmp / "out")}
    argv = [paths.get(a, a) for a in batch.container_argv(gen)]
    subprocess.run(argv, check=True, capture_output=True, timeout=60, env={"PATH": os.environ.get("PATH", "")})
    return tmp / "out"


class ScriptRunner:
    """A `BatchRunner` that runs the generator script on the host, then reads the predictions back strictly."""

    def __init__(self, root: Path) -> None:
        self.root, self.calls = root, []

    def run_batch(self, gen: BatchGenerator, model: dict, items: list[dict], sampler: dict, seeds: list) -> list[dict]:
        self.calls.append({"items": [dict(i) for i in items], "seeds": list(seeds)})
        tmp = Path(tempfile.mkdtemp(dir=self.root))
        out = run_script(gen.script, tmp, items, sampler, seeds, model["table"])
        return batch.read_predictions(out, len(items), seeds)


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)


class ProtocolTest(Base):
    def test_container_argv(self) -> None:
        gen = BatchGenerator(script=Path("/site/pkg/gen.py"), interpreter="/usr/bin/python3")
        self.assertEqual(batch.container_argv(gen),
                         ("/usr/bin/python3", "-I", "/gen/gen.py", "--items", "/in/items.jsonl", "--sampler",
                          "/in/sampler.json", "--model", "/in/model", "--out", "/out"))
        self.assertEqual(batch.RULE, "AN-25")
        self.assertEqual(batch.MAX_PREDICTIONS_BYTES, 256 * 1024 * 1024)
        with self.assertRaises(Exception):
            gen.gpu = True                                       # frozen

    def test_synthetic_batch_declaration(self) -> None:
        self.assertEqual(pack.BATCH_GENERATOR, "synthetic_table_batch")
        self.assertEqual(pack.BATCH.interpreter, "/usr/local/bin/python3")
        self.assertEqual(pack.BATCH.time_limit_s, 600)
        self.assertFalse(pack.BATCH.gpu)
        self.assertTrue(pack.BATCH.script.is_absolute())
        self.assertEqual(pack.BATCH.script, Path(pack.__file__).resolve().with_name("synthetic_batch_generate.py"))
        self.assertIs(pack.SCENARIO.batch_generators[pack.BATCH_GENERATOR], pack.BATCH)
        with self.assertRaises(TypeError):
            pack.SCENARIO.batch_generators["x"] = pack.BATCH

    def test_script_is_standard_library_only(self) -> None:
        tree = ast.parse(pack.BATCH.script.read_text())
        mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                self.assertEqual(n.level, 0)                             # no relative import
                mods.add(n.module.split(".")[0])
        self.assertTrue(mods)
        self.assertLessEqual(mods - {"__future__"}, set(sys.stdlib_module_names))

    def test_feature_has_one_source(self) -> None:
        from gatedtrain.testing import synthetic_batch_generate as script
        self.assertIs(pack.feature, script.feature)
        self.assertEqual(pack.feature("tok3-x-y"), "tok3")
        self.assertIsNone(pack.feature({"q": 1}))

    def test_script_matches_per_item(self) -> None:
        items, model = toy_items(), toy_model()
        out = run_script(pack.BATCH.script, self.tmp, [{"input": i["input"]} for i in items], pack.SAMPLER, SEEDS,
                         model["table"])
        preds = batch.read_predictions(out, len(items), SEEDS)
        want = [{"item_id": i, "seed": s, "prediction": pack.generate(model, {"input": it["input"]}, sampler=pack.SAMPLER,
                                                                      seed=s)}
                for s in SEEDS for i, it in enumerate(items)]
        self.assertEqual(preds, want)
        self.assertIn(None, [p["prediction"] for p in preds])          # an unknown token answers nothing
        self.assertGreater(sum(p["prediction"] is not None for p in preds), 0)

    def test_items_file_has_no_label(self) -> None:
        items = [{"input": "tok1-1", "label": "SECRET-LABEL-1"}, {"input": {"q": "x"}, "label": "SECRET-LABEL-2"}]
        path = self.tmp / "items.jsonl"
        batch.write_items(path, items)
        raw = path.read_bytes()
        self.assertNotIn(b"SECRET-LABEL", raw)
        lines = raw.split(b"\n")
        self.assertEqual(lines[-1], b"")
        self.assertEqual([json.loads(x) for x in lines[:-1]],
                         [{"item_id": 0, "input": "tok1-1"}, {"item_id": 1, "input": {"q": "x"}}])
        for line in lines[:-1]:
            self.assertEqual(canonical_json(json.loads(line)), line)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        with self.assertRaises(FileExistsError):                        # a new file only (O_EXCL)
            batch.write_items(path, items)
        link = self.tmp / "link.jsonl"
        link.symlink_to(self.tmp / "elsewhere.jsonl")
        with self.assertRaises(OSError):
            batch.write_items(link, items)
        self.assertFalse((self.tmp / "elsewhere.jsonl").exists())
        # generate_batch hands the runner the inputs only
        runner = ScriptRunner(self.tmp)
        batch.generate_batch(pack.BATCH, toy_model(), items, pack.SAMPLER, SEEDS, runner=runner)
        self.assertEqual(runner.calls[0]["items"], [{"input": "tok1-1"}, {"input": {"q": "x"}}])

    def test_sampler_doc(self) -> None:
        self.assertEqual(batch.sampler_doc({"t": 1}, [11]), {"sampler": {"t": 1}, "seeds": [11]})


class ReadPredictionsTest(Base):
    N = 3

    def lines(self) -> list[bytes]:
        return [canonical_json({"item_id": i, "seed": s, "prediction": f"SECRET-PRED-{s}-{i}"})
                for s in SEEDS for i in range(self.N)]

    def out(self, lines: list[bytes]) -> Path:
        d = Path(tempfile.mkdtemp(dir=self.tmp))
        (d / batch.PREDICTIONS).write_bytes(b"".join(x + b"\n" for x in lines))
        return d

    def assert_crash(self, out: Path) -> None:
        with self.assertRaises(Refused) as cm:
            batch.read_predictions(out, self.N, SEEDS)
        p = cm.exception.problems[0]
        self.assertEqual(p.rule, "AN-9")
        self.assertIn("the inference output is not one prediction per item and seed (crash)", p.message)
        self.assertNotIn("SECRET", str(cm.exception))
        self.assertNotIn("NaN", str(cm.exception))

    def test_valid(self) -> None:
        got = batch.read_predictions(self.out(list(reversed(self.lines()))), self.N, SEEDS)
        self.assertEqual(got, [{"item_id": i, "seed": s, "prediction": f"SECRET-PRED-{s}-{i}"}
                               for s in SEEDS for i in range(self.N)])

    def test_read_predictions_refusals(self) -> None:
        good = self.lines()
        cases = {
            "missing pair": good[:-1],
            "duplicate": good + [good[0]],
            "extra pair": good + [canonical_json({"item_id": self.N, "seed": 11, "prediction": "SECRET-x"})],
            "item out of range": good[:-1] + [canonical_json({"item_id": self.N, "seed": 12,
                                                              "prediction": "SECRET-PRED-12-2"})],
            "unknown seed": good[:-1] + [canonical_json({"item_id": self.N - 1, "seed": 99, "prediction": "SECRET"})],
            "NaN": good[:-1] + [b'{"item_id":2,"prediction":NaN,"seed":12}'],
            "extra key": good[:-1] + [b'{"item_id":1,"seed":11,"prediction":1.0,"x":0}'],
            "key order": good[1:] + [b'{"seed":11,"item_id":0,"prediction":"SECRET-PRED-11-0"}'],
            "whitespace": good[1:] + [b'{"item_id": 0,"prediction":"SECRET-PRED-11-0","seed":11}'],
            "not an object": good + [b'["SECRET"]'],
            "bool item": good[1:] + [b'{"item_id":false,"prediction":"SECRET","seed":11}'],
            "blank line": good + [b""],
        }
        for name, lines in cases.items():
            with self.subTest(case=name):
                self.assert_crash(self.out(lines))
        with self.subTest(case="not utf-8"):
            d = self.out(good)
            (d / batch.PREDICTIONS).write_bytes(b"\xff\xfe" + (d / batch.PREDICTIONS).read_bytes())
            self.assert_crash(d)
        with self.subTest(case="over the cap"), mock.patch.object(batch, "MAX_PREDICTIONS_BYTES", 10):
            self.assert_crash(self.out(good))
        with self.subTest(case="link"):
            target = self.out(good) / batch.PREDICTIONS
            d = Path(tempfile.mkdtemp(dir=self.tmp))
            (d / batch.PREDICTIONS).symlink_to(target)
            self.assert_crash(d)
        with self.subTest(case="extra file"):
            d = self.out(good)
            (d / "notes.txt").write_text("SECRET")
            self.assert_crash(d)
        with self.subTest(case="no file"):
            self.assert_crash(Path(tempfile.mkdtemp(dir=self.tmp)))
        with self.subTest(case="a directory"):
            d = Path(tempfile.mkdtemp(dir=self.tmp))
            (d / batch.PREDICTIONS).mkdir()
            self.assert_crash(d)


class ScoreArmTest(Base):
    def score_arm(self, **kw) -> list[dict]:
        return sealed_eval.score_arm(toy_items(), pack.generate, toy_model(), FULL, scorers.exact_match,
                                     score_range=(0.0, 1.0), scorer_name="exact_match", **kw)

    def test_score_arm_batch_equals_per_item(self) -> None:
        per_item = self.score_arm()
        runner = ScriptRunner(self.tmp)
        got = self.score_arm(batch=pack.BATCH, runner=runner)
        self.assertEqual(got, per_item)
        self.assertEqual(len(runner.calls), 1)                         # one call per arm for all seeds
        self.assertEqual(runner.calls[0]["seeds"], SEEDS)
        self.assertTrue(all(set(i) == {"input"} for i in runner.calls[0]["items"]))
        self.assertGreater(sum(r["score"] for r in got), 0)

    def test_batch_without_runner_refused(self) -> None:
        with self.assertRaises(Refused) as cm:
            self.score_arm(batch=pack.BATCH)
        p = cm.exception.problems[0]
        self.assertEqual(p.rule, "AN-8")
        self.assertIn("a batch generator runs in the keeper's inference container", p.message)
        self.assertIn("bind the project to a keeper", p.fix)

    def test_runner_order_checked(self) -> None:
        class Shuffled:
            def run_batch(self, gen, model, items, sampler, seeds):
                return [{"item_id": i, "seed": s, "prediction": None}
                        for i in range(len(items)) for s in seeds]           # items outer: wrong order
        with self.assertRaises(Refused) as cm:
            self.score_arm(batch=pack.BATCH, runner=Shuffled())
        self.assertEqual(cm.exception.problems[0].rule, "AN-9")

    def test_scorer_failure_names_the_class_only(self) -> None:
        def bad(prediction, label):
            raise ValueError(f"SECRET {label}")
        with self.assertRaises(Refused) as cm:
            sealed_eval.score_arm(toy_items(), None, toy_model(), FULL, bad, score_range=(0.0, 1.0),
                                  scorer_name="x", batch=pack.BATCH, runner=ScriptRunner(self.tmp))
        self.assertEqual(cm.exception.problems[0].rule, "AN-8")
        self.assertIn("ValueError", cm.exception.problems[0].message)
        self.assertNotIn("SECRET", str(cm.exception))


class _EP:
    def __init__(self, name: str, scenario: pack.Scenario) -> None:
        self.name, self.value, self._s = name, f"x:{name}", scenario

    def load(self) -> pack.Scenario:
        return self._s


class SnapshotTest(Base):
    def inject(self, *eps: _EP) -> None:
        real = trust_stages._scenario_entry_points
        trust_stages._scenario_entry_points = lambda: list(eps)
        self.addCleanup(setattr, trust_stages, "_scenario_entry_points", real)
        trust_stages._reset_scenario_snapshot()
        self.addCleanup(trust_stages._reset_scenario_snapshot)

    def test_snapshot_holds_batch_generators(self) -> None:
        self.inject(_EP("synthetic", pack.SCENARIO))
        self.assertIs(trust_stages.scenario_batch_generators()[pack.BATCH_GENERATOR], pack.BATCH)
        self.assertEqual(trust_stages.scenario_pack_of(pack.BATCH_GENERATOR), "synthetic")
        self.assertNotIn(pack.BATCH_GENERATOR, trust_stages.scenario_generators())
        self.assertEqual(trust_stages.scenario_pack_of(pack.GENERATOR), "synthetic")

    def test_name_clash_refused(self) -> None:
        both = pack.Scenario("a", {"x": pack.generate}, batch_generators={"x": pack.BATCH})
        item_only = pack.Scenario("b", {"x": pack.generate})
        batch_only = pack.Scenario("c", batch_generators={"x": pack.BATCH})
        for name, eps in {"one pack, both forms": [_EP("a", both)],
                          "per-item then batch": [_EP("b", item_only), _EP("c", batch_only)],
                          "batch then per-item": [_EP("c", batch_only), _EP("b", item_only)],
                          "two batch packs": [_EP("c", batch_only), _EP("d", batch_only)]}.items():
            with self.subTest(case=name):
                self.inject(*eps)
                with self.assertRaises(Refused) as cm:
                    trust_stages.scenario_batch_generators()
                self.assertEqual(cm.exception.problems[0].rule, "AN-25")
                self.assertIn("'x'", cm.exception.problems[0].message)

    def test_malformed_batch_generators_refused(self) -> None:
        class Odd:
            generators: dict = {}
            batch_generators = {"x": "not a declaration"}
        self.inject(_EP("odd", Odd()))
        with self.assertRaises(Refused) as cm:
            trust_stages.scenario_generators()
        self.assertEqual(cm.exception.problems[0].rule, "AN-25")


class InProcessTest(PackBase):
    def test_in_process_refuses_batch(self) -> None:
        # The pack's models name the batch generator, so the registration binds it.
        real = pack.GENERATOR
        pack.GENERATOR = pack.BATCH_GENERATOR
        self.addCleanup(setattr, pack, "GENERATOR", real)
        proj = self.make()
        out = self.run_all(proj, "train")
        reg = InProcessTrust(proj).registration(proj.read_artifact(out["preregister"]["registration"])["hash"])
        self.assertEqual(trust_stages.registered_generator(reg), ("synthetic", pack.BATCH_GENERATOR))
        ledger = proj.root / FixedAlphaLedger.REL
        before = (FixedAlphaLedger(proj).available(), ledger.read_bytes() if ledger.exists() else None)
        with self.assertRaises(Refused) as cm:
            self.run_(proj, "evaluate_sealed")
        p = cm.exception.problems[0]
        self.assertEqual(p.rule, "AN-8")
        self.assertIn("a batch generator runs in the keeper's inference container", p.message)
        self.assertIsNone(InProcessTrust(proj).sealed_evaluation())   # refused before the claim
        self.assertEqual((FixedAlphaLedger(proj).available(), ledger.read_bytes() if ledger.exists() else None),
                         before)
        self.assertGreater(before[0], 0)


if __name__ == "__main__":
    unittest.main()
