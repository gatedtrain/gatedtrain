"""Task 9: `gatedtrain project` end to end, as subprocesses, on the synthetic scenario pack.

The pack is not registered through entry points, so every subprocess runs `tests/fixtures/ainative/run_cli.py`, which
injects it and then calls `gatedtrain.cli.main`. Every call uses `--json`.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from _support import EXIT_OK, FIXTURES, cli_env

from gatedtrain.testing import synthetic_pack as pack

WRAPPER = FIXTURES / "ainative" / "run_cli.py"
TOY = FIXTURES / "ainative" / "toy.jsonl"
EXIT_NOT_PASS, EXIT_USAGE, EXIT_REFUSED = 1, 2, 3
STAGES = ["intake_seal", "profile_data", "choose_task_and_base", "preregister", "build_train_set", "write_recipe",
          "smoke", "train", "evaluate_sealed", "gate", "report"]
DECISIONS = {"preregister": {"metric_confirmed": True}}


class E2EBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        for stage, decision in DECISIONS.items():
            (self.tmp / f"{stage}.json").write_text(json.dumps(decision))

    def cli(self, *args: str, expect: int | None = EXIT_OK) -> dict:
        """One `--json` call; asserts the exit code and the one-line envelope, returns the parsed envelope."""
        proc = subprocess.run([sys.executable, str(WRAPPER), "--json", *args], cwd=self.tmp, env=cli_env(),
                              capture_output=True, text=True, timeout=300)
        where = f"{args!r}\nstdout: {proc.stdout[-3000:]}\nstderr: {proc.stderr[-3000:]}"
        self.assertEqual(proc.stderr, "", where)
        self.assertEqual(proc.stdout.count("\n"), 1, where)
        doc = json.loads(proc.stdout)
        self.assertEqual((doc["schema"], doc["exit"]), ("gatedtrain.cli_result", proc.returncode), where)
        if expect is not None:
            self.assertEqual(proc.returncode, expect, where)
        return doc

    def pipeline(self, noise: float, name: str = "pipe") -> Path:
        raw = yaml.safe_load(pack.SCENARIO.pipeline.read_text())
        next(s for s in raw["stages"] if s["name"] == "write_recipe")["params"]["noise"] = noise
        path = self.tmp / f"{name}.yaml"
        path.write_text(yaml.safe_dump(raw))
        return path

    def create(self, noise: float = 0.0, name: str = "proj", gpu_hours: str | None = None) -> Path:
        root = self.tmp / name
        extra = ["--gpu-hours", gpu_hours] if gpu_hours else []
        doc = self.cli("project", "create", str(root), "--goal", "lookup task", "--data", str(TOY),
                       "--pipeline", str(self.pipeline(noise, name + "-pipe")), *extra)
        self.assertTrue(doc["ok"])
        return root

    def run_stage(self, root: Path, stage: str, *, expect: int | None = EXIT_OK, decision: dict | None = None) -> dict:
        args = ["project", "run", str(root), stage]
        path = self.tmp / f"{stage}.json"
        if decision is not None:
            path = self.tmp / f"{stage}.custom.json"
            path.write_text(json.dumps(decision))
        if path.exists():
            args += ["--decision", str(path)]
        return self.cli(*args, expect=expect)

    def run_all(self, root: Path, upto: str = "report") -> dict[str, dict]:
        out = {}
        for stage in STAGES:
            out[stage] = self.run_stage(root, stage)
            self.assertTrue(out[stage]["ok"], stage)
            if stage == upto:
                break
        return out

    def attempts(self, root: Path) -> bytes:
        return (root / "attempts.jsonl").read_bytes()


class FullRunTest(E2EBase):
    def test_noise_zero_passes_and_every_stage_is_done(self) -> None:
        root = self.create()
        for stage in STAGES:
            if stage != "intake_seal":
                plan = self.cli("project", "plan", str(root), stage)
                self.assertEqual(plan["data"]["stage"], stage)
                self.assertIn("summary", plan["data"])
            doc = self.run_stage(root, stage)
            self.assertEqual(doc["data"]["stage"], stage)
            self.assertFalse(doc["data"]["reused"])
        status = self.cli("project", "status", str(root))
        self.assertEqual({k: v["status"] for k, v in status["data"]["stages"].items()}, {s: "done" for s in STAGES})
        self.assertEqual(status["data"]["gate"], "PASS")
        self.assertTrue(status["data"]["passed"])
        report = self.cli("project", "report", str(root))
        self.assertTrue(report["ok"])
        self.assertEqual(report["data"]["report"]["status"], "PASS")
        self.assertTrue((root / "stages" / "report" / "report.md").is_file())

    def test_rerunning_the_sequence_is_a_no_op(self) -> None:
        root = self.create()
        first = self.run_all(root)
        log = self.attempts(root)
        second = self.run_all(root)
        for stage in STAGES:
            self.assertTrue(second[stage]["data"]["reused"], stage)
            self.assertEqual(second[stage]["data"]["outputs"], first[stage]["data"]["outputs"], stage)
        self.assertEqual(self.attempts(root), log)       # no new `start` or `done` event, the gate included

    def test_a_second_gate_run_after_a_changed_decision_refuses_alpha_spent(self) -> None:
        root = self.create()
        self.run_all(root, "gate")
        log = self.attempts(root)
        doc = self.run_stage(root, "gate", expect=EXIT_REFUSED, decision={"rerun": True, "again": 1})
        self.assertFalse(doc["ok"])
        self.assertEqual(doc["problems"][0]["rule"], "AN-9")
        self.assertIn("alpha", doc["problems"][0]["message"])
        self.assertEqual(self.attempts(root).count(b'"event":"done"'), log.count(b'"event":"done"'))

    def test_an_amended_registration_after_the_gate_refuses_alpha_spent(self) -> None:
        """Changing the inputs of the gate (a new registration) is a new gate attempt; the spent alpha refuses it
        already at `preregister`, which could not grant its alpha (AN-7), so the gate cannot run again."""
        root = self.create()
        self.run_all(root, "gate")
        log = self.attempts(root)
        doc = self.run_stage(root, "preregister", expect=EXIT_REFUSED,
                             decision={"metric_confirmed": True, "margin": 0.05, "rerun": True})
        self.assertEqual(doc["problems"][0]["rule"], "AN-7")
        self.assertIn("alpha", doc["problems"][0]["message"])
        self.assertEqual(self.attempts(root).count(b'"event":"done"'), log.count(b'"event":"done"'))


class DegenerateRecipeTest(E2EBase):
    def test_noise_one_is_refused_at_smoke_and_never_reaches_the_gate(self) -> None:
        root = self.create(noise=1.0)
        for stage in STAGES[:STAGES.index("smoke")]:
            self.run_stage(root, stage)
        doc = self.run_stage(root, "smoke", expect=EXIT_REFUSED)
        self.assertFalse(doc["ok"])
        self.assertEqual(doc["problems"][0]["rule"], "MM-38")
        self.assertFalse((root / "stages" / "train").exists())
        gate = self.run_stage(root, "gate", expect=EXIT_REFUSED)
        self.assertFalse(gate["ok"])
        self.assertFalse((root / "stages" / "gate").exists())
        status = self.cli("project", "status", str(root))
        self.assertEqual(status["data"]["stages"]["smoke"]["status"], "failed")
        self.assertEqual(status["data"]["stages"]["gate"]["status"], "pending")
        self.assertIsNone(status["data"]["gate"])
        self.assertFalse(status["data"]["passed"])
        report = self.cli("project", "report", str(root), expect=EXIT_REFUSED)
        self.assertFalse(report["ok"])
        self.assertNotIn("PASS", json.dumps(report["data"]))
        self.assertIn("report", report["problems"][0]["message"])


class CreateTest(E2EBase):
    def test_the_gpu_budget_is_recorded_and_nothing_enforces_it(self) -> None:
        root = self.create(gpu_hours="2.5")
        doc = yaml.safe_load((root / "project.yaml").read_text())
        self.assertEqual(doc["budget"]["gpu_hours"], 2.5)
        self.run_all(root)                                      # a project with a budget still runs to the end
        help_text = subprocess.run([sys.executable, str(WRAPPER), "project", "create", "--help"],
                                   capture_output=True, text=True, env=cli_env()).stdout
        self.assertIn("--gpu-hours", help_text)
        self.assertIn("does not enforce", " ".join(help_text.split()))

    def test_create_refuses_an_existing_project_and_a_missing_pipeline(self) -> None:
        root = self.create()
        again = self.cli("project", "create", str(root), "--goal", "x", "--data", str(TOY),
                         "--pipeline", str(self.tmp / "pipe.yaml"), expect=EXIT_REFUSED)
        self.assertFalse(again["ok"])
        missing = self.cli("project", "create", str(self.tmp / "other"), "--goal", "x", "--data", str(TOY),
                           "--pipeline", str(self.tmp / "nope.yaml"), expect=EXIT_REFUSED)
        self.assertFalse(missing["ok"])

    def test_a_decision_file_that_is_not_an_object_is_refused(self) -> None:
        root = self.create()
        bad = self.tmp / "bad.json"
        bad.write_text("[1]")
        doc = self.cli("project", "run", str(root), "intake_seal", "--decision", str(bad), expect=EXIT_REFUSED)
        self.assertFalse(doc["ok"])
        self.assertEqual(self.cli("project", "run", str(root), "nostage", expect=EXIT_REFUSED)["ok"], False)

    def test_text_mode_prints_the_status(self) -> None:
        root = self.create()
        proc = subprocess.run([sys.executable, str(WRAPPER), "project", "status", str(root)], capture_output=True,
                              text=True, env=cli_env(), cwd=self.tmp)
        self.assertEqual(proc.returncode, EXIT_OK, proc.stderr)
        self.assertIn("intake_seal", proc.stdout)
        self.assertIn("pending", proc.stdout)


if __name__ == "__main__":
    unittest.main()
