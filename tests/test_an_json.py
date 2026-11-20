"""AN-1: `--json` on every CLI command (tests/README.md section 12).

`--json` goes before the subcommand and makes stdout exactly one JSON object
(`schema` gatedtrain.cli_result, `schema_version`, `command`, `ok`, `exit`, `data`, `problems`, `warnings`); stderr
stays empty; the exit code is the one the text mode returns. Without `--json` the text output is unchanged.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import test_ext_launch as tel
from _support import DEMO_DIR, EXIT_OK, EXIT_REFUSED, describe, run_cli

from gatedtrain.cli import build_parser
from gatedtrain.errors import Problem
from gatedtrain.output import Result, emit, result_json

# The smallest valid argument list of each subcommand (parse only; nothing runs).
MINIMAL_ARGS = {
    "validate": ["spec.yaml"],
    "run": ["spec.yaml"],
    "warmup": ["warmup.yaml", "--out", "out"],
    "impact": ["sha256:" + "0" * 64, "--run-dir", "run"],
    "export-gsm8k": ["--out", "out"],
    "verify-export": ["dir"],
    "merge-eval": ["--registration", "r.yaml", "--out", "out", "run1"],
    "nel-config": ["--registration", "r.yaml", "--model-url", "u", "--model-id", "m", "--eval-root", "e", "--out", "o"],
    "tierb-collect": ["spec.yaml", "--stage-out", "s", "--eval-baseline", "b", "--eval-candidate", "c",
                      "--image-id", "i", "--nemo-rl-commit", "c"],
    "ext-collect": ["spec.yaml", "--adapter", "a", "--stage-out", "s", "--launch-record", "l", "--smoke-record", "m",
                    "--eval-baseline", "b", "--eval-candidate", "c"],
    "ext-merge-eval": ["--registration", "r.yaml", "--out", "out", "run1"],
    "ext-launch": ["--adapter", "a"],
    "ext-from-nel": ["--registration", "r.yaml", "--adapter", "a", "--evaluator", "e", "--items", "i",
                     "--checkpoint", "c", "--out", "o", "run1"],
    "ext-arm-fill": ["--adapter", "a", "--checkpoint", "c", "--items", "i", "arm"],
}


def subcommand_names(parser) -> list[str]:
    for action in parser._actions:
        if hasattr(action, "choices") and isinstance(action.choices, dict) and action.dest == "command":
            return list(action.choices)
    raise AssertionError("no subparsers")


class TestJson(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-an-json-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_every_subcommand_accepts_json(self):
        names = subcommand_names(build_parser())
        self.assertEqual(sorted(names), sorted(MINIMAL_ARGS), "a new subcommand needs an entry in MINIMAL_ARGS")
        for name in names:
            args = build_parser().parse_args(["--json", name, *MINIMAL_ARGS[name]])
            self.assertIs(args.json, True, name)
            self.assertIs(build_parser().parse_args([name, *MINIMAL_ARGS[name]]).json, False, name)

    def test_refusal_is_one_json_line(self):
        proc = run_cli(["--json", "validate", "missing.yaml"], cwd=self.tmp)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertEqual(proc.stderr, "")
        self.assertEqual(proc.stdout.count("\n"), 1, describe(proc))
        doc = json.loads(proc.stdout)
        self.assertEqual((doc["schema"], doc["schema_version"], doc["command"], doc["ok"], doc["exit"]),
                         ("gatedtrain.cli_result", 1, "validate", False, 3))
        self.assertEqual(set(doc), {"schema", "schema_version", "command", "ok", "exit", "data", "problems", "warnings"})
        self.assertTrue(doc["problems"])
        self.assertTrue(all(set(p) == {"rule", "field", "message", "fix"} for p in doc["problems"]))
        self.assertEqual(doc["warnings"], [])

    def test_success_is_one_json_line(self):
        proc = run_cli(["--json", "validate", str(DEMO_DIR / "spec.yaml")], cwd=self.tmp)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertEqual(proc.stderr, "")
        doc = json.loads(proc.stdout)
        self.assertEqual((doc["ok"], doc["exit"], doc["data"], doc["problems"]), (True, 0, {}, []))

    def test_usage_error_keeps_exit_2_in_json(self):
        proc = run_cli(["--json", "ext-launch", "--adapter", "synthetic"], cwd=self.tmp)
        self.assertEqual(proc.returncode, 2, describe(proc))
        self.assertEqual(proc.stderr, "")
        doc = json.loads(proc.stdout)
        self.assertEqual((doc["ok"], doc["exit"]), (False, 2))
        self.assertTrue(doc["problems"])

    def test_text_output_unchanged_without_json(self):
        ok = run_cli(["validate", str(DEMO_DIR / "spec.yaml")], cwd=self.tmp)
        self.assertEqual((ok.returncode, ok.stdout, ok.stderr), (0, "ok\n", ""))
        bad = run_cli(["validate", "missing.yaml"], cwd=self.tmp)
        self.assertEqual((bad.returncode, bad.stdout), (3, ""))
        self.assertTrue(bad.stderr.startswith("error["), bad.stderr)
        self.assertEqual(bad.stderr.count("\n"), len(bad.stderr.splitlines()))
        usage = run_cli(["ext-launch", "--adapter", "synthetic"], cwd=self.tmp)
        self.assertEqual((usage.returncode, usage.stdout), (2, ""))
        self.assertEqual(usage.stderr, "gatedtrain ext-launch: error: --profile, --request, --gpu-log, --out-record "
                                       "required (or --recheck RECORD)\n")

    def test_ext_launch_json_carries_the_command(self):
        (self.tmp / "p").mkdir()
        plain, _ = tel.launch(self.tmp / "p", tel.GOLDEN_REQUEST)
        self.assertEqual(plain.returncode, EXIT_OK, describe(plain))
        (self.tmp / "j").mkdir()
        proc, out = tel.launch(self.tmp / "j", tel.GOLDEN_REQUEST, pre=["--json"])
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        doc = json.loads(proc.stdout)
        self.assertEqual((doc["command"], doc["ok"], doc["exit"]), ("ext-launch", True, 0))
        self.assertEqual(set(doc["data"]), {"command"})
        # same command as the text mode, up to the per-run temp root and the hashed label
        label = json.loads(out.read_text(encoding="utf-8"))["label"]
        plain_label = json.loads((self.tmp / "p" / "launch.json").read_text(encoding="utf-8"))["label"]
        want = tel.unrooted(plain.stdout, self.tmp / "p" / "host", plain_label)
        got = tel.unrooted(doc["data"]["command"] + "\n", self.tmp / "j" / "host", label)
        self.assertEqual(got, want)

    def test_progress_lines_are_text_mode_only(self):
        """The `[gatedtrain] ...` progress lines (runner._log: run, tierb-collect, ext-collect) go to stderr in text
        mode only; with `--json` stderr stays empty. (The real `run` of the demo spec was checked by hand: ~4.5 min.)"""
        import contextlib
        import io
        from unittest import mock

        from gatedtrain import cli, runner

        def fake_run(spec, runs_root):
            runner._log("run r1: progress")
            return self.tmp / "r1", {"status": "PASS"}, "report text\n"

        for as_json in (False, True):
            out, err = io.StringIO(), io.StringIO()
            with mock.patch("gatedtrain.runner.run", fake_run), contextlib.redirect_stdout(out), \
                    contextlib.redirect_stderr(err):
                code = cli.main([*(["--json"] if as_json else []), "run", "spec.yaml"])
            self.assertEqual(code, EXIT_OK)
            if as_json:
                self.assertEqual(err.getvalue(), "")
                self.assertEqual(json.loads(out.getvalue())["data"]["report"], {"status": "PASS"})
            else:
                self.assertEqual(err.getvalue(), "[gatedtrain] run r1: progress\n")
                self.assertEqual(out.getvalue(), f"run_dir: {self.tmp / 'r1'}\nreport text\n")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            runner._log("after")                       # the flag is reset once main returns
        self.assertEqual(err.getvalue(), "[gatedtrain] after\n")

    def test_emit_and_result_json(self):
        r = Result("x", False, 3, {"a": 1}, [Problem("R", "f", "m", "x")], ["w"])
        self.assertEqual(result_json(r), {"schema": "gatedtrain.cli_result", "schema_version": 1, "command": "x",
                                          "ok": False, "exit": 3, "data": {"a": 1},
                                          "problems": [{"rule": "R", "field": "f", "message": "m", "fix": "x"}],
                                          "warnings": ["w"]})


if __name__ == "__main__":
    unittest.main()
