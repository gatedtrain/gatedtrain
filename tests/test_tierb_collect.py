"""H-6 `gatedtrain tierb-collect` on recorded fixture outputs (TIER-B-SPEC sections 1 and 3;
interim rules G-7, G-14 to G-17; review round decisions HIGH-2, LOW-2, LOW-3, LOW-4, LOW-6;
tests/README.md sections 11.7 and 11.12).

Each scenario is built on disk: the GSM8K fixture is exported with H-1, the registration and
spec are rendered from tests/fixtures/tierb templates, and the NeMo RL outputs and merged NeMo
Evaluator results (K = 2) are recorded fixtures built by _tierb. The NeMo RL outputs follow the
trainer.yaml profile (LOW-3): smoke = 10 steps of 2 prompts x 4 generations, save period 5;
long = 150 steps of 32 prompts x 8 generations, save period 25.

Expected gate: baseline reward 1 on prompts i % 3 == 0, candidate on i % 3 != 2, so the paired
per-prompt differences are 1 on a third of the 30 prompts and 0 elsewhere: Delta-hat = 1/3,
SD = 0.479, lower 97.5% bound ~ 0.16 > 0, observed MDE 0.245 < 0.30 -> PASS, not underpowered.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from _support import (EXIT_GATE_NOT_PASS, EXIT_OK, EXIT_REFUSED, SKEL_ROOT, canonical_bytes, count_lines, describe, expected_step_shards,
                      file_sha256, load_report, manifest_hash, parse_errors, parse_run_dir, run_cli)
from _tierb import (PROFILES, REVISION, baseline_reward, build_eval_run, build_merged, build_stage_out,
                    candidate_reward, heldout_rows, registration_seeds, render_inputs, run_export)

IMAGE_ID = "sha256:" + "ab" * 32
NEMO_RL_COMMIT = "4f39fd86289a2ffd5aaf5d202702f7fd7185c924"


class Scenario:
    def __init__(self, root: Path, profile: str = "smoke", stage_profile: str | None = None):
        self.root = root
        proc = run_export(root / "export", cwd=SKEL_ROOT)
        if proc.returncode != EXIT_OK:
            raise AssertionError("export-gsm8k (H-1) failed: " + describe(proc))
        self.paths = render_inputs(root, root / "export")
        (root / "trainer.yaml").write_text(
            (root / "trainer.yaml").read_text(encoding="utf-8").replace("profile: smoke", f"profile: {profile}"),
            encoding="utf-8")
        self.profile = PROFILES[stage_profile or profile]
        self.stage_out = self.build_stage(root / "out" / "rl")
        self.rows = heldout_rows(root / "export")
        self.seeds = registration_seeds(self.paths["registration"])
        self.eval_root = root / "out" / "eval"
        self.baseline = self.arm("baseline", baseline_reward)
        self.candidate = self.arm("candidate", candidate_reward)

    def build_stage(self, path: Path, **kw) -> Path:
        p = self.profile
        return build_stage_out(path, prompts_per_step=p["prompts_per_step"], gens=p["gens"], steps=p["steps"],
                               save_period=p["save_period"], **kw)

    def arm(self, name: str, reward_fn, root: Path | None = None, **kw) -> Path:
        return build_merged((root or self.eval_root) / name, self.rows, reward_fn=reward_fn, seeds=self.seeds, **kw)

    def collect(self, *, baseline: Path | None = None, candidate: Path | None = None, runs_root: Path | None = None,
                stage_out: Path | None = None):
        runs_root = runs_root or self.root / "runs"
        args = ["tierb-collect", str(self.paths["spec"]), "--stage-out", str(stage_out or self.stage_out),
                "--eval-baseline", str(baseline or self.baseline), "--eval-candidate", str(candidate or self.candidate),
                "--image-id", IMAGE_ID, "--nemo-rl-commit", NEMO_RL_COMMIT, "--runs-root", str(runs_root)]
        return runs_root, run_cli(args, cwd=self.root, timeout=300)


def no_run_dir(runs_root: Path) -> bool:
    return not (runs_root.exists() and any(p.is_dir() for p in runs_root.iterdir()))


class RefusalMixin:
    def assert_refused(self, runs_root: Path, proc, rule: str, field: str | None = None):
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = parse_errors(proc.stderr)
        match = [e for e in errs if e["rule"] == rule and (field is None or e["field"] == field)]
        self.assertTrue(match, f"expected error[{rule}]{' ' + field if field else ''}; got {errs}\n" + describe(proc))
        self.assertTrue(no_run_dir(runs_root), "no run directory may be created")
        self.assertNotIn("MISSING", proc.stdout)
        return match[0]


class TestCollectPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-collect-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)   # runs even if setUpClass fails
        cls.sc = Scenario(cls.tmp)
        cls.runs_root, cls.proc = cls.sc.collect()

    def run_dir(self) -> Path:
        self.assertEqual(self.proc.returncode, EXIT_OK, describe(self.proc))
        return parse_run_dir(self.proc.stdout)

    def test_fixture_exercises_comma_rule(self):
        """HIGH-2 (b): some heldout answers carry a thousands comma that the evaluator drops."""
        self.assertTrue(any("," in r["answer"] for r in self.sc.rows))

    def test_gate_passes(self):
        rep = load_report(self.run_dir())
        self.assertEqual(rep["status"], "PASS")
        self.assertAlmostEqual(rep["delta_hat"], 1 / 3, places=12)
        self.assertFalse(rep["underpowered"])
        self.assertEqual(rep["n_items"], 30)                       # R11A-4: item names (was n_prompts)
        self.assertEqual((rep["unpaired_baseline"], rep["unpaired_candidate"]), (0, 0))
        self.assertIs(rep["sd_zero"], False)

    def test_report_provenance(self):
        run_dir = self.run_dir()
        rep = load_report(run_dir)
        self.assertEqual(rep["tags"], ["prototype", "unpromotable", "registration: unsigned"])
        self.assertEqual(rep["dataset_revision"], REVISION, "G-17")
        self.assertEqual(rep["item_set_hash"], manifest_hash(self.tmp / "export" / "heldout"))   # R11A-4
        self.assertEqual(rep["registration_hash"], file_sha256(self.sc.paths["registration"]))
        self.assertEqual(rep["baseline_checkpoint_hash"], manifest_hash(self.sc.paths["base"]))
        final = self.sc.stage_out / "checkpoints" / "step_10"
        self.assertEqual(rep["candidate_checkpoint_hash"], manifest_hash(final))
        self.assertTrue(any("vLLM" in n and "batch" in n for n in rep["notes"]), "G-16 note in the report")
        text = (run_dir / "report" / "gate_report.txt").read_text(encoding="utf-8")
        self.assertIn(f"dataset_revision: {REVISION}", text)

    def test_lockfile(self):
        run_dir = self.run_dir()
        raw = (run_dir / "lockfile.json").read_bytes()
        lock = json.loads(raw)
        self.assertEqual(raw, canonical_bytes(lock))
        tr = lock["backend"]["trainer"]
        self.assertEqual((tr["name"], tr["image_id"], tr["nemo_rl_commit"]), ("nemo_rl", IMAGE_ID, NEMO_RL_COMMIT))
        self.assertEqual(lock["backend"]["evaluator"]["name"], "nemo_evaluator")
        self.assertEqual(lock["backend"]["evaluator"]["nemo_evaluator_commit"], "ea35ba3")
        self.assertEqual(lock["dataset_revision"], REVISION)
        self.assertEqual(lock["config_hash"], file_sha256(self.tmp / "trainer.yaml"))
        self.assertEqual(lock["nemo_rl_config_sha256"],
                         file_sha256(self.sc.stage_out / "checkpoints" / "step_10" / "config.yaml"))
        self.assertIs(lock["allow_list_checked"], False)

    def test_layout_like_run(self):
        run_dir = self.run_dir()
        for rel in ("lockfile.json", "plan.json", "registration/registration.yaml", "ledger/consumption.parquet",
                    "report/gate_report.json", "report/gate_report.txt", "eval/baseline/results.jsonl",
                    "eval/candidate/results.jsonl"):
            self.assertTrue((run_dir / rel).is_file(), rel)
        for name in ("baseline_checkpoint", "trained_checkpoint", "training_shards", "item_set", "registration",
                     "eval_baseline", "eval_candidate", "gate_report", "ledger"):
            doc = json.loads((run_dir / "manifests" / f"{name}.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["manifest_hash"], manifest_hash(Path(doc["root"])), name)

    def test_ledger_smoke(self):
        """G-7: 8 lines per step = 2 prompts x 4 generations; consumed_samples counts prompts."""
        run_dir = self.run_dir()
        rows = pq.read_table(run_dir / "ledger" / "consumption.parquet").to_pylist()
        expected = {(s, "shard-00000.jsonl") for s in range(1, 6)} | {(s, "shard-00001.jsonl") for s in range(6, 11)}
        self.assertEqual({(r["step"], r["data_ref"]["shard"]) for r in rows}, expected)
        self.assertEqual({r["precision"] for r in rows}, {"iteration"})

    def test_training_started_from_the_hashed_base_model(self):
        """L-1 pass case: step_<last>/config.yaml names a local path whose final component is the
        spec's base-model directory name (`base-model`)."""
        import yaml

        cfg = yaml.safe_load((self.sc.stage_out / "checkpoints" / "step_10" / "config.yaml").read_text())
        self.assertEqual(Path(cfg["policy"]["model_name"]).name, self.sc.paths["base"].name)
        self.run_dir()

    def test_impact_after_collect(self):
        run_dir = self.run_dir()
        ck = {s: manifest_hash(self.sc.stage_out / "checkpoints" / f"step_{s}") for s in (5, 10)}
        train = self.tmp / "export" / "train"

        def impact(shard):
            p = run_cli(["impact", str(train / shard), "--run-dir", str(run_dir), "--json"], cwd=self.tmp, timeout=60)
            self.assertEqual(p.returncode, EXIT_OK, describe(p))
            return {r["checkpoint_hash"] for r in json.loads(p.stdout)["results"]}

        self.assertEqual(impact("shard-00000.jsonl"), {ck[5], ck[10]})
        self.assertEqual(impact("shard-00001.jsonl"), {ck[10]}, "step_5 was saved before shard-00001 was read")
        self.assertEqual(impact("shard-00002.jsonl"), set())


class TestCollectLongProfile(unittest.TestCase):
    """G-7 with 8 generations per prompt: the long profile (150 steps, 32 x 8, save period 25).
    32-row batches over the 40-row fixture under drop_last (D-1): every step reads rows 0-31."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-collect-long-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = Scenario(cls.tmp, profile="long")
        cls.runs_root, cls.proc = cls.sc.collect()

    def test_ledger_with_eight_generations(self):
        self.assertEqual(self.proc.returncode, EXIT_OK, describe(self.proc))
        run_dir = parse_run_dir(self.proc.stdout)
        rows = pq.read_table(run_dir / "ledger" / "consumption.parquet").to_pylist()
        exp = self.sc.stage_out / "logs" / "exp_001"
        sizes = [(s, count_lines(exp / f"train_data_step{s}.jsonl")) for s in range(1, 151)]
        self.assertEqual({n for _, n in sizes}, {256})
        shards = [(p.name, count_lines(p)) for p in sorted((self.tmp / "export" / "train").glob("*.jsonl"))]
        expected = expected_step_shards(sizes, shards, samples_per_position=8)
        self.assertEqual(expected, {(s, f"shard-0000{k}.jsonl") for s in range(1, 151) for k in range(4)})
        self.assertEqual({(r["step"], r["data_ref"]["shard"]) for r in rows}, expected)
        self.assertEqual(load_report(run_dir)["status"], "PASS")


class TestCollectRefusals(RefusalMixin, unittest.TestCase):
    """Refusals happen before anything is created (exit 3, no run directory)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-collect-ref-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)   # runs even if setUpClass fails
        cls.sc = Scenario(cls.tmp)
        out = cls.tmp / "out"
        cls.empty = cls.sc.arm("baseline", baseline_reward, root=out / "eval-empty", empty=True)
        cls.unpaired = cls.sc.arm("candidate", candidate_reward, root=out / "eval-unpaired", drop={4})
        cls.unmerged = build_eval_run(out / "eval-raw" / "baseline" / "rep0", cls.sc.rows, reward_fn=baseline_reward,
                                      seed=cls.sc.seeds[0], model="baseline")
        cls.no_reward = cls.sc.arm("baseline", baseline_reward, root=out / "eval-no-reward", no_reward={6})
        # HIGH-2 variants: both arms see the same change (a different evaluated set or Hub revision).
        cls.drift = {}
        for key, kw in (("missing", {"drop": {7}}),
                        ("answer", {"answers": {5: "123456"}}),
                        ("question", {"questions": {5: "TEST 5: A different question from another revision?"}})):
            root = out / f"eval-{key}"
            cls.drift[key] = (cls.sc.arm("baseline", baseline_reward, root=root, **kw),
                              cls.sc.arm("candidate", candidate_reward, root=root, **kw))

    def refuse(self, key: str, rule: str, field: str | None = None, **kw):
        runs_root, proc = self.sc.collect(runs_root=self.tmp / f"runs-{key}", **kw)
        return self.assert_refused(runs_root, proc, rule, field)

    def test_silent_missing_regression_parent_dir(self):
        """Draft section 2 item 1: pointing at the parent directory used to give a silent MISSING gate."""
        self.refuse("parent", "TIERB-RESULTS", baseline=self.sc.eval_root)

    def test_unmerged_run_dir(self):
        """A raw NeMo Evaluator run (<run>/gsm8k/results.jsonl) is not a merged arm."""
        self.refuse("unmerged", "TIERB-RESULTS", baseline=self.unmerged)

    def test_missing_results(self):
        self.refuse("missing", "TIERB-RESULTS", candidate=self.tmp / "out" / "eval" / "nowhere")

    def test_empty_results(self):
        self.refuse("empty", "TIERB-RESULTS", baseline=self.empty)

    def test_record_without_reward(self):
        """LOW-6: a record without `reward` is refused, not read as 0."""
        self.refuse("no-reward", "TIERB-RESULTS", baseline=self.no_reward)

    # test_unpaired_beyond_tolerance (rev-9 TIERB-PAIRING refusal, interim rule G-18) is superseded by design
    # revision 10 (R10-8): unpaired items above `pairing.max_unpaired` make the gate MISSING in the report.
    # Replacement: test_r10_tierb.TestMissingnessDefault (tests/README.md section 11.15, R10A-3).

    def test_problem_missing_in_every_run(self):
        """HIGH-2 (a) refused this (TIERB-PROMPTS). Design revision 11 (R11-2, R11A-17) reverses it: the expected
        items are the registered held-out set, so problem 7, dropped from both arms, is two missing items per
        arm (K = 2, reason `absent`) and the gate is MISSING at the default bound, run written (superseded by
        test_r11_tierb.TestRegisteredItemSet)."""
        b, c = self.drift["missing"]
        runs_root, proc = self.sc.collect(runs_root=self.tmp / "runs-drift-missing", baseline=b, candidate=c)
        self.assertEqual(proc.returncode, EXIT_GATE_NOT_PASS, describe(proc))
        rep = load_report(parse_run_dir(proc.stdout))
        self.assertEqual(rep["status"], "MISSING")
        self.assertIn("missing_above_bound", rep["status_reasons"])
        self.assertEqual(rep["missing_reasons"], {"baseline": {"absent": 2}, "candidate": {"absent": 2}})

    def test_answer_changed(self):
        """HIGH-2 (b): expected_answer must equal the heldout answer without thousands commas
        (NeMo Evaluator `_prepare`, benchmarks/gsm8k.py:29-31) -- Hub revision drift."""
        b, c = self.drift["answer"]
        self.refuse("drift-answer", "TIERB-PROMPTS", baseline=b, candidate=c)

    def test_question_changed(self):
        """HIGH-2 (c): the heldout question must appear in the record's prompt."""
        b, c = self.drift["question"]
        self.refuse("drift-question", "TIERB-PROMPTS", baseline=b, candidate=c)


class TestCollectStageRefusals(RefusalMixin, unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-collect-st-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_second_attempt_directory(self):
        """A manual restart writes a second exp_* directory (draft risk R9): refuse, do not guess."""
        sc = Scenario(self.tmp)
        shutil.rmtree(sc.stage_out)
        sc.build_stage(sc.stage_out, exp_dirs=2)
        runs_root, proc = sc.collect()
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue(no_run_dir(runs_root))

    def test_consumed_samples_mismatch(self):
        sc = Scenario(self.tmp)
        shutil.rmtree(sc.stage_out)
        sc.build_stage(sc.stage_out, consumed_offset=1)
        runs_root, proc = sc.collect()
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue(no_run_dir(runs_root))

    def test_profile_mismatch(self):
        """LOW-3: a smoke run collected as long (trainer.yaml profile long; step config 10 x 2 x 4)."""
        sc = Scenario(self.tmp, profile="long", stage_profile="smoke")
        runs_root, proc = sc.collect()
        self.assert_refused(runs_root, proc, "TIERB-PROFILE")

    def test_artifact_pin_mismatch(self):
        """LOW-2: a non-registration `@sha256:` pin is checked as in Tier A preflight (04:80)."""
        sc = Scenario(self.tmp)
        spec = sc.paths["spec"]
        text = spec.read_text(encoding="utf-8")
        self.assertIn('uri: "file:export/heldout"', text)
        spec.write_text(text.replace('uri: "file:export/heldout"', 'uri: "file:export/heldout@sha256:' + "0" * 64 + '"'),
                        encoding="utf-8")
        runs_root, proc = sc.collect()
        self.assert_refused(runs_root, proc, "ARTIFACT-HASH", "artifacts.heldout.uri")

    def test_artifact_pin_match_accepted(self):
        """The same pin with the right hash is accepted (the check is not a blanket refusal)."""
        sc = Scenario(self.tmp)
        spec = sc.paths["spec"]
        good = manifest_hash(self.tmp / "export" / "heldout")
        spec.write_text(spec.read_text(encoding="utf-8").replace(
            'uri: "file:export/heldout"', f'uri: "file:export/heldout@{good}"'), encoding="utf-8")
        _, proc = sc.collect()
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_training_started_from_hub_id(self):
        """L-1: a Hub id means training did not start from the hashed base-model directory."""
        sc = Scenario(self.tmp)
        shutil.rmtree(sc.stage_out)
        sc.build_stage(sc.stage_out, model_name="Qwen/Qwen3-0.6B")
        runs_root, proc = sc.collect()
        self.assert_refused(runs_root, proc, "TIERB-BASE")

    def test_training_started_from_other_directory(self):
        """L-1: a local path whose final component is not the spec's base-model directory name."""
        sc = Scenario(self.tmp)
        shutil.rmtree(sc.stage_out)
        sc.build_stage(sc.stage_out, model_name="/work/tierb/other-model")
        runs_root, proc = sc.collect()
        self.assert_refused(runs_root, proc, "TIERB-BASE")

    def test_step_config_is_a_list(self):
        """Final re-review LOW-1: a step_<last>/config.yaml that parses to a YAML list (not a mapping)
        is refused (TIERB-* , TIERB-PROFILE expected), not an AttributeError, and nothing is created."""
        sc = Scenario(self.tmp)
        (sc.stage_out / "checkpoints" / "step_10" / "config.yaml").write_text("- grpo\n- policy\n", encoding="utf-8")
        runs_root, proc = sc.collect()
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        rules = {e["rule"] for e in parse_errors(proc.stderr)}
        self.assertTrue(any(r.startswith("TIERB-") for r in rules), describe(proc))
        self.assertNotIn("Traceback", proc.stderr)
        self.assertTrue(no_run_dir(runs_root), "no run directory may be created")

    def test_unreadable_checkpoint_file(self):
        """Final re-review LOW-3: a file under the trained checkpoint step_<last>/ that cannot be read
        is refused with exit 3 before the run directory is created (not a PermissionError later)."""
        import os

        if os.geteuid() == 0:
            self.skipTest("running as root: root can read mode-0000 files")
        sc = Scenario(self.tmp)
        target = sc.stage_out / "checkpoints" / "step_10" / "policy" / "tokenizer" / "tokenizer.json"
        mode = target.stat().st_mode
        os.chmod(target, 0)
        self.addCleanup(os.chmod, target, mode)     # restored before the temp dir is removed
        runs_root, proc = sc.collect()
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue(parse_errors(proc.stderr), "the refusal names the rule, the field and the fix")
        self.assertNotIn("Traceback", proc.stderr)
        self.assertTrue(no_run_dir(runs_root), "no run directory may be created")

    def test_unreadable_checkpoint_directory(self):
        """F-26: a directory under the trained checkpoint step_<last>/ that cannot be listed is refused
        with exactly TIERB-READ (exit 3) before the run directory is created; an rglob-based walk
        would skip it silently and hash a partial checkpoint."""
        import os

        if os.geteuid() == 0:
            self.skipTest("running as root: root can list mode-000 directories")
        sc = Scenario(self.tmp)
        target = sc.stage_out / "checkpoints" / "step_10" / "policy" / "tokenizer"
        mode = target.stat().st_mode
        os.chmod(target, 0)
        self.addCleanup(os.chmod, target, mode)
        runs_root, proc = sc.collect()
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        rules = [e["rule"] for e in parse_errors(proc.stderr)]
        self.assertTrue(rules, describe(proc))
        self.assertEqual(set(rules), {"TIERB-READ"}, describe(proc))
        self.assertNotIn("Traceback", proc.stderr)
        self.assertTrue(no_run_dir(runs_root), "no run directory may be created")

    def test_baseline_with_hf_cache_dir(self):
        """LOW-4: `huggingface-cli download --local-dir` leaves `.cache/` metadata in the base model."""
        sc = Scenario(self.tmp)
        (sc.paths["base"] / ".cache" / "huggingface").mkdir(parents=True)
        (sc.paths["base"] / ".cache" / "huggingface" / "download.lock").write_text("", encoding="utf-8")
        runs_root, proc = sc.collect()
        err = self.assert_refused(runs_root, proc, "TIERB-BASELINE")
        self.assertIn(".cache", err["fix"])
        self.assertRegex(err["fix"], r"(?i)delete|remove")


if __name__ == "__main__":
    unittest.main()
