"""G-21 fix B: the registration selects NeMo Evaluator's built-in `numeric_match` scorer and the
Tier B gate reads that scorer's reward (interim rule G-21, chosen by the maintainers
2026-11-11; tests/README.md section 11.14).

Facts at NeMo Evaluator ea35ba3 (a clone named by GATEDTRAIN_NEMO_EVALUATOR_DIR, commit
ea35ba36756abd59de201f60af8a2feba014744d), re-checked by TestScorerCitations:
- BenchmarkConfig accepts `scoring: ScoringConfig` (src/nemo_evaluator/config/benchmarks.py:63);
  `ScoringConfig.metrics` is a list; a function scorer is `{type: scorer, name: <registered name>}`
  (config/scoring.py:24-30, :160-184). `include_defaults` and `primary` are in the schema but are not
  consumed at ea35ba3, so nothing here relies on them.
- `numeric_match` is in the scorer registry (scoring/__init__.py:38-44); it compares the last number in
  the response, commas removed, after stripping trailing zeros and a trailing dot on both sides
  (scoring/pattern.py:39-43), so 10 and 1 match.
- NeMo Evaluator's orchestration layer passes scorer-type metric names as extra scorers (orchestration/orchestrator.py:1050,
  :1064). They do not replace the top-level `reward`; each writes
  `scoring_details["scorer:<name>"] = {correct, extracted, reward}` (engine/eval_loop.py:704-721) and the
  record gets `scorer_rewards = {"scorer:<name>": 0.0 | 1.0}` (eval_loop.py:785-794).
- The composed-config snapshot `<run>/full_config.yaml` is the raw config dumped as written
  (config/snapshot.py:72-81), so a run made with the scorer declares it there.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from _support import (DEMO_DIR, EXIT_OK, EXIT_REFUSED, NEMO_EVALUATOR_DIR_ENV, SKEL_ROOT, canonical_bytes, describe,
                      env_dir, file_sha256, load_report, manifest_hash, parse_errors, parse_run_dir, read_jsonl, run_cli)
from _tierb import (PROFILES, TIERB, baseline_reward, build_eval_run, build_merged, build_stage_out,
                    candidate_reward, heldout_rows, registration_seeds, render_inputs, run_export, snapshot_text,
                    with_scorer)
from test_tierb_merge import SEEDS, MergeCase
from test_tierb_spec import reference_export

SCORER = "numeric_match"
SCORER_KEY = f"scorer:{SCORER}"
SCORING_BLOCK = {"metrics": [{"type": "scorer", "name": SCORER}]}
GOLDEN = TIERB / "golden"
# GATEDTRAIN_NEMO_EVALUATOR_DIR; the class skips if it is unset or the clone is absent.
NEMO_EVAL_CLONE = env_dir(NEMO_EVALUATOR_DIR_ENV)
NEMO_EVAL_SRC = NEMO_EVAL_CLONE / "src" / "nemo_evaluator"
IMAGE_ID = "sha256:" + "ab" * 32
EVALUATOR_ID = "gsm8k.nemo_evaluator@0.1"
# Decision 4 (maintainers, 2026-11-11): with no scorer registered, evaluator_hash stays today's value,
# SHA-256 of the canonical JSON {"benchmark", "evaluator", "nemo_evaluator_commit"} (observed before G-21).
NO_SCORER_EVALUATOR_HASH = "sha256:032ad6d4e93ce7d39a33641679b6bec9d534df21ba7af4b5738858531648739e"
TODAY_LOCK_KEYS = {"schema", "schema_version", "tags", "run_id", "spec_sha256", "registration_sha256", "config_hash",
                   "nemo_rl_config_sha256", "allow_list_checked", "backend", "dataset_revision",
                   "analysis_code_version", "python_version", "artifacts"}
TODAY_LOCK_EVALUATOR = {"name": "nemo_evaluator", "nemo_evaluator_commit": "ea35ba3", "uses": EVALUATOR_ID}
NEMO_RL_COMMIT = "4f39fd86289a2ffd5aaf5d202702f7fd7185c924"


def zero_reward(i: int) -> float:
    """The G-21 smoke situation: the default `answer_line` scorer gives 0 everywhere."""
    return 0.0


def plain_registration_text(prompt_set_hash: str) -> str:
    text = (TIERB / "registration.template.yaml").read_text(encoding="utf-8")
    return text.replace("{prompt_set_hash}", prompt_set_hash)


def scorer_errors(proc) -> list[dict[str, str]]:
    return [e for e in parse_errors(proc.stderr)
            if e["rule"] == "REG-SCORER" and e["field"] == "registration.evaluators[0].scorer"]


# ---------------------------------------------------------------------------
# Requirement 1: registration field evaluators[0].scorer
# ---------------------------------------------------------------------------

class TestRegistrationScorer(unittest.TestCase):
    """`evaluators[0].scorer` is optional; at ea35ba3 the prototype allowlist is {numeric_match}.
    Absent or null keeps today's behaviour. Anything else is refused with
    `error[REG-SCORER] registration.evaluators[0].scorer: ...`, exit 3, by `validate` (spec loading)
    and by every command that loads the registration (nel-config, merge-eval)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-scorer-reg-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.export = reference_export(self.tmp)

    def validate_with(self, scorer):
        render_inputs(self.tmp, self.export, scorer=scorer, set_scorer=True)
        return run_cli(["validate", "spec.yaml"], cwd=self.tmp, timeout=60)

    def test_numeric_match_accepted(self):
        proc = self.validate_with(SCORER)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_null_accepted(self):
        """`scorer: null` is the same as no scorer."""
        proc = self.validate_with(None)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_other_scorer_names_refused(self):
        """Registered NeMo Evaluator scorers other than numeric_match (scoring/__init__.py:38-44), and
        unknown names, are outside the prototype allowlist."""
        for name in ("answer_line", "exact_match", "fuzzy_match", "Numeric_Match", "not_a_scorer", ""):
            with self.subTest(scorer=name):
                proc = self.validate_with(name)
                self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
                self.assertTrue(scorer_errors(proc), describe(proc))

    def test_non_string_refused(self):
        for value in (1, True, [SCORER], {"name": SCORER}):
            with self.subTest(scorer=value):
                proc = self.validate_with(value)
                self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
                self.assertTrue(scorer_errors(proc), describe(proc))

    def test_refusal_names_the_allowlist(self):
        proc = self.validate_with("answer_line")
        errs = scorer_errors(proc)
        self.assertTrue(errs, describe(proc))
        self.assertIn(SCORER, errs[0]["fix"])

    def _bad_registration(self) -> Path:
        reg = self.tmp / "bad-registration.yaml"
        reg.write_text(with_scorer(plain_registration_text("sha256:" + "0" * 64), "answer_line"), encoding="utf-8")
        return reg

    def test_nel_config_refuses_unknown_scorer(self):
        out = self.tmp / "cfg"
        proc = run_cli(["nel-config", "--registration", str(self._bad_registration()), "--model-url",
                        "http://localhost:8000/v1", "--model-id", "baseline", "--eval-root", "/work/out/eval/baseline",
                        "--out", str(out)], cwd=SKEL_ROOT, timeout=60)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue(scorer_errors(proc), describe(proc))
        self.assertFalse(out.exists())

    def test_merge_eval_refuses_unknown_scorer(self):
        out = self.tmp / "merged"
        proc = run_cli(["merge-eval", "--registration", str(self._bad_registration()), "--out", str(out),
                        str(self.tmp / "rep0"), str(self.tmp / "rep1")], cwd=SKEL_ROOT, timeout=60)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue(scorer_errors(proc), describe(proc))
        self.assertFalse(out.exists())


# ---------------------------------------------------------------------------
# Requirement 2: nel-config emits the scoring block
# ---------------------------------------------------------------------------

class TestNelConfigScorer(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-scorer-nel-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.plain = plain_registration_text("sha256:" + "0" * 64)

    def render(self, reg_text: str, name: str) -> Path:
        reg = self.tmp / f"{name}.yaml"
        reg.write_text(reg_text, encoding="utf-8")
        out = self.tmp / name
        proc = run_cli(["nel-config", "--registration", str(reg), "--model-url", "http://localhost:8000/v1",
                        "--model-id", "baseline", "--eval-root", "/work/out/eval/baseline", "--out", str(out)],
                       cwd=SKEL_ROOT, timeout=60)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["nel-gsm8k-rep0.yaml", "nel-gsm8k-rep1.yaml"])
        return out

    def test_scoring_block_emitted(self):
        """Each nel-gsm8k-rep<r>.yaml has, under its single gsm8k benchmark,
        `scoring: {metrics: [{type: scorer, name: numeric_match}]}` (config/benchmarks.py:63;
        config/scoring.py:24-30, :160-184); `include_defaults` and `primary` are not emitted."""
        out = self.render(with_scorer(self.plain, SCORER), "scored")
        for r in (0, 1):
            with self.subTest(repeat=r):
                got = yaml.safe_load((out / f"nel-gsm8k-rep{r}.yaml").read_text(encoding="utf-8"))
                benches = got["benchmarks"]
                self.assertEqual(len(benches), 1)
                self.assertEqual(benches[0]["name"], "gsm8k")
                self.assertEqual(benches[0].get("scoring"), SCORING_BLOCK)

    def test_everything_else_unchanged(self):
        """Removing the scoring block gives the golden config exactly."""
        out = self.render(with_scorer(self.plain, SCORER), "scored")
        for r in (0, 1):
            with self.subTest(repeat=r):
                got = yaml.safe_load((out / f"nel-gsm8k-rep{r}.yaml").read_text(encoding="utf-8"))
                got["benchmarks"][0].pop("scoring", None)
                want = yaml.safe_load((GOLDEN / f"nel-gsm8k-rep{r}.yaml").read_text(encoding="utf-8"))
                self.assertEqual(got, want)

    def test_null_scorer_is_byte_identical_to_no_scorer(self):
        """`scorer: null` and an absent scorer write the same bytes, with no scoring block; the absent
        case is pinned against the golden files by test_tierb_nel_config.TestNelConfig.test_golden."""
        absent = self.render(self.plain, "absent")
        null = self.render(with_scorer(self.plain, None), "null")
        for r in (0, 1):
            with self.subTest(repeat=r):
                a = (absent / f"nel-gsm8k-rep{r}.yaml").read_bytes()
                self.assertEqual((null / f"nel-gsm8k-rep{r}.yaml").read_bytes(), a)
                self.assertNotIn(b"scoring", a)


# ---------------------------------------------------------------------------
# Requirement 3: merge-eval keeps scorer_rewards
# ---------------------------------------------------------------------------

class TestMergeKeepsScorerRewards(MergeCase):
    def test_scorer_fields_unchanged(self):
        """Merged records keep `scorer_rewards` and `scoring_details["scorer:numeric_match"]` exactly
        as each source run wrote them (eval_loop.py:704-721, :785-794)."""
        self.reg.write_text(with_scorer(self.reg.read_text(encoding="utf-8"), SCORER), encoding="utf-8")
        kw = {"scorer": SCORER, "scorer_fn": candidate_reward}
        runs = self.runs({0: dict(kw), 1: dict(kw)})
        out, proc = self.merge(runs)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        merged = read_jsonl(out / "results.jsonl")
        self.assertEqual(len(merged), len(self.rows) * len(SEEDS))
        sources = {r: {x["problem_idx"]: x for x in read_jsonl(run / "gsm8k" / "results.jsonl")}
                   for r, run in enumerate(runs)}
        for rec in merged:
            src = sources[rec["repeat"]][rec["problem_idx"]]
            self.assertEqual(rec["scorer_rewards"], {SCORER_KEY: candidate_reward(rec["problem_idx"])})
            self.assertEqual(rec["scorer_rewards"], src["scorer_rewards"])
            self.assertEqual(rec["scoring_details"], src["scoring_details"])
            self.assertEqual(rec["reward"], src["reward"])


class TestMergeCopiesConfigs(MergeCase):
    """Decision 1 (maintainers, 2026-11-11): merge-eval copies each input run's full_config.yaml to
    `OUT/configs/rep<r>.yaml` (r in registered seed order) and records it in merge.json as
    `sources[r].config = "configs/rep<r>.yaml"` and `sources[r].config_sha256`."""

    def test_configs_copied(self):
        runs = self.runs()
        out, proc = self.merge(runs)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertTrue((out / "configs").is_dir(), "merge-eval writes OUT/configs/ (decision 1)")
        self.assertEqual(sorted(p.name for p in (out / "configs").iterdir()), ["rep0.yaml", "rep1.yaml"])
        doc = json.loads((out / "merge.json").read_text(encoding="utf-8"))
        for r, run in enumerate(runs):
            with self.subTest(repeat=r):
                copied = out / "configs" / f"rep{r}.yaml"
                self.assertEqual(copied.read_bytes(), (run / "full_config.yaml").read_bytes())
                self.assertEqual(doc["sources"][r]["config"], f"configs/rep{r}.yaml")
                self.assertEqual(doc["sources"][r]["config_sha256"], file_sha256(copied))

    def test_configs_copied_with_scorer(self):
        self.reg.write_text(with_scorer(self.reg.read_text(encoding="utf-8"), SCORER), encoding="utf-8")
        kw = {"scorer": SCORER, "scorer_fn": candidate_reward}
        runs = self.runs({0: dict(kw), 1: dict(kw)})
        out, proc = self.merge(runs)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        for r, run in enumerate(runs):
            copied = out / "configs" / f"rep{r}.yaml"
            self.assertTrue(copied.is_file(), f"{copied} (decision 1)")
            self.assertEqual(copied.read_bytes(), (run / "full_config.yaml").read_bytes())


class TestMergeScorerRefusals(MergeCase):
    """Decision 2 (maintainers, 2026-11-11): with scorer X registered, merge-eval refuses (MERGE-EVAL,
    field runs[r], exit 3, OUT not created) any run whose full_config.yaml does not declare
    `{type: scorer, name: X}` under the gsm8k benchmark's `scoring.metrics`."""

    def setUp(self):
        super().setUp()
        self.reg.write_text(with_scorer(self.reg.read_text(encoding="utf-8"), SCORER), encoding="utf-8")
        self.kw = {"scorer": SCORER, "scorer_fn": candidate_reward}

    def test_run_without_scorer(self):
        self.assert_refused(self.runs({0: dict(self.kw), 1: {**self.kw, "declared_scorers": []}}), "runs[1]")

    def test_run_with_another_scorer(self):
        self.assert_refused(self.runs({0: {**self.kw, "declared_scorers": ["exact_match"]}, 1: dict(self.kw)}),
                            "runs[0]")

    def test_no_run_declares_scorer(self):
        out, proc = self.merge(self.runs({0: {**self.kw, "declared_scorers": []},
                                           1: {**self.kw, "declared_scorers": []}}))
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        fields = {e["field"] for e in parse_errors(proc.stderr) if e["rule"] == "MERGE-EVAL"}
        self.assertEqual(fields & {"runs[0]", "runs[1]"}, {"runs[0]", "runs[1]"}, describe(proc))
        self.assertFalse(out.exists())

    def test_no_scorer_registered_no_check(self):
        """Without a registered scorer, runs that declare none merge as today."""
        self.reg.write_text(self.reg.read_text(encoding="utf-8").replace(f"    scorer: {SCORER}\n", ""),
                            encoding="utf-8")
        out, proc = self.merge(self.runs())
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))


# ---------------------------------------------------------------------------
# Requirements 4-6: tierb-collect reads the registered scorer
# ---------------------------------------------------------------------------

class ScorerScenario:
    """A Tier B scenario whose arms are built the way the runbook builds them: K raw NeMo Evaluator runs
    per arm (recorded fixtures) merged with the real `gatedtrain merge-eval`. Top-level `reward` is 0 on
    every record (the G-21 situation); `scorer_rewards["scorer:numeric_match"]` follows baseline_reward /
    candidate_reward, which gives the same expected gate as test_tierb_collect.TestCollectPass:
    Delta-hat = 1/3, PASS."""

    def __init__(self, root: Path, *, scorer: str | None):
        self.root = root
        proc = run_export(root / "export", cwd=SKEL_ROOT)
        if proc.returncode != EXIT_OK:
            raise AssertionError("export-gsm8k (H-1) failed: " + describe(proc))
        self.paths = render_inputs(root, root / "export", scorer=scorer, set_scorer=scorer is not None)
        self.plain_reg = root / "registration-plain.yaml"
        self.plain_reg.write_text(plain_registration_text(manifest_hash(root / "export" / "heldout")), encoding="utf-8")
        p = PROFILES["smoke"]
        self.stage_out = build_stage_out(root / "out" / "rl", prompts_per_step=p["prompts_per_step"], gens=p["gens"],
                                         steps=p["steps"], save_period=p["save_period"])
        self.rows = heldout_rows(root / "export")
        self.seeds = registration_seeds(self.paths["registration"])

    def arm(self, name: str, scorer_fn, tag: str, *, registration: Path | None = None,
            per_run: dict | None = None, **kw) -> Path:
        """K raw runs under out/raw-<tag>/<name>/rep<r>, merged into out/eval-<tag>/<name>."""
        per_run = per_run or {}
        runs = []
        for r, seed in enumerate(self.seeds):
            args = {"reward_fn": zero_reward, "seed": seed, "model": name, "scorer": SCORER, "scorer_fn": scorer_fn,
                    **kw, **per_run.get(r, {})}
            runs.append(build_eval_run(self.root / "out" / f"raw-{tag}" / name / f"rep{r}", self.rows, **args))
        out = self.root / "out" / f"eval-{tag}" / name
        proc = run_cli(["merge-eval", "--registration", str(registration or self.paths["registration"]),
                        "--out", str(out), *map(str, runs)], cwd=SKEL_ROOT, timeout=60)
        self.last_merge = proc
        return out

    def collect(self, baseline: Path, candidate: Path, runs_root: Path):
        args = ["tierb-collect", str(self.paths["spec"]), "--stage-out", str(self.stage_out),
                "--eval-baseline", str(baseline), "--eval-candidate", str(candidate),
                "--image-id", IMAGE_ID, "--nemo-rl-commit", NEMO_RL_COMMIT, "--runs-root", str(runs_root)]
        return run_cli(args, cwd=self.root, timeout=300)


def hand_built_arm(root: Path, rows, seeds, *, scorer_fn, model: str, declared_scorers: list[str]) -> Path:
    """A merged arm written without merge-eval, in decision 1's layout: results.jsonl, configs/rep<r>.yaml
    (composed-config snapshots declaring `declared_scorers`) and a merge.json that lists them."""
    arm = build_merged(root, rows, reward_fn=zero_reward, seeds=seeds, scorer=SCORER, scorer_fn=scorer_fn)
    (arm / "configs").mkdir()
    sources = []
    for r, seed in enumerate(seeds):
        cfg = arm / "configs" / f"rep{r}.yaml"
        cfg.write_text(snapshot_text(seed, 1, model, f"/gone/rep{r}", scorers=declared_scorers), encoding="utf-8")
        sources.append({"repeat": r, "seed": seed, "run_dir": f"/gone/rep{r}",
                        "results_sha256": "sha256:" + "0" * 64, "config": f"configs/rep{r}.yaml",
                        "config_sha256": file_sha256(cfg)})
    (arm / "merge.json").write_text(json.dumps({"schema": "gatedtrain.merge_eval", "schema_version": "1.0",
                                                "tags": ["prototype", "unpromotable", "registration: unsigned"],
                                                "sources": sources, "notes": []}), encoding="utf-8")
    return arm


def no_run_dir(runs_root: Path) -> bool:
    return not (runs_root.exists() and any(p.is_dir() for p in runs_root.iterdir()))


def numeric_match_note(notes: list[str]) -> list[str]:
    """Decision 7: the note names numeric_match and the trailing-zero effect (integers without a trailing
    dot: "10" vs "1" match, "10." does not), and gives 25/5,276 = 0.47% (G-21)."""
    return [n for n in notes if SCORER in n and re.search(r"(?i)trailing zero", n)
            and "0.47%" in n and "25/5,276" in n and "G-21" in n]


class TestCollectWithScorer(unittest.TestCase):
    """Requirements 4, 5 and 6: the gate is computed from scorer_rewards["scorer:numeric_match"]."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-scorer-collect-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = ScorerScenario(cls.tmp, scorer=SCORER)
        cls.baseline = cls.sc.arm("baseline", baseline_reward, "ok")
        cls.merge_b = cls.sc.last_merge
        cls.candidate = cls.sc.arm("candidate", candidate_reward, "ok")
        cls.merge_c = cls.sc.last_merge
        cls.proc = cls.sc.collect(cls.baseline, cls.candidate, cls.tmp / "runs")

    def run_dir(self) -> Path:
        self.assertEqual(self.merge_b.returncode, EXIT_OK, describe(self.merge_b))
        self.assertEqual(self.merge_c.returncode, EXIT_OK, describe(self.merge_c))
        self.assertEqual(self.proc.returncode, EXIT_OK, describe(self.proc))
        return parse_run_dir(self.proc.stdout)

    def test_fixture_reward_is_zero_everywhere(self):
        """Guard on the fixture itself: top-level reward carries no signal, scorer_rewards do."""
        for arm in (self.baseline, self.candidate):
            recs = read_jsonl(arm / "results.jsonl")
            self.assertEqual({r["reward"] for r in recs}, {0.0})
            self.assertEqual({tuple(r["scorer_rewards"]) for r in recs}, {(SCORER_KEY,)})

    def test_gate_uses_scorer_rewards(self):
        """Requirement 5: reward is all 0 but the scorer values differ between arms, and the gate is
        the scorer's: Delta-hat = 1/3, PASS (with `reward` it would be Delta-hat = 0, see the control)."""
        rep = load_report(self.run_dir())
        self.assertEqual(rep["status"], "PASS")
        self.assertAlmostEqual(rep["delta_hat"], 1 / 3, places=12)
        self.assertAlmostEqual(rep["baseline_score"], 10 / 30, places=12)
        self.assertAlmostEqual(rep["candidate_score"], 20 / 30, places=12)
        self.assertEqual(rep["n_items"], 30)                       # R11A-4: item names (was n_prompts)
        self.assertIs(rep["sd_zero"], False)
        self.assertFalse(rep["underpowered"])

    def test_metric_source_line(self):
        text = (self.run_dir() / "report" / "gate_report.txt").read_text(encoding="utf-8")
        self.assertIn(f"metric_source: scorer_rewards[{SCORER_KEY}]", text.splitlines())

    def test_numeric_match_note(self):
        """Requirement 6: the report says numeric_match compares the last number after stripping trailing
        zeros (pattern.py:39-43), so 10 and 1 match, with the observed false-positive rate 25/5,276 = 0.47%
        on the smoke baseline (G-21)."""
        run_dir = self.run_dir()
        rep = load_report(run_dir)
        self.assertTrue(numeric_match_note(rep.get("notes", [])), rep.get("notes"))
        self.assertTrue(any("vLLM" in n and "batch" in n for n in rep["notes"]), "the G-16 note stays")
        text = (run_dir / "report" / "gate_report.txt").read_text(encoding="utf-8")
        self.assertIn(numeric_match_note(rep["notes"])[0], text)

    def test_evaluator_identity_covers_scorer(self):
        """Decision 4: the scorer is part of the evaluator identity."""
        run_dir = self.run_dir()
        rep = load_report(run_dir)
        self.assertNotEqual(rep["evaluator_hash"], NO_SCORER_EVALUATOR_HASH)
        lock = json.loads((run_dir / "lockfile.json").read_bytes())
        self.assertEqual(lock["backend"]["evaluator"], {**TODAY_LOCK_EVALUATOR, "scorer": SCORER})

    def test_arm_holds_copied_configs(self):
        """Decision 1: each merged arm holds configs/rep<r>.yaml, byte-equal to the raw runs' snapshots."""
        for name, arm in (("baseline", self.baseline), ("candidate", self.candidate)):
            for r in range(len(self.sc.seeds)):
                with self.subTest(arm=name, repeat=r):
                    raw = self.tmp / "out" / "raw-ok" / name / f"rep{r}" / "full_config.yaml"
                    copied = arm / "configs" / f"rep{r}.yaml"
                    self.assertTrue(copied.is_file(), f"{copied} (decision 1)")
                    self.assertEqual(copied.read_bytes(), raw.read_bytes())

    def test_collect_after_raw_runs_moved(self):
        """Decision 1: collect reads the copied configs, not merge.json's absolute run_dir paths, so it
        still works after the raw rep<r> directories are moved away."""
        raw, moved = self.tmp / "out" / "raw-ok", self.tmp / "moved-raw-ok"
        shutil.move(str(raw), str(moved))
        self.addCleanup(shutil.move, str(moved), str(raw))
        proc = self.sc.collect(self.baseline, self.candidate, self.tmp / "runs-moved")
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertEqual(load_report(parse_run_dir(proc.stdout))["status"], "PASS")

    def test_results_copied_unchanged(self):
        """The run directory keeps the evaluator's records as they are; only the metric read changes.
        Independent of the gate outcome (exit 0 or 1)."""
        self.assertIn(self.proc.returncode, (0, 1), describe(self.proc))
        run_dir = parse_run_dir(self.proc.stdout)
        for arm, src in (("baseline", self.baseline), ("candidate", self.candidate)):
            self.assertEqual((run_dir / "eval" / arm / "results.jsonl").read_bytes(),
                             (src / "results.jsonl").read_bytes(), arm)


class TestCollectWithoutScorerControl(unittest.TestCase):
    """Control for requirement 5 and the no-scorer half of requirement 4: the same records, but the
    registration names no scorer, so the gate reads the top-level `reward` (all 0) as today."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-scorer-control-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = ScorerScenario(cls.tmp, scorer=None)
        cls.baseline = cls.sc.arm("baseline", baseline_reward, "ok")
        cls.merge_b = cls.sc.last_merge
        cls.candidate = cls.sc.arm("candidate", candidate_reward, "ok")
        cls.merge_c = cls.sc.last_merge
        cls.proc = cls.sc.collect(cls.baseline, cls.candidate, cls.tmp / "runs")

    def run_dir(self) -> Path:
        self.assertEqual(self.merge_b.returncode, EXIT_OK, describe(self.merge_b))
        self.assertEqual(self.merge_c.returncode, EXIT_OK, describe(self.merge_c))
        self.assertIn(self.proc.returncode, (0, 1), describe(self.proc))
        return parse_run_dir(self.proc.stdout)

    def test_gate_reads_reward(self):
        rep = load_report(self.run_dir())
        self.assertNotEqual(rep["status"], "PASS")
        self.assertEqual(rep["delta_hat"], 0.0)
        self.assertEqual((rep["baseline_score"], rep["candidate_score"]), (0.0, 0.0))

    def test_metric_source_reward(self):
        text = (self.run_dir() / "report" / "gate_report.txt").read_text(encoding="utf-8")
        self.assertIn("metric_source: reward", text.splitlines())

    def test_no_numeric_match_note(self):
        rep = load_report(self.run_dir())
        self.assertFalse([n for n in rep.get("notes", []) if SCORER in n], rep.get("notes"))

    def test_evaluator_hash_unchanged(self):
        """Decision 4: no scorer, so evaluator_hash is today's value (pinned, and recomputed here)."""
        want = "sha256:" + __import__("hashlib").sha256(canonical_bytes(
            {"evaluator": EVALUATOR_ID, "benchmark": "gsm8k", "nemo_evaluator_commit": "ea35ba3"})).hexdigest()
        self.assertEqual(want, NO_SCORER_EVALUATOR_HASH)
        self.assertEqual(load_report(self.run_dir())["evaluator_hash"], NO_SCORER_EVALUATOR_HASH)

    def test_lockfile_unchanged(self):
        """Decision 4: no scorer, so the lockfile has today's keys and today's backend.evaluator."""
        lock = json.loads((self.run_dir() / "lockfile.json").read_bytes())
        self.assertEqual(set(lock), TODAY_LOCK_KEYS)
        self.assertEqual(set(lock["backend"]), {"trainer", "evaluator"})
        self.assertEqual(lock["backend"]["evaluator"], TODAY_LOCK_EVALUATOR)


class TestCollectScorerRefusals(unittest.TestCase):
    """Requirement 4 refusals, exit 3, no run directory:
    - TIERB-RESULTS when a record lacks scorer_rewards["scorer:numeric_match"] (never read as 0;
      mirrors LOW-6);
    - TIERB-SCORER when an arm's per-run full_config.yaml files do not declare the registered scorer
      under the gsm8k benchmark's `scoring.metrics`, or when collect cannot find them."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-scorer-refuse-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = ScorerScenario(cls.tmp, scorer=SCORER)
        cls.good_b = cls.sc.arm("baseline", baseline_reward, "ok")
        cls.good_c = cls.sc.arm("candidate", candidate_reward, "ok")
        cls.missing = cls.sc.arm("baseline", baseline_reward, "missing", scorer_missing={6})
        cls.missing_merge = cls.sc.last_merge
        # Runs made without the scorer, merged under a registration that names none (merge-eval has
        # nothing to check), then gated under the scorer registration.
        cls.undeclared = cls.sc.arm("baseline", baseline_reward, "undeclared", declared_scorers=[],
                                    registration=cls.sc.plain_reg)
        cls.undeclared_merge = cls.sc.last_merge
        cls.other = cls.sc.arm("candidate", candidate_reward, "other", declared_scorers=["exact_match"],
                               registration=cls.sc.plain_reg)
        cls.other_merge = cls.sc.last_merge
        cls.one_run = cls.sc.arm("baseline", baseline_reward, "one-run", per_run={1: {"declared_scorers": []}})
        cls.one_run_merge = cls.sc.last_merge
        cls.errored = cls.sc.arm("baseline", baseline_reward, "errored", scorer_error={4})
        cls.errored_merge = cls.sc.last_merge
        # Decision 2 defence in depth: an arm in merge-eval's layout whose copied configs lack the scorer.
        cls.hand = hand_built_arm(cls.tmp / "out" / "eval-hand" / "baseline", cls.sc.rows, cls.sc.seeds,
                                  scorer_fn=baseline_reward, model="baseline", declared_scorers=[])
        # An arm with merged records but no copied configs (no configs/, no merge.json).
        cls.bare = build_merged(cls.tmp / "out" / "eval-bare" / "baseline", cls.sc.rows, reward_fn=zero_reward,
                                seeds=cls.sc.seeds, scorer=SCORER, scorer_fn=baseline_reward)

    def assert_refused(self, key: str, rule: str, field: str, **arms):
        runs_root = self.tmp / f"runs-{key}"
        proc = self.sc.collect(arms.get("baseline", self.good_b), arms.get("candidate", self.good_c), runs_root)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = parse_errors(proc.stderr)
        self.assertTrue([e for e in errs if e["rule"] == rule and e["field"] == field],
                        f"expected error[{rule}] {field}; got {errs}\n" + describe(proc))
        self.assertTrue(no_run_dir(runs_root), "no run directory may be created")
        self.assertNotIn("Traceback", proc.stderr)
        return proc

    def test_good_arms_accepted(self):
        """The refusals below are not blanket refusals: the good arms collect (exit 0, PASS)."""
        runs_root = self.tmp / "runs-good"
        proc = self.sc.collect(self.good_b, self.good_c, runs_root)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_record_without_scorer_reward(self):
        self.assertEqual(self.missing_merge.returncode, EXIT_OK, describe(self.missing_merge))
        self.assert_refused("missing", "TIERB-RESULTS", "--eval-baseline", baseline=self.missing)

    def test_runs_without_scorer(self):
        self.assertEqual(self.undeclared_merge.returncode, EXIT_OK, describe(self.undeclared_merge))
        self.assert_refused("undeclared", "TIERB-SCORER", "--eval-baseline", baseline=self.undeclared)

    def test_runs_with_another_scorer(self):
        self.assertEqual(self.other_merge.returncode, EXIT_OK, describe(self.other_merge))
        self.assert_refused("other", "TIERB-SCORER", "--eval-candidate", candidate=self.other)

    def test_one_run_without_scorer(self):
        """Decision 2: runs[1] of the baseline was made without the scorer; merge-eval refuses it early
        (MERGE-EVAL runs[1], exit 3) and writes no arm."""
        self.assertEqual(self.one_run_merge.returncode, EXIT_REFUSED, describe(self.one_run_merge))
        errs = parse_errors(self.one_run_merge.stderr)
        self.assertTrue([e for e in errs if e["rule"] == "MERGE-EVAL" and e["field"] == "runs[1]"],
                        describe(self.one_run_merge))
        self.assertFalse(self.one_run.exists())

    def test_hand_built_arm_configs_lack_scorer(self):
        """Decision 2, defence in depth: collect keeps its own check on the copied configs."""
        self.assert_refused("hand", "TIERB-SCORER", "--eval-baseline", baseline=self.hand)

    def test_scorer_error_record(self):
        """Decision 3: a record whose scoring_details["scorer:numeric_match"] has an `error` key
        (eval_loop.py:719-721) is refused, although scorer_rewards holds 0.0."""
        self.assertEqual(self.errored_merge.returncode, EXIT_OK, describe(self.errored_merge))
        rec = next(r for r in read_jsonl(self.errored / "results.jsonl") if r["problem_idx"] == 4)
        self.assertEqual(rec["scorer_rewards"], {SCORER_KEY: 0.0})
        self.assertIn("error", rec["scoring_details"][SCORER_KEY])
        self.assert_refused("errored", "TIERB-RESULTS", "--eval-baseline", baseline=self.errored)

    def test_arm_without_run_configs(self):
        """Merged records alone cannot show the runs were made with the scorer: refused, not trusted."""
        self.assert_refused("bare", "TIERB-SCORER", "--eval-baseline", baseline=self.bare)


class TestScorerOnlyForTierBEvaluator(unittest.TestCase):
    """Decision D-10 (review MEDIUM-1, maintainers 2026-11-11): `evaluators[0].scorer` is allowed only when
    `evaluators[0].id` is the Tier B evaluator `gsm8k.nemo_evaluator@0.1`. Any other evaluator with a
    scorer is refused by `validate` with REG-SCORER, exit 3, and nothing is written. The Tier A demo
    (`addition_em.cpu_stub@0.1`) is the case under test; its spec is re-pinned to the edited registration."""

    # Match the evaluator entry regardless of the pinned hash (R10-5).
    TIER_A_ANCHOR = re.compile(r"  - id: addition_em\.cpu_stub@0\.1\n    manifest_hash: [^\n]*\n")

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-scorer-tier-a-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.demo = self.tmp / "demo"
        shutil.copytree(DEMO_DIR, self.demo,
                        ignore=shutil.ignore_patterns("baseline", "runs", "tierb", "__pycache__", ".DS_Store"))

    def validate_with(self, scorer):
        reg = self.demo / "registration.yaml"
        text = reg.read_text(encoding="utf-8")
        found = self.TIER_A_ANCHOR.findall(text)
        self.assertEqual(len(found), 1, "Tier A demo registration layout changed")
        value = yaml.safe_dump(scorer, default_flow_style=True).split("\n")[0]
        reg.write_text(text.replace(found[0], found[0] + f"    scorer: {value}\n"), encoding="utf-8")
        spec = self.demo / "spec.yaml"
        spec_text, n = re.subn(r"registration\.yaml@sha256:[0-9a-f]{64}",
                               "registration.yaml@" + file_sha256(reg), spec.read_text(encoding="utf-8"))
        self.assertEqual(n, 1, "demo/spec.yaml pins registration.yaml once")
        spec.write_text(spec_text, encoding="utf-8")
        self.before = sorted(str(p.relative_to(self.demo)) for p in self.demo.rglob("*"))
        return run_cli(["validate", "spec.yaml"], cwd=self.demo, timeout=60)

    def test_tier_a_evaluator_with_scorer_refused(self):
        proc = self.validate_with(SCORER)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = scorer_errors(proc)
        self.assertTrue(errs, describe(proc))
        self.assertIn(EVALUATOR_ID, errs[0]["message"] + errs[0]["fix"],
                      "the refusal names the only evaluator that takes a scorer")
        self.assertNotIn("Traceback", proc.stderr)
        self.assertEqual(sorted(str(p.relative_to(self.demo)) for p in self.demo.rglob("*")), self.before,
                         "validate writes nothing")

    def test_tier_a_null_scorer_accepted(self):
        """Control: `scorer: null` is no scorer, so the re-pinned Tier A demo validates as today."""
        proc = self.validate_with(None)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))


def derive_arm(src: Path, dst: Path, edit, *, problem_idx: int = 6, repeat: int = 0) -> Path:
    """A copy of the merged arm `src` whose record (problem_idx, repeat) is passed through `edit` and
    written back with json.dumps (so NaN and Infinity are written as the JSON literals Python accepts)."""
    shutil.copytree(src, dst)
    lines = []
    for line in (dst / "results.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["problem_idx"] == problem_idx and rec.get("repeat", 0) == repeat:
            rec = edit(rec)
        lines.append(json.dumps(rec))
    (dst / "results.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dst


BAD_SCORES = {"nan": float("nan"), "pos-inf": float("inf"), "neg-inf": float("-inf"), "negative": -0.1,
              "above-one": 1.5}


class ScoreRefusalCase(unittest.TestCase):
    """Shared refusal check: exit 3, the rule and field named, no run directory, no traceback."""

    def assert_refused(self, sc: ScorerScenario, key: str, rule: str, field: str, *, baseline: Path,
                       candidate: Path):
        runs_root = self.tmp / f"runs-{key}"
        proc = sc.collect(baseline, candidate, runs_root)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = parse_errors(proc.stderr)
        self.assertTrue([e for e in errs if e["rule"] == rule and e["field"] == field],
                        f"expected error[{rule}] {field}; got {errs}\n" + describe(proc))
        self.assertTrue(no_run_dir(runs_root), "no run directory may be created")
        self.assertNotIn("Traceback", proc.stderr)


class TestCollectScoreRange(ScoreRefusalCase):
    """Decision D-11 (review LOW-1, maintainers 2026-11-11): every per-record score the gate reads, the
    registered scorer's `scorer_rewards["scorer:X"]` or the plain `reward` without a scorer, must be a
    finite number in [0, 1]. NaN, +inf, -inf, -0.1 and 1.5 are refused with TIERB-RESULTS (field of the
    arm), exit 3, before any run directory exists. Python's json reads NaN/Infinity literals, so such a
    results.jsonl loads without error and only this check stops it."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-scorer-range-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.scored = ScorerScenario(cls.tmp / "scored", scorer=SCORER)
        cls.s_b = cls.scored.arm("baseline", baseline_reward, "ok")
        cls.s_c = cls.scored.arm("candidate", candidate_reward, "ok")
        cls.plain = ScorerScenario(cls.tmp / "plain", scorer=None)
        cls.p_b = cls.plain.arm("baseline", baseline_reward, "ok", reward_fn=baseline_reward)
        cls.p_c = cls.plain.arm("candidate", candidate_reward, "ok", reward_fn=candidate_reward)

    @staticmethod
    def set_scorer_reward(value):
        def edit(rec):
            rec["scorer_rewards"][SCORER_KEY] = value
            rec["scoring_details"][SCORER_KEY]["reward"] = value
            return rec
        return edit

    @staticmethod
    def set_reward(value):
        def edit(rec):
            rec["reward"] = value
            return rec
        return edit

    def test_good_arms_accepted(self):
        """Control: the unedited arms of both scenarios collect."""
        for name, sc, b, c in (("scored", self.scored, self.s_b, self.s_c), ("plain", self.plain, self.p_b, self.p_c)):
            with self.subTest(scenario=name):
                proc = sc.collect(b, c, self.tmp / f"runs-good-{name}")
                self.assertIn(proc.returncode, (0, 1), describe(proc))

    def test_scorer_reward_out_of_range(self):
        for name, value in BAD_SCORES.items():
            with self.subTest(value=name):
                arm = derive_arm(self.s_b, self.tmp / "arms" / f"scored-{name}", self.set_scorer_reward(value))
                self.assert_refused(self.scored, f"scored-{name}", "TIERB-RESULTS", "--eval-baseline",
                                    baseline=arm, candidate=self.s_c)

    def test_scorer_reward_out_of_range_candidate(self):
        arm = derive_arm(self.s_c, self.tmp / "arms" / "scored-cand-nan", self.set_scorer_reward(float("nan")))
        self.assert_refused(self.scored, "scored-cand-nan", "TIERB-RESULTS", "--eval-candidate",
                            baseline=self.s_b, candidate=arm)

    def test_plain_reward_out_of_range(self):
        for name, value in BAD_SCORES.items():
            with self.subTest(value=name):
                arm = derive_arm(self.p_b, self.tmp / "arms" / f"plain-{name}", self.set_reward(value))
                self.assert_refused(self.plain, f"plain-{name}", "TIERB-RESULTS", "--eval-baseline",
                                    baseline=arm, candidate=self.p_c)

    def test_plain_reward_out_of_range_candidate(self):
        arm = derive_arm(self.p_c, self.tmp / "arms" / "plain-cand-inf", self.set_reward(float("inf")))
        self.assert_refused(self.plain, "plain-cand-inf", "TIERB-RESULTS", "--eval-candidate",
                            baseline=self.p_b, candidate=arm)


class TestCollectMalformedScorerFields(ScoreRefusalCase):
    """Decision D-12 (review LOW-2, maintainers 2026-11-11), with a scorer registered:
    - a record whose `scorer_rewards` or `scoring_details` is not a JSON object (a list or a string) is
      refused with TIERB-RESULTS, field of the arm, exit 3, not a traceback;
    - an arm whose `configs/rep<r>.yaml` holds bytes that are not UTF-8 is refused with TIERB-SCORER,
      field of the arm, exit 3, not a traceback.
    No run directory is created in either case."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-scorer-malformed-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = ScorerScenario(cls.tmp, scorer=SCORER)
        cls.good_b = cls.sc.arm("baseline", baseline_reward, "ok")
        cls.good_c = cls.sc.arm("candidate", candidate_reward, "ok")

    def test_non_dict_fields(self):
        cases = {
            "rewards-list": ("scorer_rewards", [SCORER_KEY, 1.0]),
            "rewards-string": ("scorer_rewards", f"{SCORER_KEY}=1.0"),
            "details-list": ("scoring_details", [SCORER_KEY, {"correct": True, "reward": 1.0}]),
            "details-string": ("scoring_details", "numeric_match: correct"),
        }
        for name, (field, value) in cases.items():
            with self.subTest(case=name):
                def edit(rec, field=field, value=value):
                    rec[field] = value
                    return rec
                arm = derive_arm(self.good_b, self.tmp / "arms" / name, edit)
                self.assert_refused(self.sc, name, "TIERB-RESULTS", "--eval-baseline",
                                    baseline=arm, candidate=self.good_c)

    def test_non_utf8_run_config(self):
        arm = self.tmp / "arms" / "non-utf8"
        shutil.copytree(self.good_c, arm)
        cfg = arm / "configs" / "rep0.yaml"
        cfg.write_bytes(b"# \xff\xfe not UTF-8\n" + cfg.read_bytes())
        self.assert_refused(self.sc, "non-utf8", "TIERB-SCORER", "--eval-candidate",
                            baseline=self.good_b, candidate=arm)


class TestDemoTemplateAmendment(unittest.TestCase):
    """Decision 6 (maintainers, 2026-11-11): demo/tierb/registration.yaml is the G-21 amendment (05:142);
    demo/tierb/spec.yaml re-pins its SHA-256 (also checked by test_tierb_spec.TestTierBTemplates.test_contents)."""

    def setUp(self):
        self.path = DEMO_DIR / "tierb" / "registration.yaml"
        self.text = self.path.read_text(encoding="utf-8")
        self.reg = yaml.safe_load(self.text)

    def test_amendment_fields(self):
        self.assertEqual(self.reg["registration_version"], 2)
        self.assertEqual(self.reg["registration_id"], "tierb-4090-gsm8k-superiority-v2")
        self.assertEqual(self.reg["evaluators"][0]["scorer"], SCORER)

    def test_scorer_comment_cites_g21(self):
        lines = [l for l in self.text.splitlines() if re.match(r"\s*scorer:\s*numeric_match\b", l)]
        self.assertEqual(len(lines), 1, "one `scorer: numeric_match` line")
        idx = self.text.splitlines().index(lines[0])
        context = "\n".join(self.text.splitlines()[max(0, idx - 3):idx + 1])
        self.assertIn("G-21", context, "the scorer line or the comment just above it cites G-21")

    def test_spec_pins_template(self):
        spec = yaml.safe_load((DEMO_DIR / "tierb" / "spec.yaml").read_text(encoding="utf-8"))
        uri = spec["artifacts"]["registration"]["uri"]
        self.assertTrue(uri.endswith("@" + file_sha256(self.path)), uri)

    def test_template_validates(self):
        proc = run_cli(["validate", "spec.yaml"], cwd=DEMO_DIR / "tierb", timeout=60)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))


# ---------------------------------------------------------------------------
# Citations against the pinned clone (read-only)
# ---------------------------------------------------------------------------

@unittest.skipUnless(NEMO_EVAL_SRC.is_dir(), f"pinned NeMo Evaluator clone not found at {NEMO_EVAL_CLONE}")
class TestScorerCitations(unittest.TestCase):
    """Every NeMo Evaluator fact the G-21 tests rely on, read from the clone at ea35ba3."""

    def lines(self, rel: str) -> list[str]:
        return (NEMO_EVAL_SRC / rel).read_text(encoding="utf-8").splitlines()

    def test_clone_commit(self):
        head = (NEMO_EVAL_CLONE / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = NEMO_EVAL_CLONE / ".git" / head[5:]
            if ref.is_file():
                head = ref.read_text(encoding="utf-8").strip()
            else:
                packed = (NEMO_EVAL_CLONE / ".git" / "packed-refs").read_text(encoding="utf-8")
                head = next(l.split()[0] for l in packed.splitlines() if l.endswith(head[5:]))
        self.assertTrue(head.startswith("ea35ba3"), head)

    def test_benchmark_scoring_field(self):
        """config/benchmarks.py:63."""
        self.assertEqual(self.lines("config/benchmarks.py")[62].strip(),
                         "scoring: ScoringConfig = Field(default_factory=ScoringConfig)")

    def test_scorer_metric_schema(self):
        """config/scoring.py:24-30: ScorerMetric {type: "scorer", name: str}, extra keys forbidden."""
        block = "\n".join(self.lines("config/scoring.py")[23:30])
        self.assertIn("class ScorerMetric(BaseModel):", block)
        self.assertIn('model_config = ConfigDict(extra="forbid")', block)
        self.assertIn('type: Literal["scorer"] = "scorer"', block)
        self.assertIn("name: str", block)

    def test_scoring_config_metrics(self):
        """config/scoring.py:160-184: ScoringConfig.metrics is a list; `primary` is only required with
        more than one metric, so one scorer metric needs neither `primary` nor `include_defaults`."""
        block = "\n".join(self.lines("config/scoring.py")[159:184])
        self.assertIn("class ScoringConfig(BaseModel):", block)
        self.assertIn("metrics: list[MetricConfig]", block)
        self.assertIn("if len(self.metrics) > 1 and self.primary is None:", block)

    def test_numeric_match_registered(self):
        """scoring/__init__.py:38-44."""
        block = "\n".join(self.lines("scoring/__init__.py")[37:44])
        self.assertIn('"numeric_match": numeric_match,', block)

    def test_numeric_match_strips_trailing_zeros(self):
        """scoring/pattern.py:39-43: last number, commas removed, trailing zeros and dot stripped on both
        sides, so "10" and "1" compare equal (the false positive the report note names)."""
        block = "\n".join(self.lines("scoring/pattern.py")[38:43])
        self.assertIn("def numeric_match(sample: ScorerInput) -> dict:", block)
        self.assertIn('nums[-1].rstrip("0").rstrip(".")', block)
        self.assertIn('str(sample.target).strip().rstrip("0").rstrip(".")', block)
        ns: dict = {}
        exec(compile("import re\n" + "\n".join(self.lines("scoring/pattern.py")[38:43]).replace(
            "sample: ScorerInput", "sample"), "pattern.py", "exec"), ns)

        class S:
            def __init__(self, response, target):
                self.response, self.target = response, target

        self.assertTrue(ns["numeric_match"](S("The answer is 10", "1"))["correct"])
        self.assertTrue(ns["numeric_match"](S("The answer is 1", "10"))["correct"])
        # Decision D-7: the effect is on integers without a trailing dot; "10." keeps its zero because the
        # regex captures the dot, so rstrip("0") has nothing to strip and the comparison is "10" vs "1".
        self.assertFalse(ns["numeric_match"](S("The answer is 10.", "1"))["correct"])
        self.assertTrue(ns["numeric_match"](S("The answer is: **1,234**.", "1234"))["correct"])
        self.assertFalse(ns["numeric_match"](S("The answer is 12", "13"))["correct"])

    def test_orchestration_passes_scorer_metrics(self):
        """orchestration/orchestrator.py:1050 and :1064."""
        lines = self.lines("orchestration/orchestrator.py")
        self.assertIn('scorer_names = [m.name for m in bench.scoring.metrics if getattr(m, "type", None) == "scorer"]',
                      lines[1049])
        self.assertIn('"scorers": scorer_names or None,', lines[1063])

    def test_extra_scorer_record_fields(self):
        """engine/eval_loop.py:704-721 (scoring_details["scorer:<name>"]) and :785-794 (scorer_rewards,
        top-level reward unchanged)."""
        lines = self.lines("engine/eval_loop.py")
        self.assertIn('vr.scoring_details[f"scorer:{scorer_name}"] = sresult', "\n".join(lines[703:721]))
        block = "\n".join(lines[784:794])
        self.assertIn('if sk.startswith("scorer:") and isinstance(sv, dict):', block)
        self.assertIn('"reward": vr.reward,', block)
        self.assertIn('"scorer_rewards": scorer_rewards,', block)

    def test_snapshot_dumps_raw_config(self):
        """config/snapshot.py:72-81: the snapshot body is the raw config, so `scoring` appears as written."""
        block = "\n".join(self.lines("config/snapshot.py")[71:81])
        self.assertIn("body = yaml.dump(raw, default_flow_style=False, sort_keys=False)", block)


if __name__ == "__main__":
    unittest.main()
