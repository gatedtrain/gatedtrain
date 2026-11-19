"""Design revision 10 alignment of `gatedtrain tierb-collect` (tests/README.md section 11.15).

Scenarios are the recorded fixtures of test_tierb_collect (smoke profile, K = 2, 30 held-out problems;
expected gate Delta-hat = 1/3, PASS) with one change each. Registrations are edited through YAML and
the spec is re-pinned.

- R10A-2 (R10-5, 05 section 6.1, 05:217): an evaluator hash that differs at run time is MISSING.
- R10A-3 (R10-8, 05 section 3.4, 05:116): unpaired items above `pairing.max_unpaired` make the gate
  MISSING (report, exit 1) instead of the rev-9 TIERB-PAIRING refusal; per-arm missing counts.
- R10A-4 (R10-9, 05 section 3.7, 05:151): `amends` is reported next to the registration hash; a
  registration with `post_hoc: true` cannot PASS (INCONCLUSIVE, reason post_hoc).
- R10A-7 (review finding 7): the run's own data config must keep the ledger mapping valid.
- R10A-8 (review finding 8): actual K and the repeat -> seed mapping must match the registration.
- R10A-10 (review finding 10): p-value in the report.
- R10A-11 (review finding 11): the lockfile records whether the NeMo RL image carries a local
  modification; an absent flag is recorded as unknown, never as unmodified.
- R10A-12: the long-run-shaped regression (K = 4, numeric_match, long profile, new defaults) is PASS.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from _support import (EXIT_GATE_NOT_PASS, EXIT_OK, EXIT_REFUSED, EXIT_USAGE, SKEL_ROOT, describe, file_sha256,
                      load_report, parse_errors, parse_run_dir, run_cli)
from _tierb import (PROFILES, baseline_reward, build_eval_run, build_merged, build_stage_out, candidate_reward,
                    evaluator_identity_hash, heldout_rows, nemo_rl_data_config, registration_seeds, render_inputs,
                    run_export)
from test_r10_registration import DUMMY_HASH, PRIOR_HASH, repin, rewrite_registration
from test_tierb_collect import IMAGE_ID, NEMO_RL_COMMIT, Scenario, no_run_dir

IMAGE_NOTES = ("local edit of the build-context checkout: pyproject.toml:617 flash-attn MAX_JOBS 4 -> 2, "
               "NVCC_THREADS 2 (the first run's build notes, 2026-11-10, option B)")


def collect(sc, runs_root: Path, *, baseline: Path | None = None, candidate: Path | None = None,
            stage_out: Path | None = None, extra: list[str] = ()):
    args = ["tierb-collect", str(sc.paths["spec"]), "--stage-out", str(stage_out or sc.stage_out),
            "--eval-baseline", str(baseline or sc.baseline), "--eval-candidate", str(candidate or sc.candidate),
            "--image-id", IMAGE_ID, "--nemo-rl-commit", NEMO_RL_COMMIT, "--runs-root", str(runs_root), *extra]
    return run_cli(args, cwd=sc.root, timeout=300)


def drop_records(src: Path, dst: Path, keys: set[tuple[int, int]]) -> Path:
    """A copy of the merged arm `src` without the records whose (problem_idx, repeat) is in `keys`."""
    shutil.copytree(src, dst)
    lines = [l for l in (dst / "results.jsonl").read_text(encoding="utf-8").splitlines()
             if l.strip() and (json.loads(l)["problem_idx"], json.loads(l).get("repeat", 0)) not in keys]
    (dst / "results.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dst


def text_lines(run_dir: Path) -> list[str]:
    return (run_dir / "report" / "gate_report.txt").read_text(encoding="utf-8").splitlines()


class GateCase(unittest.TestCase):
    """One Scenario per class; each test collects into its own runs root."""

    mutate = None    # registration edit applied before the arms are built

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-r10-tierb-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = Scenario(cls.tmp)
        if cls.mutate is not None:
            rewrite_registration(cls.sc.paths["registration"], cls.sc.paths["spec"], cls.mutate)
        cls.n = 0

    def collect(self, **kw):
        type(self).n += 1
        runs_root = self.tmp / f"runs-{type(self).n}"
        return runs_root, collect(self.sc, runs_root, **kw)

    def gate(self, proc, status: str, exit_code: int = EXIT_GATE_NOT_PASS) -> tuple[Path, dict]:
        self.assertEqual(proc.returncode, exit_code, describe(proc))
        run_dir = parse_run_dir(proc.stdout)
        rep = load_report(run_dir)
        self.assertEqual(rep["status"], status, {k: rep.get(k) for k in ("status", "status_reasons")})
        return run_dir, rep

    def refused(self, runs_root, proc, rule: str, field: str | None = None):
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = [e for e in parse_errors(proc.stderr) if e["rule"] == rule and (field is None or e["field"] == field)]
        self.assertTrue(errs, f"expected error[{rule}]\n" + describe(proc))
        self.assertTrue(no_run_dir(runs_root))


# ---------------------------------------------------------------------------
# New report fields on the unchanged fixture (R10A-3, -4, -8, -10, -11)
# ---------------------------------------------------------------------------

class TestReportFields(GateCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.proc = collect(cls.sc, cls.tmp / "runs")

    def test_pass_with_new_fields(self):
        run_dir, rep = self.gate(self.proc, "PASS", EXIT_OK)
        self.assertEqual(rep["status_reasons"], [])
        self.assertEqual((rep["missing_baseline"], rep["missing_candidate"]), (0, 0))
        self.assertEqual(rep["max_unpaired"], 0, "absent pairing means 0 (05 section 3.4)")
        self.assertEqual(rep["min_paired"], 10, "absent min_paired means 10 (05 section 3.6)")
        self.assertEqual(rep["k_actual"], {"baseline": [2, 2], "candidate": [2, 2]})
        self.assertIs(rep["post_hoc"], False)
        self.assertIsNone(rep["registration_amends"])
        self.assertEqual(rep["p_value_method"], "bca_inversion")
        self.assertLess(rep["p_value"], 0.025)

    def test_evaluator_hashes(self):
        _, rep = self.gate(self.proc, "PASS", EXIT_OK)
        self.assertEqual(rep["evaluator_hash"], evaluator_identity_hash())
        self.assertEqual(rep["evaluator_hash_registered"], evaluator_identity_hash())

    def test_text_lines(self):
        run_dir, rep = self.gate(self.proc, "PASS", EXIT_OK)
        lines = text_lines(run_dir)
        for prefix in ("p_value: ", "status_reasons:", "k_actual: ", "missing_baseline: 0", "missing_candidate: 0",
                       "post_hoc: false", "registration_amends: null", "min_paired: 10", "max_unpaired: 0"):
            self.assertTrue([l for l in lines if l.startswith(prefix)], f"no line starting {prefix!r}")
        k_line = [l for l in lines if l.startswith("k_actual: ")][0]
        self.assertIn("2", k_line)

    def test_image_modification_unknown_without_flag(self):
        """R10A-11: no flag -> `nemo_rl_modified: "unknown"`, never false; `image_build_notes: null`."""
        run_dir, _ = self.gate(self.proc, "PASS", EXIT_OK)
        tr = json.loads((run_dir / "lockfile.json").read_bytes())["backend"]["trainer"]
        self.assertEqual(tr["nemo_rl_modified"], "unknown")
        self.assertIsNone(tr["image_build_notes"])


# ---------------------------------------------------------------------------
# R10A-2 (R10-5): evaluator hash mismatch at run time is MISSING
# ---------------------------------------------------------------------------

class TestEvaluatorHashMismatch(GateCase):
    mutate = staticmethod(lambda d: d["evaluators"][0].__setitem__("manifest_hash", DUMMY_HASH))

    def test_missing(self):
        _, proc = self.collect()
        run_dir, rep = self.gate(proc, "MISSING")
        self.assertIn("evaluator_hash_mismatch", rep["status_reasons"])
        self.assertEqual(rep["evaluator_hash_registered"], DUMMY_HASH)
        self.assertEqual(rep["evaluator_hash"], evaluator_identity_hash())
        self.assertTrue([l for l in text_lines(run_dir) if l.startswith("evaluator_hash_registered: ")])


# ---------------------------------------------------------------------------
# R10A-3 (R10-8): MISSING above the unpaired bound, per-arm counts
# ---------------------------------------------------------------------------

class TestMissingnessDefault(GateCase):
    """Registration without `pairing` (default 0)."""

    mutate = staticmethod(lambda d: d.pop("pairing"))

    def test_one_record_missing_is_missing_not_refused(self):
        cand = drop_records(self.sc.candidate, self.tmp / "eval-drop1" / "candidate", {(5, 1)})
        _, proc = self.collect(candidate=cand)
        run_dir, rep = self.gate(proc, "MISSING")
        self.assertIn("missing_above_bound", rep["status_reasons"])
        self.assertEqual((rep["missing_baseline"], rep["missing_candidate"]), (0, 1))
        self.assertIn("missing_candidate: 1", text_lines(run_dir))

    def test_whole_problem_missing_in_one_arm(self):
        """Replaces test_tierb_collect.test_unpaired_beyond_tolerance (rev-9 TIERB-PAIRING refusal).
        Problem 4 is evaluated in the baseline only: TIERB-PROMPTS (a) checks the union of both arms'
        problems against the held-out set, and the missing items make the gate MISSING."""
        cand = self.sc.arm("candidate", candidate_reward, root=self.tmp / "eval-drop4", drop={4})
        _, proc = self.collect(candidate=cand)
        _, rep = self.gate(proc, "MISSING")
        self.assertEqual((rep["missing_baseline"], rep["missing_candidate"]), (0, 2))
        self.assertEqual((rep["unpaired_baseline"], rep["unpaired_candidate"]), (2, 0))


class TestMissingnessCount(GateCase):
    mutate = staticmethod(lambda d: d.__setitem__("pairing", {"max_unpaired": 2}))

    def test_within_count_bound(self):
        cand = self.sc.arm("candidate", candidate_reward, root=self.tmp / "eval-drop4", drop={4})
        _, proc = self.collect(candidate=cand)
        _, rep = self.gate(proc, "PASS", EXIT_OK)
        self.assertEqual(rep["missing_candidate"], 2)
        self.assertEqual(rep["n_items"], 29)                       # R11A-4: item names (was n_prompts)

    def test_above_count_bound(self):
        cand = self.sc.arm("candidate", candidate_reward, root=self.tmp / "eval-drop45", drop={4, 5})
        _, proc = self.collect(candidate=cand)
        self.gate(proc, "MISSING")


class TestMissingnessFraction(GateCase):
    """2 missing items of 2 |E| = 2 * 30 * 2 = 120 expected = 1.67%."""

    mutate = staticmethod(lambda d: d.__setitem__("pairing", {"max_unpaired": 0.02}))

    def test_within_fraction(self):
        cand = self.sc.arm("candidate", candidate_reward, root=self.tmp / "eval-drop4", drop={4})
        _, proc = self.collect(candidate=cand)
        self.gate(proc, "PASS", EXIT_OK)

    def test_above_fraction(self):
        cand = self.sc.arm("candidate", candidate_reward, root=self.tmp / "eval-drop456", drop={4, 5, 6})
        _, proc = self.collect(candidate=cand)
        self.gate(proc, "MISSING")


# ---------------------------------------------------------------------------
# R10A-4 (R10-9): amendments and post hoc
# ---------------------------------------------------------------------------

class TestAmendment(GateCase):
    mutate = staticmethod(lambda d: d.update(amends=PRIOR_HASH, created_at="2026-11-11T09:00:00Z"))  # R11A-29

    def test_both_hashes_reported(self):
        _, proc = self.collect()
        run_dir, rep = self.gate(proc, "PASS", EXIT_OK)
        self.assertEqual(rep["registration_amends"], PRIOR_HASH)
        self.assertEqual(rep["registration_hash"], file_sha256(self.sc.paths["registration"]))
        self.assertIn(f"registration_amends: {PRIOR_HASH}", text_lines(run_dir))


class TestPostHoc(GateCase):
    # R11A-29: `created_at` with `amends`. The gate reads the declared flag only (post_hoc evidence is not
    # wired, R11A-16), so the time does not change the result.
    mutate = staticmethod(lambda d: d.update(amends=PRIOR_HASH, created_at="2026-11-11T09:00:00Z", post_hoc=True))

    def test_cannot_pass(self):
        """The fixture is a clear PASS; declared post hoc it is INCONCLUSIVE with reason post_hoc."""
        _, proc = self.collect()
        run_dir, rep = self.gate(proc, "INCONCLUSIVE")
        self.assertIn("post_hoc", rep["status_reasons"])
        self.assertIs(rep["post_hoc"], True)
        self.assertAlmostEqual(rep["delta_hat"], 1 / 3, places=12, msg="the statistics are still reported")
        self.assertIn("post_hoc: true", text_lines(run_dir))

    def test_fail_stays_fail(self):
        """Maintainers' decision of 2026-11-11: a post hoc registration whose result is FAIL stays FAIL. A candidate
        with reward 0 everywhere: Delta-hat = -1/3, observed SD ~0.48, power at theta = 0.30 ~0.93
        (powered), so the superiority non-rejection is FAIL; post_hoc does not soften it."""
        cand = self.sc.arm("candidate", lambda i: 0.0, root=self.tmp / "eval-zero")
        _, proc = self.collect(candidate=cand)
        run_dir, rep = self.gate(proc, "FAIL")
        self.assertIs(rep["post_hoc"], True)
        self.assertFalse(rep["underpowered"])
        self.assertNotIn("post_hoc", rep["status_reasons"])
        self.assertAlmostEqual(rep["delta_hat"], -1 / 3, places=12)

    def test_missing_stays_missing(self):
        cand = self.sc.arm("candidate", candidate_reward, root=self.tmp / "eval-drop4", drop={4})
        _, proc = self.collect(candidate=cand)
        _, rep = self.gate(proc, "MISSING")
        self.assertIs(rep["post_hoc"], True)
        self.assertIn("missing_above_bound", rep["status_reasons"])
        self.assertNotIn("post_hoc", rep["status_reasons"])


# ---------------------------------------------------------------------------
# R10A-8: actual K and the repeat -> seed mapping
# ---------------------------------------------------------------------------

class TestKAndSeeds(GateCase):
    def arm(self, name, fn, root: Path, seeds: list[int]) -> Path:
        return build_merged(root / name, self.sc.rows, reward_fn=fn, seeds=seeds)

    def test_both_arms_k1_under_registered_k2(self):
        """R10A-8 made this MISSING with `k_mismatch`. Design revision 11 (R11-7, R11A-7): fewer than K records
        are missing items only, so it stays MISSING at bound 0, but for `missing_above_bound` (30 missing
        items per arm, reason `absent`), not `k_mismatch`."""
        root = self.tmp / "eval-k1"
        b = self.arm("baseline", baseline_reward, root, self.sc.seeds[:1])
        c = self.arm("candidate", candidate_reward, root, self.sc.seeds[:1])
        _, proc = self.collect(baseline=b, candidate=c)
        run_dir, rep = self.gate(proc, "MISSING")
        self.assertIn("missing_above_bound", rep["status_reasons"])
        self.assertNotIn("k_mismatch", rep["status_reasons"])
        self.assertEqual((rep["missing_baseline"], rep["missing_candidate"]), (30, 30))
        self.assertEqual(rep["k_actual"], {"baseline": [1, 1], "candidate": [1, 1]})
        self.assertEqual(rep["samples_per_item"], 2, "the registered K stays in the report")

    def test_baseline_k2_candidate_k1(self):
        """As above (R11A-7): MISSING for `missing_above_bound`, not `k_mismatch`."""
        c = self.arm("candidate", candidate_reward, self.tmp / "eval-c-k1", self.sc.seeds[:1])
        _, proc = self.collect(candidate=c)
        _, rep = self.gate(proc, "MISSING")
        self.assertIn("missing_above_bound", rep["status_reasons"])
        self.assertNotIn("k_mismatch", rep["status_reasons"])
        self.assertEqual(rep["k_actual"], {"baseline": [2, 2], "candidate": [1, 1]})

    def test_seed_mapping_swapped(self):
        """repeat 0 carries eval_seed 12 and repeat 1 eval_seed 11 in both arms: no unpaired item and
        K = 2, but the repeat -> seed mapping is not the registered one."""
        root = self.tmp / "eval-swapped"
        swapped = list(reversed(self.sc.seeds))
        b = self.arm("baseline", baseline_reward, root, swapped)
        c = self.arm("candidate", candidate_reward, root, swapped)
        _, proc = self.collect(baseline=b, candidate=c)
        _, rep = self.gate(proc, "MISSING")
        self.assertIn("seed_mismatch", rep["status_reasons"])


# ---------------------------------------------------------------------------
# R10A-7: the run's data config
# ---------------------------------------------------------------------------

class TestDataConfig(GateCase):
    """Refused with TIERB-DATA, field --stage-out, exit 3, nothing written, when step_<last>/config.yaml
    has `data.shuffle` other than false, more or fewer than one training dataset, a training
    `data_path` whose file name is not train.jsonl or whose parent directory name is not the spec's
    export directory name, or `split_validation_size` other than 0 (response_dataset.py:76 splits and
    reorders rows). `data.train` may be a mapping or a one-element list (nemo_rl/data/utils.py:215-216)."""

    def stage(self, name: str, **data_kw) -> Path:
        p = PROFILES["smoke"]
        data = data_kw.pop("data", None) or nemo_rl_data_config(**data_kw)
        return build_stage_out(self.tmp / "stages" / name, prompts_per_step=p["prompts_per_step"], gens=p["gens"],
                               steps=p["steps"], save_period=p["save_period"], data=data)

    def refuse(self, name: str, **data_kw):
        runs_root, proc = self.collect(stage_out=self.stage(name, **data_kw))
        self.refused(runs_root, proc, "TIERB-DATA", "--stage-out")

    def test_shuffle_true(self):
        self.refuse("shuffle", shuffle=True)

    def test_shuffle_absent(self):
        cfg = nemo_rl_data_config()
        cfg.pop("shuffle")
        self.refuse("no-shuffle", data=cfg)

    def test_other_file(self):
        self.refuse("other-file", data_path="/work/tierb/export/test.jsonl")

    def test_other_export_dir(self):
        self.refuse("other-dir", data_path="/work/tierb/export-old/train.jsonl")

    def test_two_datasets(self):
        self.refuse("two", extra_datasets=1)

    def test_validation_split(self):
        self.refuse("split", split_validation_size=0.1)

    def test_mapping_shape_accepted(self):
        cfg = nemo_rl_data_config()
        cfg["train"] = cfg["train"][0]
        _, proc = self.collect(stage_out=self.stage("mapping", data=cfg))
        self.gate(proc, "PASS", EXIT_OK)

    def test_real_shape_accepted(self):
        """Control: the default fixture shape is the real long run's (data_path /work/tierb/export/train.jsonl)."""
        _, proc = self.collect(stage_out=self.stage("real"))
        self.gate(proc, "PASS", EXIT_OK)


# ---------------------------------------------------------------------------
# R10A-11: image modification flags
# ---------------------------------------------------------------------------

class TestImageModification(GateCase):
    def trainer(self, extra):
        _, proc = self.collect(extra=extra)
        run_dir, _ = self.gate(proc, "PASS", EXIT_OK)
        return json.loads((run_dir / "lockfile.json").read_bytes())["backend"]["trainer"]

    def test_modified_with_notes(self):
        tr = self.trainer(["--nemo-rl-modified", "yes", "--image-build-notes", IMAGE_NOTES])
        self.assertIs(tr["nemo_rl_modified"], True)
        self.assertEqual(tr["image_build_notes"], IMAGE_NOTES)

    def test_unmodified(self):
        tr = self.trainer(["--nemo-rl-modified", "no"])
        self.assertIs(tr["nemo_rl_modified"], False)
        self.assertIsNone(tr["image_build_notes"])

    def test_bad_value_is_usage_error(self):
        runs_root, proc = self.collect(extra=["--nemo-rl-modified", "maybe"])
        self.assertEqual(proc.returncode, EXIT_USAGE, describe(proc))
        self.assertTrue(no_run_dir(runs_root))


# ---------------------------------------------------------------------------
# R10A-12: regression shaped like the real long run
# ---------------------------------------------------------------------------

class TestLongRunShapedRegression(unittest.TestCase):
    """Shaped like the real long run <HOST_WORK>/runs/run-20261111T110313Z-7adda9b9 (read-only, not
    used): long profile (150 steps, 32 x 8, save period 25), K = 4 with seeds [11, 12, 13, 14] from four
    single-seed runs merged by the real merge-eval, the numeric_match scorer, superiority with a fixed
    sigma plan, the saved data config of that run, and a registration that carries the rev-10 defaults
    explicitly (min_paired 10, pairing.max_unpaired 0) and the resolved evaluator hash. 30 held-out
    problems instead of 1,319. Expected: Delta-hat = 1/3, PASS, no status reasons."""

    SEEDS = [11, 12, 13, 14]
    SCORER = "numeric_match"

    @classmethod
    def setUpClass(cls):
        cls.tmp = root = Path(tempfile.mkdtemp(prefix="gatedtrain-r10-long-"))
        cls.addClassCleanup(shutil.rmtree, root, True)
        proc = run_export(root / "export", cwd=SKEL_ROOT)
        assert proc.returncode == EXIT_OK, describe(proc)
        cls.paths = render_inputs(root, root / "export", scorer=cls.SCORER, set_scorer=True)
        (root / "trainer.yaml").write_text((root / "trainer.yaml").read_text(encoding="utf-8")
                                           .replace("profile: smoke", "profile: long"), encoding="utf-8")
        spec = cls.paths["spec"]
        spec.write_text(spec.read_text(encoding="utf-8").replace("seeds: 2}", "seeds: 4}"), encoding="utf-8")

        def rev10(d):
            d["clause"]["samples_per_prompt"] = 4
            d["clause"]["eval_seeds"] = list(cls.SEEDS)
            d["min_paired"] = 10
            d["pairing"] = {"max_unpaired": 0}
            d["amends"] = PRIOR_HASH
            d["created_at"] = "2026-11-11T09:00:00Z"                  # R11A-29
            d["post_hoc"] = False

        rewrite_registration(cls.paths["registration"], spec, rev10)
        assert registration_seeds(cls.paths["registration"]) == cls.SEEDS
        assert yaml.safe_load(cls.paths["registration"].read_text())["evaluators"][0]["manifest_hash"] == \
            evaluator_identity_hash(cls.SCORER)
        p = PROFILES["long"]
        cls.stage_out = build_stage_out(root / "out" / "rl", prompts_per_step=p["prompts_per_step"], gens=p["gens"],
                                        steps=p["steps"], save_period=p["save_period"])
        rows = heldout_rows(root / "export")
        arms = {}
        for name, fn in (("baseline", baseline_reward), ("candidate", candidate_reward)):
            runs = [build_eval_run(root / "out" / "raw" / name / f"rep{r}", rows, reward_fn=lambda i: 0.0, seed=s,
                                   model=name, scorer=cls.SCORER, scorer_fn=fn) for r, s in enumerate(cls.SEEDS)]
            out = root / "out" / "eval" / name
            mp = run_cli(["merge-eval", "--registration", str(cls.paths["registration"]), "--out", str(out),
                          *map(str, runs)], cwd=SKEL_ROOT, timeout=60)
            assert mp.returncode == EXIT_OK, describe(mp)
            arms[name] = out
        cls.arms = arms

    def collect(self, runs_root: Path, extra=()):
        args = ["tierb-collect", str(self.paths["spec"]), "--stage-out", str(self.stage_out),
                "--eval-baseline", str(self.arms["baseline"]), "--eval-candidate", str(self.arms["candidate"]),
                "--image-id", IMAGE_ID, "--nemo-rl-commit", NEMO_RL_COMMIT, "--runs-root", str(runs_root), *extra]
        return run_cli(args, cwd=self.tmp, timeout=300)

    def check_pass(self, proc) -> Path:
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        run_dir = parse_run_dir(proc.stdout)
        rep = load_report(run_dir)
        self.assertEqual(rep["status"], "PASS", rep.get("status_reasons"))
        self.assertEqual(rep["status_reasons"], [])
        self.assertAlmostEqual(rep["delta_hat"], 1 / 3, places=12)
        self.assertEqual(rep["n_items"], 30)                       # R11A-4: item names (was n_prompts)
        self.assertEqual(rep["k_actual"], {"baseline": [4, 4], "candidate": [4, 4]})
        self.assertEqual(rep["evaluator_hash"], evaluator_identity_hash(self.SCORER))
        self.assertEqual(rep["evaluator_hash_registered"], rep["evaluator_hash"])
        self.assertEqual(rep["registration_amends"], PRIOR_HASH)
        self.assertLess(rep["p_value"], 0.025)
        return run_dir

    def test_pass_without_new_flags(self):
        run_dir = self.check_pass(self.collect(self.tmp / "runs-plain"))
        tr = json.loads((run_dir / "lockfile.json").read_bytes())["backend"]["trainer"]
        self.assertEqual(tr["nemo_rl_modified"], "unknown")

    def test_pass_with_image_notes(self):
        run_dir = self.check_pass(self.collect(self.tmp / "runs-notes",
                                               ["--nemo-rl-modified", "yes", "--image-build-notes", IMAGE_NOTES]))
        tr = json.loads((run_dir / "lockfile.json").read_bytes())["backend"]["trainer"]
        self.assertEqual((tr["nemo_rl_modified"], tr["image_build_notes"]), (True, IMAGE_NOTES))


if __name__ == "__main__":
    unittest.main()
