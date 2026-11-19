"""End-to-end acceptance tests (MVP-SPEC section 5), SLOW.

These run the real CLI on CPU: one warm-up and six full runs, shared across
the classes of this module through _support.WORKSPACE. Set
GATEDTRAIN_SKIP_SLOW=1 to skip them. Budget: each `gatedtrain run` of a demo
spec should take at most about 90 s so the suite stays under 15 minutes.

Design references are given per class; the interface (CLI, run-directory
layout, report fields) is fixed in tests/README.md.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unittest
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from _support import (
    DEMO_DIR,
    NEMO_EVALUATOR_DIR_ENV,
    EXIT_GATE_NOT_PASS,
    EXIT_OK,
    EXIT_REFUSED,
    NOT_ASSESSED,
    REQUIRED_TAGS,
    SHA256_ID,
    STUB_MARK,
    STUB_MARKER_FILE,
    WORKSPACE,
    canonical_bytes,
    checkpoint_dirs,
    count_lines,
    describe,
    expected_step_shards,
    file_sha256,
    load_manifest,
    load_report,
    manifest_entries,
    manifest_hash,
    parse_errors,
    read_jsonl,
    run_cli,
    slow,
    stage_dir,
    train_step_files,
)
from test_pairing import upstream_load_paired_records

# NeMo Evaluator per-sample record keys (eval_loop.py:790-803 at ea35ba3),
# written to results.jsonl by write_all (artifacts.py:98-102).
NEMO_RESULT_KEYS = {
    "problem_idx", "repeat", "reward", "scorer_rewards", "prompt", "model_response", "extracted_answer",
    "expected_answer", "scoring_details", "metadata", "tokens", "latency_ms",
}
# NeMo RL train_data_step<N>.jsonl keys (grpo_sync.py:1429-1456 at 4f39fd8) plus `idx` (logger.py:1436).
GRPO_LOG_KEYS = {
    "agent_ref", "attribution_key", "rewards", "filtered_rewards", "input_lengths", "token_loss_mask",
    "sample_loss_mask", "advantages", "generation_logprobs", "prev_logprobs", "token_ids", "content", "idx",
}
GRPO_LOG_REQUIRED = {"token_ids", "rewards", "token_loss_mask", "sample_loss_mask", "idx"}

# Design revision 11 (tests/README.md section 11.16, R11A-4): reports, manifests and the lockfile use item names
# (`n_items`, `samples_per_item`, `item_set_hash`, manifests/item_set.json).
REPORT_KEYS = {
    "schema_version", "tags", "stubs", "workflow", "gate_id", "hypothesis_type", "metric", "status",
    "delta_hat", "interval", "margin", "alpha_used", "power_target", "achieved_power", "mde", "mde_declared",
    "underpowered", "n_items", "samples_per_item", "eval_seeds", "sd_diff", "design_effect", "plan",
    "baseline_score", "candidate_score", "registration_hash", "registration_signed_by", "analysis_code_version",
    "item_set_hash", "evaluator_hash", "sampler_hash", "baseline_checkpoint_hash", "candidate_checkpoint_hash",
    "lockfile_hash", "independence_tier", "transfer_test", "contamination", "family_alpha", "created_at",
}
PLAN_KEYS = {"n_required", "mde", "design_effect", "sigma_plan", "sigma_used", "n_available", "underpowered"}
MANIFEST_NAMES = (
    "baseline_checkpoint", "trained_checkpoint", "training_shards", "item_set", "registration",
    "eval_baseline", "eval_candidate", "gate_report", "ledger",
)
GATE_STAT_KEYS = ("status", "delta_hat", "achieved_power", "mde", "underpowered", "n_items", "sd_diff",
                  "baseline_score", "candidate_score")
RUN_IDENTITY_COLUMNS = ("run_id", "attempt_id", "sample_id", "writer_id")
# The demo registration's hash, computed from the file (it changes whenever the file does).
# test_report_values checks that demo/spec.yaml pins this value.
DEMO_REG_SHA = hashlib.sha256((DEMO_DIR / "registration.yaml").read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Independent computations over a run directory
# ---------------------------------------------------------------------------

def eval_records(run_dir: Path, arm: str) -> dict[tuple[int, int], dict]:
    return upstream_load_paired_records()(run_dir / "eval" / arm)


def per_prompt_means(recs: dict[tuple[int, int], dict]) -> dict[int, float]:
    acc: dict[int, list[float]] = defaultdict(list)
    for (pid, _), r in recs.items():
        acc[pid].append(float(r["reward"]))
    return {pid: sum(v) / len(v) for pid, v in acc.items()}


def em_rate(recs) -> float:
    return sum(float(r["reward"]) for r in recs.values()) / len(recs)


def shard_list(dataset: Path) -> list[tuple[str, int]]:
    return [(p.name, count_lines(p)) for p in sorted(dataset.glob("*.jsonl"))]


def step_sizes(run_dir: Path) -> list[tuple[int, int]]:
    return [(n, count_lines(p)) for n, p in train_step_files(run_dir)]


def first_steps(run_dir: Path, dataset: Path) -> dict[str, int]:
    first: dict[str, int] = {}
    for step, shard in sorted(expected_step_shards(step_sizes(run_dir), shard_list(dataset))):
        first.setdefault(shard, step)
    return first


def impact(target: str, run_dir: Path) -> list[dict]:
    proc = run_cli(["impact", target, "--run-dir", str(run_dir), "--json"], cwd=WORKSPACE.demo, timeout=60)
    if proc.returncode != EXIT_OK:
        raise AssertionError(describe(proc))
    doc = json.loads(proc.stdout)
    return doc["results"]


# ---------------------------------------------------------------------------
# E2E-pass
# ---------------------------------------------------------------------------

@slow
class SlowE2EPass(unittest.TestCase):
    """MVP-SPEC section 5 "E2E-pass"; E2E-SKEL (verification-plan.md:256)."""

    @classmethod
    def setUpClass(cls):
        cls.proc, cls.run_dir = WORKSPACE.run("pass", "spec.yaml")

    def test_exit_zero(self):
        self.assertEqual(self.proc.returncode, EXIT_OK, describe(self.proc))
        self.assertEqual(self.run_dir.parent, WORKSPACE.runs_root("pass"))

    def test_gate_passes(self):
        rep = load_report(self.run_dir)
        self.assertEqual(rep["hypothesis_type"], "superiority")
        self.assertEqual(rep["status"], "PASS")
        self.assertGreater(rep["delta_hat"], 0.0)
        self.assertGreater(rep["interval"]["lower"], rep["margin"])
        self.assertEqual(rep["margin"], 0.0)
        self.assertFalse(rep["underpowered"])

    def test_trained_beats_baseline(self):
        base = eval_records(self.run_dir, "baseline")
        cand = eval_records(self.run_dir, "candidate")
        b, c = em_rate(base), em_rate(cand)
        self.assertGreater(c, b)
        # Baseline is partly capable: neither 0% nor 100% (MVP-SPEC section 3).
        self.assertGreater(b, 0.02)
        self.assertLess(b, 0.95)
        rep = load_report(self.run_dir)
        self.assertAlmostEqual(rep["baseline_score"], b, places=12)
        self.assertAlmostEqual(rep["candidate_score"], c, places=12)

    def test_delta_matches_independent_computation(self):
        """Per-prompt paired differences of raw scores, K averaged (05:97-99)."""
        base = per_prompt_means(eval_records(self.run_dir, "baseline"))
        cand = per_prompt_means(eval_records(self.run_dir, "candidate"))
        self.assertEqual(set(base), set(cand))
        diffs = [cand[p] - base[p] for p in sorted(base)]
        mean = sum(diffs) / len(diffs)
        sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1))
        rep = load_report(self.run_dir)
        self.assertEqual(rep["n_items"], len(diffs))
        self.assertAlmostEqual(rep["delta_hat"], mean, places=12)
        self.assertAlmostEqual(rep["sd_diff"], sd, places=12)

    def test_report_has_every_required_field(self):
        rep = load_report(self.run_dir)
        self.assertLessEqual(REPORT_KEYS, set(rep), REPORT_KEYS - set(rep))
        self.assertLessEqual(PLAN_KEYS, set(rep["plan"]))
        self.assertLessEqual({"lower", "upper", "level", "method"}, set(rep["interval"]))
        self.assertEqual(rep["tags"], list(REQUIRED_TAGS))
        for key in ("independence_tier", "transfer_test", "contamination", "family_alpha"):
            self.assertEqual(rep[key], NOT_ASSESSED, key)
        self.assertTrue(rep["stubs"])
        self.assertTrue(all(STUB_MARK in s for s in rep["stubs"]), rep["stubs"])

    def test_report_values(self):
        rep = load_report(self.run_dir)
        demo = WORKSPACE.demo
        self.assertEqual(rep["workflow"], "skeleton-addition-demo")
        self.assertEqual(rep["gate_id"], "quality_gate")
        self.assertEqual(rep["metric"], "exact_match")
        self.assertEqual(rep["registration_hash"], "sha256:" + DEMO_REG_SHA)
        self.assertEqual(rep["registration_hash"], file_sha256(demo / "registration.yaml"))
        # The spec pins the registration it was written against (05 section 3.7).
        self.assertIn(f"registration.yaml@sha256:{DEMO_REG_SHA}", (DEMO_DIR / "spec.yaml").read_text(encoding="utf-8"))
        self.assertEqual(rep["registration_signed_by"], "prototype-unsigned")
        self.assertEqual(rep["item_set_hash"], manifest_hash(demo / "data" / "heldout"))
        self.assertAlmostEqual(rep["alpha_used"], 0.025, places=12)
        self.assertAlmostEqual(rep["power_target"], 0.8, places=12)
        self.assertAlmostEqual(rep["mde_declared"], 0.10, places=12)
        self.assertAlmostEqual(rep["interval"]["level"], 0.95, places=12)
        self.assertEqual(rep["interval"]["method"], "cluster_bootstrap")
        self.assertEqual(rep["n_items"], 500)
        self.assertEqual(rep["samples_per_item"], 4)
        self.assertEqual(rep["eval_seeds"], [11, 12, 13, 14])
        self.assertEqual(rep["plan"]["n_required"], 172)          # see test_stats_plan
        self.assertEqual(rep["plan"]["n_available"], 500)
        self.assertFalse(rep["plan"]["underpowered"])
        self.assertEqual(rep["plan"]["sigma_plan"], "ucb80")
        self.assertEqual(rep["underpowered"], rep["achieved_power"] < rep["power_target"] or rep["plan"]["underpowered"])
        for key in ("evaluator_hash", "sampler_hash", "baseline_checkpoint_hash", "candidate_checkpoint_hash",
                    "lockfile_hash", "item_set_hash", "registration_hash"):
            self.assertRegex(rep[key], SHA256_ID, key)
        self.assertEqual(rep["baseline_checkpoint_hash"], manifest_hash(demo / "baseline"))
        final_step, final_dir = checkpoint_dirs(self.run_dir)[-1]
        self.assertEqual(rep["candidate_checkpoint_hash"], manifest_hash(final_dir))
        self.assertEqual(rep["lockfile_hash"], file_sha256(self.run_dir / "lockfile.json"))
        self.assertEqual(rep["analysis_code_version"], "gatedtrain 0.3.0")

    def test_text_report(self):
        text = (self.run_dir / "report" / "gate_report.txt").read_text(encoding="utf-8")
        self.assertRegex(text, r"(?m)^status: PASS$")
        for needle in ("delta_hat:", "interval:", "underpowered: false", "achieved_power:", "mde:",
                       "prototype", "unpromotable", "registration: unsigned", STUB_MARK):
            self.assertIn(needle, text)
        for key in ("independence_tier", "transfer_test", "contamination", "family_alpha"):
            self.assertRegex(text, rf"(?m)^{key}: {re.escape(NOT_ASSESSED)}$")
        self.assertIn("status: PASS", self.proc.stdout, "the CLI prints the text report")

    def test_results_in_nemo_evaluator_format(self):
        prompts = read_jsonl(WORKSPACE.demo / "data" / "heldout" / "prompts.jsonl")
        expected_keys = {(i, r) for i in range(len(prompts)) for r in range(4)}
        per_arm = {}
        for arm in ("baseline", "candidate"):
            with self.subTest(arm=arm):
                path = self.run_dir / "eval" / arm / "results.jsonl"
                lines = read_jsonl(path)
                self.assertEqual(len(lines), len(expected_keys))
                recs = eval_records(self.run_dir, arm)
                self.assertEqual(set(recs), expected_keys)
                for (pid, rep_idx), r in recs.items():
                    self.assertEqual(set(r), NEMO_RESULT_KEYS)
                    self.assertEqual(r["prompt"], prompts[pid]["prompt"])
                    self.assertEqual(r["expected_answer"], prompts[pid]["answer"])
                    self.assertIn(r["reward"], (0.0, 1.0))
                    self.assertEqual(r["reward"], 1.0 if r["model_response"] == r["expected_answer"] else 0.0)
                    # Registered evaluation seeds, repeat r uses eval_seeds[r] (05:97).
                    self.assertEqual(r["metadata"]["eval_seed"], [11, 12, 13, 14][rep_idx])
                    self.assertIs(r["metadata"]["skeleton_stub"], True)
                per_arm[arm] = recs
        if set(per_arm) != {"baseline", "candidate"}:      # an arm's subtest skipped (the pinned clone is unset)
            self.skipTest(f"an arm was skipped; set {NEMO_EVALUATOR_DIR_ENV}")
        # Common random numbers: both arms use the same seed per key (05:97).
        for key in expected_keys:
            self.assertEqual(per_arm["baseline"][key]["metadata"]["eval_seed"],
                             per_arm["candidate"][key]["metadata"]["eval_seed"])

    def test_manifests(self):
        """Content-addressed manifests, canonical JSON, manifest rule H7 hash (MVP-SPEC 2.8)."""
        for name in MANIFEST_NAMES:
            with self.subTest(manifest=name):
                path = self.run_dir / "manifests" / f"{name}.json"
                raw = path.read_bytes()
                doc = json.loads(raw)
                self.assertEqual(raw, canonical_bytes(doc), "canonical JSON (canonical-JSON rule C1-C3)")
                self.assertEqual(doc["artifact"], name)
                self.assertEqual(doc["manifest_hash_kind"], "p2_filelist_sha256")
                root = Path(doc["root"])
                self.assertTrue(root.is_dir(), root)
                self.assertEqual(doc["entries"], manifest_entries(root))
                self.assertEqual(doc["manifest_hash"], manifest_hash(root))
                self.assertIn("prototype", doc["tags"])
                self.assertIn("unpromotable", doc["tags"])
        m = lambda n: load_manifest(self.run_dir, n)  # noqa: E731
        self.assertEqual(Path(m("baseline_checkpoint")["root"]).resolve(), (WORKSPACE.demo / "baseline").resolve())
        self.assertEqual(Path(m("trained_checkpoint")["root"]).resolve(), checkpoint_dirs(self.run_dir)[-1][1].resolve())
        self.assertEqual(Path(m("training_shards")["root"]).resolve(), (WORKSPACE.demo / "data" / "train").resolve())
        self.assertEqual(Path(m("item_set")["root"]).resolve(), (WORKSPACE.demo / "data" / "heldout").resolve())
        self.assertEqual(Path(m("eval_baseline")["root"]).resolve(), (self.run_dir / "eval" / "baseline").resolve())
        self.assertEqual(Path(m("eval_candidate")["root"]).resolve(), (self.run_dir / "eval" / "candidate").resolve())
        self.assertEqual(Path(m("gate_report")["root"]).resolve(), (self.run_dir / "report").resolve())
        self.assertEqual(Path(m("ledger")["root"]).resolve(), (self.run_dir / "ledger").resolve())
        reg_root = Path(m("registration")["root"])
        self.assertEqual(file_sha256(reg_root / "registration.yaml"), "sha256:" + DEMO_REG_SHA)

    def test_lockfile(self):
        """Backend identity and config hash, no allow-list checks (MVP-SPEC 2.8)."""
        import torch

        raw = (self.run_dir / "lockfile.json").read_bytes()
        lock = json.loads(raw)
        self.assertEqual(raw, canonical_bytes(lock))
        self.assertEqual(lock["config_hash"], file_sha256(WORKSPACE.demo / "trainer.yaml"))
        self.assertEqual(lock["spec_sha256"], file_sha256(WORKSPACE.demo / "spec.yaml"))
        self.assertEqual(lock["registration_sha256"], "sha256:" + DEMO_REG_SHA)
        self.assertIs(lock["allow_list_checked"], False)
        for role in ("trainer", "evaluator"):
            b = lock["backend"][role]
            self.assertEqual(b["name"], "cpu_stub")
            self.assertEqual(b["package"], "gatedtrain")
            self.assertEqual(b["package_version"], "0.3.0")
            self.assertEqual(b["torch_version"], torch.__version__)
        for name in MANIFEST_NAMES:
            if name in ("gate_report",):
                continue
            self.assertEqual(lock["artifacts"][name], load_manifest(self.run_dir, name)["manifest_hash"], name)
        self.assertIn("prototype", lock["tags"])
        self.assertIn("unpromotable", lock["tags"])

    def test_trainer_outputs_have_nemo_rl_names_and_shapes(self):
        """grpo_sync.py:1323, :1353, :1408-1414, :1458-1460 at 4f39fd8; MVP-SPEC section 3."""
        steps = train_step_files(self.run_dir)
        self.assertTrue(steps)
        numbers = [n for n, _ in steps]
        self.assertEqual(numbers, list(range(1, len(numbers) + 1)), "train_data_step1..N, contiguous")
        total = 0
        for _, path in steps:
            lines = read_jsonl(path)
            self.assertTrue(lines)
            for i, line in enumerate(lines):
                self.assertLessEqual(set(line), GRPO_LOG_KEYS, f"{path.name}: no fields NeMo RL lacks")
                self.assertLessEqual(GRPO_LOG_REQUIRED, set(line), path.name)
                self.assertEqual(line["idx"], i)
            total += len(lines)
        ckpts = checkpoint_dirs(self.run_dir)
        final_step, final_dir = ckpts[-1]
        self.assertEqual(final_step, numbers[-1], "the last step is checkpointed")
        info = json.loads((final_dir / "training_info.json").read_text(encoding="utf-8"))
        self.assertEqual(info["consumed_samples"], total)
        self.assertEqual(info["total_steps"], final_step)
        for step, d in ckpts:
            self.assertTrue((d / "train_dataloader.pt").is_file(), d)
            self.assertTrue((d / "training_info.json").is_file(), d)

    def test_ledger(self):
        """Parquet ConsumptionRecord rows at precision `iteration` (05:341-350, 05:372-382)."""
        table = pq.read_table(self.run_dir / "ledger" / "consumption.parquet")
        rows = table.to_pylist()
        self.assertTrue(rows)
        from test_ledger_unit import CONSUMPTION_RECORD_FIELDS, DATA_REF_FIELDS

        self.assertLessEqual(CONSUMPTION_RECORD_FIELDS, set(table.column_names))
        self.assertLessEqual(DATA_REF_FIELDS, set(rows[0]["data_ref"]))
        dataset = WORKSPACE.demo / "data" / "train"
        expected = expected_step_shards(step_sizes(self.run_dir), shard_list(dataset))
        self.assertEqual({(r["step"], r["data_ref"]["shard"]) for r in rows}, expected)
        snapshot = manifest_hash(dataset)
        for r in rows:
            self.assertEqual(r["precision"], "iteration")
            self.assertIsNone(r["data_ref"]["row"])
            self.assertEqual(r["data_ref"]["snapshot"], snapshot)
            self.assertEqual(r["outcome"], "consumed")

    def test_impact_used_shards(self):
        """05 section 11.3 and 12.4 (05:352-354, 05:403-408): every checkpoint whose
        last_step_included >= first_step(S) possibly consumed S, at precision iteration."""
        dataset = WORKSPACE.demo / "data" / "train"
        first = first_steps(self.run_dir, dataset)
        self.assertTrue(first)
        baseline_hash = manifest_hash(WORKSPACE.demo / "baseline")
        ckpts = {manifest_hash(d): step for step, d in checkpoint_dirs(self.run_dir)}
        final_hash = manifest_hash(checkpoint_dirs(self.run_dir)[-1][1])
        for shard, f_step in sorted(first.items()):
            with self.subTest(shard=shard):
                results = impact(file_sha256(dataset / shard), self.run_dir)
                got = {r["checkpoint_hash"] for r in results}
                self.assertIn(final_hash, got)
                self.assertNotIn(baseline_hash, got)
                want = {h for h, step in ckpts.items() if step >= f_step}
                self.assertEqual(got, want)
                for r in results:
                    self.assertEqual(r["precision"], "iteration")
                    self.assertEqual(r["answer"], "possibly consumed")
                    self.assertEqual(r["stage_id"], "sft")
                    self.assertEqual(r["run_id"], self.run_dir.name)
                    self.assertEqual(r["last_step_included"], ckpts[r["checkpoint_hash"]])

    def test_impact_by_path_equals_by_hash(self):
        dataset = WORKSPACE.demo / "data" / "train"
        shard = sorted(first_steps(self.run_dir, dataset))[0]
        by_hash = impact(file_sha256(dataset / shard), self.run_dir)
        by_path = impact(str(dataset / shard), self.run_dir)
        key = lambda r: r["checkpoint_hash"]  # noqa: E731
        self.assertEqual(sorted(by_hash, key=key), sorted(by_path, key=key))

    def test_impact_unused_shards(self):
        unused = WORKSPACE.demo / "data" / "unused" / "shard-00000.jsonl"
        self.assertEqual(impact(file_sha256(unused), self.run_dir), [])
        self.assertEqual(impact(str(unused), self.run_dir), [])
        dataset = WORKSPACE.demo / "data" / "train"
        used = set(first_steps(self.run_dir, dataset))
        for name, _ in shard_list(dataset):
            if name not in used:
                with self.subTest(shard=name):
                    self.assertEqual(impact(file_sha256(dataset / name), self.run_dir), [])

    def test_impact_never_returns_baseline_for_stage_shard(self):
        baseline_hash = manifest_hash(WORKSPACE.demo / "baseline")
        dataset = WORKSPACE.demo / "data" / "train"
        for name, _ in shard_list(dataset):
            with self.subTest(shard=name):
                got = {r["checkpoint_hash"] for r in impact(file_sha256(dataset / name), self.run_dir)}
                self.assertNotIn(baseline_hash, got)

    def test_sd_zero_false(self):
        """Interim rule G-13: `sd_zero` is in the JSON and text reports; false on a real effect."""
        rep = load_report(self.run_dir)
        self.assertIs(rep["sd_zero"], False)
        text = (self.run_dir / "report" / "gate_report.txt").read_text(encoding="utf-8")
        self.assertRegex(text, r"(?m)^sd_zero: false$")

    def test_scores_over_paired_items_and_unpaired_counts(self):
        """Decision D-3: scores over the paired items only; `unpaired_baseline` and
        `unpaired_candidate` counts; a warning line when either is above 0."""
        from gatedtrain.report import render_text

        rep = load_report(self.run_dir)
        self.assertEqual(rep["unpaired_baseline"], 0)
        self.assertEqual(rep["unpaired_candidate"], 0)
        text = (self.run_dir / "report" / "gate_report.txt").read_text(encoding="utf-8")
        self.assertRegex(text, r"(?m)^unpaired_baseline: 0$")
        self.assertRegex(text, r"(?m)^unpaired_candidate: 0$")
        self.assertNotRegex(text, r"(?mi)^warning:.*unpaired")
        for key in ("unpaired_baseline", "unpaired_candidate"):
            with self.subTest(key=key):
                changed = json.loads(json.dumps(rep))
                changed[key] = 3
                self.assertRegex(render_text(changed), r"(?mi)^warning:.*unpaired")

    def test_disjoint_from_not_assessed(self):
        """Decision D-5: the spec declares `disjoint_from`; the skeleton does not check it
        and says so in the report."""
        rep = load_report(self.run_dir)
        self.assertEqual(rep["disjoint_from"], NOT_ASSESSED)
        text = (self.run_dir / "report" / "gate_report.txt").read_text(encoding="utf-8")
        self.assertRegex(text, rf"(?m)^disjoint_from: {re.escape(NOT_ASSESSED)}$")

    def test_stub_markers_on_outputs(self):
        """Every stand-in output carries skeleton-stub (MVP-SPEC sections 3 and 5)."""
        dirs = [WORKSPACE.demo / "baseline", stage_dir(self.run_dir) / "logs",
                self.run_dir / "eval" / "baseline", self.run_dir / "eval" / "candidate"]
        dirs += [d for _, d in checkpoint_dirs(self.run_dir)]
        for d in dirs:
            with self.subTest(dir=str(d)):
                marker = d / STUB_MARKER_FILE
                self.assertTrue(marker.is_file(), marker)
                self.assertIn(STUB_MARK, marker.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Second run: never overwrite, and determinism
# ---------------------------------------------------------------------------

@slow
class SlowNoOverwriteAndDeterminism(unittest.TestCase):
    """MVP-SPEC section 2 (last line) and section 5 "Determinism"."""

    @classmethod
    def setUpClass(cls):
        _, cls.first = WORKSPACE.run("pass", "spec.yaml")
        if WORKSPACE.has_run("pass2"):
            raise AssertionError("second run started before the snapshot was taken")
        cls.snapshot = manifest_hash(cls.first)
        cls.proc2, cls.second = WORKSPACE.run("pass2", "spec.yaml", runs_root_key="pass")

    def test_new_run_directory_and_first_untouched(self):
        self.assertEqual(self.proc2.returncode, EXIT_OK, describe(self.proc2))
        self.assertNotEqual(self.first, self.second)
        self.assertEqual(self.first.parent, self.second.parent)
        self.assertEqual(manifest_hash(self.first), self.snapshot, "the earlier run was modified")

    def test_identical_gate_statistics(self):
        a, b = load_report(self.first), load_report(self.second)
        for key in GATE_STAT_KEYS:
            self.assertEqual(a[key], b[key], key)
        self.assertEqual(a["interval"]["lower"], b["interval"]["lower"])
        self.assertEqual(a["interval"]["upper"], b["interval"]["upper"])
        self.assertEqual(a["candidate_checkpoint_hash"], b["candidate_checkpoint_hash"])

    def test_identical_content_hashes(self):
        for name in ("trained_checkpoint", "eval_baseline", "eval_candidate", "training_shards", "item_set"):
            with self.subTest(manifest=name):
                ha = load_manifest(self.first, name)["manifest_hash"]
                hb = load_manifest(self.second, name)["manifest_hash"]
                self.assertEqual(ha, hb)
        for arm in ("baseline", "candidate"):
            self.assertEqual(file_sha256(self.first / "eval" / arm / "results.jsonl"),
                             file_sha256(self.second / "eval" / arm / "results.jsonl"))
        for (sa, da), (sb, db) in zip(checkpoint_dirs(self.first), checkpoint_dirs(self.second)):
            self.assertEqual(sa, sb)
            self.assertEqual(manifest_hash(da), manifest_hash(db))

    def test_identical_ledger_content(self):
        """Ledger rows are equal once run-instance identity columns are dropped
        (gap G-5 in tests/README.md: 05 section 12.2 keys rows by run)."""
        def rows(run_dir):
            t = pq.read_table(run_dir / "ledger" / "consumption.parquet")
            keep = [c for c in t.column_names if c not in RUN_IDENTITY_COLUMNS]
            out = t.select(keep).to_pylist()
            return sorted(out, key=lambda r: json.dumps(r, sort_keys=True, default=str))

        self.assertEqual(rows(self.first), rows(self.second))
        self.assertEqual(json.loads((self.first / "lockfile.json").read_text())["config_hash"],
                         json.loads((self.second / "lockfile.json").read_text())["config_hash"])


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------

@slow
class SlowNegativeNoTraining(unittest.TestCase):
    """MVP-SPEC section 5: learning rate 0 gives a gate that is not PASS."""

    @classmethod
    def setUpClass(cls):
        cls.proc, cls.run_dir = WORKSPACE.run("no-train", "spec-no-train.yaml")

    def test_not_pass(self):
        rep = load_report(self.run_dir)
        self.assertNotEqual(rep["status"], "PASS")
        # R10A-5 (05 section 3.6, 05:145): identical arms give sd = 0 and a zero-width interval, which is
        # INCONCLUSIVE with reason zero_width_interval whatever the limiting power; `underpowered` is not
        # what makes it INCONCLUSIVE (the G-13 power limit stays 1.0). Replaces the rev-9 rule
        # "INCONCLUSIVE iff underpowered".
        self.assertEqual(rep["status"], "INCONCLUSIVE")
        self.assertIn("zero_width_interval", rep["status_reasons"])
        self.assertIs(rep["sd_zero"], True)
        self.assertNotIn("min_paired", rep["status_reasons"])
        self.assertEqual(self.proc.returncode, EXIT_GATE_NOT_PASS, describe(self.proc))

    def test_identical_model_gives_identical_samples(self):
        """lr = 0 leaves the weights unchanged; with common random numbers the
        candidate's samples equal the baseline's (05:97)."""
        base = eval_records(self.run_dir, "baseline")
        cand = eval_records(self.run_dir, "candidate")
        self.assertEqual(set(base), set(cand))
        for key in base:
            self.assertEqual(base[key]["model_response"], cand[key]["model_response"], key)
        self.assertEqual(load_report(self.run_dir)["delta_hat"], 0.0)

    def test_sd_zero_true(self):
        """Interim rule G-13: identical arms print `sd_zero: true` in the JSON and text reports."""
        rep = load_report(self.run_dir)
        self.assertIs(rep["sd_zero"], True)
        self.assertEqual(rep["sd_diff"], 0.0)
        text = (self.run_dir / "report" / "gate_report.txt").read_text(encoding="utf-8")
        self.assertRegex(text, r"(?m)^sd_zero: true$")


@slow
class SlowNegativeTooFewPrompts(unittest.TestCase):
    """MVP-SPEC section 5: below the planned n -> underpowered; a non-rejection is
    INCONCLUSIVE, never FAIL (05:82)."""

    @classmethod
    def setUpClass(cls):
        cls.proc, cls.run_dir = WORKSPACE.run("few-prompts", "spec-few-prompts.yaml")

    def test_underpowered(self):
        rep = load_report(self.run_dir)
        self.assertEqual(rep["n_items"], 30)
        self.assertEqual(rep["plan"]["n_required"], 172)
        self.assertTrue(rep["plan"]["underpowered"])
        self.assertTrue(rep["underpowered"])

    def test_never_fail(self):
        rep = load_report(self.run_dir)
        self.assertIn(rep["status"], {"PASS", "INCONCLUSIVE"})
        if rep["status"] == "PASS":
            self.assertGreater(rep["interval"]["lower"], rep["margin"])
        expected_exit = EXIT_OK if rep["status"] == "PASS" else EXIT_GATE_NOT_PASS
        self.assertEqual(self.proc.returncode, expected_exit, describe(self.proc))
        text = (self.run_dir / "report" / "gate_report.txt").read_text(encoding="utf-8")
        self.assertIn("underpowered: true", text)


@slow
class SlowNegativeBadTraining(unittest.TestCase):
    """MVP-SPEC section 5: shuffled labels damage the model -> FAIL on superiority."""

    @classmethod
    def setUpClass(cls):
        cls.proc, cls.run_dir = WORKSPACE.run("bad-train", "spec-bad-train.yaml")

    def test_superiority_fails(self):
        rep = load_report(self.run_dir)
        self.assertEqual(rep["status"], "FAIL")
        self.assertLess(rep["delta_hat"], 0.0)
        self.assertFalse(rep["underpowered"])
        self.assertEqual(self.proc.returncode, EXIT_GATE_NOT_PASS, describe(self.proc))


@slow
class SlowNegativeBadTrainingNonInferiority(unittest.TestCase):
    """MVP-SPEC section 5: the non-inferiority variant also FAILs."""

    @classmethod
    def setUpClass(cls):
        cls.proc, cls.run_dir = WORKSPACE.run("bad-train-ni", "spec-bad-train-ni.yaml")

    def test_non_inferiority_fails(self):
        rep = load_report(self.run_dir)
        self.assertEqual(rep["hypothesis_type"], "non_inferiority")
        self.assertEqual(rep["gate_id"], "quality_gate_ni")
        self.assertAlmostEqual(rep["margin"], 0.10, places=12)
        self.assertEqual(rep["status"], "FAIL")
        self.assertLess(rep["interval"]["lower"], -0.10)
        self.assertFalse(rep["underpowered"])
        self.assertEqual(self.proc.returncode, EXIT_GATE_NOT_PASS, describe(self.proc))


@slow
class SlowNegativeNoTrainFewPrompts(unittest.TestCase):
    """MVP review follow-up: an end-to-end underpowered non-rejection is INCONCLUSIVE
    (05:82; interim rule G-3), exit 1. lr = 0 on the 30-prompt set: Delta-hat = 0 with
    SD 0 (power limit 1.0, G-13), but n = 30 < n_required = 172 from the plan."""

    @classmethod
    def setUpClass(cls):
        cls.proc, cls.run_dir = WORKSPACE.run("no-train-few", "spec-no-train-few-prompts.yaml")

    def test_inconclusive(self):
        rep = load_report(self.run_dir)
        self.assertEqual(rep["status"], "INCONCLUSIVE")
        self.assertTrue(rep["underpowered"])
        self.assertTrue(rep["plan"]["underpowered"])
        self.assertEqual(rep["n_items"], 30)
        self.assertEqual(rep["delta_hat"], 0.0)
        self.assertIs(rep["sd_zero"], True)
        self.assertEqual(self.proc.returncode, EXIT_GATE_NOT_PASS, describe(self.proc))
        self.assertIn("status: INCONCLUSIVE", self.proc.stdout)


@slow
class SlowRegistrationChangesDuringCopy(unittest.TestCase):
    """Decision D-4: a registration whose bytes change while the run copies it gives a
    clean refusal: exit 3, REG-HASH, and no run directory left behind.

    The change is simulated in-process by making `shutil.copyfile` write altered bytes
    when it copies the registration. If the implementation does not copy the
    registration with `shutil.copyfile`, the test fails (finding L-4(b)) so that the
    copy point is re-pinned rather than silently skipped."""

    def test_clean_refusal(self):
        import contextlib
        import io
        import shutil as _shutil
        from unittest import mock

        WORKSPACE.ensure_baseline()
        demo = WORKSPACE.fresh_demo_copy("reg-copy-race")
        _shutil.copytree(WORKSPACE.demo / "baseline", demo / "baseline")
        runs_root = WORKSPACE.root / "runs-reg-copy-race"
        real_copyfile = _shutil.copyfile
        hits = []

        def racing_copyfile(src, dst, *a, **kw):
            result = real_copyfile(src, dst, *a, **kw)
            if Path(src).name == "registration.yaml":
                hits.append(str(src))
                with open(dst, "ab") as f:
                    f.write(b"# changed during the copy\n")
            return result

        from gatedtrain import cli

        err = io.StringIO()
        out = io.StringIO()
        with mock.patch("shutil.copyfile", racing_copyfile), contextlib.redirect_stderr(err), \
                contextlib.redirect_stdout(out):
            code = cli.main(["run", str(demo / "spec.yaml"), "--runs-root", str(runs_root)])
        # L-4(b): the copy point is pinned to shutil.copyfile; a change of copy mechanism
        # must fail here (and the test be re-pinned), never skip silently.
        self.assertTrue(hits, "the registration was not copied with shutil.copyfile; re-pin the D-4 copy point")
        self.assertEqual(code, EXIT_REFUSED, err.getvalue()[-3000:])
        errors = parse_errors(err.getvalue())
        self.assertTrue(any(e["rule"] == "REG-HASH" for e in errors), err.getvalue()[-3000:])
        self.assertFalse(runs_root.exists() and any(p.is_dir() for p in runs_root.iterdir()),
                         "no run directory may be left behind")


@slow
class SlowRegistrationVanishesAfterPreflight(unittest.TestCase):
    """Finding L-4(a): if copying the registration itself raises (here the file is
    deleted right after preflight), the run gives a clean refusal: exit 3 and no run
    directory left behind. The deletion point is pinned to `gatedtrain.runner.preflight`."""

    def test_clean_refusal(self):
        import contextlib
        import io
        import shutil as _shutil
        from unittest import mock

        WORKSPACE.ensure_baseline()
        demo = WORKSPACE.fresh_demo_copy("reg-vanishes")
        _shutil.copytree(WORKSPACE.demo / "baseline", demo / "baseline")
        runs_root = WORKSPACE.root / "runs-reg-vanishes"
        from gatedtrain import cli, runner

        real_preflight = runner.preflight
        calls = []

        def preflight_then_delete(*a, **kw):
            result = real_preflight(*a, **kw)
            calls.append(1)
            (demo / "registration.yaml").unlink()
            return result

        err, out = io.StringIO(), io.StringIO()
        with mock.patch.object(runner, "preflight", preflight_then_delete), contextlib.redirect_stderr(err), \
                contextlib.redirect_stdout(out):
            code = cli.main(["run", str(demo / "spec.yaml"), "--runs-root", str(runs_root)])
        self.assertTrue(calls, "preflight was not called through gatedtrain.runner.preflight; re-pin the test")
        self.assertEqual(code, EXIT_REFUSED, err.getvalue()[-3000:])
        self.assertTrue(parse_errors(err.getvalue()), "the refusal names the rule, the field and the fix")
        self.assertFalse(runs_root.exists() and any(p.is_dir() for p in runs_root.iterdir()),
                         "no run directory may be left behind")


@slow
class SlowBatchLargerThanData(unittest.TestCase):
    """Finding L-3: a trainer batch size larger than the training rows can never fill a
    batch under drop_last (D-1). `run` refuses in preflight: exit 3, an error on
    `artifacts.trainer_cfg.batch_size`, and no run directory."""

    def test_refused_in_preflight(self):
        WORKSPACE.ensure_baseline()
        runs_root = WORKSPACE.root / "runs-big-batch"
        proc = run_cli(["run", "spec-big-batch.yaml", "--runs-root", str(runs_root)], cwd=WORKSPACE.demo)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errors = parse_errors(proc.stderr)
        self.assertTrue(any(e["field"] == "artifacts.trainer_cfg.batch_size" for e in errors), describe(proc))
        self.assertFalse(runs_root.exists() and any(p.is_dir() for p in runs_root.iterdir()),
                         "no run directory may be created")


# ---------------------------------------------------------------------------
# Registration integrity through `run`
# ---------------------------------------------------------------------------

@slow
class SlowRegistrationIntegrity(unittest.TestCase):
    """MVP-SPEC section 5: a registration changed after the spec recorded its hash
    makes `run` refuse with a clear error, before any run directory exists (05:142)."""

    def test_run_refuses_changed_registration(self):
        WORKSPACE.ensure_baseline()
        demo = WORKSPACE.fresh_demo_copy("reg-integrity")
        import shutil

        shutil.copytree(WORKSPACE.demo / "baseline", demo / "baseline")
        reg = demo / "registration.yaml"
        reg.write_text(reg.read_text(encoding="utf-8") + "# amended after the hash was recorded\n", encoding="utf-8")
        runs_root = WORKSPACE.root / "runs-reg-integrity"
        proc = run_cli(["run", "spec.yaml", "--runs-root", str(runs_root)], cwd=demo)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errors = parse_errors(proc.stderr)
        self.assertTrue(any(e["rule"] == "REG-HASH" and e["field"] == "artifacts.registration.uri" for e in errors),
                        describe(proc))
        self.assertFalse(runs_root.exists() and any(runs_root.iterdir()), "no run directory may be created")


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------

@slow
class SlowWarmup(unittest.TestCase):
    """Baseline from a short warm-up run (MVP-SPEC section 3)."""

    def test_warmup_is_deterministic_and_never_overwrites(self):
        baseline = WORKSPACE.ensure_baseline()
        out = WORKSPACE.root / "warmup-again"
        proc = run_cli(["warmup", "warmup.yaml", "--out", str(out)], cwd=WORKSPACE.demo)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertEqual(manifest_hash(out), manifest_hash(baseline))
        before = manifest_hash(out)
        proc = run_cli(["warmup", "warmup.yaml", "--out", str(out)], cwd=WORKSPACE.demo)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertEqual(manifest_hash(out), before)

    def test_model_size(self):
        """A tiny character-level transformer, a few hundred thousand parameters."""
        from gatedtrain.stubs.model import count_parameters

        n = count_parameters(WORKSPACE.ensure_baseline())
        self.assertGreaterEqual(n, 100_000)
        self.assertLessEqual(n, 1_000_000)


if __name__ == "__main__":
    unittest.main()
