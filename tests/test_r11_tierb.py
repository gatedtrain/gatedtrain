"""Design revision 11 in `gatedtrain tierb-collect` (tests/README.md section 11.16).

Design (revision 12 files):
- R11-19 (C5): new writes use item names: report (`n_items`, `samples_per_item`, `eval_distribution`,
  resampling unit `item`), lineage gate role `items`, evaluation-set artifact type `item_set`
  (05 section 3.7, 04 section 4.2). Legacy inputs are read and written out in item names.
- R11-2, 05 section 3.4: expected items come from the registered item set, so a held-out problem absent from
  both arms is a missing item of each arm (reason `absent`), not a refusal; the report gives each arm's
  missing count and the count per reason. R11-9, 05 section 3.8: `seed` is required; a record without
  `metadata.eval_seed` is a missing item (`no_seed`, design choice).
- R11-10: the report gives `direction` and the natural-unit difference beside the oriented one.
- R11-15, 08 section 2.4 item 1: a cut is used only when complete under the completeness rule the adapter
  declares per library version; an interrupted write is never a cut. NeMo RL at 4f39fd8 publishes a
  checkpoint by renaming tmp_step_N to step_N (nemo_rl/utils/checkpoint_storage.py:47, :69-:98;
  nemo_rl/utils/checkpoint.py:379-392, :423-434). The lockfile records the declared rule.
- R11-18 (C4), 05 sections 11.2, 12.2: every ConsumptionRecord carries `derivation`; the Tier B ledger maps
  steps to shards from step sample counts and the sequential loader order (decision D-1), not from records
  emitted during training, so its records are `reconstructed` at precision `iteration` (design choice).

Fixture: the H-6 Scenario of test_tierb_collect (30 held-out problems, K = 2, seeds 11 and 12). Arms are
rewritten as files; `gatedtrain` is only called through the CLI.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from _r11 import rewrite, to_item_names
from _support import (EXIT_GATE_NOT_PASS, EXIT_OK, EXIT_REFUSED, describe, file_sha256, load_report,
                      manifest_hash, parse_errors, parse_run_dir, run_cli)
from _tierb import baseline_reward, candidate_reward
from test_r10_tierb import collect, drop_records, text_lines
from test_tierb_collect import Scenario, no_run_dir

NEMO_RL_SHORT = "4f39fd8"


def strip_seed(src: Path, dst: Path, problems: set[int]) -> Path:
    """A copy of merged arm `src` whose records for `problems` carry no metadata.eval_seed."""
    shutil.copytree(src, dst)
    out = []
    for line in (dst / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["problem_idx"] in problems:
            rec["metadata"].pop("eval_seed", None)
        out.append(json.dumps(rec))
    (dst / "results.jsonl").write_text("\n".join(out) + "\n", encoding="utf-8")
    return dst


class R11Case(unittest.TestCase):
    """One Scenario per class, inputs in revision 11 names unless `legacy`; `reg_extra` edits the
    registration after the rename."""

    legacy = False
    reg_extra = None

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-r11-tierb-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = Scenario(cls.tmp)
        cls.legacy_reg_sha = file_sha256(cls.sc.paths["registration"])
        if not cls.legacy:
            to_item_names(cls.sc.paths["spec"], cls.sc.paths["registration"], reg_extra=cls.reg_extra)
        elif cls.reg_extra is not None:
            rewrite(cls.sc.paths["spec"], cls.sc.paths["registration"], reg_mutate=cls.reg_extra)
        cls.n = 0

    def collect(self, **kw):
        type(self).n += 1
        runs_root = self.tmp / f"runs-{type(self).n}"
        return runs_root, collect(self.sc, runs_root, **kw)

    def gate(self, proc, status: str, exit_code: int = EXIT_GATE_NOT_PASS):
        self.assertEqual(proc.returncode, exit_code, describe(proc))
        run_dir = parse_run_dir(proc.stdout)
        rep = load_report(run_dir)
        self.assertEqual(rep["status"], status, {k: rep.get(k) for k in ("status", "status_reasons")})
        return run_dir, rep


# ---------------------------------------------------------------------------
# R11A-4: new writes use item names (revision 11 inputs)
# ---------------------------------------------------------------------------

class TestItemNamesWritten(R11Case):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.proc = collect(cls.sc, cls.tmp / "runs")

    def run_dir(self):
        self.assertEqual(self.proc.returncode, EXIT_OK, describe(self.proc))
        return parse_run_dir(self.proc.stdout)

    def test_report_fields(self):
        rep = load_report(self.run_dir())
        heldout = manifest_hash(self.tmp / "export" / "heldout")
        self.assertEqual(rep["status"], "PASS")
        self.assertEqual(rep["n_items"], 30)
        self.assertEqual(rep["samples_per_item"], 2)
        self.assertEqual(rep["unit"], "item")
        self.assertEqual(rep["direction"], "higher_is_better")
        self.assertEqual(rep["eval_distribution"], {"item_set_hash": heldout, "env_version": None,
                                                    "env_params_version": None})
        self.assertEqual(rep["interval"]["bootstrap"]["resampling_unit"], "item")
        self.assertEqual(rep["missing_reasons"], {"baseline": {}, "candidate": {}})
        self.assertEqual(rep["natural_units"], {"delta_hat": rep["delta_hat"], "lower": rep["interval"]["lower"],
                                                "upper": rep["interval"]["upper"]})
        for old in ("n_prompts", "samples_per_prompt", "prompt_set_hash"):
            self.assertNotIn(old, rep)

    def test_text_lines(self):
        run_dir = self.run_dir()
        lines = text_lines(run_dir)
        heldout = manifest_hash(self.tmp / "export" / "heldout")
        for line in ("n_items: 30", "direction: higher_is_better", f"item_set_hash: {heldout}"):
            self.assertIn(line, lines)
        self.assertFalse([ln for ln in lines if ln.startswith(("n_prompts:", "prompt_set_hash:"))])
        self.assertTrue([ln for ln in lines if ln.startswith("natural_units: ")])

    def test_lineage_role_items(self):
        edges = json.loads((self.run_dir() / "lineage" / "edges.json").read_text(encoding="utf-8"))
        gate = [a for a in edges["activities"] if a["kind"] == "gate"][0]
        roles = {u["role"]: u.get("hash") for u in gate["used"]}
        self.assertEqual(set(roles), {"model", "baseline", "items", "registration"})
        self.assertEqual(roles["items"], manifest_hash(self.tmp / "export" / "heldout"))

    def test_manifest_and_lockfile_names(self):
        run_dir = self.run_dir()
        self.assertFalse((run_dir / "manifests" / "prompt_set.json").exists())
        doc = json.loads((run_dir / "manifests" / "item_set.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["type"], "item_set")
        self.assertEqual(doc["manifest_hash"], manifest_hash(self.tmp / "export" / "heldout"))
        lock = json.loads((run_dir / "lockfile.json").read_bytes())
        self.assertIn("item_set", lock["artifacts"])
        self.assertNotIn("prompt_set", lock["artifacts"])

    def test_ledger_derivation_reconstructed(self):
        """R11A-18: every Tier B record carries `derivation: reconstructed` at precision `iteration`."""
        table = pq.read_table(self.run_dir() / "ledger" / "consumption.parquet")
        self.assertIn("derivation", table.schema.names)
        rows = table.to_pylist()
        self.assertTrue(rows)
        self.assertEqual({r["derivation"] for r in rows}, {"reconstructed"})
        self.assertEqual({r["precision"] for r in rows}, {"iteration"})

    def test_impact_shows_derivation(self):
        run_dir = self.run_dir()
        proc = run_cli(["impact", str(self.tmp / "export" / "train" / "shard-00001.jsonl"), "--run-dir",
                        str(run_dir), "--json"], cwd=self.tmp, timeout=60)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        results = json.loads(proc.stdout)["results"]
        self.assertTrue(results)
        self.assertEqual({r["derivation"] for r in results}, {"reconstructed"})
        text = run_cli(["impact", str(self.tmp / "export" / "train" / "shard-00001.jsonl"), "--run-dir",
                        str(run_dir)], cwd=self.tmp, timeout=60)
        self.assertIn("derivation reconstructed", text.stdout)

    def test_completeness_rule_declared(self):
        """R11A-20: the lockfile's trainer backend records the completeness rule the adapter declares for this
        NeMo RL version."""
        tr = json.loads((self.run_dir() / "lockfile.json").read_bytes())["backend"]["trainer"]
        rule = tr.get("checkpoint_completeness")
        self.assertIsInstance(rule, dict, tr)
        self.assertTrue(str(rule.get("nemo_rl_commit", "")).startswith(NEMO_RL_SHORT), rule)
        self.assertIn("tmp_step", json.dumps(rule))


# ---------------------------------------------------------------------------
# R11A-2 / R11A-4: legacy inputs are read and written out in item names
# ---------------------------------------------------------------------------

class TestLegacyInputs(R11Case):
    legacy = True

    def test_collect_writes_item_names(self):
        _, proc = self.collect()
        run_dir, rep = self.gate(proc, "PASS", EXIT_OK)
        self.assertEqual(rep["registration_hash"], self.legacy_reg_sha, "the hash of the file as written")
        self.assertEqual(rep["n_items"], 30)
        self.assertEqual(rep["samples_per_item"], 2)
        self.assertEqual(rep["direction"], "higher_is_better")
        self.assertEqual(rep["eval_distribution"]["item_set_hash"], manifest_hash(self.tmp / "export" / "heldout"))
        for old in ("n_prompts", "samples_per_prompt", "prompt_set_hash"):
            self.assertNotIn(old, rep)
        self.assertTrue((run_dir / "manifests" / "item_set.json").is_file())


# ---------------------------------------------------------------------------
# R11A-17: the gate's expected items are the registered item set; seedless records are missing
# ---------------------------------------------------------------------------

class TestRegisteredItemSet(R11Case):

    def test_problem_absent_from_both_arms_is_missing(self):
        """Problem 29 is in the held-out set and in neither arm: two missing items per arm (K = 2), MISSING
        at the default bound. Revision 10 refused it (TIERB-PROMPTS (a), union of the arms)."""
        base = self.sc.arm("baseline", baseline_reward, root=self.tmp / "e29", drop={29})
        cand = self.sc.arm("candidate", candidate_reward, root=self.tmp / "e29", drop={29})
        _, proc = self.collect(baseline=base, candidate=cand)
        run_dir, rep = self.gate(proc, "MISSING")
        self.assertIn("missing_above_bound", rep["status_reasons"])
        self.assertEqual((rep["missing_baseline"], rep["missing_candidate"]), (2, 2))
        self.assertEqual(rep["missing_reasons"], {"baseline": {"absent": 2}, "candidate": {"absent": 2}})
        self.assertEqual(rep["n_items"], 29)

    def test_extra_problem_still_refused(self):
        """A problem outside the registered set stays a refusal (TIERB-PROMPTS)."""
        src = self.sc.candidate
        dst = self.tmp / "e-extra" / "candidate"
        shutil.copytree(src, dst)
        lines = (dst / "results.jsonl").read_text(encoding="utf-8").splitlines()
        rec = json.loads(lines[0])
        rec["problem_idx"] = 30
        (dst / "results.jsonl").write_text("\n".join(lines + [json.dumps(rec)]) + "\n", encoding="utf-8")
        runs_root, proc = self.collect(candidate=dst)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue([e for e in parse_errors(proc.stderr) if e["rule"] == "TIERB-PROMPTS"], describe(proc))
        self.assertTrue(no_run_dir(runs_root))

    def test_records_without_seed_are_missing(self):
        """Problem 5's candidate records carry no metadata.eval_seed: two missing items (`no_seed`)."""
        cand = strip_seed(self.sc.candidate, self.tmp / "e-noseed" / "candidate", {5})
        _, proc = self.collect(candidate=cand)
        run_dir, rep = self.gate(proc, "MISSING")
        self.assertEqual(rep["missing_reasons"]["candidate"], {"no_seed": 2})
        self.assertEqual(rep["missing_candidate"], 2)
        self.assertEqual(rep["n_items"], 29)

    def test_one_repeat_missing_reason_absent(self):
        cand = drop_records(self.sc.candidate, self.tmp / "e-drop1" / "candidate", {(5, 1)})
        _, proc = self.collect(candidate=cand)
        _, rep = self.gate(proc, "MISSING")
        self.assertEqual(rep["missing_reasons"]["candidate"], {"absent": 1})
        self.assertEqual(rep["n_items"], 30, "item 5 keeps repeat 0 in both arms: still a complete pair")


class TestRegisteredItemSetWithinBound(R11Case):
    reg_extra = staticmethod(lambda d: d.__setitem__("pairing", {"max_unpaired": 4}))

    def test_problem_absent_from_both_arms_within_bound(self):
        base = self.sc.arm("baseline", baseline_reward, root=self.tmp / "e29", drop={29})
        cand = self.sc.arm("candidate", candidate_reward, root=self.tmp / "e29", drop={29})
        _, proc = self.collect(baseline=base, candidate=cand)
        _, rep = self.gate(proc, "PASS", EXIT_OK)
        self.assertEqual((rep["missing_baseline"], rep["missing_candidate"]), (2, 2))
        self.assertEqual(rep["n_items"], 29)


class TestItemSetHashChecked(R11Case):
    reg_extra = staticmethod(lambda d: d["eval_distribution"].__setitem__("item_set_hash", "sha256:" + "7" * 64))

    def test_wrong_item_set_hash_refused(self):
        runs_root, proc = self.collect()
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue([e for e in parse_errors(proc.stderr)
                         if e["rule"] == "REG-FIELD" and e["field"] == "registration.eval_distribution.item_set_hash"],
                        describe(proc))
        self.assertTrue(no_run_dir(runs_root))


# ---------------------------------------------------------------------------
# R11A-20: an interrupted checkpoint write is never a cut
# ---------------------------------------------------------------------------

class TestInterruptedCheckpoint(R11Case):

    def test_tmp_step_directory_is_ignored(self):
        """NeMo RL writes step 15 into tmp_step_15 and was interrupted before the rename: no cut, no lineage row,
        no manifest entry; the collect still PASSes on the complete cuts."""
        stage = self.tmp / "stage-interrupted"
        shutil.copytree(self.sc.stage_out, stage)
        tmp_ckpt = stage / "checkpoints" / "tmp_step_15"
        (tmp_ckpt / "policy" / "weights").mkdir(parents=True)
        (tmp_ckpt / "training_info.json").write_text(json.dumps({"consumed_samples": 30, "current_step": 15}),
                                                     encoding="utf-8")
        _, proc = self.collect(stage_out=stage)
        run_dir, rep = self.gate(proc, "PASS", EXIT_OK)
        ckpts = pq.read_table(run_dir / "lineage" / "checkpoints.parquet").to_pylist()
        self.assertEqual(sorted(r["last_step_included"] for r in ckpts), [5, 10])
        self.assertFalse(any("tmp_step" in str(r.get("path", "")) for r in ckpts))
        tr = json.loads((run_dir / "lockfile.json").read_bytes())["backend"]["trainer"]
        self.assertIn("checkpoint_completeness", tr)


if __name__ == "__main__":
    unittest.main()
