"""Ledger writer on NeMo RL-shaped trainer outputs (MVP-SPEC section 2.9;
05 sections 11.2 and 12.2).

The fixture has the names and shapes NeMo RL writes at 4f39fd8:
- per-step `train_data_step<N>.jsonl` written by `log_batched_dict_as_jsonl`
  (grpo_sync.py:1458-1460; logger.py:1409-1438): one JSON object per sample
  with the log_data keys of grpo_sync.py:1429-1456 plus `idx` (logger.py:1436), and no dataset
  row ID;
- `step_<N>/training_info.json` holding `consumed_samples` (grpo_sync.py:1323,
  :1353; checkpoint.py:406-412) and `step_<N>/train_dataloader.pt`
  (grpo_sync.py:1408-1414; data/utils.py:69-81).

Shards (file-name order) with row counts 10, 7, 5, 5; five steps of 4 samples.
Global sample ranges per step and the shards they fall in:
  step 1: [0, 4)   -> shard-00000
  step 2: [4, 8)   -> shard-00000
  step 3: [8, 12)  -> shard-00000 (8, 9), shard-00001 (10, 11)
  step 4: [12, 16) -> shard-00001
  step 5: [16, 20) -> shard-00001 (16), shard-00002 (17, 18, 19)
shard-00003 is never reached.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from _support import expected_step_shards, manifest_hash

CONSUMPTION_RECORD_FIELDS = {
    # 05 section 12.2 (05:372-382)
    "run_id", "attempt_id", "step", "sample_id", "data_ref", "parent_ref", "outcome", "reason",
    "rule_version", "gen_version", "reward_epoch", "rewards", "sampler_config_hash", "seed_inputs",
    "chunk", "precision", "writer_id",
}
DATA_REF_FIELDS = {"snapshot", "shard", "row", "mixture", "component"}  # 05 section 12.1 (05:361)

EXPECTED = {
    (1, "shard-00000.jsonl"), (2, "shard-00000.jsonl"),
    (3, "shard-00000.jsonl"), (3, "shard-00001.jsonl"),
    (4, "shard-00001.jsonl"),
    (5, "shard-00001.jsonl"), (5, "shard-00002.jsonl"),
}


def build_outputs(root: Path, *, shard_rows=(10, 7, 5, 5), step_lines=(4, 4, 4, 4, 4),
                  checkpoints=((3, 12), (5, 20))) -> dict[str, Path]:
    """NeMo RL-shaped trainer outputs: shards, per-step logs, step_<N> checkpoints.

    step_lines: lines in train_data_step1..N.jsonl (one per sample).
    checkpoints: (step, consumed_samples) pairs written to training_info.json.
    """
    import torch

    dataset = root / "dataset"
    dataset.mkdir(parents=True)
    for i, rows in enumerate(shard_rows):
        with (dataset / f"shard-{i:05d}.jsonl").open("w", encoding="utf-8") as f:
            for j in range(rows):
                f.write(json.dumps({"prompt": f"{i}+{j}=", "answer": str(i + j)}, sort_keys=True) + "\n")
    logs = root / "logs"
    logs.mkdir()
    for step, n_lines in enumerate(step_lines, start=1):
        with (logs / f"train_data_step{step}.jsonl").open("w", encoding="utf-8") as f:
            for idx in range(n_lines):
                sample = {
                    "rewards": 1.0, "input_lengths": 6, "token_loss_mask": [0, 0, 0, 1, 1, 1],
                    "sample_loss_mask": 1.0, "advantages": [0.0] * 6, "generation_logprobs": [0.0] * 6,
                    "prev_logprobs": [0.0] * 6, "token_ids": [1, 2, 3, 4, 5, 6], "idx": idx,
                }
                f.write(json.dumps(sample) + "\n")
    ckpts = root / "checkpoints"
    for step, consumed in checkpoints:
        d = ckpts / f"step_{step}"
        d.mkdir(parents=True)
        info = {"consumed_samples": consumed, "current_step": step, "total_steps": step,
                "current_epoch": 0, "total_valid_tokens": 0}
        (d / "training_info.json").write_text(json.dumps(info), encoding="utf-8")
        torch.save({"samples_yielded": consumed}, d / "train_dataloader.pt")
    return {"dataset": dataset, "logs": logs, "checkpoints": ckpts}


def build_fixture(root: Path) -> dict[str, Path]:
    return build_outputs(root)


def ledger_rows(paths: dict[str, Path], root: Path, **kw) -> list[dict]:
    from gatedtrain.ledger import write_ledger

    out = root / "ledger" / "consumption.parquet"
    write_ledger(paths["logs"], paths["checkpoints"], paths["dataset"], out, run_id="run-fixture",
                 attempt_id="0", **kw)
    return pq.read_table(out).to_pylist()


class TestLedgerWriter(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gatedtrain-ledger-"))
        self.paths = build_fixture(self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def write(self):
        from gatedtrain.ledger import write_ledger

        out = self.root / "ledger" / "consumption.parquet"
        write_ledger(self.paths["logs"], self.paths["checkpoints"], self.paths["dataset"], out,
                     run_id="run-fixture", attempt_id="0")
        self.assertTrue(out.is_file())
        return pq.read_table(out).to_pylist()

    def test_reference_mapping(self):
        """Guards the independent helper against the hand computation in the docstring."""
        sizes = [(s, 4) for s in range(1, 6)]
        shards = [("shard-00000.jsonl", 10), ("shard-00001.jsonl", 7), ("shard-00002.jsonl", 5),
                  ("shard-00003.jsonl", 5)]
        self.assertEqual(expected_step_shards(sizes, shards), EXPECTED)

    def test_schema(self):
        rows = self.write()
        self.assertTrue(rows)
        self.assertLessEqual(CONSUMPTION_RECORD_FIELDS, set(rows[0]), "05 section 12.2 fields")
        self.assertLessEqual(DATA_REF_FIELDS, set(rows[0]["data_ref"]), "05 section 12.1 data_ref fields")

    def test_every_step_maps_to_its_shards(self):
        rows = self.write()
        pairs = {(r["step"], r["data_ref"]["shard"]) for r in rows}
        self.assertEqual(pairs, EXPECTED)
        self.assertEqual({r["step"] for r in rows}, {1, 2, 3, 4, 5}, "no step may be omitted")

    def test_precision_iteration_and_no_row_ids(self):
        rows = self.write()
        snapshot = manifest_hash(self.paths["dataset"])
        for r in rows:
            self.assertEqual(r["precision"], "iteration")                   # 05 section 11.2
            self.assertIsNone(r["data_ref"]["row"], "no row-level lineage")  # no row-level lineage in the prototype
            self.assertEqual(r["data_ref"]["snapshot"], snapshot)
            self.assertEqual(r["outcome"], "consumed")                       # 05 section 12.3
            self.assertEqual(r["run_id"], "run-fixture")

    def test_unused_shard_absent(self):
        rows = self.write()
        self.assertNotIn("shard-00003.jsonl", {r["data_ref"]["shard"] for r in rows})


class TestLedgerLoaderSemantics(unittest.TestCase):
    """MVP review follow-up: decisions D-1, D-2 and interim rule G-7."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gatedtrain-ledger-sem-"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_wrap_around_drop_last(self):
        """D-1: StatefulDataLoader(..., drop_last=True) (nemo_rl/algorithms/grpo.py:883-890,
        drop_last=True at :888, 4f39fd8). Shards of 6, 2 and 3 rows (11 rows), batches of 4:
        each epoch yields 2 batches (rows 0-7); rows 8-10 (all of shard-00002) are skipped
        and epoch 2 restarts at row 0.
          step 1: rows 0-3 -> shard-00000
          step 2: rows 4-7 -> shard-00000 (4, 5), shard-00001 (6, 7)
          step 3: epoch 2, rows 0-3 -> shard-00000
          step 4: rows 4-7 -> shard-00000, shard-00001
        A continuous wrap would put rows 8-10 (shard-00002) and row 0 into step 3."""
        paths = build_outputs(self.root, shard_rows=(6, 2, 3), step_lines=(4, 4, 4, 4), checkpoints=((4, 16),))
        expected = {(1, "shard-00000.jsonl"), (2, "shard-00000.jsonl"), (2, "shard-00001.jsonl"),
                    (3, "shard-00000.jsonl"), (4, "shard-00000.jsonl"), (4, "shard-00001.jsonl")}
        self.assertEqual(expected_step_shards([(s, 4) for s in range(1, 5)],
                                              [("shard-00000.jsonl", 6), ("shard-00001.jsonl", 2),
                                               ("shard-00002.jsonl", 3)]), expected)
        rows = ledger_rows(paths, self.root)
        self.assertEqual({(r["step"], r["data_ref"]["shard"]) for r in rows}, expected)
        self.assertNotIn("shard-00002.jsonl", {r["data_ref"]["shard"] for r in rows})

    def test_zero_sample_step_maps_to_no_shard(self):
        """D-2: step 3 logs no samples. Steps 1, 2: rows 0-7 (shard-00000); step 4: rows 8-11
        (shard-00000 8, 9; shard-00001 10, 11); step 5: rows 12-15 (shard-00001)."""
        paths = build_outputs(self.root, step_lines=(4, 4, 0, 4, 4), checkpoints=((5, 16),))
        rows = ledger_rows(paths, self.root)
        self.assertEqual({(r["step"], r["data_ref"]["shard"]) for r in rows},
                         {(1, "shard-00000.jsonl"), (2, "shard-00000.jsonl"), (4, "shard-00000.jsonl"),
                          (4, "shard-00001.jsonl"), (5, "shard-00001.jsonl")})
        self.assertNotIn(3, {r["step"] for r in rows})

    def test_consumed_samples_mismatch_is_an_error(self):
        """consumed_samples (grpo_sync.py:1353) must equal the samples logged up to the cut."""
        paths = build_outputs(self.root, checkpoints=((3, 12), (5, 21)))
        with self.assertRaises(ValueError):
            ledger_rows(paths, self.root)

    def test_orphan_steps_after_last_cut_excluded(self):
        """Records not covered by any committed cut are orphaned and excluded (05:409)."""
        paths = build_outputs(self.root, step_lines=(4, 4, 4, 4, 4, 4, 4), checkpoints=((3, 12), (5, 20)))
        rows = ledger_rows(paths, self.root)
        self.assertEqual({r["step"] for r in rows}, {1, 2, 3, 4, 5})
        self.assertEqual({(r["step"], r["data_ref"]["shard"]) for r in rows}, EXPECTED)

    def test_samples_per_position_tier_b(self):
        """G-7: NeMo RL's consumed_samples counts prompts (grpo_sync.py:1323) while the
        step log holds prompts x generations. 4 prompts x 4 generations = 16 lines per
        step; with samples_per_position=4 the mapping equals the 4-row-per-step fixture."""
        paths = build_outputs(self.root, step_lines=(16, 16, 16, 16, 16), checkpoints=((3, 12), (5, 20)))
        rows = ledger_rows(paths, self.root, samples_per_position=4)
        self.assertEqual({(r["step"], r["data_ref"]["shard"]) for r in rows}, EXPECTED)
        self.assertEqual(expected_step_shards([(s, 16) for s in range(1, 6)],
                                              [("shard-00000.jsonl", 10), ("shard-00001.jsonl", 7),
                                               ("shard-00002.jsonl", 5), ("shard-00003.jsonl", 5)],
                                              samples_per_position=4), EXPECTED)


if __name__ == "__main__":
    unittest.main()
