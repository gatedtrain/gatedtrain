"""Impact query on the ledger fixture (05 section 11.3, 05:352-354; section 12.4,
05:403-408). MVP review follow-up, item 5: the query must tell checkpoints
apart, not only say whether the stage used a shard.

Fixture (test_ledger_unit.build_fixture): shards of 10, 7, 5, 5 rows, five
steps of 4 samples, checkpoints step_3 and step_5. First steps:
shard-00000 -> 1, shard-00001 -> 3, shard-00002 -> 5; shard-00003 is never read.

So step_3 (last_step_included 3) possibly consumed shard-00000 and
shard-00001 but not shard-00002, which was first reached at step 5.

The lineage tables are written in the layout `gatedtrain.impact` documents
(lineage/shards.parquet and lineage/checkpoints.parquet; impact.py module
docstring), so this test pins that layout as part of the interface.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from _support import file_sha256, manifest_hash
from test_ledger_unit import build_fixture

RUN_ID = "run-fixture"
STAGE_ID = "sft"


class TestImpactQueryUnit(unittest.TestCase):
    def setUp(self):
        from gatedtrain.ledger import write_ledger

        self.root = Path(tempfile.mkdtemp(prefix="gatedtrain-impact-"))
        self.run_dir = self.root / RUN_ID
        self.run_dir.mkdir()
        self.paths = build_fixture(self.run_dir / "fixture")
        write_ledger(self.paths["logs"], self.paths["checkpoints"], self.paths["dataset"],
                     self.run_dir / "ledger" / "consumption.parquet", run_id=RUN_ID, attempt_id="0")
        self.ckpt = {step: manifest_hash(self.paths["checkpoints"] / f"step_{step}") for step in (3, 5)}
        snapshot = manifest_hash(self.paths["dataset"])
        (self.run_dir / "lineage").mkdir()
        pq.write_table(pa.Table.from_pylist([
            {"run_id": RUN_ID, "stage_id": STAGE_ID, "checkpoint_hash": h, "last_step_included": step,
             "path": f"fixture/checkpoints/step_{step}"} for step, h in sorted(self.ckpt.items())
        ]), self.run_dir / "lineage" / "checkpoints.parquet")
        pq.write_table(pa.Table.from_pylist([
            {"run_id": RUN_ID, "stage_id": STAGE_ID, "snapshot": snapshot, "shard": p.name,
             "shard_sha256": file_sha256(p)} for p in sorted(self.paths["dataset"].glob("*.jsonl"))
        ]), self.run_dir / "lineage" / "shards.parquet")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def query(self, shard_name: str) -> list[dict]:
        from gatedtrain.impact import impact_query

        return impact_query(file_sha256(self.paths["dataset"] / shard_name), run_dir=self.run_dir)

    def test_shard_read_from_step_1(self):
        got = self.query("shard-00000.jsonl")
        self.assertEqual({r["checkpoint_hash"] for r in got}, {self.ckpt[3], self.ckpt[5]})
        for r in got:
            self.assertEqual(r["precision"], "iteration")
            self.assertEqual(r["answer"], "possibly consumed")
            self.assertEqual(r["first_step"], 1)

    def test_shard_first_read_at_step_3(self):
        got = self.query("shard-00001.jsonl")
        self.assertEqual({r["checkpoint_hash"] for r in got}, {self.ckpt[3], self.ckpt[5]})

    def test_checkpoint_saved_before_shard_reached_is_not_returned(self):
        got = self.query("shard-00002.jsonl")
        self.assertEqual({r["checkpoint_hash"] for r in got}, {self.ckpt[5]})
        self.assertEqual([r["last_step_included"] for r in got], [5])
        self.assertEqual(got[0]["first_step"], 5)

    def test_shard_in_dataset_never_read_returns_empty(self):
        self.assertEqual(self.query("shard-00003.jsonl"), [])

    def test_by_path_equals_by_hash(self):
        from gatedtrain.impact import impact_query

        path = str(self.paths["dataset"] / "shard-00002.jsonl")
        self.assertEqual(impact_query(path, run_dir=self.run_dir), self.query("shard-00002.jsonl"))


if __name__ == "__main__":
    unittest.main()
