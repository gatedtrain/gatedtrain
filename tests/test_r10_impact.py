"""Impact output for orphaned and superseded consumption records (R10A-9).

The fixture adds two step logs after the final committed checkpoint. Both read
shard-00002, which contributes two orphaned entries without a checkpoint.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from _support import EXIT_OK, describe, manifest_hash, parse_run_dir, run_cli
from test_tierb_collect import Scenario


class TestImpactCounts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-r10-impact-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = Scenario(cls.tmp)
        exp = cls.sc.stage_out / "logs" / "exp_001"
        for step in (11, 12):
            shutil.copyfile(exp / "train_data_step10.jsonl", exp / f"train_data_step{step}.jsonl")
        cls.proc = cls.sc.collect()[1]

    def run_dir(self) -> Path:
        self.assertEqual(self.proc.returncode, EXIT_OK, describe(self.proc))
        return parse_run_dir(self.proc.stdout)

    def impact(self, shard: str, *, as_json: bool = True):
        args = ["impact", str(self.tmp / "export" / "train" / shard), "--run-dir", str(self.run_dir())]
        proc = run_cli(args + (["--json"] if as_json else []), cwd=self.tmp, timeout=60)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        return json.loads(proc.stdout) if as_json else proc.stdout

    def test_orphaned_shard(self):
        out = self.impact("shard-00002.jsonl")
        self.assertEqual(out["results"], [], "orphaned records enter no rollup (05:424)")
        self.assertEqual(out["counts"], {"orphaned": 2, "superseded": 0})

    def test_covered_shard(self):
        out = self.impact("shard-00001.jsonl")
        ck10 = manifest_hash(self.sc.stage_out / "checkpoints" / "step_10")
        self.assertEqual({r["checkpoint_hash"] for r in out["results"]}, {ck10})
        self.assertEqual(out["counts"], {"orphaned": 0, "superseded": 0})

    def test_unused_shard(self):
        out = self.impact("shard-00003.jsonl")
        self.assertEqual(out["results"], [])
        self.assertEqual(out["counts"], {"orphaned": 0, "superseded": 0})

    def test_text_line(self):
        text = self.impact("shard-00002.jsonl", as_json=False)
        self.assertIn("orphaned: 2, superseded: 0", text.splitlines())


if __name__ == "__main__":
    unittest.main()
