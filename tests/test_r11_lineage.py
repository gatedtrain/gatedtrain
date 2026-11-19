"""Design revision 11 in the ledger and the impact query (tests/README.md section 11.16).

Design (revision 12 files):
- R11-18 (C4), 05 section 11.2 "Derivation", section 12.2 (`derivation` field): each ConsumptionRecord carries
  `derivation` in {observed, reconstructed}; a reconstructed record claims at most `iteration` precision
  unless the adapter passed a replay conformance test, after which `row` is allowed; query results and
  impact answers show the derivation.
- R11-16 (C2), 05 section 12.4 "Superseded and orphaned records", section 13.2 impact join: a membership
  question reports superseded and orphaned step ranges per segment and shard (a row-level question reports
  record counts with their precision). The prototype's impact query ("which checkpoints possibly consumed
  shard S") is a membership question.
- R11-19: existing artifacts stay readable. The real long run's run directory (fixtures/r11/longrun, written
  by revision 9/10 code: no `derivation` column, no lineage/orphaned.parquet, gate role `prompts`) answers
  impact queries unchanged.

Interfaces (proposal): `ledger.check_precision(precision, derivation, *, replay_conformance=False)`
raises ValueError for a disallowed pair; `ledger.SCHEMA` has a string field `derivation`; `gatedtrain impact
--json` results carry `derivation` (null for records written before revision 11) and a top-level
`ranges: {orphaned: [...], superseded: [...]}`, each range `{run_id, attempt_id, shard, first_step,
last_step}`; the text output adds `derivation <value>` to each result line and one line per range,
`orphaned_range: attempt <a> shard <shard> steps <first>-<last>`, or `orphaned_ranges: none`, and the same
for `superseded`.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from _r11 import LONGRUN
from _support import EXIT_OK, describe, parse_run_dir, run_cli
from test_tierb_collect import Scenario


# ---------------------------------------------------------------------------
# R11A-18: derivation
# ---------------------------------------------------------------------------

class TestDerivationRule(unittest.TestCase):

    def check(self, *a, **kw):
        from gatedtrain.ledger import check_precision

        return check_precision(*a, **kw)

    def test_allowed(self):
        for precision in ("row", "step_range", "iteration", "stage"):
            self.check(precision, "observed")
        for precision in ("iteration", "stage"):
            self.check(precision, "reconstructed")
        self.check("row", "reconstructed", replay_conformance=True)

    def test_reconstructed_finer_than_iteration_refused(self):
        for precision in ("row", "step_range"):
            with self.subTest(precision=precision):
                with self.assertRaises(ValueError):
                    self.check(precision, "reconstructed")

    def test_unknown_values_refused(self):
        with self.assertRaises(ValueError):
            self.check("iteration", "inferred")
        with self.assertRaises(ValueError):
            self.check("sample", "observed")

    def test_schema_field(self):
        from gatedtrain.ledger import SCHEMA

        self.assertIn("derivation", SCHEMA.names)
        self.assertEqual(str(SCHEMA.field("derivation").type), "string")


# ---------------------------------------------------------------------------
# R11A-19: orphaned and superseded step ranges per segment and shard
# ---------------------------------------------------------------------------

class TestOrphanRanges(unittest.TestCase):
    """The R10A-9 scenario (test_r10_impact): steps 11 and 12 logged after the last cut (step_10) read rows
    20-23, all in shard-00002: one orphaned range, attempt 0, steps 11-12. One attempt, one segment: nothing
    superseded. The R10A-9 `counts` stay as they are."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-r11-lineage-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = Scenario(cls.tmp)
        exp = cls.sc.stage_out / "logs" / "exp_001"
        for step in (11, 12):
            shutil.copyfile(exp / "train_data_step10.jsonl", exp / f"train_data_step{step}.jsonl")
        cls.proc = cls.sc.collect()[1]

    def impact(self, shard: str, *, as_json: bool = True):
        self.assertEqual(self.proc.returncode, EXIT_OK, describe(self.proc))
        args = ["impact", str(self.tmp / "export" / "train" / shard), "--run-dir", str(parse_run_dir(self.proc.stdout))]
        proc = run_cli(args + (["--json"] if as_json else []), cwd=self.tmp, timeout=60)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        return json.loads(proc.stdout) if as_json else proc.stdout

    def test_orphaned_range(self):
        out = self.impact("shard-00002.jsonl")
        run_id = parse_run_dir(self.proc.stdout).name
        self.assertEqual(out["ranges"], {
            "orphaned": [{"run_id": run_id, "attempt_id": "0", "shard": "shard-00002.jsonl", "first_step": 11,
                          "last_step": 12}],
            "superseded": []})
        self.assertEqual(out["counts"], {"orphaned": 2, "superseded": 0}, "R10A-9 counts unchanged")

    def test_covered_shard_has_no_ranges(self):
        out = self.impact("shard-00001.jsonl")
        self.assertEqual(out["ranges"], {"orphaned": [], "superseded": []})

    def test_text_lines(self):
        lines = self.impact("shard-00002.jsonl", as_json=False).splitlines()
        self.assertIn("orphaned_range: attempt 0 shard shard-00002.jsonl steps 11-12", lines)
        self.assertIn("superseded_ranges: none", lines)


# ---------------------------------------------------------------------------
# R11A-21: the real long run's run directory stays readable
# ---------------------------------------------------------------------------

class TestLegacyRunDirectory(unittest.TestCase):
    """Impact queries on a copy of the real long run's lineage and ledger (no `derivation` column, no
    orphaned.parquet). Expected answers are computed here from the ledger and checkpoint tables with pyarrow,
    by the 05 section 12.4 rule for one segment: every checkpoint with last_step_included >= the shard's first
    consuming step."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-r11-legacy-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.run_dir = cls.tmp / "run-20261111T110313Z-7adda9b9"
        shutil.copytree(LONGRUN, cls.run_dir)
        cls.shards = {r["shard"]: r["shard_sha256"]
                      for r in pq.read_table(cls.run_dir / "lineage" / "shards.parquet").to_pylist()}
        cls.ckpts = pq.read_table(cls.run_dir / "lineage" / "checkpoints.parquet").to_pylist()
        first: dict[str, int] = {}
        for r in pq.read_table(cls.run_dir / "ledger" / "consumption.parquet").to_pylist():
            if r["outcome"] == "consumed":
                s = r["data_ref"]["shard"]
                first[s] = min(first.get(s, r["step"]), r["step"])
        cls.first = first

    def impact(self, shard: str, *, as_json: bool = True):
        args = ["impact", self.shards[shard], "--run-dir", str(self.run_dir)]
        proc = run_cli(args + (["--json"] if as_json else []), cwd=self.tmp, timeout=60)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        return json.loads(proc.stdout) if as_json else proc.stdout

    def expected(self, shard: str) -> set[str]:
        if shard not in self.first:
            return set()
        return {c["checkpoint_hash"] for c in self.ckpts if c["last_step_included"] >= self.first[shard]}

    def test_fixture_guard(self):
        self.assertEqual(self.first["shard-00003.jsonl"], 47)
        self.assertEqual(len(self.expected("shard-00003.jsonl")), 5)
        self.assertNotIn("shard-00012.jsonl", self.first)
        names = pq.read_table(self.run_dir / "ledger" / "consumption.parquet").schema.names
        self.assertNotIn("derivation", names, "the fixture is a pre-revision-11 ledger")

    def test_consumed_shards(self):
        for shard in ("shard-00000.jsonl", "shard-00003.jsonl", "shard-00009.jsonl"):
            with self.subTest(shard=shard):
                out = self.impact(shard)
                self.assertEqual({r["checkpoint_hash"] for r in out["results"]}, self.expected(shard))
                self.assertEqual({r["precision"] for r in out["results"]}, {"iteration"})
                self.assertEqual({r["answer"] for r in out["results"]}, {"possibly consumed"})
                self.assertEqual({r.get("derivation", "key absent") for r in out["results"]}, {None},
                                 "records written before revision 11 have no derivation")

    def test_unused_shard(self):
        out = self.impact("shard-00012.jsonl")
        self.assertEqual(out["results"], [])
        self.assertEqual(out["counts"], {"orphaned": 0, "superseded": 0})
        self.assertEqual(out["ranges"], {"orphaned": [], "superseded": []})


if __name__ == "__main__":
    unittest.main()
