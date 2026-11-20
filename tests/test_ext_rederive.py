"""A merged arm is re-derived from its sources (EXT-52; EXT-47), in process.

`test_ext_review2.TestRederivedArm` runs the rule through `ext-collect` in a subprocess; these tests call
`arm.provenance_problems` on arms written by the in-process writers (`merge.merge_eval`, `from_nel.from_nel`), so the
re-derivation itself (`arm._rederived_problems`) is pinned line by line: an untouched arm passes, an edited score or a
reordered line is refused, a source whose lines no longer read (its recorded SHA-256 updated with it) cannot re-derive
the arm, and without registered item rows the `ext_from_nel` arm is not re-derived here (EXT-5 refuses it elsewhere).
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from _ext import EVALUATOR_ID, SEEDS, ExtScenario, baseline_score, build_arm
from _support import file_sha256
from test_ext_from_nel import nel_runs, scorer_candidate

from gatedtrain.ext import arm, registry
from gatedtrain.ext.from_nel import from_nel
from gatedtrain.ext.merge import merge_eval

EDITED = "the arm was edited after it was written (EXT-52)"


def rewrite(path: Path, fn) -> None:
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    path.write_text("".join(json.dumps(r) + "\n" for r in fn(recs)), encoding="utf-8")


class RederiveCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-rederive-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = ExtScenario(cls.tmp / "sc")
        cls.row = registry.EVALUATOR_ROWS[EVALUATOR_ID]
        cls.item_rows = [json.loads(l) for l in (cls.sc.items_dir / "items.jsonl").read_text(encoding="utf-8")
                         .splitlines() if l.strip()]
        cls.n = 0

    def out(self) -> Path:
        type(self).n += 1
        return self.tmp / f"arm-{type(self).n}"

    def problems(self, d: Path, *, item_rows=..., row=...) -> list[str]:
        meta = json.loads((d / "eval_meta.json").read_text(encoding="utf-8"))
        return arm.provenance_problems(d, meta, nel_benchmark=self.row.get("nel_benchmark"),
                                       row=self.row if row is ... else row,
                                       item_rows=self.item_rows if item_rows is ... else item_rows)


class TestMergedArm(RederiveCase):
    """`gatedtrain.ext_merge_eval`: the arm is `merge_records` of its source arms."""

    def merged(self) -> Path:
        out = self.out()
        srcs = [build_arm(out.parent / f"{out.name}-s{s}", self.sc.items, baseline_score, model=self.sc.base_model,
                          item_set=self.sc.item_set_hash, seeds=[s]) for s in SEEDS]
        merge_eval(self.sc.paths["registration"], out, srcs)
        return out

    def test_untouched_passes(self):
        self.assertEqual(self.problems(self.merged()), [])

    def test_edited_score_refused(self):
        out = self.merged()
        rewrite(out / "items.jsonl", lambda recs: [dict(recs[0], score=1.0 - recs[0]["score"])] + recs[1:])
        self.assertTrue(any(EDITED in m for m in self.problems(out)), self.problems(out))

    def test_reordered_lines_refused(self):
        out = self.merged()
        rewrite(out / "items.jsonl", lambda recs: list(reversed(recs)))
        self.assertTrue(any(EDITED in m for m in self.problems(out)))

    def test_formatting_counts(self):
        """Byte equality: the same records written with other JSON spacing are not the file ext-merge-eval writes."""
        out = self.merged()
        recs = [json.loads(l) for l in (out / "items.jsonl").read_text(encoding="utf-8").splitlines()]
        (out / "items.jsonl").write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in recs),
                                         encoding="utf-8")
        self.assertTrue(any(EDITED in m for m in self.problems(out)))

    def test_source_with_bad_lines(self):
        """A source arm whose recorded SHA-256 was updated after a bad line was added cannot re-derive the arm."""
        out = self.merged()
        doc = json.loads((out / "merge.json").read_text(encoding="utf-8"))
        src = Path(doc["sources"][1]["path"])
        with (src / "items.jsonl").open("a", encoding="utf-8") as f:
            f.write("not json\n")
        doc["sources"][1]["items_sha256"] = file_sha256(src / "items.jsonl")
        (out / "merge.json").write_text(json.dumps(doc), encoding="utf-8")
        msgs = self.problems(out)
        self.assertTrue(any(m.startswith("merge.json sources[1] has bad lines") and "cannot be re-derived" in m
                            for m in msgs), msgs)

    def test_without_row_not_rederived(self):
        """Without the evaluator row (no line rules) the provenance checks run, the re-derivation does not."""
        out = self.merged()
        rewrite(out / "items.jsonl", lambda recs: list(reversed(recs)))
        self.assertEqual(self.problems(out, row=None), [])


class TestFromNelArm(RederiveCase):
    """`gatedtrain.ext_from_nel`: the arm is `from_nel.convert_records` of the runs' results.jsonl."""

    def converted(self) -> Path:
        out = self.out()
        runs = nel_runs(out.parent / f"{out.name}-runs", self.sc.rows, score_fn=scorer_candidate, model="cand")
        from_nel(registration=self.sc.paths["registration"], adapter="synthetic", evaluator=EVALUATOR_ID,
                 items=self.sc.items_dir, checkpoint=self.sc.final_cut, out=out, runs=runs)
        return out

    def test_untouched_passes(self):
        self.assertEqual(self.problems(self.converted()), [])

    def test_edited_score_refused(self):
        out = self.converted()
        rewrite(out / "items.jsonl", lambda recs: [dict(r, score=1.0) if r["score"] is not None else r for r in recs])
        self.assertTrue(any(EDITED in m for m in self.problems(out)))

    def test_dropped_line_refused(self):
        out = self.converted()
        rewrite(out / "items.jsonl", lambda recs: recs[:-1])
        self.assertTrue(any(EDITED in m for m in self.problems(out)))

    def test_without_item_rows_not_rederived(self):
        out = self.converted()
        rewrite(out / "items.jsonl", lambda recs: list(reversed(recs)))
        self.assertEqual(self.problems(out, item_rows=[]), [])

    def test_run_with_bad_lines(self):
        out = self.converted()
        doc = json.loads((out / "merge.json").read_text(encoding="utf-8"))
        results = Path(doc["sources"][0]["run_dir"]) / self.row["nel_benchmark"] / "results.jsonl"
        with results.open("a", encoding="utf-8") as f:
            f.write("{broken\n")
        doc["sources"][0]["results_sha256"] = file_sha256(results)
        (out / "merge.json").write_text(json.dumps(doc), encoding="utf-8")
        msgs = self.problems(out)
        self.assertTrue(any("results.jsonl has bad lines" in m and "cannot be re-derived" in m for m in msgs), msgs)


if __name__ == "__main__":
    unittest.main()
