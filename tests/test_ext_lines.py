"""Line splitting of JSONL and line-oriented files (tests/README.md section 11.18, EXT-90).

`str.splitlines()` also breaks on U+2028, U+2029, U+0085, \\x0b, \\x0c and \\x1c to \\x1e. JSON allows raw U+2028 and
U+2029 inside strings (RFC 8259 escapes only U+0000 to U+001F, `"` and `\\`), and `export-gsm8k` writes
`ensure_ascii=False`, so the real GSM8K training export holds a row with a raw U+2028
(`<tierb-root>/tierb/export/train/shard-00004.jsonl` line 382, read 2026-11-11). Every reader splits on `\\n` (with
`\\r\\n` and `\\r` as before) through `canonical.text_lines`, as `ledger._count_lines` (file iteration) and the
trainer's JSON loader do, so row numbers and row counts agree.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from _ext import gsm8k_rows
from _support import EXIT_REFUSED, describe, parse_errors
from test_ext_hardening import CollectCase

# U+2028 LINE SEPARATOR, U+2029 PARAGRAPH SEPARATOR, U+0085 NEXT LINE: legal raw inside a JSON string.
SEPARATORS = ("\u2028", "\u2029", "\u0085")


def dumps(obj) -> str:
    """One JSONL line as export-gsm8k writes it (raw non-ASCII, tierb.py `_row_bytes`)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def separator_rows(field: str = "question", **extra) -> list[dict]:
    return [{field: f"before{sep}after {i}", **extra} for i, sep in enumerate(SEPARATORS)]


class TmpCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-lines-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def write(self, rel: str, lines: list[str]) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# The shared helper
# ---------------------------------------------------------------------------

class TestTextLines(unittest.TestCase):

    def test_only_line_feeds_split(self):
        from gatedtrain.canonical import text_lines

        for sep in SEPARATORS + ("\x0b", "\x0c", "\x1c", "\x1d", "\x1e"):
            with self.subTest(sep=hex(ord(sep))):
                self.assertEqual(text_lines(f"a{sep}b\nc\n"), [f"a{sep}b", "c"])

    def test_line_ends_as_splitlines(self):
        """`\\n`, `\\r\\n` and `\\r` end a line and a final line end adds no empty line, as before."""
        from gatedtrain.canonical import text_lines

        for text in ("", "\n", "a", "a\n", "a\nb", "a\r\nb\r\n", "a\rb\r", "a\n\nb\n", "a\r\n\r\nb", "\n\n", "a\r"):
            with self.subTest(text=text):
                self.assertEqual(text_lines(text), text.splitlines())


# ---------------------------------------------------------------------------
# JSONL readers: rows with a separator inside a string parse, line numbers stay the file's
# ---------------------------------------------------------------------------

class TestJsonlReaders(TmpCase):

    def test_item_set(self):
        from gatedtrain.ext.collect import read_item_set

        rows = [{"item": str(i), "question": r["question"], "answer": "1"} for i, r in enumerate(separator_rows())]
        self.write("items/items.jsonl", [dumps(r) for r in rows])
        self.assertEqual(read_item_set(self.root / "items", ("question", "answer")), (rows, []))
        self.write("items/items.jsonl", [dumps(r) for r in rows] + ["not json"])
        self.assertEqual(read_item_set(self.root / "items", ("question", "answer"))[1], ["line 4 does not parse"])

    def test_exact_overlap(self):
        from gatedtrain.ext.collect import exact_overlap

        train = separator_rows(answer="1")
        self.write("train/shard-00000.jsonl", [dumps(r) for r in train])
        items = [{"question": train[0]["question"]}, {"question": train[2]["question"]}, {"question": "other"}]
        self.assertEqual(exact_overlap(items, self.root / "train", "question"), 2)
        self.write("train/shard-00001.jsonl", [dumps(train[1]), "{"])
        with self.assertRaisesRegex(ValueError, r"^shard-00001\.jsonl:2 does not parse$"):
            exact_overlap(items, self.root / "train", "question")

    def test_samples_log(self):
        from gatedtrain.testing.synthetic import read_samples

        lines = [dumps({"step": 1, "row": i, "completion": f"x{sep}y"}) for i, sep in enumerate(SEPARATORS)]
        path = self.write("samples.jsonl", lines)
        self.assertEqual(read_samples(path), ({1: {0, 1, 2}}, None))
        path = self.write("samples.jsonl", lines + ["{"])
        self.assertEqual(read_samples(path), (None, "samples.jsonl line 4 does not parse"))

    def test_arm_lines(self):
        from gatedtrain.ext.arm import read_lines

        self.write("arm/items.jsonl", [dumps({"item": "a\u2028b"}), "not json"])
        _, bad = read_lines(self.root / "arm", seeds=[1], score_range=[0, 1], error_classes=[])
        self.assertIn("line 2 does not parse", bad)
        self.assertFalse([b for b in bad if b.startswith(("line 1 does not parse", "line 3"))], bad)

    def test_results_lines(self):
        from gatedtrain.runner import results_lines

        recs = [{"problem_idx": i, "response": f"r{sep}s"} for i, sep in enumerate(SEPARATORS)]
        path = self.write("results.jsonl", [dumps(r) for r in recs] + ["{"])
        self.assertEqual(results_lines(path), [(1, recs[0]), (2, recs[1]), (3, recs[2]), (4, None)])

    def test_item_ids(self):
        from gatedtrain.runner import item_ids

        self.write("prompts/prompts.jsonl", [dumps(r) for r in separator_rows("prompt")])
        self.assertEqual(item_ids(self.root / "prompts"), [0, 1, 2])

    def test_tierb_read_jsonl(self):
        from gatedtrain.tierb import _read_jsonl

        rows = separator_rows(answer="#### 1")
        self.assertEqual(_read_jsonl(self.write("train.jsonl", [dumps(r) for r in rows])), rows)

    def test_row_counts_agree(self):
        """The ledger's shard table (file iteration), the step rows' total, the contamination reader and the
        item-set readers count the same rows for a file holding every separator."""
        from gatedtrain.canonical import text_lines
        from gatedtrain.ext.collect import read_item_set
        from gatedtrain.ledger import shard_table
        from gatedtrain.runner import item_ids

        rows = [{"item": str(i), "question": r["question"], "answer": "1"}
                for i, r in enumerate(separator_rows() + separator_rows())]
        self.write("train/shard-00000.jsonl", [dumps(r) for r in rows])
        self.write("data/items.jsonl", [dumps(r) for r in rows])
        self.write("data/prompts.jsonl", [dumps(r) for r in rows])
        text = (self.root / "train/shard-00000.jsonl").read_text(encoding="utf-8")
        self.assertEqual(shard_table(self.root / "train"), [("shard-00000.jsonl", len(rows))])
        self.assertEqual(len([l for l in text_lines(text) if l.strip()]), len(rows))
        self.assertEqual(len(read_item_set(self.root / "data", ("question",))[0]), len(rows))
        self.assertEqual(item_ids(self.root / "data"), list(range(len(rows))))


# ---------------------------------------------------------------------------
# Line-oriented text files: a separator inside a line does not start a new line
# ---------------------------------------------------------------------------

class TestTextReaders(TmpCase):

    def test_gpu_guard_log_first_start(self):
        """A note line holding a START after a U+2028 is one line; the label's first START is line 2's."""
        from gatedtrain.ext.launch import first_start

        log = self.write("gpu-guard.log", ["2026-11-11T08:00:00Z NOTE x\u20282026-11-11T07:00:00Z START lbl",
                                           "2026-11-11T09:00:00Z START lbl", "2026-11-11T09:10:00Z END lbl rc=0"])
        self.assertEqual(first_start(log, "lbl"), 1794387600.0)

    def test_gpu_guard_log_seconds(self):
        from gatedtrain.ext.launch import gpu_seconds

        log = self.write("gpu-guard.log", ["2026-11-11T09:00:00Z START lbl",
                                           "2026-11-11T09:05:00Z NOTE y\u20282026-11-11T09:05:00Z END lbl rc=0",
                                           "2026-11-11T09:10:00Z END lbl rc=0"])
        self.assertEqual(gpu_seconds(log, "lbl"), 600)

    def test_gpu_step_script(self):
        """bash ends a line at `\\n` only: a comment holding `\\u2028WORK=/b` assigns nothing."""
        from gatedtrain.ext.launch import gpu_step_assignments, gpu_step_log_problem

        work = self.root / "work"
        text = [f"WORK={work}", "# was\u2028WORK=/b", 'LOG="$WORK/gpu-guard.log"']
        self.write("work/bin/gpu-step.sh", text)
        self.assertEqual(gpu_step_assignments("".join(l + "\n" for l in text)), {"WORK": [1], "LOG": [3]})
        self.assertIsNone(gpu_step_log_problem(work))

    def test_gpu_log_md(self):
        from gatedtrain.ext.launch import gpu_used_h

        md = self.write("gpu-log.md", ["| Date | GPU h | Running total |", "|---|---|---|",
                                       "| 2026-11-10 | 1.0 | 1.0 |", "| 2026-11-11 | 2.0 a\u2028b | 3.0 |"])
        self.assertEqual(gpu_used_h(md), 3.0)

    def test_env_freeze(self):
        from gatedtrain.testing.synthetic import SyntheticAdapter

        self.write("out/env-freeze.txt", ["synthetic-trainer==1.0.0", "# was\u2028synthetic-trainer==0.0.1"])
        got = SyntheticAdapter().framework_identity(self.root / "out", {"framework": {"name": "synthetic-trainer",
                                                                                    "version": "1.0.0"}})
        self.assertEqual(got["version"], "1.0.0")

    def test_nel_snapshot_header(self):
        from gatedtrain.ext.from_nel import _framework_version

        text = "# nemo-evaluator snapshot\u2028key: v\n# nemo-evaluator version: 1.2.3\nkey: 1\n"
        self.assertEqual(_framework_version(text), "1.2.3")


# ---------------------------------------------------------------------------
# ext-collect: training rows with a separator inside a string are read, not refused
# ---------------------------------------------------------------------------

def separator_shard(path: Path, extra: list[dict] = ()) -> None:
    rows = separator_rows(answer="1") + list(extra)
    path.write_text("".join(dumps(r) + "\n" for r in rows), encoding="utf-8")


class TestCollectSeparatorRows(CollectCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        separator_shard(cls.sc.train / "shard-00009.jsonl")
        cls.sc.retrain()                                   # the run read the new rows (EXT-65)

    def test_passes_without_overlap(self):
        rep = self.passes()
        self.assertEqual(rep["status"], "PASS")
        self.assertEqual(rep["contamination"], {"exact_overlap": 0, "allowed": False})


class TestCollectSeparatorOverlap(CollectCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        row = gsm8k_rows("test")[4]
        separator_shard(cls.sc.train / "shard-00009.jsonl", [{"answer": row["answer"], "question": row["question"]}])
        cls.sc.retrain()

    def test_overlap_counted_and_refused(self):
        proc = self.sc.collect(self.runs())
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = [e for e in parse_errors(proc.stderr) if e["rule"] == "EXT-CONTAMINATION"]
        self.assertEqual([e["field"] for e in errs], ["registration.contamination.allow_exact_overlap"], errs)
        self.assertTrue(errs[0]["message"].startswith("1 item(s) "), errs)


if __name__ == "__main__":
    unittest.main()
