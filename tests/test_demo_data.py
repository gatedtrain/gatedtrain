"""Demo task data fixture (MVP-SPEC section 3, "Task").

These tests check the fixture data under demo/data and need no
implementation: held-out prompts disjoint from all training data, at least
200 prompts, shards split into files, and a reproducible generator.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from _support import DEMO_DIR, VENV_PYTHON, manifest_hash, read_jsonl

DATA = DEMO_DIR / "data"


def prompts_of(directory: Path) -> set[str]:
    out = set()
    for f in sorted(directory.glob("*.jsonl")):
        out |= {r["prompt"] for r in read_jsonl(f)}
    return out


class TestDemoData(unittest.TestCase):
    def test_heldout_size(self):
        rows = read_jsonl(DATA / "heldout" / "prompts.jsonl")
        self.assertGreaterEqual(len(rows), 200, "MVP-SPEC section 3: at least 200 prompts")
        self.assertEqual(len({r["prompt"] for r in rows}), len(rows), "no duplicate prompts")

    def test_small_heldout_is_prefix(self):
        full = read_jsonl(DATA / "heldout" / "prompts.jsonl")
        small = read_jsonl(DATA / "heldout-small" / "prompts.jsonl")
        self.assertEqual(small, full[: len(small)])
        self.assertLess(len(small), 172, "below the planned n of the demo registration (see test_stats_plan)")

    def test_heldout_disjoint_from_all_training_data(self):
        held = prompts_of(DATA / "heldout")
        for sub in ("train", "train-shuffled", "unused", "warmup"):
            with self.subTest(data=sub):
                self.assertFalse(held & prompts_of(DATA / sub), f"held-out prompts overlap {sub}")

    def test_training_data_split_into_shards(self):
        shards = sorted((DATA / "train").glob("shard-*.jsonl"))
        self.assertGreaterEqual(len(shards), 2)
        self.assertFalse(prompts_of(DATA / "unused") & prompts_of(DATA / "train"))

    def test_rows_are_exact_match_addition(self):
        for f in sorted(DATA.rglob("*.jsonl")):
            if f.parent.name == "train-shuffled":
                continue
            for r in read_jsonl(f):
                a, b = r["prompt"].rstrip("=").split("+")
                self.assertEqual(r["answer"], str(int(a) + int(b)), f"{f.name}: {r}")

    def test_shuffled_labels_are_wrong_for_most_rows(self):
        for f in sorted((DATA / "train-shuffled").glob("*.jsonl")):
            rows = read_jsonl(f)
            wrong = 0
            for r in rows:
                a, b = r["prompt"].rstrip("=").split("+")
                wrong += r["answer"] != str(int(a) + int(b))
            self.assertGreater(wrong / len(rows), 0.9, f.name)

    def test_generator_is_reproducible(self):
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-data-"))
        try:
            shutil.copy2(DEMO_DIR / "make_data.py", tmp / "make_data.py")
            subprocess.run([str(VENV_PYTHON), "-I", str(tmp / "make_data.py")], check=True, cwd=str(tmp))
            for sub in ("train", "train-shuffled", "unused", "warmup", "heldout", "heldout-small"):
                with self.subTest(data=sub):
                    self.assertEqual(manifest_hash(tmp / "data" / sub), manifest_hash(DATA / sub))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
