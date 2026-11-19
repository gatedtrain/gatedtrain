"""H-1 `gatedtrain export-gsm8k` and H-2 `gatedtrain verify-export` (TIER-B-SPEC sections 1 and 3;
interim rules G-14 and G-17; tests/README.md section 11.1-11.2).

The export runs on the tiny local fixture tests/fixtures/tierb/gsm8k-source (no network).
Expected bytes come from the independent reference in _tierb.expected_export.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from _support import (EXIT_OK, EXIT_REFUSED, SKEL_ROOT, canonical_bytes, describe, file_sha256, manifest_entries,
                      manifest_hash, parse_errors, run_cli)
from _tierb import NEL_PROMPT_FOR_NEMO_RL, REVISION, ROWS_PER_SHARD, expected_export, run_export, upstream_prompt

EXPORT_KEYS = {"schema", "schema_version", "tags", "dataset", "config", "revision", "rows", "rows_per_shard",
               "train_manifest_hash", "prompt_set_hash", "concat_sha256", "prompt_file_sha256"}


class ExportCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-export-"))
        self.out = self.tmp / "export"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def export(self):
        proc = run_export(self.out, cwd=SKEL_ROOT)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        return proc


class TestExportGsm8k(ExportCase):
    def test_reference_reads_the_fixture(self):
        """Guards the independent reference (passes without gatedtrain)."""
        ref = expected_export()
        self.assertEqual((ref["n_train"], ref["n_test"]), (40, 30))
        self.assertEqual(len(ref["shards"]), 4)
        self.assertRegex(ref["concat"], rb'"answer":"\d+,\d{3}"', "gsm8k.py:22-25 keeps the thousands comma")
        self.assertNotRegex(ref["concat"], rb'"answer":" ', "the answer is stripped")
        up = upstream_prompt()
        if up is not None:
            self.assertEqual(up.replace("{question}", "{}"), NEL_PROMPT_FOR_NEMO_RL)

    def test_layout_and_bytes(self):
        self.export()
        ref = expected_export()
        shards = sorted(p.name for p in (self.out / "train").iterdir())
        self.assertEqual(shards, sorted(ref["shards"]), "only shard files in train/")
        for name, data in ref["shards"].items():
            self.assertEqual((self.out / "train" / name).read_bytes(), data, name)
        self.assertEqual((self.out / "train.jsonl").read_bytes(), ref["concat"], "concatenation in file-name order")
        self.assertEqual((self.out / "heldout" / "prompts.jsonl").read_bytes(), ref["heldout"])
        prompt = (self.out / "nel_gsm8k_prompt.txt").read_bytes()
        self.assertEqual(prompt, NEL_PROMPT_FOR_NEMO_RL.encode("utf-8"), "G-15 prompt, no trailing newline")

    def test_hashes_file(self):
        """G-17: the export records the dataset revision and the concatenation's SHA-256."""
        self.export()
        raw = (self.out / "export.json").read_bytes()
        doc = json.loads(raw)
        self.assertEqual(raw, canonical_bytes(doc), "canonical JSON")
        self.assertEqual(set(doc), EXPORT_KEYS)
        self.assertEqual(doc["dataset"], "openai/gsm8k")
        self.assertEqual(doc["config"], "main")
        self.assertEqual(doc["revision"], REVISION)
        self.assertEqual(doc["rows"], {"train": 40, "test": 30})
        self.assertEqual(doc["rows_per_shard"], ROWS_PER_SHARD)
        self.assertEqual(doc["concat_sha256"], file_sha256(self.out / "train.jsonl"))
        self.assertEqual(doc["train_manifest_hash"], manifest_hash(self.out / "train"))
        self.assertEqual(doc["prompt_set_hash"], manifest_hash(self.out / "heldout"))
        self.assertEqual(doc["prompt_file_sha256"], file_sha256(self.out / "nel_gsm8k_prompt.txt"))
        self.assertIn("prototype", doc["tags"])
        self.assertIn("unpromotable", doc["tags"])

    def test_manifests(self):
        """A content-addressed manifest for the shards and the held-out set (H7)."""
        self.export()
        for name, sub in (("training_shards", "train"), ("prompt_set", "heldout")):
            with self.subTest(manifest=name):
                raw = (self.out / "manifests" / f"{name}.json").read_bytes()
                doc = json.loads(raw)
                self.assertEqual(raw, canonical_bytes(doc))
                self.assertEqual(doc["artifact"], name)
                self.assertEqual(doc["manifest_hash_kind"], "p2_filelist_sha256")
                self.assertEqual(doc["entries"], manifest_entries(self.out / sub))
                self.assertEqual(doc["manifest_hash"], manifest_hash(self.out / sub))

    def test_never_overwrites(self):
        self.export()
        before = manifest_hash(self.out)
        proc = run_export(self.out, cwd=SKEL_ROOT)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertEqual(manifest_hash(self.out), before)

    def test_source_needs_revision(self):
        """G-17: a local copy carries no revision of its own, so --revision is required with --source."""
        proc = run_export(self.out, cwd=SKEL_ROOT, revision=None)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue(any(e["rule"] == "EXPORT-REVISION" for e in parse_errors(proc.stderr)), describe(proc))
        self.assertFalse(self.out.exists())


class TestVerifyExport(ExportCase):
    def verify(self):
        return run_cli(["verify-export", str(self.out)], cwd=SKEL_ROOT, timeout=60)

    def assert_refused(self, rule: str):
        proc = self.verify()
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        rules = {e["rule"] for e in parse_errors(proc.stderr)}
        self.assertIn(rule, rules, describe(proc))

    def test_clean_export_verifies(self):
        self.export()
        proc = self.verify()
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_tampered_shard(self):
        self.export()
        shard = self.out / "train" / "shard-00001.jsonl"
        shard.write_bytes(shard.read_bytes().replace(b'"answer":"', b'"answer":"9', 1))
        self.assert_refused("EXPORT-HASH")

    def test_tampered_concatenation(self):
        self.export()
        concat = self.out / "train.jsonl"
        concat.write_bytes(concat.read_bytes() + b'{"answer":"1","question":"extra"}\n')
        self.assert_refused("EXPORT-HASH")

    def test_reordered_shards(self):
        """The concatenation is checked in file-name order: swapping two shard names breaks it."""
        self.export()
        a, b = self.out / "train" / "shard-00000.jsonl", self.out / "train" / "shard-00002.jsonl"
        tmp = self.out / "swap.tmp"
        a.rename(tmp)
        b.rename(a)
        tmp.rename(b)
        self.assert_refused("EXPORT-HASH")

    def test_extra_file_in_shard_directory(self):
        self.export()
        (self.out / "train" / "shard-00099.jsonl").write_bytes(b'{"answer":"1","question":"stray"}\n')
        self.assert_refused("EXPORT-HASH")

    def test_concatenation_inside_shard_directory(self):
        """L-5 / G-14: train.jsonl must not sit inside the shard directory (it would become a shard)."""
        self.export()
        shutil.copy2(self.out / "train.jsonl", self.out / "train" / "train.jsonl")
        self.assert_refused("EXPORT-LAYOUT")



class _Response:
    def __init__(self, data: bytes):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self.data


class TestDownloadHash(unittest.TestCase):
    """The pinned revision's parquet files must have their recorded SHA-256 (EXPORT-SHA256); urlopen is replaced."""

    def test_wrong_bytes_refused(self):
        from unittest import mock

        from gatedtrain import tierb
        from gatedtrain.errors import Refused

        with mock.patch("urllib.request.urlopen", return_value=_Response(b"not the pinned parquet")):
            with self.assertRaises(Refused) as cm:
                tierb._download(None)
        self.assertEqual([p.rule for p in cm.exception.problems], ["EXPORT-SHA256"])

    def test_default_revision_is_pinned(self):
        from gatedtrain import tierb

        self.assertEqual(set(tierb.DATASET_PARQUET_SHA256[tierb.DATASET_REVISION]), {"train", "test"})

if __name__ == "__main__":
    unittest.main()
