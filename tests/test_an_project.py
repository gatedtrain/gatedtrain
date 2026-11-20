"""AN-2: the AI-native project store (tests/README.md section 12).

One goal is one project directory: `project.yaml`, `stages/<stage>/<name>.json`, `attempts.jsonl`. Created once;
every artifact is canonical JSON written by temp file plus rename and read back against its recorded sha256.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from gatedtrain.ainative.project import ArtifactRef, Project, create_project
from gatedtrain.canonical import canonical_json, file_sha256, sha256_bytes
from gatedtrain.errors import Refused


class ProjectStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.data = self.tmp / "data.jsonl"
        self.data.write_text('{"q": "1+1", "a": "2"}\n')
        self.root = self.tmp / "proj"

    def make(self) -> Project:
        return create_project(self.root, goal="improve math", data=self.data, pipeline="default",
                              budget={"gpu_hours": 2, "llm_usd": 5})

    def refusal(self, fn) -> str:
        with self.assertRaises(Refused) as cm:
            fn()
        self.assertEqual([p.rule for p in cm.exception.problems], ["AN-2"] * len(cm.exception.problems))
        return str(cm.exception)

    def test_create_then_load_round_trips_every_key(self) -> None:
        self.make()
        doc = yaml.safe_load((self.root / "project.yaml").read_text())
        self.assertEqual(doc["schema"], "gatedtrain.ainative.project")
        self.assertEqual(doc["schema_version"], 1)
        self.assertEqual(doc["goal"], "improve math")
        self.assertEqual(doc["data"], {"path": str(self.data), "sha256": file_sha256(self.data)})
        self.assertEqual(doc["pipeline"], "default")
        self.assertEqual(doc["budget"], {"gpu_hours": 2, "llm_usd": 5})
        self.assertRegex(doc["created_at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z$")
        p = Project.load(self.root)
        self.assertEqual((p.goal, p.pipeline, p.budget), ("improve math", "default",
                                                          {"gpu_hours": 2, "llm_usd": 5}))
        self.assertEqual(p.data, {"path": str(self.data), "sha256": file_sha256(self.data)})
        self.assertEqual(p.created_at, doc["created_at"])
        self.assertEqual(p.root, self.root)

    def test_create_on_non_empty_root_refuses(self) -> None:
        self.root.mkdir()
        (self.root / "x").write_text("x")
        msg = self.refusal(self.make)
        self.assertIn("error[AN-2]", msg)
        self.make_empty_ok()

    def make_empty_ok(self) -> None:
        empty = self.tmp / "empty"
        empty.mkdir()
        create_project(empty, goal="g", data=self.data, pipeline="p", budget={"gpu_hours": 1, "llm_usd": 1})

    def test_create_twice_refuses(self) -> None:
        self.make()
        self.refusal(self.make)

    def test_load_refuses_missing_or_wrong_schema(self) -> None:
        self.refusal(lambda: Project.load(self.root))
        self.make()
        path = self.root / "project.yaml"
        doc = yaml.safe_load(path.read_text())
        doc["schema"] = "other"
        path.write_text(yaml.safe_dump(doc))
        self.refusal(lambda: Project.load(self.root))

    def test_write_artifact_hashes_canonical_json(self) -> None:
        p = self.make()
        obj = {"b": 1, "a": [1, 2]}
        ref = p.write_artifact("intake_seal", "split", obj, "gatedtrain.ainative.split")
        self.assertEqual(ref, ArtifactRef("intake_seal", "split", "gatedtrain.ainative.split",
                                          sha256_bytes(canonical_json(obj)), "stages/intake_seal/split.json"))
        self.assertEqual((self.root / ref.path).read_bytes(), canonical_json(obj))
        self.assertEqual(p.read_artifact(ref), obj)
        self.assertEqual(p.read_artifact(ref), json.loads((self.root / ref.path).read_text()))
        lst = p.write_artifact("gate", "rows", [1, 2], "gatedtrain.ainative.rows")
        self.assertEqual(p.read_artifact(lst), [1, 2])

    def test_read_refuses_changed_file(self) -> None:
        p = self.make()
        ref = p.write_artifact("gate", "verdict", {"ok": True}, "gatedtrain.ainative.verdict")
        (self.root / ref.path).write_bytes(canonical_json({"ok": False}))
        msg = self.refusal(lambda: p.read_artifact(ref))
        self.assertIn("sha256", msg)

    def test_read_refuses_missing_file(self) -> None:
        p = self.make()
        ref = p.write_artifact("gate", "verdict", {"ok": True}, "gatedtrain.ainative.verdict")
        (self.root / ref.path).unlink()
        self.refusal(lambda: p.read_artifact(ref))

    def test_write_is_temp_plus_rename(self) -> None:
        p = self.make()
        with mock.patch("os.replace", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                p.write_artifact("gate", "verdict", {"ok": True}, "gatedtrain.ainative.verdict")
        stage = self.root / "stages" / "gate"
        self.assertFalse((stage / "verdict.json").exists())
        self.assertEqual([f for f in os.listdir(stage)], [])   # the temp file is cleaned up

    def test_rewrite_replaces_whole_file(self) -> None:
        p = self.make()
        p.write_artifact("gate", "v", {"n": 1}, "gatedtrain.ainative.v")
        ref = p.write_artifact("gate", "v", {"n": 2}, "gatedtrain.ainative.v")
        self.assertEqual(p.read_artifact(ref), {"n": 2})

    def test_unsafe_names_refused(self) -> None:
        p = self.make()
        for bad in ("", "..", ".", "a/b", "a\\b", "a\0b", "../x"):
            with self.subTest(bad=bad):
                self.refusal(lambda: p.write_artifact(bad, "n", {}, "s"))
                self.refusal(lambda: p.write_artifact("st", bad, {}, "s"))
        self.assertEqual([x.name for x in self.root.iterdir()].count("x"), 0)

    def test_read_refuses_forged_ref_path(self) -> None:
        p = self.make()
        (self.tmp / "evil.json").write_bytes(canonical_json({"a": 1}))
        ref = ArtifactRef("gate", "v", "s", sha256_bytes(canonical_json({"a": 1})), "../evil.json")
        self.refusal(lambda: p.read_artifact(ref))

    def test_symlinked_stage_dirs_refused(self) -> None:
        p = self.make()
        outside = self.tmp / "outside"
        outside.mkdir()
        (outside / "v.json").write_bytes(canonical_json({"a": 1}))
        ref = ArtifactRef("gate", "v", "s", sha256_bytes(canonical_json({"a": 1})), "stages/gate/v.json")
        # stages/<stage> is a link
        (self.root / "stages").mkdir()
        (self.root / "stages" / "gate").symlink_to(outside, target_is_directory=True)
        self.refusal(lambda: p.write_artifact("gate", "new", {"a": 1}, "s"))
        self.refusal(lambda: p.read_artifact(ref))
        self.assertEqual(sorted(os.listdir(outside)), ["v.json"])
        # stages itself is a link
        (self.root / "stages" / "gate").unlink()
        (self.root / "stages").rmdir()
        (self.root / "stages").symlink_to(outside, target_is_directory=True)
        (outside / "gate").mkdir()
        (outside / "gate" / "v.json").write_bytes(canonical_json({"a": 1}))
        self.refusal(lambda: p.write_artifact("gate", "new", {"a": 1}, "s"))
        self.refusal(lambda: p.read_artifact(ref))
        self.assertEqual(os.listdir(outside / "gate"), ["v.json"])

    def test_symlinked_attempt_log_refused(self) -> None:
        p = self.make()
        outside = self.tmp / "outside.jsonl"
        outside.write_text("keep\n")
        (self.root / "attempts.jsonl").symlink_to(outside)
        self.refusal(lambda: p.append_attempt({"stage": "gate"}))
        self.refusal(lambda: p.read_attempts())
        self.assertEqual(outside.read_text(), "keep\n")

    def test_symlinked_project_file_refused(self) -> None:
        self.make()
        outside = self.tmp / "outside.yaml"
        outside.write_bytes((self.root / "project.yaml").read_bytes())
        before = outside.read_bytes()
        (self.root / "project.yaml").unlink()
        (self.root / "project.yaml").symlink_to(outside)
        self.refusal(lambda: Project.load(self.root))
        self.assertEqual(outside.read_bytes(), before)

    def test_append_attempt_is_jsonl(self) -> None:
        p = self.make()
        p.append_attempt({"stage": "gate", "n": 1})
        p.append_attempt({"stage": "gate", "n": 2})
        lines = (self.root / "attempts.jsonl").read_text().splitlines()
        rows = [json.loads(x) for x in lines]
        self.assertEqual([r["n"] for r in rows], [1, 2])
        for r in rows:
            self.assertEqual(r["schema"], "gatedtrain.ainative.attempt")
            self.assertEqual(r["schema_version"], 1)
        self.assertEqual(lines[0], canonical_json(rows[0]).decode())


if __name__ == "__main__":
    unittest.main()
