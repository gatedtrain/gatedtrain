"""Canonical JSON and content-addressed manifests (MVP-SPEC section 2.8),
following the format conventions: canonical-JSON rules C1-C3 and manifest rules H1-H7.

Reference values come from the independent helpers in _support.py.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from _support import canonical_bytes, manifest_entries, manifest_hash


class TestCanonicalJson(unittest.TestCase):
    def test_canonical_json_bytes(self):
        from gatedtrain.canonical import canonical_json

        obj = {"b": [1, 2, {"z": None, "a": True}], "a": "xé", "c": 9007199254740991}
        # JCS: sorted keys, no whitespace, UTF-8 non-ASCII, no trailing newline (C1-C3).
        expected = '{"a":"xé","b":[1,2,{"a":true,"z":null}],"c":9007199254740991}'.encode("utf-8")
        self.assertEqual(canonical_json(obj), expected)
        self.assertEqual(canonical_json(obj), canonical_bytes(obj))

    def test_rejects_nan_and_large_int(self):
        from gatedtrain.canonical import canonical_json

        with self.assertRaises(ValueError):
            canonical_json({"x": float("nan")})          # C4: NaN rejected
        with self.assertRaises(ValueError):
            canonical_json({"x": 2 ** 53})               # C4: outside +/-(2^53 - 1)


class TestManifestHash(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gatedtrain-manifest-"))
        files = {
            "a-b": b"1",
            "a.b": b"22",
            "a/b": b"333",
            "a0": b"",                       # empty file (an empty file is a valid entry)
            "nested/deep/x.bin": bytes(range(256)),
            "café.txt": b"nfc name",   # non-ASCII NFC name
        }
        for rel, data in files.items():
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        (self.root / "emptydir").mkdir()    # H2: empty directories are not represented

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_entries_sorted_flat_string_order(self):
        from gatedtrain.canonical import manifest_entries as impl_entries

        entries = impl_entries(self.root)
        # H6: a-b < a.b < a/b < a0 by UTF-8 bytes, not tree-walk order.
        paths = [e["path"] for e in entries]
        self.assertEqual(paths[:4], ["a-b", "a.b", "a/b", "a0"])
        self.assertEqual(entries, manifest_entries(self.root))
        empty = [e for e in entries if e["path"] == "a0"][0]
        self.assertEqual(empty, {"path": "a0", "sha256": hashlib.sha256(b"").hexdigest(), "size": 0})

    def test_manifest_hash(self):
        from gatedtrain.canonical import manifest_hash as impl_hash

        self.assertEqual(impl_hash(self.root), manifest_hash(self.root))
        self.assertRegex(impl_hash(self.root), r"^sha256:[0-9a-f]{64}$")

    def test_hash_independent_of_location(self):
        from gatedtrain.canonical import manifest_hash as impl_hash

        other = Path(tempfile.mkdtemp(prefix="gatedtrain-manifest-copy-")) / "copy"
        try:
            shutil.copytree(self.root, other)
            self.assertEqual(impl_hash(other), impl_hash(self.root))  # H3: paths relative to the root
        finally:
            shutil.rmtree(other.parent, ignore_errors=True)

    def test_content_change_changes_hash(self):
        from gatedtrain.canonical import manifest_hash as impl_hash

        before = impl_hash(self.root)
        (self.root / "a0").write_bytes(b"x")
        self.assertNotEqual(impl_hash(self.root), before)

    def test_symlink_rejected(self):
        from gatedtrain.canonical import manifest_hash as impl_hash

        os.symlink(self.root / "a-b", self.root / "link")
        with self.assertRaises(ValueError):  # H2
            impl_hash(self.root)



class TestManifestUnreadableSubtree(unittest.TestCase):
    """F-26: a subdirectory that cannot be listed is an error, never silently left out of the hash
    (pathlib's rglob skips directories it cannot scan, so an rglob walk would hash a partial tree)."""

    def setUp(self):
        if os.geteuid() == 0:
            self.skipTest("running as root: root can list mode-000 directories")
        self.root = Path(tempfile.mkdtemp(prefix="gatedtrain-manifest-perm-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        (self.root / "visible.txt").write_bytes(b"v")
        sub = self.root / "locked"
        sub.mkdir()
        (sub / "hidden.txt").write_bytes(b"h")
        mode = sub.stat().st_mode
        os.chmod(sub, 0)
        self.addCleanup(os.chmod, sub, mode)        # runs before the rmtree above (LIFO)

    def test_raises_permission_error(self):
        from gatedtrain.canonical import manifest_entries as impl_entries

        with self.assertRaises(PermissionError):
            impl_entries(self.root)


if __name__ == "__main__":
    unittest.main()
