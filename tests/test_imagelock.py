"""IMG-1 .. IMG-4: the image lock (data), the owner's config, precedence, validation, no silent widening."""

from __future__ import annotations

import copy
import json
import os
import re
import stat
import tempfile
import unittest
from pathlib import Path

from gatedtrain import imagelock
from gatedtrain.errors import Refused
from gatedtrain.families import nemo_rl

SRC = Path(imagelock.__file__).resolve().parent
TREE = SRC.parent.parent
NOT_A_CHECKOUT = "not a source checkout (no containers/ or pyproject.toml beside src/): an sdist or wheel install"
IS_CHECKOUT = (TREE / "containers").is_dir() and (TREE / "pyproject.toml").is_file()
ID_RE = re.compile(r"sha256:[0-9a-f]{64}")
A = "sha256:" + "a" * 64
B = "sha256:" + "b" * 64
C = "sha256:" + "c" * 64
COMMIT = "1" * 40


def entry(image_id, origin="maintainer-built", **over):
    e = {"id": image_id, "origin": origin,
         "recipe": {"name": "r", "source": None, "commit": COMMIT, "containerfile_sha256": None, "build": None},
         "base_id": None, "built_utc": "2026-11-12", "added_by": "test", "note": None}
    e.update(over)
    return e


def lock_doc(**images):
    imgs = images or {"x/one": [entry(A)], "x/two": []}
    return {"schema": "gatedtrain.image-lock", "schema_version": 1,
            "images": {k: {"purpose": "p", "accepted": v} for k, v in imgs.items()}}


def config_doc(**images):
    return {"schema": "gatedtrain.image-config", "schema_version": 1,
            "images": {k: {"accepted": v} for k, v in images.items()}}


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def write(self, name, doc, mode=0o600, raw=None):
        p = self.dir / name
        p.write_text(raw if raw is not None else json.dumps(doc))
        p.chmod(mode)
        return p

    def lock(self, doc=None):
        return imagelock.load_lock(self.write("lock.json", doc or lock_doc()))

    def config(self, doc, lock=None, **kw):
        return imagelock.load_user_config(self.write("images.json", doc), lock=lock or self.lock(), **kw)

    def refused(self, fn, rule):
        with self.assertRaises(Refused) as cm:
            fn()
        self.assertEqual(cm.exception.problems[0].rule, rule, cm.exception.problems)
        return cm.exception.problems[0]


class TestShippedLock(unittest.TestCase):
    def test_shipped_lock_loads_and_has_the_three_logical_images(self):
        lock = imagelock.load_lock()
        self.assertEqual(set(lock["images"]), {"nemo-rl/tierb", "nemo-rl/tierb-uid1000", "nemo-rl/tierb-rootless"})
        for logical in ("nemo-rl/tierb", "nemo-rl/tierb-uid1000"):
            self.assertEqual(len(imagelock.maintainer_ids(logical, lock=lock)), 1)
        self.assertEqual(imagelock.maintainer_ids("nemo-rl/tierb-rootless", lock=lock), [])

    def test_ids_recorded_before_the_lock_stay_accepted(self):
        """Records that name the maintainer images keep verifying because the IDs are lock entries (the IDs are read
        from the shipped data file, never written in a test)."""
        raw = json.loads((SRC / "data" / "image-lock.json").read_text())
        lock = imagelock.load_lock()
        for logical in ("nemo-rl/tierb", "nemo-rl/tierb-uid1000"):
            image_id = raw["images"][logical]["accepted"][0]["id"]
            self.assertIn(image_id, imagelock.maintainer_ids(logical, lock=lock))
            self.assertTrue(imagelock.provenance(image_id, lock=lock, config=None)["in_maintainer_lock"])
            record = {"runtime": {"kind": "container", "image": image_id}}          # a registration-like record
            self.assertIn(record["runtime"]["image"], imagelock.accepted_ids(logical, lock=lock, config=None))

    @unittest.skipUnless(IS_CHECKOUT, NOT_A_CHECKOUT)
    def test_uid1000_recipe_claims_no_containerfile_hash_for_the_pre_arg_build(self):
        """The maintainer image was built from the Containerfile before it took BASE as a build argument (same chown
        list, same layers). The shipped file has a different hash, so the lock records none rather than a hash of a file
        that did not build the image, and says why."""
        lock = imagelock.load_lock()
        acc = imagelock.resolve("nemo-rl/tierb-uid1000", lock=lock, config=None)[0]
        self.assertIsNone(acc.recipe["containerfile_sha256"])
        entry = lock["images"]["nemo-rl/tierb-uid1000"]["accepted"][0]
        self.assertIn("pre-ARG", entry["note"])
        self.assertIn("same chown list", entry["note"])
        self.assertIn("containers/nemo-rl-uid1000", entry["note"])
        self.assertEqual(acc.base_id, imagelock.maintainer_ids("nemo-rl/tierb", lock=lock)[0])

    @unittest.skipUnless(IS_CHECKOUT, NOT_A_CHECKOUT)
    def test_lock_ships_as_package_data(self):
        text = (TREE / "pyproject.toml").read_text()
        self.assertIn("image-lock.json", text)

    def test_lock_file_and_package_marker_are_present(self):
        self.assertTrue((SRC / "data" / "__init__.py").exists())
        self.assertTrue(imagelock.LOCK_PATH.is_file())

    def test_lock_sha256_is_the_hash_of_the_shipped_file(self):
        import hashlib
        self.assertEqual(imagelock.lock_sha256(), "sha256:" + hashlib.sha256(imagelock.LOCK_PATH.read_bytes()).hexdigest())


class TestLockValidation(Base):
    def test_valid(self):
        self.assertEqual(self.lock()["schema_version"], 1)

    def test_unknown_top_field(self):
        d = lock_doc(); d["extra"] = 1
        self.refused(lambda: self.lock(d), "IMG-1")

    def test_missing_top_field(self):
        d = lock_doc(); del d["images"]
        self.refused(lambda: self.lock(d), "IMG-1")

    def test_unknown_and_missing_entry_fields(self):
        e = entry(A); e["surprise"] = 1
        self.refused(lambda: self.lock(lock_doc(x=[e])), "IMG-1")
        e = entry(A); del e["recipe"]
        self.refused(lambda: self.lock(lock_doc(x=[e])), "IMG-1")
        e = entry(A); e["recipe"]["extra"] = 1
        self.refused(lambda: self.lock(lock_doc(x=[e])), "IMG-1")

    def test_schema_version_both_ways(self):
        for v in (0, 2, "1", True):
            d = lock_doc(); d["schema_version"] = v
            p = self.refused(lambda: self.lock(d), "IMG-1")
            self.assertEqual(p.field, "schema_version")

    def test_wrong_schema_name(self):
        d = lock_doc(); d["schema"] = "other"
        self.refused(lambda: self.lock(d), "IMG-1")

    def test_duplicate_json_keys(self):
        raw = '{"schema": "gatedtrain.image-lock", "schema_version": 1, "images": {}, "images": {}}'
        self.refused(lambda: imagelock.load_lock(self.write("l.json", None, raw=raw)), "IMG-1")

    def test_bad_ids(self):
        for bad in ("sha256:" + "A" * 64, "sha256:abc", A[7:], "sha256:" + "a" * 65, 5):
            self.refused(lambda: self.lock(lock_doc(x=[entry(bad)])), "IMG-1")

    def test_lock_may_only_hold_maintainer_built(self):
        self.refused(lambda: self.lock(lock_doc(x=[entry(A, origin="user-built")])), "IMG-1")
        self.refused(lambda: self.lock(lock_doc(x=[entry(A, origin="other")])), "IMG-1")

    def test_recipe_rules(self):
        e = entry(A); e["recipe"]["name"] = ""
        self.refused(lambda: self.lock(lock_doc(x=[e])), "IMG-1")
        e = entry(A); e["recipe"]["commit"] = "zz"
        self.refused(lambda: self.lock(lock_doc(x=[e])), "IMG-1")

    def test_duplicate_id_inside_lock(self):
        self.refused(lambda: self.lock(lock_doc(x=[entry(A)], y=[entry(A)])), "IMG-1")

    def test_oversize_file_and_entries(self):
        self.refused(lambda: imagelock.load_lock(self.write("l.json", None, raw=" " * (256 * 1024 + 1))), "IMG-1")
        many = [entry("sha256:" + f"{i:064x}") for i in range(65)]
        self.refused(lambda: self.lock(lock_doc(x=many)), "IMG-1")

    def test_not_json(self):
        self.refused(lambda: imagelock.load_lock(self.write("l.json", None, raw="{")), "IMG-1")

    def test_built_utc_must_be_a_date(self):
        self.refused(lambda: self.lock(lock_doc(x=[entry(A, built_utc="yesterday")])), "IMG-1")
        self.refused(lambda: self.lock(lock_doc(x=[entry(A, built_utc="\uff12\uff10\uff12\uff16-11-12")])), "IMG-1")
        self.refused(lambda: self.lock(lock_doc(x=[entry(A, built_utc="2026-11-12\n")])), "IMG-1")

    def test_free_text_must_be_one_printable_line(self):
        """IMG-1/IMG-2: every string that reaches report.md or CLI output is one printable line (no newline, no control
        or format character), so a lock or config cannot forge a report line."""
        bad = ["x\n## Verdict", "x\r- keeper mode: separate-user", "a\x00b", "a\x1b[31mb", "a\u202eb", "a\u2028b", ""]
        for text in bad:
            for key in ("added_by", "note"):
                self.refused(lambda: self.lock(lock_doc(x=[entry(A, **{key: text})])), "IMG-1")
            for key in ("name", "source", "build"):
                e = entry(A); e["recipe"][key] = text
                self.refused(lambda: self.lock(lock_doc(x=[e])), "IMG-1")
        d = lock_doc(); d["images"]["x/one"]["purpose"] = "p\n## Verdict"
        self.refused(lambda: self.lock(d), "IMG-1")
        self.lock(lock_doc(x=[entry(A, note="a plain note, with punctuation: 1/2")]))

    def test_free_text_length_is_capped(self):
        self.refused(lambda: self.lock(lock_doc(x=[entry(A, note="n" * 513)])), "IMG-1")


class TestConfig(Base):
    def test_missing_file_is_empty(self):
        cfg = imagelock.load_user_config(self.dir / "none.json", lock=self.lock())
        self.assertEqual(cfg["images"], {})

    def test_valid_config(self):
        cfg = self.config(config_doc(**{"x/one": [entry(B, "user-built")]}))
        self.assertEqual(cfg["images"]["x/one"]["accepted"][0]["id"], B)

    def test_config_may_only_hold_user_built(self):
        self.refused(lambda: self.config(config_doc(**{"x/one": [entry(B)]})), "IMG-2")

    def test_user_built_needs_commit_or_containerfile_hash(self):
        e = entry(B, "user-built"); e["recipe"]["commit"] = None
        self.refused(lambda: self.config(config_doc(**{"x/one": [e]})), "IMG-2")
        e["recipe"]["containerfile_sha256"] = "sha256:" + "d" * 64
        self.config(config_doc(**{"x/one": [e]}))

    def test_unknown_logical_name(self):
        self.refused(lambda: self.config(config_doc(**{"x/invented": [entry(B, "user-built")]})), "IMG-2")

    def test_unknown_fields_and_schema(self):
        d = config_doc(); d["extra"] = 1
        self.refused(lambda: self.config(d), "IMG-2")
        d = config_doc(); d["schema_version"] = 2
        self.refused(lambda: self.config(d), "IMG-2")
        d = config_doc(); d["schema"] = "gatedtrain.image-lock"
        self.refused(lambda: self.config(d), "IMG-2")

    def test_symlink_refused(self):
        real = self.write("real.json", config_doc())
        link = self.dir / "link.json"
        link.symlink_to(real)
        self.refused(lambda: imagelock.load_user_config(link, lock=self.lock()), "IMG-3")

    def test_group_or_other_writable_refused(self):
        for mode in (0o620, 0o602, 0o666):
            p = self.write("images.json", config_doc(), mode=mode)
            self.refused(lambda: imagelock.load_user_config(p, lock=self.lock()), "IMG-3")

    def test_readable_by_others_is_fine(self):
        p = self.write("images.json", config_doc(), mode=0o644)
        imagelock.load_user_config(p, lock=self.lock())

    def test_wrong_owner_refused(self):
        p = self.write("images.json", config_doc())
        self.refused(lambda: imagelock.load_user_config(p, lock=self.lock(), uid=os.getuid() + 1), "IMG-3")

    def test_not_a_regular_file_refused(self):
        d = self.dir / "dir.json"
        d.mkdir()
        self.refused(lambda: imagelock.load_user_config(d, lock=self.lock()), "IMG-3")

    def test_default_path_follows_xdg_config_home_only_for_the_path(self):
        old = {k: os.environ.get(k) for k in ("XDG_CONFIG_HOME", "HOME")}
        try:
            os.environ["XDG_CONFIG_HOME"] = "/x/cfg"
            self.assertEqual(imagelock.user_config_path(), Path("/x/cfg/gatedtrain/images.json"))
            del os.environ["XDG_CONFIG_HOME"]
            os.environ["HOME"] = "/h"
            self.assertEqual(imagelock.user_config_path(), Path("/h/.config/gatedtrain/images.json"))
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


class TestResolve(Base):
    def test_lock_first_then_config_in_file_order(self):
        lock = self.lock(lock_doc(**{"x/one": [entry(A)]}))
        cfg = self.config(config_doc(**{"x/one": [entry(C, "user-built"), entry(B, "user-built")]}), lock=lock)
        acc = imagelock.resolve("x/one", lock=lock, config=cfg)
        self.assertEqual([a.id for a in acc], [A, C, B])
        self.assertEqual([a.source for a in acc], ["lock", "owner_config", "owner_config"])

    def test_config_cannot_hold_an_id_the_lock_lists_under_any_logical_name(self):
        """IMG-2: a config entry for a lock ID (same logical image or another) is refused, so the owner file can never
        re-label or re-attribute a shipped image."""
        lock = self.lock(lock_doc(**{"x/one": [entry(A)], "x/two": []}))
        for logical in ("x/one", "x/two"):
            p = self.refused(lambda: self.config(config_doc(**{logical: [entry(A, "user-built")]}), lock=lock), "IMG-2")
            self.assertIn("lock", p.message)

    def test_same_id_under_two_logicals_in_config_refused(self):
        lock = self.lock(lock_doc(a=[entry(A)], b=[]))
        self.refused(lambda: self.config(config_doc(b=[entry(B, "user-built")], a=[entry(B, "user-built")]), lock=lock),
                     "IMG-2")

    def test_unknown_logical_refused(self):
        self.refused(lambda: imagelock.resolve("nope", lock=self.lock(), config=None), "IMG-4")

    def test_provenance(self):
        lock = self.lock()
        cfg = self.config(config_doc(**{"x/two": [entry(B, "user-built")]}), lock=lock)
        p = imagelock.provenance(A, lock=lock, config=cfg)
        self.assertEqual((p["logical"], p["origin"], p["in_maintainer_lock"], p["source"]),
                         ("x/one", "maintainer-built", True, "lock"))
        p = imagelock.provenance(B, lock=lock, config=cfg)
        self.assertEqual((p["logical"], p["origin"], p["in_maintainer_lock"], p["source"]),
                         ("x/two", "user-built", False, "owner_config"))
        self.assertEqual(p["recipe"]["name"], "r")
        p = imagelock.provenance(C, lock=lock, config=cfg)
        self.assertEqual((p["logical"], p["origin"], p["recipe"], p["in_maintainer_lock"], p["source"]),
                         (None, None, None, False, "unlisted"))

    def test_in_maintainer_lock_ignores_config(self):
        lock = self.lock()
        self.assertTrue(imagelock.in_maintainer_lock(A, lock=lock))
        self.assertFalse(imagelock.in_maintainer_lock(B, lock=lock))


class TestNoSilentWidening(Base):
    """An agent controls its environment and its project, not the owner's config file or the shipped lock."""

    def test_env_vars_and_project_files_have_no_effect(self):
        before = imagelock.accepted_ids("nemo-rl/tierb", config=None)
        proj = self.dir / "project"
        proj.mkdir()
        for name in ("images.json", "image-lock.json", "gatedtrain-images.json"):
            (proj / name).write_text(json.dumps(config_doc(**{"nemo-rl/tierb": [entry(A, "user-built")]})))
        saved = dict(os.environ)
        try:
            for k in ("GATEDTRAIN_IMAGES", "GATEDTRAIN_IMAGE_LOCK", "GATEDTRAIN_ACCEPTED_IMAGES",
                      "GATEDTRAIN_EXTRA_IMAGES", "GATEDTRAIN_PACK_VLM_CLEVR_EXTRA_IMAGES"):
                os.environ[k] = A
            cwd = os.getcwd()
            os.chdir(proj)
            try:
                after = imagelock.accepted_ids("nemo-rl/tierb", config=None)
            finally:
                os.chdir(cwd)
        finally:
            os.environ.clear()
            os.environ.update(saved)
        self.assertEqual(before, after)
        self.assertNotIn(A, after)

    def test_source_reads_the_environment_only_for_the_config_path(self):
        src = (SRC / "imagelock.py").read_text()
        uses = re.findall(r"os\.(?:environ|getenv)[^\n]*", src)
        self.assertTrue(uses)
        for use in uses:
            self.assertTrue("XDG_CONFIG_HOME" in use or "HOME" in use, use)


class TestBackwardCompat(unittest.TestCase):
    def test_module_attributes_come_from_the_lock(self):
        lock = imagelock.load_lock()
        tierb = imagelock.maintainer_ids("nemo-rl/tierb", lock=lock)[0]
        uid1000 = imagelock.maintainer_ids("nemo-rl/tierb-uid1000", lock=lock)[0]
        self.assertEqual(nemo_rl.NEMO_RL_IMAGE, tierb)
        self.assertEqual(nemo_rl.NEMO_RL_UID1000_IMAGE, uid1000)
        self.assertEqual(nemo_rl.NEMO_RL_IMAGES, (tierb, uid1000))
        self.assertEqual(nemo_rl.NEMO_RL_IMAGES[0], nemo_rl.NEMO_RL_IMAGE)
        ns = {}
        exec("from gatedtrain.families.nemo_rl import NEMO_RL_IMAGE, NEMO_RL_IMAGES, NEMO_RL_UID1000_IMAGE", ns)
        self.assertEqual(ns["NEMO_RL_IMAGE"], tierb)

    def test_empty_maintainer_list_raises_attribute_error_not_index_error(self):
        from unittest import mock
        empty = lock_doc(**{"nemo-rl/tierb": [], "nemo-rl/tierb-uid1000": []})
        with mock.patch.object(imagelock, "load_lock", return_value=empty):
            for name in ("NEMO_RL_IMAGE", "NEMO_RL_UID1000_IMAGE", "NEMO_RL_IMAGES"):
                with self.assertRaises(AttributeError):
                    getattr(nemo_rl, name)

    def test_unknown_attribute_still_raises(self):
        with self.assertRaises(AttributeError):
            nemo_rl.NEMO_RL_NOPE

    def test_accepted_images_includes_the_owner_config(self):
        self.assertEqual(nemo_rl.accepted_images("nemo-rl/tierb", config=None),
                         imagelock.accepted_ids("nemo-rl/tierb", config=None))

    def test_no_lock_image_id_in_src_outside_the_lock(self):
        """The IDs the lock holds appear nowhere else in the source tree (synthetic placeholder digests are not IDs)."""
        lock_ids = [e["id"] for v in imagelock.load_lock()["images"].values() for e in v["accepted"]]
        self.assertTrue(lock_ids)
        offenders = []
        for p in SRC.rglob("*"):
            if p.is_file() and p != SRC / "data" / "image-lock.json" and "__pycache__" not in p.parts:
                text = p.read_text(errors="replace")
                if any(i in text or i[7:15] in text for i in lock_ids):
                    offenders.append(str(p.relative_to(SRC)))
        self.assertEqual(offenders, [])


class TestReviewRound2(Base):
    """M3, M4, L5, L11: hostile files cannot crash or forge output; one read for lock and hash; safe writes."""

    def test_deeply_nested_json_is_refused_not_a_crash(self):
        deep = "[" * 100_000 + "]" * 100_000
        self.assertLess(len(deep), imagelock.MAX_FILE_BYTES)
        self.refused(lambda: imagelock.load_lock(self.write("lock.json", None, raw=deep)), "IMG-1")
        lock = self.lock()
        cfg = self.write("images.json", None, raw=deep)
        self.refused(lambda: imagelock.load_user_config(cfg, lock=lock), "IMG-2")

    def test_hostile_logical_key_cannot_inject_terminal_text(self):
        evil = "a\x1b[31m\nerror[IMG-0] fake: forged | fix: x"
        problem = self.refused(lambda: self.config(config_doc(**{evil: []})), "IMG-2")
        line = problem.line()
        self.assertNotIn("\x1b", line)
        self.assertNotIn("\n", line)
        lock_problem = self.refused(lambda: imagelock.load_lock(self.write("l.json", lock_doc(**{evil: []}))), "IMG-1")
        self.assertNotIn("\x1b", lock_problem.line())

    def test_problem_line_escapes_control_characters_in_every_part(self):
        from gatedtrain.errors import Problem
        line = Problem("R-1", "f\x1b[0m", "m\x07\u202e", "x\x00").line()
        for bad in ("\x1b", "\x07", "\u202e", "\x00"):
            self.assertNotIn(bad, line)

    def test_lock_and_hash_come_from_the_same_bytes(self):
        import hashlib
        from unittest import mock
        p = self.write("lock.json", lock_doc())
        data = p.read_bytes()
        reads = []
        real = Path.read_bytes

        def counting(self_):
            reads.append(self_)
            return real(self_)
        with mock.patch.object(Path, "read_bytes", counting):
            doc, sha = imagelock.load_lock_with_sha(p)
        self.assertEqual(len(reads), 1)
        self.assertEqual(sha, "sha256:" + hashlib.sha256(data).hexdigest())
        self.assertEqual(doc["images"]["x/one"]["accepted"][0]["id"], A)

    def test_built_utc_must_be_a_real_date(self):
        for bad in ("2026-13-45", "2026-02-30", "0000-00-00"):
            self.refused(lambda bad=bad: self.lock(lock_doc(**{"x/one": [entry(A, built_utc=bad)]})), "IMG-1")
        self.lock(lock_doc(**{"x/one": [entry(A, built_utc="2024-02-29")]}))

    def test_write_survives_a_stale_temp_file_of_the_old_name_and_leaves_none(self):
        lock = self.lock()
        target = self.dir / "gt" / "images.json"
        target.parent.mkdir()
        stale = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        stale.write_text("planted")
        imagelock.write_user_config(imagelock.empty_config(), target, lock=lock)
        self.assertEqual(stale.read_text(), "planted")                   # not clobbered, not followed
        self.assertEqual(sorted(x.name for x in target.parent.iterdir() if x != target and x != stale
                                and not x.name.endswith(imagelock.LOCK_SUFFIX)), [])

    def test_temp_names_are_unique(self):
        names = {imagelock._tmp_name(Path("/x/images.json")).name for _ in range(20)}
        self.assertEqual(len(names), 20)

    def test_read_modify_write_holds_an_advisory_lock(self):
        import fcntl
        target = self.dir / "gt" / "images.json"
        with imagelock.locked_config(target):
            fd = os.open(str(target) + imagelock.LOCK_SUFFIX, os.O_RDWR)
            try:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(fd)
        fd = os.open(str(target) + imagelock.LOCK_SUFFIX, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)                # released on exit
        finally:
            os.close(fd)


class TestPackaging(unittest.TestCase):
    """L8: the lock is data the installed package reads, so the sdist and the wheel must carry it (a source tree test
    passes either way). Nothing is built into the tree: the build runs on a copy in a temp directory."""

    LOCK_REL = "gatedtrain/data/image-lock.json"

    @unittest.skipUnless(IS_CHECKOUT, NOT_A_CHECKOUT)
    def test_pyproject_and_manifest_list_the_lock(self):
        import tomllib
        conf = tomllib.loads((TREE / "pyproject.toml").read_text())
        self.assertIn("image-lock.json", conf["tool"]["setuptools"]["package-data"]["gatedtrain.data"])
        manifest = (TREE / "MANIFEST.in").read_text()
        for line in manifest.splitlines():
            if line.startswith(("prune ", "exclude ", "global-exclude ")):
                self.assertNotRegex(line, r"\b(src|data|gatedtrain)\b", line)      # nothing prunes the package data

    def _build(self, kind: str) -> list[str]:
        import shutil
        import subprocess
        import sys
        try:
            import setuptools  # noqa: F401
        except ImportError:
            self.skipTest("setuptools is not installed here; the sdist and wheel contents were not checked")
        with tempfile.TemporaryDirectory() as tmp:
            copy, out = Path(tmp) / "src", Path(tmp) / "out"
            out.mkdir()
            shutil.copytree(TREE, copy, ignore=shutil.ignore_patterns(
                ".git", ".venv", "__pycache__", "tests", "demo", "docs", "containers", "*.egg-info", "build", "dist"))
            code = f"import setuptools.build_meta as b; print(b.build_{kind}({str(out)!r}))"
            done = subprocess.run([sys.executable, "-I", "-c", code], cwd=copy, capture_output=True, text=True)
            if done.returncode != 0:
                self.skipTest(f"cannot build the {kind} offline here: {done.stderr.strip().splitlines()[-1:]}")
            name = done.stdout.strip().splitlines()[-1]
            if kind == "sdist":
                import tarfile
                with tarfile.open(out / name) as tar:
                    return tar.getnames()
            import zipfile
            with zipfile.ZipFile(out / name) as zf:
                return zf.namelist()

    @unittest.skipUnless(IS_CHECKOUT, NOT_A_CHECKOUT)
    def test_the_sdist_carries_the_lock(self):
        self.assertTrue(any(n.endswith("src/" + self.LOCK_REL) for n in self._build("sdist")))

    @unittest.skipUnless(IS_CHECKOUT, NOT_A_CHECKOUT)
    def test_the_wheel_carries_the_lock_and_no_recipes(self):
        names = self._build("wheel")
        self.assertIn(self.LOCK_REL, names)
        self.assertFalse([n for n in names if n.startswith("containers/")])    # the recipes need a source checkout


if __name__ == "__main__":
    unittest.main()
