"""Review of the 3.12 path fix (2026-11-19; docs/findings.md, "Review of the 3.12 path fix"; tests/README.md
section 11.34).

- S1, S2, S5, T3 (MULTIMODAL-SPEC MM-1, EXT-45): an `output_ref` path contains no symbolic link at all. It is opened
  component by component from the arm directory without following links, so a link anywhere (also one whose lexical
  resolution exceeds `PATH_MAX`, where the lenient `realpath` gave up and kept a link), a loop, a dangling link or a
  path longer than `PATH_MAX` is a problem, never a file outside the arm and never a traceback; collect and merge
  copy the bytes they hash, so a file swapped after the check is refused.
- Arm files under an arm path longer than `PATH_MAX` give problems, not tracebacks (`read_meta`, `read_lines`,
  `provenance_problems`).
- S3: a spec `file:` URI through a link loop is refused on Python 3.12 (lenient `gatedtrain.paths.resolve`).
- T4: a loop and an unreadable link at the launch record's `output_dir` against `--stage-out`, the SDK's
  `bound_mounts` and the diffusers family's mount checks.
- S6 (EXT-113): a host interpreter whose path does not resolve is in no interpreter tree.

The verifier's reproductions (`verify-paths/scripts/s1.py`, `s2.py`, `s3.py`, `s5.py`) are kept here as tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _ext import SEEDS, ExtScenario, build_arm, candidate_score
from _support import SKEL_ROOT, read_jsonl

from gatedtrain.errors import Refused
from gatedtrain.ext import arm as arm_mod

PATH_MAX = os.pathconf("/", "PC_PATH_MAX")
DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY


def sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def deep_dirs(base: Path, names: list[str]) -> str:
    """Create `base/names[0]/names[1]/...` through directory descriptors (the path may exceed `PATH_MAX`); return
    it as a string."""
    fd = os.open(base, DIR_FLAGS)
    try:
        for name in names:
            os.mkdir(name, dir_fd=fd)
            nxt = os.open(name, DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = nxt
    finally:
        os.close(fd)
    return "/".join([str(base), *names])


def at(path: str, fn):
    """Call `fn(dir_fd, name)` for the last component of `path`, reaching its directory by descriptors."""
    parts = path.split("/")
    fd = os.open("/", DIR_FLAGS)
    try:
        for name in parts[1:-1]:
            nxt = os.open(name, DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = nxt
        return fn(fd, parts[-1])
    finally:
        os.close(fd)


class OutputRefCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-path-review-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.arm = self.tmp / "arm"
        (self.arm / "out").mkdir(parents=True)
        self.secret = self.tmp / "SECRET"
        self.secret.write_bytes(b"host secret\n")
        self.good = sha(self.secret.read_bytes())
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)

    def problem(self, path: str, digest: str | None = None, arm: Path | str | None = None) -> str | None:
        return arm_mod._output_ref_problem(Path(arm or self.arm), {"path": path, "sha256": digest or self.good})


class TestOutputRefNoLink(OutputRefCase):

    def test_plain_file_and_hard_link_accepted(self):
        (self.arm / "out" / "a.txt").write_bytes(b"output")
        os.link(self.arm / "out" / "a.txt", self.arm / "hard.txt")
        self.assertIsNone(self.problem("out/a.txt", sha(b"output")))
        self.assertEqual(self.problem("./out//a.txt", sha(b"output")),
                         "output_ref path './out//a.txt' is not in canonical form (no `.` or empty component)")
        self.assertIsNone(self.problem("hard.txt", sha(b"output")), "a hard link is a regular file (MM-1)")

    def test_symbolic_links_refused(self):
        """Every link is refused, also one that stays inside the arm, a dangling one and a loop (T3)."""
        (self.arm / "out" / "a.txt").write_bytes(b"host secret\n")
        cases = {"file link inside": ("in.txt", "out/a.txt", "in.txt"),
                 "file link outside": ("leak", str(self.secret), "leak"),
                 "directory link inside": ("dir", "out", "dir/a.txt"),
                 "directory link outside": ("up", str(self.tmp), "up/SECRET"),
                 "dangling": ("dangling", "missing", "dangling"),
                 "loop": ("loop", "loop", "loop/x"),
                 "loop as the file": ("loop2", "loop2", "loop2")}
        for name, (link, target, ref) in cases.items():
            with self.subTest(case=name):
                os.symlink(target, self.arm / link)
                why = self.problem(ref)
                self.assertIsNotNone(why)
                self.assertIn("contains a symbolic link", why)
                self.assertIn(repr(link), why)

    def test_s2_link_chain_beyond_path_max_refused(self):
        """Verifier's S2: `a` -> a 2.5 KB relative chain, `a/b` -> another, `b/leak` -> a host file; the lexical
        resolution exceeds PATH_MAX, where the lenient `realpath` kept the rest and the old check accepted it."""
        r1 = ["d" * 250] * 10
        r2 = ["e" * 250] * 10
        end = deep_dirs(self.arm, r1)
        end2 = deep_dirs(Path(end), r2)
        at(end2 + "/leak", lambda fd, name: os.symlink(str(self.secret), name, dir_fd=fd))
        os.symlink("/".join(r1), self.arm / "a")
        at(end + "/b", lambda fd, name: os.symlink("/".join(r2), name, dir_fd=fd))
        with open(self.arm / "a" / "b" / "leak", "rb") as f:          # the kernel reaches the host file
            self.assertEqual(f.read(), b"host secret\n")
        why = self.problem("a/b/leak")
        self.assertIn("contains a symbolic link ('a'", why or "")

    def test_s1_relative_arm_beyond_path_max_refused(self):
        """Verifier's S1: a relative arm, the output path longer than PATH_MAX, a link to a host file at its end."""
        os.chdir(self.tmp)
        names = ["d" * 200] * 21
        end = deep_dirs(self.arm, names)
        at(end + "/leak", lambda fd, name: os.symlink(str(self.secret), name, dir_fd=fd))
        ref = "/".join(names) + "/leak"
        self.assertGreater(len("arm/" + ref), PATH_MAX)
        for digest in (self.good, sha(b"other")):
            why = self.problem(ref, digest, arm="arm")
            self.assertIn("contains a symbolic link ('leak'", why or "")
        at(end + "/plain", lambda fd, name: os.close(os.open(name, os.O_WRONLY | os.O_CREAT, 0o644, dir_fd=fd)))
        self.assertIsNone(self.problem("/".join(names) + "/plain", sha(b""), arm="arm"),
                          "a regular file deeper than PATH_MAX is opened by descriptors")

    def test_s5_absent_beyond_path_max(self):
        """Verifier's S5: an absent path longer than PATH_MAX is absent, not ENAMETOOLONG out of `realpath`."""
        ref = "/".join(["x" * 250] * 17) + "/f"
        self.assertEqual(self.problem(ref), f"output_ref file {ref!r} is absent")

    def test_other_open_failures(self):
        (self.arm / "f").write_bytes(b"x")
        long = "n" * 300
        self.assertEqual(self.problem("f/x"), "output_ref file 'f/x' is absent")
        self.assertEqual(self.problem("out"), "output_ref file 'out' is not a regular file")
        os.mkfifo(self.arm / "fifo")
        self.assertEqual(self.problem("fifo"), "output_ref file 'fifo' is not a regular file", "never blocks")
        self.assertEqual(self.problem(long), f"output_ref file {long!r} cannot be opened (ENAMETOOLONG)")
        self.assertEqual(self.problem("out/a.txt", sha(b"y")), "output_ref file 'out/a.txt' is absent")
        (self.arm / "out" / "a.txt").write_bytes(b"x")
        self.assertEqual(self.problem("out/a.txt", sha(b"y")), "output_ref file 'out/a.txt' differs from its sha256")
        missing = self.tmp / "no-arm"
        self.assertIn("the arm directory is absent", self.problem("f", arm=missing))

    @unittest.skipIf(os.geteuid() == 0, "root opens a 0000 file")
    def test_unreadable_file(self):
        p = self.arm / "secret.txt"
        p.write_bytes(b"x")
        p.chmod(0)
        self.addCleanup(p.chmod, 0o644)
        self.assertEqual(self.problem("secret.txt"), "output_ref file 'secret.txt' cannot be opened (EACCES)")

    def test_arm_path_beyond_path_max(self):
        """The arm files under an arm path longer than PATH_MAX: problems, never a traceback (Python 3.12's
        `Path.is_file()` raises ENAMETOOLONG)."""
        arm = deep_dirs(self.tmp, ["a" * 250] * 17)
        self.assertGreater(len(arm), PATH_MAX)
        meta, why = arm_mod.read_meta(Path(arm))
        self.assertIsNone(meta)
        self.assertTrue(why and why[0].endswith("eval_meta.json cannot be read (ENAMETOOLONG)"), why)
        recs, why = arm_mod.read_lines(Path(arm), seeds=[1], score_range=[0, 1], error_classes=())
        self.assertTrue(why and why[0].endswith("items.jsonl cannot be read (ENAMETOOLONG)"), why)
        self.assertEqual(arm_mod.provenance_problems(Path(arm), {"eval_seeds": [1]}, nel_benchmark=None),
                         ["merge.json cannot be read (ENAMETOOLONG); provenance cannot be checked"])
        self.assertIn("the arm directory cannot be opened (ENAMETOOLONG)", self.problem("x", arm=arm))


class TestCopyChecked(OutputRefCase):
    """collect and merge copy the bytes they hash (MM-1): a file changed or replaced by a link after the line checks
    is refused, and nothing is left at the destination."""

    def setUp(self):
        super().setUp()
        self.src = self.arm / "out" / "a.txt"
        self.src.write_bytes(b"output")
        self.ref = {"path": "out/a.txt", "sha256": sha(b"output")}
        self.dst = self.tmp / "dst.txt"

    def test_copy(self):
        arm_mod.copy_output_ref(self.arm, self.ref, self.dst, flag="--eval-candidate")
        self.assertEqual(self.dst.read_bytes(), b"output")

    def test_swapped_for_a_link(self):
        self.src.unlink()
        os.symlink(self.secret, self.src)
        with self.assertRaises(Refused) as cm:
            arm_mod.copy_output_ref(self.arm, self.ref, self.dst, flag="--eval-candidate")
        (p,) = cm.exception.problems
        self.assertEqual((p.rule, p.field), ("EXT-ARM", "--eval-candidate"))
        self.assertIn("contains a symbolic link", p.message)
        self.assertFalse(self.dst.exists())

    def test_changed_content(self):
        self.src.write_bytes(b"host secret\n")
        with self.assertRaises(Refused) as cm:
            arm_mod.copy_output_ref(self.arm, self.ref, self.dst, flag="--eval-candidate")
        self.assertIn("differs from its sha256 (changed after it was checked)", cm.exception.problems[0].message)
        self.assertFalse(self.dst.exists())


class TestMergeCopiesCheckedBytes(unittest.TestCase):
    """ext-merge-eval, in process: an output swapped for a link between the line checks and the copy refuses the
    merge (EXT-ARM) and leaves no output directory."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-path-review-m-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = ExtScenario(cls.tmp / "sc")

    def arms(self, name: str) -> list[Path]:
        out = []
        for i, seed in enumerate(SEEDS):
            a = build_arm(self.tmp / name / f"s{i}", self.sc.items, candidate_score, model=self.sc.cand_model,
                          item_set=self.sc.item_set_hash, seeds=[seed])
            lines = read_jsonl(a / "items.jsonl")
            (a / "outputs").mkdir()
            (a / "outputs" / f"{i}.txt").write_bytes(b"output %d" % i)
            lines[0]["output_ref"] = {"path": f"outputs/{i}.txt", "sha256": sha(b"output %d" % i)}
            (a / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
            out.append(a)
        return out

    def test_merge_copies(self):
        from gatedtrain.ext.merge import merge_eval

        out = merge_eval(self.sc.paths["registration"], self.tmp / "ok" / "out", self.arms("ok"))
        self.assertEqual((out / "outputs" / "1.txt").read_bytes(), b"output 1")

    def test_swap_after_the_check_refused(self):
        from gatedtrain.ext import merge as merge_mod

        secret = self.tmp / "SECRET"
        secret.write_bytes(b"host secret\n")
        runs = self.arms("swap")
        real = arm_mod.copy_output_ref

        def swap_then_copy(arm, ref, dst, *, flag, key=None):
            p = Path(arm) / ref["path"]
            p.unlink()
            p.symlink_to(secret)
            return real(arm, ref, dst, flag=flag, key=key)

        out = self.tmp / "swap" / "out"
        with mock.patch.object(merge_mod, "copy_output_ref", swap_then_copy):
            with self.assertRaises(Refused) as cm:
                merge_mod.merge_eval(self.sc.paths["registration"], out, runs)
        self.assertIn("contains a symbolic link", cm.exception.problems[0].message)
        self.assertFalse(out.exists())
        self.assertEqual([p.name for p in out.parent.iterdir() if p.name.startswith(".out.tmp-")], [])


class TestSpecLinkLoop(unittest.TestCase):
    """S3: on Python 3.12 `Path.resolve()` raises RuntimeError for a link loop; the spec's `file:` URIs and the
    spec path resolve leniently (`gatedtrain.paths`), so the loop is a refusal."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-path-review-s-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        os.symlink("loop", self.tmp / "loop")

    def test_uri_through_a_loop(self):
        from gatedtrain.spec import _uri_path, load_spec

        self.assertEqual(_uri_path("file:loop/x", self.tmp), (self.tmp / "loop" / "x", None))
        demo = SKEL_ROOT / "demo"
        for name in ("spec.yaml", "registration.yaml", "trainer.yaml"):
            shutil.copyfile(demo / name, self.tmp / name)
        text = (self.tmp / "spec.yaml").read_text(encoding="utf-8")
        self.assertEqual(text.count('uri: "file:registration.yaml@'), 1)
        (self.tmp / "spec.yaml").write_text(text.replace('uri: "file:registration.yaml@', 'uri: "file:loop/r.yaml@'),
                                            encoding="utf-8")
        with self.assertRaises(Refused):
            load_spec(self.tmp / "spec.yaml")
        with self.assertRaises(Refused):
            load_spec(self.tmp / "loop")


UNREADABLE_LINK = "/proc/1/root"
NO_UNREADABLE_LINK = os.access(UNREADABLE_LINK, os.R_OK) or not os.path.islink(UNREADABLE_LINK)


class TestLenientSites(unittest.TestCase):
    """T4: a link loop and an unreadable link where a launch record, the SDK and the diffusers family resolve paths
    leniently: a mismatch or no binding, never a traceback."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-path-review-t4-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = ExtScenario(cls.tmp / "sc")
        os.symlink("loop", cls.tmp / "loop")

    def bad_paths(self) -> dict[str, str]:
        out = {"loop": str(self.tmp / "loop" / "x")}
        if not NO_UNREADABLE_LINK:
            out["unreadable link"] = UNREADABLE_LINK + "/x"
        return out

    def test_output_dir_against_stage_out(self):
        from gatedtrain.ext.launch import host_config, record_problems

        for name, path in self.bad_paths().items():
            with self.subTest(case=name):
                rec = self.sc.launch_record(output_dir=path)
                got = record_problems(rec, adapter="synthetic", stage_out=self.sc.stage_out,
                                      framework="synthetic-trainer", config=host_config(self.sc.env()))
                self.assertIn("--launch-record.output_dir", [p.field for p in got])

    def test_sdk_bound_mounts(self):
        from gatedtrain.ext.sdk import bound_mounts

        for name, path in self.bad_paths().items():
            with self.subTest(case=name):
                mounts = [{"host": path, "container": "/inputs/x", "mode": "ro"}]
                self.assertEqual(bound_mounts(mounts, self.sc.base), [])
                self.assertEqual(bound_mounts(mounts, Path(path)), mounts, "the same lenient form on both sides")

    def test_diffusers_mount_binding(self):
        from gatedtrain.families.diffusers_lora import DiffusersLoraAdapter

        class Family(DiffusersLoraAdapter):             # a recipe adapter supplies the fixes
            BASE_FIX = DATA_FIX = "fix"

        ad = Family()
        good = [{"host": str(self.sc.base), "container": "/inputs/base", "mode": "ro"},
                {"host": str(self.sc.export), "container": "/inputs/export", "mode": "ro"}]
        for name, path in self.bad_paths().items():
            with self.subTest(case=name):
                got = ad.mount_binding_problems(good, base=Path(path), data=Path(path), base_field="b", data_field="d")
                self.assertEqual([(p.rule, p.field) for p in got], [("EXT-BASE", "b"), ("EXT-DATA", "d")])
                mounts = good + [{"host": path, "container": "/inputs/bad", "mode": "ro"}]
                got = ad.mount_binding_problems(mounts, base=self.sc.base, data=self.sc.export, base_field="b",
                                                data_field="d")
                self.assertEqual(got, [])


class TestInterpreterTree(unittest.TestCase):
    """S6 (EXT-113): the interpreter's host file must resolve (strict) inside a tree; a loop or an unreadable link
    is in no tree, even below a tree's lexical path."""

    def test_unresolvable_is_in_no_tree(self):
        from gatedtrain.ext.launch import _in_tree

        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-path-review-s6-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "venv" / "bin").mkdir(parents=True)
        (tmp / "venv" / "bin" / "python").write_text("stub", encoding="utf-8")
        os.symlink("loop", tmp / "venv" / "bin" / "loop")
        trees = (str(tmp / "venv"),)
        self.assertTrue(_in_tree(str(tmp / "venv" / "bin" / "python"), trees))
        self.assertFalse(_in_tree(str(tmp / "venv" / "bin" / "loop"), trees))
        self.assertFalse(_in_tree(str(tmp / "venv" / "bin" / "missing"), trees))
        if not NO_UNREADABLE_LINK:
            self.assertFalse(_in_tree(UNREADABLE_LINK + "/usr/bin/python3", ("/proc/1/root/usr",)))


if __name__ == "__main__":
    unittest.main()
