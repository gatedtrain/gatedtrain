"""Review of PR 2 (2026-11-19; docs/findings.md, "Review of PR 2"; tests/README.md section 11.34): arm files are read
once, without following links, and the run holds the bytes that were checked.

- A1, C2: `ext-collect` writes `items.jsonl`, `eval_meta.json` and `merge.json` from the bytes it checked; files
  swapped after the checks change nothing, and an output swapped for a link refuses (one clause here; several
  clauses in `test_r18_clauses.CollectTwoClauses`).
- A2: a merged arm's sources lie under the sources root (the output root, or `--sources-root`), are read without
  following links (no FIFO hang), and a mismatch never prints the source's hash.
- A3: a linked `eval_meta.json` is refused without its content in the message; a FIFO is never opened.
- A4: no file that is not a regular file is opened for reading (`O_PATH` first).
- A5: output paths are canonical (one spelling per file); `merge.json` records the hashes of the bytes merged.
- A6: a NUL or a lone surrogate in an output path is a problem, not a traceback.
- B1: a directory the user may only search (mode 0o111) is walked.
- C3: `GATEDTRAIN_REQUIRE_CONTAINER_USER=1` turns the launch fixtures' skip into a failure.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _ext import SEEDS, ExtScenario, build_arm, candidate_score, merge
from _support import EXIT_REFUSED, describe, parse_errors, read_jsonl

from gatedtrain.errors import Refused
from gatedtrain.ext import arm as arm_mod
from gatedtrain.ext import armfs

WATCHDOG_S = 20


def sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def write_lines(arm: Path, lines: list[dict]) -> None:
    (arm / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")


class Watchdog(unittest.TestCase):
    """A read that blocks (a FIFO opened for reading) fails the test instead of hanging the suite."""

    def setUp(self):
        def expired(signum, frame):
            raise AssertionError(f"blocked for {WATCHDOG_S} s (a FIFO opened for reading?)")

        old = signal.signal(signal.SIGALRM, expired)
        self.addCleanup(signal.signal, signal.SIGALRM, old)
        signal.alarm(WATCHDOG_S)
        self.addCleanup(signal.alarm, 0)
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-arm-reads-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.secret = self.tmp / "SECRET"
        self.secret.write_text(json.dumps({"auths": {"registry": {"auth": "c2VjcmV0"}}, "schema": "TOPSECRET"}),
                               encoding="utf-8")


class ScenarioCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-arm-reads-c-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = ExtScenario(cls.tmp / "sc")
        cls.n = 0
        cls.secret = cls.tmp / "HOST_SECRET"
        cls.secret.write_bytes(b"host secret\n")

    def fresh(self, name: str) -> Path:
        type(self).n += 1
        return self.sc.arm(f"{name}-{type(self).n}", candidate_score, root=self.tmp / "arms")

    def with_output(self, arm: Path, rel: str = "outputs/a.txt", data: bytes = b"output") -> Path:
        lines = read_jsonl(arm / "items.jsonl")
        (arm / rel).parent.mkdir(parents=True, exist_ok=True)
        (arm / rel).write_bytes(data)
        lines[0]["output_ref"] = {"path": rel, "sha256": sha(data)}
        write_lines(arm, lines)
        return arm

    def collect(self, candidate: Path, hook=None, **kw):
        """ext-collect in this process; `hook()` runs after every check passed, before the arms are copied."""
        from gatedtrain import runner
        from gatedtrain.ext.collect import collect

        real = runner.copy_registration

        def hooked(wf, run_dir):
            if hook is not None:
                hook()
            return real(wf, run_dir)

        type(self).n += 1
        runs = self.tmp / f"runs-{type(self).n}"
        sc = self.sc
        with mock.patch.dict(os.environ, sc.env()), mock.patch.object(runner, "copy_registration", hooked):
            return runs, collect(sc.paths["spec"], adapter="synthetic", stage_out=sc.stage_out,
                                 launch_record=sc.launch, smoke_record=sc.smoke, eval_baseline=sc.baseline,
                                 eval_candidate=candidate, runs_root=runs, **kw)


class TestCollectCopiesCheckedBytes(ScenarioCase):
    """A1, C2 (one clause; the copy sites of `_write_run`)."""

    def test_arm_files_swapped_after_the_checks(self):
        cand = self.fresh("swap")
        before = {n: (cand / n).read_bytes() for n in ("items.jsonl", "eval_meta.json")}

        def swap():
            (cand / "items.jsonl").unlink()
            (cand / "items.jsonl").symlink_to(self.secret)
            (cand / "eval_meta.json").write_bytes(b"{}")
            (cand / "merge.json").symlink_to(self.secret)

        _, (run_dir, report, _) = self.collect(cand, swap)
        got = run_dir / "eval" / "candidate"
        self.assertEqual({n: (got / n).read_bytes() for n in before}, before)
        self.assertFalse((got / "merge.json").exists())
        self.assertFalse(any(p.is_symlink() for p in got.rglob("*")))

    def test_output_swapped_for_a_link_refused(self):
        cand = self.with_output(self.fresh("out-link"))

        def swap():
            (cand / "outputs" / "a.txt").unlink()
            (cand / "outputs" / "a.txt").symlink_to(self.secret)

        with self.assertRaises(Refused) as cm:
            runs, _ = self.collect(cand, swap)
        (p,) = cm.exception.problems
        self.assertEqual((p.rule, p.field), ("EXT-ARM", "--eval-candidate"))
        self.assertIn("output_ref path 'outputs/a.txt' contains a symbolic link", p.message)
        self.assertFalse(any((self.tmp / f"runs-{type(self).n}").glob("run-*")))

    def test_output_rewritten_refused(self):
        cand = self.with_output(self.fresh("out-edit"))
        with self.assertRaises(Refused) as cm:
            self.collect(cand, lambda: (cand / "outputs" / "a.txt").write_bytes(b"other"))
        self.assertIn("differs from its sha256 (changed after it was checked)", cm.exception.problems[0].message)

    def test_unreadable_output_is_an_arm_problem(self):
        """B3: a read error of an output is the arm's (EXT-ARM), not a failed write of the run (`--runs-root`)."""
        cand = self.with_output(self.fresh("out-eio"))
        real = os.read

        def eio(fd, n):
            if stat.S_ISREG(os.fstat(fd).st_mode) and os.fstat(fd).st_size == len(b"output"):
                raise OSError(5, "Input/output error")
            return real(fd, n)

        self.addCleanup(mock.patch.stopall)
        with self.assertRaises(Refused) as cm:
            self.collect(cand, lambda: mock.patch.object(arm_mod.os, "read", eio).start())
        (p,) = cm.exception.problems
        self.assertEqual((p.rule, p.field), ("EXT-ARM", "--eval-candidate"))
        self.assertIn("cannot be opened (EIO)", p.message)


class TestProvenanceSources(ScenarioCase):
    """A2: merge.json sources lie under the sources root and are read without following links."""

    def merged(self, name: str) -> tuple[Path, list[Path]]:
        type(self).n += 1
        d = self.tmp / f"{name}-{type(self).n}"
        srcs = [build_arm(d / f"s{s}", self.sc.items, candidate_score, model=self.sc.cand_model,
                          item_set=self.sc.item_set_hash, seeds=[s]) for s in SEEDS]
        out = d / "merged"
        proc = merge(self.sc.paths["registration"], out, srcs, cwd=self.sc.root)
        self.assertEqual(proc.returncode, 0, describe(proc))
        return out, srcs

    def problems(self, arm: Path, root: Path | None) -> list[str]:
        meta, why = arm_mod.read_meta(arm)
        self.assertEqual(why, [])
        return arm_mod.provenance_problems(arm, meta, nel_benchmark=None, sources_root=root)

    def test_sources_under_the_root(self):
        out, _ = self.merged("ok")
        self.assertEqual(self.problems(out, self.tmp), [])

    def test_source_outside_the_root_refused_without_its_hash(self):
        out, srcs = self.merged("outside")
        got = self.problems(out, self.tmp / "elsewhere")
        self.assertTrue(got and all("provenance cannot be checked" in m for m in got), got)
        self.assertFalse(any("sha256:" in m for m in got), got)
        self.assertEqual(self.problems(out, None),
                         ["merge.json names sources but no sources root is configured; provenance cannot be checked"])

    def test_mismatch_says_only_that_it_differs(self):
        out, srcs = self.merged("differs")
        (srcs[0] / "items.jsonl").write_bytes((srcs[0] / "items.jsonl").read_bytes() + b"\n")
        got = self.problems(out, self.tmp)
        self.assertIn("merge.json source sources[0] items.jsonl differs from the recorded hash", got)
        self.assertFalse(any("sha256:" in m for m in got), got)

    def test_fifo_and_linked_sources_refused(self):
        def expired(signum, frame):
            raise AssertionError("a provenance source blocked")

        old = signal.signal(signal.SIGALRM, expired)
        self.addCleanup(signal.signal, signal.SIGALRM, old)
        signal.alarm(WATCHDOG_S)
        self.addCleanup(signal.alarm, 0)
        out, srcs = self.merged("fifo")
        (srcs[0] / "items.jsonl").unlink()
        os.mkfifo(srcs[0] / "items.jsonl")
        (srcs[1] / "eval_meta.json").unlink()
        (srcs[1] / "eval_meta.json").symlink_to(self.secret)
        got = self.problems(out, self.tmp)
        self.assertIn("merge.json source sources[0] items.jsonl is not a regular file; provenance cannot be checked",
                      got)
        self.assertIn("merge.json source sources[1] eval_meta.json contains a symbolic link ('eval_meta.json'; MM-1 "
                      "allows none); provenance cannot be checked", got)

    def test_source_through_a_link_refused(self):
        out, srcs = self.merged("dirlink")
        doc = json.loads((out / "merge.json").read_text(encoding="utf-8"))
        moved = srcs[0].parent / "moved"
        srcs[0].rename(moved)
        srcs[0].symlink_to(moved)
        got = self.problems(out, self.tmp)
        self.assertTrue(any(f"sources[0] path {doc['sources'][0]['path']!r} contains a symbolic link" in m
                            for m in got), got)

    def test_collect_defaults_to_the_output_root(self):
        """Without `--sources-root`, sources must lie under the host's output root (the fixture's `<root>/out`)."""
        out, _ = self.merged("default")
        proc = self.sc.collect(self.tmp / f"runs-cli-{type(self).n}", candidate=out)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        msgs = [e["message"] for e in parse_errors(proc.stderr) if e["rule"] == "EXT-ARM"]
        self.assertTrue(any("under the sources root" in m for m in msgs), msgs)
        self.assertNotIn("Traceback", proc.stderr)


class TestArmFiles(Watchdog):
    """A3, A4, B1."""

    def arm(self) -> Path:
        arm = self.tmp / "arm"
        arm.mkdir()
        return arm

    def test_linked_eval_meta_refused_without_its_content(self):
        arm = self.arm()
        (arm / "eval_meta.json").symlink_to(self.secret)
        meta, why = arm_mod.read_meta(arm)
        self.assertIsNone(meta)
        self.assertEqual(why, [f"{arm}/eval_meta.json contains a symbolic link ('eval_meta.json'; MM-1 allows none)"])
        self.assertFalse(any(x in why[0] for x in ("auths", "TOPSECRET", "registry")))

    def test_fifo_arm_files_never_block(self):
        arm = self.arm()
        for name in ("eval_meta.json", "items.jsonl", "merge.json"):
            os.mkfifo(arm / name)
        self.assertEqual(arm_mod.read_meta(arm)[1], [f"{arm}/eval_meta.json is not a regular file"])
        self.assertEqual(arm_mod.read_lines(arm, seeds=[1], score_range=[0, 1], error_classes=())[1],
                         [f"{arm}/items.jsonl is not a regular file"])
        self.assertEqual(arm_mod.provenance_problems(arm, {"eval_seeds": [1]}, nel_benchmark=None),
                         ["merge.json is not a regular file; provenance cannot be checked"])

    def test_swap_to_fifo_between_inspection_and_open(self):
        """A file inspected as regular and then replaced by a FIFO is never opened for reading."""
        arm = self.arm()
        (arm / "eval_meta.json").write_text("{}", encoding="utf-8")
        real = os.fstat
        swapped = []

        def fstat_then_swap(fd):
            st = real(fd)
            if not swapped and stat.S_ISREG(st.st_mode):
                swapped.append(True)
                (arm / "eval_meta.json").unlink()
                os.mkfifo(arm / "eval_meta.json")
            return st

        with mock.patch.object(armfs.os, "fstat", fstat_then_swap):
            snap = armfs.ArmSnapshot.read(arm)
        self.assertTrue(swapped)
        # O_PATH holds the inode inspected; the reopened file is that inode or the read is refused, never the FIFO
        self.assertIn(snap.files["eval_meta.json"].data, (b"{}", None))

    def test_nothing_but_regular_files_opened_for_reading(self):
        """A4: every open without O_PATH that returns a descriptor is of a directory or a regular file."""
        if not armfs._PROC_FD:
            self.skipTest("no O_PATH and /proc/self/fd here")
        arm = self.arm()
        os.mkfifo(arm / "eval_meta.json")
        os.mkfifo(arm / "fifo")
        (arm / "items.jsonl").write_text(json.dumps({"item": "a", "repeat": 0, "seed": 1, "score": 1, "error": None,
                                                     "output_ref": {"path": "fifo", "sha256": sha(b"")}}) + "\n",
                                         encoding="utf-8")
        real = os.open
        bad = []

        def watched(path, flags, *a, **k):
            fd = real(path, flags, *a, **k)
            if not flags & os.O_PATH:
                mode = os.fstat(fd).st_mode
                if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                    bad.append(path)
            return fd

        with mock.patch.object(armfs.os, "open", watched):
            arm_mod.read_meta(arm)
            _, why = arm_mod.read_lines(arm, seeds=[1], score_range=[0, 1], error_classes=())
        self.assertEqual(bad, [])
        self.assertEqual(why, ["line 1 output_ref file 'fifo' is not a regular file"])

    @unittest.skipUnless(os.geteuid() == 0, "creating a device node needs root")
    def test_device_node_not_opened(self):
        arm = self.arm()
        os.mknod(arm / "tty", stat.S_IFCHR | 0o600, os.makedev(5, 0))
        self.assertEqual(arm_mod._output_ref_problem(arm, {"path": "tty", "sha256": sha(b"")}),
                         "output_ref file 'tty' is not a regular file")

    def test_search_only_directory_is_walked(self):
        """B1: a directory with mode 0o111 (search, no read) on the way to an output is fine."""
        arm = self.arm()
        (arm / "x").mkdir()
        (arm / "x" / "f").write_bytes(b"ok")
        (arm / "x").chmod(0o111)
        self.addCleanup((arm / "x").chmod, 0o755)
        self.assertIsNone(arm_mod._output_ref_problem(arm, {"path": "x/f", "sha256": sha(b"ok")}))


class TestPathStrings(ScenarioCase):
    """A5, A6."""

    def test_nul_and_surrogates_are_problems(self):
        for name, path, why in (("nul", "outputs/a\x00b", "contains a NUL character"),
                                ("surrogate", "outputs/\ud800", "is not valid UTF-8"),
                                ("dot", "outputs/./a.txt", "is not in canonical form"),
                                ("empty", "outputs//a.txt", "is not in canonical form")):
            with self.subTest(case=name):
                cand = self.fresh(name)
                lines = read_jsonl(cand / "items.jsonl")
                lines[0]["output_ref"] = {"path": path, "sha256": sha(b"x")}
                write_lines(cand, lines)
                _, bad = arm_mod.read_lines(cand, seeds=list(SEEDS), score_range=[0, 1], error_classes=())
                self.assertTrue(bad and why in bad[0], bad)
                proc = merge(self.sc.paths["registration"], self.tmp / f"merged-{name}", [cand], cwd=self.sc.root)
                self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
                self.assertNotIn("Traceback", proc.stderr)
                self.assertIn(why, proc.stderr)

    def test_merge_records_the_hashes_of_the_bytes_merged(self):
        """A5: merge.json's source hashes are those of the bytes merged, even if a source changes meanwhile."""
        from gatedtrain.ext import merge as merge_mod

        type(self).n += 1
        d = self.tmp / f"hashes-{type(self).n}"
        srcs = [build_arm(d / f"s{s}", self.sc.items, candidate_score, model=self.sc.cand_model,
                          item_set=self.sc.item_set_hash, seeds=[s]) for s in SEEDS]
        before = [(s / "items.jsonl").read_bytes() for s in srcs]
        real = merge_mod.read_lines

        def read_then_change(run, **kw):
            out = real(run, **kw)
            (Path(run) / "items.jsonl").write_bytes(b"changed\n")
            return out

        with mock.patch.object(merge_mod, "read_lines", read_then_change):
            out = merge_mod.merge_eval(self.sc.paths["registration"], d / "merged", srcs)
        doc = json.loads((out / "merge.json").read_text(encoding="utf-8"))
        self.assertEqual([s["items_sha256"] for s in doc["sources"]], [sha(b) for b in before])


class TestRequireContainerUser(unittest.TestCase):
    """C3: in CI the launch fixtures fail instead of skipping when the running user cannot launch a container."""

    def test_switch(self):
        import _ext

        with mock.patch.object(_ext.os, "getuid", lambda: 0):
            with mock.patch.dict(os.environ, {_ext.REQUIRE_CONTAINER_USER: "0"}):
                self.assertRaises(unittest.SkipTest, _ext.container_user)
            with mock.patch.dict(os.environ, {_ext.REQUIRE_CONTAINER_USER: "1"}):
                with self.assertRaises(AssertionError) as cm:
                    _ext.container_user()
        self.assertIn("root", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
