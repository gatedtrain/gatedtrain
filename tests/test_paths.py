"""Lenient path resolution (`gatedtrain.paths`; docs/findings.md, "Python 3.12: lenient `realpath`").

`realpath` must return what Python 3.13's `os.path.realpath` returns, on every supported Python, and must not
raise where Python 3.12's does: a symbolic link the user may not read (`/proc/1/root` of another user). The tree
below holds link chains, a loop, a dangling link, links through `..` and absolute links; the random paths combine
them with `.`, `..`, empty and missing components.
"""

from __future__ import annotations

import os
import random
import shutil
import signal
import sys
import tempfile
import unittest
from pathlib import Path

from gatedtrain.paths import realpath, resolve


WATCHDOG_S = 30                                   # a resolution that never ends (an undetected loop) fails, never hangs


class TestRealpath(unittest.TestCase):
    def setUp(self):
        self.expired = False

        def expired(signum, frame):                # AssertionError: realpath catches OSError (TimeoutError is one)
            self.expired = True
            raise AssertionError(f"the test did not end within {WATCHDOG_S} s (a symbolic link loop not detected?)")

        old = signal.signal(signal.SIGALRM, expired)
        self.addCleanup(signal.signal, signal.SIGALRM, old)
        signal.alarm(WATCHDOG_S)
        self.addCleanup(signal.alarm, 0)
        self.tmp = tempfile.mkdtemp(prefix="gatedtrain-paths-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        root = self.tmp
        for d in ("a/sub", "b/sub", "c"):
            os.makedirs(os.path.join(root, d))
        links = {"to_a": "a", "to_bsub": "b/sub", "up": "..", "chain": "to_a", "loop1": "loop2", "loop2": "loop1",
                 "dangling": "missing/x", "abs_c": os.path.join(root, "c"), "a/back": "../b", "a/self": "."}
        for name, target in links.items():
            os.symlink(target, os.path.join(root, name))
        self.parts = ["a", "b", "c", "sub", "to_a", "to_bsub", "up", "chain", "loop1", "dangling", "abs_c", "back",
                      "self", ".", "..", "", "missing"]
        cwd = os.getcwd()
        os.chdir(root)
        self.addCleanup(os.chdir, cwd)

    def cases(self):
        rng = random.Random(11)
        out = [self.tmp, "", ".", "relative/x", os.path.join(self.tmp, "loop1", "x")]
        for _ in range(2000):
            p = "/".join(rng.choice(self.parts) for _ in range(rng.randint(1, 6)))
            out.append(os.path.join(self.tmp, p) if rng.random() < 0.9 else p)
        return out

    def test_same_as_the_standard_library(self):
        """Python 3.13's `realpath`; Python 3.12's agrees except after a link loop, where 3.12 stops resolving and
        3.13 (and `gatedtrain.paths`) resolve the rest."""
        compared = 0
        for case in self.cases():
            if self.expired:
                break
            if sys.version_info < (3, 13) and "loop1" in case:
                continue
            try:
                want = os.path.realpath(case)
            except OSError:
                continue
            with self.subTest(case=case):
                self.assertEqual(realpath(case), want)
                self.assertEqual(resolve(case), Path(want))
            compared += 1
        self.assertFalse(self.expired)
        self.assertGreater(compared, 1500)

    def test_fixed_values(self):
        """Review of the 3.12 path fix (T1, T2): expected values that Python 3.12 and 3.13 both give, so they hold
        on every supported Python without the standard library as the oracle (the random comparison leaves the loop
        out on 3.12): a link loop is kept and what follows it resolved, `..` after the loop leaves it; below a
        regular file (`a/f/x`: ENOTDIR from `lstat`) the rest is kept, and `a/f/..` is `a`. The watchdog of `setUp`
        turns a resolution that never ends (a loop not detected) into a failure instead of a hang."""
        t = self.tmp
        Path(t, "a", "f").write_text("regular file", encoding="utf-8")
        cases = {f"{t}/loop1/x": f"{t}/loop1/x", f"{t}/loop1": f"{t}/loop1", f"{t}/loop2/x/y": f"{t}/loop2/x/y",
                 f"{t}/loop1/../a": f"{t}/a", f"{t}/a/f": f"{t}/a/f", f"{t}/a/f/x": f"{t}/a/f/x", f"{t}/a/f/x/y": f"{t}/a/f/x/y",
                 f"{t}/a/f/..": f"{t}/a", f"{t}/a/f/../sub": f"{t}/a/sub", f"{t}/to_a/f/x": f"{t}/a/f/x"}
        for case, want in cases.items():
            if self.expired:
                break
            with self.subTest(case=case):
                self.assertEqual(realpath(case), want)
                self.assertEqual(os.path.realpath(case), want, "the standard library agrees on this Python")

    @unittest.skipIf(os.access("/proc/1/root", os.R_OK) or not os.path.islink("/proc/1/root"),
                     "/proc/1/root is readable here (root) or absent")
    def test_unreadable_link_is_kept(self):
        for case, want in (("/proc/1/root", "/proc/1/root"), ("/proc/1/root/run", "/proc/1/root/run"),
                           ("//proc/1/root", "/proc/1/root"), ("/proc/1/cwd", "/proc/1/cwd")):
            with self.subTest(case=case):
                self.assertEqual(realpath(case), want)
                if sys.version_info >= (3, 13):
                    self.assertEqual(os.path.realpath(case), want)
        link = os.path.join(self.tmp, "to_init_root")
        os.symlink("/proc/1/root", link)
        self.assertEqual(realpath(os.path.join(link, "etc")), "/proc/1/root/etc")


if __name__ == "__main__":
    unittest.main()
