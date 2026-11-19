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
import sys
import tempfile
import unittest
from pathlib import Path

from gatedtrain.paths import realpath, resolve


class TestRealpath(unittest.TestCase):
    def setUp(self):
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
        self.assertGreater(compared, 1500)

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
