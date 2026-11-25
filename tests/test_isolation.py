"""Test isolation: the suite runs the code of this checkout under the interpreter that runs it, and nothing in it finds a
Python environment by looking around (tests/README.md section 15, ENGINEERING-STANDARDS rules 8 and 12).

A keeper test is a trust test: a stale or foreign environment on the import path could change its result. The tests
start their subprocesses (keeper, CLI) with `sys.executable`, so the environment they exercise is the one that started
the suite. These tests pin that and fail when the suite was started from an environment that is not its own:

- the `gatedtrain` under test is this checkout's `src/`, not an installed copy (a site-packages copy of another version
  would be tested instead);
- no directory of the import path or the interpreter's prefix lies in the temporary directory unless the checkout lies
  there too (a virtual environment someone left under `/tmp` is not the checkout's);
- no test or fixture file locates an interpreter or a virtual environment by scanning (`glob`/`rglob`/`which`
  over a temporary directory, or a hard-coded keeper venv path).
"""

from __future__ import annotations

import re
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path

import gatedtrain
from _support import FIXTURES, PKG_DIR, SKEL_ROOT, TESTS_DIR

# A path under the temporary directory is foreign to a checkout that is not itself there (see the module docstring).
_TEMP = Path(tempfile.gettempdir()).resolve()

# Patterns that would make a test find an environment by looking around instead of using `sys.executable`.
_SCANNING = (
    re.compile(r"(?:glob|rglob|iterdir|scandir|listdir|walk)\([^)]*(?:gettempdir|/tmp|TMPDIR)"),
    re.compile(r"which\(\s*[\"']python"),
    re.compile(r"[\"']/opt/gatedtrain-keeper/venv"),
    re.compile(r"site-packages[\"'][^\n]*(?:glob|rglob)"),
)


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except (OSError, ValueError):
        return False


class EnvironmentUnderTestTest(unittest.TestCase):
    def test_gatedtrain_is_this_checkouts_source(self):
        self.assertTrue(_under(Path(gatedtrain.__file__), PKG_DIR.resolve()),
                        f"gatedtrain was imported from {gatedtrain.__file__}, not from {PKG_DIR}: install this "
                        "checkout (pip install -e .) in the environment that runs the suite")

    def test_the_import_path_has_no_foreign_temporary_environment(self):
        if _under(SKEL_ROOT, _TEMP):
            self.skipTest("the checkout itself lies under the temporary directory")
        paths = [Path(p) for p in sys.path if p] + [Path(sys.prefix), Path(sys.exec_prefix),
                                                    Path(sysconfig.get_paths()["purelib"])]
        foreign = sorted({str(p) for p in paths if _under(p, _TEMP)})
        self.assertEqual(foreign, [], "the suite was started from an environment under the temporary directory, "
                                     "which is not this checkout's own; use the checkout's virtual environment")


class NoEnvironmentDiscoveryTest(unittest.TestCase):
    def test_no_scanning_for_an_interpreter_or_environment(self):
        offenders = []
        files = [p for root in (TESTS_DIR, FIXTURES) for p in root.rglob("*.py")
                 if p.resolve() != Path(__file__).resolve()]
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in _SCANNING:
                if pattern.search(text):
                    offenders.append(f"{path.relative_to(SKEL_ROOT)}: {pattern.pattern}")
        self.assertEqual(offenders, [], "a test file finds an environment by scanning; start subprocesses with "
                                        "sys.executable instead")

    def test_the_scan_patterns_catch_what_they_name(self):
        for bad in ("root.glob(f'{tempfile.gettempdir()}/*/venv')", "shutil.which('python3')",
                    "Path('/opt/gatedtrain-keeper/venv/bin/python')"):
            self.assertTrue(any(p.search(bad) for p in _SCANNING), bad)
        self.assertFalse(any(p.search("subprocess.run([sys.executable, 'x'])") for p in _SCANNING))


if __name__ == "__main__":
    unittest.main()
