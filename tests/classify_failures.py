"""Run the test suite and group failures by cause.

Run from the repository root with
    .venv/bin/python tests/classify_failures.py [--json OUT]

Missing modules and files get their own categories. Other failures are listed
with their tracebacks.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parent
SKEL = TESTS.parent

MISSING_MODULE = ("No module named 'gatedtrain'", "No module named gatedtrain")
MISSING_FILES = (str(SKEL / "src" / "gatedtrain"), str(SKEL / "docs"))


def classify(text: str) -> str:
    if any(m in text for m in MISSING_MODULE):
        return "MISSING_MODULE"
    if "is not true" in text and any(p in text for p in MISSING_FILES):
        return "MISSING_FILE"
    return "OTHER"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    sys.path.insert(0, str(TESTS))
    sys.path.insert(1, str(SKEL))   # `import gatedtrain` from the source tree, as with `python -m unittest` from SKEL
    suite = unittest.defaultTestLoader.discover(str(TESTS), pattern="test_*.py", top_level_dir=str(TESTS))
    discovered = suite.countTestCases()
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    start = time.monotonic()
    result = runner.run(suite)
    elapsed = time.monotonic() - start

    rows = []
    for kind, items in (("FAIL", result.failures), ("ERROR", result.errors)):
        for test, tb in items:
            rows.append({"test": test.id() if hasattr(test, "id") else str(test), "kind": kind,
                         "cause": classify(tb), "last_line": tb.strip().splitlines()[-1][:300], "traceback": tb})
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["cause"]] = counts.get(r["cause"], 0) + 1

    # Subtest failures and class-level errors are reported per item; count
    # test methods once. A setUpClass error stands for every test of its class.
    failing_methods = {r["test"].split(" ")[0] for r in rows if not r["test"].startswith("setUpClass")}
    class_errors = [r["test"] for r in rows if r["test"].startswith("setUpClass")]
    passed = result.testsRun - len(failing_methods) - len(result.skipped)
    print(f"discovered {discovered} test methods; ran {result.testsRun} in {elapsed:.1f}s; passed {passed}; "
          f"failing methods {len(failing_methods)}; "
          f"class-level errors {len(class_errors)}; skipped {len(result.skipped)}; "
          f"items reported {len(rows)} (failures {len(result.failures)}, errors {len(result.errors)})")
    print("causes:", json.dumps(counts, sort_keys=True))
    for r in rows:
        print(f"  {r['kind']:5} {r['cause']:14} {r['test']}  ::  {r['last_line']}")
    for r in rows:
        if r["cause"] == "OTHER":
            print("\n--- OTHER:", r["test"], "\n", r["traceback"])
    if args.json:
        args.json.write_text(json.dumps({"elapsed_s": elapsed, "tests_run": result.testsRun, "rows": rows},
                                        indent=2), encoding="utf-8")
    return 0 if counts.get("OTHER", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
