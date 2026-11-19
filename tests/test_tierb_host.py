"""F-27: the Tier B host path runs on Linux without torch, from `requirements-tierb-host.txt`
(maintainers' decision; tests/README.md section 11.13).

`pyproject.toml` restricts the uv lock to macOS (`[tool.uv] environments`, interim rule G-11), so
`uv sync --locked` cannot build the environment on a Linux host. The host instead runs
    uv venv --python 3.13 <dir>
    uv pip install --python <dir>/bin/python -r requirements-tierb-host.txt
with exact pins equal to uv.lock, and no torch.

How torch is blocked: a `sys.meta_path` finder raises ModuleNotFoundError for `torch` and
`torch.*`, which is what a host without torch does. Setting `sys.modules['torch'] = None` is not
equivalent: scipy's array-API helper then finds a `torch` key and fails on `None.Tensor` at
import time (observed with scipy 1.18.1), a failure no real torch-free host would see.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

from _support import (DEMO_DIR, DOCS_DIR, EXIT_OK, PKG_DIR, SKEL_ROOT, VENV_PYTHON, cli_env, describe, manifest_hash,
                      parse_run_dir)
from _tierb import (SOURCE, baseline_reward, build_eval_run, build_stage_out, candidate_reward, heldout_rows,
                    registration_seeds, render_inputs)

REQUIREMENTS = SKEL_ROOT / "requirements-tierb-host.txt"
REQUIRED = {"numpy", "scipy", "pyarrow", "duckdb", "pyyaml"}
# import name -> distribution name, for the third-party modules gatedtrain imports
DIST_OF = {"yaml": "pyyaml"}

BLOCK = '''
import importlib.abc, runpy, sys
class _NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named %r (blocked: Tier B host has no torch)" % name, name=name)
        return None
sys.meta_path.insert(0, _NoTorch())
'''
LAUNCH = BLOCK + '''
sys.argv = ["gatedtrain"] + sys.argv[1:]
runpy.run_module("gatedtrain", run_name="__main__", alter_sys=True)
'''


def no_torch(args: list[str], *, cwd: Path, timeout: int = 300) -> subprocess.CompletedProcess:
    """`gatedtrain <args>` with torch unavailable."""
    return subprocess.run([str(VENV_PYTHON), "-c", LAUNCH, *args], cwd=str(cwd), env=cli_env(),
                          capture_output=True, text=True, timeout=timeout)


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()       # PEP 503


def lock_versions() -> dict[str, dict]:
    with (SKEL_ROOT / "uv.lock").open("rb") as f:
        return {norm(p["name"]): p for p in tomllib.load(f)["package"]}


def read_requirements(test: unittest.TestCase) -> dict[str, str]:
    test.assertTrue(REQUIREMENTS.is_file(), REQUIREMENTS)
    pins = {}
    hashes: dict[str, int] = {}
    text = REQUIREMENTS.read_text(encoding="utf-8").replace("\\\n", " ")
    for raw in text.splitlines():
        if raw.lstrip().startswith("#"):
            continue
        line = raw.strip()
        if not line:
            continue
        m = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9._+!-]+)((?:\s+--hash=sha256:[0-9a-f]{64})*)", line)
        test.assertIsNotNone(m, f"not an exact `name==version` pin with `--hash=sha256:` entries: {raw!r}")
        test.assertNotIn(norm(m.group(1)), pins, f"duplicate pin {m.group(1)}")
        pins[norm(m.group(1))] = m.group(2)
        hashes[norm(m.group(1))] = m.group(3).count("--hash=")
    for name, n in hashes.items():
        test.assertGreater(n, 0, f"{name} has no --hash (the host installs with --require-hashes)")
    return pins


class TestTorchBlockIsEffective(unittest.TestCase):
    """Guards the harness itself: under the launcher, `import torch` fails, scipy still imports."""

    def test_block(self):
        code = BLOCK + "import scipy.stats\ntry:\n    import torch\nexcept ModuleNotFoundError:\n    print('blocked')\n"
        proc = subprocess.run([str(VENV_PYTHON), "-c", code], capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("blocked", proc.stdout)


class TestTierBHostPathWithoutTorch(unittest.TestCase):
    """Every host-side Tier B command exits 0 with torch blocked (F-27 item 1)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-host-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)

    def ok(self, args, cwd=None):
        proc = no_torch(args, cwd=cwd or self.tmp)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertNotIn("Traceback", proc.stderr)
        return proc

    def test_validate_demo_tierb_spec(self):
        self.ok(["validate", "spec.yaml"], cwd=DEMO_DIR / "tierb")

    def test_end_to_end_host_path(self):
        root = self.tmp / "e2e"
        root.mkdir()
        export = root / "export"
        self.ok(["export-gsm8k", "--out", str(export), "--source", str(SOURCE), "--rows-per-shard", "10",
                 "--revision", "0123456789abcdef0123456789abcdef01234567"])
        self.ok(["verify-export", str(export)])
        paths = render_inputs(root, export)
        self.ok(["nel-config", "--registration", str(paths["registration"]), "--model-url",
                 "http://localhost:8000/v1", "--model-id", "baseline", "--eval-root", "/work/out/eval/baseline",
                 "--out", str(root / "nel")])
        rows = heldout_rows(export)
        seeds = registration_seeds(paths["registration"])
        arms = {}
        for arm, fn in (("baseline", baseline_reward), ("candidate", candidate_reward)):
            runs = [build_eval_run(root / "raw" / arm / f"rep{r}", rows, reward_fn=fn, seed=s, model=arm)
                    for r, s in enumerate(seeds)]
            arms[arm] = root / "eval" / arm
            self.ok(["merge-eval", "--registration", str(paths["registration"]), "--out", str(arms[arm]),
                     *map(str, runs)])
        stage = build_stage_out(root / "out" / "rl", prompts_per_step=2, gens=4, steps=10, save_period=5)
        proc = self.ok(["tierb-collect", str(paths["spec"]), "--stage-out", str(stage),
                        "--eval-baseline", str(arms["baseline"]), "--eval-candidate", str(arms["candidate"]),
                        "--image-id", "sha256:" + "ab" * 32, "--nemo-rl-commit", "4f39fd8",
                        "--runs-root", str(root / "runs")])
        run_dir = parse_run_dir(proc.stdout)
        imp = self.ok(["impact", str(export / "train" / "shard-00000.jsonl"), "--run-dir", str(run_dir), "--json"])
        got = {r["checkpoint_hash"] for r in json.loads(imp.stdout)["results"]}
        self.assertEqual(got, {manifest_hash(stage / "checkpoints" / f"step_{s}") for s in (5, 10)})


class TestHostRequirementsPinned(unittest.TestCase):
    """F-27 item 2: exact pins equal to uv.lock, no torch."""

    def test_pins_match_lock(self):
        pins = read_requirements(self)
        lock = lock_versions()
        self.assertLessEqual(REQUIRED, set(pins), "numpy, scipy, pyarrow, duckdb, pyyaml")
        self.assertNotIn("torch", pins)
        for name, ver in pins.items():
            with self.subTest(package=name):
                self.assertIn(name, lock, f"{name} is not in uv.lock")
                self.assertEqual(ver, lock[name]["version"], f"{name}=={ver} but uv.lock has {lock[name]['version']}")

    def test_lock_dependency_closure_pinned(self):
        """Every uv.lock dependency of a pinned package is pinned too (scipy -> numpy), so the host
        resolves the same versions as the lock."""
        pins = read_requirements(self)
        lock = lock_versions()
        todo, seen = list(pins), set()
        while todo:
            name = todo.pop()
            if name in seen or name not in lock:
                continue
            seen.add(name)
            for dep in lock[name].get("dependencies", []):
                dn = norm(dep["name"])
                with self.subTest(package=name, dependency=dn):
                    self.assertIn(dn, pins, f"{name} depends on {dn} in uv.lock")
                todo.append(dn)


def third_party_imports() -> dict[str, list[str]]:
    """Top-level third-party modules imported anywhere in gatedtrain/ except stubs/, statically or
    lazily, minus the lazy imports inside runner.py's Tier A entry points (`run`, `preflight`)."""
    std = set(sys.stdlib_module_names)
    found: dict[str, list[str]] = {}
    for path in sorted(PKG_DIR.rglob("*.py")):
        rel = path.relative_to(PKG_DIR).as_posix()
        if rel.startswith("stubs/"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tier_a_only: set[int] = set()
        if rel == "runner.py":
            for fn in tree.body:
                if isinstance(fn, ast.FunctionDef) and fn.name in ("run", "preflight"):
                    tier_a_only |= {id(n) for n in ast.walk(fn)}
        for node in ast.walk(tree):
            if id(node) in tier_a_only:
                continue
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
            for m in mods:
                top = m.split(".")[0]
                if top not in std and top != "gatedtrain":
                    found.setdefault(top, []).append(f"{rel}:{node.lineno}")
    return found


class TestHostRequirementsComplete(unittest.TestCase):
    """F-27 item 3: every third-party module the Tier B host path imports is pinned."""

    def test_scan_sees_the_known_imports(self):
        """Guards the scanner (passes without the requirements file)."""
        found = third_party_imports()
        self.assertLessEqual({"yaml", "numpy", "scipy", "duckdb", "pyarrow"}, set(found), found)
        self.assertNotIn("torch", found, "torch is imported outside stubs/ and outside runner.run/preflight: "
                                          f"{found.get('torch')}")

    def test_every_import_pinned(self):
        pins = read_requirements(self)
        for module, where in sorted(third_party_imports().items()):
            with self.subTest(module=module):
                self.assertIn(norm(DIST_OF.get(module, module)), pins, f"{module} imported at {where}")


class TestRunbookHostInstall(unittest.TestCase):
    """F-27 item 4: a Linux host install subsection with the two uv commands; `gatedtrain` points to
    that venv; no `uv sync --locked` for the host."""

    def test_runbook(self):
        text = (DOCS_DIR / "tier-b-runbook.md").read_text(encoding="utf-8")
        self.assertRegex(text, r"(?im)^#+ .*linux.*host.*install|^#+ .*host.*install.*linux|^#+ .*install.*linux",
                         "a Linux host install subsection")
        self.assertRegex(text, r"uv venv --python 3\.13 \S+")
        self.assertRegex(text, r"uv pip install --python \S+/bin/python -r \S*requirements-tierb-host\.txt")
        venv = re.search(r"uv venv --python 3\.13 (\S+)", text).group(1)
        self.assertRegex(text, re.escape(venv) + r"/bin/python -m gatedtrain",
                         "the gatedtrain notation runs the host venv's python")
        for line in text.splitlines():
            if "uv sync" in line and "--locked" in line:
                self.assertRegex(line, r"(?i)tier a|macos",
                                 f"`uv sync --locked` is for the macOS Tier A lock only, never the host: {line!r}")


if __name__ == "__main__":
    unittest.main()
