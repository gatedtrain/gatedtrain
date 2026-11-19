"""Shared helpers for the walking-skeleton test suite.

Everything here is independent of the implementation in ``gatedtrain``:
manifest hashes, canonical JSON and the run-directory parsing are written
from the format conventions and the interface in tests/README.md, not by
calling the code under test.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import unittest
from pathlib import Path
from typing import Any

SKEL_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = SKEL_ROOT / "tests"
FIXTURES = TESTS_DIR / "fixtures"
DEMO_DIR = SKEL_ROOT / "demo"
DOCS_DIR = SKEL_ROOT / "docs"
PKG_DIR = SKEL_ROOT / "src" / "gatedtrain"
# The interpreter that runs the suite also runs the CLI subprocesses (the repository `.venv` when the suite is run
# with `.venv/bin/python`), so the tests exercise one installation.
VENV_PYTHON = Path(sys.executable)
# The host layout (gatedtrain.ext.launch.host_config) the tests run against unless a test passes its own: a fixture host
# file with a neutral layout, since gatedtrain itself has no built-in host path. Set for in-process code and for the CLI
# subprocesses.
HOST_CONFIG = FIXTURES / "host.toml"
if os.environ.get("GATEDTRAIN_HOST_CONFIG") in (None, os.devnull):   # gatedtrain.testing sets the empty one
    os.environ["GATEDTRAIN_HOST_CONFIG"] = str(HOST_CONFIG)
HOST_WORK = "/srv/gatedtrain/tierb"          # = HOST_CONFIG host_work
HOST_ENV = {"GATEDTRAIN_HOST_CONFIG": str(HOST_CONFIG)}     # for a test that passes host_config its own environment
# gatedtrain's external-run tests use the synthetic test adapter (`gatedtrain.testing.synthetic`), which no installed
# distribution registers: in process it is installed here, and the CLI subprocesses run through its module.
CLI_MODULE = "gatedtrain.testing.synthetic"
from gatedtrain.testing import synthetic as _synthetic  # noqa: E402

_synthetic.install()

# Read-only design set and pinned clones (tests/README.md section 1). Each root is read only from its environment
# variable; there is no default. Tests that need an unset or missing root skip and name the variable or the path.
DESIGN_DIR_ENV = "GATEDTRAIN_DESIGN_DIR"
NEMO_EVALUATOR_DIR_ENV = "GATEDTRAIN_NEMO_EVALUATOR_DIR"
NEMO_RL_DIR_ENV = "GATEDTRAIN_NEMO_RL_DIR"
# Real records and data of a host that ran the recipes, read only: the Tier B host work directory (holding `bin/`,
# `tierb/`, `runs/`) and the multimodal recipes' runtime and output root. Tests that read them skip when unset.
TIERB_ROOT_ENV = "GATEDTRAIN_TEST_TIERB_ROOT"
MM_ROOT_ENV = "GATEDTRAIN_TEST_MM_ROOT"


def env_dir(var: str) -> Path:
    """The directory named by environment variable ``var``. When it is unset or empty, a placeholder path that never
    exists and names the variable (``<unset GATEDTRAIN_DESIGN_DIR>``), so `is_dir()` / `is_file()` checks fail and the
    skip message names the variable."""
    value = os.environ.get(var)
    return Path(value) if value else Path(f"<unset {var}>")


def require_env_dir(var: str) -> Path:
    """The directory named by ``var``; skips the calling test when the variable is unset or the directory is absent."""
    value = os.environ.get(var)
    if not value:
        raise unittest.SkipTest(f"{var} is not set")
    path = Path(value)
    if not path.is_dir():
        raise unittest.SkipTest(f"{var} names a missing directory: {path}")
    return path


DESIGN_DIR = env_dir(DESIGN_DIR_ENV)
NEMO_EVALUATOR_DIR = env_dir(NEMO_EVALUATOR_DIR_ENV)
NEMO_RL_DIR = env_dir(NEMO_RL_DIR_ENV)
TIERB_ROOT = env_dir(TIERB_ROOT_ENV)
MM_ROOT = env_dir(MM_ROOT_ENV)
NEMO_EVAL_COMPARISON = NEMO_EVALUATOR_DIR / "src" / "nemo_evaluator" / "engine" / "comparison.py"
NEMO_RL_GRPO_SYNC = NEMO_RL_DIR / "nemo_rl" / "algorithms" / "grpo_sync.py"

# Exit codes fixed in tests/README.md section 2.
EXIT_OK = 0
EXIT_GATE_NOT_PASS = 1
EXIT_USAGE = 2
EXIT_REFUSED = 3

STUB_MARK = "skeleton-stub"
STUB_MARKER_FILE = "SKELETON_STUB"
NOT_ASSESSED = "not-assessed (skeleton)"
REQUIRED_TAGS = ("prototype", "unpromotable", "registration: unsigned")

SLOW = os.environ.get("GATEDTRAIN_SKIP_SLOW", "") != "1"
slow = unittest.skipUnless(SLOW, "slow end-to-end test skipped (GATEDTRAIN_SKIP_SLOW=1)")

# Per-command time limit. The whole suite must finish within 15 minutes on
# CPU (MVP-SPEC section 1); about 7 full runs plus a warm-up are made.
RUN_TIMEOUT_S = 150

ERROR_LINE = re.compile(
    r"^error\[(?P<rule>[A-Z0-9-]+)\] (?P<field>\S+): (?P<message>.+?) \| fix: (?P<fix>\S.*)$"
)
RUN_DIR_LINE = re.compile(r"^run_dir: (?P<path>/.+)$", re.MULTILINE)
SHA256_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Canonical JSON and manifest hashes (canonical JSON and manifest hashes), independent copy.
# ---------------------------------------------------------------------------

def canonical_bytes(obj: Any) -> bytes:
    """RFC 8785 bytes for objects made of strings, integers, bools, null,
    lists and dicts with BMP keys (canonical-JSON rule C1-C3). Floats are not used by the
    callers; JCS float formatting differs from Python's repr in edge cases."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def manifest_entries(root: Path) -> list[dict[str, Any]]:
    """Entries {path, sha256, size} for every regular file under root,
    paths relative to root, '/'-joined, sorted by UTF-8 bytes (manifest rule H1-H6)."""
    root = Path(root)
    entries = []
    for p in root.rglob("*"):
        if p.is_symlink():
            raise AssertionError(f"symlink inside manifest root (manifest rule H2): {p}")
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        assert unicodedata.is_normalized("NFC", rel), rel
        data = p.read_bytes()
        entries.append({"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)})
    entries.sort(key=lambda e: e["path"].encode("utf-8"))
    return entries


def manifest_hash(root: Path) -> str:
    """P2 manifest_hash = "sha256:" + hex(SHA-256(JCS(sorted entries))) (manifest rule H7)."""
    return "sha256:" + hashlib.sha256(canonical_bytes(manifest_entries(root))).hexdigest()


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def cli_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SKEL_ROOT / "src")
    env.setdefault("PYTHONHASHSEED", "0")
    return env


def run_cli(args: list[str], *, cwd: Path, timeout: int = RUN_TIMEOUT_S,
            env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run the `gatedtrain` CLI with the project venv, the synthetic test adapter registered (``python -m
    gatedtrain.testing.synthetic <args>``); `env` adds variables (the external-run host layout, tests/README.md section
    11.18 EXT-63)."""
    return subprocess.run(
        [str(VENV_PYTHON), "-m", CLI_MODULE, *args],
        cwd=str(cwd),
        env={**cli_env(), **(env or {})},
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def describe(proc: subprocess.CompletedProcess) -> str:
    return (
        f"command {proc.args!r} exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout[-4000:]}\n--- stderr ---\n{proc.stderr[-4000:]}"
    )


def parse_run_dir(stdout: str) -> Path:
    m = RUN_DIR_LINE.search(stdout)
    if not m:
        raise AssertionError(f"no 'run_dir: <absolute path>' line on stdout:\n{stdout[-2000:]}")
    return Path(m.group("path"))


def parse_errors(stderr: str) -> list[dict[str, str]]:
    out = []
    for line in stderr.splitlines():
        m = ERROR_LINE.match(line.strip())
        if m:
            out.append(m.groupdict())
    return out


# ---------------------------------------------------------------------------
# Shared workspace: one demo copy, one warm-up, and cached runs per process.
# ---------------------------------------------------------------------------

class CommandFailed(AssertionError):
    pass


class _Workspace:
    def __init__(self) -> None:
        self._root: Path | None = None
        self._warmup_error: str | None = None
        self._runs: dict[str, tuple[subprocess.CompletedProcess, Path | None]] = {}

    @property
    def root(self) -> Path:
        if self._root is None:
            self._root = Path(tempfile.mkdtemp(prefix="gatedtrain-tests-"))
            if os.environ.get("GATEDTRAIN_KEEP_TMP") != "1":
                atexit.register(shutil.rmtree, self._root, True)
        return self._root

    @property
    def demo(self) -> Path:
        return self.root / "demo"

    def fresh_demo_copy(self, name: str) -> Path:
        """A private copy of demo/ (without any baseline) for tests that edit files."""
        dst = self.root / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(DEMO_DIR, dst, ignore=shutil.ignore_patterns("baseline", "runs", "__pycache__", ".DS_Store"))
        return dst

    def ensure_baseline(self) -> Path:
        """Copy demo/ and run the warm-up once: ``gatedtrain warmup warmup.yaml --out baseline``."""
        if self._warmup_error is not None:
            raise CommandFailed(self._warmup_error)
        baseline = self.demo / "baseline"
        if baseline.is_dir():
            return baseline
        if not self.demo.exists():
            shutil.copytree(DEMO_DIR, self.demo, ignore=shutil.ignore_patterns("baseline", "runs", "__pycache__", ".DS_Store"))
        proc = run_cli(["warmup", "warmup.yaml", "--out", "baseline"], cwd=self.demo)
        if proc.returncode != EXIT_OK or not baseline.is_dir():
            self._warmup_error = "warm-up failed: " + describe(proc)
            raise CommandFailed(self._warmup_error)
        return baseline

    def runs_root(self, key: str) -> Path:
        return self.root / f"runs-{key}"

    def has_run(self, key: str) -> bool:
        return key in self._runs

    def run(self, key: str, spec: str, runs_root_key: str | None = None) -> tuple[subprocess.CompletedProcess, Path]:
        """Run ``gatedtrain run <spec> --runs-root <root>/runs-<runs_root_key or key>`` once per key."""
        if key not in self._runs:
            self.ensure_baseline()
            runs_root = self.runs_root(runs_root_key or key)
            proc = run_cli(["run", spec, "--runs-root", str(runs_root)], cwd=self.demo)
            try:
                run_dir = parse_run_dir(proc.stdout)
            except AssertionError:
                run_dir = None
            self._runs[key] = (proc, run_dir)
        proc, run_dir = self._runs[key]
        if run_dir is None or not run_dir.is_dir():
            raise CommandFailed(f"run '{key}' produced no run directory: " + describe(proc))
        return proc, run_dir


WORKSPACE = _Workspace()


# ---------------------------------------------------------------------------
# Run-directory accessors (layout fixed in tests/README.md section 4).
# ---------------------------------------------------------------------------

def load_report(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "report" / "gate_report.json").read_text(encoding="utf-8"))


def load_manifest(run_dir: Path, name: str) -> dict[str, Any]:
    return json.loads((run_dir / "manifests" / f"{name}.json").read_text(encoding="utf-8"))


def stage_dir(run_dir: Path, stage_id: str = "sft") -> Path:
    return run_dir / "stage" / stage_id


def checkpoint_dirs(run_dir: Path, stage_id: str = "sft") -> list[tuple[int, Path]]:
    ck = stage_dir(run_dir, stage_id) / "checkpoints"
    out = []
    for p in ck.iterdir() if ck.is_dir() else []:
        m = re.fullmatch(r"step_(\d+)", p.name)
        if m and p.is_dir():
            out.append((int(m.group(1)), p))
    return sorted(out)


def train_step_files(run_dir: Path, stage_id: str = "sft") -> list[tuple[int, Path]]:
    logs = stage_dir(run_dir, stage_id) / "logs"
    out = []
    for p in logs.iterdir() if logs.is_dir() else []:
        m = re.fullmatch(r"train_data_step(\d+)\.jsonl", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def expected_step_shards(step_sizes: list[tuple[int, int]], shards: list[tuple[str, int]],
                          samples_per_position: int = 1) -> set[tuple[int, str]]:
    """Independent step -> shard mapping at precision ``iteration``.

    step_sizes: [(step, lines logged at that step)] in step order.
    shards: [(shard id, row count)] in read order (file-name order).
    samples_per_position: lines per loader position (generations per prompt,
    interim rule G-7); a step consumes lines / samples_per_position rows.

    Loader semantics follow NeMo RL's ``StatefulDataLoader(..., drop_last=True)``
    (nemo_rl/algorithms/grpo.py:883-890 at 4f39fd8, drop_last=True at :888; maintainers' decision D-1):
    rows are read in order; a batch that does not fit in the rest of the epoch
    is dropped, so the tail rows are skipped and the next epoch restarts at
    row 0; no batch straddles an epoch boundary. A step with zero samples maps
    to no shard (decision D-2).
    """
    total = sum(n for _, n in shards)
    bounds = []
    start = 0
    for sid, n in shards:
        bounds.append((sid, start, start + n))
        start += n
    pairs: set[tuple[int, str]] = set()
    cursor = 0
    for step, lines in step_sizes:
        assert lines % samples_per_position == 0, (step, lines)
        size = lines // samples_per_position
        if size == 0:
            continue
        assert size <= total, "a batch larger than the dataset never fills under drop_last"
        if cursor + size > total:
            cursor = 0
        for pos in range(cursor, cursor + size):
            for sid, lo, hi in bounds:
                if lo <= pos < hi:
                    pairs.add((step, sid))
                    break
        cursor += size
    return pairs


def count_lines(path: Path) -> int:
    return sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())


def python_version_ok() -> bool:
    return sys.version_info[:2] in ((3, 12), (3, 13))
