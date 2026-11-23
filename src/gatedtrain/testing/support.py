"""Canonical JSON and manifest hashes (canonical JSON and manifest hashes, an independent copy) and the `gatedtrain` CLI in a subprocess."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any

# A test never reads the developer's host file (`~/.config/gatedtrain/host.toml`): its daemon lists enter
# `host_config_sha256`, so the layouts the tests build would not match. Unless the environment names a host file, an
# empty one applies, in process and in the CLI subprocesses the tests start. Only the test helpers (this module and
# `gatedtrain.testing.ext`, which imports it) do this; the modules a distribution registers through entry points do
# not, so an installed `gatedtrain` command that loads them still reads the user's host file.
os.environ.setdefault("GATEDTRAIN_HOST_CONFIG", os.devnull)

EXIT_OK = 0
EXIT_GATE_NOT_PASS = 1
EXIT_USAGE = 2
EXIT_REFUSED = 3

# Per-command time limit of a CLI call.
RUN_TIMEOUT_S = 150

ERROR_LINE = re.compile(
    r"^error\[(?P<rule>[A-Z0-9-]+)\] (?P<field>\S+): (?P<message>.+?) \| fix: (?P<fix>\S.*)$"
)
RUN_DIR_LINE = re.compile(r"^run_dir: (?P<path>/.+)$", re.MULTILINE)
SHA256_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


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


def cli_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "0")
    return env


def run_cli(args: list[str], *, cwd: Path, timeout: int = RUN_TIMEOUT_S,
            env: dict[str, str] | None = None, python: str | None = None) -> subprocess.CompletedProcess:
    """Run ``python -m gatedtrain <args>`` with the interpreter running the tests (gatedtrain and the recipes' plugins installed
    in it), or `python`; `env` adds variables (the external-run host layout)."""
    return subprocess.run(
        [python or sys.executable, "-m", "gatedtrain", *args],
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
