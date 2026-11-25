"""One result shape for every CLI command (tests/README.md section 12, AN-1).

In text mode `emit` writes the result's warnings and then its problems to stderr and the human text to stdout (a
command that must print warnings before its work starts, such as the EXT-85 override lines, prints them itself and
passes none); with `--json` it prints one JSON object on stdout and nothing on stderr. The exit code is the same in
both modes.
"""

from __future__ import annotations

import contextlib
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .errors import Problem, printable

RESULT_SCHEMA = "gatedtrain.cli_result"
RESULT_SCHEMA_VERSION = 1


@dataclass
class Result:
    command: str
    ok: bool
    exit: int
    data: dict[str, Any] = field(default_factory=dict)
    problems: list[Problem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def result_json(result: Result) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "command": result.command,
        "ok": result.ok,
        "exit": result.exit,
        "data": result.data,
        "problems": [{"rule": p.rule, "field": p.field, "message": p.message, "fix": p.fix} for p in result.problems],
        "warnings": list(result.warnings),
    }


# Notices: warnings raised below a command, in library code that has no `Result` to put them in.
#   * Inside `notice_scope()` (which `cli.main` enters around every command) `notice` queues the message; the command's
#     `emit` drains the queue into the result's warnings (`--json` carries them, text mode prints them to stderr, once).
#   * Outside any scope (a library caller of `KeeperTrust.report`, say) there is no `emit` to carry the message, so it
#     goes to stderr at once and is not queued.
#   * A notice still queued when the outermost scope ends (a command that never called `emit`) is printed to stderr,
#     never dropped. `drain_notices()` is the queue API for a caller that wants the messages itself.
# One process-wide queue, not thread-safe: the CLI runs one command per process.
_NOTICES: list[str] = []
_SCOPES = 0


def _to_stderr(message: str) -> None:
    print(printable(message, keep="\n\t"), file=sys.stderr)


def notice(message: str) -> None:
    """A warning raised below the command. Queued inside `notice_scope()` (duplicates are kept once), else printed to
    stderr at once."""
    if _SCOPES == 0:
        _to_stderr(message)
    elif message not in _NOTICES:
        _NOTICES.append(message)


def drain_notices() -> list[str]:
    """The pending notices, removed from the queue (empty outside a scope: there nothing is queued)."""
    out = list(_NOTICES)
    _NOTICES.clear()
    return out


@contextlib.contextmanager
def notice_scope() -> Iterator[None]:
    """Collect `notice`s for the command running inside (scopes nest); whatever is still queued when the outermost
    scope ends is printed to stderr."""
    global _SCOPES
    _SCOPES += 1
    try:
        yield
    finally:
        _SCOPES -= 1
        if _SCOPES == 0:
            for message in drain_notices():
                _to_stderr(message)


def emit(result: Result, *, as_json: bool, human: str) -> int:
    """Print `human` (text mode) or one JSON line (`as_json`); return the exit code."""
    pending = [m for m in drain_notices() if m not in result.warnings]
    if pending:
        result = Result(result.command, result.ok, result.exit, result.data, result.problems,
                        [*result.warnings, *pending])
    if as_json:
        try:
            text = json.dumps(result_json(result), sort_keys=True, default=str, allow_nan=False)
        except ValueError:                       # a NaN or infinity is not JSON: say so instead of printing it
            bad = Result(result.command, False, result.exit or 3, {},
                         [Problem("OUTPUT", "data", "the result holds a non-finite number, which JSON cannot carry",
                                  "report this as a bug")], list(result.warnings))
            text = json.dumps(result_json(bad), sort_keys=True, default=str)
            print(text)
            return bad.exit
        print(text)
        return result.exit
    for line in result.warnings:
        print(printable(line, keep="\n\t"), file=sys.stderr)
    for p in result.problems:
        print(p.line(), file=sys.stderr)
    if human:
        sys.stdout.write(human)
    return result.exit
