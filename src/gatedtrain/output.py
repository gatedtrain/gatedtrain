"""One result shape for every CLI command (tests/README.md section 12, AN-1).

`emit` prints the human text exactly as the command always did (problems and warnings on stderr), or, with `--json`,
one JSON object on stdout and nothing on stderr. The exit code is the same in both modes.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any

from .errors import Problem

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


def emit(result: Result, *, as_json: bool, human: str) -> int:
    """Print `human` (text mode) or one JSON line (`as_json`); return the exit code."""
    if as_json:
        print(json.dumps(result_json(result), sort_keys=True, default=str))
        return result.exit
    for line in result.warnings:
        print(line, file=sys.stderr)
    for p in result.problems:
        print(p.line(), file=sys.stderr)
    if human:
        sys.stdout.write(human)
    return result.exit
