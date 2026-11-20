"""Attempt records of a resumable external run and their alignment with the host guard's events
(EXT-RES-1, EXT-RES-3; tests/README.md section 11.35).

A run's `attempts.jsonl` holds, per attempt A (1, 2, ... in launch order), a `start` line
`{"event": "start", "attempt": A, "utc": ..., "restore": null | {"cut", "last_step_included",
"cut_json_sha256", "repaired"}}` and, only on a normal exit, an `end` line `{"event": "end", "attempt": A,
"utc": ..., "status": ...}`. gpu-step.sh's log holds, per label, `START`, then `PREEMPTED` or `END rc=<int>`.

Alignment rule. A "non-empty attempt" is one with a `start` line. Guard attempts are the label's `START`
events in log order (the `attempt=` token restarts at 1 on each gpu-step.sh invocation, so it is not used).
A run `start` line pairs with the LAST guard attempt whose `START` is <= the line's `utc` (the run's start may
be later than the guard's `START`), and its `utc` must be <= that attempt's terminator (or the next `START`
when unterminated); two run lines in one guard attempt refuse. Run `utc` values are ISO 8601 UTC (`Z` or
`+00:00`, optional fraction) compared at whole seconds. A guard attempt with no run `start` line is an empty
attempt (paired with None), allowed only if it ended `PREEMPTED` and is not the last guard attempt.

Chain rule (design 5.3, 5.3a). Any guard `END rc!=0` for the label refuses. The final attempt is the LAST
guard attempt: it is `END rc=0` (the label's last terminal event, EXT-53), paired with a run `start` line
whose `end` line has status "ok". Every other non-empty attempt has no `end` line and its guard attempt ended
`PREEMPTED` (EXT-RES-3)."""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..canonical import text_lines
from ..errors import Problem, Refused
from . import armfs
from .launch import _ts

ATTEMPTS_FILE = "attempts.jsonl"


@dataclass(frozen=True)
class AttemptRecord:
    attempt: int
    start_utc: str
    restore: dict | None
    end_status: str | None
    repaired: bool


@dataclass(frozen=True)
class GuardAttempt:
    index: int          # 1-based position among the label's START events
    start_utc: str
    outcome: str        # "preempted", "end" or "running"
    rc: int | None
    end_utc: str | None = field(default=None, compare=False)   # window end: terminator, or the next START


_UTC_RE = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2})(?:\.[0-9]+)?(?:Z|\+00:00)")


def _run_ts(value: str) -> _dt.datetime:
    """A run-side `utc`: ISO 8601 UTC, optional fractional seconds, `Z` or `+00:00`; truncated to whole seconds
    (the guard log has second resolution). Writers write %Y-%m-%dT%H:%M:%SZ. Raises ValueError."""
    m = _UTC_RE.fullmatch(value)
    if m is None:
        raise ValueError(value)
    return _dt.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")


def _refuse(field: str, message: str, fix: str) -> Refused:
    return Refused([Problem("EXT-RES-1", field, message, fix)])


_FIX = "run the recipe's resumable hooks, which write attempts.jsonl in the output directory, and re-launch"


def read_attempts(stage_out: Path) -> list[AttemptRecord]:
    """The attempt records of `<stage_out>/attempts.jsonl`, read without following links (armfs rules).
    Raises Refused (EXT-RES-1) when the file is absent, a link, not JSONL of the documented lines, or its
    `start` lines are not attempts 1, 2, ... in order."""
    field = ATTEMPTS_FILE
    try:
        dfd = armfs.open_arm(stage_out)
        try:
            data = armfs.read_file(dfd, ATTEMPTS_FILE)
        finally:
            os.close(dfd)
    except armfs.Unreadable as exc:
        raise _refuse(field, exc.text(f"{ATTEMPTS_FILE} under {stage_out}"), _FIX) from None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise _refuse(field, f"{ATTEMPTS_FILE} is not UTF-8", _FIX) from None
    starts: list[dict] = []
    ends: dict[int, str] = {}
    for n, line in enumerate(text_lines(text), start=1):
        try:
            row = json.loads(line)
        except ValueError:
            raise _refuse(f"{field}:{n}", "line is not JSON", _FIX) from None
        if not isinstance(row, dict) or not isinstance(row.get("attempt"), int) or isinstance(row["attempt"], bool) \
                or not isinstance(row.get("utc"), str):
            raise _refuse(f"{field}:{n}", "line is not an object with integer `attempt` and string `utc`", _FIX)
        ev = row.get("event")
        try:
            _run_ts(row["utc"])
        except ValueError:
            raise _refuse(f"{field}:{n}", f"utc {row['utc']!r} is not ISO 8601 UTC (Z or +00:00)", _FIX) from None
        if ev == "start":
            rs = row.get("restore")
            if rs is not None and not isinstance(rs, dict):
                raise _refuse(f"{field}:{n}", "`restore` is neither null nor an object", _FIX)
            if rs is not None and not isinstance(rs.get("repaired", False), bool):
                raise _refuse(f"{field}:{n}", "`restore.repaired` is not a boolean", _FIX)
            if row["attempt"] != len(starts) + 1:
                raise _refuse(f"{field}:{n}", f"start line has attempt {row['attempt']}, expected {len(starts) + 1}",
                              _FIX)
            starts.append(row)
        elif ev == "end":
            a = row["attempt"]
            if a != len(starts) or a in ends:
                raise _refuse(f"{field}:{n}", f"end line of attempt {a} does not follow that attempt's start line "
                                              "or repeats an end line", _FIX)
            if not isinstance(row.get("status"), str):
                raise _refuse(f"{field}:{n}", "end line has no string `status`", _FIX)
            ends[a] = row["status"]
        else:
            raise _refuse(f"{field}:{n}", f"unknown event {ev!r}", _FIX)
    out = []
    for row in starts:
        rs = row.get("restore")
        out.append(AttemptRecord(row["attempt"], row["utc"], rs, ends.get(row["attempt"]),
                                 bool(rs.get("repaired")) if rs else False))
    return out


def guard_attempts(log_path: Path, label: str) -> list[GuardAttempt]:
    """The guard attempts of `label` in gpu-step.sh's log, in START order. A START followed by another START
    without a terminator is `running` (unterminated). Raises ValueError on an unparseable timestamp, an END
    without `rc=<int>`, or a terminator without a START (as gpu_seconds does)."""
    log_path = Path(log_path)
    out: list[GuardAttempt] = []
    open_ga: GuardAttempt | None = None
    for n, line in enumerate(text_lines(log_path.read_text(encoding="utf-8")), start=1):
        tok = line.split()
        if len(tok) < 3 or tok[2] != label or tok[1] not in ("START", "END", "PREEMPTED"):
            continue
        try:
            _ts(tok[0])
        except ValueError:
            raise ValueError(f"{log_path}:{n}: timestamp {tok[0]!r} is not %Y-%m-%dT%H:%M:%SZ") from None
        if tok[1] == "START":
            if open_ga is not None:
                out.append(GuardAttempt(open_ga.index, open_ga.start_utc, "running", None, tok[0]))
            open_ga = GuardAttempt(len(out) + 1, tok[0], "running", None)
            continue
        if open_ga is None:
            raise ValueError(f"{log_path}:{n}: {tok[1]} of {label} without a START")
        rcs = [x[3:] for x in tok[3:] if x.startswith("rc=")]
        rc: int | None = None
        if tok[1] == "END" or rcs:
            if len(rcs) != 1 or not re.fullmatch(r"-?[0-9]+", rcs[0]):
                if tok[1] == "END":
                    raise ValueError(f"{log_path}:{n}: END of {label} has no rc=<int>")
            else:
                rc = int(rcs[0])
        outcome = "end" if tok[1] == "END" else "preempted"
        out.append(GuardAttempt(open_ga.index, open_ga.start_utc, outcome, rc, tok[0]))
        open_ga = None
    if open_ga is not None:
        out.append(open_ga)
    return out


def align(records: list[AttemptRecord], guard: list[GuardAttempt]
          ) -> tuple[list[tuple[GuardAttempt, AttemptRecord | None]], list[Problem]]:
    """Pair each guard attempt with its run attempt (None: empty) by the window rule in the module docstring and
    check the chain rule. Returns the pairs in guard order and the problems (EXT-RES-1, EXT-RES-3)."""
    problems: list[Problem] = []

    def p1(field: str, msg: str, fix: str) -> None:
        problems.append(Problem("EXT-RES-1", field, msg, fix))

    def p3(field: str, msg: str, fix: str) -> None:
        problems.append(Problem("EXT-RES-3", field, msg, fix))

    restart = "start again in a new, empty output directory (a new launch); a crash is never hidden by a later success"
    if not guard:
        p1("gpu-guard.log", "the log has no START of the run's launch label", "collect with the run's own label")
        return [], problems
    for g in guard:
        if g.outcome == "end" and g.rc != 0:
            p1("gpu-guard.log", f"guard attempt {g.index} ended with rc={g.rc}; any END rc!=0 makes the run "
                                "uncollectable (A1)", restart)

    starts = [_ts(g.start_utc) for g in guard]
    assigned: dict[int, AttemptRecord] = {}
    for r in records:
        t = _run_ts(r.start_utc)
        cand = [i for i, st in enumerate(starts) if st <= t]
        gi = cand[-1] if cand else None
        g = guard[gi] if gi is not None else None
        if g is None or (g.end_utc is not None and t > _ts(g.end_utc)):
            p1(f"attempts.jsonl attempt {r.attempt}", "the start line lies in no guard attempt window (a start "
               "before the first guard START, or after the attempt's terminator)", restart)
            return [(g, assigned.get(i)) for i, g in enumerate(guard)], problems
        if gi in assigned:
            p1(f"attempts.jsonl attempt {r.attempt}", f"guard attempt {g.index} already holds a run start line "
               "(more run start lines than guard STARTs)", restart)
            return [(g, assigned.get(i)) for i, g in enumerate(guard)], problems
        assigned[gi] = r
    pairs = [(g, assigned.get(i)) for i, g in enumerate(guard)]

    last = len(pairs) - 1
    gl = pairs[last][0]
    if gl.outcome != "running" and (gl.outcome != "end" or gl.rc != 0):
        # The run is unfinished: say so, rather than blaming earlier attempts for not being final.
        p3(f"guard attempt {gl.index}", "the final attempt (the label's last guard attempt) is not END rc=0 "
           "(EXT-53); an empty PREEMPTED attempt after the last run start line does not make an earlier attempt "
           "final", restart)
        return pairs, problems
    for i, (g, r) in enumerate(pairs):
        where = f"guard attempt {g.index}"
        if g.outcome == "running":
            p1(where, "the guard attempt has no END or PREEMPTED", restart)
            continue
        if i == last:
            if g.outcome != "end" or g.rc != 0:
                p3(where, "the final attempt (the label's last guard attempt) is not END rc=0 (EXT-53); an empty "
                          "PREEMPTED attempt after the last run start line does not make an earlier attempt final",
                   restart)
            if r is None:
                p3(where, "the final attempt has no run start line", restart)
            elif r.end_status != "ok":
                p3(f"attempts.jsonl attempt {r.attempt}",
                   "the final attempt has no end line with status \"ok\"" if r.end_status is None
                   else f"the final attempt's end status is {r.end_status!r}, not \"ok\"", restart)
            continue
        if r is None:
            if g.outcome != "preempted":
                p1(where, "the guard attempt has no run start line and did not end PREEMPTED (an empty attempt is "
                          "allowed only after PREEMPTED)", restart)
            continue
        if r.end_status is not None:
            p3(f"attempts.jsonl attempt {r.attempt}", "a non-final attempt has an end line", restart)
        if g.outcome != "preempted":
            p3(where, "a non-final attempt's guard attempt did not end PREEMPTED", restart)
    if not any(r is not None for _, r in pairs):
        p1("attempts.jsonl", "no run start line", _FIX)
    return pairs, problems
