"""Attempt records of a resumable external run, their alignment with the host guard's events, the segment chain
and the kept/orphaned split (EXT-RES-1 to EXT-RES-4; tests/README.md section 11.35).

A run's `attempts.jsonl` begins with the header line `{"schema": "gatedtrain.ext.attempts", "version": 1}` (the
marker of the new layout; a file whose first line does not name the schema id is an older launcher's file and is not
read here, and a first line that names it but is not exactly that header, integer version included, refuses), then
holds, per attempt A (1, 2, ... in launch order), a `start` line
`{"event": "start", "attempt": A, "utc": ..., "restore": null | {"cut", "last_step_included",
"cut_json_sha256", "repaired"}}` and, only on a normal exit, an `end` line `{"event": "end", "attempt": A,
"utc": ..., "status": ...}`. Start times do not decrease from one attempt to the next. gpu-step.sh's log holds, per label, `START`, then `PREEMPTED` or `END rc=<int>`.

Alignment rule. A "non-empty attempt" is one with a `start` line. Guard attempts are the label's `START`
events in log order (the `attempt=` token restarts at 1 on each gpu-step.sh invocation, so it is not used).
A run `start` line pairs with the LAST guard attempt whose `START` is <= the line's `utc` (the run's start may
be later than the guard's `START`), and its `utc` must be <= that attempt's terminator (or the next `START`
when unterminated); two run lines in one guard attempt refuse, and so does a run attempt whose guard attempt comes
before the previous run attempt's (guard order follows attempt order). Run `utc` values are ISO 8601 UTC (`Z` or
`+00:00`, optional fraction) compared at whole seconds. A guard attempt with no run `start` line is an empty
attempt (paired with None), allowed only if it ended `PREEMPTED` and is not the last guard attempt.

Chain rule (design 5.3, 5.3a). Any guard `END rc!=0` for the label refuses. The final attempt is the LAST
guard attempt: it is `END rc=0` (the label's last terminal event, EXT-53), paired with a run `start` line
whose `end` line has status "ok". Every other non-empty attempt has no `end` line and its guard attempt ended
`PREEMPTED` (EXT-RES-3).

Segment rule (design 5.2, 05 12.4; linear chains only). Empty attempts are skipped. The first non-empty attempt
has `restore` null. Attempt k>1 restores the newest committed cut written by attempts 1..k-1 (EXT-RES-2). A cut
is committed when it is in the `cuts` list ext-collect passes in. With more than one non-empty attempt, every
committed cut's `cut.json` must carry a positive integer `attempt` (its writer), naming one of the non-empty
attempts, with a step above that writer's branch step; otherwise the chain refuses (there is no step-based
fallback). A single-attempt run may omit the field. The restore's `last_step_included` and `cut_json_sha256` must
match the cut and its `cut.json` bytes (read without following links); a restored cut's `cut.json` must carry
that integer `last_step_included` (design 5.1: every restore point records it). A repaired restore is checked the same way
(EXT-RES-7 adds its own validity checks). A later attempt with no earlier cut restores nothing and starts at
branch step 0. A non-final segment ends at the next attempt's branch step; the final one at `max_steps`. A record
of attempt A is kept iff `branch_step_A < step <= end_step_A` (EXT-RES-4)."""

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
HEADER_SCHEMA, HEADER_VERSION = "gatedtrain.ext.attempts", 1
HEADER = {"schema": HEADER_SCHEMA, "version": HEADER_VERSION}


def is_new_layout(stage_out: Path) -> bool:
    """Whether `<stage_out>/attempts.jsonl` is the resumable layout: its FIRST line names the schema id
    `gatedtrain.ext.attempts` (the header `{"schema": "gatedtrain.ext.attempts", "version": 1}`). The first line ends at
    the reader's line ends (`\\n`, `\\r\\n` or `\\r`, `canonical.text_lines`). A first line that names the schema id but is
    not that exact header (a BOM before it, other keys, a float version) is the new layout too, so that read_attempts
    refuses it instead of reading the run as an old one (EXT-RES-1). Any other attempts.jsonl (the launchers' older
    formats) or none is the old layout. A link counts as new so that read_attempts refuses it."""
    path = Path(stage_out) / ATTEMPTS_FILE
    if os.path.islink(path):
        return True
    try:
        dfd = armfs.open_arm(Path(stage_out))
        try:
            data = armfs.read_file(dfd, ATTEMPTS_FILE)
        finally:
            os.close(dfd)
    except (armfs.Unreadable, OSError):
        return False
    first = data.split(b"\n", 1)[0].split(b"\r", 1)[0]           # the first line of text_lines, on bytes
    return HEADER_SCHEMA.encode() in first


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
    last_start = None                                     # the previous start line's utc (whole seconds)
    for n, line in enumerate(text_lines(text), start=1):
        try:
            row = json.loads(line)
        except ValueError:
            raise _refuse(f"{field}:{n}", "line is not JSON", _FIX) from None
        if n == 1 and HEADER_SCHEMA in line:
            # The header exactly: these two keys, the schema id and the integer version (not 1.0, not true).
            if not (isinstance(row, dict) and set(row) == {"schema", "version"} and row["schema"] == HEADER_SCHEMA):
                raise _refuse(f"{field}:1", f"the first line names {HEADER_SCHEMA} but is not the header "
                                            f"{json.dumps(HEADER)}", _FIX)
            if type(row["version"]) is not int or row["version"] != HEADER_VERSION:
                raise _refuse(f"{field}:1", f"header version {row.get('version')!r} is not supported "
                                            f"(this collector reads the integer version {HEADER_VERSION})", _FIX)
            continue
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
            t = _run_ts(row["utc"])
            if last_start is not None and t < last_start:
                # A backdated start would pair with an earlier guard window and flip the chain (align).
                raise _refuse(f"{field}:{n}", f"start line of attempt {row['attempt']} has utc {row['utc']}, earlier "
                                              "than the previous attempt's start; start times must not decrease", _FIX)
            last_start = t
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
    prev_gi = None                        # the guard attempt of the previous run attempt: guard order follows A
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
        if prev_gi is not None and gi < prev_gi:
            p1(f"attempts.jsonl attempt {r.attempt}", f"the start line lies in guard attempt {g.index}, before the "
               f"previous run attempt's guard attempt {guard[prev_gi].index}; run attempts must follow the guard "
               "attempts in order", restart)
            return [(g, assigned.get(i)) for i, g in enumerate(guard)], problems
        prev_gi = gi
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


@dataclass(frozen=True)
class Segment:
    attempt: int
    attempt_id: str              # f"{run_id}.{attempt}"
    parent: int | None           # previous non-empty attempt
    branch_step: int | None      # None for the first attempt; else the restored cut's last_step_included (0: none)
    end_step: int                # last step this segment keeps
    restore_cut: str | None


def _read_cut(path: Path) -> tuple[str | None, dict | None]:
    """(sha256 of `<path>/cut.json`, its JSON object or None), or (None, None) when unreadable (no link following)."""
    try:
        dfd = armfs.open_arm(path)
        try:
            data = armfs.read_file(dfd, "cut.json")
        finally:
            os.close(dfd)
    except armfs.Unreadable:
        return None, None
    try:
        doc = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, RecursionError):
        return armfs.sha256(data), None
    return armfs.sha256(data), doc if isinstance(doc, dict) else None


def segment_chain(aligned: list[tuple[GuardAttempt, AttemptRecord | None]], cuts: list[tuple[int, Path]], *,
                  run_id: str, max_steps: int) -> tuple[list[Segment], list[Problem]]:
    """The segments of the non-empty attempts in `aligned` (rule in the module docstring) and the EXT-RES-2 problems."""
    problems: list[Problem] = []

    def p2(attempt: int, msg: str) -> None:
        problems.append(Problem("EXT-RES-2", f"attempts.jsonl attempt {attempt}", msg,
                                "restore only the newest committed cut of the earlier attempts; a rollback is not "
                                "supported (linear chains), start a new launch instead"))

    info: dict[int, tuple[Path, str | None, dict | None]] = {}
    for step, path in cuts:
        if step in info:
            problems.append(Problem("EXT-RES-2", f"cuts/{Path(path).name}",
                                    f"two committed cuts at step {step}: {info[step][0].name} and {Path(path).name}",
                                    "collect a run with one committed cut per step"))
            continue
        sha, doc = _read_cut(Path(path))
        info[step] = (Path(path), sha, doc)
    ne = [r for _, r in aligned if r is not None]
    writer: dict[int, int] = {}
    if len(ne) > 1:
        known = {r.attempt for r in ne}
        for st in sorted(info):
            w = info[st][2].get("attempt") if info[st][2] else None
            if not isinstance(w, int) or isinstance(w, bool) or w < 1:
                problems.append(Problem("EXT-RES-2", f"cuts/{info[st][0].name}",
                                        f"cut {info[st][0].name} does not record the attempt that wrote it; a "
                                        "resumed run's cuts must", "write the integer `attempt` into every cut.json"))
            elif w not in known:
                problems.append(Problem("EXT-RES-2", f"cuts/{info[st][0].name}",
                                        f"cut {info[st][0].name} records attempt {w}, which is not one of the run's "
                                        "non-empty attempts", "collect the run's own cuts"))
            else:
                writer[st] = w
        if problems:
            return [], problems
    segs: list[Segment] = []
    steps: list[int] = []                 # branch step of each non-empty attempt (0 for the first)
    for idx, r in enumerate(ne):
        rs = r.restore
        if idx == 0:
            if rs is not None:
                p2(r.attempt, "the first attempt has a restore")
            steps.append(0)
            segs.append(Segment(r.attempt, f"{run_id}.{r.attempt}", None, None, max_steps, None))
            continue
        earlier = {e.attempt for e in ne[:idx]}
        cand = [st for st, w in writer.items() if w in earlier]
        expected = max(cand) if cand else None
        branch = 0
        if rs is None:
            if expected is not None:
                p2(r.attempt, f"the attempt restores no cut although checkpoint-{expected} was committed")
        else:
            step = rs.get("last_step_included")
            name = rs.get("cut")
            c = info.get(step) if isinstance(step, int) and not isinstance(step, bool) else None
            if c is None or c[0].name != name:
                p2(r.attempt, f"the restored cut {name!r} (last_step_included {step!r}) is not a committed cut")
            else:
                branch = step
                doc = c[2] or {}
                if rs.get("cut_json_sha256") != c[1]:
                    p2(r.attempt, f"cut_json_sha256 of {name} does not match its cut.json")
                lsi = doc.get("last_step_included")
                if not (isinstance(lsi, int) and not isinstance(lsi, bool)):
                    # Design 5.1: every restore point is a committed cut with `last_step_included`.
                    p2(r.attempt, f"the cut.json of {name} has no integer last_step_included; a restored cut must "
                                  "record it")
                elif lsi != step:
                    p2(r.attempt, f"last_step_included {step} of {name} does not match its cut.json")
                if step != expected:
                    p2(r.attempt, f"{name} is not the newest committed cut of the earlier attempts "
                                  f"(checkpoint-{expected})")
        steps.append(branch)
        prev = segs[-1]
        segs[-1] = Segment(prev.attempt, prev.attempt_id, prev.parent, prev.branch_step, branch, prev.restore_cut)
        segs.append(Segment(r.attempt, f"{run_id}.{r.attempt}", prev.attempt, branch, max_steps,
                            rs.get("cut") if rs else None))
    by_attempt = {sg.attempt: sg for sg in segs}
    for st, w in sorted(writer.items()):
        if st <= (by_attempt[w].branch_step or 0):
            p2(w, f"{info[st][0].name} is recorded as written by attempt {w} but its step is not above that "
                  f"attempt's branch step {by_attempt[w].branch_step or 0}")
    return segs, problems


def split_steps(chain: list[Segment], records: list[dict]) -> tuple[list[dict], list[dict], list[Problem]]:
    """(kept, orphaned, problems) (EXT-RES-4). A record of attempt A is kept iff `branch_step_A < step <= end_step_A`
    (branch 0 for the first attempt). It is orphaned only when A is not the final segment and `step > end_step_A`
    (steps run after A's last cut and redone by a later attempt). A record that is malformed (`attempt` or `step`
    missing, not an int, a bool, or step < 1), has step > max_steps, or step <= branch_step_A refuses, since
    orphaned records are excluded from every rollup and would hide consumption. Every step 1..max_steps (the last segment's end) must have
    kept records from exactly one attempt; a record of an attempt outside the chain refuses."""
    problems: list[Problem] = []
    fix = "collect a run whose kept records cover every step exactly once"
    seg = {s.attempt: s for s in chain}
    kept: list[dict] = []
    orphaned: list[dict] = []
    owners: dict[int, set[int]] = {}
    unknown: set = set()
    max_steps = chain[-1].end_step if chain else 0
    bad: list[str] = []
    for i, r in enumerate(records):
        a, st = (r.get("attempt"), r.get("step")) if isinstance(r, dict) else (None, None)
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in (a, st)) or st < 1:
            bad.append(f"record {i}: `attempt` and `step` must be integers with step >= 1")
            continue
        s = seg.get(a)
        if s is None:
            unknown.add(a)
            continue
        if st > max_steps:
            bad.append(f"record {i} (attempt {a}): step {st} is beyond max_steps {max_steps}")
        elif st <= (s.branch_step or 0):
            bad.append(f"record {i} (attempt {a}): step {st} is not above the attempt's branch step "
                       f"{s.branch_step or 0}")
        elif st <= s.end_step:
            kept.append(r)
            owners.setdefault(st, set()).add(s.attempt)
        else:
            orphaned.append(r)
    for m in bad[:5]:
        problems.append(Problem("EXT-RES-4", "records", m, fix))
    if len(bad) > 5:
        problems.append(Problem("EXT-RES-4", "records", f"{len(bad) - 5} more malformed or out-of-range records", fix))
    for a in sorted(unknown, key=str):
        problems.append(Problem("EXT-RES-4", "records", f"records of attempt {a!r}, which is not in the segment chain",
                                fix))
    if not chain:
        problems.append(Problem("EXT-RES-4", "records", "the segment chain is empty", fix))
        return kept, orphaned, problems
    max_steps = chain[-1].end_step
    gaps = [n for n in range(1, max_steps + 1) if n not in owners]
    over = sorted(n for n, o in owners.items() if len(o) > 1)
    if gaps:
        problems.append(Problem("EXT-RES-4", "records", f"no kept record at steps {_ranges(gaps)} of 1..{max_steps}",
                                fix))
    if over:
        problems.append(Problem("EXT-RES-4", "records", f"kept records of more than one attempt at steps "
                                                        f"{_ranges(over)}", fix))
    return kept, orphaned, problems


def _ranges(steps: list[int]) -> str:
    out: list[str] = []
    i = 0
    while i < len(steps):
        j = i
        while j + 1 < len(steps) and steps[j + 1] == steps[j] + 1:
            j += 1
        out.append(str(steps[i]) if i == j else f"{steps[i]}-{steps[j]}")
        i = j + 1
    return ", ".join(out)
