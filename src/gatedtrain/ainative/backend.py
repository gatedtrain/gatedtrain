"""AN-22: the local backend runs one project's jobs on the owner's rootless Docker daemon (design revision 25, 05 §18.10).

A `Job` is checked by the launch rules `ext-launch` applies to a request (`job_problems`: isolation, the argv binding,
the adapter's launch and argv rules), then recorded and started as a detached container (`runtime.Runtime`):

    <project>/jobs/<seq:04d>-<label>/job.json          the job (environment names only), its key, the stage, the
                                                       rootless setting, the host configuration hash, the script and
                                                       interpreter hashes, the docker argument list (names only)
    <project>/jobs/<seq:04d>-<label>/attempts.jsonl    rows `start`, `end`, `cancel_requested`, `stop`,
                                                       `refused`, `gpu_log_written`
    <project>/jobs/<seq:04d>-<label>/attempt-<n>.log   the tail of attempt n's container log (at most 10 MiB)
    <project>/jobs/<seq:04d>-<label>/lock              the job's lock (`fcntl.flock`)
    <project>/jobs/output-dirs.jsonl                   the output directories `new_output_dir` created
    <project>/jobs/lock                                the project's job lock (sequence numbers, keys, output dirs)

Each attempt writes into a fresh directory `<output dir>/a<attempt>` mounted at the job's output path, so a retry
never sees an earlier attempt's files. `submit` of a job whose key matches a queued, running or succeeded job of the
project returns that job (a stage re-run after the driver died reattaches to its container). `status` inspects the
current attempt and records its end once; a daemon error (`RuntimeFailure`) leaves the job's records unchanged.
Environment values are never recorded or printed: a job's start needs them, so a job this process did not submit
(or re-submit) is not started here.

AN-23: an attempt's GPU seconds are its container's run time (0 for a CPU job). Every start of a GPU attempt is refused
when the committed GPU time (the recorded GPU seconds plus the estimates of the project's other queued or running GPU
jobs) plus the job's estimate exceeds `budget.gpu_hours`; a refused start is a `refused` row and ends the job
`budget_exhausted`. A running attempt is stopped at its time limit, a GPU attempt also once it used the budget left when
it started, or once the project's recorded GPU time plus the run time of all its running GPU attempts reaches the budget
(`stop` rows). This is enforced per poll: only while a driver calls `status`, and an attempt may overrun by up to one
poll interval. Before a GPU attempt starts the backend waits for a free GPU (the
`probe`; none: no wait). `failed`, `preempted` and `lost` attempts are retried while attempts remain, under the job lock
that records the end. Each job that reaches a terminal state appends one row to `<host_work>/gpu-log.md`.

The budget and the records are cooperative: they hold for drivers that use this backend, and they are not a guarantee
against an agent that runs containers itself or edits the project's records (spec §5.8.3). The keeper's own container
time is the checked figure.

AN-24: a pack stage reaches the backend through `StageBackend` (`ctx.backend`), which records its jobs under the stage
and refuses a GPU job from a stage that does not declare `needs: gpu`, and every side effect during `plan`.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import posixpath
import re
import stat
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from ..canonical import canonical_json, sha256_bytes
from ..errors import Problem, Refused
from ..ext import launch
from ..ext.launch import HostConfig
from .keeper_trust import project_id
from .project import Project, _refuse_links, _write_atomic, append_jsonl, read_jsonl
from .runtime import (GPU_BUSY_MIB, GPU_POLL_S, GPU_WAIT_S, TIME_FORMAT, ContainerState, DockerCLIRuntime, GpuProbe, Mount,
                      RunSpec, Runtime, RuntimeFailure, docker_run_argv, wait_for_gpu)

JOB_SCHEMA = "gatedtrain.ainative.job"
JOB_ATTEMPT_SCHEMA = "gatedtrain.ainative.job_attempt"
OUTPUT_DIR_SCHEMA = "gatedtrain.ainative.output_dir"
SCHEMA_VERSION = 1
JOBS_DIR = "jobs"
RULE = "AN-22"
RULE_GPU = "AN-23"
RULE_STAGE = "AN-24"
GPU_LOG_HEADER = ("| Step | Started (UTC) | Ended (UTC) | GPU-hours | Running total | Outcome |\n"
                  "|---|---|---|---|---|---|\n")                # docs/gpu-step.md

STATES = ("queued", "running", "succeeded", "failed", "cancelled", "timed_out", "budget_exhausted")
TERMINAL = ("succeeded", "failed", "cancelled", "timed_out", "budget_exhausted")
OUTCOMES = ("ok", "oom", "cancelled", "timed_out", "budget_exhausted", "preempted", "lost", "failed")
RETRIED = ("failed", "preempted", "lost")
OUTCOME_STATE = {"ok": "succeeded", "oom": "failed", "cancelled": "cancelled", "timed_out": "timed_out",
                 "budget_exhausted": "budget_exhausted", "preempted": "failed", "lost": "failed", "failed": "failed"}
STOP_REASONS = ("timed_out", "budget_exhausted")
RC_SIGNALLED = (137, 143)               # SIGKILL, SIGTERM: a stop or a kill from outside
LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
JOB_ID_RE = re.compile(r"([0-9]{4,})-([A-Za-z0-9][A-Za-z0-9_.-]{0,63})")
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}")
LOG_CAP_BYTES = 10 * 1024 * 1024        # the saved container log: its tail, at most this many bytes
LOCK, JOB_FILE, ATTEMPTS, OUTPUT_DIRS = "lock", "job.json", "attempts.jsonl", "output-dirs.jsonl"
DIR_MODE = 0o755
LABEL_PROJECT, LABEL_JOB = "gatedtrain.project", "gatedtrain.job"
FIX = "fix the job and submit it again"
DEFAULT_JOB_TIME_LIMIT_S = 6 * 3600      # a job's time limit unless its adapter sets one (6 h)
DEFAULT_MAX_ATTEMPTS = 3                 # runs of a job that failed, was preempted or lost before it is final
JOB_POLL_S = GPU_POLL_S                  # how often `wait` reads a job's state
MAX_INPUT_ENTRIES = 100_000             # a read-only mount the job key fingerprints holds at most this many entries
SMALL_FILE_BYTES = 1 << 20              # inputs up to this size are fingerprinted by content, larger ones by mtime


def _problem(field: str, message: str, fix: str = FIX) -> Problem:
    return Problem(RULE, field, message, fix)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _strs(x: Any) -> bool:
    return isinstance(x, (tuple, list)) and all(isinstance(i, str) for i in x)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Job:
    """One container run. `runtime` is `{"kind": "container", "image"}` or `{"kind": "venv_in_container", "image",
    "venv_lock"}` with the image pinned by digest; `mounts` are read-only inputs; the output directory is mounted
    read-write at `output_container` (each attempt in its own `<output_dir>/a<n>`); `profile` is the ext profile the
    adapter's argv rules apply."""
    label: str
    adapter: str
    runtime: Mapping[str, str]
    framework: Mapping[str, str | None]
    argv: tuple[str, ...]
    env: Mapping[str, str]
    mounts: tuple[Mount, ...]
    output_dir: str
    output_container: str
    tmpfs: tuple[str, ...] = ()
    network: str = "none"
    gpu: bool = False
    time_limit_s: int = DEFAULT_JOB_TIME_LIMIT_S
    gpu_hours_est: float = 0.0
    training_seed: int = 0
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    profile: str = "full"

    def doc(self) -> dict[str, Any]:
        """The job as canonical JSON values, the environment as names and values."""
        return {"label": self.label, "adapter": self.adapter, "profile": self.profile,
                "runtime": dict(self.runtime), "framework": dict(self.framework), "argv": list(self.argv),
                "env": dict(self.env),
                "mounts": [{"host": m.host, "container": m.container, "read_only": m.read_only} for m in self.mounts],
                "output_dir": self.output_dir, "output_container": self.output_container, "tmpfs": list(self.tmpfs),
                "network": self.network, "gpu": self.gpu, "time_limit_s": self.time_limit_s,
                "gpu_hours_est": self.gpu_hours_est, "training_seed": self.training_seed,
                "max_attempts": self.max_attempts}

    def key(self) -> str:
        """`sha256:` of the canonical JSON of the job without its output directory, the fingerprint of its read-only
        inputs (`input_fingerprint`) and the hashes of the files `argv[0:3]` name in them (interpreter, script): a job
        whose inputs changed has another key, so it neither reattaches nor gets the old output directory. Reads the
        inputs; Refused (AN-22) for a mount over MAX_INPUT_ENTRIES entries."""
        doc = self.doc()
        del doc["output_dir"]
        return sha256_bytes(canonical_json({"job": doc, "inputs": input_fingerprint(self.mounts),
                                            "argv_files": _argv_file_hashes(self.argv, self.mounts)}))


def _text(name: str) -> str:
    """A file name as canonical JSON text (bytes that are not UTF-8 escaped, never refused)."""
    return os.fsencode(name).decode("utf-8", "backslashreplace")


def input_fingerprint(mounts: tuple[Mount, ...]) -> list[dict[str, Any]]:
    """Per read-only mount, its entries in sorted relative path order, never following a link (`lstat`): a directory,
    a link with its target, a regular file with its size and sha256 (at most SMALL_FILE_BYTES) or size and mtime_ns
    (larger), anything else as its kind. Refused (AN-22) past MAX_INPUT_ENTRIES entries in all (the walk stops there),
    and when an entry changes while it is read (a link removed, a file swapped for a FIFO)."""
    out: list[dict[str, Any]] = []
    count = 0

    def counted() -> None:
        nonlocal count
        count += 1
        if count > MAX_INPUT_ENTRIES:
            raise Refused([_problem("job.mounts", f"the read-only inputs hold more than {MAX_INPUT_ENTRIES} entries; "
                                    "the job key fingerprints every input",
                                    "mount fewer, larger inputs (an archive, a dataset file)")])
    for m in mounts:
        if not m.read_only:
            continue
        entries: list[list[Any]] = []
        counted()
        stack = [(os.path.realpath(m.host), "")]
        while stack:
            path, rel = stack.pop()
            try:
                st = os.lstat(path)
            except OSError:
                entries.append([rel, "missing"])
                continue
            if stat.S_ISLNK(st.st_mode):
                try:
                    target = os.readlink(path)
                except OSError:
                    raise Refused([_changed(path)]) from None
                entries.append([rel, "link", _text(target)])
            elif stat.S_ISDIR(st.st_mode):
                entries.append([rel, "dir"])
                names: list[str] = []
                try:
                    with os.scandir(path) as it:
                        for e in it:
                            counted()
                            names.append(e.name)
                except OSError:
                    entries.append([rel, "unreadable"])
                    continue
                stack += [(os.path.join(path, n), f"{rel}/{_text(n)}" if rel else _text(n))
                          for n in sorted(names, reverse=True)]
            elif stat.S_ISREG(st.st_mode):
                if st.st_size <= SMALL_FILE_BYTES:
                    try:
                        entries.append([rel, "file", st.st_size, _sha256_nofollow(path, field="job.mounts")])
                    except OSError:
                        entries.append([rel, "unreadable"])
                else:
                    entries.append([rel, "file", st.st_size, str(st.st_mtime_ns)])   # beyond 2^53: text
            else:
                entries.append([rel, "other"])
        out.append({"container": m.container, "entries": sorted(entries, key=lambda e: e[0])})
    return out


def _changed(path: str) -> Problem:
    return _problem("job.mounts", f"the input {path} changed while the job key was computed",
                    "submit the job again once its inputs stay unchanged")


def _argv_file_hashes(argv: tuple[str, ...], mounts: tuple[Mount, ...]) -> list[str | None]:
    """The sha256 of the file each of argv[0:3] names in a read-only mount (as the container resolves it), else None
    (also for a file that is not regular: a FIFO is never opened for reading in blocking mode)."""
    table = [{"host": m.host, "container": m.container, "mode": "ro"} for m in mounts if m.read_only]
    out: list[str | None] = []
    for a in list(argv)[:3]:
        hit = launch.container_file(a, table) if isinstance(a, str) else None
        try:
            out.append(_sha256_regular(hit[1], follow=True) if hit else None)
        except OSError:
            out.append(None)
    return out


def _record_doc(job: Job) -> dict[str, Any]:
    """`job.doc()` with the environment names only (MM-35)."""
    doc = job.doc()
    doc["env"] = sorted(doc["env"])
    return doc


def job_request(job: Job, cfg: HostConfig) -> dict[str, Any]:
    """The ext-launch request the job is checked as (`launch.isolation_problems`, kind "request")."""
    mounts = [{"host": m.host, "container": m.container, "mode": "ro" if m.read_only else "rw"} for m in job.mounts]
    mounts.append({"host": job.output_dir, "container": job.output_container, "mode": "rw"})
    return {"area": 0, "label": job.label, "host_work": cfg.host_work, "runtime": dict(job.runtime),
            "framework": dict(job.framework), "training_seed": job.training_seed, "gpus": job.gpu,
            "network": job.network, "network_reason": None, "user": launch.container_user(cfg), "env": dict(job.env),
            "mounts": mounts, "tmpfs": list(job.tmpfs), "output_dir": job.output_dir, "gpu_est_h": job.gpu_hours_est,
            "argv": list(job.argv)}


def _shape_problems(job: Any) -> list[Problem]:
    """The field types `job_request` and `Job.doc` rely on, and the backend's own limits."""
    if not isinstance(job, Job):
        return [_problem("job", f"{type(job).__name__} is not a Job")]
    out: list[Problem] = []
    if not (isinstance(job.label, str) and LABEL_RE.fullmatch(job.label)):
        out.append(_problem("job.label", f"label {job.label!r} is not a plain name of at most 64 characters "
                            "(letters, digits, _ . -; first a letter or digit)"))
    if not isinstance(job.adapter, str):
        out.append(_problem("job.adapter", "the adapter is not a name"))
    if job.profile not in launch.PROFILES:
        out.append(_problem("job.profile", f"profile {job.profile!r} is not one of {', '.join(launch.PROFILES)}"))
    for name in ("runtime", "framework", "env"):
        if not isinstance(getattr(job, name), Mapping):
            out.append(_problem(f"job.{name}", f"{name} is not a mapping"))
    if isinstance(job.env, Mapping) and not all(isinstance(k, str) for k in job.env):
        out.append(_problem("job.env", "environment names are not strings"))
    if not (_strs(job.argv) and job.argv):
        out.append(_problem("job.argv", "argv is not a non-empty sequence of strings"))
    if not (isinstance(job.mounts, (tuple, list)) and all(
            isinstance(m, Mount) and isinstance(m.host, str) and isinstance(m.container, str)
            and isinstance(m.read_only, bool) for m in job.mounts)):
        out.append(_problem("job.mounts", "mounts are not a sequence of Mount"))
    for name in ("output_dir", "output_container", "network"):
        if not isinstance(getattr(job, name), str):
            out.append(_problem(f"job.{name}", f"{name} is not a string"))
    if not _strs(job.tmpfs):
        out.append(_problem("job.tmpfs", "tmpfs is not a sequence of container paths"))
    if not isinstance(job.gpu, bool):
        out.append(_problem("job.gpu", "gpu is not true or false"))
    if not _num(job.gpu_hours_est):
        out.append(_problem("job.gpu_hours_est", "gpu_hours_est is not a number"))
    elif job.gpu is True and not job.gpu_hours_est > 0:
        out.append(_problem("job.gpu_hours_est", f"a GPU job needs a positive GPU-hour estimate, not "
                            f"{job.gpu_hours_est!r}", "give the job's estimated GPU-hours (from a smoke run)"))
    if not _int(job.training_seed):
        out.append(_problem("job.training_seed", "training_seed is not an integer"))
    for name in ("time_limit_s", "max_attempts"):
        v = getattr(job, name)
        if not (_int(v) and v >= 1):
            out.append(_problem(f"job.{name}", f"{name} {v!r} is not an integer of at least 1"))
    return out


def job_problems(job: Job, cfg: HostConfig, *, socket_gids: set[int] | None = None,
                 fresh_output: bool = True) -> list[Problem]:
    """AN-22: every rule the job breaks, before anything starts: the launch rules ext-launch applies to a request
    (isolation, the argv binding, the adapter's launch and argv rules; the EXT message kept, field `job.<request
    field>`), an unknown adapter, a GPU job without a positive estimate, a time limit or attempt count below 1, a label
    that is not a plain name of at most 64 characters. `fresh_output` False drops the empty-output rule (EXT-63), for
    the job a re-submit reattaches to."""
    from ..ext import registry

    out = _shape_problems(job)
    if out:
        return out
    try:
        ad = registry.adapter(job.adapter)
    except KeyError:
        return [_problem("job.adapter", f"no ext adapter {job.adapter!r} is registered",
                         f"use one of {', '.join(sorted(registry.ADAPTERS)) or 'the installed adapters'}")]
    acfg = launch.adapter_config(cfg, job.adapter)
    try:
        req = job_request(job, acfg)
    except ValueError as exc:                                # container_user: the owner is no user of this host
        return [_problem("job.user", str(exc), "run the backend as a user of this host")]
    for f, msg in launch.isolation_problems(req, kind="request", framework=ad.framework_name, config=acfg,
                                            socket_gids=socket_gids, fresh_output=fresh_output):
        out.append(_problem(f"job.{f}", msg))
    binding, why = launch.bind_argv(req["argv"], req["mounts"], interpreter=acfg.interpreter,
                                    trees=acfg.interpreter_trees, image_interpreters=acfg.image_interpreters,
                                    interpreter_flags=acfg.interpreter_flags)
    if binding is None:
        out.append(_problem("job.argv", why))
    out += [Problem(RULE, p.field, p.message, p.fix)                                          # EXT-190 to EXT-193
            for p in launch.adapter_launch_problems(req, adapter=job.adapter, kind="request", flag="job", fix=FIX)]
    if hasattr(ad, "argv_problems"):                                                          # EXT-133, EXT-156
        out += [_problem("job.argv", msg) for msg in ad.argv_problems(req["argv"], profile=job.profile)]
    return out


@dataclass(frozen=True)
class JobStatus:
    job_id: str
    state: str
    attempt: int
    rc: int | None
    outcome: str | None
    gpu_seconds: int
    seconds: int
    reason: str | None


class _BudgetRefused(Refused):
    """A start refused by the GPU budget (AN-23): the job ended `budget_exhausted` (its `refused` row is written)."""


@dataclass(frozen=True)
class _View:
    status: JobStatus
    pending: bool                   # an attempt is due to start (the first, or a retry)
    stop_reason: str | None         # the current attempt's `stop` row reason
    limit: tuple[int, str] | None   # the current attempt's run-time limit in seconds and its stop reason


def _summarize(job_id: str, max_attempts: int, rows: list[dict]) -> _View:
    """The job's state from its attempt rows alone (no runtime). A `refused` row (a start refused for good) ends the
    job: `budget_exhausted` for the budget (AN-23), else `failed`."""
    starts = [r for r in rows if r.get("event") == "start"]
    ends = {r["attempt"]: r for r in rows if r.get("event") == "end"}
    cancel = any(r.get("event") == "cancel_requested" for r in rows)
    refused = [r for r in rows if r.get("event") == "refused"]
    attempt = max((r["attempt"] for r in starts), default=0)
    stops = [r for r in rows if r.get("event") == "stop" and r.get("attempt") == attempt]
    stop_reason = stops[-1]["reason"] if stops else None
    current = [r for r in starts if r["attempt"] == attempt]
    limit = None
    if current and _int(current[-1].get("limit_s")) and current[-1].get("limit_reason") in STOP_REASONS:
        limit = (current[-1]["limit_s"], current[-1]["limit_reason"])
    gpu = sum(int(e.get("gpu_seconds") or 0) for e in ends.values())
    secs = sum(int(e.get("seconds") or 0) for e in ends.values())
    last = ends.get(attempt)
    rc = last.get("rc") if last else None
    outcome = last.get("outcome") if last else None
    pending = False
    reason = None
    if attempt == 0 and cancel:
        state, reason = "cancelled", "cancelled before it started"
    elif refused and (attempt == 0 or last is not None):
        state = "budget_exhausted" if refused[-1].get("rule") == RULE_GPU else "failed"
        reason = f"attempt {refused[-1].get('attempt')} was refused: {refused[-1].get('reason')}"
    elif attempt == 0:
        state, pending = "queued", True
    elif last is None:
        state = "running"
    elif outcome in RETRIED and attempt < max_attempts:
        state, pending = ("cancelled", False) if cancel else ("queued", True)
        reason = f"attempt {attempt} ended {outcome}" + ("" if cancel else "; it is retried")
    else:
        state = OUTCOME_STATE.get(outcome, "failed")
        if state != "succeeded":
            reason = f"attempt {attempt} ended {outcome}" + (f" (rc {rc})" if rc is not None else "")
    return _View(JobStatus(job_id, state, attempt, rc, outcome, gpu, secs, reason), pending, stop_reason, limit)


def _outcome(st: ContainerState, cancel: bool, stop_reason: str | None) -> str:
    if st.status == "missing":
        return "cancelled" if cancel else "lost"
    if st.exit_code == 0:
        return "ok"
    if st.oom_killed:
        return "oom"
    if cancel:
        return "cancelled"
    if stop_reason in STOP_REASONS:
        return stop_reason
    if st.exit_code in RC_SIGNALLED:
        return "preempted"
    return "failed"


def _seconds(started: str | None, finished: str | None) -> int:
    if not (started and finished):
        return 0
    a = datetime.strptime(started, TIME_FORMAT)
    b = datetime.strptime(finished, TIME_FORMAT)
    return max(0, int((b - a).total_seconds()))


def _sha256_regular(path: str, *, follow: bool = False) -> str | None:
    """The sha256 of a regular file, None when what the open found is not one. Opened non-blocking (a file swapped for
    a FIFO after a check does not block the open) and checked with `fstat`; `follow` False refuses a link (ELOOP)."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | (0 if follow else os.O_NOFOLLOW))
    with os.fdopen(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        h = hashlib.sha256()
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _sha256_nofollow(path: str, *, field: str = "path") -> str:
    """The sha256 of the regular file `path` (no link followed); Refused (AN-22) when it is not a regular file."""
    got = _sha256_regular(path)
    if got is None:
        raise Refused([_problem(field, f"{path} is not a regular file (it changed after it was checked)",
                                "write regular files only, and do not change them while they are read")])
    return got


def _sha256_bound(path: str, field: str) -> str:
    """The sha256 of a file the argv binding resolved (links followed, a FIFO never blocks); Refused (AN-22) when it is
    not a regular file."""
    got = _sha256_regular(path, follow=True)
    if got is None:
        raise Refused([_problem(field, f"{path} is not a regular file", "name a regular script and interpreter")])
    return got


def _start_row(rows: list[dict], attempt: int) -> dict | None:
    """The last `start` row of `attempt`."""
    found = [r for r in rows if r.get("event") == "start" and r.get("attempt") == attempt]
    return found[-1] if found else None


def _record_argv(argv: list[str]) -> list[str]:
    """The docker argument list with each `-e NAME=value` replaced by `-e NAME` (MM-35)."""
    out = list(argv)
    for i in range(len(out) - 1):
        if out[i] == "-e":
            out[i + 1] = out[i + 1].split("=", 1)[0]
    return out


def _table_open(text: str) -> bool:
    """True when the last line of `text` is a row of a table with a `Running total` column (`launch.gpu_used_h`'s
    reading), so a row appended after it counts."""
    col = False
    for line in text.splitlines():
        cells = line.strip()
        if not cells.startswith("|"):
            col = False
        elif not col:
            col = "Running total" in [c.strip() for c in cells.strip("|").split("|")]
    return col


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------

class LocalBackend:
    """AN-22: one project's jobs on one daemon. Nothing is read or started when it is constructed: the runtime
    (default `DockerCLIRuntime` on the context `gatedtrain`) and the host layout (default `launch.host_config()`) are
    built at first use. `probe`, `clock` and `sleep` serve the GPU wait, the attempt times and `wait`."""

    def __init__(self, project: Project, runtime: Runtime | None = None, *, host: HostConfig | None = None,
                 probe: GpuProbe | None = None, clock: Callable[[], datetime] = _utc_now,
                 sleep: Callable[[float], None] = time.sleep,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self.project = project
        self._runtime = runtime
        self._host = host
        self.probe = probe
        self.clock = clock
        self.sleep = sleep
        self.monotonic = monotonic
        self._jobs: dict[str, Job] = {}         # jobs this backend submitted: their environment values start them
        self._pid: str | None = None

    # -- lazily built ------------------------------------------------------------------------------------------------

    @property
    def runtime(self) -> Runtime:
        if self._runtime is None:
            self._runtime = DockerCLIRuntime(context=launch.DOCKER_CONTEXT)
        return self._runtime

    def host(self) -> HostConfig:
        if self._host is None:
            self._host = launch.host_config()
        return self._host

    def project_hex(self) -> str:
        if self._pid is None:
            self._pid = project_id(self.project)[len("sha256:"):]
        return self._pid

    def _utc(self) -> str:
        return self.clock().strftime(TIME_FORMAT)

    # -- records -----------------------------------------------------------------------------------------------------

    def _rel(self, *parts: str) -> str:
        return "/".join((JOBS_DIR,) + parts)

    def _jobs_dir(self) -> Path:
        _refuse_links(self.project.root, JOBS_DIR)
        d = self.project.root / JOBS_DIR
        d.mkdir(exist_ok=True)
        return d

    @contextlib.contextmanager
    def _lock(self, rel: str) -> Iterator[None]:
        _refuse_links(self.project.root, rel)
        fd = os.open(self.project.root / rel, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def _append(self, job_id: str, event: str, **fields: Any) -> None:
        row = {"event": event, **fields, "utc": self._utc(), "schema": JOB_ATTEMPT_SCHEMA,
               "schema_version": SCHEMA_VERSION}
        append_jsonl(self.project.root, self._rel(job_id, ATTEMPTS), row)

    def _rows(self, job_id: str) -> list[dict]:
        return read_jsonl(self.project.root, self._rel(job_id, ATTEMPTS))

    def _record(self, job_id: str) -> dict[str, Any]:
        if not (isinstance(job_id, str) and JOB_ID_RE.fullmatch(job_id)):
            raise Refused([_problem("job_id", f"{job_id!r} is not a job id (<seq>-<label>)",
                                    "use an id `submit` or the job list returned")])
        rel = self._rel(job_id, JOB_FILE)
        _refuse_links(self.project.root, rel)
        try:
            rec = json.loads((self.project.root / rel).read_bytes().decode("utf-8"))
        except FileNotFoundError:
            raise Refused([_problem("job_id", f"the project has no job {job_id}",
                                    "use an id `submit` or the job list returned")]) from None
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise Refused([_problem(rel, f"cannot read the job record ({exc.__class__.__name__})",
                                    "the record was damaged; restore it from a copy")]) from None
        if not (isinstance(rec, dict) and rec.get("schema") == JOB_SCHEMA
                and rec.get("schema_version") == SCHEMA_VERSION and rec.get("job_id") == job_id
                and isinstance(rec.get("job"), dict)):
            raise Refused([_problem(rel, f"not a {JOB_SCHEMA} version {SCHEMA_VERSION} record of job {job_id}",
                                    "the record was damaged; restore it from a copy")])
        return rec

    def _view(self, job_id: str, rec: Mapping[str, Any]) -> _View:
        return _summarize(job_id, int(rec["job"]["max_attempts"]), self._rows(job_id))

    def _job_ids(self) -> list[str]:
        d = self.project.root / JOBS_DIR
        _refuse_links(self.project.root, JOBS_DIR)
        if not d.is_dir():
            return []
        ids = [e.name for e in os.scandir(d) if JOB_ID_RE.fullmatch(e.name) and e.is_dir(follow_symlinks=False)]
        return sorted(ids, key=lambda i: (int(i.split("-", 1)[0]), i))

    def _container(self, job_id: str, attempt: int) -> str:
        return f"gatedtrain-{self.project_hex()[:12]}-{job_id}-a{attempt}"

    @staticmethod
    def _attempt_dir(rec: Mapping[str, Any], attempt: int) -> str:
        return os.path.join(rec["job"]["output_dir"], f"a{attempt}")

    # -- GPU accounting, budget, GPU wait (AN-23) --------------------------------------------------------------------

    def _ledger(self, exclude: str | None = None) -> tuple[int, float]:
        """The GPU seconds every job's `end` rows hold, and the estimates (hours) of the queued or running GPU jobs
        other than `exclude`."""
        gpu_s, est = 0, 0.0
        for job_id in self._job_ids():
            if not (self.project.root / JOBS_DIR / job_id / JOB_FILE).exists():
                continue                                        # a submit cut short before its record
            rec = self._record(job_id)
            st = self._view(job_id, rec).status
            gpu_s += st.gpu_seconds
            if (job_id != exclude and rec["job"].get("gpu") is True and st.state in ("queued", "running")
                    and _num(rec["job"].get("gpu_hours_est"))):
                est += float(rec["job"]["gpu_hours_est"])
        return gpu_s, est

    def gpu_seconds(self) -> int:
        """AN-23: the project's GPU seconds, summed from its jobs' `end` rows."""
        return self._ledger()[0]

    def committed_gpu_hours(self) -> float:
        """AN-23: `gpu_seconds() / 3600` plus the estimates of every queued or running GPU job."""
        gpu_s, est = self._ledger()
        return gpu_s / 3600 + est

    def _budget_hours(self) -> float:
        b = self.project.budget.get("gpu_hours") if isinstance(self.project.budget, Mapping) else None
        if not (_num(b) and math.isfinite(b) and b >= 0):
            raise Refused([Problem(RULE_GPU, "budget.gpu_hours", f"the project's GPU budget {b!r} is not a number of "
                                   "hours", "set budget.gpu_hours in the project's project.yaml")])
        return float(b)

    def _budget_left_s(self, job: Job, *, exclude: str | None) -> tuple[Problem | None, int]:
        """The budget check of one GPU attempt start: committed = the recorded GPU seconds plus
        the estimates of the other queued or running GPU jobs; the start is refused when committed plus the job's own
        estimate exceeds the budget. Else no problem and the budget left in seconds (the attempt's GPU limit)."""
        budget = self._budget_hours()
        gpu_s, est = self._ledger(exclude)
        left_s = budget * 3600 - (gpu_s + est * 3600)
        if job.gpu_hours_est * 3600 > left_s + 1e-6:
            return Problem(RULE_GPU, "job.gpu_hours_est", f"the job's estimate of {job.gpu_hours_est:g} GPU-hours "
                           f"exceeds the GPU budget left: the budget is {budget:g} GPU-hours and "
                           f"{(gpu_s + est * 3600) / 3600:.2f} are committed (recorded GPU time and the estimates of "
                           "the project's other queued or running GPU jobs)",
                           "raise budget.gpu_hours in project.yaml, or lower the job's cost"), max(0, int(left_s))
        return None, int(math.floor(left_s + 1e-6))

    def _wait_gpu(self) -> None:
        """AN-23: wait for a free GPU (no probe: no wait); a GPU still busy at GPU_WAIT_S, or no GPU tool, refuses the
        start; the job stays queued."""
        if self.probe is None:
            return
        if not wait_for_gpu(self.probe, sleep=self.sleep, clock=self.monotonic):
            raise Refused([Problem(RULE_GPU, "gpu", f"the GPU is busy: compute processes held more than {GPU_BUSY_MIB} "
                                   f"MiB through {GPU_WAIT_S} s of waiting, or this host has no nvidia-smi to read it",
                                   "let the other GPU work end, then poll or submit the job again (it stays queued)")])

    # -- the host's GPU log (AN-23) ----------------------------------------------------------------------------------

    def _gpu_log_path(self) -> str:
        return os.path.join(self.host().host_work, launch.GPU_LOG_MD)

    @staticmethod
    def _gpu_log_refused(path: str, why: str) -> Refused:
        return Refused([Problem(RULE_GPU, "gpu_log", f"the GPU log {path} {why}",
                                "fix the log by hand: its last table row needs a numeric `Running total` "
                                "(docs/gpu-step.md); the backend never resets it")])

    def _gpu_log_total(self, fd: int, path: str) -> float | None:
        """The log's running total (`launch.gpu_used_h`), None for an empty file (one created and cut short before
        its first write, which writes the header and the first row at once)."""
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise self._gpu_log_refused(path, "is not a regular file")
        if st.st_size == 0:
            return None
        try:
            return launch.gpu_used_h(Path(f"/dev/fd/{fd}"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise self._gpu_log_refused(path, f"does not parse ({exc.__class__.__name__}: no table row with a numeric "
                                        "`Running total`)") from None

    def _gpu_log_check(self) -> None:
        """Before a new job: an existing log must parse, so a job's end never fails on it after GPU time was spent."""
        path = self._gpu_log_path()
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise self._gpu_log_refused(path, f"cannot be opened ({exc.__class__.__name__}; a symbolic link is "
                                        "refused)") from None
        try:
            self._gpu_log_total(fd, path)
        finally:
            os.close(fd)

    def _gpu_log(self, job_id: str, rec: Mapping[str, Any], status: JobStatus) -> None:
        """One row for a job that reached a terminal state, appended without following a link and under a lock on the
        log (it is the host's, shared by projects): the header and the first row in one write when the log is absent;
        after a log that does not end inside its `Running total` table (notes, headings, a blank line: free text,
        docs/gpu-step.md) a new table, header and row, so the row is the last one `launch.gpu_used_h` reads. An
        existing log that does not parse is refused, never reset. Once the row is written the job records a
        `gpu_log_written` row in its own attempts.jsonl; a job with that row never opens the host's log again (job ids
        repeat across projects, and the log may have been archived or edited since), so a crash between the append
        and the record is the only way a job gets a second row."""
        rows = self._rows(job_id)
        if any(r.get("event") == "gpu_log_written" for r in rows):
            return
        starts = [r for r in rows if r.get("event") == "start"]
        ends = [r for r in rows if r.get("event") == "end"]
        first = next((e["started_utc"] for e in ends if e.get("started_utc")), None) or (
            starts[0]["utc"] if starts else "-")
        last = (ends[-1].get("finished_utc") or ends[-1].get("utc")) if ends else "-"
        hours = status.gpu_seconds / 3600
        rc = "-" if status.rc is None else status.rc
        path = self._gpu_log_path()
        try:
            fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o644)
        except OSError as exc:
            raise self._gpu_log_refused(path, f"cannot be opened ({exc.__class__.__name__}; a symbolic link is "
                                        "refused)") from None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            before = self._gpu_log_total(fd, path)
            size = os.fstat(fd).st_size
            text = os.pread(fd, size, 0).decode("utf-8", "replace") if size else ""
            row = (f"| {job_id} ({rec['job']['label']}), {len(starts)} attempt(s) | {first} | {last} | {hours:.2f} "
                   f"| {(before or 0.0) + hours:.2f} | {status.state.upper()} rc={rc} |\n")
            if before is None:
                data = GPU_LOG_HEADER + row
            else:
                sep = "" if text.endswith("\n") else "\n"
                data = sep + (row if _table_open(text) else "\n" + GPU_LOG_HEADER + row)
            os.write(fd, data.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        self._append(job_id, "gpu_log_written", state=status.state)

    # -- checks before a start ---------------------------------------------------------------------------------------

    def _daemon_and_image(self, job: Job, cfg: HostConfig) -> None:
        """AN-22 before any start: on a rootless host the daemon is rootless and its socket the owner's; the
        daemon holds the image (no run pulls one). A daemon error raises RuntimeFailure."""
        problems = []
        if cfg.rootless:
            info = self.runtime.daemon()
            uid = launch.owner_uid(cfg)
            if not info.rootless:
                problems.append(_problem("daemon", "the Docker daemon the backend reaches is not rootless, while the "
                                         "host file says `rootless = true`: a container would run as root on the "
                                         "host", "point the context `gatedtrain` at the owner's rootless daemon "
                                         f"(unix://{launch.rootless_socket(cfg)})"))
            elif info.socket_uid != uid:
                problems.append(_problem("daemon", f"the daemon's socket {info.socket or '(not a unix socket)'} is "
                                         f"owned by uid {info.socket_uid}, not the owner's uid {uid}",
                                         "point the context `gatedtrain` at the owner's rootless daemon "
                                         f"(unix://{launch.rootless_socket(cfg)})"))
        image = job.runtime["image"]
        if self.runtime.image_id(image) is None:
            problems.append(_problem("job.runtime.image", f"the daemon does not hold the image {image}; no job pulls "
                                     "an image", f"pull or build it first: docker --context {launch.DOCKER_CONTEXT} "
                                     f"pull {image}"))
        if problems:
            raise Refused(problems)

    def _spec(self, job: Job, cfg: HostConfig, job_id: str, attempt: int, out_dir: str) -> RunSpec:
        """The container of one attempt: the request's options as ext-launch prints them, with the adapter's run
        options (EXT-191) and the attempt's own output directory."""
        from ..ext import registry

        argv = list(job.argv)
        script = (posixpath.basename(argv[launch.script_index(argv, registry.interpreter_flags(job.adapter))])
                  if len(argv) > 1 else "")
        options, tmpfs_options = registry.run_options(job.adapter, script)
        shm = [o[len("--shm-size="):] for o in options if o.startswith("--shm-size=")]
        return RunSpec(name=self._container(job_id, attempt), image=job.runtime["image"], argv=tuple(argv),
                       env=tuple(sorted(job.env.items())),
                       mounts=tuple(job.mounts) + (Mount(out_dir, job.output_container, read_only=False),),
                       user=launch.container_user(cfg), network=job.network,
                       gpu_options=launch.gpu_options(cfg.rootless) if job.gpu else (),
                       tmpfs=tuple((t, tmpfs_options or launch.TMPFS_OPTIONS) for t in job.tmpfs),
                       shm_size=shm[0] if shm else launch.SHM_SIZE[len("--shm-size="):],
                       labels=((LABEL_JOB, job_id), (LABEL_PROJECT, self.project_hex())),
                       extra_options=tuple(o for o in options if not o.startswith("--shm-size=")))

    def _fresh_attempt_dir(self, path: str, cfg: HostConfig) -> None:
        """`<output dir>/a<n>`: created (owner, 0755), or an existing empty owner directory (a start cut short)."""
        try:
            os.mkdir(path, DIR_MODE)
            os.chmod(path, DIR_MODE)
        except FileExistsError:
            pass
        st = os.lstat(path)
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != launch.owner_uid(cfg) or os.listdir(path):
            raise Refused([_problem("job.output_dir", f"the attempt directory {path} exists and is not an empty "
                                    "directory of the owner; an attempt writes into a fresh directory",
                                    "remove it, or submit the job with a new output directory")])

    # -- submit ------------------------------------------------------------------------------------------------------

    def _find(self, key: str) -> str | None:
        """The newest job of the project with this key that is queued, running or succeeded."""
        for job_id in reversed(self._job_ids()):
            try:
                rec = self._record(job_id)
            except Refused:
                continue
            if rec.get("key") == key and self._view(job_id, rec).status.state in ("queued", "running", "succeeded"):
                return job_id
        return None

    def submit(self, job: Job, *, stage: str = "") -> str:
        """AN-22: the job's id. A job of this project with the same key that is queued, running or succeeded is
        returned after its record is checked against the launch rules (without the fresh-output rule) and nothing
        starts; else the job is checked (`job_problems`), the daemon and image are checked, the record is written and
        attempt 1 starts. A GPU job over the budget (AN-23) or a GPU log that does not parse is refused before its
        record."""
        problems = _shape_problems(job)
        if problems:
            raise Refused(problems)
        cfg = launch.adapter_config(self.host(), job.adapter)
        key = job.key()
        self._jobs_dir()
        with self._lock(self._rel(LOCK)):
            found = self._find(key)
            if found is not None:                                    # reattach
                rec = self._record(found)
                problems = job_problems(job, cfg, fresh_output=False)
                if rec["job"] != _record_doc(job):
                    problems.append(_problem(f"{JOBS_DIR}/{found}/{JOB_FILE}", "the job record does not match the "
                                             "job of its key: the record was changed",
                                             "restore the record, or submit the job with a new output directory"))
                for k, now in (("rootless", cfg.rootless), ("host_config_sha256", launch.host_config_sha256(cfg))):
                    if rec.get(k) != now:
                        problems.append(_problem(f"{JOBS_DIR}/{found}/{JOB_FILE}.{k}", f"the job was recorded with "
                                                 f"{k} {rec.get(k)!r}, this host has {now!r}",
                                                 "run the job on the host layout it was submitted under"))
                if problems:
                    raise Refused(problems)
                self._jobs[found] = job
                return found
            problems = job_problems(job, cfg)
            if job.gpu:
                problems += [p for p in (self._budget_left_s(job, exclude=None)[0],) if p is not None]
            if problems:
                raise Refused(problems)
            self._gpu_log_check()
            self._daemon_and_image(job, cfg)
            job_id = self._new_job(job, cfg, key, stage)
        self._jobs[job_id] = job
        with self._lock(self._rel(job_id, LOCK)):
            # Re-read under the job's lock: another driver may have reattached and started attempt 1, or a cancel may
            # have come, between the record and this lock.
            self._advance(job_id, self._record(job_id), key=key, raise_budget=True)
        return job_id

    def _new_job(self, job: Job, cfg: HostConfig, key: str, stage: str) -> str:
        ids = self._job_ids()
        seq = max((int(i.split("-", 1)[0]) for i in ids), default=0) + 1
        job_id = f"{seq:04d}-{job.label}"
        jdir = self.project.root / JOBS_DIR / job_id
        os.mkdir(jdir, DIR_MODE)
        binding, _ = launch.bind_argv(list(job.argv), job_request(job, cfg)["mounts"], interpreter=cfg.interpreter,
                                      trees=cfg.interpreter_trees, image_interpreters=cfg.image_interpreters,
                                      interpreter_flags=cfg.interpreter_flags)
        # The record is written before `a1` exists (the start creates it), so a crash leaves a queued job, never an
        # output directory without a record; the recorded `-v` names `a1` under the output's resolved path.
        spec = self._spec(job, cfg, job_id, 1, job.output_dir)
        real_out = launch.mount_source(job.output_dir)
        argv = [f"{real_out}/a1{t[len(real_out):]}" if t == f"{real_out}:{job.output_container}:rw" else t
                for t in docker_run_argv(spec, context=launch.DOCKER_CONTEXT)]
        rec = {"schema": JOB_SCHEMA, "schema_version": SCHEMA_VERSION, "job_id": job_id, "key": key, "stage": stage,
               "job": _record_doc(job), "rootless": cfg.rootless, "host_config_sha256": launch.host_config_sha256(cfg),
               "script_sha256": _sha256_bound(binding.script_host, "job.argv"),
               "interpreter_sha256": (_sha256_bound(binding.interpreter_host, "job.argv")
                                      if binding.interpreter_host is not None else None),
               "docker_argv": _record_argv(argv),
               "utc": self._utc()}
        rel = self._rel(job_id, JOB_FILE)
        _refuse_links(self.project.root, rel)
        _write_atomic(self.project.root / rel, canonical_json(rec))
        return job_id

    # -- one attempt -------------------------------------------------------------------------------------------------

    def _start(self, job_id: str, rec: Mapping[str, Any], job: Job | None, attempt: int, *,
               key: str | None = None, checked: bool = False) -> None:
        """Start attempt `attempt` under the job's lock. A container of that attempt the daemon already holds (a
        start whose `start` row a crash cut off) is adopted when it is running or exited, and removed when it was only
        created; after a failed start, a container left only created is removed, one that ran is adopted. A GPU
        attempt passes the budget (else a `refused` row ends the job: `_BudgetRefused`) and waits for a free GPU; the
        budget is checked again, and the container started, under the project's job lock, so two jobs never both take
        the last of it. `key` is the job's key when the caller just computed it."""
        if job is None:
            raise Refused([_problem("job_id", f"job {job_id} has an attempt due, and its environment values are "
                                    "never recorded", "submit the job again (the same job reattaches) and poll it")])
        if (key or job.key()) != rec["key"]:
            why = "the inputs changed since the job was recorded"
            name = self._container(job_id, attempt)
            st = self.runtime.inspect(name)
            if st.status in ("running", "exited"):          # a start a crash cut off: stop it and record it first
                self._append(job_id, "start", attempt=attempt, container=name, name=name,
                             output=self._attempt_dir(rec, attempt))
                if st.status == "running":
                    self.runtime.stop(name)
                    st = self.runtime.inspect(name)
                rows = self._rows(job_id)
                self._end(job_id, rec, _summarize(job_id, int(rec["job"]["max_attempts"]), rows), st, rows)
                attempt += 1
            self._append(job_id, "refused", attempt=attempt, rule=RULE, reason=why)
            raise Refused([_problem("job", f"job {job_id}: {why}; it ends here, its attempt {attempt} is not started",
                                    "submit the job again: with the new inputs it is a new job")])
        cfg = launch.adapter_config(self.host(), job.adapter)
        if not checked:
            self._daemon_and_image(job, cfg)
        name = self._container(job_id, attempt)
        out_dir = self._attempt_dir(rec, attempt)
        st = self.runtime.inspect(name)
        if st.status in ("running", "exited"):
            self._append(job_id, "start", attempt=attempt, container=name, name=name, output=out_dir,
                         **self._limit(job, job_id))
            return
        if st.status == "created":
            self.runtime.remove(name)
        if job.gpu:
            self._budget_gate(job_id, job, attempt)
            self._wait_gpu()
        with self._lock(self._rel(LOCK)):
            limit = self._budget_gate(job_id, job, attempt) if job.gpu else self._limit(job, job_id)
            self._fresh_attempt_dir(out_dir, cfg)
            spec = self._spec(job, cfg, job_id, attempt, out_dir)
            try:
                cid = self.runtime.run(spec)
            except RuntimeFailure:
                with contextlib.suppress(RuntimeFailure):
                    left = self.runtime.inspect(name)
                    if left.status in ("running", "exited"):        # it ran after all (a CLI timeout): adopt
                        self._append(job_id, "start", attempt=attempt, container=name, name=name, output=out_dir,
                                     **limit)
                        return
                    if left.status == "created":
                        self.runtime.remove(name)
                raise
            self._append(job_id, "start", attempt=attempt, container=cid, name=name, output=out_dir, **limit)

    def _limit(self, job: Job, job_id: str, left_s: int | None = None) -> dict[str, Any]:
        """The attempt's run-time limit for its `start` row: the time limit, or for a GPU attempt the budget left when
        it starts when that comes first."""
        if job.gpu and left_s is None:
            left_s = self._budget_left_s(job, exclude=job_id)[1]
        if job.gpu and left_s is not None and left_s < job.time_limit_s:
            return {"limit_s": left_s, "limit_reason": "budget_exhausted"}
        return {"limit_s": job.time_limit_s, "limit_reason": "timed_out"}

    def _budget_gate(self, job_id: str, job: Job, attempt: int) -> dict[str, Any]:
        problem, left_s = self._budget_left_s(job, exclude=job_id)
        if problem is not None:
            self._append(job_id, "refused", attempt=attempt, rule=RULE_GPU, reason=problem.message)
            raise _BudgetRefused([problem])
        return self._limit(job, job_id, left_s)

    def _end(self, job_id: str, rec: Mapping[str, Any], view: _View, st: ContainerState, rows: list[dict]) -> None:
        """Record the end of the current attempt once (under the job's lock): its outcome, times and the tail of its
        log; then remove the container."""
        attempt = view.status.attempt
        name = self._container(job_id, attempt)
        cancel = any(r.get("event") == "cancel_requested" for r in rows)
        outcome = _outcome(st, cancel, view.stop_reason)
        if st.status != "missing":
            log = self.runtime.logs(name)
            rel = self._rel(job_id, f"attempt-{attempt}.log")
            _refuse_links(self.project.root, rel)
            _write_atomic(self.project.root / rel, log[-LOG_CAP_BYTES:])
        seconds = _seconds(st.started_utc, st.finished_utc)
        if st.status == "missing":
            # No container times: the attempt counts from its `start` row up to now, at most its limit, so a lost GPU
            # attempt is not free and its retries cannot overspend the budget.
            limit_s = view.limit[0] if view.limit else int(rec["job"]["time_limit_s"])
            seconds = min(int(self._elapsed(_start_row(rows, attempt))), limit_s)
        self._append(job_id, "end", attempt=attempt, outcome=outcome, rc=st.exit_code, started_utc=st.started_utc,
                     finished_utc=st.finished_utc, seconds=seconds,
                     gpu_seconds=seconds if rec["job"]["gpu"] else 0)
        if st.status != "missing":
            with contextlib.suppress(RuntimeFailure):                # the end is recorded; a leftover is harmless
                self.runtime.remove(name)

    def _advance(self, job_id: str, rec: Mapping[str, Any], *, was: str | None = None, key: str | None = None,
                 raise_budget: bool = False) -> JobStatus:
        """Under the job's lock: start a due attempt, or inspect the running one, stop it past its limit and record its
        end (then start a retry when one is due). A RuntimeFailure leaves the records as they were. A job that reaches
        a terminal state here (from `was`, default its state on entry) gets its GPU log row; a job already terminal
        gets it only when it has no `gpu_log_written` row (a row a crash cut off)."""
        view = self._view(job_id, rec)
        if (was or view.status.state) in TERMINAL:
            if view.status.state in TERMINAL:
                self._gpu_log(job_id, rec, view.status)
            return view.status
        try:
            status = self._step(job_id, rec, view, key=key, raise_budget=raise_budget)
        except Refused:
            after = self._view(job_id, rec).status
            if after.state in TERMINAL:
                self._gpu_log(job_id, rec, after)
            raise
        if status.state in TERMINAL:
            self._gpu_log(job_id, rec, status)
        return status

    def _step(self, job_id: str, rec: Mapping[str, Any], view: _View, *, key: str | None,
              raise_budget: bool) -> JobStatus:
        if view.status.state in TERMINAL:
            return view.status
        if view.status.state == "running":
            name = self._container(job_id, view.status.attempt)
            st = self.runtime.inspect(name)
            reason = self._due_stop(rec, view, st) if st.status == "running" else None
            if reason is not None:
                if view.stop_reason is None:
                    self._append(job_id, "stop", attempt=view.status.attempt, reason=reason)
                self.runtime.stop(name)
                view = self._view(job_id, rec)
                st = self.runtime.inspect(name)
            if st.status in ("running", "created"):
                return view.status
            self._end(job_id, rec, view, st, self._rows(job_id))
            view = self._view(job_id, rec)
        if view.pending:
            try:
                self._start(job_id, rec, self._jobs.get(job_id), view.status.attempt + 1, key=key)
            except _BudgetRefused:
                if raise_budget:
                    raise
            view = self._view(job_id, rec)
        return view.status

    def _elapsed(self, start_row: Mapping[str, Any] | None) -> float:
        """Seconds from a `start` row's time to now (0 without one)."""
        if not (start_row and isinstance(start_row.get("utc"), str)):
            return 0.0
        started = datetime.strptime(start_row["utc"], TIME_FORMAT).replace(tzinfo=timezone.utc)
        return max(0.0, (self.clock() - started).total_seconds())

    def _budget_remaining_s(self) -> float:
        """The project's GPU budget minus its recorded GPU seconds and the time every running GPU attempt has run so
        far (from its `start` row): what all running GPU attempts together may still use."""
        used = 0.0
        for job_id in self._job_ids():
            if not (self.project.root / JOBS_DIR / job_id / JOB_FILE).exists():
                continue
            rec = self._record(job_id)
            rows = self._rows(job_id)
            st = _summarize(job_id, int(rec["job"]["max_attempts"]), rows).status
            used += st.gpu_seconds
            if rec["job"].get("gpu") is True and st.state == "running":
                used += self._elapsed(_start_row(rows, st.attempt))
        return self._budget_hours() * 3600 - used

    def _due_stop(self, rec: Mapping[str, Any], view: _View, st: ContainerState) -> str | None:
        """The reason to stop the running attempt now, else None: a stop already recorded (due again), its own limit
        (the time limit, or the budget left when it started), or for a GPU attempt the project's budget used up by
        the recorded time and every running GPU attempt (checked at each poll: enforcement is per poll)."""
        if view.stop_reason is not None:
            return view.stop_reason
        if st.started_utc:
            limit_s, reason = view.limit or (int(rec["job"]["time_limit_s"]), "timed_out")
            started = datetime.strptime(st.started_utc, TIME_FORMAT).replace(tzinfo=timezone.utc)
            if (self.clock() - started).total_seconds() >= limit_s:
                return reason
        if rec["job"].get("gpu") is True and self._budget_remaining_s() <= 0:
            return "budget_exhausted"
        return None

    # -- public operations -------------------------------------------------------------------------------------------

    def status(self, job_id: str) -> JobStatus:
        rec = self._record(job_id)
        with self._lock(self._rel(job_id, LOCK)):
            return self._advance(job_id, rec)

    def wait(self, job_id: str, *, poll_s: float = JOB_POLL_S) -> JobStatus:
        while True:
            s = self.status(job_id)
            if s.state in TERMINAL:
                return s
            self.sleep(poll_s)

    def logs(self, job_id: str, *, tail: int | None = 200) -> str:
        """The current attempt's log: from the daemon while it runs, from the saved tail once its end is recorded."""
        if tail is not None and not (_int(tail) and tail >= 0):
            raise Refused([_problem("tail", f"{tail!r} is not a count of lines", "pass a count of lines")])
        rec = self._record(job_id)
        rows = self._rows(job_id)
        attempt = _summarize(job_id, int(rec["job"]["max_attempts"]), rows).status.attempt
        if attempt == 0:
            return ""
        if any(r.get("event") == "end" and r.get("attempt") == attempt for r in rows):
            rel = self._rel(job_id, f"attempt-{attempt}.log")
            _refuse_links(self.project.root, rel)
            try:
                data = (self.project.root / rel).read_bytes()
            except FileNotFoundError:
                data = b""
            if tail is not None:
                lines = data.splitlines(keepends=True)
                data = b"".join(lines[-tail:]) if tail else b""
        else:
            data = self.runtime.logs(self._container(job_id, attempt), tail=tail)
        return data.decode("utf-8", "replace")

    def cancel(self, job_id: str) -> JobStatus:
        """Record `cancel_requested`, then stop the running container; the end is recorded `cancelled`."""
        rec = self._record(job_id)
        with self._lock(self._rel(job_id, LOCK)):
            view = self._view(job_id, rec)
            if view.status.state in TERMINAL:
                return view.status
            self._append(job_id, "cancel_requested")
            if view.status.state == "running":
                self.runtime.stop(self._container(job_id, view.status.attempt))
            return self._advance(job_id, rec, was=view.status.state)

    def jobs(self) -> list[JobStatus]:
        """Every job's state from its records alone (no runtime)."""
        out = []
        for job_id in self._job_ids():
            if not (self.project.root / JOBS_DIR / job_id / JOB_FILE).exists():
                continue                                        # a submit cut short before its record
            rec = self._record(job_id)
            out.append(self._view(job_id, rec).status)
        return out

    # -- outputs -----------------------------------------------------------------------------------------------------

    def new_output_dir(self, name: str, *, key: str | None = None) -> Path:
        """`<output_root>/gatedtrain-<project id prefix>/<name>` (owner, 0755).

        Without `key` (a job's inputs directory, which the stage rewrites): created, or returned again when it is a
        plain owner directory this project created. With `key` (`Job.key()` of the job the directory is the output
        of): the directory recorded for this name and key is returned again, so a stage re-run of the
        same job reattaches, unless every job of that key that used it ended without success (failed, cancelled,
        timed_out, budget_exhausted): a re-run then gets a fresh directory like a job with another key, `<name>` when
        free, else `<name>.<n>`. Anything else is refused."""
        if not (isinstance(name, str) and NAME_RE.fullmatch(name)):
            raise Refused([_problem("name", f"{name!r} is not a plain name (letters, digits, _ . -; first a letter or "
                                    "digit; at most 100 characters)", "pass a plain directory name")])
        if key is not None and not (isinstance(key, str) and launch.SHA_RE.fullmatch(key)):
            raise Refused([_problem("key", f"{key!r} is not a job key (sha256:<64 hex>)", "pass Job.key()")])
        cfg = self.host()
        uid = launch.owner_uid(cfg)
        root = cfg.output_root
        if not os.path.isdir(root):
            raise Refused([_problem("output_root", f"the output root {root} is not a directory",
                                    "create it, or set `output_root` in the host file")])
        base = os.path.join(root, f"gatedtrain-{self.project_hex()[:12]}")
        self._jobs_dir()
        rel = self._rel(OUTPUT_DIRS)

        def plain(path: str) -> bool:
            st = os.lstat(path)
            return stat.S_ISDIR(st.st_mode) and st.st_uid == uid

        def create(path: str) -> Path:
            os.mkdir(path, DIR_MODE)
            os.chmod(path, DIR_MODE)
            append_jsonl(self.project.root, rel, {"schema": OUTPUT_DIR_SCHEMA, "schema_version": SCHEMA_VERSION,
                                                  "name": name, "key": key, "path": path, "utc": self._utc()})
            return Path(path)

        with self._lock(self._rel(LOCK)):
            with contextlib.suppress(FileExistsError):
                os.mkdir(base, DIR_MODE)
                os.chmod(base, DIR_MODE)
            if not plain(base) or os.path.islink(base):
                raise Refused([_problem("output_root", f"{base} is not a directory of the owner (uid {uid})",
                                        "remove it; the backend creates it")])
            rows = read_jsonl(self.project.root, rel)
            if key is not None:
                for r in reversed(rows):
                    if r.get("name") == name and r.get("key") == key:
                        path = r.get("path")
                        if not (isinstance(path, str) and os.path.lexists(path) and plain(path)):
                            raise Refused([_problem("name", f"the output directory {path} of this job is gone or "
                                                    "not a plain directory of the owner", "restore or remove it")])
                        if self._spent(path, key):
                            break                       # a re-run after a failure: a fresh directory
                        return Path(path)
                n = 1
                while True:
                    path = os.path.join(base, name if n == 1 else f"{name}.{n}")
                    if not os.path.lexists(path):
                        return create(path)
                    n += 1
            target = os.path.join(base, name)
            if not os.path.lexists(target):
                return create(target)
            mine = any(r.get("path") == target and r.get("key") is None for r in rows)
            if not (mine and plain(target)):
                raise Refused([_problem("name", f"{target} exists and is not a plain directory of the owner that "
                                        "this project created for no job", "pick another name")])
            return Path(target)

    def _spent(self, path: str, key: str) -> bool:
        """Jobs of `key` used the output directory `path`, and all of them ended without success."""
        states = []
        for job_id in self._job_ids():
            if not (self.project.root / JOBS_DIR / job_id / JOB_FILE).exists():
                continue
            rec = self._record(job_id)
            if rec.get("key") == key and rec["job"].get("output_dir") == path:
                states.append(self._view(job_id, rec).status.state)
        return bool(states) and all(st in TERMINAL and st != "succeeded" for st in states)

    def _succeeded_dir(self, job_id: str, op: str) -> str:
        rec = self._record(job_id)
        view = self._view(job_id, rec)
        if view.status.state != "succeeded":
            raise Refused([_problem("job_id", f"job {job_id} is {view.status.state}; {op} reads a succeeded job's "
                                    "output only", "wait for the job to succeed")])
        return self._attempt_dir(rec, view.status.attempt)

    def artifacts(self, job_id: str) -> dict[str, Any]:
        """The succeeded attempt's files `{path, size, sha256}` sorted by path; a symbolic link or a non-regular file
        anywhere in the output is refused."""
        top = self._succeeded_dir(job_id, "artifacts")
        files: list[dict[str, Any]] = []
        problems: list[Problem] = []

        def walk(d: str, rel: str) -> None:
            with os.scandir(d) as it:
                entries = sorted(it, key=lambda e: e.name)
            for e in entries:
                r = f"{rel}{e.name}"
                if e.is_symlink():
                    problems.append(_problem(f"artifacts.{r}", "is a symbolic link; a job's output holds files and "
                                             "directories only", "write the file itself, not a link"))
                elif e.is_dir(follow_symlinks=False):
                    walk(e.path, r + "/")
                elif e.is_file(follow_symlinks=False):
                    files.append({"path": r, "size": e.stat(follow_symlinks=False).st_size,
                                  "sha256": _sha256_nofollow(e.path)})
                else:
                    problems.append(_problem(f"artifacts.{r}", "is not a regular file (a FIFO, socket or device)",
                                             "write regular files only"))

        st = os.lstat(top)
        if not stat.S_ISDIR(st.st_mode):
            raise Refused([_problem("artifacts", f"{top} is not a directory", "rerun the job")])
        walk(top, "")
        if problems:
            raise Refused(problems)
        return {"job_id": job_id, "output_dir": top, "files": sorted(files, key=lambda f: f["path"])}

    def hand_off(self, job_id: str, paths: list[str]) -> list[dict[str, str]]:
        """`[{path, sha256}]` for the model files a model reference lists, each checked readable by the keeper user:
        inside the succeeded attempt's directory, no symbolic link on the way, a regular file with `o+r`, and every
        directory from the output root down to it with `o+x`."""
        top = self._succeeded_dir(job_id, "hand_off")
        top_real = os.path.realpath(top)
        root_real = os.path.realpath(self.host().output_root)
        out: list[dict[str, str]] = []
        problems: list[Problem] = []
        fix = "make the file a regular file readable by other users, in directories they can enter (chmod o+r, o+x)"
        for i, p in enumerate(paths):
            f = f"hand_off.paths[{i}]"
            if not (isinstance(p, str) and p):
                problems.append(_problem(f, f"{p!r} is not a path", "pass the model files' paths"))
                continue
            path = os.path.normpath(p if os.path.isabs(p) else os.path.join(top, p))
            rel = os.path.relpath(path, os.path.normpath(top))
            if rel == "." or rel == ".." or rel.startswith("../"):
                problems.append(_problem(f, f"{p} is not inside the job's output {top}",
                                         "hand off files of the succeeded attempt only"))
                continue
            why = None
            cur = top_real
            parts = rel.split("/")
            for n, part in enumerate(parts):
                cur = os.path.join(cur, part)
                try:
                    st = os.lstat(cur)
                except OSError as exc:
                    why = f"{cur} cannot be read ({exc.__class__.__name__})"
                    break
                if stat.S_ISLNK(st.st_mode):
                    why = f"{cur} is a symbolic link"
                    break
                if n < len(parts) - 1 and not stat.S_ISDIR(st.st_mode):
                    why = f"{cur} is not a directory"
                    break
            else:
                if not stat.S_ISREG(st.st_mode):
                    why = f"{cur} is not a regular file"
                elif not st.st_mode & stat.S_IROTH:
                    why = f"{cur} is not readable by other users (mode {stat.S_IMODE(st.st_mode):04o}, no o+r)"
                else:
                    d = os.path.dirname(cur)
                    chain = [d]
                    while d != root_real and d.startswith(root_real + "/"):
                        d = os.path.dirname(d)
                        chain.append(d)
                    for d in reversed(chain):
                        dm = os.stat(d).st_mode
                        if not dm & stat.S_IXOTH:
                            why = f"the directory {d} cannot be entered by other users (mode {stat.S_IMODE(dm):04o}, " \
                                  "no o+x)"
                            break
            if why:
                problems.append(_problem(f, f"{p}: {why}; the keeper user must read the file", fix))
                continue
            out.append({"path": cur, "sha256": _sha256_nofollow(cur)})
        if problems:
            raise Refused(problems)
        return out



class StageBackend:
    """AN-24: what a pack stage sees as `ctx.backend`: the `LocalBackend` operations, its jobs recorded under the
    stage's ref name. Only a stage that declares `needs: gpu` (`allow_gpu`) submits a GPU job. During `plan`
    (`planning`) every operation that may change a record or a container is refused (`submit`, `new_output_dir`,
    `cancel`, and `status` and `wait`, which record an end and start a due retry); `jobs`, `logs`, `artifacts` and
    `hand_off` only read. The rule is cooperative: it holds for stages that use this object."""

    def __init__(self, backend: LocalBackend, *, stage: str, allow_gpu: bool, planning: bool = False) -> None:
        self._backend = backend
        self.stage = stage
        self.allow_gpu = bool(allow_gpu)
        self.planning = bool(planning)

    def _no_side_effect(self, op: str) -> None:
        if self.planning:
            raise Refused([Problem(RULE_STAGE, op, f"stage {self.stage!r}: `plan` has no side effects, so it cannot "
                                   f"call `{op}`", f"call `{op}` from the stage's `run`")])

    def submit(self, job: Job) -> str:
        self._no_side_effect("submit")
        if getattr(job, "gpu", False) is not False and not self.allow_gpu:
            raise Refused([Problem(RULE_STAGE, "job.gpu", f"stage {self.stage!r} does not declare `needs: gpu`; only "
                                   "a stage that declares `needs: gpu` submits GPU jobs",
                                   "declare `gpu` in the stage's `needs`, or submit a CPU job")])
        return self._backend.submit(job, stage=self.stage)

    def status(self, job_id: str) -> JobStatus:
        self._no_side_effect("status")
        return self._backend.status(job_id)

    def wait(self, job_id: str, *, poll_s: float = JOB_POLL_S) -> JobStatus:
        self._no_side_effect("wait")
        return self._backend.wait(job_id, poll_s=poll_s)

    def cancel(self, job_id: str) -> JobStatus:
        self._no_side_effect("cancel")
        return self._backend.cancel(job_id)

    def new_output_dir(self, name: str, *, key: str | None = None) -> Path:
        self._no_side_effect("new_output_dir")
        return self._backend.new_output_dir(name, key=key)

    def jobs(self) -> list[JobStatus]:
        return self._backend.jobs()

    def logs(self, job_id: str, *, tail: int | None = 200) -> str:
        return self._backend.logs(job_id, tail=tail)

    def artifacts(self, job_id: str) -> dict[str, Any]:
        return self._backend.artifacts(job_id)

    def hand_off(self, job_id: str, paths: list[str]) -> list[dict[str, str]]:
        return self._backend.hand_off(job_id, paths)
