"""A container runtime for tests (AN-22, AN-23, AN-25): the `Runtime` protocol without Docker, a Docker group or a GPU.

`FakeRuntime.run` executes the image's handler in this process, synchronously, unless the next scripted outcome says
otherwise, and keeps containers the way Docker does (a taken name, an unknown image under `--pull never`, removing a
running container and touching a missing one all fail with `RuntimeFailure`). With `state_dir`, containers, logs, the
outcome queue and the run log live in JSON files there, under a file lock, so the CLI subprocesses of one test share
them. `subprocess_handler` runs a spec's argv on the host with its container paths translated to the mounts' host paths.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from ..ainative.runtime import (TIME_FORMAT, ContainerState, DaemonInfo, Mount, RunSpec, RuntimeFailure,
                                docker_run_argv)
from ..ext import launch

Handler = Callable[[RunSpec, Callable[[str], Path]], int]
OUTCOMES = ("ok", "preempt", "oom", "hang", "vanish")
RC_KILLED, RC_STOPPED = 137, 143
STATE_FILE, RUNS_FILE, LOCK_FILE, LOG_DIR = "state.json", "runs.jsonl", "lock", "logs"


def _spec_json(spec: RunSpec) -> dict[str, Any]:
    return dataclasses.asdict(spec)


def _spec_from_json(d: Mapping[str, Any]) -> RunSpec:
    pairs = ("env", "tmpfs", "labels")
    kw = {k: (tuple(tuple(p) for p in v) if k in pairs else tuple(v) if isinstance(v, list) else v)
          for k, v in d.items() if k != "mounts"}
    return RunSpec(mounts=tuple(Mount(**m) for m in d["mounts"]), **kw)


def _host_path(spec: RunSpec, tmpfs_dirs: Mapping[str, Path]) -> Callable[[str], Path]:
    """Container path -> host path: the longest mount or tmpfs path equal to or above it; ValueError otherwise."""
    roots = [(m.container.rstrip("/") or "/", Path(os.path.realpath(m.host))) for m in spec.mounts]
    roots += [(p.rstrip("/") or "/", d) for p, d in tmpfs_dirs.items()]
    roots.sort(key=lambda r: len(r[0]), reverse=True)

    def host_path(cpath: str) -> Path:
        for root, host in roots:
            if cpath == root:
                return host
            if cpath.startswith(root.rstrip("/") + "/"):
                return host / cpath[len(root.rstrip("/")) + 1:]
        raise ValueError(f"{cpath!r} is under no mount or tmpfs of container {spec.name}")
    return host_path


def subprocess_handler(spec: RunSpec, host_path: Callable[[str], Path]) -> int:
    """Run the spec's argv on the host: `argv[0]` replaced by this interpreter, every argument equal to or under a
    mount's or tmpfs's container path translated by `host_path`, the environment the spec's plus `PATH`. The output
    (stdout and stderr, as `docker logs` gives both) goes to `sys.stdout`, which the fake runtime captures."""
    roots = [m.container for m in spec.mounts] + [p for p, _ in spec.tmpfs]

    def mapped(a: str) -> str:
        if any(a == r or a.startswith(r.rstrip("/") + "/") for r in roots):
            return str(host_path(a))
        return a
    argv = [sys.executable, *(mapped(a) for a in spec.argv[1:])]
    env = dict(spec.env)
    env["PATH"] = os.environ.get("PATH", os.defpath)
    proc = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                          check=False)
    sys.stdout.write(proc.stdout.decode("utf-8", "surrogateescape"))
    return proc.returncode


class FakeRuntime:
    """`Runtime` for tests. `handlers` maps an image (the spec's reference) to a `Handler`; an image without one exits
    0. `image_ids` maps references to image IDs (`image_id`; a run of another image fails as under `--pull never`);
    without it every reference is held, with a stable fake ID. `outcomes` are consumed one per run: `"ok"` (the
    handler's rc), an int (that rc), `"preempt"` (rc 137), `"oom"` (rc 137, `oom_killed`), `"hang"` (running until
    `stop`, then rc 143, or until `finish`), `"vanish"` (the daemon no longer lists it); only `"ok"` runs the handler.
    Each run starts `duration_s` after the previous one, counted from `start_utc`. `rootless`, `socket_uid` (default
    this process's uid) and `socket` are what `daemon()` reports."""

    def __init__(self, handlers: Mapping[str, Handler] | None = None, *, image_ids: Mapping[str, str] | None = None,
                 outcomes: Iterable[str | int] = (), duration_s: int = 60, start_utc: str = "2026-11-22T00:00:00Z",
                 state_dir: Path | None = None, rootless: bool = True, socket_uid: int | None = None,
                 socket: str | None = None) -> None:
        self.handlers = dict(handlers or {})
        self.image_ids = None if image_ids is None else dict(image_ids)
        self.duration_s = int(duration_s)
        self.start = dt.datetime.strptime(start_utc, TIME_FORMAT).replace(tzinfo=dt.timezone.utc)
        self.state_dir = None if state_dir is None else Path(state_dir)
        self._daemon = DaemonInfo(rootless, os.getuid() if socket_uid is None else socket_uid, socket)
        queued = list(outcomes)
        for o in queued:
            if not (o in OUTCOMES or (isinstance(o, int) and not isinstance(o, bool))):
                raise ValueError(f"outcome {o!r} is not an rc or one of {OUTCOMES}")
        self._mem: dict[str, Any] = {"containers": {}, "outcomes": [], "clock": 0, "seq": 0}
        self._runs: list[RunSpec] = []
        self._logs: dict[str, bytes] = {}
        if self.state_dir is not None:
            (self.state_dir / LOG_DIR).mkdir(parents=True, exist_ok=True)
        with self._state() as st:
            st["outcomes"].extend(queued)

    # -- state ------------------------------------------------------------------------------------------------------

    @contextlib.contextmanager
    def _state(self) -> Iterator[dict[str, Any]]:
        if self.state_dir is None:
            yield self._mem
            return
        with open(self.state_dir / LOCK_FILE, "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.state_dir / STATE_FILE
            st = json.loads(path.read_text()) if path.exists() else {"containers": {}, "outcomes": [], "clock": 0,
                                                                     "seq": 0}
            yield st
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(st))
            os.replace(tmp, path)

    def _time(self, offset_s: int) -> str:
        return (self.start + dt.timedelta(seconds=offset_s)).strftime(TIME_FORMAT)

    def _put_log(self, name: str, data: bytes) -> None:
        if self.state_dir is None:
            self._logs[name] = data
        else:
            (self.state_dir / LOG_DIR / name).write_bytes(data)

    def _get_log(self, name: str) -> bytes:
        if self.state_dir is None:
            return self._logs.get(name, b"")
        p = self.state_dir / LOG_DIR / name
        return p.read_bytes() if p.exists() else b""

    @property
    def runs(self) -> list[RunSpec]:
        """Every spec `run` accepted, in order (across processes with `state_dir`)."""
        if self.state_dir is None:
            return self._runs
        p = self.state_dir / RUNS_FILE
        if not p.exists():
            return []
        return [_spec_from_json(json.loads(line)) for line in p.read_text().splitlines() if line]

    def _record_run(self, spec: RunSpec) -> None:
        if self.state_dir is None:
            self._runs.append(spec)
        else:
            with open(self.state_dir / RUNS_FILE, "a") as f:
                f.write(json.dumps(_spec_json(spec), sort_keys=True) + "\n")

    @staticmethod
    def _find(st: Mapping[str, Any], container: str) -> dict[str, Any] | None:
        for c in st["containers"].values():
            if not c["vanished"] and container in (c["name"], c["id"]):
                return c
        return None

    # -- Runtime -----------------------------------------------------------------------------------------------------

    def run(self, spec: RunSpec) -> str:
        docker_run_argv(spec, context="fake")               # the same value and mount checks as the real runtime
        if self.image_id(spec.image) is None:
            raise RuntimeFailure("run", 125)
        with self._state() as st:
            if self._find(st, spec.name) is not None:
                raise RuntimeFailure("run", 125)
            outcome = st["outcomes"].pop(0) if st["outcomes"] else "ok"
            st["seq"] += 1
            cid = hashlib.sha256(f"{spec.name}/{st['seq']}".encode()).hexdigest()
            begin = st["clock"]
            st["clock"] += self.duration_s
            st["containers"][cid] = {"id": cid, "name": spec.name, "labels": [list(p) for p in spec.labels],
                                           "status": "running", "exit_code": None, "oom_killed": False,
                                           "started_utc": self._time(begin), "finished_utc": None, "vanished": False,
                                           "end": self._time(begin + self.duration_s)}
            self._record_run(spec)
        log = b""
        if outcome == "ok":
            rc, log = self._call_handler(spec)
        elif outcome in ("preempt", "oom"):
            rc = RC_KILLED
        elif isinstance(outcome, int):
            rc = outcome
        else:
            rc = None                                       # hang, vanish
        self._put_log(spec.name, log)
        with self._state() as st:
            c = st["containers"][cid]
            if outcome == "vanish":
                c["vanished"] = True
            elif rc is not None:
                c.update(status="exited", exit_code=rc, oom_killed=outcome == "oom", finished_utc=c["end"])
        return cid

    def _call_handler(self, spec: RunSpec) -> tuple[int, bytes]:
        handler = self.handlers.get(spec.image)
        if handler is None:
            return 0, b""
        tmp = Path(tempfile.mkdtemp(prefix="fake-tmpfs-"))
        try:
            dirs = {}
            for i, (p, _) in enumerate(spec.tmpfs):
                dirs[p] = tmp / str(i)
                dirs[p].mkdir()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = int(handler(spec, _host_path(spec, dirs)))
            return rc, buf.getvalue().encode("utf-8", "surrogateescape")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def inspect(self, container: str) -> ContainerState:
        with self._state() as st:
            c = self._find(st, container)
        if c is None:
            return ContainerState("missing", None, False, None, None)
        return ContainerState(c["status"], c["exit_code"], c["oom_killed"], c["started_utc"], c["finished_utc"])

    def _live(self, st: Mapping[str, Any], op: str, container: str) -> dict[str, Any]:
        c = self._find(st, container)
        if c is None:
            raise RuntimeFailure(op, 1)
        return c

    def logs(self, container: str, *, tail: int | None = None) -> bytes:
        with self._state() as st:
            name = self._live(st, "logs", container)["name"]
        data = self._get_log(name)
        if tail is None:
            return data
        lines = data.splitlines(keepends=True)
        return b"".join(lines[-tail:]) if tail else b""

    def stop(self, container: str, *, timeout_s: int = 30) -> None:
        with self._state() as st:
            c = self._live(st, "stop", container)
            if c["status"] == "running":
                c.update(status="exited", exit_code=RC_STOPPED, finished_utc=c["end"])

    def finish(self, container: str, rc: int) -> None:
        """End a `hang` container from the test with exit code `rc`."""
        with self._state() as st:
            c = self._live(st, "finish", container)
            if c["status"] != "running":
                raise RuntimeFailure("finish", 1)
            c.update(status="exited", exit_code=int(rc), finished_utc=c["end"])

    def remove(self, container: str) -> None:
        with self._state() as st:
            c = self._live(st, "remove", container)
            if c["status"] == "running":
                raise RuntimeFailure("remove", 1)
            del st["containers"][c["id"]]

    def image_id(self, ref: str) -> str | None:
        launch.option_value(ref)
        if self.image_ids is not None:
            return self.image_ids.get(ref)
        if launch.SHA_RE.fullmatch(ref):
            return ref
        return "sha256:" + hashlib.sha256(ref.encode()).hexdigest()

    def daemon(self) -> DaemonInfo:
        return self._daemon

    def containers(self, label: str) -> tuple[str, ...]:
        key, _, val = label.partition("=")
        with self._state() as st:
            return tuple(c["name"] for c in st["containers"].values()
                         if not c["vanished"] and [key, val] in c["labels"])
