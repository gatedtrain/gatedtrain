"""The container runtime the local backend and the keeper's inference container start containers through (AN-22,
AN-23, AN-25; design revision 25, 05 §18.10).

`RunSpec` is one detached container; `docker_run_argv` turns it into a docker argument list with the isolation options
fixed in code; `DockerCLIRuntime` runs the docker CLI (an argument list, never a shell; a small fixed environment; a
failure reports the operation and the exit code only, never docker's message, which may quote container output).
`gatedtrain.testing.fake_runtime.FakeRuntime` implements the same `Runtime` protocol for tests.

`NvidiaSmiProbe` and `wait_for_gpu` give the bounded GPU wait both the backend and the keeper run before a GPU
container starts: a reading that is not a number counts as busy, and a host without `nvidia-smi` refuses (AN-23).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from ..ext import launch
from ..ext.launch import mount_source, option_value

NAME_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}")      # Docker container names, at most 128 characters
CONTAINER_ID_RE = re.compile(r"[0-9a-f]{64}")
STATUSES = ("created", "running", "exited", "missing")
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
TIME_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_DOCKER_TIME_RE = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2})(?:\.[0-9]+)?Z")
_UNSET_TIME = "0001-01-01T00:00:00Z"
# Docker's own states folded into STATUSES: a paused or restarting container still holds its resources.
_STATUS_MAP = {"created": "created", "running": "running", "paused": "running", "restarting": "running",
               "removing": "exited", "exited": "exited", "dead": "exited"}
GPU_OPTION_SETS = ((), launch.CDI_GPU_OPTIONS, launch.ROOTFUL_GPU_OPTIONS)
CLI_ENV_KEYS = ("PATH", "HOME", "XDG_RUNTIME_DIR")
ROOTLESS_SECURITY_OPTION = "name=rootless"

GPU_BUSY_MIB = 1024             # AN-23: GPU memory held by compute processes at most this counts as a free GPU
GPU_WAIT_S = 900                # AN-23: the longest wait for a free GPU before a GPU container start is refused
GPU_POLL_S = 10.0
UNREADABLE_MIB = 2**31 - 1      # a probe line that is not a number: counted as busy (fail closed)
PROBE_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class Mount:
    """A bind mount: `host` (resolved to its strict `realpath` when the container starts) at `container`."""
    host: str
    container: str
    read_only: bool = True


@dataclass(frozen=True)
class RunSpec:
    """One detached container. Isolation options that never vary (`--cap-drop ALL`, no-new-privileges, `--pull never`)
    are not fields; `docker_run_argv` always adds them. `gpu_options` is one of GPU_OPTION_SETS; `extra_options` are
    an adapter's fixed run options (EXT-191), code and never request values."""
    name: str
    image: str
    argv: tuple[str, ...]
    env: tuple[tuple[str, str], ...]
    mounts: tuple[Mount, ...]
    user: str
    network: str = "none"
    gpu_options: tuple[str, ...] = ()
    read_only_root: bool = False
    tmpfs: tuple[tuple[str, str], ...] = ()
    shm_size: str | None = "8g"
    log_driver: str | None = None
    labels: tuple[tuple[str, str], ...] = ()
    extra_options: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not NAME_RE.fullmatch(self.name):
            raise ValueError(f"container name {self.name!r} is not a Docker name of at most 128 characters")
        if not isinstance(self.image, str) or not launch.IMAGE_RE.fullmatch(self.image):
            raise ValueError(f"image {self.image!r} is not pinned by digest")


@dataclass(frozen=True)
class ContainerState:
    """A container as the daemon reports it; `missing` when the daemon does not list it. Times are UTC seconds."""
    status: str
    exit_code: int | None
    oom_killed: bool
    started_utc: str | None
    finished_utc: str | None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"container status {self.status!r} is not one of {STATUSES}")
        for t in (self.started_utc, self.finished_utc):
            if t is not None and not TIME_RE.fullmatch(t):
                raise ValueError(f"time {t!r} is not {TIME_FORMAT}")


@dataclass(frozen=True)
class DaemonInfo:
    """What the daemon says about itself (pre-flight finding 3): whether it runs rootless (`docker info` security
    options hold `name=rootless`), and for a `unix://` endpoint the resolved socket path and its owner uid (None for
    another endpoint, or when the socket cannot be read)."""
    rootless: bool
    socket_uid: int | None
    socket: str | None


class RuntimeFailure(Exception):
    """A runtime operation failed: the operation and the exit code only (-1: timed out; 127: the CLI is missing; 0:
    the CLI succeeded but its output was not what the operation reads).
    Docker's message is never kept, since it may quote container output or sealed paths."""

    def __init__(self, op: str, rc: int) -> None:
        super().__init__(f"docker {op} failed with exit code {rc}")
        self.op, self.rc = op, rc


class Runtime(Protocol):
    def run(self, spec: RunSpec) -> str: ...                                         # detached; the container id
    def inspect(self, container: str) -> ContainerState: ...
    def logs(self, container: str, *, tail: int | None = None) -> bytes: ...
    def stop(self, container: str, *, timeout_s: int = 30) -> None: ...
    def remove(self, container: str) -> None: ...
    def image_id(self, ref: str) -> str | None: ...
    def daemon(self) -> DaemonInfo: ...
    def containers(self, label: str) -> tuple[str, ...]: ...                         # names, running or not


def _endpoint(context: str | None, host: str | None) -> list[str]:
    if (context is None) == (host is None):
        raise ValueError("exactly one of context and host names the daemon")
    return ["--context", option_value(context)] if context is not None else ["-H", option_value(host)]


def _container_path(p: str) -> str:
    option_value(p)
    if not p.startswith("/") or any(c in p for c in ":,") or any(ord(c) < 32 for c in p):
        raise ValueError(f"container path {p!r} is not an absolute path without `:` or `,`")
    return p


def docker_run_argv(spec: RunSpec, *, context: str | None = None, host: str | None = None,
                    docker: str = "docker") -> list[str]:
    """The `docker run` argument list of `spec`: `docker (--context C | -H H) run -d --pull never --name N
    [--log-driver D] --network NET --user U --cap-drop ALL --security-opt no-new-privileges [--read-only]
    [--shm-size=S] [gpu options] [extra options] [--tmpfs p:o]... [-e K=V sorted]... [-v src:dst:ro|rw]...
    [--label k=v sorted]... IMAGE ARGV...`. No `--rm`: the backend reads the exit state, then removes the container.
    `--pull never`: an image the daemon does not hold fails the start instead of being pulled (pre-flight finding 27).
    Each `-v` source is the mount's strict `realpath` (EXT-69, `launch.mount_source`); a value before the image that
    docker would read as an option raises ValueError (EXT-39, `launch.option_value`), as does a GPU option set other
    than GPU_OPTION_SETS or an environment name that is not a plain name."""
    value = option_value
    tok = [docker, *_endpoint(context, host), "run", "-d", "--pull", "never", "--name", value(spec.name)]
    if spec.log_driver is not None:
        tok += ["--log-driver", value(spec.log_driver)]
    tok += ["--network", value(spec.network), "--user", value(spec.user), "--cap-drop", "ALL", "--security-opt",
            "no-new-privileges"]
    if spec.read_only_root:
        tok.append("--read-only")
    if spec.shm_size is not None:
        tok.append(f"--shm-size={value(spec.shm_size)}")
    if tuple(spec.gpu_options) not in GPU_OPTION_SETS:
        raise ValueError(f"GPU options {spec.gpu_options!r} are not one of {GPU_OPTION_SETS}")
    tok += list(spec.gpu_options)
    tok += list(spec.extra_options)
    for path, opts in spec.tmpfs:
        tok += ["--tmpfs", value(f"{_container_path(path)}:{opts}")]
    for name, val in sorted(spec.env):
        if not launch.ENV_NAME_RE.fullmatch(name):
            raise ValueError(f"environment name {name!r} is not a plain name")
        tok += ["-e", value(f"{name}={val}")]
    for m in spec.mounts:
        tok += ["-v", value(f"{mount_source(m.host)}:{_container_path(m.container)}:{'ro' if m.read_only else 'rw'}")]
    for key, val in sorted(spec.labels):
        tok += ["--label", value(f"{key}={val}")]
    tok.append(value(spec.image))
    tok += list(spec.argv)
    return tok


def utc_seconds(docker_time: str | None) -> str | None:
    """A Docker timestamp (RFC 3339 with nanoseconds) as `%Y-%m-%dT%H:%M:%SZ`; None for Docker's unset time."""
    if not docker_time or docker_time == _UNSET_TIME:
        return None
    m = _DOCKER_TIME_RE.fullmatch(docker_time)
    if m is None:
        raise ValueError(f"Docker time {docker_time!r} is not UTC RFC 3339")
    return m.group(1) + "Z"


def _cli_env() -> dict[str, str]:
    return {k: os.environ[k] for k in CLI_ENV_KEYS if k in os.environ}


def _name_filter(container: str) -> str:
    """The `docker ps --filter` of one container: its full id, or its exact name (`.` escaped in the anchored
    regular expression; the other name characters are literal)."""
    if CONTAINER_ID_RE.fullmatch(container):
        return f"id={container}"
    if not NAME_RE.fullmatch(container):
        raise ValueError(f"container {container!r} is neither a full container id nor a container name")
    return "name=^" + container.replace(".", "\\.") + "$"


class DockerCLIRuntime:
    """`Runtime` over the docker CLI on one daemon (`context`, the owner's `gatedtrain`, or `host`, the keeper's
    socket). Each call is one subprocess with an argument list and the environment `PATH`, `HOME`, `XDG_RUNTIME_DIR`
    (so `DOCKER_HOST` and credentials in the caller's environment never reach docker); a non-zero exit raises
    `RuntimeFailure(op, rc)`."""

    def __init__(self, *, context: str | None = None, host: str | None = None, docker: str = "docker",
                 timeout_s: float = 120.0) -> None:
        self._endpoint = _endpoint(context, host)
        self.context, self.host, self.docker, self.timeout_s = context, host, str(docker), timeout_s

    def _cli(self, op: str, args: list[str], *, endpoint: bool = True, merge_stderr: bool = False) -> bytes:
        argv = [self.docker, *(self._endpoint if endpoint else []), *args]
        try:
            proc = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT if merge_stderr else subprocess.DEVNULL,
                                  env=_cli_env(), timeout=self.timeout_s, check=False)
        except subprocess.TimeoutExpired:
            raise RuntimeFailure(op, -1) from None
        except OSError:
            raise RuntimeFailure(op, 127) from None
        if proc.returncode != 0:
            raise RuntimeFailure(op, proc.returncode)
        return proc.stdout

    def _text(self, op: str, args: list[str], **kw) -> str:
        try:
            return self._cli(op, args, **kw).decode("utf-8").strip()
        except UnicodeDecodeError:
            raise RuntimeFailure(op, 0) from None

    def run(self, spec: RunSpec) -> str:
        argv = docker_run_argv(spec, context=self.context, host=self.host, docker=self.docker)
        cid = self._text("run", argv[1 + len(self._endpoint):])
        if not CONTAINER_ID_RE.fullmatch(cid):
            raise RuntimeFailure("run", 0)
        return cid

    def inspect(self, container: str) -> ContainerState:
        """`missing` only when `docker ps -a` succeeds and lists no such container (pre-flight finding 21); a daemon
        that cannot be reached raises RuntimeFailure, so the caller leaves the job's state unchanged."""
        listed = self._text("inspect", ["ps", "-a", "--no-trunc", "--filter", _name_filter(container), "--format",
                                        "{{.ID}}"])
        if not listed:
            return ContainerState("missing", None, False, None, None)
        out = self._text("inspect", ["inspect", "--type", "container", "--format", "{{json .State}}",
                                     option_value(container)])
        try:
            st = json.loads(out)
            status = _STATUS_MAP[st["Status"]]
            exited = status == "exited"
            return ContainerState(status, int(st["ExitCode"]) if exited else None, bool(st["OOMKilled"]),
                                  utc_seconds(st.get("StartedAt")), utc_seconds(st.get("FinishedAt")) if exited
                                  else None)
        except (ValueError, KeyError, TypeError):
            raise RuntimeFailure("inspect", 0) from None

    def logs(self, container: str, *, tail: int | None = None) -> bytes:
        args = ["logs"]
        if tail is not None:
            if not isinstance(tail, int) or isinstance(tail, bool) or tail < 0:
                raise ValueError(f"tail {tail!r} is not a count of lines")
            args += ["--tail", str(tail)]
        return self._cli("logs", [*args, option_value(container)], merge_stderr=True)

    def stop(self, container: str, *, timeout_s: int = 30) -> None:
        self._cli("stop", ["stop", "-t", str(int(timeout_s)), option_value(container)])

    def remove(self, container: str) -> None:
        self._cli("remove", ["rm", option_value(container)])

    def image_id(self, ref: str) -> str | None:
        """The ID of the image `ref` names in this daemon's store, or None when the daemon does not hold it (the
        daemon answering `docker version` tells a missing image from an unreachable daemon, which raises)."""
        option_value(ref)
        try:
            out = self._text("image_id", ["image", "inspect", "--format", "{{.Id}}", ref])
        except RuntimeFailure as exc:
            if exc.rc in (-1, 127):
                raise
            self._text("image_id", ["version", "--format", "{{.Server.Version}}"])
            return None
        if not launch.SHA_RE.fullmatch(out):
            raise RuntimeFailure("image_id", 0)
        return out

    def _socket_path(self) -> str | None:
        endpoint = self.host
        if endpoint is None:
            endpoint = self._text("daemon", ["context", "inspect", "--format", "{{.Endpoints.docker.Host}}",
                                             option_value(self.context)], endpoint=False)
        return endpoint[len("unix://"):] if endpoint.startswith("unix://") else None

    def daemon(self) -> DaemonInfo:
        """Whether the daemon is rootless, from `docker info`, and the owner of its socket (pre-flight finding 3)."""
        out = self._text("daemon", ["info", "--format", "{{json .SecurityOptions}}"])
        try:
            options = json.loads(out)
            rootless = any(ROOTLESS_SECURITY_OPTION in str(o).split(",") for o in options or ())
        except (ValueError, TypeError):
            raise RuntimeFailure("daemon", 0) from None
        path = self._socket_path()
        if path is None:
            return DaemonInfo(rootless, None, None)
        real = os.path.realpath(path)
        try:
            uid = os.stat(real).st_uid
        except OSError:
            uid = None
        return DaemonInfo(rootless, uid, real)

    def containers(self, label: str) -> tuple[str, ...]:
        """The names of the containers, running or not, labelled `label` (`key=value`); for cleanup at start."""
        out = self._text("containers", ["ps", "-a", "--filter", option_value(f"label={label}"), "--format",
                                        "{{.Names}}"])
        return tuple(line for line in out.splitlines() if line)


# ---------------------------------------------------------------------------
# AN-23: the GPU wait
# ---------------------------------------------------------------------------

class GpuProbe(Protocol):
    def used_mib(self) -> int | None: ...       # None: no GPU tool on this host


class NvidiaSmiProbe:
    """The GPU memory held by compute processes, summed from `nvidia-smi --query-compute-apps=used_memory`. A line
    that is not a number (`[N/A]`, `[Insufficient Permissions]` for another user's process) counts as busy
    (UNREADABLE_MIB); a missing tool, a failing one or one that does not answer in time gives None."""

    def __init__(self, nvidia_smi: str = "nvidia-smi") -> None:
        self.nvidia_smi = str(nvidia_smi)

    def used_mib(self) -> int | None:
        argv = [self.nvidia_smi, "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"]
        try:
            proc = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  env=_cli_env(), timeout=PROBE_TIMEOUT_S, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode != 0:
            return None
        total = 0
        for line in proc.stdout.decode("utf-8", "replace").splitlines():
            line = line.strip()
            if not line:
                continue
            if not re.fullmatch(r"[0-9]+", line):      # ASCII digits only: `str.isdigit` accepts other scripts' digits
                return UNREADABLE_MIB
            total += int(line)
        return total


def wait_for_gpu(probe: GpuProbe, *, busy_mib: int = GPU_BUSY_MIB, timeout_s: float = GPU_WAIT_S,
                 poll_s: float = GPU_POLL_S, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> bool:
    """AN-23: True once the probe reads at most `busy_mib`, polling every `poll_s` for at most `timeout_s`; False when
    the GPU is still busy at the bound, or at once when the probe has no GPU tool (a GPU job is then refused)."""
    start = clock()
    while True:
        used = probe.used_mib()
        if used is None:
            return False
        if used <= busy_mib:
            return True
        if clock() - start + poll_s > timeout_s:
            return False
        sleep(poll_s)
