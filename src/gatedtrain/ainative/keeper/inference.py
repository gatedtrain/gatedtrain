"""The keeper's inference container (AN-25, AN-8, AN-9, AN-13; design revision 25, 05 §18.9 and §18.10).

A keeper started with an inference image runs a scenario pack's batch generator (`batch.BatchGenerator`, declared as
data) in a container of its own rootless Docker daemon: one container per arm for all registered seeds, the image by
the ID its environment record holds, `--user 0:0 --cap-drop ALL --security-opt no-new-privileges --read-only --network
none --log-driver none`, a tmpfs `/tmp`, the fixed environment `KEEPER_ENV`, no image pull, and the GPU through
Docker's native CDI when the generator needs it. The container sees the sealed items (`/in/items.jsonl`, never a
label), the sampler and seeds (`/in/sampler.json`), the keeper's copy of the model files (`/in/model`, AN-13), the
script (`/gen/<name>`), the eval trees at their own paths when the generator asks for them, all read-only, and one
empty keeper-owned output directory `/out`. Nothing about the items is passed in the arguments or the environment.

`KeeperInference`:
- `remove_leftovers` (at `serve`, first): the run directories, then the checks that need no Docker call, then
  (only when those pass) the leftover containers;
- `start_problems` (AN-13, at `serve`, after `remove_leftovers` reported nothing): waits up to 60 s for the daemon's
  socket (the keeper user's daemon may start after the keeper), then refuses a daemon that is not rootless or whose
  socket is not owned by the keeper uid under a directory of mode 0700, an image ID other than the configured one, an
  eval tree that fails `envcheck.eval_tree_problems`, (for the Docker CLI runtime) a Docker CLI or `nvidia-smi` that
  fails the keeper's ownership check, and a subordinate id range of the owner that covers the keeper's uid or gid (or
  one of the keeper's that covers the owner's), read from /etc/subuid and /etc/subgid (an unreadable file refuses);
- `environment`: what the signed environment record holds under `inference`;
- `ready` (before the grant, so a refusal spends nothing): the generator's time limit fits the driver's wait for an
  evaluation, eval trees are configured when it needs them and its interpreter is a regular file inside one, its script
  is a file of a distribution the keeper runs, the daemon and the image are unchanged, and for a GPU generator the GPU
  is free (AN-23's bounded wait; no `nvidia-smi` refuses);
- `run_batch` (`batch.BatchRunner`): runs one container and reads `/out` back with `batch.read_predictions`. The daemon
  is checked again just before the start. Every failure of the container, the runtime or the files around it is a crash
  (AN-9), a daemon that changed since `ready` too, whose message names an exit code, a time limit, the daemon check or
  an exception class, never a path into the keeper's state, an item or a prediction. The container and its run directory
  `state/run/<hex>/` are removed on every path (best effort: a failed removal never changes the outcome;
  `remove_leftovers` at the next start, before the start checks, removes what is left). `last_run` is `{"arm",
  "image_id", "seconds", "gpu_seconds", "rc"}` of the last container started (`arm` None: the keeper's state sets it),
  or None when none started.
"""

from __future__ import annotations

import dataclasses
import math
import os
import pwd
import re
import secrets
import shutil
import stat
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ...canonical import canonical_json
from ...errors import Problem, Refused
from ...ext import launch
from ..batch import (CONTAINER_ITEMS, CONTAINER_MODEL, CONTAINER_OUT, CONTAINER_SAMPLER, CONTAINER_SCRIPT_DIR,
                     BatchGenerator, EVAL_SLACK_S, container_argv, evaluation_wait_bound, read_predictions, sampler_doc,
                     write_items)
from ..runtime import (GPU_BUSY_MIB, GPU_WAIT_S, MAX_GPU_BUSY_MIB, TIME_FORMAT, ContainerState, DockerCLIRuntime, GpuProbe, Mount,
                       NvidiaSmiProbe, RunSpec, Runtime, RuntimeFailure, wait_for_gpu)
from . import envcheck, protocol

RULE_ENV = "AN-13"
RULE_BATCH = "AN-25"
RULE_CRASH = "AN-9"
KEEPER_ENV = (("DO_NOT_TRACK", "1"), ("HF_HUB_OFFLINE", "1"), ("HOME", "/tmp"), ("TRANSFORMERS_OFFLINE", "1"),
              ("VLLM_NO_USAGE_STATS", "1"))
KEEPER_LABEL = ("gatedtrain.keeper", "inference")      # every keeper container; leftovers are removed at start
NAME_PREFIX = "gatedtrain-keeper-"
RUN_DIR = "run"
SUBID_FILES = ("/etc/subuid", "/etc/subgid")
CONTAINER_USER = "0:0"           # the keeper uid on the host under the keeper's rootless daemon (05 §18.10)
SHM_SIZE = launch.SHM_SIZE_VALUE
SOCKET_WAIT_S = 60.0
SOCKET_POLL_S = 1.0
POLL_S = 2.0                     # how often a running container is inspected
STOP_TIMEOUT_S = 10
DEFAULT_DOCKER = Path("/usr/bin/docker")
DEFAULT_NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
CRASH_FIX = ("check the keeper's inference image and the scenario pack's batch generator, then train a new candidate "
             "and evaluate it")
DAEMON_FIX = ("run the keeper's inference on the keeper user's own rootless Docker daemon: --docker-host "
              "unix:///run/user/<keeper uid>/docker.sock (`gatedtrain setup host` installs it)")

_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")


def _problem(rule: str, field: str, message: str, fix: str) -> Problem:
    return Problem(rule, field, message, fix)


def _crash(detail: str) -> Refused:
    return Refused([_problem(RULE_CRASH, "inference", f"{detail}; the attempt settles as not passed", CRASH_FIX)])


def _plain_abs(p: Path, what: str) -> Path:
    s = os.fspath(p)
    if not os.path.isabs(s) or any(c in s for c in ":,") or any(ord(c) < 32 for c in s):
        raise ValueError(f"the inference {what} {s!r} is not an absolute path without `:` or `,`")
    return Path(s)


@dataclass(frozen=True)
class InferenceConfig:
    """`keeper serve --inference-image ID [--inference-ref REF] [--eval-tree DIR]... [--docker-host URL] [--docker PATH]
    [--nvidia-smi PATH] [--gpu-wait-s S] [--gpu-busy-mib MIB] [--eval-python PATH]`: the image ID (`sha256:<64 hex>`)
    the keeper runs and refuses any other; the reference the daemon knows it by (default: the ID); the eval trees
    (absolute directories, mounted read-only at their own paths); the daemon; the Docker CLI and `nvidia-smi` the keeper
    calls by absolute path. The owner's, set where the keeper is started (its unit), never by a request: `gpu_wait_s`
    and `gpu_busy_mib` are how long the keeper waits for a free GPU and how much memory still counts as free (AN-23;
    a GPU shared with other work needs other values); `eval_python` replaces the interpreter an eval-tree generator
    declares (a pack's default is the eval venv `setup host` builds). A value that differs from the default is part of
    the signed environment record (`environment`)."""

    image_id: str
    docker_host: str
    eval_trees: tuple[Path, ...] = ()
    image_ref: str | None = None
    docker: Path = DEFAULT_DOCKER
    nvidia_smi: Path = DEFAULT_NVIDIA_SMI
    gpu_wait_s: float = GPU_WAIT_S
    gpu_busy_mib: int = GPU_BUSY_MIB
    eval_python: Path | None = None

    def __post_init__(self) -> None:
        if not (isinstance(self.image_id, str) and _IMAGE_ID.fullmatch(self.image_id)):
            raise ValueError(f"the inference image {self.image_id!r} is not an image ID sha256:<64 hex>")
        if not (isinstance(self.docker_host, str) and self.docker_host):
            raise ValueError("the inference Docker host is empty")
        launch.option_value(self.docker_host)
        if self.image_ref is not None:
            if not isinstance(self.image_ref, str) or not launch.IMAGE_RE.fullmatch(self.image_ref):
                raise ValueError(f"the inference image reference {self.image_ref!r} is not pinned by digest")
        object.__setattr__(self, "eval_trees", tuple(_plain_abs(Path(t), "eval tree") for t in self.eval_trees))
        object.__setattr__(self, "docker", _plain_abs(Path(self.docker), "Docker CLI"))
        object.__setattr__(self, "nvidia_smi", _plain_abs(Path(self.nvidia_smi), "nvidia-smi"))
        w = self.gpu_wait_s
        try:
            wait_ok = isinstance(w, (int, float)) and not isinstance(w, bool) and math.isfinite(w) and w > 0
        except OverflowError:
            wait_ok = False
        if not wait_ok:
            raise ValueError(f"the GPU wait {w!r} is not a number of seconds above 0")
        b = self.gpu_busy_mib
        if isinstance(b, bool) or not isinstance(b, int) or not 0 <= b <= MAX_GPU_BUSY_MIB:
            raise ValueError(f"the GPU busy threshold {b!r} is not a whole number of MiB from 0 to {MAX_GPU_BUSY_MIB}")
        if self.eval_python is not None:
            object.__setattr__(self, "eval_python", _plain_abs(Path(self.eval_python), "eval interpreter"))


def _seconds(st: ContainerState | None) -> int | None:
    if st is None or st.started_utc is None or st.finished_utc is None:
        return None
    a, b = (datetime.strptime(t, TIME_FORMAT).replace(tzinfo=timezone.utc) for t in (st.started_utc, st.finished_utc))
    return max(0, int((b - a).total_seconds()))


def _subids_cover(lines: list[str], users: tuple[str, ...], ident: int) -> bool:
    """Whether a `name:start:count` line of one of `users` (a name or a uid) covers `ident`; a malformed line of
    such a user counts as covering (fail closed)."""
    for line in lines:
        parts = line.strip().split(":")
        if len(parts) < 3 or parts[0] not in users:
            continue
        try:
            start, count = int(parts[1]), int(parts[2])
        except ValueError:
            return True
        if start <= ident < start + count:
            return True
    return False


def _open_up(path: Path, st: os.stat_result) -> None:
    """Give a directory (already `lstat`ed: not a link) its owner's read, write and search permission."""
    mode = stat.S_IMODE(st.st_mode)
    if mode & 0o700 != 0o700:
        os.chmod(path, mode | 0o700)


def _remove_tree(path: Path) -> None:
    """Remove `path` without following links; directories without their owner's permissions are opened up first."""
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode):
        os.unlink(path)
        return
    _open_up(path, st)
    for name in os.listdir(path):
        _remove_tree(path / name)
    os.rmdir(path)


def _write_new(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
    finally:
        os.close(fd)


class KeeperInference:
    """The keeper's `BatchRunner` (see the module docstring). `probe` defaults to `nvidia-smi` at the configured
    absolute path; `sleep` and `clock` drive every wait (the socket, the GPU, a running container)."""

    def __init__(self, config: InferenceConfig, runtime: Runtime, state_dir: Path, *, probe: GpuProbe | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                 owner_uid: int | None = None) -> None:
        self.config = config
        self.owner_uid = owner_uid
        self.runtime = runtime
        self.state_dir = Path(state_dir)
        self.keeper_uid = os.geteuid()
        self._own_probe = probe is None
        self.probe: GpuProbe = NvidiaSmiProbe(str(config.nvidia_smi)) if probe is None else probe
        self.sleep = sleep
        self.clock = clock
        self.last_run: dict | None = None
        self._environment: dict | None = None
        self._dist_files: frozenset[str] | None = None

    # ---- checks ---------------------------------------------------------------------------------------------------

    def _socket_path(self) -> str | None:
        host = self.config.docker_host
        return host[len("unix://"):] if host.startswith("unix://") else None

    def _wait_for_socket(self) -> list[Problem]:
        path = self._socket_path()
        if path is None:
            return [_problem(RULE_ENV, "docker_host", "the keeper's Docker host is not a unix:// socket", DAEMON_FIX)]
        start = self.clock()
        while not os.path.exists(path):
            if self.clock() - start >= SOCKET_WAIT_S:
                return [_problem(RULE_ENV, "docker_host", f"the keeper's Docker socket did not appear within "
                                 f"{SOCKET_WAIT_S:g} s", "start the keeper user's rootless Docker daemon (the keeper "
                                 "unit orders itself after user@<keeper uid>.service), then start the keeper again")]
            self.sleep(SOCKET_POLL_S)
        return []

    def _daemon_problems(self) -> list[Problem]:
        """The daemon is rootless and its socket is the keeper's, in a directory of the keeper's with mode 0700."""
        try:
            d = self.runtime.daemon()
        except RuntimeFailure as exc:
            return [_problem(RULE_ENV, "docker_host", f"the keeper's Docker daemon cannot be asked ({exc})",
                             DAEMON_FIX)]
        problems = []
        if not d.rootless:
            problems.append(_problem(RULE_ENV, "docker_host", "the keeper's Docker daemon is not rootless, so "
                                     "container root would be host root", DAEMON_FIX))
        if d.socket is None:
            problems.append(_problem(RULE_ENV, "docker_host", "the keeper's Docker daemon is not reached through a "
                                     "unix socket the keeper can check", DAEMON_FIX))
            return problems
        if d.socket_uid != self.keeper_uid:
            problems.append(_problem(RULE_ENV, "docker_host", f"the keeper's Docker socket is owned by uid "
                                     f"{d.socket_uid}, not the keeper (uid {self.keeper_uid})", DAEMON_FIX))
        try:
            st = os.lstat(os.path.dirname(d.socket))
        except OSError:
            problems.append(_problem(RULE_ENV, "docker_host", "the directory of the keeper's Docker socket cannot be "
                                     "inspected", DAEMON_FIX))
            return problems
        mode = stat.S_IMODE(st.st_mode)
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != self.keeper_uid or mode != 0o700:
            problems.append(_problem(RULE_ENV, "docker_host", f"the directory of the keeper's Docker socket is not "
                                     f"the keeper's with mode 0700 (owner uid {st.st_uid}, mode {mode:04o}), so "
                                     "another uid could replace the socket", DAEMON_FIX))
        return problems

    def _image_problems(self) -> list[Problem]:
        fix = ("pull the image by digest into the keeper's daemon (`gatedtrain setup host`) and pass its ID as "
               "--inference-image")
        ref = self.config.image_ref or self.config.image_id
        try:
            held = self.runtime.image_id(ref)
        except (RuntimeFailure, ValueError) as exc:
            what = str(exc) if isinstance(exc, RuntimeFailure) else type(exc).__name__
            return [_problem(RULE_ENV, "inference_image", f"the keeper's Docker daemon cannot be asked for the "
                             f"inference image ({what})", fix)]
        if held is None:
            return [_problem(RULE_ENV, "inference_image", "the keeper's Docker daemon does not hold the inference "
                             "image", fix)]
        if held != self.config.image_id:
            return [_problem(RULE_ENV, "inference_image", f"the keeper's Docker daemon holds image {held} under the "
                             f"inference image reference, not the configured {self.config.image_id}", fix)]
        return []

    def _cli_problems(self) -> list[Problem]:
        if not isinstance(self.runtime, DockerCLIRuntime):
            return []
        problems = []
        cli = self.config.docker
        if not os.path.isfile(cli):
            problems.append(_problem(RULE_ENV, "docker", f"the Docker CLI {cli} is not a file",
                                     "install the Docker CLI, or pass its absolute path as --docker"))
        else:
            problems += envcheck.path_problems([cli], self.keeper_uid)
        if self._own_probe:              # a missing nvidia-smi is skipped here; a GPU generator is refused then
            problems += envcheck.path_problems([self.config.nvidia_smi], self.keeper_uid)
        return problems

    def _subid_problems(self) -> list[Problem]:
        """No subordinate uid (gid) range of the owner covers the keeper's uid (gid), nor one of the keeper's the
        owner's: the owner's rootless containers could otherwise act as the keeper on the host (or the keeper's as the
        owner). Read from `SUBID_FILES`; a file that cannot be read, or a user without a passwd entry, refuses
        (fail closed). Skipped without an owner uid, and for a keeper that runs as the owner (nothing is sealed from
        the owner then)."""
        if self.owner_uid is None or self.owner_uid == self.keeper_uid:
            return []
        fix = ("give the owner and the keeper user subordinate id ranges that cover neither user's own ids "
               "(/etc/subuid, /etc/subgid), then start the keeper again")
        try:
            owner, keeper = pwd.getpwuid(self.owner_uid), pwd.getpwuid(self.keeper_uid)
        except KeyError:
            return [_problem(RULE_ENV, "subids", "the owner or the keeper user has no passwd entry, so their "
                             "subordinate id ranges cannot be checked", fix)]
        problems = []
        for path, kind, owner_id, keeper_id in ((SUBID_FILES[0], "uid", owner.pw_uid, keeper.pw_uid),
                                                (SUBID_FILES[1], "gid", owner.pw_gid, keeper.pw_gid)):
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    lines = f.read().splitlines()
            except OSError as exc:
                problems.append(_problem(RULE_ENV, "subids", f"{path} cannot be read ({type(exc).__name__}), so the "
                                         "subordinate id ranges cannot be checked", fix))
                continue
            for who, user, other_name, other_id in (("owner", owner, "keeper", keeper_id),
                                                    ("keeper", keeper, "owner", owner_id)):
                if _subids_cover(lines, (user.pw_name, str(user.pw_uid)), other_id):
                    problems.append(_problem(RULE_ENV, "subids", f"a subordinate {kind} range of the {who} in {path} "
                                             f"covers the {other_name}'s {kind} {other_id}", fix))
        return problems

    def start_problems(self) -> list[Problem]:
        """AN-13 at start; [] when the keeper may serve."""
        problems = self._wait_for_socket()
        if problems:
            return problems
        problems += self._daemon_problems()
        problems += self._image_problems()
        if self.config.eval_trees:
            problems += envcheck.eval_tree_problems(self.config.eval_trees, self.keeper_uid)
        problems += self._cli_problems()
        problems += self._subid_problems()
        return problems

    def _socket_file_problems(self) -> list[Problem]:
        """The checks of the daemon's socket that need no call to the daemon: when the configured socket exists, it is
        the keeper's and lies in a directory of the keeper's with mode 0700."""
        path = self._socket_path()
        if path is None or not os.path.lexists(path):
            return []
        try:
            sock, parent = os.lstat(path), os.lstat(os.path.dirname(path))
        except OSError:
            return [_problem(RULE_ENV, "docker_host", "the keeper's Docker socket or its directory cannot be "
                             "inspected", DAEMON_FIX)]
        problems = []
        if sock.st_uid != self.keeper_uid:
            problems.append(_problem(RULE_ENV, "docker_host", f"the keeper's Docker socket is owned by uid "
                                     f"{sock.st_uid}, not the keeper (uid {self.keeper_uid})", DAEMON_FIX))
        mode = stat.S_IMODE(parent.st_mode)
        if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != self.keeper_uid or mode != 0o700:
            problems.append(_problem(RULE_ENV, "docker_host", f"the directory of the keeper's Docker socket is not "
                                     f"the keeper's with mode 0700 (owner uid {parent.st_uid}, mode {mode:04o}), so "
                                     "another uid could replace the socket", DAEMON_FIX))
        return problems

    def remove_leftovers(self) -> list[Problem]:
        """At start, before `start_problems` (the keeper holds its state lock), in this order: (a) `state/run/*` is
        removed, strictly and without any Docker call: a directory without the owner's permissions is opened up
        first, links are never followed, and an entry that still cannot be removed is an AN-13 problem naming its
        path in the keeper's state; (b) the checks that need no Docker call: the Docker CLI and `nvidia-smi`
        (`_cli_problems`) and, when the socket exists, its owner and its 0700 directory; (c) only when (b) found
        nothing, every container labelled for the keeper is removed, best effort and only when the socket is
        already there (no wait here: the bounded wait stays in `start_problems`). Returns the problems of (a) and
        (b); the caller runs `start_problems` only when there are none, so no Docker CLI another uid could change
        and no socket another uid could replace is ever used."""
        problems = self._remove_runs()
        early = self._cli_problems() + self._socket_file_problems()
        if early:
            return problems + early
        path = self._socket_path()
        if path is not None and os.path.exists(path):
            try:
                names = self.runtime.containers("=".join(KEEPER_LABEL))
            except RuntimeFailure:
                names = ()
            for name in names:
                self._remove_container(name)
        return problems

    def _remove_runs(self) -> list[Problem]:
        run = self.state_dir / RUN_DIR
        try:
            if os.path.lexists(run):
                st = os.lstat(run)
                if not stat.S_ISDIR(st.st_mode):
                    os.unlink(run)
                else:
                    _open_up(run, st)
                    for name in os.listdir(run):
                        _remove_tree(run / name)
        except OSError as exc:
            where = exc.filename if isinstance(exc.filename, (str, os.PathLike)) else run
            return [_problem(RULE_ENV, "state", f"the keeper cannot remove the leftover run entry {os.fspath(where)} "
                             f"in its state ({type(exc).__name__})",
                             "remove it as the keeper user or as root, then start the keeper again")]
        return []

    # ---- environment record (AN-13) --------------------------------------------------------------------------------

    def _distribution_files(self) -> frozenset[str]:
        if self._dist_files is None:
            self._dist_files = frozenset(os.path.realpath(p) for dist in envcheck.keeper_distributions()
                                         for p in envcheck.distribution_files(dist).values())
        return self._dist_files

    def environment(self) -> dict:
        """`{"image_id", "docker_host", "eval_trees": [{"path", "files_sha256"}], "docker_cli_sha256"}` (the CLI's
        hash None for a runtime that is not the Docker CLI); computed once."""
        if self._environment is None:
            cli = envcheck._file_sha256(self.config.docker) if isinstance(self.runtime, DockerCLIRuntime) else None
            self._environment = {"image_id": self.config.image_id, "docker_host": self.config.docker_host,
                                 "eval_trees": [envcheck.eval_tree_record(t) for t in self.config.eval_trees],
                                 "docker_cli_sha256": cli}
            c = self.config                  # the owner's settings, in the signed record only when they are not the defaults
            if (c.gpu_wait_s, c.gpu_busy_mib) != (GPU_WAIT_S, GPU_BUSY_MIB):
                self._environment["gpu"] = {"wait_s": c.gpu_wait_s, "busy_mib": c.gpu_busy_mib}
            if c.eval_python is not None:
                self._environment["eval_python"] = os.fspath(c.eval_python)
            self._distribution_files()
        return self._environment

    # ---- before the grant ------------------------------------------------------------------------------------------

    def _interpreter_in_trees(self, interpreter: str) -> bool:
        """The eval trees are mounted at their own paths, so the container path is the host path: it must lie inside
        a configured tree, and resolve (links followed) to a regular file inside a configured tree."""
        def inside(p: str, roots: list[str]) -> bool:
            return any(p == r or p.startswith(r.rstrip("/") + "/") for r in roots)
        named = [os.path.normpath(os.fspath(t)) for t in self.config.eval_trees]
        real_trees = [os.path.realpath(t) for t in self.config.eval_trees]
        if not inside(os.path.normpath(interpreter), named):
            return False
        real = os.path.realpath(interpreter)
        if not inside(real, real_trees):
            return False
        try:
            return stat.S_ISREG(os.lstat(real).st_mode)
        except OSError:
            return False

    def _resolve(self, gen: BatchGenerator) -> BatchGenerator:
        """`gen` with the owner's eval interpreter in place of the one the pack declares, for a generator that runs from
        the eval trees (the setting names the eval venv; an image generator keeps its own interpreter)."""
        if self.config.eval_python is not None and gen.eval_trees:
            return dataclasses.replace(gen, interpreter=os.fspath(self.config.eval_python))
        return gen

    def ready(self, gen: BatchGenerator) -> None:
        """Refused (AN-25, AN-13) unless `gen` can run now; see the module docstring. Called before the grant."""
        gen = self._resolve(gen)
        gpu_wait_s = self.config.gpu_wait_s
        bound = evaluation_wait_bound(gpu_wait_s, gen.time_limit_s)
        wait = protocol.LONG_OP_TIMEOUT_S
        if bound > wait:
            most = int((wait - gpu_wait_s - EVAL_SLACK_S) // 2)
            raise Refused([_problem(RULE_BATCH, "batch_generator.time_limit_s", f"the batch generator's time limit "
                                    f"of {gen.time_limit_s} s does not fit the driver's wait for an evaluation "
                                    f"({gpu_wait_s:g} s GPU wait + 2 x {gen.time_limit_s} s + {EVAL_SLACK_S} s = "
                                    f"{bound} s, more than {wait:g} s)",
                                    f"give the scenario pack's batch generator a time limit of at most {most} s")])
        if gen.eval_trees and not self.config.eval_trees:
            raise Refused([_problem(RULE_BATCH, "batch_generator.eval_trees", "the batch generator needs the "
                                    "keeper's eval trees, and this keeper has no eval tree configured",
                                    "start the keeper with --eval-tree (`gatedtrain setup host` builds "
                                    "/opt/gatedtrain-keeper)")])
        if gen.eval_trees and not self._interpreter_in_trees(gen.interpreter):
            raise Refused([_problem(RULE_BATCH, "batch_generator.interpreter", "the batch generator needs the "
                                    "keeper's eval trees, and its interpreter is not a regular file inside a "
                                    "configured eval tree (links resolved within the trees)",
                                    "point the batch generator's interpreter at the eval venv's python under an "
                                    "--eval-tree of the keeper")])
        if os.path.realpath(gen.script) not in self._distribution_files():
            raise Refused([_problem(RULE_ENV, "batch_generator.script", "the batch generator's script is not a file "
                                    "of a distribution in the keeper's environment record",
                                    "install the scenario pack in the keeper's own environment")])
        problems = self._daemon_problems() + self._image_problems()
        if problems:
            raise Refused(problems)
        if gen.gpu:
            if self.probe.used_mib() is None:
                raise Refused([_problem(RULE_BATCH, "gpu", "the keeper cannot read the GPU's memory use "
                                        "(nvidia-smi), so it does not start a GPU generator",
                                        "install the NVIDIA driver's nvidia-smi, or pass its absolute path to "
                                        "`keeper serve --nvidia-smi`")])
            if not wait_for_gpu(self.probe, busy_mib=self.config.gpu_busy_mib, timeout_s=gpu_wait_s, sleep=self.sleep,
                                clock=self.clock):
                raise Refused([_problem(RULE_BATCH, "gpu", f"the GPU is busy: other processes held more than "
                                        f"{self.config.gpu_busy_mib} MiB of GPU memory for {gpu_wait_s:g} s",
                                        "wait until the other GPU job ends, then evaluate again")])

    # ---- the container --------------------------------------------------------------------------------------------

    def _model_dir(self, model: dict) -> str:
        """The keeper's copy directory of the arm's model files: one directory under `state/models`."""
        files = model.get("files") if isinstance(model, dict) else None
        if not (isinstance(files, list) and files and
                all(isinstance(f, dict) and isinstance(f.get("path"), str) for f in files)):
            raise _crash("the model reference lists no model files for the inference container")
        dirs = {os.path.dirname(f["path"]) for f in files}
        models = os.path.realpath(self.state_dir / "models") + os.sep
        if len(dirs) != 1 or not os.path.realpath(next(iter(dirs))).startswith(models):
            raise _crash("the model files are not one copy directory in the keeper's state")
        return next(iter(dirs))

    def _new_run_dir(self, hexid: str) -> Path:
        base = self.state_dir / RUN_DIR
        try:
            os.mkdir(base, 0o700)
        except FileExistsError:
            pass
        st = os.lstat(base)
        if not stat.S_ISDIR(st.st_mode):
            raise NotADirectoryError("run")
        run_dir = base / hexid
        os.mkdir(run_dir, 0o700)
        os.chmod(run_dir, 0o700)
        return run_dir

    def _spec(self, name: str, gen: BatchGenerator, run_dir: Path, model_dir: str) -> RunSpec:
        mounts = [Mount(str(run_dir / "items.jsonl"), CONTAINER_ITEMS),
                  Mount(str(run_dir / "sampler.json"), CONTAINER_SAMPLER),
                  Mount(model_dir, CONTAINER_MODEL),
                  Mount(str(gen.script), f"{CONTAINER_SCRIPT_DIR}/{gen.script.name}")]
        if gen.eval_trees:
            mounts += [Mount(str(t), str(t)) for t in self.config.eval_trees]
        mounts.append(Mount(str(run_dir / "out"), CONTAINER_OUT, read_only=False))
        return RunSpec(name=name, image=self.config.image_id, argv=container_argv(gen), env=KEEPER_ENV,
                       mounts=tuple(mounts), user=CONTAINER_USER, network="none",
                       gpu_options=launch.CDI_GPU_OPTIONS if gen.gpu else (), read_only_root=True,
                       tmpfs=(("/tmp", launch.TMPFS_OPTIONS),), shm_size=SHM_SIZE, log_driver="none",
                       labels=(KEEPER_LABEL,))

    def _remove_container(self, name: str) -> None:
        """Stop (when running) and remove; best effort."""
        try:
            st = self.runtime.inspect(name)
            if st.status == "missing":
                return
            if st.status != "exited":
                self.runtime.stop(name, timeout_s=STOP_TIMEOUT_S)
            self.runtime.remove(name)
        except (RuntimeFailure, OSError, ValueError):
            pass

    def run_batch(self, gen: BatchGenerator, model: dict, items: list[dict], sampler: dict,
                  seeds: list) -> list[dict]:
        """One container for every item and seed of one arm; the predictions as `batch.read_predictions` returns
        them, or Refused (AN-9): a crash, or a daemon that changed since `ready` (the grant is already made)."""
        gen = self._resolve(gen)
        self.last_run = None
        hexid = secrets.token_hex(8)
        name = NAME_PREFIX + hexid
        run_dir: Path | None = None
        attempted = started = False
        state: ContainerState | None = None
        t0 = self.clock()
        try:
            try:
                model_dir = self._model_dir(model)
                run_dir = self._new_run_dir(hexid)
                write_items(run_dir / "items.jsonl", items)
                _write_new(run_dir / "sampler.json", canonical_json(sampler_doc(sampler, seeds)))
                os.mkdir(run_dir / "out", 0o700)
                spec = self._spec(name, gen, run_dir, model_dir)
                problems = self._daemon_problems()           # immediately before the start, after the grant
                if problems:
                    raise Refused([_problem(RULE_CRASH, "inference", f"{p.message}; the attempt settles as not "
                                            "passed", p.fix) for p in problems])
                attempted = True
                self.runtime.run(spec)
                started = True
                timed_out = False
                deadline = self.clock() + gen.time_limit_s
                while True:
                    state = self.runtime.inspect(name)
                    if state.status == "exited":
                        break
                    if state.status == "missing":
                        state = None
                        raise _crash("the inference container disappeared before it exited")
                    if self.clock() >= deadline:
                        self.runtime.stop(name, timeout_s=STOP_TIMEOUT_S)
                        timed_out = True
                        try:
                            state = self.runtime.inspect(name)
                        except RuntimeFailure:
                            pass
                        break
                    self.sleep(POLL_S)
                if timed_out:
                    raise _crash(f"the inference container ran past its time limit of {gen.time_limit_s} s")
                if state.exit_code != 0:
                    raise _crash(f"the inference container exited with rc {state.exit_code}")
                return read_predictions(run_dir / "out", len(items), seeds)
            except Refused:
                raise
            except RuntimeFailure as exc:                    # the operation and the exit code only
                raise _crash(f"the inference container failed: {exc}") from None
            except Exception as exc:                         # the class only: a message may name a sealed path
                raise _crash(f"the inference container could not be run ({type(exc).__name__})") from None
            finally:
                if started:
                    seconds = _seconds(state)
                    if seconds is None:
                        seconds = max(0, int(round(self.clock() - t0)))
                    self.last_run = {"arm": None, "image_id": self.config.image_id, "seconds": seconds,
                                     "gpu_seconds": seconds if gen.gpu else 0,
                                     "rc": state.exit_code if state is not None else None}
        finally:
            if attempted:                                    # also a container created by a start that failed
                self._remove_container(name)
            if run_dir is not None:
                shutil.rmtree(run_dir, ignore_errors=True)
