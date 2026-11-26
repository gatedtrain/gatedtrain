"""AN-19: `gatedtrain setup keeper` prints the privileged commands that install the keeper; it never runs them.

`setup_commands` returns the commands, in the order the user runs them once with sudo, and the systemd unit text:
the keeper's system user, its state directory (mode 0700, owned by the keeper user), the socket directory (the unit's
`RuntimeDirectory=` when the socket sits directly under `/run/<name>/`, which systemd recreates at every start, else
one `install -d`), a root-owned virtual environment with `gatedtrain==<this version>` and the scenario packs at pinned
versions (root-owned is stricter than owned by the keeper user, design 05 §18.9 AN-13: neither the owner uid nor the
keeper can change the code the keeper runs), the unit file, `systemctl daemon-reload` and `enable --now`, the check
`gatedtrain keeper fingerprint --socket ...`, and a closing comment telling the user to keep that fingerprint out of
band (AN-14, `gatedtrain verify --fingerprint`).

Nothing here starts a process, creates a file or reads more than the installed distributions' metadata; every value
is printed as text, shell-quoted. A path holding `%`, `$`, whitespace, a quote or a backslash is refused, since the
unit's `ExecStart=` would not pass it through unchanged.
"""

from __future__ import annotations

import importlib.metadata
import math
import re
import shlex
from pathlib import PurePosixPath
from typing import Any

from ... import __version__
from ...canonical import plain_decimal
from ...errors import Problem, Refused
from ..runtime import MAX_GPU_BUSY_MIB
from . import protocol

RULE = "AN-19"
DEFAULT_USER = "gatedtrain-keeper"
DEFAULT_STATE = "/var/lib/gatedtrain-keeper"
DEFAULT_SOCKET = "/run/gatedtrain-keeper/keeper.sock"
DEFAULT_VENV = "/opt/gatedtrain-keeper/venv"
UNIT_NAME = "gatedtrain-keeper.service"
UNIT_PATH = f"/etc/systemd/system/{UNIT_NAME}"
SCENARIO_ENTRY_POINT_GROUP = "gatedtrain.scenarios"
KEEP_FINGERPRINT = ("# keep the fingerprint the last command printed somewhere the agent cannot write (a note on "
                    "another device); `gatedtrain verify <project> --fingerprint <fp>` checks a project against it. "
                    "It protects a separate-user keeper only: a same-user keeper's key is readable by your own "
                    "account, so a fingerprint kept away from the agent does not protect it")

_USER = re.compile(r"[a-z_][a-z0-9_-]{0,31}")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_PINNED = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9._,-]+\])?==[A-Za-z0-9][A-Za-z0-9.+!_-]*")


def _refuse(field: str, message: str, fix: str, rule: str = RULE) -> Refused:
    return Refused([Problem(rule, field, message, fix)])


def installed_packs() -> list[str]:
    """`name==version` of every installed distribution that provides a scenario pack (other than gatedtrain)."""
    pins = set()
    for ep in importlib.metadata.entry_points(group=SCENARIO_ENTRY_POINT_GROUP):
        dist = getattr(ep, "dist", None)
        if dist is None:
            continue
        name = dist.metadata["Name"]
        if name and name.lower() != "gatedtrain":
            pins.add(f"{name}=={dist.version}")
    return sorted(pins)


# systemd rewrites `%` (specifiers) and `$` (environment variables) in a unit's ExecStart=, and splits it on
# whitespace and quotes; a path holding one of these would not be the path the keeper is started with.
_UNIT_UNSAFE = re.compile(r"[%$\s'\"\\]")


def plain_absolute(field: str, value: str, rule: str = RULE) -> PurePosixPath:
    """`value` as an absolute path without `..` and without a non-printable character (escape, NUL, line and bidi
    separators, ...): such a character would reach a printed command line raw. The path is shell-quoted where printed
    (not part of a unit's `ExecStart=`); `_absolute` adds the unit rule."""
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or "\n" in value:
        raise _refuse(field, "the path is not an absolute path without `..`", "pass an absolute path", rule)
    if not value.isprintable():
        raise _refuse(field, "the path holds a non-printable character (a control, escape or separator character)",
                      "pass a path of letters, digits, `/`, `.`, `_` and `-`", rule)
    return path


def _absolute(field: str, value: str, rule: str = RULE) -> PurePosixPath:
    path = plain_absolute(field, value, rule)
    if _UNIT_UNSAFE.search(value):
        raise _refuse(field, "the path holds a character systemd rewrites or splits in a unit (`%`, `$`, whitespace, "
                      "a quote or a backslash)", "pass a path of letters, digits, `/`, `.`, `_` and `-`", rule)
    return path


def format_seconds(value: float) -> str:
    """`value` as printed in a unit (`canonical.plain_decimal`: never rounded, so the printed option is the value the
    keeper runs with)."""
    return plain_decimal(value)


def serve_argv(venv: PurePosixPath, state: PurePosixPath, socket: PurePosixPath, owner_uid: int, *,
               inference_image: str | None = None, eval_trees: list[str] | tuple[str, ...] = (),
               docker_host: str | None = None, gpu_wait_s: float | None = None, gpu_busy_mib: int | None = None,
               eval_python: str | None = None) -> list[str]:
    """The keeper's `serve` command as argv, the one source of the unit's `ExecStart=` (here) and of the inference
    drop-in `gatedtrain setup host` prints. The caller has validated every value. The GPU and eval-interpreter options
    (AN-23, AN-25) are printed only when given."""
    argv = [str(venv / "bin" / "gatedtrain"), "keeper", "serve", "--state", str(state), "--socket", str(socket),
            "--owner-uid", str(owner_uid)]
    if inference_image is not None:
        argv += ["--inference-image", inference_image]
    for tree in eval_trees:
        argv += ["--eval-tree", str(tree)]
    if docker_host is not None:
        argv += ["--docker-host", docker_host]
    if gpu_wait_s is not None:
        argv += ["--gpu-wait-s", format_seconds(gpu_wait_s)]
    if gpu_busy_mib is not None:
        argv += ["--gpu-busy-mib", str(gpu_busy_mib)]
    if eval_python is not None:
        argv += ["--eval-python", eval_python]
    return argv


def _unit(user: str, serve: list[str], runtime_dir: str | None) -> str:
    serve = " ".join(shlex.quote(x) for x in serve)
    lines = ["[Unit]",
             "Description=gatedtrain keeper (sealed evaluation, AI-native step 2)",
             "",
             "[Service]",
             "Type=simple",
             f"User={user}",
             f"Group={user}",
             "UMask=0077"]
    if runtime_dir is not None:
        lines += [f"RuntimeDirectory={runtime_dir}", "RuntimeDirectoryMode=0755"]
    lines += [f"ExecStart={serve}",
              "Restart=on-failure",
              "NoNewPrivileges=yes",
              "PrivateTmp=yes",
              "",
              "[Install]",
              "WantedBy=multi-user.target"]
    return "\n".join(lines) + "\n"


def check_eval_tree(tree: str, rule: str = RULE) -> None:
    """Refused unless `tree` is a plain absolute path a docker mount can carry (shared with the layout file, AN-34)."""
    _absolute("--eval-tree", tree, rule)
    if any(c in tree for c in ":,"):
        raise _refuse("--eval-tree", "the eval tree holds `:` or `,`, which a docker mount cannot carry",
                      "pass a path of letters, digits, `/`, `.`, `_` and `-`", rule)


def _inference_options(inference_image: str | None, eval_trees: list[str] | None,
                       docker_host: str | None) -> tuple[str | None, list[str], str | None]:
    """The keeper's inference options (AN-25) checked for the unit's `ExecStart=`; Refused (AN-19) for an image that
    is not an image ID, an eval tree that is not a plain absolute path, a Docker host systemd would rewrite or docker
    would read as an option, or an option without `--inference-image`, as `keeper serve` refuses it."""
    trees = list(eval_trees or [])
    if inference_image is None:
        for field, given in (("--eval-tree", trees), ("--docker-host", docker_host)):
            if given:
                raise _refuse(field, f"{field} needs --inference-image",
                              "pass the inference image ID (sha256:<64 hex>) the keeper's daemon holds")
        return None, [], None
    if not _IMAGE_ID.fullmatch(inference_image):
        raise _refuse("--inference-image", "the inference image is not an image ID sha256:<64 hex>",
                      "pass the ID the keeper's daemon prints for the image (`docker image inspect -f '{{.Id}}'`)")
    for tree in trees:
        check_eval_tree(tree)
    if docker_host is not None and (not docker_host or docker_host.startswith("-") or not docker_host.isprintable()
                                    or _UNIT_UNSAFE.search(docker_host)):
        raise _refuse("--docker-host", "the Docker host is empty, starts with `-`, holds a non-printable character, "
                      "or holds a character systemd rewrites or splits in a unit", "pass unix:///run/user/<keeper uid>/docker.sock")
    return inference_image, trees, docker_host


def check_gpu_wait(value: Any, field: str = "--gpu-wait-s") -> None:
    """Refused (AN-19) unless `value` is a finite number of seconds above 0."""
    try:
        bad = (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               or not 0 < value <= protocol.MAX_GPU_WAIT_S)
    except OverflowError:                                  # an integer too large for a float
        bad = True
    if bad:
        raise _refuse(field, "the GPU wait is not a number of seconds above 0 and at most "
                      f"{protocol.MAX_GPU_WAIT_S:g} (the client's deadline less the slack)",
                      "pass seconds, such as 900")


def check_gpu_busy(value: Any, field: str = "--gpu-busy-mib") -> None:
    """Refused (AN-19) unless `value` is a whole number of MiB from 0 to `MAX_GPU_BUSY_MIB`."""
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_GPU_BUSY_MIB:
        raise _refuse(field, f"the busy threshold is not a whole number of MiB from 0 to {MAX_GPU_BUSY_MIB}",
                      "pass MiB, such as 1024")


def check_eval_python(value: Any, field: str = "--eval-python") -> None:
    """Refused (AN-19) unless `value` is an absolute path a unit can carry (no space, `:`, `,` or systemd character)."""
    if not isinstance(value, str):
        raise _refuse(field, "the interpreter path is not a string", "pass an absolute path")
    _absolute(field, value)
    if any(c in value for c in ":, ") or _UNIT_UNSAFE.search(value):
        raise _refuse(field, "the interpreter path holds a space, `:`, `,` or a character systemd rewrites",
                      "pass a path of letters, digits, `/`, `.`, `_` and `-`")


def _gpu_options(gpu_wait_s: float | None, gpu_busy_mib: int | None, eval_python: str | None,
                 inference_image: str | None) -> None:
    """The keeper's GPU and eval-interpreter options checked for the unit's `ExecStart=` (the checks `InferenceConfig`
    makes); Refused (AN-19) for a bad value, or an option without `--inference-image`."""
    if inference_image is None:
        for field, given in (("--gpu-wait-s", gpu_wait_s), ("--gpu-busy-mib", gpu_busy_mib),
                             ("--eval-python", eval_python)):
            if given is not None:
                raise _refuse(field, f"{field} needs --inference-image",
                              "pass the inference image ID (sha256:<64 hex>) the keeper's daemon holds")
        return
    if gpu_wait_s is not None:
        check_gpu_wait(gpu_wait_s)
    if gpu_busy_mib is not None:
        check_gpu_busy(gpu_busy_mib)
    if eval_python is not None:
        check_eval_python(eval_python)


def setup_commands(*, user: str = DEFAULT_USER, state: str = DEFAULT_STATE, socket: str = DEFAULT_SOCKET,
                   venv: str = DEFAULT_VENV, owner_uid: int, packages: list[str] | None = None,
                   inference_image: str | None = None, eval_trees: list[str] | None = None,
                   docker_host: str | None = None, gpu_wait_s: float | None = None, gpu_busy_mib: int | None = None,
                   eval_python: str | None = None) -> dict:
    """`{commands: [str], unit: str}` for installing the keeper (AN-19). Refused (AN-19) on a user name, path or
    package spec that is not safe to print as a command: an unpinned package would let the keeper run whatever
    version is current when the user runs the command. The inference options (AN-25) go into `ExecStart=`."""
    if not _USER.fullmatch(user):
        raise _refuse("--user", "the user name is not a plain system user name ([a-z_][a-z0-9_-]*, at most 32 "
                      "characters)", f"pass a name such as {DEFAULT_USER}")
    if isinstance(owner_uid, bool) or not isinstance(owner_uid, int) or owner_uid < 0:
        raise _refuse("--owner-uid", "the owner uid is not a non-negative integer", "pass your uid (`id -u`)")
    state_p, socket_p, venv_p = (_absolute(f, v) for f, v in (("--state", state), ("--socket", socket),
                                                               ("--venv", venv)))
    if len(str(socket_p).encode()) > protocol.MAX_SOCKET_PATH:
        raise _refuse("--socket", f"the socket path is longer than {protocol.MAX_SOCKET_PATH} bytes",
                      "use a shorter path, such as the default")
    image, trees, host = _inference_options(inference_image, eval_trees, docker_host)
    _gpu_options(gpu_wait_s, gpu_busy_mib, eval_python, inference_image)
    pins = installed_packs() if packages is None else list(packages)
    for spec in pins:
        if not _PINNED.fullmatch(spec):
            raise _refuse("--packages", "a package is not pinned to one version (name==version)",
                          "pass each scenario pack as name==version")
    q = shlex.quote
    sock_dir = socket_p.parent
    runtime_dir = sock_dir.name if sock_dir.parent == PurePosixPath("/run") else None
    unit = _unit(user, serve_argv(venv_p, state_p, socket_p, owner_uid, inference_image=image, eval_trees=trees,
                                  docker_host=host, gpu_wait_s=gpu_wait_s, gpu_busy_mib=gpu_busy_mib,
                                  eval_python=eval_python), runtime_dir)
    commands = [
        f"sudo useradd --system --user-group --home-dir {q(str(state_p))} --no-create-home "
        f"--shell /usr/sbin/nologin {q(user)}",
        f"sudo install -d -o {q(user)} -g {q(user)} -m 0700 {q(str(state_p))}",
    ]
    if runtime_dir is None:            # systemd creates /run/<name> itself (RuntimeDirectory=), again at every boot
        commands.append(f"sudo install -d -o {q(user)} -g {q(user)} -m 0755 {q(str(sock_dir))}")
    commands += [
        f"sudo install -d -o root -g root -m 0755 {q(str(venv_p.parent))}",
        f"sudo python3 -m venv {q(str(venv_p))}",
        f"sudo {q(str(venv_p / 'bin' / 'pip'))} install --no-cache-dir "
        + " ".join(q(p) for p in [f"gatedtrain=={__version__}", *pins]),
        f"sudo chmod -R go-w {q(str(venv_p))}",
        f"sudo tee {UNIT_PATH} > /dev/null <<'EOF'\n{unit}EOF",
        "sudo systemctl daemon-reload",
        f"sudo systemctl enable --now {UNIT_NAME}",
        f"gatedtrain keeper fingerprint --socket {q(str(socket_p))}",
        KEEP_FINGERPRINT,
    ]
    return {"commands": commands, "unit": unit}
