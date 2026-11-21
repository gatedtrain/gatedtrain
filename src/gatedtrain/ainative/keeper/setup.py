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
is printed as text, shell-quoted.
"""

from __future__ import annotations

import importlib.metadata
import re
import shlex
from pathlib import PurePosixPath

from ... import __version__
from ...errors import Problem, Refused
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
                    "another device); `gatedtrain verify <project> --fingerprint <fp>` checks a project against it")

_USER = re.compile(r"[a-z_][a-z0-9_-]{0,31}")
_PINNED = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9._,-]+\])?==[A-Za-z0-9][A-Za-z0-9.+!_-]*")


def _refuse(field: str, message: str, fix: str) -> Refused:
    return Refused([Problem(RULE, field, message, fix)])


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


def _absolute(field: str, value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or "\n" in value:
        raise _refuse(field, "the path is not an absolute path without `..`", "pass an absolute path")
    return path


def _unit(user: str, state: PurePosixPath, socket: PurePosixPath, venv: PurePosixPath, owner_uid: int,
          runtime_dir: str | None) -> str:
    serve = " ".join(shlex.quote(str(x)) for x in (venv / "bin" / "gatedtrain", "keeper", "serve", "--state", state,
                                                   "--socket", socket, "--owner-uid", owner_uid))
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


def setup_commands(*, user: str = DEFAULT_USER, state: str = DEFAULT_STATE, socket: str = DEFAULT_SOCKET,
                   venv: str = DEFAULT_VENV, owner_uid: int, packages: list[str] | None = None) -> dict:
    """`{commands: [str], unit: str}` for installing the keeper (AN-19). Refused (AN-19) on a user name, path or
    package spec that is not safe to print as a command: an unpinned package would let the keeper run whatever
    version is current when the user runs the command."""
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
    pins = installed_packs() if packages is None else list(packages)
    for spec in pins:
        if not _PINNED.fullmatch(spec):
            raise _refuse("--packages", "a package is not pinned to one version (name==version)",
                          "pass each scenario pack as name==version")
    q = shlex.quote
    sock_dir = socket_p.parent
    runtime_dir = sock_dir.name if sock_dir.parent == PurePosixPath("/run") else None
    unit = _unit(user, state_p, socket_p, venv_p, owner_uid, runtime_dir)
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
