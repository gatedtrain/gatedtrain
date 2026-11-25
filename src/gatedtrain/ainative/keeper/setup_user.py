"""AN-31, AN-30: `gatedtrain setup keeper --mode same-user` prints user-level commands only; it never runs them.

Same-user mode has no privileged step. The commands create the keeper's state directory (mode 0700) under the user's
XDG data directory, a systemd *user* unit that runs `gatedtrain keeper serve` from the CURRENT environment (the
interpreter's own `gatedtrain`, with the scenario packs installed there), and print the fingerprint check. Every
location derives from the XDG variables the user's session sets, falling back to the standard XDG defaults under the
home directory and `/run/user/<uid>`; nothing is a fact about one machine. The commands are idempotent and never change an existing
directory: the state directory (and the socket directory, when systemd does not manage it) is created with `mkdir -m
0700` (the mode applies to the leaf only), and an existing one is checked, not touched: it must be owned by you with
mode 0700 or the printed script stops with a message (`install -d -m` would reset the mode of a shared directory).
The printed block is ONE script: its first line is `set -e` and every guard exits non-zero, so nothing after a failed
guard (the unit write, `systemctl --user enable --now`) runs; run it with `sh`, or paste it whole. The unit
rewrite can be run again, `systemctl --user enable --now` is a no-op for a running unit. A `--state` or `--socket`
whose directory is your home, the runtime base or a shared directory (sticky bit, or owned by another user) is
refused.

`NoNewPrivileges=yes` in the unit is UNVERIFIED against rootless Docker: the rootless daemon starts its own helpers
through setuid `newuidmap`/`newgidmap` in the user's own manager, not in this unit, but a keeper that starts rootless
containers itself was not run with the flag in this environment (no Docker was available); drop the line from the
unit if the keeper cannot start its inference containers.

This mode does NOT seal anything from the user's account (the texts say so, keeper/texts.py); `--mode separate-user`
is the hardened install (keeper/setup.py).
"""

from __future__ import annotations

import os
import shlex
import stat
import sys
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath

from ...errors import Problem, Refused, printable
from . import protocol
from .setup import (_UNIT_UNSAFE, KEEP_FINGERPRINT, UNIT_NAME, _absolute, _gpu_options, _inference_options,
                    format_seconds, plain_absolute)

RULE = "AN-31"                   # the same-user commands
MODE_RULE = "AN-30"              # an option that belongs to the other mode

# Logical names under the XDG base directories (one place, so the unit, the commands and the tests agree).
APP_DIR = "gatedtrain-keeper"                       # the state directory and the runtime directory name
SOCKET_NAME = "keeper.sock"
UNIT_SUBDIR = "systemd/user"                        # where systemd's user manager reads unit files, under CONFIG
XDG_DATA_FALLBACK = ".local/share"                  # XDG Base Directory spec defaults, relative to the home
XDG_CONFIG_FALLBACK = ".config"
RUNTIME_FALLBACK = "/run/user/{uid}"                # the runtime directory logind creates for a user (systemd docs)

HEADER = ("# same-user mode: no sudo; run these as YOUR user, not with sudo. The keeper runs under your own OS user, so it does NOT seal data from your "
          "account (an agent with your shell can read it). To harden: gatedtrain setup keeper --mode separate-user "
          "(one sudo).")
CHANGES = ("# what these commands change: they create the directories and the one user unit file named below and "
           "enable the unit, nothing else; a new state or socket directory gets mode 0700, an existing directory is "
           "never changed (it must already be yours with mode 0700, else the script stops there); rerunning is safe")
SCRIPT_NOTE = ("# run this as one script with sh (sh FILE), or paste it whole: the next line makes it stop at the first "
               "failing step, so nothing after a failed check runs")
STOP_ON_ERROR = "set -e"
# `setup keeper` without --mode printed the separate-user commands before 0.7.0; say so once (the output is the new one).
DEFAULT_MODE_WARNING = ("the default changed in 0.7.0 to same-user (no sudo); the previous output is --mode "
                        "separate-user")


def _refuse(field: str, message: str, fix: str, rule: str = RULE) -> Refused:
    return Refused([Problem(rule, field, message, fix)])


def refuse_empty(**values: str | None) -> None:
    """AN-31 (both modes): an option given as an empty string is refused. It is neither "unset" (which means the
    default) nor a path; reading it as the default in one mode and as a bad path in the other hid a typo."""
    for name, value in values.items():
        if value == "":
            flag = "--" + name
            raise _refuse(flag, f"{flag} is empty", f"give {flag} a value, or leave it out for the default")


def mode_option_problem(field: str) -> Refused:
    """AN-30: `--user` and `--venv` belong to separate-user mode; same-user has no system user and no venv."""
    return _refuse(field, f"{field} belongs to separate-user mode; same-user mode has no system user and no venv "
                   "(the keeper runs from the current environment under your own user)",
                   f"drop {field}, or use --mode separate-user", MODE_RULE)


def keeper_executable(python: str | None = None) -> str | None:
    """The `gatedtrain` console script next to this interpreter (the environment the keeper will run from), as an
    absolute path, else None. The PATH is NOT searched: a `gatedtrain` found there may belong to another environment,
    with other packs, so the caller refuses instead of guessing. The interpreter path is not resolved (a venv's python
    is a link to the base one), only made absolute."""
    here = Path(os.path.abspath(python or sys.executable)).parent / "gatedtrain"
    return str(here) if here.is_file() and os.access(here, os.X_OK) else None


def _no_executable(python: str | None = None) -> Refused:
    where = printable(str(Path(os.path.abspath(python or sys.executable)).parent))
    return _refuse("--executable", f"no `gatedtrain` command exists next to this Python ({where}), and the PATH is not "
                   "searched (a `gatedtrain` there may belong to another environment)",
                   "install gatedtrain in this environment (pip install gatedtrain), or name the command with "
                   "--executable PATH (layout file: [same_user] executable)")


def _executable_path(value: str) -> PurePosixPath:
    """The keeper's command as it goes into `ExecStart=`: refused unless absolute and plain, naming the options that
    let a venv at an unusual path be used (a link or copy at a plain path, given as `--executable`)."""
    path = plain_absolute("--executable", value, RULE)
    if _UNIT_UNSAFE.search(value):
        raise _refuse("--executable", "the path of the gatedtrain command holds a character systemd rewrites or "
                      "splits in a unit (`%`, `$`, whitespace, a quote or a backslash), so `ExecStart=` cannot carry it",
                      "make a link to it (or a copy) at a plain path and pass that as --executable PATH (layout file: "
                      "[same_user] executable); --layout cannot change the path of this environment's own command")
    return path


def executable_mode_problem() -> Refused:
    """AN-30: `--executable` belongs to same-user mode (separate-user runs the keeper from its own root-owned venv)."""
    return _refuse("--executable", "--executable belongs to same-user mode; separate-user mode runs the keeper from "
                   "its own root-owned venv (--venv)", "drop --executable, or use --mode same-user", MODE_RULE)


def _xdg(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    return value if value and os.path.isabs(value) else None      # the XDG spec: a relative value is invalid


def default_paths(owner_uid: int, env: Mapping[str, str], home: str | None) -> dict[str, str]:
    """State directory, socket and unit file under the XDG directories of this user."""
    base = home if home is not None else env.get("HOME") or ""
    if not os.path.isabs(base):
        base = ""
    data = _xdg(env, "XDG_DATA_HOME") or (f"{base}/{XDG_DATA_FALLBACK}" if base else None)
    config = _xdg(env, "XDG_CONFIG_HOME") or (f"{base}/{XDG_CONFIG_FALLBACK}" if base else None)
    runtime = _xdg(env, "XDG_RUNTIME_DIR") or RUNTIME_FALLBACK.format(uid=owner_uid)
    if data is None or config is None:
        raise _refuse("HOME", "the home directory is unknown, so the XDG data and config directories cannot be found",
                      "set HOME (or XDG_DATA_HOME and XDG_CONFIG_HOME) to absolute paths, or pass --state")
    return {"state": f"{data}/{APP_DIR}", "runtime": runtime, "socket": f"{runtime}/{APP_DIR}/{SOCKET_NAME}",
            "unit_dir": f"{config}/{UNIT_SUBDIR}"}


def _private_dir(path: PurePosixPath) -> str:
    """Shell lines that make `path` a private directory without ever changing an existing one: parents are made plain
    (`mkdir -p` leaves existing ones alone), the leaf is made `mkdir -m 0700` (the mode applies to the leaf only) when
    absent, and the leaf must then be owned by the user with mode 0700, else the script prints why and exits non-zero.
    Every step exits on failure itself, so the block also stops when it is not under `set -e`."""
    q = shlex.quote
    leaf = q(str(path))
    msg = f"stopped: {path} exists and is not a directory owned by you with mode 0700; it was not changed"
    # The message belongs to the ownership/mode check only: a failing `mkdir` prints its own error and exits.
    return (f"mkdir -p {q(str(path.parent))} || exit 1\n"
            f"[ -d {leaf} ] || mkdir -m 0700 {leaf} || exit 1\n"
            f'[ "$(stat -c %u:%a {leaf})" = "$(id -u):700" ] || {{ echo {q(msg)} >&2; exit 1; }}')


def _unit_path(field: str, value: str, key: str) -> PurePosixPath:
    """A state or socket path as it goes into `ExecStart=`; a refusal says how to choose the location yourself."""
    try:
        return _absolute(field, value, RULE)
    except Refused as exc:
        p = exc.problems[0]
        raise Refused([Problem(p.rule, p.field, p.message, f"{p.fix}, or choose the location yourself with {field} "
                               f"PATH (layout file: [same_user] {key})")]) from None


def _shared_location(field: str, directory: PurePosixPath, *, owner_uid: int, home: str | None,
                     runtime_base: PurePosixPath, lstat: Callable[[str], os.stat_result]) -> Refused | None:
    """AN-31: a directory the keeper's files must not live in: `/`, the home, the runtime base itself, the system temp
    directory, or an existing directory that is shared (sticky bit) or owned by another user."""
    import tempfile

    system = {PurePosixPath("/"), runtime_base, PurePosixPath(tempfile.gettempdir())}
    if home:
        system.add(PurePosixPath(home))
    why = None
    if directory in system:
        why = "is your home, the runtime base or a system directory"
    else:
        try:
            st = lstat(str(directory))
        except OSError:
            st = None                                              # it does not exist (yet): `mkdir` will create it
        if st is not None and stat.S_ISDIR(st.st_mode) and (st.st_mode & stat.S_ISVTX or st.st_uid != owner_uid):
            why = "is a shared directory (sticky bit) or owned by another user"
    if why is None:
        return None
    return _refuse(field, f"{directory} {why}; the keeper's private files cannot live there",
                   "pass a path under your own data directory (the default), e.g. a new subdirectory")


def _unit(serve: list[str], runtime_dir: str | None) -> str:
    lines = ["[Unit]",
             "Description=gatedtrain keeper (same-user mode: procedure followed, sealed data not protected from "
             "your account)",
             "",
             "[Service]",
             "Type=simple",
             "UMask=0077"]
    if runtime_dir is not None:
        lines.append(f"RuntimeDirectory={runtime_dir}")
    lines += ["ExecStart=" + " ".join(shlex.quote(x) for x in serve),
              "Restart=on-failure",
              "NoNewPrivileges=yes",
              "",
              "[Install]",
              "WantedBy=default.target"]
    return "\n".join(lines) + "\n"


def setup_commands_same_user(*, owner_uid: int, state: str | None = None, socket: str | None = None,
                             inference_image: str | None = None, eval_trees: list[str] | None = None,
                             docker_host: str | None = None, gpu_wait_s: float | None = None,
                             gpu_busy_mib: int | None = None, eval_python: str | None = None,
                             env: Mapping[str, str] | None = None,
                             home: str | None = None, executable: str | None = None,
                             lstat: Callable[[str], os.stat_result] = os.lstat) -> dict:
    """`{commands, unit, mode}` for a keeper under the owner's own user (AN-31). Refused (AN-31) on a path that is
    not safe to print, an unknown home, or no `gatedtrain` console script to run; the inference options (AN-25) go
    into `ExecStart=` as for separate-user mode. `env`/`home`/`executable`/`lstat` are the host's facts, injectable so the
    commands are testable on any host (defaults: the process environment, this interpreter's `gatedtrain`)."""
    if isinstance(owner_uid, bool) or not isinstance(owner_uid, int) or owner_uid < 0:
        raise _refuse("--owner-uid", "the owner uid is not a non-negative integer", "pass your uid (`id -u`)")
    env = os.environ if env is None else env
    if home is None and not env.get("HOME"):
        try:
            home = str(Path.home())
        except (RuntimeError, KeyError):
            home = None
    paths = default_paths(owner_uid, env, home)
    state_p = _unit_path("--state", state if state is not None else paths["state"], "state")
    socket_p = _unit_path("--socket", socket if socket is not None else paths["socket"], "socket")
    unit_dir = plain_absolute("unit directory", paths["unit_dir"], RULE)       # not part of ExecStart: shell-quoted
    if len(str(socket_p).encode()) > protocol.MAX_SOCKET_PATH:
        raise _refuse("--socket", f"the socket path is longer than {protocol.MAX_SOCKET_PATH} bytes",
                      "pass a shorter --socket")
    try:
        image, trees, host = _inference_options(inference_image, eval_trees, docker_host)
        _gpu_options(gpu_wait_s, gpu_busy_mib, eval_python, inference_image)
    except Refused as exc:                                  # the same checks, reported under this mode's rule
        raise Refused([Problem(RULE, p.field, p.message, p.fix) for p in exc.problems]) from None
    exe = executable if executable is not None else keeper_executable()
    if exe is None:
        raise _no_executable()
    exe_p = _executable_path(exe)
    serve = [str(exe_p), "keeper", "serve", "--state", str(state_p), "--socket", str(socket_p),
             "--owner-uid", str(owner_uid)]
    if image is not None:
        serve += ["--inference-image", image]
    for tree in trees:
        serve += ["--eval-tree", tree]
    if host is not None:
        serve += ["--docker-host", host]
    for flag, value in (("--gpu-wait-s", gpu_wait_s), ("--gpu-busy-mib", gpu_busy_mib), ("--eval-python", eval_python)):
        if value is not None:
            serve += [flag, format_seconds(value) if flag == "--gpu-wait-s" else str(value)]
    sock_dir = socket_p.parent
    runtime_base = PurePosixPath(paths["runtime"])
    for field, directory in (("--state", state_p), ("--socket", sock_dir)):
        problem = _shared_location(field, directory, owner_uid=owner_uid, home=home or env.get("HOME"),
                                   runtime_base=runtime_base, lstat=lstat)
        if problem is not None:
            raise problem
    runtime_dir = sock_dir.name if sock_dir.parent == runtime_base else None   # systemd recreates it at every start
    unit = _unit(serve, runtime_dir)
    q = shlex.quote
    unit_path = unit_dir / UNIT_NAME
    commands = [CHANGES, SCRIPT_NOTE, STOP_ON_ERROR, _private_dir(state_p)]
    if runtime_dir is None:
        commands.append(_private_dir(sock_dir))
    commands += [
        f"mkdir -p {q(str(unit_dir))}",
        f"tee {q(str(unit_path))} > /dev/null <<'EOF'\n{unit}EOF",
        "systemctl --user daemon-reload",
        f"systemctl --user enable --now {UNIT_NAME}",
        f"# to apply changed options to a unit that is already running (enable --now does not restart it): "
        f"systemctl --user restart {UNIT_NAME}",
        "# no systemd user session? run it in a terminal: " + " ".join(q(x) for x in serve),
        "# optional, so the keeper survives logout: loginctl enable-linger (may ask an administrator; "
        "without it the keeper stops when you log out)",
        f"{q(str(exe_p))} keeper fingerprint --socket {q(str(socket_p))}",
        KEEP_FINGERPRINT,
    ]
    return {"commands": commands, "unit": unit, "mode": "same_user"}
