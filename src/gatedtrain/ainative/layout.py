"""AN-34: the setup layout file, one TOML file for `gatedtrain setup host` and `gatedtrain setup keeper`.

The owner passes `--layout FILE` instead of a dozen path flags. The file is read as input and PRINTED into the setup
output; nothing reads it afterwards (the keeper never does), nothing is written, and there is no default location and
no environment variable that names one. Precedence for every value: an explicit flag, then the layout file, then the
built-in default (the existing `DEFAULT_*` constants; this module repeats none of them). The printed commands start
with the file's path and SHA-256 and list the source of every value that is not a default, so a reader of the output
sees which values the owner's file chose.

`LAYOUT_KEYS` is the one registry: table, key, type, validator, the flag each scope has for it, default, whether this
version supports it, and the line of documentation `gatedtrain setup layout` prints above it. Adding an option to
`setup host` or `setup keeper` means adding a row (or an entry in `NOT_LAYOUT_FLAGS` with the reason); a test compares
the registry with both commands' argparse flags in both directions, so a flag cannot be forgotten.

A *scope* is what a layout is read for: `host` (`setup host`), `separate-user` or `same-user` (`setup keeper --mode`).
Tables that do not apply to the scope are refused with the scope named.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from ..errors import Problem, Refused
from . import hostsetup
from .keeper import protocol, setup, setup_user

RULE = "AN-34"
SCHEMA_VERSION = 1
# A layout is a few dozen lines. The cap keeps a wrong file (a log, a dump, a device) from being read whole and parsed;
# 64 KiB is far above any real layout.
MAX_LAYOUT_BYTES = 64 * 1024
UNSUPPORTED = "not supported by this version (the keeper has no such option yet)"
SOURCES = ("flag", "layout", "default")

HOST, SEPARATE, SAME = "host", "separate-user", "same-user"
SCOPES = (HOST, SEPARATE, SAME)
SCOPE_LABEL = {HOST: "`setup host`", SEPARATE: "`setup keeper --mode separate-user`",
               SAME: "`setup keeper --mode same-user`"}

TABLES = ("keeper", "same_user", "host", "inference", "gpu")
# Why a flag of `setup host` or `setup keeper` has no layout key (the registry test requires every other flag to have
# one). `--layout` itself is the file.
NOT_LAYOUT_FLAGS: dict[tuple[str, str], str] = {
    ("host", "--layout"): "names the layout file itself",
    ("keeper", "--layout"): "names the layout file itself",
    ("host", "--owner"): "who runs the setup is a fact of this invocation, not of the machine's layout",
    ("host", "--owner-uid"): "who runs the setup is a fact of this invocation, not of the machine's layout",
    ("keeper", "--owner-uid"): "who the keeper serves is a fact of this invocation, not of the machine's layout",
    ("host", "--confirm-group-removal"): "a switch for this one run",
    ("keeper", "--mode"): "chooses which tables of the layout apply",
    ("keeper", "--packages"): "the packs to install are decided when the command runs (default: those installed)",
    ("keeper", "--inference-image"): "an image ID exists only after the keeper's daemon pulled the image; "
                                     "`setup host` prints it",
    ("keeper", "--docker-host"): "derived from the keeper's uid by default; given per invocation",
}


def _scope_command(scope: str) -> str:
    return "host" if scope == HOST else "keeper"


@dataclass(frozen=True)
class Key:
    table: str
    name: str
    kind: str                                   # "str", "list", "number" or "int"
    check: Callable[[Any, str], None] | None    # (value, scope); raises Refused with the existing validator's problem
    flags: Mapping[str, str | None]             # scope -> the flag it replaces (None: the key has no flag)
    doc: str
    default: Callable[[], Any] | None = None    # the existing DEFAULT_* constant; None: no default
    computed: str | None = None                 # a default derived from the XDG directories (same-user mode)
    placeholder: str = ""                       # what the printed layout shows for a key without a default
    supported: bool = True

    @property
    def dotted(self) -> str:
        return f"{self.table}.{self.name}"


# ---- validators: the ones the flags use; each raises Refused, converted to AN-34 problems by `_problems`

def _user(value: str, scope: str) -> None:
    if not setup._USER.fullmatch(value):
        raise setup._refuse("user", "the user name is not a plain system user name ([a-z_][a-z0-9_-]*, at most 32 "
                            "characters)", f"pass a name such as {setup.DEFAULT_USER}")


def _path(value: str, scope: str) -> None:
    if scope == HOST:
        hostsetup._path("path", value)
    else:
        setup._absolute("path", value)


def _socket(value: str, scope: str) -> None:
    _path(value, scope)
    if len(value.encode()) > protocol.MAX_SOCKET_PATH:
        raise setup._refuse("socket", f"the socket path is longer than {protocol.MAX_SOCKET_PATH} bytes",
                            "use a shorter path")


def _same_user_path(value: str, scope: str) -> None:
    setup._absolute("path", value, setup_user.RULE)


def _same_user_socket(value: str, scope: str) -> None:
    _same_user_path(value, scope)
    if len(value.encode()) > protocol.MAX_SOCKET_PATH:
        raise setup._refuse("socket", f"the socket path is longer than {protocol.MAX_SOCKET_PATH} bytes",
                            "use a shorter path")


def _same_user_executable(value: str, scope: str) -> None:
    setup_user._executable_path(value)


def _keeper_eval_python(value: str, scope: str) -> None:
    setup.check_eval_python(value, "keeper_eval_python")


def _gpu_wait(value: Any, scope: str) -> None:
    setup.check_gpu_wait(value, "wait_s")


def _gpu_busy(value: Any, scope: str) -> None:
    setup.check_gpu_busy(value, "busy_mib")


def _image(value: str, scope: str) -> None:
    hostsetup.check_inference_ref(value)


def _file(value: str, scope: str) -> None:
    """A file the setup reads (the requirements): absolute, so it does not depend on the directory it is run from."""
    if not PurePosixPath(value).is_absolute() or not value.isprintable():
        raise setup._refuse("file", "the path is not an absolute path", "pass an absolute path")


def _tree(value: str, scope: str) -> None:
    setup.check_eval_tree(value)


def _dir(name: str) -> Callable[[], str]:
    return lambda: getattr(hostsetup, name)


def _kf(sep: str | None, host: str | None) -> dict[str, str | None]:
    return {SEPARATE: sep, HOST: host}


LAYOUT_KEYS: tuple[Key, ...] = (
    Key("keeper", "user", "str", _user, _kf("--user", "--keeper-user"),
        "the keeper's system user", default=lambda: setup.DEFAULT_USER),
    Key("keeper", "state", "str", _path, _kf("--state", "--state"),
        "the keeper's state directory", default=lambda: setup.DEFAULT_STATE),
    Key("keeper", "socket", "str", _socket, _kf("--socket", "--socket"),
        "the keeper's Unix socket", default=lambda: setup.DEFAULT_SOCKET),
    Key("keeper", "venv", "str", _path, _kf("--venv", "--venv"),
        "the keeper's root-owned virtual environment", default=lambda: setup.DEFAULT_VENV),
    Key("same_user", "state", "str", _same_user_path, {SAME: "--state"},
        "the keeper's state directory (default: under your XDG data directory)", computed="state"),
    Key("same_user", "socket", "str", _same_user_socket, {SAME: "--socket"},
        "the keeper's Unix socket (default: under your XDG runtime directory)", computed="socket"),
    Key("same_user", "executable", "str", _same_user_executable, {SAME: "--executable"},
        "the gatedtrain command the keeper's unit runs (default: the one next to this Python; set it when that path "
        "holds a space, `%` or `$`: point it at a link at a plain path)", placeholder='"PATH"'),
    Key("host", "keeper_home", "str", _path, {HOST: "--keeper-home"},
        "the keeper user's home directory", default=_dir("DEFAULT_KEEPER_HOME")),
    Key("host", "keeper_data_root", "str", _path, {HOST: "--keeper-data-root"},
        "the keeper's rootless Docker data root", default=_dir("DEFAULT_KEEPER_DATA_ROOT")),
    Key("host", "owner_data_root", "str", _path, {HOST: "--owner-data-root"},
        "the owner's rootless Docker data root", default=_dir("DEFAULT_OWNER_DATA_ROOT")),
    Key("host", "handoff", "str", _path, {HOST: "--handoff"},
        "the hand-off directory (the owner's output_root)", default=_dir("DEFAULT_HANDOFF")),
    Key("host", "eval_venv", "str", _path, {HOST: "--eval-venv"},
        "the keeper's root-owned eval venv", default=_dir("DEFAULT_EVAL_VENV")),
    Key("host", "eval_python_root", "str", _path, {HOST: None},
        "the root-owned tree holding the standalone CPython the eval venv is created from "
        "(layout only: there is no flag)", default=_dir("DEFAULT_EVAL_PYTHON_ROOT")),
    Key("inference", "image", "str", _image, {HOST: "--inference-image"},
        "the inference base image, pinned by digest, the keeper's daemon pulls",
        placeholder='"NAME@sha256:<64 hex>"'),
    Key("inference", "eval_requirements", "str", _file, {HOST: "--eval-requirements"},
        "absolute path of the hash-pinned requirements of the keeper's eval venv (needs eval_python)",
        placeholder='"FILE"'),
    Key("inference", "eval_python", "str", _path, {HOST: "--eval-python"},
        "the interpreter the eval venv is created from, under eval_python_root (needs eval_requirements)",
        placeholder='"PATH"'),
    Key("inference", "eval_trees", "list", _tree, {SEPARATE: "--eval-tree", SAME: "--eval-tree"},
        "root-owned eval trees printed into ExecStart= (the flag needs --inference-image)",
        placeholder='["DIR"]'),
    Key("inference", "keeper_eval_python", "str", _keeper_eval_python, {SEPARATE: "--eval-python", SAME: "--eval-python"},
        "the eval venv's python that replaces the one a pack declares (keeper serve option; needs image)",
        placeholder='"PATH"'),
    Key("gpu", "wait_s", "number", _gpu_wait, {SEPARATE: "--gpu-wait-s", SAME: "--gpu-wait-s"},
        "seconds to wait for a free GPU before a GPU generator is refused (needs image)", placeholder="N"),
    Key("gpu", "busy_mib", "int", _gpu_busy, {SEPARATE: "--gpu-busy-mib", SAME: "--gpu-busy-mib"},
        "GPU memory in use (MiB) that still counts as a free GPU (needs image)", placeholder="N"),
)


def keys_for(scope: str, *, supported_only: bool = True) -> list[Key]:
    return [k for k in LAYOUT_KEYS if scope in k.flags and (k.supported or not supported_only)]


def _dest(flag: str) -> str:
    return flag[2:].replace("-", "_")


# ---- reading and validating the file

@dataclass(frozen=True)
class Layout:
    path: str                       # as the owner gave it
    sha256: str                     # of the bytes read
    values: dict[tuple[str, str], Any] = field(default_factory=dict)


def _refuse_layout(path: str, message: str, fix: str) -> Refused:
    return Refused([Problem(RULE, path, message, fix)])


def _read(path: str) -> bytes:
    """The file's bytes, opened without following a symlink; refused unless a regular file of at most
    `MAX_LAYOUT_BYTES`. O_NONBLOCK so a FIFO cannot hang the open (the fstat check then refuses it)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    except OSError as exc:
        raise _refuse_layout(path, f"the layout file cannot be opened without following links "
                             f"({type(exc).__name__})", "pass a regular file, not a symlink") from None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise _refuse_layout(path, "the layout file is not a regular file", "pass a regular file")
        with os.fdopen(fd, "rb", closefd=False) as fh:
            data = fh.read(MAX_LAYOUT_BYTES + 1)
    except OSError as exc:
        raise _refuse_layout(path, f"the layout file cannot be read ({type(exc).__name__})",
                             "pass a readable regular file") from None
    finally:
        os.close(fd)
    if len(data) > MAX_LAYOUT_BYTES:
        raise _refuse_layout(path, f"the layout file is larger than {MAX_LAYOUT_BYTES} bytes",
                             "a layout is a short file; pass the right one")
    return data


def _type_ok(key: Key, value: Any) -> bool:
    if key.kind == "str":
        return isinstance(value, str)
    if key.kind == "list":
        return isinstance(value, list) and all(isinstance(v, str) for v in value)
    if key.kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, (int, float)) and not isinstance(value, bool)


_KIND_WORD = {"str": "a string", "list": "a list of strings", "int": "a whole number", "number": "a number"}


def _problems(path: str, key: Key, exc: Refused) -> list[Problem]:
    return [Problem(RULE, f"{path}: [{key.table}] {key.name}", p.message, p.fix) for p in exc.problems]


def parse(data: bytes, path: str, scope: str) -> dict[tuple[str, str], Any]:
    """The layout's values for `scope` as {(table, key): value}; every problem of the document is reported at once
    (a bad `schema_version` or a TOML error stops first)."""
    try:
        doc = tomllib.loads(data.decode("utf-8"))
    except RecursionError:                      # nesting well under the size cap can still exceed the parser's depth
        raise _refuse_layout(path, "the layout file is nested too deeply for TOML (a layout is flat tables of "
                             "plain values)", "fix the file (`gatedtrain setup layout` prints a valid one)") from None
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise _refuse_layout(path, f"the layout file is not valid TOML ({type(exc).__name__}: {exc})",
                             "fix the file (`gatedtrain setup layout` prints a valid one)") from None
    version = doc.pop("schema_version", None)
    if isinstance(version, bool) or version != SCHEMA_VERSION or not isinstance(version, int):
        raise _refuse_layout(path, f"schema_version is {'missing' if version is None else repr(version)[:40]}; this "
                             f"version reads exactly {SCHEMA_VERSION}", f"start the file with `schema_version = "
                             f"{SCHEMA_VERSION}` (a newer layout needs a newer gatedtrain)")
    problems: list[Problem] = []
    values: dict[tuple[str, str], Any] = {}
    where = SCOPE_LABEL[scope]
    for table, body in doc.items():
        if table not in TABLES or not isinstance(body, dict):
            problems.append(Problem(RULE, f"{path}: {table}", "an unknown table" if table not in TABLES
                                    else "a layout key must sit in a table", f"use only the tables "
                                    f"{', '.join('[' + t + ']' for t in TABLES)}"))
            continue
        if not any(k.table == table and scope in k.flags for k in LAYOUT_KEYS):
            problems.append(Problem(RULE, f"{path}: [{table}]", f"the table [{table}] does not apply to {where}",
                                    "remove it (`gatedtrain setup layout --help` shows which tables a command reads)"))
            continue
        for name, value in body.items():
            key = next((k for k in LAYOUT_KEYS if (k.table, k.name) == (table, name)), None)
            if key is None:
                problems.append(Problem(RULE, f"{path}: [{table}] {name}", "an unknown key",
                                        "remove it or check its spelling"))
            elif scope not in key.flags:
                problems.append(Problem(RULE, f"{path}: [{table}] {name}", f"does not apply to {where}",
                                        "remove it"))
            elif not key.supported:
                problems.append(Problem(RULE, f"{path}: [{table}] {name}", UNSUPPORTED,
                                        "remove it, or pass the matching option when this version has one"))
            elif not _type_ok(key, value):
                problems.append(Problem(RULE, f"{path}: [{table}] {name}", f"must be {_KIND_WORD[key.kind]}",
                                        "fix the value's type"))
            else:
                try:
                    for item in value if key.kind == "list" else [value]:
                        key.check(item, scope)
                except Refused as exc:
                    problems += _problems(path, key, exc)
                else:
                    values[(table, name)] = value
    if problems:
        raise Refused(problems)
    return values


def load(path: str, scope: str) -> Layout:
    """Read and validate the layout file at `path` (as the owner gave it) for `scope`."""
    data = _read(path)
    return Layout(path, hashlib.sha256(data).hexdigest(), parse(data, path, scope))


# ---- the effective values

def _computed_defaults(env: Mapping[str, str] | None, uid: int | None, home: str | None) -> dict[str, str]:
    paths = setup_user.default_paths(os.getuid() if uid is None else uid, os.environ if env is None else env, home)
    return {"state": paths["state"], "socket": paths["socket"]}


def effective_layout(scope: str, *, flags: Mapping[str, Any] | None = None, layout: Layout | None = None,
                     env: Mapping[str, str] | None = None, uid: int | None = None, home: str | None = None,
                     compute_defaults: bool = True) -> dict[str, tuple[Any, str]]:
    """{"table.key": (value, source)} for every supported key of `scope`; source is `flag` (`flags` maps a flag's
    dest, e.g. `keeper_home`, to the value given), `layout` or `default`. A same-user default is derived from the XDG
    directories (`compute_defaults`; the commands compute it themselves, so they pass False and get None)."""
    flags = flags or {}
    computed = (_computed_defaults(env, uid, home)
                if compute_defaults and any(k.computed for k in keys_for(scope)) else {})
    out: dict[str, tuple[Any, str]] = {}
    for key in keys_for(scope):
        given = flags.get(_dest(key.flags[scope])) if key.flags[scope] else None
        if given is not None:
            out[key.dotted] = (given, "flag")
        elif layout is not None and (key.table, key.name) in layout.values:
            out[key.dotted] = (layout.values[(key.table, key.name)], "layout")
        else:
            value = key.default() if key.default else computed.get(key.computed) if key.computed else None
            out[key.dotted] = (value, "default")
    return out


def _shown(value: Any) -> str:
    text = json.dumps(value) if isinstance(value, list) else str(value)
    return text if text.isprintable() else json.dumps(text)


@dataclass(frozen=True)
class Resolved:
    values: dict[str, Any]          # key name -> value (defaults filled in, except a same-user default: None)
    header: list[str]               # comment lines that start the printed commands; empty without --layout
    info: dict | None               # JSON `data.layout`; None without --layout


def resolve(scope: str, args, layout_path: str | None) -> Resolved:
    """The thin entry the commands call: the values `setup host` / `setup keeper` take (flag > layout > default), the
    header that records the layout and where each non-default value came from, and the JSON record. Refused (AN-34)
    for an unusable layout file."""
    layout = load(layout_path, scope) if layout_path is not None else None
    flags = {_dest(k.flags[scope]): getattr(args, _dest(k.flags[scope]), None)
             for k in keys_for(scope) if k.flags[scope]}
    eff = effective_layout(scope, flags=flags, layout=layout, compute_defaults=False)
    values = {key.name: eff[key.dotted][0] for key in keys_for(scope)}
    if layout is None:
        return Resolved(values, [], None)
    header = [f"# layout: {_shown(layout.path)} sha256:{layout.sha256}"]
    chosen = [f"#   {dotted} = {_shown(v)} ({source})" for dotted, (v, source) in eff.items() if source != "default"]
    header += ["# source of each value" + ("" if chosen else ": every key is its default")] + chosen
    info = {"path": layout.path, "sha256": layout.sha256, "keys": {d: s for d, (_, s) in eff.items()}}
    return Resolved(values, header, info)


# ---- `gatedtrain setup layout`

_INTRO = ("# gatedtrain setup layout file (AN-34), for {where}.\n"
          "# Pass it as `--layout FILE`; a flag overrides a key here, a key here overrides the built-in default.\n"
          "# The file is read as input and printed into the setup output; the keeper never reads it.\n"
          "# A key left out takes its default; a commented key has no default.")


def _toml(value: Any) -> str:
    return json.dumps(value) if isinstance(value, (str, list)) else repr(value)


def render(scope: str, *, env: Mapping[str, str] | None = None, uid: int | None = None,
           home: str | None = None) -> str:
    """A complete, commented, valid layout for `scope` showing the effective defaults (nothing is written)."""
    eff = effective_layout(scope, env=env, uid=uid, home=home)
    lines = [_INTRO.format(where=SCOPE_LABEL[scope]), "", f"schema_version = {SCHEMA_VERSION}"]
    table = None
    for key in LAYOUT_KEYS:
        if scope not in key.flags:
            continue
        if key.table != table:
            table = key.table
            lines += ["", f"[{table}]"]
        flag = key.flags[scope]
        lines.append(f"# {key.doc}" + (f" (replaces {flag})" if flag else ""))
        if not key.supported:
            lines += [f"# reserved, {UNSUPPORTED}", f"# {key.name} = {key.placeholder}"]
            continue
        value = eff[key.dotted][0]
        lines.append(f"{key.name} = {_toml(value)}" if value is not None else f"# {key.name} = {key.placeholder}")
    return "\n".join(lines) + "\n"
