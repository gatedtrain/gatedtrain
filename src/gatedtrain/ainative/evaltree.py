"""AN-36: `gatedtrain setup eval-tree` prints the user-level script that builds the keeper's eval tree for a same-user
keeper; `setup eval-tree --check` verifies a built tree. Nothing here runs a build, starts a process or writes a file.

Why it exists. A batch generator (the packs' vLLM generators) runs in the keeper's inference container with the python of
an eval venv that the keeper mounts read-only (AN-25). The keeper's own rule for such a tree (AN-13,
`envcheck.eval_tree_problems`) is that every node and ancestor is owned by the keeper's uid or root and is not writable
by anyone else. With a same-user keeper the keeper's uid is the owner's, so a tree the owner builds under the owner's
home passes; before this command the only builder was `gatedtrain setup host`, a script for root that also creates the
keeper's user, `/opt` and `/srv` trees (AN-20), so a same-user keeper could not have a pack without sudo.

What it does not change. The keeper's rule stays the only authority: this command adds no flag, variable or key that
loosens it, and a separate-user keeper (another uid) refuses the tree this command builds (tested), so a user-owned tree
cannot weaken that mode. A same-user keeper does not seal its code or data from the owner's account (`keeper/texts.py`),
and the user-owned tree is part of that: the owner, or anything running as the owner, can change it after the keeper
checked and hashed it at start (the check is at start only; a change shows in the signed environment record only after
a restart). The printed script says so.

The script (all values shell-quoted, all of it under `root`): refuses root and another uid; installs a standalone
CPython with `uv python install --no-bin --no-registry` into `<root>/python` (no link in `~/.local/bin`, no registry
entry); creates `<root>/eval-venv` with `python -I -m venv` from the interpreter `uv python find` names (resolved, so
the venv links to the real directory, not a minor-version link uv may later move); installs the hash-pinned requirements
(`pip --isolated --require-hashes --no-deps --only-binary :all:`, the same recipe and validator as `setup host`);
removes group and other write bits from both trees (uv leaves its `.lock` file mode 0666, which the keeper refuses);
records the requirements' sha256 in `<root>/requirements.sha256` last; and ends with `setup eval-tree --check`. A rerun
with the same requirements changes nothing and checks again; other requirements, or a venv without the record (an
interrupted build), stop the script: it never clears a directory. The rollback is `rm -rf <root>`, printed as a comment.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from ..errors import Problem, Refused
from . import hostsetup
from .keeper import envcheck, setup, setup_user

RULE = "AN-36"
SCRIPT_NAME = "gatedtrain-eval-tree.sh"
COMMAND = f"sh {SCRIPT_NAME}"
# The default root is a sibling of the keeper's state directory under the XDG data directory, never inside it: the
# state directory is the keeper's private 0700 directory (AN-13), and a tree the container mounts does not belong there.
ROOT_DIR = f"{setup_user.APP_DIR}-eval"
PYTHON_DIR = "python"                   # `UV_PYTHON_INSTALL_DIR`: the standalone CPython the venv is made from
VENV_DIR = "eval-venv"
# Written last, outside both trees: the sha256 of the requirements the venv was built from, so a rerun can tell "built
# from these" from "built from others" and "interrupted" without touching anything.
REQUIREMENTS_MARK = "requirements.sha256"
# `--check` reports at most this many individual problems (a wrong-owner tree has one per file: tens of thousands),
# then the count of the rest.
MAX_CHECK_PROBLEMS = 20
# What `--check` applies to, said in its output (a claim states the mode that produced it): it runs the keeper's rule as
# the current user, so it answers for a same-user keeper only; a separate-user keeper (another uid) refuses this tree.
CHECK_SCOPE = "a same-user keeper (the keeper's rule as your uid); a separate-user keeper refuses a tree you own"
_VERSION = re.compile(r"[0-9]+\.[0-9]+(\.[0-9]+)?")


def _refuse(field: str, message: str, fix: str) -> Refused:
    return Refused([Problem(RULE, field, message, fix)])


def default_root(env: Mapping[str, str], home: str | None, owner_uid: int) -> str:
    """`<XDG data>/gatedtrain-keeper-eval`, from the session's XDG variables (the keeper's state directory's parent,
    `setup_user.default_paths`); nothing here is a fact about one machine."""
    try:
        paths = setup_user.default_paths(owner_uid, env, home)
    except Refused as exc:                  # the same refusal, reported under this command's rule
        raise Refused([Problem(RULE, p.field, p.message, p.fix) for p in exc.problems]) from None
    return str(PurePosixPath(paths["state"]).parent / ROOT_DIR)


def check_root(value: object, field: str = "--root") -> PurePosixPath:
    """Refused (AN-36) unless `value` is an absolute path the script, a unit's `ExecStart=` and a docker mount carry
    unchanged: letters, digits, `/`, `.`, `_`, `-`, `+` and `@`, no `..`, not `/`."""
    if not isinstance(value, str) or not value:
        raise _refuse(field, "the eval root is empty or not a string", "pass an absolute directory, or leave it out "
                      "for the default under your XDG data directory")
    path = setup._absolute(field, value, RULE)
    if not hostsetup._SAFE_PATH.fullmatch(value) or path == PurePosixPath("/"):
        raise _refuse(field, "the eval root is `/` or holds a character other than letters, digits, `/`, `.`, `_`, "
                      "`-`, `+` and `@`", "pass a path of letters, digits, `/`, `.`, `_` and `-`")
    if value.startswith("//"):
        raise _refuse(field, "the eval root starts with `//`", "pass a path with a single leading `/`")
    setup.check_eval_tree(value, RULE)
    return path


def trees(root: PurePosixPath) -> list[str]:
    """The eval trees of `root`, in the order `setup host` gives them (interpreter tree, then the venv)."""
    return [str(root / PYTHON_DIR), str(root / VENV_DIR)]


def eval_python(root: PurePosixPath) -> str:
    return str(root / VENV_DIR / "bin" / "python")


def keeper_args(root: PurePosixPath) -> list[str]:
    """The `setup keeper --mode same-user` options that point the keeper at the tree (it also needs
    `--inference-image`, which this command does not know)."""
    out: list[str] = []
    for tree in trees(root):
        out += ["--eval-tree", tree]
    return out + ["--eval-python", eval_python(root)]


def _version(value: object) -> str:
    if not isinstance(value, str) or not _VERSION.fullmatch(value):
        raise _refuse("--python-version", "the CPython version is not `MAJOR.MINOR` or `MAJOR.MINOR.PATCH`",
                      "pass the version the pack's eval lock names, such as 3.12 (see the pack's README)")
    return value


def eval_tree_script(*, requirements: bytes, python_version: str, root: str | None, env: Mapping[str, str],
                     home: str | None, owner_uid: int, executable: str) -> dict:
    """`{script, command, root, trees, eval_python, keeper_args, requirements_sha256, mode}` (AN-36). Refused (AN-36) on
    a root, version, requirements file, uid or executable that is not safe to print. `env`, `home` and `executable` are
    the session's facts, passed in so the output is testable on any host and the same for the same inputs."""
    if isinstance(owner_uid, bool) or not isinstance(owner_uid, int) or owner_uid <= 0:
        raise _refuse("--owner-uid", "the uid is not a positive integer; a root-owned tree is what `gatedtrain setup "
                      "host` builds", "run it as your own user")
    version = _version(python_version)
    text = hostsetup.requirements_text(requirements, "--requirements", RULE)
    exe = setup_user._executable_path(executable) if isinstance(executable, str) else None
    if exe is None:
        raise _refuse("--executable", "the gatedtrain command is not a path", "pass an absolute path")
    root_p = check_root(root if root is not None else default_root(env, home, owner_uid))
    digest = hashlib.sha256(requirements).hexdigest()
    q = shlex.quote
    ev, py = str(root_p / VENV_DIR), str(root_p / PYTHON_DIR)
    mark = str(root_p / REQUIREMENTS_MARK)
    L = [
        "#!/bin/sh",
        "# gatedtrain setup eval-tree (AN-36): printed, never run by gatedtrain. Run it as YOUR user, not with sudo,",
        f"# after reading it: sh {SCRIPT_NAME}",
        "# Builds the keeper's eval tree (the CPython and the venv a pack's batch generator runs with).",
        "# For a same-user keeper only: a separate-user keeper refuses this tree (use `gatedtrain setup host`, one sudo).",
        "# The tree is yours. Like the rest of a same-user keeper it does not seal anything from your account, and",
        "# anything running as you can change it after the keeper checked it at start (a restart re-checks it).",
        f"# What it changes: files under {root_p} only, plus uv's download cache; nothing outside it, no link in",
        "# ~/.local/bin, no uv registry entry. Rerunning with the same requirements changes nothing; it never clears a",
        f"# directory. Needs GNU coreutils (realpath -e, stat -c). To undo it: rm -rf {q(str(root_p))}   (not run by this script)",
        "set -eu",
        "umask 022",
        'fail() { echo "stopped: $1" >&2; exit 1; }',
        '[ "$(id -u)" -ne 0 ] || fail "run this as your own user, not as root (gatedtrain setup host builds the '
        'root-owned tree of a separate-user keeper)"',
        f'[ "$(id -u)" -eq {owner_uid} ] || fail "this script was printed for uid {owner_uid}; print it again as '
        'the user who runs it"',
        'UV=$(command -v uv) || fail "uv is not installed (see docs.astral.sh/uv); it installs the CPython"',
        f"ROOT={q(str(root_p))}",
        f"MARK={q(mark)}",
        f"WANT={q(f'{digest} python={version}')}",
        'mkdir -p "$(dirname "$ROOT")"',
        '[ ! -L "$ROOT" ] || fail "$ROOT is a symbolic link; pass the real directory"',
        "CREATED=no",
        'if [ ! -e "$ROOT" ]; then mkdir -m 0755 "$ROOT"; CREATED=yes; fi',
        '[ -d "$ROOT" ] || fail "$ROOT is not a directory"',
        '[ "$(stat -c %u "$ROOT")" = "$(id -u)" ] || fail "$ROOT is not owned by you"',
        f'[ -e "$MARK" ] || [ ! -e {q(ev)} ] || fail "{ev} exists without {REQUIREMENTS_MARK} (an interrupted build, or not made by '
        f'this command); remove that directory yourself (rm -rf {ev}) and run this again"',
        "# a directory that already holds something else is never adopted: the script changes only what it made",
        '[ -e "$MARK" ] || [ "$CREATED" = yes ] || [ -z "$(ls -A "$ROOT")" ] || fail "$ROOT already holds files that '
        'this command did not make; pass an empty or new --root"',
        'if [ -e "$MARK" ]; then',
        '  [ "$(cat "$MARK")" = "$WANT" ] || fail "an eval tree built from other requirements or another CPython '
        'version is in $ROOT; remove that directory yourself and run this again, or choose another --root"',
        '  echo "already built from these requirements and this CPython version: not rebuilding, checking only"',
        "else",
        f"  UV_PYTHON_INSTALL_DIR={q(py)}; export UV_PYTHON_INSTALL_DIR",
        f'  "$UV" python install --no-config --managed-python --no-bin --no-registry {q(version)}',
        f'  PY=$("$UV" python find --no-config --no-project --managed-python {q(version)})',
        '  PY=$(realpath -e "$PY")',
        f'  case "$PY" in "$(realpath -e {q(py)})"/*) ;; *) fail "the interpreter $PY is outside {py}" ;; esac',
        f'  "$PY" -I -m venv --clear {q(ev)}',
        "  REQ=$(mktemp)",
        "  trap 'rm -f \"$REQ\"' EXIT",
        f"  cat > \"$REQ\" <<'{hostsetup.REQUIREMENTS_EOF}'",
        text + hostsetup.REQUIREMENTS_EOF,
        f'  PIP_CONFIG_FILE=/dev/null {q(ev + "/bin/pip")} --isolated install --no-cache-dir --require-hashes '
        '--no-deps --only-binary :all: -r "$REQ"',
        '  rm -f "$REQ"; trap - EXIT',
        "  # uv leaves its .lock file mode 0666 and the keeper refuses a tree any other user could write",
        f"  chmod -R go-w {q(py)} {q(ev)}",
        '  [ "$CREATED" != yes ] || chmod go-w "$ROOT"',
        '  echo "$WANT" > "$MARK"',
        "fi",
        f"{q(str(exe))} setup eval-tree --check --root {q(str(root_p))}",
        "echo",
        "echo 'the eval tree is ready. Give the keeper these options (setup keeper --mode same-user, with its "
        "--inference-image):'",
        "echo " + q(" ".join(q(a) for a in keeper_args(root_p))),
    ]
    return {"script": "\n".join(L) + "\n", "command": COMMAND, "root": str(root_p), "trees": trees(root_p),
            "eval_python": eval_python(root_p), "keeper_args": keeper_args(root_p), "requirements_sha256": digest,
            "mode": "same_user"}


def check(root: str) -> dict:
    """`{root, trees, problems: 0, checked_as_uid, applies_to}` when the trees under `root` pass the keeper's rule for the current user as the
    keeper (AN-13); Refused (AN-36) otherwise, naming at most `MAX_CHECK_PROBLEMS` problems and the count of the rest.
    Read-only."""
    root_p = check_root(root)
    paths = [Path(t) for t in trees(root_p)]
    found = envcheck.eval_tree_problems(paths, os.geteuid())
    if not found:
        return {"root": str(root_p), "trees": [str(p) for p in paths], "problems": 0, "checked_as_uid": os.geteuid(),
                "applies_to": CHECK_SCOPE}
    fix_missing = (f"build it with `gatedtrain setup eval-tree --requirements FILE --python-version X --root "
                   f"{root_p}` (same-user keeper)")
    inside = (str(root_p / PYTHON_DIR), str(root_p / VENV_DIR))

    def fix(p: Problem) -> str:
        if "does not exist" in p.message:
            return fix_missing
        if "writable" in p.message and any(t in p.message for t in inside):      # a write bit inside the trees
            return f"{p.fix}; for a tree this command built: chmod -R go-w {inside[0]} {inside[1]}"
        return p.fix

    listed = [Problem(RULE, "eval tree", p.message, fix(p)) for p in found[:MAX_CHECK_PROBLEMS]]
    if len(found) > MAX_CHECK_PROBLEMS:
        listed.append(Problem(RULE, "eval tree", f"{len(found) - MAX_CHECK_PROBLEMS} more problems of {len(found)} "
                              "are not listed", "fix the ones above and check again; a tree owned by another user "
                              "has one problem per file, so rebuild it as yourself"))
    raise Refused(listed)
