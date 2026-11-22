"""AN-20: `gatedtrain setup host` prints the root bash script that sets up rootless Docker; it never runs it.

`host_script` returns one script the user saves as `SCRIPT_NAME`, reads, runs once with `sudo bash` and deletes. Every
check comes before the first change (the script first moves to `/`, so no program root runs reads the caller's
directory): the script runs as root, the owner has the uid it was printed for, an NVIDIA CDI spec (`nvidia.com/gpu`)
exists in `CDI_SPEC_DIRS`, `slirp4netns` or `pasta` is installed, the eval interpreter tree (with
`--eval-requirements`) is root-owned and not group- or other-writable, no subordinate id range of the owner or an
existing keeper user covers the other's uid or gid (checked again once the keeper user exists), and the parents of the directories it creates
are root-owned and not group- or other-writable. Then, in order: `apt-get update` and `uidmap`; `docker-ce-rootless-extras` at the
installed `docker-ce` version unless it is there already (unheld and held again when it was held; the rootful daemon
is not restarted); the AppArmor profile Ubuntu 24.04 requires for rootlesskit (only when no profile names
`/usr/bin/rootlesskit`); the keeper user when missing (as `setup keeper` creates it) and its home; the data roots and
the hand-off directory; a subuid/subgid range for each user that has none; the owner leaves the rootful `docker`
group, before any user manager or rootless daemon of the owner starts, since `runuser` and the user manager take the
owner's groups when they start (a user manager of the owner that already runs keeps the group: the script warns
and the closing comments say to restart it once the owner's sessions end); lingering; for each user a rootless daemon with its own data
root and Docker's native CDI; the owner's context `gatedtrain`; with `--eval-requirements` the keeper's root-owned
eval venv from the hash-pinned requirements (`python -I`, `pip --isolated --only-binary :all:`); with
`--inference-image` the keeper's daemon pulls the image and a drop-in of the keeper's unit passes its ID and the eval
trees to `keeper serve` (the serve command comes from `setup.serve_argv`, the builder of the unit's own `ExecStart=`),
ordered after the keeper's user manager; without it an earlier inference drop-in is removed.

Nothing here starts a process or writes a file; values are printed shell-quoted, and a path is refused unless it is
made of letters, digits, `/`, `.`, `_`, `-`, `+` and `@` (so it reads the same in a unit and in an unquoted heredoc).
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import PurePosixPath

from .. import __version__
from ..errors import Refused
from ..ext import launch
from .keeper import protocol, setup

RULE = "AN-20"
DEFAULT_HANDOFF = "/srv/gatedtrain/out"
DEFAULT_OWNER_DATA_ROOT = "/srv/gatedtrain/docker"
DEFAULT_KEEPER_HOME = "/var/lib/gatedtrain-keeper-home"
DEFAULT_KEEPER_DATA_ROOT = "/var/lib/gatedtrain-keeper-docker"
DEFAULT_EVAL_VENV = "/opt/gatedtrain-keeper/eval-venv"
DEFAULT_EVAL_PYTHON_ROOT = "/opt/gatedtrain-keeper/python"
CDI_SPEC_DIRS = ("/etc/cdi", "/var/run/cdi")
SUBID_COUNT = 65536
SCRIPT_NAME = "gatedtrain-host-setup.sh"
COMMAND = f"sudo bash {SCRIPT_NAME}"
REQUIREMENTS_EOF = "GATEDTRAIN_EVAL_REQUIREMENTS"
DROPIN_DIR = f"{setup.UNIT_PATH}.d"
DROPIN_PATH = f"{DROPIN_DIR}/inference.conf"

_SAFE_PATH = re.compile(r"/[A-Za-z0-9._/+@-]*")
_HASH = re.compile(r"--hash=sha256:[0-9a-f]{64}(\s|$)")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _refuse(field: str, message: str, fix: str) -> Refused:
    return setup._refuse(field, message, fix, RULE)


def _path(field: str, value: str) -> PurePosixPath:
    path = setup._absolute(field, value, RULE)
    if not _SAFE_PATH.fullmatch(value) or path == PurePosixPath("/"):
        raise _refuse(field, "the path is `/` or holds a character other than letters, digits, `/`, `.`, `_`, `-`, "
                      "`+` and `@`", "pass a path of letters, digits, `/`, `.`, `_` and `-`")
    return path


def _requirements(data: bytes) -> str:
    """The requirements text, refused unless every requirement carries `--hash=sha256:` (pip `--require-hashes`
    needs one for each; PyPI's hashes are the integrity anchor, design 5.8.1)."""
    fix = ("export the pack's lock with hashes: `uv export --format requirements-txt --no-emit-project` (every "
           "requirement pinned with --hash=sha256:)")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise _refuse("--eval-requirements", "the requirements file is not UTF-8 text", fix) from None
    if _CONTROL.search(text):
        raise _refuse("--eval-requirements", "the requirements file holds a control character", fix)
    if REQUIREMENTS_EOF in text.splitlines():
        raise _refuse("--eval-requirements", f"a line of the requirements file is {REQUIREMENTS_EOF}, the end marker "
                      "of the script's here-document", fix)
    logical, count = "", 0
    for line in (text + "\n").splitlines():
        if line.rstrip().endswith("\\"):
            logical += line.rstrip()[:-1] + " "
            continue
        logical, req = "", logical + line
        req = re.sub(r"(^|\s)#.*$", "", req).strip()
        if not req:
            continue
        if req.startswith("-"):
            raise _refuse("--eval-requirements", f"the requirements file holds an option line ({req.split()[0]})",
                          fix)
        if not _HASH.search(req):
            raise _refuse("--eval-requirements", f"the requirement {req.split()[0]!r} is not pinned with "
                          "--hash=sha256:", fix)
        count += 1
    if logical or not count:
        raise _refuse("--eval-requirements", "the requirements file is empty or ends in a line continuation", fix)
    return text if text.endswith("\n") else text + "\n"


def host_script(*, owner: str, owner_uid: int, keeper_user: str = setup.DEFAULT_USER,
                keeper_home: str = DEFAULT_KEEPER_HOME, keeper_data_root: str = DEFAULT_KEEPER_DATA_ROOT,
                owner_data_root: str = DEFAULT_OWNER_DATA_ROOT, handoff: str = DEFAULT_HANDOFF,
                state: str = setup.DEFAULT_STATE, socket: str = setup.DEFAULT_SOCKET, venv: str = setup.DEFAULT_VENV,
                inference_ref: str | None = None, eval_requirements: bytes | None = None,
                eval_python: str | None = None, eval_venv: str = DEFAULT_EVAL_VENV,
                confirm_group_removal: bool = False) -> dict:
    """`{"script": str, "command": str}` (AN-20). Refused (AN-20) on a user name, uid, path, image reference or
    requirements file that is not safe to print or would not hold the guarantee: an inference reference not pinned
    by digest, a requirement without `--hash=sha256:`, an eval interpreter outside `DEFAULT_EVAL_PYTHON_ROOT`."""
    for field, name in (("--owner", owner), ("--keeper-user", keeper_user)):
        if not (isinstance(name, str) and setup._USER.fullmatch(name)):
            example = "your login name" if field == "--owner" else setup.DEFAULT_USER
            raise _refuse(field, "the user name is not a plain user name ([a-z_][a-z0-9_-]*, at most 32 characters)",
                          f"pass a user name such as {example}")
    if keeper_user == owner:
        raise _refuse("--keeper-user", "the keeper user is the owner; the keeper must be a separate user",
                      f"pass a separate system user, such as {setup.DEFAULT_USER}")
    if isinstance(owner_uid, bool) or not isinstance(owner_uid, int) or owner_uid <= 0:
        raise _refuse("--owner-uid", "the owner uid is not a positive integer (rootless Docker is for a non-root "
                      "user)", "pass the owner's uid (`id -u`)")
    paths = {f: _path(f, v) for f, v in (("--keeper-home", keeper_home), ("--keeper-data-root", keeper_data_root),
                                         ("--owner-data-root", owner_data_root), ("--handoff", handoff),
                                         ("--state", state), ("--socket", socket), ("--venv", venv),
                                         ("--eval-venv", eval_venv))}
    if len(str(paths["--socket"]).encode()) > protocol.MAX_SOCKET_PATH:
        raise _refuse("--socket", f"the socket path is longer than {protocol.MAX_SOCKET_PATH} bytes",
                      "pass the socket `gatedtrain setup keeper` printed")
    created = ["--keeper-home", "--keeper-data-root", "--owner-data-root", "--handoff"]
    for i, a in enumerate(created):
        for b in created[i + 1:]:
            pa, pb = paths[a], paths[b]
            if pa == pb or pa in pb.parents or pb in pa.parents:
                raise _refuse(b, f"{b} and {a} are the same directory or one holds the other",
                              "give each of them its own directory")
    if inference_ref is not None and not (isinstance(inference_ref, str) and "@sha256:" in inference_ref
                                          and launch.IMAGE_RE.fullmatch(inference_ref)):
        raise _refuse("--inference-image", "the inference image is not a reference pinned by digest "
                      "(<name>@sha256:<64 hex>)", "pass the public CUDA base image as <name>@sha256:<digest>")
    requirements = None
    if eval_requirements is not None:
        requirements = _requirements(eval_requirements)
        if eval_python is None:
            raise _refuse("--eval-python", "--eval-requirements needs --eval-python, the interpreter the eval venv is "
                          "created from", f"install a standalone CPython under {DEFAULT_EVAL_PYTHON_ROOT} and pass "
                          "its bin/python3.x")
    elif eval_python is not None:
        raise _refuse("--eval-python", "--eval-python needs --eval-requirements", "pass the hash-pinned "
                      "requirements of the eval venv, or leave out --eval-python")
    if eval_python is not None:
        py = _path("--eval-python", eval_python)
        if PurePosixPath(DEFAULT_EVAL_PYTHON_ROOT) not in py.parents:
            raise _refuse("--eval-python", f"the eval interpreter is not under {DEFAULT_EVAL_PYTHON_ROOT}, the "
                          "root-owned tree the keeper mounts", f"install a standalone CPython under "
                          f"{DEFAULT_EVAL_PYTHON_ROOT} (`uv python install` with UV_PYTHON_INSTALL_DIR) and pass "
                          "its bin/python3.x")
    script = _script(owner=owner, owner_uid=owner_uid, keeper=keeper_user, paths=paths,
                     inference_ref=inference_ref, requirements=requirements, eval_python=eval_python,
                     confirm=bool(confirm_group_removal))
    return {"script": script, "command": COMMAND}


def _script(*, owner: str, owner_uid: int, keeper: str, paths: dict, inference_ref: str | None,
            requirements: str | None, eval_python: str | None, confirm: bool) -> str:
    q = shlex.quote
    p = {k: str(v) for k, v in paths.items()}
    khome, kroot, oroot, handoff, venv = (p["--keeper-home"], p["--keeper-data-root"], p["--owner-data-root"],
                                          p["--handoff"], p["--eval-venv"])
    owner_sock = f"unix:///run/user/{owner_uid}/docker.sock"
    keeper_sock = "unix:///run/user/$KUID/docker.sock"
    handoff_parent = str(PurePosixPath(handoff).parent)
    parents = [str(PurePosixPath(p[f]).parent) for f in ("--keeper-home", "--keeper-data-root", "--owner-data-root",
                                                          "--handoff")]
    if requirements is not None:
        parents += [str(PurePosixPath(venv).parent), str(PurePosixPath(DEFAULT_EVAL_PYTHON_ROOT).parent)]
    parents = [d for d in dict.fromkeys(parents) if d != "/"]
    cdi_globs = " ".join(f"{d}/*.{ext}" for d in CDI_SPEC_DIRS for ext in ("yaml", "json"))
    L = [
        "#!/bin/bash",
        f"# gatedtrain setup host (AN-20), printed by gatedtrain {__version__}; gatedtrain ran none of it.",
        f"# Rootless Docker for the owner {owner} (uid {owner_uid}) and the keeper user {keeper}; the owner leaves the",
        "# rootful `docker` group (the rootful daemons keep running for root-only use).",
        f"# Save this as {SCRIPT_NAME}, read it, run it once as root with:",
        f"#   {COMMAND}",
        "# and delete it afterwards. Every check comes before the first change; a failing step stops the script.",
        "set -euo pipefail",
        "# no program root runs here reads modules or configuration from the caller's directory; the keeper reads",
        "# the eval venv and the drop-in, and nothing created here is group- or other-writable",
        "cd /",
        "umask 022",
        "",
        f"OWNER={q(owner)}",
        f"OWNER_UID={owner_uid}",
        f"KEEPER={q(keeper)}",
        "KEEPER_STOPPED=",
        "OWNER_MANAGER_WAS_ACTIVE=",
        "fail() { echo \"gatedtrain host setup: $*\" >&2; exit 1; }",
        "# covers FILE USER UID ID: a subordinate range of USER (by name or uid) in FILE covers ID",
        "covers() {",
        "  awk -F: -v u=\"$2\" -v n=\"$3\" -v id=\"$4\" "
        "'($1 == u || $1 == n) && NF >= 3 && id + 0 >= $2 + 0 && id + 0 < $2 + $3 { f = 1 } END { exit !f }' "
        '"$1" 2>/dev/null',
        "}",
        "# a user's subordinate ids must not cover the other user's uid or gid: the user's rootless containers could",
        "# then act as the other user on the host",
        "check_subids() {",
        '  local kuid kgid ogid; kuid=$(id -u "$KEEPER"); kgid=$(id -g "$KEEPER"); ogid=$(id -g "$OWNER")',
        '  ! covers /etc/subuid "$OWNER" "$OWNER_UID" "$kuid" || fail "($1) a subuid range of $OWNER '
        'covers the keeper\'s uid $kuid; give $OWNER a range that does not (/etc/subuid)"',
        '  ! covers /etc/subuid "$KEEPER" "$kuid" "$OWNER_UID" || fail "($1) a subuid range of $KEEPER '
        'covers the owner\'s uid $OWNER_UID; give $KEEPER a range that does not (/etc/subuid)"',
        '  ! covers /etc/subgid "$OWNER" "$OWNER_UID" "$kgid" || fail "($1) a subgid range of $OWNER '
        'covers the keeper\'s gid $kgid; give $OWNER a range that does not (/etc/subgid)"',
        '  ! covers /etc/subgid "$KEEPER" "$kuid" "$ogid" || fail "($1) a subgid range of $KEEPER '
        'covers the owner\'s gid $ogid; give $KEEPER a range that does not (/etc/subgid)"',
        "}",
        "",
        "# ---- checks (nothing has changed yet)",
        '[ "$(id -u)" -eq 0 ] || fail "run this script as root"',
        '[ "$(id -u "$OWNER" 2>/dev/null)" = "$OWNER_UID" ] || fail "the user $OWNER does not exist or does not have '
        'uid $OWNER_UID"',
        "DOCKER_GID=$(getent group docker | cut -d: -f3 || true)",
        '[ -z "$DOCKER_GID" ] || [ "$(id -g "$OWNER")" != "$DOCKER_GID" ] || fail "docker is the primary group of '
        '$OWNER; give $OWNER another primary group first (usermod -g)"',
        "# GPU through Docker's native CDI: a spec naming nvidia.com/gpu must exist",
        f"grep -qsE '\"?kind\"?[[:space:]]*:[[:space:]]*\"?nvidia\\.com/gpu' {cdi_globs} \\",
        f"  || fail \"no NVIDIA CDI spec (kind nvidia.com/gpu) in {' or '.join(CDI_SPEC_DIRS)}; generate one as root: "
        "nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml\"",
        "DOCKER_CE_VERSION=$(dpkg-query -W -f='${Version}' docker-ce 2>/dev/null || true)",
        '[ -n "$DOCKER_CE_VERSION" ] || fail "docker-ce is not installed"',
        "# rootlesskit needs a user-mode network driver",
        "command -v slirp4netns >/dev/null || command -v pasta >/dev/null \\",
        '  || fail "neither slirp4netns nor pasta is installed; install one first: apt-get install slirp4netns"',
        "# overlapping subordinate id ranges of the owner and an existing keeper user",
        'if id -u "$KEEPER" >/dev/null 2>&1; then check_subids before-changes; fi',
    ]
    if requirements is not None:
        L += [
            "# The eval interpreter is a standalone CPython that root installs before this script runs, for example",
            f"#   env UV_PYTHON_INSTALL_DIR={DEFAULT_EVAL_PYTHON_ROOT} uv python install 3.12",
            "# (run as root). This script does not download or install it; it checks that the tree is root-owned",
            "# and not group- or other-writable, since the keeper mounts it into its inference container.",
            f"PYROOT={q(DEFAULT_EVAL_PYTHON_ROOT)}",
            f"EVAL_PYTHON={q(eval_python)}",
            '[ -d "$PYROOT" ] && [ ! -L "$PYROOT" ] || fail "$PYROOT is not a directory"',
            '[ -z "$(find "$PYROOT" \\( ! -user root -o \\( ! -type l -perm /022 \\) \\) -print -quit)" ] \\',
            '  || fail "$PYROOT holds a file not owned by root or writable by group or others"',
            '[ -x "$EVAL_PYTHON" ] || fail "$EVAL_PYTHON is not an executable"',
            'case "$(realpath -e "$EVAL_PYTHON")" in "$(realpath -e "$PYROOT")"/*) ;; '
            '*) fail "$EVAL_PYTHON resolves outside $PYROOT" ;; esac',
        ]
    L += [
        "# the parents of the directories created below: root-owned and not group- or other-writable",
        f"for d in {' '.join(q(d) for d in parents)}; do",
        '  [ ! -e "$d" ] || [ -z "$(find "$d" -maxdepth 0 \\( ! -user root -o -perm /022 \\) -print)" ] \\',
        '    || fail "$d is not owned by root or is writable by group or others"',
        "done",
        "",
        "# ---- changes",
        "apt-get update",
        "apt-get install -y uidmap",
        "# docker-ce-rootless-extras at the installed docker-ce version, kept on hold when it was held; the rootful",
        "# daemon is not restarted",
        "EXTRAS_VERSION=$(dpkg-query -W -f='${Version}' docker-ce-rootless-extras 2>/dev/null || true)",
        'if [ "$EXTRAS_VERSION" != "$DOCKER_CE_VERSION" ]; then',
        "  EXTRAS_HELD=$(apt-mark showhold docker-ce-rootless-extras)",
        '  [ -z "$EXTRAS_HELD" ] || apt-mark unhold docker-ce-rootless-extras',
        '  apt-get install -y --allow-downgrades --no-install-recommends '
        '"docker-ce-rootless-extras=$DOCKER_CE_VERSION" \\',
        '    || { [ -z "$EXTRAS_HELD" ] || apt-mark hold docker-ce-rootless-extras; '
        'fail "docker-ce-rootless-extras=$DOCKER_CE_VERSION did not install"; }',
        '  [ -z "$EXTRAS_HELD" ] || apt-mark hold docker-ce-rootless-extras',
        "fi",
        "# the AppArmor profile Ubuntu 24.04 requires for rootlesskit, only when no profile names it yet",
        "if [ -d /etc/apparmor.d ] && ! grep -qsE '^[[:space:]]*(profile[[:space:]]+[^[:space:]]+[[:space:]]+)?"
        "\"?/usr/bin/rootlesskit\"?[[:space:]]' /etc/apparmor.d/*; then",
        "  cat > /etc/apparmor.d/usr.bin.rootlesskit <<'APPARMOR'",
        "abi <abi/4.0>,",
        "include <tunables/global>",
        "",
        '"/usr/bin/rootlesskit" flags=(unconfined) {',
        "  userns,",
        "",
        "  include if exists <local/usr.bin.rootlesskit>",
        "}",
        "APPARMOR",
        "  apparmor_parser -r /etc/apparmor.d/usr.bin.rootlesskit",
        "fi",
        "# the keeper user, as `gatedtrain setup keeper` creates it, when missing",
        'if ! id -u "$KEEPER" >/dev/null 2>&1; then',
        f"  useradd --system --user-group --home-dir {q(p['--state'])} --no-create-home --shell /usr/sbin/nologin "
        '"$KEEPER"',
        "fi",
        'KUID=$(id -u "$KEEPER")',
        'KGID=$(id -g "$KEEPER")',
        "# a keeper user created just now: its uid against the owner's existing ranges",
        "check_subids after-keeper-created",
        'OGID=$(id -g "$OWNER")',
        'OWNER_HOME=$(getent passwd "$OWNER" | cut -d: -f6)',
        f'install -d -o "$KUID" -g "$KGID" -m 0700 {q(khome)}',
        f'if [ "$(getent passwd "$KEEPER" | cut -d: -f6)" != {q(khome)} ]; then',
        "  # usermod does not change the home of a user that runs a process: stop the keeper first (restarted below)",
        f"  if systemctl is-active --quiet {setup.UNIT_NAME}; then systemctl stop {setup.UNIT_NAME}; "
        "KEEPER_STOPPED=1; fi",
        f'  usermod -d {q(khome)} "$KEEPER"',
        "fi",
        f'install -d -o "$KUID" -g "$KGID" -m 0700 {q(kroot)}',
    ]
    for d in parents:
        if d == handoff_parent:
            L.append(f"install -d -o root -g root -m 0755 {q(d)}")
        else:
            L.append(f"[ -d {q(d)} ] || install -d -o root -g root -m 0755 {q(d)}")
    L += [
        f'install -d -o "$OWNER_UID" -g "$OGID" -m 0700 {q(oroot)}',
        f'install -d -o "$OWNER_UID" -g "$OGID" -m 0755 {q(handoff)}',
        f"# a subordinate uid/gid range of {SUBID_COUNT} for each user that has none, after the highest range in use",
        f"for entry in \"$OWNER:$OWNER_UID\" \"$KEEPER:$KUID\"; do",
        '  u=${entry%%:*}; n=${entry##*:}',
        "  start=$({ cat /etc/subuid 2>/dev/null || true; cat /etc/subgid 2>/dev/null || true; } \\",
        "    | awk -F: 'BEGIN { m = 100000 } NF >= 3 && $2 + $3 > m { m = $2 + $3 } END { print m }')",
        f"  end=$((start + {SUBID_COUNT} - 1))",
        '  grep -qE "^($u|$n):" /etc/subuid 2>/dev/null || usermod --add-subuids "$start-$end" "$u"',
        '  grep -qE "^($u|$n):" /etc/subgid 2>/dev/null || usermod --add-subgids "$start-$end" "$u"',
        "done",
        "# the owner leaves the rootful docker group before any user manager or rootless daemon of the owner starts:",
        "# runuser and the user manager take the owner's groups when they start",
    ]
    if confirm:
        L.append('read -r -p "Remove $OWNER from the docker group now? Enter goes on, Ctrl-C stops: " _')
    L += [
        "if getent group docker | cut -d: -f4 | tr ',' '\\n' | grep -x -- \"$OWNER\" >/dev/null; then",
        '  gpasswd -d "$OWNER" docker',
        "fi",
        "# a user manager of the owner that already runs (the owner ran sudo from a session) keeps the docker group,",
        "# and the owner's rootless daemon below starts under it",
        'if systemctl is-active --quiet "user@$OWNER_UID.service"; then',
        "  OWNER_MANAGER_WAS_ACTIVE=1",
        '  echo "gatedtrain host setup: warning: the user manager of $OWNER (user@$OWNER_UID.service) is already '
        'running and keeps the docker group, so the owner\'s rootless daemon starts with it; after all sessions of '
        '$OWNER have ended run: systemctl restart user@$OWNER_UID.service (or reboot), before expecting gatedtrain '
        'keeper fingerprint to show no rootful access" >&2',
        "fi",
        'loginctl enable-linger "$OWNER" "$KEEPER"',
        'systemctl start "user@$OWNER_UID.service" "user@$KUID.service"',
        'as_owner() { runuser -u "$OWNER" -- env HOME="$OWNER_HOME" XDG_RUNTIME_DIR=/run/user/$OWNER_UID '
        'DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$OWNER_UID/bus "$@"; }',
        f'as_keeper() {{ runuser -u "$KEEPER" -- env HOME={q(khome)} XDG_RUNTIME_DIR=/run/user/$KUID '
        'DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$KUID/bus "$@"; }',
    ]
    for who, home, root in (("owner", '"$OWNER_HOME"', oroot), ("keeper", q(khome), kroot)):
        cfg = f"{home}/.config/docker"
        cfg_text = ("$OWNER_HOME" if who == "owner" else khome) + "/.config/docker"      # inside double quotes
        daemon = json.dumps({"data-root": root, "features": {"cdi": True}, "cdi-spec-dirs": list(CDI_SPEC_DIRS)},
                            indent=2)
        L += [
            f"# rootless Docker for the {who}: data root {root}, GPUs through CDI",
            f"as_{who} mkdir -p {cfg}",
            f"if as_{who} test -e {cfg}/daemon.json && ! as_{who} test -e {cfg}/daemon.json.before-gatedtrain; then",
            f"  as_{who} cp {cfg}/daemon.json {cfg}/daemon.json.before-gatedtrain",
            f'  echo "gatedtrain host setup: the {who}\'s daemon.json is replaced; the old one is kept as '
            f'{cfg_text}/daemon.json.before-gatedtrain"',
            "fi",
            f"as_{who} tee {cfg}/daemon.json >/dev/null <<'JSON'",
            daemon,
            "JSON",
            f"as_{who} dockerd-rootless-setuptool.sh install",
            f"as_{who} systemctl --user restart docker",
        ]
        if who == "owner":
            L += [
                "# the owner's Docker context gatedtrain points at the owner's rootless daemon",
                "if as_owner docker context inspect gatedtrain >/dev/null 2>&1; then",
                f"  as_owner docker context update gatedtrain --docker host={owner_sock}",
                "else",
                f"  as_owner docker context create gatedtrain --docker host={owner_sock}",
                "fi",
            ]
    trees: list[str] = []
    if requirements is not None:
        trees = [DEFAULT_EVAL_PYTHON_ROOT, venv]
        L += [
            "# the keeper's eval venv: root-owned, from the interpreter checked above and the hash-pinned requirements",
            f'"$EVAL_PYTHON" -I -m venv --clear {q(venv)}',
            "REQ=$(mktemp)",
            f"cat > \"$REQ\" <<'{REQUIREMENTS_EOF}'",
            requirements + REQUIREMENTS_EOF,
            f'PIP_CONFIG_FILE=/dev/null {q(venv + "/bin/pip")} --isolated install --no-cache-dir --require-hashes '
            '--no-deps --only-binary :all: -r "$REQ"',
            'rm -f "$REQ"',
            f"chmod -R go-w {q(venv)}",
        ]
    if inference_ref is not None:
        serve = setup.serve_argv(PurePosixPath(p["--venv"]), PurePosixPath(p["--state"]),
                                 PurePosixPath(p["--socket"]), owner_uid, inference_image="${IMG}",
                                 eval_trees=trees, docker_host="unix:///run/user/${KUID}/docker.sock")
        L += [
            "# the keeper's daemon pulls the inference image; the keeper runs only its ID",
            f"as_keeper env DOCKER_HOST={keeper_sock} docker pull {q(inference_ref)}",
            f"IMG=$(as_keeper env DOCKER_HOST={keeper_sock} docker image inspect -f '{{{{.Id}}}}' {q(inference_ref)})",
            '[[ "$IMG" =~ ^sha256:[0-9a-f]{64}$ ]] || fail "the keeper\'s daemon gave no image ID for the image"',
            f"install -d -o root -g root -m 0755 {DROPIN_DIR}",
            f"cat > {DROPIN_PATH} <<DROPIN",
            "[Unit]",
            "After=user@${KUID}.service",
            "Wants=user@${KUID}.service",
            "",
            "[Service]",
            "ExecStart=",
            "ExecStart=" + " ".join(serve),
            "DROPIN",
        ]
    else:
        L += ["# no inference image: an inference drop-in of an earlier run goes",
              f"rm -f {DROPIN_PATH}"]
    L += [
        "systemctl daemon-reload",
        f"if [ -f {setup.UNIT_PATH} ]; then",
        f'  if [ -n "$KEEPER_STOPPED" ]; then systemctl start {setup.UNIT_NAME}; '
        f"else systemctl try-restart {setup.UNIT_NAME}; fi",
        "else",
        f'  echo "gatedtrain host setup: {setup.UNIT_NAME} is not installed; install it with gatedtrain setup keeper"',
        "fi",
        '[ -z "$OWNER_MANAGER_WAS_ACTIVE" ] || echo "gatedtrain host setup: warning: the user manager of $OWNER was '
        'already running; after all sessions of $OWNER have ended run: systemctl restart user@$OWNER_UID.service (or '
        'reboot), before expecting gatedtrain keeper fingerprint to show no rootful access" >&2',
        'echo "gatedtrain host setup: done; the next steps are the comments at the end of this script"',
        "",
        "# Next steps:",
        f"# 1. After every old session of {owner} has ended (they keep the docker group), as root:",
        f"#      sudo systemctl restart user@{owner_uid}.service   (or reboot)",
        f"#    This matters most when the script warned that the user manager of {owner} was already running:",
        "#    the owner's rootless daemon then started with the docker group, and `gatedtrain keeper fingerprint`",
        "#    reports rootful access until the restart.",
        f"# 2. As {owner}, no sudo: in ~/.config/gatedtrain/host.toml set",
        "#      rootless = true",
        f"#      output_root = \"{handoff}\"",
        f"# 3. As {owner}: pull or build the recipe images again into the rootless store, with",
        "#      docker --context gatedtrain pull <image>   (or docker --context gatedtrain build ...)",
        f"# 4. As {owner}: gatedtrain keeper fingerprint --socket {p['--socket']}",
        "#    and check it against the fingerprint you keep out of band.",
        "#    Note: a script printed again by `gatedtrain setup host` with --inference-image but",
        "#    without --eval-requirements passes no --eval-tree to the keeper, and one printed",
        "#    without --inference-image removes the inference drop-in.",
        f"# 5. Delete {SCRIPT_NAME}.",
    ]
    return "\n".join(L) + "\n"
