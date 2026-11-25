"""AN-32, AN-33: `gatedtrain setup check` names the outside prerequisites of the sudo-free setup; it changes nothing.

Rootless Docker, the NVIDIA container toolkit and the OS settings around them are not gatedtrain's to install. The
check reads the host through a `HostProbe` (files, environment, PATH, a few read-only commands, one rlimit), so the
tests use a fake host and the check can never write. Each check returns `{id, status, required, owner, summary, fix}`:
`status` is ok, missing, unknown (could not tell) or not_applicable; `owner` says who can fix it, `you` or the
`administrator` (an administrator item is a Docker/OS/NVIDIA prerequisite, never a gatedtrain step). A check that
raises is reported `unknown`, so the check never stops midway. Exit 0 when every required check is ok, 1 when a
required check is missing or unknown (AN-33).
"""

from __future__ import annotations

import json
import os
import platform
import re
import pwd
import resource
import shutil
import stat
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from ..errors import printable
from .keeper.setup_user import RUNTIME_FALLBACK

# ---- The one table of locations and thresholds (each with its reason in CONSTANT_REASONS, each tested) ----------
SUBID_MIN_COUNT = 65536
SUBID_FILES = {"subuid": "/etc/subuid", "subgid": "/etc/subgid"}
CDI_SYSTEM_DIRS = ("/etc/cdi", "/var/run/cdi")
CDI_USER_SUBDIR = "cdi"
CDI_GPU_KIND = "nvidia.com/gpu"
UID_HELPERS = ("newuidmap", "newgidmap")
UID_HELPER_CAPS = {"newuidmap": "cap_setuid", "newgidmap": "cap_setgid"}   # the file capability that replaces setuid
ROOTLESS_OPTION = "name=rootless"                  # the exact entry of `docker info`'s SecurityOptions
SECURITY_OPTIONS_FORMAT = "{{json .SecurityOptions}}"
NET_HELPERS = ("slirp4netns", "pasta")
LINGER_DIR = "/var/lib/systemd/linger"
USERNS_MAX_FILE = "/proc/sys/user/max_user_namespaces"
USERNS_APPARMOR_FILE = "/proc/sys/kernel/apparmor_restrict_unprivileged_userns"
PTRACE_SCOPE_FILE = "/proc/sys/kernel/yama/ptrace_scope"
NVIDIA_PROC_DIR = "/proc/driver/nvidia"
DOCKER_SOCKET_NAME = "docker.sock"
RUNNABLE = ("docker", "systemctl", "getcap")                  # the only commands the check runs, both read-only queries
RUN_TIMEOUT = 10                                    # seconds; a daemon that does not answer is `unknown`, not a hang
UNLIMITED = -1                                      # RLIM_INFINITY as the probe reports it
READ_LIMIT_BYTES = 1 << 20                          # the most `read_text` returns: 1 MiB, far above any file read here
DISPLAY_MAX_CHARS = 64                              # the longest environment value (a context name) printed back
SCHEME_MAX_LEN = 16                                 # the longest URL scheme printed back (one letter plus 15)
NSSWITCH_FILE = "/etc/nsswitch.conf"
SUBID_NSS_KEY = "subid"                             # the nsswitch.conf database that names the subordinate-id sources
SUBID_NSS_LOCAL = "files"                           # the source that is /etc/subuid and /etc/subgid themselves
DOCKER_CONFIG_ENV = "DOCKER_CONFIG"
DOCKER_CONFIG_SUBDIR = ".docker"
DOCKER_CONFIG_FILE = "config.json"
DEFAULT_CONTEXT = "default"                         # the docker context that means the local daemon

CONSTANT_REASONS = {
    "SUBID_MIN_COUNT": "rootless Docker maps one full 16-bit id range into the user namespace; fewer ids break images",
    "SUBID_FILES": "the files newuidmap/newgidmap read to learn which subordinate ids a user may map (shadow-utils)",
    "CDI_SYSTEM_DIRS": "the directories the Docker daemon reads CDI specs from by default (the CDI spec's static and "
                       "dynamic directories); a spec elsewhere is not seen unless the daemon is configured for it",
    "CDI_GPU_KIND": "the CDI device kind the NVIDIA container toolkit generates for GPUs",
    "UID_HELPERS": "the helpers rootless Docker uses to set up the user namespace id maps (setuid root or file capability)",
    "UID_HELPER_CAPS": "distributions that ship the helpers without setuid grant these file capabilities instead",
    "ROOTLESS_OPTION": "the SecurityOptions entry Docker reports for a rootless daemon, matched exactly (not a substring)",
    "NET_HELPERS": "either user-space network stack lets a rootless daemon give containers networking",
    "LINGER_DIR": "systemd records lingering users here; lingering keeps a user's services after logout",
    "USERNS_MAX_FILE": "0 means unprivileged user namespaces are disabled, which rootless containers need",
    "USERNS_APPARMOR_FILE": "Ubuntu 24.04 restricts unprivileged user namespaces unless a profile allows rootlesskit",
    "PTRACE_SCOPE_FILE": "Yama ptrace_scope 0 lets any process of a user read the memory of its siblings",
    "READ_LIMIT_BYTES": "every file read here is a short config or spec; a larger one is cut off rather than slurped",
    "DISPLAY_MAX_CHARS": "an environment value is echoed back only when short and printable; a long one is hidden",
    "SCHEME_MAX_LEN": "real URL schemes (tcp, ssh, npipe, unix) are short; a longer prefix is not one and is not echoed",
    "NSSWITCH_FILE": "where glibc is told which sources answer a database; `subid` may name SSSD or LDAP",
    "SUBID_NSS_KEY": "the nsswitch.conf database shadow-utils 4.9+ uses for subordinate ids",
    "SUBID_NSS_LOCAL": "the nsswitch source that is the local /etc/subuid and /etc/subgid files",
    "DOCKER_CONFIG_FILE": "the Docker CLI's own config (under $DOCKER_CONFIG or ~/.docker); its `currentContext` "
                          "redirects every docker command without any environment variable",
    "DEFAULT_CONTEXT": "the context name the Docker CLI uses for the local daemon; any other name is a redirect",
}

ADMIN_NOTE = (" This is an administrator prerequisite, not a gatedtrain step; gatedtrain itself needs no sudo for "
              "the default setup.")
CLOSING_LINE = ('gatedtrain itself needs no sudo for the same-user setup; items marked "administrator" are '
                'Docker/OS/NVIDIA prerequisites that need root on a machine that lacks them.')


class HostProbe:
    """Every way the check touches the host. Reads only; the real one is `SystemProbe`, tests pass a fake."""

    def system(self) -> str: raise NotImplementedError
    def uid(self) -> int: raise NotImplementedError
    def username(self) -> str | None: raise NotImplementedError
    def env(self, name: str) -> str | None: raise NotImplementedError
    def read_text(self, path: str) -> str | None: raise NotImplementedError
    def listdir(self, path: str) -> list[str] | None: raise NotImplementedError
    def stat(self, path: str) -> tuple[int, int] | None: raise NotImplementedError      # (st_uid, st_mode)
    def which(self, name: str) -> str | None: raise NotImplementedError
    def memlock_hard(self) -> int | None: raise NotImplementedError
    def run(self, argv: list[str], timeout: int, env: dict[str, str] | None = None) -> tuple[int, str] | None:
        raise NotImplementedError

    def cdi_dirs(self) -> list[str]:
        """The directories the Docker daemon reads CDI specs from by default (the named table)."""
        return list(CDI_SYSTEM_DIRS)

    def cdi_user_dirs(self) -> list[str]:
        """The user's CDI directory (`$XDG_CONFIG_HOME/cdi`, else `$HOME/.config/cdi`): `nvidia-ctk` can write there
        without root, but the daemon does not read it unless it is configured to."""
        config = self.env("XDG_CONFIG_HOME")
        if not config or not os.path.isabs(config):
            home = self.env("HOME")
            config = f"{home}/.config" if home and os.path.isabs(home) else None
        return [f"{config}/{CDI_USER_SUBDIR}"] if config else []

    def docker_config_path(self) -> str | None:
        """The Docker CLI's config file: `$DOCKER_CONFIG/config.json`, else `$HOME/.docker/config.json`."""
        directory = self.env(DOCKER_CONFIG_ENV)
        if not directory or not os.path.isabs(directory):
            home = self.env("HOME")
            directory = f"{home}/{DOCKER_CONFIG_SUBDIR}" if home and os.path.isabs(home) else None
        return f"{directory}/{DOCKER_CONFIG_FILE}" if directory else None


class SystemProbe(HostProbe):
    def system(self): return platform.system().lower()
    def uid(self): return os.getuid()

    def username(self):
        try:
            return pwd.getpwuid(os.getuid()).pw_name
        except KeyError:
            return None

    def env(self, name): return os.environ.get(name)

    def read_text(self, path):
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                return fh.read(READ_LIMIT_BYTES)
        except OSError:
            return None

    def listdir(self, path):
        try:
            return sorted(os.listdir(path))
        except OSError:
            return None

    def stat(self, path):
        try:
            st = os.stat(path)
        except OSError:
            return None
        return st.st_uid, st.st_mode

    def which(self, name): return shutil.which(name)

    def memlock_hard(self):
        try:
            hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)[1]
        except (OSError, ValueError):
            return None
        return UNLIMITED if hard == resource.RLIM_INFINITY else hard

    def run(self, argv, timeout, env=None):
        if argv[0] not in RUNNABLE:
            return None
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False,
                                  env={**os.environ, **(env or {})})
        except (OSError, subprocess.SubprocessError):
            return None
        return done.returncode, done.stdout


@dataclass(frozen=True)
class CheckSpec:
    id: str
    owner: str               # who usually fixes it: "you" or "administrator"
    required: bool
    gpu_only: bool = False


def docker_socket_candidates(probe: HostProbe, uid: int) -> list[str]:
    """The rootless daemon's socket: `$DOCKER_HOST` when it is a unix socket, else `$XDG_RUNTIME_DIR/docker.sock`,
    else the logind runtime directory of the uid."""
    host = probe.env("DOCKER_HOST")
    if host and host.startswith("unix://"):
        return [host[len("unix://"):]]
    runtime = probe.env("XDG_RUNTIME_DIR")
    if not runtime or not os.path.isabs(runtime):
        runtime = RUNTIME_FALLBACK.format(uid=uid)
    return [f"{runtime}/{DOCKER_SOCKET_NAME}"]


def gpu_default(probe: HostProbe) -> bool:
    return probe.listdir(NVIDIA_PROC_DIR) is not None


def _res(spec: CheckSpec, status: str, summary: str, fix: str = "nothing to do", *, owner: str | None = None,
         required: bool | None = None) -> dict:
    owner = owner or spec.owner
    if status in ("missing", "unknown") and owner == "administrator":
        fix += ADMIN_NOTE
    return {"id": spec.id, "status": status, "required": spec.required if required is None else required,
            "owner": owner, "summary": summary, "fix": fix}


def _linux(p: HostProbe, spec: CheckSpec, gpu: bool) -> dict:
    system = p.system()
    if system == "linux":
        return _res(spec, "ok", "Linux")
    return _res(spec, "missing", f"this is {system}; the keeper and rootless Docker need Linux",
                "run gatedtrain on a Linux host")


def _scheme(host: str) -> str:
    """The scheme of a DOCKER_HOST value only (never the rest: it can hold a user name or a credential)."""
    head = host.split(":", 1)[0]
    return head if re.fullmatch(rf"[a-z][a-z0-9+.-]{{0,{SCHEME_MAX_LEN - 1}}}", head) else "an unrecognised scheme"


def _shown(value: str) -> str:
    """An environment-derived value for a message: escaped, and replaced when it is long."""
    return printable(value) if len(value) <= DISPLAY_MAX_CHARS else "(unprintable or long)"


def configured_context(p: HostProbe) -> str | None:
    """The `currentContext` of the Docker CLI's config (read through the probe), None when absent, `default` or
    unreadable."""
    path = p.docker_config_path()
    text = p.read_text(path) if path else None
    try:
        doc = json.loads(text) if text else None
    except (ValueError, RecursionError):        # malformed, or nested beyond the parser's depth: not a usable config
        return None
    name = doc.get("currentContext") if isinstance(doc, dict) else None
    return name if isinstance(name, str) and name and name != DEFAULT_CONTEXT else None


def _rootless_docker(p: HostProbe, spec: CheckSpec, gpu: bool) -> dict:
    if p.which("docker") is None:
        return _res(spec, "missing", "no `docker` command on PATH",
                    "install Docker Engine with its rootless extras (docker-ce-rootless-extras)",
                    owner="administrator")
    host = p.env("DOCKER_HOST")
    if host and not host.startswith("unix://"):
        return _res(spec, "missing", f"DOCKER_HOST points at a {_scheme(host)} daemon, not at a rootless socket of "
                    "yours; the docker command would not talk to your rootless daemon",
                    "unset DOCKER_HOST, or set it to unix://$XDG_RUNTIME_DIR/docker.sock", owner="you")
    uid = p.uid()
    socks = docker_socket_candidates(p, uid)
    owned = []
    for sock in socks:
        st = p.stat(sock)
        if st is not None and stat.S_ISSOCK(st[1]) and st[0] == uid:
            owned.append(sock)
    if not owned:
        return _res(spec, "missing", f"no rootless Docker socket of yours at {_shown(socks[0])}",
                    "start your rootless daemon: dockerd-rootless-setuptool.sh install (as yourself, no sudo)",
                    owner="you")
    got = p.run(["docker", "info", "--format", SECURITY_OPTIONS_FORMAT], RUN_TIMEOUT,
                env={"DOCKER_HOST": f"unix://{owned[0]}"})
    if got is None or got[0] != 0:
        return _res(spec, "unknown", f"the daemon at {_shown(owned[0])} did not answer `docker info`",
                    "start it (systemctl --user start docker) and rerun the check", owner="you")
    try:
        options = json.loads(got[1])
    except (ValueError, RecursionError):
        options = None
    if not (isinstance(options, list) and all(isinstance(o, str) for o in options)):
        return _res(spec, "unknown", f"the daemon at {_shown(owned[0])} did not report its security options as a JSON list",
                    "rerun `docker info --format '{{json .SecurityOptions}}'` and check the output", owner="you")
    if ROOTLESS_OPTION not in options:
        return _res(spec, "missing", f"the daemon at {_shown(owned[0])} is not rootless",
                    "install a rootless daemon: dockerd-rootless-setuptool.sh install", owner="you")
    context = p.env("DOCKER_CONTEXT")
    where = "DOCKER_CONTEXT"
    if not context and not host:
        context, where = configured_context(p), f"currentContext in the Docker config ({DOCKER_CONFIG_FILE})"
    if context and not host:
        return _res(spec, "ok", f"rootless daemon at {_shown(owned[0])}; note: {where} is set to {_shown(context)!r}",
                    "your docker commands use that context, not necessarily this socket; this check asked the "
                    "socket directly (unset DOCKER_CONTEXT or `docker context use default` to make them agree)")
    return _res(spec, "ok", f"rootless daemon at {_shown(owned[0])}")


def _helper_state(p: HostProbe, name: str, path: str) -> tuple[str, str]:
    """(`ok` | `missing` | `unknown`, what was seen) for one helper file: root-owned, then setuid or the file
    capability (`getcap` when the host has it)."""
    st = p.stat(path)
    if st is None:
        return "missing", f"{name} at {path} cannot be inspected"
    if st[0] != 0:
        return "missing", f"{name} at {path} is not owned by root"
    if st[1] & stat.S_ISUID:
        return "ok", f"{name} at {path} is setuid root"
    getcap = p.which("getcap")
    if getcap is None:
        return "unknown", f"{name} at {path} is not setuid and `getcap` is not installed to check file capabilities"
    got = p.run(["getcap", path], RUN_TIMEOUT)
    if got is None or got[0] != 0:
        return "unknown", f"{name} at {path} is not setuid and `getcap` did not answer"
    if UID_HELPER_CAPS[name] in got[1]:
        return "ok", f"{name} at {path} has the file capability {UID_HELPER_CAPS[name]}"
    return "missing", f"{name} at {path} is not setuid and has no file capability {UID_HELPER_CAPS[name]}"


def _uidmap(p: HostProbe, spec: CheckSpec, gpu: bool) -> dict:
    absent, seen, worst = [], [], "ok"
    for name in UID_HELPERS:
        path = p.which(name)                                     # the first PATH hit is the one that would run
        if path is None:
            absent.append(name)
            continue
        state, what = _helper_state(p, name, path)
        seen.append(what)
        if state == "missing" or (state == "unknown" and worst == "ok"):
            worst = state
    if absent:
        return _res(spec, "missing", f"not on PATH: {', '.join(absent)}", "install the uidmap package")
    if worst == "missing":
        return _res(spec, "missing", "; ".join(seen) + " (not setuid root or without the file capability)",
                    "reinstall the uidmap package (the helpers must be root-owned and setuid root, or carry the "
                    "file capability)")
    if worst == "unknown":
        return _res(spec, "unknown", "; ".join(seen), "install `getcap` (libcap2-bin) and rerun the check")
    return _res(spec, "ok", "; ".join(seen))


def _subid_count(text: str, names: set[str]) -> int:
    total = 0
    for line in text.splitlines():
        parts = line.strip().split(":")
        if len(parts) == 3 and parts[0] in names and parts[2].isdigit():
            total += int(parts[2])
    return total


def _subid_sources(p: HostProbe) -> list[str]:
    """The sources nsswitch.conf names for subordinate ids, `[]` when it names none (then the local files are it)."""
    for line in (p.read_text(NSSWITCH_FILE) or "").splitlines():
        key, _, rest = line.partition(":")
        if key.strip() == SUBID_NSS_KEY:
            return rest.split("#", 1)[0].split()
    return []


def _subuid(p: HostProbe, spec: CheckSpec, gpu: bool) -> dict:
    names = {str(p.uid())} | ({p.username()} if p.username() else set())
    short, unreadable = [], []
    for kind, path in SUBID_FILES.items():
        text = p.read_text(path)
        if text is None:
            (unreadable if p.stat(path) is not None else short).append(kind)    # present but unreadable vs absent
        elif _subid_count(text, names) < SUBID_MIN_COUNT:
            short.append(kind)
    fix = (f"add a range of at least {SUBID_MIN_COUNT} ids that is not used in /etc/subuid and /etc/subgid (see "
           "`man subuid`): usermod --add-subuids <FIRST>-<LAST> --add-subgids <FIRST>-<LAST> <you>")
    if unreadable:
        return _res(spec, "unknown", f"cannot read {', '.join(SUBID_FILES[k] for k in unreadable)}",
                    "check its permissions, or run `grep \"$USER\" /etc/subuid /etc/subgid` as yourself")
    others = [x for x in _subid_sources(p) if x != SUBID_NSS_LOCAL]
    if short and others:
        return _res(spec, "unknown", f"the local files have no range of at least {SUBID_MIN_COUNT} ids for you in: "
                    f"{', '.join(short)}, but {NSSWITCH_FILE} also names {', '.join(map(printable, others))} for "
                    f"{SUBID_NSS_KEY}", "ask your administrator whether the directory service grants you a range; "
                    f"if not: {fix}")
    if short:
        return _res(spec, "missing", f"no range of at least {SUBID_MIN_COUNT} ids for you in: {', '.join(short)}", fix)
    return _res(spec, "ok", f"subuid and subgid ranges of at least {SUBID_MIN_COUNT} ids")


def _userns(p: HostProbe, spec: CheckSpec, gpu: bool) -> dict:
    text = p.read_text(USERNS_MAX_FILE)
    try:
        count = int((text or "").strip())
    except ValueError:
        return _res(spec, "unknown", f"cannot read {USERNS_MAX_FILE}", "check that user namespaces are enabled",
                    owner="administrator")
    if count <= 0:
        return _res(spec, "missing", "unprivileged user namespaces are disabled",
                    "set user.max_user_namespaces to a positive number")
    return _res(spec, "ok", "user namespaces enabled")


def _userns_apparmor(p: HostProbe, spec: CheckSpec, gpu: bool) -> dict:
    """Informational. With the restriction on, whether rootlesskit may create a user namespace depends on an AppArmor
    profile this check does not evaluate, so the answer is `unknown`; it is not required, so a host that has the
    profile does not fail the check."""
    if (p.read_text(USERNS_APPARMOR_FILE) or "").strip() == "1":
        return _res(spec, "unknown", "AppArmor restricts unprivileged user namespaces (Ubuntu 24.04): rootlesskit "
                    "needs a profile that allows it, and this check cannot tell whether one is installed",
                    "if rootless Docker fails to start, install the rootlesskit AppArmor profile")
    return _res(spec, "ok", "no AppArmor restriction on unprivileged user namespaces")


def _rootless_net(p: HostProbe, spec: CheckSpec, gpu: bool) -> dict:
    for name in NET_HELPERS:
        if p.which(name) is not None:
            return _res(spec, "ok", f"{name} found")
    return _res(spec, "missing", f"none of {', '.join(NET_HELPERS)} on PATH",
                "install slirp4netns (or passt, which provides pasta)")


def _nvidia_driver(p: HostProbe, spec: CheckSpec, gpu: bool) -> dict:
    if p.which("nvidia-smi") is None:
        return _res(spec, "missing", "no `nvidia-smi` on PATH", "install the NVIDIA driver")
    return _res(spec, "ok", "nvidia-smi found")


def cdi_kind(text: str) -> str | None:
    """The top-level `kind` of a CDI spec (JSON, or the YAML the toolkit writes): a `kind:` line at column 0 only, so a
    comment, a nested key or a device name never counts."""
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            doc = json.loads(text)
        except (ValueError, RecursionError):
            return None
        kind = doc.get("kind") if isinstance(doc, dict) else None
        return kind if isinstance(kind, str) else None
    for line in text.splitlines():
        m = re.match(r"""kind:[ \t]*(?:"([^"\n]*)"|'([^'\n]*)'|([^\s#'"]+))[ \t]*(?:#.*)?$""", line)
        if m:
            return next(g for g in m.groups() if g is not None)
    return None


def _cdi_scan(p: HostProbe, directories: list[str]) -> tuple[str | None, list[str]]:
    """(the first directory holding a GPU spec, the directories or specs that could not be read). A directory that
    exists but cannot be listed, or a listed spec that cannot be read, is not evidence of absence."""
    unreadable = []
    for d in directories:
        names = p.listdir(d)
        if names is None:
            if p.stat(d) is not None:
                unreadable.append(d)
            continue
        for name in names:
            if not name.endswith((".json", ".yaml", ".yml")):
                continue
            text = p.read_text(f"{d}/{name}")
            if text is None:
                unreadable.append(f"{d}/{name}")
            elif cdi_kind(text) == CDI_GPU_KIND:
                return d, unreadable
    return None, unreadable


CDI_FEATURE_NOTE = ("informational: some Docker versions need the CDI feature enabled in the daemon configuration "
                    "(daemon.json `features`) before `--device nvidia.com/gpu=...` works; check yours")


def _nvidia_cdi(p: HostProbe, spec: CheckSpec, gpu: bool) -> dict:
    dirs = p.cdi_dirs()
    found, unreadable = _cdi_scan(p, dirs)
    if found is not None:
        return _res(spec, "ok", f"{CDI_GPU_KIND} spec in {_shown(found)}", CDI_FEATURE_NOTE)
    mine, _ = _cdi_scan(p, p.cdi_user_dirs())
    if mine is not None:
        return _res(spec, "unknown", f"a {CDI_GPU_KIND} spec is in {_shown(mine)}, but dockerd does not read this "
                    "directory unless configured",
                    f"generate the spec in one of {', '.join(dirs)} (administrator), or configure the daemon to read "
                    f"{_shown(mine)}. {CDI_FEATURE_NOTE}", owner="administrator")
    if unreadable:
        return _res(spec, "unknown", f"cannot read {', '.join(map(_shown, unreadable))}, so a {CDI_GPU_KIND} spec "
                    "there cannot be ruled out", "check the permissions, or run `nvidia-ctk cdi list` as yourself",
                    owner="administrator")
    return _res(spec, "missing", f"no CDI spec of kind {CDI_GPU_KIND} in {', '.join(dirs)}",
                f"install nvidia-container-toolkit and run: nvidia-ctk cdi generate --output=<a directory the daemon "
                f"reads>. {CDI_FEATURE_NOTE}", owner="administrator")


def _systemd_user(p: HostProbe, spec: CheckSpec, gpu: bool) -> dict:
    runtime = p.env("XDG_RUNTIME_DIR")
    if not runtime:
        return _res(spec, "missing", "XDG_RUNTIME_DIR is not set (no login session)",
                    "log in through a normal session; or run the keeper in a terminal (setup keeper prints how)")
    got = p.run(["systemctl", "--user", "is-system-running"], RUN_TIMEOUT)
    if got is None or not got[1].strip():
        return _res(spec, "unknown", "the systemd user manager did not answer",
                    "run the keeper in a terminal (setup keeper prints how)")
    state = got[1].strip()
    if state in ("running", "degraded", "starting"):
        return _res(spec, "ok", f"user manager {state}")
    return _res(spec, "missing", f"user manager state: {state}", "run the keeper in a terminal")


def _linger(p: HostProbe, spec: CheckSpec, gpu: bool) -> dict:
    name = p.username()
    if name is None:
        return _res(spec, "unknown", "your user name is unknown", "loginctl enable-linger")
    if p.read_text(f"{LINGER_DIR}/{name}") is not None:
        return _res(spec, "ok", "lingering is on: the keeper survives logout")
    return _res(spec, "missing", "lingering is off: the keeper stops when you log out",
                "run `loginctl enable-linger` (it may ask an administrator), or stay logged in")


def _memlock(p: HostProbe, spec: CheckSpec, gpu: bool) -> dict:
    hard = p.memlock_hard()
    if hard is None:
        return _res(spec, "unknown", "cannot read RLIMIT_MEMLOCK", "nothing to do")
    if hard == UNLIMITED:
        return _res(spec, "ok", "the hard memlock limit is unlimited")
    return _res(spec, "missing", f"the hard memlock limit is {hard} bytes",
                "a rootless daemon cannot raise it; the VLM pack runs without it")


def _ptrace(p: HostProbe, spec: CheckSpec, gpu: bool) -> dict:
    text = (p.read_text(PTRACE_SCOPE_FILE) or "").strip()
    if not text.isdigit():
        return _res(spec, "unknown", "no Yama ptrace_scope on this kernel", "nothing to do")
    if int(text) >= 1:
        return _res(spec, "ok", f"ptrace_scope {text}")
    return _res(spec, "missing", "ptrace_scope 0: any process of your user can read another's memory",
                "informational: same-user mode does not protect against it either way")


CHECKS = (CheckSpec("linux", "you", True), CheckSpec("rootless_docker", "you", True),
          CheckSpec("uidmap", "administrator", True), CheckSpec("subuid", "administrator", True),
          CheckSpec("userns", "administrator", True), CheckSpec("userns_apparmor", "administrator", False),
          CheckSpec("rootless_net", "administrator", True),
          CheckSpec("nvidia_driver", "administrator", True, gpu_only=True),
          CheckSpec("nvidia_cdi", "administrator", True, gpu_only=True),
          CheckSpec("systemd_user", "you", False), CheckSpec("linger", "you", False),
          CheckSpec("memlock", "administrator", False), CheckSpec("ptrace_scope", "you", False))
_IMPL: dict[str, Callable[[HostProbe, CheckSpec, bool], dict]] = {
    "linux": _linux, "rootless_docker": _rootless_docker, "uidmap": _uidmap, "subuid": _subuid, "userns": _userns,
    "userns_apparmor": _userns_apparmor,
    "rootless_net": _rootless_net, "nvidia_driver": _nvidia_driver, "nvidia_cdi": _nvidia_cdi,
    "systemd_user": _systemd_user, "linger": _linger, "memlock": _memlock, "ptrace_scope": _ptrace}


def run_checks(probe: HostProbe, *, gpu: bool | None = None) -> list[dict]:
    """Every check in table order; `gpu=None` follows `gpu_default`. A check that raises is `unknown`."""
    if gpu is None:
        try:
            gpu = gpu_default(probe)
        except Exception:                                           # noqa: BLE001 - the check never stops midway
            gpu = False
    out = []
    for spec in CHECKS:
        if spec.gpu_only and not gpu:
            out.append(_res(spec, "not_applicable", "no GPU requested or found", "pass --gpu to check it",
                            required=False))
            continue
        try:
            out.append(_IMPL[spec.id](probe, spec, gpu))
        except Exception as exc:                                    # noqa: BLE001
            out.append(_res(spec, "unknown", f"the check failed: {type(exc).__name__}", "rerun the check"))
    return out


def failing_ids(checks: list[dict]) -> list[str]:
    return [c["id"] for c in checks if c["required"] and c["status"] in ("missing", "unknown")]


def exit_code(checks: list[dict]) -> int:
    return 1 if failing_ids(checks) else 0
