"""AN-21: does the owner's account reach a rootful Docker or containerd daemon, or hold a root-equivalent group?

The keeper's separate-user guarantee (design 05 §18.9) holds only when the owner cannot reach a rootful container
daemon: through it the owner can mount the keeper's state directory and read its signing key. `owner_root_docker`
returns the reasons the keeper must downgrade its guarantee for the owner uid, an empty list when there are none:

- the owner is in a group named `docker`, `lxd`, `incus-admin` or `disk` (`ROOT_EQUIVALENT_GROUPS`), or in the group
  of a rootful socket (one of `ROOTFUL_SOCKET_GLOBS` owned by root), by `etc/group`, by its primary gid in
  `etc/passwd`, or, on the real host, by the group list the name service returns (`os.getgrouplist`, so sssd or LDAP
  groups count);
- a live process of the owner (any of the ids on its `Uid:` line) carries such a gid in `Groups:`, or as its
  effective or saved gid on the `Gid:` line, of `proc/<pid>/status`, so a session started before the owner left the
  group counts until it ends;
- a rootful socket is writable by every user, or carries a POSIX access ACL (`system.posix_acl_access`), which can
  grant other users access;
- a TCP socket listens (state `0A`) on port 2375 or 2376 (`proc/net/tcp`, `proc/net/tcp6`).

It fails closed, with a reason, when `etc/group`, `etc/passwd`, `proc`, `proc/self/mountinfo` or `proc/net/tcp`
cannot be read, when `proc` is mounted with `hidepid` (other users' processes are hidden, so old sessions cannot be
seen), when the process table on the real host shows neither PID 1 nor any process of another uid than the
keeper's (a private PID namespace or another hidden /proc: `HostView.pid_sanity`), when a process's status cannot be
read for any reason but its end (ENOENT, ESRCH), when a socket that
matched a glob cannot be stat'ed for any reason but its end (ENOENT), when a rootful socket's extended attributes
cannot be listed (any error but ENOTSUP, ENODATA), and when the name service fails (an OSError; a group that does not
exist is no reason). Reasons name uids,
gids, pids with their command names, and paths only; at most `MAX_REASONS` are listed, then a count.

Not detected (stated in the guarantee text or in design 05 §18.10): root through `sudo`; a rootful daemon on another
socket or port; other container or virtualisation daemons.

`HostView` names the tree the detection reads, so tests run it on a fake tree under a temporary directory with
their own uid as `root_uid` (and real Unix sockets bound in the tree). The name service is asked only for the real
root (`nss` None means: only when `root` is `/`); a fake tree is read from its files alone.
"""

from __future__ import annotations

import errno
import glob
import grp
import os
import pwd
import stat
from dataclasses import dataclass
from pathlib import Path

from ...ext.launch import DOCKER_SOCKET_GLOBS

RULE = "AN-21"
ROOTFUL_SOCKET_GLOBS = DOCKER_SOCKET_GLOBS + ("/run/containerd/containerd.sock", "/var/run/containerd/containerd.sock")
DOCKER_TCP_PORTS = (2375, 2376)
ROOT_EQUIVALENT_GROUPS = ("docker", "lxd", "incus-admin", "disk")
TCP_LISTEN = "0A"
MAX_REASONS = 20                 # reasons listed; the rest are counted in one more line
_GONE = (errno.ENOENT, errno.ESRCH)
ACL_XATTR = "system.posix_acl_access"
_NO_XATTR = (errno.ENOTSUP, errno.ENODATA)


@dataclass(frozen=True)
class HostView:
    root: Path = Path("/")
    root_uid: int = 0
    nss: bool | None = None
    pid_sanity: bool | None = None

    @property
    def checks_pid_table(self) -> bool:
        """Whether a process table without PID 1 and without any process of another uid counts as hidden (None:
        only on the real root)."""
        return Path(self.root) == Path("/") if self.pid_sanity is None else self.pid_sanity

    def path(self, host_path: str) -> Path:
        return Path(self.root) / host_path.lstrip("/")

    @property
    def uses_nss(self) -> bool:
        return Path(self.root) == Path("/") if self.nss is None else self.nss


class _Unreadable(Exception):
    def __init__(self, host_path: str, exc: BaseException) -> None:
        super().__init__(host_path)
        self.reason = f"cannot read {host_path} ({type(exc).__name__}): counted as rootful access (fail closed)"


def _read(view: HostView, host_path: str) -> str:
    try:
        return view.path(host_path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise _Unreadable(host_path, exc) from None


def _groups(view: HostView) -> list[tuple[str, int, list[str]]]:
    """(name, gid, members) of every well-formed `etc/group` line."""
    out = []
    for line in _read(view, "/etc/group").splitlines():
        parts = line.split(":")
        if len(parts) >= 4 and parts[2].isdigit():
            out.append((parts[0], int(parts[2]), [m for m in parts[3].split(",") if m]))
    return out


def _passwd(view: HostView, uid: int) -> tuple[str, int] | None:
    """(name, primary gid) of `uid` in `etc/passwd`, or None."""
    for line in _read(view, "/etc/passwd").splitlines():
        parts = line.split(":")
        if len(parts) >= 4 and parts[2] == str(uid) and parts[3].isdigit():
            return parts[0], int(parts[3])
    return None


def _fail(what: str, exc: BaseException) -> str:
    return f"{what} ({type(exc).__name__}): counted as rootful access (fail closed)"


def _rootful_sockets(view: HostView, reasons: list[str]) -> list[tuple[str, os.stat_result]]:
    """(host path, stat) of every socket of `ROOTFUL_SOCKET_GLOBS` owned by `view.root_uid` (a link is followed);
    a stat error other than ENOENT, an ACL or an xattr listing error appends to `reasons`."""
    found = []
    for pattern in ROOTFUL_SOCKET_GLOBS:
        for p in sorted(glob.glob(str(view.path(pattern)))):
            host_path = "/" + Path(p).relative_to(view.root).as_posix()
            try:
                st = os.stat(p)
            except OSError as exc:
                if exc.errno != errno.ENOENT:          # not the socket vanishing between glob and stat
                    reasons.append(_fail(f"cannot stat {host_path}", exc))
                continue
            if not (stat.S_ISSOCK(st.st_mode) and st.st_uid == view.root_uid):
                continue
            found.append((host_path, st))
            try:
                if ACL_XATTR in os.listxattr(p):
                    reasons.append(f"{host_path} is a rootful daemon socket with a POSIX access ACL ({ACL_XATTR}), "
                                   "which can grant other users access: counted as rootful access")
            except OSError as exc:
                if exc.errno not in _NO_XATTR:
                    reasons.append(_fail(f"cannot list the extended attributes of {host_path}", exc))
    return found


def _hidepid(view: HostView) -> bool:
    """True when the mount at `/proc` has a `hidepid` option other than 0/off (`proc/self/mountinfo`)."""
    hidden = False
    for line in _read(view, "/proc/self/mountinfo").splitlines():
        fields = line.split()
        if len(fields) < 5 or fields[4] != "/proc":
            continue
        opts = fields[5].split(",") if len(fields) > 5 else []
        if " - " in line:
            post = line.split(" - ", 1)[1].split()
            opts += post[2].split(",") if len(post) > 2 else []
        values = [o.split("=", 1)[1] for o in opts if o.startswith("hidepid=")]
        hidden = any(v not in ("0", "off") for v in values) if values else False   # the last mount on /proc counts
    return hidden


def _tcp_listeners(view: HostView) -> list[str]:
    reasons = []
    for host_path, required in (("/proc/net/tcp", True), ("/proc/net/tcp6", False)):
        try:
            text = _read(view, host_path)
        except _Unreadable:
            if required:
                raise
            continue                                  # no IPv6 on this host
        for line in text.splitlines()[1:]:
            fields = line.split()
            if len(fields) < 4 or ":" not in fields[1]:
                continue
            try:
                port = int(fields[1].rsplit(":", 1)[1], 16)
            except ValueError:
                continue
            if fields[3] == TCP_LISTEN and port in DOCKER_TCP_PORTS:
                reasons.append(f"a TCP socket listens on port {port}, a Docker TCP endpoint port ({host_path})")
    return sorted(set(reasons))


def _status_field(text: str, name: str) -> list[str]:
    for line in text.splitlines():
        if line.startswith(name + ":"):
            return line[len(name) + 1:].split()
    return []


def _processes(view: HostView, owner_uid: int, risky: dict[int, str]) -> list[str]:
    reasons = []
    proc = view.path("/proc")
    try:
        names = os.listdir(proc)
    except OSError as exc:
        raise _Unreadable("/proc", exc) from None
    own = os.geteuid()
    visible = "1" in names                            # PID 1, or below: a process of another uid than the keeper's
    for name in sorted((n for n in names if n.isdigit()), key=int):
        host_path = f"/proc/{name}/status"
        try:
            text = (proc / name / "status").read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            if exc.errno in _GONE:
                continue                              # the process ended
            reasons.append(f"cannot read {host_path} ({type(exc).__name__}): counted as rootful access "
                           "(fail closed)")
            continue
        uids = _status_field(text, "Uid")
        if any(u.isdigit() and int(u) != own for u in uids):
            visible = True
        if str(owner_uid) not in uids:
            continue
        command = " ".join(_status_field(text, "Name")) or "?"
        carried = {int(g) for g in _status_field(text, "Groups") + _status_field(text, "Gid")[1:3] if g.isdigit()}
        for g in sorted(carried & set(risky)):         # supplementary, effective and saved gids
            reasons.append(f"pid {name} ({command}) of uid {owner_uid} carries gid {g} ({risky[g]})")
    if view.checks_pid_table and not visible:
        reasons.append("the process table is not fully visible (no PID 1 and no process of another uid: a private "
                       "PID namespace or a hidden /proc), so other sessions of the owner cannot be seen: counted as "
                       "rootful access (fail closed)")
    return reasons


def owner_root_docker(owner_uid: int, view: HostView = HostView()) -> list[str]:
    """AN-21: the reasons the owner uid can reach a rootful container daemon or holds a root-equivalent group, in a
    stable order; [] when there are none. Never raises for what it reads: an unreadable input is a reason."""
    reasons: list[str] = []
    try:
        if _hidepid(view):
            reasons.append("/proc is mounted with hidepid, so other sessions of the owner cannot be seen: counted "
                           "as rootful access (fail closed)")
    except _Unreadable as exc:
        reasons.append(exc.reason)
    risky: dict[int, str] = {}                        # gid -> why it is risky (names a group or a path)
    groups: list[tuple[str, int, list[str]]] = []
    try:
        groups = _groups(view)
    except _Unreadable as exc:
        reasons.append(exc.reason)
    for gname, gid, _ in groups:
        if gname in ROOT_EQUIVALENT_GROUPS:
            risky.setdefault(gid, f"group {gname}")
    if view.uses_nss:
        for gname in ROOT_EQUIVALENT_GROUPS:
            try:
                risky.setdefault(grp.getgrnam(gname).gr_gid, f"group {gname}")
            except KeyError:
                pass                                   # no such group
            except OSError as exc:
                reasons.append(_fail(f"cannot look up group {gname} in the name service", exc))
    sockets = _rootful_sockets(view, reasons)
    for host_path, st in sockets:
        risky.setdefault(st.st_gid, f"the group of {host_path}")
        if st.st_mode & stat.S_IWOTH:
            reasons.append(f"{host_path} is a rootful daemon socket writable by every user")
    entry = None
    try:
        entry = _passwd(view, owner_uid)
    except _Unreadable as exc:
        reasons.append(exc.reason)
    if entry is None and view.uses_nss:
        try:
            pw = pwd.getpwuid(owner_uid)
            entry = (pw.pw_name, pw.pw_gid)
        except KeyError:
            pass                                       # no such user
        except OSError as exc:
            reasons.append(_fail(f"cannot look up uid {owner_uid} in the name service", exc))
    member: dict[int, str] = {}
    if entry is not None:
        uname, primary = entry
        if primary in risky:
            member.setdefault(primary, "its primary gid (/etc/passwd)")
        for _, gid, members in groups:
            if gid in risky and uname in members:
                member.setdefault(gid, "/etc/group")
        if view.uses_nss:
            try:
                for gid in os.getgrouplist(uname, primary):
                    if gid in risky:
                        member.setdefault(gid, "the name service (getgrouplist)")
            except OSError as exc:
                reasons.append(_fail(f"cannot list the groups of uid {owner_uid} in the name service", exc))
    for gid in sorted(member):
        reasons.append(f"uid {owner_uid} is in gid {gid} ({risky[gid]}) by {member[gid]}")
    try:
        reasons += _processes(view, owner_uid, risky)
    except _Unreadable as exc:
        reasons.append(exc.reason)
    try:
        reasons += _tcp_listeners(view)
    except _Unreadable as exc:
        reasons.append(exc.reason)
    if len(reasons) > MAX_REASONS:
        rest = len(reasons) - MAX_REASONS
        reasons = reasons[:MAX_REASONS] + [f"and {rest} more reason{'' if rest == 1 else 's'}"]
    return reasons
