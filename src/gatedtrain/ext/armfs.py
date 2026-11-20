"""Reading arm files without following links (MULTIMODAL-SPEC MM-1, MM-5; docs/findings.md, "Review of PR 2").

An arm directory is written by a container that may still run while it is collected, so nothing in it is read by
path. The arm directory itself is opened once as the operator names it (a link to the arm is the operator's choice);
every file below it is reached component by component under directory descriptors, never following a symbolic link:

- on Linux (`O_PATH` and `/proc/self/fd`): intermediate components with `O_PATH | O_DIRECTORY | O_NOFOLLOW` (a
  directory the user may only search, mode 0o111, is fine; a link is refused); the last component with `O_PATH |
  O_NOFOLLOW`, then `fstat`: only a regular file is opened for reading, through `/proc/self/fd/<n>` with
  `O_NOCTTY`, and the reopened file must be the same (`st_dev`, `st_ino`). So a FIFO, a socket or a device node is
  never opened (no block, no side effect of `open`), and nothing swapped in between is read;
- elsewhere (macOS), with weaker guarantees: intermediate components with `O_RDONLY | O_DIRECTORY | O_NOFOLLOW` (a
  search-only directory is refused with EACCES), the last one inspected by `fstatat` without following and then
  opened with `O_NOFOLLOW | O_NONBLOCK | O_NOCTTY`. A file swapped for a FIFO or device between the two is opened
  (without blocking, `O_NONBLOCK`) and then refused by the identity check, so `open` itself may have a side effect
  on a device there.

`ArmSnapshot.read` reads `items.jsonl`, `eval_meta.json` and `merge.json` once each (at most `MAX_FILE_BYTES`), with
their SHA-256; every check parses those bytes and `ext-collect` writes them into the run, so the run holds exactly
the bytes that were checked. The arm's identity (`st_dev`, `st_ino` of the opened directory) tells arms apart
(S4) and is checked again whenever the arm is reopened (output files). A path is checked lexically first
(`path_problem`): a UTF-8 string without NUL, relative, in canonical form (no `.`, `..` or empty component).
"""

from __future__ import annotations

import errno
import hashlib
import os
import posixpath
import stat
from dataclasses import dataclass, field
from pathlib import Path

ARM_FILES = ("items.jsonl", "eval_meta.json", "merge.json")
MAX_FILE_BYTES = 1 << 30                             # an arm file or a provenance source is read into memory
_O_PATH = getattr(os, "O_PATH", None)
_PROC_FD = _O_PATH is not None and os.path.isdir("/proc/self/fd")
_WALK = (_O_PATH if _PROC_FD else os.O_RDONLY) | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_NOCTTY = getattr(os, "O_NOCTTY", 0)


class Unreadable(Exception):
    """Why a file below a directory descriptor was not opened: `kind` is `link` (`detail`: the component),
    `absent`, `not_regular`, `changed`, `too_large` or `error` (`detail`: the errno name)."""

    def __init__(self, kind: str, detail: str = "") -> None:
        super().__init__(kind, detail)
        self.kind, self.detail = kind, detail

    def reason(self) -> str:
        """The problem without its subject; never the file's content."""
        return {"link": f"contains a symbolic link ({self.detail!r}; MM-1 allows none)",
                "absent": "is absent",
                "not_regular": "is not a regular file",
                "changed": "changed while it was opened",
                "too_large": f"is larger than {self.detail} bytes"}.get(self.kind,
                                                                           f"cannot be opened ({self.detail})")

    def text(self, what: str) -> str:
        return f"{what} {self.reason()}"


def errname(exc: OSError) -> str:
    return errno.errorcode.get(exc.errno, exc.__class__.__name__) if exc.errno else exc.__class__.__name__


def path_problem(p: object) -> str | None:
    """Why `p` is not a relative path in canonical form, or None (MM-1): a UTF-8 string without NUL, not absolute,
    no `..` ("leaves the arm"), no `.` or empty component (so one file has one spelling, A5)."""
    if not isinstance(p, str):
        return "is not a string"
    try:
        p.encode("utf-8")
    except UnicodeEncodeError:
        return "is not valid UTF-8"
    if "\x00" in p:
        return "contains a NUL character"
    parts = p.split("/")
    if p.startswith("/") or ".." in parts:
        return "leaves the arm"
    if any(x in ("", ".") for x in parts):
        return "is not in canonical form (no `.` or empty component)"
    return None


def _classify(dir_fd: int, name: str, exc: OSError) -> Unreadable:
    try:
        link = stat.S_ISLNK(os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode)
    except OSError:
        link = False
    if link or exc.errno == errno.ELOOP:
        return Unreadable("link", name)
    if exc.errno in (errno.ENOENT, errno.ENOTDIR):
        return Unreadable("absent")
    return Unreadable("error", errname(exc))


def open_dir(dir_fd: int, rel: str) -> int:
    """A descriptor of the directory `rel` (canonical, relative; "" is `dir_fd` itself, duplicated) below `dir_fd`,
    no component a link. Raises Unreadable."""
    cur = os.dup(dir_fd)
    try:
        for name in rel.split("/") if rel else ():
            try:
                nxt = os.open(name, _WALK, dir_fd=cur)
            except OSError as exc:
                raise _classify(cur, name, exc) from None
            os.close(cur)
            cur = nxt
    except BaseException:
        os.close(cur)
        raise
    return cur


def open_regular(dir_fd: int, rel: str) -> int:
    """A read-only descriptor of the regular file `rel` (canonical, relative) below `dir_fd`, reached without
    following a link and opened for reading only once it is known to be a regular file. Raises Unreadable."""
    head, _, name = rel.rpartition("/")
    parent = open_dir(dir_fd, head)
    try:
        if _PROC_FD:
            try:
                pfd = os.open(name, _O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
            except OSError as exc:
                raise _classify(parent, name, exc) from None
            try:
                st = os.fstat(pfd)
                if stat.S_ISLNK(st.st_mode):
                    raise Unreadable("link", name)
                if not stat.S_ISREG(st.st_mode):
                    raise Unreadable("not_regular")
                try:
                    fd = os.open(f"/proc/self/fd/{pfd}", os.O_RDONLY | _NOCTTY | os.O_CLOEXEC)
                except OSError as exc:
                    raise Unreadable("error", errname(exc)) from None
            finally:
                os.close(pfd)
        else:
            try:
                st = os.stat(name, dir_fd=parent, follow_symlinks=False)
            except OSError as exc:
                raise _classify(parent, name, exc) from None
            if stat.S_ISLNK(st.st_mode):
                raise Unreadable("link", name)
            if not stat.S_ISREG(st.st_mode):
                raise Unreadable("not_regular")
            try:
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | _NOCTTY | os.O_CLOEXEC,
                             dir_fd=parent)
            except OSError as exc:
                raise _classify(parent, name, exc) from None
        try:
            now = os.fstat(fd)
        except OSError as exc:
            os.close(fd)
            raise Unreadable("error", errname(exc)) from None
        if (now.st_dev, now.st_ino) != (st.st_dev, st.st_ino) or not stat.S_ISREG(now.st_mode):
            os.close(fd)
            raise Unreadable("changed")
        return fd
    finally:
        os.close(parent)


def read_fd(fd: int, limit: int | None = None) -> bytes:
    """Everything `fd` holds, at most `limit` bytes (default `MAX_FILE_BYTES`, read when called; Unreadable
    `too_large` beyond; OSError as Unreadable)."""
    limit = MAX_FILE_BYTES if limit is None else limit
    chunks, n = [], 0
    try:
        while chunk := os.read(fd, 1 << 20):
            n += len(chunk)
            if n > limit:
                raise Unreadable("too_large", str(limit))
            chunks.append(chunk)
    except OSError as exc:
        raise Unreadable("error", errname(exc)) from None
    return b"".join(chunks)


def read_file(dir_fd: int, rel: str) -> bytes:
    """The bytes of the regular file `rel` below `dir_fd` (open_regular, read_fd). Raises Unreadable."""
    fd = open_regular(dir_fd, rel)
    try:
        return read_fd(fd)
    finally:
        os.close(fd)


def sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def open_arm(arm: Path | str, key: tuple[int, int] | None = None) -> int:
    """A descriptor of the arm directory, opened as named (following a link the operator chose). With `key`, the
    directory must still be that one (Unreadable `changed`). Raises Unreadable (`error`, `absent`)."""
    try:
        fd = os.open(arm, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except (OSError, ValueError) as exc:
        if isinstance(exc, OSError) and exc.errno in (errno.ENOENT, errno.ENOTDIR):
            raise Unreadable("absent") from None
        raise Unreadable("error", errname(exc) if isinstance(exc, OSError) else "ValueError") from None
    if key is not None:
        st = os.fstat(fd)
        if (st.st_dev, st.st_ino) != key:
            os.close(fd)
            raise Unreadable("changed")
    return fd


@dataclass
class ArmFile:
    """One of ARM_FILES: its bytes and SHA-256, or None and why not (`missing`: absent or the arm is absent)."""
    data: bytes | None = None
    sha256: str | None = None
    problem: str | None = None


@dataclass
class ArmSnapshot:
    """The arm's files read once (module docstring). `key` is None when the arm directory did not open."""
    path: Path
    key: tuple[int, int] | None
    files: dict[str, ArmFile] = field(default_factory=dict)

    @classmethod
    def read(cls, arm: Path | str, *, dir_fd: int | None = None) -> "ArmSnapshot":
        """Open the arm (or use `dir_fd`, a directory reached by `open_dir`) and read ARM_FILES."""
        path = Path(arm)
        if dir_fd is None:
            try:
                fd = open_arm(path)
            except Unreadable as exc:
                why = "missing" if exc.kind == "absent" else f"cannot be read ({exc.detail})"
                return cls(path, None, {n: ArmFile(problem=why) for n in ARM_FILES})
        else:
            fd = os.dup(dir_fd)
        try:
            st = os.fstat(fd)
            files = {}
            for name in ARM_FILES:
                try:
                    data = read_file(fd, name)
                except Unreadable as exc:
                    files[name] = ArmFile(problem="missing" if exc.kind == "absent" else exc.reason())
                    continue
                files[name] = ArmFile(data, sha256(data))
            return cls(path, (st.st_dev, st.st_ino), files)
        finally:
            os.close(fd)

    def problem(self, name: str) -> str | None:
        """None when the file was read, else `<arm>/<name> <why>` (`is missing` when absent)."""
        f = self.files[name]
        if f.problem is None:
            return None
        why = "is missing" if f.problem == "missing" else f.problem
        return f"{self.path / name} {why}"


def under_root(root: Path | str, path: object) -> str | None:
    """`path` relative to `root` when it is an absolute path in normal form lexically at or below `root`, else None
    (MM-5: provenance sources lie under the sources root). A relative `root` is taken from the working directory,
    and `path` may name the root as written (absolute) or by its real path (a source recorded through the link
    that the output root is, or the link's target)."""
    from ..paths import realpath

    for r in dict.fromkeys((os.path.abspath(root), realpath(os.path.abspath(root)))):
        rel = _below(r, path)
        if rel is not None:
            return rel
    return None


def _below(r: str, path: object) -> str | None:
    if not (isinstance(path, str) and path.startswith("/") and "\x00" not in path
            and posixpath.normpath(path) == path and not path.startswith("//")):
        return None
    try:
        path.encode("utf-8")
    except UnicodeEncodeError:
        return None
    if path == r:
        return ""
    prefix = r if r.endswith("/") else r + "/"
    return path[len(prefix):] if path.startswith(prefix) else None
