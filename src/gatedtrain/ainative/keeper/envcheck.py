"""AN-13: the keeper runs only code, and keeps only state, that no other uid can change; model files are data only.

Code running as the keeper user can read every sealed set and the signing key, so `serve` refuses to start when any
file it runs, or any directory above one, is writable by another uid (spec section 5.1). Precisely, a path is refused
when:
- its owner is neither the keeper's uid nor root (0);
- it has the other-write bit;
- it has the group-write bit, unless its group is the user-private group of an allowed user and it has no POSIX ACL.
  The allowed uids are the keeper's uid (which is the owner uid when the keeper runs as the user) and root. A
  user-private group is one whose membership is certain without enumerating the user database: its gid is the
  allowed user's primary gid, its name is that user's name, and its supplementary members are allowed users only.
  Any other group (a shared group, or one that cannot be resolved) refuses, since users whose primary group it is
  may not be listable (NSS without enumeration). With a POSIX ACL (`system.posix_acl_access`, or
  `system.posix_acl_default` on a directory) the group bits are the ACL mask, so a named user may write: refused. A
  filesystem without extended attributes has no ACL.
A root-owned directory with the sticky bit (such as `/tmp`) is excepted from the write-bit rules. The state directory
itself must be exactly 0700 (spec section 5.1); its ancestors follow the rules above.

The files checked are those of the interpreter, `sys.path`, every module imported when the check runs (its source and
its cached bytecode), the `__pycache__` directory next to every checked `.py` file and the files in it, every `.pth`
file in the site directories (run at every interpreter start), gatedtrain, the installed scenario packs and
gatedtrain's runtime dependencies (followed through their requirements, extras left out, so an optional torch is not
hashed at start). Known limit: a module that is first imported after the check and lies outside those distributions
is not checked; site-packages is not walked as a whole.

The environment record names those distributions with a hash of their files; the keeper signs it once at start.

Model files are read as data: a pickle-based or unknown format, pickle or zip (the torch format) content, a symbolic
link, or a hash that differs from the one the driver sent is refused before any alpha is spent. The keeper copies the
files a model reference lists into its own state and hashes the copy (`copy_model_files`); the generator receives the
record with the copies' paths (`copied_record`), so the hash in the claim is the hash of what runs.
"""

from __future__ import annotations

import errno
import grp
import hashlib
import os
import pwd
import site
import re
import shutil
import stat
import sys
import sysconfig
from collections import deque
from collections.abc import Iterable
from importlib import metadata, util
from pathlib import Path

from ...canonical import canonical_json, sha256_bytes
from ...errors import Problem, Refused

RULE = "AN-13"
ROOT_DISTRIBUTION = "gatedtrain"
SCENARIO_ENTRY_POINT_GROUP = "gatedtrain.scenarios"

MODEL_FILE_SUFFIXES = (".safetensors", ".json", ".jsonl", ".txt", ".jinja")
PICKLE_SUFFIXES = (".pt", ".pth", ".bin", ".pkl", ".pickle", ".ckpt", ".joblib")
_PICKLE_MAGIC = b"\x80"
_ZIP_MAGIC = b"PK\x03\x04"
_CHUNK = 1 << 20
_REQ_NAME = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _problem(field: str, message: str, fix: str) -> Problem:
    return Problem(RULE, field, message, fix)


# ---- ownership and write bits -------------------------------------------------------------------------------------

def _chain(path: Path) -> list[Path]:
    """The path and every ancestor up to `/`, for the path as given and for its resolved form."""
    given = Path(os.path.abspath(path))
    real = Path(os.path.realpath(given))
    chain = [given, *given.parents]
    if real != given:
        chain += [real, *real.parents]
    return chain


_ACL_ATTRS = ("system.posix_acl_access",)
_DEFAULT_ACL_ATTR = "system.posix_acl_default"


def _has_posix_acl(path: Path, is_dir: bool) -> bool:
    try:
        names = os.listxattr(path, follow_symlinks=False)
    except OSError as exc:
        if exc.errno in (errno.ENOTSUP, errno.EOPNOTSUPP):
            return False                     # no extended attributes on this filesystem: no ACL
        return True                          # cannot tell: treat as an ACL (refuse)
    wanted = _ACL_ATTRS + ((_DEFAULT_ACL_ATTR,) if is_dir else ())
    return any(n in names for n in wanted)


class _Groups:
    """Whether a gid is the user-private group of an allowed user (known without enumerating the user database)."""

    def __init__(self, allowed: frozenset[int]) -> None:
        self.users = {}
        for uid in allowed:
            try:
                self.users[uid] = pwd.getpwuid(uid)
            except KeyError:
                pass
        self.names = {u.pw_name for u in self.users.values()}
        self._cache: dict[int, bool] = {}

    def is_private(self, gid: int) -> bool:
        if gid not in self._cache:
            self._cache[gid] = self._lookup(gid)
        return self._cache[gid]

    def _lookup(self, gid: int) -> bool:
        try:
            group = grp.getgrgid(gid)
        except KeyError:
            return False
        return any(u.pw_gid == gid and group.gr_name == u.pw_name for u in self.users.values()) \
            and set(group.gr_mem) <= self.names


def _one_path_problems(path: Path, st: os.stat_result, keeper_uid: int, groups: _Groups) -> list[Problem]:
    problems = []
    if st.st_uid not in (keeper_uid, 0):
        problems.append(_problem("environment", f"{path} is owned by uid {st.st_uid}, not the keeper (uid "
                                 f"{keeper_uid}) or root", "make the keeper user or root its owner"))
    if stat.S_ISLNK(st.st_mode):             # a link's own mode bits mean nothing; its target is checked separately
        return problems
    mode = stat.S_IMODE(st.st_mode)
    if stat.S_ISDIR(st.st_mode) and st.st_uid == 0 and mode & stat.S_ISVTX:
        return problems
    if mode & 0o002:
        problems.append(_problem("environment", f"{path} is writable by every user (mode {mode:04o})",
                                 f"remove the write bit: chmod o-w {path}"))
    if mode & 0o020:
        if not groups.is_private(st.st_gid):
            problems.append(_problem("environment", f"{path} is writable by its group (mode {mode:04o}, gid "
                                     f"{st.st_gid}), which is not the private group of the keeper or root",
                                     f"remove the write bit: chmod g-w {path}"))
        elif _has_posix_acl(path, stat.S_ISDIR(st.st_mode)):
            problems.append(_problem("environment", f"{path} has a POSIX ACL and group write bits (mode "
                                     f"{mode:04o}), so a named user may write to it",
                                     f"remove the ACL (setfacl -b {path}) or the write bit (chmod g-w {path})"))
    return problems


def path_problems(paths: Iterable[Path], keeper_uid: int) -> list[Problem]:
    """Every path, or ancestor up to `/`, that a uid other than the keeper's or root could change (the rules in the
    module docstring; the keeper's uid is the owner uid in same-user mode, so it covers both); a symbolic link on
    the way is followed hop by hop, so the directory holding each link is checked too. A path that does not exist is
    skipped: creating it needs a write to its parent, which is checked."""
    groups = _Groups(frozenset({keeper_uid, 0}))
    seen: set[Path] = set()
    problems: list[Problem] = []
    todo = deque(p for path in paths for p in _chain(Path(path)))
    while todo:
        p = todo.popleft()
        if p in seen:
            continue
        seen.add(p)
        try:
            st = os.lstat(p)
        except FileNotFoundError:
            continue
        except OSError:
            problems.append(_problem("environment", f"cannot inspect {p}",
                                     "make the keeper's code readable by the keeper user"))
            continue
        problems += _one_path_problems(p, st, keeper_uid, groups)
        if stat.S_ISLNK(st.st_mode):
            try:
                todo.extend(_chain(p.parent / os.readlink(p)))
            except OSError:
                pass
    return problems


def state_dir_problems(state_dir: Path, keeper_uid: int) -> list[Problem]:
    """The state directory must exist, be a directory and not a link, be owned by the keeper, and have mode 0700."""
    fix = f"create it as the keeper user: mkdir -m 0700 {state_dir}"
    try:
        st = os.lstat(state_dir)
    except FileNotFoundError:
        return [_problem("state", f"{state_dir} does not exist", fix)]
    except OSError:
        return [_problem("state", f"cannot inspect {state_dir}", fix)]
    if stat.S_ISLNK(st.st_mode):
        return [_problem("state", f"{state_dir} is a symbolic link", "pass the real directory, not a link")]
    if not stat.S_ISDIR(st.st_mode):
        return [_problem("state", f"{state_dir} is not a directory", fix)]
    problems = []
    if st.st_uid != keeper_uid:
        problems.append(_problem("state", f"{state_dir} is owned by uid {st.st_uid}, not the keeper (uid "
                                 f"{keeper_uid})", f"chown the keeper user {state_dir}"))
    mode = stat.S_IMODE(st.st_mode)
    if mode != 0o700:
        problems.append(_problem("state", f"{state_dir} has mode {mode:04o}, not 0700",
                                 f"chmod 0700 {state_dir}"))
    return problems


# ---- the distributions the keeper runs ----------------------------------------------------------------------------

def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _distribution(name: str) -> metadata.Distribution | None:
    try:
        return metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None


def _runtime_requirements(dist: metadata.Distribution) -> list[str]:
    """Names of the distribution's requirements outside any extra. Other markers are not evaluated: an installed
    distribution a marker would leave out is included, which only widens the check."""
    names = []
    for req in dist.requires or []:
        _, _, marker = req.partition(";")
        if "extra" in marker:
            continue
        m = _REQ_NAME.match(req)
        if m:
            names.append(m.group(1))
    return names


def keeper_distributions() -> list[metadata.Distribution]:
    """gatedtrain, every distribution providing a scenario pack, and their runtime requirements, transitively."""
    roots = [d for d in [_distribution(ROOT_DISTRIBUTION)] if d is not None]
    for ep in metadata.entry_points(group=SCENARIO_ENTRY_POINT_GROUP):
        if ep.dist is not None:
            roots.append(ep.dist)
    found: dict[str, metadata.Distribution] = {}
    todo = list(roots)
    while todo:
        dist = todo.pop()
        key = _norm(dist.metadata["Name"])
        if key in found:
            continue
        found[key] = dist
        for name in _runtime_requirements(dist):
            if _norm(name) not in found and (req := _distribution(name)) is not None:
                todo.append(req)
    return [found[k] for k in sorted(found)]


def _top_level_names(dist: metadata.Distribution) -> list[str]:
    text = dist.read_text("top_level.txt") or ""
    names = [line.strip() for line in text.splitlines() if line.strip()]
    if not names:
        names = [_norm(dist.metadata["Name"]).replace("-", "_")]
    return names


def _package_files(name: str) -> list[tuple[Path, Path]]:
    """`(root, file)` pairs of an importable top-level package or module, found without importing it (an editable
    install's sources are not in the distribution's file list); `root` is the directory that holds it."""
    try:
        spec = util.find_spec(name)
    except (ImportError, ValueError):
        return []
    if spec is None:
        return []
    out = []
    if spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            root = Path(os.path.abspath(location)).parent
            for dirpath, dirnames, filenames in os.walk(location):
                dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
                out += [(root, Path(os.path.abspath(Path(dirpath) / f))) for f in filenames]
    elif spec.origin and os.path.isfile(spec.origin):
        origin = Path(os.path.abspath(spec.origin))
        out.append((origin.parent, origin))
    return out


def distribution_files(dist: metadata.Distribution) -> dict[str, Path]:
    """`{name: absolute path}` of the distribution's existing files (compiled bytecode left out), named relative to
    the install directory, or to the directory holding the package when it lies outside (an editable install)."""
    base = Path(os.path.abspath(dist.locate_file("")))
    out: dict[str, Path] = {}
    for f in dist.files or []:
        if "__pycache__" in f.parts:
            continue
        p = Path(os.path.abspath(dist.locate_file(f)))
        if p.is_file():
            out[Path(f).as_posix()] = p
    for top in _top_level_names(dist):
        for root, p in _package_files(top):
            try:
                name = p.relative_to(base).as_posix()
            except ValueError:
                name = p.relative_to(root).as_posix()
            out.setdefault(name, p)
    return out


def keeper_code_paths() -> list[Path]:
    """`sys.prefix`, the interpreter, the stdlib directory, every existing `sys.path` entry (an empty entry is the
    working directory), the file of every imported module, and the files of the keeper's distributions."""
    paths: list[Path] = [Path(sys.prefix), Path(sys.executable)]
    for key in ("stdlib", "platstdlib"):
        p = sysconfig.get_path(key)
        if p:
            paths.append(Path(p))
    for entry in sys.path:
        p = Path(entry or os.getcwd())
        if p.exists():
            paths.append(p)
    for module in list(sys.modules.values()):
        for attr in ("__file__", "__cached__"):
            f = getattr(module, attr, None)
            if isinstance(f, str) and os.path.exists(f):
                paths.append(Path(f))
    for dist in keeper_distributions():
        paths += distribution_files(dist).values()
    paths += _pth_files()
    paths += _bytecode_paths(paths)
    return paths


def _site_dirs() -> list[str]:
    dirs = list(site.getsitepackages()) if hasattr(site, "getsitepackages") else []
    dirs += [d for d in (sysconfig.get_path("purelib"), sysconfig.get_path("platlib")) if d]
    return sorted(set(dirs))


def _pth_files() -> list[Path]:
    """Every `.pth` file directly in a site directory: the interpreter runs their import lines at every start."""
    out = []
    for d in _site_dirs():
        try:
            names = os.listdir(d)
        except OSError:
            continue
        out += [Path(d) / n for n in sorted(names) if n.endswith(".pth")]
    return out


def _bytecode_paths(paths: list[Path]) -> list[Path]:
    """The `__pycache__` directory next to every `.py` file and the files in it: CPython loads a matching `.pyc`
    without reading the source."""
    out: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        if p.suffix != ".py":
            continue
        cache = Path(os.path.abspath(p)).parent / "__pycache__"
        if cache in seen:
            continue
        seen.add(cache)
        try:
            names = os.listdir(cache)
        except OSError:                      # missing: creating it needs a write to the parent, which is checked
            continue
        out.append(cache)
        out += [cache / n for n in sorted(names)]
    return out


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def environment_record() -> dict:
    """`{"python", "distributions": [{"name", "version", "files_sha256"}]}`, sorted by name; `files_sha256` is the
    sha256 of the canonical JSON of the sorted `[name, sha256]` pairs of the distribution's files."""
    dists = []
    for dist in keeper_distributions():
        pairs = []
        for name, p in distribution_files(dist).items():
            try:
                pairs.append([name, _file_sha256(p)])
            except OSError:
                raise Refused([_problem("environment", f"cannot read {p} to record the environment",
                                        "make the keeper's code readable by the keeper user")]) from None
        pairs.sort()
        dists.append({"name": dist.metadata["Name"], "version": dist.version,
                      "files_sha256": sha256_bytes(canonical_json(pairs))})
    dists.sort(key=lambda d: _norm(d["name"]))
    return {"python": sys.version, "distributions": dists}


def environment_sha256(record: dict) -> str:
    return sha256_bytes(canonical_json(record))


# ---- model files --------------------------------------------------------------------------------------------------

def _read_model_file(path: Path) -> tuple[str, bytes]:
    """`(sha256, first bytes)` of a regular file opened without following a final link."""
    # O_NONBLOCK: a FIFO swapped in after the lstat check must not block the keeper on open.
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise IsADirectoryError(path)
        h = hashlib.sha256()
        head = b""
        while chunk := handle.read(_CHUNK):
            if not head:
                head = chunk[:len(_ZIP_MAGIC)]
            h.update(chunk)
    return "sha256:" + h.hexdigest(), head


TRAVERSE_FIX = "put model files under a directory the keeper user can traverse, such as the output root (mode 0755)"


def _entry_problem(field: str, entry: object) -> tuple[Problem | None, os.stat_result | None]:
    """The checks on an entry before its file is opened: shape, suffix, and a regular file that is not a link."""
    if not (isinstance(entry, dict) and isinstance(entry.get("path"), str) and isinstance(entry.get("sha256"), str)):
        return _problem(field, "a model file entry must be {path, sha256} strings",
                        "list each model file with its path and sha256"), None
    path = Path(entry["path"])
    suffix = path.suffix.lower()
    if suffix in PICKLE_SUFFIXES:
        return _problem(field, f"{path.name}: the {suffix} format is pickle-based and runs code on load",
                        "save the model as safetensors (with JSON or text side files)"), None
    if suffix not in MODEL_FILE_SUFFIXES:
        return _problem(field, f"{path.name}: unknown model file format {suffix or '(no suffix)'}",
                        f"use one of: {', '.join(MODEL_FILE_SUFFIXES)}"), None
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return _problem(field, f"{path.name}: cannot read the file", "pass an existing model file"), None
    except OSError:
        return _problem(field, f"{path.name}: the keeper cannot reach the file", TRAVERSE_FIX), None
    if stat.S_ISLNK(st.st_mode):
        return _problem(field, f"{path.name} is a symbolic link", "pass the file itself, not a link"), None
    if not stat.S_ISREG(st.st_mode):
        return _problem(field, f"{path.name} is not a regular file", "pass a regular model file"), None
    return None, st


def _head_problem(field: str, name: str, head: bytes) -> Problem | None:
    if head.startswith(_PICKLE_MAGIC):
        return _problem(field, f"{name} holds pickle data, which runs code on load", "save the model as safetensors")
    if head.startswith(_ZIP_MAGIC):
        return _problem(field, f"{name} is a zip archive (the torch format, pickle inside)",
                        "save the model as safetensors")
    return None


def _hash_problem(field: str, name: str, digest: str, sent: str) -> Problem | None:
    if digest != sent:
        return _problem(field, f"{name}: the file's sha256 differs from the one sent",
                        "send the hash of the file as it is now, or stop changing the file")
    return None


def _model_file_problem(field: str, entry: object) -> tuple[Problem | None, dict | None]:
    problem, _ = _entry_problem(field, entry)
    if problem is not None:
        return problem, None
    path = Path(entry["path"])
    try:
        digest, head = _read_model_file(path)
    except OSError:
        return _problem(field, f"{path.name}: cannot read the file as a regular file",
                        "pass a regular model file"), None
    problem = _head_problem(field, path.name, head) or _hash_problem(field, path.name, digest, entry["sha256"])
    if problem is not None:
        return problem, None
    return None, {"path": entry["path"], "sha256": digest}


def check_model_files(record: dict) -> list[dict]:
    """The `{path, sha256}` entries of `record["files"]` hashed on read; Refused (AN-13) naming every bad entry."""
    entries = record.get("files", [])
    if not isinstance(entries, list):
        raise Refused([_problem("files", "the model files must be a list", "send a list of {path, sha256}")])
    problems, hashed = [], []
    for i, entry in enumerate(entries):
        problem, ok = _model_file_problem(f"files[{i}]", entry)
        if problem is not None:
            problems.append(problem)
        else:
            hashed.append(ok)
    if problems:
        raise Refused(problems)
    return hashed


def _copy_one(field: str, entry: dict, checked: os.stat_result, target: Path) -> tuple[Problem | None, dict | None]:
    """Open the checked file without following a link, check it is the file that was checked, and stream it into
    `target` (created exclusively, mode 0600) while hashing; the head is checked before anything is written."""
    path = Path(entry["path"])
    try:
        src = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError:
        return _problem(field, f"{path.name}: the keeper cannot open the file", TRAVERSE_FIX), None
    with os.fdopen(src, "rb") as handle:
        st = os.fstat(handle.fileno())
        if not stat.S_ISREG(st.st_mode) or (st.st_dev, st.st_ino) != (checked.st_dev, checked.st_ino):
            return _problem(field, f"{path.name} changed between the check and the copy",
                            "stop changing the model files while the keeper evaluates"), None
        h = hashlib.sha256()
        size = 0
        try:
            chunk = handle.read(_CHUNK)
        except OSError:
            return _problem(field, f"{path.name}: the keeper cannot read the file", TRAVERSE_FIX), None
        problem = _head_problem(field, path.name, chunk)
        if problem is not None:
            return problem, None
        try:
            dst = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        except OSError:
            return _problem(field, f"{path.name}: the keeper cannot copy the file into its state",
                            "check the keeper's state directory and its disk"), None
        with os.fdopen(dst, "wb") as out:
            try:
                while chunk:
                    h.update(chunk)
                    size += len(chunk)
                    out.write(chunk)
                    chunk = handle.read(_CHUNK)
                out.flush()
                os.fsync(out.fileno())
            except OSError:
                return _problem(field, f"{path.name}: the keeper cannot copy the file into its state",
                                "check the keeper's state directory and its disk"), None
    digest = "sha256:" + h.hexdigest()
    problem = _hash_problem(field, path.name, digest, entry["sha256"])
    if problem is not None:
        return problem, None
    return None, {"name": path.name, "sha256": digest, "size": size}


def copy_model_files(record: dict, dest: Path) -> list[dict]:
    """Copy each file `record["files"]` lists into `dest/<base name>` and hash the copy (AN-13): the checks of
    `check_model_files`, then the file is opened with `O_NOFOLLOW | O_NONBLOCK`, must be the regular file that was
    checked, and is streamed into a new file (`O_EXCL`, mode 0600) while hashing. `dest` must not exist; it is created
    with mode 0700. Refused (AN-13) naming every bad entry: also two entries with one base name and a file the keeper
    cannot open; on a refusal `dest` is removed. Returns `[{"name", "sha256", "size"}]` in the listed order."""
    entries = record.get("files", [])
    if not isinstance(entries, list):
        raise Refused([_problem("files", "the model files must be a list", "send a list of {path, sha256}")])
    os.mkdir(dest, 0o700)
    os.chmod(dest, 0o700)
    problems: list[Problem] = []
    copied: list[dict] = []
    names: set[str] = set()
    try:
        for i, entry in enumerate(entries):
            field = f"files[{i}]"
            problem, checked = _entry_problem(field, entry)
            if problem is None:
                name = os.path.basename(entry["path"])
                if name in names:
                    problem = _problem(field, f"{name}: another listed file has the same base name",
                                       "list files with distinct base names (the keeper copies them into one "
                                       "directory)")
                else:
                    names.add(name)
                    problem, ok = _copy_one(field, entry, checked, dest / name)
            if problem is not None:
                problems.append(problem)
            else:
                copied.append(ok)
        if problems:
            raise Refused(problems)
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    return copied


def copied_record(record: dict, dest: Path) -> dict:
    """The model record with each `files[].path` replaced by `dest/<base name>`: what the generator, per-item or
    batch, receives once `copy_model_files(record, dest)` succeeded (a record without `files` is returned as is)."""
    if "files" not in record:
        return dict(record)
    return {**record, "files": [{**e, "path": str(Path(dest) / os.path.basename(e["path"]))}
                                for e in record.get("files", [])]}
