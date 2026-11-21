"""AN-13: the keeper runs only code, and keeps only state, that no other uid can change; model files are data only.

Code running as the keeper user can read every sealed set and the signing key, so `serve` refuses to start when any
file it runs, or any directory above one, is owned by a uid other than the keeper's and root, or is writable by group
or other (a root-owned directory with the sticky bit, such as `/tmp`, excepted). The files checked are those of the
interpreter, `sys.path`, every imported module, gatedtrain, the installed scenario packs and gatedtrain's runtime
dependencies (followed through their requirements, extras left out, so an optional torch is not hashed at start).

The environment record names those distributions with a hash of their files; the keeper signs it once at start.

Model files are read as data: a pickle-based or unknown format, pickle or zip (the torch format) content, a symbolic
link, or a hash that differs from the one the driver sent is refused before any alpha is spent.
"""

from __future__ import annotations

import hashlib
import os
import re
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

MODEL_FILE_SUFFIXES = (".safetensors", ".json", ".jsonl", ".txt")
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


def _one_path_problems(path: Path, st: os.stat_result, keeper_uid: int) -> list[Problem]:
    problems = []
    if st.st_uid not in (keeper_uid, 0):
        problems.append(_problem("environment", f"{path} is owned by uid {st.st_uid}, not the keeper (uid "
                                 f"{keeper_uid}) or root", "make the keeper user or root its owner"))
    if stat.S_ISLNK(st.st_mode):             # a link's own mode bits mean nothing; its target is checked separately
        return problems
    mode = stat.S_IMODE(st.st_mode)
    sticky_root_dir = stat.S_ISDIR(st.st_mode) and st.st_uid == 0 and mode & stat.S_ISVTX
    if mode & 0o022 and not sticky_root_dir:
        problems.append(_problem("environment", f"{path} is writable by group or other (mode {mode:04o})",
                                 f"remove the write bits: chmod go-w {path}"))
    return problems


def path_problems(paths: Iterable[Path], keeper_uid: int) -> list[Problem]:
    """Every path, or ancestor up to `/`, that a uid other than the keeper's or root could change; a symbolic link on
    the way is followed hop by hop, so the directory holding each link is checked too. A path that does not exist is
    skipped: creating it needs a write to its parent, which is checked."""
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
        problems += _one_path_problems(p, st, keeper_uid)
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
        f = getattr(module, "__file__", None)
        if isinstance(f, str) and os.path.exists(f):
            paths.append(Path(f))
    for dist in keeper_distributions():
        paths += distribution_files(dist).values()
    return paths


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
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
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


def _model_file_problem(field: str, entry: object) -> tuple[Problem | None, dict | None]:
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
    except OSError:
        return _problem(field, f"{path.name}: cannot read the file", "pass an existing model file"), None
    if stat.S_ISLNK(st.st_mode):
        return _problem(field, f"{path.name} is a symbolic link", "pass the file itself, not a link"), None
    try:
        digest, head = _read_model_file(path)
    except OSError:
        return _problem(field, f"{path.name}: cannot read the file as a regular file",
                        "pass a regular model file"), None
    if head.startswith(_PICKLE_MAGIC):
        return _problem(field, f"{path.name} holds pickle data, which runs code on load",
                        "save the model as safetensors"), None
    if head.startswith(_ZIP_MAGIC):
        return _problem(field, f"{path.name} is a zip archive (the torch format, pickle inside)",
                        "save the model as safetensors"), None
    if digest != entry["sha256"]:
        return _problem(field, f"{path.name}: the file's sha256 differs from the one sent",
                        "send the hash of the file as it is now, or stop changing the file"), None
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
