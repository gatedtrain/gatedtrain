"""The image lock: which container images gatedtrain accepts, as data (IMG-1 .. IMG-4).

Code names LOGICAL images (`nemo-rl/tierb`, ...). Their IDs live in two data files:

- the shipped lock, `data/image-lock.json`, written by the maintainers and installed with the code (root-owned on a
  keeper host): one entry per logical image listing the accepted IDs, each with its recipe, base image, build date and
  origin `maintainer-built`;
- the owner's config, `$XDG_CONFIG_HOME/gatedtrain/images.json` (default `~/.config/gatedtrain/images.json`): it can
  only ADD IDs of origin `user-built` to logical images the lock already names. It cannot invent a logical image or
  remove or edit a lock entry.

Precedence: lock entries first, then config additions in file order. The config may not name an ID the lock lists,
under any logical image (IMG-2), so it can never re-label or re-attribute a shipped image. No environment variable,
command-line flag or project file supplies an accepted ID. LIMIT, stated plainly: the environment (`HOME`,
`XDG_CONFIG_HOME`) decides WHERE the owner's config is, and an agent running as the same user controls both its
environment and that user's files, so it can run `gatedtrain image add --yes` or write the file itself. The owner
config is therefore NOT a security boundary against a same-user agent. What stays reliable is the keeper's flag
(`user_added`) in the signed report, which holds when the keeper's code and shipped lock are outside the agent's reach
(a keeper that runs as a separate user); the report also carries the sha256 of that lock. The config is read without
following symlinks and must be a regular file owned by the running uid that group and others cannot write; a file that
breaks this is refused (IMG-3), never ignored. Every free-text field is one printable line (no control characters), so
a file cannot forge a line of `report.md` or of CLI output.

Both files carry `schema` and `schema_version` (exactly 1: another version is refused, never read as best effort),
have exact key sets (an unknown or a missing field is refused), reject duplicate JSON keys and are size capped.

Every user-derived string in a refusal (a logical name, a file path) goes through `errors.printable`, so a hostile key
cannot put an escape sequence or a forged line on the terminal; a file nested deeper than the JSON parser allows is
refused like any other malformed file (never a traceback).

Rules: IMG-1 the lock is malformed; IMG-2 the config is malformed; IMG-3 the config file is not trustworthy (symlink,
owner, mode, type); IMG-4 an unknown logical image. IMG-5 .. IMG-9 are in `cli_image` and `ainative.image_provenance`.
"""

from __future__ import annotations

import contextlib
import datetime
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import Problem, Refused, printable

LOCK_SCHEMA = "gatedtrain.image-lock"
CONFIG_SCHEMA = "gatedtrain.image-config"
SCHEMA_VERSION = 1
MAINTAINER = "maintainer-built"
USER = "user-built"
MAX_FILE_BYTES = 256 * 1024         # a lock or config is a short list; a larger file is not one of ours
MAX_PER_IMAGE = 64                  # accepted IDs per logical image
ID_RE = re.compile(r"sha256:[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")        # [0-9], not \d: \d also matches non-ASCII digits
LOCK_SUFFIX = ".lock"               # the advisory-lock file beside the owner's config (`locked_config`)
MAX_TEXT = 512                      # a free-text field (note, recipe name, ...) is one short line
NAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]*(/[a-z0-9][a-z0-9._-]*)*")

_ENTRY_KEYS = frozenset({"id", "origin", "recipe", "base_id", "built_utc", "added_by", "note"})
_RECIPE_KEYS = frozenset({"name", "source", "commit", "containerfile_sha256", "build"})
LOCK_PATH = Path(__file__).resolve().parent / "data" / "image-lock.json"


@dataclass(frozen=True)
class Accepted:
    logical: str
    id: str
    origin: str
    recipe: dict
    base_id: str | None
    built_utc: str
    added_by: str
    note: str | None
    source: str                       # "lock" | "owner_config"


def is_text_line(v: Any, *, maximum: int = MAX_TEXT) -> bool:
    """A string that is safe to print into `report.md` and CLI output: 1..`maximum` characters, one line, every
    character printable (no newline, carriage return, escape, other control, bidi or line-separator character)."""
    return isinstance(v, str) and 0 < len(v) <= maximum and v.isprintable()


def _refuse(rule: str, field: str, message: str, fix: str) -> Refused:
    return Refused([Problem(rule, field, message, fix)])


def user_config_path() -> Path:
    """`$XDG_CONFIG_HOME/gatedtrain/images.json`, default `~/.config/gatedtrain/images.json`. The environment names
    only WHERE the owner's file is (the owner's own account); it never supplies accepted IDs."""
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base and os.path.isabs(base) else Path(os.environ.get("HOME") or os.path.expanduser("~")) / ".config"
    return root / "gatedtrain" / "images.json"


def _no_dup_keys(pairs: list) -> dict:
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate key {k!r}")
        out[k] = v
    return out


def _parse(data: bytes, rule: str, field: str) -> Any:
    if len(data) > MAX_FILE_BYTES:
        raise _refuse(rule, field, f"the file is larger than {MAX_FILE_BYTES} bytes", "it is not an image lock; restore it")
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=_no_dup_keys)
    except RecursionError:                      # nesting under the size cap can still exceed the parser's depth
        raise _refuse(rule, field, "the file is nested too deeply for JSON", "restore the file") from None
    except (UnicodeDecodeError, ValueError) as exc:
        raise _refuse(rule, field, f"the file is not valid JSON without duplicate keys ({type(exc).__name__})",
                      "restore the file") from None


def _exact(obj: Any, keys: frozenset, rule: str, field: str) -> dict:
    if not isinstance(obj, dict):
        raise _refuse(rule, field, "is not an object", "restore the file")
    if set(obj) != keys:
        extra, missing = sorted(set(obj) - keys), sorted(keys - set(obj))
        raise _refuse(rule, field, f"unknown fields {extra} / missing fields {missing}",
                      "the file must have exactly the documented fields (schema_version 1)")
    return obj


def _is_date(v: str) -> bool:
    """YYYY-MM-DD in ASCII digits AND a day that exists (2026-02-30 is not one)."""
    if not DATE_RE.fullmatch(v):
        return False
    try:
        datetime.date.fromisoformat(v)
    except ValueError:
        return False
    return True


def _check_entry(e: Any, origin: str, rule: str, field: str) -> None:
    _exact(e, _ENTRY_KEYS, rule, field)
    if not (isinstance(e["id"], str) and ID_RE.fullmatch(e["id"])):
        raise _refuse(rule, field + ".id", "is not sha256:<64 lowercase hex>", "use the image ID `docker image inspect` prints")
    if e["origin"] != origin:
        raise _refuse(rule, field + ".origin", f"is {e['origin']!r}; this file may only hold {origin!r}",
                      "a lock holds maintainer-built images, the owner's config user-built ones")
    r = _exact(e["recipe"], _RECIPE_KEYS, rule, field + ".recipe")
    if not is_text_line(r["name"]):
        raise _refuse(rule, field + ".recipe.name", "must be a non-empty single printable line", "name the recipe")
    for k in ("source", "build"):
        if r[k] is not None and not is_text_line(r[k]):
            raise _refuse(rule, f"{field}.recipe.{k}", "must be a single printable line or null", "fix the value")
    if r["commit"] is not None and not (isinstance(r["commit"], str) and COMMIT_RE.fullmatch(r["commit"])):
        raise _refuse(rule, field + ".recipe.commit", "is not 40 lowercase hex digits or null", "give the full commit")
    if r["containerfile_sha256"] is not None and not (isinstance(r["containerfile_sha256"], str)
                                                       and ID_RE.fullmatch(r["containerfile_sha256"])):
        raise _refuse(rule, field + ".recipe.containerfile_sha256", "is not sha256:<64 hex> or null", "fix the value")
    if origin == USER and r["commit"] is None and r["containerfile_sha256"] is None:
        raise _refuse(rule, field + ".recipe", "a user-built image needs a recipe commit or a Containerfile hash",
                      "pass --recipe-commit or --containerfile-sha256")
    if e["base_id"] is not None and not (isinstance(e["base_id"], str) and ID_RE.fullmatch(e["base_id"])):
        raise _refuse(rule, field + ".base_id", "is not sha256:<64 hex> or null", "fix the value")
    if not (isinstance(e["built_utc"], str) and _is_date(e["built_utc"])):
        raise _refuse(rule, field + ".built_utc", "is not a real YYYY-MM-DD date", "fix the value")
    if not is_text_line(e["added_by"]):
        raise _refuse(rule, field + ".added_by", "must be a non-empty single printable line", "say who or what recorded it")
    if e["note"] is not None and not is_text_line(e["note"]):
        raise _refuse(rule, field + ".note", "must be a single printable line or null", "fix the value")


def _check_doc(doc: Any, schema: str, origin: str, rule: str, extra: frozenset, known: set | None,
               lock_ids: set | None = None) -> None:
    _exact(doc, frozenset({"schema", "schema_version", "images"}), rule, "file")
    if doc["schema"] != schema:
        raise _refuse(rule, "schema", f"is {doc['schema']!r}, expected {schema!r}", "this is not that file")
    v = doc["schema_version"]
    if not (isinstance(v, int) and not isinstance(v, bool) and v == SCHEMA_VERSION):
        raise _refuse(rule, "schema_version", f"is {v!r}; this gatedtrain reads exactly version {SCHEMA_VERSION}",
                      "use the gatedtrain release that wrote the file; there is no silent up- or downgrade")
    if not isinstance(doc["images"], dict):
        raise _refuse(rule, "images", "is not an object", "restore the file")
    seen: dict[str, str] = {}
    for logical, body in doc["images"].items():
        f = f"images.{printable(logical) if isinstance(logical, str) else logical}"
        if not (isinstance(logical, str) and NAME_RE.fullmatch(logical)):
            raise _refuse(rule, f, "is not a logical image name", "use names such as nemo-rl/tierb")
        if known is not None and logical not in known:
            raise _refuse(rule, f, "names a logical image the shipped lock does not have; the config cannot invent one",
                          "add only to logical images listed by `gatedtrain image list`")
        _exact(body, frozenset({"accepted"}) | extra, rule, f)
        acc = body["accepted"]
        if not isinstance(acc, list) or len(acc) > MAX_PER_IMAGE:
            raise _refuse(rule, f + ".accepted", f"is not a list of at most {MAX_PER_IMAGE} entries", "trim the list")
        for i, e in enumerate(acc):
            _check_entry(e, origin, rule, f"{f}.accepted[{i}]")
            if lock_ids is not None and e["id"] in lock_ids:
                raise _refuse(rule, f"{f}.accepted[{i}].id", "is listed in the shipped lock; the owner's config cannot "
                              "hold an ID the lock names (under any logical image)", "remove the entry")
            if e["id"] in seen:
                raise _refuse(rule, f"{f}.accepted[{i}].id", f"is also listed under {seen[e['id']]}",
                              "one ID belongs to one logical image")
            seen[e["id"]] = logical
        if "purpose" in extra and not is_text_line(body["purpose"]):
            raise _refuse(rule, f + ".purpose", "must be a non-empty single printable line", "describe the image")


def _lock_ids(lock: dict) -> set:
    return {e["id"] for v in lock["images"].values() for e in v["accepted"]}


def load_lock_with_sha(path: str | os.PathLike | None = None) -> tuple[dict, str]:
    """The shipped lock (IMG-1), validated, and the `sha256:<hex>` of the SAME bytes that were parsed (one read, so a
    file replaced between a parse and a hash cannot make the two disagree). `path` is for tests."""
    p = Path(path) if path is not None else LOCK_PATH
    try:
        data = p.read_bytes()
    except OSError as exc:
        raise _refuse("IMG-1", str(p), f"cannot read the image lock ({type(exc).__name__})", "reinstall gatedtrain") from None
    doc = _parse(data, "IMG-1", str(p))
    _check_doc(doc, LOCK_SCHEMA, MAINTAINER, "IMG-1", frozenset({"purpose"}), None)
    return doc, "sha256:" + hashlib.sha256(data).hexdigest()


def load_lock(path: str | os.PathLike | None = None) -> dict:
    """The shipped lock (IMG-1), validated; `path` is for tests."""
    return load_lock_with_sha(path)[0]


def lock_sha256() -> str:
    """`sha256:<hex>` of the shipped lock file's bytes (validated first): the keeper puts it in the signed report so
    a verifier can compare it with a trusted copy. A caller that also uses the lock's content takes both from
    `load_lock_with_sha`."""
    return load_lock_with_sha()[1]


def _read_config_bytes(p: Path, uid: int) -> bytes | None:
    try:
        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _refuse("IMG-3", str(p), f"cannot open the owner's image config without following links "
                      f"({type(exc).__name__})", "make it a regular file, not a symlink") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise _refuse("IMG-3", str(p), "is not a regular file", "replace it with a regular file")
        if st.st_uid != uid:
            raise _refuse("IMG-3", str(p), "is not owned by the running user", "the file must be the owner's own")
        if st.st_mode & 0o022:
            raise _refuse("IMG-3", str(p), "is writable by group or others", f"chmod go-w {p}")
        with os.fdopen(fd, "rb", closefd=False) as fh:
            return fh.read(MAX_FILE_BYTES + 1)
    finally:
        os.close(fd)


def empty_config() -> dict:
    return {"schema": CONFIG_SCHEMA, "schema_version": SCHEMA_VERSION, "images": {}}


def load_user_config(path: str | os.PathLike | None = None, *, uid: int | None = None, lock: dict | None = None) -> dict:
    """The owner's config (IMG-2, IMG-3), validated against `lock` (default: the shipped one). A missing file is an
    empty config; an untrustworthy or malformed file is refused, never ignored."""
    p = Path(path) if path is not None else user_config_path()
    data = _read_config_bytes(p, os.getuid() if uid is None else uid)
    if data is None:
        return empty_config()
    doc = _parse(data, "IMG-2", str(p))
    lk = lock if lock is not None else load_lock()
    _check_doc(doc, CONFIG_SCHEMA, USER, "IMG-2", frozenset(), set(lk["images"]), _lock_ids(lk))
    return doc


def validate_config(doc: Any, lock: dict | None = None) -> dict:
    """Validate an in-memory config document the way `load_user_config` validates the file (IMG-2)."""
    lk = lock if lock is not None else load_lock()
    _check_doc(doc, CONFIG_SCHEMA, USER, "IMG-2", frozenset(), set(lk["images"]), _lock_ids(lk))
    return doc


def _tmp_name(p: Path) -> Path:
    """A temp file name beside `p` that nobody can predict (random suffix; the file is also opened O_EXCL)."""
    return p.with_name(f".{p.name}.{secrets.token_hex(8)}.tmp")


@contextlib.contextmanager
def locked_config(path: str | os.PathLike | None = None) -> Iterator[Path]:
    """Hold an advisory `flock` (a `<config>.lock` file beside the config) while the caller reads, changes and writes
    the owner's config, so two `gatedtrain image add` runs cannot lose each other's entry. Advisory: a program that
    ignores it (an agent writing the file itself) is not stopped, and the config is not a boundary anyway."""
    p = Path(path) if path is not None else user_config_path()
    p.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(str(p) + LOCK_SUFFIX, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield p
    finally:
        os.close(fd)                                  # closing releases the lock


def write_user_config(doc: dict, path: str | os.PathLike | None = None, *, lock: dict | None = None) -> Path:
    """Write the owner's config atomically (unpredictable tmp name opened O_EXCL, rename, file 0600, a new directory
    0700); the document is validated first. Only `gatedtrain image add` calls this: the owner's explicit, recorded
    act, inside `locked_config`."""
    validate_config(doc, lock)
    p = Path(path) if path is not None else user_config_path()
    p.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = _tmp_name(p)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(doc, indent=2, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return p


def _accepted(logical: str, e: dict, source: str) -> Accepted:
    return Accepted(logical, e["id"], e["origin"], dict(e["recipe"]), e["base_id"], e["built_utc"], e["added_by"],
                    e["note"], source)


def _unknown(logical: str) -> Refused:
    return _refuse("IMG-4", "logical", f"{logical!r} is not a logical image of the shipped lock",
                   "list them with `gatedtrain image list`")


def resolve(logical: str, *, lock: dict | None = None, config: dict | None | bool = False) -> list[Accepted]:
    """Accepted images of `logical`: lock entries, then config additions in file order (the config never holds a lock
    ID, so there is no overlap; a hand-made `config` document that did is collapsed keeping the lock's provenance). `config=False` (default) loads the owner's file; `None` means no config."""
    lk = lock if lock is not None else load_lock()
    if logical not in lk["images"]:
        raise _unknown(logical)
    cfg = load_user_config(lock=lk) if config is False else config
    out = [_accepted(logical, e, "lock") for e in lk["images"][logical]["accepted"]]
    have = {a.id for a in out}
    if cfg:
        for e in cfg.get("images", {}).get(logical, {}).get("accepted", []):
            if e["id"] not in have:
                out.append(_accepted(logical, e, "owner_config"))
                have.add(e["id"])
    return out


def accepted_ids(logical: str, *, lock: dict | None = None, config: dict | None | bool = False) -> list[str]:
    return [a.id for a in resolve(logical, lock=lock, config=config)]


def maintainer_ids(logical: str, *, lock: dict | None = None) -> list[str]:
    """The IDs of the shipped lock only (what the keeper itself accepts)."""
    return accepted_ids(logical, lock=lock, config=None)


def in_maintainer_lock(image_id: str, *, lock: dict | None = None) -> bool:
    lk = lock if lock is not None else load_lock()
    return any(e["id"] == image_id for v in lk["images"].values() for e in v["accepted"])


def provenance(image_id: str, *, lock: dict | None = None, config: dict | None | bool = False) -> dict:
    """Where an image ID comes from: `{logical, id, origin, recipe, in_maintainer_lock, source}`; an ID in neither
    file is `source: "unlisted"` with null logical, origin and recipe."""
    lk = lock if lock is not None else load_lock()
    for logical in lk["images"]:
        for a in resolve(logical, lock=lk, config=config):
            if a.id == image_id:
                return {"logical": logical, "id": image_id, "origin": a.origin, "recipe": a.recipe,
                        "in_maintainer_lock": a.source == "lock", "source": a.source}
    return {"logical": None, "id": image_id, "origin": None, "recipe": None, "in_maintainer_lock": False,
            "source": "unlisted"}
