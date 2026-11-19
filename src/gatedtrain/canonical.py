"""Canonical JSON and content-addressed manifests (MVP-SPEC section 2.8).

Follows the project's format conventions (rule ids C1-C6 for canonical JSON, H1-H8 for manifests):

- canonical JSON is RFC 8785 (JCS) with rules C1-C4: UTF-8, no
  whitespace, keys in UTF-16 code-unit order, minimal string escaping,
  ECMAScript number form; NaN, Infinity and integers outside +/-(2^53 - 1)
  are rejected; sets serialize as sorted arrays (C6);
- a manifest lists `{path, sha256, size}` for every regular file under the
  root (H1-H2), with '/'-joined NFC relative paths (H3-H4), rejects case-fold
  collisions (H5), sorts by UTF-8 bytes (H6), and hashes the JCS bytes of the
  entry array (H7). The kind is recorded as `p2_filelist_sha256` (H8).
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from decimal import Decimal
from pathlib import Path
from typing import Any

MAX_SAFE_INT = 2**53 - 1
MANIFEST_HASH_KIND = "p2_filelist_sha256"
SCHEMA_VERSION = "1.0"   # format rule 2A, provisional initial value


_LINE_END = re.compile(r"\r\n|\r|\n")


def text_lines(text: str) -> list[str]:
    """The lines of a JSONL or line-oriented text file: `text.splitlines()` restricted to the line ends `\\n`,
    `\\r\\n` and `\\r` (EXT-90). `str.splitlines` also breaks on U+2028, U+2029, U+0085, \\x0b, \\x0c and \\x1c to
    \\x1e, which JSON allows raw inside a string, so it cuts a valid row in two; Python's file iteration
    (`ledger._count_lines`) and JSON lines loaders end a line at `\\n` only. As with `splitlines`, a final line
    end adds no empty line."""
    lines = _LINE_END.split(text)
    if lines[-1] == "":
        lines.pop()
    return lines


def _check_str(s: str) -> None:
    for ch in s:
        if 0xD800 <= ord(ch) <= 0xDFFF:
            raise ValueError("lone surrogate in string (canonical-JSON rule C3)")


def _encode_str(s: str) -> str:
    _check_str(s)
    out = ['"']
    for ch in s:
        o = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif o < 0x20:
            out.append({"\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r"}.get(ch, f"\\u{o:04x}"))
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _encode_float(x: float) -> str:
    """ECMAScript Number.prototype.toString form (RFC 8785 section 3.2.2.3)."""
    if math.isnan(x) or math.isinf(x):
        raise ValueError("NaN and Infinity are not allowed in canonical JSON (canonical-JSON rule C4)")
    if x == 0:
        return "0"                                      # -0 is written 0 (C4)
    sign = "-" if x < 0 else ""
    tup = Decimal(repr(float(abs(x)))).normalize().as_tuple()  # shortest round-trip digits
    digits = "".join(map(str, tup.digits))
    k = len(digits)
    n = tup.exponent + k                                # value = 0.digits * 10^n
    if k <= n <= 21:
        body = digits + "0" * (n - k)
    elif 0 < n <= 21:
        body = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        body = "0." + "0" * (-n) + digits
    else:
        e = n - 1
        mant = digits[0] + ("." + digits[1:] if k > 1 else "")
        body = f"{mant}e{'+' if e > 0 else '-'}{abs(e)}"
    return sign + body


def _encode(obj: Any) -> str:
    if obj is None:
        return "null"
    if obj is True:
        return "true"
    if obj is False:
        return "false"
    if isinstance(obj, int):
        if abs(obj) > MAX_SAFE_INT:
            raise ValueError(f"integer {obj} outside +/-(2^53 - 1) (canonical-JSON rule C4)")
        return str(obj)
    if isinstance(obj, float):
        return _encode_float(obj)
    if isinstance(obj, str):
        return _encode_str(obj)
    if isinstance(obj, dict):
        items = []
        for k in obj:
            if not isinstance(k, str):
                raise ValueError(f"object key {k!r} is not a string")
            _check_str(k)
            if any(ord(ch) > 0xFFFF for ch in k):
                raise ValueError(f"object key {k!r} has a supplementary-plane character (canonical-JSON rule C2)")
        for k in sorted(obj, key=lambda s: s.encode("utf-16-be")):   # C2: UTF-16 code units
            items.append(_encode_str(k) + ":" + _encode(obj[k]))
        return "{" + ",".join(items) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_encode(v) for v in obj) + "]"
    if isinstance(obj, (set, frozenset)):
        enc = sorted((_encode(v) for v in obj), key=lambda s: s.encode("utf-8"))   # C6
        return "[" + ",".join(enc) + "]"
    raise ValueError(f"type {type(obj).__name__} is not allowed in canonical JSON")


def canonical_json(obj: Any) -> bytes:
    """RFC 8785 bytes, no trailing newline (canonical-JSON rule C1)."""
    return _encode(obj).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _check_path(rel: str) -> None:
    """Manifest rule H4: reject, never rewrite."""
    if not unicodedata.is_normalized("NFC", rel):
        raise ValueError(f"manifest path is not NFC (manifest rule H4): {rel!r}")
    for seg in rel.split("/"):
        if seg in ("", ".", ".."):
            raise ValueError(f"bad path segment (manifest rule H4): {rel!r}")
    if "\\" in rel or any(ord(c) < 0x20 or ord(c) == 0x7F for c in rel):
        raise ValueError(f"bad character in manifest path (manifest rule H4): {rel!r}")


def manifest_entries(root: Path, *, exclude: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """Sorted `{path, sha256, size}` for every regular file under root (manifest rule H1-H6).

    The walk raises on a directory it cannot list (`os.walk(onerror=...)`), so an unreadable
    subtree is an error, never silently left out of the hash; an unreadable file raises OSError.
    `exclude` names directories directly under root that are left out entirely; an adapter declares them
    (tests/README.md section 11.22, EXT-162: the SD3.5 base's own `.cache/`). Default: none. An excluded name that is
    present but is a symbolic link or not a directory raises ValueError (manifest rule H2).
    """
    import os

    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"manifest root is not a directory: {root}")

    def fail(err: OSError) -> None:
        raise err

    for name in exclude:
        p = root / name
        if p.is_symlink() or (p.exists() and not p.is_dir()):
            raise ValueError(f"the excluded {p} is not a plain directory (a symbolic link or a file; manifest rule H2)")
    entries = []
    folded: dict[str, str] = {}
    top = True
    for dirpath, dirnames, filenames in os.walk(root, onerror=fail, followlinks=False):
        if top and exclude:                               # top-down walk: pruning dirnames skips the subtree
            dirnames[:] = [d for d in dirnames if d not in exclude]
        top = False
        for d in dirnames:
            if os.path.islink(os.path.join(dirpath, d)):
                raise ValueError(f"symbolic link inside a manifest root (manifest rule H2): {Path(dirpath, d)}")
        for name in filenames:
            p = Path(dirpath, name)
            if p.is_symlink():
                raise ValueError(f"symbolic link inside a manifest root (manifest rule H2): {p}")
            if not p.is_file():
                raise ValueError(f"not a regular file (manifest rule H2): {p}")
            rel = p.relative_to(root).as_posix()
            _check_path(rel)
            key = rel.casefold()
            if key in folded:
                raise ValueError(f"paths collide after case folding (manifest rule H5): {folded[key]!r}, {rel!r}")
            folded[key] = rel
            data = p.read_bytes()
            entries.append({"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)})
    entries.sort(key=lambda e: e["path"].encode("utf-8"))   # H6
    return entries


def entries_hash(entries: list[dict[str, Any]]) -> str:
    """H7: "sha256:" + hex(SHA-256(JCS(entries)))."""
    return sha256_bytes(canonical_json(entries))


def manifest_hash(root: Path) -> str:
    return entries_hash(manifest_entries(root))


def manifest_document(name: str, type_: str, root: Path, tags: tuple[str, ...],
                      entries: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The manifest document of tests/README.md section 5.1. `root` is recorded, not hashed.
    `entries` may be passed when the root was already walked (Tier B collect pre-checks)."""
    if entries is None:
        entries = manifest_entries(root)
    return {
        "schema": "gatedtrain.manifest",
        "schema_version": SCHEMA_VERSION,
        "artifact": name,
        "type": type_,
        "root": str(Path(root).resolve()),
        "entries": entries,
        "manifest_hash": entries_hash(entries),
        "manifest_hash_kind": MANIFEST_HASH_KIND,
        "tags": list(tags),
    }


def write_canonical(path: Path, obj: Any) -> bytes:
    data = canonical_json(obj)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
    return data
