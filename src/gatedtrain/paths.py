"""Lenient path resolution that behaves the same on every supported Python.

`os.path.realpath` and `Path.resolve()` without `strict` raise `PermissionError` on Python 3.12 when a component is
a symbolic link the user may not read (`/proc/1/root` of another user), where Python 3.13 keeps the component and
goes on. Paths that come from requests, launch records and run outputs are resolved with these functions, so a
check refuses such a path instead of crashing (docs/findings.md, "Python 3.12: lenient `realpath`").

`realpath` is ported from CPython 3.13's `Lib/posixpath.py`, Copyright (c) 2001-2024 Python Software Foundation,
under the PSF License Version 2 (THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

import os
import stat
from pathlib import Path


def realpath(p: str | os.PathLike[str]) -> str:
    """Lenient `realpath` (the path need not exist): Python 3.13's `posixpath.realpath(p, strict=False)` for `str`
    paths, ported line for line. A component that cannot be inspected, or a symbolic link that cannot be read, is
    kept as it is and resolution goes on; a symbolic link loop is kept as it is and the rest is resolved after it."""
    filename = os.fspath(p)
    rest = filename.split("/")[::-1]              # parts still to resolve; None marks a resolved link (popped next)
    part_count = len(rest)
    path = "/" if filename.startswith("/") else os.getcwd()
    seen: dict[str, str | None] = {}              # link -> resolved target, None while unresolved (a loop)
    while part_count:
        name = rest.pop()
        if name is None:
            seen[rest.pop()] = path
            continue
        part_count -= 1
        if not name or name == ".":
            continue
        if name == "..":
            path = path[:path.rindex("/")] or "/"
            continue
        newpath = path + name if path == "/" else path + "/" + name
        try:
            if not stat.S_ISLNK(os.lstat(newpath).st_mode):
                path = newpath
                continue
            if newpath in seen:
                path = seen[newpath]
                if path is not None:
                    continue
                path = newpath
                continue
            target = os.readlink(newpath)
        except OSError:
            pass
        else:
            if target.startswith("/"):
                path = "/"
            seen[newpath] = None
            rest.append(newpath)
            rest.append(None)
            target_parts = target.split("/")[::-1]
            rest.extend(target_parts)
            part_count += len(target_parts)
            continue
        path = newpath
    return path


def resolve(p: str | os.PathLike[str]) -> Path:
    """`Path(p).resolve()` (not strict) by `realpath`."""
    return Path(realpath(p))
