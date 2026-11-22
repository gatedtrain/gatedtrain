"""AN-2: the project store.

One goal is one project directory:

    <root>/project.yaml                     goal, data, pipeline, budget, created_at
    <root>/stages/<stage>/<name>.json       one canonical-JSON record per artifact
    <root>/attempts.jsonl                   append-only attempt log

An artifact is written to a temp file in its stage directory and renamed into place (a crash never leaves a half
record). Its `ArtifactRef` carries the sha256 of the canonical JSON bytes; `read_artifact` refuses a file whose
hash no longer matches.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..canonical import canonical_json, file_sha256, manifest_hash, sha256_bytes
from ..errors import Problem, Refused

PROJECT_SCHEMA = "gatedtrain.ainative.project"
ATTEMPT_SCHEMA = "gatedtrain.ainative.attempt"
SCHEMA_VERSION = 1


def _refuse(field: str, message: str, fix: str) -> Refused:
    return Refused([Problem("AN-2", field, message, fix)])


def _check_name(kind: str, value: object) -> str:
    if (not isinstance(value, str) or value in ("", ".", "..") or "/" in value or "\\" in value
            or "\0" in value):
        raise _refuse(kind, f"{value!r} is not a safe {kind} name (empty, '.', '..', or containing '/', '\\' or NUL)",
                      f"use a plain name for the {kind}")
    return value


@dataclass(frozen=True)
class ArtifactRef:
    stage: str
    name: str
    schema: str
    sha256: str
    path: str          # relative to the project root, '/'-joined


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_atomic(path: Path, data: bytes) -> None:
    """Temp file in the same directory, then `os.replace`; the temp file is removed on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _refuse_links(root: Path, rel: str) -> None:
    """Refuse when `stages` or `stages/<stage>` (or the file itself) under the root is a symlink."""
    cur = root
    for part in rel.split("/"):
        cur = cur / part
        if cur.is_symlink():
            raise _refuse(rel, f"{part!r} is a symbolic link; records must stay inside the project",
                          "remove the link; the project store does not follow links")


class Project:
    def __init__(self, root: Path, doc: dict[str, Any]) -> None:
        self.root = Path(root)
        self.goal: str = doc["goal"]
        self.data: dict[str, str] = dict(doc["data"])
        self.pipeline: str = doc["pipeline"]
        self.budget: dict[str, Any] = dict(doc["budget"])
        self.created_at: str = doc["created_at"]
        self.pipeline_sha256: str | None = doc.get("pipeline_sha256")

    @classmethod
    def load(cls, root: Path) -> "Project":
        path = Path(root) / "project.yaml"
        _refuse_links(Path(root), "project.yaml")
        try:
            doc = yaml.safe_load(path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise _refuse(str(path), f"cannot read the project file ({type(exc).__name__})",
                          "pass a directory made by create_project") from None
        if not isinstance(doc, dict) or doc.get("schema") != PROJECT_SCHEMA \
                or doc.get("schema_version") != SCHEMA_VERSION:
            raise _refuse(str(path), f"not a {PROJECT_SCHEMA} version {SCHEMA_VERSION} record",
                          "pass a directory made by create_project")
        missing = [k for k in ("goal", "data", "pipeline", "budget", "created_at") if k not in doc]
        if missing or not isinstance(doc["data"], dict):
            raise _refuse(str(path), f"project record is missing or malformed: {', '.join(missing) or 'data'}",
                          "recreate the project")
        return cls(Path(root), doc)

    def _doc(self) -> dict[str, Any]:
        doc = {"schema": PROJECT_SCHEMA, "schema_version": SCHEMA_VERSION, "goal": self.goal, "data": self.data,
               "pipeline": self.pipeline, "budget": self.budget, "created_at": self.created_at}
        if self.pipeline_sha256 is not None:
            doc["pipeline_sha256"] = self.pipeline_sha256
        return doc

    def write_artifact(self, stage: str, name: str, obj: dict | list, schema: str) -> ArtifactRef:
        _check_name("stage", stage)
        _check_name("artifact", name)
        try:
            data = canonical_json(obj)
        except ValueError as exc:
            raise _refuse(f"{stage}/{name}", f"the record is not canonical JSON: {exc}",
                          "use plain JSON values") from None
        rel = f"stages/{stage}/{name}.json"
        _refuse_links(self.root, rel)
        _write_atomic(self.root / rel, data)
        return ArtifactRef(stage, name, schema, sha256_bytes(data), rel)

    def read_artifact(self, ref: ArtifactRef) -> dict | list:
        _check_name("stage", ref.stage)
        _check_name("artifact", ref.name)
        if ref.path != f"stages/{ref.stage}/{ref.name}.json":
            raise _refuse(ref.path, "the reference path does not match its stage and name",
                          "use the reference returned by write_artifact")
        _refuse_links(self.root, ref.path)
        try:
            data = (self.root / ref.path).read_bytes()
        except OSError as exc:
            raise _refuse(ref.path, f"cannot read the artifact ({type(exc).__name__})",
                          "rerun the stage that writes it") from None
        got = sha256_bytes(data)
        if got != ref.sha256:
            raise _refuse(ref.path, f"sha256 mismatch: recorded {ref.sha256}, file has {got}",
                          "the record was changed after it was written; rerun the stage")
        return json.loads(data.decode("utf-8"))

    def append_attempt(self, event: dict) -> None:
        """One `O_APPEND` line in `attempts.jsonl` (`append_jsonl`)."""
        row = {**event, "schema": ATTEMPT_SCHEMA, "schema_version": SCHEMA_VERSION}   # event keys cannot override
        append_jsonl(self.root, "attempts.jsonl", row)

    def read_attempts(self) -> list[dict]:
        """The attempt log in order (`read_jsonl`)."""
        return read_jsonl(self.root, "attempts.jsonl")


def append_jsonl(root: Path, rel: str, row: dict) -> None:
    """One `O_APPEND` write per line of the log `rel` under `root`, never through a symbolic link. An unterminated last
    line (a crash in the middle of an earlier append) is not a row: `read_jsonl` ignores it, and it is cut here before
    the new line is written, so it never ends up in the middle of the log."""
    line = canonical_json(row) + b"\n"
    _refuse_links(Path(root), rel)
    fd = os.open(Path(root) / rel, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o644)
    try:
        size = os.fstat(fd).st_size
        if size and os.pread(fd, 1, size - 1) != b"\n":
            data = os.pread(fd, size, 0)
            os.ftruncate(fd, data.rfind(b"\n") + 1)
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)


def read_jsonl(root: Path, rel: str) -> list[dict]:
    """The log `rel` under `root` in order; an absent log is empty; a line that is not a JSON object is refused. An
    unterminated last line is a write cut short by a crash and is ignored (see `append_jsonl`); every other line must
    be a complete row."""
    _refuse_links(Path(root), rel)
    try:
        raw = (Path(root) / rel).read_bytes()
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise _refuse(rel, f"cannot read the log ({type(exc).__name__})", "check the file permissions") from None
    rows: list[dict] = []
    lines = raw.split(b"\n")
    lines.pop()                           # after the last newline: empty, or an unterminated (cut) write
    for n, line in enumerate(lines, 1):
        if not line:
            continue
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            row = None
        if not isinstance(row, dict):
            raise _refuse(f"{rel} line {n}", "the line is not a JSON object",
                          "the log was damaged; restore it from a copy")
        rows.append(row)
    return rows


PIPELINE_COPY = "pipeline.yaml"


def create_project(root: Path, *, goal: str, data: Path, pipeline: str, budget: dict,
                   pipeline_bytes: bytes | None = None) -> Project:
    """With `pipeline_bytes`, the pipeline is copied to `<root>/pipeline.yaml` and its sha256 recorded as
    `pipeline_sha256`; `pipeline` is then the copy's name and the machine refuses a copy that changed."""
    root = Path(root)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise _refuse(str(root), "the project root exists and is not an empty directory",
                      "choose a new or empty directory; a project is created once")
    if not isinstance(budget, dict) or not {"gpu_hours", "llm_usd"} <= budget.keys():
        raise _refuse("budget", "the budget needs 'gpu_hours' and 'llm_usd'", "pass both budget numbers")
    data = Path(data)
    try:
        digest = manifest_hash(data) if data.is_dir() else file_sha256(data)
    except (OSError, ValueError) as exc:
        raise _refuse(str(data), f"cannot hash the data ({exc})", "pass an existing readable file or directory") \
            from None
    doc = {"schema": PROJECT_SCHEMA, "schema_version": SCHEMA_VERSION, "goal": goal,
           "data": {"path": str(data), "sha256": digest}, "pipeline": pipeline,
           "budget": {"gpu_hours": budget["gpu_hours"], "llm_usd": budget["llm_usd"]}, "created_at": _utc_now()}
    root.mkdir(parents=True, exist_ok=True)
    if pipeline_bytes is not None:
        _write_atomic(root / PIPELINE_COPY, pipeline_bytes)
        doc["pipeline"], doc["pipeline_sha256"] = PIPELINE_COPY, sha256_bytes(pipeline_bytes)
    _write_atomic(root / "project.yaml", yaml.safe_dump(doc, sort_keys=False).encode("utf-8"))
    return Project(root, doc)
