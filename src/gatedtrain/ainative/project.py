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


class Project:
    def __init__(self, root: Path, doc: dict[str, Any]) -> None:
        self.root = Path(root)
        self.goal: str = doc["goal"]
        self.data: dict[str, str] = dict(doc["data"])
        self.pipeline: str = doc["pipeline"]
        self.budget: dict[str, Any] = dict(doc["budget"])
        self.created_at: str = doc["created_at"]

    @classmethod
    def load(cls, root: Path) -> "Project":
        path = Path(root) / "project.yaml"
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
        return {"schema": PROJECT_SCHEMA, "schema_version": SCHEMA_VERSION, "goal": self.goal, "data": self.data,
                "pipeline": self.pipeline, "budget": self.budget, "created_at": self.created_at}

    def write_artifact(self, stage: str, name: str, obj: dict | list, schema: str) -> ArtifactRef:
        _check_name("stage", stage)
        _check_name("artifact", name)
        try:
            data = canonical_json(obj)
        except ValueError as exc:
            raise _refuse(f"{stage}/{name}", f"the record is not canonical JSON: {exc}",
                          "use plain JSON values") from None
        rel = f"stages/{stage}/{name}.json"
        _write_atomic(self.root / rel, data)
        return ArtifactRef(stage, name, schema, sha256_bytes(data), rel)

    def read_artifact(self, ref: ArtifactRef) -> dict | list:
        _check_name("stage", ref.stage)
        _check_name("artifact", ref.name)
        if ref.path != f"stages/{ref.stage}/{ref.name}.json":
            raise _refuse(ref.path, "the reference path does not match its stage and name",
                          "use the reference returned by write_artifact")
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
        row = {"schema": ATTEMPT_SCHEMA, "schema_version": SCHEMA_VERSION, **event}
        line = canonical_json(row) + b"\n"
        fd = os.open(self.root / "attempts.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line)
            os.fsync(fd)
        finally:
            os.close(fd)


def create_project(root: Path, *, goal: str, data: Path, pipeline: str, budget: dict) -> Project:
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
    _write_atomic(root / "project.yaml", yaml.safe_dump(doc, sort_keys=False).encode("utf-8"))
    return Project(root, doc)
