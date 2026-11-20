"""AN-5: the trust interfaces and their step 1 in-process stubs.

`TrustBackend` holds the sealed items and the preregistrations; `AlphaLedger` holds the significance budget. The
stages talk only to these interfaces, so a keeper daemon in step 2 replaces the stubs without changing a stage.
The step 1 stubs run in the operator process: they keep the sealed items in memory and say so in
`sealed_guarantee`.

Both stubs persist their state under `<root>/trust/` (sealed items, registrations, `ledger.jsonl`), written by temp
file plus rename and never through a symbolic link, so a new process sees the same state. The step 1 store is
plain files the operator can read; the keeper replaces it in step 2.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ..canonical import canonical_json, sha256_bytes
from ..errors import Problem, Refused
from .project import Project, _refuse_links, _write_atomic

UNSEALED_GUARANTEE = "unsealed (step 1): the operator process can read the sealed items"


@dataclass(frozen=True)
class SealedRef:
    sealed_id: str          # sha256 of the canonical JSON of the sealed items


class TrustBackend(Protocol):
    sealed_guarantee: str

    def seal(self, items: list[dict]) -> SealedRef: ...

    def sealed_summary(self, ref: SealedRef) -> dict: ...         # n, manifest_sha256, label_counts

    def run_sealed(self, ref: SealedRef, generate: Callable[[dict], dict]) -> list[dict]: ...

    def register(self, registration: dict) -> str: ...            # returns the registration hash

    def registration(self, hash: str) -> dict: ...


class AlphaLedger(Protocol):
    def available(self) -> float: ...

    def spend(self, attempt_id: str) -> float: ...                # the alpha for this test

    def record(self, attempt_id: str, passed: bool) -> None: ...

    def entries(self) -> list[dict]: ...


def _refuse_trust(field: str, message: str, fix: str) -> Refused:
    return Refused([Problem("AN-5", field, message, fix)])


_HEX = re.compile(r"^[0-9a-f]{64}$")


def _hex(digest: object) -> str:
    """File-name form of a `sha256:<hex>` value (no colon in any file name)."""
    h = digest.removeprefix("sha256:") if isinstance(digest, str) else ""
    if not _HEX.match(h):
        raise _refuse_trust("hash", f"{digest!r} is not a sha256 hash", "pass a value returned by seal or register")
    return h


def _read_trust_file(project: Project, rel: str) -> bytes | None:
    _refuse_links(project.root, rel)
    try:
        return (project.root / rel).read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise Refused([Problem("AN-2", rel, f"cannot read the trust record ({type(exc).__name__})",
                               "check the file permissions")]) from None


def _write_trust_file(project: Project, rel: str, data: bytes) -> None:
    _refuse_links(project.root, rel)
    _write_atomic(project.root / rel, data)


class InProcessTrust:
    """Step 1 stub: sealed items and registrations are files in the project's `trust/` directory."""

    sealed_guarantee = UNSEALED_GUARANTEE

    def __init__(self, project: Project) -> None:
        self.project = project

    def seal(self, items: list[dict]) -> SealedRef:
        try:
            data = canonical_json(items)
        except ValueError as exc:
            raise _refuse_trust("seal", f"the items are not canonical JSON: {exc}", "use plain JSON values") \
                from None
        ref = SealedRef(sha256_bytes(data))
        _write_trust_file(self.project, f"trust/sealed/{_hex(ref.sealed_id)}.json", data)
        return ref

    def _items(self, ref: SealedRef) -> list[dict]:
        data = _read_trust_file(self.project, f"trust/sealed/{_hex(ref.sealed_id)}.json")
        if data is None:
            raise _refuse_trust("sealed", "unknown sealed reference in this project", "seal the items first")
        if sha256_bytes(data) != ref.sealed_id:
            raise _refuse_trust("sealed", "the sealed record no longer matches its hash", "seal the items again")
        return json.loads(data.decode("utf-8"))

    def sealed_summary(self, ref: SealedRef) -> dict:
        items = self._items(ref)
        counts: dict[str, int] = {}
        for item in items:
            label = item.get("label")
            if label is not None:
                counts[str(label)] = counts.get(str(label), 0) + 1
        return {"n": len(items), "manifest_sha256": ref.sealed_id, "label_counts": counts}

    def run_sealed(self, ref: SealedRef, generate: Callable[[dict], dict]) -> list[dict]:
        records = []
        for item in self._items(ref):
            rec = generate(dict(item))
            if not isinstance(rec, dict):
                raise _refuse_trust("run_sealed", "generate must return a record (dict) per item",
                                    "return one dict per item")
            records.append(rec)
        return records

    def register(self, registration: dict) -> str:
        try:
            data = canonical_json(registration)
        except ValueError as exc:
            raise _refuse_trust("register", f"the registration is not canonical JSON: {exc}",
                                "use plain JSON values") from None
        digest = sha256_bytes(data)
        _write_trust_file(self.project, f"trust/registrations/{_hex(digest)}.json", data)
        return digest

    def registration(self, hash: str) -> dict:
        rel = f"trust/registrations/{_hex(hash)}.json"
        data = _read_trust_file(self.project, rel)
        if data is None:
            raise _refuse_trust(rel, "no such registration", "pass a hash returned by register")
        if sha256_bytes(data) != hash:
            raise _refuse_trust(rel, "the registration no longer matches its hash", "re-register")
        return json.loads(data.decode("utf-8"))


class FixedAlphaLedger:
    """Step 1 stub: one test at a fixed alpha, then the budget is empty (rule AN-9). The ledger is
    `trust/ledger.jsonl`: one line per spend and per record."""

    REL = "trust/ledger.jsonl"

    def __init__(self, project: Project, alpha: float = 0.05) -> None:
        self.project = project
        self.alpha = alpha

    def _lines(self) -> list[dict]:
        raw = _read_trust_file(self.project, self.REL)
        rows = []
        for line in (raw or b"").split(b"\n"):
            if line:
                try:
                    row = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, ValueError):
                    row = None
                if not isinstance(row, dict):
                    raise Refused([Problem("AN-9", self.REL, "the ledger has a line that is not a JSON object",
                                           "the ledger was damaged; start a new project")])
                rows.append(row)
        return rows

    def _append(self, row: dict) -> None:
        raw = _read_trust_file(self.project, self.REL) or b""
        _write_trust_file(self.project, self.REL, raw + canonical_json(row) + b"\n")

    def available(self) -> float:
        return 0.0 if any(r.get("event") == "spend" for r in self._lines()) else self.alpha

    def spend(self, attempt_id: str) -> float:
        if any(r.get("event") == "spend" for r in self._lines()):
            raise Refused([Problem("AN-9", "gate", "the alpha budget is spent: step 1 allows one gate test per project",
                                   "start a new project for another test")])
        self._append({"event": "spend", "attempt_id": attempt_id, "alpha": self.alpha})
        return self.alpha

    def record(self, attempt_id: str, passed: bool) -> None:
        if not any(r.get("event") == "spend" and r.get("attempt_id") == attempt_id for r in self._lines()):
            raise Refused([Problem("AN-9", "gate", f"attempt {attempt_id!r} never spent alpha",
                                   "spend alpha before recording the outcome")])
        self._append({"event": "record", "attempt_id": attempt_id, "passed": bool(passed)})

    def entries(self) -> list[dict]:
        out: dict[str, dict] = {}
        for r in self._lines():
            if r.get("event") == "spend":
                out[r["attempt_id"]] = {"attempt_id": r["attempt_id"], "alpha": r["alpha"], "passed": None}
            elif r.get("event") == "record" and r.get("attempt_id") in out:
                out[r["attempt_id"]]["passed"] = r["passed"]
        return list(out.values())
