"""AN-5: the trust interfaces and their step 1 in-process stubs.

`TrustBackend` holds the sealed items and the preregistrations; `AlphaLedger` holds the significance budget. The
stages talk only to these interfaces, so a keeper daemon in step 2 replaces the stubs without changing a stage.
The step 1 stubs run in the operator process: they keep the sealed items in memory and say so in
`sealed_guarantee`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ..canonical import canonical_json, sha256_bytes
from ..errors import Problem, Refused
from .project import Project, _refuse_links

REGISTRATION_SCHEMA = "gatedtrain.ainative.registration"
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


class InProcessTrust:
    """Step 1 stub: sealed items live in this process's memory; registrations are records in the project."""

    sealed_guarantee = UNSEALED_GUARANTEE

    def __init__(self, project: Project) -> None:
        self.project = project
        self._sealed: dict[str, list[dict]] = {}

    def seal(self, items: list[dict]) -> SealedRef:
        try:
            data = canonical_json(items)
        except ValueError as exc:
            raise _refuse_trust("seal", f"the items are not canonical JSON: {exc}", "use plain JSON values") \
                from None
        ref = SealedRef(sha256_bytes(data))
        self._sealed[ref.sealed_id] = json.loads(data.decode("utf-8"))
        return ref

    def _items(self, ref: SealedRef) -> list[dict]:
        try:
            return self._sealed[ref.sealed_id]
        except KeyError:
            raise _refuse_trust("sealed", "unknown sealed reference in this process",
                                "seal the items again; step 1 keeps them in memory only") from None

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
            digest = sha256_bytes(canonical_json(registration))
        except ValueError as exc:
            raise _refuse_trust("register", f"the registration is not canonical JSON: {exc}",
                                "use plain JSON values") from None
        self.project.write_artifact("registrations", digest, registration, REGISTRATION_SCHEMA)
        return digest

    def registration(self, hash: str) -> dict:
        rel = f"stages/registrations/{hash}.json"
        if not isinstance(hash, str) or not hash or "/" in hash or "\\" in hash or "\0" in hash or hash in (".", ".."):
            raise _refuse_trust("registration", f"{hash!r} is not a registration hash", "pass a hash from register")
        _refuse_links(self.project.root, rel)
        try:
            data = (self.project.root / rel).read_bytes()
        except OSError:
            raise _refuse_trust(rel, "no such registration", "pass a hash returned by register") from None
        if sha256_bytes(data) != hash:
            raise _refuse_trust(rel, "the registration no longer matches its hash", "re-register")
        return json.loads(data.decode("utf-8"))


class FixedAlphaLedger:
    """Step 1 stub: one test at a fixed alpha, then the budget is empty (rule AN-9)."""

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self._spent: list[dict] = []

    def available(self) -> float:
        return 0.0 if self._spent else self.alpha

    def spend(self, attempt_id: str) -> float:
        if self._spent:
            raise Refused([Problem("AN-9", "gate", "the alpha budget is spent: step 1 allows one gate test per project",
                                   "start a new project for another test")])
        self._spent.append({"attempt_id": attempt_id, "alpha": self.alpha, "passed": None})
        return self.alpha

    def record(self, attempt_id: str, passed: bool) -> None:
        for entry in self._spent:
            if entry["attempt_id"] == attempt_id:
                entry["passed"] = bool(passed)
                return
        raise Refused([Problem("AN-9", "gate", f"attempt {attempt_id!r} never spent alpha",
                               "spend alpha before recording the outcome")])

    def entries(self) -> list[dict]:
        return [dict(e) for e in self._spent]
