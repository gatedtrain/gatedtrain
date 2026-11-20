"""AN-5: the trust interfaces and their step 1 in-process stubs.

`TrustBackend` holds the sealed items and the preregistrations; `AlphaLedger` holds the significance budget. The
stages talk only to these interfaces, so a keeper daemon in step 2 replaces the stubs without changing a stage.
The step 1 stubs run in the operator process: the operator and every stage can read the sealed items and read and
rewrite the trust records and the attempt log, and `sealed_guarantee` says so.

Both stubs persist their state under `<root>/trust/` (sealed items, registrations, `ledger.jsonl`), written by temp
file plus rename and never through a symbolic link, so a new process sees the same state. The step 1 store is
plain files the operator can read; the keeper replaces it in step 2.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ..canonical import canonical_json, sha256_bytes
from ..errors import Problem, Refused
from .project import Project, _refuse_links, _write_atomic

UNSEALED_GUARANTEE = ("unsealed (step 1): the operator process can read the sealed items, and every stage can read "
                      "and rewrite the trust records and the attempt log")
# A spent test: the alpha is per project in step 1, so a new project on the same data would get a fresh alpha with
# the same sealed split; the fix asks for new sealed data instead (step 2 keys the budget by the sealed set).
NEW_TEST_FIX = "this project's sealed test is used; a new test needs new sealed data"


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

    def latest_registration(self) -> str | None: ...              # the hash the last `register` returned

    def claim_sealed_evaluation(self, candidate_sha256: str, registration_hash: str) -> dict: ...  # the claim on record

    def sealed_evaluation(self) -> dict | None: ...               # the claim on record, or None


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


def _refuse_sealed_claim() -> Refused:
    return Refused([Problem("AN-8", "trust/sealed_evaluation", "the sealed evaluation claim is missing or damaged",
                            NEW_TEST_FIX)])


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
    LATEST = "trust/registrations/latest"
    SEALED_EVALUATION = "trust/sealed_evaluation"

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
        _write_trust_file(self.project, self.LATEST, digest.encode("ascii"))
        return digest

    def latest_registration(self) -> str | None:
        data = _read_trust_file(self.project, self.LATEST)
        if data is None:
            return None
        digest = data.decode("ascii", "replace").strip()
        _hex(digest)
        return digest

    def claim_sealed_evaluation(self, candidate_sha256: str, registration_hash: str) -> dict:
        """Claim the project's one sealed evaluation for this candidate. The claim is created once, complete, by a
        hard link that fails when it exists (no half-written claim); the claim on record is returned, so the caller
        sees an earlier claim for another candidate and refuses."""
        rel = self.SEALED_EVALUATION
        _refuse_links(self.project.root, rel)
        path = self.project.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        claim = {"candidate_model_sha256": candidate_sha256, "registration_hash": registration_hash}
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".sealed_evaluation.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(canonical_json(claim))
                f.flush()
                os.fsync(f.fileno())
            os.link(tmp, path)
        except FileExistsError:
            pass
        finally:
            os.unlink(tmp)
        held = self.sealed_evaluation()
        if held is None:
            raise _refuse_sealed_claim()
        return held

    def sealed_evaluation(self) -> dict | None:
        data = _read_trust_file(self.project, self.SEALED_EVALUATION)
        if data is None:
            return None
        try:
            claim = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            claim = None
        if not (isinstance(claim, dict) and isinstance(claim.get("candidate_model_sha256"), str)
                and isinstance(claim.get("registration_hash"), str)):
            raise _refuse_sealed_claim()
        return claim

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
    CLAIM = "trust/spent"

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
                                           f"the ledger was damaged; {NEW_TEST_FIX}")])
                rows.append(row)
        return rows

    def _append(self, row: dict) -> None:
        """One `O_APPEND` write per line, never through a link."""
        _refuse_links(self.project.root, self.REL)
        path = self.project.root / self.REL
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o644)
        try:
            os.write(fd, canonical_json(row) + b"\n")
            os.fsync(fd)
        finally:
            os.close(fd)

    def _claim(self) -> bool:
        """Claim the single spend atomically (`O_EXCL`); False when another spend already holds it."""
        _refuse_links(self.project.root, self.CLAIM)
        path = self.project.root / self.CLAIM
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644))
        except FileExistsError:
            return False
        return True

    def _claimed(self) -> bool:
        _refuse_links(self.project.root, self.CLAIM)
        return (self.project.root / self.CLAIM).exists()

    def available(self) -> float:
        return 0.0 if self._claimed() or any(r.get("event") == "spend" for r in self._lines()) else self.alpha

    def spend(self, attempt_id: str) -> float:
        spent = Refused([Problem("AN-9", "gate", "the alpha budget is spent: step 1 allows one gate test per project",
                                 NEW_TEST_FIX)])
        if any(r.get("event") == "spend" for r in self._lines()) or not self._claim():
            raise spent
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
