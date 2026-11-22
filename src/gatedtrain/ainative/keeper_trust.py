"""AN-8 to AN-10, AN-14, AN-15: the driver side of the keeper.

`KeeperTrust` and `KeeperAlphaLedger` implement the step 1 `TrustBackend` and `AlphaLedger` interfaces as requests to
the keeper (`keeper.protocol.KeeperClient`); the stage machine is unchanged, and the trust stages branch on
`ctx.trust.mode` where the keeper does the work (it splits, scores, gates and builds the report).

A project is bound to one keeper by `bind_project`: `trust/keeper.json` (`KEEPER_BINDING`: the socket, the key
fingerprint and the project id) and `trust/keeper.pub` (`KEEPER_PUB`: the public key the keeper served when the
project was bound). Every call first asks the keeper for its key (`hello`) and refuses (AN-14) a keeper that serves
another key than the binding's, and every envelope received is verified against `trust/keeper.pub` (whose
fingerprint must be the binding's) and appended to the project's receipts (`RECEIPTS`, append-only: one
`O_APPEND | O_NOFOLLOW` write per line, as step 1's ledger; an envelope already there is not appended again). The
receipts are what the machine counts as a side effect, so a spend the keeper reports with a refusal
(`RefusedAfterSpend`) is appended before the refusal is raised and the attempt is logged `failed`.

The receipts are synced with the keeper (`sync`) after every `evaluate`, `gate` and `report`: the keeper lists every
envelope it signed for the project (`records`), and each one the receipts lack is verified and appended. A driver can
miss an envelope without any tampering: a client that gave up during an evaluation never received that attempt's claim
and grant, and the next evaluate settles the attempt in the keeper; after a retrain the identical request that would
return the stored claim is never sent.

The project copy of the key is a convenience, not a trust anchor: `gatedtrain verify` anchors on the key the keeper
serves or on a fingerprint the reader holds (AN-14).
"""

from __future__ import annotations

import base64
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..canonical import canonical_json, sha256_bytes
from ..errors import Problem, Refused
from . import signing
from .alpha_investing import PLANNING_ALPHA
from .keeper import protocol
from .keeper.protocol import KeeperClient, RefusedAfterSpend
from .project import Project, _refuse_links, _write_atomic
from .trust import SealedRef

KEEPER_BINDING = "trust/keeper.json"
KEEPER_PUB = "trust/keeper.pub"
RECEIPTS = "trust/keeper/records.jsonl"
BINDING_SCHEMA = "gatedtrain.ainative.keeper_binding"
BINDING_SCHEMA_VERSION = 1
_BINDING_KEYS = frozenset({"schema", "schema_version", "socket", "fingerprint", "project_id"})

DRIVER_NEVER_HOLDS = "the keeper splits and scores; the driver never holds sealed items"
KEEPER_SPENDS = "the keeper spends alpha when it grants a sealed evaluation"


def _refuse(rule: str, field: str, message: str, fix: str) -> Refused:
    return Refused([Problem(rule, field, message, fix)])


def _anchor(field: str, message: str, fix: str = "check the keeper; a project stays bound to the key it was "
            "bound to (`gatedtrain verify` checks the records)") -> Refused:
    return _refuse("AN-14", field, message, fix)


def project_id(project: Project) -> str:
    """The id the keeper knows the project by: the sha256 of the canonical JSON of the project document (goal, data,
    pipeline, budget, creation time), so a copy of the binding in another project does not match."""
    return sha256_bytes(canonical_json(project._doc()))


def backend_compute(project: Project) -> dict:
    """AN-10, AN-22: the project's job totals `{jobs, seconds, gpu_seconds}` from its local backend's records
    (`LocalBackend.jobs`, which reads records only: no runtime, no host file); zeros for a project without jobs. The
    keeper states them in the report as the driver's (cooperative, not checked by the keeper)."""
    from .backend import LocalBackend          # backend imports this module (project_id)

    jobs = LocalBackend(project).jobs()
    return {"jobs": len(jobs), "seconds": sum(j.seconds for j in jobs), "gpu_seconds": sum(j.gpu_seconds for j in jobs)}


@dataclass(frozen=True)
class KeeperBinding:
    socket: str
    fingerprint: str
    project_id: str


def _read(project: Project, rel: str) -> bytes | None:
    _refuse_links(project.root, rel)
    try:
        return (project.root / rel).read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _anchor(rel, f"cannot read the keeper record ({type(exc).__name__})",
                      "check the file permissions") from None


def load_binding(project: Project) -> KeeperBinding | None:
    """The project's keeper binding, None when the project has none; refused (AN-14) when it is malformed."""
    data = _read(project, KEEPER_BINDING)
    if data is None:
        return None
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        doc = None
    if not (isinstance(doc, dict) and set(doc) == _BINDING_KEYS and doc["schema"] == BINDING_SCHEMA
            and doc["schema_version"] == BINDING_SCHEMA_VERSION
            and all(isinstance(doc[k], str) for k in ("socket", "fingerprint", "project_id"))):
        raise _anchor(KEEPER_BINDING, "the keeper binding is malformed",
                      "restore it; a project cannot be rebound to another keeper")
    return KeeperBinding(doc["socket"], doc["fingerprint"], doc["project_id"])


def _hello_key(hello: Any) -> tuple[str, bytes]:
    """(fingerprint, public PEM) of a `hello` result, checked against each other (AN-14)."""
    fp = hello.get("fingerprint") if isinstance(hello, dict) else None
    pem = hello.get("public_pem") if isinstance(hello, dict) else None
    if not (isinstance(fp, str) and isinstance(pem, str)):
        raise _anchor("hello", "the keeper serves no signing key", "start the keeper with `keeper serve`")
    if signing.fingerprint(signing.load_public_pem(pem.encode("ascii"))) != fp:
        raise _anchor("hello", "the keeper's public key does not have the fingerprint it states",
                      "check that the socket is a gatedtrain keeper's")
    return fp, pem.encode("ascii")


def bind_project(project: Project, socket: Path) -> KeeperBinding:
    """Bind the project to the keeper at `socket`: ask its key (`hello`), open the project in the keeper, then write
    `trust/keeper.pub` and `trust/keeper.json`. Binding again to the same key is a no-op; a binding to another key is
    refused (AN-14), since the project's signed records name the first one.

    Both calls wait with the long-op deadline (`protocol.LONG_OP_TIMEOUT_S`, read at call time), as `KeeperTrust.check`
    does: on a keeper busy with another client's evaluation the wait is queueing, not work; a keeper that is not
    running still fails at the connect (AN-12)."""
    socket = Path(socket).absolute()
    client = KeeperClient(socket, timeout=protocol.LONG_OP_TIMEOUT_S)
    fp, pem = _hello_key(client.call("hello", None, {}))
    pid = project_id(project)
    existing = load_binding(project)
    if existing is not None and (existing.fingerprint != fp or existing.project_id != pid):
        raise _anchor(KEEPER_BINDING, "the project is already bound to a keeper with another key",
                      "a project stays with the keeper it was bound to; create a new project for another keeper")
    client.call("open_project", pid, {})
    for rel, data in ((KEEPER_PUB, pem), (KEEPER_BINDING, canonical_json({
            "schema": BINDING_SCHEMA, "schema_version": BINDING_SCHEMA_VERSION, "socket": str(socket),
            "fingerprint": fp, "project_id": pid}))):
        _refuse_links(project.root, rel)
        _write_atomic(project.root / rel, data)
    return KeeperBinding(str(socket), fp, pid)


# The small records every keeper-mode run writes (the training part and the arm records are left out: the step 1
# ones can be large, and a keeper intake always writes a `sealed_summary` too).
_MARKED_RECORDS = ("sealed_summary", "power", "registration", "gate_result", "report")


def keeper_records(project: Project) -> bool:
    """True when the project holds keeper-mode records: receipts, the keeper's public key, or a trust stage record
    (`_MARKED_RECORDS`) of `schema_version` 2 with a `gatedtrain.ainative.` schema."""
    if any(os.path.lexists(project.root / rel) for rel in (RECEIPTS, KEEPER_PUB)):
        return True
    stages = project.root / "stages"
    paths = [p for name in _MARKED_RECORDS for p in stages.glob(f"*/{name}.json")] if stages.is_dir() else []
    for path in sorted(paths):
        try:
            doc = json.loads(path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get("schema_version") == 2 and \
                str(doc.get("schema", "")).startswith("gatedtrain.ainative."):
            return True
    return False


class KeeperTrust:
    """`TrustBackend` as keeper requests (mode `keeper`). Construction reads the binding and the key copy and refuses
    (AN-14) when either is missing or they disagree; it does not connect. `check()` asks the keeper's key (an
    unreachable keeper refuses, AN-12)."""

    mode = "keeper"

    def __init__(self, project: Project, client: KeeperClient | None = None) -> None:
        self.project = project
        binding = load_binding(project)
        if binding is None:
            raise _anchor(KEEPER_BINDING, "the project has no keeper binding",
                          "bind the project to its keeper (`gatedtrain project create ... --keeper SOCKET`)")
        if binding.project_id != project_id(project):
            raise _anchor(f"{KEEPER_BINDING}.project_id", "the keeper binding was made for another project",
                          "a binding cannot be copied between projects; bind this project to the keeper")
        pem = _read(project, KEEPER_PUB)
        if pem is None:
            raise _anchor(KEEPER_PUB, "the project's copy of the keeper key is missing",
                          "restore trust/keeper.pub from the keeper (`hello`)")
        self.public = signing.load_public_pem(pem)
        if signing.fingerprint(self.public) != binding.fingerprint:
            raise _anchor(KEEPER_PUB, "the project's copy of the keeper key is not the key the project was bound to",
                          "restore trust/keeper.pub from the keeper the project was bound to")
        self.binding = binding
        self.client = client if client is not None else KeeperClient(Path(binding.socket))
        self._guarantee: str | None = None

    # ---- transport ------------------------------------------------------------------------------------------------

    def check(self) -> dict:
        """Ask the keeper's key (`hello`) and refuse (AN-14) another key than the binding's; returns the hello.

        The `hello` names the project, so its guarantee is the one for the project's sealed set (AN-21: downgraded
        for good once rootful access was seen on the set), the text its intake summary and report state.

        The keeper serves one connection at a time, so this `hello` may queue behind another client's evaluation: it
        waits with the long-op deadline (`protocol.LONG_OP_TIMEOUT_S`, read at call time) unless the client has an
        explicit timeout. A keeper that is not running still fails at once (the connect is refused, AN-12)."""
        timeout = self.client.timeout if self.client.timeout is not None else protocol.LONG_OP_TIMEOUT_S
        hello = KeeperClient(self.client.socket_path, timeout=timeout).call("hello", self.binding.project_id, {})
        fp, _ = _hello_key(hello)
        if fp != self.binding.fingerprint:
            raise _anchor("keeper", "the keeper at the socket serves another key than the one this project was bound "
                          "to (a keeper with another state, or another keeper)",
                          "start the keeper the project was bound to; its signed records name that key")
        if not isinstance(hello.get("sealed_guarantee"), str):
            raise _anchor("hello", "the keeper states no guarantee", "check that the socket is a gatedtrain keeper's")
        self._guarantee = hello["sealed_guarantee"]
        return hello

    def _call(self, op: str, args: dict | None = None) -> Any:
        self.check()
        return self.client.call(op, self.binding.project_id, args or {})

    @property
    def sealed_guarantee(self) -> str:
        if self._guarantee is None:
            self.check()
        return self._guarantee

    # ---- envelopes and receipts -----------------------------------------------------------------------------------

    def verified(self, envelope: Any, kind: str) -> dict:
        """The record of an envelope signed by the bound key (AN-14), for this project when it names one."""
        record = signing.verify(envelope, self.public, kind=kind)
        if "project_id" in record and record["project_id"] != self.binding.project_id:
            raise _anchor(kind, "the keeper signed this record for another project", "check the keeper")
        return record

    def _receipt_lines(self) -> list[bytes]:
        data = _read(self.project, RECEIPTS)
        return [line for line in (data or b"").split(b"\n") if line]

    def receipts(self) -> list[dict]:
        """Every envelope in `RECEIPTS`, each verified against the bound key; refused (AN-14) on a bad line."""
        out = []
        for n, line in enumerate(self._receipt_lines(), 1):
            try:
                env = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                env = None
            if not isinstance(env, dict):
                raise _anchor(f"{RECEIPTS} line {n}", "the receipt is not a JSON object",
                              "the receipts were altered; restore them (`gatedtrain verify` checks them)")
            self.verified(env, env.get("kind") if isinstance(env.get("kind"), str) else "")
            out.append(env)
        return out

    def _append(self, envelope: dict) -> None:
        """Append a verified envelope to the receipts unless it is already there (one `O_APPEND` write per line)."""
        line = canonical_json(envelope)
        if line in self._receipt_lines():
            return
        _refuse_links(self.project.root, RECEIPTS)
        path = self.project.root / RECEIPTS
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o644)
        try:
            os.write(fd, line + b"\n")
            os.fsync(fd)
        finally:
            os.close(fd)

    def _keep(self, envelope: Any, kind: str) -> dict:
        record = self.verified(envelope, kind)
        self._append(envelope)
        return record

    def sync(self) -> int:
        """Append to the receipts every envelope the keeper signed for this project that they lack, each verified
        against the bound key for this project first (AN-14); returns how many were appended (final review finding 1:
        a claim and grant the driver never received)."""
        held = self._call("records")
        if not isinstance(held, list):
            raise _anchor("records", "the keeper's record list is malformed", "check the keeper")
        have = set(self._receipt_lines())
        added = 0
        for env in held:
            kind = env.get("kind") if isinstance(env, dict) else None
            self.verified(env, kind if isinstance(kind, str) else "")
            if canonical_json(env) not in have:
                self._append(env)
                have.add(canonical_json(env))
                added += 1
        return added

    def receipt(self, kind: str, **match: Any) -> dict | None:
        """The last receipt of `kind` whose record has the given values, or None."""
        found = None
        for env in self.receipts():
            if env["kind"] == kind and all(env["record"].get(k) == v for k, v in match.items()):
                found = env
        return found

    # ---- intake ---------------------------------------------------------------------------------------------------

    def seal(self, items: list[dict]) -> SealedRef:
        raise _refuse("AN-8", "seal", "the driver cannot seal items under the keeper", DRIVER_NEVER_HOLDS)

    def run_sealed(self, ref: SealedRef, generate: Any) -> list[dict]:
        raise _refuse("AN-8", "run_sealed", "the driver cannot run the sealed items under the keeper",
                      DRIVER_NEVER_HOLDS)

    def claim_sealed_evaluation(self, candidate_sha256: str, registration_hash: str) -> dict:
        raise _refuse("AN-8", "claim_sealed_evaluation", "the driver cannot claim a sealed evaluation by itself under "
                      "the keeper; `evaluate` asks the keeper, which spends alpha first", DRIVER_NEVER_HOLDS)

    def intake(self, raw: bytes, data_sha256: str, *, accept_inconclusive: bool = False) -> dict:
        """The keeper's split of the data file: `{sealed_id, train_rows, sealed_summary, power, reused}`. Data whose
        split would most likely be INCONCLUSIVE are refused (AN-6) unless `accept_inconclusive`."""
        return self._call("intake", {"data_b64": base64.b64encode(raw).decode("ascii"), "data_sha256": data_sha256,
                                     "accept_inconclusive": bool(accept_inconclusive)})

    def sealed_summary(self, ref: SealedRef | None = None) -> dict:
        """The keeper's summary of the project's sealed set; with `ref`, refused unless it is that set."""
        summary = self._call("sealed_summary")
        if not isinstance(summary, dict) or (ref is not None and summary.get("manifest_sha256") != ref.sealed_id):
            raise _refuse("AN-8", "sealed_summary", "the keeper's sealed set of this project is not the one named",
                          "pass the sealed set id the keeper returned at intake")
        return summary

    # ---- metric confirmation and registrations --------------------------------------------------------------------

    def confirm_metric(self, values: dict, sentence: str) -> dict:
        """Record the owner's confirmation of the deciding metric in the keeper (AN-18); returns the envelope."""
        env = self._call("confirm_metric", {**values, "sentence": sentence})
        self._keep(env, "metric_confirmation")
        return env

    def register(self, registration: dict) -> str:
        out = self._call("register", {"registration": registration})
        env = out.get("envelope") if isinstance(out, dict) else None
        record = self._keep(env, "registration")
        digest = out.get("hash")
        if record.get("hash") != digest or record.get("registration") != registration or \
                sha256_bytes(canonical_json(registration)) != digest:
            raise _anchor("register", "the keeper's signed registration is not the registration sent",
                          "check the keeper")
        return digest

    def registration_envelope(self, hash: str) -> dict:
        """The keeper's signed envelope of the registration `hash` (verified, appended to the receipts)."""
        out = self._call("registration", {"hash": hash})
        env = out.get("envelope") if isinstance(out, dict) else None
        record = self._keep(env, "registration")
        if record.get("hash") != hash or record.get("registration") != out.get("registration") or \
                sha256_bytes(canonical_json(record.get("registration"))) != hash:
            raise _anchor("registration", "the keeper's signed registration does not have the hash asked for",
                          "check the keeper")
        return env

    def registration(self, hash: str) -> dict:
        return self.registration_envelope(hash)["record"]["registration"]

    def latest_registration(self) -> str | None:
        return self._call("latest_registration")

    # ---- attempts -------------------------------------------------------------------------------------------------

    def evaluate(self, registration_hash: str, baseline: tuple[dict, str], candidate: tuple[dict, str]) -> dict:
        """Ask the keeper for a sealed attempt: it grants alpha, signs the claim and scores both arms. Returns the
        claim record. A refusal after the grant (the scoring failed) appends the spend's envelopes before it is
        raised, so the machine logs the attempt `failed` (preflight finding 9). Either way the receipts are then synced
        with the keeper (`sync`), which also brings in an earlier attempt this driver gave up on."""
        args = {"registration_hash": registration_hash,
                "baseline": {"record": baseline[0], "sha256": baseline[1]},
                "candidate": {"record": candidate[0], "sha256": candidate[1]}}
        try:
            out = self._call("evaluate", args)
        except RefusedAfterSpend as exc:
            res = exc.result
            for key, kind in (("claim", "claim"), ("ledger", "ledger_entry"), ("summary", "gate_summary"),
                              ("settle", "ledger_entry")):
                if res.get(key) is not None:
                    self._keep(res[key], kind)
            try:
                self.sync()
            except Refused:                          # the refusal below is the one to report; a later op syncs
                pass
            raise Refused(exc.problems) from None
        claim = self._keep(out.get("claim") if isinstance(out, dict) else None, "claim")
        grant = self._keep(out.get("ledger"), "ledger_entry")
        want = {"registration_hash": registration_hash, "baseline_model_sha256": baseline[1],
                "candidate_model_sha256": candidate[1], "attempt_id": out.get("attempt_id"), "alpha": out.get("alpha")}
        if any(claim.get(k) != v for k, v in want.items()) or grant.get("attempt_id") != claim.get("attempt_id") or \
                grant.get("alpha") != claim.get("alpha") or grant.get("entry") != "grant":
            raise _anchor("evaluate", "the keeper's signed claim is not for the attempt requested", "check the keeper")
        self.sync()
        return claim

    def sealed_evaluation(self) -> dict | None:
        """The project's latest claim record in the keeper, or None."""
        latest = None
        for env in self._call("records"):
            if isinstance(env, dict) and env.get("kind") == "claim":
                latest = self.verified(env, "claim")
        return latest

    def gate(self, attempt_id: str) -> dict:
        """The attempt's gate summary record (status and alpha only; the keeper keeps the effect and the records)."""
        out = self._call("gate", {"attempt_id": attempt_id})
        summary = self._keep(out.get("summary") if isinstance(out, dict) else None, "gate_summary")
        settle = self._keep(out.get("ledger"), "ledger_entry")
        if summary.get("attempt_id") != attempt_id or settle.get("attempt_id") != attempt_id or \
                settle.get("entry") != "settle":
            raise _anchor("gate", "the keeper's signed gate summary is not for the attempt requested",
                          "check the keeper")
        self.sync()
        return summary

    def report(self) -> dict:
        """The keeper's signed report record (`schema_version` 2); the receipts are synced before it is returned, so
        the project holds every envelope the report rests on. When the project's backend recorded at least one job,
        the request carries its job totals (`backend_compute`), which the report states as the driver's (AN-10); a
        project without jobs sends none, so a keeper that does not know `compute` still reports it. Job records that
        cannot be read never block the report: no totals are sent and a warning goes to stderr."""
        try:
            compute = backend_compute(self.project)
        except Exception as exc:                 # the class only: the report does not rest on the driver's totals
            compute = None
            print(f"warning: the project's job records cannot be read ({type(exc).__name__}); the keeper's report "
                  "states no job totals of the driver", file=sys.stderr)
        out = self._call("report", {"compute": compute} if compute and compute["jobs"] > 0 else None)
        record = self._keep(out.get("report") if isinstance(out, dict) else None, "report")
        self.sync()
        return record

    def alpha(self) -> dict:
        """The alpha-investing state of the project's sealed set, its ledger envelopes verified."""
        out = self._call("alpha")
        if not isinstance(out, dict) or not isinstance(out.get("entries"), list):
            raise _refuse("AN-15", "alpha", "the keeper's alpha state is malformed", "check the keeper")
        out["records"] = [signing.verify(e, self.public, kind="ledger_entry") for e in out["entries"]]
        return out


class KeeperAlphaLedger:
    """`AlphaLedger` read from the keeper. The keeper spends: `spend` and `record` refuse (AN-15)."""

    planning_alpha = PLANNING_ALPHA

    def __init__(self, trust: KeeperTrust) -> None:
        self.trust = trust

    def available(self) -> float:
        return self.trust.alpha()["available"]

    def spend(self, attempt_id: str) -> float:
        raise _refuse("AN-15", "spend", "the driver cannot spend alpha under the keeper", KEEPER_SPENDS)

    def record(self, attempt_id: str, passed: bool) -> None:
        raise _refuse("AN-15", "record", "the driver cannot record an outcome under the keeper; the keeper gates",
                      KEEPER_SPENDS)

    def entries(self) -> list[dict]:
        """One entry per attempt on the sealed set (every project that uses it, named by `project_id`; `j` is the
        attempt's place in the set's ledger), in grant order; an open attempt has `passed` and the settled amounts
        None. `project_entries` keeps this project's."""
        out: dict[str, dict] = {}
        for row in self.trust.alpha()["records"]:
            aid = row.get("attempt_id")
            if row.get("entry") == "grant":
                out[aid] = {"attempt_id": aid, "project_id": row.get("project_id"), "j": row.get("j"),
                            "alpha": row.get("alpha"), "passed": None, "alpha_spent": None, "alpha_earned": None,
                            "wealth_after": None}
            elif row.get("entry") == "settle" and aid in out:
                out[aid].update(passed=row.get("passed"), alpha_spent=row.get("cost"), alpha_earned=row.get("earned"),
                                wealth_after=row.get("wealth_after"))
        return list(out.values())

    def project_entries(self) -> list[dict]:
        """The entries of this project's own attempts (`entries` lists the whole sealed set's)."""
        return [e for e in self.entries() if e["project_id"] == self.trust.binding.project_id]
