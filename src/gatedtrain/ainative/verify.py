"""AN-14: `gatedtrain verify <project>` checks a keeper project's signed records against a trust anchor.

The project's copy of the keeper key (`trust/keeper.pub`) is a convenience, not an anchor: an agent running as the user
can rewrite the project. The anchor is either the key the keeper serves over its socket (`keeper`; the keeper also
lists every envelope it signed for the project, `records`) or a key fingerprint the reader got out of band
(`fingerprint`). Against the anchor `verify_project` checks:

- the project's key copy and the fingerprint its binding names are the anchor's (preflight finding 10);
- every envelope in the receipts (`RECEIPTS`) verifies under the anchor key, for this project;
- each `registration`, `gate_result` and `report` record in the project equals the signed record its envelope holds
  (by `envelope_sha256` for the first two, by record equality for the report), and `report.md` is the rendering of
  `report.json` (`render_markdown`).

Without an anchor the same checks run against the project's own key copy and binding (anchor `none`): that proves the
records are consistent with each other, not that the keeper made them. Any failure is refused (AN-14); the caller
turns the anchor into the exit code (0 anchored, 1 consistency only).

Only `--keeper` finds a deleted record: the project must hold exactly the envelopes the keeper's `records` op lists for
it, no more and no fewer. Offline (`--fingerprint` or no anchor) a report is bound to this project through its
registration hash and attempt ids, which must be among the project's signed receipts, but a record removed together
with everything naming it cannot be seen (`OFFLINE_LIMIT`, stated in the output).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from ..errors import Problem, Refused
from . import signing
from .keeper import protocol
from .keeper.protocol import KeeperClient
from .keeper_trust import KEEPER_PUB, RECEIPTS, _hello_key, load_binding, project_id
from .project import Project, _refuse_links
from .trust_stages import REPORT_MD, SCHEMA_GATE_RESULT, SCHEMA_REGISTRATION, SCHEMA_REPORT, render_markdown

CONSISTENCY_ONLY = "checked consistency only: no keeper and no fingerprint"
OFFLINE_LIMIT = ("offline verify cannot detect deleted records; verify against the keeper (--keeper) to check "
                 "that the project holds every record the keeper signed for it")
_FP = re.compile(r"sha256:[0-9a-f]{64}")
_ENVELOPE_META = ("schema", "schema_version", "envelope_sha256")


def _bad(field: str, message: str, fix: str = "the project's records were altered or come from another keeper; "
         "do not trust its report") -> Refused:
    return Refused([Problem("AN-14", field, message, fix)])


def _read_json(project: Project, rel: str) -> Any:
    _refuse_links(project.root, rel)
    try:
        return json.loads((project.root / rel).read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        raise _bad(rel, "the record cannot be read as JSON") from None


def _receipts(project: Project) -> list[tuple[str, Any]]:
    _refuse_links(project.root, RECEIPTS)
    try:
        data = (project.root / RECEIPTS).read_bytes()
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise _bad(RECEIPTS, f"the receipts cannot be read ({type(exc).__name__})") from None
    out = []
    for n, line in enumerate((x for x in data.split(b"\n") if x), 1):
        try:
            out.append((f"{RECEIPTS} line {n}", json.loads(line.decode("utf-8"))))
        except (UnicodeDecodeError, ValueError):
            raise _bad(f"{RECEIPTS} line {n}", "the receipt is not JSON") from None
    return out


def _anchor(project: Project, keeper: Path | None, fingerprint: str | None) -> tuple[str, str, Any]:
    """(anchor kind, anchor fingerprint, anchor public key)."""
    if fingerprint is not None and not _FP.fullmatch(fingerprint):
        raise _bad("--fingerprint", "the fingerprint is not sha256:<64 lowercase hex>",
                   "pass the fingerprint `gatedtrain keeper fingerprint` printed at setup")
    binding = load_binding(project)
    if binding is None:
        raise _bad("trust/keeper.json", "the project is not bound to a keeper, so it holds no signed records",
                   "verify a project created with `gatedtrain project create ... --keeper SOCKET`")
    if binding.project_id != project_id(project):
        raise _bad("trust/keeper.json.project_id", "the keeper binding was made for another project")
    _refuse_links(project.root, KEEPER_PUB)
    try:
        copy = signing.load_public_pem((project.root / KEEPER_PUB).read_bytes())
    except OSError:
        raise _bad(KEEPER_PUB, "the project's copy of the keeper key is missing") from None
    if keeper is not None:
        hello = KeeperClient(Path(keeper).absolute(), timeout=protocol.LONG_OP_TIMEOUT_S).call("hello", None, {})
        fp, pem = _hello_key(hello)
        kind, public = "keeper", signing.load_public_pem(pem)
        if fingerprint is not None and fingerprint != fp:
            raise _bad("--fingerprint", "the keeper serves another key than the fingerprint given")
    elif fingerprint is not None:
        kind, fp, public = "fingerprint", fingerprint, copy
    else:
        kind, fp, public = "none", binding.fingerprint, copy
    if signing.fingerprint(copy) != fp:
        raise _bad(KEEPER_PUB, "the project's copy of the keeper key is not the anchor's key")
    if binding.fingerprint != fp:
        raise _bad("trust/keeper.json.fingerprint", "the project is bound to another key than the anchor's")
    return kind, fp, public


_NAMED = {"registration.json": SCHEMA_REGISTRATION, "gate_result.json": SCHEMA_GATE_RESULT,
          "report.json": SCHEMA_REPORT}


def _records(project: Project) -> list[tuple[str, dict]]:
    """(path, record) of every registration, gate result and report record under `stages/`: each file named
    `registration.json`, `gate_result.json` or `report.json` must hold that record (a wrong schema is refused, never
    skipped), and any other record of those schemas is checked too. A `report.md` needs its `report.json`."""
    out = []
    stages = project.root / "stages"
    for path in sorted(stages.glob("*/*.json")) if stages.is_dir() else []:
        rel = path.relative_to(project.root).as_posix()
        doc = _read_json(project, rel)
        schema = doc.get("schema") if isinstance(doc, dict) else None
        want = _NAMED.get(path.name)
        if want is not None and schema != want:
            raise _bad(rel, f"the file does not hold a {want} record")
        if schema in _NAMED.values():
            out.append((rel, doc))
    for path in sorted(stages.glob(f"*/{REPORT_MD}")) if stages.is_dir() else []:
        if not os.path.lexists(path.with_name("report.json")):
            raise _bad(path.relative_to(project.root).as_posix(), "report.md has no report.json next to it, so it "
                       "cannot be checked against a signed report")
    return out


def _cross_bind(by_sha: dict[str, dict], rel: str, report: dict) -> None:
    """The report's registration and attempts must be this project's: its signed registration and claim receipts
    (the report record names no project; review finding 1)."""
    hashes = {e["record"].get("hash") for e in by_sha.values() if e["kind"] == "registration"}
    attempts = {e["record"].get("attempt_id") for e in by_sha.values() if e["kind"] == "claim"}
    if _dict(report.get("registration")).get("hash") not in hashes:
        raise _bad(f"{rel}.registration.hash", "the report's registration is not one this project holds signed")
    listed = report.get("attempts")
    if not isinstance(listed, list) or not listed or \
            any(_dict(a).get("attempt_id") not in attempts for a in listed):
        raise _bad(f"{rel}.attempts", "the report lists an attempt this project holds no signed claim for")


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _check_registration(rel: str, doc: dict, env: dict) -> None:
    rec = env["record"]
    reg = _dict(rec.get("registration"))
    clause = _dict(reg.get("clause"))
    want = {"hash": rec.get("hash"), "scorer": clause.get("metric"), "direction": clause.get("direction"),
            "margin": clause.get("margin"), "alpha": clause.get("alpha"),
            "item_set_hash": _dict(reg.get("eval_distribution")).get("item_set_hash"),
            "post_hoc": bool(reg.get("post_hoc", False)), "amends": reg.get("amends")}
    if any(doc.get(k) != v for k, v in want.items()):
        raise _bad(rel, "the registration record is not the registration the keeper signed")


def verify_project(root: Path, *, keeper: Path | None, fingerprint: str | None) -> dict:
    """`{anchor: "keeper" | "fingerprint" | "none", checked, fingerprint}`; refused (AN-14) on a bad signature, a
    record the keeper does not hold, or a record, key copy or binding that differs from the anchor. `checked` counts the
    envelopes in the receipts plus the registration, gate result and report records compared with them."""
    project = Project.load(Path(root))
    kind, fp, public = _anchor(project, keeper, fingerprint)
    pid = project_id(project)
    by_sha: dict[str, dict] = {}
    for where, env in _receipts(project):
        env_kind = env.get("kind") if isinstance(env, dict) else None
        record = signing.verify(env, public, kind=env_kind if isinstance(env_kind, str) else "")
        if "project_id" in record and record["project_id"] != pid:
            raise _bad(where, "the receipt was signed for another project")
        by_sha[signing.envelope_sha256(env)] = env
    if kind == "keeper":                     # the keeper's own list of what it signed for this project, both ways
        held = KeeperClient(Path(keeper).absolute(), timeout=protocol.LONG_OP_TIMEOUT_S).call("records", pid, {})
        if not isinstance(held, list):
            raise _bad("records", "the keeper's record list is malformed", "check the keeper")
        signed = set()
        for env in held:
            signing.verify(env, public)
            signed.add(signing.envelope_sha256(env))
        foreign = [sha for sha in by_sha if sha not in signed]
        if foreign:
            raise _bad(RECEIPTS, f"the keeper did not sign the receipt {foreign[0]} for this project",
                       "the receipt was made with the keeper's key but not by the keeper for this project; do not "
                       "trust it")
        missing = sorted(signed - set(by_sha))
        if missing:
            raise _bad(RECEIPTS, f"{len(missing)} record(s) the keeper signed for this project are missing from the "
                       f"project, first {missing[0]}", "a record was deleted from the project; do not trust its "
                       "report")
    checked = len(by_sha)
    for rel, doc in _records(project):
        schema = doc["schema"]
        if schema == SCHEMA_REPORT:
            if not any(e["kind"] == "report" and e["record"] == doc for e in by_sha.values()):
                raise _bad(rel, "report.json is not a report the keeper signed")
            _cross_bind(by_sha, rel, doc)
            if _dict(doc.get("keeper")).get("fingerprint") != fp:
                raise _bad(f"{rel}.keeper.fingerprint", "the report names another key than the anchor's")
            md = str(Path(rel).with_name(REPORT_MD))
            _refuse_links(project.root, md)
            try:
                text = (project.root / md).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                raise _bad(md, "report.md is missing") from None
            if text != render_markdown(doc):
                raise _bad(md, "report.md is not the rendering of report.json")
        else:
            sha = doc.get("envelope_sha256")
            env = by_sha.get(sha) if isinstance(sha, str) else None
            want = "registration" if schema == SCHEMA_REGISTRATION else "gate_summary"
            if env is None or env["kind"] != want:
                raise _bad(f"{rel}.envelope_sha256", "the record names no signed envelope the project holds")
            if schema == SCHEMA_REGISTRATION:
                _check_registration(rel, doc, env)
            elif env["record"] != {k: v for k, v in doc.items() if k not in _ENVELOPE_META}:
                raise _bad(rel, "the gate result is not the gate summary the keeper signed")
        checked += 1
    return {"anchor": kind, "checked": checked, "fingerprint": fp}
