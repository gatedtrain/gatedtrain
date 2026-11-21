"""The keeper's persistent state: projects, intake and sealed sets, metric confirmations and signed registrations
(AN-12, AN-14, AN-16, AN-17, AN-18; design 05 §18.9).

`KeeperState(state_dir)` owns the state directory for the life of the keeper: it holds an exclusive `fcntl.flock` on
`<state>/lock` (a second keeper on one state directory is refused, AN-12) and creates the Ed25519 signing key
`<state>/keeper.key` on first start. Every method takes the project id (`sha256:` of the canonical project document,
computed by the driver).

Layout under the state directory (every file mode 0600, every directory 0700, never written through a link):

- `intakes/<data hex>.json`: one per data file taken in; the commit point of an intake. Holds the training rows, the
  sealed summary and power bodies, the first project that took it in, the sorted input hashes of the sealed rows and
  of the rows released as training.
- `sets/<sealed hex>/items.json`: the sealed items (canonical JSON; the file's sha256 is the sealed set id);
  `sets/<sealed hex>/closed`: present once a report closed the set.
- `projects/<project hex>/project.json`: the project (opened time, its data and sealed set, the latest registration,
  its sealed attempts); `projects/<project hex>/registrations/<hash hex>.json`: each registration with its envelope;
  `projects/<project hex>/records.jsonl`: every envelope the keeper signed for the project, in order.

Files are written by temp file plus rename, except `records.jsonl`, which is append-only: one `O_APPEND | O_NOFOLLOW`
write per line, as step 1's ledger (a torn last line from a crash is dropped at the next start, before any append).

Item-level seal tracking (AN-16, plan interpretations 1 and 4, preflight finding 12): an item is the sha256 of its
canonical `input`. A data file whose sha256 was taken in before reuses its sealed set (refused once a report closed
it); any other data holding an input that was ever sealed is refused, naming a count only; an input released as
training is forced into the training part of every later split, so it is never sealed.

No response, record or refusal holds a sealed input or label; an exception from scorer code is counted, never shown.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ...canonical import canonical_json, sha256_bytes
from ...errors import Problem, Refused
from ...registration import check_contents, resolved_evaluator_hash
from .. import scorers, sealed_eval, signing
from ..alpha_investing import PLANNING_ALPHA
from ..project import _utc_now, _write_atomic
from ..trust_stages import MAX_DATA_BYTES, MAX_MARGIN, SCHEMA_POWER, SCHEMA_SEALED_SUMMARY
from . import texts

KEY_FILE = "keeper.key"
LOCK_FILE = "lock"
STATE_SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 2        # keeper-mode bodies of the step 1 `sealed_summary` and `power` records
SCHEMA_INTAKE = "gatedtrain.ainative.keeper_intake"
SCHEMA_PROJECT = "gatedtrain.ainative.keeper_project"
SCHEMA_REGISTRATION = "gatedtrain.ainative.keeper_registration"
NEW_DATA_FIX = "a new test needs new labelled data"
MAX_SENTENCE_CHARS = 4000
CONFIRMED_KEYS = ("scorer", "scorer_params", "direction", "margin")

_SHA = re.compile(r"sha256:[0-9a-f]{64}")


def _problem(rule: str, field: str, message: str, fix: str) -> Refused:
    return Refused([Problem(rule, field, message, fix)])


def _hex(value: str) -> str:
    return value.removeprefix("sha256:")


def _check_sha(value: Any, field: str, rule: str = "AN-12") -> str:
    if not (isinstance(value, str) and _SHA.fullmatch(value)):
        raise _problem(rule, field, f"{field} is not sha256:<64 lowercase hex>",
                       f"send the {field} the keeper returned")
    return value


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _damaged(rel: str) -> Refused:
    return _problem("AN-12", rel, "a keeper state file is damaged",
                    "restore the keeper's state directory from a backup")


class KeeperState:
    def __init__(self, state_dir: Path, *, owner_uid: int | None = None) -> None:
        self.dir = Path(state_dir)
        self.keeper_uid = os.geteuid()
        self.owner_uid = self.keeper_uid if owner_uid is None else owner_uid
        self.guarantee = texts.sealed_guarantee(self.keeper_uid, self.owner_uid)
        self._mutex = threading.RLock()
        self._lock_fd: int | None = self._take_lock()
        try:
            self.key = self._signing_key()
            self._load_index()
        except BaseException:
            self.close()
            raise

    # ---- lifetime -------------------------------------------------------------------------------------------------

    def _take_lock(self) -> int:
        try:
            fd = os.open(self.dir / LOCK_FILE, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        except OSError as exc:
            raise _problem("AN-12", "state", f"cannot open the state lock ({errno.errorcode.get(exc.errno, 'error')})",
                           "check the state directory exists and belongs to the keeper user") from None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise _problem("AN-12", "state", "another keeper holds this state directory",
                           "stop that keeper first; one state directory serves one keeper") from None
        return fd

    def close(self) -> None:
        """Release the state lock (the keeper is stopping)."""
        fd, self._lock_fd = self._lock_fd, None
        if fd is not None:
            os.close(fd)

    def _signing_key(self) -> Ed25519PrivateKey:
        path = self.dir / KEY_FILE
        if not os.path.lexists(path):
            signing.generate_key(path)
        return signing.load_private_key(path)

    # ---- files ----------------------------------------------------------------------------------------------------

    def _read(self, rel: str) -> bytes | None:
        try:
            fd = os.open(self.dir / rel, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        except FileNotFoundError:
            return None
        except OSError:
            raise _damaged(rel) from None
        with os.fdopen(fd, "rb") as f:
            return f.read()

    def _read_json(self, rel: str, schema: str) -> dict | None:
        data = self._read(rel)
        if data is None:
            return None
        try:
            doc = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise _damaged(rel) from None
        if not (isinstance(doc, dict) and doc.get("schema") == schema):
            raise _damaged(rel)
        return doc

    def _write(self, rel: str, data: bytes) -> None:
        path = self.dir / rel
        for parent in reversed(path.relative_to(self.dir).parents[:-1]):
            d = self.dir / parent
            if d.is_symlink():
                raise _damaged(str(parent))
            d.mkdir(mode=0o700, exist_ok=True)
        _write_atomic(path, data)

    def _write_json(self, rel: str, doc: dict) -> None:
        self._write(rel, canonical_json(doc))

    def _append(self, rel: str, envelope: dict) -> None:
        """One `O_APPEND` write per line, never through a link."""
        path = self.dir / rel
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        try:
            os.write(fd, canonical_json(envelope) + b"\n")
            os.fsync(fd)
        finally:
            os.close(fd)

    # ---- index ----------------------------------------------------------------------------------------------------

    def _load_index(self) -> None:
        """Seal tracking and the record index from the committed files; a torn last record line is dropped."""
        self._by_data: dict[str, str] = {}
        self._sealed: set[str] = set()
        self._released: set[str] = set()
        self._signed: set[str] = set()
        intakes = self.dir / "intakes"
        for path in sorted(intakes.glob("*.json")) if intakes.is_dir() else []:
            rec = self._read_json(f"intakes/{path.name}", SCHEMA_INTAKE)
            self._index_intake(rec)
        projects = self.dir / "projects"
        for path in sorted(projects.iterdir()) if projects.is_dir() else []:
            rel = f"projects/{path.name}/records.jsonl"
            self._drop_torn_line(rel)
            for env in self._read_records(rel):
                self._signed.add(signing.envelope_sha256(env))

    def _index_intake(self, rec: dict) -> None:
        try:
            self._by_data[rec["data_sha256"]] = rec["sealed_id"]
            self._sealed.update(rec["sealed_input_hashes"])
            self._released.update(rec["released_input_hashes"])
        except (KeyError, TypeError):
            raise _damaged(f"intakes/{_hex(str(rec.get('data_sha256')))}.json") from None

    def _drop_torn_line(self, rel: str) -> None:
        data = self._read(rel)
        if data and not data.endswith(b"\n"):
            fd = os.open(self.dir / rel, os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                os.ftruncate(fd, data.rfind(b"\n") + 1)
                os.fsync(fd)
            finally:
                os.close(fd)

    def _read_records(self, rel: str) -> list[dict]:
        out = []
        for line in (self._read(rel) or b"").split(b"\n"):
            if not line:
                continue
            try:
                env = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                env = None
            if not isinstance(env, dict):
                raise _damaged(rel)
            out.append(env)
        return out

    # ---- projects -------------------------------------------------------------------------------------------------

    @staticmethod
    def _project_rel(project_id: str) -> str:
        return f"projects/{_hex(_check_sha(project_id, 'project'))}"

    def _project(self, project_id: str) -> dict:
        rel = self._project_rel(project_id)
        doc = self._read_json(f"{rel}/project.json", SCHEMA_PROJECT)
        if doc is None:
            raise _problem("AN-12", "project", "the keeper has not opened this project",
                           "open the project first (`open_project`, done when the project is bound to the keeper)")
        return doc

    def _save_project(self, doc: dict) -> None:
        self._write_json(f"{self._project_rel(doc['project_id'])}/project.json", doc)

    def _taken_in(self, project_id: str) -> tuple[dict, dict]:
        """(project, intake record) of a project that has taken in its data."""
        doc = self._project(project_id)
        if doc.get("data_sha256") is None:
            raise _problem("AN-16", "project", "the project has taken in no data yet", "run intake first")
        rec = self._intake_record(doc["data_sha256"])
        if rec is None:
            raise _damaged(f"{self._project_rel(project_id)}/project.json")
        return doc, rec

    def open_project(self, project_id: str) -> dict:
        with self._mutex:
            rel = self._project_rel(project_id)
            doc = self._read_json(f"{rel}/project.json", SCHEMA_PROJECT)
            if doc is None:
                doc = {"schema": SCHEMA_PROJECT, "schema_version": STATE_SCHEMA_VERSION, "project_id": project_id,
                       "opened_utc": _utc_now(), "data_sha256": None, "sealed_id": None, "reused": None,
                       "latest_registration": None, "attempts": []}
                self._save_project(doc)
            return {"project_id": doc["project_id"], "opened_utc": doc["opened_utc"]}

    # ---- intake (AN-16) -------------------------------------------------------------------------------------------

    def _intake_record(self, data_sha256: str) -> dict | None:
        return self._read_json(f"intakes/{_hex(data_sha256)}.json", SCHEMA_INTAKE)

    def _closed(self, sealed_id: str) -> bool:
        return os.path.lexists(self.dir / "sets" / _hex(sealed_id) / "closed")

    def _items(self, sealed_id: str) -> list[dict]:
        rel = f"sets/{_hex(sealed_id)}/items.json"
        data = self._read(rel)
        if data is None or sha256_bytes(data) != sealed_id:
            raise _damaged(rel)
        return json.loads(data.decode("utf-8"))

    def _result(self, rec: dict, reused: bool) -> dict:
        summary = {"schema": SCHEMA_SEALED_SUMMARY, "schema_version": ARTIFACT_SCHEMA_VERSION, **rec["sealed_summary"],
                   "guarantee": self.guarantee}
        power = {"schema": SCHEMA_POWER, "schema_version": ARTIFACT_SCHEMA_VERSION, **rec["power"]}
        return {"sealed_id": rec["sealed_id"], "train_rows": rec["train_rows"], "sealed_summary": summary,
                "power": power, "reused": bool(reused)}

    def _seal_new(self, project_id: str, raw: bytes, data_sha256: str, accept_inconclusive: bool) -> dict:
        s = sealed_eval.split_data(raw, data_sha256, planning_alpha=PLANNING_ALPHA,
                                   train_only=frozenset(self._released))
        inputs = set(s["input_hashes"])                  # a set: the order of first occurrence is not kept
        again = len(inputs & self._sealed)
        if again:
            raise _problem("AN-16", "data", f"{again} of the {len(inputs)} distinct inputs were sealed in an earlier "
                           "sealed set (the item is the input; a changed label or order does not make it new)",
                           NEW_DATA_FIX)
        if s["n_sealed"] < 1:
            raise _problem("AN-16", "data", "every input was released as training in an earlier intake, so nothing "
                           "is left to seal", NEW_DATA_FIX)
        if not s["conclusive"] and not accept_inconclusive:   # as step 1's intake_seal; nothing is stored yet
            raise _problem("AN-6", "accept_inconclusive",
                           f"the data are too small: the final test will most likely be INCONCLUSIVE "
                           f"({s['n_sealed']} sealed examples, {s['n_needed']} needed); about {s['more']} more "
                           "labelled examples would make it conclusive",
                           "add labelled examples, or send accept_inconclusive true to go on")
        items = canonical_json(s["sealed"])
        sealed_id = sha256_bytes(items)
        self._write(f"sets/{_hex(sealed_id)}/items.json", items)
        shown = {str(r["label"]) for r in s["train"]}
        counts: dict[str, int] = {}
        for item in s["sealed"]:
            key = str(item["label"])
            if key in shown:                             # no label text that only the sealed part holds
                counts[key] = counts.get(key, 0) + 1
        sealed_hashes = set(s["sealed_input_hashes"])
        rec = {"schema": SCHEMA_INTAKE, "schema_version": STATE_SCHEMA_VERSION, "data_sha256": data_sha256,
               "sealed_id": sealed_id, "first_project_id": project_id, "utc": _utc_now(), "train_rows": s["train"],
               "sealed_summary": {"n": s["n_sealed"], "manifest_sha256": sealed_id, "label_counts": counts,
                                  "label_counts_other": s["n_sealed"] - sum(counts.values())},
               "power": {**sealed_eval.power_record(s, planning_alpha=PLANNING_ALPHA),
                         "planning_alpha": PLANNING_ALPHA},
               "sealed_input_hashes": sorted(sealed_hashes),
               "released_input_hashes": sorted(inputs - sealed_hashes)}
        self._write_json(f"intakes/{_hex(data_sha256)}.json", rec)       # the commit point
        self._index_intake(rec)
        return rec

    def intake(self, project_id: str, raw: bytes, data_sha256: str, *, accept_inconclusive: bool = False) -> dict:
        """Split and seal the data file (AN-16); returns `{sealed_id, train_rows, sealed_summary, power, reused}`.
        New data whose split would most likely be INCONCLUSIVE are refused (AN-6) unless `accept_inconclusive`,
        before anything is stored, so the same rows plus more can be taken in afterwards."""
        with self._mutex:
            doc = self._project(project_id)
            if not isinstance(accept_inconclusive, bool):
                raise _problem("AN-6", "accept_inconclusive", "accept_inconclusive is not true or false",
                               "send true or false, or leave it out")
            _check_sha(data_sha256, "data_sha256", "AN-16")
            if not isinstance(raw, bytes) or len(raw) > MAX_DATA_BYTES:
                raise _problem("AN-6", "data", f"the data file is larger than 64 MiB ({MAX_DATA_BYTES} bytes)",
                               "split the data or use a smaller sample")
            if sha256_bytes(raw) != data_sha256:
                raise _problem("AN-16", "data_sha256", "the data received do not have the sha256 that was sent",
                               "send the data file and its sha256 unchanged")
            if doc["data_sha256"] is not None:
                if doc["data_sha256"] != data_sha256:
                    raise _problem("AN-16", "data", "this project has already taken in another data file",
                                   "create a new project for other data")
                rec = self._intake_record(data_sha256)
                if rec is None:
                    raise _damaged(f"{self._project_rel(project_id)}/project.json")
                return self._result(rec, doc["reused"])
            rec = self._intake_record(data_sha256)
            if rec is not None:
                if self._closed(rec["sealed_id"]):
                    raise _problem("AN-16", "data", "the sealed set of these data was closed by a report",
                                   NEW_DATA_FIX)
                reused = rec["first_project_id"] != project_id
            else:
                rec = self._seal_new(project_id, raw, data_sha256, accept_inconclusive)
                reused = False
            doc.update(data_sha256=data_sha256, sealed_id=rec["sealed_id"], reused=reused)
            self._save_project(doc)
            return self._result(rec, reused)

    def sealed_summary(self, project_id: str) -> dict:
        with self._mutex:
            doc, rec = self._taken_in(project_id)
            return self._result(rec, doc["reused"])["sealed_summary"]

    # ---- signed records -------------------------------------------------------------------------------------------

    def _sign_and_append(self, project_id: str, kind: str, record: dict) -> dict:
        env = signing.sign(self.key, kind, record)
        self._append(f"{self._project_rel(project_id)}/records.jsonl", env)
        self._signed.add(signing.envelope_sha256(env))
        return env

    def records(self, project_id: str) -> list[dict]:
        """Every envelope the keeper signed for the project, in order."""
        with self._mutex:
            self._project(project_id)
            return self._read_records(f"{self._project_rel(project_id)}/records.jsonl")

    def has_record(self, sha256: str) -> bool:
        with self._mutex:
            return isinstance(sha256, str) and sha256 in self._signed

    # ---- metric confirmation (AN-18) ------------------------------------------------------------------------------

    def _confirmations(self, project_id: str) -> list[dict]:
        return [env["record"] for env in self.records(project_id) if env.get("kind") == "metric_confirmation"]

    def confirm_metric(self, project_id: str, values: dict, sentence: str, peer_uid: int) -> dict:
        """Record and sign the owner's confirmation of the deciding metric (AN-18); returns the envelope."""
        with self._mutex:
            doc, _ = self._taken_in(project_id)
            if peer_uid != self.owner_uid:
                raise _problem("AN-18", "peer", "the confirmation did not come from the owner uid",
                               "run `gatedtrain keeper confirm` as the user the keeper serves")
            if not isinstance(values, dict) or set(values) - set(CONFIRMED_KEYS) or "scorer" not in values:
                raise _problem("AN-18", "values", "the confirmed values are not {scorer, scorer_params, direction, "
                               "margin}", "send the scorer, its parameters, the direction and the margin")
            spec = scorers.resolve(values["scorer"])
            params = scorers.evaluator_entry(spec, values.get("scorer_params", {})).get("params", {})
            direction = values.get("direction", spec.direction)
            if direction != spec.direction:
                raise _problem("AN-7", "direction", f"scorer {spec.name} is {spec.direction}",
                               f"use direction {spec.direction}, or pick another scorer")
            margin = values.get("margin")
            if not (_num(margin) and 0 <= margin <= MAX_MARGIN):
                raise _problem("AN-7", "margin", f"the margin is not a number from 0 to {MAX_MARGIN}",
                               "give the smallest gain that counts, as a fraction of the score (0.05 is 5 points)")
            if not (isinstance(sentence, str) and sentence.strip() and len(sentence) <= MAX_SENTENCE_CHARS):
                raise _problem("AN-18", "sentence", f"the sentence is not a text of 1 to {MAX_SENTENCE_CHARS} "
                               "characters", "send the plain sentence that was shown to the user")
            record = {"project_id": project_id, "sealed_id": doc["sealed_id"], "scorer": spec.name,
                      "scorer_params": params, "direction": direction, "margin": float(margin), "sentence": sentence,
                      "peer_uid": peer_uid}
            for env in self.records(project_id):            # the same confirmation again: the one on record
                if env.get("kind") == "metric_confirmation" and \
                        {k: v for k, v in env["record"].items() if k != "utc"} == record:
                    return env
            return self._sign_and_append(project_id, "metric_confirmation", {**record, "utc": _utc_now()})

    # ---- registrations (AN-7, AN-8, AN-17, AN-18) -----------------------------------------------------------------

    def _registration_rel(self, project_id: str, digest: str) -> str:
        return f"{self._project_rel(project_id)}/registrations/{_hex(digest)}.json"

    def _stored_registration(self, project_id: str, digest: str) -> dict | None:
        rel = self._registration_rel(project_id, digest)
        doc = self._read_json(rel, SCHEMA_REGISTRATION)
        if doc is not None and (doc.get("hash") != digest or sha256_bytes(canonical_json(doc["registration"]))
                                != digest):
            raise _damaged(rel)
        return doc

    def _check_registration(self, doc: dict, rec: dict, reg: Any) -> tuple[str, scorers.ScorerSpec]:
        if not isinstance(reg, dict):
            raise _problem("AN-7", "registration", "the registration is not a JSON object", "send the registration")
        try:
            digest = sha256_bytes(canonical_json(reg))
        except (ValueError, TypeError, RecursionError):
            raise _problem("AN-7", "registration", "the registration is not canonical JSON",
                           "use plain JSON values") from None
        try:
            problems = check_contents(reg)
        except Exception as exc:                        # a malformed value the checker did not expect
            raise _problem("AN-7", "registration", f"the registration cannot be checked ({type(exc).__name__})",
                           "send a registration built by preregister") from None
        if problems:
            raise Refused(problems)
        clause = reg.get("clause")
        if not isinstance(clause, dict):
            raise _problem("AN-7", "registration.clause", "the keeper takes a registration with one clause",
                           "register one superiority clause")
        if (reg.get("eval_distribution") or {}).get("item_set_hash") != doc["sealed_id"]:
            raise _problem("AN-7", "registration.eval_distribution.item_set_hash",
                           "the registered item set is not this project's sealed set",
                           "register the sealed set the keeper returned at intake")
        if clause.get("alpha") != PLANNING_ALPHA:
            raise _problem("AN-15", "registration.clause.alpha",
                           f"the registered alpha is not the planning alpha {PLANNING_ALPHA}; under the keeper the "
                           "test level of each attempt is set by alpha-investing",
                           f"register alpha {PLANNING_ALPHA} (the first bid at the initial wealth)")
        evs = reg.get("evaluators")
        ev = evs[0] if isinstance(evs, list) and evs and isinstance(evs[0], dict) else {}
        found = scorers.spec_for_evaluator(ev)
        if found is None or found[0].name != clause.get("metric") or \
                ev.get("manifest_hash") != resolved_evaluator_hash(ev):
            raise _problem("AN-17", "registration.evaluators[0]", "the registered evaluator is not the one the scorer "
                           "library resolves for the registered metric",
                           "register the evaluator entry of the confirmed scorer")
        spec, params = found
        if not any(c.get("scorer") == spec.name and c.get("scorer_params") == params
                   and c.get("direction") == clause.get("direction") and c.get("margin") == clause.get("margin")
                   for c in self._confirmations(doc["project_id"])):
            raise _problem("AN-18", "registration", "no metric confirmation of this project matches the "
                           "registration's scorer, parameters, direction and margin",
                           "confirm the metric for this project (`gatedtrain keeper confirm`), then register again")
        sigma = rec["power"].get("sigma")
        if not _num(sigma) or spec.planning_sd > sigma:
            raise _problem("AN-17", "registration.clause.metric",
                           f"scorer {spec.name} has a planning SD of {spec.planning_sd:g}, larger than the sealed set "
                           "was sized for, so the test would be underpowered",
                           f"the sealed set was sized for a planning SD of {sigma}; pick a library scorer or run "
                           "intake again with new data")
        items = self._items(doc["sealed_id"])
        bad = sum(1 for item in items if not _takes(spec, item["label"]))
        if bad:
            raise _problem("AN-17", "registration.clause.metric",
                           f"{bad} of {len(items)} sealed labels are not what scorer {spec.name} takes",
                           "pick a scorer that takes these labels, or label new data for this scorer and take it in")
        latest = doc.get("latest_registration")
        if doc.get("attempts") and digest != latest and not (reg.get("post_hoc") is True and
                                                             reg.get("amends") == latest):
            raise _problem("AN-8", "registration", "after the project's first sealed attempt a changed registration "
                           "is an amendment (attempts differ only in the candidate)",
                           f"register it with post_hoc true and amends {latest}")
        return digest, spec

    def register(self, project_id: str, registration: dict) -> dict:
        """Check, store and sign the registration; it becomes the project's latest. Returns `{hash, envelope}`."""
        with self._mutex:
            doc, rec = self._taken_in(project_id)
            digest, _ = self._check_registration(doc, rec, registration)
            stored = self._stored_registration(project_id, digest)
            if stored is None:
                env = signing.sign(self.key, "registration", {
                    "project_id": project_id, "sealed_id": doc["sealed_id"], "hash": digest,
                    "registration": registration, "utc": _utc_now()})
                stored = {"schema": SCHEMA_REGISTRATION, "schema_version": STATE_SCHEMA_VERSION, "hash": digest,
                          "registration": registration, "envelope": env}
                self._write_json(self._registration_rel(project_id, digest), stored)
            env = stored["envelope"]
            if signing.envelope_sha256(env) not in self._signed:
                self._append(f"{self._project_rel(project_id)}/records.jsonl", env)
                self._signed.add(signing.envelope_sha256(env))
            if doc.get("latest_registration") != digest:
                doc["latest_registration"] = digest
                self._save_project(doc)
            return {"hash": digest, "envelope": env}

    def registration(self, project_id: str, hash: str) -> dict:
        with self._mutex:
            self._project(project_id)
            stored = self._stored_registration(project_id, _check_sha(hash, "hash", "AN-7"))
            if stored is None:
                raise _problem("AN-7", "hash", "no registration with this hash in this project",
                               "pass a hash that register returned for this project")
            return {"registration": stored["registration"], "envelope": stored["envelope"]}

    def latest_registration(self, project_id: str) -> str | None:
        with self._mutex:
            return self._project(project_id).get("latest_registration")

    # ---- hooks for the attempt ops --------------------------------------------------------------------------------

    def _record_attempt(self, project_id: str, attempt_id: str) -> None:
        """Note a sealed attempt of the project: from then on a changed registration is an amendment (AN-8)."""
        doc = self._project(project_id)
        doc["attempts"] = [*doc.get("attempts", []), attempt_id]
        self._save_project(doc)

    def _close_set(self, sealed_id: str) -> None:
        """Close the sealed set (a report on it was built): no reuse, no further claim (AN-16)."""
        self._write(f"sets/{_hex(sealed_id)}/closed", b"")


def _takes(spec: scorers.ScorerSpec, label: Any) -> bool:
    """Whether the scorer takes the label; a `label_ok` that raises counts as not taken (it is never shown)."""
    try:
        return bool(spec.label_ok(label))
    except Exception:
        return False
