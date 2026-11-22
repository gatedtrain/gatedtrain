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
  `sets/<sealed hex>/closed`: present once a report closed the set;
  `sets/<sealed hex>/root_docker_seen.json`: written once, at the first check on the set that found rootful access
  by the owner (AN-21: first time and reasons); from then on the set's guarantee is the downgraded text;
  `sets/<sealed hex>/attempts/<j>.json`: sealed attempt j on the set (from any project): its claim and ledger `grant`
  envelopes, and once settled its gate record, gate summary and ledger `settle` envelopes. The attempt files are the
  set's alpha-investing ledger; the grant is committed (this file written) before anything is scored;
  `sets/<sealed hex>/baseline/<key hex>.json`: the baseline's per-item records, scored once per set and key
  (baseline sha256, sampler, eval_seeds, evaluator entry); `sets/<sealed hex>/records/<j>.json`: the candidate's
  per-item records of attempt j. Per-item records and gate records never leave the keeper.
- `projects/<project hex>/project.json`: the project (opened time, its data and sealed set, the latest registration,
  its sealed attempts); `projects/<project hex>/registrations/<hash hex>.json`: each registration with its envelope;
  `projects/<project hex>/records.jsonl`: every envelope the keeper signed for the project and showed its driver, in
  order (not the gate records); `projects/<project hex>/report.json`: the project's signed report.
- `models/<attempt hex>/<arm>/`: the keeper's copy of an arm's model files (AN-13), made before the grant under a fresh
  random hex per `evaluate` call; the generator runs only the copy. Removed once the arm is scored, on every other
  exit of `evaluate`, and (any leftover) when the keeper starts.
- `run/<hex>/`: one inference container's items, sampler and output directory (`inference.KeeperInference`), removed
  with the container on every path and (any leftover) when the keeper starts.

Files are written by temp file plus rename, except `records.jsonl`, which is append-only: one `O_APPEND | O_NOFOLLOW`
write per line, as step 1's ledger (a torn last line from a crash is dropped at the next start, before any append).

Item-level seal tracking (AN-16): an item is the sha256 of its
canonical `input`. A data file whose sha256 was taken in before reuses its sealed set (refused once a report closed
it); any other data holding an input that was ever sealed is refused, naming a count only; an input released as
training is forced into the training part of every later split, so it is never sealed.

No response, record or refusal holds a sealed input or label; an exception from scorer code is counted, never shown.

Attempts (AN-8, AN-9, AN-10, AN-15; spec §5.4): `evaluate` settles the set's open attempt, checks, grants alpha_j
(Foster-Stine, `alpha_investing`), commits and signs the claim, and only then scores (an exception while scoring,
a refusal or not, settles the attempt as not passed and returns the spend's envelopes in `RefusedAfterSpend`, naming
the exception class only); `gate` decides at alpha_j once
and settles the ledger; `report` builds and signs `report.json` (`schema_version` 2) from keeper records and closes
the set. An attempt still open at a restart is gated normally when its per-item records were stored, else it settles
as not passed (MISSING); either happens on the next `evaluate`, `gate` or `report` on the set.
At start, envelopes an attempt file or report holds but `records.jsonl` lacks (a crash between the two writes) are
appended again.

Batch generators (AN-25, AN-8, revision 25): when the registered generator is a scenario pack's batch generator,
each arm runs in the keeper's inference container (`inference`, a `KeeperInference`) on the keeper's copy of its
model files. Before the grant, a keeper without an inference container or an arm that lists no model files is
refused (AN-8), and `inference.ready` runs after the copies, last before the grant (image, daemon, script, time
bound, GPU wait). A container that fails after the grant is a crash (AN-9): the attempt settles as not passed. Each
container run is noted in the attempt as `inference` (arm, image ID, seconds, GPU seconds, exit code), crashed ones
too, and signed in its gate record; `report` sums the project's runs into `compute.keeper` and states the driver's
optional `compute` totals as `compute.backend`, marked as the driver's (AN-10).

Rootful-access detection (AN-21): `root_docker` is called at intake, sealed summary, evaluate,
gate and report on a set (and `hello` passes its own result); a positive result is a fact about the host, so it is
recorded once on every sealed set not yet closed, whichever set the op touched (a closed set and an existing record
are never changed). The intake summary, the report and `hello` for every project on such a set state the downgraded
guarantee from then on, also after the access ended. Without `root_docker` (tests that construct the state in
process) nothing is detected.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import pwd
import re
import secrets
import shutil
import stat
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ...canonical import canonical_json, sha256_bytes
from ... import gates
from ...errors import Problem, Refused
from ...registration import check_contents, resolved_evaluator_hash
from .. import alpha_investing as ai, scorers, sealed_eval, signing
from ..batch import BatchGenerator, BatchRunner
from ..alpha_investing import PLANNING_ALPHA
from ..project import _utc_now, _write_atomic
from ..trust_stages import (MAX_DATA_BYTES, MAX_MARGIN, SCHEMA_POWER, SCHEMA_REPORT, SCHEMA_SEALED_SUMMARY,
                            registered_base, registered_generator, scenario_batch_generators, scenario_generators,
                            scenario_pack_of)
from . import envcheck, texts
from .protocol import RefusedAfterSpend

KEY_FILE = "keeper.key"
LOCK_FILE = "lock"
MODELS_DIR = "models"
STATE_SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 2        # keeper-mode bodies of the step 1 `sealed_summary` and `power` records
SCHEMA_INTAKE = "gatedtrain.ainative.keeper_intake"
SCHEMA_PROJECT = "gatedtrain.ainative.keeper_project"
SCHEMA_REGISTRATION = "gatedtrain.ainative.keeper_registration"
SCHEMA_ATTEMPT = "gatedtrain.ainative.keeper_attempt"
SCHEMA_ARM_RECORDS = "gatedtrain.ainative.keeper_arm_records"
SCHEMA_STORED_REPORT = "gatedtrain.ainative.keeper_report"
SCHEMA_ROOT_DOCKER_SEEN = "gatedtrain.ainative.keeper_root_docker_seen"
REPORT_SCHEMA_VERSION = 2
NEW_DATA_FIX = "a new test needs new labelled data"
PASSED_FIX = "this project passed; write its report"
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
    def __init__(self, state_dir: Path, *, owner_uid: int | None = None,
                 root_docker: Callable[[], list[str]] | None = None, inference: BatchRunner | None = None) -> None:
        self.dir = Path(state_dir)
        self.keeper_uid = os.geteuid()
        self.owner_uid = self.keeper_uid if owner_uid is None else owner_uid
        self.root_docker = root_docker
        self.inference = inference          # the inference container (AN-25); None: a batch generator is refused
        self._mutex = threading.RLock()
        self.environment_sha256: str | None = None     # set by `serve` once it signed the environment record
        self._lock_fd: int | None = self._take_lock()
        try:
            self.key = self._signing_key()
            self._remove_copies(self.dir / MODELS_DIR)          # leftovers of a keeper that stopped while scoring
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
        self._repair()

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
                   "guarantee": self._guarantee(rec["sealed_id"])}
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
                self._check_set(rec["sealed_id"])
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
            self._check_set(rec["sealed_id"])
            return self._result(rec, reused)

    def sealed_summary(self, project_id: str) -> dict:
        with self._mutex:
            doc, rec = self._taken_in(project_id)
            self._check_set(rec["sealed_id"])
            return self._result(rec, doc["reused"])["sealed_summary"]

    # ---- rootful-access detection (AN-21) ------------------------------------------------------------------------

    def detect_root_docker(self) -> list[str]:
        """The current detection's reasons ([] without a detector); a detector that fails counts as positive."""
        if self.root_docker is None:
            return []
        try:
            return list(self.root_docker())
        except Exception as exc:                     # fail closed
            return [f"the rootful-access detection failed ({type(exc).__name__}): counted as rootful access"]

    def root_docker_seen(self, sealed_id: str) -> dict | None:
        """The set's `root_docker_seen` record (`{first_utc, reasons}`), or None."""
        return self._read_json(f"{self._set_rel(sealed_id)}/root_docker_seen.json", SCHEMA_ROOT_DOCKER_SEEN)

    def _open_sets(self) -> list[str]:
        """The ids of the sealed sets that hold their items and are not closed."""
        d = self.dir / "sets"
        names = sorted(p.name for p in d.iterdir()) if d.is_dir() else []
        return [f"sha256:{n}" for n in names if re.fullmatch(r"[0-9a-f]{64}", n)
                and os.path.lexists(d / n / "items.json") and not self._closed(f"sha256:{n}")]

    def record_root_docker(self, reasons: list[str]) -> None:
        """A positive detection: write `root_docker_seen` once on every open set that has
        none; closed sets and existing records stay as they are. Nothing for an empty `reasons`."""
        if not reasons:
            return
        with self._mutex:
            for sealed_id in self._open_sets():
                if self.root_docker_seen(sealed_id) is None:
                    self._write_json(f"{self._set_rel(sealed_id)}/root_docker_seen.json", {
                        "schema": SCHEMA_ROOT_DOCKER_SEEN, "schema_version": STATE_SCHEMA_VERSION,
                        "sealed_id": sealed_id, "first_utc": _utc_now(), "reasons": list(reasons)})

    def _check_set(self, sealed_id: str, reasons: list[str] | None = None) -> bool:
        """Run the detection (or take `reasons`) and record a positive result on every open set; True when the set
        was ever seen with rootful access (recorded) or the detection is positive now."""
        reasons = self.detect_root_docker() if reasons is None else reasons
        self.record_root_docker(reasons)
        return self.root_docker_seen(sealed_id) is not None or bool(reasons)

    def _guarantee(self, sealed_id: str) -> str:
        """The guarantee for the set (its recorded detection; run `_check_set` first)."""
        return texts.sealed_guarantee(self.keeper_uid, self.owner_uid,
                                      root_docker=self.root_docker_seen(sealed_id) is not None)

    def root_docker_for(self, project_id: str, reasons: list[str]) -> tuple[bool, list[str]]:
        """`hello` for a project: (rootful access now or recorded on the project's set, the
        reasons: the current ones, else the recorded ones with their first time). A project the keeper does not know,
        or one without a sealed set yet, reports the current state."""
        with self._mutex:
            doc = self._read_json(f"{self._project_rel(project_id)}/project.json", SCHEMA_PROJECT)
            sealed_id = doc.get("sealed_id") if doc is not None else None
            if not sealed_id:
                return bool(reasons), list(reasons)
            if not self._check_set(sealed_id, reasons):
                return False, []
            if reasons:
                return True, list(reasons)
            seen = self.root_docker_seen(sealed_id) or {}
            return True, [f"recorded on this sealed set at {seen.get('first_utc')}: {r}"
                          for r in seen.get("reasons", [])]

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

    # ---- attempts: storage --------------------------------------------------------------------------------------

    def _set_rel(self, sealed_id: str) -> str:
        return f"sets/{_hex(sealed_id)}"

    def _attempt_rel(self, sealed_id: str, j: int) -> str:
        return f"{self._set_rel(sealed_id)}/attempts/{j:06d}.json"

    def _raw_attempts(self, sealed_id: str) -> list[dict]:
        """The attempt files of the set in j order, checked for shape and numbering only."""
        d = self.dir / self._set_rel(sealed_id) / "attempts"
        out = []
        for path in sorted(d.glob("*.json")) if d.is_dir() else []:
            rel = f"{self._set_rel(sealed_id)}/attempts/{path.name}"
            doc = self._read_json(rel, SCHEMA_ATTEMPT)
            if doc is None or doc.get("j") != len(out) + 1 or path.name != f"{len(out) + 1:06d}.json" or \
                    doc.get("sealed_id") != sealed_id or \
                    not all(isinstance(doc.get(k), dict) for k in ("claim", "grant")):
                raise _damaged(rel)
            out.append(doc)
        return out

    def _attempts(self, sealed_id: str) -> list[dict]:
        """The set's attempts with every envelope verified (AN-14) and the ledger replayed (AN-15): each grant is the
        bid of the wealth the earlier settles leave, at most the last attempt is open, and every stored number equals
        the recomputed one."""
        public = self.key.public_key()
        atts = self._raw_attempts(sealed_id)
        settled: list[dict] = []
        for att in atts:
            kinds = {"claim": "claim", "grant": "ledger_entry", "gate_record": "gate_record",
                     "summary": "gate_summary", "settle": "ledger_entry"}
            recs = {}
            for key, kind in kinds.items():
                env = att.get(key)
                if env is not None:
                    recs[key] = signing.verify(env, public, kind=kind)
            ident = (att["attempt_id"], att["j"], att["project_id"])
            if any((r.get("attempt_id"), r.get("j"), r.get("project_id")) != ident for r in recs.values()) or \
                    recs["grant"].get("entry") != "grant" or recs["claim"]["alpha"] != recs["grant"]["alpha"]:
                raise _damaged(self._attempt_rel(sealed_id, att["j"]))
            is_open = "settle" not in recs
            if is_open != any(att.get(k) is None for k in ("gate_record", "summary", "settle")) or \
                    (is_open and att is not atts[-1]):
                raise _damaged(self._attempt_rel(sealed_id, att["j"]))
            wealth = ai.replay(settled)
            if recs["grant"]["wealth_before"] != wealth or not ai.can_bid(wealth) or \
                    recs["grant"]["alpha"] != ai.bid(wealth):
                raise _problem("AN-15", f"ledger[{att['j'] - 1}]", "the grant is not the bid of the replayed wealth",
                               "the ledger was altered; start a new test on new sealed data")
            if not is_open:
                row = recs["settle"]
                if row.get("entry") != "settle" or row["alpha"] != recs["grant"]["alpha"] or \
                        row["passed"] is not (recs["summary"]["status"] == gates.PASS):
                    raise _damaged(self._attempt_rel(sealed_id, att["j"]))
                settled.append(row)
        ai.replay(settled)
        return atts

    def _wealth(self, atts: list[dict]) -> float:
        """The wealth after the settled attempts; an open attempt counts as not passed."""
        wealth = ai.replay([a["settle"]["record"] for a in atts if a.get("settle") is not None])
        for a in atts:
            if a.get("settle") is None:
                wealth = ai.settle(wealth, a["grant"]["record"]["alpha"], False)["wealth_after"]
        return wealth

    def _append_record(self, project_id: str, env: dict) -> None:
        """Append an envelope to the project's records unless it is already there."""
        digest = signing.envelope_sha256(env)
        if digest not in self._signed:
            self._append(f"{self._project_rel(project_id)}/records.jsonl", env)
            self._signed.add(digest)

    def _repair(self) -> None:
        """After a crash between committing an attempt or report and appending to `records.jsonl`: append what is
        missing and note the attempt in its project."""
        sets = self.dir / "sets"
        for d in sorted(sets.iterdir()) if sets.is_dir() else []:
            sealed_id = f"sha256:{d.name}"
            for att in self._raw_attempts(sealed_id):
                doc = self._project(att["project_id"])
                if att["attempt_id"] not in doc.get("attempts", []):
                    self._record_attempt(att["project_id"], att["attempt_id"])
                for key in ("claim", "grant", "summary", "settle"):
                    if att.get(key) is not None:
                        self._append_record(att["project_id"], att[key])
        projects = self.dir / "projects"
        for path in sorted(projects.iterdir()) if projects.is_dir() else []:
            stored = self._read_json(f"projects/{path.name}/report.json", SCHEMA_STORED_REPORT)
            if stored is not None:
                self._close_set(stored["sealed_id"])
                self._append_record(stored["project_id"], stored["envelope"])

    def _read_arm(self, rel: str) -> list | None:
        doc = self._read_json(rel, SCHEMA_ARM_RECORDS)
        if doc is None:
            return None
        if not isinstance(doc.get("records"), list):
            raise _damaged(rel)
        return doc["records"]

    def _write_arm(self, rel: str, model_sha: str, records: list) -> None:
        self._write_json(rel, {"schema": SCHEMA_ARM_RECORDS, "schema_version": STATE_SCHEMA_VERSION,
                               "model_ref_sha256": model_sha, "records": records})

    # ---- attempts: checks before the spend ------------------------------------------------------------------------

    def _owner_groups(self) -> set[int]:
        try:
            pw = pwd.getpwuid(self.owner_uid)
            return set(os.getgrouplist(pw.pw_name, pw.pw_gid))
        except (KeyError, OSError):
            return set()

    def _owner_may(self, st: os.stat_result, bits: tuple[int, int, int], groups: set[int] | None = None) -> bool:
        """Whether the owner uid has the (user, group, other) permission `bits` on `st` by its mode: the owner's
        class if it owns the file, else the group class if one of its groups (`os.getgrouplist`) owns it, else
        other. Root may."""
        if self.owner_uid == 0:
            return True
        if st.st_uid == self.owner_uid:
            return bool(st.st_mode & bits[0])
        if st.st_gid in (self._owner_groups() if groups is None else groups):
            return bool(st.st_mode & bits[1])
        return bool(st.st_mode & bits[2])

    def _owner_can_read(self, st: os.stat_result) -> bool:
        return self._owner_may(st, (stat.S_IRUSR, stat.S_IRGRP, stat.S_IROTH))

    def _owner_can_reach(self, real: str) -> bool:
        """Whether the owner uid may search (x) every ancestor directory of the resolved path `real`, so the file
        counts as readable by the owner only when the owner could also get to it. A directory that cannot be
        inspected counts as not searchable (fail closed)."""
        if self.owner_uid == 0:
            return True
        groups = self._owner_groups()
        parent = os.path.dirname(real)
        while True:
            try:
                st = os.stat(parent)
            except OSError:
                return False
            if not self._owner_may(st, (stat.S_IXUSR, stat.S_IXGRP, stat.S_IXOTH), groups):
                return False
            if parent == os.path.dirname(parent):
                return True
            parent = os.path.dirname(parent)

    @staticmethod
    def _remove_copies(path: Path) -> None:
        """Remove a model copy directory (never following a link); strict, as at start."""
        if os.path.islink(path) or os.path.isfile(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

    @classmethod
    def _discard_copies(cls, path: Path) -> None:
        """Best effort while evaluating: a failed removal never changes the result (the sweep at start removes
        what is left)."""
        try:
            cls._remove_copies(path)
        except OSError:
            pass

    def _opened_ok(self, fd: int, st: os.stat_result, name: str) -> tuple[str, str] | None:
        """AN-13 on the file the copy actually opened (a parent directory swapped for a link after the path checks
        is caught here): it lies outside the keeper's state directory, the owner can read it and can search every
        directory above its kernel-reported path. Fails closed when the descriptor's path cannot be read."""
        try:
            real = os.path.realpath(os.readlink(f"/proc/self/fd/{fd}"))
        except OSError:
            return f"{name}: the keeper cannot tell where the opened file lies", \
                "run the keeper where /proc is mounted"
        state = os.path.realpath(self.dir)
        if not real.startswith("/") or real == state or real.startswith(state + os.sep):
            return f"{name} is inside the keeper's state directory", \
                "keep model files outside the keeper's state directory"
        if not self._owner_can_read(st):
            return f"{name} is not readable by the owner uid", "pass a file the user the keeper serves can read"
        if not self._owner_can_reach(real):
            return f"{name}: the owner uid cannot search a directory above the file", envcheck.TRAVERSE_FIX
        return None

    def _model_files(self, arm: str, record: dict, dest: Path) -> dict:
        """AN-13: model files must be absolute paths outside the keeper's state directory that the owner can read,
        under directories the owner can search (the keeper is never a way to read a file the owner cannot). The checks on the paths here are an early filter;
        `envcheck.copy_model_files` repeats them on each opened descriptor (`_opened_ok`, authoritative), copies the
        files into `dest` and checks the copy. Returns the record the generator receives (the copies' paths)."""
        problems = []
        entries = record.get("files", [])
        state = os.path.realpath(self.dir)
        for i, entry in enumerate(entries if isinstance(entries, list) else []):
            path = entry.get("path") if isinstance(entry, dict) else None
            if not isinstance(path, str):
                continue
            field = f"{arm}.record.files[{i}]"
            name = os.path.basename(path)
            if not os.path.isabs(path):
                problems.append(Problem(envcheck.RULE, field, f"{name}: the path is not absolute",
                                        "list each model file by its absolute path"))
                continue
            real = os.path.realpath(path)
            if real == state or real.startswith(state + os.sep):
                problems.append(Problem(envcheck.RULE, field, f"{name} is inside the keeper's state directory",
                                        "keep model files outside the keeper's state directory"))
                continue
            try:
                st = os.stat(path)
            except OSError:
                continue                                  # copy_model_files names it
            if not self._owner_can_read(st):
                problems.append(Problem(envcheck.RULE, field, f"{name} is not readable by the owner uid",
                                        "pass a file the user the keeper serves can read"))
            elif not self._owner_can_reach(real):
                problems.append(Problem(envcheck.RULE, field, f"{name}: the owner uid cannot search a directory "
                                        "above the file", envcheck.TRAVERSE_FIX))
        if problems:
            raise Refused(problems)
        try:
            envcheck.copy_model_files(record, dest, opened_ok=self._opened_ok)
        except Refused as exc:
            raise Refused([Problem(p.rule, f"{arm}.record.{p.field}", p.message, p.fix) for p in exc.problems]) \
                from None
        return envcheck.copied_record(record, dest)

    def _arm_args(self, arm: str, value: Any) -> tuple[dict, str]:
        if not (isinstance(value, dict) and set(value) == {"record", "sha256"} and isinstance(value["record"], dict)):
            raise _problem("AN-8", arm, f"the {arm} is not {{record, sha256}} with a model record",
                           "send the model reference record and its sha256")
        sha = _check_sha(value["sha256"], f"{arm}.sha256", "AN-8")
        try:
            actual = sha256_bytes(canonical_json(value["record"]))
        except (ValueError, TypeError, RecursionError):
            raise _problem("AN-8", f"{arm}.record", "the model record is not canonical JSON",
                           "send the model reference record as the stage wrote it") from None
        if actual != sha:
            raise _problem("AN-8", f"{arm}.sha256", f"the {arm} record does not have the sha256 that was sent",
                           "send the record and the sha256 of its canonical JSON")
        return value["record"], sha

    # ---- attempts: ops ----------------------------------------------------------------------------------------------

    def alpha(self, project_id: str) -> dict:
        """The alpha-investing state of the project's sealed set (an open attempt counts as not passed)."""
        with self._mutex:
            doc, _ = self._taken_in(project_id)
            atts = self._attempts(doc["sealed_id"])
            wealth = self._wealth(atts)
            entries = [e for a in atts for e in (a["grant"], a.get("settle")) if e is not None]
            return {"sealed_id": doc["sealed_id"], "wealth": wealth,
                    "available": ai.bid(wealth) if ai.can_bid(wealth) else 0.0, "planning_alpha": PLANNING_ALPHA,
                    "rule": ai.RULE, "entries": entries}

    @staticmethod
    def _claim_response(att: dict) -> dict:
        return {"attempt_id": att["attempt_id"], "j": att["j"], "alpha": att["grant"]["record"]["alpha"],
                "claim": att["claim"], "ledger": att["grant"]}

    def evaluate(self, project_id: str, registration_hash: str, baseline: dict, candidate: dict) -> dict:
        """Grant, sign and commit a sealed attempt, then score both arms (AN-8, AN-15); see the module docstring."""
        with self._mutex:
            doc, _ = self._taken_in(project_id)
            sealed_id = doc["sealed_id"]
            self._check_set(sealed_id)
            self._settle_open(sealed_id)
            if self._closed(sealed_id):
                raise _problem("AN-16", "sealed set", "a report closed this sealed set; no further attempt is allowed "
                               "on it from any project", NEW_DATA_FIX)
            _check_sha(registration_hash, "registration_hash", "AN-8")
            base_rec, base_sha = self._arm_args("baseline", baseline)
            cand_rec, cand_sha = self._arm_args("candidate", candidate)
            atts = self._attempts(sealed_id)
            mine = [a for a in atts if a["project_id"] == project_id]
            if mine and mine[-1]["registration_hash"] == registration_hash and \
                    mine[-1]["candidate_model_sha256"] == cand_sha:
                return self._claim_response(mine[-1])        # the same request again: the stored claim
            if any(a["summary"] is not None and a["summary"]["record"]["status"] == gates.PASS for a in mine):
                raise _problem("AN-9", "project", "this project has a PASS attempt; a PASS ends the project's "
                               "attempts", PASSED_FIX)
            if registration_hash != doc.get("latest_registration"):
                raise _problem("AN-8", "registration_hash", "the registration is not the project's latest",
                               "send the hash of the project's latest registration")
            full = self._stored_registration(project_id, registration_hash)["registration"]
            if base_sha != registered_base(full):
                raise _problem("AN-8", "baseline.sha256", "the baseline is not the base model that was registered",
                               "send the registered base model's record")
            if cand_sha == base_sha:
                raise _problem("AN-8", "candidate", "the candidate and the baseline are the same model",
                               "send the trained candidate's record")
            served = registered_generator(full)
            fn = scenario_generators().get(served[1]) if served else None
            gen = scenario_batch_generators().get(served[1]) if served else None      # AN-25: a batch generator
            if (fn is None and gen is None) or scenario_pack_of(served[1]) != served[0]:
                raise _problem("AN-8", "registration.base_model.generator", "the keeper does not serve the registered "
                               "scenario pack and generator", "install the registered scenario pack in the keeper")
            for arm, rec in (("baseline", base_rec), ("candidate", cand_rec)):
                if rec.get("generator") != served[1]:
                    raise _problem("AN-8", f"{arm}.record.generator", f"the {arm} model names another generator than "
                                   "the registered one", "both models must be served by the registered generator")
            evs = full.get("evaluators")
            ev = evs[0] if isinstance(evs, list) and evs and isinstance(evs[0], dict) else {}
            found = scorers.spec_for_evaluator(ev)
            if found is None or ev.get("manifest_hash") != resolved_evaluator_hash(ev):
                raise _problem("AN-8", "registration.evaluators[0]", "the registered evaluator is not the scorer the "
                               "keeper runs", "register again with the scorer the keeper serves")
            if gen is not None:
                self._check_batch(base_rec, cand_rec)
            copies = self.dir / MODELS_DIR / secrets.token_hex(16)      # this attempt's copies (AN-13)
            try:
                (self.dir / MODELS_DIR).mkdir(mode=0o700, exist_ok=True)
                copies.mkdir(mode=0o700)
                runs = {arm: self._model_files(arm, rec, copies / arm)    # before the grant: a refusal spends nothing
                        for arm, rec in (("baseline", base_rec), ("candidate", cand_rec))}
                wealth = self._wealth(atts)
                if not ai.can_bid(wealth):
                    raise _problem("AN-15", "alpha", f"the next bid {ai.bid(wealth):.6g} is below the smallest "
                                   f"allowed {ai.MIN_BID}: the alpha-investing wealth of this sealed set is spent",
                                   NEW_DATA_FIX)
                if gen is not None:
                    self.inference.ready(gen)      # image, daemon, script, time bound, GPU wait: last before the grant
                att = self._grant(project_id, sealed_id, registration_hash, base_sha, cand_sha, wealth,
                                  len(atts) + 1, full, ev)
                try:
                    self._score(att, full, found, fn, runs["baseline"], runs["candidate"], ev, copies, batch=gen)
                except Exception as exc:                    # after the grant: the attempt settles, not passed
                    problems = exc.problems if isinstance(exc, Refused) else [Problem(
                        "AN-8", "evaluate", f"the sealed evaluation failed after alpha was granted "
                        f"({type(exc).__name__}); the attempt settles as not passed", "check the keeper (its disk "
                        "and its scenario pack), then train a new candidate")]  # class name only: may hold an item
                    settled = self._settle(att)
                    raise RefusedAfterSpend(problems, {**self._claim_response(settled),
                                                           "summary": settled["summary"],
                                                           "settle": settled["settle"]}) from None
                return self._claim_response(att)
            finally:
                self._discard_copies(copies)

    def _check_batch(self, base_rec: dict, cand_rec: dict) -> None:
        """AN-8 (revision 25), before the grant: a batch generator runs only in the keeper's inference container, on
        the copy of the model files each arm lists."""
        if self.inference is None or not hasattr(self.inference, "ready"):
            raise _problem("AN-8", "registration.base_model.generator", "this keeper runs no inference container",
                           "start the keeper with an inference image (--inference-image; `gatedtrain setup host` "
                           "prints the command), or register a per-item generator")
        for arm, rec in (("baseline", base_rec), ("candidate", cand_rec)):
            files = rec.get("files")
            if not (isinstance(files, list) and files):
                raise _problem("AN-8", f"{arm}.record.files", "a batch generator's model reference must list its "
                               "model files (the keeper mounts its copy of them)",
                               "list each model file with its sha256 under `files`")

    def _baseline_key(self, base_sha: str, full: dict, ev: dict) -> str:
        return sha256_bytes(canonical_json({"baseline_model_sha256": base_sha, "sampler": full.get("sampler"),
                                            "eval_seeds": (full.get("clause") or {}).get("eval_seeds"),
                                            "evaluator": ev}))

    def _grant(self, project_id: str, sealed_id: str, registration_hash: str, base_sha: str, cand_sha: str,
               wealth: float, j: int, full: dict, ev: dict) -> dict:
        alpha = ai.bid(wealth)
        attempt_id = f"{_hex(sealed_id)[:16]}.{j}"
        utc = _utc_now()
        ident = {"attempt_id": attempt_id, "j": j, "project_id": project_id, "sealed_id": sealed_id}
        claim = signing.sign(self.key, "claim", {**ident, "registration_hash": registration_hash,
                                                 "baseline_model_sha256": base_sha, "candidate_model_sha256": cand_sha,
                                                 "alpha": alpha, "utc": utc})
        grant = signing.sign(self.key, "ledger_entry", {"entry": "grant", **ident, "alpha": alpha,
                                                        "wealth_before": wealth, "utc": utc})
        att = {"schema": SCHEMA_ATTEMPT, "schema_version": STATE_SCHEMA_VERSION, **ident,
               "registration_hash": registration_hash, "baseline_model_sha256": base_sha,
               "candidate_model_sha256": cand_sha, "baseline_key": self._baseline_key(base_sha, full, ev),
               "claim": claim, "grant": grant, "gate_record": None, "summary": None, "settle": None}
        self._write_json(self._attempt_rel(sealed_id, j), att)            # the commit point of the spend
        self._record_attempt(project_id, attempt_id)
        self._append_record(project_id, claim)
        self._append_record(project_id, grant)
        return att

    def _score(self, att: dict, full: dict, found: tuple[scorers.ScorerSpec, dict], fn: Any, base_rec: dict,
               cand_rec: dict, ev: dict, copies: Path, *, batch: BatchGenerator | None = None) -> None:
        """Score both arms on the records that name the keeper's copies; each arm's copy is removed (best effort)
        once it is scored. With `batch`, each arm runs in the keeper's inference container, and each container run
        is noted in the attempt (`inference`, signed in its gate record)."""
        spec, params = found
        score = scorers.bind(spec, params)
        items = self._items(att["sealed_id"])
        generate = _canonical_output(fn) if fn is not None else None
        set_rel = self._set_rel(att["sealed_id"])
        base_rel = f"{set_rel}/baseline/{_hex(att['baseline_key'])}.json"
        if self._read_arm(base_rel) is None:                  # scored once per set and key
            records = self._score_arm(att, "baseline", items, generate, base_rec, full, score, spec, batch)
            self._write_arm(base_rel, att["baseline_model_sha256"], records)
        self._discard_copies(copies / "baseline")
        records = self._score_arm(att, "candidate", items, generate, cand_rec, full, score, spec, batch)
        self._discard_copies(copies / "candidate")
        self._write_arm(f"{set_rel}/records/{att['j']:06d}.json", att["candidate_model_sha256"], records)

    def _score_arm(self, att: dict, arm: str, items: list[dict], generate: Any, rec: dict, full: dict, score: Any,
                   spec: scorers.ScorerSpec, batch: BatchGenerator | None) -> list:
        if batch is None:
            return sealed_eval.score_arm(items, generate, rec, full, score, score_range=spec.score_range,
                                         scorer_name=spec.name)
        self.inference.last_run = None
        try:
            records = sealed_eval.score_arm(items, None, rec, full, score, score_range=spec.score_range,
                                            scorer_name=spec.name, batch=batch, runner=self.inference)
        finally:                       # a crashed container counts too: the settle signs what ran
            run = self.inference.last_run
            if run is not None:
                att.setdefault("inference", []).append({**run, "arm": arm})      # the state sets the arm
                self.inference.last_run = None
        self._write_json(self._attempt_rel(att["sealed_id"], att["j"]), att)   # before the arm's records
        return records

    # ---- attempts: gate and settle ----------------------------------------------------------------------------------

    def _settle_open(self, sealed_id: str) -> None:
        for att in self._attempts(sealed_id):
            if att.get("settle") is None:
                self._settle(att)

    def _decide(self, att: dict, full: dict) -> dict:
        """The gate result at the claim's alpha; missing per-item records (a crash, a kill during scoring) make it
        MISSING."""
        alpha = att["grant"]["record"]["alpha"]
        set_rel = self._set_rel(att["sealed_id"])
        evs = full.get("evaluators")
        found = scorers.spec_for_evaluator(evs[0] if isinstance(evs, list) and evs and isinstance(evs[0], dict)
                                           else {})
        score_range = found[0].score_range if found else (0.0, 1.0)
        base = self._read_arm(f"{set_rel}/baseline/{_hex(att['baseline_key'])}.json")
        cand = self._read_arm(f"{set_rel}/records/{att['j']:06d}.json")
        expected = {"baseline": att["baseline_model_sha256"], "candidate": att["candidate_model_sha256"]}
        n_sealed = len(self._items(att["sealed_id"]))

        def run(b: list | None, c: list | None) -> dict:
            return sealed_eval.decide(full, {"records": b or [], "model_ref_sha256": expected["baseline"]},
                                      {"records": c or [], "model_ref_sha256": expected["candidate"]},
                                      alpha=alpha, n_sealed=n_sealed, expected_models=expected,
                                      score_range=score_range)
        if found is None or base is None or cand is None:
            return run(None, None)
        try:
            return run(base, cand)
        except Exception:                                    # never leave a granted attempt unsettled
            return run(None, None)

    def _settle(self, att: dict) -> dict:
        """Gate the open attempt once, sign the gate record, summary and ledger settle, commit them in the attempt
        file, then append the driver-visible ones to the project's records. Returns the settled attempt."""
        pid = att["project_id"]
        full = self._stored_registration(pid, att["registration_hash"])["registration"]
        grant = att["grant"]["record"]
        alpha = grant["alpha"]
        post_hoc = bool(full.get("post_hoc", False))
        result = self._decide(att, full)
        gate_result = {**result, "alpha": alpha, "attempt_id": att["attempt_id"],
                       "registration_hash": att["registration_hash"], "post_hoc": post_hoc}
        status = result["status"]
        row = ai.settle(grant["wealth_before"], alpha, status == gates.PASS)
        ident = {"attempt_id": att["attempt_id"], "j": att["j"], "project_id": pid, "sealed_id": att["sealed_id"]}
        utc = _utc_now()
        gate_record = signing.sign(self.key, "gate_record", {
            **ident, "registration_hash": att["registration_hash"],
            "candidate_model_sha256": att["candidate_model_sha256"], "result": gate_result,
            "inference": list(att.get("inference") or []), "utc": utc})
        summary = signing.sign(self.key, "gate_summary", {
            **ident, "registration_hash": att["registration_hash"], "status": status, "alpha": alpha,
            "alpha_spent": row["cost"], "alpha_earned": row["earned"], "wealth_after": row["wealth_after"],
            "post_hoc": post_hoc})
        settle = signing.sign(self.key, "ledger_entry", {"entry": "settle", **ident, **row, "utc": utc})
        att = {**att, "gate_record": gate_record, "summary": summary, "settle": settle}
        self._write_json(self._attempt_rel(att["sealed_id"], att["j"]), att)     # the commit point of the settle
        self._append_record(pid, summary)
        self._append_record(pid, settle)
        return att

    def gate(self, project_id: str, attempt_id: str) -> dict:
        """The attempt's gate summary and ledger settle envelopes (AN-9); computed once, then the stored ones."""
        with self._mutex:
            doc, _ = self._taken_in(project_id)
            self._check_set(doc["sealed_id"])
            found = [a for a in self._attempts(doc["sealed_id"])
                     if a["attempt_id"] == attempt_id and a["project_id"] == project_id]
            if not found:
                raise _problem("AN-9", "attempt_id", "this project has no sealed attempt with this id",
                               "pass the attempt_id that evaluate returned for this project")
            att = found[0]
            if att.get("settle") is None:
                att = self._settle(att)
            return {"summary": att["summary"], "ledger": att["settle"]}

    # ---- report (AN-10, AN-16) --------------------------------------------------------------------------------------

    def report(self, project_id: str, compute: dict | None = None) -> dict:
        """Settle the set's open attempt, build and sign `report.json` (`schema_version` 2) from keeper records, close
        the sealed set (AN-16). A second call returns the stored envelope. `compute` (optional, AN-10): the driver's
        backend totals `{jobs, seconds, gpu_seconds}`, stated in the report as the driver's."""
        with self._mutex:
            doc, rec = self._taken_in(project_id)
            backend = _backend_compute(compute)
            sealed_id = doc["sealed_id"]
            self._check_set(sealed_id)
            rel = f"{self._project_rel(project_id)}/report.json"
            stored = self._read_json(rel, SCHEMA_STORED_REPORT)
            if stored is not None:
                signing.verify(stored["envelope"], self.key.public_key(), kind="report")
                self._close_set(sealed_id)
                self._append_record(project_id, stored["envelope"])
                return {"report": stored["envelope"]}
            self._settle_open(sealed_id)
            atts = self._attempts(sealed_id)
            mine = [a for a in atts if a["project_id"] == project_id]
            if not mine:
                raise _problem("AN-10", "project", "the project has no settled sealed attempt to report",
                               "evaluate a candidate and gate it first")
            report = self._build_report(project_id, rec, atts, mine, backend)
            env = signing.sign(self.key, "report", report)
            self._close_set(sealed_id)              # before the report is stored: never a stored report on an open set
            self._write_json(rel, {"schema": SCHEMA_STORED_REPORT, "schema_version": STATE_SCHEMA_VERSION,
                                   "project_id": project_id, "sealed_id": sealed_id, "envelope": env})
            self._append_record(project_id, env)
            return {"report": env}

    def _build_report(self, project_id: str, rec: dict, atts: list[dict], mine: list[dict],
                      backend: dict | None = None) -> dict:
        passed = [a for a in mine if a["summary"]["record"]["status"] == gates.PASS]
        head = passed[0] if passed else mine[-1]
        stored = self._stored_registration(project_id, head["registration_hash"])
        reg_record = signing.verify(stored["envelope"], self.key.public_key(), kind="registration")
        if reg_record.get("registration") != stored["registration"]:
            raise _damaged(self._registration_rel(project_id, head["registration_hash"]))
        full = stored["registration"]
        clause = full.get("clause") or {}
        evs = full.get("evaluators")
        found = scorers.spec_for_evaluator(evs[0] if isinstance(evs, list) and evs and isinstance(evs[0], dict)
                                           else {})
        if found is None:
            raise _problem("AN-10", "registration.evaluators[0]", "the registered evaluator is not a scorer the "
                           "keeper runs, so the metric's unit is unknown", "register again with a scorer it serves")
        gate = head["gate_record"]["record"]["result"]
        power = dict(rec["power"])
        guarantee = self._guarantee(rec["sealed_id"])
        registration = {"hash": head["registration_hash"], "scorer": clause.get("metric"),
                        "direction": clause.get("direction"), "margin": clause.get("margin"),
                        "alpha": clause.get("alpha"),
                        "item_set_hash": (full.get("eval_distribution") or {}).get("item_set_hash"),
                        "post_hoc": bool(full.get("post_hoc", False)), "amends": full.get("amends")}
        ledger = [{k: a["settle"]["record"][k] for k in ("attempt_id", "j", "project_id", "alpha", "passed", "cost",
                                                         "earned", "wealth_after")} for a in atts]
        attempts = [{"attempt_id": a["attempt_id"], "j": a["j"], "candidate_model_sha256": a["candidate_model_sha256"],
                     "alpha": a["grant"]["record"]["alpha"], "status": a["summary"]["record"]["status"],
                     "utc": a["claim"]["record"]["utc"]} for a in mine]
        return {"schema": SCHEMA_REPORT, "schema_version": REPORT_SCHEMA_VERSION,
                "status": sealed_eval.effective_status(gate["status"], registration["post_hoc"]),
                "metric": {"scorer": clause.get("metric"), "unit": found[0].unit},
                "n_sealed": power["n_sealed"],
                "registration": registration,
                "power": power,
                "gate_result": gate,
                "display": sealed_eval.display_block(gate, power),
                "ledger": ledger, "attempts": attempts,
                "sealed_guarantee": guarantee,
                "known_limits": texts.known_limits(guarantee, len(atts)),
                "alpha_rule": {"rule": ai.RULE, "initial_wealth": ai.INITIAL_WEALTH, "payout": ai.PAYOUT,
                               "min_bid": ai.MIN_BID, "planning_alpha": PLANNING_ALPHA, "guarantee": ai.GUARANTEE},
                "keeper": {"fingerprint": signing.fingerprint(self.key),
                           "environment_sha256": self.environment_sha256},
                "compute": {"keeper": _keeper_compute(mine), "backend": backend}}


MAX_COMPUTE = 2 ** 53 - 1
COMPUTE_KEYS = frozenset({"jobs", "seconds", "gpu_seconds"})


def _backend_compute(compute: Any) -> dict | None:
    """AN-10: the driver's backend totals as the report states them (marked as the driver's), None when not sent."""
    if compute is None:
        return None
    if not (isinstance(compute, dict) and set(compute) == COMPUTE_KEYS and all(
            isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= MAX_COMPUTE for v in compute.values())):
        raise _problem("AN-10", "compute", "compute is not {jobs, seconds, gpu_seconds}, each a whole number >= 0",
                       "send the backend's job totals, or leave compute out")
    return {"jobs": compute["jobs"], "seconds": compute["seconds"], "gpu_seconds": compute["gpu_seconds"],
            "source": texts.COMPUTE_BACKEND_SOURCE}


def _keeper_compute(mine: list[dict]) -> dict:
    """AN-10: the containers, seconds and GPU seconds of the project's sealed evaluations, from the signed gate
    records (settled attempts only, which `report` ensures)."""
    rows = [r for a in mine for r in (a["gate_record"]["record"].get("inference") or [])]
    return {"containers": len(rows), "seconds": sum(int(r.get("seconds") or 0) for r in rows),
            "gpu_seconds": sum(int(r.get("gpu_seconds") or 0) for r in rows)}


def _canonical_output(fn: Any) -> Any:
    """The generator with its output checked: a value canonical JSON cannot encode is a crash (ValueError), so the
    attempt settles as not passed."""
    def generate(*args: Any, **kwargs: Any) -> Any:
        out = fn(*args, **kwargs)
        try:
            canonical_json(out)
        except (ValueError, TypeError, RecursionError):
            raise ValueError("the generator returned a value that is not canonical JSON") from None
        return out
    return generate


def _takes(spec: scorers.ScorerSpec, label: Any) -> bool:
    """Whether the scorer takes the label; a `label_ok` that raises counts as not taken (it is never shown)."""
    try:
        return bool(spec.label_ok(label))
    except Exception:
        return False
