"""AN-16, AN-17, AN-18, AN-14 and AN-12: keeper intake, item-level seal tracking, metric confirmation and signed
registrations (tests/README.md section 13).

The socket tests run the keeper as a subprocess of the test user (`tests/_keeper.py`); the `KeeperState` tests call
the state directly where a test needs an injected pack scorer or a claim that only a later step makes.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from _keeper import RUN_CLI, KeeperProcess, skip_unless_linux
from _support import FIXTURES, cli_env

import test_an_scorers as tsc
from gatedtrain.ainative import scorers, sealed_eval, signing, trust_stages
from gatedtrain.ainative.alpha_investing import PLANNING_ALPHA
from gatedtrain.ainative.keeper import texts
from gatedtrain.ainative.keeper.state import KeeperState
from gatedtrain.canonical import sha256_bytes
from gatedtrain.errors import Refused

TOY = (FIXTURES / "ainative" / "toy.jsonl").read_bytes()
NEW_DATA_FIX = "a new test needs new labelled data"
DEP = {"sampler": {"name": "greedy", "temperature": 1.0, "max_new_tokens": 8}, "eval_seeds": [0, 1]}
BASE_SHA = "sha256:" + "b" * 64


def project_id(name: str) -> str:
    return "sha256:" + hashlib.sha256(name.encode("utf-8")).hexdigest()


def rows_of(raw: bytes) -> list[dict]:
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode("utf-8")


def intake_args(raw: bytes, digest: str | None = None) -> dict:
    return {"data_b64": base64.b64encode(raw).decode("ascii"), "data_sha256": digest or sha256_bytes(raw)}


def registration(item_set_hash: str, scorer: str = "exact_match", *, margin: float = 0.05,
                 params: dict | None = None, alpha: float = PLANNING_ALPHA) -> dict:
    spec = scorers.resolve(scorer)
    v = {"scorer": scorer, "direction": spec.direction, "margin": margin, "alpha": alpha}
    return trust_stages._registration(v, item_set_hash, DEP, BASE_SHA, ("synthetic", "synthetic"),
                                      scorers.evaluator_entry(spec, params or {}))


def confirm_args(scorer: str = "exact_match", *, margin: float = 0.05, params: dict | None = None) -> dict:
    spec = scorers.resolve(scorer)
    return {"scorer": scorer, "scorer_params": params or {}, "direction": spec.direction, "margin": margin,
            "sentence": f"I will count {spec.what}; the new model must beat the base model by more than "
                        f"{margin * 100:g} points."}


def problem(cm) -> object:
    return cm.exception.problems[0]


class TextsTest(unittest.TestCase):
    def test_keeper_guarantee_is_the_spec_sentence(self):
        self.assertEqual(texts.KEEPER_GUARANTEE,
                         "the sealed items, registrations, claims, ledger and per-item records are held by a separate "
                         "OS user running code the user's account cannot change, and every record is signed; the "
                         "candidate runs inside the keeper process, not in a container (a model format that executes "
                         "code is refused); the original data file and the metric confirmation are only as strong as "
                         "the user's own account; a root user or the keeper's own account can break all of it.")

    def test_separate_user_text(self):                  # preflight finding 24: the separate-user text is pinned in CI
        self.assertEqual(texts.sealed_guarantee(1001, 1000), texts.KEEPER_GUARANTEE)

    def test_same_user_text(self):
        self.assertEqual(texts.sealed_guarantee(1000, 1000), texts.KEEPER_SAME_USER_GUARANTEE)
        self.assertIn("nothing is sealed", texts.KEEPER_SAME_USER_GUARANTEE)


@skip_unless_linux
class KeeperIntakeTest(unittest.TestCase):
    """Through `KeeperProcess` and `KeeperClient`."""

    def setUp(self):
        self.kp = KeeperProcess()
        self.kp.__enter__()
        self.addCleanup(self.kp.__exit__, None, None, None)
        self.client = self.kp.client
        self.project_dir = Path(tempfile.mkdtemp(prefix="gk-project-"))
        self.addCleanup(shutil.rmtree, self.project_dir, True)
        self.saved = 0

    def save(self, name: str, value: object) -> object:
        """Keep a response under the project-side temp dir, as a driver would."""
        self.saved += 1
        (self.project_dir / f"{self.saved:03d}-{name}.json").write_text(json.dumps(value, sort_keys=True),
                                                                        encoding="utf-8")
        return value

    def call(self, op: str, pid: str | None, args: dict | None = None) -> object:
        return self.save(op, self.client.call(op, pid, args or {}))

    def opened(self, name: str) -> str:
        pid = project_id(name)
        self.call("open_project", pid)
        return pid

    def intake(self, pid: str, raw: bytes = TOY) -> dict:
        return self.call("intake", pid, intake_args(raw))

    def refused(self, op: str, pid: str, args: dict) -> object:
        with self.assertRaises(Refused) as cm:
            self.client.call(op, pid, args)
        self.save(f"{op}-refused", [p.__dict__ for p in cm.exception.problems])
        return problem(cm)

    def assert_no_sealed_text(self, sealed: list[str]) -> None:
        quoted = [json.dumps(s) for s in sealed]
        for path in [*self.project_dir.rglob("*"), self.kp.log_path]:
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="replace")
                hits = [q for q in quoted if q in text]
                self.assertEqual(hits, [], f"sealed inputs in {path.name}")

    # ---- intake ---------------------------------------------------------------------------------------------------

    def test_intake_returns_training_rows_summary_and_power(self):      # AN-16, AN-6
        pid = self.opened("p1")
        self.assertEqual(self.call("open_project", pid)["project_id"], pid)      # idempotent
        r = self.intake(pid)
        self.assertEqual(set(r), {"sealed_id", "train_rows", "sealed_summary", "power", "reused"})
        self.assertIs(r["reused"], False)
        summary, power = r["sealed_summary"], r["power"]
        self.assertEqual(summary["manifest_sha256"], r["sealed_id"])
        self.assertEqual(summary["schema"], "gatedtrain.ainative.sealed_summary")
        self.assertEqual(power["schema"], "gatedtrain.ainative.power")
        self.assertEqual(summary["guarantee"], texts.KEEPER_SAME_USER_GUARANTEE)
        self.assertEqual(power["alpha"], 0.02375)
        self.assertEqual(power["planning_alpha"], 0.02375)
        step1 = sealed_eval.split_data(TOY, sha256_bytes(TOY), planning_alpha=0.05)
        self.assertGreater(power["n_needed"], step1["n_needed"])
        self.assertEqual(len(r["train_rows"]) + summary["n"], power["n_unique"])
        train_inputs = {row["input"] for row in r["train_rows"]}
        sealed = [row["input"] for row in rows_of(TOY) if row["input"] not in train_inputs]
        self.assertEqual(len(sealed), summary["n"])
        shown = {row["label"] for row in r["train_rows"]}
        self.assertTrue(set(summary["label_counts"]) <= shown)
        self.assertEqual(self.call("sealed_summary", pid), summary)
        self.call("hello", None)
        self.call("records", pid)
        # survives a restart; the same intake again returns the stored result
        self.kp.restart()
        self.client = self.kp.client
        self.assertEqual(self.call("sealed_summary", pid), summary)
        self.assertEqual(self.intake(pid), r)
        self.assert_no_sealed_text(sealed)

    def test_intake_refuses_a_wrong_digest_and_a_second_data_file(self):
        pid = self.opened("p1")
        p = self.refused("intake", pid, intake_args(TOY, "sha256:" + "0" * 64))
        self.assertEqual((p.rule, p.field), ("AN-16", "data_sha256"))
        self.intake(pid)
        other = jsonl([{"input": f"fresh-{i}", "label": "x"} for i in range(50)])
        p = self.refused("intake", pid, intake_args(other))
        self.assertEqual(p.rule, "AN-16")

    def test_same_data_second_project_reuses_the_sealed_set(self):     # AN-16, plan interpretation 1
        first = self.intake(self.opened("p1"))
        second = self.intake(self.opened("p2"))
        self.assertIs(second["reused"], True)
        self.assertEqual(second["sealed_id"], first["sealed_id"])
        self.assertEqual(second["train_rows"], first["train_rows"])

    def test_data_holding_a_sealed_item_refused(self):                 # AN-16
        first = self.intake(self.opened("p1"))
        train_inputs = {row["input"] for row in first["train_rows"]}
        rows = rows_of(TOY)
        sealed = [row["input"] for row in rows if row["input"] not in train_inputs]
        target = sealed[0]
        relabelled = [{**row, "label": "changed"} if row["input"] == target else row for row in rows]
        reshuffled = list(reversed(rows)) + [{"input": "a brand new row", "label": "L0"}]
        for name, data in (("relabelled", relabelled), ("reshuffled", reshuffled)):
            with self.subTest(name):
                p = self.refused("intake", self.opened(f"p-{name}"), intake_args(jsonl(data)))
                self.assertEqual(p.rule, "AN-16")
                self.assertEqual(p.fix, NEW_DATA_FIX)
                self.assertRegex(p.message, rf"\b{len(sealed)}\b")
                self.assertNotIn(target, p.message + p.field + p.fix)
        self.assert_no_sealed_text(sealed)

    def test_released_training_inputs_are_never_sealed_later(self):    # AN-16, preflight finding 12
        first = self.intake(self.opened("p1"))
        released = first["train_rows"]
        fresh = [{"input": f"fresh-{i}", "label": f"L{i % 5}"} for i in range(400)]
        second = self.intake(self.opened("p2"), jsonl(fresh + released))
        self.assertIs(second["reused"], False)
        self.assertNotEqual(second["sealed_id"], first["sealed_id"])
        train2 = {row["input"] for row in second["train_rows"]}
        self.assertTrue({row["input"] for row in released} <= train2)
        self.assertEqual(len(second["train_rows"]) + second["sealed_summary"]["n"], len(fresh) + len(released))

    # ---- metric confirmation and registration ---------------------------------------------------------------------

    def test_register_needs_a_matching_confirmation(self):            # AN-18, AN-14
        pid = self.opened("p1")
        sealed_id = self.intake(pid)["sealed_id"]
        reg = registration(sealed_id, margin=0.05)
        p = self.refused("register", pid, {"registration": reg})
        self.assertEqual(p.rule, "AN-18")
        conf = self.call("confirm_metric", pid, confirm_args(margin=0.05))
        p = self.refused("register", pid, {"registration": registration(sealed_id, margin=0.10)})
        self.assertEqual(p.rule, "AN-18")
        out = self.call("register", pid, {"registration": reg})
        self.assertEqual(set(out), {"hash", "envelope"})
        hello = self.call("hello", None)
        public = signing.load_public_pem(hello["public_pem"].encode("ascii"))
        record = signing.verify(out["envelope"], public, kind="registration")
        self.assertEqual(record["hash"], out["hash"])
        self.assertEqual(record["registration"], reg)
        self.assertEqual((record["project_id"], record["sealed_id"]), (pid, sealed_id))
        crec = signing.verify(conf, public, kind="metric_confirmation")
        self.assertEqual(set(crec), {"project_id", "sealed_id", "scorer", "scorer_params", "direction", "margin",
                                     "sentence", "peer_uid", "utc"})
        self.assertEqual(crec["peer_uid"], os.getuid())
        self.assertEqual(crec["sentence"], confirm_args()["sentence"])
        # identical content again: same hash and envelope
        self.assertEqual(self.call("register", pid, {"registration": reg}), out)
        self.assertEqual(self.call("registration", pid, {"hash": out["hash"]}),
                         {"registration": reg, "envelope": out["envelope"]})
        self.assertEqual(self.call("latest_registration", pid), out["hash"])
        records = self.call("records", pid)
        self.assertEqual(records, [conf, out["envelope"]])
        self.assertIs(self.call("has_record", pid, {"sha256": signing.envelope_sha256(out["envelope"])}), True)
        self.assertIs(self.call("has_record", pid, {"sha256": "sha256:" + "0" * 64}), False)
        self.kp.restart()
        self.client = self.kp.client
        self.assertEqual(self.call("records", pid), records)
        self.assertEqual(self.call("latest_registration", pid), out["hash"])

    def test_scorer_that_cannot_take_the_sealed_labels_refused(self):  # Review Focus 1, AN-17
        pid = self.opened("p1")
        r = self.intake(pid)
        self.call("confirm_metric", pid, confirm_args("set_f1"))
        p = self.refused("register", pid, {"registration": registration(r["sealed_id"], "set_f1")})
        self.assertEqual(p.rule, "AN-17")
        n = r["sealed_summary"]["n"]
        self.assertEqual(p.message, f"{n} of {n} sealed labels are not what scorer set_f1 takes")
        labels = {row["label"] for row in rows_of(TOY)}
        self.assertFalse([lab for lab in labels if re.search(rf"\b{re.escape(lab)}\b", p.message + p.fix)])
        self.assertIsNone(self.call("latest_registration", pid))

    def test_hello_states_the_same_user_guarantee(self):
        self.assertEqual(self.call("hello", None)["sealed_guarantee"], texts.KEEPER_SAME_USER_GUARANTEE)

    def test_second_keeper_on_the_same_state_refused(self):           # AN-12
        cmd = [sys.executable, str(RUN_CLI), "keeper", "serve", "--state", str(self.kp.state),
               "--socket", str(self.kp.tmp / "second.sock"), "--owner-uid", str(os.getuid())]
        proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60,
                              env=cli_env(), cwd=str(self.kp.tmp))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("AN-12", proc.stdout + proc.stderr)
        self.assertFalse((self.kp.tmp / "second.sock").exists())
        self.assertIn("keeper_version", self.call("hello", None))


@skip_unless_linux
class KeeperStateTest(unittest.TestCase):
    """`KeeperState` called directly (no socket)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gk-state-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.dir = self.tmp / "state"
        self.dir.mkdir(mode=0o700)
        self.state = self.open()

    def open(self) -> KeeperState:
        st = KeeperState(self.dir, owner_uid=os.getuid())
        self.addCleanup(st.close)
        return st

    def ready(self, name: str = "p1", raw: bytes = TOY) -> tuple[str, dict]:
        pid = project_id(name)
        self.state.open_project(pid)
        return pid, self.state.intake(pid, raw, sha256_bytes(raw))

    def test_a_second_state_on_one_directory_refused(self):
        with self.assertRaises(Refused) as cm:
            KeeperState(self.dir, owner_uid=os.getuid())
        self.assertEqual(problem(cm).rule, "AN-12")
        self.state.close()
        self.open()                                     # released on close

    def test_state_survives_a_reopen(self):
        pid, r = self.ready()
        self.state.confirm_metric(pid, {k: v for k, v in confirm_args().items() if k != "sentence"},
                                  confirm_args()["sentence"], os.getuid())
        out = self.state.register(pid, registration(r["sealed_id"]))
        fp = signing.fingerprint(self.state.key)
        self.state.close()
        again = self.open()
        self.assertEqual(signing.fingerprint(again.key), fp)
        self.assertEqual(again.latest_registration(pid), out["hash"])
        self.assertEqual(again.sealed_summary(pid), r["sealed_summary"])
        self.assertTrue(again.has_record(signing.envelope_sha256(out["envelope"])))
        self.assertEqual(len(again.records(pid)), 2)
        lines = (self.dir / "projects" / pid.removeprefix("sha256:") / "records.jsonl").read_bytes().splitlines()
        self.assertEqual(len(lines), 2)

    def confirm(self, pid: str, scorer: str = "exact_match", **kw) -> dict:
        args = confirm_args(scorer, **kw)
        sentence = args.pop("sentence")
        return self.state.confirm_metric(pid, args, sentence, os.getuid())

    def test_confirmation_from_another_uid_refused(self):
        pid, _ = self.ready()
        args = confirm_args()
        sentence = args.pop("sentence")
        with self.assertRaises(Refused) as cm:
            self.state.confirm_metric(pid, args, sentence, os.getuid() + 1)
        self.assertEqual(problem(cm).rule, "AN-18")

    def test_registration_checks(self):
        pid, r = self.ready()
        self.confirm(pid)
        cases = {
            "item_set": (registration("sha256:" + "c" * 64), "AN-7"),
            "alpha": (registration(r["sealed_id"], alpha=0.05), "AN-15"),
            "contents": ({**registration(r["sealed_id"]), "unknown": 1}, "REG-FIELD"),
        }
        for name, (reg, rule) in cases.items():
            with self.subTest(name), self.assertRaises(Refused) as cm:
                self.state.register(pid, reg)
            self.assertIn(rule, {p.rule for p in cm.exception.problems})
        self.assertIsNone(self.state.latest_registration(pid))

    def test_scorer_params_must_match_the_confirmation(self):
        pid, r = self.ready("num", jsonl([{"input": f"q{i}", "label": i} for i in range(300)]))
        self.confirm(pid, "numeric_match", params={"abs_tol": 0.5})
        with self.assertRaises(Refused) as cm:
            self.state.register(pid, registration(r["sealed_id"], "numeric_match", params={"abs_tol": 0.25}))
        self.assertEqual(problem(cm).rule, "AN-18")
        out = self.state.register(pid, registration(r["sealed_id"], "numeric_match", params={"abs_tol": 0.5}))
        self.assertEqual(out["hash"], self.state.latest_registration(pid))

    def test_pack_scorer_with_a_larger_planning_sd_refused(self):     # preflight finding 19, AN-17
        tsc.inject_pack(self)
        tsc.TEST_SCORERS["wide"] = tsc.pack_decl(id="wide.t17@0.1", planning_sd=0.9)
        tsc.TEST_SCORERS["pack_iou"] = tsc.pack_decl()
        trust_stages._reset_scenario_snapshot()
        pid, r = self.ready()
        self.confirm(pid, "wide")
        with self.assertRaises(Refused) as cm:
            self.state.register(pid, registration(r["sealed_id"], "wide"))
        self.assertEqual(problem(cm).rule, "AN-17")
        self.assertIn(f"{math.sqrt(0.5)}", problem(cm).fix)
        self.confirm(pid, "pack_iou")                   # no label_ok: any JSON label is taken
        self.state.register(pid, registration(r["sealed_id"], "pack_iou"))

    def test_after_the_first_claim_a_change_must_be_post_hoc(self):    # AN-8
        pid, r = self.ready()
        self.confirm(pid)
        self.confirm(pid, margin=0.1)
        first = self.state.register(pid, registration(r["sealed_id"]))
        changed = registration(r["sealed_id"], margin=0.1)
        second = self.state.register(pid, changed)               # before any claim: a change is free
        self.assertEqual(self.state.latest_registration(pid), second["hash"])
        self.state._record_attempt(pid, "attempt-1")             # what a claim (Task 9) records
        with self.assertRaises(Refused) as cm:
            self.state.register(pid, registration(r["sealed_id"]))
        self.assertEqual(problem(cm).rule, "AN-8")
        self.assertIn(second["hash"], problem(cm).fix)
        self.assertEqual(self.state.register(pid, changed), second)   # the latest again: unchanged
        amended = {**registration(r["sealed_id"]), "post_hoc": True, "amends": second["hash"],
                   "created_at": "2026-11-21T00:00:00Z"}
        out = self.state.register(pid, amended)
        self.assertEqual(self.state.latest_registration(pid), out["hash"])
        self.assertNotEqual(out["hash"], first["hash"])

    def test_reuse_of_a_closed_set_refused(self):                      # AN-16
        _, r = self.ready()
        self.state._close_set(r["sealed_id"])                    # what a report (Task 9) does
        pid2 = project_id("p2")
        self.state.open_project(pid2)
        with self.assertRaises(Refused) as cm:
            self.state.intake(pid2, TOY, sha256_bytes(TOY))
        self.assertEqual((problem(cm).rule, problem(cm).fix), ("AN-16", NEW_DATA_FIX))

    def test_intake_before_open_project_refused(self):
        with self.assertRaises(Refused) as cm:
            self.state.intake(project_id("nope"), TOY, sha256_bytes(TOY))
        self.assertEqual(problem(cm).rule, "AN-12")


if __name__ == "__main__":
    unittest.main()
