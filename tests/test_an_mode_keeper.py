"""AN-27, AN-28: the keeper side of the observed mode (tests/README.md section 13).

`KeeperState` is called directly with a peer uid, as the server does per connection. The state runs under the test user
with an owner uid that is not the test user's (so the keeper is a separate-user keeper by its own uids and its 0700
state directory); a peer with the keeper's own uid is the rogue case and sees same-user mode. Uids are derived from
the running process, no host name or number is assumed.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _keeper import skip_unless_linux, write_host_tree

from test_an_keeper_attempts import _ORIGINAL_EPS, _ScenarioEP, BASE, TOY, arm, model, registration
from test_an_keeper_intake import confirm_args, project_id
from gatedtrain.ainative import signing, trust_stages
from gatedtrain.ainative.keeper import modes, rootful, server, texts
from gatedtrain.ainative.keeper.state import SCHEMA_MODE_SEEN, REPORT_SCHEMA_VERSION, KeeperState
from gatedtrain.canonical import sha256_bytes
from gatedtrain.errors import Refused
from gatedtrain.testing import synthetic_pack as pack

KEEPER = os.getuid()
OWNER = KEEPER + 1               # the user the keeper serves: any uid that is not the keeper's
STRONG = OWNER                   # a peer that is not the keeper's own uid
WEAK = KEEPER                    # a peer with the keeper's own uid (what an agent running as the keeper's user is)


def rule(cm) -> str:
    return cm.exception.problems[0].rule


@skip_unless_linux
class ModeStateCase(unittest.TestCase):
    """A separate-user `KeeperState` (owner is another uid, state directory 0700) in a temporary directory."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gk-mode-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.dir = self.tmp / "state"
        self.dir.mkdir(mode=0o700)
        trust_stages._scenario_entry_points = lambda: [_ScenarioEP()]
        trust_stages._reset_scenario_snapshot()
        self.addCleanup(trust_stages._reset_scenario_snapshot)
        self.addCleanup(setattr, trust_stages, "_scenario_entry_points", _ORIGINAL_EPS)
        self.state = self.open()

    def open(self, owner: int = OWNER) -> KeeperState:
        st = KeeperState(self.dir, owner_uid=owner)
        self.addCleanup(st.close)
        return st

    def ready(self, name: str = "p1", *, peer: int = STRONG, require: str | None = None) -> tuple[str, dict]:
        pid = project_id(name)
        self.state.open_project(pid, require, peer)
        r = self.state.intake(pid, TOY, sha256_bytes(TOY), accept_inconclusive=True, peer_uid=peer)
        return pid, r

    def registered(self, name: str = "p1", *, peer: int = STRONG, require: str | None = None):
        pid, r = self.ready(name, peer=peer, require=require)
        args = confirm_args()
        sentence = args.pop("sentence")
        self.state.confirm_metric(pid, args, sentence, OWNER)
        reg = self.state.register(pid, registration(r["sealed_id"]), peer)
        return pid, r, reg

    def candidate(self, r: dict, k: int = 0) -> dict:
        full = pack.fit(r["train_rows"], 0.0)
        tok = sorted(full)[k]
        return arm(model({tok: full[tok]}, f"one-token-{k}"))

    def seen(self, r: dict) -> list[str]:
        return [m["mode"] for m in self.state.modes_seen(r["sealed_id"])]


class ObserveModeTest(ModeStateCase):
    def test_the_state_observes_its_own_directory(self):
        out = self.state.observe_mode()
        self.assertEqual(out["mode"], "separate_user")
        self.assertEqual(out["facts"]["state_dir_mode"], "0700")
        self.assertEqual((out["facts"]["keeper_euid"], out["facts"]["owner_uid"]), (KEEPER, OWNER))

    def test_peer_with_the_keeper_uid_is_same_user(self):          # the rogue `--owner-uid` case
        self.assertEqual(self.state.observe_mode(WEAK)["mode"], "same_user")
        self.assertEqual(self.state.observe_mode(STRONG)["mode"], "separate_user")

    def test_a_keeper_for_its_own_uid_is_same_user(self):
        self.state.close()
        self.assertEqual(self.open(KEEPER).observe_mode()["mode"], "same_user")

    def test_a_state_directory_open_to_others_is_same_user(self):
        os.chmod(self.dir, 0o755)
        self.assertEqual(self.state.observe_mode()["mode"], "same_user")


class StickyModeTest(ModeStateCase):
    def test_mode_seen_is_written_once_per_mode_and_never_upgraded(self):
        pid, r = self.ready()
        self.assertEqual(self.seen(r), ["separate_user"])
        doc = json.loads((self.dir / "sets" / r["sealed_id"].split(":")[1] / "mode_seen.json").read_bytes())
        self.assertEqual((doc["schema"], doc["sealed_id"]), (SCHEMA_MODE_SEEN, r["sealed_id"]))
        entry = doc["modes"][0]
        self.assertEqual(set(entry), {"mode", "first_utc", "fingerprint"})
        self.assertEqual(entry["fingerprint"], signing.fingerprint(self.state.key))
        self.state.sealed_summary(pid, STRONG)                       # the same mode again: nothing appended
        self.assertEqual(self.seen(r), ["separate_user"])
        self.assertEqual(self.state.set_mode(r["sealed_id"], STRONG), "separate_user")
        self.state.sealed_summary(pid, WEAK)                         # one request as a same-user peer
        self.assertEqual(self.seen(r), ["separate_user", "same_user"])
        self.assertEqual(self.state.set_mode(r["sealed_id"], STRONG), "same_user")      # weaker for good
        self.state.sealed_summary(pid, STRONG)
        self.assertEqual(self.seen(r), ["separate_user", "same_user"])

    def test_a_restart_under_a_stronger_observation_never_upgrades(self):
        pid, r = self.ready(peer=WEAK)
        self.assertEqual(self.seen(r), ["same_user"])
        self.state.close()
        again = self.open()
        self.assertEqual(again.set_mode(r["sealed_id"], STRONG), "same_user")
        summary = again.sealed_summary(pid, STRONG)
        self.assertEqual(summary["keeper_mode"], "same_user")
        self.assertEqual(summary["guarantee"], texts.KEEPER_OWNER_RUN_GUARANTEE)

    def test_the_summary_states_the_mode_and_the_matching_text(self):
        pid, r = self.ready()
        s = r["sealed_summary"]
        self.assertEqual((s["keeper_mode"], s["guarantee"]), ("separate_user", texts.KEEPER_GUARANTEE))
        s = self.state.sealed_summary(pid, WEAK)
        self.assertEqual((s["keeper_mode"], s["guarantee"]), ("same_user", texts.KEEPER_OWNER_RUN_GUARANTEE))

    def test_deleting_mode_seen_does_not_unstick_a_set(self):
        pid, r, reg = self.registered(peer=WEAK)                      # the signed registration states same_user
        (self.dir / "sets" / r["sealed_id"].split(":")[1] / "mode_seen.json").unlink()
        self.assertEqual(self.state.set_mode(r["sealed_id"], STRONG), "same_user")
        self.state.close()
        self.assertEqual(self.open().set_mode(r["sealed_id"], STRONG), "same_user")

    def test_deleting_mode_seen_does_not_unstick_a_set_signed_by_a_claim(self):
        pid, r, reg = self.registered()
        self.state.evaluate(pid, reg["hash"], arm(BASE), self.candidate(r), WEAK)     # a same-user claim
        (self.dir / "sets" / r["sealed_id"].split(":")[1] / "mode_seen.json").unlink()
        for path in (self.dir / "projects").glob("*/registrations/*.json"):          # and no weak registration is left
            doc = json.loads(path.read_bytes())
            self.assertEqual(doc["envelope"]["record"]["keeper_mode"], "separate_user")
        self.assertEqual(self.state.set_mode(r["sealed_id"], STRONG), "same_user")

    def test_a_damaged_mode_record_fails_closed(self):
        pid, r = self.ready()
        (self.dir / "sets" / r["sealed_id"].split(":")[1] / "mode_seen.json").write_bytes(b"not json")
        with self.assertRaises(Refused):
            self.state.set_mode(r["sealed_id"], STRONG)

    def test_a_closed_set_is_untouched(self):
        pid, r, reg = self.registered()
        a = self.state.evaluate(pid, reg["hash"], arm(BASE), self.candidate(r), STRONG)
        self.state.gate(pid, a["attempt_id"], STRONG)
        self.state.report(pid, None, STRONG)
        self.state.report(pid, None, WEAK)                           # the set is closed: nothing is recorded
        self.assertEqual(self.seen(r), ["separate_user"])


class SignedRecordsTest(ModeStateCase):
    def test_registration_claim_and_report_carry_the_mode(self):
        pid, r, reg = self.registered()
        rec = signing.verify(reg["envelope"], self.state.key.public_key(), kind="registration")
        self.assertEqual((rec["keeper_mode"], rec["require_keeper_mode"]), ("separate_user", "any"))
        a = self.state.evaluate(pid, reg["hash"], arm(BASE), self.candidate(r), STRONG)
        self.assertEqual(a["claim"]["record"]["keeper_mode"], "separate_user")
        self.state.gate(pid, a["attempt_id"], STRONG)
        report = signing.verify(self.state.report(pid, None, STRONG)["report"], self.state.key.public_key(),
                                kind="report")
        self.assertEqual(report["schema_version"], REPORT_SCHEMA_VERSION)
        self.assertEqual(REPORT_SCHEMA_VERSION, 3)
        self.assertEqual((report["keeper_mode"], report["require_keeper_mode"]), ("separate_user", "any"))
        self.assertEqual([m["mode"] for m in report["modes_seen"]], ["separate_user"])
        self.assertTrue(all(set(m) == {"mode", "first_utc"} for m in report["modes_seen"]))
        self.assertEqual([x["keeper_mode"] for x in report["attempts"]], ["separate_user"])
        self.assertEqual(report["sealed_guarantee"], texts.KEEPER_GUARANTEE)

    def test_a_same_user_peer_makes_a_same_user_report_with_the_owner_run_text(self):
        pid, r, reg = self.registered(peer=WEAK)
        a = self.state.evaluate(pid, reg["hash"], arm(BASE), self.candidate(r), WEAK)
        self.state.gate(pid, a["attempt_id"], WEAK)
        report = signing.verify(self.state.report(pid, None, WEAK)["report"], self.state.key.public_key(),
                                kind="report")
        self.assertEqual(report["keeper_mode"], "same_user")
        self.assertEqual(report["sealed_guarantee"], texts.KEEPER_OWNER_RUN_GUARANTEE)
        self.assertEqual(report["known_limits"][0]["text"], texts.KEEPER_OWNER_RUN_GUARANTEE)

    def test_mixed_modes_are_listed_and_the_weakest_wins(self):
        pid, r, reg = self.registered()                               # strong registration
        a1 = self.state.evaluate(pid, reg["hash"], arm(BASE), self.candidate(r, 0), STRONG)
        self.state.gate(pid, a1["attempt_id"], STRONG)
        a2 = self.state.evaluate(pid, reg["hash"], arm(BASE), self.candidate(r, 1), WEAK)     # one weak attempt
        self.state.gate(pid, a2["attempt_id"], WEAK)
        report = signing.verify(self.state.report(pid, None, STRONG)["report"], self.state.key.public_key(),
                                kind="report")
        self.assertEqual([m["mode"] for m in report["modes_seen"]], ["separate_user", "same_user"])
        self.assertEqual(report["keeper_mode"], "same_user")
        self.assertEqual([x["keeper_mode"] for x in report["attempts"]], ["separate_user", "same_user"])
        self.assertEqual(report["sealed_guarantee"], texts.KEEPER_OWNER_RUN_GUARANTEE)


class PolicyTest(ModeStateCase):
    def test_open_project_stores_the_requirement_and_fixes_it(self):
        pid = project_id("pol")
        self.assertEqual(self.state.open_project(pid, "separate_user", STRONG)["require_keeper_mode"],
                         "separate_user")
        self.assertEqual(self.state.open_project(pid, None, STRONG)["require_keeper_mode"], "separate_user")
        self.assertEqual(self.state.open_project(pid, "separate_user", STRONG)["require_keeper_mode"],
                         "separate_user")
        with self.assertRaises(Refused) as cm:                        # a later open with another value
            self.state.open_project(pid, "any", STRONG)
        self.assertEqual(rule(cm), "AN-28")
        self.assertTrue(cm.exception.problems[0].fix)

    def test_the_default_is_any_and_stays_any(self):
        pid = project_id("pol2")
        self.assertEqual(self.state.open_project(pid)["require_keeper_mode"], "any")
        with self.assertRaises(Refused) as cm:
            self.state.open_project(pid, "separate_user", STRONG)
        self.assertEqual(rule(cm), "AN-28")

    def test_a_value_outside_the_policy_values_is_refused(self):
        for bad in ("same_user", "separate-user", "strict", ""):
            with self.subTest(bad), self.assertRaises(Refused) as cm:
                self.state.open_project(project_id("pol3"), bad, STRONG)
            self.assertEqual(rule(cm), "AN-28")

    def test_a_same_user_keeper_refuses_the_requirement_at_open_and_stores_nothing(self):
        with self.assertRaises(Refused) as cm:
            self.state.open_project(project_id("pol4"), "separate_user", WEAK)
        self.assertEqual(rule(cm), "AN-28")
        self.assertFalse((self.dir / "projects").exists() and list((self.dir / "projects").iterdir()))

    def test_evaluate_gate_and_register_are_refused_for_a_weaker_peer(self):
        pid, r, reg = self.registered(require="separate_user")
        for op, call in (("register", lambda: self.state.register(pid, registration(r["sealed_id"]), WEAK)),
                         ("evaluate", lambda: self.state.evaluate(pid, reg["hash"], arm(BASE), self.candidate(r),
                                                                  WEAK))):
            with self.subTest(op), self.assertRaises(Refused) as cm:
                call()
            self.assertEqual(rule(cm), "AN-28")
            self.assertTrue(cm.exception.problems[0].fix)
        self.assertEqual(self.seen(r), ["separate_user"], "a refused peer must not weaken the set")
        a = self.state.evaluate(pid, reg["hash"], arm(BASE), self.candidate(r), STRONG)     # a strong peer still can
        with self.assertRaises(Refused) as cm:
            self.state.gate(pid, a["attempt_id"], WEAK)
        self.assertEqual(rule(cm), "AN-28")
        self.assertEqual(self.state.gate(pid, a["attempt_id"], STRONG)["summary"]["record"]["attempt_id"],
                         a["attempt_id"])

    def test_a_set_that_was_once_same_user_fails_the_requirement_for_good(self):
        pid, r, reg = self.registered(require="separate_user")
        self.state.sealed_summary(pid, WEAK)                          # the set records same-user (no policy on reads)
        with self.assertRaises(Refused) as cm:
            self.state.evaluate(pid, reg["hash"], arm(BASE), self.candidate(r), STRONG)
        self.assertEqual(rule(cm), "AN-28")

    def test_intake_enforces_the_requirement_on_a_weakened_set(self):
        self.ready("weak", peer=WEAK)                                 # the same data taken in by a same-user peer
        pid = project_id("strict")
        self.state.open_project(pid, "separate_user", STRONG)
        with self.assertRaises(Refused) as cm:
            self.state.intake(pid, TOY, sha256_bytes(TOY), accept_inconclusive=True, peer_uid=STRONG)
        self.assertEqual(rule(cm), "AN-28")
        self.assertTrue(cm.exception.problems[0].fix)

    def test_any_is_not_enforced(self):
        pid, r, reg = self.registered(peer=WEAK)
        a = self.state.evaluate(pid, reg["hash"], arm(BASE), self.candidate(r), WEAK)
        self.assertEqual(a["claim"]["record"]["keeper_mode"], "same_user")


@skip_unless_linux
class HelloTest(unittest.TestCase):
    """`hello` through `Keeper.handle`, with the peer the server would pass."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gk-mode-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.dir = self.tmp / "state"
        self.dir.mkdir(mode=0o700)
        self.view = rootful.HostView(root=write_host_tree(self.tmp / "host", owner_uid=OWNER), root_uid=KEEPER)

    def keeper(self, owner: int = OWNER) -> server.Keeper:
        st = KeeperState(self.dir, owner_uid=owner)
        self.addCleanup(st.close)
        return server.Keeper(self.dir, owner, key=st.key, state=st, host_view=self.view)

    def hello(self, keeper, peer, project=None) -> dict:
        line = json.dumps({"op": "hello", "project": project, "args": {}}).encode()
        resp = keeper.handle(line, peer)
        self.assertTrue(resp["ok"], resp)
        return resp["result"]

    def test_separate_user_hello(self):
        h = self.hello(self.keeper(), STRONG)
        self.assertEqual((h["keeper_mode"], h["same_user"], h["modes_seen"]), ("separate_user", False, []))
        self.assertEqual(h["sealed_guarantee"], texts.KEEPER_GUARANTEE)
        self.assertEqual(h["mode_facts"]["peer_uid"], STRONG)
        self.assertEqual(h["mode_facts"]["reasons"], [])

    def test_rogue_owner_uid_is_same_user_for_the_peer_that_is_the_keeper(self):
        k = self.keeper(owner=OWNER + 12345)          # an agent started this keeper with an arbitrary --owner-uid
        h = self.hello(k, KEEPER)
        self.assertEqual((h["keeper_mode"], h["same_user"]), ("same_user", True))
        self.assertEqual(h["sealed_guarantee"], texts.KEEPER_OWNER_RUN_GUARANTEE)
        self.assertEqual(h["mode_facts"]["peer_uid"], KEEPER)
        self.assertTrue(h["mode_facts"]["reasons"])

    def test_a_keeper_for_its_own_uid_is_same_user(self):
        h = self.hello(self.keeper(owner=KEEPER), KEEPER)
        self.assertEqual((h["keeper_mode"], h["same_user"]), ("same_user", True))      # the existing field is kept

    def test_hello_for_a_project_lists_the_set_modes_and_is_weakened_by_them(self):
        k = self.keeper()
        pid = project_id("hello")
        k.state.open_project(pid, None, STRONG)
        k.state.intake(pid, TOY, sha256_bytes(TOY), accept_inconclusive=True, peer_uid=STRONG)
        h = self.hello(k, STRONG, pid)
        self.assertEqual((h["keeper_mode"], h["modes_seen"]), ("separate_user", ["separate_user"]))
        k.state.sealed_summary(pid, KEEPER)
        h = self.hello(k, STRONG, pid)
        self.assertEqual((h["keeper_mode"], h["same_user"], h["modes_seen"]),
                         ("same_user", True, ["separate_user", "same_user"]))
        self.assertEqual(h["sealed_guarantee"], texts.KEEPER_OWNER_RUN_GUARANTEE)
        self.assertEqual(self.hello(k, STRONG)["keeper_mode"], "separate_user")      # no project: current peer only

    def test_hello_for_a_project_reads_the_signed_records_as_set_mode_does(self):
        # the signed registration states same_user; with mode_seen.json deleted, hello must still say same_user, as the
        # operations (set_mode) do, so a project is never told "separate_user" while its requests are refused
        k = self.keeper()
        pid = project_id("hello-signed")
        k.state.open_project(pid, None, KEEPER)
        r = k.state.intake(pid, TOY, sha256_bytes(TOY), accept_inconclusive=True, peer_uid=KEEPER)
        args = confirm_args()
        sentence = args.pop("sentence")
        k.state.confirm_metric(pid, args, sentence, OWNER)
        k.state.register(pid, registration(r["sealed_id"]), KEEPER)
        (self.dir / "sets" / r["sealed_id"].split(":")[1] / "mode_seen.json").unlink()
        self.assertEqual(k.state.set_mode(r["sealed_id"], STRONG), "same_user")
        h = self.hello(k, STRONG, pid)
        self.assertEqual((h["keeper_mode"], h["same_user"]), ("same_user", True))
        self.assertEqual(h["sealed_guarantee"], texts.KEEPER_OWNER_RUN_GUARANTEE)
        self.assertIn("same_user", h["modes_seen"])
        self.assertEqual(k.state.project_modes(pid), h["modes_seen"])

    def test_a_keeper_without_a_state_directory_it_can_read_is_same_user(self):
        k = self.keeper()
        shutil.rmtree(self.dir)
        out = k.observe_mode(STRONG)
        self.assertEqual(out["mode"], "same_user")
        self.assertIn("cannot be read", out["facts"]["reasons"][-1])

    def test_open_project_op_takes_the_requirement(self):
        k = self.keeper()
        pid = project_id("op")
        line = json.dumps({"op": "open_project", "project": pid, "args": {"require_keeper_mode": "separate_user"}})
        self.assertTrue(k.handle(line.encode(), STRONG)["ok"])
        resp = k.handle(line.encode(), KEEPER)                       # as the keeper's own uid: the project exists
        self.assertTrue(resp["ok"])                                  # reopening with the same value is fine
        bad = json.dumps({"op": "open_project", "project": project_id("op2"),
                          "args": {"require_keeper_mode": "separate_user"}})
        resp = k.handle(bad.encode(), KEEPER)
        self.assertEqual((resp["ok"], resp["problems"][0]["rule"]), (False, "AN-28"))


@skip_unless_linux
class EnvironmentRecordTest(unittest.TestCase):
    def test_the_signed_environment_record_states_the_mode(self):
        tmp = Path(tempfile.mkdtemp(prefix="gk-mode-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        state = tmp / "state"
        state.mkdir(mode=0o700)
        view = rootful.HostView(root=write_host_tree(tmp / "host", owner_uid=OWNER), root_uid=KEEPER)
        caught = []
        with mock.patch.object(server, "_serve_loop", lambda keeper, path, stop: caught.append(keeper) or 0):
            server.serve(state, tmp / "k.sock", OWNER, host_view=view)
        record = caught[0].environment["record"]
        self.assertEqual(record["mode"], "separate_user")
        self.assertEqual(set(record["mode_facts"]), {"keeper_euid", "owner_uid", "state_dir_uid", "state_dir_mode",
                                                     "reasons"})
        self.assertNotIn("peer_uid", record["mode_facts"])
        caught.clear()
        state2 = tmp / "state2"
        state2.mkdir(mode=0o700)
        with mock.patch.object(server, "_serve_loop", lambda keeper, path, stop: caught.append(keeper) or 0):
            server.serve(state2, tmp / "k2.sock", KEEPER, host_view=view)
        self.assertEqual(caught[0].environment["record"]["mode"], "same_user")


if __name__ == "__main__":
    unittest.main()
