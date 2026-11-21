"""AN-8, AN-9, AN-10, AN-14, AN-15: the driver side of the keeper. `KeeperTrust` and `KeeperAlphaLedger` implement the
step 1 interfaces as keeper requests, `machine.backends` picks them from the project's binding, and the five trust
stages branch on `ctx.trust.mode` (tests/README.md section 13).

The keeper runs as a subprocess of the test user (`tests/_keeper.py`); the synthetic pack is injected in this process
too (`inject_pack`), so the pipeline's own stages run here. A second candidate in one project comes from the test-only
`train` decision key `noise_override` (preflight finding 1). Verified statuses on `toy.jsonl` (exact match, margin
0.05, n_sealed 399): `noise_override` 1.0 (every answer wrong, like the empty base) is INCONCLUSIVE (zero-width
interval); the noise-free table is PASS.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from _keeper import KeeperProcess, skip_unless_linux
from _support import FIXTURES

from test_an_synthetic_pack import fixture_rows, inject_pack, jsonl
from gatedtrain.ainative import alpha_investing as ai, machine, scorers, signing, trust_stages
from gatedtrain.ainative.keeper import protocol, texts
from gatedtrain.ainative.keeper_trust import (KEEPER_BINDING, KEEPER_PUB, RECEIPTS, KeeperAlphaLedger, KeeperTrust,
                                              bind_project, project_id)
from gatedtrain.ainative.machine import backends, run_stage, status
from gatedtrain.ainative.project import create_project
from gatedtrain.ainative.stage import stage_registry
from gatedtrain.ainative.trust import UNSEALED_GUARANTEE, FixedAlphaLedger, InProcessTrust
from gatedtrain.errors import Refused
from gatedtrain.testing import synthetic_pack as pack

TOY = (FIXTURES / "ainative" / "toy.jsonl").read_bytes()
MARGIN = 0.05
BID2 = 0.011586107554417413
FAIL_COST = 0.024327784891165175
RECEIPT_KEYS = {"schema", "schema_version", "arm", "held_by", "attempt_id", "registration_hash", "item_set_hash",
                "model_ref_sha256", "claim_sha256"}
UPTO_TRAIN = ("intake_seal", "profile_data", "choose_task_and_base", "preregister", "build_train_set", "write_recipe",
              "smoke", "train")


def rule_of(exc: Refused) -> str:
    return exc.problems[0].rule


class KeeperProject:
    """A project on `toy.jsonl` (outside the project root) bound to a keeper subprocess; a test mixes it in."""

    env: dict[str, str] = {}

    def start(self, env: dict | None = None) -> None:
        inject_pack(self)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.kp = KeeperProcess(env=env if env is not None else self.env)
        self.kp.__enter__()
        self.addCleanup(self.kp.__exit__, None, None, None)

    def make(self, name: str = "proj", data: bytes = TOY, *, bind: bool = True):
        path = self.tmp / f"{name}.jsonl"
        path.write_bytes(data)
        proj = create_project(self.tmp / name, goal="lookup", data=path, pipeline=str(pack.SCENARIO.pipeline),
                              budget={"gpu_hours": 1, "llm_usd": 1})
        if bind:
            bind_project(proj, self.kp.socket)
        return proj

    def go(self, proj, ref: str, decision: dict | None = None):
        if decision is None:
            decision = {"intake_seal": {"accept_inconclusive": True},
                        "preregister": {"metric_confirmed": True, "margin": MARGIN}}.get(ref, {})
        return run_stage(proj, ref, decision, registry=stage_registry())

    def confirm(self, proj, margin: float = MARGIN) -> dict:
        spec = scorers.resolve("exact_match")
        return KeeperTrust(proj).confirm_metric(
            {"scorer": "exact_match", "scorer_params": {}, "direction": spec.direction, "margin": margin},
            f"I will count {spec.what}; the new model must beat the base model by more than {margin * 100:g} points.")

    def upto_train(self, proj, train: dict | None = None) -> None:
        for ref in UPTO_TRAIN:
            if ref == "preregister":
                self.confirm(proj)
            self.go(proj, ref, train if ref == "train" and train is not None else None)

    def read(self, proj, stage: str, name: str) -> dict:
        return proj.read_artifact(machine._recorded_outputs(proj, machine._states(
            proj, machine._load_spec(proj), stage_registry())[stage])[name])

    def public(self):
        return signing.load_public_pem(self.kp.client.call("hello", None, {})["public_pem"].encode("ascii"))

    def receipts(self, proj) -> list[dict]:
        pub = self.public()
        lines = (proj.root / RECEIPTS).read_bytes().decode("utf-8").splitlines()
        return [{"kind": e["kind"], "record": signing.verify(e, pub), "sha256": signing.envelope_sha256(e)}
                for e in (json.loads(x) for x in lines)]

    def sealed_inputs(self, proj) -> set[str]:
        train = {r["input"] for r in self.read(proj, "intake_seal", "train_part")["rows"]}
        every = {json.loads(line)["input"] for line in (self.tmp / "proj.jsonl").read_text().splitlines() if line}
        return every - train

    def assert_no_sealed_input(self, proj, sealed: set[str]) -> None:
        self.assertTrue(sealed)
        quoted = [json.dumps(s) for s in sealed]
        for path in proj.root.rglob("*"):
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="replace")
                hit = [q for q in quoted if q in text]
                self.assertEqual(hit, [], f"{path.relative_to(proj.root)} holds a sealed input")


class Step1PairTest(unittest.TestCase):
    def test_no_binding_gives_the_step1_pair(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / "d.jsonl"
            data.write_bytes(jsonl(fixture_rows()))
            proj = create_project(Path(d) / "p", goal="g", data=data, pipeline=str(pack.SCENARIO.pipeline),
                                  budget={"gpu_hours": 1, "llm_usd": 1})
            trust, ledger = backends(proj)
            self.assertIsInstance(trust, InProcessTrust)
            self.assertIsInstance(ledger, FixedAlphaLedger)
            self.assertEqual(trust.mode, "in_process")


@skip_unless_linux
class KeeperPipelineTest(KeeperProject, unittest.TestCase):
    def setUp(self):
        self.start()

    def test_two_attempts_through_the_stage_machine(self):         # AN-8, AN-9, AN-10, AN-15
        proj = self.make()
        trust, ledger = backends(proj)
        self.assertIsInstance(trust, KeeperTrust)
        self.assertIsInstance(ledger, KeeperAlphaLedger)
        self.assertEqual((trust.mode, ledger.planning_alpha), ("keeper", 0.02375))
        self.assertEqual(trust.sealed_guarantee, texts.KEEPER_SAME_USER_GUARANTEE)
        binding = json.loads((proj.root / KEEPER_BINDING).read_text())
        self.assertEqual(set(binding), {"schema", "schema_version", "socket", "fingerprint", "project_id"})
        self.assertEqual((binding["schema"], binding["schema_version"], binding["project_id"]),
                         ("gatedtrain.ainative.keeper_binding", 1, project_id(proj)))
        self.assertEqual(signing.fingerprint(signing.load_public_pem((proj.root / KEEPER_PUB).read_bytes())),
                         binding["fingerprint"])

        self.upto_train(proj, {"noise_override": 1.0})
        for name in ("train_part", "sealed_summary", "power"):
            self.assertEqual(self.read(proj, "intake_seal", name)["schema_version"], 2)
        self.assertEqual(self.read(proj, "intake_seal", "power")["alpha"], 0.02375)
        reg = self.read(proj, "preregister", "registration")
        self.assertEqual((reg["schema_version"], reg["alpha"]), (2, 0.02375))
        full = KeeperTrust(proj).registration(reg["hash"])
        self.assertEqual(full["clause"]["alpha"], 0.02375)
        self.assertIn(ai.RULE, full["signed_by"])                       # the rule id (preflight finding 13)
        reg_receipts = [r for r in self.receipts(proj) if r["kind"] == "registration"]
        self.assertEqual(reg["envelope_sha256"], reg_receipts[-1]["sha256"])

        self.go(proj, "evaluate_sealed")
        base = self.read(proj, "evaluate_sealed", "baseline_records")
        cand = self.read(proj, "evaluate_sealed", "candidate_records")
        for arm, rec in (("baseline", base), ("candidate", cand)):
            self.assertEqual(set(rec), RECEIPT_KEYS)
            self.assertEqual((rec["arm"], rec["held_by"], rec["schema_version"]), (arm, "keeper", 2))
        self.assertEqual(base["claim_sha256"], cand["claim_sha256"])
        claims = [r for r in self.receipts(proj) if r["kind"] == "claim"]
        self.assertEqual(cand["claim_sha256"], claims[-1]["sha256"])
        self.assertEqual(claims[-1]["record"]["candidate_model_sha256"], cand["model_ref_sha256"])

        self.go(proj, "gate")
        g1 = self.read(proj, "gate", "gate_result")
        self.assertEqual(g1["status"], "INCONCLUSIVE")                  # verified: noise 1.0 against the empty base
        self.assertEqual((g1["alpha"], g1["alpha_earned"], g1["schema_version"]), (0.02375, 0.0, 2))
        self.assertAlmostEqual(g1["alpha_spent"], FAIL_COST)
        self.assertNotIn("delta_hat_natural", g1)
        self.assertNotIn("delta_hat", g1)
        summaries = [r for r in self.receipts(proj) if r["kind"] == "gate_summary"]
        self.assertEqual(g1["envelope_sha256"], summaries[-1]["sha256"])

        self.go(proj, "train", {"rerun": True})                        # a new candidate: the pipeline's noise 0
        self.assertEqual(status(proj)["evaluate_sealed"]["status"], "stale")
        self.go(proj, "evaluate_sealed")
        self.go(proj, "gate")
        g2 = self.read(proj, "gate", "gate_result")
        self.assertEqual(g2["status"], "PASS")
        self.assertEqual(g2["alpha"], BID2)
        self.assertNotEqual(g2["attempt_id"], g1["attempt_id"])
        entries = KeeperAlphaLedger(KeeperTrust(proj)).entries()
        self.assertEqual([(e["alpha"], e["passed"]) for e in entries], [(0.02375, False), (BID2, True)])
        self.assertEqual(set(entries[0]), {"attempt_id", "alpha", "passed", "alpha_spent", "alpha_earned",
                                           "wealth_after"})

        self.go(proj, "report")
        self.assertTrue(all(s["status"] == "done" for s in status(proj).values()), status(proj))
        report = self.read(proj, "report", "report")
        self.assertEqual((report["schema_version"], report["status"]), (2, "PASS"))
        self.assertEqual([a["alpha"] for a in report["attempts"]], [0.02375, BID2])
        self.assertEqual(report["sealed_guarantee"], texts.KEEPER_SAME_USER_GUARANTEE)
        self.assertEqual(report["keeper"]["fingerprint"], binding["fingerprint"])
        md = (proj.root / "stages" / "report" / "report.md").read_text()
        self.assertEqual(md, trust_stages.render_markdown(report))
        self.assertIn(ai.GUARANTEE, md)
        self.assertEqual([r for r in self.receipts(proj) if r["kind"] == "report"][-1]["record"], report)
        self.assertNotIn(UNSEALED_GUARANTEE, md)
        self.assertFalse((proj.root / "trust" / "sealed").exists())
        self.assert_no_sealed_input(proj, self.sealed_inputs(proj))

    def test_registration_plan_states_alpha_investing(self):       # preflight finding 13
        proj = self.make()
        for ref in UPTO_TRAIN[:3]:
            self.go(proj, ref)
        prop = machine.plan_stage(proj, "preregister")
        self.assertEqual(prop.choices["alpha"], 0.02375)
        self.assertIn("alpha-investing", prop.summary)
        with self.assertRaises(Refused) as cm:                       # no confirmation in the keeper: AN-18
            self.go(proj, "preregister", {"metric_confirmed": True, "margin": MARGIN})
        self.assertEqual(rule_of(cm.exception), "AN-18")
        self.assertEqual(proj.read_attempts()[-1]["event"], "refused")
        self.confirm(proj)
        with self.assertRaises(Refused) as cm:                       # the user's decision is still required
            self.go(proj, "preregister", {"margin": MARGIN})
        self.assertEqual(rule_of(cm.exception), "AN-7")
        self.go(proj, "preregister", {"metric_confirmed": True, "margin": MARGIN})

    def test_ledger_and_trust_refuse_driver_side_sealed_operations(self):   # AN-8, AN-15
        proj = self.make()
        trust, ledger = backends(proj)
        for call in (lambda: ledger.spend("a"), lambda: ledger.record("a", True)):
            with self.assertRaises(Refused) as cm:
                call()
            self.assertEqual(rule_of(cm.exception), "AN-15")
        for call in (lambda: trust.seal([{"input": "x", "label": "y"}]),
                     lambda: trust.run_sealed(trust_stages.SealedRef("sha256:" + "0" * 64), lambda item: {}),
                     lambda: trust.claim_sealed_evaluation("sha256:" + "0" * 64, "sha256:" + "1" * 64)):
            with self.assertRaises(Refused) as cm:
                call()
            self.assertEqual(rule_of(cm.exception), "AN-8")

    def test_intake_forwards_accept_inconclusive(self):             # Task 8 carry
        proj = self.make(data=jsonl(fixture_rows(100)))
        with self.assertRaises(Refused) as cm:
            self.go(proj, "intake_seal", {})
        self.assertEqual(rule_of(cm.exception), "AN-6")
        self.assertEqual(proj.read_attempts()[-1]["event"], "refused")
        self.assertFalse((proj.root / "stages" / "intake_seal").exists())
        self.go(proj, "intake_seal", {"accept_inconclusive": True})
        self.assertFalse(self.read(proj, "intake_seal", "power")["conclusive_expected"])


@skip_unless_linux
class KeeperBindingTest(KeeperProject, unittest.TestCase):
    def setUp(self):
        self.start()

    def _new_state(self) -> None:
        self.kp.stop()
        shutil.rmtree(self.kp.state)
        self.kp.state.mkdir(mode=0o700)
        self.kp.start()

    def test_a_keeper_with_another_key_is_refused(self):           # AN-14
        proj = self.make()
        self.go(proj, "intake_seal")
        self._new_state()                                            # same socket, new state, new key
        with self.assertRaises(Refused) as cm:
            self.go(proj, "profile_data")
        self.assertEqual(rule_of(cm.exception), "AN-14")
        with self.assertRaises(Refused) as cm:
            backends(proj)
        self.assertEqual(rule_of(cm.exception), "AN-14")
        with self.assertRaises(Refused) as cm:
            bind_project(proj, self.kp.socket)                       # a binding to another key is never replaced
        self.assertEqual(rule_of(cm.exception), "AN-14")

    def test_deleting_the_binding_never_falls_back(self):          # AN-14, preflight finding 11
        proj = self.make()
        self.upto_train(proj)
        for ref in ("evaluate_sealed", "gate"):
            self.go(proj, ref)
        (proj.root / KEEPER_BINDING).unlink()
        with self.assertRaises(Refused) as cm:
            backends(proj)
        self.assertEqual(rule_of(cm.exception), "AN-14")
        rows = len(proj.read_attempts())
        with self.assertRaises(Refused) as cm:
            self.go(proj, "report")
        self.assertEqual(rule_of(cm.exception), "AN-14")
        self.assertEqual(len(proj.read_attempts()), rows)            # refused before the attempt started
        self.assertNotEqual(status(proj)["report"]["status"], "done")
        self.assertFalse((proj.root / "stages" / "report").exists())
        for path in proj.root.rglob("*"):
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="replace")
                self.assertNotIn(texts.KEEPER_GUARANTEE, text)

    def test_keeper_records_without_receipts_still_refuse(self):   # schema_version 2 artifacts alone
        proj = self.make()
        self.go(proj, "intake_seal")
        (proj.root / KEEPER_BINDING).unlink()
        self.assertFalse((proj.root / RECEIPTS).exists())
        with self.assertRaises(Refused) as cm:
            backends(proj)
        self.assertEqual(rule_of(cm.exception), "AN-14")

    def test_an_unreachable_keeper_refuses(self):                   # AN-12: never the step 1 pair
        proj = self.make()
        self.kp.stop()
        with self.assertRaises(Refused) as cm:
            backends(proj)
        self.assertEqual(rule_of(cm.exception), "AN-12")
        with self.assertRaises(Refused):
            self.go(proj, "intake_seal")
        self.assertFalse((proj.root / "trust" / "sealed").exists())

    def test_a_replaced_public_key_is_refused(self):                # AN-14: the copy must match the binding
        proj = self.make()
        other = self.make("other", jsonl(fixture_rows()), bind=False)
        os.makedirs(other.root / "trust")
        for rel in (KEEPER_BINDING, KEEPER_PUB):                     # a binding copied from another project
            shutil.copy(proj.root / rel, other.root / rel)
        with self.assertRaises(Refused) as cm:
            backends(other)
        self.assertEqual(rule_of(cm.exception), "AN-14")
        (proj.root / KEEPER_PUB).write_bytes(signing.public_pem(Ed25519PrivateKey.generate()))
        with self.assertRaises(Refused) as cm:
            backends(proj)
        self.assertEqual(rule_of(cm.exception), "AN-14")


@skip_unless_linux
class RefusalAfterSpendTest(KeeperProject, unittest.TestCase):
    env = {"GATEDTRAIN_TEST_GENERATOR": "raise"}

    def setUp(self):
        self.start()

    def test_a_refusal_after_the_grant_is_a_failed_attempt(self):  # preflight finding 9
        proj = self.make()
        self.upto_train(proj)
        with self.assertRaises(Refused) as cm:
            self.go(proj, "evaluate_sealed")
        self.assertEqual(rule_of(cm.exception), "AN-8")
        self.assertEqual(proj.read_attempts()[-1]["event"], "failed")
        self.assertEqual(status(proj)["evaluate_sealed"]["status"], "failed")
        kinds = [r["kind"] for r in self.receipts(proj)]
        self.assertEqual(kinds[-4:], ["claim", "ledger_entry", "gate_summary", "ledger_entry"])
        sealed = self.sealed_inputs(proj)
        text = " ".join(f"{p.field} {p.message} {p.fix}" for p in cm.exception.problems)
        self.assertFalse([s for s in sealed if s in text])
        self.assert_no_sealed_input(proj, sealed)
        self.assertAlmostEqual(KeeperAlphaLedger(KeeperTrust(proj)).available(), BID2)


@skip_unless_linux
class BusyKeeperTest(KeeperProject, unittest.TestCase):
    """The key check (`hello`) before each op waits behind another client's long op with the long-op deadline: the
    wait is queueing, not work (Task 10 review). Every candidate call sleeps 5 ms in the keeper, so an evaluation
    takes a few seconds; the client's short deadline is patched to 0.5 s in this process (the keeper subprocess keeps
    its own)."""

    env = {"GATEDTRAIN_TEST_GENERATOR": "slow:0.005"}

    def setUp(self):
        self.start()
        patcher = mock.patch.object(protocol, "SOCKET_TIMEOUT_S", 0.5)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_key_check_waits_behind_a_long_op(self):
        a = self.make("a")
        b = self.make("b", jsonl([{**r, "input": f"b{r['input']}"} for r in fixture_rows()]))   # no toy input
        self.upto_train(a, {"noise_override": 1.0})                  # not a PASS, so a second attempt is allowed
        errors: list[BaseException] = []

        def evaluate() -> None:
            try:
                self.go(a, "evaluate_sealed")
            except BaseException as exc:                             # noqa: BLE001  (reported below)
                errors.append(exc)

        worker = threading.Thread(target=evaluate)
        worker.start()
        self.addCleanup(worker.join, 120)
        time.sleep(1.0)                                              # the keeper is scoring a's candidate
        self.assertTrue(worker.is_alive(), "the evaluation ended before the second client called")
        began = time.monotonic()
        KeeperTrust(b).check()                                       # queued behind a's evaluate
        self.assertGreater(time.monotonic() - began, 0.5)
        worker.join(120)
        self.assertEqual(errors, [])
        self.assertEqual(status(a)["evaluate_sealed"]["status"], "done")

        worker = threading.Thread(target=evaluate)                   # a stage run of b, queued the same way
        self.go(a, "train", {"rerun": True})
        worker.start()
        time.sleep(1.0)
        self.assertTrue(worker.is_alive())
        self.go(b, "intake_seal")
        worker.join(120)
        self.assertEqual(errors, [])
        self.assertEqual(status(b)["intake_seal"]["status"], "done")

    def test_a_stopped_keeper_still_fails_fast(self):
        proj = self.make()
        self.kp.stop()
        began = time.monotonic()
        with self.assertRaises(Refused) as cm:
            KeeperTrust(proj).check()
        self.assertEqual(rule_of(cm.exception), "AN-12")
        self.assertLess(time.monotonic() - began, 5.0)


if __name__ == "__main__":
    unittest.main()
