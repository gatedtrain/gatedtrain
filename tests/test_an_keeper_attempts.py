"""AN-8, AN-9, AN-10, AN-13, AN-14, AN-15, AN-16: keeper attempts: each sealed evaluation spends alpha first, the
keeper gates, settles the ledger and builds the signed report; a report closes the sealed set (tests/README.md
section 13).

The keeper runs as a subprocess of the test user (`tests/_keeper.py`). Candidates are model records built here from
the training rows the keeper returned, so several candidates fit in one project without a pipeline.
Verified statuses of the fixtures on `toy.jsonl` (exact match, margin 0.05, n_sealed 399): `noise=1.0`
(every answer wrong, like the base) is INCONCLUSIVE (zero-width interval), a table that knows one token is FAIL at
each of the first three bids, the noise-free table is PASS.

The generator switch `GATEDTRAIN_TEST_GENERATOR` (`tests/fixtures/ainative/run_cli.py`) counts, raises, blocks or
slows the generator inside the keeper subprocess.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

from _keeper import KeeperProcess, skip_unless_linux
from _support import FIXTURES

from test_an_keeper_intake import confirm_args, intake_args, project_id
from gatedtrain.ainative import alpha_investing as ai, scorers, signing, trust_stages
from gatedtrain.ainative.alpha_investing import PLANNING_ALPHA
from gatedtrain.ainative.keeper import texts
from gatedtrain.ainative.keeper.protocol import KeeperClient, RefusedAfterSpend
from gatedtrain.ainative.keeper.state import KeeperState
from gatedtrain.canonical import canonical_json, sha256_bytes
from gatedtrain.errors import Refused
from gatedtrain.testing import synthetic_pack as pack

TOY = (FIXTURES / "ainative" / "toy.jsonl").read_bytes()
NEW_DATA_FIX = "a new test needs new labelled data"
PASSED_FIX = "this project passed; write its report"
DEP = {"sampler": dict(pack.SAMPLER), "eval_seeds": list(pack.EVAL_SEEDS)}
BASE = {"schema": trust_stages.SCHEMA_MODEL_REF, "schema_version": 1, "generator": pack.GENERATOR, "label": "base",
        "table": {}}
BASE_SHA = sha256_bytes(canonical_json(BASE))
W0 = 0.0475
FAIL_COST = 0.024327784891165175
BID2 = 0.011586107554417413
BID3 = 0.005725148070253095
SUMMARY_KEYS = {"attempt_id", "j", "project_id", "sealed_id", "registration_hash", "status", "alpha", "alpha_spent",
                "alpha_earned", "wealth_after", "post_hoc"}
EFFECT_KEYS = ("delta_hat", "ci_lower", "ci_upper", "achieved_power", "p_value", "candidate_score", "baseline_score")


def arm(record: dict) -> dict:
    return {"record": record, "sha256": sha256_bytes(canonical_json(record))}


def model(table: dict, variant: str) -> dict:
    return {**BASE, "label": "candidate", "variant": variant, "table": table}


def registration(sealed_id: str, *, margin: float = 0.05) -> dict:
    spec = scorers.resolve("exact_match")
    v = {"scorer": "exact_match", "direction": spec.direction, "margin": margin, "alpha": PLANNING_ALPHA}
    return trust_stages._registration(v, sealed_id, DEP, BASE_SHA, ("synthetic", pack.GENERATOR),
                                      scorers.evaluator_entry(spec, {}))


class Attempts:
    """Helpers over one keeper subprocess; a test case mixes it in and sets `self.kp`."""

    env: dict[str, str] = {}

    def start(self, env: dict | None = None) -> None:
        self.kp = KeeperProcess(env=env if env is not None else self.env)
        self.kp.__enter__()
        self.addCleanup(self.kp.__exit__, None, None, None)
        self.train: dict[str, list] = {}
        self.sealed: dict[str, str] = {}
        self.reg: dict[str, str] = {}

    @property
    def client(self) -> KeeperClient:
        return self.kp.client

    def call(self, op: str, pid: str | None, args: dict | None = None):
        return self.client.call(op, pid, args or {})

    def refused(self, op: str, pid: str | None, args: dict | None = None):
        with self.assertRaises(Refused) as cm:
            self.call(op, pid, args)
        return cm.exception

    def public(self):
        return signing.load_public_pem(self.call("hello", None)["public_pem"].encode("ascii"))

    def ready(self, name: str, *, margin: float = 0.05) -> str:
        pid = project_id(name)
        self.call("open_project", pid)
        r = self.call("intake", pid, intake_args(TOY))
        self.train[pid], self.sealed[pid] = r["train_rows"], r["sealed_id"]
        self.call("confirm_metric", pid, confirm_args(margin=margin))
        reg = registration(r["sealed_id"], margin=margin)
        self.reg[pid] = self.call("register", pid, {"registration": reg})["hash"]
        return pid

    # candidates (statuses verified on toy.jsonl, see the module docstring)
    def noisy(self, pid: str, noise: float) -> dict:
        return model(pack.fit(self.train[pid], noise), f"noise-{noise}")

    def one_token(self, pid: str, k: int = 0) -> dict:          # FAIL: beats the base by less than the margin
        full = pack.fit(self.train[pid], 0.0)
        tok = sorted(full)[k]
        return model({tok: full[tok]}, f"one-token-{k}")

    def evaluate(self, pid: str, candidate: dict, *, baseline: dict = BASE, reg: str | None = None) -> dict:
        return self.call("evaluate", pid, {"registration_hash": reg or self.reg[pid], "baseline": arm(baseline),
                                           "candidate": arm(candidate)})

    def gate(self, pid: str, attempt: dict | str) -> dict:
        aid = attempt if isinstance(attempt, str) else attempt["attempt_id"]
        out = self.call("gate", pid, {"attempt_id": aid})
        self.assertEqual(set(out), {"summary", "ledger"})
        return signing.verify(out["summary"], self.public(), kind="gate_summary")

    def alpha(self, pid: str) -> dict:
        return self.call("alpha", pid)

    def ledger_records(self, pid: str) -> list[dict]:
        pub = self.public()
        return [signing.verify(e, pub, kind="ledger_entry") for e in self.alpha(pid)["entries"]]


@skip_unless_linux
class KeeperAttemptsTest(Attempts, unittest.TestCase):
    def setUp(self):
        self.start()

    def test_each_evaluation_spends_first(self):               # AN-8, AN-15
        pid = self.ready("p1")
        a = self.alpha(pid)
        self.assertEqual((a["sealed_id"], a["rule"], a["planning_alpha"]), (self.sealed[pid], "foster-stine", 0.02375))
        self.assertEqual((a["wealth"], a["available"], a["entries"]), (W0, 0.02375, []))
        a1 = self.evaluate(pid, self.noisy(pid, 1.0))
        self.assertEqual(set(a1), {"attempt_id", "j", "alpha", "claim", "ledger"})
        self.assertAlmostEqual(a1["alpha"], 0.02375)
        self.assertEqual((a1["j"], a1["attempt_id"]), (1, f"{self.sealed[pid].removeprefix('sha256:')[:16]}.1"))
        pub = self.public()
        claim = signing.verify(a1["claim"], pub, kind="claim")
        self.assertEqual(claim["candidate_model_sha256"], arm(self.noisy(pid, 1.0))["sha256"])
        self.assertEqual((claim["registration_hash"], claim["alpha"], claim["attempt_id"]),
                         (self.reg[pid], 0.02375, a1["attempt_id"]))
        grant = signing.verify(a1["ledger"], pub, kind="ledger_entry")
        self.assertEqual((grant["entry"], grant["alpha"], grant["wealth_before"]), ("grant", 0.02375, W0))
        open_ = self.alpha(pid)                                  # an open attempt counts as not passed
        self.assertAlmostEqual(open_["wealth"], W0 - FAIL_COST)
        self.assertAlmostEqual(open_["available"], BID2)
        s1 = self.gate(pid, a1)
        self.assertEqual(s1["status"], "INCONCLUSIVE")           # verified: noise 1.0 against the empty base
        self.assertEqual((s1["alpha"], s1["alpha_earned"]), (0.02375, 0.0))
        self.assertAlmostEqual(s1["alpha_spent"], FAIL_COST)
        self.assertEqual(self.gate(pid, a1), s1)                 # computed once
        a2 = self.evaluate(pid, self.one_token(pid))
        self.assertAlmostEqual(a2["alpha"], BID2)
        self.assertEqual(a2["j"], 2)
        self.assertEqual(self.gate(pid, a2)["status"], "FAIL")   # verified: one known token
        rows = self.ledger_records(pid)
        self.assertEqual([r["entry"] for r in rows], ["grant", "settle", "grant", "settle"])
        self.assertAlmostEqual(self.alpha(pid)["available"], BID3)

    def test_refusals_before_the_spend(self):                   # AN-8: nothing granted, nothing scored
        pid = self.ready("p1")
        cand = self.one_token(pid)
        good = {"registration_hash": self.reg[pid], "baseline": arm(BASE), "candidate": arm(cand)}
        other_base = {**BASE, "variant": "other"}
        stale = {**arm(cand), "sha256": "sha256:" + "0" * 64}
        foreign = {**cand, "generator": "other.generator"}
        cases = {
            "registration": {**good, "registration_hash": "sha256:" + "1" * 64},
            "baseline": {**good, "baseline": arm(other_base)},
            "same": {**good, "candidate": arm(BASE)},
            "sha": {**good, "candidate": stale},
            "generator": {**good, "candidate": arm(foreign)},
            "shape": {**good, "candidate": {"record": cand}},
        }
        for name, args in cases.items():
            with self.subTest(name):
                exc = self.refused("evaluate", pid, args)
                self.assertEqual(exc.problems[0].rule, "AN-8")
                self.assertNotIsInstance(exc, RefusedAfterSpend)
        self.assertEqual(self.alpha(pid)["entries"], [])
        self.assertEqual([e["kind"] for e in self.call("records", pid)], ["metric_confirmation", "registration"])

    def test_fourth_failing_attempt_refused(self):             # AN-15
        pid = self.ready("p1")
        alphas = []
        for k in range(3):
            a = self.evaluate(pid, self.one_token(pid, k))
            alphas.append(a["alpha"])
            self.assertEqual(self.gate(pid, a)["status"], "FAIL")
        self.assertEqual(alphas, [0.02375, BID2, BID3])
        before = self.alpha(pid)
        self.assertEqual(before["available"], 0.0)
        exc = self.refused("evaluate", pid, {"registration_hash": self.reg[pid], "baseline": arm(BASE),
                                             "candidate": arm(self.one_token(pid, 3))})
        self.assertEqual((exc.problems[0].rule, exc.problems[0].fix), ("AN-15", NEW_DATA_FIX))
        self.assertEqual(self.alpha(pid), before)                # nothing granted

    def test_pass_earns_back(self):                            # AN-15
        pid = self.ready("p1")
        self.gate(pid, self.evaluate(pid, self.one_token(pid)))
        a2 = self.evaluate(pid, self.noisy(pid, 0.0))
        s2 = self.gate(pid, a2)
        self.assertEqual(s2["status"], "PASS")
        self.assertEqual((s2["alpha"], s2["alpha_spent"], s2["alpha_earned"]), (a2["alpha"], 0.0, 0.05))
        self.assertAlmostEqual(s2["wealth_after"], 0.0975 - FAIL_COST)
        self.assertAlmostEqual(self.alpha(pid)["wealth"], 0.0975 - FAIL_COST)
        exc = self.refused("evaluate", pid, {"registration_hash": self.reg[pid], "baseline": arm(BASE),
                                             "candidate": arm(self.one_token(pid, 5))})
        self.assertEqual((exc.problems[0].rule, exc.problems[0].fix), ("AN-9", PASSED_FIX))
        # another project on the same data shares the wealth and may still test
        p2 = self.ready("p2")
        self.assertAlmostEqual(self.evaluate(p2, self.one_token(p2, 6))["alpha"], (0.0975 - FAIL_COST) / 2)

    def test_identical_request_returns_the_stored_claim(self):
        pid = self.ready("p1")
        a1 = self.evaluate(pid, self.one_token(pid))
        self.assertEqual(self.evaluate(pid, self.one_token(pid)), a1)
        s1 = self.gate(pid, a1)
        self.assertEqual(self.evaluate(pid, self.one_token(pid)), a1)
        self.assertEqual(self.gate(pid, a1), s1)
        self.assertEqual(len(self.alpha(pid)["entries"]), 2)

    def test_driver_sees_no_effect(self):                      # AN-9
        pid = self.ready("p1")
        a1 = self.evaluate(pid, self.noisy(pid, 0.0))
        out = self.call("gate", pid, {"attempt_id": a1["attempt_id"]})
        summary = signing.verify(out["summary"], self.public(), kind="gate_summary")
        self.assertEqual(set(summary), SUMMARY_KEYS)
        settle = signing.verify(out["ledger"], self.public(), kind="ledger_entry")
        self.assertEqual(set(settle), {"entry", "sealed_id", "attempt_id", "j", "project_id", "alpha", "wealth_before",
                                       "passed", "cost", "earned", "wealth_after", "utc"})
        records = self.call("records", pid)
        self.assertNotIn("gate_record", [e["kind"] for e in records])
        seen = json.dumps([a1, out, records, self.alpha(pid)])
        for key in EFFECT_KEYS:
            self.assertNotIn(f'"{key}"', seen)
        with self.assertRaises(Refused) as cm:
            self.call("gate", project_id("other"), {"attempt_id": a1["attempt_id"]})
        self.assertIn(cm.exception.problems[0].rule, {"AN-9", "AN-12"})
        p2 = self.ready("p2")
        exc = self.refused("gate", p2, {"attempt_id": a1["attempt_id"]})   # another project's attempt
        self.assertEqual(exc.problems[0].rule, "AN-9")

    def test_registration_changed_after_claim_needs_post_hoc(self):    # AN-8
        pid = self.ready("p1")
        first = self.reg[pid]
        self.gate(pid, self.evaluate(pid, self.one_token(pid)))
        self.call("confirm_metric", pid, confirm_args(margin=0.1))
        changed = registration(self.sealed[pid], margin=0.1)
        exc = self.refused("register", pid, {"registration": changed})
        self.assertEqual(exc.problems[0].rule, "AN-8")
        amended = {**changed, "post_hoc": True, "amends": first, "created_at": "2026-11-21T00:00:00Z"}
        new = self.call("register", pid, {"registration": amended})["hash"]
        exc = self.refused("evaluate", pid, {"registration_hash": first, "baseline": arm(BASE),
                                             "candidate": arm(self.noisy(pid, 0.0))})
        self.assertEqual((exc.problems[0].rule, exc.problems[0].field), ("AN-8", "registration_hash"))
        s = self.gate(pid, self.evaluate(pid, self.noisy(pid, 0.0), reg=new))
        self.assertIs(s["post_hoc"], True)
        self.assertEqual(s["status"], "INCONCLUSIVE")            # a post hoc PASS is not a PASS
        self.assertAlmostEqual(s["alpha_spent"], BID2 / (1 - BID2))

    def test_report_closes_the_set(self):                      # AN-16, AN-10
        p1, p2 = self.ready("p1"), self.ready("p2")
        exc = self.refused("report", p1)
        self.assertEqual(exc.problems[0].rule, "AN-10")          # no settled attempt yet
        self.gate(p1, self.evaluate(p1, self.one_token(p1)))
        r1 = self.call("report", p1)
        self.assertEqual(set(r1), {"report"})
        self.assertEqual(self.call("report", p1), r1)            # the stored envelope
        exc = self.refused("evaluate", p2, {"registration_hash": self.reg[p2], "baseline": arm(BASE),
                                            "candidate": arm(self.one_token(p2, 1))})
        self.assertEqual((exc.problems[0].rule, exc.problems[0].fix), ("AN-16", NEW_DATA_FIX))
        exc = self.refused("evaluate", p1, {"registration_hash": self.reg[p1], "baseline": arm(BASE),
                                            "candidate": arm(self.one_token(p1, 2))})
        self.assertEqual(exc.problems[0].rule, "AN-16")
        p3 = project_id("p3")
        self.call("open_project", p3)
        self.assertEqual(self.refused("intake", p3, intake_args(TOY)).problems[0].rule, "AN-16")
        self.assertEqual(self.refused("report", p2).problems[0].rule, "AN-10")

    def test_report_is_signed_and_from_keeper_records(self):   # AN-10, AN-14
        p1, p2 = self.ready("p1"), self.ready("p2")
        a1 = self.evaluate(p1, self.one_token(p1))
        self.gate(p1, a1)
        b1 = self.evaluate(p2, self.one_token(p2, 1))          # another project's attempt on the same set
        self.gate(p2, b1)
        a2 = self.evaluate(p1, self.noisy(p1, 0.0))            # left open: the report settles it
        env = self.call("report", p1)["report"]
        hello = self.call("hello", None)
        report = signing.verify(env, self.public(), kind="report")
        self.assertEqual((report["schema"], report["schema_version"]), ("gatedtrain.ainative.report", 2))
        self.assertEqual(set(report), {"schema", "schema_version", "status", "metric", "n_sealed", "registration",
                                       "power", "gate_result", "display", "ledger", "attempts", "sealed_guarantee",
                                       "known_limits", "alpha_rule", "keeper", "compute"})
        self.assertEqual(report["status"], "PASS")               # the PASS attempt is the headline
        gate = report["gate_result"]
        self.assertEqual((gate["attempt_id"], gate["alpha"], gate["status"]), (a2["attempt_id"], a2["alpha"], "PASS"))
        self.assertEqual(report["display"]["effect"], round(gate["delta_hat_natural"], 4))
        self.assertEqual(report["metric"], {"scorer": "exact_match", "unit": scorers.resolve("exact_match").unit})
        self.assertEqual(report["registration"]["hash"], self.reg[p1])
        self.assertEqual(report["n_sealed"], 399)
        self.assertEqual(report["power"]["planning_alpha"], 0.02375)
        self.assertEqual([r["attempt_id"] for r in report["ledger"]],
                         [a1["attempt_id"], b1["attempt_id"], a2["attempt_id"]])
        self.assertEqual(set(report["ledger"][0]), {"attempt_id", "j", "project_id", "alpha", "passed", "cost",
                                                    "earned", "wealth_after"})
        self.assertEqual([r["project_id"] for r in report["ledger"]], [p1, p2, p1])
        self.assertEqual([r["alpha"] for r in report["ledger"]], [0.02375, BID2, BID3])
        self.assertEqual([(a["attempt_id"], a["status"]) for a in report["attempts"]],
                         [(a1["attempt_id"], "FAIL"), (a2["attempt_id"], "PASS")])
        self.assertEqual(set(report["attempts"][0]), {"attempt_id", "j", "candidate_model_sha256", "alpha", "status",
                                                      "utc"})
        self.assertEqual([x["id"] for x in report["known_limits"]],
                         ["sealed_guarantee", "original_data", "oracles", "metric_confirmation", "model_reference",
                          "alpha_rule", "sealed_evaluations"])
        limits = {x["id"]: x["text"] for x in report["known_limits"]}
        self.assertEqual(limits["oracles"], texts.LIMIT_ORACLES)
        self.assertEqual(limits["alpha_rule"], texts.LIMIT_ALPHA_RULE)
        self.assertIn("valid given the earlier outcomes", texts.LIMIT_ALPHA_RULE)   # final review 6e
        self.assertEqual(report["known_limits"][0]["text"], texts.KEEPER_SAME_USER_GUARANTEE)
        self.assertEqual(report["known_limits"][-1]["attempts"], 3)
        self.assertEqual(report["alpha_rule"], {"rule": "foster-stine", "initial_wealth": ai.INITIAL_WEALTH,
                                                "payout": 0.05, "min_bid": 0.005, "planning_alpha": 0.02375,
                                                "guarantee": ai.GUARANTEE})
        self.assertEqual(report["keeper"], {"fingerprint": hello["fingerprint"],
                                            "environment_sha256": hello["environment_sha256"]})
        md = trust_stages.render_markdown(report)
        for text in ("alpha spent", "alpha earned", ai.GUARANTEE, hello["fingerprint"], a2["attempt_id"]):
            self.assertIn(text, md)
        # the report is among the project's records, and the keeper confirms it signed it
        self.assertIn(env, self.call("records", p1))
        self.assertIs(self.call("has_record", p1, {"sha256": signing.envelope_sha256(env)}), True)

    def test_pickle_model_file_refused_before_spend(self):     # AN-13
        pid = self.ready("p1")
        tmp = Path(tempfile.mkdtemp(prefix="gk-model-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        pkl = tmp / "model.pkl"
        pkl.write_bytes(b"\x80\x04K\x01.")
        ok = tmp / "model.safetensors"
        ok.write_bytes(b"{}")
        inside = self.kp.state / "model.safetensors"
        inside.write_bytes(b"{}")
        entries = {"pickle": str(pkl), "state": str(inside), "relative": "model.safetensors"}
        for name, path in entries.items():
            with self.subTest(name):
                files = [{"path": path, "sha256": sha256_bytes(Path(path if os.path.isabs(path) else ok).read_bytes())}]
                cand = {**self.one_token(pid), "files": files}
                exc = self.refused("evaluate", pid, {"registration_hash": self.reg[pid], "baseline": arm(BASE),
                                                     "candidate": arm(cand)})
                self.assertEqual(exc.problems[0].rule, "AN-13")
        self.assertEqual(self.alpha(pid)["entries"], [])
        good = {**self.one_token(pid), "files": [{"path": str(ok), "sha256": sha256_bytes(b"{}")}]}
        self.assertEqual(self.evaluate(pid, good)["j"], 1)


@skip_unless_linux
class KeeperGeneratorTest(Attempts, unittest.TestCase):
    """The generator switch of run_cli.py, inside the keeper subprocess."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gk-gen-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_generator_crash_costs_alpha(self):
        for mode in ("raise", "nan"):
            with self.subTest(mode):
                self.start({"GATEDTRAIN_TEST_GENERATOR": mode})
                pid = self.ready("p1")
                with self.assertRaises(RefusedAfterSpend) as cm:
                    self.evaluate(pid, self.one_token(pid))
                p = cm.exception.problems[0]
                self.assertEqual((p.rule, p.field), ("AN-8", "generate"))
                self.assertIn("ValueError", p.message)
                spent = cm.exception.result
                self.assertEqual(set(spent), {"attempt_id", "j", "alpha", "claim", "ledger", "summary", "settle"})
                pub = self.public()
                signing.verify(spent["claim"], pub, kind="claim")
                settle = signing.verify(spent["settle"], pub, kind="ledger_entry")
                self.assertEqual((settle["entry"], settle["passed"]), ("settle", False))
                self.assertAlmostEqual(settle["cost"], FAIL_COST)
                summary = signing.verify(spent["summary"], pub, kind="gate_summary")
                self.assertEqual(summary["status"], "MISSING")
                self.assertAlmostEqual(self.alpha(pid)["wealth"], W0 - FAIL_COST)
                self.assertEqual(self.gate(pid, spent["attempt_id"]), summary)
                sealed = {r["input"] for r in map(json.loads, TOY.decode().splitlines())} - \
                    {r["input"] for r in self.train[pid]}
                seen = json.dumps([p.__dict__ for p in cm.exception.problems]) + json.dumps(spent) + self.kp.output()
                self.assertFalse([s for s in sealed if json.dumps(s) in seen])
                self.kp.__exit__(None, None, None)

    def test_baseline_scored_once_per_set(self):
        counts = self.tmp / "calls"
        self.start({"GATEDTRAIN_TEST_GENERATOR": f"count:{counts}"})
        p1, p2 = self.ready("p1"), self.ready("p2")
        self.gate(p1, self.evaluate(p1, self.one_token(p1, 0)))
        self.gate(p1, self.evaluate(p1, self.one_token(p1, 1)))
        self.gate(p2, self.evaluate(p2, self.one_token(p2, 2)))
        calls = counts.read_bytes()
        per_arm = 399 * len(pack.EVAL_SEEDS)
        self.assertEqual((calls.count(b"b"), calls.count(b"c")), (per_arm, 3 * per_arm))

    def test_concurrent_claims_serialised(self):
        self.start()
        p1, p2 = self.ready("p1"), self.ready("p2")
        cands = {p1: self.one_token(p1, 0), p2: self.one_token(p2, 1)}
        barrier = threading.Barrier(2)
        out: dict[str, object] = {}

        def run(pid: str) -> None:
            barrier.wait()
            try:
                out[pid] = self.evaluate(pid, cands[pid])
            except Exception as exc:                           # reported by the assertion below
                out[pid] = exc

        threads = [threading.Thread(target=run, args=(p,)) for p in (p1, p2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(120)
        self.assertTrue(all(isinstance(v, dict) for v in out.values()), out)
        first, second = sorted(out.values(), key=lambda a: a["j"])
        self.assertEqual((first["j"], second["j"]), (1, 2))
        self.assertEqual(first["alpha"], 0.02375)
        self.assertAlmostEqual(second["alpha"], BID2)             # bid from the wealth after the first settled
        rows = self.ledger_records(p1)
        self.assertEqual([(r["entry"], r["j"]) for r in rows], [("grant", 1), ("settle", 1), ("grant", 2)])
        self.assertEqual(rows[2]["wealth_before"], rows[1]["wealth_after"])

    def test_slow_evaluate_ahead_of_a_second_client(self):      # no silent spend behind a slow op
        self.start({"GATEDTRAIN_TEST_GENERATOR": "slow:0.003"})
        p1, p2 = self.ready("p1"), self.ready("p2")
        done: dict[str, object] = {}
        t = threading.Thread(target=lambda: done.setdefault("a", self.evaluate(p1, self.one_token(p1))))
        t.start()
        time.sleep(0.5)
        short = KeeperClient(self.kp.socket, timeout=0.5)
        with self.assertRaises(Refused) as cm:
            short.call("evaluate", p2, {"registration_hash": self.reg[p2], "baseline": arm(BASE),
                                        "candidate": arm(self.one_token(p2, 1))})
        p = cm.exception.problems[0]
        self.assertEqual((p.rule, p.field), ("AN-12", "response"))
        self.assertIn("one request at a time", p.fix)
        t.join(120)
        self.assertEqual(done["a"]["j"], 1)
        # the request the client gave up on was not served: no second attempt was granted
        self.assertEqual([r["j"] for r in self.ledger_records(p1)], [1])
        self.assertEqual(self.evaluate(p2, self.one_token(p2, 1))["j"], 2)

    def test_killed_between_claim_and_gate(self):
        # kill point 1: after the records are stored -> gated normally after the restart
        self.start()
        pid = self.ready("p1")
        a1 = self.evaluate(pid, self.one_token(pid))
        self.kp.kill()
        self.kp.start()
        self.assert_state_whole()
        s1 = self.gate(pid, a1)
        self.assertEqual(s1["status"], "FAIL")
        self.assertAlmostEqual(s1["alpha_spent"], FAIL_COST)
        # kill point 2: during scoring -> no records, settles not passed (MISSING) on the next gate
        marker = self.tmp / "scoring"
        self.kp.stop()
        self.kp.env = {"GATEDTRAIN_TEST_GENERATOR": f"block:{marker}"}
        self.kp.start()
        t = threading.Thread(target=lambda: self.assertRaises(Refused, self.evaluate, pid, self.one_token(pid, 1)))
        t.start()
        deadline = time.monotonic() + 60
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(marker.exists())
        self.kp.kill()
        t.join(60)
        self.kp.env = {}
        self.kp.start()
        self.assert_state_whole()
        claims = [e["record"] for e in self.call("records", pid) if e["kind"] == "claim"]
        self.assertEqual([c["j"] for c in claims], [1, 2])
        s2 = self.gate(pid, claims[1]["attempt_id"])
        self.assertEqual(s2["status"], "MISSING")
        self.assertAlmostEqual(s2["alpha_spent"], BID2 / (1 - BID2))
        self.assertAlmostEqual(self.alpha(pid)["wealth"], W0 - FAIL_COST - BID2 / (1 - BID2))

    def test_open_attempt_after_a_kill_settles_on_evaluate(self):
        marker = self.tmp / "scoring"
        self.start({"GATEDTRAIN_TEST_GENERATOR": f"block:{marker}"})
        pid = self.ready("p1")
        t = threading.Thread(target=lambda: self.assertRaises(Refused, self.evaluate, pid, self.one_token(pid)))
        t.start()
        deadline = time.monotonic() + 60
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.kp.kill()
        t.join(60)
        self.kp.env = {}
        self.kp.start()
        a2 = self.evaluate(pid, self.one_token(pid, 1))         # settles attempt 1 (not passed) first
        self.assertEqual(a2["j"], 2)
        self.assertAlmostEqual(a2["alpha"], BID2)
        rows = self.ledger_records(pid)
        self.assertEqual([(r["entry"], r["j"]) for r in rows], [("grant", 1), ("settle", 1), ("grant", 2)])
        self.assertIs(rows[1]["passed"], False)

    def assert_state_whole(self) -> None:
        """No temp file left behind and every JSON state file parses."""
        for path in self.kp.state.rglob("*"):
            if path.is_file():
                self.assertNotIn(".tmp", path.name)
                if path.suffix == ".json":
                    json.loads(path.read_bytes())


@skip_unless_linux
class KeeperStateAttemptsTest(unittest.TestCase):
    """`KeeperState` called directly: tampered stored records."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gk-state-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.dir = self.tmp / "state"
        self.dir.mkdir(mode=0o700)
        trust_stages._scenario_entry_points = lambda: [_ScenarioEP()]
        trust_stages._reset_scenario_snapshot()
        self.addCleanup(trust_stages._reset_scenario_snapshot)
        self.addCleanup(setattr, trust_stages, "_scenario_entry_points", _ORIGINAL_EPS)
        self.state = self.open()

    def open(self) -> KeeperState:
        st = KeeperState(self.dir, owner_uid=os.getuid())
        self.addCleanup(st.close)
        return st

    def ready(self) -> tuple[str, dict, str]:
        pid = project_id("p1")
        self.state.open_project(pid)
        r = self.state.intake(pid, TOY, sha256_bytes(TOY))
        args = confirm_args()
        sentence = args.pop("sentence")
        self.state.confirm_metric(pid, args, sentence, os.getuid())
        reg = self.state.register(pid, registration(r["sealed_id"]))["hash"]
        return pid, r, reg

    def candidate(self, r: dict, k: int) -> dict:
        full = pack.fit(r["train_rows"], 0.0)
        tok = sorted(full)[k]
        return arm(model({tok: full[tok]}, f"one-token-{k}"))

    def test_tampered_ledger_refused(self):                    # replay checks every stored number
        pid, r, reg = self.ready()
        a = self.state.evaluate(pid, reg, arm(BASE), self.candidate(r, 0))
        self.state.gate(pid, a["attempt_id"])
        path = next((self.dir / "sets").rglob("attempts/*.json"))
        doc = json.loads(path.read_bytes())
        settle = doc["settle"]["record"]
        settle["wealth_after"] = settle["wealth_after"] + 0.01      # breaks the signature too
        path.write_bytes(canonical_json(doc))
        self.state.close()
        with self.assertRaises(Refused) as cm:
            again = self.open()
            again.alpha(pid)
        self.assertEqual(cm.exception.problems[0].rule, "AN-14")

    def test_resigned_ledger_with_wrong_arithmetic_refused(self):    # a valid signature is not enough
        pid, r, reg = self.ready()
        a = self.state.evaluate(pid, reg, arm(BASE), self.candidate(r, 0))
        self.state.gate(pid, a["attempt_id"])
        key = self.state.key
        path = next((self.dir / "sets").rglob("attempts/*.json"))
        doc = json.loads(path.read_bytes())
        settle = dict(doc["settle"]["record"], wealth_after=doc["settle"]["record"]["wealth_after"] + 0.01)
        doc["settle"] = signing.sign(key, "ledger_entry", settle)
        path.write_bytes(canonical_json(doc))
        self.state.close()
        again = self.open()
        with self.assertRaises(Refused) as cm:
            again.alpha(pid)
        self.assertEqual(cm.exception.problems[0].rule, "AN-15")

    def test_report_refuses_a_bad_signature(self):             # AN-14
        pid, r, reg = self.ready()
        a = self.state.evaluate(pid, reg, arm(BASE), self.candidate(r, 0))
        self.state.gate(pid, a["attempt_id"])
        path = next((self.dir / "sets").rglob("attempts/*.json"))
        doc = json.loads(path.read_bytes())
        sig = doc["gate_record"]["signature"]
        doc["gate_record"]["signature"] = ("A" if sig[0] != "A" else "B") + sig[1:]
        path.write_bytes(canonical_json(doc))
        self.state.close()
        with self.assertRaises(Refused) as cm:
            again = self.open()
            again.report(pid)
        self.assertEqual(cm.exception.problems[0].rule, "AN-14")
        self.assertFalse((self.dir / "sets" / r["sealed_id"].removeprefix("sha256:") / "closed").exists())


class _ScenarioEP:
    name = "synthetic"
    value = "gatedtrain.testing.synthetic_pack:SCENARIO"

    def load(self):
        return pack.SCENARIO


_ORIGINAL_EPS = trust_stages._scenario_entry_points


class ReplayChecksTest(unittest.TestCase):
    def rows(self, outcomes) -> list[dict]:
        rows, w = [], ai.INITIAL_WEALTH
        for passed in outcomes:
            row = ai.settle(w, ai.bid(w), passed)
            rows.append(row)
            w = row["wealth_after"]
        return rows

    def test_replay_compares_every_stored_number(self):
        rows = self.rows([False, True])
        for key, delta in (("wealth_before", 0.001), ("cost", 0.001), ("earned", -0.01), ("wealth_after", 0.01)):
            with self.subTest(key):
                bad = [dict(rows[0], **{key: rows[0][key] + delta}), rows[1]]
                with self.assertRaises(Refused) as cm:
                    ai.replay(bad)
                self.assertEqual(cm.exception.problems[0].rule, "AN-15")

    def test_replay_refuses_a_row_past_the_run_out(self):
        rows = self.rows([False, False, False])
        w = rows[-1]["wealth_after"]
        self.assertFalse(ai.can_bid(w))
        with self.assertRaises(Refused):
            ai.replay(rows + [ai.settle(w, ai.bid(w), False)])


if __name__ == "__main__":
    unittest.main()
