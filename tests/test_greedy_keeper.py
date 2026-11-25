"""AN-26 in the keeper (revision 26): a greedy registration's sealed evaluation carries a control seed in the same
inference container call, compares the predictions itself and settles a seed-dependent run as a crash that costs
alpha. The container's own claims are never read: the keeper compares the predictions the container wrote."""

from __future__ import annotations

import json
import os
import unittest

from _keeper import skip_unless_linux

import test_an_keeper_inference as base
from gatedtrain.ainative import scorers, trust_stages
from gatedtrain.ainative.alpha_investing import PLANNING_ALPHA
from gatedtrain.ainative.keeper.protocol import RefusedAfterSpend
from gatedtrain.errors import Refused
from gatedtrain.testing import synthetic_pack as pack
from gatedtrain.testing.fake_runtime import subprocess_handler

from test_an_keeper_attempts import arm

REGISTERED = 11
GREEDY = {"name": "greedy", "decoding": "greedy", "temperature": 0, "max_new_tokens": 1}


def greedy_registration(sealed_id: str, base_sha: str) -> dict:
    spec = scorers.resolve("exact_match")
    v = {"scorer": "exact_match", "direction": spec.direction, "margin": 0.05, "alpha": PLANNING_ALPHA}
    return trust_stages._registration(v, sealed_id, {"sampler": dict(GREEDY), "eval_seeds": [REGISTERED]}, base_sha,
                                      ("synthetic", pack.BATCH_GENERATOR), scorers.evaluator_entry(spec, {}))


def tampering(change):
    """The pack's script, then `change(seed, line_doc)` applied to every prediction line of the candidate's run (the
    baseline is left alone), written back canonically."""
    def handler(spec, host_path):
        rc = subprocess_handler(spec, host_path)
        if host_path("/in/model").name != "baseline":
            out = host_path("/out") / "predictions.jsonl"
            docs = [json.loads(x) for x in out.read_bytes().split(b"\n")[:-1]]
            for doc in docs:
                change(doc)
            out.write_bytes(b"".join(json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                                allow_nan=False).encode() + b"\n" for d in docs))
        return rc
    return handler


@skip_unless_linux
class KeeperGreedyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.h = base.StateInferenceTest("test_pass_and_inference_records")
        self.h.setUp()
        self.addCleanup(self.h.doCleanups)
        self.reopen(subprocess_handler)

    def reopen(self, handler) -> None:
        h = self.h
        h.rt = h.runtime(handler)
        h.state.close()
        h.state = h.open()
        h.reg = h.state.register(h.pid, greedy_registration(h.sealed, arm(h.base)["sha256"]))["hash"]

    def test_a_deterministic_pack_is_scored_with_one_container_per_arm(self) -> None:
        att = self.h.evaluate(self.h.candidate())
        self.assertEqual(len(self.h.rt.runs), 2)              # the control seed adds no container
        gate = self.h.state.gate(self.h.pid, att["attempt_id"])["summary"]["record"]
        self.assertEqual(gate["status"], "PASS")
        rows = self.h.attempt_file()["gate_record"]["record"]["inference"]
        self.assertEqual([r["arm"] for r in rows], ["baseline", "candidate"])

    def test_the_container_gets_the_registered_seed_and_one_other(self) -> None:
        seen = []

        def spy(spec, host_path):
            seen.append(json.loads(host_path("/in/sampler.json").read_text()))
            return subprocess_handler(spec, host_path)
        self.reopen(spy)
        self.h.evaluate(self.h.candidate())
        self.assertEqual(len(seen), 2)
        for doc in seen:
            self.assertEqual(doc["sampler"], GREEDY)
            self.assertEqual(len(doc["seeds"]), 2)
            self.assertIn(REGISTERED, doc["seeds"])
            (control,) = [s for s in doc["seeds"] if s != REGISTERED]
            self.assertTrue(0 <= control < 2 ** 31)
        self.assertNotEqual(seen[0]["seeds"], [REGISTERED])

    def assert_not_deterministic(self, change) -> None:
        self.reopen(tampering(change))
        before = self.h.state.alpha(self.h.pid)
        with self.assertRaises(RefusedAfterSpend) as cm:
            self.h.evaluate(self.h.candidate())
        p = cm.exception.problems[0]
        self.assertEqual(p.rule, "AN-26")
        self.assertIn("not deterministic", p.message)
        self.assertNotIn(str(self.h.dir), p.message + p.fix)
        self.assertNotIn(base.SECRET, p.message + p.fix)
        self.assertNotEqual(cm.exception.result["summary"]["record"]["status"], "PASS")
        self.assertEqual(cm.exception.result["settle"]["record"]["passed"], False)
        after = self.h.state.alpha(self.h.pid)
        self.assertLess(after["wealth"], before["wealth"])    # the comparison is a bit of the sealed data: it costs
        self.assertEqual(self.h.left("models"), [])
        self.assertEqual(self.h.left("run"), [])

    def test_a_container_that_samples_under_other_seeds_is_refused_after_the_spend(self) -> None:
        def change(doc):
            if doc["seed"] != REGISTERED:
                doc["prediction"] = "sampled"
        self.assert_not_deterministic(change)

    def test_a_container_that_samples_only_under_the_registered_seed_is_refused_too(self) -> None:
        def change(doc):
            if doc["seed"] == REGISTERED:
                doc["prediction"] = "sampled"
        self.assert_not_deterministic(change)

    def test_one_differing_item_is_enough(self) -> None:
        def change(doc):
            if doc["seed"] == REGISTERED and doc["item_id"] == 3:
                doc["prediction"] = "drift"
        self.assert_not_deterministic(change)

    def test_a_pack_that_claims_greedy_in_its_output_is_not_believed(self) -> None:
        """No file or field the container writes can stand in for the comparison: an extra file is a crash (AN-9),
        and a prediction line may carry only the three keys."""
        def change(doc):
            doc["deterministic"] = True
            if doc["seed"] != REGISTERED:
                doc["prediction"] = "sampled"
        self.reopen(tampering(change))
        with self.assertRaises(RefusedAfterSpend) as cm:
            self.h.evaluate(self.h.candidate())
        self.assertEqual(cm.exception.problems[0].rule, "AN-9")      # the strict reader first

    def test_the_sampled_registration_still_runs_its_own_seeds(self) -> None:
        seen = []

        def spy(spec, host_path):
            seen.append(json.loads(host_path("/in/sampler.json").read_text())["seeds"])
            return subprocess_handler(spec, host_path)
        h = self.h
        h.rt = h.runtime(spy)
        h.state.close()
        h.state = h.open()
        h.reg = h.state.register(h.pid, base.batch_registration(h.sealed, arm(h.base)["sha256"]))["hash"]
        h.evaluate(h.candidate())
        self.assertEqual(seen, [list(pack.EVAL_SEEDS)] * 2)


if __name__ == "__main__":
    unittest.main()
