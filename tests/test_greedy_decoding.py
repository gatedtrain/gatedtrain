"""REG-GREEDY and AN-26 (revision 26): registered greedy decoding.

Greedy decoding is an explicit opt-in (`sampler.decoding: greedy`, temperature 0, one evaluation seed) that only a
path with a determinism check accepts. The check runs the generator once more under a seed the registration does not
name and requires identical predictions: a repeat under the same seed would not tell greedy decoding from seeded
sampling. These tests cover the registration rules, the statistics with one sample per item, and the check in the
shared scoring core (the keeper calls the same `sealed_eval.score_arm`)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gatedtrain import gates
from gatedtrain.ainative import batch, scorers, sealed_eval, trust_stages
from gatedtrain.ainative.batch import BatchGenerator
from gatedtrain.ainative.machine import run_stage
from gatedtrain.ainative.trust import InProcessTrust
from gatedtrain.errors import Refused
from gatedtrain.registration import check_contents, is_greedy

from test_an_intake import (Base as IntakeBase, ChooseDeployment, DEPLOY, N_DATA, jsonl, rows)

BASE_SHA = "sha256:" + "b" * 64
GREEDY = {"name": "greedy", "decoding": "greedy", "temperature": 0, "max_new_tokens": 16}
SAMPLED = {"name": "sampled", "temperature": 0.7, "max_new_tokens": 16}


def registration(sampler: dict, seeds: list) -> dict:
    v = {"scorer": "exact_match", "direction": "higher_is_better", "margin": 0.05, "alpha": 0.05}
    return trust_stages._registration(v, "sha256:" + "a" * 64, {"sampler": sampler, "eval_seeds": seeds}, BASE_SHA)


def problems(reg: dict, **kw) -> list:
    return [p for p in check_contents(reg, **kw) if p.field.startswith("registration.sampler")
            or p.field.startswith("registration.clause")]


def rules(ps: list) -> set:
    return {p.rule for p in ps}


class RegistrationTest(unittest.TestCase):
    def test_the_sampled_registration_is_unchanged(self) -> None:
        self.assertEqual(check_contents(registration(SAMPLED, [11, 12])), [])
        self.assertFalse(is_greedy(SAMPLED))
        self.assertTrue(is_greedy(GREEDY))

    def test_greedy_is_refused_unless_the_path_has_a_determinism_check(self) -> None:
        ps = check_contents(registration(GREEDY, [11]))
        self.assertEqual(rules(ps), {"REG-GREEDY"})
        self.assertEqual([p.field for p in ps], ["registration.sampler.decoding"])
        self.assertIn("determinism check", ps[0].message)

    def test_greedy_with_one_seed_is_accepted_where_allowed(self) -> None:
        self.assertEqual(check_contents(registration(GREEDY, [11]), greedy_allowed=True), [])

    def test_temperature_zero_needs_the_greedy_switch(self) -> None:
        ps = check_contents(registration({"name": "s", "temperature": 0, "max_new_tokens": 16}, [11]),
                            greedy_allowed=True)
        self.assertEqual(rules(ps), {"REG-GREEDY"})
        self.assertEqual([p.field for p in ps], ["registration.sampler.temperature"])
        self.assertIn("decoding", ps[0].fix)

    def test_greedy_switch_needs_temperature_zero(self) -> None:
        for temperature in (0.7, 1, 1e-9):
            with self.subTest(temperature=temperature):
                ps = check_contents(registration({**GREEDY, "temperature": temperature}, [11]), greedy_allowed=True)
                self.assertEqual(rules(ps), {"REG-GREEDY"})
        ps = check_contents(registration({k: v for k, v in GREEDY.items() if k != "temperature"}, [11]),
                            greedy_allowed=True)
        self.assertEqual(rules(ps), {"REG-GREEDY"})

    def test_decoding_has_one_value(self) -> None:
        for value in ("beam", "Greedy", True, None, 0):
            with self.subTest(value=value):
                ps = problems(registration({**GREEDY, "decoding": value}, [11]), greedy_allowed=True)
                self.assertIn("REG-GREEDY", rules(ps))

    def test_negative_temperature_is_still_an_ordinary_refusal(self) -> None:
        ps = check_contents(registration({"name": "s", "temperature": -1, "max_new_tokens": 16}, [11]),
                            greedy_allowed=True)
        self.assertEqual(rules(ps), {"REG-FIELD"})

    def test_top_p_is_absent_or_one(self) -> None:
        self.assertEqual(check_contents(registration({**GREEDY, "top_p": 1.0}, [11]), greedy_allowed=True), [])
        ps = check_contents(registration({**GREEDY, "top_p": 0.9}, [11]), greedy_allowed=True)
        self.assertEqual(rules(ps), {"REG-GREEDY"})
        self.assertEqual([p.field for p in ps], ["registration.sampler.top_p"])

    def test_one_seed_only(self) -> None:
        for seeds in ([11, 12], [1, 2, 3, 4]):
            with self.subTest(seeds=seeds):
                ps = check_contents(registration(GREEDY, seeds), greedy_allowed=True)
                self.assertEqual(rules(ps), {"REG-GREEDY"})
                self.assertIn("registration.clause.eval_seeds", [p.field for p in ps])
                self.assertIn("one", ps[0].message)

    def test_samples_per_item_is_one(self) -> None:
        reg = registration(GREEDY, [11])
        reg["clause"] = {**reg["clause"], "samples_per_item": 2, "eval_seeds": [11, 12]}
        ps = check_contents(reg, greedy_allowed=True)
        self.assertEqual(rules(ps), {"REG-GREEDY"})
        self.assertIn("registration.clause.samples_per_item", [p.field for p in ps])

    def test_greedy_is_a_one_clause_registration(self) -> None:
        reg = registration(GREEDY, [11])
        clause = reg.pop("clause")
        reg["clauses"] = [{**clause, "name": "a", "evaluator": reg["evaluators"][0]["id"]}]
        self.assertIn("REG-GREEDY", rules(check_contents(reg, greedy_allowed=True)))

    def test_unknown_decoding_key_is_still_refused_on_other_paths(self) -> None:
        ps = check_contents(registration(SAMPLED, [11, 12]))
        self.assertEqual(ps, [])


class StatisticsTest(unittest.TestCase):
    """One sample per item: the item is the resampled unit, so n is the number of items and K never enters."""

    def arms(self, k: int):
        rng = np.random.default_rng(5)
        n = 80
        base = (rng.random(n) < 0.4).astype(float)
        cand = (rng.random(n) < 0.6).astype(float)
        seeds = list(range(k))

        def recs(scores):
            return {"records": [{"item": i, "repeat": r, "seed": seeds[r], "score": float(scores[i])}
                                for r in range(k) for i in range(n)]}
        return {"eval_seeds": seeds, "samples_per_item": k, "alpha": 0.05, "power": 0.8, "margin": 0.0,
                "type": "superiority", "metric": "m", "direction": "higher_is_better", "sigma_plan": "fixed",
                "planning_effect": 0.1, "unit": "item", "interval": "cluster_bootstrap"}, recs(base), recs(cand), n

    def decide(self, k: int) -> dict:
        clause, base, cand, n = self.arms(k)
        return gates.evaluate_items(clause, base, cand, items=list(range(n)), seed=3, n_boot=2000, min_paired=10,
                                    max_unpaired=0, score_range=(0.0, 1.0))

    def test_identical_repeats_do_not_shrink_the_interval(self) -> None:
        one, four = self.decide(1), self.decide(4)
        for key in ("ci_lower", "ci_upper", "delta_hat", "n_items", "achieved_power", "status"):
            self.assertEqual(one[key], four[key], key)
        self.assertEqual(one["n_items"], 80)

    def test_one_seed_runs_through_the_gate(self) -> None:
        res = self.decide(1)
        self.assertIn(res["status"], (gates.PASS, gates.FAIL, gates.INCONCLUSIVE))
        self.assertEqual(res["n_items"], 80)


# ---------------------------------------------------------------------------------------------------------------------
# AN-26: the control seed check in the shared scoring core
# ---------------------------------------------------------------------------------------------------------------------

ITEMS = [{"input": f"secret-in-{i}", "label": f"secret-in-{i}"} for i in range(6)]
FULL = {"sampler": dict(GREEDY), "clause": {"eval_seeds": [11], "samples_per_item": 1}}
MODEL = {"generator": "g"}


def exact(p, label):
    return 1.0 if p == label else 0.0


def greedy_generate(model, item, *, sampler, seed):
    return item["input"]


def seeded_sampling(model, item, *, sampler, seed):
    """Looks greedy under one seed but depends on it: a repeat under the same seed would not show it."""
    return item["input"] + ("!" if seed % 2 else "")


class ControlSeedTest(unittest.TestCase):
    def score(self, generate, full=FULL, **kw):
        return sealed_eval.score_arm(ITEMS, generate, MODEL, full, exact, score_range=(0.0, 1.0),
                                     scorer_name="exact_match", **kw)

    def test_a_greedy_generator_is_scored_under_the_registered_seed_only(self) -> None:
        seen = []

        def generate(model, item, *, sampler, seed):
            seen.append(seed)
            return item["input"]
        rows_ = self.score(generate, control_seed=7)
        self.assertEqual(rows_, [{"item_id": i, "seed": 11, "score": 1.0} for i in range(len(ITEMS))])
        self.assertEqual(sorted(set(seen)), [7, 11])
        self.assertEqual(seen.count(7), len(ITEMS))

    def test_a_prediction_that_depends_on_the_seed_is_refused(self) -> None:
        with self.assertRaises(Refused) as cm:
            self.score(seeded_sampling, control_seed=12)
        self.assertEqual({p.rule for p in cm.exception.problems}, {"AN-26"})
        self.assertNotIn("secret", str(cm.exception))
        self.assertIsNone(cm.exception.__cause__)

    def test_the_control_seed_is_not_registered_and_changes_per_call(self) -> None:
        seeds = {sealed_eval.draw_control_seed(FULL) for _ in range(40)}
        self.assertTrue(len(seeds) > 20)
        self.assertNotIn(11, seeds)
        self.assertTrue(all(isinstance(s, int) and not isinstance(s, bool) and 0 <= s < 2 ** 31 for s in seeds))

    def test_a_control_seed_equal_to_the_registered_one_is_refused(self) -> None:
        with self.assertRaises(Refused):
            self.score(greedy_generate, control_seed=11)

    def test_the_sampled_path_is_unchanged(self) -> None:
        seen = []

        def generate(model, item, *, sampler, seed):
            seen.append(seed)
            return item["input"] + str(seed)
        full = {"sampler": dict(SAMPLED), "clause": {"eval_seeds": [11, 12]}}
        rows_ = self.score(generate, full=full)
        self.assertEqual(len(rows_), 2 * len(ITEMS))
        self.assertEqual(set(seen), {11, 12})

    def test_unequal_predictions_are_compared_as_canonical_json(self) -> None:
        def generate(model, item, *, sampler, seed):
            return {"a": 1, "b": [1, 2]} if seed == 11 else {"b": [1, 2], "a": 1}
        self.assertEqual(len(self.score(generate, control_seed=5)), len(ITEMS))

        def generate2(model, item, *, sampler, seed):
            return 1 if seed == 11 else 1.0
        with self.assertRaises(Refused):                      # 1 and 1.0 are different predictions
            self.score(generate2, control_seed=5)

    def test_a_prediction_that_is_not_json_is_refused_without_its_content(self) -> None:
        def generate(model, item, *, sampler, seed):
            return {1, 2}
        with self.assertRaises(Refused) as cm:
            self.score(generate, control_seed=5)
        self.assertEqual({p.rule for p in cm.exception.problems}, {"AN-26"})
        self.assertNotIn("secret", str(cm.exception))


class RecordingRunner:
    """A `BatchRunner` that answers from a function of (item, seed) and records the seeds it was asked for."""

    def __init__(self, fn) -> None:
        self.fn, self.calls = fn, []

    def run_batch(self, gen, model, items, sampler, seeds):
        self.calls.append({"seeds": list(seeds), "sampler": dict(sampler), "n": len(items)})
        return [{"item_id": i, "seed": s, "prediction": self.fn(it["input"], s)}
                for s in seeds for i, it in enumerate(items)]


GEN = BatchGenerator(script=Path("/site/pkg/gen.py"), interpreter="/usr/bin/python3")


class BatchControlTest(unittest.TestCase):
    def score(self, runner, full=FULL, **kw):
        return sealed_eval.score_arm(ITEMS, None, MODEL, full, exact, score_range=(0.0, 1.0),
                                     scorer_name="exact_match", batch=GEN, runner=runner, **kw)

    def test_one_container_call_carries_the_registered_and_the_control_seed(self) -> None:
        runner = RecordingRunner(lambda text, seed: text)
        rows_ = self.score(runner, control_seed=7)
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(sorted(runner.calls[0]["seeds"]), [7, 11])
        self.assertEqual(rows_, [{"item_id": i, "seed": 11, "score": 1.0} for i in range(len(ITEMS))])

    def test_the_seed_order_is_not_fixed(self) -> None:
        orders = set()
        for _ in range(60):
            runner = RecordingRunner(lambda text, seed: text)
            self.score(runner)
            orders.add(tuple(runner.calls[0]["seeds"][:1]))
            self.assertNotIn(11, [s for s in runner.calls[0]["seeds"] if s != 11])
        self.assertGreaterEqual(len(orders), 2)

    def test_a_seed_dependent_batch_is_refused_with_counts_only(self) -> None:
        runner = RecordingRunner(lambda text, seed: text + ("!" if seed != 11 else ""))
        with self.assertRaises(Refused) as cm:
            self.score(runner, control_seed=12)
        (p,) = cm.exception.problems
        self.assertEqual(p.rule, "AN-26")
        self.assertNotIn("secret", str(cm.exception))

    def test_the_refusal_does_not_say_how_many_items_differ(self) -> None:
        """A count of differing items is a number about the sealed items: one item and all items read the same."""
        messages = set()
        for bad in ("3", ""):
            runner = RecordingRunner(lambda text, seed, b=bad: text + ("!" if seed != 11 and text.endswith(b) else ""))
            with self.assertRaises(Refused) as cm:
                self.score(runner, control_seed=12)
            messages.add(cm.exception.problems[0].message)
        self.assertEqual(len(messages), 1)
        self.assertFalse(any(ch.isdigit() for ch in messages.pop()))

    def test_a_pack_that_answers_only_the_registered_seed_is_refused_as_a_crash(self) -> None:
        class Short(RecordingRunner):
            def run_batch(self, gen, model, items, sampler, seeds):
                return super().run_batch(gen, model, items, sampler, seeds[:1])
        with self.assertRaises(Refused) as cm:
            self.score(Short(lambda text, seed: text))
        self.assertEqual({p.rule for p in cm.exception.problems}, {"AN-9"})

    def test_a_sampled_registration_runs_its_registered_seeds_only(self) -> None:
        runner = RecordingRunner(lambda text, seed: text)
        full = {"sampler": dict(SAMPLED), "clause": {"eval_seeds": [11, 12]}}
        self.score(runner, full=full)
        self.assertEqual(runner.calls[0]["seeds"], [11, 12])


class RegisteredByThePipelineTest(IntakeBase):
    """`preregister` of the AI-native path accepts a greedy deployment and refuses the invalid ones."""

    def deploy(self, record: dict, name: str):
        proj = self.make(jsonl(rows(N_DATA)), name=name)
        self.run_(proj, "intake_seal")
        reg = {**self.registry(), "choose_x": type("C", (ChooseDeployment,), {"record": record})}
        run_stage(proj, "choose_x", {}, registry=reg)
        return proj, reg

    def test_a_greedy_deployment_registers(self) -> None:
        greedy = {**DEPLOY, "sampler": dict(GREEDY), "eval_seeds": [11]}
        proj, reg = self.deploy(greedy, "g1")
        out = run_stage(proj, "preregister", {"metric_confirmed": True}, registry=reg).outputs
        full = InProcessTrust(proj).registration(proj.read_artifact(out["registration"])["hash"])
        self.assertEqual(full["sampler"], GREEDY)
        self.assertEqual(full["clause"]["samples_per_item"], 1)
        self.assertEqual(full["clause"]["eval_seeds"], [11])

    def test_switching_to_greedy_after_training_is_post_hoc(self) -> None:
        proj = self.make(jsonl(rows(N_DATA)), name="ph")
        self.run_(proj, "intake_seal")
        sampled = {**self.registry(), "choose_x": type("C", (ChooseDeployment,), {"record": DEPLOY})}
        run_stage(proj, "choose_x", {}, registry=sampled)
        first = proj.read_artifact(run_stage(proj, "preregister", {"metric_confirmed": True},
                                             registry=sampled).outputs["registration"])
        run_stage(proj, "train_x", {}, registry=sampled)
        greedy = {**self.registry(), "choose_x": type("C", (ChooseDeployment,), {
            "record": {**DEPLOY, "sampler": dict(GREEDY), "eval_seeds": [11]}})}
        run_stage(proj, "choose_x", {"note": "greedy", "rerun": True}, registry=greedy)
        second = proj.read_artifact(run_stage(proj, "preregister", {"metric_confirmed": True, "rerun": True},
                                              registry=greedy).outputs["registration"])
        self.assertTrue(second["post_hoc"])
        self.assertEqual(second["amends"], first["hash"])

    def test_invalid_greedy_deployments_are_refused(self) -> None:
        bads = [{**DEPLOY, "sampler": dict(GREEDY), "eval_seeds": [11, 12]},
                {**DEPLOY, "sampler": {**GREEDY, "temperature": 0.5}, "eval_seeds": [11]},
                {**DEPLOY, "sampler": {"name": "s", "temperature": 0, "max_new_tokens": 8}, "eval_seeds": [11]}]
        for i, rec in enumerate(bads):
            with self.subTest(i=i):
                proj, reg = self.deploy(rec, f"b{i}")
                with self.assertRaises(Refused) as cm:
                    run_stage(proj, "preregister", {"metric_confirmed": True}, registry=reg)
                self.assertIn("REG-GREEDY", {p.rule for p in cm.exception.problems})
                self.assertFalse((proj.root / "trust" / "registrations").exists())


if __name__ == "__main__":
    unittest.main()
