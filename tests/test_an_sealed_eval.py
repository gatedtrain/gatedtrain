"""AN-6, AN-8, AN-9: the shared sealed-evaluation core (`gatedtrain.ainative.sealed_eval`).

The in-process stages (`trust_stages`) and the keeper call the same split, scoring and gate statistics. These tests
exercise the core directly, with no project and no trust backend; the step 1 stage tests pin the in-process path.
"""

from __future__ import annotations

import json
import unittest

from _support import FIXTURES

from gatedtrain import gates
from gatedtrain.ainative import alpha_investing, sealed_eval, trust_stages
from gatedtrain.canonical import canonical_json, sha256_bytes
from gatedtrain.errors import Refused

TOY = FIXTURES / "ainative" / "toy.jsonl"

# `trust_stages._split` on the toy fixture, computed before the code moved into `sealed_eval`.
GOLDEN_TOY = {"n_unique": 800, "n_removed": 0, "n_needed": 310, "n_sealed": 310, "mde": 0.0998592334401666,
              "conclusive": True, "more": 0}
GOLDEN_SEALED = "sha256:7662df3e90c4c4c558007167f92f3510dff499dbc9b38fad79f633057c89ed96"
GOLDEN_TRAIN = "sha256:a5c08f3e5760925d4238f33b40c92e5b79a9e34ea10e60298c6efe6932deb6c5"
STEP1_POWER_KEYS = ["n_unique", "n_sealed", "n_needed", "mde", "target_mde", "alpha", "power", "sigma",
                    "conclusive_expected", "n_more_examples"]

BASE_SHA = "sha256:" + "b" * 64
CAND_SHA = "sha256:" + "c" * 64
SEEDS = [11, 12]


def jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r) + "\n" for r in rows).encode()


def split(raw: bytes, planning_alpha: float = 0.05) -> dict:
    return sealed_eval.split_data(raw, sha256_bytes(raw), planning_alpha=planning_alpha)


def registration(post_hoc: bool = False) -> dict:
    v = {"scorer": "exact_match", "direction": "higher_is_better", "margin": 0.0, "alpha": 0.05}
    dep = {"sampler": {"name": "s", "temperature": 0.0}, "eval_seeds": list(SEEDS)}
    full = trust_stages._registration(v, "sha256:" + "a" * 64, dep, BASE_SHA)
    return {**full, "post_hoc": True} if post_hoc else full


class SplitDataTest(unittest.TestCase):
    def test_the_toy_fixture_splits_as_step_1_did(self) -> None:
        raw = TOY.read_bytes()
        s = split(raw)
        self.assertEqual({k: s[k] for k in GOLDEN_TOY}, GOLDEN_TOY)
        self.assertEqual(s["digest"], sha256_bytes(raw))
        self.assertEqual(sha256_bytes(canonical_json(s["sealed"])), GOLDEN_SEALED)
        self.assertEqual(sha256_bytes(canonical_json(s["train"])), GOLDEN_TRAIN)

    def test_a_smaller_planning_alpha_needs_more_sealed_items(self) -> None:
        raw = TOY.read_bytes()
        fixed, keeper = split(raw, 0.05), split(raw, alpha_investing.PLANNING_ALPHA)
        self.assertEqual(alpha_investing.PLANNING_ALPHA, 0.02375)
        self.assertGreater(keeper["n_needed"], fixed["n_needed"])
        self.assertGreaterEqual(keeper["n_sealed"], fixed["n_sealed"])

    def test_input_hashes_have_one_entry_per_distinct_row(self) -> None:
        rows = [{"input": f"q{i}", "label": f"a{i % 3}"} for i in range(40)]
        raw = jsonl(rows + rows[:5] + [{"input": {"k": 1, "j": 2}, "label": "x"}, {"input": {"j": 2, "k": 1},
                                                                                     "label": "x"}])
        s = split(raw)
        self.assertEqual(s["n_unique"], 41)
        self.assertEqual(s["n_removed"], 6)
        self.assertEqual(len(s["input_hashes"]), 41)
        self.assertEqual(len(set(s["input_hashes"])), 41)
        want = {sha256_bytes(canonical_json(r["input"])) for r in rows} | {sha256_bytes(canonical_json({"j": 2,
                                                                                                        "k": 1}))}
        self.assertEqual(set(s["input_hashes"]), want)
        sealed = {sha256_bytes(canonical_json(r["input"])) for r in s["sealed"]}
        self.assertEqual(set(s["sealed_input_hashes"]), sealed)
        self.assertEqual(len(s["sealed_input_hashes"]), s["n_sealed"])
        self.assertTrue(set(s["sealed_input_hashes"]) <= set(s["input_hashes"]))

    def test_refusals_are_an_6(self) -> None:
        for raw in (b"", b'{"input": "only"}\n', jsonl([{"input": "a", "label": "x"}]),
                    jsonl([{"input": "a", "label": "x"}, {"input": "a", "label": "y"}])):
            with self.subTest(raw=raw):
                with self.assertRaises(Refused) as cm:
                    split(raw)
                self.assertEqual({p.rule for p in cm.exception.problems}, {"AN-6"})

    def test_power_record_is_the_step_1_body(self) -> None:
        s = split(TOY.read_bytes())
        rec = sealed_eval.power_record(s, planning_alpha=0.05)
        self.assertEqual(list(rec), STEP1_POWER_KEYS)               # no `planning_alpha` key on this path
        self.assertEqual(rec["alpha"], 0.05)
        self.assertEqual((rec["n_sealed"], rec["n_needed"], rec["mde"]), (s["n_sealed"], s["n_needed"], s["mde"]))
        self.assertEqual(rec["sigma"], sealed_eval.PILOT_SIGMA)
        self.assertEqual(rec["conclusive_expected"], s["conclusive"])


def echo(model, item, *, sampler, seed):
    return item["input"]


class ScoreArmTest(unittest.TestCase):
    items = [{"input": f"secret-input-{i}", "label": f"secret-label-{i}"} for i in range(5)]

    def score_arm(self, generate, score=None, items=None, rng=(0.0, 1.0)):
        score = score or (lambda p, label: 1.0 if p == label else 0.0)
        return sealed_eval.score_arm(items or self.items, generate, {"generator": "g"}, registration(), score,
                                     score_range=rng, scorer_name="exact_match")

    def assert_no_leak(self, refused: Refused) -> None:
        for p in refused.problems:
            for text in (p.rule, p.field, p.message, p.fix):
                self.assertNotIn("secret", text)

    def test_one_record_per_item_and_seed(self) -> None:
        rows = self.score_arm(lambda model, item, *, sampler, seed: item["input"].replace("input", "label"))
        self.assertEqual(rows, [{"item_id": i, "seed": s, "score": 1.0} for s in SEEDS for i in range(5)])

    def test_generate_gets_the_input_only_and_the_registered_sampler(self) -> None:
        seen = []

        def generate(model, item, *, sampler, seed):
            seen.append((model, item, sampler, seed))
            return ""
        self.score_arm(generate)
        self.assertEqual(seen[0], ({"generator": "g"}, {"input": "secret-input-0"}, registration()["sampler"], 11))
        self.assertEqual({s for *_, s in seen}, set(SEEDS))

    def test_a_raising_generate_is_refused_without_the_item(self) -> None:
        def generate(model, item, *, sampler, seed):
            raise ValueError(item["input"])
        with self.assertRaises(Refused) as cm:
            self.score_arm(generate)
        self.assertEqual({p.rule for p in cm.exception.problems}, {"AN-8"})
        self.assertIn("ValueError", cm.exception.problems[0].message)
        self.assertNotIn("secret", str(cm.exception))
        self.assert_no_leak(cm.exception)
        self.assertIsNone(cm.exception.__cause__)

    def test_a_raising_scorer_is_refused_without_the_label(self) -> None:
        def score(prediction, label):
            raise KeyError(label)
        with self.assertRaises(Refused) as cm:
            self.score_arm(echo, score)
        self.assertEqual({p.rule for p in cm.exception.problems}, {"AN-8"})
        self.assertIn("KeyError", cm.exception.problems[0].message)
        self.assert_no_leak(cm.exception)

    def test_a_scorer_raising_refused_with_the_label_is_not_passed_on(self) -> None:
        def score(prediction, label):
            raise Refused([trust_stages.Problem("AN-8", label, label, label)])
        with self.assertRaises(Refused) as cm:
            self.score_arm(echo, score)
        self.assert_no_leak(cm.exception)

    def test_out_of_range_and_non_finite_scores_are_refused(self) -> None:
        for value in (2.0, -0.1, float("nan"), float("inf"), True, "1"):
            with self.subTest(value=value):
                with self.assertRaises(Refused) as cm:
                    self.score_arm(echo, lambda p, label, v=value: v)
                p = cm.exception.problems[0]
                self.assertEqual(p.rule, "AN-8")
                self.assertIn("range [0, 1]", p.message)

    def test_the_score_range_is_the_scorers(self) -> None:
        rows = self.score_arm(echo, lambda p, label: 10.0, rng=(0.0, 10.0))
        self.assertTrue(all(r["score"] == 10.0 for r in rows))


def rows(scores: list[float]) -> list[dict]:
    return [{"item_id": i, "seed": s, "score": float(scores[i])} for s in SEEDS for i in range(len(scores))]


class DecideTest(unittest.TestCase):
    n = 200

    def decide(self, base: list[float], cand: list[float], full: dict | None = None, alpha: float = 0.05) -> dict:
        return sealed_eval.decide(full or registration(), {"records": rows(base), "model_ref_sha256": BASE_SHA},
                                  {"records": rows(cand), "model_ref_sha256": CAND_SHA}, alpha=alpha,
                                  n_sealed=self.n, expected_models={"baseline": BASE_SHA, "candidate": CAND_SHA},
                                  score_range=(0.0, 1.0))

    def half(self, shift: int = 0) -> list[float]:
        return [float((i + shift) % 2) for i in range(self.n)]

    def test_better_worse_and_equal_arms(self) -> None:
        self.assertEqual(self.decide(self.half(), [1.0] * self.n)["status"], gates.PASS)
        self.assertEqual(self.decide(self.half(), [0.0] * self.n)["status"], gates.FAIL)
        self.assertEqual(self.decide(self.half(), self.half(1))["status"], gates.INCONCLUSIVE)

    def test_the_alpha_is_the_one_given_not_the_registered_one(self) -> None:
        res = self.decide(self.half(), [1.0] * self.n, alpha=0.2)
        self.assertEqual(registration()["clause"]["alpha"], 0.05)
        self.assertEqual((res["alpha"], res["n_items"]), (0.2, self.n))
        self.assertAlmostEqual(res["ci_level"], 0.6)

    def test_post_hoc_downgrades_a_pass(self) -> None:
        res = self.decide(self.half(), [1.0] * self.n, registration(post_hoc=True))
        self.assertEqual(res["status"], gates.INCONCLUSIVE)
        self.assertIn("post_hoc", res["status_reasons"])

    def test_a_record_of_another_model_is_missing(self) -> None:
        res = sealed_eval.decide(registration(), {"records": rows(self.half()), "model_ref_sha256": BASE_SHA},
                                 {"records": rows([1.0] * self.n), "model_ref_sha256": "sha256:" + "d" * 64},
                                 alpha=0.05, n_sealed=self.n,
                                 expected_models={"baseline": BASE_SHA, "candidate": CAND_SHA},
                                 score_range=(0.0, 1.0))
        self.assertEqual(res["status"], gates.MISSING)

    def test_convert_record_maps_item_id_and_seed(self) -> None:
        self.assertEqual(sealed_eval.convert_record({"item_id": 3, "seed": 12, "score": 1.0}, SEEDS),
                         {"item": 3, "seed": 12, "score": 1.0, "repeat": 1})
        self.assertEqual(sealed_eval.convert_record({"item_id": 3, "seed": 99, "score": 1.0}, SEEDS)["repeat"], -1)
        self.assertEqual(sealed_eval.convert_record("junk", SEEDS), "junk")


class ReportPiecesTest(unittest.TestCase):
    """The step 1 report pieces live in `sealed_eval` and `trust_stages` uses them, so the keeper can reuse them."""

    def test_trust_stages_uses_the_shared_pieces(self) -> None:
        self.assertIs(trust_stages.known_limits, sealed_eval.known_limits)
        self.assertIs(trust_stages._effective_status, sealed_eval.effective_status)
        self.assertIs(trust_stages._headline, sealed_eval.headline)

    def test_display_block(self) -> None:
        gate = {"delta_hat_natural": 0.123456, "ci_lower_natural": 0.1, "ci_upper_natural": 0.15556,
                "ci_level": 0.95, "baseline_score": 0.5, "candidate_score": 0.623456}
        self.assertEqual(sealed_eval.display_block(gate, {"mde": 0.09985923}),
                         {"effect": 0.1235, "ci_lower": 0.1, "ci_upper": 0.1556, "ci_level": 0.95,
                          "baseline_score": 0.5, "candidate_score": 0.6235, "mde_planned": 0.0999})

    def test_effective_status_and_headline(self) -> None:
        self.assertEqual(sealed_eval.effective_status(gates.PASS, True), gates.INCONCLUSIVE)
        self.assertEqual(sealed_eval.effective_status(gates.FAIL, True), gates.FAIL)
        self.assertEqual(sealed_eval.headline(gates.PASS, False),
                         "# PASS: the new model beats the base model by more than the registered margin")
        self.assertEqual(sealed_eval.headline(gates.PASS, True), "# INCONCLUSIVE: post hoc, not a confirmed gain")

    def test_known_limits_counts_evaluate_sealed_starts(self) -> None:
        attempts = [{"stage": "evaluate_sealed", "event": "start"}, {"stage": "gate", "event": "start"},
                    {"stage": "evaluate_sealed", "event": "refused"}]
        limits = sealed_eval.known_limits("stated", attempts)
        self.assertEqual([x["id"] for x in limits], ["sealed_guarantee", "alpha_per_project", "metric_confirmation",
                                                     "model_reference", "sealed_evaluations"])
        self.assertEqual(limits[0]["text"], "stated")
        self.assertEqual(limits[-1]["attempts"], 1)


if __name__ == "__main__":
    unittest.main()
