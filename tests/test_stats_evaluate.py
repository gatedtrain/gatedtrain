"""Statistics unit tests: `evaluate()` and `combine()`.

Design:
- 05 section 3.1 gate types, pass rules and the underpowered rule (05:54-82);
- 05 section 3.2 alpha defaults (05:84-93);
- 05 section 3.3 paired design on raw per-prompt differences, K averaged (05:95-99);
- 05 section 3.4 prompt-clustered bootstrap (05:105);
- 05 section 3.6 MDE (05:124); 04 section 4.2 rule 1, power at the declared MDE (04:102);
- 05 section 3.8 statuses PASS / FAIL / INCONCLUSIVE / MISSING (05:155);
- MVP-SPEC section 2.3 (plan-level underpowered) and section 2.7.

Interpretation used by these tests (listed as gap G-3 in tests/README.md):
`evaluate()` reports achieved power and MDE with the observed SD of the
per-prompt paired differences (ddof = 1), normal approximation, at the
planned distance d (superiority: planning_effect - margin;
non-inferiority: planning_effect + margin).
"""

from __future__ import annotations

import math
import random
import unittest

import numpy as np
from scipy.stats import norm

Z_975 = norm.ppf(0.975)
Z_80 = norm.ppf(0.80)


def clause(type_="superiority", **over):
    c = {
        "type": type_, "metric": "exact_match", "margin": 0.0, "alpha": 0.025, "power": 0.8,
        "planning_effect": 0.10, "sigma_plan": "fixed", "unit": "prompt", "samples_per_prompt": 1,
        "eval_seeds": [11], "interval": "cluster_bootstrap", "blocking": True,
    }
    if type_ == "non_inferiority":
        c.update(margin=0.10, planning_effect=0.0)
    if type_ == "equivalence":
        c.update(margin=0.10, planning_effect=0.0, alpha=0.05)
    c.update(over)
    return c


def records(matrix):
    """{(problem_idx, repeat): record} in NeMo Evaluator's per-sample shape (artifacts.py:98)."""
    out = {}
    for i, row in enumerate(matrix):
        for r, reward in enumerate(row):
            out[(i, r)] = {"problem_idx": i, "repeat": r, "reward": float(reward)}
    return out


def alternating(n, center, amp):
    """Per-prompt differences center +/- amp, alternating: mean = center, deterministic."""
    return [center + amp * (1 if i % 2 == 0 else -1) for i in range(n)]


def diff_records(diffs):
    """K = 1: baseline reward 0.5 everywhere, candidate 0.5 + d_i."""
    base = records([[0.5] for _ in diffs])
    cand = records([[0.5 + d] for d in diffs])
    return base, cand


class TestPointEstimateOnRawPairedDifferences(unittest.TestCase):
    """05 section 3.3 (05:97-99): Delta-hat is the mean over prompts of
    (mean_k candidate - mean_k baseline) on raw scores; never group-normalized."""

    BASE = [[1, 0], [0, 0], [1, 1], [0, 1]]
    CAND = [[1, 1], [1, 0], [1, 1], [0, 0]]
    # Per-prompt means: base 0.5, 0, 1, 0.5; cand 1, 0.5, 1, 0
    # Differences: 0.5, 0.5, 0, -0.5 -> mean 0.125
    # SD (ddof = 1): deviations .375, .375, -.125, -.625; squares sum .6875; /3 = .2291667; sqrt = .4787136
    DELTA = 0.125
    SD = math.sqrt(0.6875 / 3)

    def test_delta_hat_and_sd(self):
        from gatedtrain.gates import evaluate

        res = evaluate(clause(samples_per_prompt=2, eval_seeds=[11, 12]), records(self.BASE), records(self.CAND),
                       seed=1, n_boot=2000)
        self.assertAlmostEqual(res["delta_hat"], self.DELTA, places=12)
        self.assertAlmostEqual(res["sd_diff"], self.SD, places=12)
        self.assertEqual(res["n_prompts"], 4)
        self.assertEqual(res["hypothesis_type"], "superiority")

    def test_record_order_does_not_matter(self):
        from gatedtrain.gates import evaluate

        base = records(self.BASE)
        cand = records(self.CAND)
        keys = list(cand)
        random.Random(3).shuffle(keys)
        cand_shuffled = {k: cand[k] for k in keys}
        c = clause(samples_per_prompt=2, eval_seeds=[11, 12])
        a = evaluate(c, base, cand, seed=1, n_boot=2000)
        b = evaluate(c, base, cand_shuffled, seed=1, n_boot=2000)
        for key in ("delta_hat", "sd_diff", "ci_lower", "ci_upper", "status"):
            self.assertEqual(a[key], b[key], key)

    def test_unpaired_keys_are_ignored(self):
        """Pairs are the intersection of (problem_idx, repeat) keys, as in
        NeMo Evaluator's build_flip_report (comparison.py:387 at ea35ba3)."""
        from gatedtrain.gates import evaluate

        base = records(self.BASE)
        cand = records(self.CAND)
        cand[(99, 0)] = {"problem_idx": 99, "repeat": 0, "reward": 1.0}
        cand[(99, 1)] = {"problem_idx": 99, "repeat": 1, "reward": 1.0}
        res = evaluate(clause(samples_per_prompt=2, eval_seeds=[11, 12]), base, cand, seed=1, n_boot=2000)
        self.assertEqual(res["n_prompts"], 4)
        self.assertAlmostEqual(res["delta_hat"], self.DELTA, places=12)


class TestPairingNarrowsTheInterval(unittest.TestCase):
    """Pairing removes prompt difficulty from Delta (05:97). Prompts alternate
    between always-wrong and always-right; the candidate fixes 1 in 8 of the
    always-wrong prompts on one of K = 4 samples."""

    def setUp(self):
        n, k = 400, 4
        base, cand = [], []
        for i in range(n):
            row = [0] * k if i % 2 == 0 else [1] * k
            base.append(list(row))
            c = list(row)
            if i % 16 == 0:
                c[0] = 1
            cand.append(c)
        self.base, self.cand, self.n = base, cand, n
        diffs = np.array(cand).mean(axis=1) - np.array(base).mean(axis=1)
        self.delta = diffs.mean()             # 25 prompts * 0.25 / 400 = 0.015625
        self.sd = diffs.std(ddof=1)

    def test_interval_width_matches_paired_theory(self):
        from gatedtrain.gates import evaluate

        res = evaluate(clause(samples_per_prompt=4, eval_seeds=[11, 12, 13, 14]),
                       records(self.base), records(self.cand), seed=7, n_boot=4000)
        self.assertAlmostEqual(res["delta_hat"], self.delta, places=12)
        # Paired normal-theory width 2 * z_.975 * sd / sqrt(n); an unpaired
        # analysis would be about 6x wider here (per-arm SD ~ 0.5).
        paired_width = 2 * Z_975 * self.sd / math.sqrt(self.n)
        width = res["ci_upper"] - res["ci_lower"]
        self.assertLess(width, 1.5 * paired_width, (width, paired_width))
        self.assertGreater(width, 0.5 * paired_width, (width, paired_width))
        self.assertEqual(res["status"], "PASS")


class TestIntervalReporting(unittest.TestCase):
    def test_level_alpha_and_seed(self):
        from gatedtrain.gates import evaluate

        base, cand = diff_records(alternating(200, 0.1, 0.3))
        a = evaluate(clause(), base, cand, seed=11, n_boot=3000)
        b = evaluate(clause(), base, cand, seed=11, n_boot=3000)
        # 05 section 3.1: superiority and non-inferiority use the lower (1 - alpha)
        # bound; the reported interval is the two-sided (1 - 2 alpha) interval.
        self.assertAlmostEqual(a["ci_level"], 0.95, places=12)
        self.assertAlmostEqual(a["alpha"], 0.025, places=12)
        self.assertEqual((a["ci_lower"], a["ci_upper"]), (b["ci_lower"], b["ci_upper"]), "seedable (MVP-SPEC 2.7)")
        self.assertLessEqual(a["ci_lower"], a["delta_hat"])
        self.assertGreaterEqual(a["ci_upper"], a["delta_hat"])
        # Normal-theory reference: 0.1 +/- 1.96 * sd / sqrt(200)
        d = np.array(alternating(200, 0.1, 0.3))
        half = Z_975 * d.std(ddof=1) / math.sqrt(200)
        self.assertAlmostEqual(a["ci_lower"], 0.1 - half, delta=0.25 * half)
        self.assertAlmostEqual(a["ci_upper"], 0.1 + half, delta=0.25 * half)

    def test_equivalence_level(self):
        from gatedtrain.gates import evaluate

        base, cand = diff_records(alternating(200, 0.0, 0.3))
        res = evaluate(clause("equivalence"), base, cand, seed=11, n_boot=3000)
        self.assertAlmostEqual(res["alpha"], 0.05, places=12)
        self.assertAlmostEqual(res["ci_level"], 0.90, places=12)  # (1 - 2 alpha), 05:62


class TestAchievedPowerAndMde(unittest.TestCase):
    """04 section 4.2 rule 1 (04:102): achieved power at the declared MDE.
    Reference: power = Phi(d * sqrt(n) / (s * sqrt(DE)) - z_{1-alpha}),
    the 05:124 MDE formula solved for power; MDE = (z_{1-a} + z_{1-b}) s sqrt(DE / n)."""

    def test_superiority(self):
        from gatedtrain.gates import evaluate

        diffs = alternating(400, 0.2, 0.3)
        s = np.std(diffs, ddof=1)  # 0.3 * sqrt(400 / 399) = 0.3003757
        base, cand = diff_records(diffs)
        res = evaluate(clause(), base, cand, seed=1, n_boot=2000)
        self.assertAlmostEqual(res["mde"], (Z_975 + Z_80) * s / math.sqrt(400), places=9)
        self.assertAlmostEqual(res["achieved_power"], norm.cdf(0.10 * math.sqrt(400) / s - Z_975), places=9)
        self.assertAlmostEqual(res["power_target"], 0.8, places=12)

    def test_non_inferiority_distance_includes_margin(self):
        from gatedtrain.gates import evaluate

        diffs = alternating(30, 0.0, 0.3)
        s = np.std(diffs, ddof=1)
        base, cand = diff_records(diffs)
        res = evaluate(clause("non_inferiority"), base, cand, seed=1, n_boot=2000)
        # d = planning_effect + margin = 0.0 + 0.10
        self.assertAlmostEqual(res["achieved_power"], norm.cdf(0.10 * math.sqrt(30) / s - Z_975), places=9)
        self.assertAlmostEqual(res["mde"], (Z_975 + Z_80) * s / math.sqrt(30), places=9)

    def test_underpowered_flag_follows_power(self):
        from gatedtrain.gates import evaluate

        for n in (16, 40, 120, 400):
            with self.subTest(n=n):
                base, cand = diff_records(alternating(n, 0.0, 0.3))
                res = evaluate(clause(), base, cand, seed=1, n_boot=2000)
                self.assertEqual(res["underpowered"], res["achieved_power"] < 0.8)


class TestStatusRules(unittest.TestCase):
    """05 section 3.1 (05:82) and 04 section 4.2 rule 1 (04:102):
    an underpowered pass is PASS and flagged; an underpowered non-rejection is
    INCONCLUSIVE, never FAIL; a powered non-rejection is FAIL."""

    def check(self, type_, diffs, status, underpowered, **kw):
        from gatedtrain.gates import evaluate

        base, cand = diff_records(diffs)
        res = evaluate(clause(type_), base, cand, seed=5, n_boot=3000, **kw)
        self.assertEqual(res["status"], status, res)
        self.assertEqual(res["underpowered"], underpowered, res)
        return res

    def test_superiority_powered_pass(self):
        # lower 97.5% bound ~ 0.2 - 1.96 * 0.3 / 20 = 0.171 > 0; MDE 0.042 < 0.10
        res = self.check("superiority", alternating(400, 0.2, 0.3), "PASS", False)
        self.assertGreater(res["ci_lower"], 0.0)

    def test_superiority_underpowered_pass(self):
        # n = 16: lower ~ 0.5 - 2.13 * 0.075 = 0.34 > 0, but MDE 0.217 > 0.10
        self.check("superiority", alternating(16, 0.5, 0.3), "PASS", True)

    def test_superiority_underpowered_non_rejection_is_inconclusive(self):
        # n = 16, mean 0: lower < 0, MDE 0.217 > 0.10
        self.check("superiority", alternating(16, 0.0, 0.3), "INCONCLUSIVE", True)

    def test_superiority_powered_non_rejection_is_fail(self):
        # n = 2000, mean 0: lower ~ -0.013 < 0; MDE 0.0188 < 0.10
        self.check("superiority", alternating(2000, 0.0, 0.3), "FAIL", False)

    def test_superiority_margin_is_used(self):
        # mean 0.2, n = 400, delta_sup = 0.25 > lower bound 0.171 -> not rejected; powered -> FAIL
        from gatedtrain.gates import evaluate

        base, cand = diff_records(alternating(400, 0.2, 0.3))
        res = evaluate(clause(margin=0.25, planning_effect=0.40), base, cand, seed=5, n_boot=3000)
        self.assertEqual(res["status"], "FAIL")

    def test_plan_underpowered_forces_flag(self):
        """MVP-SPEC sections 2.3 and 5: a prompt set below the planned n is underpowered."""
        res = self.check("superiority", alternating(400, 0.2, 0.3), "PASS", True,
                         plan={"n_required": 1000, "underpowered": True})
        self.assertGreater(res["ci_lower"], 0.0)
        self.check("superiority", alternating(400, 0.0, 0.3), "INCONCLUSIVE", True,
                   plan={"n_required": 1000, "underpowered": True})

    def test_non_inferiority(self):
        # mean 0, n = 400: lower ~ -0.029 > -0.10 -> PASS
        self.check("non_inferiority", alternating(400, 0.0, 0.3), "PASS", False)
        # mean -0.3, n = 400: lower ~ -0.329 < -0.10; MDE 0.042 < 0.10 -> FAIL
        res = self.check("non_inferiority", alternating(400, -0.3, 0.3), "FAIL", False)
        self.assertLess(res["ci_lower"], -0.10)

    def test_equivalence(self):
        from gatedtrain.gates import evaluate

        # mean 0, n = 2000: 90% interval ~ +/- 0.011, inside (-0.10, 0.10) -> PASS
        base, cand = diff_records(alternating(2000, 0.0, 0.3))
        res = evaluate(clause("equivalence"), base, cand, seed=5, n_boot=3000)
        self.assertEqual(res["status"], "PASS")
        self.assertGreater(res["ci_lower"], -0.10)
        self.assertLess(res["ci_upper"], 0.10)

    def test_equivalence_non_rejection_is_inconclusive(self):
        """Revised for design revision 10 (R10-3, tests/README.md section 11.15; supersedes interim rule
        G-12): 05 section 3.6 now gives the TOST power formula, so only an underpowered equivalence
        non-rejection is INCONCLUSIVE; a powered one is FAIL (test_r10_stats.TestEquivalenceAchievedPower)."""
        from gatedtrain.gates import evaluate

        # mean 0.3, n = 16: the 90% interval is far outside (-0.10, 0.10); TOST power is 0 at n = 16
        base, cand = diff_records(alternating(16, 0.3, 0.3))
        res = evaluate(clause("equivalence"), base, cand, seed=5, n_boot=3000)
        self.assertEqual(res["achieved_power"], 0.0)
        self.assertTrue(res["underpowered"])
        self.assertEqual(res["status"], "INCONCLUSIVE")
        # With n = 400 the same effect is powered, so the non-rejection is FAIL.
        base, cand = diff_records(alternating(400, 0.3, 0.3))
        res = evaluate(clause("equivalence"), base, cand, seed=5, n_boot=3000)
        self.assertEqual(res["status"], "FAIL")
        self.assertFalse(res["underpowered"])

    def test_missing(self):
        """No paired records -> MISSING (NeMo Evaluator gate.py:152 at ea35ba3)."""
        from gatedtrain.gates import evaluate

        base, _ = diff_records(alternating(10, 0.0, 0.3))
        res = evaluate(clause(), base, {}, seed=5, n_boot=500)
        self.assertEqual(res["status"], "MISSING")
        res = evaluate(clause(), {}, {}, seed=5, n_boot=500)
        self.assertEqual(res["status"], "MISSING")

    def test_zero_variance(self):
        """Identical arms (no-training control): no exception, never PASS."""
        from gatedtrain.gates import evaluate

        base, cand = diff_records([0.0] * 300)
        res = evaluate(clause(), base, cand, seed=5, n_boot=500)
        self.assertEqual(res["delta_hat"], 0.0)
        self.assertIn(res["status"], {"FAIL", "INCONCLUSIVE"})

    def test_sd_zero_flag(self):
        """Interim rule G-13: SD of the paired differences = 0 keeps the power limit 1.0
        and sets `sd_zero: true`, so identical arms are told apart from a powered result."""
        from gatedtrain.gates import evaluate

        base, cand = diff_records([0.0] * 300)
        res = evaluate(clause(), base, cand, seed=5, n_boot=500)
        self.assertIs(res["sd_zero"], True)
        self.assertEqual(res["sd_diff"], 0.0)
        self.assertEqual(res["achieved_power"], 1.0)
        base, cand = diff_records(alternating(400, 0.2, 0.3))
        res = evaluate(clause(), base, cand, seed=5, n_boot=500)
        self.assertIs(res["sd_zero"], False)


class TestPairedScoresAndUnpairedCounts(unittest.TestCase):
    """Decision D-3 (MVP review LOW-3): scores are over the paired items only, and the
    unpaired items of each arm are counted. Items are (problem_idx, repeat) keys."""

    def test_counts_and_scores(self):
        from gatedtrain.gates import evaluate

        base = records(TestPointEstimateOnRawPairedDifferences.BASE)
        cand = records(TestPointEstimateOnRawPairedDifferences.CAND)
        cand[(99, 0)] = {"problem_idx": 99, "repeat": 0, "reward": 1.0}
        cand[(99, 1)] = {"problem_idx": 99, "repeat": 1, "reward": 1.0}
        base[(98, 0)] = {"problem_idx": 98, "repeat": 0, "reward": 0.0}
        res = evaluate(clause(samples_per_prompt=2, eval_seeds=[11, 12]), base, cand, seed=1, n_boot=500)
        self.assertEqual(res["unpaired_baseline"], 1)
        self.assertEqual(res["unpaired_candidate"], 2)
        # Paired rewards: base 1,0,0,0,1,1,0,1 -> 4/8; cand 1,1,1,0,1,1,0,0 -> 5/8.
        self.assertAlmostEqual(res["baseline_score"], 0.5, places=12)
        self.assertAlmostEqual(res["candidate_score"], 0.625, places=12)

    def test_no_unpaired(self):
        from gatedtrain.gates import evaluate

        base, cand = diff_records(alternating(10, 0.0, 0.3))
        res = evaluate(clause(), base, cand, seed=1, n_boot=500)
        self.assertEqual((res["unpaired_baseline"], res["unpaired_candidate"]), (0, 0))


class TestCombine(unittest.TestCase):
    """`combine(results, "and")`: intersection-union test (05 section 3.5, 05:114),
    verdict semantics of NeMo Evaluator `_aggregate_verdict` (gate.py:537 at ea35ba3):
    any FAIL or MISSING blocks; otherwise any INCONCLUSIVE gives INCONCLUSIVE."""

    def r(self, status):
        return {"status": status}

    def test_and(self):
        from gatedtrain.gates import combine

        self.assertEqual(combine([self.r("PASS"), self.r("PASS")], "and"), "PASS")
        self.assertEqual(combine([self.r("PASS"), self.r("FAIL")], "and"), "FAIL")
        self.assertEqual(combine([self.r("PASS"), self.r("INCONCLUSIVE")], "and"), "INCONCLUSIVE")
        self.assertEqual(combine([self.r("INCONCLUSIVE"), self.r("FAIL")], "and"), "FAIL")
        self.assertIn(combine([self.r("PASS"), self.r("MISSING")], "and"), {"FAIL", "MISSING"})
        self.assertIn(combine([self.r("INCONCLUSIVE"), self.r("MISSING")], "and"), {"FAIL", "MISSING"})
        self.assertEqual(combine([self.r("PASS")], "and"), "PASS")


if __name__ == "__main__":
    unittest.main()
