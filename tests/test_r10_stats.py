"""Design revision 10 alignment of the gate library (tests/README.md section 11.15).

Design (current files, revision 10):
- R10-3: 05 section 3.6 "Achieved power and `underpowered`" (05:138-143): power at the registered
  planning effect theta with the observed SD of the paired differences (ddof = 1), SE = sd * sqrt(DE / n);
  superiority / non-inferiority Phi(d / SE - z_{1-alpha}); equivalence (TOST)
  max(0, Phi((delta - theta) / SE - z) + Phi((delta + theta) / SE - z) - 1); TOST required n
  DE * ((z_{1-alpha} + z_{1-beta/2}) sigma / delta)^2 at theta = 0, DE * ((z_{1-alpha} + z_{1-beta})
  sigma / (delta - |theta|))^2 otherwise. `underpowered` on low power, n below plan, or power not
  computable. Supersedes interim rules G-3 (equivalence part) and G-12.
- R10-8: 05 section 3.4 "Missingness check" (05:116): `pairing.max_unpaired` (count or fraction,
  default 0), complete pairs only, per-arm missing counts, MISSING above the bound.
- R10-10: 05 section 3.6 "Minimum paired sample" (05:145): `min_paired` default 10; fewer pairs, or a
  zero-width interval (n < 2, or sd = 0), is INCONCLUSIVE, never PASS.
- R10-4: 05 section 3.8 status precedence for `and` (05:160).
- Review finding 8 (decision R10A-8): the actual K per prompt and the repeat -> seed mapping must match
  the registration (05 section 3.3, common random numbers, 05:97); otherwise MISSING.
- Review finding 10 (decision R10A-10): p-values in the evaluate() result (05 section 3.8 table, 05:166
  "Delta-hat, CI, p-value, ..."), computed by inverting the BCa interval (one-sided for superiority and
  non-inferiority, TOST = max of the two one-sided p-values for equivalence).
- Review follow-up (decisions R10A-14 to R10A-16): one zero-width criterion with a relative float tolerance
  (05:156); p-values pinned to the BCa inversion on skewed data; `p_value_at_floor` when the bootstrap tail
  behind the p-value is empty.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.stats import norm

from test_stats_evaluate import alternating, clause, diff_records, records

Z_975 = norm.ppf(0.975)
Z_95 = norm.ppf(0.95)
Z_90 = norm.ppf(0.90)
Z_80 = norm.ppf(0.80)


def evaluate(*args, **kw):
    from gatedtrain.gates import evaluate as impl

    return impl(*args, **kw)


def seeded_records(matrix, seeds):
    """{(i, r): record} whose metadata.eval_seed is seeds[r], as merge-eval and the cpu_stub write it."""
    out = records(matrix)
    for (i, r), rec in out.items():
        rec["metadata"] = {"eval_seed": seeds[r]}
    return out


def tost_power(delta, theta, se, alpha):
    z = norm.ppf(1 - alpha)
    return max(0.0, norm.cdf((delta - theta) / se - z) + norm.cdf((delta + theta) / se - z) - 1.0)


# ---------------------------------------------------------------------------
# R10A-1 (R10-3): achieved power and required n for equivalence
# ---------------------------------------------------------------------------

class TestEquivalencePlan(unittest.TestCase):
    """05 section 3.6 TOST required n; `plan()` no longer returns n_required = None for equivalence."""

    def test_required_n_at_theta_zero(self):
        """theta = 0: n = DE ((z_{0.95} + z_{1 - beta/2}) sigma / delta)^2 = ((1.6449 + 1.2816) 0.2 / 0.1)^2
        = 34.26 -> 35 (alpha 0.05, power 0.8, so beta / 2 = 0.1)."""
        from gatedtrain.gates import plan

        p = plan(clause("equivalence", sigma_plan="fixed"), {"sigma": 0.2, "source": "t"}, n_available=100)
        self.assertEqual(p["n_required"], math.ceil(((Z_95 + Z_90) * 0.2 / 0.10) ** 2))
        self.assertEqual(p["n_required"], 35)
        self.assertFalse(p["underpowered"])

    def test_required_n_theta_nonzero(self):
        """theta = 0.02: conservative n = ((z_{0.95} + z_{0.8}) 0.2 / (0.10 - 0.02))^2 = 38.6 -> 39."""
        from gatedtrain.gates import plan

        p = plan(clause("equivalence", sigma_plan="fixed", planning_effect=0.02), {"sigma": 0.2, "source": "t"},
                 n_available=100)
        self.assertEqual(p["n_required"], math.ceil(((Z_95 + Z_80) * 0.2 / 0.08) ** 2))
        self.assertEqual(p["n_required"], 39)

    def test_plan_underpowered_when_n_available_below(self):
        from gatedtrain.gates import plan

        p = plan(clause("equivalence", sigma_plan="fixed"), {"sigma": 0.2, "source": "t"}, n_available=30)
        self.assertEqual(p["n_required"], 35)
        self.assertTrue(p["underpowered"])

    def test_superiority_and_ni_unchanged(self):
        """R10-3 keeps the superiority and non-inferiority n formula (test_stats_plan pins 172 and 223)."""
        from gatedtrain.gates import plan

        p = plan(clause(sigma_plan="fixed"), {"sigma": 0.25, "source": "t"}, n_available=60)
        self.assertEqual(p["n_required"], math.ceil(((Z_975 + Z_80) * 0.25 / 0.10) ** 2))


class TestEquivalenceAchievedPower(unittest.TestCase):
    """evaluate() reports TOST achieved power by the 05 section 3.6 formula, at theta = planning_effect,
    with the observed SD (ddof = 1); it is no longer null."""

    def test_formula_theta_zero(self):
        diffs = alternating(40, 0.0, 0.3)
        s = float(np.std(diffs, ddof=1))
        res = evaluate(clause("equivalence"), *diff_records(diffs), seed=5, n_boot=2000)
        self.assertIsNotNone(res["achieved_power"])
        self.assertAlmostEqual(res["achieved_power"], tost_power(0.10, 0.0, s / math.sqrt(40), 0.05), places=9)

    def test_formula_theta_nonzero(self):
        diffs = alternating(400, 0.0, 0.3)
        s = float(np.std(diffs, ddof=1))
        res = evaluate(clause("equivalence", planning_effect=0.03), *diff_records(diffs), seed=5, n_boot=2000)
        self.assertAlmostEqual(res["achieved_power"], tost_power(0.10, 0.03, s / math.sqrt(400), 0.05), places=9)

    def test_power_is_floored_at_zero(self):
        """n = 16, sd 0.3: Phi(0.1 / 0.0775 - 1.645) = 0.36, so 2 * 0.36 - 1 < 0 -> 0."""
        res = evaluate(clause("equivalence"), *diff_records(alternating(16, 0.0, 0.3)), seed=5, n_boot=2000)
        self.assertEqual(res["achieved_power"], 0.0)
        self.assertTrue(res["underpowered"])

    def test_powered_non_rejection_is_fail(self):
        """Supersedes G-12: with a computable power at or above target, an equivalence non-rejection is
        FAIL (n = 400, sd 0.3, mean 0.3: power ~ 1, interval far outside (-0.1, 0.1))."""
        res = evaluate(clause("equivalence"), *diff_records(alternating(400, 0.3, 0.3)), seed=5, n_boot=3000)
        self.assertGreaterEqual(res["achieved_power"], 0.8)
        self.assertFalse(res["underpowered"])
        self.assertEqual(res["status"], "FAIL")

    def test_underpowered_non_rejection_is_inconclusive(self):
        res = evaluate(clause("equivalence"), *diff_records(alternating(16, 0.3, 0.3)), seed=5, n_boot=3000)
        self.assertTrue(res["underpowered"])
        self.assertEqual(res["status"], "INCONCLUSIVE")

    def test_plan_shortfall_sets_underpowered(self):
        """`underpowered` also when n is below the planned n (05 section 3.6)."""
        from gatedtrain.gates import plan

        c = clause("equivalence", sigma_plan="fixed")
        p = plan(c, {"sigma": 0.6, "source": "t"}, n_available=200)   # n_required = 309
        self.assertGreater(p["n_required"], 200)
        res = evaluate(c, *diff_records(alternating(200, 0.0, 0.05)), plan=p, seed=5, n_boot=2000)
        self.assertGreaterEqual(res["achieved_power"], 0.8, "observed power alone would not flag it")
        self.assertTrue(res["underpowered"])
        self.assertEqual(res["status"], "PASS", "an underpowered pass stays PASS, flagged")


# ---------------------------------------------------------------------------
# R10A-5 (R10-10): minimum paired sample and zero-width intervals
# ---------------------------------------------------------------------------

class TestMinPaired(unittest.TestCase):
    """Default min_paired = 10. Today's code returns PASS for n = 1 and for n = 3 identical
    differences; both must be INCONCLUSIVE."""

    def check_inconclusive(self, res, reason):
        self.assertEqual(res["status"], "INCONCLUSIVE", res)
        self.assertIn(reason, res.get("status_reasons", []), res)

    def test_n1_is_inconclusive(self):
        res = evaluate(clause(), *diff_records([0.5]), seed=5, n_boot=500)
        self.check_inconclusive(res, "min_paired")
        self.assertIn("zero_width_interval", res["status_reasons"])

    def test_n3_identical_differences_is_inconclusive(self):
        res = evaluate(clause(), *diff_records([0.5, 0.5, 0.5]), seed=5, n_boot=500)
        self.check_inconclusive(res, "min_paired")
        self.assertIs(res["sd_zero"], True)

    def test_n9_below_default(self):
        """n = 9 with a clear effect: lower bound > 0 today (PASS), but 9 < 10."""
        res = evaluate(clause(), *diff_records(alternating(9, 0.5, 0.1)), seed=5, n_boot=2000)
        self.assertGreater(res["ci_lower"], 0.0)
        self.check_inconclusive(res, "min_paired")

    def test_n10_at_default_passes(self):
        res = evaluate(clause(), *diff_records(alternating(10, 0.5, 0.1)), seed=5, n_boot=2000)
        self.assertEqual(res["status"], "PASS", res)
        self.assertNotIn("min_paired", res.get("status_reasons", []))
        self.assertEqual(res["min_paired"], 10)

    def test_registered_min_paired_is_used(self):
        res = evaluate(clause(), *diff_records(alternating(16, 0.5, 0.3)), seed=5, n_boot=2000, min_paired=20)
        self.check_inconclusive(res, "min_paired")
        self.assertEqual(res["min_paired"], 20)

    def test_sd_zero_large_n_is_inconclusive(self):
        """n = 300, every difference +0.2: zero-width interval at 0.2 > 0 is PASS today; R10-10 makes a
        `sd_zero: true` zero-width interval INCONCLUSIVE whatever its limiting power."""
        res = evaluate(clause(), *diff_records([0.2] * 300), seed=5, n_boot=500)
        self.assertIs(res["sd_zero"], True)
        self.assertEqual(res["achieved_power"], 1.0, "the G-13 limit stays")
        self.check_inconclusive(res, "zero_width_interval")

    def test_ordinary_pass_has_no_reasons(self):
        res = evaluate(clause(), *diff_records(alternating(400, 0.2, 0.3)), seed=5, n_boot=2000)
        self.assertEqual(res["status"], "PASS")
        self.assertEqual(res["status_reasons"], [])


class TestPostHocStatus(unittest.TestCase):
    """R10A-4 (05 section 3.7, 05:151) and the maintainers' decision of 2026-11-11: `post_hoc: true` turns only a
    PASS into INCONCLUSIVE (reason post_hoc); a FAIL stays FAIL, and INCONCLUSIVE and MISSING are
    unchanged. The statistics are kept. Inputs are real evaluate() results."""

    @staticmethod
    def post_hoc(res):
        from gatedtrain.gates import apply_post_hoc
        return apply_post_hoc(dict(res))

    def check_unchanged(self, res, status):
        self.assertEqual(res["status"], status, res)
        out = self.post_hoc(res)
        self.assertEqual(out, res, "post_hoc must not change a non-PASS result")
        self.assertNotIn("post_hoc", out["status_reasons"])

    def test_pass_becomes_inconclusive(self):
        res = evaluate(clause(), *diff_records(alternating(400, 0.2, 0.3)), seed=5, n_boot=2000)
        self.assertEqual(res["status"], "PASS")
        out = self.post_hoc(res)
        self.assertEqual(out["status"], "INCONCLUSIVE")
        self.assertEqual(out["status_reasons"], ["post_hoc"])
        for key in ("delta_hat", "ci_lower", "ci_upper", "achieved_power", "p_value", "underpowered"):
            self.assertEqual(out[key], res[key], key)

    def test_fail_stays_fail(self):
        """Powered superiority non-rejection (Delta-hat = -0.2, n = 400): FAIL, also when post hoc."""
        res = evaluate(clause(), *diff_records(alternating(400, -0.2, 0.3)), seed=5, n_boot=2000)
        self.assertFalse(res["underpowered"])
        self.check_unchanged(res, "FAIL")

    def test_inconclusive_unchanged(self):
        res = evaluate(clause(), *diff_records(alternating(9, 0.5, 0.1)), seed=5, n_boot=2000)
        self.check_unchanged(res, "INCONCLUSIVE")
        self.assertEqual(res["status_reasons"], ["min_paired"])

    def test_underpowered_inconclusive_unchanged(self):
        res = evaluate(clause(), *diff_records(alternating(20, -0.05, 0.3)), seed=5, n_boot=2000)
        self.check_unchanged(res, "INCONCLUSIVE")
        self.assertIn("underpowered", res["status_reasons"])

    def test_missing_unchanged(self):
        base, cand = diff_records(alternating(20, 0.2, 0.3))
        del cand[(3, 0)]
        res = evaluate(clause(), base, cand, seed=5, n_boot=500)
        self.check_unchanged(res, "MISSING")
        self.assertIn("missing_above_bound", res["status_reasons"])


# ---------------------------------------------------------------------------
# R10A-3 (R10-8): missingness check in evaluate()
# ---------------------------------------------------------------------------

class TestMissingness(unittest.TestCase):
    """Expected items E = {(p, r): p in either arm, r in range(K)}. missing_<arm> = |E - arm|.
    The bound: missing_baseline + missing_candidate <= max_unpaired (an int count) or
    <= max_unpaired * 2|E| (a fraction in [0, 1)). Above it the clause is MISSING
    (reason `missing_above_bound`); Delta-hat and the interval are still reported on complete pairs."""

    def arms(self, n=40, drop_candidate=()):
        base, cand = diff_records(alternating(n, 0.5, 0.1))
        for k in drop_candidate:
            cand.pop(k)
        return base, cand

    def test_default_zero_one_missing_item_is_missing(self):
        base, cand = self.arms(drop_candidate=[(3, 0)])
        res = evaluate(clause(), base, cand, seed=5, n_boot=1000)
        self.assertEqual(res["status"], "MISSING", res)
        self.assertIn("missing_above_bound", res["status_reasons"])
        self.assertEqual((res["missing_baseline"], res["missing_candidate"]), (0, 1))
        self.assertEqual(res["max_unpaired"], 0)
        self.assertIsNotNone(res["delta_hat"], "complete pairs are still analysed and reported")
        self.assertEqual(res["n_prompts"], 39)

    def test_count_bound(self):
        base, cand = self.arms(drop_candidate=[(3, 0), (5, 0)])
        self.assertEqual(evaluate(clause(), base, cand, seed=5, n_boot=1000, max_unpaired=2)["status"], "PASS")
        self.assertEqual(evaluate(clause(), base, cand, seed=5, n_boot=1000, max_unpaired=1)["status"], "MISSING")

    def test_fraction_bound(self):
        """2 missing of 2|E| = 80 expected items = 2.5%."""
        base, cand = self.arms(drop_candidate=[(3, 0), (5, 0)])
        self.assertEqual(evaluate(clause(), base, cand, seed=5, n_boot=1000, max_unpaired=0.03)["status"], "PASS")
        self.assertEqual(evaluate(clause(), base, cand, seed=5, n_boot=1000, max_unpaired=0.02)["status"], "MISSING")

    def test_item_missing_in_both_arms_counts(self):
        """Repeat 1 of prompt 0 is absent from both arms: not unpaired, but missing from the registered
        K = 2 design in each arm."""
        b = records([[1, 1]] * 20)
        c = records([[1, 0]] * 20)
        b.pop((0, 1))
        c.pop((0, 1))
        res = evaluate(clause(samples_per_prompt=2, eval_seeds=[11, 12]), b, c, seed=5, n_boot=500)
        self.assertEqual((res["missing_baseline"], res["missing_candidate"]), (1, 1))
        self.assertEqual(res["status"], "MISSING")

    def test_complete_arms(self):
        base, cand = self.arms()
        res = evaluate(clause(), base, cand, seed=5, n_boot=1000)
        self.assertEqual((res["missing_baseline"], res["missing_candidate"]), (0, 0))
        self.assertEqual(res["status"], "PASS")


# ---------------------------------------------------------------------------
# R10A-8 (review finding 8): actual K and the repeat -> seed mapping
# ---------------------------------------------------------------------------

class TestKAndSeeds(unittest.TestCase):
    K4 = dict(samples_per_prompt=4, eval_seeds=[11, 12, 13, 14])

    def strong(self, n, k):
        return [[0] * k for _ in range(n)], [[1] * k if i % 4 else [0] * k for i in range(n)]

    def test_baseline_k4_candidate_k1(self):
        """R10A-8 made this MISSING (`k_mismatch`). Design revision 11 (R11-7, R11A-7) reverses it: fewer
        than K records are missing items only, here 1200 of 3200 expected, below max_unpaired 0.99, so the
        gate PASSes on repeat 0 (superseded by test_r11_items.TestClauseMissing.test_candidate_k1_against_k4).
        `k_actual` still reports the repeats per prompt."""
        b, _ = self.strong(400, 4)
        _, c = self.strong(400, 1)
        res = evaluate(clause(**self.K4), records(b), records(c), seed=5, n_boot=1000, max_unpaired=0.99)
        self.assertEqual(res["status"], "PASS", res)
        self.assertNotIn("k_mismatch", res["status_reasons"])
        self.assertEqual((res["missing_baseline"], res["missing_candidate"]), (0, 1200))
        self.assertEqual(res["k_actual"], {"baseline": [4, 4], "candidate": [1, 1]})

    def test_both_arms_k1_under_registered_k4(self):
        """R10A-8 made this MISSING (`k_mismatch`). Design revision 11 (R11A-7) reverses it: 1200 missing items
        per arm, below max_unpaired 0.99, so the gate PASSes (superseded by test_r11_items.TestClauseMissing
        .test_fewer_than_k_in_every_item_is_missing_items_only)."""
        b, c = self.strong(400, 1)
        res = evaluate(clause(**self.K4), records(b), records(c), seed=5, n_boot=1000, max_unpaired=0.99)
        self.assertEqual(res["status"], "PASS", res)
        self.assertNotIn("k_mismatch", res["status_reasons"])
        self.assertEqual((res["missing_baseline"], res["missing_candidate"]), (1200, 1200))
        self.assertEqual(res["k_actual"], {"baseline": [1, 1], "candidate": [1, 1]})

    def test_repeat_outside_registered_range(self):
        b, c = self.strong(40, 4)
        rb, rc = records(b), records(c)
        rb[(0, 4)] = {"problem_idx": 0, "repeat": 4, "reward": 0.0}
        rc[(0, 4)] = {"problem_idx": 0, "repeat": 4, "reward": 1.0}
        res = evaluate(clause(**self.K4), rb, rc, seed=5, n_boot=500)
        self.assertEqual(res["status"], "MISSING")
        self.assertIn("k_mismatch", res["status_reasons"])

    def test_seed_mapping_differs(self):
        """metadata.eval_seed of repeat r must be eval_seeds[r] (common random numbers, 05:97)."""
        b, c = self.strong(40, 4)
        rb = seeded_records(b, [11, 12, 13, 14])
        rc = seeded_records(c, [12, 11, 13, 14])
        res = evaluate(clause(**self.K4), rb, rc, seed=5, n_boot=500)
        self.assertEqual(res["status"], "MISSING", res)
        self.assertIn("seed_mismatch", res["status_reasons"])

    def test_matching_k_and_seeds(self):
        b, c = self.strong(40, 4)
        res = evaluate(clause(**self.K4), seeded_records(b, [11, 12, 13, 14]), seeded_records(c, [11, 12, 13, 14]),
                       seed=5, n_boot=500)
        self.assertEqual(res["k_actual"], {"baseline": [4, 4], "candidate": [4, 4]})
        self.assertNotIn("k_mismatch", res["status_reasons"])
        self.assertNotIn("seed_mismatch", res["status_reasons"])
        self.assertEqual(res["status"], "PASS")

    def test_records_without_seed_metadata_are_not_a_mismatch(self):
        """The unit-test records of test_stats_evaluate carry no metadata; only a present seed is checked."""
        b, c = self.strong(40, 4)
        res = evaluate(clause(**self.K4), records(b), records(c), seed=5, n_boot=500)
        self.assertNotIn("seed_mismatch", res["status_reasons"])


# ---------------------------------------------------------------------------
# R10A-10 (review finding 10): p-values
# ---------------------------------------------------------------------------

class TestPValues(unittest.TestCase):
    """`p_value` and `p_value_method = "bca_inversion"`. One-sided p for H0: Delta <= delta_sup
    (superiority) or Delta <= -delta (non-inferiority): the smallest alpha' whose BCa lower bound
    reaches the null value, p = Phi(w / (1 + a w) - z0) with w = Phi^-1(G(null)) - z0, G the bootstrap
    CDF with ties counted half and clipped to [1/(B+1), B/(B+1)] as z0 is. Equivalence: TOST p = max of
    the two one-sided p-values (H0: Delta <= -delta; H0: Delta >= delta). Null when the interval has zero
    width or there are no pairs. By construction p < alpha exactly when the bound rejects, up to the
    bootstrap quantile interpolation, so the tests stay away from the boundary."""

    def test_keys_and_method(self):
        res = evaluate(clause(), *diff_records(alternating(400, 0.2, 0.3)), seed=5, n_boot=2000)
        self.assertEqual(res["p_value_method"], "bca_inversion")
        self.assertGreaterEqual(res["p_value"], 0.0)
        self.assertLessEqual(res["p_value"], 1.0)

    def test_strong_effect_small_p(self):
        res = evaluate(clause(), *diff_records(alternating(400, 0.2, 0.3)), seed=5, n_boot=2000)
        self.assertEqual(res["status"], "PASS")
        self.assertLess(res["p_value"], 0.025)

    def test_null_effect_p_near_half(self):
        res = evaluate(clause(), *diff_records(alternating(2000, 0.0, 0.3)), seed=5, n_boot=2000)
        self.assertNotEqual(res["status"], "PASS")
        self.assertGreater(res["p_value"], 0.3)
        self.assertLess(res["p_value"], 0.7)

    def test_agrees_with_normal_approximation(self):
        """Mean 0.03, sd 0.3, n = 400: z = 0.03 / (0.3 / 20) = 2, normal p = 0.0228."""
        diffs = alternating(400, 0.03, 0.3)
        s = float(np.std(diffs, ddof=1))
        p_norm = 1 - norm.cdf(0.03 / (s / math.sqrt(400)))
        res = evaluate(clause(), *diff_records(diffs), seed=5, n_boot=10000)
        self.assertAlmostEqual(res["p_value"], p_norm, delta=0.01)

    def test_consistent_with_the_decision(self):
        for mean in (-0.1, 0.0, 0.01, 0.08, 0.2):
            with self.subTest(mean=mean):
                res = evaluate(clause(), *diff_records(alternating(400, mean, 0.3)), seed=5, n_boot=4000)
                if res["ci_lower"] > 0.005:
                    self.assertLess(res["p_value"], 0.025, res)
                if res["ci_lower"] < -0.005:
                    self.assertGreater(res["p_value"], 0.025, res)

    def test_non_inferiority_uses_the_margin(self):
        """Mean 0, delta 0.10, n = 400: z = 0.10 / 0.015 = 6.7, so p is tiny although Delta-hat = 0."""
        res = evaluate(clause("non_inferiority"), *diff_records(alternating(400, 0.0, 0.3)), seed=5, n_boot=2000)
        self.assertEqual(res["status"], "PASS")
        self.assertLess(res["p_value"], 0.001)

    def test_equivalence_tost(self):
        inside = evaluate(clause("equivalence"), *diff_records(alternating(2000, 0.0, 0.3)), seed=5, n_boot=2000)
        self.assertEqual(inside["status"], "PASS")
        self.assertLess(inside["p_value"], 0.05)
        outside = evaluate(clause("equivalence"), *diff_records(alternating(400, 0.3, 0.3)), seed=5, n_boot=2000)
        self.assertGreater(outside["p_value"], 0.5, "the upper one-sided test cannot reject")

    def test_zero_width_gives_null(self):
        res = evaluate(clause(), *diff_records([0.2] * 300), seed=5, n_boot=500)
        self.assertIsNone(res["p_value"])

    def test_no_pairs_gives_null(self):
        base, _ = diff_records(alternating(10, 0.0, 0.3))
        self.assertIsNone(evaluate(clause(), base, {}, seed=5, n_boot=500)["p_value"])


# ---------------------------------------------------------------------------
# R10A-14 (review follow-up MEDIUM-1): one zero-width criterion with a relative tolerance
# ---------------------------------------------------------------------------

ZW_REL_TOL = 1e-12   # R10A-14: degenerate when max(d) - min(d) <= 1e-12 * max(1, |mean(d)|)


def k3_one_more_correct(n=30):
    """K = 3 binary rewards; the candidate has exactly one more correct repeat on every prompt, so every
    per-prompt difference is 1/3. Float means give 0.3333333333333333 and 0.33333333333333337 (spread
    5.6e-17, sd 9.8e-17): review probe p1.py."""
    base = records([[1.0 if r < i % 3 else 0.0 for r in range(3)] for i in range(n)])
    cand = records([[1.0 if r < i % 3 + 1 else 0.0 for r in range(3)] for i in range(n)])
    return base, cand


K3 = {"samples_per_prompt": 3, "eval_seeds": [11, 12, 13]}


class TestZeroWidthTolerance(unittest.TestCase):
    """05 section 3.6 (05:156) and R10A-5: a zero-width interval (n < 2, or no between-item variation) is
    INCONCLUSIVE with reason `zero_width_interval`, `sd_zero: true` and `p_value: null`. R10A-14: the
    interval, the p-value, `sd_zero` and the status use one criterion, which tolerates float noise:
    max(d) - min(d) <= 1e-12 * max(1, |mean(d)|). Today each of them tests `std() == 0.0` exactly, so the
    p1 data (sd 9.8e-17) pass with ci_lower == ci_upper and a p-value."""

    def check_degenerate(self, res):
        self.assertEqual(res["status"], "INCONCLUSIVE", res)
        self.assertIn("zero_width_interval", res["status_reasons"], res)
        self.assertIs(res["sd_zero"], True, res)
        self.assertIsNone(res["p_value"], res)
        self.assertEqual(res["ci_lower"], res["ci_upper"], res)
        # G-13: power keeps its limit (1.0 here: planning distance > 0, or delta > |theta| for TOST) although
        # the clause is INCONCLUSIVE (05 section 3.6 "whatever its limiting power").
        self.assertEqual(res["achieved_power"], 1.0, res)

    def check_genuine(self, res, status="PASS"):
        self.assertEqual(res["status"], status, res)
        self.assertNotIn("zero_width_interval", res["status_reasons"], res)
        self.assertIs(res["sd_zero"], False, res)
        self.assertIsNotNone(res["p_value"], res)
        self.assertLess(res["ci_lower"], res["ci_upper"], res)

    def test_probe_data_is_float_noise(self):
        """Guards the fixture: the differences are 1/3 up to float noise, not exactly equal."""
        from gatedtrain.gates import paired_differences

        d = paired_differences(*k3_one_more_correct())[3]
        self.assertGreater(float(d.std()), 0.0, "the fixture must carry float noise")
        self.assertLessEqual(float(d.max() - d.min()), ZW_REL_TOL)

    def test_superiority_float_noise_is_zero_width(self):
        res = evaluate(clause(**K3), *k3_one_more_correct(), seed=1, n_boot=2000)
        self.check_degenerate(res)
        self.assertAlmostEqual(res["delta_hat"], 1 / 3, places=12)

    def test_equivalence_float_noise_is_zero_width(self):
        """Same data, equivalence with delta = 0.5: the point interval at 1/3 lies inside (-0.5, 0.5),
        PASS today."""
        res = evaluate(clause("equivalence", margin=0.5, **K3), *k3_one_more_correct(), seed=1, n_boot=2000)
        self.check_degenerate(res)

    def test_library_functions_share_the_criterion(self):
        """`bca_interval` gives a zero-width interval and `bca_p_value` None on the same data."""
        from gatedtrain.gates import bca_interval, bca_p_value, paired_differences

        d = paired_differences(*k3_one_more_correct())[3]
        lo, hi = bca_interval(d, alpha=0.025, seed=1, n_boot=2000)
        self.assertEqual(lo, hi)
        for hyp, m in (("superiority", 0.0), ("non_inferiority", 0.1), ("equivalence", 0.5)):
            with self.subTest(hypothesis=hyp):
                self.assertIsNone(bca_p_value(d, hypothesis=hyp, margin=m, seed=1, n_boot=2000))

    def test_spread_below_tolerance_is_zero_width(self):
        """0.2 +/- 5e-14: spread 1e-13, a tenth of the tolerance."""
        res = evaluate(clause(), *diff_records(alternating(40, 0.2, 5e-14)), seed=5, n_boot=2000)
        self.check_degenerate(res)

    def test_tolerance_is_relative_to_the_mean(self):
        """Mean 1000, spread 2e-10: below 1e-12 * 1000 = 1e-9, so zero width although sd ~ 1e-10."""
        res = evaluate(clause(), *diff_records(alternating(40, 1000.0, 1e-10)), seed=5, n_boot=2000)
        self.check_degenerate(res)

    def test_tiny_genuine_spread_superiority(self):
        """0.2 +/- 5e-12: spread 1e-11, ten times the tolerance, stays a real (narrow) interval."""
        res = evaluate(clause(), *diff_records(alternating(40, 0.2, 5e-12)), seed=5, n_boot=2000)
        self.check_genuine(res)

    def test_tiny_genuine_spread_equivalence(self):
        res = evaluate(clause("equivalence"), *diff_records(alternating(40, 0.0, 5e-12)), seed=5, n_boot=2000)
        self.check_genuine(res)


# ---------------------------------------------------------------------------
# R10A-15 (review follow-up MEDIUM-2): p-values pinned to the BCa inversion on skewed data
# ---------------------------------------------------------------------------

def skewed(n, n_out, out, lo, hi, sign=1.0):
    """n - n_out evenly spaced differences in [lo, hi] and n_out outliers at `out` (times `sign`):
    right-skewed for sign = 1, so the BCa acceleration a and bias z0 are clearly nonzero (a ~ 0.06-0.07,
    z0 ~ 0.05-0.06 for the cases below)."""
    return [sign * round(float(x), 6) for x in [*np.linspace(lo, hi, n - n_out), *[out] * n_out]]


def p_by_bisection(diffs, null, *, seed, n_boot, upper=False):
    """Independent expected p: the alpha' at which the real `bca_interval` (same seed, so the same bootstrap
    draws) has its lower bound at `null` (or, upper=True, its upper bound at `null`, for H0: Delta >= null).
    Uses only the interval, never the p-value code."""
    from gatedtrain.gates import bca_interval

    lo_a, hi_a = 1e-7, 0.5
    for _ in range(60):
        mid = 0.5 * (lo_a + hi_a)
        lower, upper_b = bca_interval(diffs, alpha=mid, seed=seed, n_boot=n_boot)
        rejects = (upper_b < null) if upper else (lower > null)
        if rejects:
            hi_a = mid
        else:
            lo_a = mid
    return 0.5 * (lo_a + hi_a)


class TestPValuesSkewed(unittest.TestCase):
    """R10A-15: the p-value is the BCa inversion of R10A-10, checked on skewed data where it differs from
    the percentile p and from sign errors. Expected p by bisection on alpha' of the real `bca_interval`
    (independent of `bca_p_value`); tolerance 3/B, the quantile-interpolation gap. TestPValues uses symmetric
    `alternating` data (a = 0, z0 ~ 0), where `w/(1 - a w)`, `+z0` and the percentile `Phi(w)` all give the
    same p and survive. On these cases each of the three mutants moves p by at least 0.004 (> 5/B)."""

    SEED, B = 7, 4000
    TOL = 3.0 / 4000

    def check(self, clause_, diffs, expected_parts):
        from gatedtrain.gates import paired_differences

        base, cand = diff_records(diffs)
        d = paired_differences(base, cand)[3]
        res = evaluate(clause_, base, cand, seed=self.SEED, n_boot=self.B)
        expected = max(p_by_bisection(d, null, seed=self.SEED, n_boot=self.B, upper=up)
                       for null, up in expected_parts)
        self.assertIsNotNone(res["p_value"], res)
        self.assertAlmostEqual(res["p_value"], expected, delta=self.TOL,
                               msg=f"p_value {res['p_value']} vs BCa inversion {expected}")
        return res, expected

    def test_superiority_n30(self):
        """n = 30, three outliers at 2.0, null 0.05: p = 0.0438 (percentile 0.070, +z0 0.056, 1 - a w 0.081)."""
        self.check(clause(margin=0.05), skewed(30, 3, 2.0, -0.3, 0.3), [(0.05, False)])

    def test_superiority_n30_null_0_1(self):
        """Same data, null 0.10: p = 0.140 (percentile 0.171, +z0 0.169, 1 - a w 0.170)."""
        self.check(clause(margin=0.10), skewed(30, 3, 2.0, -0.3, 0.3), [(0.10, False)])

    def test_superiority_n24(self):
        """n = 24, two outliers at 1.5, null 0.10: p = 0.0389 (percentile 0.064, +z0 0.049, 1 - a w 0.078)."""
        self.check(clause(margin=0.10), skewed(24, 2, 1.5, -0.2, 0.4), [(0.10, False)])

    def test_superiority_n40(self):
        """n = 40, four outliers at 3.0, null 0.05: p = 0.0145 (percentile 0.029, +z0 0.019, 1 - a w 0.040)."""
        self.check(clause(margin=0.05), skewed(40, 4, 3.0, -0.5, 0.5), [(0.05, False)])

    def test_non_inferiority(self):
        """The n = 30 data shifted by -0.15, delta = 0.10: the null -0.10 sits where 0.05 sat above."""
        diffs = [x - 0.15 for x in skewed(30, 3, 2.0, -0.3, 0.3)]
        self.check(clause("non_inferiority", margin=0.10), diffs, [(-0.10, False)])

    def test_equivalence_upper_side_decides(self):
        """Right-skewed, delta = 0.4: TOST p = max(lower p at -0.4, upper p at 0.4) = the upper one, 0.0878
        (percentile 0.058, +z0 0.070, 1 - a w 0.044)."""
        res, expected = self.check(clause("equivalence", margin=0.4), skewed(30, 3, 2.0, -0.3, 0.3),
                                   [(-0.4, False), (0.4, True)])
        self.assertGreater(expected, 0.05)

    def test_equivalence_lower_side_decides(self):
        """The mirror image (left-skewed): the lower one-sided p decides, 0.0879."""
        res, expected = self.check(clause("equivalence", margin=0.4), skewed(30, 3, 2.0, -0.3, 0.3, sign=-1.0),
                                   [(-0.4, False), (0.4, True)])
        self.assertGreater(expected, 0.05)


# ---------------------------------------------------------------------------
# R10A-16 (review follow-up LOW-1): p-value at the bootstrap floor
# ---------------------------------------------------------------------------

def report_for(res, c):
    """build_report with placeholder provenance, for the p-value fields only."""
    from gatedtrain.report import build_report

    hashes = {k: "sha256:" + "0" * 64 for k in (
        "registration_hash", "item_set_hash", "evaluator_hash", "evaluator_hash_registered", "sampler_hash",
        "baseline_checkpoint_hash", "candidate_checkpoint_hash", "lockfile_hash")}
    plan_ = {"n_required": 10, "mde": 0.1, "design_effect": 1.0, "sigma_plan": "fixed", "sigma_used": 0.3,
             "n_available": res["n_prompts"], "underpowered": False}
    return build_report(workflow="w", gate_id="g", clause=c, result=res, plan=plan_, registration={},
                        hashes=hashes, stubs=[], created_at="2026-11-11T00:00:00Z", run_id="r",
                        disjoint_declared=False)


class TestPValueAtFloor(unittest.TestCase):
    """R10A-16: when the bootstrap tail count behind the reported p is 0 (or B for the upper TOST side), G is
    clipped to 1/(B+1) (or B/(B+1)) and the p-value is a floor, not an estimate (the real long run reported
    8.54e-05 this way). `evaluate()` and the report give `p_value_at_floor` (true then, false otherwise,
    including a null p-value); the text line reads `p_value: <= <p> (bca_inversion)` at the floor."""

    def floor_line(self, rep):
        from gatedtrain.report import render_text

        lines = [ln for ln in render_text(rep).splitlines() if ln.startswith("p_value:")]
        self.assertEqual(len(lines), 1, lines)
        return lines[0]

    def test_strong_superiority_is_at_floor(self):
        """Mean 0.2, SE 0.015, null 0: no bootstrap mean is <= 0."""
        c = clause()
        res = evaluate(c, *diff_records(alternating(400, 0.2, 0.3)), seed=5, n_boot=2000)
        self.assertIs(res.get("p_value_at_floor"), True, res)
        rep = report_for(res, c)
        self.assertIs(rep.get("p_value_at_floor"), True)
        self.assertRegex(self.floor_line(rep), r"^p_value: <= \d")

    def test_strong_equivalence_is_at_floor(self):
        """n = 2000, mean 0, delta 0.10 (15 SE): both TOST tails are empty."""
        c = clause("equivalence")
        res = evaluate(c, *diff_records(alternating(2000, 0.0, 0.3)), seed=5, n_boot=2000)
        self.assertIs(res.get("p_value_at_floor"), True, res)
        self.assertRegex(self.floor_line(report_for(res, c)), r"^p_value: <= \d")

    def test_interior_p_is_not_at_floor(self):
        """The skewed n = 30 superiority case (p = 0.044) has a nonempty tail."""
        c = clause(margin=0.05)
        res = evaluate(c, *diff_records(skewed(30, 3, 2.0, -0.3, 0.3)), seed=7, n_boot=4000)
        self.assertIs(res.get("p_value_at_floor"), False, res)
        rep = report_for(res, c)
        self.assertIs(rep.get("p_value_at_floor"), False)
        self.assertRegex(self.floor_line(rep), r"^p_value: \d")

    def test_p_near_one_is_not_a_floor(self):
        """Mean -0.2: every bootstrap mean is below 0, G clips at B/(B+1) and p ~ 1 is a ceiling, not a floor."""
        res = evaluate(clause(), *diff_records(alternating(400, -0.2, 0.3)), seed=5, n_boot=2000)
        self.assertGreater(res["p_value"], 0.99)
        self.assertIs(res.get("p_value_at_floor"), False, res)

    def test_null_p_is_not_at_floor(self):
        c = clause()
        res = evaluate(c, *diff_records([0.2] * 300), seed=5, n_boot=500)
        self.assertIsNone(res["p_value"])
        self.assertIs(res.get("p_value_at_floor"), False, res)
        self.assertEqual(self.floor_line(report_for(res, c)), "p_value: null (bca_inversion)")

    # Branch coverage of `bca_p_value_detail` (review follow-up, 2026-11-11; section 11.16 R11A-22).

    def test_no_pairs_is_not_at_floor(self):
        """The n == 0 MISSING path returns early: the key is present and false."""
        base, _ = diff_records(alternating(10, 0.0, 0.3))
        res = evaluate(clause(), base, {}, seed=5, n_boot=500)
        self.assertEqual(res["status"], "MISSING")
        self.assertIsNone(res["p_value"])
        self.assertIn("p_value_at_floor", res)
        self.assertIs(res["p_value_at_floor"], False)

    def test_non_inferiority_at_floor(self):
        """Mean 0, delta 0.10, SE 0.015: no bootstrap mean is <= -0.10 (p = 7.1e-4 at B = 2000)."""
        res = evaluate(clause("non_inferiority"), *diff_records(alternating(400, 0.0, 0.3)), seed=5, n_boot=2000)
        self.assertEqual(res["status"], "PASS")
        self.assertIs(res["p_value_at_floor"], True, res)

    def test_non_inferiority_at_ceiling_is_not_a_floor(self):
        """Mean -0.3, delta 0.10: every bootstrap mean is below -0.10, G clips at B/(B+1), p ~ 0.9994."""
        res = evaluate(clause("non_inferiority"), *diff_records(alternating(400, -0.3, 0.3)), seed=5, n_boot=2000)
        self.assertGreater(res["p_value"], 0.99)
        self.assertIs(res["p_value_at_floor"], False, res)

    def test_tost_upper_side_decides_at_floor(self):
        """Right-skewed n = 60 (one outlier at 2.0), shifted 2 SE below 0, delta = 4 SE = 0.1614: the lower side
        is interior (17 of 2000 bootstrap means <= -delta, p_lo = 4.7e-4); no bootstrap mean is >= delta, so the
        upper side is at its floor, and with the acceleration a = 0.089 its p (8.7e-3) is the larger and is
        reported. The mutant that takes the lower side's floor flag reports false."""
        base = [round(float(x), 6) for x in np.linspace(-0.3, 0.3, 59)] + [2.0]
        m = float(np.mean(base))
        se = float(np.std(base, ddof=1)) / math.sqrt(60)
        diffs = [round(x - m - 2.0 * se, 6) for x in base]
        delta = round(4.0 * se, 4)
        self.assertEqual(delta, 0.1614, "fixture guard")
        res = evaluate(clause("equivalence", margin=delta), *diff_records(diffs), seed=5, n_boot=2000)
        self.assertAlmostEqual(res["p_value"], 8.70e-3, delta=1e-4)
        self.assertIs(res["p_value_at_floor"], True, res)

    def test_tost_deciding_side_interior_is_not_a_floor(self):
        """Mean 0.065, delta 0.10, SE 0.015: the lower side (11 SE away) is at its floor, the upper side (2.3 SE
        away) is interior and decides (p = 0.0091): false."""
        res = evaluate(clause("equivalence"), *diff_records(alternating(400, 0.065, 0.3)), seed=5, n_boot=4000)
        self.assertAlmostEqual(res["p_value"], 0.0091, delta=5e-4)
        self.assertIs(res["p_value_at_floor"], False, res)


# ---------------------------------------------------------------------------
# R10-4: combine precedence (pins today's behaviour more tightly than test_stats_evaluate)
# ---------------------------------------------------------------------------

class TestCombinePrecedence(unittest.TestCase):
    """05 section 3.8 (05:160): for `and`, FAIL if any fails, else MISSING if any is missing, else
    INCONCLUSIVE if any is inconclusive, else PASS."""

    def test_and(self):
        from gatedtrain.gates import combine

        r = lambda s: {"status": s}   # noqa: E731
        self.assertEqual(combine([r("PASS"), r("MISSING")], "and"), "MISSING")
        self.assertEqual(combine([r("INCONCLUSIVE"), r("MISSING")], "and"), "MISSING")
        self.assertEqual(combine([r("FAIL"), r("MISSING")], "and"), "FAIL")
        self.assertEqual(combine([r("INCONCLUSIVE"), r("PASS")], "and"), "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
