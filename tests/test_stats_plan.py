"""Statistics unit tests: power, MDE, sigma planning and sample size.

Design: 05 section 3.6 (05-evaluation-and-lineage.md:120-132),
section 3.8 `plan()` (05:148), 04 section 4.2 field mapping (04:98).

Every reference value is computed independently with SciPy (norm, chi2)
or by hand, and the computation is shown next to it.
"""

from __future__ import annotations

import math
import unittest

from scipy.stats import chi2, norm

Z_975 = norm.ppf(0.975)  # 1.959963984540054
Z_95 = norm.ppf(0.95)    # 1.644853626951472
Z_80 = norm.ppf(0.80)    # 0.8416212335729143
Z_90 = norm.ppf(0.90)    # 1.2815515655446004


def sup_clause(**over):
    c = {
        "type": "superiority", "metric": "exact_match", "margin": 0.0, "alpha": 0.025,
        "power": 0.8, "planning_effect": 0.10, "sigma_plan": "ucb80", "unit": "prompt",
        "samples_per_prompt": 4, "eval_seeds": [11, 12, 13, 14],
        "interval": "cluster_bootstrap", "blocking": True,
    }
    c.update(over)
    return c


class TestSigmaPlanning(unittest.TestCase):
    """05 section 3.6 (05:130): sigma_U = s * sqrt(df / chi2_{0.20,df})."""

    def test_ucb80_factor_df4(self):
        from gatedtrain.gates import sigma_ucb80_factor

        # chi2.ppf(0.20, 4) = 1.648776618065969 -> sqrt(4 / 1.648776...) = 1.55757542255663
        ref = math.sqrt(4 / chi2.ppf(0.20, 4))
        self.assertAlmostEqual(sigma_ucb80_factor(4), ref, places=10)
        self.assertEqual(round(sigma_ucb80_factor(4), 2), 1.56)  # "sigma_U = 1.56 s" (05:130)

    def test_ucb80_factor_df1(self):
        from gatedtrain.gates import sigma_ucb80_factor

        # chi2.ppf(0.20, 1) = 0.0641847546673016 -> sqrt(1 / 0.06418...) = 3.9471538755427455
        ref = math.sqrt(1 / chi2.ppf(0.20, 1))
        self.assertAlmostEqual(sigma_ucb80_factor(1), ref, places=10)
        self.assertEqual(round(sigma_ucb80_factor(1), 2), 3.95)  # "sigma_U = 3.95 s" (05:130)

    def test_ucb80_factor_df9(self):
        from gatedtrain.gates import sigma_ucb80_factor

        # chi2.ppf(0.20, 9) = 5.380053211773292 -> sqrt(9 / 5.380...) = 1.2933854444729525
        self.assertAlmostEqual(sigma_ucb80_factor(9), math.sqrt(9 / chi2.ppf(0.20, 9)), places=10)


class TestDesignEffectAndMde(unittest.TestCase):
    """05 section 3.6 (05:124): MDE = (z_{1-a} + z_{1-b}) * sigma_d * sqrt(DE / n), DE = 1 + (m - 1) rho."""

    def test_design_effect(self):
        from gatedtrain.gates import design_effect

        self.assertAlmostEqual(design_effect(4, 0.1), 1.3, places=12)   # 1 + 3 * 0.1
        self.assertAlmostEqual(design_effect(1, 0.5), 1.0, places=12)   # m = 1: no clustering
        self.assertAlmostEqual(design_effect(10, 0.0), 1.0, places=12)  # rho = 0

    def test_mde_basic(self):
        from gatedtrain.gates import mde

        # (1.959964 + 0.841621) * 0.30 * sqrt(1 / 500) = 2.801585 * 0.30 * 0.044721 = 0.0375872
        ref = (Z_975 + Z_80) * 0.30 * math.sqrt(1.0 / 500)
        self.assertAlmostEqual(ref, 0.037587209954755035, places=12)
        self.assertAlmostEqual(mde(alpha=0.025, power=0.8, sigma=0.30, n=500), ref, places=10)

    def test_mde_with_design_effect(self):
        from gatedtrain.gates import mde

        # 2.801585 * 0.25 * sqrt(1.3 / 200) = 0.0564678
        ref = (Z_975 + Z_80) * 0.25 * math.sqrt(1.3 / 200)
        self.assertAlmostEqual(ref, 0.056467755330624904, places=12)
        self.assertAlmostEqual(mde(alpha=0.025, power=0.8, sigma=0.25, n=200, design_effect=1.3), ref, places=10)

    def test_mde_power_90(self):
        from gatedtrain.gates import mde

        # (1.959964 + 1.281552) * 0.2 * sqrt(1 / 100)
        ref = (Z_975 + Z_90) * 0.2 * 0.1
        self.assertAlmostEqual(mde(alpha=0.025, power=0.9, sigma=0.2, n=100), ref, places=10)


class TestPlanPerGateType(unittest.TestCase):
    """`plan(clause, pilot, n_available=...)` (05 section 3.8) for each gate type.

    Required n = ceil(DE * ((z_{1-a} + z_{1-b}) * sigma_used / d)^2), the
    05:124 formula solved for n, where d is the planned distance from the
    null: superiority d = planning_effect - margin (H0: Delta <= delta_sup),
    non-inferiority d = planning_effect + margin (H0: Delta <= -delta),
    05 section 3.1 table (05:58-62).
    """

    def test_superiority_demo_registration(self):
        from gatedtrain.gates import plan

        # sigma_U = 0.30 * 1.5575754 = 0.4672726; n = (2.8015852 * 0.4672726 / 0.10)^2 = 171.375 -> 172
        sigma_u = 0.30 * math.sqrt(4 / chi2.ppf(0.20, 4))
        n_ref = math.ceil(((Z_975 + Z_80) * sigma_u / 0.10) ** 2)
        self.assertEqual(n_ref, 172)
        p = plan(sup_clause(), {"s": 0.30, "df": 4}, n_available=500)
        self.assertEqual(p["n_required"], 172)
        self.assertAlmostEqual(p["sigma_used"], sigma_u, places=9)
        self.assertEqual(p["sigma_plan"], "ucb80")
        self.assertAlmostEqual(p["design_effect"], 1.0, places=12)
        self.assertEqual(p["n_available"], 500)
        self.assertFalse(p["underpowered"])
        # MDE at the available n under the sigma plan: 2.8015852 * 0.4672726 / sqrt(500) = 0.0585441
        self.assertAlmostEqual(p["mde"], (Z_975 + Z_80) * sigma_u / math.sqrt(500), places=9)

    def test_superiority_too_few_prompts_is_underpowered(self):
        from gatedtrain.gates import plan

        p = plan(sup_clause(), {"s": 0.30, "df": 4}, n_available=30)
        self.assertEqual(p["n_required"], 172)
        self.assertTrue(p["underpowered"], "MVP-SPEC section 2.3: report underpowered up front when n_available < n")

    def test_superiority_nonzero_margin(self):
        from gatedtrain.gates import plan

        # d = 0.10 - 0.02 = 0.08; sigma fixed 0.25: n = (2.8015852 * 0.25 / 0.08)^2 = 76.64 -> 77
        p = plan(sup_clause(margin=0.02, sigma_plan="fixed"), {"sigma": 0.25}, n_available=100)
        self.assertEqual(p["n_required"], math.ceil(((Z_975 + Z_80) * 0.25 / 0.08) ** 2))
        self.assertEqual(p["n_required"], 77)

    def test_non_inferiority(self):
        from gatedtrain.gates import plan

        # d = 0.0 + 0.10; same sigma_U as the demo -> 172
        c = sup_clause(type="non_inferiority", margin=0.10, planning_effect=0.0)
        self.assertEqual(plan(c, {"s": 0.30, "df": 4}, n_available=500)["n_required"], 172)
        # d = 0.0 + 0.05 -> (2.8015852 * 0.4672726 / 0.05)^2 = 685.50 -> 686
        c = sup_clause(type="non_inferiority", margin=0.05, planning_effect=0.0)
        sigma_u = 0.30 * math.sqrt(4 / chi2.ppf(0.20, 4))
        n_ref = math.ceil(((Z_975 + Z_80) * sigma_u / 0.05) ** 2)
        self.assertEqual(n_ref, 686)
        self.assertEqual(plan(c, {"s": 0.30, "df": 4}, n_available=500)["n_required"], 686)

    def test_design_effect_inflates_n(self):
        from gatedtrain.gates import plan

        # 171.375 * 1.3 = 222.788 -> 223 (cluster size 4, rho 0.1)
        p = plan(sup_clause(), {"s": 0.30, "df": 4}, n_available=500, cluster_size=4, icc=0.1)
        self.assertAlmostEqual(p["design_effect"], 1.3, places=12)
        self.assertEqual(p["n_required"], 223)

    def test_fixed_sigma(self):
        from gatedtrain.gates import plan

        # sigma = 0.25 named in the registration: n = (2.8015852 * 0.25 / 0.10)^2 = 49.05 -> 50
        p = plan(sup_clause(sigma_plan="fixed"), {"sigma": 0.25}, n_available=60)
        self.assertEqual(p["n_required"], 50)
        self.assertAlmostEqual(p["sigma_used"], 0.25, places=12)
        self.assertFalse(p["underpowered"])

    def test_equivalence_mde_uses_clause_alpha(self):
        from gatedtrain.gates import plan

        # Equivalence default alpha is 0.05 per one-sided test (05 section 3.2, 05:88).
        # MDE by the 05:124 formula at n = 100, sigma = 0.2:
        # (1.6448536 + 0.8416212) * 0.2 * sqrt(1 / 100) = 0.0497295
        c = sup_clause(type="equivalence", margin=0.10, planning_effect=0.0, alpha=0.05, sigma_plan="fixed")
        p = plan(c, {"sigma": 0.2}, n_available=100)
        self.assertAlmostEqual(p["mde"], (Z_95 + Z_80) * 0.2 * 0.1, places=9)


if __name__ == "__main__":
    unittest.main()
