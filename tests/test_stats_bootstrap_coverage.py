"""Bootstrap interval coverage on synthetic data (MVP-SPEC section 5,
"Statistics unit tests"; 05 section 3.4, 05:105).

Model: each simulated gate draws n = 100 fresh prompts with baseline success
probability p_b ~ Uniform(0.2, 0.8) and candidate p_c = p_b + 0.05, and K = 4
Bernoulli samples per prompt and arm. The population effect is exactly
Delta = E[p_c - p_b] = 0.05 (no clipping: p_c <= 0.85).

With alpha = 0.05 the one-sided lower (1 - alpha) bound should cover Delta in
about 95% of simulations and the two-sided (1 - 2 alpha) interval in about
90%. With 400 simulations the Monte Carlo SD of a coverage estimate is
sqrt(0.95 * 0.05 / 400) = 0.0109 and sqrt(0.90 * 0.10 / 400) = 0.0150; the
tolerances below are about 3.5 and 3.3 Monte Carlo SDs.

The data generator uses its own seeded numpy Generator; evaluate() gets a
distinct bootstrap seed per simulation.
"""

from __future__ import annotations

import unittest

import numpy as np

N_SIMS = 400
N_PROMPTS = 100
K = 4
TRUE_DELTA = 0.05


def simulate(rng: np.random.Generator):
    p_b = rng.uniform(0.2, 0.8, size=N_PROMPTS)
    p_c = p_b + TRUE_DELTA
    base = (rng.random((N_PROMPTS, K)) < p_b[:, None]).astype(float)
    cand = (rng.random((N_PROMPTS, K)) < p_c[:, None]).astype(float)
    b = {(i, r): {"problem_idx": i, "repeat": r, "reward": base[i, r]} for i in range(N_PROMPTS) for r in range(K)}
    c = {(i, r): {"problem_idx": i, "repeat": r, "reward": cand[i, r]} for i in range(N_PROMPTS) for r in range(K)}
    return b, c


class TestBootstrapCoverage(unittest.TestCase):
    def test_coverage_superiority_alpha_005(self):
        from gatedtrain.gates import evaluate

        clause = {
            "type": "superiority", "metric": "exact_match", "margin": 0.0, "alpha": 0.05, "power": 0.8,
            "planning_effect": 0.10, "sigma_plan": "fixed", "unit": "prompt", "samples_per_prompt": K,
            "eval_seeds": [11, 12, 13, 14], "interval": "cluster_bootstrap", "blocking": True,
        }
        rng = np.random.default_rng(20261109)
        lower_cover = 0
        two_sided_cover = 0
        for sim in range(N_SIMS):
            b, c = simulate(rng)
            res = evaluate(clause, b, c, seed=1000 + sim, n_boot=2000)
            self.assertAlmostEqual(res["ci_level"], 0.90, places=12)
            lower_cover += res["ci_lower"] <= TRUE_DELTA
            two_sided_cover += res["ci_lower"] <= TRUE_DELTA <= res["ci_upper"]
        lower_rate = lower_cover / N_SIMS
        two_rate = two_sided_cover / N_SIMS
        self.assertLessEqual(abs(lower_rate - 0.95), 0.038, f"one-sided coverage {lower_rate:.3f}")
        self.assertLessEqual(abs(two_rate - 0.90), 0.05, f"two-sided coverage {two_rate:.3f}")


if __name__ == "__main__":
    unittest.main()
