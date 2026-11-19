"""Registration sampler rules of an LLM evaluator (the synthetic test adapter's), unchanged by the SB3 recipe's
deterministic-policy sampler (EXT-106)."""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

import _support  # noqa: F401  (the test host layout)

# The SB3 recipe's registered deterministic-policy sampler (its eval_hover.py SAMPLER), which the LLM rules refuse.
SAMPLER = {"policy": "deterministic (Gaussian mean)", "vecnormalize": {"training": False, "norm_reward": False},
           "episodes_per_item": 1, "max_steps": 242}


class TestRegistrationSampler(unittest.TestCase):

    def test_llm_rules_unchanged(self):
        from gatedtrain.registration import check_contents
        reg = yaml.safe_load((Path(__file__).parent / "fixtures" / "ext" / "registration.template.yaml")
                             .read_text(encoding="utf-8").replace("{item_set_hash}", "sha256:" + "a" * 64)
                             .replace("{evaluator_hash}", "sha256:" + "b" * 64))
        reg["sampler"] = dict(SAMPLER)
        fields = [p.field for p in check_contents(reg, evaluator_id="synthetic.items@0.1")]
        self.assertIn("registration.sampler.temperature", fields)


if __name__ == "__main__":
    unittest.main()
