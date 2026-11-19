"""External-run adapters, slice 6: area "CAD" (tests/README.md section 11.23, EXT-180 to EXT-199), the parts that stay
in gatedtrain. The adapter `nemo_rl_ext` and its tests live in the recipe repository `gatedtrain-recipe-nemo-rl-cad`; kept here are
the documentation table (MM-37), the rules other adapters keep (EXT-191: per-script options and the sidecar row are a
recipe's own rows); the other adapters' cases of review 11 L1 live in their recipe repositories.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import _support  # noqa: F401  (the test host layout and the synthetic test adapter)


def registry():
    from gatedtrain.ext import registry as reg
    return reg


class TestDocsRows(unittest.TestCase):

    def test_docs_rows(self):
        """MM-37: one training row names `nemo_rl_ext` with both ids; rows 6 and 6s name the two evaluators."""
        text = (Path(__file__).resolve().parents[1] / "MULTIMODAL-SPEC.md").read_text(encoding="utf-8")
        rows = [l for l in text.splitlines() if l.startswith("|") and "`nemo_rl_ext`" in l]
        self.assertEqual(len(rows), 1, rows)
        self.assertIn("`sft.nemo_rl@0.1`", rows[0])
        self.assertIn("`rl.nemo_rl_env@0.1`", rows[0])
        self.assertNotIn("MM-GAP-12 until a decision", text)
        self.assertIn("`cad.cadquery_valid_items@0.1`", text)


class TestOtherAdapters(unittest.TestCase):

    def test_other_adapters_unchanged(self):
        """EXT-191: the per-script options are the CAD adapter's rows only."""
        self.assertEqual(registry().run_options("synthetic", "train.py"), ((), None))
        self.assertEqual(registry().image_interpreters("synthetic"), {})
        self.assertIsNone(registry().sidecar("synthetic"))

    def test_other_adapters_refuse_a_sidecar(self):
        from gatedtrain.ext.launch import HostConfig, sidecar_problems
        cfg = HostConfig(sidecar=None)
        msgs = sidecar_problems({"sidecar": {"name": "x"}}, kind="request", cfg=cfg)
        self.assertTrue(any("sidecar" in f for f, _ in msgs), msgs)




if __name__ == "__main__":
    unittest.main()
