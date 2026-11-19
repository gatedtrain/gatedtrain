"""H-7 spec and registration support for Tier B, and H-8 templates under demo/tierb
(TIER-B-SPEC section 1; interim rules G-10, G-15 to G-17; tests/README.md sections 11.8-11.9).
"""

from __future__ import annotations

import re
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from _support import DEMO_DIR, EXIT_OK, EXIT_REFUSED, describe, file_sha256, parse_errors, run_cli
from _tierb import expected_export, render_inputs
from test_l2_skel_compat import grammar_from_design

TEMPLATES = DEMO_DIR / "tierb"


def reference_export(root: Path) -> Path:
    """The fixture export written by the independent reference (no gatedtrain needed)."""
    ref = expected_export()
    exp = root / "export"
    (exp / "train").mkdir(parents=True)
    for name, data in ref["shards"].items():
        (exp / "train" / name).write_bytes(data)
    (exp / "heldout").mkdir()
    (exp / "heldout" / "prompts.jsonl").write_bytes(ref["heldout"])
    return exp


class TestTierBSpecSupport(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-tierb-spec-"))
        self.paths = render_inputs(self.tmp, reference_export(self.tmp))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_validate_accepts_tier_b(self):
        """gsm8k.nemo_evaluator@0.1, rl.nemo_rl@0.1, budget.gpu_hours > 0, sampler top_p and
        enable_thinking, and pairing.max_unpaired are accepted."""
        proc = run_cli(["validate", "spec.yaml"], cwd=self.tmp, timeout=60)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_run_still_refuses_tier_b(self):
        """Tier A behaviour unchanged: `gatedtrain run` never launches a GPU stage."""
        runs = self.tmp / "runs"
        proc = run_cli(["run", "spec.yaml", "--runs-root", str(runs)], cwd=self.tmp, timeout=60)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue(any(e["field"] == "stages[0].uses" for e in parse_errors(proc.stderr)), describe(proc))
        self.assertFalse(runs.exists() and any(runs.iterdir()))

    def test_enable_thinking_true_refused(self):
        """G-15: Tier B registrations keep thinking off."""
        reg = self.paths["registration"]
        reg.write_text(reg.read_text(encoding="utf-8").replace("enable_thinking: false", "enable_thinking: true"),
                       encoding="utf-8")
        spec = self.paths["spec"]
        spec.write_text(re.sub(r"@sha256:[0-9a-f]{64}", "@" + file_sha256(reg), spec.read_text(encoding="utf-8")),
                        encoding="utf-8")
        proc = run_cli(["validate", "spec.yaml"], cwd=self.tmp, timeout=60)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue(any(e["field"] == "registration.sampler.enable_thinking" for e in parse_errors(proc.stderr)),
                        describe(proc))


class TestTierBTemplates(unittest.TestCase):
    """H-8: demo/tierb/spec.yaml and demo/tierb/registration.yaml."""

    def load(self, name):
        path = TEMPLATES / name
        self.assertTrue(path.is_file(), path)
        return path, path.read_text(encoding="utf-8"), yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_validates(self):
        proc = run_cli(["validate", "spec.yaml"], cwd=TEMPLATES, timeout=60)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_contents(self):
        _, _, spec = self.load("spec.yaml")
        reg_path, reg_text, reg = self.load("registration.yaml")
        self.assertEqual(spec["budget"]["gpu_hours"], 8, "GPU-hour budget 8 (interim rule G-10)")
        train, gate = spec["stages"]
        self.assertEqual(train["uses"], "rl.nemo_rl@0.1")
        self.assertEqual(gate["uses"], "gate.superiority@1.0")
        self.assertEqual(gate["evaluator"], "gsm8k.nemo_evaluator@0.1")
        reg_uri = spec["artifacts"][gate["inputs"]["registration"]]["uri"]
        self.assertTrue(reg_uri.endswith("@" + file_sha256(reg_path)), "the spec pins the template registration")
        clause = reg["clause"]
        self.assertEqual(clause["type"], "superiority")
        self.assertEqual(clause["samples_per_prompt"], 4)
        self.assertEqual(len(set(clause["eval_seeds"])), 4)
        self.assertIsNotNone(clause["margin"])
        self.assertIn(clause["sigma_plan"], {"fixed", "ucb80"})
        comments = "\n".join(l for l in reg_text.splitlines() if l.lstrip().startswith("#") or " #" in l)
        self.assertIn("internal_pilot", comments, "margin and planning sigma are internal_pilot placeholders")
        self.assertEqual(reg["evaluators"][0]["id"], "gsm8k.nemo_evaluator@0.1")
        self.assertIs(reg["sampler"]["enable_thinking"], False)
        self.assertIn("top_p", reg["sampler"])
        self.assertEqual(reg["signed_by"], "prototype-unsigned")

    def test_subset_of_04_grammar(self):
        g = grammar_from_design()
        _, _, spec = self.load("spec.yaml")
        self.assertLessEqual(set(spec), g["top"])
        for entry in spec["artifacts"].values():
            self.assertLessEqual(set(entry), g["artifact"])
        for stage in spec["stages"]:
            self.assertLessEqual(set(stage), g["stage"])
            self.assertLessEqual(set(stage.get("test", {}) or {}), g["test"])


if __name__ == "__main__":
    unittest.main()
