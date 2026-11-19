"""H-4 `gatedtrain nel-config` (TIER-B-SPEC section 1; interim rules G-15, G-16, G-17;
tests/README.md section 11.4): K NeMo Evaluator config files from the registration.

Golden files: tests/fixtures/tierb/golden/nel-gsm8k-rep{0,1}.yaml, compared as parsed YAML.
Every key is checked against the pinned NeMo Evaluator schema in test_tierb_citations.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from _support import EXIT_OK, EXIT_REFUSED, SKEL_ROOT, describe, parse_errors, run_cli
from _tierb import TIERB

GOLDEN = TIERB / "golden"


class TestNelConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-nelcfg-"))
        self.reg = self.tmp / "registration.yaml"
        text = (TIERB / "registration.template.yaml").read_text(encoding="utf-8")
        self.reg.write_text(text.replace("{prompt_set_hash}", "sha256:" + "0" * 64), encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def render(self, out: Path):
        return run_cli(["nel-config", "--registration", str(self.reg), "--model-url", "http://localhost:8000/v1",
                        "--model-id", "baseline", "--eval-root", "/work/out/eval/baseline", "--out", str(out)],
                       cwd=SKEL_ROOT, timeout=60)

    def test_golden(self):
        out = self.tmp / "cfg"
        proc = self.render(out)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["nel-gsm8k-rep0.yaml", "nel-gsm8k-rep1.yaml"])
        for r in (0, 1):
            with self.subTest(repeat=r):
                got = yaml.safe_load((out / f"nel-gsm8k-rep{r}.yaml").read_text(encoding="utf-8"))
                want = yaml.safe_load((GOLDEN / f"nel-gsm8k-rep{r}.yaml").read_text(encoding="utf-8"))
                self.assertEqual(got, want)

    def test_enable_thinking_must_be_false(self):
        """G-15: thinking is off on both sides; a registration that turns it on is refused."""
        self.reg.write_text(self.reg.read_text(encoding="utf-8").replace("enable_thinking: false",
                                                                          "enable_thinking: true"), encoding="utf-8")
        out = self.tmp / "cfg"
        proc = self.render(out)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue(any(e["field"] == "registration.sampler.enable_thinking" for e in parse_errors(proc.stderr)),
                        describe(proc))
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
