"""External-run adapters, slice 10: area "agentic RL" (tests/README.md section 11.27, EXT-248 to EXT-262), the part that
stays in gatedtrain. The adapter `nemo_rl_agentic` and its tests live in the recipe repository
`gatedtrain-recipe-nemo-rl-agentic-math`; kept here is the documentation table (MM-37).
"""

from __future__ import annotations

import unittest
from pathlib import Path


class TestDocsRows(unittest.TestCase):

    def test_docs_rows(self):
        """MM-37: one training row names `nemo_rl_agentic` with its id; row 10 names the evaluator."""
        text = (Path(__file__).resolve().parents[1] / "MULTIMODAL-SPEC.md").read_text(encoding="utf-8")
        rows = [l for l in text.splitlines() if l.startswith("|") and "`nemo_rl_agentic`" in l]
        self.assertEqual(len(rows), 1, rows)
        self.assertIn("`rl.nemo_rl_agentic@0.1`", rows[0])
        self.assertIn("**Row 10 deviations", text)


if __name__ == "__main__":
    unittest.main()
