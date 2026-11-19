"""Tier B smoke check for interim rules G-15 and G-19 (docs/tier-b-runbook.md step 5).

prototype, unpromotable. An operator runs this inside the NeMo RL 4f39fd8 container
(`uv run python /work/tierb/reward_check.py`); gatedtrain never runs it.

It builds the training reward exactly as NeMo RL's math environment does with
`math_verify_impl: hf_math_verify` (nemo_rl/environments/math_environment.py:93-99
builds `math_metric`; :133-137 wraps the ground truth as `\\boxed{gt}`), and
scores answers in NeMo Evaluator's GSM8K format, `The answer is N`
(benchmarks/gsm8k.py:22-26, :42-44 at ea35ba3). The ground truth keeps the
thousands separator, as the export does (G-19).

Exit 0 when every case scores as expected; exit 1 otherwise (stop and report).
"""

import sys

from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

verify = math_metric(
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
)

# (ground truth as exported, model response, expected reward)
CASES = [
    ("18", "Janet sells 16 - 3 - 4 = 9 eggs at $2 each. The answer is 18.", 1.0),       # G-15
    ("18", "The answer is 18", 1.0),                                                     # G-15, no period
    ("3,082", "The answer is 3082.", 1.0),                                               # G-19, on its own
    ("3,082", "The total is 3,082. The answer is 3082.", 1.0),                           # G-19, comma earlier
    ("3,082", "The answer is 3,082.", 1.0),                                              # G-19, comma kept
    ("18", "The answer is 17.", 0.0),                                                    # a wrong answer scores 0
]

failed = 0
for gt, response, want in CASES:
    score, _ = verify(["\\boxed{" + gt + "}"], [response])
    ok = float(score) == want
    failed += not ok
    print(f"{'ok  ' if ok else 'FAIL'} gt={gt!r} response={response!r} reward={float(score)} expected={want}")
print("G-15/G-19 reward check:", "PASS" if not failed else f"FAIL ({failed} case(s)); stop and report")
sys.exit(1 if failed else 0)
