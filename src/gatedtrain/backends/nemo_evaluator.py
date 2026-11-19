"""Tier B evaluator backend `nemo_evaluator` (MVP-SPEC section 4).

An operator runs NeMo Evaluator at ea35ba3 (`nel eval run`, cli/eval.py:71-88)
on the baseline and on the trained checkpoint with the registered repeats
(K) and sampler. NeMo Evaluator writes one per-sample record per line to
`<output dir>/results.jsonl` (engine/artifacts.py:96-102), with the keys of
engine/eval_loop.py:790-803. This module only reads those files; it pairs
records by (problem_idx, repeat) exactly as NeMo Evaluator's
`load_paired_records` does (engine/comparison.py:360).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..gates import load_paired_records

NEMO_EVALUATOR_COMMIT = "ea35ba3"
RESULT_KEYS = ("problem_idx", "repeat", "reward", "scorer_rewards", "prompt", "model_response", "extracted_answer",
               "expected_answer", "scoring_details", "metadata", "tokens", "latency_ms")


def read_results(task_dir: Path) -> dict[tuple[int, int], dict[str, Any]]:
    """{(problem_idx, repeat): record} from a NeMo Evaluator task directory."""
    return load_paired_records(Path(task_dir))


def build_eval_command(*, model_url: str, model_id: str, bench: str, repeats: int, temperature: float,
                       max_tokens: int, output_dir: str) -> list[str]:
    """argv for one NeMo Evaluator run (cli/eval.py:72-88). Never executed by gatedtrain."""
    return ["nel", "eval", "run", "--bench", bench, "--model-url", model_url, "--model-id", model_id,
            "--repeats", str(int(repeats)), "--temperature", str(float(temperature)),
            "--max-tokens", str(int(max_tokens)), "--output-dir", output_dir]
