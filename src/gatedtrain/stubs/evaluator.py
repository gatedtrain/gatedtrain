"""skeleton-stub evaluator backend `cpu_stub` (MVP-SPEC section 3 "Evaluator backend").

Samples K completions per prompt with temperature sampling and writes
`results.jsonl` with exactly NeMo Evaluator's per-sample keys
(src/nemo_evaluator/engine/eval_loop.py:790-803 at ea35ba3, written one JSON
object per line by artifacts.py:98-102), keyed by `problem_idx` and `repeat`.

Common random numbers (05-evaluation-and-lineage.md:97): repeat r of every
prompt uses evaluation seed `eval_seeds[r]` in both arms. The seed fixes one
uniform draw per (prompt, token position), and a token is sampled by
inverting the model's CDF at that draw, so two identical models give
identical samples and two similar models give coupled samples.
`latency_ms` is fixed at 0.0 so results do not differ between runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from . import model as model_mod
from . import task, write_marker

STUB = "skeleton-stub"
EVALUATOR_ID = "addition_em.cpu_stub@0.1"


@torch.no_grad()
def sample_answers(model, prompts: list[str], *, seed: int, temperature: float, max_new_tokens: int) -> list[list[int]]:
    """One sampled completion per prompt under one evaluation seed."""
    if temperature <= 0:
        raise ValueError("the cpu_stub evaluator samples with temperature > 0")
    gen = torch.Generator().manual_seed(int(seed))
    uniforms = torch.rand((len(prompts), max_new_tokens), generator=gen, dtype=torch.float64)
    ids = torch.tensor([task.encode_prompt(p) for p in prompts], dtype=torch.long)
    out = []
    for t in range(max_new_tokens):
        if ids.shape[1] >= task.SEQ_LEN:
            break
        logits = model(ids)[:, -1, :].to(torch.float64) / temperature
        cdf = torch.softmax(logits, dim=-1).cumsum(dim=-1)
        u = uniforms[:, t : t + 1] * cdf[:, -1:]
        nxt = torch.searchsorted(cdf, u).clamp_(max=cdf.shape[1] - 1)
        ids = torch.cat([ids, nxt], dim=1)
        out.append(nxt.squeeze(1))
    gen_ids = torch.stack(out, dim=1).tolist() if out else [[] for _ in prompts]
    return gen_ids


def evaluate(checkpoint_dir: Path, prompt_dir: Path, out_dir: Path, *, eval_seeds: list[int],
             temperature: float, max_new_tokens: int) -> Path:
    """Evaluate one checkpoint on the prompt set; write `out_dir/results.jsonl`."""
    model_mod.configure_torch()
    model = model_mod.load(checkpoint_dir)
    model.eval()
    rows = task.load_prompts(prompt_dir)
    prompts = [r["prompt"] for r in rows]
    per_repeat = [sample_answers(model, prompts, seed=s, temperature=temperature, max_new_tokens=max_new_tokens)
                  for s in eval_seeds]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    path = out_dir / "results.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            for rep, seed in enumerate(eval_seeds):
                gen_ids = per_repeat[rep][idx]
                response = task.decode_answer(gen_ids)
                reward = task.exact_match(response, row["answer"])
                n_tokens = next((i + 1 for i, t in enumerate(gen_ids) if t == task.EOS), len(gen_ids))
                record = {
                    "problem_idx": idx,
                    "repeat": rep,
                    "reward": reward,
                    "scorer_rewards": {},
                    "prompt": row["prompt"],
                    "model_response": response,
                    "extracted_answer": response,
                    "expected_answer": row["answer"],
                    "scoring_details": {"method": "exact_match", "evaluator": EVALUATOR_ID},
                    "metadata": {"eval_seed": int(seed), "skeleton_stub": True},
                    "tokens": n_tokens,
                    "latency_ms": 0.0,
                }
                f.write(json.dumps(record) + "\n")
    write_marker(out_dir, f"cpu_stub evaluator {EVALUATOR_ID} results in NeMo Evaluator results.jsonl format")
    return path
