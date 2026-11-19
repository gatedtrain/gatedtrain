"""Tier B test helpers: reference export, recorded-output builders.

Everything here is written from the pinned sources, not from gatedtrain:
- GSM8K answer extraction: nemo_rl/data/datasets/response_datasets/gsm8k.py:22-25 (4f39fd8);
- NeMo Evaluator GSM8K prompt: src/nemo_evaluator/benchmarks/gsm8k.py:22-26 (ea35ba3);
- NeMo RL outputs: logs/exp_<NNN>/train_data_step<N>.jsonl (examples/run_grpo.py:106,
  utils/logger.py:1993-2025, :1409-1438; grpo_sync.py:1429-1460) and checkpoints/step_<N>/ with
  training_info.json and config.yaml (utils/checkpoint.py:404-419) and train_dataloader.pt
  (data/utils.py:69-81), consolidated HF weights under policy/weights/model/consolidated
  (Automodel checkpointing.py:415-421) and policy/tokenizer (grpo_sync.py:1383-1400);
- NeMo Evaluator outputs: <run>/gsm8k/results.jsonl (orchestration/orchestrator.py:1046-1047, :1081;
  engine/artifacts.py:96-102; record keys engine/eval_loop.py:790-803) and the composed-config
  snapshot <run>/full_config.yaml (config/snapshot.py:39, :72-81; executors/local_executor.py:69).
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from _support import FIXTURES, NEMO_EVAL_COMPARISON, file_sha256, manifest_hash, run_cli

TIERB = FIXTURES / "tierb"
SOURCE = TIERB / "gsm8k-source"
REVISION = "0123456789abcdef0123456789abcdef01234567"   # fixture "HF dataset revision"
ROWS_PER_SHARD = 10
NEMO_EVAL_GSM8K = NEMO_EVAL_COMPARISON.parents[1] / "benchmarks" / "gsm8k.py"

# NeMo Evaluator's GSM8K prompt (benchmarks/gsm8k.py:22-26 at ea35ba3) with `{question}`
# replaced by `{}` for NeMo RL's `prompt.format(problem)` (nemo_rl/data/processors.py:574-576).
NEL_PROMPT_FOR_NEMO_RL = (
    "Solve the following math problem step by step. "
    "Put your final numerical answer after 'The answer is'.\n\n"
    "Problem: {}"
)
NEMO_RESULT_KEYS = (
    "problem_idx", "repeat", "reward", "scorer_rewards", "prompt", "model_response", "extracted_answer",
    "expected_answer", "scoring_details", "metadata", "tokens", "latency_ms",
)


def upstream_prompt() -> str | None:
    """`_PROMPT` from the pinned NeMo Evaluator clone, or None if the clone is absent."""
    if not NEMO_EVAL_GSM8K.is_file():
        return None
    tree = ast.parse(NEMO_EVAL_GSM8K.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_PROMPT" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("_PROMPT not found in the pinned benchmarks/gsm8k.py")


# ---------------------------------------------------------------------------
# Reference export (H-1), independent of gatedtrain
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str:
    """nemo_rl/data/datasets/response_datasets/gsm8k.py:22-25: the text after `####`, stripped."""
    assert "####" in text, text
    return text.split("####")[1].strip()


def export_line(question: str, answer: str) -> bytes:
    row = {"answer": answer, "question": question}
    return (json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def read_source(split: str) -> list[dict[str, str]]:
    return [json.loads(l) for l in (SOURCE / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def expected_export(rows_per_shard: int = ROWS_PER_SHARD) -> dict[str, Any]:
    train = [export_line(r["question"], extract_answer(r["answer"])) for r in read_source("train")]
    test = [export_line(r["question"], extract_answer(r["answer"])) for r in read_source("test")]
    shards = {}
    for i in range(0, len(train), rows_per_shard):
        shards[f"shard-{i // rows_per_shard:05d}.jsonl"] = b"".join(train[i:i + rows_per_shard])
    return {"shards": shards, "concat": b"".join(shards[k] for k in sorted(shards)),
            "heldout": b"".join(test), "n_train": len(train), "n_test": len(test)}


def run_export(out: Path, *, cwd: Path, rows_per_shard: int = ROWS_PER_SHARD, revision: str | None = REVISION):
    args = ["export-gsm8k", "--out", str(out), "--source", str(SOURCE), "--rows-per-shard", str(rows_per_shard)]
    if revision is not None:
        args += ["--revision", revision]
    return run_cli(args, cwd=cwd, timeout=120)


# ---------------------------------------------------------------------------
# Recorded NeMo RL outputs
# ---------------------------------------------------------------------------

def nemo_rl_data_config(*, shuffle: Any = False, data_path: str = "/work/tierb/export/train.jsonl",
                        split_validation_size: Any = 0.0, extra_datasets: int = 0) -> dict[str, Any]:
    """The `data` section NeMo RL saves in step_<N>/config.yaml for the build_command overrides
    (golden common_overrides: data.shuffle=false, ++data.train.data_path={export}/train.jsonl,
    data.train.split_validation_size=0). setup_response_data turns a `data.train` mapping into a
    one-element list in place (nemo_rl/data/utils.py:215-216 at 4f39fd8) before the config is saved, so
    the saved shape is a list; the real long run's step_150/config.yaml has exactly this shape, with
    data_path /work/tierb/export/train.jsonl (container path of <host>/tierb/export/train.jsonl)."""
    entry = {"data_path": data_path, "dataset_name": "ResponseDataset", "env_name": "math",
             "input_key": "question", "output_key": "answer", "processor": "math_hf_data_processor",
             "prompt_file": "/work/tierb/export/nel_gsm8k_prompt.txt", "seed": 42,
             "split_validation_size": split_validation_size, "system_prompt_file": None}
    train = [entry] + [dict(entry, data_path=f"/work/tierb/extra{i}/train.jsonl") for i in range(extra_datasets)]
    return {"default": {"env_name": "math", "processor": "math_hf_data_processor",
                        "prompt_file": "/work/tierb/export/nel_gsm8k_prompt.txt", "system_prompt_file": None},
            "max_input_seq_length": 512, "num_workers": 1, "shuffle": shuffle, "train": train,
            "use_multiple_dataloader": False, "validation": None}


def build_stage_out(root: Path, *, prompts_per_step: int = 2, gens: int = 8, steps: int = 10,
                    save_period: int = 5, exp_dirs: int = 1, consumed_offset: int = 0,
                    model_name: str = "/work/tierb/base-model", data: dict | None = None) -> Path:
    """`checkpoint_dir=<root>/checkpoints`, `log_dir=<root>/logs` as build_command sets them.
    `data` replaces the saved `data` section (default: nemo_rl_data_config(), R10A-7)."""
    import torch

    logs = root / "logs"
    for e in range(1, exp_dirs + 1):
        (logs / f"exp_{e:03d}").mkdir(parents=True, exist_ok=True)
    exp = logs / "exp_001"
    for step in range(1, steps + 1):
        with (exp / f"train_data_step{step}.jsonl").open("w", encoding="utf-8") as f:
            for idx in range(prompts_per_step * gens):
                sample = {"rewards": float((step + idx) % 2), "input_lengths": 12,
                          "token_loss_mask": [0] * 6 + [1] * 6, "sample_loss_mask": 1.0,
                          "advantages": [0.0] * 12, "generation_logprobs": [0.0] * 12,
                          "prev_logprobs": [0.0] * 12, "token_ids": list(range(12)),
                          "content": ["The answer is 42."], "idx": idx}
                f.write(json.dumps(sample) + "\n")
    for step in range(save_period, steps + 1, save_period):
        d = root / "checkpoints" / f"step_{step}"
        (d / "policy" / "weights" / "model" / "consolidated").mkdir(parents=True)
        (d / "policy" / "tokenizer").mkdir(parents=True)
        info = {"consumed_samples": prompts_per_step * step + consumed_offset, "current_step": step,
                "total_steps": step, "current_epoch": 0, "total_valid_tokens": 0}
        (d / "training_info.json").write_text(json.dumps(info), encoding="utf-8")
        cfg = {"grpo": {"num_prompts_per_step": prompts_per_step, "num_generations_per_prompt": gens,
                        "max_num_steps": steps},
               "data": data if data is not None else nemo_rl_data_config(), "checkpointing": {"save_period": save_period, "keep_top_k": None},
               # F-17: build_command sets policy.model_name to the base-model dir inside the container.
               "policy": {"model_name": model_name}}
        (d / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        torch.save({"samples_yielded": prompts_per_step * step}, d / "train_dataloader.pt")
        (d / "policy" / "weights" / "model" / "consolidated" / "model.safetensors").write_bytes(
            hashlib.sha256(f"weights-{step}".encode()).digest())
        (d / "policy" / "weights" / "model" / "consolidated" / "config.json").write_text(
            '{"architectures": ["Qwen3ForCausalLM"]}', encoding="utf-8")
        (d / "policy" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Recorded NeMo Evaluator outputs
# ---------------------------------------------------------------------------

def baseline_reward(i: int) -> float:
    return 1.0 if i % 3 == 0 else 0.0


def candidate_reward(i: int) -> float:
    return 1.0 if i % 3 != 2 else 0.0


def nel_expected_answer(answer: str) -> str:
    """NeMo Evaluator's `_prepare` drops thousands commas (benchmarks/gsm8k.py:29-31 at ea35ba3);
    the export keeps NeMo RL's form (gsm8k.py:22-25 at 4f39fd8)."""
    return answer.replace(",", "")


def nel_record(i: int, rep: int, question: str, answer: str, reward: float, seed: int | None, *,
               scorer: str | None = None, scorer_reward: float | None = None,
               scorer_error: bool = False) -> dict[str, Any]:
    """One per-sample record (eval_loop.py:790-803). `prompt` is the GSM8K template filled with the
    question (benchmarks/gsm8k.py:22-26); `expected_answer` is `_prepare`'s comma-free answer.

    G-21: with `scorer` and `scorer_reward`, the record carries what an extra scorer writes at
    ea35ba3: `scoring_details["scorer:<name>"] = {correct, extracted, reward}` (engine/eval_loop.py:704-721)
    and `scorer_rewards = {"scorer:<name>": reward}` (eval_loop.py:785-794). The top-level `reward`
    stays the default scorer's. With `scorer_error` the record is what a raising scorer leaves
    (eval_loop.py:719-721): `{"error": ..., "reward": 0.0}` and `scorer_rewards` 0.0."""
    expected = nel_expected_answer(answer)
    resp = f"... The answer is {expected if reward else 'unknown'}"
    meta = {} if seed is None else {"eval_seed": seed}
    details: dict[str, Any] = {"method": "pattern"}
    scorer_rewards: dict[str, float] = {}
    if scorer is not None and scorer_error:
        key = f"scorer:{scorer}"
        details[key] = {"error": "ValueError: fixture scorer failure", "reward": 0.0}
        scorer_rewards[key] = 0.0
    elif scorer is not None and scorer_reward is not None:
        key = f"scorer:{scorer}"
        details[key] = {"correct": bool(scorer_reward), "extracted": expected if scorer_reward else "0",
                        "reward": float(scorer_reward)}
        scorer_rewards[key] = float(scorer_reward)
    return {"problem_idx": i, "repeat": rep, "reward": reward, "scorer_rewards": scorer_rewards,
            "prompt": NEL_PROMPT_FOR_NEMO_RL.replace("{}", question),
            "model_response": resp, "extracted_answer": expected if reward else "unknown", "expected_answer": expected,
            "scoring_details": details, "metadata": meta, "tokens": 20, "latency_ms": 12.5}


def heldout_rows(export_dir: Path) -> list[dict[str, str]]:
    return [json.loads(l) for l in (export_dir / "heldout" / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]


def snapshot_text(seed: int, repeats: int, model: str, out_dir: str, *, generation: dict | None = None,
                  enable_thinking: bool = False, scorers: list[str] | None = None) -> str:
    """config/snapshot.py:72-81: provenance comment header + composed YAML body.
    Default sampler = the fixture registration's (temperature 0.7, top_p 1.0, max_tokens 1024).
    `scorers` (G-21) adds `scoring: {metrics: [{type: scorer, name: <n>}, ...]}` to the gsm8k benchmark
    (BenchmarkConfig.scoring, config/benchmarks.py:63; ScorerMetric, config/scoring.py:24-30); the
    snapshot dumps the raw config as written (snapshot.py:72-81), so the block appears verbatim."""
    gen = {"temperature": 0.7, "top_p": 1.0, "max_tokens": 1024, "seed": seed, **(generation or {})}
    raw = {"services": {"model": {"type": "api", "url": "http://localhost:8000/v1/chat/completions",
                                  "protocol": "chat_completions", "model": model,
                                  "generation": gen,
                                  "proxy": {"extra_body": {"chat_template_kwargs": {"enable_thinking": enable_thinking}}}}},
           "benchmarks": [{"name": "gsm8k", "repeats": repeats, "solver": {"type": "simple", "service": "model"}}],
           "output": {"dir": out_dir, "timestamped": False}}
    if scorers:
        raw["benchmarks"][0]["scoring"] = {"metrics": [{"type": "scorer", "name": n} for n in scorers]}
    header = "# Re-run with:  nel eval run <this file>\n# nemo-evaluator version: 0.0.0-fixture\n"
    return header + yaml.dump(raw, default_flow_style=False, sort_keys=False)


def build_eval_run(run_dir: Path, rows: list[dict[str, str]], *, reward_fn, seed: int, model: str,
                   repeats: int = 1, drop: set[int] = frozenset(), empty: bool = False,
                   generation: dict | None = None, enable_thinking: bool = False,
                   scorer: str | None = None, scorer_fn=None, declared_scorers: list[str] | None = None,
                   scorer_missing: set[int] = frozenset(), scorer_error: set[int] = frozenset()) -> Path:
    """One single-seed NeMo Evaluator run: <run>/gsm8k/results.jsonl, <run>/full_config.yaml.

    G-21: with `scorer` and `scorer_fn`, each record carries `scorer_rewards["scorer:<scorer>"] =
    scorer_fn(i)` except for problems in `scorer_missing`. `declared_scorers` is the scorer list in the
    run's full_config.yaml; it defaults to `[scorer]` (a run made with the scorer) and `[]` declares none.
    Problems in `scorer_error` get the record a raising scorer leaves (eval_loop.py:719-721)."""
    task = run_dir / "gsm8k"
    task.mkdir(parents=True)
    lines = []
    if not empty:
        for rep in range(repeats):
            for i, r in enumerate(rows):
                if i in drop:
                    continue
                sr = scorer_fn(i) if scorer_fn is not None and i not in scorer_missing else None
                lines.append(json.dumps(nel_record(i, rep, r["question"], r["answer"], reward_fn(i), None,
                                                   scorer=scorer, scorer_reward=sr,
                                                   scorer_error=i in scorer_error)) + "\n")
    (task / "results.jsonl").write_text("".join(lines), encoding="utf-8")
    if declared_scorers is None:
        declared_scorers = [scorer] if scorer else []
    (run_dir / "full_config.yaml").write_text(
        snapshot_text(seed, repeats, model, str(run_dir), generation=generation, enable_thinking=enable_thinking,
                      scorers=declared_scorers),
        encoding="utf-8")
    bundle = {"run_id": f"eval-fixture-{model}-{seed}", "config": {"benchmark": "gsm8k", "repeats": repeats},
              "benchmark": {"name": "gsm8k", "samples": len(rows), "repeats": repeats}, "n_results": len(lines)}
    (task / f"eval-fixture-{model}-{seed}.json").write_text(json.dumps(bundle), encoding="utf-8")
    return run_dir


def build_merged(arm_dir: Path, rows: list[dict[str, str]], *, reward_fn, seeds: list[int],
                 drop: set[int] = frozenset(), empty: bool = False, answers: dict[int, str] | None = None,
                 questions: dict[int, str] | None = None, no_reward: set[int] = frozenset(),
                 scorer: str | None = None, scorer_fn=None) -> Path:
    """What merge-eval (H-3) produces: <arm>/results.jsonl, repeat r <-> seeds[r].
    answers / questions replace a problem's heldout answer or question (dataset revision drift);
    no_reward drops the `reward` key from that problem's records. `scorer` / `scorer_fn` add
    `scorer_rewards` as in build_eval_run (G-21); no merge.json and no run configs are written."""
    arm_dir.mkdir(parents=True)
    answers, questions = answers or {}, questions or {}
    lines = []
    if not empty:
        for i, r in enumerate(rows):
            if i in drop:
                continue
            for rep, seed in enumerate(seeds):
                rec = nel_record(i, rep, questions.get(i, r["question"]), answers.get(i, r["answer"]), reward_fn(i), seed,
                                 scorer=scorer, scorer_reward=scorer_fn(i) if scorer_fn is not None else None)
                if i in no_reward:
                    rec.pop("reward")
                lines.append(json.dumps(rec) + "\n")
    (arm_dir / "results.jsonl").write_text("".join(lines), encoding="utf-8")
    return arm_dir


# ---------------------------------------------------------------------------
# A whole Tier B scenario on disk
# ---------------------------------------------------------------------------

# build_command profiles (tests/README.md section 11.6): (steps, prompts per step, generations, save period)
PROFILES = {"smoke": {"steps": 10, "prompts_per_step": 2, "gens": 4, "save_period": 5},
            "long": {"steps": 150, "prompts_per_step": 32, "gens": 8, "save_period": 25}}


EVALUATOR_ID = "gsm8k.nemo_evaluator@0.1"
NEMO_EVALUATOR_SHORT = "ea35ba3"


def evaluator_identity_hash(scorer: Any = None) -> str:
    """The Tier B evaluator identity hash (decision D-4, tests/README.md section 11.14): SHA-256 of the
    canonical JSON {benchmark, evaluator, nemo_evaluator_commit[, scorer]}. Without a scorer it is
    sha256:032ad6d4..., with numeric_match sha256:2c196d9d... (the value in the real long-run report).
    R10-5 (decision R10A-2): the registration carries this resolved hash as evaluators[0].manifest_hash."""
    identity = {"benchmark": "gsm8k", "evaluator": EVALUATOR_ID, "nemo_evaluator_commit": NEMO_EVALUATOR_SHORT}
    if scorer is not None:
        identity["scorer"] = scorer
    return "sha256:" + hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                                 ensure_ascii=False).encode("utf-8")).hexdigest()


NO_SCORER_HASH_LINE = f'    manifest_hash: "{evaluator_identity_hash()}"\n'


def with_scorer(registration_text: str, scorer: Any) -> str:
    """G-21: the fixture registration with `evaluators[0].scorer` set to `scorer` (written as YAML,
    so None gives `scorer: null`). R10-5 (decision R10A-2): a string scorer is part of the evaluator
    identity, so `manifest_hash` becomes the identity hash with that scorer; None keeps the no-scorer hash."""
    anchor = NO_SCORER_HASH_LINE
    assert registration_text.count(anchor) == 1, "fixture registration layout changed"
    value = yaml.safe_dump(scorer, default_flow_style=True).split("\n")[0]   # drop the "..." end marker
    hash_line = anchor if not isinstance(scorer, str) else f'    manifest_hash: "{evaluator_identity_hash(scorer)}"\n'
    return registration_text.replace(anchor, hash_line + f"    scorer: {value}\n")


def render_inputs(root: Path, export_dir: Path, *, scorer: Any = None, set_scorer: bool = False) -> dict[str, Path]:
    """base-model/, trainer.yaml, registration.yaml and spec.yaml next to <root>/export.
    With `set_scorer` the registration carries `evaluators[0].scorer: <scorer>` (G-21)."""
    base = root / "base-model"
    base.mkdir(parents=True, exist_ok=True)
    (base / "config.json").write_text('{"architectures": ["Qwen3ForCausalLM"]}', encoding="utf-8")
    (base / "model.safetensors").write_bytes(hashlib.sha256(b"base-weights").digest())
    (root / "trainer.yaml").write_bytes((TIERB / "trainer.yaml").read_bytes())
    reg = (TIERB / "registration.template.yaml").read_text(encoding="utf-8")
    reg = reg.replace("{prompt_set_hash}", manifest_hash(export_dir / "heldout"))
    if set_scorer:
        reg = with_scorer(reg, scorer)
    (root / "registration.yaml").write_text(reg, encoding="utf-8")
    spec = (TIERB / "spec.template.yaml").read_text(encoding="utf-8")
    spec = spec.replace("{registration_sha256}", file_sha256(root / "registration.yaml")[len("sha256:"):])
    (root / "spec.yaml").write_text(spec, encoding="utf-8")
    return {"base": base, "registration": root / "registration.yaml", "spec": root / "spec.yaml"}


def registration_seeds(path: Path) -> list[int]:
    return list(yaml.safe_load(path.read_text(encoding="utf-8"))["clause"]["eval_seeds"])


def strip_comments(text: str) -> str:
    return re.sub(r"(?m)^\s*#.*\n", "", text)
