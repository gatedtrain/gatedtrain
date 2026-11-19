"""Smoke records and their verdict (MULTIMODAL-SPEC.md MM-38, MM-39; tests/README.md section 11.18 EXT-29, EXT-30).

A smoke run (a few training steps and a 20-item baseline evaluation) writes `smoke.json`. Its verdict is
recomputed here, never trusted: PASS only when no reason below holds.

- `gpu_memory`: peak GPU memory above 22.5 GB (of 24.5 GB);
- `budget`: projected GPU-hours plus the GPU log's running total above the GPU-hour budget ceiling
  (launch.GPU_CAP_H, 1000: high enough not to limit runs; hours are still logged);
- `evaluator_errors`: any evaluator error;
- `degenerate_mean`: a non-binary metric with both bounds finite whose baseline mean lies outside
  [lo + 5 % w, hi - 5 % w] (the boundary is allowed); a binary metric (registry `binary: true`) whose mean
  is 0 or 1 (EXT-G3);
- `degenerate_mode` (non-binary only): at least 50 % of the baseline scores share one value; design revision 18
  (R18-8): a score exactly equal to the finite bound at the metric's best end (the lower bound of a lower-is-better
  metric, a WER of 0; the upper bound of a higher-is-better one) does not count, so the rule reads the other scores.
  A score on the worst end (an OCR accuracy of 0) still counts. The share stays a share of the whole sample, so a
  verdict changes only where the best bound itself made up half. Without a direction nothing is exempt;
- `degenerate_bound` (non-binary only, R18-8): fewer than a quarter of the scores (at least 2) lie off that best bound,
  so the evaluator is pinned at its best value and the scores off it are too few to show it varies;
- `degenerate_sd` (non-binary only): the baseline scores have standard deviation 0.

A record with no score on the best bound gets the verdict it had before revision 18.

Design revision 18 (R18-8): a registration with several clauses checks the smoke baseline of every clause's evaluator:
the record then carries `baseline_scores_by_evaluator` ({evaluator id: scores}, one entry per clause evaluator, the
first clause's equal to `baseline_scores`), each recomputed with that evaluator's row (`smoke_problems(evaluators=)`).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from ..renamed import old_schema_note

SMOKE_SCHEMA = "gatedtrain.smoke_record"
SMOKE_KEYS = ("schema", "adapter", "area", "base_model_hash", "item_set_hash", "peak_gpu_mem_gb", "host_ram_peak_gb",
              "swap_peak_gb", "seconds_per_step", "projected_gpu_h", "gpu_log_total_h", "baseline_scores",
              "error_count", "verdict", "reasons")
GPU_MEM_LIMIT_GB = 22.5
GPU_CAP_H = 1000            # the budget ceiling passed to gpu-step.sh (launch.GPU_CAP_H)
MEAN_MARGIN = 0.05
MODE_SHARE = 0.5
OFF_BOUND_SHARE = 0.25      # R18-8: the least share of baseline scores off the declared bounds
OFF_BOUND_MIN = 2
_EPS = 1e-9


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def smoke_verdict(record: dict, *, score_range, binary: bool, direction: str | None = None) -> dict[str, Any]:
    """{verdict: PASS | FAIL, reasons} recomputed from the record's measurements (EXT-29). Raises ValueError
    when a measurement is missing or not a finite number."""
    for key in ("peak_gpu_mem_gb", "projected_gpu_h", "gpu_log_total_h"):
        if not _num(record.get(key)):
            raise ValueError(f"`{key}` is {record.get(key)!r}, not a finite number")
    errors = record.get("error_count")
    if not (isinstance(errors, int) and not isinstance(errors, bool) and errors >= 0):
        raise ValueError(f"`error_count` is {errors!r}, not a count")
    scores = record.get("baseline_scores")
    if not (isinstance(scores, list) and all(_num(s) for s in scores)):
        raise ValueError("`baseline_scores` is not a list of finite numbers")
    reasons: list[str] = []
    if record["peak_gpu_mem_gb"] > GPU_MEM_LIMIT_GB:
        reasons.append("gpu_memory")
    if record["projected_gpu_h"] + record["gpu_log_total_h"] > GPU_CAP_H + _EPS:
        reasons.append("budget")
    if errors > 0:
        reasons.append("evaluator_errors")
    if not scores:
        reasons.append("degenerate_mean")
    else:
        mean = sum(scores) / len(scores)
        lo, hi = (list(score_range) + [None, None])[:2]
        if binary:
            if mean == 0 or mean == 1:
                reasons.append("degenerate_mean")
        else:
            if lo is not None and hi is not None:
                w = hi - lo
                if mean < lo + MEAN_MARGIN * w - _EPS or mean > hi - MEAN_MARGIN * w + _EPS:
                    reasons.append("degenerate_mean")
            # R18-8: scores on the finite bound at the metric's best end are left out of the mode rule; enough must
            # remain. The worst end counts as before.
            best = lo if direction == "lower_is_better" else hi if direction == "higher_is_better" else None
            off = [x for x in scores if best is None or x != best]
            if len(off) < len(scores):
                if len(off) < max(OFF_BOUND_MIN, math.ceil(OFF_BOUND_SHARE * len(scores))):
                    reasons.append("degenerate_bound")
                elif Counter(off).most_common(1)[0][1] >= MODE_SHARE * len(scores):   # share of the whole sample
                    reasons.append("degenerate_mode")
            elif Counter(scores).most_common(1)[0][1] >= MODE_SHARE * len(scores):
                reasons.append("degenerate_mode")
            sd = math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores))
            if len(scores) < 2 or sd == 0:
                reasons.append("degenerate_sd")
    return {"verdict": "FAIL" if reasons else "PASS", "reasons": reasons}


def smoke_problems(record: Any, *, adapter: str, score_range, binary: bool,
                   base_model_hash: str | None = None, item_set_hash: str | None = None,
                   evaluators: list[tuple[str, dict]] | None = None, direction: str | None = None) -> list[str]:
    """Why a smoke record does not allow a full run or a collect (MM-39, EXT-30): empty when it does.

    `evaluators` (R18-8): with several clauses, (evaluator id, registry row) per clause in clause order; each clause
    evaluator's baseline scores (`baseline_scores_by_evaluator`) are recomputed with its row, and `score_range` and
    `binary` are ignored."""
    if not isinstance(record, dict):
        return ["the smoke record is not a JSON object"]
    out = []
    missing = [k for k in SMOKE_KEYS if k not in record]
    if missing:
        out.append(f"missing keys {missing}")
    if record.get("schema") != SMOKE_SCHEMA:
        out.append(f"schema is {record.get('schema')!r}, not {SMOKE_SCHEMA}{old_schema_note(record.get('schema'))}")
    if record.get("adapter") != adapter:
        out.append(f"the record is for adapter {record.get('adapter')!r}, not {adapter!r}")
    if record.get("verdict") != "PASS":
        out.append(f"the record claims verdict {record.get('verdict')!r}, not PASS")
    if evaluators:
        out += _per_evaluator_problems(record, evaluators)
    else:
        try:
            v = smoke_verdict(record, score_range=score_range, binary=binary, direction=direction)
            if v["verdict"] != "PASS":
                out.append(f"the recomputed verdict is FAIL: {', '.join(v['reasons'])}")
        except ValueError as exc:
            out.append(f"the verdict cannot be recomputed: {exc}")
    if base_model_hash is not None and record.get("base_model_hash") != base_model_hash:
        out.append(f"base_model_hash {record.get('base_model_hash')!r} is not the base's model hash {base_model_hash}")
    if item_set_hash is not None and record.get("item_set_hash") != item_set_hash:
        out.append(f"item_set_hash {record.get('item_set_hash')!r} is not the registered {item_set_hash}")
    return out


def _per_evaluator_problems(record: dict, evaluators: list[tuple[str, dict]]) -> list[str]:
    """R18-8: every clause evaluator's smoke baseline, each with its own row."""
    by = record.get("baseline_scores_by_evaluator")
    if not isinstance(by, dict):
        return ["the gate has several clauses, but the record has no `baseline_scores_by_evaluator` "
                f"({{evaluator id: scores}} for {', '.join(ev for ev, _ in evaluators)})"]
    out = []
    first = evaluators[0][0]
    if first in by and by[first] != record.get("baseline_scores"):
        out.append(f"baseline_scores is not baseline_scores_by_evaluator[{first!r}] (the first clause's evaluator)")
    for ev, row in evaluators:
        if ev not in by:
            out.append(f"baseline_scores_by_evaluator has no scores of evaluator {ev}")
            continue
        try:
            v = smoke_verdict({**record, "baseline_scores": by[ev]}, score_range=row["score_range"],
                              binary=row["binary"], direction=row.get("direction"))
            if v["verdict"] != "PASS":
                out.append(f"the recomputed verdict of evaluator {ev} is FAIL: {', '.join(v['reasons'])}")
        except ValueError as exc:
            out.append(f"the verdict of evaluator {ev} cannot be recomputed: {exc}")
    for ev in sorted(set(by) - {e for e, _ in evaluators}, key=str):
        out.append(f"baseline_scores_by_evaluator names {ev!r}, which decides no clause")
    return out
