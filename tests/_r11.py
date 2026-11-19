"""Fixtures and helpers for registered item sets and per-item evaluation.

Import gatedtrain inside helpers so a missing API fails the affected tests
individually.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any

import yaml

from _support import FIXTURES, file_sha256

R11_FIXTURES = FIXTURES / "r11"
# Copy of the real long run's metadata (run-20261111T110313Z-7adda9b9, read-only source under
# <HOST_WORK>/runs/ on a single-GPU host; host paths in the copy are replaced by /srv/ptc/..., the copy keeps its bytes): registration, report, lineage, ledger, manifests, lockfile, plan.
# The two arms are compact per-item records (item = problem_idx, seed = metadata.eval_seed,
# score = scorer_rewards["scorer:numeric_match"], output_ref null), and raw_samples.json holds the first two
# NeMo Evaluator records of each arm verbatim. Extraction: fixtures/r11/extract_longrun.py.
LONGRUN = R11_FIXTURES / "longrun"
LONGRUN_ITEMS = 1319
RECORD_FIELDS = ("item", "repeat", "seed", "score", "error", "sub_metrics", "output_ref")


def clause(type_: str = "superiority", **over: Any) -> dict:
    """A revision 11 clause (05 section 3.1). Same defaults as test_stats_evaluate.clause."""
    c = {
        "type": type_, "metric": "score", "direction": "higher_is_better", "margin": 0.0, "alpha": 0.025,
        "power": 0.8, "planning_effect": 0.10, "sigma_plan": "fixed", "unit": "item", "samples_per_item": 1,
        "eval_seeds": [11], "interval": "cluster_bootstrap", "blocking": True,
    }
    if type_ == "non_inferiority":
        c.update(margin=0.10, planning_effect=0.0)
    if type_ == "equivalence":
        c.update(margin=0.10, planning_effect=0.0, alpha=0.05)
    c.update(over)
    return c


def rec(item: Any, repeat: int, score: Any, seed: Any, **over: Any) -> dict:
    """One per-item record (05 section 3.8)."""
    r = {"item": item, "repeat": repeat, "seed": seed, "score": score, "error": None, "sub_metrics": None,
         "output_ref": None}
    r.update(over)
    return r


def err(item: Any, repeat: int, seed: Any, cls: str = "timeout") -> dict:
    """An evaluator-error record: score null, error {class, message} (MULTIMODAL-SPEC MM-1, MM-C1)."""
    return rec(item, repeat, None, seed, error={"class": cls, "message": f"fixture {cls}"})


def arm(records, model: str | None = None) -> dict:
    """An arm as `gates.evaluate_items` reads it: the records plus the evaluated checkpoint's
    `model_manifest_hash`, recorded once per arm (05 section 3.8)."""
    return {"records": list(records), "model_manifest_hash": model}


def ids(n: int) -> list[str]:
    """Opaque string item IDs (MM-1: a key from the item set, never a row index)."""
    return [f"u{i:03d}" for i in range(n)]


def matrix_arms(base, cand, seeds, item_ids=None):
    """(items, baseline arm, candidate arm) from per-item lists of K scores; a None score skips the record."""
    item_ids = list(item_ids or ids(len(base)))

    def build(m):
        return [rec(item_ids[i], r, s, seeds[r]) for i, row in enumerate(m) for r, s in enumerate(row)
                if s is not None]

    return item_ids, arm(build(base)), arm(build(cand))


def alternating(n: int, center: float, amp: float) -> list[float]:
    """Per-item differences center +/- amp, alternating: mean = center, deterministic."""
    return [center + amp * (1 if i % 2 == 0 else -1) for i in range(n)]


def diff_arms(diffs, base: float = 0.2):
    """K = 1, seed 11: baseline `base` everywhere, candidate base + d_i. Scores stay in [0, 1] for the
    differences used here (|d| <= 0.8 - base)."""
    return matrix_arms([[base] for _ in diffs], [[base + d] for d in diffs], [11])


def evaluate_items(*args, **kw):
    from gatedtrain.gates import evaluate_items as impl

    return impl(*args, **kw)


def load_longrun_arm(name: str) -> list[dict]:
    with gzip.open(LONGRUN / f"{name}.items.jsonl.gz", "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def longrun_registration() -> dict:
    return yaml.safe_load((LONGRUN / "registration" / "registration.yaml").read_text(encoding="utf-8"))


def longrun_report() -> dict:
    return json.loads((LONGRUN / "report" / "gate_report.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Revision 10 names -> revision 11 names (fixture rewriting)
# ---------------------------------------------------------------------------

def to_item_names_registration(d: dict, *, direction: str = "higher_is_better") -> None:
    """Rewrite a revision 10 registration (prompt names) in place to the revision 11 names."""
    c = d["clause"]
    new_clause = {}
    for k, v in c.items():
        if k == "unit":
            new_clause["unit"] = "item"
        elif k == "samples_per_prompt":
            new_clause["samples_per_item"] = v
        else:
            new_clause[k] = v
        if k == "metric":
            new_clause["direction"] = direction
    d["clause"] = new_clause
    out = {}
    for k, v in d.items():
        if k == "prompt_set_hash":
            out["eval_distribution"] = {"item_set_hash": v, "env_version": None, "env_params_version": None}
        else:
            out[k] = v
    d.clear()
    d.update(out)


def to_item_names_spec(spec: dict) -> None:
    """Rewrite a revision 10 spec in place: gate input role `prompts` -> `items`, its artifact type
    `prompt_dataset` -> `item_set`, `cluster_by: prompt` -> `item` (04 section 3, section 4.2)."""
    gate = spec["stages"][1]
    inputs = gate["inputs"]
    name = inputs.pop("prompts")
    gate["inputs"] = {**{k: v for k, v in inputs.items() if k in ("model", "baseline")}, "items": name,
                      **{k: v for k, v in inputs.items() if k not in ("model", "baseline")}}
    spec["artifacts"][name]["type"] = "item_set"
    if gate.get("test", {}).get("cluster_by") == "prompt":
        gate["test"]["cluster_by"] = "item"


def repin(spec: Path, reg: Path) -> None:
    pat = re.escape(reg.name) + r"@sha256:[0-9a-f]{64}"
    text, n = re.subn(pat, f"{reg.name}@{file_sha256(reg)}", spec.read_text(encoding="utf-8"))
    assert n == 1, f"{spec} pins {reg.name} {n} times"
    spec.write_text(text, encoding="utf-8")


def rewrite(spec: Path, reg: Path, *, reg_mutate=None, spec_mutate=None) -> None:
    """Apply the mutations through YAML and re-pin the registration in the spec."""
    if reg_mutate is not None:
        d = yaml.safe_load(reg.read_text(encoding="utf-8"))
        reg_mutate(d)
        reg.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    if spec_mutate is not None:
        s = yaml.safe_load(spec.read_text(encoding="utf-8"))
        spec_mutate(s)
        spec.write_text(yaml.safe_dump(s, sort_keys=False), encoding="utf-8")
    repin(spec, reg)


def to_item_names(spec: Path, reg: Path, *, reg_extra=None, spec_extra=None) -> None:
    """Both documents in revision 11 names, plus optional further mutations."""
    def r(d):
        to_item_names_registration(d)
        if reg_extra is not None:
            reg_extra(d)

    def s(d):
        to_item_names_spec(d)
        if spec_extra is not None:
            spec_extra(d)

    rewrite(spec, reg, reg_mutate=r, spec_mutate=s)
