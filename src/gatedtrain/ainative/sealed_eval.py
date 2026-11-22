"""AN-6, AN-8, AN-9, AN-10: the sealed-evaluation core shared by the in-process stages and the keeper.

Everything here is pure: no project, no trust backend, no ledger. `trust_stages` (step 1, in the operator's process)
and the keeper (step 2, a separate process that holds the sealed items) call the same code, so the split, the
scoring and the gate statistics cannot drift apart between the two paths.

- `split_data`: parse the JSONL rows, remove duplicate inputs BEFORE the split, size the sealed part with the power
  plan at the given planning alpha, and split deterministically by the data hash.
- `power_record`: the body of the `power` record (without `schema`).
- `score_arm` / `item_scorer`: score one arm with the registered sampler, eval_seeds and scorer; any exception from
  pack code becomes `Refused` AN-8 with the exception's class name only, never its message (it may quote a sealed
  input or label). With a batch generator (AN-25), `score_arm` makes one `batch.generate_batch` call per arm for all
  seeds through the caller's runner (the keeper's inference container) and scores its predictions the same way.
- `decide`: the gate statistics on two arms of `score_arm` rows (`gates.evaluate_items`, then `gates.apply_post_hoc`
  for a post hoc registration).
- The shared report pieces: `known_limits`, `round_display`, `effective_status`, `headline`, `display_block`.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable

from .. import gates
from ..canonical import canonical_json, sha256_bytes
from ..errors import Problem, Refused
from ..registration import max_unpaired, min_paired
from .batch import CRASH, CRASH_FIX, BatchGenerator, BatchRunner, generate_batch

DEFAULT_MDE = 0.10
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8
# Planning SD of the paired per-item difference (candidate minus base) of two 0/1 scores: the bound when the two
# scores are independent and each has the largest SD of a 0/1 score, 0.5, so sqrt(0.5^2 * 2) (design 05 §18.4;
# user decision, 2026-11-20). Positively correlated arms have a smaller SD, so the plan does not under-size the
# sealed part for them.
PILOT_SIGMA = math.sqrt(0.5)
PILOT_SOURCE = ("the independence bound for the SD of the paired difference of two 0/1 scores (exact match): "
                "sqrt(0.5^2 * 2)")
MAX_SEALED_FRACTION = 0.5


def _refuse(rule: str, field: str, message: str, fix: str) -> Refused:
    return Refused([Problem(rule, field, message, fix)])


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


# ---------------------------------------------------------------------------
# split (AN-6)
# ---------------------------------------------------------------------------

def parse_rows(raw: bytes) -> list[tuple[int, Any, Any]]:
    """(line number, input, label) of every non-blank JSONL line; refuses (AN-6) a malformed file or row."""
    if not raw.strip():
        raise _refuse("AN-6", "data", "the data file is empty", "add labelled rows, one JSON object per line")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise _refuse("AN-6", "data", "the data file is not valid UTF-8", "save the file as UTF-8") from None
    out: list[tuple[int, Any, Any]] = []
    for n, line in enumerate(text.split("\n"), 1):
        line = line.rstrip("\r")
        if not line.strip():
            continue
        field = f"data line {n}"
        try:
            row = json.loads(line)
        except ValueError:
            raise _refuse("AN-6", field, "the line is not valid JSON", "write one JSON object per line") from None
        if not isinstance(row, dict):
            raise _refuse("AN-6", field, "the line is not a JSON object", 'use {"input": ..., "label": ...}')
        extra = sorted(set(row) - {"input", "label"})
        if extra:
            raise _refuse("AN-6", field, f"the row has fields other than input and label ({', '.join(extra)})",
                          'keep only "input" and "label"')
        if "input" not in row:
            raise _refuse("AN-6", field, "the row has no `input`", 'add an "input" (text or object)')
        if row.get("label") is None:
            raise _refuse("AN-6", field, "the row has no `label`; every example needs a verifiable answer or label",
                          'add a "label" (text, number, list or object); unlabelled data cannot be sealed')
        if not isinstance(row["input"], (str, dict)):
            raise _refuse("AN-6", field, "`input` is neither text nor an object", "use text or an object as input")
        # A list label is for set_f1 (design rev-23 AN-6, extended in step 2).
        if not (isinstance(row["label"], (str, dict, list)) or _num(row["label"])):
            raise _refuse("AN-6", field, "`label` is not text, a number, a list or an object",
                          "use text, a number, a list or an object as label")
        out.append((n, row["input"], row["label"]))
    if not out:
        raise _refuse("AN-6", "data", "the data file is empty", "add labelled rows, one JSON object per line")
    return out


def _canon(value: Any, field: str) -> bytes:
    """The refusal names the row only: the encoder's own message can quote a key or a number of the data."""
    try:
        return canonical_json(value)
    except ValueError:
        raise _refuse("AN-6", field, "the row holds a value canonical JSON cannot encode (a non-finite number, an "
                      "integer beyond 2^53 - 1, or a key or text with a character it refuses)",
                      "use plain JSON values") from None


def dedup(rows: list[tuple[int, Any, Any]]) -> tuple[list[dict], int]:
    """Unique rows in first-occurrence order, keyed by the sha256 of the canonical `input`; the same input with a
    different label is refused. Returns (rows with `ihash`, number of exact duplicates removed)."""
    seen: dict[str, dict] = {}
    conflicts: list[tuple[int, int]] = []
    removed = 0
    for n, inp, label in rows:
        ihash = sha256_bytes(_canon(inp, f"data line {n}"))
        lcanon = _canon(label, f"data line {n}")
        prior = seen.get(ihash)
        if prior is None:
            seen[ihash] = {"ihash": ihash, "input": inp, "label": label, "lcanon": lcanon, "line": n}
        elif prior["lcanon"] == lcanon:
            removed += 1
        else:
            conflicts.append((prior["line"], n))
    if conflicts:
        shown = ", ".join(f"{a} and {b}" for a, b in conflicts[:5])
        raise _refuse("AN-6", "data", f"the same input has different labels (lines {shown}"
                      f"{', ...' if len(conflicts) > 5 else ''})", "keep one label per distinct input")
    return list(seen.values()), removed


def power_clause(planning_alpha: float) -> dict:
    """The superiority clause intake sizes the sealed part with (margin 0, the planning MDE, `planning_alpha`)."""
    return {"type": "superiority", "margin": 0.0, "planning_effect": DEFAULT_MDE, "alpha": planning_alpha,
            "power": DEFAULT_POWER, "sigma_plan": "fixed"}


def split_data(raw: bytes, digest: str, *, planning_alpha: float, train_only: frozenset[str] = frozenset()) -> dict:
    """Dedup, size and split the data file's bytes (`digest` is their `sha256:<hex>`).

    The sealed size is the power plan's `n_required` at `planning_alpha`, capped at half the distinct rows; the
    split orders the rows by sha256(`digest:ihash`) and seals the first ones. A row whose input hash is in
    `train_only` (the keeper: inputs it released as training before) is never sealed; the sealed size is then also
    capped at the number of the other rows. Returns `digest`, `n_unique`,
    `n_removed`, `n_needed`, `n_sealed`, `mde` (achieved), `conclusive`, `more` (rows still needed), `sealed` and
    `train` (`{input, label}` rows), `input_hashes` (sha256 of the canonical `input` of every distinct row, in
    first-occurrence order) and `sealed_input_hashes` (the same for the sealed rows, in sealed order)."""
    unique, removed = dedup(parse_rows(raw))
    n = len(unique)
    if n < 2:
        raise _refuse("AN-6", "data", "the data has fewer than 2 distinct examples, so it cannot be split",
                      "add more labelled examples")
    cap = int(n * MAX_SEALED_FRACTION)
    pilot = {"sigma": PILOT_SIGMA, "source": PILOT_SOURCE}
    clause = power_clause(planning_alpha)
    needed = gates.plan(clause, pilot, n_available=cap)["n_required"]
    eligible = [r for r in unique if r["ihash"] not in train_only] if train_only else unique
    n_sealed = min(needed, cap, len(eligible))
    order = sorted(eligible, key=lambda r: hashlib.sha256(f"{digest}:{r['ihash']}".encode()).hexdigest())
    sealed = order[:n_sealed]
    sealed_ids = {r["ihash"] for r in sealed}
    train = [r for r in unique if r["ihash"] not in sealed_ids]
    achieved = gates.plan(clause, pilot, n_available=n_sealed)["mde"]
    return {"digest": digest, "n_unique": n, "n_removed": removed, "n_needed": needed, "n_sealed": n_sealed,
            "mde": achieved, "conclusive": n_sealed >= needed,
            "more": max(0, int(round(needed / MAX_SEALED_FRACTION)) - n),
            "sealed": [{"input": r["input"], "label": r["label"]} for r in sealed],
            "train": [{"input": r["input"], "label": r["label"]} for r in train],
            "input_hashes": [r["ihash"] for r in unique],
            "sealed_input_hashes": [r["ihash"] for r in sealed]}


def power_record(s: dict, *, planning_alpha: float) -> dict:
    """The `power` record body (no `schema`, `schema_version`) of a `split_data` result. The in-process path writes
    exactly these keys; a caller that needs more (the keeper's `planning_alpha`) adds them itself."""
    return {"n_unique": s["n_unique"], "n_sealed": s["n_sealed"], "n_needed": s["n_needed"], "mde": s["mde"],
            "target_mde": DEFAULT_MDE, "alpha": planning_alpha, "power": DEFAULT_POWER, "sigma": PILOT_SIGMA,
            "conclusive_expected": s["conclusive"], "n_more_examples": s["more"]}


# ---------------------------------------------------------------------------
# scoring (AN-8)
# ---------------------------------------------------------------------------

def eval_seeds(full: dict) -> list:
    """The registration's non-empty `clause.eval_seeds`; refuses (AN-8) otherwise."""
    seeds = (full.get("clause") or {}).get("eval_seeds")
    if not (isinstance(seeds, list) and seeds):
        raise _refuse("AN-8", "registration.clause.eval_seeds", "the registration has no eval_seeds",
                      "rerun preregister")
    return seeds


def item_scorer(generate: Callable[..., Any], model: dict, full: dict, score: Callable[[Any, Any], float], seed: Any,
                *, score_range: tuple[float, float], scorer_name: str) -> Callable[[dict], dict]:
    """A function of one sealed item (`{input, label}`) to its `{item_id, seed, score}` record under one seed.

    `item_id` is the item's position in the order the function is called in (counted per seed). `generate` is the
    registered generator, called as `generate(model, {"input": ...}, sampler=..., seed=seed)` with copies of the
    model reference and the registered sampler; the label goes to `score` only. An exception from `generate` or
    `score` (a `Refused` too: pack code may build one with the label in it) is refused (AN-8) with its class name
    only, and so is a score that is not a finite number in `score_range`."""
    sampler = full["sampler"]
    counter = [0]

    def run(item: dict) -> dict:
        index = counter[0]
        counter[0] += 1
        try:
            prediction = generate(dict(model), {"input": item["input"]}, sampler=dict(sampler), seed=seed)
        except Exception as exc:       # never carry the item (or the pack's message about it) out
            raise _refuse("AN-8", "generate", f"the generation function failed ({type(exc).__name__})",
                          "fix the scenario pack's generate") from None
        return {"item_id": index, "seed": seed, "score": _score_one(score, prediction, item["label"], score_range,
                                                                    scorer_name)}

    return run


def _score_one(score: Callable[[Any, Any], float], prediction: Any, label: Any, score_range: tuple[float, float],
               scorer_name: str) -> float:
    """The registered scorer's value for one prediction, checked to be a finite number in `score_range`; refusals
    (AN-8) name the exception's class only."""
    lo, hi = score_range
    try:
        value = score(prediction, label)
    except Exception as exc:       # a pack scorer's message may quote the label: class name only
        raise _refuse("AN-8", "score", f"the scorer failed ({type(exc).__name__})",
                      "fix the scorer of the scenario pack") from None
    if not (_num(value) and math.isfinite(value) and lo <= value <= hi):
        raise _refuse("AN-8", "score", f"the scorer {scorer_name} returned a value that is not a number in its "
                      f"range [{lo:g}, {hi:g}]", "fix the scorer of the scenario pack")
    return float(value)


def score_arm(items: list[dict], generate: Callable[..., Any] | None, model: dict, full: dict,
              score: Callable[[Any, Any], float], *, score_range: tuple[float, float], scorer_name: str,
              batch: BatchGenerator | None = None, runner: BatchRunner | None = None) -> list[dict]:
    """One `{item_id, seed, score}` record per registered seed and item (seeds outer, items in the given order) of
    one arm: `items` are the sealed `{input, label}` items, `generate` the registered generator, `model` the arm's
    model reference, `full` the registration (its `sampler` and `clause.eval_seeds`), `score` the bound registered
    scorer with its `score_range` and name. Refusals as in `item_scorer`.

    With `batch` (AN-25), `generate` is not used: one `generate_batch` call through `runner` gives the predictions
    for all seeds and items, each scored as in `item_scorer`; the records are the same in shape and order. Without a
    `runner` it is refused (AN-8): a batch generator runs only in the keeper's inference container. Predictions not
    in (seed, item) order, one per pair, are a crash (AN-9)."""
    if batch is not None:
        return _score_arm_batch(items, model, full, score, batch, runner, score_range=score_range,
                                scorer_name=scorer_name)
    records: list[dict] = []
    for seed in eval_seeds(full):
        run = item_scorer(generate, model, full, score, seed, score_range=score_range, scorer_name=scorer_name)
        records.extend(run(dict(item)) for item in items)
    return records


def _score_arm_batch(items: list[dict], model: dict, full: dict, score: Callable[[Any, Any], float],
                     gen: BatchGenerator, runner: BatchRunner | None, *, score_range: tuple[float, float],
                     scorer_name: str) -> list[dict]:
    if runner is None:
        raise _refuse("AN-8", "generate", "a batch generator runs in the keeper's inference container",
                      "bind the project to a keeper")
    seeds = eval_seeds(full)
    predictions = generate_batch(gen, dict(model), [{"input": item["input"]} for item in items],
                                 dict(full["sampler"]), list(seeds), runner=runner)
    want = [(s, i) for s in seeds for i in range(len(items))]
    if not (isinstance(predictions, list) and len(predictions) == len(want) and all(
            isinstance(p, dict) and p.get("seed") == s and p.get("item_id") == i and "prediction" in p
            for p, (s, i) in zip(predictions, want))):
        raise _refuse("AN-9", "predictions", f"{CRASH}: the runner did not return {len(want)} predictions in seed "
                      "and item order", CRASH_FIX)
    return [{"item_id": i, "seed": s, "score": _score_one(score, p["prediction"], items[i]["label"], score_range,
                                                          scorer_name)}
            for p, (s, i) in zip(predictions, want)]


# ---------------------------------------------------------------------------
# gate statistics (AN-9)
# ---------------------------------------------------------------------------

def convert_record(rec: Any, seeds: list) -> Any:
    """A sealed record as an `evaluate_items` record. A row that is not a well formed record is passed on as it is
    (or with the fields it lacks left out), so the gate counts it as missing instead of dropping it."""
    if not isinstance(rec, dict):
        return rec
    out = {k: v for k, v in rec.items() if k != "item_id"}
    if "item_id" in rec:
        out["item"] = rec["item_id"]
    seed = rec.get("seed")
    out["repeat"] = seeds.index(seed) if isinstance(seed, int) and seed in seeds else -1
    return out


def decide(full: dict, base: dict, cand: dict, *, alpha: float, n_sealed: int, expected_models: dict,
           score_range: tuple[float, float]) -> dict:
    """The gate result of one test at `alpha` (the alpha the ledger granted, not the registered one).

    `base` and `cand` are arms of the shape `{"records": rows, "model_ref_sha256": hash}`: `rows` are `score_arm`
    records (`{item_id, seed, score}`, as a `sealed_records` record holds them; a malformed row is passed on and
    counted as missing), `hash` the model reference hash the records were made for. `expected_models` maps
    `baseline` and `candidate` to the hashes the gate expects. The registration gives the clause, the eval_seeds,
    the bootstrap, `min_paired` and `max_unpaired`; the items are `range(n_sealed)`. A post hoc registration's PASS
    is downgraded (`gates.apply_post_hoc`)."""
    seeds = eval_seeds(full)
    clause = dict(full["clause"])
    clause["alpha"] = alpha
    arms = [{"records": [convert_record(r, seeds) for r in arm["records"]],
             "model_manifest_hash": arm.get("model_ref_sha256")} for arm in (base, cand)]
    boot = full["bootstrap"]
    result = gates.evaluate_items(clause, arms[0], arms[1], items=list(range(n_sealed)),
                                  seed=int(boot["seed"]), n_boot=int(boot["replicates"]),
                                  min_paired=min_paired(full), max_unpaired=max_unpaired(full),
                                  score_range=score_range, expected_models=expected_models)
    if bool(full.get("post_hoc", False)):
        result = gates.apply_post_hoc(result)
    return result


# ---------------------------------------------------------------------------
# report pieces (AN-10)
# ---------------------------------------------------------------------------

LIMIT_ALPHA_PER_PROJECT = ("the alpha is per project in step 1: a new project on the same data gets a fresh alpha "
                           "and the same sealed split, so its test would not be independent of this one (step 2 keys "
                           "the alpha by the sealed set)")
LIMIT_MODEL_REFERENCE = ("a model reference must content-address its weights; step 1 does not check this, so after a "
                         "retrain the gate records may describe weights the reference no longer points to")
LIMIT_METRIC_CONFIRMATION = ("the metric confirmation is the decision flag metric_confirmed, set by whoever runs the "
                             "stages; step 1 has no separate channel that shows the sentence to the user")


def known_limits(guarantee: str, attempts: list[dict]) -> list[dict]:
    """The report's known limits, from constants and records: the backend's guarantee, the per-project alpha, the
    metric confirmation flag, that model references must content-address their weights, and the number of
    `evaluate_sealed` attempts in the attempt log (refused ones too)."""
    n = sum(1 for a in attempts if a.get("stage") == "evaluate_sealed" and a.get("event") == "start")
    return [{"id": "sealed_guarantee", "text": guarantee},
            {"id": "alpha_per_project", "text": LIMIT_ALPHA_PER_PROJECT},
            {"id": "metric_confirmation", "text": LIMIT_METRIC_CONFIRMATION},
            {"id": "model_reference", "text": LIMIT_MODEL_REFERENCE},
            {"id": "sealed_evaluations", "attempts": n,
             "text": f"the sealed set was scored in {n} evaluate_sealed attempt{'' if n == 1 else 's'} (refused "
                     "attempts included); one project allows one scoring of the sealed set"}]


def round_display(x: Any) -> Any:
    """A float rounded to 4 places for display; anything else as it is."""
    return round(x, 4) if isinstance(x, float) else x


def effective_status(status: str, post_hoc: bool) -> str:
    """The status a report shows: a post hoc PASS is INCONCLUSIVE."""
    return "INCONCLUSIVE" if post_hoc and status == gates.PASS else status


def headline(status: str, post_hoc: bool) -> str:
    """The report's first markdown line."""
    status = effective_status(status, post_hoc)
    if post_hoc:
        return f"# {status}: post hoc, not a confirmed gain"
    said = {gates.PASS: "the new model beats the base model by more than the registered margin",
            gates.FAIL: "the new model does not beat the base model by the registered margin",
            gates.INCONCLUSIVE: "the sealed test cannot tell whether the new model is better",
            gates.MISSING: "the sealed records are incomplete, so no verdict"}
    return f"# {status}: {said.get(status, 'no verdict')}"


def display_block(gate: dict, power: dict) -> dict:
    """The report's `display` block: the gate result's effect, interval and arm scores and the power plan's MDE,
    rounded for display."""
    return {"effect": round_display(gate["delta_hat_natural"]), "ci_lower": round_display(gate["ci_lower_natural"]),
            "ci_upper": round_display(gate["ci_upper_natural"]), "ci_level": round_display(gate["ci_level"]),
            "baseline_score": round_display(gate["baseline_score"]),
            "candidate_score": round_display(gate["candidate_score"]), "mde_planned": round_display(power["mde"])}
