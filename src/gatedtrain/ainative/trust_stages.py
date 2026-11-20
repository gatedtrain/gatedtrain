"""AN-6 and AN-7: the first two trust stages, `intake_seal` and `preregister`.

`intake_seal` reads the project's data (one JSONL file of `{"input", "label"}` rows), removes duplicate inputs
BEFORE it splits, seals a deterministic part for evaluation and passes on only the allowed views: the training part,
the sealed part's manifest hash and size, a label distribution limited to labels the training part already shows, and
the power numbers. The sealed items go only to `ctx.trust.seal`; no stage output holds a sealed input or label text.

`preregister` writes the registration before anything trains. The metric comes from a small scorer library, the user
confirms it in one plain sentence (`decision["metric_confirmed"]` must be true), and the registration passes
`registration.check_contents`. A later run with different values after a training stage ran amends the
registration and marks it `post_hoc` (an output field the report reads).

`preregister` reads the deployment record (`gatedtrain.ainative.deployment`: `{"sampler": {...}, "eval_seeds": [...]}`)
that a pluggable stage before it produces, and copies the sampler and the evaluation seeds into the registration.
`evaluate_sealed` must use exactly the registered sampler and eval_seeds and score only with
`scorers.exact_match`; the registered evaluator is `scorers.EVALUATOR_ID` with the hash of the scorer's source.

Both stages take no pipeline params: their behaviour comes from the project's data, the user's decision and the
registration, never from pipeline configuration.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .. import ANALYSIS_CODE_VERSION, gates
from ..canonical import canonical_json, file_sha256, sha256_bytes
from ..errors import Problem, Refused
from ..registration import check_contents, max_unpaired, min_paired, resolved_evaluator_hash
from . import scorers
from .project import ArtifactRef, Project, _refuse_links, _utc_now, _write_atomic
from .stage import Proposal, StageContext
from .trust import SealedRef

SCHEMA_VERSION = 1
SCHEMA_TRAIN_PART = "gatedtrain.ainative.train_part"
SCHEMA_SEALED_SUMMARY = "gatedtrain.ainative.sealed_summary"
SCHEMA_POWER = "gatedtrain.ainative.power"
SCHEMA_REGISTRATION = "gatedtrain.ainative.registration"
SCHEMA_DEPLOYMENT = "gatedtrain.ainative.deployment"
SCHEMA_MODEL_REF = "gatedtrain.ainative.model_ref"
SCHEMA_SEALED_RECORDS = "gatedtrain.ainative.sealed_records"
SCHEMA_GATE_RESULT = "gatedtrain.ainative.gate_result"
SCHEMA_REPORT = "gatedtrain.ainative.report"

MAX_DATA_BYTES = 64 * 1024 * 1024
DEFAULT_MDE = 0.10
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8
PILOT_SIGMA = 0.5            # the maximum SD of a 0/1 score, so the plan never under-sizes the sealed part
MAX_SEALED_FRACTION = 0.5


def _refuse(rule: str, field: str, message: str, fix: str) -> Refused:
    return Refused([Problem(rule, field, message, fix)])


def _meta(schema: str) -> dict:
    return {"schema": schema, "schema_version": SCHEMA_VERSION}


# ---------------------------------------------------------------------------
# intake_seal
# ---------------------------------------------------------------------------

def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _read_data(project: Project) -> tuple[bytes, str]:
    path = Path(project.data["path"])
    fix_file = "pass one JSONL file of {\"input\": ..., \"label\": ...} rows when the project is created"
    if not path.is_file():
        raise _refuse("AN-6", "data", "the project data is not a single file; step 1 reads one JSONL file", fix_file)
    try:
        size = path.stat().st_size
        if size > MAX_DATA_BYTES:
            raise _refuse("AN-6", "data", f"the data file is {size} bytes; the limit is 64 MiB ({MAX_DATA_BYTES} bytes)",
                          "split the data or use a smaller sample")
        raw = path.read_bytes()
    except OSError as exc:
        raise _refuse("AN-6", "data", f"cannot read the data file ({type(exc).__name__})",
                      "check the file permissions") from None
    if len(raw) > MAX_DATA_BYTES:
        raise _refuse("AN-6", "data", "the data file is larger than 64 MiB", "split the data or use a smaller sample")
    digest = sha256_bytes(raw)
    if digest != project.data.get("sha256"):
        raise _refuse("AN-6", "data", "the data file changed after the project was created",
                      "create a new project for the changed data")
    return raw, digest


def _parse_rows(raw: bytes) -> list[tuple[int, Any, Any]]:
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
                          'add a "label" (text, number or object); unlabelled data cannot be sealed')
        if not isinstance(row["input"], (str, dict)):
            raise _refuse("AN-6", field, "`input` is neither text nor an object", "use text or an object as input")
        if not (isinstance(row["label"], (str, dict)) or _num(row["label"])):
            raise _refuse("AN-6", field, "`label` is not text, a number or an object",
                          "use text, a number or an object as label")
        out.append((n, row["input"], row["label"]))
    if not out:
        raise _refuse("AN-6", "data", "the data file is empty", "add labelled rows, one JSON object per line")
    return out


def _canon(value: Any, field: str) -> bytes:
    try:
        return canonical_json(value)
    except ValueError as exc:
        raise _refuse("AN-6", field, f"the value is not canonical JSON: {exc}", "use plain JSON values") from None


def _dedup(rows: list[tuple[int, Any, Any]]) -> tuple[list[dict], int]:
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


def _power_clause() -> dict:
    return {"type": "superiority", "margin": 0.0, "planning_effect": DEFAULT_MDE, "alpha": DEFAULT_ALPHA,
            "power": DEFAULT_POWER, "sigma_plan": "fixed"}


def _split(project: Project) -> dict:
    raw, digest = _read_data(project)
    unique, removed = _dedup(_parse_rows(raw))
    n = len(unique)
    if n < 2:
        raise _refuse("AN-6", "data", "the data has fewer than 2 distinct examples, so it cannot be split",
                      "add more labelled examples")
    cap = int(n * MAX_SEALED_FRACTION)
    pilot = {"sigma": PILOT_SIGMA, "source": "the maximum SD of a 0/1 score (exact match)"}
    needed = gates.plan(_power_clause(), pilot, n_available=cap)["n_required"]
    n_sealed = min(needed, cap)
    order = sorted(unique, key=lambda r: hashlib.sha256(f"{digest}:{r['ihash']}".encode()).hexdigest())
    sealed = order[:n_sealed]
    sealed_ids = {r["ihash"] for r in sealed}
    train = [r for r in unique if r["ihash"] not in sealed_ids]
    achieved = gates.plan(_power_clause(), pilot, n_available=n_sealed)["mde"]
    return {"digest": digest, "n_unique": n, "n_removed": removed, "n_needed": needed, "n_sealed": n_sealed,
            "mde": achieved, "conclusive": n_sealed >= needed,
            "more": max(0, int(round(needed / MAX_SEALED_FRACTION)) - n),
            "sealed": [{"input": r["input"], "label": r["label"]} for r in sealed],
            "train": [{"input": r["input"], "label": r["label"]} for r in train]}


class IntakeSeal:
    name = "intake_seal"
    version = "1"
    inputs: dict[str, str] = {}
    outputs = {"train_part": SCHEMA_TRAIN_PART, "sealed_summary": SCHEMA_SEALED_SUMMARY, "power": SCHEMA_POWER}
    needs: frozenset[str] = frozenset()
    params: frozenset[str] = frozenset()
    trains = False

    def plan(self, ctx: StageContext) -> Proposal:
        s = _split(ctx.project)
        if s["conclusive"]:
            summary = (f"{s['n_unique']} distinct labelled examples: {s['n_sealed']} will be sealed for the final "
                       f"test and {len(s['train'])} are for training; the test can detect a gain of "
                       f"{DEFAULT_MDE:g} or more.")
        else:
            summary = (f"{s['n_unique']} distinct labelled examples are too few: the final test will most likely be "
                       f"INCONCLUSIVE (it needs {s['n_needed']} sealed examples to detect a gain of "
                       f"{DEFAULT_MDE:g}); about {s['more']} more labelled examples would make it conclusive.")
        estimate = {"n_needed": s["n_needed"], "n_sealed": s["n_sealed"], "mde": s["mde"],
                    "conclusive_expected": s["conclusive"], "n_more_examples": s["more"]}
        choices = {"n_unique": s["n_unique"], "n_duplicates_removed": s["n_removed"], "n_train": len(s["train"]),
                   "n_needed": s["n_needed"], "mde": DEFAULT_MDE, "alpha": DEFAULT_ALPHA, "power": DEFAULT_POWER}
        return Proposal(self.name, summary, choices, estimate, None if s["conclusive"] else "data_too_small")

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        s = _split(ctx.project)
        if not s["conclusive"] and decision.get("accept_inconclusive") is not True:
            raise _refuse("AN-6", "decision.accept_inconclusive",
                          f"the data are too small: the final test will most likely be INCONCLUSIVE "
                          f"({s['n_sealed']} sealed examples, {s['n_needed']} needed); about {s['more']} more labelled "
                          "examples would make it conclusive",
                          "add labelled examples, or set decision accept_inconclusive to true to go on")
        trust = ctx.trust
        ref = trust.seal(s["sealed"])
        summary = trust.sealed_summary(ref)
        shown = {str(r["label"]) for r in s["train"]}
        counts = {k: v for k, v in summary["label_counts"].items() if k in shown}   # no label text only the seal has
        project = ctx.project
        train = {**_meta(SCHEMA_TRAIN_PART), "n": len(s["train"]), "n_duplicates_removed": s["n_removed"],
                 "data_sha256": s["digest"], "rows": s["train"]}
        sealed = {**_meta(SCHEMA_SEALED_SUMMARY), "n": summary["n"], "manifest_sha256": summary["manifest_sha256"],
                  "label_counts": counts, "label_counts_other": summary["n"] - sum(counts.values()),
                  "guarantee": trust.sealed_guarantee}
        power = {**_meta(SCHEMA_POWER), "n_unique": s["n_unique"], "n_sealed": s["n_sealed"],
                 "n_needed": s["n_needed"], "mde": s["mde"], "target_mde": DEFAULT_MDE, "alpha": DEFAULT_ALPHA,
                 "power": DEFAULT_POWER, "sigma": PILOT_SIGMA, "conclusive_expected": s["conclusive"],
                 "n_more_examples": s["more"]}
        return {"train_part": project.write_artifact(self.name, "train_part", train, SCHEMA_TRAIN_PART),
                "sealed_summary": project.write_artifact(self.name, "sealed_summary", sealed, SCHEMA_SEALED_SUMMARY),
                "power": project.write_artifact(self.name, "power", power, SCHEMA_POWER)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        problems: list[Problem] = []

        def bad(field: str, message: str) -> None:
            problems.append(Problem("AN-6", field, message, "rerun intake_seal"))

        train = ctx.project.read_artifact(outputs["train_part"])
        sealed = ctx.project.read_artifact(outputs["sealed_summary"])
        power = ctx.project.read_artifact(outputs["power"])
        if not isinstance(train, dict) or not isinstance(sealed, dict) or not isinstance(power, dict):
            return [Problem("AN-6", "outputs", "an output record is not an object", "rerun intake_seal")]
        if train.get("n") != len(train.get("rows", [])):
            bad("train_part.n", "the training row count does not match its rows")
        if sealed.get("n") != power.get("n_sealed"):
            bad("sealed_summary.n", "the sealed size differs between the summary and the power record")
        if train.get("n", 0) + sealed.get("n", 0) != power.get("n_unique"):
            bad("power.n_unique", "training plus sealed rows do not add up to the distinct examples")
        h = sealed.get("manifest_sha256")
        if not (isinstance(h, str) and len(h) == 71 and h.startswith("sha256:")):
            bad("sealed_summary.manifest_sha256", "the manifest hash is not sha256:<64 hex>")
        if power.get("conclusive_expected") != (power.get("n_sealed", 0) >= power.get("n_needed", 1)):
            bad("power.conclusive_expected", "the flag does not match n_sealed and n_needed")
        return problems


# ---------------------------------------------------------------------------
# preregister
# ---------------------------------------------------------------------------

# The scorer library of step 1. Each scorer names the evaluator id that registers it and its direction; the evaluator's
# manifest hash comes from the scorer's source (`scorers.manifest_hash`). The training reward may differ.
SCORERS: dict[str, dict] = {
    "exact_match": {"evaluator": scorers.EVALUATOR_ID, "direction": scorers.DIRECTION,
                    "what": "the share of items whose answer matches yours exactly"},
}
DEFAULT_SCORER = "exact_match"
MAX_MARGIN = 0.5


def _refuse7(field: str, message: str, fix: str) -> Refused:
    return _refuse("AN-7", field, message, fix)


def _values(ctx: StageContext, decision: dict) -> dict:
    """The scorer, direction, margin and alpha the registration will carry."""
    scorer = decision.get("scorer", DEFAULT_SCORER)
    if scorer not in SCORERS:
        raise _refuse7("decision.scorer", f"{scorer!r} is not in the scorer library",
                       f"use one of: {', '.join(sorted(SCORERS))}")
    row = SCORERS[scorer]
    direction = decision.get("direction", row["direction"])
    if direction != row["direction"]:
        raise _refuse7("decision.direction", f"scorer {scorer} is {row['direction']}, not {direction!r}",
                       f"use direction {row['direction']}, or pick another scorer")
    margin = decision.get("margin", 0.0)
    if not _num(margin) or not 0 <= margin <= MAX_MARGIN:
        raise _refuse7("decision.margin", f"margin {margin!r} is not a number from 0 to {MAX_MARGIN}",
                       "give the smallest gain that counts, as a fraction of the score (0.05 is 5 points)")
    alpha = ctx.ledger.available()
    if not _num(alpha) or not 0 < alpha < 0.5:
        raise _refuse7("alpha", f"no alpha is available to register (available: {alpha!r})",
                       "the alpha budget is spent; start a new project")
    return {"scorer": scorer, "direction": direction, "margin": float(margin), "alpha": float(alpha)}


def _sealed_summary(ctx: StageContext) -> dict:
    summary = ctx.project.read_artifact(ctx.inputs["sealed_summary"])
    if not isinstance(summary, dict) or "manifest_sha256" not in summary:
        raise _refuse7("inputs.sealed_summary", "the sealed summary is malformed", "rerun intake_seal")
    return summary


def _sentence(v: dict, n_sealed: int) -> str:
    points = f"{v['margin'] * 100:g}"
    return (f"I will count {SCORERS[v['scorer']]['what']}; the new model must beat the base model by more than "
            f"{points} points, tested at the {v['alpha']:g} significance level on {n_sealed} sealed items.")


def _deployment(ctx: StageContext) -> dict:
    dep = ctx.project.read_artifact(ctx.inputs["deployment"])
    seeds = dep.get("eval_seeds") if isinstance(dep, dict) else None
    if not (isinstance(dep, dict) and isinstance(dep.get("sampler"), dict) and isinstance(seeds, list) and seeds):
        raise _refuse7("inputs.deployment", "the deployment record has no sampler mapping and non-empty eval_seeds",
                       "rerun the stage that produces the deployment record")
    return {"sampler": dep["sampler"], "eval_seeds": seeds}


def _registration(v: dict, item_set_hash: str, dep: dict) -> dict:
    row = SCORERS[v["scorer"]]
    return {
        "registration_version": 1,
        "registration_id": f"ainative-{item_set_hash.removeprefix('sha256:')[:16]}",
        "signed_by": "prototype-unsigned",
        "workflow": "ainative",
        "gate": "quality_gate",
        "clause": {"type": "superiority", "metric": v["scorer"], "direction": v["direction"], "margin": v["margin"],
                   "alpha": v["alpha"], "power": DEFAULT_POWER, "planning_effect": v["margin"] + DEFAULT_MDE,
                   "sigma_plan": "fixed", "unit": "item", "samples_per_item": len(dep["eval_seeds"]),
                   "eval_seeds": list(dep["eval_seeds"]),
                   "interval": "cluster_bootstrap", "blocking": True},
        "margin_justification": f"Margin {v['margin']:g} (fraction of the {v['scorer']} score) as confirmed by the "
                                "user before any training.",
        "sigma_pilot": {"sigma": PILOT_SIGMA, "source": "the maximum SD of a 0/1 score (exact match)"},
        "eval_distribution": {"item_set_hash": item_set_hash, "env_version": None, "env_params_version": None},
        "sampler": dict(dep["sampler"]),
        "evaluators": [{"id": row["evaluator"], "manifest_hash": scorers.manifest_hash()}],
        "multiplicity": {"plan": "single_gate", "family": None},
        "analysis_code_version": ANALYSIS_CODE_VERSION,
        "bootstrap": {"replicates": 10000, "seed": int(item_set_hash.removeprefix("sha256:")[:8], 16)},
    }


_AMEND_KEYS = ("amends", "created_at", "post_hoc")


def _training_ran(project: Project) -> bool:
    """True when the attempt log shows a stage that trains starting after the first `preregister` run was done.
    Reads only the log (`start` rows carry `trains`, resolved at run time), never the editable pipeline file."""
    done = False
    for row in project.read_attempts():
        if row.get("event") == "done" and row.get("stage_type") == "preregister":
            done = True
        elif done and row.get("event") == "start" and row.get("trains") is True:
            return True
    return False


def _prior(ctx: StageContext) -> tuple[dict, dict] | None:
    """(registration output record, full registration) of the last done `preregister` run, or None."""
    last = None
    for row in ctx.project.read_attempts():
        if row.get("event") == "done" and row.get("stage_type") == "preregister":
            last = row
    if last is None:
        return None
    r = (last.get("outputs") or {}).get("registration")
    if not isinstance(r, dict):
        return None
    ref = ArtifactRef(r["stage"], r["name"], r["schema"], r["sha256"], r["path"])
    out = ctx.project.read_artifact(ref)
    return out, ctx.trust.registration(out["hash"])


class Preregister:
    name = "preregister"
    version = "1"
    inputs = {"sealed_summary": SCHEMA_SEALED_SUMMARY, "deployment": SCHEMA_DEPLOYMENT}
    outputs = {"registration": SCHEMA_REGISTRATION}
    needs = frozenset({"user"})
    params: frozenset[str] = frozenset()
    trains = False

    def plan(self, ctx: StageContext) -> Proposal:
        v = _values(ctx, {})
        n = _sealed_summary(ctx)["n"]
        choices = {**v, "available_scorers": {k: r["what"] for k, r in SCORERS.items()}, "n_sealed": n}
        return Proposal(self.name, _sentence(v, n), choices, {}, "confirm_metric")

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        if decision.get("metric_confirmed") is not True:
            raise _refuse7("decision.metric_confirmed",
                           "the user has not confirmed the deciding metric, direction and margin; nothing trains "
                           "before that",
                           "show the plan() summary to the user and set decision metric_confirmed to true once "
                           "they agree (they may change scorer or margin first)")
        v = _values(ctx, decision)
        item_hash = _sealed_summary(ctx)["manifest_sha256"]
        base = _registration(v, item_hash, _deployment(ctx))
        problems = check_contents(base)
        if problems:
            raise Refused(problems)
        final, amends, post_hoc = base, None, False
        prior = _prior(ctx)
        if prior is not None:
            prior_out, prior_full = prior
            changed = {k: x for k, x in prior_full.items() if k not in _AMEND_KEYS} != base
            if prior_full.get("post_hoc") is True or (changed and _training_ran(ctx.project)):
                post_hoc, amends = True, prior_out["hash"]
                final = {**base, "post_hoc": True, "amends": amends, "created_at": _utc_now()}
                problems = check_contents(final)
                if problems:
                    raise Refused(problems)
        digest = ctx.trust.register(final)
        out = {**_meta(SCHEMA_REGISTRATION), "hash": digest, "scorer": v["scorer"], "direction": v["direction"],
               "margin": v["margin"], "alpha": v["alpha"], "item_set_hash": item_hash, "post_hoc": post_hoc,
               "amends": amends}
        return {"registration": ctx.project.write_artifact(self.name, "registration", out, SCHEMA_REGISTRATION)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        out = ctx.project.read_artifact(outputs["registration"])
        full = ctx.trust.registration(out["hash"])
        problems = list(check_contents(full))

        def bad(field: str, message: str) -> None:
            problems.append(Problem("AN-7", field, message, "rerun preregister"))

        item_hash = _sealed_summary(ctx)["manifest_sha256"]
        if out.get("item_set_hash") != item_hash or full.get("eval_distribution", {}).get("item_set_hash") != item_hash:
            bad("registration.item_set_hash", "the registered item set is not the sealed set")
        clause = full.get("clause", {})
        for key in ("margin", "alpha", "direction"):
            if clause.get(key) != out.get(key):
                bad(f"registration.{key}", f"the output record and the registration disagree on {key}")
        if clause.get("metric") != out.get("scorer"):
            bad("registration.scorer", "the output record and the registration disagree on the scorer")
        evs = full.get("evaluators")
        ev = evs[0] if isinstance(evs, list) and evs and isinstance(evs[0], dict) else {}
        if ev.get("id") != scorers.EVALUATOR_ID or ev.get("manifest_hash") != resolved_evaluator_hash(ev):
            bad("registration.evaluators[0].manifest_hash",
                "the registered evaluator hash is not the one the library resolves for the scorer")
        if bool(full.get("post_hoc", False)) != out.get("post_hoc"):
            bad("registration.post_hoc", "the output record and the registration disagree on post_hoc")
        return problems


# ---------------------------------------------------------------------------
# evaluate_sealed, gate, report (AN-8, AN-9, AN-10)
# ---------------------------------------------------------------------------

# The generation functions of the scenario packs, by the name a model reference gives in `generator`. A pack
# registers `generate(model_ref, item, *, sampler, seed) -> prediction`; `item` is `{"input": ...}` only, the label
# stays with the scorer. Task 8's synthetic pack registers itself here.
GENERATORS: dict[str, Callable[..., Any]] = {}


def register_generator(name: str, fn: Callable[..., Any]) -> None:
    GENERATORS[name] = fn


_REG_KEYS = ("hash", "scorer", "direction", "margin", "alpha", "item_set_hash", "post_hoc", "amends")


def _registered(ctx: StageContext, rule: str) -> tuple[dict, dict]:
    """(registration output record, full registration) of the `registration` input."""
    out = ctx.project.read_artifact(ctx.inputs["registration"])
    if not isinstance(out, dict) or not isinstance(out.get("hash"), str):
        raise _refuse(rule, "inputs.registration", "the registration record is malformed", "rerun preregister")
    return out, ctx.trust.registration(out["hash"])


def _evaluator_problem(full: dict) -> str | None:
    evs = full.get("evaluators")
    ev = evs[0] if isinstance(evs, list) and evs and isinstance(evs[0], dict) else {}
    if ev.get("id") != scorers.EVALUATOR_ID or ev.get("manifest_hash") != scorers.manifest_hash():
        return ("the registered evaluator is not the scorer this build runs (its id or manifest hash differs), so the "
                "sealed items cannot be scored as registered")
    return None


def _eval_seeds(full: dict) -> list:
    seeds = (full.get("clause") or {}).get("eval_seeds")
    if not (isinstance(seeds, list) and seeds):
        raise _refuse("AN-8", "registration.clause.eval_seeds", "the registration has no eval_seeds",
                      "rerun preregister")
    return seeds


def _score_arm(ctx: StageContext, sealed: SealedRef, model: dict, full: dict) -> list[dict]:
    fn = GENERATORS.get(model.get("generator"))
    if fn is None:
        raise _refuse("AN-8", "model_ref.generator", f"no generation function is registered for {model.get('generator')!r}",
                      "register the scenario pack that serves this model reference")
    sampler = full["sampler"]
    records: list[dict] = []
    for seed in _eval_seeds(full):
        counter = [0]

        def generate(item: dict, seed=seed) -> dict:
            index = counter[0]
            counter[0] += 1
            try:
                prediction = fn(dict(model), {"input": item["input"]}, sampler=dict(sampler), seed=seed)
            except Exception as exc:       # never carry the item (or the pack's message about it) out
                raise _refuse("AN-8", "generate", f"the generation function failed ({type(exc).__name__})",
                              "fix the scenario pack's generate") from None
            return {"item_id": index, "seed": seed, "score": scorers.exact_match(prediction, item["label"])}

        records.extend(ctx.trust.run_sealed(sealed, generate))
    return records


class EvaluateSealed:
    """AN-8: scores the sealed items with exactly the registered sampler and eval_seeds and exact match only. What
    leaves the trust backend is one `{item_id, seed, score}` record per item and seed (`item_id` is an index the
    backend assigns, not a content hash). In step 1 the operator process can still read the sealed items; the
    guarantee is stated (`sealed_guarantee`), not enforced."""

    name = "evaluate_sealed"
    version = "1"
    inputs = {"registration": SCHEMA_REGISTRATION, "baseline_model": SCHEMA_MODEL_REF,
              "candidate_model": SCHEMA_MODEL_REF}
    outputs = {"baseline_records": SCHEMA_SEALED_RECORDS, "candidate_records": SCHEMA_SEALED_RECORDS}
    needs: frozenset[str] = frozenset()
    params: frozenset[str] = frozenset()
    trains = False

    def plan(self, ctx: StageContext) -> Proposal:
        _, full = _registered(ctx, "AN-8")
        n_seeds = len(_eval_seeds(full))
        return Proposal(self.name, f"score the sealed items for both models with the registered sampler on "
                        f"{n_seeds} evaluation seeds; only per-item scores leave the trust backend", {}, {})

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        reg, full = _registered(ctx, "AN-8")
        problem = _evaluator_problem(full)
        if problem:
            raise _refuse("AN-8", "registration.evaluators[0].manifest_hash", problem,
                          "register again with preregister, or restore the scorer")
        seeds = _eval_seeds(full)
        item_set = full["eval_distribution"]["item_set_hash"]
        sealed = SealedRef(item_set)
        out: dict[str, ArtifactRef] = {}
        for arm in ("baseline", "candidate"):
            ref = ctx.inputs[f"{arm}_model"]
            model = ctx.project.read_artifact(ref)
            if not isinstance(model, dict):
                raise _refuse("AN-8", f"inputs.{arm}_model", "the model reference is malformed", "rerun its stage")
            records = _score_arm(ctx, sealed, model, full)
            doc = {**_meta(SCHEMA_SEALED_RECORDS), "arm": arm, "registration_hash": reg["hash"],
                   "item_set_hash": item_set, "n_items": len(records) // len(seeds), "eval_seeds": list(seeds),
                   "sampler": dict(full["sampler"]), "model_ref_sha256": ref.sha256, "records": records}
            out[f"{arm}_records"] = ctx.project.write_artifact(self.name, f"{arm}_records", doc,
                                                                SCHEMA_SEALED_RECORDS)
        return out

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        problems: list[Problem] = []

        def bad(field: str, message: str) -> None:
            problems.append(Problem("AN-8", field, message, "rerun evaluate_sealed"))

        counts = set()
        for arm in ("baseline", "candidate"):
            rec = ctx.project.read_artifact(outputs[f"{arm}_records"])
            rows = rec.get("records") if isinstance(rec, dict) else None
            if not isinstance(rows, list):
                bad(f"{arm}_records", "the record has no list of records")
                continue
            if any(not isinstance(r, dict) or set(r) != {"item_id", "seed", "score"} for r in rows):
                bad(f"{arm}_records.records", "a record has fields other than item_id, seed and score")
            if len(rows) != rec.get("n_items", -1) * len(rec.get("eval_seeds", [])):
                bad(f"{arm}_records.n_items", "the record count is not n_items times the number of seeds")
            counts.add(rec.get("n_items"))
        if len(counts) > 1:
            bad("n_items", "the two arms have different numbers of items")
        return problems


def _attempt_key(project: Project, stage: str) -> str:
    """The key of the stage's current attempt: the last `start` row the machine wrote for it."""
    key = None
    for row in project.read_attempts():
        if row.get("stage") == stage and row.get("event") == "start":
            key = row.get("key")
    if not isinstance(key, str) or not key:
        raise _refuse("AN-9", "gate", "the gate has no attempt key in the attempt log",
                      "run the gate through the pipeline (run_stage), which records the attempt")
    return key


def _arm(project: Project, ref: ArtifactRef, reg: dict, seeds: list) -> dict:
    rec = project.read_artifact(ref)
    if not isinstance(rec, dict) or not isinstance(rec.get("records"), list):
        raise _refuse("AN-9", f"inputs.{ref.name}", "the sealed records are malformed", "rerun evaluate_sealed")
    if rec.get("item_set_hash") != reg["item_set_hash"] or rec.get("registration_hash") != reg["hash"]:
        raise _refuse("AN-9", f"inputs.{ref.name}", "the records were not made for the registered item set",
                      "rerun evaluate_sealed on the registered sealed set")
    if rec.get("eval_seeds") != seeds:
        raise _refuse("AN-9", f"inputs.{ref.name}", "the records use other evaluation seeds than the registration",
                      "rerun evaluate_sealed")
    rows = [{"item": r["item_id"], "repeat": seeds.index(r["seed"]) if r.get("seed") in seeds else -1,
             "seed": r.get("seed"), "score": r["score"]} for r in rec["records"] if isinstance(r, dict)
            and "item_id" in r and "score" in r]
    return {"records": rows, "model_manifest_hash": rec.get("model_ref_sha256"), "n_items": rec.get("n_items")}


class Gate:
    """AN-9: one test on the sealed records at the alpha the ledger grants. The alpha is spent before the statistics
    run (a second gate attempt refuses) and the outcome is recorded after."""

    name = "gate"
    version = "1"
    inputs = {"registration": SCHEMA_REGISTRATION, "baseline_records": SCHEMA_SEALED_RECORDS,
              "candidate_records": SCHEMA_SEALED_RECORDS}
    outputs = {"gate_result": SCHEMA_GATE_RESULT}
    needs: frozenset[str] = frozenset()
    params: frozenset[str] = frozenset()
    trains = False

    def plan(self, ctx: StageContext) -> Proposal:
        reg, _ = _registered(ctx, "AN-9")
        return Proposal(self.name, f"test the new model against the base model on the sealed records at alpha "
                        f"{ctx.ledger.available():g}; this spends the alpha budget", {"registration": reg["hash"]}, {})

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        available = ctx.ledger.available()
        if not (_num(available) and available > 0):
            raise _refuse("AN-9", "gate", "the alpha budget is spent: step 1 allows one gate test per project",
                          "start a new project for another test")
        reg, full = _registered(ctx, "AN-9")
        clause = dict(full["clause"])
        if clause.get("direction") != reg["direction"] or clause.get("margin") != reg["margin"]:
            raise _refuse("AN-9", "registration.clause", "the registration and its output record disagree on the "
                          "direction or the margin", "rerun preregister")
        seeds = _eval_seeds(full)
        base = _arm(ctx.project, ctx.inputs["baseline_records"], reg, seeds)
        cand = _arm(ctx.project, ctx.inputs["candidate_records"], reg, seeds)
        if base["n_items"] != cand["n_items"] or not isinstance(base["n_items"], int) or base["n_items"] < 1:
            raise _refuse("AN-9", "inputs", "the two arms do not score the same items", "rerun evaluate_sealed")
        attempt_id = _attempt_key(ctx.project, self.name)
        alpha = ctx.ledger.spend(attempt_id)                      # the ledger decides alpha, not the registration
        clause["alpha"] = alpha
        boot = full["bootstrap"]
        try:
            result = gates.evaluate_items(clause, base, cand, items=list(range(base["n_items"])),
                                          seed=int(boot["seed"]), n_boot=int(boot["replicates"]),
                                          min_paired=min_paired(full), max_unpaired=max_unpaired(full),
                                          score_range=(0.0, 1.0))
        except Exception:
            ctx.ledger.record(attempt_id, False)
            raise
        if reg["post_hoc"] is True:
            result = gates.apply_post_hoc(result)
        ctx.ledger.record(attempt_id, result["status"] == gates.PASS)
        doc = {**_meta(SCHEMA_GATE_RESULT), **result, "alpha": alpha, "attempt_id": attempt_id,
               "registration_hash": reg["hash"], "post_hoc": bool(reg["post_hoc"])}
        return {"gate_result": ctx.project.write_artifact(self.name, "gate_result", doc, SCHEMA_GATE_RESULT)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        res = ctx.project.read_artifact(outputs["gate_result"])
        reg = ctx.project.read_artifact(ctx.inputs["registration"])
        problems: list[Problem] = []

        def bad(field: str, message: str) -> None:
            problems.append(Problem("AN-9", field, message, "rerun gate"))

        if res.get("registration_hash") != reg.get("hash"):
            bad("gate_result.registration_hash", "the result is for another registration")
        if res.get("status") not in (gates.PASS, gates.FAIL, gates.INCONCLUSIVE, gates.MISSING):
            bad("gate_result.status", "the status is not PASS, FAIL, INCONCLUSIVE or MISSING")
        if reg.get("post_hoc") is True and res.get("status") == gates.PASS:
            bad("gate_result.status", "a post hoc registration cannot PASS")
        entry = [e for e in ctx.ledger.entries() if e["attempt_id"] == res.get("attempt_id")]
        if len(entry) != 1 or entry[0]["alpha"] != res.get("alpha") or \
                entry[0]["passed"] is not (res.get("status") == gates.PASS):
            bad("gate_result.attempt_id", "the ledger does not show this attempt with this alpha and outcome")
        return problems


METRIC_UNITS = {"exact_match": "share of items answered exactly (0 to 1)"}
REPORT_MD = "report.md"


def _round(x: Any) -> Any:
    return round(x, 4) if isinstance(x, float) else x


def _effective_status(status: str, post_hoc: bool) -> str:
    return "INCONCLUSIVE" if post_hoc and status == gates.PASS else status


def _headline(status: str, post_hoc: bool) -> str:
    status = _effective_status(status, post_hoc)
    if post_hoc:
        return f"# {status}: post hoc, not a confirmed gain"
    said = {gates.PASS: "the new model beats the base model by more than the registered margin",
            gates.FAIL: "the new model does not beat the base model by the registered margin",
            gates.INCONCLUSIVE: "the sealed test cannot tell whether the new model is better",
            gates.MISSING: "the sealed records are incomplete, so no verdict"}
    return f"# {status}: {said.get(status, 'no verdict')}"


def render_markdown(report: dict) -> str:
    """`report.md`, rendered from `report.json` alone: every number and name in it is read from `report`."""
    reg, gate, power, shown = report["registration"], report["gate_result"], report["power"], report["display"]
    status = _effective_status(report["status"], reg["post_hoc"])
    lines = [_headline(report["status"], reg["post_hoc"]), ""]
    if reg["post_hoc"]:
        lines += ["The registration was amended after training started, so this result is post hoc and is not a "
                  "confirmed gain.", ""]
    lines += [f"Status: {status}", f"Metric: {reg['scorer']}, {report['metric']['unit']}",
              f"Sealed items: n_sealed = {report['n_sealed']}", "",
              "## Effect",
              f"- effect (new minus base): {shown['effect']}",
              f"- interval: {shown['ci_lower']} to {shown['ci_upper']} (level {shown['ci_level']})",
              f"- base model score: {shown['baseline_score']}",
              f"- new model score: {shown['candidate_score']}",
              f"- registered margin: {reg['margin']}",
              f"- alpha used: {gate['alpha']}",
              f"- items compared: {gate['n_items']}",
              f"- status reasons: {', '.join(gate['status_reasons']) or 'none'}", "",
              "## Registration",
              f"- registration: {reg['hash']}",
              f"- sealed set: {reg['item_set_hash']}",
              f"- direction: {reg['direction']}",
              f"- post_hoc: {'yes' if reg['post_hoc'] else 'no'}"]
    if reg["amends"]:
        lines.append(f"- amends: {reg['amends']}")
    lines += ["", "## Power",
              f"- sealed items planned: {power['n_sealed']}, needed: {power['n_needed']}",
              f"- smallest gain the sealed set can detect: {shown['mde_planned']}", "",
              "## Gate attempts"]
    for e in report["ledger"]:
        outcome = {True: "passed", False: "did not pass", None: "no outcome recorded"}[e["passed"]]
        lines.append(f"- attempt {e['attempt_id']}: alpha {e['alpha']}, {outcome}")
    for a in report["attempts"]:
        lines.append(f"- log: attempt {a['attempt']} {a['event']} at {a['utc']}")
    lines += ["", "## Known limits", report["sealed_guarantee"], ""]
    return "\n".join(lines)


class Report:
    """AN-10: `report.json` is built from records only (the registration, the power plan, the gate result, the ledger,
    the attempt log and the backend's `sealed_guarantee`); `report.md` is rendered from `report.json` and nothing
    else. No text is generated. The markdown file sits next to the JSON record (`stages/report/report.md`); the JSON
    record is the declared output the machine tracks, `check` re-renders the markdown and refuses a difference."""

    name = "report"
    version = "1"
    inputs = {"registration": SCHEMA_REGISTRATION, "power": SCHEMA_POWER, "gate_result": SCHEMA_GATE_RESULT}
    outputs = {"report": SCHEMA_REPORT}
    needs: frozenset[str] = frozenset()
    params: frozenset[str] = frozenset()
    trains = False

    def plan(self, ctx: StageContext) -> Proposal:
        return Proposal(self.name, "write report.json from the records and render report.md from it", {}, {})

    @staticmethod
    def _build(ctx: StageContext) -> dict:
        reg = ctx.project.read_artifact(ctx.inputs["registration"])
        power = ctx.project.read_artifact(ctx.inputs["power"])
        gate = ctx.project.read_artifact(ctx.inputs["gate_result"])
        keep = {"schema", "schema_version"}
        attempts = [{"attempt": r.get("attempt"), "event": r.get("event"), "utc": r.get("utc"), "key": r.get("key")}
                    for r in ctx.project.read_attempts() if r.get("stage") == "gate"]
        return {**_meta(SCHEMA_REPORT),
                "status": _effective_status(gate["status"], bool(reg["post_hoc"])),
                "metric": {"scorer": reg["scorer"], "unit": METRIC_UNITS[reg["scorer"]]},
                "n_sealed": power["n_sealed"],
                "registration": {k: reg.get(k) for k in _REG_KEYS},
                "power": {k: v for k, v in power.items() if k not in keep},
                "gate_result": {k: v for k, v in gate.items() if k not in keep},
                "display": {"effect": _round(gate["delta_hat_natural"]), "ci_lower": _round(gate["ci_lower_natural"]),
                            "ci_upper": _round(gate["ci_upper_natural"]), "ci_level": _round(gate["ci_level"]),
                            "baseline_score": _round(gate["baseline_score"]),
                            "candidate_score": _round(gate["candidate_score"]), "mde_planned": _round(power["mde"])},
                "ledger": ctx.ledger.entries(), "attempts": attempts,
                "sealed_guarantee": ctx.trust.sealed_guarantee}

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        report = self._build(ctx)
        ref = ctx.project.write_artifact(self.name, "report", report, SCHEMA_REPORT)
        rel = f"stages/{self.name}/{REPORT_MD}"
        _refuse_links(ctx.project.root, rel)
        _write_atomic(ctx.project.root / rel, render_markdown(ctx.project.read_artifact(ref)).encode("utf-8"))
        return {"report": ref}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        report = ctx.project.read_artifact(outputs["report"])
        rel = f"stages/{self.name}/{REPORT_MD}"
        _refuse_links(ctx.project.root, rel)
        try:
            text = (ctx.project.root / rel).read_text(encoding="utf-8")
        except OSError:
            return [Problem("AN-10", rel, "report.md is missing", "rerun report")]
        problems: list[Problem] = []
        if text != render_markdown(report):
            problems.append(Problem("AN-10", rel, "report.md is not the rendering of report.json",
                                    "rerun report; do not edit report.md"))
        gate = ctx.project.read_artifact(ctx.inputs["gate_result"])
        reg = ctx.project.read_artifact(ctx.inputs["registration"])
        if report.get("status") != _effective_status(gate.get("status"), bool(reg.get("post_hoc"))):
            problems.append(Problem("AN-10", "report.status", "the report status differs from the gate result",
                                    "rerun report"))
        if report.get("sealed_guarantee") != ctx.trust.sealed_guarantee:
            problems.append(Problem("AN-10", "report.sealed_guarantee",
                                    "the report does not state the trust backend's guarantee", "rerun report"))
        return problems
