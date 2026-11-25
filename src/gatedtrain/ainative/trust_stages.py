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
`evaluate_sealed` must use exactly the registered sampler and eval_seeds and score only with the registered scorer
(`scorers`: a library scorer or a scenario pack's scorer by reference, AN-17); the registered evaluator is
`scorers.evaluator_entry` of the scorer and its parameters, with the hash of the scorer's code.

Both stages take no pipeline params: their behaviour comes from the project's data, the user's decision and the
registration, never from pipeline configuration.

The split, the scoring, the gate statistics and the shared report pieces are in `sealed_eval`, which the keeper
calls too; the stages here add the project, the trust backend and the ledger around them.

Keeper mode (`ctx.trust.mode == "keeper"`, AN-8 to AN-10, AN-14): each stage becomes a request to the keeper and
writes the same schema ids at `schema_version` 2. `intake_seal` sends the data file and writes the training part,
sealed summary and power record the keeper returned (the user's `accept_inconclusive` decision goes with it);
`preregister` registers at the fixed planning alpha with the alpha rule named in `signed_by` (the test level of each
attempt is set by alpha-investing; a registration takes no new key) and adds the registration's `envelope_sha256`;
`evaluate_sealed` asks the keeper for a sealed attempt and writes per-arm receipts (no records: they stay in the
keeper); `gate` copies the keeper's signed gate summary (status and alpha only) with its `envelope_sha256`; `report`
verifies every receipt, copies the keeper's signed report and renders `report.md` from it. The in-process path is
unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

from .. import ANALYSIS_CODE_VERSION, gates
from ..canonical import sha256_bytes
from ..errors import Problem, Refused
from ..registration import check_contents, resolved_evaluator_hash
from . import alpha_investing, image_provenance, scorers, sealed_eval, signing
from .batch import BatchGenerator
from .keeper import modes, texts as keeper_texts
from .project import ArtifactRef, Project, _refuse_links, _utc_now, _write_atomic
from .stage import Proposal, StageContext
from .trust import NEW_TEST_FIX, SealedRef
# The split, scoring, gate statistics and report pieces are shared with the keeper (`sealed_eval`); the step 1
# names stay importable from here.
from .sealed_eval import (DEFAULT_ALPHA, DEFAULT_MDE, DEFAULT_POWER, LIMIT_ALPHA_PER_PROJECT,  # noqa: F401
                          LIMIT_METRIC_CONFIRMATION, LIMIT_MODEL_REFERENCE, MAX_SEALED_FRACTION, PILOT_SIGMA,
                          PILOT_SOURCE, _num, display_block, known_limits)
from .sealed_eval import effective_status as _effective_status, eval_seeds as _eval_seeds, headline as _headline

SCHEMA_VERSION = 1
KEEPER_REPORT_VERSIONS = sealed_eval.REPORT_SCHEMA_VERSIONS   # a version 2 report still checks (see `sealed_eval`)
KEEPER_SCHEMA_VERSION = sealed_eval.KEEPER_SCHEMA_VERSION
# The registration field that names the alpha rule in keeper mode: `signed_by` names the signer,
# which under the keeper is the keeper, and the rule it sets each attempt's test level by.
KEEPER_SIGNED_BY = f"gatedtrain-keeper (alpha rule: {alpha_investing.RULE})"
RECEIPT_KEYS = frozenset({"schema", "schema_version", "arm", "held_by", "attempt_id", "registration_hash",
                          "item_set_hash", "model_ref_sha256", "claim_sha256"})
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
SEALED_TEST_USED = "this project's sealed test is used; testing another candidate needs new sealed data"


def _refuse(rule: str, field: str, message: str, fix: str) -> Refused:
    return Refused([Problem(rule, field, message, fix)])


def _meta(schema: str, version: int = SCHEMA_VERSION) -> dict:
    return {"schema": schema, "schema_version": version}


def _keeper(ctx: StageContext) -> bool:
    return getattr(ctx.trust, "mode", None) == "keeper"


# ---------------------------------------------------------------------------
# intake_seal
# ---------------------------------------------------------------------------

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


def _split(project: Project, planning_alpha: float = DEFAULT_ALPHA) -> dict:
    """`sealed_eval.split_data` of the project's data file."""
    raw, digest = _read_data(project)
    return sealed_eval.split_data(raw, digest, planning_alpha=planning_alpha)


class IntakeSeal:
    name = "intake_seal"
    version = "1"
    inputs: dict[str, str] = {}
    outputs = {"train_part": SCHEMA_TRAIN_PART, "sealed_summary": SCHEMA_SEALED_SUMMARY, "power": SCHEMA_POWER}
    needs: frozenset[str] = frozenset()
    params: frozenset[str] = frozenset()
    trains = False

    def plan(self, ctx: StageContext) -> Proposal:
        planning_alpha = ctx.ledger.planning_alpha
        s = _split(ctx.project, planning_alpha)
        if s["conclusive"]:
            summary = (f"{s['n_unique']} distinct labelled examples: {s['n_sealed']} will be sealed for the final "
                       f"test and {len(s['train'])} are for training; the test can detect a gain of "
                       f"{s['mde']:.2g} or more.")
            if _keeper(ctx) and getattr(ctx.trust, "keeper_mode", None) == modes.SAME_USER:
                summary += " " + keeper_texts.PLAN_SEALED_SAME_USER       # AN-27: never "sealed" without the qualifier
        else:
            summary = (f"{s['n_unique']} distinct labelled examples are too few: the final test will most likely be "
                       f"INCONCLUSIVE (it needs {s['n_needed']} sealed examples to detect a gain of "
                       f"{DEFAULT_MDE:g}); about {s['more']} more labelled examples would make it conclusive.")
        estimate = {"n_needed": s["n_needed"], "n_sealed": s["n_sealed"], "mde": s["mde"],
                    "conclusive_expected": s["conclusive"], "n_more_examples": s["more"]}
        choices = {"n_unique": s["n_unique"], "n_duplicates_removed": s["n_removed"], "n_train": len(s["train"]),
                   "n_needed": s["n_needed"], "mde": DEFAULT_MDE, "alpha": planning_alpha, "power": DEFAULT_POWER}
        return Proposal(self.name, summary, choices, estimate, None if s["conclusive"] else "data_too_small")

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        if _keeper(ctx):
            return self._run_keeper(ctx, decision)
        planning_alpha = ctx.ledger.planning_alpha
        s = _split(ctx.project, planning_alpha)
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
        power = {**_meta(SCHEMA_POWER), **sealed_eval.power_record(s, planning_alpha=planning_alpha)}
        return {"train_part": project.write_artifact(ctx.stage or self.name, "train_part", train, SCHEMA_TRAIN_PART),
                "sealed_summary": project.write_artifact(ctx.stage or self.name, "sealed_summary", sealed, SCHEMA_SEALED_SUMMARY),
                "power": project.write_artifact(ctx.stage or self.name, "power", power, SCHEMA_POWER)}

    def _run_keeper(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        """The keeper splits and seals (AN-6, AN-16); the records are the ones it returned. Too small data are refused
        by the keeper unless the user accepted an INCONCLUSIVE test."""
        raw, digest = _read_data(ctx.project)
        res = ctx.trust.intake(raw, digest, accept_inconclusive=decision.get("accept_inconclusive") is True)
        rows = res.get("train_rows") if isinstance(res, dict) else None
        summary, power = (res.get("sealed_summary"), res.get("power")) if isinstance(res, dict) else (None, None)
        if not (isinstance(rows, list) and isinstance(summary, dict) and isinstance(power, dict)
                and summary.get("schema") == SCHEMA_SEALED_SUMMARY and power.get("schema") == SCHEMA_POWER
                and summary.get("schema_version") == power.get("schema_version") == KEEPER_SCHEMA_VERSION
                and summary.get("manifest_sha256") == res.get("sealed_id")):
            raise _refuse("AN-6", "intake", "the keeper's intake result is malformed", "check the keeper")
        _, removed = sealed_eval.dedup(sealed_eval.parse_rows(raw))
        train = {**_meta(SCHEMA_TRAIN_PART, KEEPER_SCHEMA_VERSION), "n": len(rows), "n_duplicates_removed": removed,
                 "data_sha256": digest, "rows": rows}
        stage = ctx.stage or self.name
        return {"train_part": ctx.project.write_artifact(stage, "train_part", train, SCHEMA_TRAIN_PART),
                "sealed_summary": ctx.project.write_artifact(stage, "sealed_summary", summary, SCHEMA_SEALED_SUMMARY),
                "power": ctx.project.write_artifact(stage, "power", power, SCHEMA_POWER)}

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
        if _keeper(ctx) and ({r.get("schema_version") for r in (train, sealed, power)} != {KEEPER_SCHEMA_VERSION}
                             or power.get("alpha") != ctx.ledger.planning_alpha
                             or sealed.get("guarantee") != ctx.trust.sealed_guarantee):
            bad("outputs", "the records are not the keeper's intake at the planning alpha with its guarantee")
        return problems


# ---------------------------------------------------------------------------
# preregister
# ---------------------------------------------------------------------------

# The scorers a registration may name: the library of `scorers` and the scorers of the installed scenario packs
# (AN-17). Each names the evaluator id that registers it, its direction and planning SD; the evaluator's manifest
# hash comes from the scorer's code and parameters (`scorers.evaluator_entry`). The training reward may differ.
DEFAULT_SCORER = "exact_match"
MAX_MARGIN = 0.5


def _refuse7(field: str, message: str, fix: str) -> Refused:
    return _refuse("AN-7", field, message, fix)


def _scorer_names() -> list[str]:
    return sorted({*scorers.library(), *scenario_scorers()})


def _available_scorers() -> dict[str, str]:
    """What each scorer a registration may name counts (pack scorers resolved, so a bad declaration is refused)."""
    return {name: scorers.resolve(name).what for name in _scorer_names()}


def _scorer(decision: dict) -> tuple[scorers.ScorerSpec, dict]:
    """(spec, bound parameters) of the decision's `scorer` and `scorer_params`."""
    name = decision.get("scorer", DEFAULT_SCORER)
    if not isinstance(name, str) or name not in _scorer_names():
        raise _refuse7("decision.scorer", f"{name!r} is not in the scorer library",
                       f"use one of: {', '.join(_scorer_names())}")
    spec = scorers.resolve(name)
    entry = scorers.evaluator_entry(spec, decision.get("scorer_params", {}))
    return spec, entry.get("params", {})


def _metric(decision: dict) -> dict:
    """The scorer, direction and margin of a preregister decision, checked (AN-7)."""
    spec, _ = _scorer(decision)
    scorer = spec.name
    direction = decision.get("direction", spec.direction)
    if direction != spec.direction:
        raise _refuse7("decision.direction", f"scorer {scorer} is {spec.direction}, not {direction!r}",
                       f"use direction {spec.direction}, or pick another scorer")
    margin = decision.get("margin", 0.0)
    if not _num(margin) or not 0 <= margin <= MAX_MARGIN:
        raise _refuse7("decision.margin", f"margin {margin!r} is not a number from 0 to {MAX_MARGIN}",
                       "give the smallest gain that counts, as a fraction of the score (0.05 is 5 points)")
    return {"scorer": scorer, "direction": direction, "margin": float(margin)}


def keeper_confirmation(decision: dict, n_sealed: int) -> tuple[dict, str]:
    """AN-18: what `gatedtrain keeper confirm` asks the keeper to record for a preregister decision (its `scorer`,
    `scorer_params` and `margin`; the direction is the scorer's) and the plan sentence it shows first, the one
    `preregister` states in keeper mode. The keeper matches a registration against exactly these values."""
    v = _metric(decision)
    _, params = _scorer(decision)
    sentence = _sentence({**v, "alpha": alpha_investing.PLANNING_ALPHA}, n_sealed, keeper=True)
    return {"scorer": v["scorer"], "scorer_params": params, "direction": v["direction"], "margin": v["margin"]}, \
        sentence


def _values(ctx: StageContext, decision: dict) -> dict:
    """The scorer, direction, margin and alpha the registration will carry."""
    v = _metric(decision)
    # Under the keeper the registration carries the fixed planning alpha: the test level of each attempt is set by
    # alpha-investing at the gate, so a rerun after an attempt registers the same content.
    alpha = ctx.ledger.planning_alpha if _keeper(ctx) else ctx.ledger.available()
    if not _num(alpha) or not 0 < alpha < 0.5:
        raise _refuse7("alpha", f"no alpha is available to register (available: {alpha!r})",
                       NEW_TEST_FIX)
    return {**v, "alpha": float(alpha)}


def _sealed_summary(ctx: StageContext) -> dict:
    summary = ctx.project.read_artifact(ctx.inputs["sealed_summary"])
    if not isinstance(summary, dict) or "manifest_sha256" not in summary:
        raise _refuse7("inputs.sealed_summary", "the sealed summary is malformed", "rerun intake_seal")
    return summary


def _intake_sigma(ctx: StageContext) -> float:
    """The planning SD the sealed set was sized for: `sigma` of the power record written with the `sealed_summary`
    input (found through the attempt log, so `preregister` keeps its step 1 inputs)."""
    ref = ctx.inputs["sealed_summary"]
    power = None
    for row in ctx.project.read_attempts():
        outs = row.get("outputs") if row.get("event") == "done" and row.get("stage") == ref.stage else None
        got = (outs or {}).get("sealed_summary")
        if isinstance(got, dict) and got.get("sha256") == ref.sha256 and isinstance(outs.get("power"), dict):
            power = outs["power"]
    sigma = None
    if power is not None:
        rec = ctx.project.read_artifact(ArtifactRef(power["stage"], power["name"], power["schema"], power["sha256"],
                                                    power["path"]))
        sigma = rec.get("sigma") if isinstance(rec, dict) else None
    if not (_num(sigma) and math.isfinite(sigma) and sigma > 0):
        raise _refuse7("inputs.sealed_summary", "the power record written with the sealed summary is not found, so "
                       "the planning SD the sealed set was sized for is unknown", "rerun intake_seal")
    return float(sigma)


def _sentence(v: dict, n_sealed: int, keeper: bool = False) -> str:
    points = f"{v['margin'] * 100:g}"
    if keeper:
        return (f"I will count {scorers.resolve(v['scorer']).what}; the new model must beat the base model by more "
                f"than {points} points on {n_sealed} sealed items; the test level of each attempt is set by "
                f"alpha-investing ({alpha_investing.GUARANTEE}), {v['alpha']:g} for the first attempt.")
    return (f"I will count {scorers.resolve(v['scorer']).what}; the new model must beat the base model by more than "
            f"{points} points, tested at the {v['alpha']:g} significance level on {n_sealed} sealed items.")


def _deployment(ctx: StageContext) -> dict:
    dep = ctx.project.read_artifact(ctx.inputs["deployment"])
    seeds = dep.get("eval_seeds") if isinstance(dep, dict) else None
    if not (isinstance(dep, dict) and isinstance(dep.get("sampler"), dict) and isinstance(seeds, list) and seeds):
        raise _refuse7("inputs.deployment", "the deployment record has no sampler mapping and non-empty eval_seeds",
                       "rerun the stage that produces the deployment record")
    return {"sampler": dep["sampler"], "eval_seeds": seeds}


def registered_base(full: dict) -> str | None:
    """The `sha256:<hex>` of the base model reference a registration binds (the `base_model` field)."""
    bm = full.get("base_model")
    h = bm.get("model_ref_sha256") if isinstance(bm, dict) else None
    return h if isinstance(h, str) else None


def _registration(v: dict, item_set_hash: str, dep: dict, base_sha: str, served: tuple[str, str] | None = None,
                  evaluator: dict | None = None, signed_by: str = "prototype-unsigned") -> dict:
    """`evaluator` is the `scorers.evaluator_entry` of the scorer; absent, exact match without parameters.
    `signed_by` is `KEEPER_SIGNED_BY` in keeper mode."""
    if evaluator is None:
        evaluator = scorers.evaluator_entry(scorers.library()[DEFAULT_SCORER], {})
    return {
        "registration_version": 1,
        "registration_id": f"ainative-{item_set_hash.removeprefix('sha256:')[:16]}",
        "signed_by": signed_by,
        "workflow": "ainative",
        "gate": "quality_gate",
        "clause": {"type": "superiority", "metric": v["scorer"], "direction": v["direction"], "margin": v["margin"],
                   "alpha": v["alpha"], "power": DEFAULT_POWER, "planning_effect": v["margin"] + DEFAULT_MDE,
                   "sigma_plan": "fixed", "unit": "item", "samples_per_item": len(dep["eval_seeds"]),
                   "eval_seeds": list(dep["eval_seeds"]),
                   "interval": "cluster_bootstrap", "blocking": True},
        "margin_justification": f"Margin {v['margin']:g} (fraction of the {v['scorer']} score) as confirmed by the "
                                "user before any training.",
        "sigma_pilot": {"sigma": PILOT_SIGMA, "source": PILOT_SOURCE},
        "eval_distribution": {"item_set_hash": item_set_hash, "env_version": None, "env_params_version": None},
        "sampler": dict(dep["sampler"]),
        "base_model": {"model_ref_sha256": base_sha,
                       **({"scenario_pack": served[0], "generator": served[1]} if served else {})},
        "evaluators": [dict(evaluator)],
        "multiplicity": {"plan": "single_gate", "family": None},
        "analysis_code_version": ANALYSIS_CODE_VERSION,
        "bootstrap": {"replicates": 10000, "seed": int(item_set_hash.removeprefix("sha256:")[:8], 16)},
    }


_AMEND_KEYS = ("amends", "created_at", "post_hoc")


def _training_ran(project: Project) -> bool:
    """True when the attempt log shows a stage that trains starting after the first `preregister` run was done.
    Reads only the log (`start` rows carry `trains`, resolved at run time), never the editable pipeline file. It
    fails closed: a `start` row that does not record `trains: false` counts as training, so a later change of the
    registration is an amendment (post hoc)."""
    done = False
    for row in project.read_attempts():
        if row.get("event") == "done" and row.get("stage_type") == "preregister":
            done = True
        elif done and row.get("event") == "start" and row.get("trains") is not False:
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
    inputs = {"sealed_summary": SCHEMA_SEALED_SUMMARY, "deployment": SCHEMA_DEPLOYMENT,
              "base_model": SCHEMA_MODEL_REF}
    outputs = {"registration": SCHEMA_REGISTRATION}
    needs = frozenset({"user"})
    params: frozenset[str] = frozenset()
    trains = False

    def plan(self, ctx: StageContext) -> Proposal:
        v = _values(ctx, {})
        n = _sealed_summary(ctx)["n"]
        choices = {**v, "available_scorers": _available_scorers(), "n_sealed": n}
        return Proposal(self.name, _sentence(v, n, _keeper(ctx)), choices, {}, "confirm_metric")

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        if decision.get("metric_confirmed") is not True:
            raise _refuse7("decision.metric_confirmed",
                           "the user has not confirmed the deciding metric, direction and margin; nothing trains "
                           "before that",
                           "show the plan() summary to the user and set decision metric_confirmed to true once "
                           "they agree (they may change scorer or margin first)")
        v = _values(ctx, decision)
        spec, params = _scorer(decision)
        sigma = _intake_sigma(ctx)
        if spec.planning_sd > sigma:
            raise _refuse("AN-17", "decision.scorer",
                          f"scorer {spec.name} has a planning SD of {spec.planning_sd:g}, larger than the sealed set "
                          "was sized for, so the test would be underpowered",
                          f"the sealed set was sized for a planning SD of {sigma}; pick a library scorer or run "
                          "intake again with new data")
        item_hash = _sealed_summary(ctx)["manifest_sha256"]
        base_ref = ctx.project.read_artifact(ctx.inputs["base_model"])
        gen = base_ref.get("generator") if isinstance(base_ref, dict) else None
        if not isinstance(gen, str) or not gen:
            # evaluate_sealed resolves both arms through the registered generator; refusing only there would come
            # after training, so a base model reference without one is refused here, before anything trains.
            raise _refuse7("inputs.base_model.generator", "the base model reference names no `generator`, so the "
                           "sealed evaluation could not run the models",
                           "have the stage that produces the base model write a model reference whose `generator` "
                           "an installed scenario pack serves")
        pack = scenario_pack_of(gen)
        if pack is None:
            raise _refuse7("inputs.base_model.generator", f"no installed scenario pack serves the generator {gen!r}",
                           "install the scenario pack that serves the base model, or fix the model reference")
        base = _registration(v, item_hash, _deployment(ctx), ctx.inputs["base_model"].sha256, (pack, gen),
                             scorers.evaluator_entry(spec, params),
                             KEEPER_SIGNED_BY if _keeper(ctx) else "prototype-unsigned")
        problems = check_contents(base, greedy_allowed=True)
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
                problems = check_contents(final, greedy_allowed=True)
                if problems:
                    raise Refused(problems)
        digest = ctx.trust.register(final)
        out = {**_meta(SCHEMA_REGISTRATION), "hash": digest, "scorer": v["scorer"], "direction": v["direction"],
               "margin": v["margin"], "alpha": v["alpha"], "item_set_hash": item_hash, "post_hoc": post_hoc,
               "amends": amends}
        if _keeper(ctx):                     # the keeper signed it (AN-14); the project keeps the envelope's hash
            out = {**out, "schema_version": KEEPER_SCHEMA_VERSION,
                   "envelope_sha256": signing.envelope_sha256(ctx.trust.registration_envelope(digest))}
        return {"registration": ctx.project.write_artifact(ctx.stage or self.name, "registration", out, SCHEMA_REGISTRATION)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        out = ctx.project.read_artifact(outputs["registration"])
        full = ctx.trust.registration(out["hash"])
        problems = list(check_contents(full, greedy_allowed=True))

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
        found = scorers.spec_for_evaluator(ev)
        if found is None or found[0].name != out.get("scorer") or \
                ev.get("manifest_hash") != resolved_evaluator_hash(ev):
            bad("registration.evaluators[0].manifest_hash",
                "the registered evaluator hash is not the one the library resolves for the scorer")
        if registered_base(full) != ctx.inputs["base_model"].sha256:
            bad("registration.base_model", "the registered base model is not the base model input")
        if bool(full.get("post_hoc", False)) != out.get("post_hoc"):
            bad("registration.post_hoc", "the output record and the registration disagree on post_hoc")
        if _keeper(ctx) and (out.get("schema_version") != KEEPER_SCHEMA_VERSION or out.get("envelope_sha256") !=
                             signing.envelope_sha256(ctx.trust.registration_envelope(out["hash"]))):
            bad("registration.envelope_sha256", "the record does not name the keeper's signed registration")
        return problems


# ---------------------------------------------------------------------------
# evaluate_sealed, gate, report (AN-8, AN-9, AN-10)
# ---------------------------------------------------------------------------

# The generation functions of the scenario packs, by the name a model reference gives in `generator`. A scenario pack
# is a plugin of the entry point group `gatedtrain.scenarios`: the loaded object carries `generators`, a mapping of
# name -> `generate(model_ref, item, *, sampler, seed) -> prediction`; `item` is `{"input": ...}` only, the label
# stays with the scorer. A pack may also carry `batch_generators`, a mapping of name -> `BatchGenerator` (AN-25): a
# declaration the keeper runs in its inference container, never in process. A name is a per-item or a batch generator
# of one pack. There is no run-time registration: a stage cannot add or replace a generator, only the installed
# distributions can.
SCENARIO_ENTRY_POINT_GROUP = "gatedtrain.scenarios"


def _scenario_entry_points() -> list[Any]:
    return list(metadata.entry_points(group=SCENARIO_ENTRY_POINT_GROUP))


# Step-1 boundary: the stages share the operator's process, so a stage can still monkeypatch a module attribute
# (this snapshot, `_scenario_entry_points`, the scorers). What is closed here is the ordinary path: a pack's own
# mutable `generators` dict is copied once and never re-read. Enforcement against hostile in-process code comes with
# the step-2 keeper, which runs the sealed evaluation outside the stages' process.
_SCENARIO_SNAPSHOT: tuple[MappingProxyType, MappingProxyType, MappingProxyType, MappingProxyType] | None = None


def _reset_scenario_snapshot() -> None:
    """Forget the snapshot so the next lookup reloads the entry points (tests that patch the loader)."""
    global _SCENARIO_SNAPSHOT
    _SCENARIO_SNAPSHOT = None


def _scenario_snapshot() -> tuple[MappingProxyType, MappingProxyType, MappingProxyType, MappingProxyType]:
    """(generator name -> function, generator name -> pack name, scorer name -> (pack name, declaration), batch
    generator name -> `BatchGenerator`) of every installed scenario pack, loaded once per process and frozen; the
    pack names cover per-item and batch generators alike. A pack that fails to load, has no `generators` mapping of
    callables or reuses a generator name is refused (`AN-8`) without echoing its message; a pack whose
    `batch_generators` is not a mapping of name to `BatchGenerator`, or a name that is both a per-item and a batch
    generator or a batch generator of two packs, is refused (`AN-25`); a pack whose `scorers` is not a mapping of
    name to declaration, or that reuses another pack's scorer name, is refused (`AN-17`). The declarations
    themselves are checked by `scorers.resolve`."""
    global _SCENARIO_SNAPSHOT
    if _SCENARIO_SNAPSHOT is not None:
        return _SCENARIO_SNAPSHOT
    fns: dict[str, Callable[..., Any]] = {}
    packs: dict[str, str] = {}
    batches: dict[str, BatchGenerator] = {}
    pack_scorers: dict[str, tuple[str, MappingProxyType]] = {}
    for ep in _scenario_entry_points():
        try:
            pack = ep.load()
        except Exception as exc:
            raise _refuse("AN-8", f"entry point {ep.name}", f"the scenario pack failed to load ({type(exc).__name__})",
                          "fix or uninstall the scenario pack") from None
        gens = getattr(pack, "generators", None)
        if not (isinstance(gens, Mapping) and all(isinstance(k, str) and callable(v) for k, v in gens.items())):
            raise _refuse("AN-8", f"entry point {ep.name}", "the scenario pack has no `generators` mapping of name "
                          "to function", "expose `generators = {name: generate}`")
        for name, fn in dict(gens).items():
            if name in fns:
                raise _refuse("AN-8", f"entry point {ep.name}", f"generator {name!r} is provided by two scenario packs",
                              "give each generator a unique name")
            if name in batches:
                raise _refuse("AN-25", f"entry point {ep.name}", f"generator {name!r} is both a per-item and a batch "
                              "generator", "give each generator a unique name")
            fns[name] = fn
            packs[name] = ep.name
        bgens = getattr(pack, "batch_generators", None)
        bgens = {} if bgens is None else bgens
        if not (isinstance(bgens, Mapping) and all(isinstance(k, str) and isinstance(v, BatchGenerator)
                                                   for k, v in bgens.items())):
            raise _refuse("AN-25", f"entry point {ep.name}", "the scenario pack's `batch_generators` is not a mapping "
                          "of name to BatchGenerator", "expose `batch_generators = {name: BatchGenerator(...)}` or "
                          "leave it out")
        for name, gen in dict(bgens).items():
            if name in batches:
                raise _refuse("AN-25", f"entry point {ep.name}", f"batch generator {name!r} is provided by two "
                              "scenario packs", "give each generator a unique name")
            if name in fns:
                raise _refuse("AN-25", f"entry point {ep.name}", f"generator {name!r} is both a per-item and a batch "
                              "generator", "give each generator a unique name")
            batches[name] = gen
            packs[name] = ep.name
        decls = getattr(pack, "scorers", None)
        decls = {} if decls is None else decls
        if not (isinstance(decls, Mapping) and all(isinstance(k, str) and isinstance(v, Mapping)
                                                   for k, v in decls.items())):
            raise _refuse("AN-17", f"entry point {ep.name}", "the scenario pack's `scorers` is not a mapping of name "
                          "to declaration", "expose `scorers = {name: {id, fn, code_sha256, ...}}` or leave it out")
        for name, decl in dict(decls).items():
            if name in pack_scorers:
                raise _refuse("AN-17", f"entry point {ep.name}", f"scorer {name!r} is provided by two scenario packs",
                              "give each pack scorer a unique name")
            pack_scorers[name] = (ep.name, MappingProxyType(dict(decl)))
    _SCENARIO_SNAPSHOT = (MappingProxyType(fns), MappingProxyType(packs), MappingProxyType(pack_scorers),
                          MappingProxyType(batches))
    return _SCENARIO_SNAPSHOT


def scenario_generators() -> Mapping[str, Callable[..., Any]]:
    return _scenario_snapshot()[0]


def scenario_batch_generators() -> Mapping[str, BatchGenerator]:
    """Batch generator name -> declaration of the installed scenario packs (AN-25)."""
    return _scenario_snapshot()[3]


def scenario_scorers() -> Mapping[str, tuple[str, Mapping]]:
    """Scorer name -> (pack name, frozen declaration) of the installed scenario packs (`scorers.resolve` checks
    each declaration)."""
    return _scenario_snapshot()[2]


def scenario_pack_of(generator: Any) -> str | None:
    """The name of the entry point whose pack provides the generator (per-item or batch), or None."""
    return _scenario_snapshot()[1].get(generator) if isinstance(generator, str) else None


def registered_generator(full: dict) -> tuple[str, str] | None:
    """(scenario pack, generator) a registration binds in its `base_model` field, or None."""
    bm = full.get("base_model")
    if isinstance(bm, dict) and isinstance(bm.get("scenario_pack"), str) and isinstance(bm.get("generator"), str):
        return bm["scenario_pack"], bm["generator"]
    return None


_REG_KEYS = ("hash", "scorer", "direction", "margin", "alpha", "item_set_hash", "post_hoc", "amends")


def _registered(ctx: StageContext, rule: str) -> tuple[dict, dict]:
    """(registration facts, full registration) of the `registration` input. The facts are read from the trust store,
    never from the output record: the record must name the latest registration held there and agree with it."""
    out = ctx.project.read_artifact(ctx.inputs["registration"])
    if not isinstance(out, dict) or not isinstance(out.get("hash"), str):
        raise _refuse(rule, "inputs.registration", "the registration record is malformed", "rerun preregister")
    if out["hash"] != ctx.trust.latest_registration():
        raise _refuse(rule, "inputs.registration", "the registration is not the latest one in the trust store",
                      "rerun preregister")
    full = ctx.trust.registration(out["hash"])
    clause = full.get("clause") or {}
    facts = {"hash": out["hash"], "scorer": clause.get("metric"), "direction": clause.get("direction"),
             "margin": clause.get("margin"), "alpha": clause.get("alpha"),
             "item_set_hash": (full.get("eval_distribution") or {}).get("item_set_hash"),
             "post_hoc": bool(full.get("post_hoc", False)), "amends": full.get("amends")}
    for key, want in facts.items():
        if out.get(key) != want:
            raise _refuse(rule, f"inputs.registration.{key}",
                          f"the registration record differs from the trust store on {key}", "rerun preregister")
    return facts, full


def _done_row(project: Project, stage: str) -> dict | None:
    row = None
    for r in project.read_attempts():
        if r.get("stage") == stage and r.get("event") == "done":
            row = r
    return row


def _produced_by_training(project: Project, ref: ArtifactRef) -> bool:
    """True when the artifact is an output of a done run of its stage whose start line recorded `trains: true`. It
    fails closed: a start line without the flag (or with another value) does not count as training."""
    attempt = None
    for r in project.read_attempts():
        if r.get("stage") == ref.stage and r.get("event") == "done":
            outs = r.get("outputs") or {}
            attempt = r.get("attempt") if any(isinstance(o, dict) and o.get("sha256") == ref.sha256
                                              and o.get("path") == ref.path for o in outs.values()) else None
    if attempt is None:
        return False
    return any(r.get("stage") == ref.stage and r.get("event") == "start" and r.get("attempt") == attempt
               and r.get("trains") is True for r in project.read_attempts())


def _registered_scorer(full: dict) -> tuple[scorers.ScorerSpec, dict] | None:
    """(spec, bound parameters) of the registration's `evaluators[0]`, or None when no scorer has its id."""
    evs = full.get("evaluators")
    ev = evs[0] if isinstance(evs, list) and evs and isinstance(evs[0], dict) else {}
    return scorers.spec_for_evaluator(ev)


def _evaluator_problem(full: dict) -> str | None:
    evs = full.get("evaluators")
    ev = evs[0] if isinstance(evs, list) and evs and isinstance(evs[0], dict) else {}
    found = _registered_scorer(full)
    if found is None or (full.get("clause") or {}).get("metric") != found[0].name or \
            ev.get("manifest_hash") != scorers.manifest_hash(*found):
        return ("the registered evaluator is not the scorer this build runs (its id or manifest hash differs), so the "
                "sealed items cannot be scored as registered")
    return None


def _score_arm(ctx: StageContext, sealed: SealedRef, fn: Callable[..., Any], model: dict, full: dict,
               spec: scorers.ScorerSpec, params: dict) -> list[dict]:
    """Per-item records of one arm, scored with the registered scorer (`spec` bound to `params`): the
    `sealed_eval.item_scorer` of each registered seed, run by the trust backend over the sealed items."""
    score = scorers.bind(spec, params)
    records: list[dict] = []
    for seed in _eval_seeds(full):
        records.extend(ctx.trust.run_sealed(sealed, sealed_eval.item_scorer(
            fn, model, full, score, seed, score_range=spec.score_range, scorer_name=spec.name)))
    return records


class EvaluateSealed:
    """AN-8: scores the sealed items with exactly the registered sampler and eval_seeds and the registered scorer
    only. What leaves the trust backend is one `{item_id, seed, score}` record per item and seed (`item_id` is the item's
    position in the backend's item order, counted by `sealed_eval.item_scorer` for each seed, not a content hash). In step 1 the operator process can still read the sealed items; the
    guarantee is stated (`sealed_guarantee`), not enforced.

    A project has one sealed scoring: before scoring, `run` claims it in the trust store for the candidate's model
    reference hash and the registration, and once the claim exists every further scoring is refused (`AN-8`), for
    another candidate and for the same reference alike (a reference need not change when its weights do), because
    comparing several trainings on the sealed set and gating the best would select on the test data without
    spending alpha. A done `evaluate_sealed` whose inputs are unchanged is still reused by the machine."""

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
        if _keeper(ctx):
            return Proposal(self.name, f"ask the keeper to score the sealed items for both models with the registered "
                            f"sampler on {n_seeds} evaluation seeds; the keeper first spends alpha "
                            f"{ctx.ledger.available():g} on this attempt, and the per-item records stay in the keeper",
                            {}, {})
        return Proposal(self.name, f"score the sealed items for both models with the registered sampler on "
                        f"{n_seeds} evaluation seeds; only per-item scores leave the trust backend", {}, {})

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        reg, full = _registered(ctx, "AN-8")
        problem = _evaluator_problem(full)
        if problem:
            raise _refuse("AN-8", "registration.evaluators[0].manifest_hash", problem,
                          "register again with preregister, or restore the scorer")
        spec, params = _registered_scorer(full)       # not None: `_evaluator_problem` passed
        seeds = _eval_seeds(full)
        item_set = full["eval_distribution"]["item_set_hash"]
        sealed = SealedRef(item_set)
        base_ref, cand_ref = ctx.inputs["baseline_model"], ctx.inputs["candidate_model"]
        if base_ref.sha256 != registered_base(full):
            raise _refuse("AN-8", "inputs.baseline_model", "the baseline is not the base model that was registered",
                          "wire baseline_model from the stage that produced the registered base model")
        if cand_ref.sha256 == base_ref.sha256:
            raise _refuse("AN-8", "inputs.candidate_model", "the candidate and the baseline are the same model",
                          "wire candidate_model from the training stage")
        if not _produced_by_training(ctx.project, cand_ref):
            raise _refuse("AN-8", "inputs.candidate_model", "the candidate was not produced by a training stage",
                          "wire candidate_model from a stage that trains")
        served = registered_generator(full)
        if served is None:
            raise _refuse("AN-8", "registration.base_model", "the registration names no scenario pack and generator",
                          "register again with preregister, with a base model whose `generator` a scenario pack serves")
        if _keeper(ctx):
            return self._run_keeper(ctx, reg, served[1], item_set)
        if served[1] in scenario_batch_generators():
            # AN-8 (revision 25): a batch generator runs only in the keeper's inference container. Refused here,
            # before the claim, so the project's one sealed scoring is not used up.
            raise _refuse("AN-8", "registration.base_model.generator",
                          "a batch generator runs in the keeper's inference container", "bind the project to a keeper")
        fn = scenario_generators().get(served[1])
        if fn is None or scenario_pack_of(served[1]) != served[0]:
            raise _refuse("AN-8", "registration.base_model.generator",
                          f"the registered scenario pack {served[0]!r} no longer serves the generator {served[1]!r}",
                          "install the scenario pack that was registered")
        models: dict[str, dict] = {}
        for arm in ("baseline", "candidate"):
            model = ctx.project.read_artifact(ctx.inputs[f"{arm}_model"])
            if not isinstance(model, dict):
                raise _refuse("AN-8", f"inputs.{arm}_model", "the model reference is malformed", "rerun its stage")
            if model.get("generator") != served[1]:
                raise _refuse("AN-8", f"inputs.{arm}_model.generator",
                              f"the {arm} model names the generator {model.get('generator')!r}, not the registered "
                              f"{served[1]!r}", "both models must be served by the registered generator")
            models[arm] = model
        # One sealed scoring per project (05 §18.5: no repeated testing). Once the claim exists nothing is scored
        # again, for the same model reference either: a reference need not change when its weights do (a mutable
        # checkpoint path), so rescoring it could pick the best of several trainings. The machine still returns the
        # recorded scoring of a done stage whose inputs are unchanged, without calling `run`.
        held = ctx.trust.sealed_evaluation()
        if held is None:
            claim = ctx.trust.claim_sealed_evaluation(cand_ref.sha256, reg["hash"])
            fresh = claim.get("candidate_model_sha256") == cand_ref.sha256 and claim.get("registration_hash") == \
                reg["hash"]
        else:
            claim, fresh = held, False
        if not fresh:
            same = claim.get("candidate_model_sha256") == cand_ref.sha256
            raise _refuse("AN-8", "inputs.candidate_model",
                          "the sealed set of this project was already scored"
                          + (" for this model reference; its weights may have changed since" if same else
                             f" for another candidate ({claim.get('candidate_model_sha256')})")
                          + "; a second scoring would select on the test data", SEALED_TEST_USED)
        out: dict[str, ArtifactRef] = {}
        for arm in ("baseline", "candidate"):
            ref = ctx.inputs[f"{arm}_model"]
            records = _score_arm(ctx, sealed, fn, models[arm], full, spec, params)
            doc = {**_meta(SCHEMA_SEALED_RECORDS), "arm": arm, "registration_hash": reg["hash"],
                   "item_set_hash": item_set, "n_items": len(records) // len(seeds), "eval_seeds": list(seeds),
                   "sampler": dict(full["sampler"]), "model_ref_sha256": ref.sha256, "records": records}
            out[f"{arm}_records"] = ctx.project.write_artifact(ctx.stage or self.name, f"{arm}_records", doc,
                                                                SCHEMA_SEALED_RECORDS)
        return out

    def _run_keeper(self, ctx: StageContext, reg: dict, generator: str, item_set: str) -> dict[str, ArtifactRef]:
        """One sealed attempt in the keeper (AN-8, AN-15): it spends alpha, signs the claim and scores; the project
        gets a receipt per arm that names the claim, never a record."""
        models: dict[str, tuple[dict, str]] = {}
        for arm in ("baseline", "candidate"):
            ref = ctx.inputs[f"{arm}_model"]
            model = ctx.project.read_artifact(ref)
            if not isinstance(model, dict):
                raise _refuse("AN-8", f"inputs.{arm}_model", "the model reference is malformed", "rerun its stage")
            if model.get("generator") != generator:
                raise _refuse("AN-8", f"inputs.{arm}_model.generator",
                              f"the {arm} model names the generator {model.get('generator')!r}, not the registered "
                              f"{generator!r}", "both models must be served by the registered generator")
            models[arm] = (model, ref.sha256)
        claim = ctx.trust.evaluate(reg["hash"], models["baseline"], models["candidate"])
        env = ctx.trust.receipt("claim", attempt_id=claim["attempt_id"])
        out: dict[str, ArtifactRef] = {}
        for arm in ("baseline", "candidate"):
            doc = {**_meta(SCHEMA_SEALED_RECORDS, KEEPER_SCHEMA_VERSION), "arm": arm, "held_by": "keeper",
                   "attempt_id": claim["attempt_id"], "registration_hash": reg["hash"], "item_set_hash": item_set,
                   "model_ref_sha256": models[arm][1], "claim_sha256": signing.envelope_sha256(env)}
            out[f"{arm}_records"] = ctx.project.write_artifact(ctx.stage or self.name, f"{arm}_records", doc,
                                                                SCHEMA_SEALED_RECORDS)
        return out

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        problems: list[Problem] = []

        def bad(field: str, message: str) -> None:
            problems.append(Problem("AN-8", field, message, "rerun evaluate_sealed"))

        if _keeper(ctx):
            try:
                _receipts(ctx, outputs["baseline_records"], outputs["candidate_records"], None)
            except Refused as exc:
                problems.extend(exc.problems)
            return problems
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


def _receipts(ctx: StageContext, base_ref: ArtifactRef, cand_ref: ArtifactRef, reg: dict | None) -> dict:
    """Keeper mode: the claim record the two arm receipts name, checked (AN-8, AN-14): exactly the receipt keys, one
    attempt and claim, the claim envelope among the project's receipts with the same registration and models, and,
    with `reg`, the registered item set and registration."""
    recs = {}
    for arm, ref in (("baseline", base_ref), ("candidate", cand_ref)):
        rec = ctx.project.read_artifact(ref)
        if not (isinstance(rec, dict) and set(rec) == RECEIPT_KEYS and rec["arm"] == arm and rec["held_by"] == "keeper"
                and rec["schema_version"] == KEEPER_SCHEMA_VERSION):
            raise _refuse("AN-8", f"{arm}_records", "the record is not a keeper receipt of this arm",
                          "rerun evaluate_sealed")
        recs[arm] = rec
    base, cand = recs["baseline"], recs["candidate"]
    if any(base[k] != cand[k] for k in ("attempt_id", "registration_hash", "item_set_hash", "claim_sha256")):
        raise _refuse("AN-8", "candidate_records", "the two receipts name different attempts", "rerun evaluate_sealed")
    if reg is not None and (cand["registration_hash"] != reg["hash"] or cand["item_set_hash"] != reg["item_set_hash"]):
        raise _refuse("AN-9", "candidate_records", "the receipts were not made for the registered item set",
                      "rerun evaluate_sealed on the latest registration")
    env = ctx.trust.receipt("claim", attempt_id=cand["attempt_id"])
    claim = env["record"] if env is not None else {}
    if env is None or signing.envelope_sha256(env) != cand["claim_sha256"] or \
            claim.get("registration_hash") != cand["registration_hash"] or \
            claim.get("baseline_model_sha256") != base["model_ref_sha256"] or \
            claim.get("candidate_model_sha256") != cand["model_ref_sha256"]:
        raise _refuse("AN-14", "candidate_records.claim_sha256", "the receipts do not name a claim the keeper signed "
                      "for these models", "rerun evaluate_sealed; do not edit the receipts")
    return claim


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


def _arm(project: Project, ref: ArtifactRef, reg: dict, seeds: list, n_sealed: int, ev_row: dict | None) -> dict:
    rec = project.read_artifact(ref)
    if not isinstance(rec, dict) or not isinstance(rec.get("records"), list):
        raise _refuse("AN-9", f"inputs.{ref.name}", "the sealed records are malformed", "rerun evaluate_sealed")
    if rec.get("item_set_hash") != reg["item_set_hash"] or rec.get("registration_hash") != reg["hash"]:
        raise _refuse("AN-9", f"inputs.{ref.name}", "the records were not made for the registered item set",
                      "rerun evaluate_sealed on the registered sealed set")
    if rec.get("eval_seeds") != seeds:
        raise _refuse("AN-9", f"inputs.{ref.name}", "the records use other evaluation seeds than the registration",
                      "rerun evaluate_sealed")
    recorded = ((ev_row or {}).get("outputs") or {}).get(ref.name)
    if not isinstance(recorded, dict) or recorded.get("sha256") != ref.sha256:
        raise _refuse("AN-9", f"inputs.{ref.name}", "the records are not the ones evaluate_sealed recorded",
                      "wire the records from the evaluate_sealed stage and rerun it")
    if rec.get("n_items") != n_sealed:
        raise _refuse("AN-9", f"inputs.{ref.name}", "the records do not cover the registered sealed set size",
                      "rerun evaluate_sealed")
    return {"records": rec["records"], "model_ref_sha256": rec.get("model_ref_sha256")}


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
        if _keeper(ctx):
            return Proposal(self.name, "ask the keeper to test the new model against the base model at the alpha it "
                            "granted the attempt; the project gets the signed status and alpha",
                            {"registration": reg["hash"]}, {})
        return Proposal(self.name, f"test the new model against the base model on the sealed records at alpha "
                        f"{ctx.ledger.available():g}; this spends the alpha budget", {"registration": reg["hash"]}, {})

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        if _keeper(ctx):
            return self._run_keeper(ctx)
        available = ctx.ledger.available()
        if not (_num(available) and available > 0):
            raise _refuse("AN-9", "gate", "the alpha budget is spent: step 1 allows one gate test per project",
                          NEW_TEST_FIX)
        reg, full = _registered(ctx, "AN-9")
        clause = dict(full["clause"])
        if clause.get("direction") != reg["direction"] or clause.get("margin") != reg["margin"]:
            raise _refuse("AN-9", "registration.clause", "the registration and its output record disagree on the "
                          "direction or the margin", "rerun preregister")
        seeds = _eval_seeds(full)
        n_sealed = ctx.trust.sealed_summary(SealedRef(reg["item_set_hash"]))["n"]
        ev_row = _done_row(ctx.project, "evaluate_sealed")
        if ev_row is None:
            raise _refuse("AN-9", "inputs", "evaluate_sealed has no done record", "run evaluate_sealed first")
        base = _arm(ctx.project, ctx.inputs["baseline_records"], reg, seeds, n_sealed, ev_row)
        cand = _arm(ctx.project, ctx.inputs["candidate_records"], reg, seeds, n_sealed, ev_row)
        cand_model = (ev_row.get("input_sha256s") or {}).get("candidate_model")
        found = _registered_scorer(full)
        if found is None:
            raise _refuse("AN-9", "registration.evaluators[0]", "the registered evaluator is not a scorer this build "
                          "runs, so its score range is unknown", "register again with preregister")
        expected = {"baseline": registered_base(full), "candidate": cand_model}
        attempt_id = _attempt_key(ctx.project, ctx.stage or self.name)
        alpha = ctx.ledger.spend(attempt_id)                      # the ledger decides alpha, not the registration
        try:
            result = sealed_eval.decide(full, base, cand, alpha=alpha, n_sealed=n_sealed, expected_models=expected,
                                        score_range=found[0].score_range)
        except Exception:
            ctx.ledger.record(attempt_id, False)
            raise
        ctx.ledger.record(attempt_id, result["status"] == gates.PASS)
        doc = {**_meta(SCHEMA_GATE_RESULT), **result, "alpha": alpha, "attempt_id": attempt_id,
               "registration_hash": reg["hash"], "post_hoc": bool(reg["post_hoc"])}
        return {"gate_result": ctx.project.write_artifact(ctx.stage or self.name, "gate_result", doc, SCHEMA_GATE_RESULT)}

    def _run_keeper(self, ctx: StageContext) -> dict[str, ArtifactRef]:
        """The keeper gates the attempt the receipts name at the alpha it granted and settles the ledger (AN-9,
        AN-15); the project gets the signed summary: status and alpha only."""
        reg, _ = _registered(ctx, "AN-9")
        ev_row = _done_row(ctx.project, "evaluate_sealed")
        if ev_row is None:
            raise _refuse("AN-9", "inputs", "evaluate_sealed has no done record", "run evaluate_sealed first")
        for arm in ("baseline", "candidate"):
            ref = ctx.inputs[f"{arm}_records"]
            recorded = (ev_row.get("outputs") or {}).get(ref.name)
            if not isinstance(recorded, dict) or recorded.get("sha256") != ref.sha256:
                raise _refuse("AN-9", f"inputs.{ref.name}", "the receipts are not the ones evaluate_sealed recorded",
                              "wire the records from the evaluate_sealed stage and rerun it")
        claim = _receipts(ctx, ctx.inputs["baseline_records"], ctx.inputs["candidate_records"], reg)
        summary = ctx.trust.gate(claim["attempt_id"])
        if summary.get("registration_hash") != reg["hash"] or summary.get("post_hoc") is not reg["post_hoc"]:
            raise _refuse("AN-9", "gate", "the keeper gated the attempt under another registration",
                          "rerun evaluate_sealed on the latest registration")
        env = ctx.trust.receipt("gate_summary", attempt_id=claim["attempt_id"])
        doc = {**_meta(SCHEMA_GATE_RESULT, KEEPER_SCHEMA_VERSION), **summary,
               "envelope_sha256": signing.envelope_sha256(env)}
        return {"gate_result": ctx.project.write_artifact(ctx.stage or self.name, "gate_result", doc, SCHEMA_GATE_RESULT)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        res = ctx.project.read_artifact(outputs["gate_result"])
        reg, _ = _registered(ctx, "AN-9")
        problems: list[Problem] = []

        def bad(field: str, message: str) -> None:
            problems.append(Problem("AN-9", field, message, "rerun gate"))

        if res.get("registration_hash") != reg.get("hash"):
            bad("gate_result.registration_hash", "the result is for another registration")
        if res.get("status") not in (gates.PASS, gates.FAIL, gates.INCONCLUSIVE, gates.MISSING):
            bad("gate_result.status", "the status is not PASS, FAIL, INCONCLUSIVE or MISSING")
        if res.get("post_hoc") is not reg["post_hoc"]:
            bad("gate_result.post_hoc", "the result disagrees with the registration on post_hoc")
        if reg["post_hoc"] is True and res.get("status") == gates.PASS:
            bad("gate_result.status", "a post hoc registration cannot PASS")
        if _keeper(ctx):
            env = ctx.trust.receipt("gate_summary", attempt_id=res.get("attempt_id"))
            body = {k: v for k, v in res.items() if k not in ("schema", "schema_version", "envelope_sha256")}
            if env is None or signing.envelope_sha256(env) != res.get("envelope_sha256") or env["record"] != body:
                bad("gate_result.envelope_sha256", "the result is not the gate summary the keeper signed")
            return problems
        entry = [e for e in ctx.ledger.entries() if e["attempt_id"] == res.get("attempt_id")]
        if len(entry) != 1 or entry[0]["alpha"] != res.get("alpha") or \
                entry[0]["passed"] is not (res.get("status") == gates.PASS):
            bad("gate_result.attempt_id", "the ledger does not show this attempt with this alpha and outcome")
        return problems


_LOGGED_STAGES = ("evaluate_sealed", "gate")      # every attempt row (start, done, failed, refused) goes in the report
REPORT_MD = "report.md"


def _render_head(report: dict) -> list[str]:
    """The lines every report version shares: headline, effect, registration and power."""
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
              f"- smallest gain the sealed set can detect: {shown['mde_planned']}"]
    return lines


def _render_keeper(report: dict) -> list[str]:
    """Report `schema_version` 2 (built and signed by the keeper): every attempt with its alpha, the set's ledger, the
    alpha rule and its guarantee, the keeper's key fingerprint, and (revision 25) the compute block: the keeper's
    inference containers, and the backend's jobs when the driver sent them."""
    rule = report["alpha_rule"]
    ledger = {row["attempt_id"]: row for row in report["ledger"]}
    lines = ["", "## Sealed attempts of this project"]
    v3 = report.get("schema_version") == sealed_eval.REPORT_SCHEMA_VERSION              # AN-27: the attempt's keeper mode
    for a in report["attempts"]:
        row = ledger.get(a["attempt_id"], {})
        lines.append(f"- attempt {a['attempt_id']} (j = {a['j']}) at {a['utc']}: status {a['status']}, alpha "
                     f"{a['alpha']}, alpha spent {row.get('cost')}, alpha earned {row.get('earned')}, candidate "
                     f"{a['candidate_model_sha256']}" + (f", keeper mode {a['keeper_mode']}" if v3 else ""))
    lines += ["", "## Alpha-investing ledger of the sealed set"]
    for row in report["ledger"]:
        lines.append(f"- attempt {row['attempt_id']} of project {row['project_id']}: alpha {row['alpha']}, "
                     f"{'passed' if row['passed'] else 'did not pass'}, alpha spent {row['cost']}, alpha earned "
                     f"{row['earned']}, wealth after {row['wealth_after']}")
    lines += ["", "## Alpha rule",
              f"- rule: {rule['rule']}",
              f"- initial wealth: {rule['initial_wealth']}, payout on a PASS: {rule['payout']}, smallest bid: "
              f"{rule['min_bid']}, planning alpha: {rule['planning_alpha']}",
              f"- guarantee: {rule['guarantee']}", "",
              "## Keeper",
              f"- key fingerprint: {report['keeper']['fingerprint']}",
              f"- environment: {report['keeper']['environment_sha256']}"]
    compute = report.get("compute")
    if isinstance(compute, dict):                     # AN-10 (revision 25): the keeper's and the driver's compute
        k = compute["keeper"]
        lines += ["", "## Compute",
                  f"- keeper: {k['containers']} inference containers, {k['seconds']} s, {k['gpu_seconds']} GPU s "
                  "(the keeper's signed records)"]
        b = compute.get("backend")
        if b is not None:
            lines.append(f"- backend: {b['jobs']} jobs, {b['seconds']} s, {b['gpu_seconds']} GPU s ({b['source']})")
            lines += image_provenance.render_lines(b.get("images") or [])         # IMG-7
            if b.get("images"):
                lines.append(f"- {keeper_texts.LIMIT_IMAGES_NOT_OBSERVED}")
    return lines


def _mode_head(report: dict, lines: list[str]) -> list[str]:
    """AN-27: a keeper report (version 3) opens with the observed keeper mode, and in same-user mode its headline
    carries the qualifier, so no surface reads "PASS" next to "sealed" without it. A set served in more than one
    mode says so."""
    mode = report["keeper_mode"]
    out = [keeper_texts.MODE_LINE[mode], "", lines[0] + (keeper_texts.HEADLINE_SUFFIX_SAME_USER
                                                         if mode == modes.SAME_USER else ""), *lines[1:]]
    seen = [m["mode"] for m in report.get("modes_seen") or []]
    if len(set(seen)) > 1:
        out[2:2] = [f"This sealed set was served in more than one keeper mode ({', '.join(seen)}); the report states "
                    f"the weakest, {mode}.", ""]
    return out


def render_markdown(report: dict) -> str:
    """`report.md`, rendered from `report.json` alone: every number and name in it is read from `report`. A report of
    `schema_version` 2 (the keeper's) lists every attempt with alpha spent and earned, the alpha rule and the key
    fingerprint; version 1 renders as in step 1."""
    lines = _render_head(report)
    if report.get("schema_version") == sealed_eval.REPORT_SCHEMA_VERSION:
        lines = _mode_head(report, lines)
    if report.get("schema_version") in KEEPER_REPORT_VERSIONS:
        lines += _render_keeper(report)
    else:
        lines += ["", "## Gate attempts"]
        for e in report["ledger"]:
            outcome = {True: "passed", False: "did not pass", None: "no outcome recorded"}[e["passed"]]
            lines.append(f"- attempt {e['attempt_id']}: alpha {e['alpha']}, {outcome}")
        lines += ["", "## Sealed evaluation and gate log"]
        for a in report["attempts"]:
            lines.append(f"- {a['stage']} attempt {a['attempt']} {a['event']} at {a['utc']}")
    lines += ["", "## Known limits"]
    lines += [f"- {x['text']}" for x in report["known_limits"]]
    lines.append("")
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
        reg, full = _registered(ctx, "AN-10")
        found = _registered_scorer(full)
        if found is None:
            raise _refuse("AN-10", "registration.evaluators[0]", "the registered evaluator is not a scorer this build "
                          "runs, so the metric's unit is unknown", "register again with preregister")
        power = ctx.project.read_artifact(ctx.inputs["power"])
        gate = ctx.project.read_artifact(ctx.inputs["gate_result"])
        if gate.get("registration_hash") != reg["hash"] or gate.get("post_hoc") is not reg["post_hoc"]:
            raise _refuse("AN-10", "inputs.gate_result", "the gate result is not for the latest registration",
                          "rerun report with the gate result of the latest registration")
        keep = {"schema", "schema_version"}
        attempts = [{"stage": r.get("stage"), "attempt": r.get("attempt"), "event": r.get("event"), "utc": r.get("utc"),
                     "key": r.get("key")}
                    for r in ctx.project.read_attempts() if r.get("stage") in _LOGGED_STAGES and r.get("event") != "stale"]
        return {**_meta(SCHEMA_REPORT),
                "status": _effective_status(gate["status"], bool(reg["post_hoc"])),
                "metric": {"scorer": reg["scorer"], "unit": found[0].unit},
                "n_sealed": power["n_sealed"],
                "registration": {k: reg.get(k) for k in _REG_KEYS},
                "power": {k: v for k, v in power.items() if k not in keep},
                "gate_result": {k: v for k, v in gate.items() if k not in keep},
                "display": display_block(gate, power),
                "ledger": ctx.ledger.entries(), "attempts": attempts,
                "sealed_guarantee": ctx.trust.sealed_guarantee,
                "known_limits": known_limits(ctx.trust.sealed_guarantee, attempts)}

    @staticmethod
    def _build_keeper(ctx: StageContext) -> dict:
        """The keeper's signed report (`schema_version` 2), after every receipt in the project verified (AN-14)."""
        reg, _ = _registered(ctx, "AN-10")
        gate = ctx.project.read_artifact(ctx.inputs["gate_result"])
        if gate.get("registration_hash") != reg["hash"] or gate.get("post_hoc") is not reg["post_hoc"]:
            raise _refuse("AN-10", "inputs.gate_result", "the gate result is not for the latest registration",
                          "rerun report with the gate result of the latest registration")
        ctx.trust.receipts()
        report = ctx.trust.report()
        if report.get("schema") != SCHEMA_REPORT or report.get("schema_version") not in KEEPER_REPORT_VERSIONS or \
                gate.get("attempt_id") not in [a.get("attempt_id") for a in report.get("attempts") or []]:
            raise _refuse("AN-10", "report", "the keeper's report does not cover this project's gate result",
                          "check the keeper")
        return report

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        report = self._build_keeper(ctx) if _keeper(ctx) else self._build(ctx)
        ref = ctx.project.write_artifact(ctx.stage or self.name, "report", report, SCHEMA_REPORT)
        rel = f"stages/{ctx.stage or self.name}/{REPORT_MD}"
        _refuse_links(ctx.project.root, rel)
        _write_atomic(ctx.project.root / rel, render_markdown(ctx.project.read_artifact(ref)).encode("utf-8"))
        return {"report": ref}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        report = ctx.project.read_artifact(outputs["report"])
        rel = f"stages/{ctx.stage or self.name}/{REPORT_MD}"
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
        reg, _ = _registered(ctx, "AN-10")
        if report.get("status") != _effective_status(gate.get("status"), reg["post_hoc"]):
            problems.append(Problem("AN-10", "report.status", "the report status differs from the gate result",
                                    "rerun report"))
        limits = report.get("known_limits")
        if _keeper(ctx):
            env = ctx.trust.receipt("report")
            if env is None or env["record"] != report:
                problems.append(Problem("AN-10", "report", "report.json is not the report the keeper signed",
                                        "rerun report; do not edit report.json"))
            want = keeper_texts.known_limits(ctx.trust.sealed_guarantee, len(report.get("ledger") or []))
        else:
            want = known_limits(ctx.trust.sealed_guarantee, report.get("attempts") or [])
        if report.get("sealed_guarantee") != ctx.trust.sealed_guarantee or not isinstance(limits, list) or \
                limits != want:
            problems.append(Problem("AN-10", "report.sealed_guarantee",
                                    "the report does not state the trust backend's guarantee and the known limits "
                                    "built from the records", "rerun report"))
        return problems
