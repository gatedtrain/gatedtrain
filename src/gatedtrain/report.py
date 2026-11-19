"""Gate report: JSON and a short text rendering (MVP-SPEC section 2.11).

Carries every 05 section 3.7 report field the skeleton has (05:142):
hypothesis type, Delta-hat and interval, margin, alpha used, achieved power,
MDE, `underpowered`, registration hash, and evaluator, item-set and
sampler hashes. Fields it cannot fill (independence tier and transfer test,
05 section 6; contamination, 05 section 10; family alpha, 05 section 5) are
printed as `not-assessed (skeleton)`, and so is `disjoint_from` when the
spec declares it (decision D-5). `sd_zero` flags identical arms (G-13);
`unpaired_*` count items present in one arm only, and the scores use the
paired items only (decision D-3). Design revision 10 (tests/README.md
section 11.15) adds `status_reasons`, `p_value` (R10A-10) and
`p_value_at_floor` (R10A-16; the text line prints `<= <p>` then), `k_actual`
(R10A-8), `missing_*`, `max_unpaired` (R10A-3), `min_paired` (R10A-5),
`post_hoc` and `registration_amends` (R10A-4) and `evaluator_hash_registered`
(R10A-2); `registration_amendment` (`version_only` for a version-only amendment, 05 section 3.7, R17-1; the
text line only then). Tags: prototype, unpromotable,
registration: unsigned. `created_at` is the
only wall-clock field and enters no hash compared by the determinism check.
Reports are immutable artifacts (05:142): the run never rewrites one.

Design revision 11 (tests/README.md section 11.16, R11A-4): new reports use item names whatever the
input's names: `n_items`, `samples_per_item`, `unit: item`, `direction`, the natural-unit difference
`natural_units: {delta_hat, lower, upper}` beside the oriented Delta-hat and interval (05 section 3.1),
`eval_distribution` from the normalized registration, `item_set_hash`, `missing_reasons` per arm and
reason, and `resampling_unit: item`. Older reports keep their old keys (they are never rewritten).

Design revisions 12 and 13 (tests/README.md section 11.17, R12A-16): `constraint_report` gives the 05
section 3.7 report fields of a constraint clause (violation count, n, exact upper bound, the judges' miss and
false-alarm rates with their upper bounds, canary results; for the non-inferiority form b, c, n, the
difference and the bound, R13-1) from a `constraints.evaluate_constraint` result.

Design revision 18 (05 sections 3.4, 3.5, 3.7; R18-3, R18-4, R18-6): a clause that resamples clusters adds
`cluster_key` and `n_clusters` to `interval.bootstrap` (`resampling_unit: cluster`), a percentile clause names its
variant there; `build_multi_report` reports a gate over several clauses: the combined `status` and
`status_reasons`, `multiplicity`, the joint power of independent clauses, and `clauses`, one entry per clause with
its name, evaluator, evaluator hashes and the per-clause fields of the one-clause report. A one-clause report is
unchanged.
"""

from __future__ import annotations

import copy
from typing import Any

from . import NOT_ASSESSED, REPORT_TAGS, STUB_MARK
from .registration import normalize, normalize_clause

NOT_ASSESSED_KEYS = ("independence_tier", "transfer_test", "contamination", "family_alpha")


def build_report(*, workflow: str, gate_id: str, clause: dict, result: dict, plan: dict, registration: dict,
                 hashes: dict[str, str], stubs: list[str], created_at: str, run_id: str,
                 disjoint_declared: bool, extra: dict[str, Any] | None = None,
                 amendment: str | None = None) -> dict[str, Any]:
    clause = normalize_clause(clause)
    registration = normalize(registration)
    dist = registration.get("eval_distribution")
    rep = {
        "schema": "gatedtrain.gate_report",
        "schema_version": "1.0",
        "tags": list(REPORT_TAGS),
        "stubs": stubs,
        "run_id": run_id,
        "workflow": workflow,
        "gate_id": gate_id,
        "hypothesis_type": result["hypothesis_type"],
        "metric": clause["metric"],
        "status": result["status"],
        "status_reasons": list(result["status_reasons"]),
        "delta_hat": result["delta_hat"],
        "interval": {
            "lower": result["ci_lower"],
            "upper": result["ci_upper"],
            "level": result["ci_level"],
            "method": result["interval_method"],
            "bootstrap": {"variant": result["bootstrap_variant"], "replicates": result["n_boot"],
                          "seed": result["bootstrap_seed"], "resampling_unit": result.get("resampling_unit", "item"),
                          **({"cluster_key": result["cluster_key"], "n_clusters": result["n_clusters"]}
                             if "cluster_key" in result else {})},                                   # R18-3
        },
        "direction": result["direction"],                              # R11-10
        "natural_units": {"delta_hat": result["delta_hat_natural"], "lower": result["ci_lower_natural"],
                          "upper": result["ci_upper_natural"]},
        "p_value": result["p_value"],
        "p_value_method": result["p_value_method"],
        "p_value_at_floor": result["p_value_at_floor"],                # R10A-16
        "margin": result["margin"],
        "alpha_used": result["alpha"],
        "power_target": result["power_target"],
        "achieved_power": result["achieved_power"],
        "mde": result["mde"],
        "mde_declared": float(clause["planning_effect"]),
        "underpowered": result["underpowered"],
        "n_items": result["n_items"],
        "unit": clause["unit"],
        "samples_per_item": int(clause["samples_per_item"]),
        "eval_seeds": list(clause["eval_seeds"]),
        "sd_diff": result["sd_diff"],
        "sd_zero": result["sd_zero"],                                  # G-13
        "unpaired_baseline": result["unpaired_baseline"],              # D-3
        "unpaired_candidate": result["unpaired_candidate"],
        "missing_baseline": result["missing_baseline"],               # R10A-3
        "missing_candidate": result["missing_candidate"],
        "missing_reasons": result["missing_reasons"],                 # R11A-6
        "max_unpaired": result["max_unpaired"],
        "min_paired": result["min_paired"],                           # R10A-5
        "k_actual": result["k_actual"],                               # R10A-8
        "design_effect": result["design_effect"],
        "plan": {k: plan[k] for k in ("n_required", "mde", "design_effect", "sigma_plan", "sigma_used", "n_available",
                                      "underpowered")},
        "baseline_score": result["baseline_score"],                    # D-3: paired items only
        "candidate_score": result["candidate_score"],
        "registration_hash": hashes["registration_hash"],
        "registration_amends": registration.get("amends"),            # R10A-4
        "registration_amendment": amendment,                          # R17-1: "version_only" or null
        "post_hoc": bool(registration.get("post_hoc", False)),
        "registration_id": registration.get("registration_id"),
        "registration_signed_by": registration.get("signed_by"),
        "analysis_code_version": registration.get("analysis_code_version"),
        "eval_distribution": dict(dist) if isinstance(dist, dict) else None,      # R11-19
        "item_set_hash": hashes["item_set_hash"],
        "evaluator_hash": hashes["evaluator_hash"],
        "evaluator_hash_registered": hashes["evaluator_hash_registered"],   # R10A-2
        "sampler_hash": hashes["sampler_hash"],
        "baseline_checkpoint_hash": hashes["baseline_checkpoint_hash"],
        "candidate_checkpoint_hash": hashes["candidate_checkpoint_hash"],
        "lockfile_hash": hashes["lockfile_hash"],
        **{k: NOT_ASSESSED for k in NOT_ASSESSED_KEYS},
        "created_at": created_at,
    }
    if disjoint_declared:
        # D-5: the spec declares `disjoint_from` (04:43); the skeleton does not check it.
        rep["disjoint_from"] = NOT_ASSESSED
    rep.update(extra or {})          # Tier B: metric_source (G-21), dataset_revision (G-17), notes (G-16)
    return rep


# The keys of a one-clause report that describe its clause; a report over several clauses (R18-6) gives them per clause.
CLAUSE_REPORT_KEYS = (
    "hypothesis_type", "metric", "status", "status_reasons", "delta_hat", "interval", "direction", "natural_units",
    "p_value", "p_value_method", "p_value_at_floor", "margin", "alpha_used", "power_target", "achieved_power", "mde",
    "mde_declared", "underpowered", "n_items", "unit", "samples_per_item", "eval_seeds", "sd_diff", "sd_zero",
    "unpaired_baseline", "unpaired_candidate", "missing_baseline", "missing_candidate", "missing_reasons",
    "max_unpaired", "min_paired", "k_actual", "design_effect", "plan", "baseline_score", "candidate_score",
    "evaluator_hash", "evaluator_hash_registered")


def build_multi_report(*, workflow: str, gate_id: str, clauses: list[tuple[dict, dict, dict, str | None]],
                       status: str, status_reasons: list[str], joint_power: float, registration: dict,
                       hashes: dict[str, Any], stubs: list[str], created_at: str, run_id: str,
                       disjoint_declared: bool, extra: dict[str, Any] | None = None,
                       clause_extra: list[dict[str, Any]] | None = None,
                       amendment: str | None = None) -> dict[str, Any]:
    """A gate over several clauses (05 section 3.7, R18-6). `clauses` holds (clause, result, plan, registered
    evaluator hash) per clause in registration order; `hashes["evaluator_hashes"]` maps an evaluator id to the hash
    the run resolved. Each clause entry is the one-clause report's clause part (`CLAUSE_REPORT_KEYS`), built by
    `build_report`, with `name`, `evaluator` and `clause_extra[i]` (the metric source) in front."""
    entries = []
    common: dict[str, Any] = {}
    for i, (clause, result, plan, registered) in enumerate(clauses):
        ev = clause["evaluator"]
        one = build_report(
            workflow=workflow, gate_id=gate_id, clause=clause, result=result, plan=plan, registration=registration,
            hashes={**{k: v for k, v in hashes.items() if k != "evaluator_hashes"},
                    "evaluator_hash": hashes["evaluator_hashes"][ev], "evaluator_hash_registered": registered},
            stubs=stubs, created_at=created_at, run_id=run_id, disjoint_declared=disjoint_declared,
            amendment=amendment)
        entries.append({"name": clause["name"], "evaluator": ev, **((clause_extra or [{}] * len(clauses))[i]),
                        **{k: one[k] for k in CLAUSE_REPORT_KEYS}})
        if not common:
            common = {k: v for k, v in one.items() if k not in CLAUSE_REPORT_KEYS}
    rep = {k: common[k] for k in ("schema", "schema_version", "tags", "stubs", "run_id", "workflow", "gate_id")}
    rep.update({
        "multiplicity": registration["multiplicity"]["plan"],
        "status": status,
        "status_reasons": list(status_reasons),
        "joint_power_independent": joint_power,                          # 05 section 3.5: about the product
        "clauses": entries,
    })
    rep.update({k: v for k, v in common.items() if k not in rep})
    rep.update(extra or {})
    return rep


def constraint_report(result: dict[str, Any]) -> dict[str, Any]:
    """Report fields of a constraint clause (05 section 3.7, section 3.9, section 6.1; R12A-16)."""
    from .constraints import CANARY_REPORT_KEYS, JUDGE_REPORT_KEYS, NI_REPORT_KEYS, REPORT_KEYS

    keys = REPORT_KEYS + (("bar",) if result["form"] == "threshold" else ("margin",) + NI_REPORT_KEYS)
    rep = {k: copy.deepcopy(result.get(k)) for k in keys}
    rep["judges"] = {jid: {k: copy.deepcopy(j.get(k)) for k in JUDGE_REPORT_KEYS}
                     for jid, j in (result.get("judges") or {}).items()}
    canaries = result.get("canaries") or {}
    rep["canaries"] = {k: copy.deepcopy(canaries.get(k)) for k in CANARY_REPORT_KEYS}
    return rep


def _fmt(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _reasons(counts: dict[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in counts.items()) or "none"


# MULTIMODAL-SPEC MM-16: printed beside the status when the deciding metric's independence tier is T0 (the scorer
# is the training reward, 05 section 6.1).
T0_NOTE = "independence tier T0: non-blocking under 05 section 6.1; prototype plumbing only"


def _interval_text(iv: dict[str, Any]) -> str:
    """The interval line; R18-3/R18-4 add the resampled clusters and a non-BCa variant, nothing otherwise."""
    boot = iv.get("bootstrap") or {}
    extra = ""
    if boot.get("variant", "bca") != "bca":
        extra += f", {boot['variant']} bootstrap"
    if boot.get("resampling_unit") == "cluster":
        extra += f", resampling {boot['n_clusters']} clusters by {boot['cluster_key']}"
    return f"interval: [{_fmt(iv['lower'])}, {_fmt(iv['upper'])}] level {_fmt(iv['level'])} {iv['method']}{extra}"


def _clause_stat_lines(rep: dict[str, Any]) -> list[str]:
    """The statistics lines of one clause, from `direction` to `candidate_score`."""
    nat = rep["natural_units"]
    plan = rep["plan"]
    return [
        f"direction: {rep['direction']}",
        f"delta_hat: {_fmt(rep['delta_hat'])}",
        _interval_text(rep["interval"]),
        f"natural_units: delta_hat {_fmt(nat['delta_hat'])}, interval [{_fmt(nat['lower'])}, {_fmt(nat['upper'])}] "
        f"(candidate - baseline in metric units)",
        f"p_value: {'<= ' if rep['p_value_at_floor'] else ''}{_fmt(rep['p_value'])} ({rep['p_value_method']})",
        f"margin: {_fmt(rep['margin'])}",
        f"alpha_used: {_fmt(rep['alpha_used'])}",
        f"power_target: {_fmt(rep['power_target'])}",
        f"achieved_power: {_fmt(rep['achieved_power'])}",
        f"mde: {_fmt(rep['mde'])}",
        f"mde_declared: {_fmt(rep['mde_declared'])}",
        f"underpowered: {_fmt(rep['underpowered'])}",
        f"sd_diff: {_fmt(rep['sd_diff'])}",
        f"sd_zero: {_fmt(rep['sd_zero'])}",
        f"n_items: {rep['n_items']}",
        f"unpaired_baseline: {rep['unpaired_baseline']}",
        f"unpaired_candidate: {rep['unpaired_candidate']}",
        f"missing_baseline: {rep['missing_baseline']}",
        f"missing_candidate: {rep['missing_candidate']}",
        f"missing_reasons: baseline {_reasons(rep['missing_reasons']['baseline'])}; "
        f"candidate {_reasons(rep['missing_reasons']['candidate'])}",
        f"max_unpaired: {_fmt(rep['max_unpaired'])}",
        f"min_paired: {rep['min_paired']}",
        f"k_actual: baseline {rep['k_actual']['baseline']}, candidate {rep['k_actual']['candidate']} "
        f"(registered K {rep['samples_per_item']})",
        f"plan: n_required {_fmt(plan['n_required'])}, n_available {plan['n_available']}, "
        f"mde {_fmt(plan['mde'])}, sigma_used {_fmt(plan['sigma_used'])} ({plan['sigma_plan']}), "
        f"underpowered {_fmt(plan['underpowered'])}",
        f"baseline_score: {_fmt(rep['baseline_score'])}",
        f"candidate_score: {_fmt(rep['candidate_score'])}",
    ]


def _registration_lines(rep: dict[str, Any]) -> list[str]:
    return [
        f"registration_hash: {rep['registration_hash']}",
        f"registration_amends: {_fmt(rep['registration_amends'])}",
        *([f"registration_amendment: {rep['registration_amendment']}"]
          if rep.get("registration_amendment") else []),
        f"post_hoc: {_fmt(rep['post_hoc'])}",
        f"registration_signed_by: {rep['registration_signed_by']}",
        f"item_set_hash: {rep['item_set_hash']}",
    ]


def _tail_lines(rep: dict[str, Any], unpaired: bool) -> list[str]:
    lines = []
    if "dataset_revision" in rep:
        lines.append(f"dataset_revision: {rep['dataset_revision']}")
    if "adapter" in rep:
        # External runs (MULTIMODAL-SPEC MM-11, MM-16, MM-17, MM-23, MM-36; tests/README.md section 11.18 EXT-17).
        cont = rep["contamination"]
        lines += [f"adapter: {rep['adapter']}",
                  f"independence_tier: {rep['independence_tier']}",
                  f"gpu_seconds: {rep['gpu_seconds']}",
                  *([f"wall_seconds: {rep['wall_seconds']}"] if "wall_seconds" in rep else []),   # EXT-105
                  f"contamination: exact_overlap={cont['exact_overlap']}",
                  f"contamination_allowed: {_fmt(cont['allowed'])}",
                  f"consumption_precision: {rep['consumption_precision']}",
                  f"incomplete_cuts: {', '.join(rep['incomplete_cuts']) or 'none'}",
                  f"baseline_model_hash: {rep['baseline_model_hash']}",
                  f"candidate_model_hash: {rep['candidate_model_hash']}"]
        # 05 section 6.1 disclosures: R15-1 (tests/README.md EXT-128, EXT-134), R16-1 (EXT-148).
        from .ext.registry import DISCLOSURES
        for key in (d["field"] for d in DISCLOSURES):
            if key in rep:
                ca = rep[key]
                lines.append(f"{key}: {' '.join(str(ca['statement']).split())}")
                if " ".join(str(ca["statement"]).split()) != " ".join(str(ca.get("manifest_statement")).split()):
                    lines.append(f"{key}_manifest: {ca.get('manifest_statement')}")
    lines += [f"{k}: {rep[k]}" for k in NOT_ASSESSED_KEYS if not (k == "independence_tier" and "adapter" in rep
                                                                   and "clauses" in rep)]
    if "disjoint_from" in rep:
        lines.append(f"disjoint_from: {rep['disjoint_from']}")
    if unpaired:
        lines.append(f"warning: {rep['unpaired_baseline']} baseline and {rep['unpaired_candidate']} candidate "
                     "items are unpaired and excluded from the gate and the scores (decision D-3)")
    lines.append(f"stubs: {'; '.join(rep['stubs']) if rep['stubs'] else 'none'}")
    for note in rep.get("notes", []):
        lines.append(f"note: {note}")
    return lines


def render_text(rep: dict[str, Any]) -> str:
    if "clauses" in rep:
        return _render_multi(rep)
    lines = [
        f"tags: {', '.join(rep['tags'])}; stand-ins: {STUB_MARK if rep['stubs'] else 'none'}",
        f"workflow: {rep['workflow']}",
        f"gate_id: {rep['gate_id']}",
        f"hypothesis_type: {rep['hypothesis_type']}",
        f"metric: {rep['metric']}",
        *([f"metric_source: {rep['metric_source']}"] if "metric_source" in rep else []),
        f"status: {rep['status']}" + (f" ({T0_NOTE})" if rep.get("independence_tier") == "T0" else ""),
        f"status_reasons: {', '.join(rep['status_reasons'])}".rstrip(),
        *_clause_stat_lines(rep),
        *_registration_lines(rep),
        f"evaluator_hash: {rep['evaluator_hash']}",
        f"evaluator_hash_registered: {_fmt(rep['evaluator_hash_registered'])}",
        f"sampler_hash: {rep['sampler_hash']}",
        f"baseline_checkpoint_hash: {rep['baseline_checkpoint_hash']}",
        f"candidate_checkpoint_hash: {rep['candidate_checkpoint_hash']}",
        f"lockfile_hash: {rep['lockfile_hash']}",
    ]
    lines += _tail_lines(rep, bool(rep["unpaired_baseline"] or rep["unpaired_candidate"]))
    return "\n".join(lines) + "\n"


def _render_multi(rep: dict[str, Any]) -> str:
    """A gate over several clauses (R18-6): the combined verdict, then one indented block per clause."""
    lines = [
        f"tags: {', '.join(rep['tags'])}; stand-ins: {STUB_MARK if rep['stubs'] else 'none'}",
        f"workflow: {rep['workflow']}",
        f"gate_id: {rep['gate_id']}",
        f"multiplicity: {rep['multiplicity']} (intersection-union: PASS only if every clause passes, each at its "
        "own alpha)",
        f"status: {rep['status']}",
        f"status_reasons: {', '.join(rep['status_reasons'])}".rstrip(),
        f"joint_power_independent: {_fmt(rep['joint_power_independent'])}",
    ]
    for c in rep["clauses"]:
        tier = c.get("independence_tier")
        lines += [f"clause {c['name']}: {c['status']}" + (f" ({T0_NOTE})" if tier == "T0" else ""),
                  *(f"  {line}" for line in [
                      f"evaluator: {c['evaluator']}",
                      f"hypothesis_type: {c['hypothesis_type']}",
                      f"metric: {c['metric']}",
                      *([f"metric_source: {c['metric_source']}"] if "metric_source" in c else []),
                      *([f"independence_tier: {tier}"] if tier is not None else []),
                      f"status_reasons: {', '.join(c['status_reasons'])}".rstrip(),
                      *_clause_stat_lines(c),
                      f"evaluator_hash: {c['evaluator_hash']}",
                      f"evaluator_hash_registered: {_fmt(c['evaluator_hash_registered'])}",
                      *([f"warning: {c['unpaired_baseline']} baseline and {c['unpaired_candidate']} candidate items "
                         "are unpaired and excluded from this clause (decision D-3)"]
                        if c["unpaired_baseline"] or c["unpaired_candidate"] else []),
                  ])]
    lines += [
        *_registration_lines(rep),
        f"sampler_hash: {rep['sampler_hash']}",
        f"baseline_checkpoint_hash: {rep['baseline_checkpoint_hash']}",
        f"candidate_checkpoint_hash: {rep['candidate_checkpoint_hash']}",
        f"lockfile_hash: {rep['lockfile_hash']}",
    ]
    lines += _tail_lines(rep, False)
    return "\n".join(lines) + "\n"
