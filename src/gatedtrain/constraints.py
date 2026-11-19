"""Constraint clauses and blocking judges (05 section 3.9, section 6.1 (a)-(e); design revisions 12 and 13).

Tests: tests/README.md section 11.17 (R12A-1 to R12A-17), tests/test_r12_constraints.py. Record format:
MULTIMODAL-SPEC.md section 1a (MM-C1 to MM-C4).

A constraint clause is a threshold or non-inferiority clause with `constraint_set: <id>` (05 section 3.1,
section 3.9; R12-1, R12-4). Its metric is the per-item violation indicator v_i (1 when any repeat present
violates any registered constraint, `direction: lower_is_better`); the constraint texts, checkers, judges'
scorer fields and canaries are the registered constraint set, whose canonical-JSON SHA-256 is the evaluator
(scorer) hash (`constraint_set_hash`, prototype choice P9).

- Threshold form (05 section 3.9, R12-2; prototype choice P1: `type: superiority` with `bar`): one arm;
  PASS when the exact one-sided upper (1 - alpha) Clopper-Pearson bound of k / n is below `bar`. Power and
  n from the binomial at the planning violation rate `bar - planning_effect` (`threshold_power`,
  `binomial_n_required`).
- Non-inferiority form (05 section 3.9 as revised by R13-1 and R13-2): paired on items over the repeats
  present in both arms; c counts items where only the candidate violates, b the reverse, n complete pairs.
  `interval: clopper_pearson_discordant`: PASS when the Clopper-Pearson upper (1 - alpha) bound of c / n
  is below delta (conservative: (c - b) / n <= c / n); positive width at c = 0. Power and n from the
  binomial as in the threshold form with delta for `bar` and the planning rate of c / n. Prototype choice
  (R13-2 names "a registered planning rate of c / n" without a field): the rate is `margin -
  planning_effect`, the threshold form's construction with delta in place of `bar`, so no clause field is
  added; `planning_effect` lies in (0, margin].
- Judges (05 section 6.1 (a)-(e), R12-3, R13-3, R13-4): `judge_qualification` checks validity (miss and
  false-alarm rates with Clopper-Pearson upper bounds at the clause's alpha), independence (prototype
  choice P5: the inputs are declared under the registration's `judge_qualification`), registered canaries,
  reproducibility and the external-endpoint rules. A judge on a blocking constraint clause blocks alone
  (each constraint has one deciding checker); one failing a condition is refused at registration
  (REG-JUDGE) and makes the clause MISSING at run time (`judge_not_qualified`). Per-batch canaries, and
  MISSING on a missed or absent canary, apply only to such a judge (R13-4); a supporting judge (non-blocking
  clause) is reported without them. An external judge needs a recorded egress approval, blocking or not
  (REG-EGRESS; MISSING `egress_unapproved` at run time; 06 section 9, B20).

`items` is required, `expected_models` checks the arms' model hashes, batch
checks run per judge (`malformed_batch` for a batch of an unregistered judge or a record batched twice by one
judge), a registered canary must violate a judge-checked constraint, floating model versions are denied,
families compare case-insensitively, and an external judge's arms must carry the registered approval id.

The gate reads recorded verdicts only (MM-C1 to MM-C4); nothing here calls a model or opens a socket.
Function and key names are the maintainers' choice.
"""

from __future__ import annotations

import functools
import math
import re
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import beta, binom

from .canonical import canonical_json, sha256_bytes
from .errors import Problem
from .gates import FAIL, INCONCLUSIVE, MISSING, PASS, _clause_k, _screen_arm

SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
FORMS = {"superiority": "threshold", "non_inferiority": "non_inferiority"}
INTERVALS = {"threshold": "clopper_pearson", "non_inferiority": "clopper_pearson_discordant"}
CONSTRAINT_SET_KEYS = ("id", "constraints", "judges", "canaries")
JUDGE_FIELDS = ("model_id", "model_version", "family", "llm_judge", "endpoint", "version_pinnable",
                "prompt_template_sha256", "decoding", "parser", "response_cache")
ENDPOINTS = ("local", "external")
QUALIFICATION_KEYS = ("policy_family", "training_reward_families", "judges")
JUDGE_QUALIFICATION_KEYS = ("tier", "transfer_test", "validity", "egress_approval")
BLOCKING_TIERS = ("T2", "T3", "T4")                # 05 section 6.1 table: T2 or higher
RESPONSE_CACHE = "JudgeResponseCache"              # 05 section 6.1 (d)
FAILED_CODES = ("a_validity", "b_independence", "c_canaries", "d_reproducibility", "e_external")
# 05 section 6.1 (d): floating aliases that name no fixed model version (R12A-22), compared after stripping
# and case folding; a version that merely contains one (`latest-2025-01-28`) is pinned.
FLOATING_VERSIONS = ("latest", "current", "stable", "default", "newest")
N_MAX = 1_000_000                                  # search limit of binomial_n_required


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _text(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _norm(x: Any) -> Any:
    """A backbone family or version string for comparison: stripped and case folded (R12A-22)."""
    return x.strip().casefold() if isinstance(x, str) else x


def is_constraint_clause(clause: Any) -> bool:
    """A clause with a non-null `constraint_set` (05 section 3.1)."""
    return isinstance(clause, dict) and clause.get("constraint_set") is not None


def clause_form(clause: dict) -> str | None:
    """`threshold` (type superiority, P1) or `non_inferiority`; None for any other type."""
    return FORMS.get(clause.get("type"))


# ---------------------------------------------------------------------------
# Exact binomial bound, power and n (05 section 3.9)
# ---------------------------------------------------------------------------

def clopper_pearson_upper(k: int, n: int, alpha: float) -> float:
    """Exact one-sided upper (1 - alpha) Clopper-Pearson bound of k / n: Beta^-1(1 - alpha; k + 1, n - k),
    1 when k = n. Zero events give 1 - alpha^(1/n)."""
    if n < 1 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n and n >= 1, got k={k}, n={n}")
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, k + 1, n - k))


def _k_max(n: int, bar: float, alpha: float, start: int = -1) -> int:
    """The largest k whose upper bound is below `bar` (-1 when none); the bound grows with k."""
    k = start
    while k + 1 < n and clopper_pearson_upper(k + 1, n, alpha) < bar:
        k += 1
    while k >= 0 and clopper_pearson_upper(k, n, alpha) >= bar:
        k -= 1
    return k


def threshold_power(*, n: int, bar: float, planning_rate: float, alpha: float) -> float:
    """P(upper bound < bar) with violations ~ Bin(n, planning_rate) (05 section 3.9 binomial power)."""
    # Start from the dual exact test (bound < bar iff P(X <= k; n, bar) < alpha), then settle on the bound.
    start = int(np.count_nonzero(binom.cdf(np.arange(n + 1), n, bar) < alpha)) - 1
    k = _k_max(n, bar, alpha, start=min(start, n - 1))
    return float(binom.cdf(k, n, planning_rate)) if k >= 0 else 0.0


@functools.lru_cache(maxsize=256)
def binomial_n_required(*, bar: float, planning_rate: float, alpha: float, power: float) -> int | None:
    """The smallest n whose binomial power at `planning_rate` reaches `power` (05 section 3.9); None when
    the planning rate is not below `bar` or no n up to N_MAX reaches it."""
    if not (0 < bar < 1 and 0 <= planning_rate < bar and 0 < alpha < 1 and 0 < power < 1):
        return None
    # Below n0 not even zero violations give a bound under `bar` (1 - alpha^(1/n) >= bar).
    n0 = max(1, int(math.floor(math.log(alpha) / math.log1p(-bar))))
    k = -1
    for n in range(n0, N_MAX + 1):
        k = _k_max(n, bar, alpha, start=k)
        if k >= 0 and binom.cdf(k, n, planning_rate) >= power:
            return n
    return None


# ---------------------------------------------------------------------------
# Constraint set (05 section 3.9) and scorer hash (R12A-3, P9)
# ---------------------------------------------------------------------------

def constraint_set_hash(cs: dict) -> str:
    """"sha256:" + SHA-256 of the RFC 8785 canonical JSON of the whole constraint set (P9)."""
    return sha256_bytes(canonical_json(cs))


def _safe_hash(cs: Any) -> str | None:
    try:
        return constraint_set_hash(cs)
    except (ValueError, TypeError):
        return None


def violation_indicators(records: list) -> dict[Any, int]:
    """v_i = max over the item's usable repeats (05 section 3.9): records with an `error`, or a score other
    than 0 or 1, are not usable; an item with no usable repeat has no indicator. `sub_metrics` are ignored."""
    out: dict[Any, int] = {}
    for rec in records:
        if not isinstance(rec, dict) or rec.get("error") is not None:
            continue
        s = rec.get("score")
        if isinstance(s, bool) or not _num(s) or s not in (0, 1):
            continue
        out[rec["item"]] = max(out.get(rec["item"], 0), int(s))
    return out


# ---------------------------------------------------------------------------
# Judge qualification (05 section 6.1 (a)-(e))
# ---------------------------------------------------------------------------

def _judge_findings(judge_id: str, cs: dict, q: Any, alpha: float) -> tuple[list[tuple[str, str, str, str]], dict]:
    """Failing conditions as (code, rule, field, message), field relative to `registration.`, and the
    validity numbers. `rule` is REG-EGRESS for a missing egress approval, else REG-JUDGE."""
    judge = ((cs.get("judges") or {}) if isinstance(cs, dict) else {}).get(judge_id)
    judge = judge if isinstance(judge, dict) else {}
    q = q if isinstance(q, dict) else {}
    qj = (q.get("judges") or {}).get(judge_id) if isinstance(q.get("judges"), dict) else None
    qj = qj if isinstance(qj, dict) else {}
    jq_field = f"judge_qualification.judges.{judge_id}"
    cs_field = f"constraint_set.judges.{judge_id}"
    found: list[tuple[str, str, str, str]] = []

    # (a) validity: rates and exact upper bounds at the clause's alpha (R13-3).
    v = qj.get("validity")
    nums: dict[str, Any] = {"miss_rate": None, "miss_rate_upper": None, "false_alarm_rate": None,
                            "false_alarm_upper": None, "max_miss_rate": None, "max_false_alarm_rate": None}
    if not isinstance(v, dict):
        found.append(("a_validity", "REG-JUDGE", f"{jq_field}.validity",
                      "no validity results against human labels (05 section 6.1 (a))"))
    else:
        lv, misses = v.get("labelled_violations"), v.get("misses")
        lc, fa = v.get("labelled_compliant"), v.get("false_alarms")
        mmr, mfar = v.get("max_miss_rate"), v.get("max_false_alarm_rate")
        nums["max_miss_rate"] = mmr if _num(mmr) else None
        nums["max_false_alarm_rate"] = mfar if _num(mfar) else None
        ok_miss = _int(lv) and lv >= 1 and _int(misses) and 0 <= misses <= lv
        ok_fa = _int(lc) and lc >= 1 and _int(fa) and 0 <= fa <= lc
        if ok_miss:
            nums["miss_rate"] = misses / lv
            nums["miss_rate_upper"] = clopper_pearson_upper(misses, lv, alpha)
        if ok_fa:
            nums["false_alarm_rate"] = fa / lc
            nums["false_alarm_upper"] = clopper_pearson_upper(fa, lc, alpha)
        problems = []
        if not (ok_miss and ok_fa):
            problems.append("labelled_violations, misses, labelled_compliant and false_alarms must be counts "
                            "with misses <= labelled_violations and false_alarms <= labelled_compliant")
        if not (_num(mmr) and 0 < mmr < 1):
            problems.append("max_miss_rate must be a number in (0, 1)")
        if "max_false_alarm_rate" in v and mfar is not None and not (_num(mfar) and 0 < mfar < 1):
            problems.append("max_false_alarm_rate must be a number in (0, 1) or absent")
        if not _text(v.get("label_source")):
            problems.append("label_source must name the 05 section 8 annotation")
        if ok_miss and _num(mmr) and nums["miss_rate_upper"] > mmr:
            problems.append(f"miss-rate upper bound {nums['miss_rate_upper']:.4f} exceeds max_miss_rate {mmr}")
        if ok_fa and _num(mfar) and nums["false_alarm_upper"] > mfar:
            problems.append(f"false-alarm upper bound {nums['false_alarm_upper']:.4f} exceeds "
                            f"max_false_alarm_rate {mfar}")
        if problems:
            found.append(("a_validity", "REG-JUDGE", f"{jq_field}.validity",
                          "; ".join(problems) + " (05 section 6.1 (a))"))

    # (b) independence (P5: declared inputs).
    if qj.get("tier") not in BLOCKING_TIERS:
        found.append(("b_independence", "REG-JUDGE", f"{jq_field}.tier",
                      f"tier {qj.get('tier')!r} is not T2 or higher (05 section 6.1 (b))"))
    if not _text(qj.get("transfer_test")):
        found.append(("b_independence", "REG-JUDGE", f"{jq_field}.transfer_test",
                      "no transfer-test reference (05 section 6.1 (b), section 6.2)"))
    family = judge.get("family")
    policy = q.get("policy_family")
    rewards = q.get("training_reward_families")
    if not _text(policy):
        found.append(("b_independence", "REG-JUDGE", "judge_qualification.policy_family",
                      "the policy's backbone family is not declared (05 section 6.1 (b))"))
    if not (isinstance(rewards, list) and all(_text(r) for r in rewards)):
        found.append(("b_independence", "REG-JUDGE", "judge_qualification.training_reward_families",
                      "the training rewards' backbone families are not declared (05 section 6.1 (b))"))
    if not _text(family):
        found.append(("b_independence", "REG-JUDGE", f"{cs_field}.family",
                      "the judge's backbone family is not declared (05 section 6.1 (b))"))
    elif _norm(family) == _norm(policy) or (isinstance(rewards, list)
                                            and _norm(family) in {_norm(r) for r in rewards}):
        found.append(("b_independence", "REG-JUDGE", f"{cs_field}.family",
                      f"the judge's backbone family {family!r} is the policy's or a training reward's "
                      "(05 section 6.1 (b))"))

    # (c) a registered canary set (05 section 6.1 (c)).
    canaries = cs.get("canaries") if isinstance(cs, dict) else None
    if not (isinstance(canaries, list) and canaries):
        found.append(("c_canaries", "REG-JUDGE", "constraint_set.canaries",
                      "no registered injection canaries (05 section 6.1 (c))"))

    # (d) reproducibility.
    decoding = judge.get("decoding")
    temp = decoding.get("temperature") if isinstance(decoding, dict) else None
    if not (_num(temp) and temp == 0):
        found.append(("d_reproducibility", "REG-JUDGE", f"{cs_field}.decoding",
                      f"decoding temperature {temp!r} is not 0 (05 section 6.1 (d))"))
    mv = judge.get("model_version")
    if not _text(mv) or _norm(mv) in FLOATING_VERSIONS:
        found.append(("d_reproducibility", "REG-JUDGE", f"{cs_field}.model_version",
                      f"model version {mv!r} is not pinned to a version or date (05 section 6.1 (d))"))
    if judge.get("response_cache") != RESPONSE_CACHE:
        found.append(("d_reproducibility", "REG-JUDGE", f"{cs_field}.response_cache",
                      f"responses are not cached in {RESPONSE_CACHE} (05 section 6.1 (d))"))

    # (e) external judges.
    if judge.get("endpoint") == "external":
        if not egress_approved(qj.get("egress_approval")):
            found.append(("e_external", "REG-EGRESS", f"{jq_field}.egress_approval",
                          "an external judge receives candidate outputs: an `upload` destination without a "
                          "recorded egress approval (06 section 9, B20; 05 section 6.1 (e))"))
        if judge.get("version_pinnable") is not True:
            found.append(("e_external", "REG-JUDGE", f"{cs_field}.version_pinnable",
                          "the external judge's model version cannot be pinned, so it may only support "
                          "(05 section 6.1 (e))"))
    return found, nums


def egress_approved(approval: Any) -> bool:
    """`{approval_id, destination: upload, data_tags}` with a non-empty id and tags (06 section 9, B20)."""
    return (isinstance(approval, dict) and _text(approval.get("approval_id"))
            and approval.get("destination") == "upload"
            and isinstance(approval.get("data_tags"), list) and bool(approval["data_tags"])
            and all(_text(t) for t in approval["data_tags"]))


def judge_qualification(judge_id: str, cs: dict, q: dict | None, *, alpha: float) -> dict[str, Any]:
    """05 section 6.1 (a)-(e) for one judge of the constraint set: `can_block_alone`, `failed` (codes of
    FAILED_CODES), the miss and false-alarm rates with their exact upper bounds at `alpha` (the constraint
    clause's alpha, R13-3), and the judge's identity fields for the report."""
    findings, nums = _judge_findings(judge_id, cs, q, alpha)
    judge = (cs.get("judges") or {}).get(judge_id) if isinstance(cs, dict) else None
    judge = judge if isinstance(judge, dict) else {}
    qj = ((q or {}).get("judges") or {}).get(judge_id) if isinstance(q, dict) else None
    qj = qj if isinstance(qj, dict) else {}
    approval = qj.get("egress_approval")
    failed = [c for c in FAILED_CODES if any(f[0] == c for f in findings)]
    return {
        "judge": judge_id, "model_id": judge.get("model_id"), "model_version": judge.get("model_version"),
        "endpoint": judge.get("endpoint"), "family": judge.get("family"), "tier": qj.get("tier"),
        "transfer_test": qj.get("transfer_test"), **nums, "alpha": alpha,
        "can_block_alone": not failed, "failed": failed,
        "egress_approval_id": approval.get("approval_id") if isinstance(approval, dict) else None,
        "findings": [{"code": c, "rule": r, "field": f"registration.{fld}", "message": m}
                     for c, r, fld, m in findings],
    }


# ---------------------------------------------------------------------------
# Registration (R12A-1, R12A-2, R12A-3, R12A-9 to R12A-14)
# ---------------------------------------------------------------------------

def check_registration(reg: dict) -> list[Problem]:
    """Constraint-clause checks of a registration; `registration.check_contents` adds them. Rules:
    REG-CONSTRAINT (constraint set or constraint clause malformed), REG-JUDGE (a judge on a blocking constraint
    clause fails 05 section 6.1 (a)-(e)), REG-EGRESS (an external judge without a recorded approval)."""
    p: list[Problem] = []

    def bad(rule: str, field: str, message: str, fix: str) -> None:
        p.append(Problem(rule, f"registration.{field}", message, fix))

    clause = reg.get("clause") if isinstance(reg.get("clause"), dict) else None
    if not is_constraint_clause(clause):
        for key in ("constraint_set", "judge_qualification"):
            if key in reg:
                bad("REG-CONSTRAINT", key, f"`{key}` is registered but the clause names no constraint set "
                    "(05 section 3.9)", f"set `clause.constraint_set` to the set's id, or remove `{key}`")
        return p

    # --- the clause (R12A-1) ---
    form = clause_form(clause)
    blocking = clause.get("blocking") is True
    if form is None:
        bad("REG-CONSTRAINT", "clause.type", f"a constraint clause is a threshold (`superiority` with `bar`) "
            f"or `non_inferiority` clause, not {clause.get('type')!r} (05 section 3.9)",
            "use `type: superiority` with `bar`, or `type: non_inferiority` with `margin`")
    if clause.get("direction") != "lower_is_better":
        bad("REG-CONSTRAINT", "clause.direction", "the violation indicator is lower-is-better (05 section 3.9)",
            "set `direction: lower_is_better`")
    if form is not None and clause.get("interval") != INTERVALS[form]:
        bad("REG-CONSTRAINT", "clause.interval",
            f"the {form} form uses `interval: {INTERVALS[form]}`, not {clause.get('interval')!r} "
            "(05 section 3.9, R13-1)", f"set `interval: {INTERVALS[form]}`")
    pe = clause.get("planning_effect")
    if form == "threshold":
        bar = clause.get("bar")
        if not (_num(bar) and 0 < bar < 1):
            bad("REG-CONSTRAINT", "clause.bar", f"bar {bar!r} is not a violation rate in (0, 1) "
                "(05 section 3.9)", "set `bar`, the highest acceptable violation rate, for example 0.01")
        elif _num(pe) and not 0 < pe <= bar:
            bad("REG-CONSTRAINT", "clause.planning_effect",
                f"planning_effect {pe} puts the planning violation rate bar - planning_effect outside [0, bar) "
                "(05 section 3.9)", f"set planning_effect in (0, {bar}]; {bar} plans at zero violations")
        if clause.get("margin") is not None:
            bad("REG-CONSTRAINT", "clause.margin", "the threshold form uses `bar`, not `margin` (04 section 4.2, P1)",
                "remove `margin`")
    elif form == "non_inferiority":
        margin = clause.get("margin")
        if margin is not None and not (_num(margin) and 0 < margin < 1):
            bad("REG-CONSTRAINT", "clause.margin", f"margin {margin!r} is not a violation-rate increase in (0, 1) "
                "(05 section 3.9)", "set delta, for example 0.02")
        elif _num(margin) and _num(pe) and not 0 < pe <= margin:
            bad("REG-CONSTRAINT", "clause.planning_effect",
                f"planning_effect {pe} puts the planning rate of c / n, margin - planning_effect, outside "
                "[0, margin) (05 section 3.9, R13-2)",
                f"set planning_effect in (0, {margin}]; {margin} plans at no candidate-only violations")
        if "bar" in clause:
            bad("REG-CONSTRAINT", "clause.bar", "the non-inferiority form uses `margin`, not `bar` (05 section 3.9)",
                "remove `bar`")

    # --- the constraint set (R12A-2) ---
    cs = reg.get("constraint_set")
    if not isinstance(cs, dict) or cs.get("id") != clause.get("constraint_set"):
        bad("REG-CONSTRAINT", "constraint_set",
            f"the registration holds no constraint set with id {clause.get('constraint_set')!r} "
            "(05 section 3.7, section 3.9)", "register the constraint set the clause names")
        return p
    for key in sorted(set(cs) - set(CONSTRAINT_SET_KEYS), key=str):
        bad("REG-CONSTRAINT", f"constraint_set.{key}", "unknown constraint-set field",
            f"remove `{key}`; allowed: {', '.join(CONSTRAINT_SET_KEYS)}")
    judges = cs.get("judges", {})
    if not isinstance(judges, dict):
        bad("REG-CONSTRAINT", "constraint_set.judges", "judges must be a mapping of judge id to scorer fields",
            "write `judges: {<id>: {model_id, model_version, ...}}` or `{}`")
        judges = {}
    constraints = cs.get("constraints")
    ids: list[str] = []
    if not (isinstance(constraints, list) and constraints and all(isinstance(c, dict) for c in constraints)
            and [c.get("id") for c in constraints] == [f"c{i + 1}" for i in range(len(constraints))]):
        bad("REG-CONSTRAINT", "constraint_set.constraints",
            "constraints must be a non-empty list numbered c1, c2, ... in order (05 section 3.9)",
            "number the constraints c1, c2, ...")
    if isinstance(constraints, list):
        for i, c in enumerate(constraints):
            if not isinstance(c, dict):
                continue
            ids.append(c.get("id"))
            if not _text(c.get("text")):
                bad("REG-CONSTRAINT", f"constraint_set.constraints[{i}].text", "empty constraint text",
                    "write the constraint (\"the output must not ...\")")
            _check_checker(c.get("checker"), i, judges, bad)
    for jid, judge in judges.items():
        _check_judge(jid, judge, bad)
    judge_checked = {c.get("id") for c in constraints or [] if isinstance(c, dict)
                     and isinstance(c.get("checker"), dict) and c["checker"].get("kind") == "judge"} \
        if isinstance(constraints, list) else set()
    canaries = cs.get("canaries", [])
    if not isinstance(canaries, list):
        bad("REG-CONSTRAINT", "constraint_set.canaries", "canaries must be a list",
            "write `canaries: [{id, violates, output_sha256, injection}]`")
        canaries = []
    seen: set = set()
    for i, k in enumerate(canaries):
        ok = (isinstance(k, dict) and _text(k.get("id")) and k["id"] not in seen
              and isinstance(k.get("violates"), list) and k["violates"]
              and all(v in ids for v in k["violates"])
              and isinstance(k.get("output_sha256"), str) and SHA_RE.match(k["output_sha256"])
              and _text(k.get("injection")) and set(k) <= {"id", "violates", "output_sha256", "injection"})
        if isinstance(k, dict) and _text(k.get("id")):
            seen.add(k["id"])
        if not ok:
            bad("REG-CONSTRAINT", f"constraint_set.canaries[{i}]",
                "a canary is {id (unique), violates: [constraint ids], output_sha256: sha256:<64 hex>, "
                "injection} (05 section 6.1 (c))", "fix the canary entry")
        elif blocking and judges and not set(k["violates"]) & judge_checked:
            # R12A-21: a canary that violates only tool-checked constraints never tests a judge.
            bad("REG-JUDGE", f"constraint_set.canaries[{i}]",
                f"canary {k['id']!r} violates only tool-checked constraints {k['violates']}: it never tests the "
                "judge that blocks the clause (05 section 6.1 (c))",
                "make the canary violate a judge-checked constraint")

    # --- scorer hash (R12A-3) ---
    evaluators = reg.get("evaluators")
    mh = (evaluators[0].get("manifest_hash") if isinstance(evaluators, list) and evaluators
          and isinstance(evaluators[0], dict) else None)
    h = _safe_hash(cs)
    if h is None:
        bad("REG-CONSTRAINT", "constraint_set", "the constraint set has no canonical JSON form",
            "use strings, integers, finite numbers, booleans, null, lists and mappings")
    elif isinstance(mh, str) and SHA_RE.match(mh) and mh != h:
        bad("REG-CONSTRAINT", "evaluators[0].manifest_hash",
            f"the evaluator (scorer) hash {mh} is not the constraint-set hash {h} (05 section 3.9, R12A-3)",
            f"set `manifest_hash: \"{h}\"`, or restore the registered constraint set")

    # --- judges (R12A-9 to R12A-14) ---
    q = reg.get("judge_qualification")
    if judges:
        q_missing = not isinstance(q, dict)
        if q_missing:
            bad("REG-CONSTRAINT", "judge_qualification",
                "a constraint set with judges needs `judge_qualification` (05 section 6.1, P5)",
                "add {policy_family, training_reward_families, judges: {<id>: {tier, transfer_test, validity, "
                "egress_approval}}}")
            q = {}
        else:
            for key in sorted(set(q) - set(QUALIFICATION_KEYS), key=str):
                bad("REG-CONSTRAINT", f"judge_qualification.{key}", "unknown judge_qualification field",
                    f"remove `{key}`; allowed: {', '.join(QUALIFICATION_KEYS)}")
            qjs = q.get("judges") if isinstance(q.get("judges"), dict) else {}
            for jid, entry in qjs.items():
                if jid not in judges:
                    bad("REG-CONSTRAINT", f"judge_qualification.judges.{jid}", "not a judge of the constraint set",
                        f"remove `{jid}`")
                elif isinstance(entry, dict):
                    for key in sorted(set(entry) - set(JUDGE_QUALIFICATION_KEYS), key=str):
                        bad("REG-CONSTRAINT", f"judge_qualification.judges.{jid}.{key}",
                            "unknown judge qualification field",
                            f"remove `{key}`; allowed: {', '.join(JUDGE_QUALIFICATION_KEYS)}")
        alpha = clause.get("alpha") if _num(clause.get("alpha")) and 0 < clause["alpha"] < 1 else 0.025
        reported: set[tuple[str, str]] = set()
        for jid in judges:
            findings, _ = _judge_findings(jid, cs, q, alpha)
            for code, rule, field, message in findings:
                if rule == "REG-JUDGE" and (not blocking or q_missing):
                    continue                                   # a supporting judge (05 section 6.1, R13-4)
                if (rule, field) in reported:
                    continue
                reported.add((rule, field))
                fix = ("record the approval {approval_id, destination: upload, data_tags} (06 section 9)"
                       if rule == "REG-EGRESS" else
                       "qualify the judge, use a programmatic checker, or make the clause non-blocking "
                       "(the judge then only supports)")
                bad(rule, field, f"judge {jid}: {message}", fix)
    elif q is not None and not isinstance(q, dict):
        bad("REG-CONSTRAINT", "judge_qualification", "judge_qualification must be a mapping",
            "remove it or write the mapping")
    return p


def _check_checker(checker: Any, i: int, judges: dict, bad) -> None:
    field = f"constraint_set.constraints[{i}].checker"
    if not isinstance(checker, dict) or checker.get("kind") not in ("tool", "judge"):
        bad("REG-CONSTRAINT", field, "the checker is {kind: tool, tool, manifest_hash} or "
            "{kind: judge, judge, no_tool_reason} (05 section 3.9)", "name one checker for the constraint")
        return
    if checker["kind"] == "tool":
        mh = checker.get("manifest_hash")
        if not (_text(checker.get("tool")) and isinstance(mh, str) and SHA_RE.match(mh)
                and set(checker) <= {"kind", "tool", "manifest_hash"}):
            bad("REG-CONSTRAINT", field, "a tool checker names the tool and its manifest hash (sha256:<64 hex>)",
                "set `tool` and `manifest_hash`")
        return
    if checker.get("judge") not in judges:
        bad("REG-CONSTRAINT", f"{field}.judge", f"judge {checker.get('judge')!r} is not in constraint_set.judges",
            "name a registered judge")
    if not _text(checker.get("no_tool_reason")):
        bad("REG-CONSTRAINT", f"{field}.no_tool_reason",
            "a judge checks only constraints no programmatic tool can (05 section 3.9, P7)",
            "state why no tool checks this constraint, or use a tool checker")
    for key in sorted(set(checker) - {"kind", "judge", "no_tool_reason"}, key=str):
        bad("REG-CONSTRAINT", f"{field}.{key}", "unknown checker field", f"remove `{key}`")


def _check_judge(jid: Any, judge: Any, bad) -> None:
    """Scorer fields of one judge (R12A-2). Values that decide qualification (decoding temperature, model
    version, response cache, version pinning) are judged by `_judge_findings`, not here."""
    field = f"constraint_set.judges.{jid}"
    if not isinstance(judge, dict):
        bad("REG-CONSTRAINT", field, "a judge is a mapping of scorer fields", f"give {', '.join(JUDGE_FIELDS)}")
        return
    for f in JUDGE_FIELDS:
        if f not in judge:
            bad("REG-CONSTRAINT", f"{field}.{f}", "missing judge scorer field (05 section 3.9)", f"add `{f}`")
    for f in sorted(set(judge) - set(JUDGE_FIELDS), key=str):
        bad("REG-CONSTRAINT", f"{field}.{f}", "unknown judge field", f"remove `{f}`")
    if "model_id" in judge and not _text(judge["model_id"]):
        bad("REG-CONSTRAINT", f"{field}.model_id", "empty model id", "name the judge model")
    if "family" in judge and not _text(judge["family"]):
        bad("REG-CONSTRAINT", f"{field}.family", "empty backbone family", "name the judge's backbone family")
    for f in ("llm_judge", "version_pinnable"):
        if f in judge and not isinstance(judge[f], bool):
            bad("REG-CONSTRAINT", f"{field}.{f}", f"{f} must be true or false", f"set `{f}`")
    if "endpoint" in judge and judge["endpoint"] not in ENDPOINTS:
        bad("REG-CONSTRAINT", f"{field}.endpoint", f"endpoint {judge['endpoint']!r} is not local or external",
            "set `endpoint: local` or `external`")
    pt = judge.get("prompt_template_sha256")
    if "prompt_template_sha256" in judge and not (isinstance(pt, str) and SHA_RE.match(pt)):
        bad("REG-CONSTRAINT", f"{field}.prompt_template_sha256", "not sha256:<64 hex>", "record the template hash")
    parser = judge.get("parser")
    if "parser" in judge and not (isinstance(parser, dict) and _text(parser.get("name"))
                                  and isinstance(parser.get("sha256"), str) and SHA_RE.match(parser["sha256"])):
        bad("REG-CONSTRAINT", f"{field}.parser", "the parser is {name, sha256: sha256:<64 hex>}",
            "name the verdict parser and its hash")
    if "decoding" in judge and not isinstance(judge["decoding"], dict):
        bad("REG-CONSTRAINT", f"{field}.decoding", "decoding must be a mapping", "give {temperature: 0, ...}")


# ---------------------------------------------------------------------------
# Run time (R12A-4 to R12A-8, R12A-11, R12A-13, R12A-14, R12A-16, R12A-17)
# ---------------------------------------------------------------------------

def _screen(records: list, *, index: dict, k: int, seeds: list) -> dict[str, Any]:
    """`gates._screen_arm` with range [0, 1], then a score other than 0 or 1 is `out_of_range` (P8)."""
    s = _screen_arm(records, index=index, k=k, seeds=seeds, ratio=False, score_range=(0.0, 1.0))
    reasons = defaultdict(int, s["reasons"])
    for key in [kr for kr, rec in s["usable"].items() if rec["score"] not in (0, 1)]:
        del s["usable"][key]
        reasons["out_of_range"] += 1
    s["reasons"] = dict(sorted(reasons.items()))
    s["missing"] = len(index) * k - len(s["usable"])
    return s


def _record_keys(arm: dict) -> tuple[set, int]:
    """The (item, repeat) keys of an arm's judged records, and the count of records without a usable key."""
    keys, bad = set(), 0
    for rec in arm.get("records") or []:
        if isinstance(rec, dict):
            try:
                keys.add((rec.get("item"), rec.get("repeat")))
            except TypeError:
                bad += 1
    return keys, bad


def _canaries(arm: dict, registered: list[str], judge_ids: list[str]) -> dict[str, Any]:
    """MM-C2 batch checks of one arm, judge by judge (R12A-21): batches without a registered canary,
    judged records outside every batch of a judge (every judge checks every record), registered canaries
    the judge did not flag (`flagged` other than `true`), and malformed batches (a batch whose `judge` is not
    a registered judge, a record entry that is not [item, repeat], or a record in two batches of one judge)."""
    batches = arm.get("batches") or []
    per_judge = {jid: {"batched": set(), "without": [], "missed": [], "checked": 0} for jid in judge_ids}
    malformed: list[Any] = []
    for b in batches:
        jid = b.get("judge") if isinstance(b, dict) else None
        if not (isinstance(jid, str) and jid in per_judge):
            malformed.append(b.get("batch") if isinstance(b, dict) else None)
            continue
        pj = per_judge[jid]
        for rec in b.get("records") or []:
            try:
                key = (rec[0], rec[1]) if isinstance(rec, (list, tuple)) and len(rec) == 2 else None
                duplicate = key is None or key in pj["batched"]
            except TypeError:
                duplicate = True
            if duplicate:
                malformed.append(b.get("batch"))
            else:
                pj["batched"].add(key)
        regs = [c for c in b.get("canaries") or [] if isinstance(c, dict) and c.get("canary") in registered]
        if not regs:
            pj["without"].append(b.get("batch"))
        pj["checked"] += len(regs)
        pj["missed"] += [{"batch": b.get("batch"), "canary": c["canary"]} for c in regs if c.get("flagged") is not True]
    keys, unkeyed = _record_keys(arm)
    unbatched = sum(len(keys - pj["batched"]) + unkeyed for pj in per_judge.values())
    return {"batches": len(batches), "checked": sum(pj["checked"] for pj in per_judge.values()),
            "missed": [m for pj in per_judge.values() for m in pj["missed"]],
            "batches_without_canary": [x for pj in per_judge.values() for x in pj["without"]],
            "unbatched_records": unbatched, "malformed_batches": malformed}


def _bound_exceeded(missing: int, expected: int, max_unpaired: int | float) -> bool:
    """05 section 3.4: a count bounds the total missing items over all arms; a fraction in [0, 1) bounds
    that total over the expected items of all arms (`expected`)."""
    if isinstance(max_unpaired, float):
        return missing > max_unpaired * expected
    return missing > max_unpaired


def evaluate_constraint(clause: dict, constraint_set: dict, judge_qualification: dict | None, candidate: dict,
                        baseline: dict | None = None, *, items, min_paired: int = 10, max_unpaired: int | float = 0,
                        expected_models: dict[str, str | None] | None = None) -> dict[str, Any]:
    """The constraint clause on recorded verdicts (MM-C4). An arm is `{records, batches, constraint_set_hash,
    model_manifest_hash, egress_approval_id}` (MM-C3). `items` is the registered item set and is required:
    the expected items are `items` x K per arm, never the items found in the records (R12A-18; a repeated ID
    raises ValueError). `expected_models` is `{arm: model manifest hash}` as in `gates.evaluate_items`: an
    arm whose hash differs, is null or absent is MISSING `model_hash_mismatch` (R12A-19); None, no check.
    Returns the clause statistics, `judges` (per judge, `judge_qualification`) and `canaries`."""
    form = clause_form(clause)
    if form is None:
        raise ValueError(f"a constraint clause is superiority (threshold) or non_inferiority, not "
                         f"{clause.get('type')!r}")
    if form == "threshold" and baseline is not None:
        raise ValueError("the threshold form has one arm; pass no baseline")
    if form == "non_inferiority" and baseline is None:
        raise ValueError("the non-inferiority form is paired against a baseline arm")
    alpha, power_target = float(clause["alpha"]), float(clause["power"])
    bound_param = float(clause["bar"] if form == "threshold" else clause["margin"])
    planning_rate = bound_param - float(clause["planning_effect"])
    blocking = clause.get("blocking") is True
    k = _clause_k(clause)
    seeds = list(clause["eval_seeds"])
    arms = {"candidate": candidate} if form == "threshold" else {"candidate": candidate, "baseline": baseline}

    item_list = list(items)
    index = {it: i for i, it in enumerate(item_list)}
    if len(index) != len(item_list):
        raise ValueError("`items` repeats an item ID; pass the registered item set")
    screened = {name: _screen(list((a or {}).get("records") or []), index=index, k=k, seeds=seeds)
                for name, a in arms.items()}

    # Clause-level MISSING reasons (05 section 3.8).
    missing_reasons: list[str] = []
    missing_total = sum(s["missing"] for s in screened.values())
    flags = set().union(*(s["flags"] for s in screened.values()))
    expected_hash = _safe_hash(constraint_set)
    if any((a or {}).get("constraint_set_hash") != expected_hash or expected_hash is None for a in arms.values()):
        missing_reasons.append("evaluator_hash_mismatch")
    if _bound_exceeded(missing_total, len(item_list) * k * len(arms), max_unpaired):
        missing_reasons.append("missing_above_bound")
    if expected_models is not None:
        for name, a in arms.items():
            got = (a or {}).get("model_manifest_hash")
            if got is None or got != expected_models.get(name):
                flags.add("model_hash_mismatch")               # 05 sections 3.3, 3.8 (R11-14; R12A-19)
    missing_reasons += [r for r in ("k_mismatch", "seed_mismatch", "malformed_record", "unknown_item",
                                    "model_hash_mismatch") if r in flags]

    judges = {jid: judge_qualification_(jid, constraint_set, judge_qualification, alpha=alpha)
              for jid in (constraint_set.get("judges") or {})}
    egress_ok = not any(f["rule"] == "REG-EGRESS" for j in judges.values() for f in j["findings"])
    if egress_ok:
        # R12A-23 (MM-C3): each arm records the approval its external judge calls ran under.
        for jid, j in judges.items():
            if j["endpoint"] == "external" and any((a or {}).get("egress_approval_id") != j["egress_approval_id"]
                                                   for a in arms.values()):
                egress_ok = False
    if not egress_ok:
        missing_reasons.append("egress_unapproved")
    if blocking and any(f["rule"] == "REG-JUDGE" for j in judges.values() for f in j["findings"]):
        missing_reasons.append("judge_not_qualified")
    registered = [c.get("id") for c in constraint_set.get("canaries") or [] if isinstance(c, dict)]
    canary_required = bool(judges) and blocking                 # R13-4: only a judge that blocks alone
    by_arm = {name: _canaries(a or {}, registered, list(judges)) for name, a in arms.items()}
    canaries = {"registered": registered, "required": canary_required,
                "batches": sum(c["batches"] for c in by_arm.values()),
                "checked": sum(c["checked"] for c in by_arm.values()),
                "missed": [m for c in by_arm.values() for m in c["missed"]],
                "batches_without_canary": [b for c in by_arm.values() for b in c["batches_without_canary"]],
                "unbatched_records": sum(c["unbatched_records"] for c in by_arm.values()) if judges else 0,
                "malformed_batches": [b for c in by_arm.values() for b in c["malformed_batches"]],
                "by_arm": by_arm}
    if canaries["malformed_batches"]:
        missing_reasons.append("malformed_batch")                # MM-C2, R12A-21 (design choice)
    if canary_required:
        if canaries["batches_without_canary"] or canaries["unbatched_records"]:
            missing_reasons.append("canary_absent")
        if canaries["missed"]:
            missing_reasons.append("canary_missed")

    # Per-item violation indicators (05 section 3.9): the threshold form uses the candidate's usable
    # repeats; the non-inferiority form the repeats usable in both arms.
    uc = screened["candidate"]["usable"]
    if form == "threshold":
        reps: dict[Any, list[int]] = defaultdict(list)
        for it, r in uc:
            reps[it].append(r)
    else:
        ub = screened["baseline"]["usable"]
        reps = defaultdict(list)
        for it, r in set(uc) & set(ub):
            reps[it].append(r)
    pair_items = [it for it in item_list if it in reps]
    n = len(pair_items)
    v_c = [max(int(uc[(it, r)]["score"]) for r in reps[it]) for it in pair_items]
    out: dict[str, Any] = {
        "form": form, "hypothesis_type": clause["type"], "direction": "lower_is_better",
        "interval_method": INTERVALS[form], "alpha": alpha, "alpha_used": alpha, "power_target": power_target,
        "planning_rate": planning_rate, "blocking": blocking, "n_items": n, "min_paired": int(min_paired),
        "max_unpaired": max_unpaired, "missing_candidate": screened["candidate"]["missing"],
        "missing_reasons": {name: s["reasons"] for name, s in screened.items()},
        "constraint_set_id": constraint_set.get("id"), "constraint_set_hash": expected_hash,
        "constraints": [{"id": c.get("id"), "checker": (c.get("checker") or {}).get("kind")}
                        for c in constraint_set.get("constraints") or [] if isinstance(c, dict)],
        "judges": {jid: {key: val for key, val in j.items() if key != "findings"} for jid, j in judges.items()},
        "canaries": canaries,
    }
    out["bar" if form == "threshold" else "margin"] = bound_param
    if form == "non_inferiority":
        out["missing_baseline"] = screened["baseline"]["missing"]

    if form == "threshold":
        events = sum(v_c)
        diffs = v_c
        out.update(violation_count=events)
    else:
        ub = screened["baseline"]["usable"]
        v_b = [max(int(ub[(it, r)]["score"]) for r in reps[it]) for it in pair_items]
        c_only = sum(1 for c, b in zip(v_c, v_b) if c == 1 and b == 0)
        b_only = sum(1 for c, b in zip(v_c, v_b) if c == 0 and b == 1)
        events = c_only
        diffs = [c - b for c, b in zip(v_c, v_b)]
        out.update(violation_count=sum(v_c), baseline_violation_count=sum(v_b), candidate_only=c_only,
                   baseline_only=b_only, delta_hat=(b_only - c_only) / n if n else None,
                   difference=(c_only - b_only) / n if n else None,
                   baseline_violation_rate=sum(v_b) / n if n else None)
    out["violation_rate"] = sum(v_c) / n if n else None

    plannable = 0 <= planning_rate < bound_param
    n_required = binomial_n_required(bar=bound_param, planning_rate=planning_rate, alpha=alpha,
                                     power=power_target) if plannable else None
    if n == 0:
        missing_reasons.append("no_pairs")
        out.update(upper_bound=None, p_value=None, sd_zero=False, achieved_power=None, n_required=n_required,
                   underpowered=True, status=MISSING, status_reasons=missing_reasons)
        return out

    upper = clopper_pearson_upper(events, n, alpha)
    p_value = float(binom.cdf(events, n, bound_param))           # dual to the bound: p < alpha iff upper < bar
    sd_zero = len(set(diffs)) == 1                               # positive-width bound: reported, not INCONCLUSIVE
    achieved = threshold_power(n=n, bar=bound_param, planning_rate=planning_rate, alpha=alpha) if plannable else None
    underpowered = achieved is None or achieved < power_target or n_required is None or n < n_required
    passes = upper < bound_param
    inconclusive = ["min_paired"] if n < min_paired else []
    if missing_reasons:
        status = MISSING
    elif inconclusive:
        status = INCONCLUSIVE
    elif passes:
        status = PASS
    elif underpowered:
        status = INCONCLUSIVE
    else:
        status = FAIL
    reasons = missing_reasons + inconclusive
    if not passes and underpowered:
        reasons.append("underpowered")
    out.update(upper_bound=upper, p_value=p_value, sd_zero=sd_zero, achieved_power=achieved, n_required=n_required,
               underpowered=bool(underpowered), status=status, status_reasons=reasons)
    return out


judge_qualification_ = judge_qualification     # the parameter of evaluate_constraint shadows the function name

REPORT_KEYS = ("form", "status", "status_reasons", "violation_count", "violation_rate", "n_items", "upper_bound",
               "interval_method", "alpha_used", "p_value", "achieved_power", "underpowered", "n_required",
               "power_target", "planning_rate", "sd_zero", "blocking", "missing_candidate", "missing_reasons",
               "constraint_set_id", "constraint_set_hash", "constraints")
NI_REPORT_KEYS = ("missing_baseline", "baseline_violation_count", "candidate_only", "baseline_only", "delta_hat",
                  "difference")
JUDGE_REPORT_KEYS = ("model_id", "model_version", "endpoint", "tier", "family", "miss_rate", "miss_rate_upper",
                     "false_alarm_rate", "false_alarm_upper", "max_miss_rate", "max_false_alarm_rate",
                     "can_block_alone", "failed", "egress_approval_id", "transfer_test")
CANARY_REPORT_KEYS = ("registered", "required", "batches", "checked", "missed")
