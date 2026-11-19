"""E-stats gate library for the skeleton: plan, evaluate, combine (05 section 3.8).

Design references (05 = 05-evaluation-and-lineage.md, 04 = 04-workflow-and-composition.md):

- gate types and pass rules: 05:58-62. Superiority passes when the lower
  (1 - alpha) bound of Delta exceeds delta_sup; non-inferiority when it
  exceeds -delta; equivalence when the (1 - 2 alpha) interval lies inside
  (-delta, delta);
- the underpowered rule: 05:82 and 04:102. An underpowered pass is PASS and
  flagged; an underpowered non-rejection is INCONCLUSIVE, never FAIL;
- paired design on raw per-prompt differences, K averaged per prompt: 05:97-99;
- prompt-clustered bootstrap, BCa: 05:105;
- MDE and design effect: 05:124; sigma planning ucb80 / fixed: 05:130-132;
- statuses PASS / FAIL / INCONCLUSIVE / MISSING: 05:155;
- combine("and") follows NeMo Evaluator `_aggregate_verdict` semantics
  (05:151; gate.py:537 at ea35ba3), with MISSING propagated as MISSING
  (interim rule G-9).

Design revision 10 (tests/README.md section 11.15):

- achieved power and `underpowered` (05 section 3.6, 05:138-143; R10A-1): power at the registered
  planning effect theta with the observed SD (ddof = 1) and the normal approximation,
  SE = sd * sqrt(DE / n). Superiority / non-inferiority: Phi(d / SE - z_{1-alpha}), d = theta - delta_sup
  or theta + delta. Equivalence (TOST): max(0, Phi((delta - theta) / SE - z) + Phi((delta + theta) / SE - z)
  - 1), and `plan()` gives the TOST required n. `underpowered` on low power, on n below the plan's
  n_required, or when power cannot be computed. G-13: an SD of 0 keeps the power limit and sets `sd_zero`;
- missingness (05 section 3.4, 05:116; R10A-3): expected items E = {(p, r): p in either arm,
  r in range(K)}; above `max_unpaired` (a count, or a fraction of 2|E|) the clause is MISSING;
- minimum paired sample (05 section 3.6, 05:145; R10A-5): fewer than `min_paired` pairs, or a
  zero-width interval, is INCONCLUSIVE;
- K and seeds (05 section 3.3; R10A-8): a repeat outside range(K) and a repeat -> seed mapping other than
  the registration's are MISSING (fewer than K repeats per prompt are missing items only, R11-7);
- p-values (05 section 3.8 table; R10A-10) by inverting the BCa interval; `p_value_at_floor` (R10A-16)
  marks a p-value whose bootstrap tail is empty, so it is a bound at the clip of G, not an estimate;
- zero width (05 section 3.6, 05:156; R10A-14): one criterion, `_zero_width`, decides the zero-width
  interval, the null p-value, `sd_zero` and the `zero_width_interval` reason, with a relative float tolerance;
- `status_reasons` name why a result is MISSING or INCONCLUSIVE; MISSING takes precedence over
  INCONCLUSIVE within a clause, and `combine` follows 05 section 3.8 (05:160).

Design revision 11 (tests/README.md section 11.16): `evaluate_items` is the gate on per-item records
(05 section 3.8; R11-2, R11-3, R11-5, R11-7, R11-9 to R11-11, R11-14, R11-21) and is what `run` and
`tierb-collect` call; the mapping form `evaluate` stays as the revision 10 statistics reference.
`is_post_hoc` scopes `post_hoc` to results of the judged candidate checkpoint (R11-1). Design revision 13:
`no_seed` missing items (R13-6), the ratio-of-sums interval is BCa with the item jackknife and per-item
sums over the repeats used (R13-8).
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2, norm

HYPOTHESES = ("superiority", "non_inferiority", "equivalence")
PASS, FAIL, INCONCLUSIVE, MISSING = "PASS", "FAIL", "INCONCLUSIVE", "MISSING"


# ---------------------------------------------------------------------------
# Planning (05 section 3.6)
# ---------------------------------------------------------------------------

def sigma_ucb80_factor(df: int) -> float:
    """sigma_U / s = sqrt(df / chi2_{0.20, df}) (05:130)."""
    if df < 1:
        raise ValueError("df must be >= 1")
    return float(math.sqrt(df / chi2.ppf(0.20, df)))


def design_effect(m: float, rho: float) -> float:
    """DE = 1 + (m - 1) rho (05:124)."""
    return 1.0 + (m - 1.0) * rho


def mde(*, alpha: float, power: float, sigma: float, n: int, design_effect: float = 1.0) -> float:
    """MDE = (z_{1-alpha} + z_{power}) * sigma * sqrt(DE / n) (05:124)."""
    return float((norm.ppf(1.0 - alpha) + norm.ppf(power)) * sigma * math.sqrt(design_effect / n))


def planning_distance(clause: dict) -> float | None:
    """Distance from the null at the planning effect (05:58-62); None for equivalence, whose power
    has two sides (`equivalence_power`, 05 section 3.6)."""
    t = clause["type"]
    if t == "superiority":
        return float(clause["planning_effect"]) - float(clause["margin"])
    if t == "non_inferiority":
        return float(clause["planning_effect"]) + float(clause["margin"])
    if t == "equivalence":
        return None
    raise ValueError(f"unknown clause type {t!r}")


def planning_sigma(clause: dict, pilot: dict) -> float:
    """Planning sigma under the clause's sigma plan (05:128-132)."""
    plan_kind = clause["sigma_plan"]
    if plan_kind == "ucb80":
        return float(pilot["s"]) * sigma_ucb80_factor(int(pilot["df"]))
    if plan_kind == "fixed":
        return float(pilot["sigma"])
    raise ValueError(f"sigma_plan {plan_kind!r} is not supported by the skeleton (ucb80 or fixed)")


def equivalence_power(*, margin: float, theta: float, se: float, alpha: float) -> float:
    """TOST power (05 section 3.6, 05:142): max(0, Phi((delta - theta) / SE - z) + Phi((delta + theta) / SE - z) - 1)."""
    z = norm.ppf(1.0 - alpha)
    return float(max(0.0, norm.cdf((margin - theta) / se - z) + norm.cdf((margin + theta) / se - z) - 1.0))


def plan(clause: dict, pilot: dict, *, n_available: int, cluster_size: float = 1.0, icc: float = 0.0) -> dict[str, Any]:
    """Required n, MDE and design effect under the sigma plan (05 section 3.8 `plan`).

    n_required = ceil(DE * ((z_{1-alpha} + z_power) * sigma_used / d)^2), the
    05:124 formula solved for n. Equivalence (TOST, 05 section 3.6; R10A-1):
    ceil(DE * ((z_{1-alpha} + z_{1-beta/2}) * sigma / delta)^2) at theta = 0 and
    ceil(DE * ((z_{1-alpha} + z_{1-beta}) * sigma / (delta - |theta|))^2) otherwise.
    `underpowered` is reported up front when the prompt set is smaller than n
    (MVP-SPEC section 2.3).
    """
    alpha, power = float(clause["alpha"]), float(clause["power"])
    sigma_used = planning_sigma(clause, pilot)
    de = design_effect(cluster_size, icc)
    d = planning_distance(clause)
    if d is None:
        margin, theta = float(clause["margin"]), abs(float(clause["planning_effect"]))
        beta = 1.0 - power
        if theta == 0.0:
            z_sum, dist = norm.ppf(1 - alpha) + norm.ppf(1 - beta / 2), margin
        else:
            z_sum, dist = norm.ppf(1 - alpha) + norm.ppf(1 - beta), margin - theta
        if dist <= 0:
            raise ValueError(f"equivalence margin {margin} must exceed |planning_effect| {theta}")
        n_required = int(math.ceil(de * (z_sum * sigma_used / dist) ** 2))
    else:
        if d <= 0:
            raise ValueError(f"planning distance {d} must be positive for {clause['type']}")
        n_required = int(math.ceil(de * ((norm.ppf(1 - alpha) + norm.ppf(power)) * sigma_used / d) ** 2))
    return {
        "hypothesis_type": clause["type"],
        "alpha": alpha,
        "power": power,
        "planning_distance": d,
        "n_required": n_required,
        "mde": mde(alpha=alpha, power=power, sigma=sigma_used, n=n_available, design_effect=de),
        "design_effect": de,
        "sigma_plan": clause["sigma_plan"],
        "sigma_used": sigma_used,
        "n_available": int(n_available),
        "underpowered": bool(n_required is not None and n_available < n_required),
    }


# ---------------------------------------------------------------------------
# Pairing (NeMo Evaluator comparison.py:360 at ea35ba3)
# ---------------------------------------------------------------------------

def load_paired_records(task_dir: Path) -> dict[tuple[int, int], dict[str, Any]]:
    """Reimplementation of NeMo Evaluator `load_paired_records`
    (src/nemo_evaluator/engine/comparison.py:360-379 at ea35ba3).

    nemo_evaluator is not an allowed dependency (MVP-SPEC section 1), so the
    function is copied in behavior: missing file gives {}, blank and bad
    lines are skipped, `repeat` defaults to 0, and a duplicate key keeps the
    last line. tests/test_pairing.py checks agreement with the pinned source.
    """
    results_path = Path(task_dir) / "results.jsonl"
    if not results_path.exists():
        return {}
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            pid = record.get("problem_idx")
            rep = record.get("repeat", 0)
            if pid is not None:
                records[(int(pid), int(rep))] = record
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return records


def paired_differences(baseline: dict, candidate: dict) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Per-prompt means over the paired repeats, candidate - baseline, raw scores (05:97-99).

    Pairs are the key intersection, as in build_flip_report (comparison.py:387).
    Returns (prompt ids, baseline means, candidate means, differences), sorted by prompt.
    """
    keys = sorted(set(baseline) & set(candidate))
    b_acc: dict[int, list[float]] = defaultdict(list)
    c_acc: dict[int, list[float]] = defaultdict(list)
    for pid, rep in keys:
        b_acc[pid].append(float(baseline[(pid, rep)]["reward"]))
        c_acc[pid].append(float(candidate[(pid, rep)]["reward"]))
    pids = sorted(b_acc)
    b = np.array([sum(b_acc[p]) / len(b_acc[p]) for p in pids], dtype=np.float64)
    c = np.array([sum(c_acc[p]) / len(c_acc[p]) for p in pids], dtype=np.float64)
    return pids, b, c, c - b


# ---------------------------------------------------------------------------
# Interval: prompt-clustered bootstrap, BCa (05:105)
# ---------------------------------------------------------------------------

def _bca_parts(diffs: np.ndarray, *, seed: int, n_boot: int) -> tuple[np.ndarray, float, float]:
    """Bootstrap means, bias correction z0 and acceleration a of the BCa interval."""
    n = len(diffs)
    theta = float(diffs.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diffs[idx].mean(axis=1)
    # Bias correction with ties counted half (mid-rank), clipped away from 0 and 1.
    z0 = float(norm.ppf(_boot_cdf(boots, theta)))
    # Acceleration from the jackknife of the mean.
    jack = (diffs.sum() - diffs) / (n - 1)
    dev = jack.mean() - jack
    denom = 6.0 * float(np.sum(dev**2)) ** 1.5
    a = float(np.sum(dev**3)) / denom if denom > 0 else 0.0
    return boots, z0, a


def _boot_cdf_clip(boots: np.ndarray, x: float) -> tuple[float, int]:
    """G(x) as `_boot_cdf`, and -1 / +1 when G was clipped up to 1/(B+1) / down to B/(B+1), else 0."""
    b = len(boots)
    g = float((np.sum(boots < x) + 0.5 * np.sum(boots == x)) / b)
    lo, hi = 1.0 / (b + 1), b / (b + 1)
    if g < lo:
        return lo, -1
    if g > hi:
        return hi, 1
    return g, 0


def _boot_cdf(boots: np.ndarray, x: float) -> float:
    """G(x): share of bootstrap means below x, ties counted half, clipped to [1/(B+1), B/(B+1)]."""
    return _boot_cdf_clip(boots, x)[0]


ZERO_WIDTH_REL_TOL = 1e-12


def _zero_width(diffs: np.ndarray) -> bool:
    """R10A-14: the one zero-width criterion (05 section 3.6, 05:156 "sigma-hat_d = 0").

    Degenerate when n < 2 or max(d) - min(d) <= 1e-12 * max(1, |mean(d)|), so differences equal up to
    float noise (for example 1/3 computed two ways) count as no between-prompt variation. Used by
    `bca_interval`, `bca_p_value` and `evaluate` so the interval, the p-value, `sd_zero` and the
    `zero_width_interval` reason always agree.
    """
    if len(diffs) < 2:
        return True
    spread = float(diffs.max() - diffs.min())
    return spread <= ZERO_WIDTH_REL_TOL * max(1.0, abs(float(diffs.mean())))


def bca_interval(diffs: np.ndarray, *, alpha: float, seed: int, n_boot: int) -> tuple[float, float]:
    """Two-sided (1 - 2 alpha) BCa interval for the mean of per-prompt differences.

    Resampling prompts carries all K samples of both arms with the prompt
    (the per-prompt difference already averages them), so this is the
    prompt-clustered bootstrap of 05:105. The lower end is the one-sided
    (1 - alpha) bound used by the pass rules (05:58-62).
    """
    theta = float(diffs.mean())
    if _zero_width(diffs):
        return theta, theta
    boots, z0, a = _bca_parts(diffs, seed=seed, n_boot=n_boot)
    return _bca_bounds(boots, z0, a, alpha=alpha)


def _bca_bounds(boots: np.ndarray, z0: float, a: float, *, alpha: float) -> tuple[float, float]:
    """The BCa (alpha, 1 - alpha) quantiles of the bootstrap distribution, given z0 and a."""

    def adjusted(q: float) -> float:
        zq = norm.ppf(q)
        return float(norm.cdf(z0 + (z0 + zq) / (1 - a * (z0 + zq))))

    lo_q, hi_q = adjusted(alpha), adjusted(1 - alpha)
    lower = float(np.quantile(boots, lo_q))
    upper = float(np.quantile(boots, hi_q))
    return lower, upper


def bca_p_value_detail(diffs: np.ndarray, *, hypothesis: str, margin: float, seed: int,
                       n_boot: int) -> tuple[float | None, bool]:
    """`bca_p_value` and whether it is at the bootstrap floor (R10A-16).

    At the floor the bootstrap tail behind the reported one-sided p is empty: G(null) was clipped up to
    1/(B+1) for H0: Delta <= null, or G(delta) down to B/(B+1) for the upper TOST side (H0: Delta >= delta).
    For equivalence the side that gives the reported maximum decides (on a tie, both must be at the
    floor). The p-value is then a bound, not an estimate. A clip toward p near 1 (a ceiling) and a null
    p-value are not at the floor.
    """
    if _zero_width(diffs):
        return None, False
    boots, z0, a = _bca_parts(diffs, seed=seed, n_boot=n_boot)
    return _bca_p_detail(boots, z0, a, hypothesis=hypothesis, margin=margin)


def _bca_p_detail(boots: np.ndarray, z0: float, a: float, *, hypothesis: str,
                  margin: float) -> tuple[float, bool]:
    """p-value by BCa inversion and the at-floor flag, given the bootstrap draws, z0 and a."""

    def inverted(null: float) -> tuple[float, int]:
        g, clip = _boot_cdf_clip(boots, null)
        w = float(norm.ppf(g)) - z0
        return float(norm.cdf(w / (1 + a * w) - z0)), clip

    if hypothesis == "superiority":
        p, clip = inverted(margin)
        return p, clip < 0
    if hypothesis == "non_inferiority":
        p, clip = inverted(-margin)
        return p, clip < 0
    if hypothesis == "equivalence":
        p_lo, clip_lo = inverted(-margin)
        q_hi, clip_hi = inverted(margin)
        p_hi = 1.0 - q_hi
        floor_lo, floor_hi = clip_lo < 0, clip_hi > 0
        if p_lo > p_hi:
            return p_lo, floor_lo
        if p_hi > p_lo:
            return p_hi, floor_hi
        return p_lo, floor_lo and floor_hi
    raise ValueError(f"unknown clause type {hypothesis!r}")


def bca_p_value(diffs: np.ndarray, *, hypothesis: str, margin: float, seed: int, n_boot: int) -> float | None:
    """One-sided p-value by inverting the BCa bound (R10A-10; 05 section 3.8 table).

    For H0: Delta <= null (superiority: null = delta_sup; non-inferiority: null = -delta) p is the
    alpha' whose BCa lower bound equals the null: p = Phi(w / (1 + a w) - z0), w = Phi^-1(G(null)) - z0.
    Equivalence (TOST): the larger of that p at -delta and of 1 - Phi(w / (1 + a w) - z0) at G(delta)
    (H0: Delta >= delta). The same bootstrap draws as `bca_interval`. None for a zero-width interval
    (`_zero_width`, R10A-14).
    """
    return bca_p_value_detail(diffs, hypothesis=hypothesis, margin=margin, seed=seed, n_boot=n_boot)[0]


# ---------------------------------------------------------------------------
# evaluate (05 section 3.8)
# ---------------------------------------------------------------------------

def _passes(hypothesis: str, lower: float, upper: float, margin: float) -> bool:
    if hypothesis == "superiority":
        return lower > margin                       # 05:60
    if hypothesis == "non_inferiority":
        return lower > -margin                      # 05:61
    if hypothesis == "equivalence":
        return lower > -margin and upper < margin   # 05:62
    raise ValueError(f"unknown clause type {hypothesis!r}")


def _k_actual(records: dict) -> list[int]:
    """[min, max] repeats per prompt in one arm; [0, 0] for an empty arm (R10A-8)."""
    per_prompt: dict[int, int] = defaultdict(int)
    for pid, _ in records:
        per_prompt[pid] += 1
    counts = list(per_prompt.values())
    return [min(counts), max(counts)] if counts else [0, 0]


def _seed_mismatch(records: dict, seeds: list[int]) -> bool:
    """A record whose `metadata.eval_seed` is present and differs from eval_seeds[repeat] (R10A-8)."""
    for (_, rep), rec in records.items():
        meta = rec.get("metadata") if isinstance(rec, dict) else None
        if isinstance(meta, dict) and meta.get("eval_seed") is not None and 0 <= rep < len(seeds):
            if meta["eval_seed"] != seeds[rep]:
                return True
    return False


def _above_bound(missing: int, expected: int, max_unpaired: int | float) -> bool:
    """05 section 3.4 (05:116): an integer bound is a count, a float in [0, 1) a fraction of 2|E|."""
    if isinstance(max_unpaired, float):
        return missing > max_unpaired * 2 * expected
    return missing > max_unpaired


def evaluate(clause: dict, baseline: dict, candidate: dict, *, plan: dict | None = None, seed: int,
             n_boot: int = 10000, min_paired: int = 10, max_unpaired: int | float = 0) -> dict[str, Any]:
    """Delta-hat, interval, p-value, achieved power, MDE, underpowered and status for one clause.

    `baseline` and `candidate` map (problem_idx, repeat) to NeMo Evaluator
    per-sample records (tests/README.md section 7). `min_paired` and
    `max_unpaired` are the registration's (05 sections 3.6 and 3.4; defaults 10 and 0).

    The revision 10 mapping form, kept as the statistics reference for the unit tests: the expected items
    are the prompts of either arm (R10A-3), and only the mean estimator of a higher-is-better clause is
    read. The gate itself runs `evaluate_items` (design revision 11). The result carries the revision 10
    names (`n_prompts`) and, so it renders in a revision 11 report, `n_items`, `direction`, `estimator`,
    the natural-unit difference (equal to the oriented one) and `missing_reasons` (all `absent`).
    """
    hypothesis = clause["type"]
    if hypothesis not in HYPOTHESES:
        raise ValueError(f"unknown clause type {hypothesis!r}")
    if (clause.get("direction") or "higher_is_better") != "higher_is_better" or \
            (clause.get("estimator") or "mean") != "mean":
        raise ValueError("the mapping form reads higher_is_better mean clauses only; use evaluate_items")
    alpha = float(clause["alpha"])
    power_target = float(clause["power"])
    margin = float(clause["margin"])
    k = _clause_k(clause)
    seeds = list(clause["eval_seeds"])
    de = 1.0   # unit = prompt: per-prompt means absorb K (05:126)
    pids, b, c, diffs = paired_differences(baseline, candidate)
    n = len(pids)
    paired = set(baseline) & set(candidate)

    # 05 section 3.4: expected items E over the prompts of either arm and the registered repeats.
    prompts = {pid for pid, _ in baseline} | {pid for pid, _ in candidate}
    expected = {(pid, r) for pid in prompts for r in range(k)}
    missing_b, missing_c = len(expected - set(baseline)), len(expected - set(candidate))
    k_actual = {"baseline": _k_actual(baseline), "candidate": _k_actual(candidate)}
    missing_reasons: list[str] = []
    if n == 0:
        missing_reasons.append("no_pairs")                      # 05:155; NeMo Evaluator gate.py:152
    if _above_bound(missing_b + missing_c, len(expected), max_unpaired):
        missing_reasons.append("missing_above_bound")
    # Design revision 11 (R11-7, R11A-7): a repeat outside range(K) is MISSING; fewer than K records per
    # prompt are missing items only (supersedes R10A-8's "maximum repeats != K" rule).
    if any(not 0 <= rep < k for rep_set in (baseline, candidate) for _, rep in rep_set):
        missing_reasons.append("k_mismatch")
    if _seed_mismatch(baseline, seeds) or _seed_mismatch(candidate, seeds):
        missing_reasons.append("seed_mismatch")

    out: dict[str, Any] = {
        "hypothesis_type": hypothesis, "alpha": alpha, "ci_level": 1.0 - 2.0 * alpha, "margin": margin,
        "power_target": power_target, "design_effect": de, "n_prompts": n, "n_items": n, "n_boot": int(n_boot),
        "direction": "higher_is_better", "estimator": "mean",
        "missing_reasons": {"baseline": {"absent": missing_b} if missing_b else {},
                            "candidate": {"absent": missing_c} if missing_c else {}},
        "bootstrap_seed": int(seed), "interval_method": "cluster_bootstrap", "bootstrap_variant": "bca",
        # Decision D-3: items present in one arm only are counted, and scores use the paired items only.
        "unpaired_baseline": len(set(baseline) - paired),
        "unpaired_candidate": len(set(candidate) - paired),
        "missing_baseline": missing_b, "missing_candidate": missing_c,
        "max_unpaired": max_unpaired, "min_paired": int(min_paired), "k_actual": k_actual,
        "p_value_method": "bca_inversion",
        "baseline_score": (sum(float(baseline[k_]["reward"]) for k_ in paired) / len(paired)) if paired else None,
        "candidate_score": (sum(float(candidate[k_]["reward"]) for k_ in paired) / len(paired)) if paired else None,
    }
    if n == 0:
        out.update(delta_hat=None, ci_lower=None, ci_upper=None, delta_hat_natural=None, ci_lower_natural=None,
                   ci_upper_natural=None, sd_diff=None, sd_zero=False, achieved_power=None,
                   mde=None, p_value=None, p_value_at_floor=False, underpowered=True, status=MISSING,
                   status_reasons=missing_reasons)
        return out
    delta_hat = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if n > 1 else None
    # R10A-14: one criterion for the zero-width interval, the null p-value, `sd_zero` and the reason.
    zero_width = _zero_width(diffs)
    sd_zero = sd is not None and zero_width                          # G-13: SD 0 up to float noise
    lower, upper = bca_interval(diffs, alpha=alpha, seed=seed, n_boot=n_boot)
    p_value, p_value_at_floor = bca_p_value_detail(diffs, hypothesis=hypothesis, margin=margin, seed=seed,
                                                   n_boot=n_boot)

    achieved_power, observed_mde = _power_and_mde(clause, sd=sd, sd_zero=sd_zero, n=n, alpha=alpha,
                                                  power_target=power_target, de=de)

    # 05 section 3.6: underpowered when achieved power < target (04:102, 05:82), when n < n_required
    # (plan), or when power cannot be computed, so a non-rejection is INCONCLUSIVE, never FAIL (05:82).
    underpowered = achieved_power is None or achieved_power < power_target
    if plan is not None:                                                                  # MVP-SPEC 2.3
        n_req = plan.get("n_required")
        if plan.get("underpowered") or (n_req is not None and n < n_req):
            underpowered = True

    # 05 section 3.6 (05:145): too few pairs or a zero-width interval is never PASS.
    inconclusive_reasons: list[str] = []
    if n < min_paired:
        inconclusive_reasons.append("min_paired")
    if zero_width:
        inconclusive_reasons.append("zero_width_interval")

    passes = _passes(hypothesis, lower, upper, margin)
    if missing_reasons:
        status = MISSING
    elif inconclusive_reasons:
        status = INCONCLUSIVE
    elif passes:
        status = PASS
    elif underpowered:
        status = INCONCLUSIVE
    else:
        status = FAIL
    reasons = missing_reasons + inconclusive_reasons
    if not passes and underpowered:
        reasons.append("underpowered")
    out.update(delta_hat=delta_hat, ci_lower=lower, ci_upper=upper, sd_diff=sd,
               delta_hat_natural=delta_hat, ci_lower_natural=lower, ci_upper_natural=upper,
               sd_zero=bool(sd_zero),                                # G-13, R10A-14
               achieved_power=achieved_power, mde=observed_mde, p_value=p_value,
               p_value_at_floor=bool(p_value_at_floor),              # R10A-16
               underpowered=bool(underpowered),
               status=status, status_reasons=reasons)
    return out


def apply_post_hoc(result: dict) -> dict:
    """Downgrade a post hoc PASS to INCONCLUSIVE while keeping its statistics (R10A-4)."""
    if result["status"] == PASS:
        result = {**result, "status": INCONCLUSIVE, "status_reasons": [*result["status_reasons"], "post_hoc"]}
    return result


# ---------------------------------------------------------------------------
# evaluate_items: the gate on per-item records (design revision 11; tests/README.md section 11.16)
# ---------------------------------------------------------------------------

def _clause_k(clause: dict) -> int:
    """K from `samples_per_item`, or the legacy alias `samples_per_prompt` (R11A-2)."""
    return int(clause["samples_per_item"] if "samples_per_item" in clause else clause["samples_per_prompt"])


def _hashable(x: Any) -> bool:
    try:
        hash(x)
    except TypeError:
        return False
    return True


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _malformed(rec: Any, ratio: bool) -> bool:
    """R11A-7 `malformed_record` (design choice): not a mapping, an unknown key, a missing `item` or
    `repeat`, both or neither of score and error, a non-numeric or boolean score, an error that is not
    {class, message}, or (ratio clauses) a scored record without numeric numerator and denominator."""
    from .items import RATIO_FIELDS, RECORD_FIELDS

    if not isinstance(rec, dict):
        return True
    allowed = set(RECORD_FIELDS) | (set(RATIO_FIELDS) if ratio else set())
    if set(rec) - allowed:
        return True
    if "item" not in rec or rec["item"] is None or not _hashable(rec["item"]) or isinstance(rec["item"], bool):
        return True
    if not (isinstance(rec.get("repeat"), int) and not isinstance(rec.get("repeat"), bool)):
        return True
    seed = rec.get("seed")
    if seed is not None and not (isinstance(seed, int) and not isinstance(seed, bool)):
        return True
    score, error = rec.get("score"), rec.get("error")
    if (score is None) == (error is None):
        return True
    if error is not None:
        return not (isinstance(error, dict) and set(error) <= {"class", "message"}
                    and isinstance(error.get("class"), str) and error["class"])
    if not _is_number(score):
        return True
    if ratio:
        return not all(_is_number(rec.get(f)) for f in RATIO_FIELDS)
    return False


def _out_of_range(score: float, score_range: tuple[float | None, float | None]) -> bool:
    """05 section 6.1 (R11-11): outside the declared range; NaN and +/-inf always (R11A-6), and an integer too
    large for a float (R12A-28)."""
    try:
        if not math.isfinite(float(score)):
            return True
    except OverflowError:
        return True
    lo, hi = score_range
    return (lo is not None and score < lo) or (hi is not None and score > hi)


# R12A-28 (design choice): the finite range of a ratio part. |numerator| <= RATIO_PART_LIMIT and
# 1 / RATIO_PART_LIMIT <= denominator <= RATIO_PART_LIMIT keep every arm sum, ratio, contribution, its square and
# cube (jackknife acceleration) and every bootstrap ratio finite for any realistic number of records, so no
# overflow, NaN or exception reaches the estimator.
RATIO_PART_LIMIT = 1e12


def _ratio_parts_usable(rec: dict) -> bool:
    """R11A-26: a ratio record's numerator is finite and its denominator finite and positive. R12A-28: a part
    too large to convert (`OverflowError`) or outside `RATIO_PART_LIMIT` is not usable either."""
    try:
        num, den = float(rec["numerator"]), float(rec["denominator"])
    except OverflowError:
        return False
    return (math.isfinite(num) and math.isfinite(den) and abs(num) <= RATIO_PART_LIMIT
            and 1.0 / RATIO_PART_LIMIT <= den <= RATIO_PART_LIMIT)


def _screen_arm(records: list, *, index: dict, k: int, seeds: list, ratio: bool,
                score_range: tuple[float | None, float | None]) -> dict[str, Any]:
    """One arm's records sorted into usable records, missing-item reasons and clause-level flags."""
    flags: set[str] = set()
    usable: dict[tuple[Any, int], dict] = {}
    seen: set[tuple[Any, int]] = set()
    reasons: dict[str, int] = defaultdict(int)
    per_item: dict[Any, int] = defaultdict(int)
    for rec in records:
        if _malformed(rec, ratio):
            flags.add("malformed_record")
            continue
        key = (rec["item"], rec["repeat"])
        if key in seen:
            flags.add("malformed_record")                       # one record per (item, repeat)
            continue
        seen.add(key)
        if rec["item"] not in index:
            flags.add("unknown_item")
            continue
        rep = rec["repeat"]
        if not 0 <= rep < k:
            flags.add("k_mismatch")                             # 05 section 3.8 (R11-7)
            continue
        per_item[rec["item"]] += 1
        seed = rec.get("seed")
        if seed is not None and seed != seeds[rep]:
            flags.add("seed_mismatch")                          # 05 section 3.8 (R11-7)
        if rec.get("error") is not None:
            reasons[rec["error"]["class"]] += 1                 # never scored, never 0 (05 section 3.4)
        elif seed is None:
            reasons["no_seed"] += 1                             # 05 section 3.4 (R13-6)
        elif _out_of_range(rec["score"], score_range):
            reasons["out_of_range"] += 1                        # 05 section 6.1 (R11-11)
        elif ratio and not _ratio_parts_usable(rec):
            reasons["out_of_range"] += 1                        # R11A-26: never NaN or inf in the sums
        else:
            usable[key] = rec
    expected = len(index) * k
    absent = expected - len(usable) - sum(reasons.values())
    if absent:
        reasons["absent"] += absent
    counts = list(per_item.values())
    return {"usable": usable, "reasons": dict(sorted(reasons.items())), "missing": expected - len(usable),
            "flags": flags, "k_actual": [min(counts), max(counts)] if counts else [0, 0]}


def _ratio_bca(num_b: np.ndarray, den_b: np.ndarray, num_c: np.ndarray, den_c: np.ndarray, *, sign: float,
               seed: int, n_boot: int) -> tuple[np.ndarray, float, float]:
    """Item-clustered bootstrap of the oriented ratio difference (05 section 3.4, R11-21): items are
    resampled with all their repeats, both arms together. BCa with the item jackknife (R13-8)."""
    n = len(num_b)

    def stat(nb, db, nc, dc):
        return sign * (nc / dc - nb / db)

    theta = float(stat(num_b.sum(), den_b.sum(), num_c.sum(), den_c.sum()))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = stat(num_b[idx].sum(axis=1), den_b[idx].sum(axis=1), num_c[idx].sum(axis=1), den_c[idx].sum(axis=1))
    z0 = float(norm.ppf(_boot_cdf(boots, theta)))
    jack = stat(num_b.sum() - num_b, den_b.sum() - den_b, num_c.sum() - num_c, den_c.sum() - den_c)
    dev = jack.mean() - jack
    denom = 6.0 * float(np.sum(dev**2)) ** 1.5
    a = float(np.sum(dev**3)) / denom if denom > 0 else 0.0
    return boots, z0, a


BOOTSTRAP_VARIANTS = ("bca", "percentile")


def _cluster_value_ok(v: Any) -> bool:
    """R18-3: a cluster value is a string or an integer (not a boolean), so clusters have one sort order."""
    return isinstance(v, str) or (isinstance(v, int) and not isinstance(v, bool))


def _cluster_order(labels: list) -> list:
    """The clusters present, sorted by value (integers before strings): the bootstrap's cluster order (R18-3)."""
    return sorted(set(labels), key=lambda v: (isinstance(v, str), v))


def _cluster_sums(values: np.ndarray, labels: list, ckeys: list) -> np.ndarray:
    """Per-cluster sums of per-item values, in `ckeys` order; each sum adds the cluster's items in item order."""
    lab = np.empty(len(labels), dtype=object)
    lab[:] = labels
    return np.array([values[lab == k].sum() for k in ckeys], dtype=np.float64)


def _cluster_counts(labels: list, ckeys: list) -> np.ndarray:
    lab = np.empty(len(labels), dtype=object)
    lab[:] = labels
    return np.array([int((lab == k).sum()) for k in ckeys])


def _cluster_mean_parts(diffs: np.ndarray, labels: list, ckeys: list, *, theta: float, seed: int,
                        n_boot: int) -> tuple[np.ndarray, float, float]:
    """Cluster bootstrap of the mean paired difference (05 section 3.4, R18-3): G clusters drawn with replacement,
    each replicate the drawn clusters' summed differences over their item count; z0 against `theta` and the
    acceleration from the leave-one-cluster-out jackknife."""
    sums, counts = _cluster_sums(diffs, labels, ckeys), _cluster_counts(labels, ckeys)
    g = len(ckeys)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, g, size=(n_boot, g))
    boots = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    z0 = float(norm.ppf(_boot_cdf(boots, theta)))
    jack = (sums.sum() - sums) / (counts.sum() - counts)
    dev = jack.mean() - jack
    denom = 6.0 * float(np.sum(dev**2)) ** 1.5
    a = float(np.sum(dev**3)) / denom if denom > 0 else 0.0
    return boots, z0, a


def _cluster_design_effect(values: np.ndarray, labels: list, ckeys: list) -> float:
    """Observed design effect of a clustered clause (05 section 3.6, R18-3): the cluster-robust variance of the mean
    of the per-item values, G/(G-1) * sum_g (S_g - mean * m_g)^2 / n^2, over the item-level variance sd^2 / n;
    at least 1. 1 when the item-level SD is 0 or there are fewer than two clusters."""
    n, g = len(values), len(ckeys)
    sd2 = float(values.var(ddof=1)) if n > 1 else 0.0
    if g < 2 or sd2 <= 0.0:
        return 1.0
    mean = float(values.mean())
    resid = _cluster_sums(values, labels, ckeys) - mean * _cluster_counts(labels, ckeys)
    var_cluster = g / (g - 1) * float(np.sum(resid**2)) / n**2
    return max(1.0, var_cluster / (sd2 / n))


def _power_and_mde(clause: dict, *, sd: float | None, sd_zero: bool, n: int, alpha: float, power_target: float,
                   de: float = 1.0) -> tuple[float | None, float | None]:
    """Achieved power at the registered planning effect and the observed MDE (05 section 3.6; R10A-1)."""
    if sd is None:
        return None, None
    margin, theta = float(clause["margin"]), float(clause["planning_effect"])
    observed_mde = mde(alpha=alpha, power=power_target, sigma=sd, n=n, design_effect=de)
    d = planning_distance(clause)
    if d is None:                                              # equivalence, TOST (05 section 3.6)
        if sd_zero:
            return (1.0 if margin > abs(theta) else 0.0), observed_mde      # G-13: limit of the formula
        return equivalence_power(margin=margin, theta=theta, se=sd * math.sqrt(de / n), alpha=alpha), observed_mde
    if sd_zero:
        return (1.0 if d > 0 else 0.0), observed_mde                         # G-13: limit of the formula
    return float(norm.cdf(d * math.sqrt(n) / (sd * math.sqrt(de)) - norm.ppf(1 - alpha))), observed_mde


def evaluate_items(clause: dict, baseline: dict, candidate: dict, *, items, seed: int, n_boot: int = 10000,
                   min_paired: int = 10, max_unpaired: int | float = 0,
                   score_range: tuple[float | None, float | None] = (None, None),
                   expected_models: dict[str, str | None] | None = None, plan: dict | None = None,
                   clusters: dict | None = None) -> dict[str, Any]:
    """The gate on per-item records (05 section 3.8; design revision 11, tests/README.md section 11.16).

    An arm is `{"records": [record, ...], "model_manifest_hash": str | None}` (other keys ignored); a record
    has the fields of `items.RECORD_FIELDS`. `items` is the registered item set: expected items are
    items x range(K) per arm (05 section 3.4, R11A-6), and the item order fixes the bootstrap's item order, so
    the result does not depend on the record order. Revision 11 rules:

    - missing items (R11-2): no record (`absent`), an evaluator error (its class), a score outside
      `score_range` or not finite, or (ratio clauses) a numerator that is not finite or a denominator that
      is not finite and positive (`out_of_range`, R11A-26), no seed (`no_seed`, R13-6); above `max_unpaired` (a count,
      or a fraction of 2 |items| K) the clause is MISSING (`missing_above_bound`);
    - MISSING clause (R11-7, R11-14): a repeat outside range(K) (`k_mismatch`), a seed other than
      eval_seeds[repeat] (`seed_mismatch`), a malformed record (`malformed_record`), an item outside the set
      (`unknown_item`), an arm model hash other than `expected_models[arm]` or null (`model_hash_mismatch`);
    - complete pairs (R11-3): an item with at least one repeat usable in both arms; its difference uses the
      repeats usable in both; `n_items` counts them and is the n of `min_paired` and the power formulas;
    - direction (R11-10): `lower_is_better` scores are negated, Delta > 0 favours the candidate, and
      `delta_hat_natural`, `ci_lower_natural`, `ci_upper_natural` give the natural-unit difference;
    - `estimator: ratio_of_sums` (R11-21, R13-8): Delta is the difference of the arm ratios of summed
      numerators and denominators over the repeats of complete pairs; BCa item-clustered bootstrap with the
      item jackknife; `sd_diff` and power from the linearized per-item contributions
      d_i = (num_i - R den_i) / mean(den) per arm, oriented paired difference;
    - p-values by test-interval duality (R11-5), none for no pairs or a zero-width interval.

    Design revision 18 (05 sections 3.4 and 3.6, R18-3, R18-4): a clause with `cluster_key` resamples clusters of
    complete-pair items (`clusters` maps every registered item to its value of that item field; both arms move
    together, clusters in sorted order of their values); the mean estimator's replicate is the sum of the drawn
    clusters' item differences over their item count, the ratio estimator's the ratio difference of the drawn
    clusters' sums; BCa uses the leave-one-cluster-out jackknife; fewer than two clusters is a zero-width interval;
    power and MDE use the observed design effect (cluster-robust variance over the item variance, at least 1).
    `bootstrap_variant: percentile` takes the plain (alpha, 1 - alpha) quantiles of the replicates (BCa with
    z0 = a = 0) and the p-value G(null). Without either field the result is the revision 11 result, key for key.

    Raises ValueError for a repeated ID in `items` (R11A-27) and for a constraint clause (R12A-20).
    """
    if clause.get("constraint_set") is not None:
        # R12A-20: a constraint clause's violation indicator is analysed by constraints.evaluate_constraint.
        raise ValueError("a constraint clause (with `constraint_set`) is gated by "
                         "gatedtrain.constraints.evaluate_constraint, not as a mean")
    hypothesis = clause["type"]
    if hypothesis not in HYPOTHESES:
        raise ValueError(f"unknown clause type {hypothesis!r}")
    alpha, power_target, margin = float(clause["alpha"]), float(clause["power"]), float(clause["margin"])
    k = _clause_k(clause)
    seeds = list(clause["eval_seeds"])
    direction = clause.get("direction") or "higher_is_better"
    if direction not in ("higher_is_better", "lower_is_better"):
        raise ValueError(f"unknown direction {direction!r}")
    sign = 1.0 if direction == "higher_is_better" else -1.0
    estimator = clause.get("estimator") or "mean"
    if estimator not in ("mean", "ratio_of_sums"):
        raise ValueError(f"unknown estimator {estimator!r}")
    ratio = estimator == "ratio_of_sums"
    item_list = list(items)
    index = {it: i for i, it in enumerate(item_list)}
    if len(index) != len(item_list):
        # R11A-27 (design choice): the registered item set is a set; a repeated ID is a caller error.
        raise ValueError("`items` repeats an item ID; pass the registered item set")
    variant = clause.get("bootstrap_variant") or "bca"
    if variant not in BOOTSTRAP_VARIANTS:
        raise ValueError(f"unknown bootstrap_variant {variant!r}")
    cluster_key = clause.get("cluster_key")
    if cluster_key is not None:
        # R18-3: the cluster of every registered item, read from the item set by the caller.
        if not isinstance(clusters, dict) or any(it not in clusters for it in item_list):
            raise ValueError(f"clause clusters by {cluster_key!r}: pass `clusters` with a value for every item")
        if any(not _cluster_value_ok(clusters[it]) for it in item_list):
            raise ValueError(f"clause clusters by {cluster_key!r}: a cluster value is not a string or an integer")
    de = 1.0                                                   # unit = item: per-item means absorb K

    arms = {name: _screen_arm(list((a or {}).get("records") or []), index=index, k=k, seeds=seeds, ratio=ratio,
                              score_range=score_range)
            for name, a in (("baseline", baseline), ("candidate", candidate))}
    ub, uc = arms["baseline"]["usable"], arms["candidate"]["usable"]
    paired = set(ub) & set(uc)

    # Complete pairs in registered item order (R11-3).
    common: dict[Any, list[int]] = defaultdict(list)
    for it, rep in paired:
        common[it].append(rep)
    pair_items = [it for it in item_list if it in common]
    n = len(pair_items)

    missing_reasons: list[str] = []
    if n == 0:
        missing_reasons.append("no_pairs")
    missing_b, missing_c = arms["baseline"]["missing"], arms["candidate"]["missing"]
    if _above_bound(missing_b + missing_c, len(item_list) * k, max_unpaired):
        missing_reasons.append("missing_above_bound")
    flags = arms["baseline"]["flags"] | arms["candidate"]["flags"]
    if expected_models is not None:
        for name, a in (("baseline", baseline), ("candidate", candidate)):
            got = (a or {}).get("model_manifest_hash")
            if got is None or got != expected_models.get(name):
                flags.add("model_hash_mismatch")               # 05 sections 3.3, 3.8 (R11-14)
    missing_reasons += [r for r in ("k_mismatch", "seed_mismatch", "malformed_record", "unknown_item",
                                    "model_hash_mismatch") if r in flags]

    keys = sorted(paired, key=lambda kr: (index[kr[0]], kr[1]))
    out: dict[str, Any] = {
        "hypothesis_type": hypothesis, "alpha": alpha, "ci_level": 1.0 - 2.0 * alpha, "margin": margin,
        "power_target": power_target, "design_effect": de, "n_items": n, "n_boot": int(n_boot),
        "bootstrap_seed": int(seed), "interval_method": "cluster_bootstrap", "bootstrap_variant": "bca",
        "direction": direction, "estimator": estimator,
        "unpaired_baseline": len(set(ub) - paired), "unpaired_candidate": len(set(uc) - paired),
        "missing_baseline": missing_b, "missing_candidate": missing_c,
        "missing_reasons": {"baseline": arms["baseline"]["reasons"], "candidate": arms["candidate"]["reasons"]},
        "max_unpaired": max_unpaired, "min_paired": int(min_paired),
        "k_actual": {"baseline": arms["baseline"]["k_actual"], "candidate": arms["candidate"]["k_actual"]},
        "p_value_method": "bca_inversion",
        "baseline_score": None, "candidate_score": None,
    }
    if variant != "bca":
        out.update(bootstrap_variant=variant, p_value_method=f"{variant}_inversion")      # R18-4
    if cluster_key is not None:
        out.update(resampling_unit="cluster", cluster_key=cluster_key, n_clusters=0)       # R18-3
    if n == 0:
        out.update(delta_hat=None, ci_lower=None, ci_upper=None, delta_hat_natural=None, ci_lower_natural=None,
                   ci_upper_natural=None, sd_diff=None, sd_zero=False, achieved_power=None, mde=None, p_value=None,
                   p_value_at_floor=False, underpowered=True, status=MISSING, status_reasons=missing_reasons)
        return out

    reps = {it: sorted(common[it]) for it in pair_items}
    labels = [clusters[it] for it in pair_items] if cluster_key is not None else None
    ckeys = _cluster_order(labels) if labels is not None else None
    if ckeys is not None:
        out["n_clusters"] = len(ckeys)
    if ratio:
        def sums(arm_recs, field):
            return np.array([float(sum(arm_recs[(it, r)][field] for r in reps[it])) for it in pair_items])

        num_b, den_b = sums(ub, "numerator"), sums(ub, "denominator")
        num_c, den_c = sums(uc, "numerator"), sums(uc, "denominator")
        r_b, r_c = float(num_b.sum() / den_b.sum()), float(num_c.sum() / den_c.sum())
        out.update(baseline_score=r_b, candidate_score=r_c)
        delta_hat = sign * (r_c - r_b)
        contrib = sign * ((num_c - r_c * den_c) / den_c.mean() - (num_b - r_b * den_b) / den_b.mean())
        sd = float(contrib.std(ddof=1)) if n > 1 else None
        zero_width = _zero_width(contrib) or (ckeys is not None and len(ckeys) < 2)
        if ckeys is not None and not zero_width:
            de = _cluster_design_effect(contrib, labels, ckeys)
        if zero_width:
            lower = upper = delta_hat
            p_value, p_value_at_floor = None, False
        else:
            if ckeys is not None:                               # R18-3: the ratio of the drawn clusters' sums
                num_b, den_b, num_c, den_c = (_cluster_sums(x, labels, ckeys) for x in (num_b, den_b, num_c, den_c))
            boots, z0, a = _ratio_bca(num_b, den_b, num_c, den_c, sign=sign, seed=seed, n_boot=n_boot)
            if variant == "percentile":
                z0, a = 0.0, 0.0                                # R18-4: plain quantiles
            lower, upper = _bca_bounds(boots, z0, a, alpha=alpha)
            p_value, p_value_at_floor = _bca_p_detail(boots, z0, a, hypothesis=hypothesis, margin=margin)
    else:
        b_means = np.array([sum(float(ub[(it, r)]["score"]) for r in reps[it]) / len(reps[it]) for it in pair_items],
                           dtype=np.float64)
        c_means = np.array([sum(float(uc[(it, r)]["score"]) for r in reps[it]) / len(reps[it]) for it in pair_items],
                           dtype=np.float64)
        out.update(baseline_score=sum(float(ub[kr]["score"]) for kr in keys) / len(keys),
                   candidate_score=sum(float(uc[kr]["score"]) for kr in keys) / len(keys))
        diffs = (c_means - b_means) if sign > 0 else (b_means - c_means)
        delta_hat = float(diffs.mean())
        sd = float(diffs.std(ddof=1)) if n > 1 else None
        zero_width = _zero_width(diffs)                         # R10A-14: one criterion
        if ckeys is None and variant == "bca":                  # the revision 11 path, unchanged
            lower, upper = bca_interval(diffs, alpha=alpha, seed=seed, n_boot=n_boot)
            p_value, p_value_at_floor = bca_p_value_detail(diffs, hypothesis=hypothesis, margin=margin, seed=seed,
                                                           n_boot=n_boot)
        else:
            zero_width = zero_width or (ckeys is not None and len(ckeys) < 2)
            if zero_width:
                lower = upper = delta_hat
                p_value, p_value_at_floor = None, False
            else:
                if ckeys is not None:
                    de = _cluster_design_effect(diffs, labels, ckeys)
                    boots, z0, a = _cluster_mean_parts(diffs, labels, ckeys, theta=delta_hat, seed=seed,
                                                       n_boot=n_boot)
                else:
                    boots, z0, a = _bca_parts(diffs, seed=seed, n_boot=n_boot)
                if variant == "percentile":
                    z0, a = 0.0, 0.0                            # R18-4: plain quantiles
                lower, upper = _bca_bounds(boots, z0, a, alpha=alpha)
                p_value, p_value_at_floor = _bca_p_detail(boots, z0, a, hypothesis=hypothesis, margin=margin)
    out["design_effect"] = de
    sd_zero = sd is not None and zero_width
    achieved_power, observed_mde = _power_and_mde(clause, sd=sd, sd_zero=sd_zero, n=n, alpha=alpha,
                                                  power_target=power_target, de=de)
    underpowered = achieved_power is None or achieved_power < power_target
    if plan is not None:
        n_req = plan.get("n_required")
        if plan.get("underpowered") or (n_req is not None and n < n_req):
            underpowered = True
    inconclusive_reasons: list[str] = []
    if n < min_paired:
        inconclusive_reasons.append("min_paired")
    if zero_width:
        inconclusive_reasons.append("zero_width_interval")
    passes = _passes(hypothesis, lower, upper, margin)
    if missing_reasons:
        status = MISSING
    elif inconclusive_reasons:
        status = INCONCLUSIVE
    elif passes:
        status = PASS
    elif underpowered:
        status = INCONCLUSIVE
    else:
        status = FAIL
    reasons = missing_reasons + inconclusive_reasons
    if not passes and underpowered:
        reasons.append("underpowered")
    if sign > 0:
        natural = (delta_hat, lower, upper)
    else:
        natural = (-delta_hat, -upper, -lower)
    out.update(delta_hat=delta_hat, ci_lower=lower, ci_upper=upper, delta_hat_natural=natural[0],
               ci_lower_natural=natural[1], ci_upper_natural=natural[2], sd_diff=sd, sd_zero=bool(sd_zero),
               achieved_power=achieved_power, mde=observed_mde, p_value=p_value,
               p_value_at_floor=bool(p_value_at_floor), underpowered=bool(underpowered), status=status,
               status_reasons=reasons)
    return out


def _instant(value: str) -> _dt.datetime:
    """An RFC 3339 time with an offset as an aware instant; a naive time is refused."""
    t = _dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ValueError(f"{value!r} has no UTC offset")
    return t


def is_post_hoc(*, amends: str | None, amended_at: str, judged_checkpoint: str, observed: list[dict]) -> bool:
    """05 section 3.7 (R11-1, R11A-16): an amendment is post hoc iff some observed result of the judged
    candidate checkpoint (`checkpoint == judged_checkpoint`, `arm == "candidate"`) was created strictly
    before the amendment. Times are RFC 3339 and compare as instants; `amended_at` is the amended
    registration's `created_at` (R13-7). Results of other checkpoints and baseline-arm results do not
    count."""
    if not amends:
        return False
    cut = _instant(amended_at)
    return any(o.get("checkpoint") == judged_checkpoint and o.get("arm") == "candidate"
               and _instant(o["created_at"]) < cut for o in observed)


def combine(results: list[dict], rule: str = "and") -> str:
    """Verdict over clause results (05:151; NeMo Evaluator gate.py:537 semantics; G-9)."""
    if rule != "and":
        raise ValueError(f"combine rule {rule!r} is not implemented in the skeleton (only 'and')")
    statuses = [r["status"] for r in results]
    if not statuses:
        return MISSING
    if any(s == FAIL for s in statuses):
        return FAIL
    if any(s == MISSING for s in statuses):
        return MISSING
    if any(s == INCONCLUSIVE for s in statuses):
        return INCONCLUSIVE
    if all(s == PASS for s in statuses):
        return PASS
    raise ValueError(f"unknown status in {statuses}")
