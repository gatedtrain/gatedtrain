"""Design revision 18 (tests/README.md section 11.30; design.md "Revision 18 changes" R18-1 to R18-6).

Sources: 05 section 3.1 (several blocking clauses in `clauses`, each with `name` and its deciding `evaluator`), section
3.4 (`cluster_key`: clusters of items resampled together, sorted by value, leave-one-cluster-out jackknife;
`bootstrap_variant: percentile`), section 3.5 (`multiplicity: {plan: all_clauses_must_pass}`, the `and` precedence of
section 3.8, joint power of independent clauses), section 3.6 (observed design effect), section 3.7 (per-clause report
and combined status); 04 section 4.2 (a gate lists one evaluator per clause, `uses` names the first clause's type).

The golden test reproduces the TTS recipe's pre-registered analysis, `analyze_gate.py` of area 11 (SHA-256
7b815399aa6f01e6dc330fbf852bcb288fd3aa454cac6d6af47c78b026fa9c9a, read 2026-11-17): a one-sided percentile bound of
the mean paired difference from a cluster bootstrap by speaker (`np.random.default_rng(seed).integers(0, G, (B, G))`
over the sorted speakers, replicate = summed differences of the drawn speakers over their item count), superiority on
a lower-is-better WER (PASS if the upper (1 - alpha) bound < -margin) and non-inferiority on a higher-is-better speaker
similarity (PASS if the lower bound > -margin). The expected numbers below were computed by that script's `analyze`
on the synthetic data `tts_like` generates; `gatedtrain.gates` draws the same replicates (same generator, seed, shape and
cluster order), so the rounded bounds agree exactly and the unrounded ones to float rounding.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

import _r18
from _ext import EVALUATOR_ID, ExtScenario, baseline_score, build_arm, candidate_score, evaluator_identity_hash
from _support import (EXIT_GATE_NOT_PASS, EXIT_OK, EXIT_REFUSED, SKEL_ROOT, VENV_PYTHON, cli_env, describe,
                      parse_errors, parse_run_dir)
from gatedtrain import ANALYSIS_CODE_VERSION, gates
from gatedtrain import items as items_mod
from gatedtrain.registration import (check_against_gate, check_contents, clause_for_evaluator, clause_list, is_multi,
                                  normalize, pilot_for)

WER_ID, SIM_ID = "tts.wer_qwen_canon@0.1", "tts.sim_prompt@0.1"
TTS_EVALUATORS = {
    WER_ID: {"direction": "lower_is_better", "score_range": (0.0, None), "estimators": ("mean",)},
    SIM_ID: {"direction": "higher_is_better", "score_range": (-1.0, 1.0), "estimators": ("mean",)},
}
HASH = "sha256:" + "ab" * 32


def tts_registration() -> dict:
    """The TTS recipe's registration-v2 (area 11, gate/registration-v2.yaml) in gatedtrain fields: two blocking clauses,
    quality (WER superiority, margin 0) and speaker_guard (similarity non-inferiority, margin 0.01), alpha 0.025
    each, power 0.8, fixed planning sigma 0.12 and 0.03, cluster bootstrap by speaker with percentile bounds, 10,000
    replicates, seed 20261117, max_unpaired 0."""
    common = {"unit": "item", "samples_per_item": 1, "eval_seeds": [11], "interval": "cluster_bootstrap",
              "blocking": True, "cluster_key": "speaker", "bootstrap_variant": "percentile", "sigma_plan": "fixed",
              "alpha": 0.025, "power": 0.8}
    return {
        "registration_version": 1,
        "registration_id": "mm-a11-tts-wer-superiority-v2",
        "signed_by": "prototype-unsigned",
        "workflow": "mm-a11-tts-dpo",
        "gate": "quality_gate",
        "clauses": [
            {"name": "quality", "evaluator": WER_ID, "type": "superiority", "metric": "wer_qwen_canon",
             "direction": "lower_is_better", "margin": 0.0, "planning_effect": 0.02, **common},
            {"name": "speaker_guard", "evaluator": SIM_ID, "type": "non_inferiority", "metric": "sim_prompt",
             "direction": "higher_is_better", "margin": 0.01, "planning_effect": 0.0, **common},
        ],
        "margin_justification": "quality: superiority, delta_sup = 0; speaker_guard: at most 0.01 lower similarity.",
        "sigma_pilot": {"quality": {"sigma": 0.12, "source": "dev SD 0.0993 rounded up"},
                        "speaker_guard": {"sigma": 0.03, "source": "dev SD 0.0255 rounded up"}},
        "eval_distribution": {"item_set_hash": HASH},
        "sampler": {"name": "deployment", "temperature": 1.0, "max_new_tokens": 2048},
        "evaluators": [{"id": WER_ID, "manifest_hash": HASH}, {"id": SIM_ID, "manifest_hash": HASH}],
        "multiplicity": {"plan": "all_clauses_must_pass"},
        "analysis_code_version": ANALYSIS_CODE_VERSION,
        "bootstrap": {"replicates": 10000, "seed": 20261117},
        "pairing": {"max_unpaired": 0},
    }


def tts_gate() -> dict:
    return {"uses": "gate.superiority@1.0", "evaluator": [WER_ID, SIM_ID], "sampler": "deployment",
            "test": {"alpha": 0.025, "power": 0.8, "cluster_by": "item", "seeds": 1}}


def fields(problems) -> list[tuple[str, str]]:
    return [(p.rule, p.field) for p in problems]


class TtsEvaluators(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(items_mod.EVALUATORS, TTS_EVALUATORS)
        patcher.start()
        self.addCleanup(patcher.stop)


# ---------------------------------------------------------------------------
# R18-1, R18-2: the `clauses` form of the registration
# ---------------------------------------------------------------------------

class TestTtsRegistration(TtsEvaluators):
    def test_tts_two_clause_registration_validates(self):
        self.assertEqual(check_contents(tts_registration()), [])

    def test_validates_whatever_evaluator_the_spec_names(self):
        # The clauses name their own deciding evaluators; the spec gate's evaluator does not pick one (R18-1).
        self.assertEqual(check_contents(tts_registration(), evaluator_id=WER_ID), [])
        self.assertEqual(check_contents(tts_registration(), evaluator_id="other@0.1"), [])

    def test_helpers(self):
        reg = tts_registration()
        self.assertTrue(is_multi(reg))
        self.assertEqual([c["name"] for c in clause_list(reg)], ["quality", "speaker_guard"])
        self.assertEqual(clause_for_evaluator(reg, SIM_ID)["metric"], "sim_prompt")
        self.assertIsNone(clause_for_evaluator(reg, "other@0.1"))
        self.assertEqual(pilot_for(reg, clause_list(reg)[1])["sigma"], 0.03)
        self.assertEqual([c["name"] for c in normalize(reg)["clauses"]], ["quality", "speaker_guard"])

    def test_sigma_plan_fixed_plans_each_clause(self):
        # quality: ceil((2.8016 x 0.12 / 0.02)^2) = 283, as the registration's comment says. speaker_guard: 05
        # section 3.6 plans non-inferiority at d = theta + delta = 0.01, ceil((2.8016 x 0.03 / 0.01)^2) = 71 (the
        # registration's comment says 63); both are below the 450 items.
        reg = tts_registration()
        plans = [gates.plan(c, pilot_for(reg, c), n_available=450) for c in clause_list(reg)]
        self.assertEqual([p["n_required"] for p in plans], [283, 71])
        self.assertEqual([p["sigma_used"] for p in plans], [0.12, 0.03])
        self.assertEqual([p["underpowered"] for p in plans], [False, False])

    def test_singular_clause_form_unchanged(self):
        reg = tts_registration()
        single = {k: v for k, v in reg.items() if k not in ("clauses",)}
        clause = {k: v for k, v in reg["clauses"][0].items() if k not in ("name", "evaluator")}
        single.update(clause=clause, sigma_pilot=reg["sigma_pilot"]["quality"],
                      multiplicity={"plan": "single_gate", "family": None}, evaluators=[reg["evaluators"][0]])
        self.assertFalse(is_multi(single))
        self.assertEqual(check_contents(single, evaluator_id=WER_ID), [])
        self.assertEqual(clause_list(single)[0]["metric"], "wer_qwen_canon")
        self.assertEqual(clause_for_evaluator(single, "anything")["metric"], "wer_qwen_canon")


class TestClausesRefused(TtsEvaluators):
    def refused(self, mutate, rule: str, field: str):
        reg = tts_registration()
        mutate(reg)
        got = fields(check_contents(reg))
        self.assertIn((rule, f"registration.{field}"), got, got)
        return got

    def test_both_forms(self):
        self.refused(lambda r: r.update(clause=copy.deepcopy(r["clauses"][0])), "REG-FIELD", "clauses")

    def test_one_entry(self):
        self.refused(lambda r: r["clauses"].pop(), "REG-FIELD", "clauses")

    def test_not_a_list(self):
        self.refused(lambda r: r.update(clauses={"quality": r["clauses"][0]}), "REG-FIELD", "clauses")

    def test_missing_name_and_evaluator(self):
        def m(r):
            del r["clauses"][0]["name"]
            del r["clauses"][1]["evaluator"]
        got = self.refused(m, "REG-FIELD", "clauses[0].name")
        self.assertIn(("REG-FIELD", "registration.clauses[1].evaluator"), got)

    def test_bad_and_repeated_name(self):
        self.refused(lambda r: r["clauses"][1].update(name="quality"), "REG-FIELD", "clauses[1].name")
        self.refused(lambda r: r["clauses"][1].update(name="a b"), "REG-FIELD", "clauses[1].name")

    def test_evaluator_not_listed(self):
        self.refused(lambda r: r["evaluators"].pop(), "REG-FIELD", "clauses[1].evaluator")

    def test_one_evaluator_per_clause(self):
        self.refused(lambda r: r["clauses"][1].update(evaluator=WER_ID, direction="lower_is_better"),
                     "REG-FIELD", "clauses[1].evaluator")

    def test_undeclared_evaluator(self):
        def m(r):
            r["clauses"][1]["evaluator"] = "unknown.metric@0.1"
            r["evaluators"][1]["id"] = "unknown.metric@0.1"
        self.refused(m, "REG-FIELD", "clauses[1].evaluator")

    def test_direction_against_its_own_evaluator(self):
        self.refused(lambda r: r["clauses"][1].update(direction="lower_is_better"), "REG-FIELD",
                     "clauses[1].direction")

    def test_every_clause_blocks(self):
        self.refused(lambda r: r["clauses"][1].update(blocking=False), "REG-FIELD", "clauses[1].blocking")

    def test_missing_clause_field_and_margin(self):
        self.refused(lambda r: r["clauses"][1].pop("alpha"), "REG-FIELD", "clauses[1].alpha")
        self.refused(lambda r: r["clauses"][1].pop("margin"), "REG-MARGIN", "clauses[1].margin")

    def test_unknown_and_constraint_fields(self):
        self.refused(lambda r: r["clauses"][0].update(colour="red"), "REG-FIELD", "clauses[0].colour")
        self.refused(lambda r: r["clauses"][0].update(constraint_set="cs1"), "REG-FIELD",
                     "clauses[0].constraint_set")
        self.refused(lambda r: r["clauses"][0].update(samples_per_prompt=1), "REG-FIELD",
                     "clauses[0].samples_per_prompt")

    def test_clause_value_checks_use_the_clause_path(self):
        self.refused(lambda r: r["clauses"][1].update(alpha=0.7), "REG-FIELD", "clauses[1].alpha")
        self.refused(lambda r: r["clauses"][0].update(eval_seeds=[11, 12]), "REG-FIELD", "clauses[0].eval_seeds")

    def test_multiplicity_plan(self):
        self.refused(lambda r: r.update(multiplicity={"plan": "single_gate", "family": None}), "REG-FIELD",
                     "multiplicity")
        self.refused(lambda r: r.update(multiplicity={"plan": "all_clauses_must_pass", "family": "f"}),
                     "REG-FIELD", "multiplicity")
        reg = tts_registration()
        reg["multiplicity"] = {"plan": "all_clauses_must_pass", "family": None}
        self.assertEqual(check_contents(reg), [])

    def test_all_clauses_must_pass_needs_clauses(self):
        reg = tts_registration()
        single = {k: v for k, v in reg.items() if k != "clauses"}
        single.update(clause={k: v for k, v in reg["clauses"][0].items() if k not in ("name", "evaluator")},
                      sigma_pilot=reg["sigma_pilot"]["quality"], evaluators=[reg["evaluators"][0]])
        self.assertIn(("REG-FIELD", "registration.multiplicity"), fields(check_contents(single, evaluator_id=WER_ID)))

    def test_sigma_pilot_per_clause(self):
        self.refused(lambda r: r.update(sigma_pilot={"sigma": 0.12, "source": "flat"}), "REG-FIELD",
                     "sigma_pilot.quality")
        self.refused(lambda r: r["sigma_pilot"].pop("speaker_guard"), "REG-FIELD", "sigma_pilot.speaker_guard")
        self.refused(lambda r: r["sigma_pilot"].update(other={"sigma": 1, "source": "x"}), "REG-FIELD",
                     "sigma_pilot.other")
        self.refused(lambda r: r["sigma_pilot"]["speaker_guard"].pop("sigma"), "REG-FIELD",
                     "sigma_pilot.speaker_guard.sigma")
        self.refused(lambda r: r.update(sigma_pilot=[1]), "REG-FIELD", "sigma_pilot")

    def test_planning_distance_per_clause(self):
        self.refused(lambda r: r["clauses"][1].update(planning_effect=-0.02), "REG-FIELD",
                     "clauses[1].planning_effect")

    def test_baseline_pilot_mean_per_clause(self):
        reg = tts_registration()
        reg["baseline_pilot_mean"] = {"quality": 0.0668, "speaker_guard": 0.62}
        self.assertEqual(check_contents(reg), [])
        self.refused(lambda r: r.update(baseline_pilot_mean=0.0668), "REG-FIELD", "baseline_pilot_mean")
        self.refused(lambda r: r.update(baseline_pilot_mean={"other": 0.5}), "REG-FIELD", "baseline_pilot_mean.other")
        self.refused(lambda r: r.update(baseline_pilot_mean={"speaker_guard": 1.0}), "REG-FIELD",
                     "baseline_pilot_mean.speaker_guard")


class TestClusterAndVariantFields(TtsEvaluators):
    """R18-3, R18-4: valid in both forms, refused when malformed."""

    def single(self, **clause_extra) -> dict:
        reg = tts_registration()
        single = {k: v for k, v in reg.items() if k != "clauses"}
        clause = {k: v for k, v in reg["clauses"][0].items() if k not in ("name", "evaluator")}
        clause.update(clause_extra)
        single.update(clause=clause, sigma_pilot=reg["sigma_pilot"]["quality"],
                      multiplicity={"plan": "single_gate", "family": None}, evaluators=[reg["evaluators"][0]])
        return single

    def test_valid(self):
        self.assertEqual(check_contents(self.single(), evaluator_id=WER_ID), [])
        self.assertEqual(check_contents(self.single(bootstrap_variant="bca"), evaluator_id=WER_ID), [])

    def test_cluster_key_refused(self):
        for bad in ("", "item", 3, None, ["speaker"]):
            with self.subTest(cluster_key=bad):
                got = fields(check_contents(self.single(cluster_key=bad), evaluator_id=WER_ID))
                self.assertIn(("REG-FIELD", "registration.clause.cluster_key"), got)
        reg = tts_registration()
        reg["clauses"][1]["cluster_key"] = "item"
        self.assertIn(("REG-FIELD", "registration.clauses[1].cluster_key"), fields(check_contents(reg)))

    def test_bootstrap_variant_refused(self):
        for bad in ("basic", "BCa", None, 1):
            with self.subTest(bootstrap_variant=bad):
                got = fields(check_contents(self.single(bootstrap_variant=bad), evaluator_id=WER_ID))
                self.assertIn(("REG-FIELD", "registration.clause.bootstrap_variant"), got)

    def test_constraint_clause_takes_neither(self):
        from gatedtrain.constraints import is_constraint_clause

        reg = self.single(constraint_set="cs1")
        self.assertTrue(is_constraint_clause(reg["clause"]))
        got = fields(check_contents(reg, evaluator_id=WER_ID))
        self.assertIn(("REG-FIELD", "registration.clause.cluster_key"), got)
        self.assertIn(("REG-FIELD", "registration.clause.bootstrap_variant"), got)


class TestSpecGate(TtsEvaluators):
    """R18-5 (04 section 4.2): the spec gate against the registration (interim rule G-4)."""

    def test_matching_gate(self):
        self.assertEqual(check_against_gate(tts_registration(), tts_gate(), "stages[1]"), [])

    def test_evaluator_order_and_form(self):
        gate = tts_gate()
        gate["evaluator"] = [SIM_ID, WER_ID]
        self.assertEqual(fields(check_against_gate(tts_registration(), gate, "stages[1]")),
                         [("REG-FIELD", "registration.clauses")])
        gate["evaluator"] = WER_ID
        self.assertEqual(fields(check_against_gate(tts_registration(), gate, "stages[1]")),
                         [("REG-FIELD", "registration.clauses")])

    def test_list_against_one_clause(self):
        reg = TestClusterAndVariantFields.single(self)
        self.assertEqual(fields(check_against_gate(reg, tts_gate(), "stages[1]")),
                         [("REG-FIELD", "registration.clause")])

    def test_uses_names_the_first_clause(self):
        gate = tts_gate()
        gate["uses"] = "gate.noninferiority@1.0"
        self.assertEqual(fields(check_against_gate(tts_registration(), gate, "stages[1]")),
                         [("REG-FIELD", "registration.clauses[0].type")])

    def test_test_field_applies_to_every_clause(self):
        gate = tts_gate()
        gate["test"]["margin"] = 0.0
        self.assertEqual(fields(check_against_gate(tts_registration(), gate, "stages[1]")),
                         [("REG-FIELD", "registration.clauses[1].margin")])
        gate["test"] = {"seeds": 2}
        self.assertEqual(fields(check_against_gate(tts_registration(), gate, "stages[1]")),
                         [("REG-FIELD", "registration.clauses[0].eval_seeds"),
                          ("REG-FIELD", "registration.clauses[1].eval_seeds")])


# ---------------------------------------------------------------------------
# R18-3, R18-4: cluster bootstrap and percentile bounds (gates.evaluate_items)
# ---------------------------------------------------------------------------

def tts_like(seed: int, *, wer_shift: float, sim_shift: float, n_items: int = 450, n_speakers: int = 38):
    """Synthetic TTS-shaped arms: items `item-<i>` with a string `speaker`, per-item WER (lower is better) and speaker
    similarity (higher is better) for the baseline and the candidate; a speaker effect shared by both arms and a
    speaker-level candidate effect make items of one speaker correlated."""
    rng = np.random.default_rng(seed)
    speakers = [str(s) for s in rng.choice(np.arange(100, 9000), size=n_speakers, replace=False)]
    spk_of = rng.integers(0, n_speakers, size=n_items)
    spk_wer = rng.normal(0.0, 0.03, size=n_speakers)
    spk_gain = rng.normal(0.0, 0.01, size=n_speakers)
    spk_sim = rng.normal(0.0, 0.05, size=n_speakers)
    base, cand = {}, {}
    for i in range(n_items):
        s = int(spk_of[i])
        wb = max(0.0, 0.07 + spk_wer[s] + rng.normal(0.0, 0.08))
        wc = max(0.0, wb + wer_shift + spk_gain[s] + rng.normal(0.0, 0.06))
        sb = min(1.0, 0.62 + spk_sim[s] + rng.normal(0.0, 0.03))
        sc = min(1.0, sb + sim_shift + rng.normal(0.0, 0.02))
        item = f"item-{i:03d}"
        base[item] = {"item": item, "speaker": speakers[s], "wer_qwen_canon": round(float(wb), 6),
                      "sim_prompt": round(float(sb), 6)}
        cand[item] = {"item": item, "speaker": speakers[s], "wer_qwen_canon": round(float(wc), 6),
                      "sim_prompt": round(float(sc), 6)}
    return base, cand


def cluster_bootstrap_means(d: np.ndarray, clusters: np.ndarray, replicates: int, seed: int) -> np.ndarray:
    """Verbatim from area 11 analyze_gate.py (SHA-256 7b815399...), the reference replicates."""
    keys = np.unique(clusters)
    sums = np.array([d[clusters == k].sum() for k in keys])
    counts = np.array([(clusters == k).sum() for k in keys])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(replicates, len(keys)))
    return sums[draws].sum(axis=1) / counts[draws].sum(axis=1)


# analyze_gate.analyze on tts_like(seed, ...): {scenario: (seed, wer_shift, sim_shift, gate verdict,
# [(clause, mean_diff, bound, verdict), ...])}; bound is the upper (1 - alpha) bound for quality (lower is better)
# and the lower one for speaker_guard (higher is better), natural units, rounded to 5 decimals as the script reports.
GOLDEN = {
    "pass": (7, -0.03, 0.0, "PASS", [("quality", -0.01082, -0.0052, "PASS"),
                                     ("speaker_guard", 0.00108, -0.00087, "PASS")]),
    "guard_fails": (8, -0.02, -0.02, "FAIL", [("quality", -0.01057, -0.00423, "PASS"),
                                              ("speaker_guard", -0.02025, -0.02182, "FAIL")]),
    "quality_fails": (9, 0.01, 0.0, "FAIL", [("quality", 0.02065, 0.02646, "FAIL"),
                                             ("speaker_guard", 0.00115, -0.00051, "PASS")]),
}


def records(arm: dict, metric: str, items: list) -> dict:
    return {"records": [{"item": i, "repeat": 0, "seed": 11, "score": arm[i][metric], "error": None} for i in items]}


def evaluate_tts(base: dict, cand: dict, clause: dict, **kw) -> dict:
    items = sorted(base)
    return gates.evaluate_items(clause, records(base, clause["metric"], items), records(cand, clause["metric"], items),
                                items=items, seed=20261117, n_boot=kw.pop("n_boot", 10000), min_paired=10,
                                max_unpaired=0, clusters={i: base[i]["speaker"] for i in items}, **kw)


class TestGoldenAnalyzeGate(unittest.TestCase):
    def test_reproduces_analyze_gate(self):
        clauses = clause_list(tts_registration())
        for scenario, (seed, ws, ss, verdict, expected) in GOLDEN.items():
            base, cand = tts_like(seed, wer_shift=ws, sim_shift=ss)
            results = []
            for clause, (name, mean_diff, bound, clause_verdict) in zip(clauses, expected):
                with self.subTest(scenario=scenario, clause=name):
                    r = evaluate_tts(base, cand, clause)
                    lower_better = clause["direction"] == "lower_is_better"
                    got_bound = r["ci_upper_natural"] if lower_better else r["ci_lower_natural"]
                    self.assertEqual(round(r["delta_hat_natural"], 5), mean_diff)
                    self.assertEqual(round(got_bound, 5), bound)
                    self.assertEqual(r["n_items"], 450)
                    self.assertEqual(r["n_clusters"], 38)
                    self.assertEqual(r["status"] == gates.PASS, clause_verdict == "PASS")
                    # Unrounded: the reference replicates' quantile, to float rounding.
                    items = sorted(base)
                    d = np.array([cand[i][clause["metric"]] - base[i][clause["metric"]] for i in items])
                    boot = cluster_bootstrap_means(d, np.array([base[i]["speaker"] for i in items]), 10000,
                                                   20261117)
                    alpha = clause["alpha"]
                    ref = float(np.quantile(boot, 1 - alpha if lower_better else alpha))
                    self.assertAlmostEqual(got_bound, ref, delta=1e-12)
                    results.append(r)
            self.assertEqual(gates.combine(results, "and") == gates.PASS, verdict == "PASS", scenario)

    def test_result_fields(self):
        base, cand = tts_like(7, wer_shift=-0.03, sim_shift=0.0)
        r = evaluate_tts(base, cand, clause_list(tts_registration())[0])
        self.assertEqual((r["resampling_unit"], r["cluster_key"], r["bootstrap_variant"], r["p_value_method"]),
                         ("cluster", "speaker", "percentile", "percentile_inversion"))
        self.assertGreater(r["design_effect"], 1.0)                 # speakers carry a shared effect
        self.assertGreater(r["p_value"], 0)
        self.assertLess(r["p_value"], 0.025)                        # PASS at alpha 0.025, by duality


def gate_inputs(n: int = 40, *, seed: int = 3, k: int = 1):
    rng = np.random.default_rng(seed)
    items = [f"i{j:02d}" for j in range(n)]
    base = {"records": [], "model_manifest_hash": None}
    cand = {"records": [], "model_manifest_hash": None}
    for it in items:
        for r in range(k):
            b = float(rng.normal(0.5, 0.2))
            base["records"].append({"item": it, "repeat": r, "seed": 11 + r, "score": b, "error": None})
            cand["records"].append({"item": it, "repeat": r, "seed": 11 + r, "score": b + float(rng.normal(0.1, 0.1)),
                                    "error": None})
    return items, base, cand


def clause(**extra) -> dict:
    c = {"type": "superiority", "metric": "m", "direction": "higher_is_better", "margin": 0.0, "alpha": 0.025,
         "power": 0.8, "planning_effect": 0.1, "sigma_plan": "fixed", "unit": "item", "samples_per_item": 1,
         "eval_seeds": [11], "interval": "cluster_bootstrap", "blocking": True}
    c.update(extra)
    return c


class TestClusterBootstrap(unittest.TestCase):
    def test_default_unchanged(self):
        items, base, cand = gate_inputs()
        plain = gates.evaluate_items(clause(), base, cand, items=items, seed=5, n_boot=2000)
        explicit = gates.evaluate_items(clause(bootstrap_variant="bca"), base, cand, items=items, seed=5, n_boot=2000)
        self.assertEqual(plain, explicit)
        self.assertNotIn("resampling_unit", plain)
        self.assertEqual((plain["bootstrap_variant"], plain["p_value_method"]), ("bca", "bca_inversion"))

    def test_singleton_clusters_equal_the_item_bootstrap(self):
        # Each item its own cluster, items in sorted order: the same draws, so the same interval (BCa and percentile)
        # up to the float rounding of sum/count against mean.
        items, base, cand = gate_inputs()
        for variant in ("bca", "percentile"):
            with self.subTest(variant=variant):
                plain = gates.evaluate_items(clause(bootstrap_variant=variant), base, cand, items=items, seed=5,
                                             n_boot=2000)
                clustered = gates.evaluate_items(clause(bootstrap_variant=variant, cluster_key="id"), base, cand,
                                                 items=items, seed=5, n_boot=2000, clusters={i: i for i in items})
                for key in ("ci_lower", "ci_upper", "p_value", "delta_hat"):
                    self.assertAlmostEqual(plain[key], clustered[key], delta=1e-12)
                self.assertEqual(clustered["n_clusters"], len(items))
                self.assertEqual(clustered["design_effect"], 1.0)

    def test_percentile_is_plain_quantiles(self):
        items, base, cand = gate_inputs()
        r = gates.evaluate_items(clause(bootstrap_variant="percentile"), base, cand, items=items, seed=5, n_boot=2000)
        d = np.array([c["score"] - b["score"] for b, c in zip(base["records"], cand["records"])])
        rng = np.random.default_rng(5)
        boots = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(axis=1)
        self.assertAlmostEqual(r["ci_lower"], float(np.quantile(boots, 0.025)), delta=1e-15)
        self.assertAlmostEqual(r["ci_upper"], float(np.quantile(boots, 0.975)), delta=1e-15)
        g = (np.sum(boots < 0.0) + 0.5 * np.sum(boots == 0.0)) / len(boots)
        self.assertAlmostEqual(r["p_value"], max(float(g), 1 / 2001), delta=1e-15)

    def test_lower_is_better_orientation(self):
        items, base, cand = gate_inputs()
        up = gates.evaluate_items(clause(bootstrap_variant="percentile"), base, cand, items=items, seed=5, n_boot=2000)
        down = gates.evaluate_items(clause(bootstrap_variant="percentile", direction="lower_is_better"), base, cand,
                                    items=items, seed=5, n_boot=2000)
        self.assertAlmostEqual(down["delta_hat"], -up["delta_hat"], delta=1e-15)
        self.assertAlmostEqual(down["ci_upper_natural"], up["ci_upper"], delta=1e-12)
        self.assertAlmostEqual(down["ci_lower_natural"], up["ci_lower"], delta=1e-12)
        self.assertEqual(down["status"], gates.FAIL)                  # the candidate is worse on a lower-is-better metric

    def test_clusters_widen_the_interval(self):
        items, base, cand = gate_inputs(n=60)
        groups = {it: f"g{j // 10}" for j, it in enumerate(items)}
        # A shared shift per group: correlated differences within groups.
        shift = {f"g{g}": (g - 2.5) * 0.08 for g in range(6)}
        for rec in cand["records"]:
            rec["score"] += shift[groups[rec["item"]]]
        plain = gates.evaluate_items(clause(), base, cand, items=items, seed=5, n_boot=2000)
        clustered = gates.evaluate_items(clause(cluster_key="g"), base, cand, items=items, seed=5, n_boot=2000,
                                         clusters=groups)
        self.assertEqual(clustered["n_clusters"], 6)
        self.assertGreater(clustered["ci_upper"] - clustered["ci_lower"], plain["ci_upper"] - plain["ci_lower"])
        self.assertGreater(clustered["design_effect"], 1.5)
        self.assertLessEqual(clustered["achieved_power"], plain["achieved_power"])
        self.assertAlmostEqual(clustered["delta_hat"], plain["delta_hat"], delta=1e-15)

    def test_cluster_order_is_by_value_not_item_order(self):
        items, base, cand = gate_inputs(n=30)
        groups = {it: str(j % 7) for j, it in enumerate(items)}           # 7 clusters of 4 or 5 items
        renamed = {it: str((int(g) + 1) % 7) for it, g in groups.items()}  # the same clusters, other sort order
        a = gates.evaluate_items(clause(cluster_key="g"), base, cand, items=items, seed=5, n_boot=500, clusters=groups)
        b = gates.evaluate_items(clause(cluster_key="g"), base, cand, items=list(reversed(items)), seed=5,
                                 n_boot=500, clusters=groups)
        c = gates.evaluate_items(clause(cluster_key="g"), base, cand, items=items, seed=5, n_boot=500,
                                 clusters=renamed)
        self.assertAlmostEqual(a["ci_lower"], b["ci_lower"], delta=1e-12)     # item order does not matter
        self.assertNotEqual(a["ci_lower"], c["ci_lower"])                    # the value order does

    def test_integer_cluster_values(self):
        items, base, cand = gate_inputs(n=30)
        r = gates.evaluate_items(clause(cluster_key="g"), base, cand, items=items, seed=5, n_boot=500,
                                 clusters={it: j % 5 for j, it in enumerate(items)})
        self.assertEqual(r["n_clusters"], 5)

    def test_one_cluster_is_zero_width(self):
        items, base, cand = gate_inputs()
        r = gates.evaluate_items(clause(cluster_key="g"), base, cand, items=items, seed=5, n_boot=500,
                                 clusters={it: "all" for it in items})
        self.assertEqual(r["status"], gates.INCONCLUSIVE)
        self.assertIn("zero_width_interval", r["status_reasons"])
        self.assertIsNone(r["p_value"])
        self.assertEqual(r["ci_lower"], r["ci_upper"])

    def test_clusters_required(self):
        items, base, cand = gate_inputs(n=12)
        with self.assertRaises(ValueError):
            gates.evaluate_items(clause(cluster_key="g"), base, cand, items=items, seed=5, n_boot=100)
        with self.assertRaises(ValueError):
            gates.evaluate_items(clause(cluster_key="g"), base, cand, items=items, seed=5, n_boot=100,
                                 clusters={it: "a" for it in items[1:]})
        with self.assertRaises(ValueError):
            gates.evaluate_items(clause(cluster_key="g"), base, cand, items=items, seed=5, n_boot=100,
                                 clusters={it: True for it in items})
        with self.assertRaises(ValueError):
            gates.evaluate_items(clause(bootstrap_variant="basic"), base, cand, items=items, seed=5, n_boot=100)

    def test_clusters_of_complete_pairs_only(self):
        # A cluster whose items are all missing in one arm takes no part (G counts clusters of complete pairs).
        items, base, cand = gate_inputs(n=30)
        groups = {it: f"g{j // 10}" for j, it in enumerate(items)}
        cand["records"] = [r for r in cand["records"] if groups[r["item"]] != "g2"]
        r = gates.evaluate_items(clause(cluster_key="g"), base, cand, items=items, seed=5, n_boot=500,
                                 clusters=groups, max_unpaired=10)
        self.assertEqual((r["n_items"], r["n_clusters"]), (20, 2))

    def test_ratio_of_sums_clusters(self):
        items, base, cand = gate_inputs(n=30)
        for arm in (base, cand):
            for rec in arm["records"]:
                rec["numerator"], rec["denominator"] = rec["score"] * 10, 10.0
        c = clause(estimator="ratio_of_sums", bootstrap_variant="percentile")
        plain = gates.evaluate_items(c, base, cand, items=items, seed=5, n_boot=1000)
        single = gates.evaluate_items({**c, "cluster_key": "id"}, base, cand, items=items, seed=5, n_boot=1000,
                                      clusters={i: i for i in items})
        self.assertAlmostEqual(plain["ci_lower"], single["ci_lower"], delta=1e-12)
        grouped = gates.evaluate_items({**c, "cluster_key": "g"}, base, cand, items=items, seed=5, n_boot=1000,
                                       clusters={it: f"g{j // 5}" for j, it in enumerate(items)})
        self.assertEqual(grouped["n_clusters"], 6)
        self.assertAlmostEqual(grouped["delta_hat"], plain["delta_hat"], delta=1e-15)


# ---------------------------------------------------------------------------
# R18-1 to R18-6 end to end: ext-collect with two clauses (synthetic adapter with a second evaluator, `_r18`)
# ---------------------------------------------------------------------------

def run_r18(args: list[str], *, cwd: Path, env: dict[str, str]):
    import subprocess

    full = {**cli_env(), **env}
    full["PYTHONPATH"] = os.pathsep.join([str(SKEL_ROOT / "src"), str(SKEL_ROOT / "tests")])
    return subprocess.run([str(VENV_PYTHON), "-m", "_r18", *args], cwd=str(cwd), env=full, capture_output=True,
                          text=True, timeout=300)


def length_score(i: int) -> float:
    return float(100 + (i * 37) % 50)


def two_clause_registration(reg: dict) -> None:
    """The fixture registration with a second, lower-is-better clause on synthetic.length@0.1, clustered by `group`
    with percentile bounds (non-inferiority: the candidate may be at most 15 tokens longer)."""
    first = reg.pop("clause")
    second = {"name": "length", "evaluator": _r18.LENGTH_ID, "type": "non_inferiority", "metric": _r18.LENGTH_METRIC,
              "direction": "lower_is_better", "margin": 15.0, "alpha": 0.025, "power": 0.8, "planning_effect": 0.0,
              "sigma_plan": "fixed", "unit": "item", "samples_per_item": 2, "eval_seeds": [11, 12],
              "interval": "cluster_bootstrap", "blocking": True, "cluster_key": "group",
              "bootstrap_variant": "percentile"}
    reg["clauses"] = [{"name": "accuracy", "evaluator": EVALUATOR_ID, **first}, second]
    reg["sigma_pilot"] = {"accuracy": reg["sigma_pilot"], "length": {"sigma": 10.0, "source": "fixture value"}}
    reg["evaluators"].append({"id": _r18.LENGTH_ID, "manifest_hash": _r18.length_identity_hash()})
    reg["multiplicity"] = {"plan": "all_clauses_must_pass"}


def two_clause_spec(doc: dict) -> None:
    doc["stages"][1]["evaluator"] = [EVALUATOR_ID, _r18.LENGTH_ID]
    doc["stages"][1]["test"] = {"alpha": 0.025, "power": 0.8, "cluster_by": "item"}


def with_groups(rows: list[dict]) -> list[dict]:
    return [{**r, "group": f"g{j % 6}"} for j, r in enumerate(rows)]


class CollectTwoClauses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import _ext

        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-r18-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        real_build = _ext.build_item_set

        def build(directory, rows, *, ids=None):        # the fixture item set with a `group` field per item
            real_build(directory, rows, ids=ids)
            lines = [json.loads(l) for l in (directory / "items.jsonl").read_text(encoding="utf-8").splitlines()]
            (directory / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in with_groups(lines)),
                                                   encoding="utf-8")
            return directory

        with mock.patch.object(_ext, "build_item_set", build):
            cls.sc = ExtScenario(cls.tmp / "sc", reg_mutate=two_clause_registration)
        cls.sc.render(reg_mutate=two_clause_registration, spec_mutate=two_clause_spec)
        # R18-8: the smoke record carries both clause evaluators' baseline scores.
        cls.smoke_scores = {EVALUATOR_ID: [1.0] * 11 + [0.0] * 9,
                            _r18.LENGTH_ID: [float(100 + 7 * j % 23) for j in range(20)]}
        cls.write_smoke(cls.sc.smoke, cls.smoke_scores)
        cls.length_base = cls.length_arm("baseline-length", length_score)
        cls.length_cand = cls.length_arm("candidate-length", lambda i: length_score(i) + (i % 5))   # mean +2
        cls.n = 0

    @classmethod
    def write_smoke(cls, path: Path, by_evaluator: dict | None) -> Path:
        from _ext import smoke_record, write_canonical

        extra = {"baseline_scores_by_evaluator": by_evaluator} if by_evaluator is not None else {}
        return write_canonical(path, smoke_record(base_model=cls.sc.base_model, item_set=cls.sc.item_set_hash,
                                                  **extra))

    @classmethod
    def length_arm(cls, name: str, score_fn, **kw) -> Path:
        meta = {"evaluator_id": _r18.LENGTH_ID, "metric": _r18.LENGTH_METRIC, "direction": "lower_is_better",
                "score_range": [0, None], "scorer": dict(_r18.LENGTH_SCORER)}
        return cls.sc.arm(name, score_fn, meta=meta, **kw)

    def runs(self) -> Path:
        type(self).n += 1
        return self.tmp / f"runs-{type(self).n}"

    def collect(self, runs: Path, *, arms: list[tuple[Path, Path]] | None = None, spec: Path | None = None,
                smoke: Path | None = None):
        sc = self.sc
        arms = arms if arms is not None else [(sc.baseline, sc.candidate), (self.length_base, self.length_cand)]
        args = ["ext-collect", str(spec or sc.paths["spec"]), "--adapter", "synthetic", "--stage-out",
                str(sc.stage_out), "--launch-record", str(sc.launch), "--smoke-record", str(smoke or sc.smoke)]
        for b, c in arms:
            args += ["--eval-baseline", str(b), "--eval-candidate", str(c)]
        return run_r18(args + ["--runs-root", str(runs)], cwd=sc.root, env=sc.env())

    def test_validate(self):
        proc = run_r18(["validate", str(self.sc.paths["spec"])], cwd=self.sc.root, env=self.sc.env())
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_collect_two_clauses_pass(self):
        runs = self.runs()
        proc = self.collect(runs)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        run = parse_run_dir(proc.stdout)
        rep = json.loads((run / "report" / "gate_report.json").read_text(encoding="utf-8"))
        self.assertEqual((rep["status"], rep["status_reasons"], rep["multiplicity"]),
                         ("PASS", [], "all_clauses_must_pass"))
        self.assertAlmostEqual(rep["joint_power_independent"], 0.64)
        self.assertEqual([c["name"] for c in rep["clauses"]], ["accuracy", "length"])
        acc, length = rep["clauses"]
        self.assertEqual((acc["evaluator"], acc["hypothesis_type"], acc["status"]),
                         (EVALUATOR_ID, "superiority", "PASS"))
        self.assertEqual((length["evaluator"], length["hypothesis_type"], length["direction"], length["status"]),
                         (_r18.LENGTH_ID, "non_inferiority", "lower_is_better", "PASS"))
        self.assertEqual(length["interval"]["bootstrap"],
                         {"variant": "percentile", "replicates": 10000, "seed": 20261109, "resampling_unit": "cluster",
                          "cluster_key": "group", "n_clusters": 6})
        self.assertEqual(acc["interval"]["bootstrap"]["resampling_unit"], "item")
        self.assertAlmostEqual(length["natural_units"]["delta_hat"], 2.0)
        self.assertEqual(acc["evaluator_hash"], evaluator_identity_hash())
        self.assertEqual(length["evaluator_hash"], _r18.length_identity_hash())
        self.assertEqual(length["evaluator_hash"], length["evaluator_hash_registered"])
        self.assertEqual(length["metric_source"], "items.jsonl score (token_count)")
        for key in ("hypothesis_type", "metric", "delta_hat", "interval"):
            self.assertNotIn(key, rep)
        plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual([c["name"] for c in plan["clauses"]], ["accuracy", "length"])
        self.assertEqual(plan["clauses"][1]["sigma_used"], 10.0)
        lock = json.loads((run / "lockfile.json").read_text(encoding="utf-8"))
        self.assertEqual(lock["backend"]["evaluator"]["uses"], [EVALUATOR_ID, _r18.LENGTH_ID])
        self.assertEqual(lock["direction"], ["higher_is_better", "lower_is_better"])
        for arm in ("baseline", "candidate"):
            for name in ("accuracy", "length"):
                self.assertTrue((run / "eval" / arm / name / "items.jsonl").is_file())
        text = (run / "report" / "gate_report.txt").read_text(encoding="utf-8")
        self.assertIn("multiplicity: all_clauses_must_pass", text)
        self.assertIn("clause length: PASS", text)
        self.assertIn("resampling 6 clusters by group", text)

    def test_one_clause_fails_the_gate(self):
        cand = self.length_arm("candidate-length-long", lambda i: length_score(i) + 40.0 + (i % 5))
        proc = self.collect(self.runs(), arms=[(self.sc.baseline, self.sc.candidate), (self.length_base, cand)])
        self.assertEqual(proc.returncode, EXIT_GATE_NOT_PASS, describe(proc))
        rep = json.loads((parse_run_dir(proc.stdout) / "report" / "gate_report.json").read_text(encoding="utf-8"))
        self.assertEqual((rep["status"], rep["status_reasons"]), ("FAIL", ["length: FAIL"]))
        self.assertEqual([c["status"] for c in rep["clauses"]], ["PASS", "FAIL"])

    def test_missing_clause_and_failing_clause(self):
        # `and` precedence (05 section 3.8): FAIL before MISSING.
        cand_acc = self.sc.arm("candidate-acc-missing", candidate_score, drop={(self.sc.items[0], 0)})
        cand_len = self.length_arm("candidate-length-long2", lambda i: length_score(i) + 40.0 + (i % 5))
        proc = self.collect(self.runs(), arms=[(self.sc.baseline, cand_acc), (self.length_base, cand_len)])
        self.assertEqual(proc.returncode, EXIT_GATE_NOT_PASS, describe(proc))
        rep = json.loads((parse_run_dir(proc.stdout) / "report" / "gate_report.json").read_text(encoding="utf-8"))
        self.assertEqual([c["status"] for c in rep["clauses"]], ["MISSING", "FAIL"])
        self.assertEqual((rep["status"], rep["status_reasons"]), ("FAIL", ["accuracy: MISSING", "length: FAIL"]))

    def test_smoke_checks_every_clause_evaluator(self):
        # R18-8: without per-evaluator scores, or with a degenerate second evaluator, the collect is refused.
        cases = {"none": None,
                 "constant": {**self.smoke_scores, _r18.LENGTH_ID: [120.0] * 20},
                 "missing": {EVALUATOR_ID: self.smoke_scores[EVALUATOR_ID]}}
        for name, by in cases.items():
            with self.subTest(case=name):
                smoke = self.write_smoke(self.tmp / f"smoke-{name}.json", by)
                runs = self.runs()
                proc = self.collect(runs, smoke=smoke)
                self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
                errs = parse_errors(proc.stderr)
                self.assertTrue(errs and all((e["rule"], e["field"]) == ("EXT-SMOKE", "--smoke-record") for e in errs),
                                errs)
                self.assertFalse(runs.exists() and any(runs.iterdir()))

    def test_launch_reads_every_clause_evaluator(self):
        # R18-8: ext-launch with --spec checks the smoke record against every clause's evaluator.
        from gatedtrain.ext.launch import _clause_evaluators

        with _r18.patched():
            got = _clause_evaluators(self.sc.paths["spec"])
        self.assertEqual([ev for ev, _ in got], [EVALUATOR_ID, _r18.LENGTH_ID])
        self.assertEqual(got[1][1]["direction"], "lower_is_better")

    def test_arm_count_refused(self):
        runs = self.runs()
        proc = self.collect(runs, arms=[(self.sc.baseline, self.sc.candidate)])
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        got = {(e["rule"], e["field"]) for e in parse_errors(proc.stderr)}
        self.assertEqual(got, {("EXT-SPEC", "--eval-baseline"), ("EXT-SPEC", "--eval-candidate")})
        self.assertFalse(runs.exists() and any(runs.iterdir()))

    def test_arms_in_the_wrong_order_refused(self):
        runs = self.runs()
        proc = self.collect(runs, arms=[(self.length_base, self.length_cand), (self.sc.baseline, self.sc.candidate)])
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        rules = {e["rule"] for e in parse_errors(proc.stderr)}
        self.assertEqual(rules, {"EXT-ARM"})
        self.assertFalse(runs.exists() and any(runs.iterdir()))

    def test_same_arm_through_a_link_refused(self):
        """Review of the 3.12 path fix (S4): arms are told apart by the directory itself (`st_dev`, `st_ino`), not by
        a resolved path string, so one arm passed again through a symbolic link is passed twice."""
        link = self.tmp / "length-base-link"
        if not link.is_symlink():
            link.symlink_to(self.length_base)
        runs = self.runs()
        proc = self.collect(runs, arms=[(self.sc.baseline, self.sc.candidate), (self.length_base, link)])
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        hits = [e for e in parse_errors(proc.stderr) if e["rule"] == "EXT-ARM" and "passed twice" in e["message"]]
        self.assertEqual([e["field"] for e in hits], ["--eval-candidate"], describe(proc))
        self.assertNotIn("Traceback", proc.stderr)
        self.assertFalse(runs.exists() and any(runs.iterdir()))

    def collect_in_process(self, arms: list[tuple[Path, Path]], runs: Path, hook=None):
        """ext-collect in this process (review of PR 2): `hook()` runs after every check passed, before the arms are
        copied (at `runner.copy_registration`), as a container still writing its arms would."""
        from gatedtrain import runner
        from gatedtrain.ext.collect import collect

        real = runner.copy_registration

        def hooked(wf, run_dir):
            if hook is not None:
                hook()
            return real(wf, run_dir)

        sc = self.sc
        with _r18.patched(), mock.patch.dict(os.environ, sc.env()), \
                mock.patch.object(runner, "copy_registration", hooked):
            return collect(sc.paths["spec"], adapter="synthetic", stage_out=sc.stage_out, launch_record=sc.launch,
                           smoke_record=sc.smoke, eval_baseline=[b for b, _ in arms],
                           eval_candidate=[c for _, c in arms], runs_root=runs)

    def test_same_arm_past_path_max_refused(self):
        """Review of PR 2 (C1): the S4 alias that a resolved path string misses. With the working directory deeper
        than PATH_MAX, `arm` (a link to the length baseline) and `lnk` (a link to `arm`) cannot be inspected by path,
        so the lenient `realpath` keeps both names and gives two keys; the opened directory is one."""
        from gatedtrain.errors import Refused

        deep = self.tmp / f"deep-{type(self).n}"
        deep.mkdir()
        fd = os.open(deep, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for name in ["d" * 250] * 17:
                os.mkdir(name, dir_fd=fd)
                nxt = os.open(name, os.O_RDONLY | os.O_DIRECTORY, dir_fd=fd)
                os.close(fd)
                fd = nxt
            os.symlink(str(self.length_base), "arm", dir_fd=fd)
            os.symlink("arm", "lnk", dir_fd=fd)
            cwd = os.getcwd()
            os.fchdir(fd)
        finally:
            os.close(fd)
        try:
            self.assertGreater(len(os.getcwd()), os.pathconf("/", "PC_PATH_MAX"))
            with self.assertRaises(Refused) as cm:
                self.collect_in_process([(self.sc.baseline, self.sc.candidate), (Path("arm"), Path("lnk"))],
                                        self.runs())
        finally:
            os.chdir(cwd)
        hits = [p for p in cm.exception.problems if p.rule == "EXT-ARM" and "passed twice" in p.message]
        self.assertEqual([p.field for p in hits], ["--eval-candidate"], cm.exception.problems)

    def test_arm_files_changed_after_the_checks(self):
        """Review of PR 2 (A1, C2), several clauses: the run holds the bytes that were checked (`items.jsonl`
        swapped for a link to a host file, `eval_meta.json` rewritten and a `merge.json` link added after the checks
        change nothing), and an output swapped for a link after the checks refuses the collect, leaving no run."""
        from gatedtrain.errors import Refused

        secret = self.tmp / "HOST_SECRET"
        secret.write_bytes(b"host secret\n")
        cand = self.length_arm("candidate-length-swap", lambda i: length_score(i) + (i % 5))
        before = {n: (cand / n).read_bytes() for n in ("items.jsonl", "eval_meta.json")}

        def swap_files():
            (cand / "items.jsonl").unlink()
            (cand / "items.jsonl").symlink_to(secret)
            (cand / "eval_meta.json").write_bytes(b"{}")
            (cand / "merge.json").symlink_to(secret)

        run_dir, _, _ = self.collect_in_process([(self.sc.baseline, self.sc.candidate), (self.length_base, cand)],
                                                self.runs(), swap_files)
        got = run_dir / "eval" / "candidate" / "length"
        self.assertEqual({n: (got / n).read_bytes() for n in before}, before)
        self.assertFalse((got / "merge.json").exists())

        out = self.length_arm("candidate-length-ref", lambda i: length_score(i) + (i % 5))
        lines = [json.loads(l) for l in (out / "items.jsonl").read_text(encoding="utf-8").splitlines()]
        (out / "outputs").mkdir()
        (out / "outputs" / "a.txt").write_bytes(b"output")
        lines[0]["output_ref"] = {"path": "outputs/a.txt", "sha256": "sha256:" + hashlib.sha256(b"output").hexdigest()}
        (out / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")

        def swap_output():
            (out / "outputs" / "a.txt").unlink()
            (out / "outputs" / "a.txt").symlink_to(secret)

        runs = self.runs()
        with self.assertRaises(Refused) as cm:
            self.collect_in_process([(self.sc.baseline, self.sc.candidate), (self.length_base, out)], runs,
                                    swap_output)
        (p,) = cm.exception.problems
        self.assertEqual((p.rule, p.field), ("EXT-ARM", "--eval-candidate"))
        self.assertIn("contains a symbolic link", p.message)
        self.assertFalse(runs.exists() and any(runs.iterdir()))

    def test_item_without_cluster_field_refused(self):
        # R18-3: an item set without the clause's cluster field refuses the collect; the registration follows the
        # new item set hash, so the field is the only problem.
        sc = self.sc
        alt = self.tmp / "alt"
        shutil.copytree(sc.root, alt, symlinks=True)
        lines = [json.loads(l) for l in (alt / "items" / "items.jsonl").read_text(encoding="utf-8").splitlines()]
        del lines[3]["group"]
        lines[5]["group"] = True
        (alt / "items" / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
        from _support import manifest_hash

        new_hash = manifest_hash(alt / "items")
        reg = yaml.safe_load((alt / "registration.yaml").read_text(encoding="utf-8"))
        reg["eval_distribution"]["item_set_hash"] = new_hash
        (alt / "registration.yaml").write_text(yaml.safe_dump(reg, sort_keys=False), encoding="utf-8")
        from _support import file_sha256

        spec = yaml.safe_load((alt / "spec.yaml").read_text(encoding="utf-8"))
        spec["artifacts"]["registration"]["uri"] = \
            f"file:registration.yaml@{file_sha256(alt / 'registration.yaml')}"
        (alt / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        runs = self.runs()
        proc = self.collect(runs, spec=alt / "spec.yaml")
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = parse_errors(proc.stderr)
        hit = [e for e in errs if (e["rule"], e["field"]) == ("EXT-ITEMS", "registration.clauses[1].cluster_key")]
        self.assertEqual(len(hit), 1, errs)
        self.assertIn("2 item(s)", hit[0]["message"])
        self.assertFalse(runs.exists() and any(runs.iterdir()))


class TestSmokeBound(unittest.TestCase):
    """R18-8 (05 section 6.1 pre-run baseline check; MULTIMODAL-SPEC MM-38; maintainers' decision B, 2026-11-17): scores on the
    finite bound at the metric's best end (lower bound if lower is better, upper bound if higher is better) do not
    count toward the one-value-makes-half rule; at least a quarter (and 2) of the scores must lie off that bound."""

    WER_RANGE = (0, None)

    def verdict(self, scores, score_range=WER_RANGE, binary=False, direction="lower_is_better"):
        from _ext import smoke_record
        from gatedtrain.ext.smoke import smoke_verdict

        return smoke_verdict(smoke_record(base_model="m", item_set="i", baseline_scores=scores),
                             score_range=score_range, binary=binary, direction=direction)

    def test_tts_shaped_passes(self):
        scores = [0.0] * 14 + [0.05, 0.111, 0.0625, 0.2, 0.0833, 0.3]
        self.assertEqual(self.verdict(scores), {"verdict": "PASS", "reasons": []})
        self.assertEqual(self.verdict(scores[7:] + scores[:7])["verdict"], "PASS")        # order does not matter

    def test_without_direction_nothing_is_exempt(self):
        scores = [0.0] * 14 + [0.05, 0.111, 0.0625, 0.2, 0.0833, 0.3]
        self.assertEqual(self.verdict(scores, direction=None)["reasons"], ["degenerate_mode"])
        self.assertIn("degenerate_mode", self.verdict(scores, score_range=(None, None))["reasons"])

    def test_ocr_shaped_worst_bound_counts(self):
        # Higher is better on [0, 1]: half the scores at 0 (nothing rendered) is the worst end and still degenerate.
        scores = [0.0] * 20 + [0.25 * (i % 4) + 0.1 for i in range(20)]
        got = self.verdict(scores, score_range=(0, 1), direction="higher_is_better")
        self.assertEqual(got, {"verdict": "FAIL", "reasons": ["degenerate_mode"]})

    def test_all_on_the_best_bound_fails(self):
        self.assertEqual(self.verdict([0.0] * 20), {"verdict": "FAIL", "reasons": ["degenerate_bound", "degenerate_sd"]})
        self.assertEqual(self.verdict([1.0] * 20, score_range=(0, 1), direction="higher_is_better")["reasons"],
                         ["degenerate_mean", "degenerate_bound", "degenerate_sd"])

    def test_constant_or_mode_off_the_bound_fails(self):
        self.assertEqual(self.verdict([0.25] * 20)["reasons"], ["degenerate_mode", "degenerate_sd"])
        self.assertEqual(self.verdict([0.0] * 10 + [0.25] * 10)["reasons"], ["degenerate_mode"])

    def test_too_few_off_the_bound(self):
        four = [0.0] * 16 + [0.1, 0.2, 0.3, 0.4]                 # 4 < ceil(20 / 4) = 5
        five = [0.0] * 15 + [0.1, 0.2, 0.3, 0.4, 0.5]
        self.assertEqual(self.verdict(four)["reasons"], ["degenerate_bound"])
        self.assertEqual(self.verdict(five)["verdict"], "PASS")

    def test_only_the_best_end_is_exempt(self):
        # Higher is better on [0, 1]: the 1.0s (best) are exempt, the 0.0s (worst) count.
        best = [1.0] * 11 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.assertEqual(self.verdict(best, score_range=(0, 1), direction="higher_is_better")["verdict"], "PASS")
        worst = [0.0] * 11 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.assertIn("degenerate_mode", self.verdict(worst, score_range=(0, 1),
                                                      direction="higher_is_better")["reasons"])

    def test_share_of_the_whole_sample(self):
        # Drones-shaped (area 2's real record: 147 of 300 at the worst bound 0, 34 at the best bound 1): the exempt best
        # scores stay in the denominator, so 9 of 20 at 0 is 45 % and passes as before revision 18.
        scores = [0.0] * 9 + [1.0] * 3 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        for direction in ("higher_is_better", None):
            self.assertEqual(self.verdict(scores, score_range=(0, 1), direction=direction)["verdict"], "PASS")

    def test_binary_unchanged(self):
        self.assertEqual(self.verdict([1.0] * 11 + [0.0] * 9, score_range=(0, 1), binary=True)["verdict"], "PASS")
        self.assertEqual(self.verdict([0.0] * 20, score_range=(0, 1), binary=True)["reasons"], ["degenerate_mean"])

    def test_per_evaluator_problems(self):
        from _ext import smoke_record
        from gatedtrain.ext.smoke import smoke_problems

        rows = [("a@0.1", {"score_range": (0, 1), "binary": True, "direction": "higher_is_better"}),
                ("b@0.1", {"score_range": (0, None), "binary": False, "direction": "lower_is_better"})]
        good = {"a@0.1": [1.0] * 11 + [0.0] * 9, "b@0.1": [0.0] * 14 + [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]}
        rec = smoke_record(base_model="m", item_set="i", baseline_scores_by_evaluator=good)
        self.assertEqual(smoke_problems(rec, adapter="synthetic", score_range=None, binary=False, evaluators=rows), [])
        bad = dict(good, **{"b@0.1": [0.0] * 20})
        got = smoke_problems({**rec, "baseline_scores_by_evaluator": bad}, adapter="synthetic", score_range=None,
                             binary=False, evaluators=rows)
        self.assertEqual(got, ["the recomputed verdict of evaluator b@0.1 is FAIL: degenerate_bound, degenerate_sd"])
        got = smoke_problems({**rec, "baseline_scores": [0.0] * 20}, adapter="synthetic", score_range=None,
                             binary=False, evaluators=rows)
        self.assertEqual(len(got), 1)
        self.assertIn("baseline_scores is not baseline_scores_by_evaluator", got[0])
        self.assertEqual(len(smoke_problems({k: v for k, v in rec.items() if k != "baseline_scores_by_evaluator"},
                                            adapter="synthetic", score_range=None, binary=False, evaluators=rows)), 1)


class TestRunRefusesClauses(TtsEvaluators):
    """R18-1: only ext-collect gates several clauses; a Tier A (or Tier B) workflow with `clauses` is refused."""

    def test_tier_a_spec_refused(self):
        from _support import DEMO_DIR
        from gatedtrain.errors import Refused
        from gatedtrain.spec import load_spec

        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-r18-tiera-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        reg = tts_registration()
        reg["clauses"][0]["evaluator"] = "addition_em.cpu_stub@0.1"
        reg["clauses"][0]["direction"] = "higher_is_better"
        reg["evaluators"][0]["id"] = "addition_em.cpu_stub@0.1"
        (tmp / "registration.yaml").write_text(yaml.safe_dump(reg, sort_keys=False), encoding="utf-8")
        from _support import file_sha256

        spec_path = DEMO_DIR / "spec.yaml"
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        rname = spec["stages"][1]["inputs"]["registration"]
        spec["artifacts"][rname]["uri"] = f"file:registration.yaml@{file_sha256(tmp / 'registration.yaml')}"
        spec["stages"][1]["evaluator"] = ["addition_em.cpu_stub@0.1", SIM_ID]
        spec["stages"][1]["test"] = {}
        (tmp / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        self.assertEqual(spec["stages"][0]["uses"], "sft.cpu_stub@0.1")
        with self.assertRaises(Refused) as ctx:
            load_spec(tmp / "spec.yaml")
        hit = [p for p in ctx.exception.problems if (p.rule, p.field) == ("REG-FIELD", "registration.clauses")]
        self.assertEqual(len(hit), 1, ctx.exception.problems)
        self.assertIn("ext-collect", hit[0].message)


if __name__ == "__main__":
    unittest.main()
