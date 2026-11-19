"""Design revision 11: the gate library on per-item records (tests/README.md section 11.16).

Design (revision 12 files; revision 11 changes R11-1 to R11-21 in design/design.md):
- R11-9, 05 section 3.8 "Per-item record (package A)": records `item`, `repeat` (0 <= repeat < K), `seed`
  (required, eval_seeds[repeat]), `score` (natural units or null), `error` (class or null; exactly one of
  score and error is null), `sub_metrics` (never deciding), `output_ref`; each arm records the evaluated
  checkpoint's `model_manifest_hash` once; NeMo Evaluator records are converted into this shape.
- R11-2, 05 section 3.4 "Missingness check": expected items = registered item set x K per arm; missing =
  no record (`absent`), an evaluator error (the error class), or a score outside the declared range
  (`out_of_range`); `max_unpaired` count, or fraction of the expected items over all arms.
- R11-3, 05 section 3.6 "Minimum paired sample": a complete pair is an item with at least one repeat present
  in both arms; its difference averages the repeats present in both arms; n counts complete pairs.
- R11-7, 05 section 3.8 statuses: a repeat outside range(K) or a seed other than eval_seeds[repeat] is
  MISSING; fewer than K records are missing items, not a MISSING clause by themselves (supersedes the
  R10A-8 "maximum repeats != K" rule). R11-14, 05 section 3.3: a `model_manifest_hash` other than the
  gate's pinned checkpoint is MISSING.
- R11-10, 05 section 3.1: `direction`; lower-is-better metrics negated; margins in oriented units; Delta > 0
  favours the candidate; natural-unit difference reported beside the oriented one.
- R11-11, 05 section 6.1: the declared score range; out-of-range scores are missing items.
- R11-21, 05 sections 3.1, 3.4, 3.6: ratio-of-sums metrics; Delta = difference of the arm ratios;
  item-clustered bootstrap; power from linearized per-item contributions (delta method).
- R11-5, 05 section 3.8 "p-values": by test-interval duality; none for a zero-width interval or no pairs.
- R11-1, 05 section 3.7: post_hoc only from results of the judged candidate checkpoint; a candidate
  raw-results artifact created before the amended registration counts as observed.
- R11-8, 05 section 3.6: |theta| >= delta has no finite TOST n; `plan()` refuses it.

Interface (proposal, section 11.16): `gates.evaluate_items(clause, baseline, candidate, *, items,
seed, n_boot=10000, min_paired=10, max_unpaired=0, score_range=(None, None), expected_models=None,
plan=None)`, arms `{"records": [...], "model_manifest_hash": str | None}`; `gates.is_post_hoc(...)`;
`items.from_nemo_evaluator(record, *, scorer_key=None)`.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import unittest

import numpy as np
from scipy.stats import norm

from _r11 import (LONGRUN, LONGRUN_ITEMS, RECORD_FIELDS, alternating, arm, clause, diff_arms, err, evaluate_items,
                  ids, load_longrun_arm, longrun_registration, longrun_report, matrix_arms, rec)

Z_975 = norm.ppf(0.975)


# ---------------------------------------------------------------------------
# R11A-5: the per-item record and evaluate_items
# ---------------------------------------------------------------------------

class TestRecordShape(unittest.TestCase):
    """05 section 3.8: `evaluate` reads per-item records; pairing is by (item, repeat), never by position."""

    def strong(self):
        return diff_arms(alternating(40, 0.5, 0.1))

    def test_result_keys_use_item_names(self):
        items, b, c = self.strong()
        res = evaluate_items(clause(), b, c, items=items, seed=5, n_boot=1000)
        for key in ("n_items", "missing_baseline", "missing_candidate", "missing_reasons", "direction",
                    "delta_hat_natural", "ci_lower_natural", "ci_upper_natural", "status", "status_reasons",
                    "p_value", "p_value_at_floor", "achieved_power", "k_actual"):
            self.assertIn(key, res)
        self.assertNotIn("n_prompts", res, "new writes use item names (R11-19)")
        self.assertEqual(res["n_items"], 40)
        self.assertEqual(res["status"], "PASS", res)
        self.assertEqual(res["missing_reasons"], {"baseline": {}, "candidate": {}})

    def test_item_ids_are_opaque_and_order_free(self):
        """String IDs, integer IDs, and shuffled record order give the same statistics (mutant: pairing by
        line order instead of by item)."""
        diffs = alternating(40, 0.5, 0.1)
        items_s, b_s, c_s = diff_arms(diffs)
        ref = evaluate_items(clause(), b_s, c_s, items=items_s, seed=5, n_boot=1000)
        items_i, b_i, c_i = matrix_arms([[0.2] for _ in diffs], [[0.2 + d] for d in diffs], [11],
                                        item_ids=list(range(100, 140)))
        rng = random.Random(3)
        rng.shuffle(c_i["records"])
        res = evaluate_items(clause(), b_i, c_i, items=items_i, seed=5, n_boot=1000)
        for key in ("delta_hat", "ci_lower", "ci_upper", "p_value", "achieved_power", "n_items", "status"):
            self.assertEqual(res[key], ref[key], key)

    def test_sub_metrics_and_output_ref_never_decide(self):
        items, b, c = self.strong()
        ref = evaluate_items(clause(), b, c, items=items, seed=5, n_boot=1000)
        for r in c["records"]:
            r["sub_metrics"] = {"aesthetic": -1e6, "violated": ["c1"]}
            r["output_ref"] = {"path": "x.png", "sha256": "sha256:" + "0" * 64}
        res = evaluate_items(clause(), b, c, items=items, seed=5, n_boot=1000)
        for key in ("delta_hat", "ci_lower", "ci_upper", "status", "status_reasons"):
            self.assertEqual(res[key], ref[key], key)


# ---------------------------------------------------------------------------
# R11A-6: missing items, expected set from the registered item set, reasons
# ---------------------------------------------------------------------------

class TestMissingItems(unittest.TestCase):
    """Expected items = `items` x range(K) per arm (05 section 3.4), not the union of the arms (R10A-3)."""

    def arms(self):
        return diff_arms(alternating(40, 0.5, 0.1))

    def replace(self, a, item, new):
        a["records"] = [r for r in a["records"] if r["item"] != item] + [new]

    def test_item_absent_from_both_arms_is_missing_in_each(self):
        """u999 is in the registered set and in neither arm: one missing item per arm (K = 1). Revision 10
        took E from the arms' union and saw nothing missing."""
        items, b, c = self.arms()
        items = items + ["u999"]
        res = evaluate_items(clause(), b, c, items=items, seed=5, n_boot=1000)
        self.assertEqual((res["missing_baseline"], res["missing_candidate"]), (1, 1))
        self.assertEqual(res["missing_reasons"], {"baseline": {"absent": 1}, "candidate": {"absent": 1}})
        self.assertEqual(res["status"], "MISSING", res)
        self.assertIn("missing_above_bound", res["status_reasons"])
        self.assertEqual(res["n_items"], 40, "complete pairs are still analysed")

    def test_count_bound_over_all_arms(self):
        items, b, c = self.arms()
        items = items + ["u999"]
        self.assertEqual(evaluate_items(clause(), b, c, items=items, seed=5, n_boot=1000, max_unpaired=2)["status"],
                         "PASS")
        self.assertEqual(evaluate_items(clause(), b, c, items=items, seed=5, n_boot=1000, max_unpaired=1)["status"],
                         "MISSING")

    def test_fraction_of_expected_items_over_all_arms(self):
        """2 missing of 2 x 41 x 1 = 82 expected items = 2.44%."""
        items, b, c = self.arms()
        items = items + ["u999"]
        self.assertEqual(evaluate_items(clause(), b, c, items=items, seed=5, n_boot=1000,
                                        max_unpaired=0.03)["status"], "PASS")
        self.assertEqual(evaluate_items(clause(), b, c, items=items, seed=5, n_boot=1000,
                                        max_unpaired=0.02)["status"], "MISSING")

    def test_error_record_is_missing_with_its_class(self):
        items, b, c = self.arms()
        self.replace(c, "u003", err("u003", 0, 11, "timeout"))
        res = evaluate_items(clause(), b, c, items=items, seed=5, n_boot=1000)
        self.assertEqual(res["missing_reasons"]["candidate"], {"timeout": 1})
        self.assertEqual(res["missing_candidate"], 1)
        self.assertEqual(res["n_items"], 39, "an error is never scored (never 0)")
        self.assertEqual(res["status"], "MISSING")
        ok = evaluate_items(clause(), b, c, items=items, seed=5, n_boot=1000, max_unpaired=1)
        self.assertEqual(ok["status"], "PASS", ok)

    def test_out_of_range_score_is_missing(self):
        items, b, c = self.arms()
        self.replace(c, "u003", rec("u003", 0, 1.5, 11))
        res = evaluate_items(clause(), b, c, items=items, seed=5, n_boot=1000, score_range=(0.0, 1.0))
        self.assertEqual(res["missing_reasons"]["candidate"], {"out_of_range": 1})
        self.assertEqual(res["n_items"], 39)
        self.assertEqual(res["status"], "MISSING")

    def test_open_upper_bound(self):
        items, b, c = self.arms()
        self.replace(c, "u003", rec("u003", 0, 1.5, 11))
        res = evaluate_items(clause(), b, c, items=items, seed=5, n_boot=1000, score_range=(0.0, None))
        self.assertEqual(res["missing_reasons"]["candidate"], {})
        self.assertEqual(res["n_items"], 40)

    def test_non_finite_score_is_out_of_range(self):
        for bad in (float("nan"), float("inf")):
            with self.subTest(score=bad):
                items, b, c = self.arms()
                self.replace(c, "u003", rec("u003", 0, bad, 11))
                res = evaluate_items(clause(), b, c, items=items, seed=5, n_boot=1000)
                self.assertEqual(res["missing_reasons"]["candidate"], {"out_of_range": 1})

    def test_record_without_seed_is_missing(self):
        """`seed` is required (05 section 3.8); a record without one is a missing item, reason `no_seed`
        (design choice, section 11.16). Revision 10 did not check records without a seed."""
        for form in ("null", "absent"):
            with self.subTest(form=form):
                items, b, c = self.arms()
                r = rec("u003", 0, 0.7, None)
                if form == "absent":
                    r.pop("seed")
                self.replace(c, "u003", r)
                res = evaluate_items(clause(), b, c, items=items, seed=5, n_boot=1000)
                self.assertEqual(res["missing_reasons"]["candidate"], {"no_seed": 1})
                self.assertEqual(res["n_items"], 39)
                self.assertEqual(res["status"], "MISSING")


# ---------------------------------------------------------------------------
# R11A-7: clause-level MISSING from records; fewer than K is missing items only
# ---------------------------------------------------------------------------

K4 = {"samples_per_item": 4, "eval_seeds": [11, 12, 13, 14]}


def strong_k(n, k, cand_k=None):
    """Baseline 0 everywhere, candidate 1 on items i % 4 != 0: a large effect."""
    cand_k = k if cand_k is None else cand_k
    base = [[0.0] * k for _ in range(n)]
    cand = [[1.0 if i % 4 else 0.0] * cand_k for i in range(n)]
    return base, cand


class TestClauseMissing(unittest.TestCase):

    def check_missing(self, res, reason):
        self.assertEqual(res["status"], "MISSING", res)
        self.assertIn(reason, res["status_reasons"], res)

    def test_repeat_outside_range_k(self):
        items, b, c = diff_arms(alternating(40, 0.5, 0.1))
        b["records"].append(rec("u000", 1, 0.2, 12))
        c["records"].append(rec("u000", 1, 0.7, 12))
        self.check_missing(evaluate_items(clause(), b, c, items=items, seed=5, n_boot=500), "k_mismatch")

    def test_seed_other_than_registered(self):
        items, b, c = diff_arms(alternating(40, 0.5, 0.1))
        c["records"][0]["seed"] = 12
        self.check_missing(evaluate_items(clause(), b, c, items=items, seed=5, n_boot=500), "seed_mismatch")

    def test_fewer_than_k_in_every_item_is_missing_items_only(self):
        """K = 4 registered, both arms hold 3 repeats per item. Revision 11: 400 missing items per arm,
        below max_unpaired 0.99, so the clause is analysed on repeats 0-2 and PASSes. Revision 10
        (R10A-8) made it MISSING with `k_mismatch`."""
        base, cand = strong_k(400, 3)
        items, b, c = matrix_arms(base, cand, [11, 12, 13])
        res = evaluate_items(clause(**K4), b, c, items=items, seed=5, n_boot=1000, max_unpaired=0.99)
        self.assertNotIn("k_mismatch", res["status_reasons"], res)
        self.assertEqual((res["missing_baseline"], res["missing_candidate"]), (400, 400))
        self.assertEqual(res["missing_reasons"]["baseline"], {"absent": 400})
        self.assertEqual(res["n_items"], 400)
        self.assertEqual(res["status"], "PASS", res)

    def test_candidate_k1_against_k4(self):
        base, _ = strong_k(400, 4)
        _, cand = strong_k(400, 1)
        items, b, c = matrix_arms(base, cand, [11, 12, 13, 14])
        res = evaluate_items(clause(**K4), b, c, items=items, seed=5, n_boot=1000, max_unpaired=0.99)
        self.assertNotIn("k_mismatch", res["status_reasons"], res)
        self.assertEqual((res["missing_baseline"], res["missing_candidate"]), (0, 1200))
        self.assertEqual(res["status"], "PASS", res)

    def test_fewer_than_k_at_bound_zero_is_missing_above_bound(self):
        base, cand = strong_k(40, 3)
        items, b, c = matrix_arms(base, cand, [11, 12, 13])
        res = evaluate_items(clause(**K4), b, c, items=items, seed=5, n_boot=500)
        self.check_missing(res, "missing_above_bound")
        self.assertNotIn("k_mismatch", res["status_reasons"])

    def test_malformed_records(self):
        """Exactly one of score and error is null; no unknown keys; one record per (item, repeat). A
        violation makes the clause MISSING with reason `malformed_record` (design choice)."""
        cases = {
            "score_and_error": lambda r: r.update(error={"class": "timeout", "message": "x"}),
            "neither": lambda r: r.update(score=None),
            "unknown_key": lambda r: r.update(prompt_text="leaked"),
            "bool_score": lambda r: r.update(score=True),
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                items, b, c = diff_arms(alternating(40, 0.5, 0.1))
                mutate(c["records"][5])
                self.check_missing(evaluate_items(clause(), b, c, items=items, seed=5, n_boot=500),
                                   "malformed_record")
        with self.subTest(case="duplicate"):
            items, b, c = diff_arms(alternating(40, 0.5, 0.1))
            c["records"].append(dict(c["records"][5]))
            self.check_missing(evaluate_items(clause(), b, c, items=items, seed=5, n_boot=500), "malformed_record")

    def test_item_outside_the_registered_set(self):
        items, b, c = diff_arms(alternating(40, 0.5, 0.1))
        c["records"].append(rec("zzz", 0, 0.5, 11))
        self.check_missing(evaluate_items(clause(), b, c, items=items, seed=5, n_boot=500), "unknown_item")

    def test_model_manifest_hash_must_match_the_pinned_checkpoint(self):
        """05 sections 3.3, 3.8 (R11-14): an evaluated `model_manifest_hash` other than the gate's pinned
        checkpoint is MISSING (`model_hash_mismatch`); a null arm hash cannot be matched."""
        hb, hc = "sha256:" + "b" * 64, "sha256:" + "c" * 64
        items, b, c = diff_arms(alternating(40, 0.5, 0.1))
        b["model_manifest_hash"], c["model_manifest_hash"] = hb, hc
        ok = evaluate_items(clause(), b, c, items=items, seed=5, n_boot=500,
                            expected_models={"baseline": hb, "candidate": hc})
        self.assertNotIn("model_hash_mismatch", ok["status_reasons"])
        self.assertEqual(ok["status"], "PASS")
        bad = evaluate_items(clause(), b, c, items=items, seed=5, n_boot=500,
                             expected_models={"baseline": hb, "candidate": "sha256:" + "d" * 64})
        self.check_missing(bad, "model_hash_mismatch")
        c["model_manifest_hash"] = None
        self.check_missing(evaluate_items(clause(), b, c, items=items, seed=5, n_boot=500,
                                          expected_models={"baseline": hb, "candidate": hc}),
                           "model_hash_mismatch")

    def test_no_expected_models_means_no_check(self):
        items, b, c = diff_arms(alternating(40, 0.5, 0.1))
        res = evaluate_items(clause(), b, c, items=items, seed=5, n_boot=500)
        self.assertNotIn("model_hash_mismatch", res["status_reasons"])


# ---------------------------------------------------------------------------
# R11A-8: complete pairs (C1)
# ---------------------------------------------------------------------------

class TestCompletePairs(unittest.TestCase):
    """05 section 3.6: a complete pair has at least one repeat in both arms; its difference is the mean
    over the repeats present in both arms; n counts complete pairs."""

    K3 = {"samples_per_item": 3, "eval_seeds": [11, 12, 13]}

    def data(self, n=12):
        base = [[0.1 * (i % 3), 0.2, 0.3] for i in range(n)]
        cand = [[0.5, 0.6 + 0.01 * i, 0.9] for i in range(n)]
        return base, cand

    @staticmethod
    def expected_delta(base, cand):
        diffs = []
        for b, c in zip(base, cand):
            common = [r for r in range(3) if b[r] is not None and c[r] is not None]
            if common:
                diffs.append(sum(c[r] - b[r] for r in common) / len(common))
        return sum(diffs) / len(diffs), len(diffs)

    def test_partial_repeats_use_the_common_repeats(self):
        """Item 0: baseline lacks repeat 2, candidate lacks repeat 0; its difference is c1 - b1 = 0.4. The
        mutant that averages every present repeat per arm gives 0.65 for item 0."""
        base, cand = self.data()
        base[0][2] = None
        cand[0][0] = None
        items, b, c = matrix_arms(base, cand, [11, 12, 13])
        res = evaluate_items(clause(**self.K3), b, c, items=items, seed=5, n_boot=500, max_unpaired=2,
                             min_paired=1)
        expected, n = self.expected_delta(base, cand)
        self.assertEqual(n, 12)
        self.assertEqual(res["n_items"], 12)
        self.assertAlmostEqual(res["delta_hat"], expected, places=12)

    def test_no_common_repeat_is_not_a_pair(self):
        base, cand = self.data()
        base[0] = [0.0, None, None]
        cand[0] = [None, 0.9, None]
        items, b, c = matrix_arms(base, cand, [11, 12, 13])
        res = evaluate_items(clause(**self.K3), b, c, items=items, seed=5, n_boot=500, max_unpaired=0.5,
                             min_paired=1)
        expected, n = self.expected_delta(base, cand)
        self.assertEqual(res["n_items"], 11)
        self.assertAlmostEqual(res["delta_hat"], expected, places=12)
        self.assertEqual((res["missing_baseline"], res["missing_candidate"]), (2, 2))

    def test_min_paired_counts_complete_pairs(self):
        """10 items, one without a common repeat: n = 9 < 10, INCONCLUSIVE (`min_paired`)."""
        base, cand = strong_k(10, 2)
        base[0] = [0.0, None]
        cand[0] = [None, 1.0]
        items, b, c = matrix_arms(base, cand, [11, 12])
        res = evaluate_items(clause(samples_per_item=2, eval_seeds=[11, 12]), b, c, items=items, seed=5,
                             n_boot=500, max_unpaired=0.5)
        self.assertEqual(res["n_items"], 9)
        self.assertEqual(res["status"], "INCONCLUSIVE", res)
        self.assertIn("min_paired", res["status_reasons"])


# ---------------------------------------------------------------------------
# R11A-9: direction and natural units
# ---------------------------------------------------------------------------

def lower_better_arms(n, natural_shift, amp=0.05, base=0.4):
    """WER-like scores: baseline `base`, candidate base + natural_shift +/- amp (lower is better)."""
    return matrix_arms([[base] for _ in range(n)], [[base + d] for d in alternating(n, natural_shift, amp)], [11])


class TestDirection(unittest.TestCase):

    def test_lower_is_better_is_negated(self):
        """Candidate WER 0.1 lower: oriented Delta-hat = +0.1, PASS; natural difference -0.1."""
        items, b, c = lower_better_arms(40, -0.1)
        res = evaluate_items(clause(direction="lower_is_better"), b, c, items=items, seed=5, n_boot=1000)
        self.assertEqual(res["direction"], "lower_is_better")
        self.assertAlmostEqual(res["delta_hat"], 0.1, places=12)
        self.assertEqual(res["status"], "PASS", res)
        self.assertAlmostEqual(res["delta_hat_natural"], -0.1, places=12)
        self.assertAlmostEqual(res["ci_lower_natural"], -res["ci_upper"], places=12)
        self.assertAlmostEqual(res["ci_upper_natural"], -res["ci_lower"], places=12)
        self.assertAlmostEqual(res["baseline_score"], 0.4, places=12, msg="scores stay in natural units")
        self.assertAlmostEqual(res["candidate_score"], 0.3, places=12)

    def test_without_negation_the_same_data_fail(self):
        """Mutant `no negation for lower-is-better` (MULTIMODAL-SPEC section 9): reading the WER data as
        higher-is-better gives Delta-hat = -0.1 and FAIL."""
        items, b, c = lower_better_arms(40, -0.1)
        res = evaluate_items(clause(direction="higher_is_better"), b, c, items=items, seed=5, n_boot=1000)
        self.assertAlmostEqual(res["delta_hat"], -0.1, places=12)
        self.assertEqual(res["status"], "FAIL", res)
        self.assertAlmostEqual(res["delta_hat_natural"], res["delta_hat"], places=12)

    def test_margin_is_in_oriented_units(self):
        """Non-inferiority, delta = 0.10, candidate WER 0.15 higher (worse): oriented Delta = -0.15 < -delta,
        powered FAIL. In natural units (+0.15) it would pass."""
        items, b, c = lower_better_arms(200, 0.15)
        res = evaluate_items(clause("non_inferiority", direction="lower_is_better"), b, c, items=items, seed=5,
                             n_boot=1000)
        self.assertAlmostEqual(res["delta_hat"], -0.15, places=12)
        self.assertEqual(res["status"], "FAIL", res)

    def test_report_shows_both_units(self):
        """Report (05 section 3.1): oriented Delta-hat and interval, and beside them the labelled natural-unit
        difference; `direction`; item names in JSON and text."""
        from gatedtrain.report import build_report, render_text

        c_ = clause(direction="lower_is_better")
        items, b, c = lower_better_arms(40, -0.1)
        res = evaluate_items(c_, b, c, items=items, seed=5, n_boot=1000)
        h = "sha256:" + "0" * 64
        hashes = {k: h for k in ("registration_hash", "item_set_hash", "evaluator_hash", "evaluator_hash_registered",
                                 "sampler_hash", "baseline_checkpoint_hash", "candidate_checkpoint_hash",
                                 "lockfile_hash")}
        plan_ = {"n_required": 10, "mde": 0.1, "design_effect": 1.0, "sigma_plan": "fixed", "sigma_used": 0.3,
                 "n_available": 40, "underpowered": False}
        dist = {"item_set_hash": h, "env_version": None, "env_params_version": None}
        rep = build_report(workflow="w", gate_id="g", clause=c_, result=res, plan=plan_,
                           registration={"eval_distribution": dist}, hashes=hashes, stubs=[],
                           created_at="2026-11-11T00:00:00Z", run_id="r", disjoint_declared=False)
        self.assertEqual(rep["direction"], "lower_is_better")
        self.assertEqual(rep["delta_hat"], res["delta_hat"])
        self.assertEqual(rep["natural_units"], {"delta_hat": res["delta_hat_natural"],
                                                "lower": res["ci_lower_natural"], "upper": res["ci_upper_natural"]})
        self.assertEqual(rep["n_items"], 40)
        self.assertEqual(rep["samples_per_item"], 1)
        self.assertEqual(rep["eval_distribution"], dist)
        self.assertEqual(rep["interval"]["bootstrap"]["resampling_unit"], "item")
        self.assertEqual(rep["missing_reasons"], {"baseline": {}, "candidate": {}})
        for old in ("n_prompts", "samples_per_prompt", "prompt_set_hash"):
            self.assertNotIn(old, rep)
        lines = render_text(rep).splitlines()
        self.assertIn("direction: lower_is_better", lines)
        self.assertIn("n_items: 40", lines)
        self.assertIn(f"item_set_hash: {h}", lines)
        self.assertTrue(any(ln.startswith("natural_units: ") for ln in lines), lines)
        self.assertFalse(any(ln.startswith(("n_prompts:", "prompt_set_hash:")) for ln in lines), lines)


# ---------------------------------------------------------------------------
# R11A-12: ratio-of-sums metrics (C7)
# ---------------------------------------------------------------------------

# 60 utterances: words per utterance (denominator) and word errors (numerator) per arm. Baseline corpus WER
# 0.300, candidate 0.219: oriented (lower is better) Delta = 0.0810, while the mean of per-item WER differences
# is 0.0741 (a mean-of-ratios mutant is off by about half an SE).
DEN = [3, 10, 17, 24, 8, 15, 22, 6, 13, 20, 4, 11, 18, 25, 9, 16, 23, 7, 14, 21, 5, 12, 19, 3, 10, 17, 24, 8, 15,
       22, 6, 13, 20, 4, 11, 18, 25, 9, 16, 23, 7, 14, 21, 5, 12, 19, 3, 10, 17, 24, 8, 15, 22, 6, 13, 20, 4, 11, 18,
       25]
NUM_B = [0, 2, 5, 8, 4, 3, 6, 2, 4, 7, 0, 3, 5, 8, 4, 4, 6, 2, 5, 8, 0, 3, 6, 2, 4, 4, 7, 2, 5, 8, 1, 3, 6, 2, 4, 4,
         7, 3, 5, 8, 1, 4, 6, 2, 5, 4, 0, 3, 6, 8, 1, 4, 7, 2, 5, 5, 1, 3, 6, 9]
NUM_C = [0, 1, 3, 5, 2, 4, 6, 0, 2, 4, 1, 3, 5, 7, 0, 3, 5, 2, 4, 6, 3, 1, 3, 0, 2, 4, 6, 3, 2, 4, 1, 3, 5, 2, 4, 2,
         4, 1, 4, 6, 3, 5, 3, 0, 2, 4, 1, 3, 5, 4, 1, 3, 5, 2, 4, 6, 0, 1, 3, 6]


def ratio_clause(**over):
    c = clause(metric="wer", direction="lower_is_better", estimator="ratio_of_sums", planning_effect=0.03)
    c.update(over)
    return c


def ratio_arms(k=1, skip=frozenset()):
    """K identical repeats per utterance; `skip` = item indices whose candidate record is an error."""
    seeds = [11, 12, 13, 14][:k]
    item_ids = ids(len(DEN))

    def build(num, errors=frozenset()):
        out = []
        for i, (n_, d_) in enumerate(zip(num, DEN)):
            for r in range(k):
                if i in errors:
                    out.append(err(item_ids[i], r, seeds[r], "decode_crash"))
                else:
                    out.append(rec(item_ids[i], r, n_ / d_, seeds[r], numerator=n_, denominator=d_))
        return out

    return item_ids, arm(build(NUM_B)), arm(build(NUM_C, skip)), seeds


def ratio_reference(keep=None):
    """Independent numbers: arm ratios, oriented Delta, linearized contributions (05 section 3.6)."""
    keep = range(len(DEN)) if keep is None else keep
    den = np.array([DEN[i] for i in keep], float)
    nb = np.array([NUM_B[i] for i in keep], float)
    nc = np.array([NUM_C[i] for i in keep], float)
    rb, rc = nb.sum() / den.sum(), nc.sum() / den.sum()
    db = (nb - rb * den) / den.mean()
    dc = (nc - rc * den) / den.mean()
    contrib = -(dc - db)                       # oriented: lower is better
    sd = float(contrib.std(ddof=1))
    se = sd / math.sqrt(len(den))
    return {"rb": rb, "rc": rc, "delta": rb - rc, "sd": sd, "se": se,
            "mean_of_ratios": float(np.mean(nb / den) - np.mean(nc / den))}


class TestRatioOfSums(unittest.TestCase):
    """05 section 3.1 "Ratio metrics", section 3.4 table (ratio row), section 3.6 (delta method). Clause field
    `estimator: ratio_of_sums` (design choice; absent means the per-item mean)."""

    def test_reference_numbers(self):
        ref = ratio_reference()
        self.assertAlmostEqual(ref["rb"], 0.3, places=12)
        self.assertAlmostEqual(ref["delta"], 0.08095238095238094, places=12)
        self.assertGreater(abs(ref["delta"] - ref["mean_of_ratios"]), 0.005)

    def test_point_estimate_is_the_difference_of_arm_ratios(self):
        items, b, c, seeds = ratio_arms()
        res = evaluate_items(ratio_clause(), b, c, items=items, seed=5, n_boot=2000)
        ref = ratio_reference()
        self.assertAlmostEqual(res["delta_hat"], ref["delta"], places=12)
        self.assertAlmostEqual(res["delta_hat_natural"], ref["rc"] - ref["rb"], places=12)
        self.assertEqual(res["n_items"], 60)

    def test_power_from_linearized_contributions(self):
        """sd of the paired per-item contributions d_i = (num_i - R den_i) / mean(den), oriented, ddof = 1;
        power at theta = 0.03: Phi(0.03 / SE - z_{0.975}) = 0.649, below 0.8 (underpowered)."""
        items, b, c, _ = ratio_arms()
        res = evaluate_items(ratio_clause(), b, c, items=items, seed=5, n_boot=2000)
        ref = ratio_reference()
        self.assertAlmostEqual(res["sd_diff"], ref["sd"], places=9)
        self.assertAlmostEqual(res["achieved_power"], float(norm.cdf(0.03 / ref["se"] - Z_975)), places=6)
        self.assertAlmostEqual(res["achieved_power"], 0.6488142911468089, places=6)
        self.assertIs(res["underpowered"], True)
        self.assertEqual(res["status"], "PASS", "an underpowered pass is PASS, flagged (05 section 3.1)")

    def test_interval_resamples_items(self):
        """K = 4 identical repeats per utterance: the arm ratios, Delta-hat and power are those of K = 1, and
        the interval width is about 2 z SE of the item-level delta method (a record-level bootstrap would
        give about half)."""
        items, b, c, _ = ratio_arms(k=4)
        res = evaluate_items(ratio_clause(**{"samples_per_item": 4, "eval_seeds": [11, 12, 13, 14]}), b, c,
                             items=items, seed=5, n_boot=4000)
        ref = ratio_reference()
        self.assertAlmostEqual(res["delta_hat"], ref["delta"], places=12)
        self.assertAlmostEqual(res["sd_diff"], ref["sd"], places=9)
        width = res["ci_upper"] - res["ci_lower"]
        self.assertGreater(width, 0.75 * 2 * Z_975 * ref["se"], res)
        self.assertLess(width, 1.35 * 2 * Z_975 * ref["se"], res)
        self.assertLess(res["ci_lower"], res["delta_hat"])
        self.assertGreater(res["ci_upper"], res["delta_hat"])

    def test_p_value_by_duality(self):
        """05 section 3.8: the p-value comes from the same bootstrap; the lower bound is well above 0
        (about 0.056), so p < alpha."""
        items, b, c, _ = ratio_arms()
        res = evaluate_items(ratio_clause(), b, c, items=items, seed=5, n_boot=4000)
        self.assertGreater(res["ci_lower"], 0.03)
        self.assertIsNotNone(res["p_value"])
        self.assertLess(res["p_value"], 0.025)
        nul = evaluate_items(ratio_clause(margin=0.2), b, c, items=items, seed=5, n_boot=4000)
        self.assertLess(nul["ci_lower"], 0.2)
        self.assertGreater(nul["p_value"], 0.5, "null far above Delta-hat: p near 1")

    def test_missing_pair_leaves_both_sums(self):
        """An evaluator error on candidate utterance 3 (max_unpaired 1): utterance 3 leaves both arms' sums."""
        items, b, c, _ = ratio_arms(skip={3})
        res = evaluate_items(ratio_clause(), b, c, items=items, seed=5, n_boot=2000, max_unpaired=1)
        ref = ratio_reference(keep=[i for i in range(60) if i != 3])
        self.assertEqual(res["n_items"], 59)
        self.assertEqual(res["missing_reasons"]["candidate"], {"decode_crash": 1})
        self.assertAlmostEqual(res["delta_hat"], ref["delta"], places=12)

    def test_records_need_numerator_and_denominator(self):
        items, b, c, _ = ratio_arms()
        del c["records"][7]["numerator"]
        res = evaluate_items(ratio_clause(), b, c, items=items, seed=5, n_boot=500)
        self.assertEqual(res["status"], "MISSING", res)
        self.assertIn("malformed_record", res["status_reasons"])


# ---------------------------------------------------------------------------
# R11A-11: p-values by duality on the per-item path
# ---------------------------------------------------------------------------

class TestPValuesItems(unittest.TestCase):

    def test_no_pairs_gives_no_p_value(self):
        items = ids(20)
        res = evaluate_items(clause(), arm([]), arm([]), items=items, seed=5, n_boot=500)
        self.assertEqual(res["status"], "MISSING")
        self.assertIn("no_pairs", res["status_reasons"])
        self.assertIsNone(res["p_value"])
        self.assertIs(res["p_value_at_floor"], False)
        self.assertEqual((res["missing_baseline"], res["missing_candidate"]), (20, 20))

    def test_zero_width_gives_no_p_value(self):
        items, b, c = diff_arms([0.2] * 30)
        res = evaluate_items(clause(), b, c, items=items, seed=5, n_boot=500)
        self.assertIsNone(res["p_value"])
        self.assertIn("zero_width_interval", res["status_reasons"])


# ---------------------------------------------------------------------------
# R11A-16: post_hoc scoped to the judged candidate checkpoint, auditable "observed"
# ---------------------------------------------------------------------------

class TestPostHocRule(unittest.TestCase):
    """05 section 3.7 (R11-1): an amendment is post hoc only when a result of the candidate checkpoint the gate
    judges was observed before it; a candidate-arm raw-results artifact of that checkpoint created before the
    amended registration counts as observed. Results of other checkpoints (the Tier B smoke checkpoint,
    other loop candidates) do not. Interface: `gates.is_post_hoc(*, amends, amended_at, judged_checkpoint,
    observed)`, `observed` = [{checkpoint, arm, created_at}] (RFC 3339)."""

    JUDGED = "sha256:" + "1" * 64
    OTHER = "sha256:" + "2" * 64
    AMENDED_AT = "2026-11-11T09:00:00Z"

    def call(self, observed, amends="sha256:" + "9a" * 32, amended_at=AMENDED_AT):
        from gatedtrain.gates import is_post_hoc

        return is_post_hoc(amends=amends, amended_at=amended_at, judged_checkpoint=self.JUDGED, observed=observed)

    def obs(self, checkpoint, created_at, arm_="candidate"):
        return {"checkpoint": checkpoint, "arm": arm_, "created_at": created_at}

    def test_result_of_judged_checkpoint_before_amendment(self):
        self.assertIs(self.call([self.obs(self.JUDGED, "2026-11-11T08:00:00Z")]), True)

    def test_result_after_amendment(self):
        self.assertIs(self.call([self.obs(self.JUDGED, "2026-11-11T10:00:00Z")]), False)

    def test_other_checkpoint_before_amendment(self):
        """The Tier B case (maintainers' decision of 2026-11-11): smoke results of another checkpoint do not make the
        long-run gate post hoc."""
        self.assertIs(self.call([self.obs(self.OTHER, "2026-11-10T08:00:00Z")]), False)

    def test_baseline_arm_result_does_not_count(self):
        self.assertIs(self.call([self.obs(self.JUDGED, "2026-11-11T08:00:00Z", arm_="baseline")]), False)

    def test_not_an_amendment(self):
        self.assertIs(self.call([self.obs(self.JUDGED, "2026-11-11T08:00:00Z")], amends=None), False)

    def test_times_compare_as_instants(self):
        """10:00+02:00 is 08:00Z, before 09:00Z (a string comparison would say after)."""
        self.assertIs(self.call([self.obs(self.JUDGED, "2026-11-11T10:00:00+02:00")]), True)

    def test_fail_stays_fail(self):
        from gatedtrain.gates import apply_post_hoc

        items, b, c = diff_arms(alternating(400, -0.2, 0.3))
        res = evaluate_items(clause(), b, c, items=items, seed=5, n_boot=1000)
        self.assertEqual(res["status"], "FAIL")
        self.assertEqual(apply_post_hoc(dict(res))["status"], "FAIL")


# ---------------------------------------------------------------------------
# R11A-15: TOST planning effect |theta| >= delta (regression)
# ---------------------------------------------------------------------------

class TestTostPlanRefusal(unittest.TestCase):

    def test_plan_refuses(self):
        from gatedtrain.gates import plan

        for theta in (0.10, -0.10, 0.12):
            with self.subTest(theta=theta):
                with self.assertRaises(ValueError):
                    plan(clause("equivalence", planning_effect=theta), {"sigma": 0.2, "source": "t"},
                         n_available=100)


# ---------------------------------------------------------------------------
# R11A-5: NeMo Evaluator records converted into the per-item shape
# ---------------------------------------------------------------------------

class TestNemoEvaluatorConversion(unittest.TestCase):
    """05 section 3.8 "NeMo Evaluator records are converted into this shape". Input: the first two records of
    each arm of the real long run, verbatim (fixtures/r11/longrun/raw_samples.json)."""

    KEY = "scorer:numeric_match"

    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads((LONGRUN / "raw_samples.json").read_text(encoding="utf-8"))

    def convert(self, record, **kw):
        from gatedtrain.items import from_nemo_evaluator

        return from_nemo_evaluator(record, **kw)

    def test_scorer_record(self):
        raw = self.raw["baseline"][0]
        out = self.convert(raw, scorer_key=self.KEY)
        self.assertEqual(set(out), set(RECORD_FIELDS))
        self.assertEqual(out["item"], raw["problem_idx"])
        self.assertEqual(out["repeat"], raw["repeat"])
        self.assertEqual(out["seed"], raw["metadata"]["eval_seed"])
        self.assertEqual(out["score"], raw["scorer_rewards"][self.KEY])
        self.assertIsNone(out["error"])
        digest = hashlib.sha256(raw["model_response"].encode("utf-8")).hexdigest()
        self.assertIn(digest, json.dumps(out["output_ref"]), "output_ref names the generated text by hash")

    def test_default_reward(self):
        raw = self.raw["baseline"][0]
        out = self.convert(raw)
        self.assertEqual(out["score"], raw["reward"])
        self.assertNotEqual(raw["reward"], raw["scorer_rewards"][self.KEY], "fixture guard: the two differ")

    def test_record_without_seed_converts_to_null_seed(self):
        raw = json.loads(json.dumps(self.raw["candidate"][0]))
        raw["metadata"].pop("eval_seed")
        self.assertIsNone(self.convert(raw, scorer_key=self.KEY)["seed"])

    def test_scorer_error_is_an_error_not_a_zero(self):
        """A raising scorer leaves scoring_details[key] = {error, reward: 0.0} (eval_loop.py:719-721 at ea35ba3):
        converted to score null and an error, never a 0."""
        raw = json.loads(json.dumps(self.raw["candidate"][1]))
        raw["scoring_details"][self.KEY] = {"error": "ValueError: boom", "reward": 0.0}
        raw["scorer_rewards"][self.KEY] = 0.0
        out = self.convert(raw, scorer_key=self.KEY)
        self.assertIsNone(out["score"])
        self.assertIsNotNone(out["error"])
        self.assertIsInstance(out["error"]["class"], str)


# ---------------------------------------------------------------------------
# R11A-21: the real long run through the per-item path
# ---------------------------------------------------------------------------

class TestLongRunRegression(unittest.TestCase):
    """The real long run (report in fixtures/r11/longrun/report) re-evaluated from its per-item records with the
    normalized legacy registration gives the reported statistics exactly (same bootstrap seed and B)."""

    @classmethod
    def setUpClass(cls):
        cls.report = longrun_report()
        cls.base = load_longrun_arm("baseline")
        cls.cand = load_longrun_arm("candidate")

    def test_fixture_guard(self):
        self.assertEqual(len(self.base), 4 * LONGRUN_ITEMS)
        self.assertEqual(len(self.cand), 4 * LONGRUN_ITEMS)
        self.assertEqual(self.report["n_prompts"], LONGRUN_ITEMS, "the old report keeps its old keys")

    def test_reproduces_the_report(self):
        from gatedtrain.registration import normalize

        reg = normalize(longrun_registration())
        rep = self.report
        hb, hc = rep["baseline_checkpoint_hash"], rep["candidate_checkpoint_hash"]
        res = evaluate_items(reg["clause"], arm(self.base, hb), arm(self.cand, hc), items=range(LONGRUN_ITEMS),
                             seed=int(reg["bootstrap"]["seed"]), n_boot=int(reg["bootstrap"]["replicates"]),
                             min_paired=10, max_unpaired=0, score_range=(0.0, 1.0),
                             expected_models={"baseline": hb, "candidate": hc})
        self.assertEqual(res["status"], "PASS", res)
        self.assertEqual(res["status_reasons"], [])
        self.assertEqual(res["n_items"], LONGRUN_ITEMS)
        for key in ("delta_hat", "sd_diff", "achieved_power", "mde"):
            self.assertAlmostEqual(res[key], rep[key], places=12, msg=key)
        self.assertAlmostEqual(res["ci_lower"], rep["interval"]["lower"], places=12)
        self.assertAlmostEqual(res["ci_upper"], rep["interval"]["upper"], places=12)
        self.assertAlmostEqual(res["p_value"], 8.53658039653515e-05, places=12)
        self.assertIs(res["p_value_at_floor"], True)
        self.assertEqual((res["missing_baseline"], res["missing_candidate"]), (0, 0))
        self.assertEqual(res["direction"], "higher_is_better")


if __name__ == "__main__":
    unittest.main()
