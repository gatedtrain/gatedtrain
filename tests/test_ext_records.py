"""Per-item arms, item set, merge and pairing through `gatedtrain ext-collect` / `ext-merge-eval`
(tests/README.md section 11.18: EXT-1 to EXT-5, EXT-8 to EXT-11).

Sources: MULTIMODAL-SPEC.md MM-1 to MM-8, MM-10; design 05 section 3.3, 3.4, 3.8 (revision 11 per-item record,
R11-2, R11-9, R11-14), 05 section 6.1 (score range, R11-11). Statistics are `gates.evaluate_items`'s; these
tests check what ext-collect feeds it and what it refuses before writing.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from _ext import (ADAPTER, EVALUATOR_ID, SCORER, SEEDS, ExtScenario, baseline_score, build_arm, build_item_set,
                  candidate_score, gsm8k_rows, merge, record, write_canonical)
from _support import (EXIT_GATE_NOT_PASS, EXIT_OK, EXIT_REFUSED, describe, file_sha256, load_report, parse_errors,
                      parse_run_dir, read_jsonl)


def no_run_dir(runs_root: Path) -> bool:
    return not (runs_root.exists() and any(p.is_dir() for p in runs_root.iterdir()))


class ExtCase(unittest.TestCase):
    """One scenario per class; each collect gets a new runs root."""

    reg_mutate = None

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-rec-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = ExtScenario(cls.tmp / "sc", reg_mutate=cls.reg_mutate)
        cls.n = 0

    def runs(self) -> Path:
        type(self).n += 1
        return self.tmp / f"runs-{type(self).n}"

    def refused(self, runs: Path, proc, rule: str, field_prefix: str | None = None):
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = parse_errors(proc.stderr)
        hits = [e for e in errs if e["rule"] == rule and (field_prefix is None or e["field"].startswith(field_prefix))]
        self.assertTrue(hits, f"expected error[{rule}] {field_prefix}...; got {errs}\n{describe(proc)}")
        self.assertTrue(no_run_dir(runs), "a refusal creates no run directory (MM-10)")
        return hits[0]

    def gate(self, proc, status: str, code: int):
        self.assertEqual(proc.returncode, code, describe(proc))
        rep = load_report(parse_run_dir(proc.stdout))
        self.assertEqual(rep["status"], status, {k: rep.get(k) for k in ("status", "status_reasons", "missing_reasons")})
        return rep

    def cand(self, name: str, **kw) -> Path:
        return self.sc.arm(name, candidate_score, root=self.tmp / "arms", **kw)


# ---------------------------------------------------------------------------
# EXT-1 to EXT-3 (MM-1 to MM-3): arm schema refusals
# ---------------------------------------------------------------------------

def _set(**kw):
    def edit(rec):
        if rec["item"] == "3" and rec["repeat"] == 1:
            rec.update(kw)
        return rec
    return edit


LINE_CASES = [
    ("score and error both set", _set(error={"class": "scorer_error", "message": "x"})),
    ("neither score nor error", _set(score=None)),
    ("unknown key", _set(prompt="not a record field")),
    ("seed is not the registered seed", _set(seed=99)),
    ("repeat outside K", _set(repeat=2)),
    ("item not a string", _set(item=3)),
    ("score above range", _set(score=1.5)),
    ("score below range", _set(score=-0.5)),
    ("score boolean", _set(score=True)),
    ("score string", _set(score="1")),
    ("error not an object", _set(score=None, error="scorer raised")),
    ("error class not declared", _set(score=None, error={"class": "model_crashed", "message": "x"})),
    ("sub_metrics not finite", _set(sub_metrics={"tokens": float("inf")})),
    ("output_ref escapes the arm", _set(output_ref={"path": "../x.txt", "sha256": "sha256:" + "0" * 64})),
    ("output_ref file absent", _set(output_ref={"path": "outputs/3-1.txt", "sha256": "sha256:" + "0" * 64})),
]


class TestArmLineRefusals(ExtCase):

    def test_each_line_rule(self):
        for name, edit in LINE_CASES:
            with self.subTest(case=name):
                arm = self.cand(name.replace(" ", "_"), edit=edit)
                runs = self.runs()
                self.refused(runs, self.sc.collect(runs, candidate=arm), "EXT-ARM", "--eval-candidate")

    def test_non_finite_score(self):
        arm = self.cand("nan", edit=_set(score=float("nan")))
        runs = self.runs()
        self.refused(runs, self.sc.collect(runs, candidate=arm), "EXT-ARM", "--eval-candidate")

    def test_duplicate_key(self):
        arm = self.cand("dup")
        dup = json.dumps(record("3", 1, SEEDS[1], 1.0))
        with (arm / "items.jsonl").open("a", encoding="utf-8") as f:
            f.write(dup + "\n")
        runs = self.runs()
        self.refused(runs, self.sc.collect(runs, candidate=arm), "EXT-ARM", "--eval-candidate")

    def test_unparseable_line(self):
        arm = self.cand("garbled", extra_lines=["{not json"])
        runs = self.runs()
        self.refused(runs, self.sc.collect(runs, candidate=arm), "EXT-ARM", "--eval-candidate")

    def test_empty_arm(self):
        arm = self.cand("empty")
        (arm / "items.jsonl").write_text("", encoding="utf-8")
        runs = self.runs()
        self.refused(runs, self.sc.collect(runs, candidate=arm), "EXT-ARM", "--eval-candidate")

    def test_parent_directory(self):
        """H-6 regression: the directory holding the arms is not an arm."""
        runs = self.runs()
        self.refused(runs, self.sc.collect(runs, baseline=self.sc.baseline.parent), "EXT-ARM", "--eval-baseline")


META_CASES = [
    ("eval_meta absent", {"__delete_file__": True}),
    ("unknown key", {"note": "x"}),
    ("schema", {"schema": "gatedtrain.something_else"}),
    ("evaluator is not the gate's", {"evaluator_id": "gsm8k.nemo_evaluator@0.1"}),
    ("adapter", {"adapter": "other_adapter"}),
    ("metric", {"metric": "reward"}),
    ("direction", {"direction": "lower_is_better"}),
    ("score range", {"score_range": [0, 100]}),
    ("scorer", {"scorer": dict(SCORER, name="pattern")}),
    ("scorer code hash", {"scorer": dict(SCORER, code_sha256="sha256:" + "f" * 64)}),
    ("item set hash", {"item_set_hash": "sha256:" + "e" * 64}),
    ("sampler hash", {"sampler_hash": "sha256:" + "d" * 64}),
    ("sampler", {"sampler": {"name": "deployment", "temperature": 1.0, "top_p": 1.0, "max_new_tokens": 1024,
                             "enable_thinking": False}}),
    ("normalizer for an LLM", {"normalizer_sha256": "sha256:" + "c" * 64}),
    ("env version for an LLM", {"env_version_sha256": "sha256:" + "c" * 64}),
    ("checkpoint shape", {"checkpoint": "sha256:" + "a" * 64}),
    ("framework missing", {"framework": ...}),
]


class TestArmMetaRefusals(ExtCase):

    def test_each_meta_rule(self):
        for name, meta in META_CASES:
            with self.subTest(case=name):
                delete_file = meta.pop("__delete_file__", False)
                arm = self.cand("meta_" + name.replace(" ", "_"), meta=meta)
                if delete_file:
                    (arm / "eval_meta.json").unlink()
                runs = self.runs()
                self.refused(runs, self.sc.collect(runs, candidate=arm), "EXT-ARM", "--eval-candidate")

    def test_unmerged_seed_arm(self):
        """eval_seeds must be the registered seeds: one single-seed run is not an arm (MM-5, MM-2)."""
        arm = self.cand("one_seed", seeds=[SEEDS[0]])
        runs = self.runs()
        self.refused(runs, self.sc.collect(runs, candidate=arm), "EXT-ARM", "--eval-candidate")


class TestArmAccepted(ExtCase):

    def test_item_meta_and_output_ref(self):
        """`item_meta` never pairs and is accepted; an `output_ref` naming a file in the arm with its hash is
        accepted. Both leave the statistics unchanged."""
        def edit(rec):
            rec["item_meta"] = {"task": "gsm8k", "row": int(rec["item"])}
            rec["sub_metrics"] = {"tokens": 20}
            if rec["item"] == "0":
                rec["output_ref"] = {"path": f"outputs/{rec['item']}-{rec['repeat']}.txt", "sha256": None}
            return rec
        arm = self.cand("annotated", edit=edit)
        (arm / "outputs").mkdir()
        lines = read_jsonl(arm / "items.jsonl")
        for rec in lines:
            if rec["output_ref"]:
                p = arm / rec["output_ref"]["path"]
                p.write_text(f"The answer is {rec['item']}", encoding="utf-8")
                rec["output_ref"]["sha256"] = file_sha256(p)
        (arm / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
        plain = self.gate(self.sc.collect(self.runs()), "PASS", EXIT_OK)
        rep = self.gate(self.sc.collect(self.runs(), candidate=arm), "PASS", EXIT_OK)
        for k in ("delta_hat", "interval", "p_value", "n_items", "achieved_power"):
            self.assertEqual(rep[k], plain[k], k)


# ---------------------------------------------------------------------------
# EXT-4 (MM-4): evaluator errors are missing items
# ---------------------------------------------------------------------------

class TestErrorRecords(ExtCase):

    def errored(self, name):
        return self.cand(name, edit=_set(score=None, error={"class": "scorer_error", "message": "ValueError"}))

    def test_error_missing_at_bound_zero(self):
        rep = self.gate(self.sc.collect(self.runs(), candidate=self.errored("err0")), "MISSING", EXIT_GATE_NOT_PASS)
        self.assertIn("missing_above_bound", rep["status_reasons"])
        self.assertEqual(rep["missing_reasons"], {"baseline": {}, "candidate": {"scorer_error": 1}})
        self.assertEqual(rep["missing_candidate"], 1)


class TestErrorRecordsWithinBound(ExtCase):
    reg_mutate = staticmethod(lambda reg: reg.update(pairing={"max_unpaired": 1}))

    def test_error_within_bound(self):
        arm = self.cand("err1", edit=_set(score=None, error={"class": "scorer_error", "message": "ValueError"}))
        rep = self.gate(self.sc.collect(self.runs(), candidate=arm), "PASS", EXIT_OK)
        self.assertEqual(rep["missing_reasons"]["candidate"], {"scorer_error": 1})
        self.assertEqual(rep["n_items"], 30)      # item 3 still pairs on repeat 0 (R11A-8)
        self.assertEqual(rep["missing_candidate"], 1)     # never scored 0 (mutant "error counted as score 0")


# ---------------------------------------------------------------------------
# EXT-5, EXT-11 (MM-6, MM-3 model identity): item set and evaluated checkpoint
# ---------------------------------------------------------------------------

class TestItemSet(ExtCase):

    def test_absent_item_is_missing(self):
        drop = {("7", 0), ("7", 1)}
        b = self.sc.arm("baseline", baseline_score, root=self.tmp / "absent", drop=drop)
        c = self.sc.arm("candidate", candidate_score, root=self.tmp / "absent", drop=drop)
        rep = self.gate(self.sc.collect(self.runs(), baseline=b, candidate=c), "MISSING", EXIT_GATE_NOT_PASS)
        self.assertEqual(rep["missing_reasons"], {"baseline": {"absent": 2}, "candidate": {"absent": 2}})
        self.assertEqual(rep["n_items"], 29)

    def test_item_not_in_set_is_missing(self):
        """Design decision 2026-11-11 (05 section 3.8, revision 11): a record whose item is not in the
        registered set makes the clause MISSING `unknown_item`; the run is written (exit 1), not refused."""
        arm = self.cand("unknown_item", edit=_set(item="999"))
        rep = self.gate(self.sc.collect(self.runs(), candidate=arm), "MISSING", EXIT_GATE_NOT_PASS)
        self.assertIn("unknown_item", rep["status_reasons"])

    def test_item_set_hash_checked(self):
        runs = self.runs()
        self.sc.render(reg_mutate=lambda reg: reg["eval_distribution"].update(item_set_hash="sha256:" + "9" * 64))
        self.addCleanup(self.sc.render)
        self.refused(runs, self.sc.collect(runs), "REG-FIELD", "registration.eval_distribution.item_set_hash")

    def test_candidate_evaluated_another_model(self):
        """05 section 3.3 (R11-14): the arm's model_hash must be the evaluated cut's; otherwise MISSING."""
        arm = self.cand("wrong_model", model=self.sc.base_model)
        rep = self.gate(self.sc.collect(self.runs(), candidate=arm), "MISSING", EXIT_GATE_NOT_PASS)
        self.assertIn("model_hash_mismatch", rep["status_reasons"])


class TestItemSetShape(unittest.TestCase):

    def test_duplicate_and_non_string_items_refused(self):
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-items-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        for name, ids in (("duplicate", ["0", "1", "1"]), ("not_string", ["0", 1, "2"])):
            with self.subTest(case=name):
                sc = ExtScenario(tmp / name)
                rows = gsm8k_rows("test")[:3]
                shutil.rmtree(sc.items_dir)
                build_item_set(sc.items_dir, rows, ids=ids)
                sc.render()
                runs = tmp / f"runs-{name}"
                proc = sc.collect(runs)
                self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
                self.assertIn("EXT-ITEMS", [e["rule"] for e in parse_errors(proc.stderr)], proc.stderr)
                self.assertTrue(no_run_dir(runs))


# ---------------------------------------------------------------------------
# EXT-8 (MM-7): pairing by (item, repeat), never by line order
# ---------------------------------------------------------------------------

class TestPairingByKey(ExtCase):

    def test_line_order_irrelevant(self):
        plain = self.gate(self.sc.collect(self.runs()), "PASS", EXIT_OK)
        shuffled = self.cand("shuffled", shuffle=True)
        rep = self.gate(self.sc.collect(self.runs(), candidate=shuffled), "PASS", EXIT_OK)
        for k in ("delta_hat", "interval", "p_value", "sd_diff", "baseline_score", "candidate_score"):
            self.assertEqual(rep[k], plain[k], k)
        self.assertNotEqual(read_jsonl(shuffled / "items.jsonl")[0]["item"], "0")


# ---------------------------------------------------------------------------
# EXT-9 (MM acceptance "equivalence with Tier B")
# ---------------------------------------------------------------------------

class TestTierBEquivalence(unittest.TestCase):
    """The Tier B fixture (30 GSM8K problems, K = 2, seeds 11 and 12, the same clause and bootstrap) through
    tierb-collect, and the same scores as item records through ext-collect: equal statistics."""

    KEYS = ("status", "status_reasons", "delta_hat", "interval", "p_value", "p_value_at_floor", "achieved_power",
            "mde", "underpowered", "n_items", "sd_diff", "sd_zero", "baseline_score", "candidate_score",
            "missing_baseline", "missing_candidate", "natural_units", "plan")

    def test_statistics_equal(self):
        from test_tierb_collect import Scenario
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-equiv-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        tb = Scenario(tmp / "tierb")
        _, tb_proc = tb.collect(runs_root=tmp / "runs-tierb")
        self.assertEqual(tb_proc.returncode, EXIT_OK, describe(tb_proc))
        tb_rep = load_report(parse_run_dir(tb_proc.stdout))
        sc = ExtScenario(tmp / "ext")
        ext_proc = sc.collect(tmp / "runs-ext")
        self.assertEqual(ext_proc.returncode, EXIT_OK, describe(ext_proc))
        ext_rep = load_report(parse_run_dir(ext_proc.stdout))
        for k in self.KEYS:
            self.assertEqual(ext_rep[k], tb_rep[k], k)


# ---------------------------------------------------------------------------
# EXT-10 (MM-5): ext-merge-eval
# ---------------------------------------------------------------------------

class TestMerge(ExtCase):

    def single(self, name: str, seed: int, *, items=None, score=baseline_score, **kw) -> Path:
        return build_arm(self.tmp / "merge-src" / name, items if items is not None else self.sc.items, score,
                         model=self.sc.base_model, item_set=self.sc.item_set_hash, seeds=[seed], **kw)

    def out(self) -> Path:
        type(self).n += 1
        return self.tmp / f"merged-{type(self).n}"

    def test_merge_relabels_by_registered_seed(self):
        r12 = self.single("s12", SEEDS[1])
        r11 = self.single("s11", SEEDS[0])
        out = self.out()
        proc = merge(self.sc.paths["registration"], out, [r12, r11], cwd=self.sc.root)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        recs = read_jsonl(out / "items.jsonl")
        self.assertEqual(len(recs), 2 * len(self.sc.items))
        self.assertTrue(all(r["seed"] == SEEDS[r["repeat"]] for r in recs))
        meta = json.loads((out / "eval_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["eval_seeds"], SEEDS)
        self.assertEqual(meta["checkpoint"]["model_hash"], self.sc.base_model)
        m = json.loads((out / "merge.json").read_text(encoding="utf-8"))
        self.assertEqual(m["schema"], "gatedtrain.ext_merge_eval")
        by_seed = {tuple(s["seeds"]): s for s in m["sources"]}
        self.assertEqual(set(by_seed), {(11,), (12,)})
        self.assertEqual(by_seed[(11,)]["items_sha256"], file_sha256(r11 / "items.jsonl"))
        self.assertEqual(by_seed[(12,)]["eval_meta_sha256"], file_sha256(r12 / "eval_meta.json"))
        # the merged arm is a valid ext-collect arm
        rep = self.gate(self.sc.collect(self.runs(), baseline=out), "PASS", EXIT_OK)
        self.assertEqual(rep["n_items"], 30)

    def test_multi_seed_run_alone(self):
        src = build_arm(self.tmp / "merge-src" / "both", self.sc.items, baseline_score, model=self.sc.base_model,
                        item_set=self.sc.item_set_hash)
        out = self.out()
        proc = merge(self.sc.paths["registration"], out, [src], cwd=self.sc.root)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertEqual(len(read_jsonl(out / "items.jsonl")), 60)

    def refuse(self, runs: list[Path], out: Path | None = None):
        out = out or self.out()
        proc = merge(self.sc.paths["registration"], out, runs, cwd=self.sc.root)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn("EXT-MERGE", [e["rule"] for e in parse_errors(proc.stderr)], proc.stderr)
        return out

    def test_refusals(self):
        s11, s12 = self.single("r11", SEEDS[0]), self.single("r12", SEEDS[1])
        empty = self.single("empty", SEEDS[1])
        (empty / "items.jsonl").write_text("", encoding="utf-8")
        badseed = self.single("badseed", SEEDS[1], edit=lambda r: dict(r, seed=SEEDS[0]) if r["item"] == "4" else r)
        fewer = self.single("fewer", SEEDS[1], items=self.sc.items[:-1])
        cases = {
            "missing run": [s11, self.tmp / "no-such-run"],
            "empty run": [s11, empty],
            "seed not covered": [s11],
            "seed twice": [s11, self.single("r11b", SEEDS[0])],
            "seed outside registration": [s11, s12, self.single("r13", 13)],
            "record seed disagrees with its repeat": [s11, badseed],
            "item sets differ": [s11, fewer],
            "checkpoint differs": [s11, self.single("ck", SEEDS[1], meta={
                "checkpoint": {"model_hash": "sha256:" + "a" * 64, "base_model_hash": None}})],
            "scorer differs": [s11, self.single("sc", SEEDS[1], meta={"scorer": dict(SCORER, version="other")})],
            "sampler hash differs": [s11, self.single("sh", SEEDS[1], meta={"sampler_hash": "sha256:" + "b" * 64})],
            "normalizer differs": [s11, self.single("nm", SEEDS[1], meta={"normalizer_sha256": "sha256:" + "c" * 64})],
            "env version differs": [s11, self.single("ev", SEEDS[1], meta={"env_version_sha256": "sha256:" + "d" * 64})],
        }
        for name, runs in cases.items():
            with self.subTest(case=name):
                out = self.refuse(runs)
                self.assertFalse(out.exists(), "a refused merge writes nothing")

    def test_out_exists(self):
        out = self.out()
        out.mkdir()
        self.refuse([self.single("o11", SEEDS[0]), self.single("o12", SEEDS[1])], out)
        self.assertEqual(list(out.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
