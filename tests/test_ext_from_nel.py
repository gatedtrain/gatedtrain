"""`gatedtrain ext-from-nel`: NeMo Evaluator runs to an item arm (tests/README.md section 11.18, EXT-35 to EXT-38).

Design decision (2026-11-11, EXT-G4): the converter is in slice 1. It reads K single-seed NeMo Evaluator
runs as `merge-eval` does (`<run>/gsm8k/results.jsonl`, orchestration/orchestrator.py:1046-1047, :1081 at
ea35ba3; `<run>/full_config.yaml`, config/snapshot.py:72-81), converts each record with
`items.from_nemo_evaluator` semantics (R11A-5) under the deciding evaluator's scorer `numeric_match`
(`scorer_rewards["scorer:numeric_match"]`, engine/eval_loop.py:785-794; a raising scorer leaves
`scoring_details[...] = {error, reward: 0.0}`, eval_loop.py:719-721), and writes `items.jsonl` and
`eval_meta.json` (MULTIMODAL-SPEC MM-1, MM-2) plus `merge.json`. Repeat r and seed eval_seeds[r] come from
the registration and the run order; `item` is the registered item set's key of row `problem_idx`;
`checkpoint.model_hash` is the evaluated checkpoint directory's deployable hash (EXT-14).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from _ext import (EVALUATOR_ID, SEEDS, ExtScenario, eval_meta, model_hash, write_canonical)
from _support import (EXIT_OK, EXIT_REFUSED, describe, file_sha256, load_report, parse_errors, parse_run_dir,
                      read_jsonl, run_cli)
from _tierb import baseline_reward, build_eval_run, candidate_reward, render_inputs

SCORER = "numeric_match"
KEY = f"scorer:{SCORER}"
RECORD_KEYS = {"item", "repeat", "seed", "score", "error", "sub_metrics", "output_ref", "item_meta"}


def scorer_baseline(i: int) -> float:
    """numeric_match scores, differing from the strict `reward` so the converter must read the scorer."""
    return 1.0 if i % 2 == 0 else 0.0


def scorer_candidate(i: int) -> float:
    return 1.0 if i % 5 != 4 else 0.0


def nel_runs(root: Path, rows, *, score_fn, model: str, seeds=SEEDS, **kw) -> list[Path]:
    """One single-seed run per seed (G-16), records carrying `scorer_rewards[KEY]`."""
    return [build_eval_run(root / f"rep{r}", rows, reward_fn=lambda i: 0.0, seed=s, model=model, scorer=SCORER,
                           scorer_fn=score_fn, **kw) for r, s in enumerate(seeds)]


def convert(sc: ExtScenario, out: Path, runs: list[Path], *, checkpoint: Path, items: Path | None = None,
            registration: Path | None = None, adapter: str = "synthetic", evaluator: str = EVALUATOR_ID):
    return run_cli(["ext-from-nel", "--registration", str(registration or sc.paths["registration"]),
                    "--adapter", adapter, "--evaluator", evaluator, "--items", str(items or sc.items_dir),
                    "--checkpoint", str(checkpoint), "--out", str(out), *map(str, runs)], cwd=sc.root, timeout=120)


class FromNelCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-nel-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = ExtScenario(cls.tmp / "sc")
        cls.n = 0

    def out(self) -> Path:
        type(self).n += 1
        return self.tmp / f"arm-{type(self).n}"


# ---------------------------------------------------------------------------
# EXT-35, EXT-36: field mapping and arm metadata
# ---------------------------------------------------------------------------

class TestConversion(FromNelCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.runs = nel_runs(cls.tmp / "nel-cand", cls.sc.rows, score_fn=scorer_candidate, model="cand",
                            scorer_error={4}, scorer_missing={5})
        cls.arm = cls.tmp / "cand-arm"
        cls.proc = convert(cls.sc, cls.arm, cls.runs, checkpoint=cls.sc.final_cut)

    def setUp(self):
        self.assertEqual(self.proc.returncode, EXIT_OK, describe(self.proc))

    def records(self):
        return read_jsonl(self.arm / "items.jsonl")

    def test_layout(self):
        self.assertEqual(sorted(p.name for p in self.arm.iterdir()), ["eval_meta.json", "items.jsonl", "merge.json"])

    def test_field_mapping(self):
        recs = self.records()
        self.assertEqual(len(recs), len(self.sc.items) * len(SEEDS))
        nel = {(json.loads(l)["problem_idx"], r): json.loads(l)
               for r, run in enumerate(self.runs)
               for l in (run / "gsm8k" / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
        for rec in recs:
            self.assertEqual(set(rec), RECORD_KEYS)
            i = rec["item_meta"]["problem_idx"]
            self.assertEqual(rec["item"], self.sc.items[i])            # the registered key, never the row index
            self.assertIsInstance(rec["item"], str)
            self.assertEqual(rec["seed"], SEEDS[rec["repeat"]])       # from the registration, not the record
            src = nel[(i, rec["repeat"])]
            self.assertEqual(rec["item_meta"]["model_response_sha256"],
                             "sha256:" + hashlib.sha256(src["model_response"].encode("utf-8")).hexdigest())
            self.assertIsNone(rec["output_ref"])
            self.assertIsNone(rec["sub_metrics"])
            if i in (4, 5):
                continue
            self.assertEqual(rec["score"], scorer_candidate(i))       # scorer_rewards[KEY], not `reward`
            self.assertIsNone(rec["error"])

    def test_scorer_errors_become_error_records(self):
        by = {(r["item_meta"]["problem_idx"], r["repeat"]): r for r in self.records()}
        for rep in range(len(SEEDS)):
            raised, missing = by[(4, rep)], by[(5, rep)]
            self.assertIsNone(raised["score"])
            self.assertEqual(raised["error"]["class"], "scorer_error")     # never 0 (eval_loop.py:719-721)
            self.assertIsNone(missing["score"])
            self.assertEqual(missing["error"]["class"], "missing_score")

    def test_eval_meta(self):
        meta = json.loads((self.arm / "eval_meta.json").read_text(encoding="utf-8"))
        want = eval_meta(model=model_hash(self.sc.final_cut), item_set=self.sc.item_set_hash,
                         framework={"name": "nemo-evaluator", "version": "0.0.0-fixture"})
        meta.pop("created_at", None)
        self.assertEqual(meta, json.loads(json.dumps(want)))
        self.assertEqual(meta["checkpoint"]["model_hash"], self.sc.cand_model)

    def test_merge_json(self):
        m = json.loads((self.arm / "merge.json").read_text(encoding="utf-8"))
        self.assertEqual(m["schema"], "gatedtrain.ext_from_nel")
        self.assertEqual([(s["repeat"], s["seed"]) for s in m["sources"]], [(0, 11), (1, 12)])
        for s, run in zip(m["sources"], self.runs):
            self.assertEqual(s["results_sha256"], file_sha256(run / "gsm8k" / "results.jsonl"))
            self.assertEqual(s["config_sha256"], file_sha256(run / "full_config.yaml"))


# ---------------------------------------------------------------------------
# EXT-37: refusals (EXT-NEL; nothing written)
# ---------------------------------------------------------------------------

class TestRefusals(FromNelCase):

    def refused(self, proc, out: Path, rule: str = "EXT-NEL"):
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn(rule, [e["rule"] for e in parse_errors(proc.stderr)], proc.stderr)
        self.assertFalse(out.exists() and any(out.iterdir()), "a refused conversion writes nothing")

    def runs(self, name: str, rows=None, **kw) -> list[Path]:
        return nel_runs(self.tmp / name, rows if rows is not None else self.sc.rows, score_fn=scorer_baseline,
                        model="base", **kw)

    def test_cases(self):
        good = self.runs("good")
        drifted = [dict(r) for r in self.sc.rows]
        drifted[2] = dict(drifted[2], question="A different question, from another dataset revision?")
        empty_ckpt = self.tmp / "not-a-model"
        empty_ckpt.mkdir()
        cases = {
            "wrong number of runs": dict(runs=good[:1]),
            "runs out of seed order": dict(runs=list(reversed(good))),
            "run without results": dict(runs=[good[0], self.tmp / "no-such-run"]),
            "run with repeats 2": dict(runs=[good[0], build_eval_run(self.tmp / "rep2", self.sc.rows,
                                                                      reward_fn=lambda i: 0.0, seed=SEEDS[1],
                                                                      model="base", scorer=SCORER,
                                                                      scorer_fn=scorer_baseline, repeats=2)]),
            "scorer not declared": dict(runs=self.runs("undeclared", declared_scorers=[])),
            "sampler differs": dict(runs=self.runs("hot", generation={"temperature": 1.0})),
            "thinking on": dict(runs=self.runs("think", enable_thinking=True)),
            "problem sets differ": dict(runs=[good[0], self.runs("drop", drop={7})[1]]),
            "problem outside the item set": dict(runs=self.runs("extra", rows=self.sc.rows + [self.sc.rows[0]])),
            "question differs from the item set": dict(runs=self.runs("drift", rows=drifted)),
            "checkpoint is not a model": dict(runs=good, checkpoint=empty_ckpt),
            "evaluator not of the adapter": dict(runs=good, evaluator="gsm8k.nemo_evaluator@0.1"),
        }
        for name, kw in cases.items():
            with self.subTest(case=name):
                out = self.out()
                proc = convert(self.sc, out, kw.pop("runs"), checkpoint=kw.pop("checkpoint", self.sc.base), **kw)
                self.refused(proc, out)

    def test_item_set_hash_differs(self):
        other = self.tmp / "other-items"
        shutil.copytree(self.sc.items_dir, other)
        with (other / "items.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"item": "extra", "question": "q?", "answer": "1"}) + "\n")
        out = self.out()
        proc = convert(self.sc, out, self.runs("ish"), checkpoint=self.sc.base, items=other)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn(("REG-FIELD", "registration.eval_distribution.item_set_hash"),
                      [(e["rule"], e["field"]) for e in parse_errors(proc.stderr)])
        self.assertFalse(out.exists())

    def test_out_exists(self):
        out = self.out()
        out.mkdir()
        self.refused(convert(self.sc, out, self.runs("ox"), checkpoint=self.sc.base), out)

    def test_unknown_adapter(self):
        out = self.out()
        self.refused(convert(self.sc, out, self.runs("ua"), checkpoint=self.sc.base, adapter="nope"), out,
                     "EXT-ADAPTER")


# ---------------------------------------------------------------------------
# EXT-38: round trip into ext-collect equals tierb-collect on the same NeMo Evaluator runs
# ---------------------------------------------------------------------------

class TestRoundTripEqualsTierB(unittest.TestCase):

    KEYS = ("status", "status_reasons", "delta_hat", "interval", "p_value", "p_value_at_floor", "achieved_power",
            "mde", "underpowered", "n_items", "sd_diff", "sd_zero", "baseline_score", "candidate_score",
            "missing_baseline", "missing_candidate", "natural_units", "plan")

    def test_statistics_equal(self):
        from test_tierb_collect import Scenario
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-nel-rt-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        tb = Scenario(tmp / "tierb")
        render_inputs(tb.root, tb.root / "export", scorer=SCORER, set_scorer=True)   # G-21: gate on numeric_match
        runs = {arm: nel_runs(tmp / "nel" / arm, tb.rows, score_fn=fn, model=arm, seeds=tb.seeds)
                for arm, fn in (("baseline", scorer_baseline), ("candidate", scorer_candidate))}
        merged = {}
        for arm, rs in runs.items():
            merged[arm] = tmp / "merged" / arm
            p = run_cli(["merge-eval", "--registration", str(tb.paths["registration"]), "--out", str(merged[arm]),
                         *map(str, rs)], cwd=tb.root)
            self.assertEqual(p.returncode, EXIT_OK, describe(p))
        _, tb_proc = tb.collect(baseline=merged["baseline"], candidate=merged["candidate"], runs_root=tmp / "runs-tb")
        self.assertEqual(tb_proc.returncode, EXIT_OK, describe(tb_proc))
        tb_rep = load_report(parse_run_dir(tb_proc.stdout))

        sc = ExtScenario(tmp / "ext")
        arms = {"baseline": tmp / "arms" / "baseline", "candidate": tmp / "arms" / "candidate"}
        for arm, ckpt in (("baseline", sc.base), ("candidate", sc.final_cut)):
            p = convert(sc, arms[arm], runs[arm], checkpoint=ckpt)
            self.assertEqual(p.returncode, EXIT_OK, describe(p))
        ext_proc = sc.collect(tmp / "runs-ext", baseline=arms["baseline"], candidate=arms["candidate"],
                              sources_root=tmp)
        self.assertEqual(ext_proc.returncode, EXIT_OK, describe(ext_proc))
        ext_rep = load_report(parse_run_dir(ext_proc.stdout))
        for k in self.KEYS:
            self.assertEqual(ext_rep[k], tb_rep[k], k)
        self.assertEqual(ext_rep["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
