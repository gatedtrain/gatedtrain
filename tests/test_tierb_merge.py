"""H-3 `gatedtrain merge-eval` (TIER-B-SPEC section 1; interim rule G-16; tests/README.md section 11.3).

NeMo Evaluator sends one `seed` per service with every request (orchestrator.py:881-889 at ea35ba3),
so K single-seed runs with `repeats: 1` are merged and `repeat` is relabelled by registered seed
order. Each run is a recorded NeMo Evaluator output dir: <run>/gsm8k/results.jsonl
(orchestrator.py:1046-1047, :1081) and the composed-config snapshot <run>/full_config.yaml
(config/snapshot.py:39), which is where the run's seed and repeats are read.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from _support import EXIT_OK, EXIT_REFUSED, SKEL_ROOT, describe, file_sha256, parse_errors, read_jsonl, run_cli
from _tierb import NEMO_RESULT_KEYS, TIERB, baseline_reward, build_eval_run, read_source, extract_answer

SEEDS = [11, 12]


def fixture_rows():
    return [{"question": r["question"], "answer": extract_answer(r["answer"])} for r in read_source("test")]


class MergeCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-merge-"))
        reg = (TIERB / "registration.template.yaml").read_text(encoding="utf-8")
        self.reg = self.tmp / "registration.yaml"
        self.reg.write_text(reg.replace("{prompt_set_hash}", "sha256:" + "0" * 64), encoding="utf-8")
        self.rows = fixture_rows()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def runs(self, per_run: dict | None = None):
        per_run = per_run or {}
        out = []
        for r, seed in enumerate(SEEDS):
            kw = per_run.get(r, {})
            kw.setdefault("seed", seed)
            kw.setdefault("model", "baseline")
            out.append(build_eval_run(self.tmp / f"rep{r}", self.rows, reward_fn=baseline_reward, **kw))
        return out

    def merge(self, runs, out=None):
        out = out or self.tmp / "merged"
        return out, run_cli(["merge-eval", "--registration", str(self.reg), "--out", str(out), *map(str, runs)],
                            cwd=SKEL_ROOT, timeout=60)

    def assert_refused(self, runs, field: str):
        out, proc = self.merge(runs)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = parse_errors(proc.stderr)
        self.assertTrue(any(e["rule"] == "MERGE-EVAL" and e["field"] == field for e in errs),
                        f"expected error[MERGE-EVAL] {field}; got {errs}")
        self.assertFalse(out.exists(), "nothing is written on a refusal")


class TestMergeEval(MergeCase):
    def test_merge_relabels_repeat_by_seed_order(self):
        runs = self.runs()
        out, proc = self.merge(runs)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        merged = read_jsonl(out / "results.jsonl")
        self.assertEqual(len(merged), len(self.rows) * len(SEEDS))
        keys = [(r["problem_idx"], r["repeat"]) for r in merged]
        self.assertEqual(keys, sorted(keys), "sorted by (problem_idx, repeat)")
        self.assertEqual(set(keys), {(i, r) for i in range(len(self.rows)) for r in range(len(SEEDS))})
        sources = {r: {(x["problem_idx"]): x for x in read_jsonl(run / "gsm8k" / "results.jsonl")}
                   for r, run in enumerate(runs)}
        for rec in merged:
            self.assertEqual(set(rec), set(NEMO_RESULT_KEYS))
            self.assertEqual(rec["metadata"]["eval_seed"], SEEDS[rec["repeat"]])
            src = dict(sources[rec["repeat"]][rec["problem_idx"]])
            expect = {**src, "repeat": rec["repeat"], "metadata": {**src["metadata"], "eval_seed": SEEDS[rec["repeat"]]}}
            self.assertEqual(rec, expect, "only repeat and metadata.eval_seed change")

    def test_provenance(self):
        runs = self.runs()
        out, proc = self.merge(runs)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        doc = json.loads((out / "merge.json").read_text(encoding="utf-8"))
        self.assertEqual([s["repeat"] for s in doc["sources"]], [0, 1])
        self.assertEqual([s["seed"] for s in doc["sources"]], SEEDS)
        for s, run in zip(doc["sources"], runs):
            self.assertEqual(s["results_sha256"], file_sha256(run / "gsm8k" / "results.jsonl"))
        self.assertTrue(any("vLLM" in n and "batch" in n for n in doc["notes"]), "G-16 note")

    def test_refuses_existing_out(self):
        runs = self.runs()
        (self.tmp / "merged").mkdir()
        _, proc = self.merge(runs)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))


class TestMergeRefusals(MergeCase):
    def test_wrong_number_of_runs(self):
        runs = self.runs()
        self.assert_refused(runs[:1], "runs")

    def test_missing_run(self):
        runs = self.runs()
        shutil.rmtree(runs[1] / "gsm8k")
        self.assert_refused(runs, "runs[1]")

    def test_empty_run(self):
        runs = self.runs({1: {"empty": True}})
        self.assert_refused(runs, "runs[1]")

    def test_mismatched_problem_sets(self):
        runs = self.runs({1: {"drop": {3}}})
        self.assert_refused(runs, "runs[1]")

    def test_repeats_not_one(self):
        runs = self.runs({0: {"repeats": 2}})
        self.assert_refused(runs, "runs[0]")

    def test_seed_not_registered(self):
        runs = self.runs({1: {"seed": 99}})
        self.assert_refused(runs, "runs[1]")

    def test_seed_order_swapped(self):
        runs = self.runs({0: {"seed": SEEDS[1]}, 1: {"seed": SEEDS[0]}})
        self.assert_refused(runs, "runs[0]")



class TestMergeSamplerConsistency(MergeCase):
    """LOW-1: each run's full_config.yaml must match the registration sampler (temperature 0.7,
    top_p 1.0, max_new_tokens 1024 -> NeMo Evaluator `max_tokens`, enable_thinking false; G-15) and
    the other K runs (model). GenerationConfig fields: config/services.py:38-44 at ea35ba3;
    extra_body: services.py:101."""

    def test_temperature(self):
        self.assert_refused(self.runs({1: {"generation": {"temperature": 0.8}}}), "runs[1]")

    def test_top_p(self):
        self.assert_refused(self.runs({1: {"generation": {"top_p": 0.9}}}), "runs[1]")

    def test_max_tokens(self):
        self.assert_refused(self.runs({1: {"generation": {"max_tokens": 512}}}), "runs[1]")

    def test_model_differs_across_runs(self):
        self.assert_refused(self.runs({1: {"model": "candidate"}}), "runs[1]")

    def test_enable_thinking(self):
        self.assert_refused(self.runs({1: {"enable_thinking": True}}), "runs[1]")

    def test_all_runs_differ_from_registration(self):
        """Consistent across runs but not the registered sampler: still refused."""
        runs = self.runs({0: {"generation": {"temperature": 1.0}}, 1: {"generation": {"temperature": 1.0}}})
        out, proc = self.merge(runs)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue(any(e["rule"] == "MERGE-EVAL" for e in parse_errors(proc.stderr)), describe(proc))
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
