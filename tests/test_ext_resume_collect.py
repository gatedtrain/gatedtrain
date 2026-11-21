"""`gatedtrain ext-collect` of a resumed run: the segment chain, the ledger and the report (tests/README.md section
11.35: EXT-RES-1 to EXT-RES-5; design 05 section 12.4, decisions/RESUMABLE-RUNS-DESIGN.md sections 5.2 to 5.5).

The fixture is the synthetic adapter's run with `attempts.jsonl` that starts with the header line (the new layout), `cut.json` in every cut and a
`samples.jsonl` whose lines carry the attempt that logged them. An `attempts.jsonl` without that header (the
launchers' older formats) or none is the old layout and collects as before (attempt_id "0").
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

from _ext import ExtScenario, guard_log, shuffle_args, train_row_count
from _support import EXIT_OK, EXIT_REFUSED, describe, file_sha256, load_report, parse_run_dir, run_cli

HEADER = {"schema": "gatedtrain.ext.attempts", "version": 1}
SENTENCE = "resumed run: not bit-identical to an uninterrupted run"

# Three attempts over 12 steps, cuts at 4 (attempt 1), 8 (attempt 2) and 12 (attempt 3):
#   attempt 1 runs steps 1-6, is preempted; steps 5-6 follow its last cut and are redone;
#   attempt 2 restores checkpoint-4, runs steps 5-9, is preempted; step 9 is redone;
#   attempt 3 restores checkpoint-8, runs steps 9-12 and ends ok.
STEPS = (4, 8, 12)
RUNS = {1: range(1, 7), 2: range(5, 10), 3: range(9, 13)}
KEPT = {1: range(1, 5), 2: range(5, 9), 3: range(9, 13)}
ORPHANED = {(1, 5), (1, 6), (2, 9)}


def write_cut_json(cut: Path, step: int, attempt: int) -> str:
    (cut / "cut.json").write_text(json.dumps({"last_step_included": step, "attempt": attempt}), encoding="utf-8")
    return file_sha256(cut / "cut.json")


def write_samples(stage_out: Path, runs: dict[int, range]) -> None:
    """samples.jsonl lines of the fixture run's steps (the rows of step s are fixed), tagged with the attempt that
    ran each step."""
    by_step: dict[int, list[dict]] = {}
    for line in (stage_out / "samples.jsonl").read_text(encoding="utf-8").splitlines():
        doc = json.loads(line)
        by_step.setdefault(doc["step"], []).append(doc)
    with (stage_out / "samples.jsonl").open("w", encoding="utf-8") as f:
        for a, steps in runs.items():
            for s in steps:
                for doc in by_step[s]:
                    f.write(json.dumps({**doc, "attempt": a}, sort_keys=True) + "\n")


def three_attempts(sc: ExtScenario, *, restore_step: dict[int, int] | None = None) -> None:
    """Rewrite the scenario's run as the three-attempt run above (new layout)."""
    out = sc.stage_out
    shas = {}
    for step, attempt in zip(STEPS, (1, 2, 3)):
        shas[step] = write_cut_json(out / f"checkpoint-{step}", step, attempt)
    restore_step = restore_step or {2: 4, 3: 8}

    def restore(step: int) -> dict:
        return {"cut": f"checkpoint-{step}", "last_step_included": step, "cut_json_sha256": shas[step],
                "repaired": False}
    rows = [
        HEADER,
        {"event": "start", "attempt": 1, "utc": "2026-11-11T09:00:05Z", "restore": None},
        {"event": "start", "attempt": 2, "utc": "2026-11-11T09:12:05Z", "restore": restore(restore_step[2])},
        {"event": "start", "attempt": 3, "utc": "2026-11-11T09:32:05Z", "restore": restore(restore_step[3])},
        {"event": "end", "attempt": 3, "utc": "2026-11-11T10:00:00Z", "status": "ok"},
    ]
    (out / "attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    write_samples(out, RUNS)
    guard_log(sc.host_work, sc.label, lines=[
        "2026-11-11T09:00:00Z GUARD started pid=1",
        f"2026-11-11T09:00:00Z START {sc.label} attempt=1",
        f"2026-11-11T09:10:00Z PREEMPTED {sc.label} attempt=1 rc=137",
        f"2026-11-11T09:12:00Z START {sc.label} attempt=2",
        f"2026-11-11T09:30:00Z PREEMPTED {sc.label} attempt=2 rc=137",
        f"2026-11-11T09:32:00Z START {sc.label} attempt=3",
        f"2026-11-11T10:00:00Z END {sc.label} attempt=3 rc=0",
    ])


def one_attempt(sc: ExtScenario) -> None:
    """The scenario's default run (cuts 5 and 10) as a new-layout run of one attempt."""
    out = sc.stage_out
    for step in (5, 10):
        write_cut_json(out / f"checkpoint-{step}", step, 1)
    rows = [HEADER, {"event": "start", "attempt": 1, "utc": "2026-11-11T09:00:05Z", "restore": None},
            {"event": "end", "attempt": 1, "utc": "2026-11-11T10:00:00Z", "status": "ok"}]
    (out / "attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    write_samples(out, {1: range(1, 11)})
    guard_log(sc.host_work, sc.label, lines=[
        f"2026-11-11T09:00:00Z START {sc.label} attempt=1",
        f"2026-11-11T10:00:00Z END {sc.label} attempt=1 rc=0",
    ])


def ledger(run_dir: Path) -> list[dict]:
    return pq.read_table(run_dir / "ledger" / "consumption.parquet").to_pylist()


class Case(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-resume-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.n = 0

    def runs(self) -> Path:
        type(self).n += 1
        return self.tmp / f"runs-{type(self).n}"

    def scenario(self, **kw) -> ExtScenario:
        type(self).n += 1
        return ExtScenario(self.tmp / f"sc-{type(self).n}", **kw)

    def refused(self, proc, rule: str) -> None:
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertNotIn("Traceback", proc.stderr)
        self.assertIn(f"error[{rule}]", proc.stderr, describe(proc))


class TestThreeAttempts(Case):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.sc = ExtScenario(cls.tmp / "sc", stage_steps=STEPS)
        three_attempts(cls.sc)
        cls.proc = cls.sc.collect(cls.tmp / "runs")

    def run_dir(self) -> Path:
        self.assertEqual(self.proc.returncode, EXIT_OK, describe(self.proc))
        return parse_run_dir(self.proc.stdout)

    def test_attempt_ids_and_epochs(self):
        d = self.run_dir()
        rows = ledger(d)
        ids = {r["attempt_id"] for r in rows}
        self.assertEqual(ids, {f"{d.name}.{a}" for a in (1, 2, 3)})
        for r in rows:
            self.assertEqual(r["attempt_id"], f"{d.name}.{r['controller_epoch']}")

    def test_parent_ref(self):
        d = self.run_dir()
        for r in ledger(d):
            a = r["controller_epoch"]
            want = None if a == 1 else {"kind": "attempt", "id": f"{d.name}.{a - 1}:{ {2: 4, 3: 8}[a] }"}
            self.assertEqual(r["parent_ref"], want)

    def test_orphaned_rows(self):
        d = self.run_dir()
        orphaned = [r for r in ledger(d) if r["status"] == "orphaned"]
        self.assertEqual({(r["controller_epoch"], r["step"]) for r in orphaned}, ORPHANED)
        self.assertEqual(len(orphaned), len({r["sample_id"] for r in orphaned}))

    def test_kept_rows_cover_every_step_once(self):
        d = self.run_dir()
        kept = [r for r in ledger(d) if r["status"] == "kept"]
        self.assertEqual({r["status"] for r in ledger(d)}, {"kept", "orphaned"})
        for r in kept:
            self.assertIn(r["step"], KEPT[r["controller_epoch"]])
        self.assertEqual(sorted({r["step"] for r in kept}), list(range(1, 13)))
        keys = [(r["step"], r["data_ref"]["shard"]) for r in kept]
        self.assertEqual(len(keys), len(set(keys)), "one kept row per (step, shard)")
        self.assertEqual({r["step"]: r["controller_epoch"] for r in kept},
                         {s: a for a, steps in KEPT.items() for s in steps})

    def test_report_json(self):
        rep = load_report(self.run_dir())
        self.assertEqual(rep["status"], "PASS")
        self.assertEqual(rep["attempts"], 3)
        self.assertEqual(rep["restored_at"], [4, 8])
        self.assertEqual(rep["steps_redone"], 3)
        self.assertEqual(rep["resume_chain"], [{"attempt": 2, "parent": 1, "branch_step": 4},
                                               {"attempt": 3, "parent": 2, "branch_step": 8}])
        self.assertEqual(rep["consumption_precision"], "iteration")
        self.assertEqual(rep["gpu_seconds"], 600 + 1080 + 1680)

    def test_report_text(self):
        text = (self.run_dir() / "report" / "gate_report.txt").read_text(encoding="utf-8").splitlines()
        for line in ("attempts: 3", "restored_at: [4, 8]", "steps_redone: 3", SENTENCE):
            self.assertIn(line, text)

    def test_lineage_keeps_orphaned_counts(self):
        d = self.run_dir()
        orph = pq.read_table(d / "lineage" / "orphaned.parquet").to_pylist()
        self.assertEqual({(int(r["attempt_id"].rsplit(".", 1)[1]), r["step"]) for r in orph}, ORPHANED)

    def test_impact_counts_and_ranges_report_orphaned(self):
        d = self.run_dir()
        total, seen = 0, []
        for shard in sorted(p.name for p in (self.sc.train).glob("*.jsonl")):
            proc = run_cli(["impact", str(self.sc.train / shard), "--run-dir", str(d), "--json"], cwd=self.sc.root)
            self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
            doc = json.loads(proc.stdout)
            total += doc["counts"]["orphaned"]
            seen += doc["ranges"]["orphaned"]
        self.assertEqual(total, len([r for r in ledger(d) if r["status"] == "orphaned"]))
        self.assertTrue(seen)
        self.assertTrue(all(g["attempt_id"].startswith(d.name + ".") for g in seen))

    def test_impact_ignores_orphaned_ledger_rows(self):
        d = self.run_dir()
        shard = "shard-00003.jsonl"
        before = json.loads(run_cli(["impact", str(self.sc.train / shard), "--run-dir", str(d), "--json"],
                                    cwd=self.sc.root).stdout)["results"]
        first = {r["first_step"] for r in before}
        self.assertTrue(first and min(first) > 1)
        copy = self.tmp / "impact-copy"
        shutil.copytree(d, copy)
        rows = ledger(copy)
        extra = dict(next(r for r in rows if r["data_ref"]["shard"] == shard))
        extra.update(step=1, status="orphaned", sample_id="sha256:" + "0" * 64)
        pq.write_table(pa.Table.from_pylist(rows + [extra], schema=pq.read_schema(copy / "ledger" / "consumption.parquet")),
                       copy / "ledger" / "consumption.parquet")
        after = json.loads(run_cli(["impact", str(self.sc.train / shard), "--run-dir", str(copy), "--json"],
                                   cwd=self.sc.root).stdout)["results"]
        self.assertEqual({r["first_step"] for r in after}, first)


    # ---- EXT-RES-5: impact attribution over segments -------------------------------------------------------

    def impact(self, run_dir: Path, shard: str) -> dict:
        proc = run_cli(["impact", str(self.sc.train / shard), "--run-dir", str(run_dir), "--json"], cwd=self.sc.root)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        return json.loads(proc.stdout)

    def edited_copy(self, name: str, drop) -> Path:
        """A copy of the run whose ledger lacks the rows `drop` selects."""
        copy = self.tmp / name
        shutil.copytree(self.run_dir(), copy)
        path = copy / "ledger" / "consumption.parquet"
        table = pq.read_table(path)
        keep = [r for r in table.to_pylist() if not drop(r)]
        pq.write_table(pa.Table.from_pylist(keep, schema=table.schema), path)
        return copy

    def test_kept_rows_attribute_to_covering_cuts(self):
        d = self.run_dir()
        want = {  # shard -> (first kept step; cuts with last_step_included >= it)
            "shard-00000.jsonl": [4, 8, 12], "shard-00001.jsonl": [4, 8, 12],
            "shard-00002.jsonl": [8, 12], "shard-00003.jsonl": [8, 12]}
        for shard, cuts in want.items():
            doc = self.impact(d, shard)
            self.assertEqual([r["last_step_included"] for r in doc["results"]], cuts, shard)

    def test_orphaned_section_per_attempt_ranges(self):
        d = self.run_dir()
        doc = self.impact(d, "shard-00001.jsonl")
        self.assertEqual(doc["orphaned"], {"count": 1, "attempts": {f"{d.name}.1": [[5, 5]]}})
        doc = self.impact(d, "shard-00003.jsonl")
        self.assertEqual(doc["orphaned"], {"count": 1, "attempts": {f"{d.name}.2": [[9, 9]]}})
        self.assertNotIn("orphaned", self.impact(d, "shard-00000.jsonl"), "no orphaned record, no section")

    def test_shard_consumed_only_in_orphaned_steps_has_no_checkpoint(self):
        d = self.run_dir()
        copy = self.edited_copy("only-orphaned", lambda r: r["data_ref"]["shard"] == "shard-00002.jsonl"
                                and r["status"] == "kept")
        doc = self.impact(copy, "shard-00002.jsonl")
        self.assertEqual(doc["results"], [])
        self.assertEqual(doc["orphaned"], {"count": 1, "attempts": {f"{d.name}.1": [[6, 6]]}})
        self.assertEqual(doc["counts"]["orphaned"], 1)

    def test_orphaned_ranges_are_contiguous_runs_per_attempt(self):
        d = self.run_dir()
        copy = self.tmp / "ranges-copy"
        shutil.copytree(d, copy)
        path = copy / "lineage" / "orphaned.parquet"
        table = pq.read_table(path)
        rows = [r for r in table.to_pylist() if r["shard"] == "shard-00001.jsonl"]
        extra = [dict(rows[0], step=s) for s in (6, 7, 9)]
        pq.write_table(pa.Table.from_pylist(rows + extra, schema=table.schema), path)
        got = self.impact(copy, "shard-00001.jsonl")["orphaned"]
        self.assertEqual(got, {"count": 4, "attempts": {f"{d.name}.1": [[5, 7], [9, 9]]}})

    def test_final_attempt_tail_after_last_cut_is_attributed_to_no_checkpoint(self):
        d = self.run_dir()
        copy = self.edited_copy("tail", lambda r: r["data_ref"]["shard"] == "shard-00000.jsonl" and r["step"] <= 3)
        ckpt = copy / "lineage" / "checkpoints.parquet"
        table = pq.read_table(ckpt)
        pq.write_table(pa.Table.from_pylist([r for r in table.to_pylist() if r["last_step_included"] != 12],
                                            schema=table.schema), ckpt)
        doc = self.impact(copy, "shard-00000.jsonl")  # kept only at steps 11 and 12, last cut is 8
        self.assertEqual(doc["results"], [])
        self.assertNotIn("orphaned", doc)


class TestOldLayoutImpactUnchanged(Case):
    def test_old_layout_json_answer_is_byte_for_byte_the_pre_resume_answer(self):
        """The fixture was captured from the implementation at a3afd25 (before EXT-RES-5) for this scenario."""
        sc = self.scenario()
        (sc.stage_out / "attempts.jsonl").unlink()
        proc = sc.collect(self.runs())
        d = parse_run_dir(proc.stdout)
        out = run_cli(["impact", str(sc.train / "shard-00001.jsonl"), "--run-dir", str(d), "--json"], cwd=sc.root)
        self.assertEqual(out.returncode, EXIT_OK, describe(out))
        doc = json.loads(out.stdout)
        doc["query"] = None
        text = json.dumps(doc, indent=1, sort_keys=True).replace(d.name, "<run>")
        want = (Path(__file__).parent / "fixtures" / "ext_resume_old_layout_impact.json").read_text(encoding="utf-8")
        self.assertEqual(text, want)
        self.assertEqual(set(doc), {"query", "results", "counts", "ranges"})


class TestOneAttempt(Case):
    def test_single_attempt_new_layout(self):
        sc = self.scenario()
        one_attempt(sc)
        proc = sc.collect(self.runs())
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        d = parse_run_dir(proc.stdout)
        rows = ledger(d)
        self.assertEqual({r["attempt_id"] for r in rows}, {f"{d.name}.1"})
        self.assertEqual({r["controller_epoch"] for r in rows}, {1})
        self.assertEqual({r["status"] for r in rows}, {"kept"})
        rep = load_report(d)
        self.assertEqual((rep["attempts"], rep["restored_at"], rep["steps_redone"]), (1, [], 0))
        text = (d / "report" / "gate_report.txt").read_text(encoding="utf-8").splitlines()
        for line in ("attempts: 1", "restored_at: []", "steps_redone: 0"):
            self.assertIn(line, text)
        self.assertNotIn(SENTENCE, text)
        self.assertFalse([l for l in text if "bit-identical" in l])


class TestOldLayout(Case):
    def test_no_attempts_file_is_the_old_layout(self):
        sc = self.scenario()
        (sc.stage_out / "attempts.jsonl").unlink()
        self.assert_old(sc)

    def test_launcher_attempts_file_is_the_old_layout(self):
        """TRL, SB3, LeRobot, ddpo, flow-grpo and whisper launchers: {attempt, started_at, argv}."""
        sc = self.scenario()
        (sc.stage_out / "attempts.jsonl").write_text(
            json.dumps({"attempt": 1, "started_at": "2026-11-11T09:30:00Z", "argv": ["train.py"]}) + "\n",
            encoding="utf-8")
        self.assert_old(sc)

    def test_nemo_hook_attempts_file_is_the_old_layout(self):
        """NeMo RL cad, clevr and agentic hooks: start and end lines without a header or an end status."""
        sc = self.scenario()
        rows = [{"event": "start", "attempt": 1, "utc": "2026-11-11T09:30:05Z", "pid": 1, "stage": "x"},
                {"event": "end", "attempt": 1, "utc": "2026-11-11T09:59:00Z", "pid": 1, "stage": "x"}]
        (sc.stage_out / "attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.assert_old(sc)

    def assert_old(self, sc):
        proc = sc.collect(self.runs())
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        d = parse_run_dir(proc.stdout)
        self.assertEqual({r["attempt_id"] for r in ledger(d)}, {"0"})
        self.assertNotIn("status", {k for r in ledger(d) for k, v in r.items() if v is not None})
        rep = load_report(d)
        self.assertNotIn("attempts", rep)
        text = (d / "report" / "gate_report.txt").read_text(encoding="utf-8")
        self.assertNotIn("attempts:", text)
        self.assertNotIn("bit-identical", text)

    def test_end_rc_nonzero_then_zero_still_collects_in_the_old_layout(self):
        sc = self.scenario()
        guard_log(sc.host_work, sc.label, lines=[
            f"2026-11-11T09:00:00Z START {sc.label} attempt=1",
            f"2026-11-11T09:10:00Z END {sc.label} attempt=1 rc=1",
            f"2026-11-11T09:30:00Z START {sc.label} attempt=1",
            f"2026-11-11T10:00:00Z END {sc.label} attempt=1 rc=0",
        ])
        proc = sc.collect(self.runs())
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_ledger_without_status_column_reads_as_kept(self):
        sc = self.scenario()
        proc = sc.collect(self.runs())
        d = parse_run_dir(proc.stdout)
        table = pq.read_table(d / "ledger" / "consumption.parquet")
        old = table.drop_columns([c for c in ("status", "controller_epoch") if c in table.column_names])
        pq.write_table(old, d / "ledger" / "consumption.parquet")
        shard = "shard-00001.jsonl"
        out = run_cli(["impact", str(sc.train / shard), "--run-dir", str(d), "--json"], cwd=sc.root)
        self.assertEqual(out.returncode, EXIT_OK, describe(out))
        self.assertTrue(json.loads(out.stdout)["results"])


class TestRefusals(Case):
    def test_any_end_rc_nonzero_refuses_in_the_new_layout(self):
        sc = self.scenario()
        one_attempt(sc)
        guard_log(sc.host_work, sc.label, lines=[
            f"2026-11-11T09:00:00Z START {sc.label} attempt=1",
            f"2026-11-11T09:00:03Z END {sc.label} attempt=1 rc=1",
            f"2026-11-11T09:30:00Z START {sc.label} attempt=1",
            f"2026-11-11T10:00:00Z END {sc.label} attempt=1 rc=0",
        ])
        runs = self.runs()
        self.refused(sc.collect(runs), "EXT-RES-1")
        self.assertFalse(runs.exists() and any(runs.iterdir()))

    def test_malformed_guard_log_is_a_problem_not_a_traceback(self):
        sc = self.scenario()
        one_attempt(sc)
        guard_log(sc.host_work, sc.label, lines=[
            f"2026-11-11T09:00:00Z START {sc.label} attempt=1",
            f"2026-11-11T10:00:00Z END {sc.label} attempt=1",
        ])
        self.refused(sc.collect(self.runs()), "EXT-RES-3")

    def test_rollback_is_refused(self):
        sc = self.scenario(stage_steps=STEPS)
        three_attempts(sc, restore_step={2: 4, 3: 4})
        self.refused(sc.collect(self.runs()), "EXT-RES-2")

    def test_missing_attempt_in_the_records_refuses(self):
        sc = self.scenario(stage_steps=STEPS)
        three_attempts(sc)
        write_samples(sc.stage_out, {1: range(1, 7), 3: range(9, 13)})      # attempt 2's records are gone
        self.refused(sc.collect(self.runs()), "EXT-RES-4")


    def test_missing_attempt_in_the_records_refuses_before_the_run_directory(self):
        sc = self.scenario(stage_steps=STEPS)
        three_attempts(sc)
        write_samples(sc.stage_out, {1: range(1, 7), 3: range(9, 13)})
        runs = self.runs()
        self.refused(sc.collect(runs), "EXT-RES-4")
        self.assertFalse(runs.exists() and any(runs.iterdir()), "no run directory exists after the refusal")

    def test_unknown_header_version_refuses(self):
        sc = self.scenario()
        one_attempt(sc)
        lines = (sc.stage_out / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
        lines[0] = json.dumps({"schema": "gatedtrain.ext.attempts", "version": 2})
        (sc.stage_out / "attempts.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.refused(sc.collect(self.runs()), "EXT-RES-1")


def set_max_steps(sc: ExtScenario, value) -> None:
    path = sc.stage_out / "run_args.json"
    args = json.loads(path.read_text(encoding="utf-8"))
    if value is None:
        args.pop("max_steps")
    else:
        args["max_steps"] = value
    path.write_text(json.dumps(args, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extend_samples(sc: ExtScenario, steps, attempt: int) -> None:
    per_step, per_epoch = 4, train_row_count(sc.train) // 4
    with (sc.stage_out / "samples.jsonl").open("a", encoding="utf-8") as f:
        for s in steps:
            for pos in range(per_step):
                f.write(json.dumps({"step": s, "position": pos, "row": ((s - 1) % per_epoch) * per_step + pos,
                                    "attempt": attempt}, sort_keys=True) + "\n")


class TestFinalAttemptEnd(Case):
    """The final attempt ends at run_args.json `max_steps` when it is an integer (design 5.2). The synthetic adapter's
    own EXT-68 check wants the last cut at max_steps, so these runs use a wrapper without that check."""

    def collect(self, sc):
        from gatedtrain.ext import collect as mod
        from gatedtrain.ext import registry

        class NoStepCheck(Wrapped):
            def checks(self, **kw):
                return []

        real = registry.adapter
        with mock.patch.dict("os.environ", sc.env()), \
                mock.patch.object(mod.registry, "adapter", lambda name: NoStepCheck(real(name))):
            return mod.collect(sc.paths["spec"], adapter="synthetic", stage_out=sc.stage_out, launch_record=sc.launch,
                               smoke_record=sc.smoke, eval_baseline=sc.baseline, eval_candidate=sc.candidate,
                               runs_root=self.runs())

    def test_records_after_the_last_cut_are_kept_up_to_max_steps(self):
        sc = self.scenario()
        one_attempt(sc)
        set_max_steps(sc, 12)
        extend_samples(sc, (11, 12), 1)
        run_dir, _, _ = self.collect(sc)
        rows = ledger(run_dir)
        self.assertEqual(sorted({r["step"] for r in rows if r["status"] == "kept"}), list(range(1, 13)))

    def test_without_max_steps_records_after_the_last_cut_refuse(self):
        # Without `max_steps` the final segment ends at the last cut; later records are not dropped silently (they
        # would hide consumption), they refuse as records beyond the run's end (EXT-RES-4).
        from gatedtrain.errors import Refused

        sc = self.scenario()
        one_attempt(sc)
        set_max_steps(sc, None)
        extend_samples(sc, (11, 12), 1)
        with self.assertRaises(Refused) as cm:
            self.collect(sc)
        self.assertEqual({p.rule for p in cm.exception.problems}, {"EXT-RES-4"})
        self.assertIn("beyond max_steps 10", " ".join(p.message for p in cm.exception.problems))

    def test_without_max_steps_the_run_ends_at_the_last_cut(self):
        sc = self.scenario()
        one_attempt(sc)
        set_max_steps(sc, None)
        run_dir, _, _ = self.collect(sc)
        self.assertEqual(sorted({r["step"] for r in ledger(run_dir)}), list(range(1, 11)))

    def test_max_steps_below_the_last_cut_or_not_an_integer_refuses(self):
        # The last cut is step 10: a `max_steps` of 9 (or a non-integer) is not replaced by the cut step.
        from gatedtrain.errors import Refused

        for value in (9, "12", True, 12.0):
            sc = self.scenario()
            one_attempt(sc)
            set_max_steps(sc, value)
            with self.assertRaises(Refused, msg=value) as cm:
                self.collect(sc)
            hits = [p for p in cm.exception.problems if p.rule == "EXT-RES-4"]
            self.assertTrue(hits and "max_steps" in hits[0].message, (value, cm.exception.problems))

    def test_records_beyond_max_steps_refuse(self):
        from gatedtrain.errors import Refused

        sc = self.scenario()
        one_attempt(sc)
        set_max_steps(sc, 11)
        extend_samples(sc, (11, 12), 1)
        with self.assertRaises(Refused) as cm:
            self.collect(sc)
        self.assertEqual({p.rule for p in cm.exception.problems}, {"EXT-RES-4"})
        self.assertIn("beyond max_steps 11", " ".join(p.message for p in cm.exception.problems))


class TestStagePrecisionOnAResumedRun(Case):
    def test_stage_fallback_still_counts_redone_steps(self):
        sc = self.scenario(stage_steps=STEPS, args_mutate=shuffle_args)
        three_attempts(sc)
        proc = sc.collect(self.runs())
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        rep = load_report(parse_run_dir(proc.stdout))
        self.assertEqual((rep["consumption_precision"], rep["attempts"], rep["steps_redone"]), ("stage", 3, 3))

    def test_stage_fallback_missing_attempt_records_refuses(self):
        sc = self.scenario(stage_steps=STEPS, args_mutate=shuffle_args)
        three_attempts(sc)
        write_samples(sc.stage_out, {1: range(1, 7), 3: range(9, 13)})
        self.refused(sc.collect(self.runs()), "EXT-RES-4")

    def test_stage_fallback_without_samples_refuses(self):
        sc = self.scenario(stage_steps=STEPS, args_mutate=shuffle_args)
        three_attempts(sc)
        (sc.stage_out / "samples.jsonl").unlink()
        self.refused(sc.collect(self.runs()), "EXT-RES-4")


class Wrapped:
    """The synthetic adapter (chain-aware) whose consumption result `edit` changes."""
    edit = staticmethod(lambda cons: cons)

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def consumption(self, stage_out, data, cuts, *, run_id="", chain=None):
        return type(self).edit(dict(self._inner.consumption(stage_out, data, cuts, run_id=run_id, chain=chain)))


class TestAdapterResults(Case):
    def collect(self, sc, edit):
        from gatedtrain.ext import collect as mod
        from gatedtrain.ext import registry

        real = registry.adapter
        cls = type("W", (Wrapped,), {"edit": staticmethod(edit)})
        with mock.patch.dict("os.environ", sc.env()), \
                mock.patch.object(mod.registry, "adapter", lambda name: cls(real(name))):
            return mod.collect(sc.paths["spec"], adapter="synthetic", stage_out=sc.stage_out, launch_record=sc.launch,
                               smoke_record=sc.smoke, eval_baseline=sc.baseline, eval_candidate=sc.candidate,
                               runs_root=self.runs())

    def test_collector_stamps_status_and_parent(self):
        def edit(cons):
            cons["rows"] = [{k: v for k, v in r.items() if k not in ("status", "controller_epoch")}
                            for r in cons["rows"]]
            cons["orphaned"] = [{k: v for k, v in r.items() if k not in ("status", "controller_epoch")}
                                for r in cons["orphaned"]]
            return cons

        sc = self.scenario(stage_steps=STEPS)
        three_attempts(sc)
        run_dir, report, _ = self.collect(sc, edit)
        rows = ledger(run_dir)
        self.assertEqual({(r["status"] is None) for r in rows}, {False})
        self.assertEqual({r["status"] for r in rows if (r["controller_epoch"], r["step"]) in ORPHANED
                          and r["attempt_id"].endswith(f".{r['controller_epoch']}")} - {"kept"}, {"orphaned"})
        for r in rows:
            a = r["controller_epoch"]
            want = None if a == 1 else {"kind": "attempt", "id": f"{run_dir.name}.{a - 1}:{ {2: 4, 3: 8}[a] }"}
            self.assertEqual(r["parent_ref"], want)

    def test_row_naming_no_segment_refuses(self):
        from gatedtrain.errors import Refused

        def edit(cons):
            cons["orphaned"] = [{**r, "attempt_id": "elsewhere.9"} for r in cons["orphaned"]]
            return cons

        sc = self.scenario(stage_steps=STEPS)
        three_attempts(sc)
        with self.assertRaises(Refused) as cm:
            self.collect(sc, edit)
        self.assertEqual({p.rule for p in cm.exception.problems}, {"EXT-RES-4"})

    def test_epoch_not_of_the_segment_refuses(self):
        from gatedtrain.errors import Refused

        def edit(cons):
            cons["rows"] = [{**r, "controller_epoch": 3} for r in cons["rows"]]
            return cons

        sc = self.scenario(stage_steps=STEPS)
        three_attempts(sc)
        with self.assertRaises(Refused) as cm:
            self.collect(sc, edit)
        self.assertEqual({p.rule for p in cm.exception.problems}, {"EXT-RES-4"})

    def refuses(self, edit, words: str = "") -> None:
        from gatedtrain.errors import Refused

        sc = self.scenario(stage_steps=STEPS)
        three_attempts(sc)
        runs = self.runs()
        with mock.patch.object(self, "runs", lambda: runs), self.assertRaises(Refused) as cm:
            self.collect(sc, edit)
        self.assertEqual({p.rule for p in cm.exception.problems}, {"EXT-RES-4"}, cm.exception.problems)
        self.assertIn(words, " ".join(p.message for p in cm.exception.problems))
        self.assertFalse(runs.exists() and any(runs.iterdir()), "no run directory remains after the refusal")

    def test_kept_row_outside_its_segment_refuses(self):
        def edit(cons):
            row = next(r for r in cons["rows"] if r["controller_epoch"] == 1 and r["step"] == 2)
            row["step"] = 6                                   # attempt 1 keeps steps 1-4 only
            return cons

        self.refuses(edit, "kept row")

    def test_orphaned_row_of_the_final_segment_refuses(self):
        def edit(cons):
            cons["orphaned"] = cons["orphaned"] + [dict(next(r for r in cons["rows"] if r["controller_epoch"] == 3))]
            return cons

        self.refuses(edit, "orphaned row")

    def test_kept_rows_moved_to_orphaned_refuse(self):
        # Attempt 2 keeps step 6; the adapter lists it as orphaned (step not after the segment's end).
        def edit(cons):
            moved = [r for r in cons["rows"] if r["step"] == 6]
            cons["rows"] = [r for r in cons["rows"] if r["step"] != 6]
            cons["orphaned"] = cons["orphaned"] + moved
            return cons

        self.refuses(edit, "orphaned row")

    def test_gap_in_the_written_kept_rows_refuses(self):
        # The optional chain_problems hook is bypassed; the collector checks the rows it writes.
        def edit(cons):
            cons["rows"] = [r for r in cons["rows"] if r["step"] != 7]
            return cons

        self.refuses(edit, "no kept row at steps 7")

    def test_steps_redone_must_be_a_non_negative_integer(self):
        for bad in (True, -1, "3", 2.5):
            def edit(cons, bad=bad):
                cons["steps_redone"] = bad
                return cons

            self.refuses(edit, "steps_redone")

    def test_adapter_parent_ref_is_replaced_by_the_collectors(self):
        def edit(cons):
            for r in cons["rows"] + cons["orphaned"]:
                r["parent_ref"] = {"kind": "attempt", "id": "elsewhere.1:0"}
            return cons

        sc = self.scenario(stage_steps=STEPS)
        three_attempts(sc)
        run_dir, _, _ = self.collect(sc, edit)
        for r in ledger(run_dir):
            a = r["controller_epoch"]
            want = None if a == 1 else {"kind": "attempt", "id": f"{run_dir.name}.{a - 1}:{ {2: 4, 3: 8}[a] }"}
            self.assertEqual(r["parent_ref"], want)

    def test_no_orphaned_list_on_a_multi_attempt_chain_is_unknown_not_zero(self):
        def edit(cons):
            cons.pop("orphaned")
            cons.pop("steps_redone", None)
            return cons

        sc = self.scenario(stage_steps=STEPS)
        three_attempts(sc)
        run_dir, report, text = self.collect(sc, edit)
        self.assertIsNone(report["steps_redone"])
        self.assertIn("steps_redone: unknown", text.splitlines())


class NoChain:
    """The synthetic adapter with a `consumption` that does not take `chain`."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def consumption(self, stage_out, data, cuts, *, run_id=""):
        return self._inner.consumption(stage_out, data, cuts, run_id=run_id)


class TestAdapterWithoutChain(Case):
    def collect(self, sc: ExtScenario):
        from gatedtrain.ext import collect as mod
        from gatedtrain.ext import registry

        real = registry.adapter
        with mock.patch.dict("os.environ", sc.env()), \
                mock.patch.object(mod.registry, "adapter", lambda name: NoChain(real(name))):
            return mod.collect(sc.paths["spec"], adapter="synthetic", stage_out=sc.stage_out, launch_record=sc.launch,
                               smoke_record=sc.smoke, eval_baseline=sc.baseline, eval_candidate=sc.candidate,
                               runs_root=self.runs())

    def test_multi_segment_run_refuses(self):
        from gatedtrain.errors import Refused

        sc = self.scenario(stage_steps=STEPS)
        three_attempts(sc)
        with self.assertRaises(Refused) as cm:
            self.collect(sc)
        hits = [p for p in cm.exception.problems if p.rule == "EXT-RES-3"]
        self.assertTrue(hits, cm.exception.problems)
        self.assertIn("this adapter does not support resumed runs", hits[0].message)

    def test_single_segment_run_collects_the_old_way(self):
        sc = self.scenario()
        one_attempt(sc)
        run_dir, report, text = self.collect(sc)
        self.assertEqual(report["attempts"], 1)
        self.assertEqual({r["attempt_id"] for r in ledger(run_dir)}, {f"{run_dir.name}.1"})


if __name__ == "__main__":
    unittest.main()
