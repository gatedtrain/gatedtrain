"""Attempt records and guard alignment (tests/README.md section 11.35: EXT-RES-1, EXT-RES-3)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from gatedtrain.errors import Problem, Refused
from gatedtrain.ext.attempts import AttemptRecord, GuardAttempt, align, guard_attempts, read_attempts

LABEL = "run-a"


def start(a, utc, restore=None):
    return {"event": "start", "attempt": a, "utc": utc, "restore": restore}


def end(a, utc, status="ok"):
    return {"event": "end", "attempt": a, "utc": utc, "status": status}


def restore(n=10, repaired=False):
    return {"cut": f"checkpoint-{n}", "last_step_included": n, "cut_json_sha256": "sha256:" + "0" * 64,
            "repaired": repaired}


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def write_attempts(self, rows):
        (self.dir / "attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def write_log(self, lines):
        p = self.dir / "gpu-guard.log"
        p.write_text("".join(x + "\n" for x in lines), encoding="utf-8")
        return p

    def check(self, rows, log):
        self.write_attempts(rows)
        recs = read_attempts(self.dir)
        guard = guard_attempts(self.write_log(log), LABEL)
        return align(recs, guard)

    def rules(self, problems):
        return sorted({p.rule for p in problems})


class ReadTests(Base):
    def test_reads_records(self):
        self.write_attempts([start(1, "2026-11-20T10:00:05Z"), start(2, "2026-11-20T11:00:05Z", restore(10, True)),
                             end(2, "2026-11-20T12:00:00Z")])
        recs = read_attempts(self.dir)
        self.assertEqual(recs, [AttemptRecord(1, "2026-11-20T10:00:05Z", None, None, False),
                                AttemptRecord(2, "2026-11-20T11:00:05Z", restore(10, True), "ok", True)])

    def test_symlinked_attempts_refuses(self):
        real = self.dir / "real.jsonl"
        real.write_text(json.dumps(start(1, "2026-11-20T10:00:05Z")) + "\n", encoding="utf-8")
        os.symlink(real, self.dir / "attempts.jsonl")
        with self.assertRaises(Refused) as cm:
            read_attempts(self.dir)
        self.assertEqual(cm.exception.problems[0].rule, "EXT-RES-1")

    def test_absent_and_malformed_refuse(self):
        with self.assertRaises(Refused):
            read_attempts(self.dir)
        (self.dir / "attempts.jsonl").write_text("not json\n", encoding="utf-8")
        with self.assertRaises(Refused):
            read_attempts(self.dir)

    def test_attempt_numbers_must_run_from_one(self):
        self.write_attempts([start(2, "2026-11-20T10:00:05Z")])
        with self.assertRaises(Refused):
            read_attempts(self.dir)


class GuardTests(Base):
    def test_guard_attempts(self):
        p = self.write_log([
            "2026-11-20T10:00:00Z START other attempt=1",
            "2026-11-20T10:00:00Z START run-a attempt=1",
            "2026-11-20T10:30:00Z PREEMPTED run-a attempt=1 rc=143",
            "2026-11-20T10:31:00Z WAIT run-a",
            "2026-11-20T11:00:00Z START run-a attempt=2",
            "2026-11-20T12:00:00Z END run-a attempt=2 rc=0"])
        self.assertEqual(guard_attempts(p, LABEL), [
            GuardAttempt(1, "2026-11-20T10:00:00Z", "preempted", 143),
            GuardAttempt(2, "2026-11-20T11:00:00Z", "end", 0)])

    def test_unterminated_is_running(self):
        p = self.write_log(["2026-11-20T10:00:00Z START run-a attempt=1"])
        self.assertEqual(guard_attempts(p, LABEL)[0].outcome, "running")


class AlignTests(Base):
    def test_single_clean_attempt(self):
        pairs, probs = self.check(
            [start(1, "2026-11-20T10:00:05Z"), end(1, "2026-11-20T12:00:00Z")],
            ["2026-11-20T10:00:00Z START run-a attempt=1", "2026-11-20T12:00:01Z END run-a attempt=1 rc=0"])
        self.assertEqual(probs, [])
        self.assertEqual([(g.index, r.attempt) for g, r in pairs], [(1, 1)])

    def test_preempted_then_ok(self):
        pairs, probs = self.check(
            [start(1, "2026-11-20T10:00:05Z"), start(2, "2026-11-20T11:00:05Z", restore()),
             end(2, "2026-11-20T12:00:00Z")],
            ["2026-11-20T10:00:00Z START run-a attempt=1", "2026-11-20T10:30:00Z PREEMPTED run-a attempt=1 rc=143",
             "2026-11-20T11:00:00Z START run-a attempt=2", "2026-11-20T12:00:01Z END run-a attempt=2 rc=0"])
        self.assertEqual(probs, [])
        self.assertEqual(len(pairs), 2)

    def test_empty_preempted_attempt_aligns_as_none(self):
        pairs, probs = self.check(
            [start(1, "2026-11-20T11:00:05Z"), end(1, "2026-11-20T12:00:00Z")],
            ["2026-11-20T10:00:00Z START run-a attempt=1", "2026-11-20T10:00:30Z PREEMPTED run-a attempt=1 rc=143",
             "2026-11-20T11:00:00Z START run-a attempt=2", "2026-11-20T12:00:01Z END run-a attempt=2 rc=0"])
        self.assertEqual(probs, [])
        self.assertIsNone(pairs[0][1])
        self.assertEqual(pairs[1][1].attempt, 1)

    def test_empty_attempt_without_preempted_refuses(self):
        # Guard attempt 1 ended rc=0 yet has no run start line; attempt 2 holds the start line.
        _, probs = self.check(
            [start(1, "2026-11-20T11:00:05Z"), end(1, "2026-11-20T12:00:00Z")],
            ["2026-11-20T10:00:00Z START run-a attempt=1", "2026-11-20T10:00:30Z END run-a attempt=1 rc=0",
             "2026-11-20T11:00:00Z START run-a attempt=2", "2026-11-20T12:00:01Z END run-a attempt=2 rc=0"])
        self.assertIn("EXT-RES-1", self.rules(probs))

    def test_any_end_nonzero_refuses(self):
        _, probs = self.check(
            [start(1, "2026-11-20T10:00:05Z"), start(2, "2026-11-20T11:00:05Z", restore()),
             end(2, "2026-11-20T12:00:00Z")],
            ["2026-11-20T10:00:00Z START run-a attempt=1", "2026-11-20T10:30:00Z END run-a attempt=1 rc=1",
             "2026-11-20T11:00:00Z START run-a attempt=2", "2026-11-20T12:00:01Z END run-a attempt=2 rc=0"])
        self.assertIn("EXT-RES-1", self.rules(probs))

    def test_non_final_with_end_line_refuses(self):
        _, probs = self.check(
            [start(1, "2026-11-20T10:00:05Z"), end(1, "2026-11-20T10:29:00Z"),
             start(2, "2026-11-20T11:00:05Z", restore()), end(2, "2026-11-20T12:00:00Z")],
            ["2026-11-20T10:00:00Z START run-a attempt=1", "2026-11-20T10:30:00Z PREEMPTED run-a attempt=1 rc=143",
             "2026-11-20T11:00:00Z START run-a attempt=2", "2026-11-20T12:00:01Z END run-a attempt=2 rc=0"])
        self.assertIn("EXT-RES-3", self.rules(probs))

    def test_non_final_without_preempted_refuses(self):
        _, probs = self.check(
            [start(1, "2026-11-20T10:00:05Z"), start(2, "2026-11-20T11:00:05Z", restore()),
             end(2, "2026-11-20T12:00:00Z")],
            ["2026-11-20T10:00:00Z START run-a attempt=1", "2026-11-20T10:30:00Z END run-a attempt=1 rc=0",
             "2026-11-20T11:00:00Z START run-a attempt=2", "2026-11-20T12:00:01Z END run-a attempt=2 rc=0"])
        self.assertIn("EXT-RES-3", self.rules(probs))

    def test_final_without_end_or_bad_status_refuses(self):
        log = ["2026-11-20T10:00:00Z START run-a attempt=1", "2026-11-20T12:00:01Z END run-a attempt=1 rc=0"]
        _, p1 = self.check([start(1, "2026-11-20T10:00:05Z")], log)
        self.assertIn("EXT-RES-3", self.rules(p1))
        _, p2 = self.check([start(1, "2026-11-20T10:00:05Z"), end(1, "2026-11-20T12:00:00Z", "failed")], log)
        self.assertIn("EXT-RES-3", self.rules(p2))

    def test_final_guard_preempted_refuses(self):
        _, probs = self.check(
            [start(1, "2026-11-20T10:00:05Z")],
            ["2026-11-20T10:00:00Z START run-a attempt=1", "2026-11-20T10:30:00Z PREEMPTED run-a attempt=1 rc=143"])
        self.assertIn("EXT-RES-3", self.rules(probs))

    def test_more_run_starts_than_guard_starts_refuses(self):
        _, probs = self.check(
            [start(1, "2026-11-20T10:00:05Z"), start(2, "2026-11-20T11:00:05Z", restore()),
             end(2, "2026-11-20T12:00:00Z")],
            ["2026-11-20T10:00:00Z START run-a attempt=1", "2026-11-20T12:00:01Z END run-a attempt=1 rc=0"])
        self.assertIn("EXT-RES-1", self.rules(probs))

    def test_window_pairing_not_index_pairing(self):
        # One run start line whose time lies in the second guard window: naive index pairing would pair it with the
        # first guard attempt (and refuse the first as having no end); window pairing yields an empty first attempt.
        pairs, probs = self.check(
            [start(1, "2026-11-20T11:00:05Z"), end(1, "2026-11-20T12:00:00Z")],
            ["2026-11-20T10:00:00Z START run-a attempt=1", "2026-11-20T10:30:00Z PREEMPTED run-a attempt=1 rc=143",
             "2026-11-20T11:00:00Z START run-a attempt=2", "2026-11-20T12:00:01Z END run-a attempt=2 rc=0"])
        self.assertEqual(probs, [])
        self.assertIsNone(pairs[0][1])
        self.assertIs(pairs[1][1].attempt, 1)

    def test_run_start_outside_every_window_refuses(self):
        _, probs = self.check(
            [start(1, "2026-11-20T09:00:00Z"), end(1, "2026-11-20T12:00:00Z")],
            ["2026-11-20T10:00:00Z START run-a attempt=1", "2026-11-20T12:00:01Z END run-a attempt=1 rc=0"])
        self.assertIn("EXT-RES-1", self.rules(probs))


if __name__ == "__main__":
    unittest.main()
