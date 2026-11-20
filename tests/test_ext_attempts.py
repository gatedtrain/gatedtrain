"""Attempt records and guard alignment (tests/README.md section 11.35: EXT-RES-1, EXT-RES-3)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from gatedtrain.errors import Problem, Refused
from gatedtrain.ext.attempts import (AttemptRecord, GuardAttempt, Segment, align, guard_attempts, read_attempts,
                                     segment_chain, split_steps)

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


class FixRoundTests(Base):
    G1 = ["2026-11-20T10:00:00Z START run-a attempt=1", "2026-11-20T10:30:00Z PREEMPTED run-a attempt=1 rc=143"]

    def test_trailing_empty_preempted_refuses(self):
        _, probs = self.check([start(1, "2026-11-20T10:00:05Z")], self.G1 + [
            "2026-11-20T11:00:00Z START run-a attempt=2", "2026-11-20T11:10:00Z PREEMPTED run-a attempt=2 rc=143"])
        self.assertIn("EXT-RES-3", self.rules(probs))
        self.assertTrue(any("final attempt" in p.message for p in probs))

    def test_end_ok_then_later_empty_preempted_refuses_about_final(self):
        _, probs = self.check([start(1, "2026-11-20T10:00:05Z"), end(1, "2026-11-20T10:20:00Z")], [
            "2026-11-20T10:00:00Z START run-a attempt=1", "2026-11-20T10:25:00Z END run-a attempt=1 rc=0",
            "2026-11-20T11:00:00Z START run-a attempt=2", "2026-11-20T11:10:00Z PREEMPTED run-a attempt=2 rc=143"])
        self.assertTrue(any("final attempt" in p.message for p in probs))
        self.assertFalse(any("non-final attempt has an end line" in p.message for p in probs))

    def test_fractional_and_offset_utc(self):
        pairs, probs = self.check(
            [start(1, "2026-11-20T10:00:05.123Z"), end(1, "2026-11-20T12:00:00+00:00")],
            ["2026-11-20T10:00:00Z START run-a attempt=1", "2026-11-20T12:00:01Z END run-a attempt=1 rc=0"])
        self.assertEqual(probs, [])
        self.assertEqual(len(pairs), 1)

    def test_bad_utc_refuses(self):
        self.write_attempts([start(1, "2026-11-20 10:00:05")])
        with self.assertRaises(Refused):
            read_attempts(self.dir)

    def test_same_second_boundary_pairs_with_later_attempt(self):
        # The run's start is in the same second as PREEMPTED of attempt 1 and START of attempt 2.
        pairs, probs = self.check(
            [start(1, "2026-11-20T11:00:00Z"), end(1, "2026-11-20T12:00:00Z")],
            self.G1[:1] + ["2026-11-20T11:00:00Z PREEMPTED run-a attempt=1 rc=143",
                           "2026-11-20T11:00:00Z START run-a attempt=2", "2026-11-20T12:00:01Z END run-a attempt=2 rc=0"])
        self.assertEqual(probs, [])
        self.assertIsNone(pairs[0][1])
        self.assertIsNotNone(pairs[1][1])

    def test_start_after_terminator_refuses(self):
        _, probs = self.check(
            [start(1, "2026-11-20T10:45:00Z"), end(1, "2026-11-20T12:00:00Z")],
            self.G1 + ["2026-11-20T11:00:00Z START run-a attempt=2", "2026-11-20T12:00:01Z END run-a attempt=2 rc=0"])
        self.assertIn("EXT-RES-1", self.rules(probs))

    def test_repaired_must_be_bool(self):
        self.write_attempts([start(1, "2026-11-20T10:00:05Z"),
                             start(2, "2026-11-20T11:00:05Z", dict(restore(), repaired="false"))])
        with self.assertRaises(Refused):
            read_attempts(self.dir)

    def test_multiple_invocations_attempt_counter_resets(self):
        pairs, probs = self.check(
            [start(1, "2026-11-20T10:00:05Z"), start(2, "2026-11-20T11:00:05Z", restore()),
             end(2, "2026-11-20T12:00:00Z")],
            self.G1 + ["2026-11-20T11:00:00Z START run-a attempt=1", "2026-11-20T12:00:01Z END run-a attempt=1 rc=0"])
        self.assertEqual(probs, [])
        self.assertEqual([g.index for g, _ in pairs], [1, 2])

    def test_running_attempt_in_the_middle_refuses(self):
        _, probs = self.check(
            [start(1, "2026-11-20T10:00:05Z"), start(2, "2026-11-20T11:00:05Z", restore()),
             end(2, "2026-11-20T12:00:00Z")],
            ["2026-11-20T10:00:00Z START run-a attempt=1", "2026-11-20T11:00:00Z START run-a attempt=2",
             "2026-11-20T12:00:01Z END run-a attempt=2 rc=0"])
        self.assertIn("EXT-RES-1", self.rules(probs))

    def test_end_before_start_and_duplicate_start_refuse(self):
        self.write_attempts([end(1, "2026-11-20T10:00:05Z"), start(1, "2026-11-20T10:00:05Z")])
        with self.assertRaises(Refused):
            read_attempts(self.dir)
        self.write_attempts([start(1, "2026-11-20T10:00:05Z"), start(1, "2026-11-20T10:00:06Z")])
        with self.assertRaises(Refused):
            read_attempts(self.dir)

    def test_two_run_starts_in_one_guard_attempt_refuse(self):
        _, probs = self.check(
            [start(1, "2026-11-20T10:00:05Z"), start(2, "2026-11-20T10:10:05Z", restore()),
             end(2, "2026-11-20T12:00:00Z")],
            ["2026-11-20T10:00:00Z START run-a attempt=1", "2026-11-20T12:00:01Z END run-a attempt=1 rc=0"])
        self.assertIn("EXT-RES-1", self.rules(probs))



def rec(a, restore_=None, end_status=None, repaired=False):
    return AttemptRecord(a, "2026-11-20T10:00:00Z", restore_, end_status, repaired)


def pairs_of(*recs):
    """Aligned pairs with a dummy guard attempt (None records are empty attempts)."""
    return [(GuardAttempt(i + 1, "2026-11-20T10:00:00Z", "preempted", 143), r) for i, r in enumerate(recs)]


class ChainTests(Base):
    def make_cut(self, step, attempt=None, extra=None):
        d = self.dir / "cuts" / f"checkpoint-{step}"
        d.mkdir(parents=True, exist_ok=True)
        doc = {"files": {}, "last_step_included": step}
        if attempt is not None:
            doc["attempt"] = attempt
        doc.update(extra or {})
        data = json.dumps(doc).encode()
        (d / "cut.json").write_bytes(data)
        return step, d, "sha256:" + __import__("hashlib").sha256(data).hexdigest()

    def rs(self, cut, step=None, repaired=False):
        s, d, sha = cut
        return {"cut": d.name, "last_step_included": step if step is not None else s, "cut_json_sha256": sha,
                "repaired": repaired}

    def chain(self, recs, cuts, max_steps=30):
        return segment_chain(pairs_of(*recs), [(s, d) for s, d, _ in cuts], run_id="r", max_steps=max_steps)

    def test_single_segment(self):
        segs, probs = self.chain([rec(1, None, "ok")], [])
        self.assertEqual(probs, [])
        self.assertEqual(segs, [Segment(1, "r.1", None, None, 30, None)])

    def test_two_attempts_restore_newest_cut(self):
        c10, c20 = self.make_cut(10, 1), self.make_cut(20, 1)
        segs, probs = self.chain([rec(1), rec(2, self.rs(c20), "ok")], [c10, c20])
        self.assertEqual(probs, [])
        self.assertEqual(segs, [Segment(1, "r.1", None, None, 20, None), Segment(2, "r.2", 1, 20, 30, "checkpoint-20")])

    def test_restore_of_older_cut_refuses(self):
        c10, c20 = self.make_cut(10, 1), self.make_cut(20, 1)
        _, probs = self.chain([rec(1), rec(2, self.rs(c10), "ok")], [c10, c20])
        self.assertEqual(self.rules(probs), ["EXT-RES-2"])

    def test_older_cut_restore_when_attempt_one_wrote_newer_refuses(self):
        # The case a step-only rule cannot catch: attempt 1 wrote 10 and 20, attempt 2 restores 10 and is final.
        c10, c20 = self.make_cut(10, 1), self.make_cut(20, 1)
        _, probs = self.chain([rec(1), rec(2, self.rs(c10), "ok")], [c10, c20])
        self.assertEqual(self.rules(probs), ["EXT-RES-2"])

    def test_missing_attempt_in_resumed_run_refuses(self):
        c10 = self.make_cut(10)
        _, probs = self.chain([rec(1), rec(2, self.rs(c10), "ok")], [c10])
        self.assertEqual(self.rules(probs), ["EXT-RES-2"])
        self.assertIn("does not record the attempt that wrote it", probs[0].message)

    def test_non_positive_or_bool_attempt_refuses(self):
        for bad in (0, -1, True, "1"):
            c10 = self.make_cut(10, bad)
            _, probs = self.chain([rec(1), rec(2, self.rs(c10), "ok")], [c10])
            self.assertEqual(self.rules(probs), ["EXT-RES-2"], bad)

    def test_single_attempt_may_omit_attempt(self):
        c10 = self.make_cut(10)
        segs, probs = self.chain([rec(1, None, "ok")], [c10])
        self.assertEqual((probs, len(segs)), ([], 1))

    def test_cut_attributed_to_nonexistent_attempt_refuses(self):
        c10 = self.make_cut(10, 7)
        _, probs = self.chain([rec(1), rec(2, self.rs(c10), "ok")], [c10])
        self.assertEqual(self.rules(probs), ["EXT-RES-2"])

    def test_cut_step_not_above_writers_branch_step_refuses(self):
        c10, c20 = self.make_cut(10, 1), self.make_cut(20, 2)
        c5 = self.make_cut(5, 3)
        _, probs = self.chain([rec(1), rec(2, self.rs(c10)), rec(3, self.rs(c20), "ok")], [c5, c10, c20])
        self.assertEqual(self.rules(probs), ["EXT-RES-2"])

    def test_restore_of_uncommitted_cut_refuses(self):
        c10 = self.make_cut(10, 1)
        ghost = (20, self.dir / "cuts" / "checkpoint-20", "sha256:" + "1" * 64)
        _, probs = self.chain([rec(1), rec(2, self.rs(ghost), "ok")], [c10])
        self.assertEqual(self.rules(probs), ["EXT-RES-2"])

    def test_restore_sha_mismatch_refuses(self):
        c10 = self.make_cut(10, 1)
        bad = dict(self.rs(c10), cut_json_sha256="sha256:" + "2" * 64)
        _, probs = self.chain([rec(1), rec(2, bad, "ok")], [c10])
        self.assertEqual(self.rules(probs), ["EXT-RES-2"])

    def test_restore_last_step_mismatch_refuses(self):
        c10 = self.make_cut(10, 1)
        _, probs = self.chain([rec(1), rec(2, self.rs(c10, step=9), "ok")], [c10])
        self.assertEqual(self.rules(probs), ["EXT-RES-2"])

    def test_first_attempt_with_restore_refuses(self):
        c10 = self.make_cut(10, 1)
        _, probs = self.chain([rec(1, self.rs(c10), "ok")], [c10])
        self.assertEqual(self.rules(probs), ["EXT-RES-2"])

    def test_empty_attempt_is_skipped(self):
        c10 = self.make_cut(10, 1)
        segs, probs = self.chain([rec(1), None, rec(2, self.rs(c10), "ok")], [c10])
        self.assertEqual(probs, [])
        self.assertEqual([(s.attempt, s.parent, s.branch_step) for s in segs], [(1, None, None), (2, 1, 10)])

    def test_later_attempt_without_prior_cut_restarts_from_scratch(self):
        segs, probs = self.chain([rec(1), rec(2, None, "ok")], [])
        self.assertEqual(probs, [])
        self.assertEqual((segs[0].end_step, segs[1].branch_step), (0, 0))

    def test_later_attempt_without_restore_despite_cut_refuses(self):
        c10 = self.make_cut(10, 1)
        _, probs = self.chain([rec(1), rec(2, None, "ok")], [c10])
        self.assertEqual(self.rules(probs), ["EXT-RES-2"])

    def test_three_attempts_chain(self):
        c10, c20 = self.make_cut(10, 1), self.make_cut(20, 2)
        segs, probs = self.chain([rec(1), rec(2, self.rs(c10)), rec(3, self.rs(c20), "ok")], [c10, c20])
        self.assertEqual(probs, [])
        self.assertEqual([(s.attempt, s.parent, s.branch_step, s.end_step) for s in segs],
                         [(1, None, None, 10), (2, 1, 10, 20), (3, 2, 20, 30)])

    def test_restore_missing_newest_cut_of_attempt_one_refuses(self):
        c10, c20, c30 = self.make_cut(10, 1), self.make_cut(20, 1), self.make_cut(30, 2)
        _, probs = self.chain([rec(1), rec(2, self.rs(c10)), rec(3, self.rs(c30), "ok")], [c10, c20, c30])
        self.assertEqual(self.rules(probs), ["EXT-RES-2"])

    def test_two_cuts_at_one_step_refuse(self):
        c10, dup = self.make_cut(10, 1), self.make_cut(10, 1)
        other = self.dir / "cuts" / "other-10"
        other.mkdir()
        (other / "cut.json").write_bytes(b"{}")
        _, probs = segment_chain(pairs_of(rec(1), rec(2, self.rs(c10), "ok")), [(10, c10[1]), (10, other)],
                                 run_id="r", max_steps=30)
        self.assertEqual(self.rules(probs), ["EXT-RES-2"])

    def test_non_final_attempt_without_cut_restores_same_cut(self):
        c10 = self.make_cut(10, 1)
        segs, probs = self.chain([rec(1), rec(2, self.rs(c10)), rec(3, self.rs(c10), "ok")], [c10])
        self.assertEqual(probs, [])
        self.assertEqual([(s.attempt, s.branch_step, s.end_step) for s in segs],
                         [(1, None, 10), (2, 10, 10), (3, 10, 30)])

    def test_repaired_restore_accepted_when_committed(self):
        c10 = self.make_cut(10, 1)
        segs, probs = self.chain([rec(1), rec(2, self.rs(c10, repaired=True), "ok", True)], [c10])
        self.assertEqual(probs, [])
        self.assertEqual(segs[1].restore_cut, "checkpoint-10")

    def test_repaired_restore_of_uncommitted_cut_refuses(self):
        ghost = (10, self.dir / "cuts" / "checkpoint-10", "sha256:" + "1" * 64)
        _, probs = self.chain([rec(1), rec(2, self.rs(ghost, repaired=True), "ok", True)], [])
        self.assertEqual(self.rules(probs), ["EXT-RES-2"])


class SplitTests(unittest.TestCase):
    CHAIN = [Segment(1, "r.1", None, None, 10, None), Segment(2, "r.2", 1, 10, 20, "checkpoint-10")]

    @staticmethod
    def rows(attempt, steps):
        return [{"attempt": attempt, "step": s} for s in steps]

    def test_orphaned_records_after_last_cut(self):
        records = self.rows(1, range(1, 14)) + self.rows(2, range(11, 21))
        kept, orphaned, probs = split_steps(self.CHAIN, records)
        self.assertEqual(probs, [])
        self.assertEqual(sorted((r["attempt"], r["step"]) for r in orphaned), [(1, 11), (1, 12), (1, 13)])
        self.assertEqual(sorted(r["step"] for r in kept), list(range(1, 21)))

    def test_multiple_records_per_step_from_one_attempt_are_fine(self):
        records = self.rows(1, [1, 1, 2]) + self.rows(2, [3, 3])
        chain = [Segment(1, "r.1", None, None, 2, None), Segment(2, "r.2", 1, 2, 3, "checkpoint-2")]
        kept, orphaned, probs = split_steps(chain, records)
        self.assertEqual((probs, len(kept), orphaned), ([], 5, []))

    def test_gap_refuses(self):
        records = self.rows(1, range(1, 11)) + self.rows(2, range(12, 21))
        _, _, probs = split_steps(self.CHAIN, records)
        self.assertEqual(sorted({p.rule for p in probs}), ["EXT-RES-4"])

    def test_missing_tail_refuses(self):
        _, _, probs = split_steps(self.CHAIN, self.rows(1, range(1, 11)) + self.rows(2, range(11, 20)))
        self.assertEqual(sorted({p.rule for p in probs}), ["EXT-RES-4"])

    def test_overlap_refuses(self):
        # A malformed chain whose kept ranges overlap: both attempts keep step 10.
        chain = [Segment(1, "r.1", None, None, 10, None), Segment(2, "r.2", 1, 9, 20, "checkpoint-9")]
        _, _, probs = split_steps(chain, self.rows(1, range(1, 11)) + self.rows(2, range(10, 21)))
        self.assertEqual(sorted({p.rule for p in probs}), ["EXT-RES-4"])

    def test_unknown_attempt_refuses(self):
        _, _, probs = split_steps(self.CHAIN, self.rows(1, range(1, 11)) + self.rows(2, range(11, 21)) + self.rows(7, [3]))
        self.assertEqual(sorted({p.rule for p in probs}), ["EXT-RES-4"])

    def test_final_attempt_step_beyond_max_refuses(self):
        _, _, probs = split_steps(self.CHAIN, self.rows(1, range(1, 11)) + self.rows(2, range(11, 22)))
        self.assertEqual(sorted({p.rule for p in probs}), ["EXT-RES-4"])

    def test_step_at_or_below_branch_step_refuses(self):
        # Attempt 2 (branch 10) records step 10 again: not orphaned, a defect.
        _, orphaned, probs = split_steps(self.CHAIN, self.rows(1, range(1, 11)) + self.rows(2, range(10, 21)))
        self.assertEqual((sorted({p.rule for p in probs}), orphaned), (["EXT-RES-4"], []))

    def test_step_zero_and_negative_refuse(self):
        for bad in (0, -3):
            _, orphaned, probs = split_steps(self.CHAIN, self.rows(1, range(1, 11)) + self.rows(2, range(11, 21))
                                             + self.rows(1, [bad]))
            self.assertEqual((sorted({p.rule for p in probs}), orphaned), (["EXT-RES-4"], []), bad)

    def test_malformed_records_refuse(self):
        good = self.rows(1, range(1, 11)) + self.rows(2, range(11, 21))
        for bad in ({"step": 3}, {"attempt": 1}, {"attempt": 1, "step": "3"}, {"attempt": True, "step": 3},
                    {"attempt": 1, "step": True}, {"attempt": 1, "step": 2.0}, {"attempt": None, "step": 1}):
            _, _, probs = split_steps(self.CHAIN, good + [bad])
            self.assertEqual(sorted({p.rule for p in probs}), ["EXT-RES-4"], bad)

    def test_non_final_attempt_without_cut_is_all_orphaned(self):
        # Attempt 2 wrote no cut; attempt 3 restores cut 10 again. Attempt 2's records are orphaned.
        chain = [Segment(1, "r.1", None, None, 10, None), Segment(2, "r.2", 1, 10, 10, "checkpoint-10"),
                 Segment(3, "r.3", 2, 10, 20, "checkpoint-10")]
        records = self.rows(1, range(1, 14)) + self.rows(2, range(11, 16)) + self.rows(3, range(11, 21))
        kept, orphaned, probs = split_steps(chain, records)
        self.assertEqual(probs, [])
        self.assertEqual(sorted(r["step"] for r in kept), list(range(1, 21)))
        self.assertEqual({r["attempt"] for r in orphaned}, {1, 2})
        self.assertEqual(len(orphaned), 3 + 5)


if __name__ == "__main__":
    unittest.main()
