"""AN-23: the local backend's GPU accounting, budget, retries, GPU wait and GPU log (design revision 25, 05 §18.10).

The backend runs on `FakeRuntime` with a fake GPU probe and a fake clock (the stand-in adapter and the host layout of
`test_an_backend`); the project's budget is 1.0 GPU-hour. No test starts Docker or reads a GPU.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import os
import shutil
import threading
from pathlib import Path
from unittest import mock

from gatedtrain.ainative import backend as be
from gatedtrain.ainative.backend import LocalBackend
from gatedtrain.ainative.runtime import GPU_BUSY_MIB
from gatedtrain.errors import Refused
from gatedtrain.ext import launch
from gatedtrain.testing.fake_runtime import RC_STOPPED, FakeRuntime

from test_an_backend import IMAGE, SCRIPT, _Base, _train, setUpModule, tearDownModule  # noqa: F401

START = dt.datetime(2026, 11, 22, tzinfo=dt.timezone.utc)      # FakeRuntime's first container start


class Clock:
    """A fake clock the backend's `sleep` advances."""

    def __init__(self, at: dt.datetime = START) -> None:
        self.now = at

    def __call__(self) -> dt.datetime:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += dt.timedelta(seconds=s)

    def at(self, seconds: float) -> None:
        self.now = START + dt.timedelta(seconds=seconds)

    def mono(self) -> float:
        return (self.now - START).total_seconds()


class Probe:
    def __init__(self, *readings: int | None) -> None:
        self.readings = list(readings)
        self.calls = 0

    def used_mib(self) -> int | None:
        self.calls += 1
        return self.readings.pop(0) if len(self.readings) > 1 else self.readings[0]


class _Acc(_Base):
    def setUp(self) -> None:
        self.clock = Clock()
        super().setUp()

    def make_backend(self, rt: FakeRuntime | None = None, probe=None) -> LocalBackend:
        return LocalBackend(self.project, rt or self.rt, host=self.host, probe=probe, clock=self.clock,
                            sleep=self.clock.sleep, monotonic=self.clock.mono)

    def gpu_job(self, name: str = "train", est: float = 0.4, **kw):
        return self.job(name, gpu=True, gpu_hours_est=est, **kw)

    def of(self, jid: str, event: str) -> list[dict]:
        return [r for r in self.rows(jid) if r["event"] == event]

    def gpu_log(self) -> Path:
        return self.work / launch.GPU_LOG_MD


class TestAccounting(_Acc):
    def test_gpu_seconds_from_container_times(self) -> None:
        rt = FakeRuntime({IMAGE: _train}, duration_s=600)
        b = self.make_backend(rt)
        jid = b.submit(self.gpu_job())
        self.assertEqual(b.wait(jid).gpu_seconds, 600)
        self.assertEqual(b.gpu_seconds(), 600)
        cpu = b.submit(self.job("cpu"))
        s = b.wait(cpu)
        self.assertEqual((s.seconds, s.gpu_seconds), (600, 0))
        self.assertEqual(b.gpu_seconds(), 600)
        self.assertAlmostEqual(b.committed_gpu_hours(), 600 / 3600)

    def test_committed_counts_running_estimates(self) -> None:
        rt = FakeRuntime({IMAGE: _train}, outcomes=["hang"], duration_s=600)
        b = self.make_backend(rt)
        jid = b.submit(self.gpu_job(est=0.3))
        self.assertAlmostEqual(b.committed_gpu_hours(), 0.3)
        rt.finish(self.name(jid), 0)
        b.wait(jid)
        self.assertAlmostEqual(b.committed_gpu_hours(), 600 / 3600)

    def test_retry_after_preempt(self) -> None:
        rt = FakeRuntime({IMAGE: _train}, outcomes=["preempt", "ok"], duration_s=900)
        b = self.make_backend(rt)
        jid = b.submit(self.gpu_job())
        s = b.wait(jid)
        self.assertEqual((s.state, s.attempt, s.gpu_seconds), ("succeeded", 2, 1800))
        self.assertEqual(len(self.of(jid, "start")), 2)
        self.assertEqual(b.gpu_seconds(), 1800)

    def test_retry_stops_at_budget(self) -> None:
        """Review Focus 5, pre-flight finding 8: the job's own estimate counts once; the third start would pass the
        budget, so it is refused and the job ends budget_exhausted."""
        rt = FakeRuntime({IMAGE: _train}, outcomes=[1, 1, 1], duration_s=1800)
        b = self.make_backend(rt)
        jid = b.submit(self.gpu_job(est=0.4))
        seen = [b.committed_gpu_hours()]
        for _ in range(4):
            s = b.status(jid)
            seen.append(b.committed_gpu_hours())
        self.assertEqual((s.state, s.attempt), ("budget_exhausted", 2))
        self.assertEqual(len(rt.runs), 2)
        self.assertEqual(len(self.of(jid, "start")), 2)
        refused = self.of(jid, "refused")
        self.assertEqual([(r["attempt"], r["rule"]) for r in refused], [(3, "AN-23")])
        self.assertIn("budget", refused[0]["reason"])
        self.assertTrue(refused[0]["utc"])
        self.assertTrue(all(c <= 1.0 for c in seen), seen)
        self.assertEqual(b.gpu_seconds(), 3600)

    def test_timeout_not_retried(self) -> None:
        rt = FakeRuntime({IMAGE: _train}, outcomes=["hang", "ok"])
        b = self.make_backend(rt)
        jid = b.submit(self.job(time_limit_s=100))
        self.clock.at(99)
        self.assertEqual(b.status(jid).state, "running")
        self.clock.at(101)
        s = b.status(jid)
        self.assertEqual((s.state, s.outcome), ("timed_out", "timed_out"))
        self.assertEqual([(r["attempt"], r["reason"]) for r in self.of(jid, "stop")], [(1, "timed_out")])
        self.assertEqual(len(rt.runs), 1)
        self.assertEqual(b.status(jid).state, "timed_out")
        self.assertEqual(self.events(jid), ["start", "stop", "end"])

    def test_running_gpu_attempt_stopped_at_budget(self) -> None:
        """Pre-flight finding 8: a GPU attempt is stopped once it used the budget left when it started."""
        rt = FakeRuntime({IMAGE: _train}, outcomes=["ok", "hang"])
        b = self.make_backend(rt)
        other = b.submit(self.gpu_job("first", est=0.25))            # 60 s recorded
        b.wait(other)
        self.clock.at(60)                                             # the start row and the container agree
        jid = b.submit(self.gpu_job(est=0.4))                         # starts at +60 s; budget left 3540 s
        self.assertEqual(self.of(jid, "start")[0]["limit_s"], 3540)
        self.clock.at(60 + 3539)
        self.assertEqual(b.status(jid).state, "running")
        self.clock.at(60 + 3540)
        s = b.status(jid)
        self.assertEqual((s.state, s.outcome), ("budget_exhausted", "budget_exhausted"))
        self.assertEqual([r["reason"] for r in self.of(jid, "stop")], ["budget_exhausted"])
        self.assertEqual(len(rt.runs), 2)

    def test_stop_failure_retried_on_next_poll(self) -> None:
        rt = FakeRuntime({IMAGE: _train}, outcomes=["hang"])
        b = self.make_backend(rt)
        jid = b.submit(self.job(time_limit_s=10))
        self.clock.at(11)
        with mock.patch.object(rt, "stop", side_effect=be.RuntimeFailure("stop", 1)):
            with self.assertRaises(be.RuntimeFailure):
                b.status(jid)
        self.assertEqual(self.events(jid), ["start", "stop"])
        self.assertEqual(b.status(jid).state, "timed_out")
        self.assertEqual(self.events(jid), ["start", "stop", "end"])

    def test_oom_not_retried(self) -> None:
        rt = FakeRuntime({IMAGE: _train}, outcomes=["oom", "ok"])
        b = self.make_backend(rt)
        jid = b.submit(self.gpu_job())
        s = b.wait(jid)
        self.assertEqual((s.state, s.outcome, s.attempt), ("failed", "oom", 1))
        self.assertEqual(len(rt.runs), 1)

    def test_cancel_and_fail_not_retried_past_attempts(self) -> None:
        rt = FakeRuntime({IMAGE: _train}, outcomes=[1, 1, 1, "ok"])
        b = self.make_backend(rt)
        jid = b.submit(self.job(max_attempts=3))
        s = b.wait(jid)
        self.assertEqual((s.state, s.attempt), ("failed", 3))
        self.assertEqual(len(rt.runs), 3)

    def test_gpu_busy_refuses(self) -> None:
        """Review Focus 5: a GPU busy past GPU_WAIT_S refuses the start; no `start` row; the next poll starts it."""
        probe = Probe(2048)
        b = self.make_backend(probe=probe)
        with self.assertRaises(Refused) as cm:
            b.submit(self.gpu_job())
        p = cm.exception.problems[0]
        self.assertEqual(p.rule, "AN-23")
        self.assertIn("the GPU is busy", p.message)
        self.assertEqual(self.rt.runs, [])
        self.assertEqual(self.events("0001-train"), [])
        self.assertGreater(probe.calls, 2)
        self.assertGreaterEqual(self.clock.now - START, dt.timedelta(seconds=800))
        self.assertEqual(b.jobs()[0].state, "queued")
        probe.readings = [GPU_BUSY_MIB]
        self.assertEqual(b.wait("0001-train").state, "succeeded")
        self.assertEqual(self.events("0001-train"), ["start", "end"])

    def test_no_gpu_tool_refuses_gpu_job_only(self) -> None:
        b = self.make_backend(probe=Probe(None))
        with self.assertRaises(Refused) as cm:
            b.submit(self.gpu_job())
        self.assertEqual(cm.exception.problems[0].rule, "AN-23")
        cpu = b.submit(self.job("cpu"))                               # a CPU job does not wait for the GPU
        self.assertEqual(b.wait(cpu).state, "succeeded")

    def test_busy_gpu_on_retry(self) -> None:
        rt = FakeRuntime({IMAGE: _train}, outcomes=["preempt", "ok"])
        probe = Probe(0)
        b = self.make_backend(rt, probe=probe)
        jid = b.submit(self.gpu_job())
        probe.readings = [4096]
        with self.assertRaises(Refused):
            b.status(jid)
        self.assertEqual(self.events(jid), ["start", "end"])
        self.assertEqual(b.jobs()[0].state, "queued")
        probe.readings = [0]
        self.assertEqual(b.wait(jid).state, "succeeded")

    def test_submit_over_budget_refused(self) -> None:
        with self.assertRaises(Refused) as cm:
            self.backend.submit(self.gpu_job(est=1.5))
        p = cm.exception.problems[0]
        self.assertEqual((p.rule, p.field), ("AN-23", "job.gpu_hours_est"))
        self.assertEqual(self.rt.runs, [])
        self.assertEqual([p.name for p in (self.project.root / "jobs").iterdir() if p.name[0].isdigit()], [])
        rt = FakeRuntime({IMAGE: _train}, outcomes=["hang"])
        b = self.make_backend(rt)
        b.submit(self.gpu_job("a", est=0.6))
        with self.assertRaises(Refused) as cm:                        # the running job's estimate is committed
            b.submit(self.gpu_job("b", est=0.6))
        self.assertEqual(cm.exception.problems[0].rule, "AN-23")
        self.assertEqual(len(rt.runs), 1)
        b.submit(self.gpu_job("c", est=0.4))                          # exactly the budget: allowed
        self.assertEqual(len(rt.runs), 2)

    def test_first_start_refused_when_budget_used_meanwhile(self) -> None:
        """A queued job (its first start refused: the GPU was busy) whose start the budget no longer covers, since
        another job ran longer than its estimate: refused under the lock, the job ends budget_exhausted."""
        rt = FakeRuntime({IMAGE: _train}, outcomes=["hang", "ok"], duration_s=3000)
        x = self.make_backend(rt)
        xid = x.submit(self.gpu_job("x", est=0.3))
        probe = Probe(4096)
        b = self.make_backend(rt, probe=probe)
        with self.assertRaises(Refused):
            b.submit(self.gpu_job("a", est=0.6))                      # 0.3 + 0.6 fits; the GPU is busy
        rt.finish(self.name(xid), 0)
        self.assertEqual(x.wait(xid).gpu_seconds, 3000)
        probe.readings = [0]
        s = b.status("0002-a")
        self.assertEqual((s.state, s.attempt), ("budget_exhausted", 0))
        self.assertEqual([(r["attempt"], r["rule"]) for r in self.of("0002-a", "refused")], [(1, "AN-23")])
        self.assertEqual(len(rt.runs), 1)

    def test_bad_budget_refused(self) -> None:
        self.project.budget["gpu_hours"] = "lots"
        with self.assertRaises(Refused) as cm:
            self.backend.submit(self.gpu_job())
        self.assertEqual((cm.exception.problems[0].rule, cm.exception.problems[0].field), ("AN-23", "budget.gpu_hours"))


class ClockedRuntime(FakeRuntime):
    """A fake runtime whose `stop` ends the container at the test clock's time (its run time is then real)."""

    def __init__(self, clock: Clock, *args, **kw) -> None:
        super().__init__(*args, **kw)
        self.clock = clock

    def stop(self, container: str, *, timeout_s: int = 30) -> None:
        with self._state() as st:
            c = self._live(st, "stop", container)
            if c["status"] == "running":
                c.update(status="exited", exit_code=RC_STOPPED, finished_utc=self.clock().strftime(be.TIME_FORMAT))


class TestRoundOne(_Acc):
    def test_new_driver_stops_attempt_past_limit(self) -> None:
        """A driver crash: a new backend that never submitted the job still stops the attempt at its limit."""
        rt = FakeRuntime({IMAGE: _train}, outcomes=["hang"])
        jid = self.make_backend(rt).submit(self.job(time_limit_s=100))
        self.clock.at(101)
        s = self.make_backend(rt).status(jid)
        self.assertEqual((s.state, s.outcome), ("timed_out", "timed_out"))
        self.assertEqual(self.events(jid), ["start", "stop", "end"])

    def test_lost_attempt_counts_its_time(self) -> None:
        rt = FakeRuntime({IMAGE: _train}, outcomes=["vanish", "vanish"])
        b = self.make_backend(rt)
        jid = b.submit(self.gpu_job(est=0.4, max_attempts=2))
        self.clock.at(500)
        s = b.status(jid)                                             # a1 lost after 500 s; a2 started at +500 s
        ends = self.of(jid, "end")
        self.assertEqual([(e["outcome"], e["seconds"], e["gpu_seconds"]) for e in ends], [("lost", 500, 500)])
        self.assertEqual(s.state, "running")
        limit = self.of(jid, "start")[1]["limit_s"]
        self.assertEqual(limit, 3600 - 500)
        self.clock.at(500 + 99999)
        s = b.status(jid)
        self.assertEqual(s.state, "failed")
        self.assertEqual(self.of(jid, "end")[1]["gpu_seconds"], limit)   # at most its limit
        self.assertLessEqual(b.gpu_seconds(), 3600)

    def test_concurrent_gpu_jobs_stay_within_budget(self) -> None:
        """Review round 1, Important 3: two GPU jobs (0.4 h each) run at once and longer than estimated; each poll
        recomputes the budget left from the recorded time and both attempts' run times, so together they overrun the
        1 h budget by at most one poll interval each."""
        rt = ClockedRuntime(self.clock, {IMAGE: _train}, outcomes=["hang", "hang"], duration_s=0)
        b = self.make_backend(rt)
        ids = [b.submit(self.gpu_job("a", est=0.4)), b.submit(self.gpu_job("b", est=0.4))]
        poll = 70
        t = 0
        while any(b.status(j).state == "running" for j in ids):
            t += poll
            self.clock.at(t)
            self.assertLess(t, 4000)
        states = [b.status(j) for j in ids]
        self.assertEqual([s.state for s in states], ["budget_exhausted", "budget_exhausted"])
        self.assertLessEqual(b.gpu_seconds(), 3600 + 2 * poll)
        self.assertGreaterEqual(b.gpu_seconds(), 3600)

    def test_missing_log_row_added_by_later_status(self) -> None:
        """Minor 4: a crash cut off the log row; the next status adds it, once."""
        old = be.GPU_LOG_HEADER + "| old | a | b | 1.00 | 1.00 | PASS rc=0 |\n"
        self.gpu_log().write_text(old)
        jid = self.backend.submit(self.job())
        self.assertEqual(self.backend.wait(jid).state, "succeeded")
        self.gpu_log().write_text(old)                                # the row was lost
        self.backend.status(jid)
        self.backend.status(jid)
        lines = self.gpu_log().read_text().splitlines()
        self.assertEqual([l for l in lines if l.startswith(f"| {jid} (")], [lines[-1]])
        self.assertEqual(len(lines), 4)

    def test_inputs_changed_stops_cut_off_container(self) -> None:
        """Minor 8: a container of the due attempt runs (its `start` row was cut off); it is stopped and recorded
        before the refusal."""
        rt = FakeRuntime({IMAGE: _train}, outcomes=["preempt", "hang"])
        b = self.make_backend(rt)
        jid = b.submit(self.job())
        rt.run(dataclasses.replace(rt.runs[0], name=self.name(jid, 2)))
        (self.inputs / SCRIPT).write_text("print('changed')\n")
        with self.assertRaises(Refused):
            b.status(jid)
        self.assertEqual(self.events(jid), ["start", "end", "start", "end", "refused"])
        self.assertEqual(rt.inspect(self.name(jid, 2)).status, "missing")
        self.assertEqual(self.of(jid, "refused")[0]["attempt"], 3)
        self.assertEqual(b.status(jid).state, "failed")


class TestGpuLog(_Acc):
    def test_log_ending_in_notes(self) -> None:
        """Important 1: the documented log (tests/fixtures/ext/gpu-log.fixture.md) ends in notes; a new table is
        appended and its total counts."""
        fixture = Path(__file__).parent / "fixtures" / "ext" / "gpu-log.fixture.md"
        shutil.copyfile(fixture, self.gpu_log())
        before_text = self.gpu_log().read_text()
        before = launch.gpu_used_h(self.gpu_log())
        rt = FakeRuntime({IMAGE: _train}, duration_s=1800)
        b = self.make_backend(rt)
        jid = b.submit(self.gpu_job())
        b.wait(jid)
        text = self.gpu_log().read_text()
        self.assertTrue(text.startswith(before_text))
        self.assertAlmostEqual(launch.gpu_used_h(self.gpu_log()), round(before + 0.5, 2))
        self.assertEqual(text.splitlines()[-3], be.GPU_LOG_HEADER.splitlines()[0])

    def test_log_with_trailing_blank_line(self) -> None:
        self.gpu_log().write_text("| Step | Started (UTC) | Ended (UTC) | GPU-hours | Running total | Outcome |\n"
                                  "|---|---|---|---|---|---|\n| old | a | b | 2.00 | 2.00 | PASS rc=0 |\n\n")
        b = self.make_backend(FakeRuntime({IMAGE: _train}, duration_s=1800))
        jid = b.submit(self.gpu_job())
        b.wait(jid)
        self.assertEqual(launch.gpu_used_h(self.gpu_log()), 2.5)
        b2 = b.submit(self.job("cpu"))
        b.wait(b2)
        self.assertEqual(launch.gpu_used_h(self.gpu_log()), 2.5)
        self.assertEqual(self.gpu_log().read_text().count("| Step |"), 2)  # the second row joined the new table

    def test_gpu_log_row(self) -> None:
        """Pre-flight finding 7: an absent log is created with the header and the first row at once (total from 0);
        later rows add to the last running total; CPU jobs get a row too."""
        self.assertFalse(self.gpu_log().exists())
        rt = FakeRuntime({IMAGE: _train}, outcomes=["preempt", "ok"], duration_s=900)
        b = self.make_backend(rt)
        jid = b.submit(self.gpu_job())
        b.wait(jid)
        text = self.gpu_log().read_text()
        lines = text.splitlines()
        self.assertEqual(lines[0], "| Step | Started (UTC) | Ended (UTC) | GPU-hours | Running total | Outcome |")
        self.assertEqual(lines[1], "|---|---|---|---|---|---|")
        self.assertEqual(lines[2], f"| {jid} (train), 2 attempt(s) | 2026-11-22T00:00:00Z | 2026-11-22T00:30:00Z "
                                   "| 0.50 | 0.50 | SUCCEEDED rc=0 |")
        self.assertEqual(launch.gpu_used_h(self.gpu_log()), 0.5)
        cpu = b.submit(self.job("cpu", max_attempts=1), stage="s")
        rt2 = b.wait(cpu)
        self.assertEqual(rt2.state, "succeeded")
        last = self.gpu_log().read_text().splitlines()[-1]
        self.assertTrue(last.startswith(f"| {cpu} (cpu), 1 attempt(s) |"), last)
        self.assertIn("| 0.00 | 0.50 | SUCCEEDED rc=0 |", last)
        self.assertEqual(len(self.gpu_log().read_text().splitlines()), 4)
        self.assertEqual(b.status(cpu).state, "succeeded")            # a later poll appends nothing
        self.assertEqual(len(self.gpu_log().read_text().splitlines()), 4)

    def test_gpu_log_existing_total(self) -> None:
        self.gpu_log().write_text("# GPU log\n\n| Step | Started (UTC) | Ended (UTC) | GPU-hours | Running total | "
                                  "Outcome |\n|---|---|---|---|---|---|\n| old | a | b | 2.00 | 2.00 | PASS rc=0 |\n")
        rt = FakeRuntime({IMAGE: _train}, outcomes=[3], duration_s=1800)
        b = self.make_backend(rt)
        jid = b.submit(self.gpu_job(max_attempts=1))
        b.wait(jid)
        self.assertEqual(launch.gpu_used_h(self.gpu_log()), 2.5)
        self.assertIn("| 0.50 | 2.50 | FAILED rc=3 |", self.gpu_log().read_text().splitlines()[-1])

    def test_gpu_log_terminal_without_attempts(self) -> None:
        b = self.make_backend(FakeRuntime({IMAGE: _train}))

        class NoStart(LocalBackend):
            def _advance(self_, job_id, rec, **kw):
                return self_._view(job_id, rec).status
        jid = NoStart(self.project, self.rt, host=self.host, clock=self.clock).submit(self.job())
        self.assertEqual(b.cancel(jid).state, "cancelled")
        self.assertEqual(self.gpu_log().read_text().splitlines()[-1],
                         f"| {jid} (train), 0 attempt(s) | - | - | 0.00 | 0.00 | CANCELLED rc=- |")

    def test_gpu_log_unparsable_or_link_refused(self) -> None:
        self.gpu_log().write_text("no table here\n")
        with self.assertRaises(Refused) as cm:
            self.backend.submit(self.job())
        self.assertEqual((cm.exception.problems[0].rule, cm.exception.problems[0].field), ("AN-23", "gpu_log"))
        self.assertEqual(self.rt.runs, [])
        self.assertEqual(self.gpu_log().read_text(), "no table here\n")       # never reset
        self.gpu_log().unlink()
        target = self.tmp / "elsewhere.md"
        os.symlink(target, self.gpu_log())
        with self.assertRaises(Refused) as cm:
            self.backend.submit(self.job())
        self.assertEqual(cm.exception.problems[0].field, "gpu_log")
        self.assertFalse(target.exists())

    def test_end_recorded_once(self) -> None:
        """Review Focus 1, pre-flight finding 16: two drivers poll at once; the fake runtime's inspect holds the first
        until the second arrives (or a timeout, when the job lock keeps the second out): one `end`, one retry."""
        rt = FakeRuntime({IMAGE: _train}, outcomes=["preempt", "hang"])
        job = self.gpu_job()
        a, b = self.make_backend(rt), self.make_backend(rt)
        jid = a.submit(job)
        self.assertEqual(b.submit(job), jid)
        barrier = threading.Barrier(2, timeout=1.0)
        inspect = rt.inspect

        def gated(name):
            if name == self.name(jid, 1):
                try:
                    barrier.wait()
                except threading.BrokenBarrierError:
                    pass
            return inspect(name)
        errors = []

        def poll(backend):
            try:
                backend.status(jid)
            except Exception as exc:                                  # pragma: no cover - reported below
                errors.append(exc)
        with mock.patch.object(rt, "inspect", gated):
            threads = [threading.Thread(target=poll, args=(x,)) for x in (a, b)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(errors, [])
        self.assertEqual(self.events(jid), ["start", "end", "start"])
        self.assertEqual(len(rt.runs), 2)


class TestCarries(_Acc):
    def test_rerun_after_failure_gets_new_output_dir(self) -> None:
        rt = FakeRuntime({IMAGE: _train}, outcomes=[2])
        b = self.make_backend(rt)
        first = self.job(max_attempts=1)
        jid = b.submit(first)
        self.assertEqual(b.wait(jid).state, "failed")
        again = self.job(max_attempts=1)
        self.assertEqual(again.key(), first.key())
        self.assertNotEqual(again.output_dir, first.output_dir)
        self.assertEqual(Path(again.output_dir).name, "train.2")
        jid2 = b.submit(again)
        self.assertNotEqual(jid2, jid)
        self.assertEqual(b.wait(jid2).state, "succeeded")
        self.assertEqual(self.job(max_attempts=1).output_dir, again.output_dir)   # succeeded: kept

    def test_inputs_changed_before_retry(self) -> None:
        rt = FakeRuntime({IMAGE: _train}, outcomes=["preempt"])
        b = self.make_backend(rt)
        jid = b.submit(self.job())
        (self.inputs / SCRIPT).write_text("print('changed')\n")
        with self.assertRaises(Refused) as cm:
            b.status(jid)
        self.assertIn("the inputs changed since the job was recorded", cm.exception.problems[0].message)
        self.assertEqual(b.status(jid).state, "failed")
        self.assertEqual([(r["attempt"], r["rule"]) for r in self.of(jid, "refused")], [(2, "AN-22")])
        self.assertEqual(len(rt.runs), 1)
        self.assertEqual(len(self.gpu_log().read_text().splitlines()), 3)

    def test_key_computed_once_per_submit(self) -> None:
        job = self.job()
        with mock.patch.object(be, "input_fingerprint", wraps=be.input_fingerprint) as fp:
            self.backend.submit(job)
        self.assertEqual(fp.call_count, 1)

    def test_readlink_race_and_fifo_refused(self) -> None:
        os.symlink("x", self.inputs / "link")
        with mock.patch.object(be.os, "readlink", side_effect=FileNotFoundError()), self.assertRaises(Refused) as cm:
            self.job().key()
        self.assertEqual((cm.exception.problems[0].rule, cm.exception.problems[0].field), ("AN-22", "job.mounts"))
        fifo = self.tmp / "fifo"
        os.mkfifo(fifo)
        with self.assertRaises(Refused):                              # never blocks on a FIFO
            be._sha256_nofollow(str(fifo))

    def test_fingerprint_stops_at_bound(self) -> None:
        big = self.inputs / "many"
        big.mkdir()
        for i in range(50):
            (big / str(i)).write_text("")
        seen = []
        real = be.os.scandir

        def counting(path):
            it = real(path)

            class Wrap:
                def __enter__(self_):
                    return self_

                def __exit__(self_, *a):
                    it.close()

                def __iter__(self_):
                    for e in it:
                        seen.append(e.name)
                        yield e
            return Wrap()
        with mock.patch.object(be, "MAX_INPUT_ENTRIES", 5), mock.patch.object(be.os, "scandir", counting), \
                self.assertRaises(Refused):
            be.input_fingerprint(self.job(output_dir="/x").mounts)
        self.assertLess(len(seen), 10)


if __name__ == "__main__":
    import unittest
    unittest.main()
