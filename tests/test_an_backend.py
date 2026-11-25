"""AN-22: the local backend (design revision 25, 05 §18.10; tests/README.md section 14).

No test starts Docker: the backend runs on `FakeRuntime`, with the stand-in ext adapter (`gatedtrain.testing.standin`)
and a host layout in a temporary directory (`GATEDTRAIN_EXT_*` and a host file). The owner's primary group is injected
(`HostConfig.primary_gid`), so no result depends on the host's Docker group.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gatedtrain.ainative import backend as be
from gatedtrain.ainative.backend import Job, LocalBackend, job_problems
from gatedtrain.ainative.project import append_jsonl, create_project, read_jsonl
from gatedtrain.ainative.runtime import Mount, RuntimeFailure
from gatedtrain.errors import Refused
from gatedtrain.ext import launch, registry
from gatedtrain.testing import standin
from gatedtrain.testing.fake_runtime import FakeRuntime

IMAGE = "sha256:" + "ab" * 32
FAKE_GID = 2**31 - 3                     # the owner's primary group: no Docker socket's group on any host
SCRIPT = "synthetic_train_job.py"
SECRET = "s3cr3t-value"
CLOCK = datetime.datetime(2026, 11, 22, tzinfo=datetime.timezone.utc)


def _train(spec, host_path) -> int:
    """The fake container: writes a model directory into /out and prints a line."""
    out = host_path("/out")
    (out / "model").mkdir()
    (out / "model" / "model.safetensors").write_bytes(b"weights")
    (out / "model" / "config.json").write_text("{}")
    for p in (out / "model", out / "model" / "model.safetensors", out / "model" / "config.json"):
        p.chmod(0o755 if p.is_dir() else 0o644)
    print("trained", spec.name)
    return 0


_PATCHES: list = []


def setUpModule() -> None:
    """The stand-in adapter is registered for this module only: registry-wide tests elsewhere (the docs tables) list
    the recipe adapters."""
    from gatedtrain import spec

    for table in (registry.ADAPTERS, registry.TRAINER_ROWS, registry.EVALUATOR_ROWS, spec.TRAINER_BACKENDS):
        _PATCHES.append(mock.patch.dict(table))
        _PATCHES[-1].start()
    standin.install()


def tearDownModule() -> None:
    while _PATCHES:
        _PATCHES.pop().stop()


class _Base(unittest.TestCase):
    ROOTLESS = True

    def setUp(self) -> None:
        self.tmp = Path(os.path.realpath(tempfile.mkdtemp()))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.work = self.tmp / "work"
        self.out_root = self.tmp / "out"
        for d in (self.work, self.out_root):
            d.mkdir()
            d.chmod(0o755)
        host_file = self.tmp / "host.toml"
        host_file.write_text("rootless = true\n" if self.ROOTLESS else "")
        env = {k: v for k, v in os.environ.items() if not k.startswith(launch.ENV_PREFIX)}
        env.update({launch.ENV_HOST_WORK: str(self.work), launch.ENV_OUTPUT_ROOT: str(self.out_root),
                    launch.ENV_HOST_CONFIG: str(host_file)})
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.host = dataclasses.replace(launch.host_config(), primary_gid=lambda uid: FAKE_GID)
        data = self.tmp / "data.txt"
        data.write_text("data")
        self.project = create_project(self.tmp / "proj", goal="g", data=data, pipeline="p",
                                      budget={"gpu_hours": 1.0, "llm_usd": 0})
        self.rt = FakeRuntime({IMAGE: _train})
        self.backend = self.make_backend()
        self.inputs = self.backend.new_output_dir("inputs")
        (self.inputs / SCRIPT).write_text("print('train')\n")

    def make_backend(self, rt: FakeRuntime | None = None) -> LocalBackend:
        """The clock is fixed at the fake runtime's first start, so no attempt reaches its time limit (AN-23)."""
        return LocalBackend(self.project, rt or self.rt, host=self.host, sleep=lambda s: None, clock=lambda: CLOCK)

    def job(self, name: str = "train", **kw) -> Job:
        """A job of the stand-in adapter; without `output_dir`, the directory `new_output_dir` keeps for its key."""
        out = kw.pop("output_dir", None)
        base = dict(label=name, adapter=standin.NAME, runtime={"kind": "container", "image": IMAGE},
                    framework={"name": standin.FRAMEWORK_PACKAGE, "version": "0.1", "commit": None},
                    argv=(standin.IMAGE_PYTHON, f"/inputs/code/{SCRIPT}", "--out", "/out"),
                    env={"HF_HUB_OFFLINE": "1", "HOME": f"/tmp/{SECRET}"},
                    mounts=(Mount(str(self.inputs), "/inputs/code"),), output_dir=out or "", output_container="/out",
                    tmpfs=("/tmp",))
        base.update(kw)
        job = Job(**base)
        if out is None:
            job = dataclasses.replace(job, output_dir=str(self.backend.new_output_dir(name, key=job.key())))
        return job

    def rows(self, jid: str) -> list[dict]:
        return read_jsonl(self.project.root, f"jobs/{jid}/attempts.jsonl")

    def events(self, jid: str) -> list[str]:
        """The attempt events, without the GPU log's bookkeeping row (`gpu_log_written`)."""
        return [r["event"] for r in self.rows(jid) if r["event"] != "gpu_log_written"]

    def name(self, jid: str, attempt: int = 1) -> str:
        return f"gatedtrain-{self.backend.project_hex()[:12]}-{jid}-a{attempt}"


class TestSubmit(_Base):
    def test_submit_runs_with_shared_rules(self) -> None:
        jid = self.backend.submit(self.job(), stage="train")
        self.assertEqual(jid, "0001-train")
        spec = self.rt.runs[0]
        self.assertEqual(spec.user, "0:0")
        self.assertEqual(spec.network, "none")
        self.assertEqual(spec.gpu_options, ())
        self.assertEqual(spec.name, self.name(jid))
        self.assertEqual(dict(spec.labels), {"gatedtrain.job": jid, "gatedtrain.project": self.backend.project_hex()})
        self.assertEqual(spec.mounts[-1], Mount(os.path.join(self.job().output_dir, "a1"), "/out", read_only=False))
        self.assertEqual(spec.tmpfs, (("/tmp", launch.TMPFS_OPTIONS),))
        self.assertEqual(self.backend.wait(jid).state, "succeeded")
        rec = json.loads((self.project.root / "jobs" / jid / "job.json").read_text())
        self.assertEqual((rec["schema"], rec["schema_version"], rec["stage"], rec["rootless"]),
                         (be.JOB_SCHEMA, 1, "train", True))
        self.assertEqual(rec["key"], self.job().key())
        self.assertEqual(rec["host_config_sha256"], launch.host_config_sha256(launch.adapter_config(self.host,
                                                                                                  standin.NAME)))
        script_sha = "sha256:" + hashlib.sha256((self.inputs / SCRIPT).read_bytes()).hexdigest()
        self.assertEqual((rec["script_sha256"], rec["interpreter_sha256"]), (script_sha, None))
        self.assertIn("--pull", rec["docker_argv"])
        self.assertNotIn("--rm", rec["docker_argv"])

    def test_gpu_job_gets_cdi(self) -> None:
        jid = self.backend.submit(self.job(gpu=True, gpu_hours_est=0.5))
        self.assertEqual(self.rt.runs[0].gpu_options, launch.CDI_GPU_OPTIONS)
        self.backend.wait(jid)
        end = [r for r in self.rows(jid) if r["event"] == "end"][0]
        self.assertEqual((end["seconds"], end["gpu_seconds"]), (60, 60))

    def test_rule_violations_refused_before_start(self) -> None:
        full = self.backend.new_output_dir("full")
        (full / "stale.txt").write_text("x")
        bad_jobs = {
            "rw input mount": self.job("a", mounts=(Mount(str(self.inputs), "/inputs/code", read_only=False),)),
            "host network": self.job("b", network="host"),
            "credential env name": self.job("c", env={"HF_HUB_OFFLINE": "1", "HF_TOKEN": "x"}),
            "unpinned image": self.job("d", runtime={"kind": "container", "image": "python:3.12"}),
            "output not empty": self.job("e", output_dir=str(full)),
            "mount of host_work": self.job("f", mounts=(Mount(str(self.inputs), "/inputs/code"),
                                                        Mount(str(self.work), "/inputs/work"))),
            "unknown adapter": self.job("g", adapter="no-such-adapter"),
            "gpu without estimate": self.job("h", gpu=True, gpu_hours_est=0.0),
            "no attempts": self.job("i", max_attempts=0),
            "no time": self.job("j", time_limit_s=0),
            "long label": self.job("k", label="x" * 65),
        }
        for what, bad in bad_jobs.items():
            with self.subTest(what), self.assertRaises(Refused) as cm:
                self.backend.submit(bad)
            self.assertEqual(cm.exception.problems[0].rule, "AN-22")
            self.assertTrue(cm.exception.problems[0].field.startswith("job"), cm.exception.problems[0].field)
        self.assertEqual(self.rt.runs, [])
        self.assertEqual([p.name for p in (self.project.root / "jobs").iterdir() if p.name[0].isdigit()], [])
        msg = str(cm.exception)
        self.assertNotIn(SECRET, msg)

    def test_daemon_not_rootless_or_foreign_socket_refused(self) -> None:
        for rt in (FakeRuntime({IMAGE: _train}, rootless=False),
                   FakeRuntime({IMAGE: _train}, socket_uid=os.getuid() + 1, socket="/run/user/x/docker.sock")):
            with self.subTest(daemon=rt.daemon()), self.assertRaises(Refused) as cm:
                self.make_backend(rt).submit(self.job())
            self.assertEqual((cm.exception.problems[0].rule, cm.exception.problems[0].field), ("AN-22", "daemon"))
            self.assertEqual(rt.runs, [])

    def test_image_not_held_refused(self) -> None:
        rt = FakeRuntime({IMAGE: _train}, image_ids={})
        with self.assertRaises(Refused) as cm:
            self.make_backend(rt).submit(self.job())
        self.assertEqual(cm.exception.problems[0].field, "job.runtime.image")
        self.assertEqual(rt.runs, [])

    def test_argv_rules_and_run_options(self) -> None:
        """The adapter's argv rules refuse; its run options reach the container."""
        row = {"run_options": {SCRIPT: ("--ulimit", "memlock=-1", "--shm-size=16g")},
               "tmpfs_options": {SCRIPT: "rw,size=1g"}}
        calls = []

        def argv_problems(self_, argv, *, profile):
            calls.append(profile)
            return ["--max-steps is required"] if profile == "full" else []
        with mock.patch.dict(registry.TRAINER_ROWS[standin.NAME], row), \
                mock.patch.object(standin.StandinAdapter, "argv_problems", argv_problems, create=True):
            with self.assertRaises(Refused) as cm:
                self.backend.submit(self.job())
            self.assertEqual([(p.field, p.message) for p in cm.exception.problems],
                             [("job.argv", "--max-steps is required")])
            jid = self.backend.submit(self.job(profile="smoke"))
        self.assertEqual(calls, ["full", "smoke"])
        spec = self.rt.runs[0]
        self.assertEqual((spec.extra_options, spec.shm_size, spec.tmpfs),
                         (("--ulimit", "memlock=-1"), "16g", (("/tmp", "rw,size=1g"),)))
        self.assertEqual(self.backend.status(jid).state, "succeeded")


class TestRootful(_Base):
    ROOTLESS = False

    def test_rootful_host_uses_owner_user(self) -> None:
        rt = FakeRuntime({IMAGE: _train}, rootless=False, socket_uid=0)     # no daemon check on a rootful host
        jid = self.make_backend(rt).submit(self.job(gpu=True, gpu_hours_est=0.5))
        spec = rt.runs[0]
        self.assertEqual(spec.user, f"{os.getuid()}:{FAKE_GID}")
        self.assertEqual(spec.gpu_options, launch.ROOTFUL_GPU_OPTIONS)
        rec = json.loads((self.project.root / "jobs" / jid / "job.json").read_text())
        self.assertFalse(rec["rootless"])


class TestLifecycle(_Base):
    def test_resubmit_reattaches(self) -> None:
        """The driver dies while the job runs and its output is not empty; the
        stage re-run in a new process gets the same job and starts nothing."""
        rt = FakeRuntime({IMAGE: _train}, outcomes=["hang"])
        job = self.job()
        jid = self.make_backend(rt).submit(job)
        (Path(job.output_dir) / "a1" / "partial.bin").write_bytes(b"x")       # the running container wrote
        again = self.make_backend(rt)                                          # a new driver process
        self.assertEqual(again.submit(job), jid)
        self.assertEqual(len(rt.runs), 1)
        self.assertEqual(again.status(jid).state, "running")
        rt.finish(self.name(jid), 0)
        self.assertEqual(again.wait(jid).state, "succeeded")
        self.assertEqual(again.submit(job), jid)                               # succeeded: still reattaches
        self.assertEqual(len(rt.runs), 1)
        self.assertEqual(self.events(jid), ["start", "end"])

    def test_reattach_rechecks_rules(self) -> None:
        rt = FakeRuntime({IMAGE: _train}, outcomes=["hang"])
        job = self.job()
        jid = self.make_backend(rt).submit(job)
        names = ("HF_HUB_OFFLINE",)                                            # the adapter no longer allows HOME
        with mock.patch.dict(registry.TRAINER_ROWS[standin.NAME], {"env_names": names, "env_paths": ()}):
            with self.assertRaises(Refused) as cm:
                self.make_backend(rt).submit(job)
        self.assertEqual([(p.rule, p.field) for p in cm.exception.problems], [("AN-22", "job.env.HOME")])
        self.assertEqual(len(rt.runs), 1)
        self.assertEqual(self.make_backend(rt).status(jid).state, "running")

    def test_env_values_never_recorded(self) -> None:
        jid = self.backend.submit(self.job())
        self.backend.wait(jid)
        self.backend.artifacts(jid)
        for p in (self.project.root / "jobs").rglob("*"):
            if p.is_file():
                self.assertNotIn(SECRET.encode(), p.read_bytes(), p)
        rec = json.loads((self.project.root / "jobs" / jid / "job.json").read_text())
        self.assertEqual(rec["job"]["env"], ["HF_HUB_OFFLINE", "HOME"])
        argv = rec["docker_argv"]
        self.assertEqual([argv[i + 1] for i, t in enumerate(argv) if t == "-e"], ["HF_HUB_OFFLINE", "HOME"])

    def test_status_records_end_once_and_removes(self) -> None:
        jid = self.backend.submit(self.job())
        first = self.backend.status(jid)
        second = self.make_backend().status(jid)
        self.assertEqual(first, second)
        self.assertEqual((first.state, first.attempt, first.rc, first.outcome, first.seconds, first.gpu_seconds),
                         ("succeeded", 1, 0, "ok", 60, 0))
        self.assertEqual(self.events(jid), ["start", "end"])
        end = self.rows(jid)[1]
        self.assertEqual((end["started_utc"], end["finished_utc"]), ("2026-11-22T00:00:00Z", "2026-11-22T00:01:00Z"))
        self.assertEqual(end["schema"], be.JOB_ATTEMPT_SCHEMA)
        self.assertEqual(self.rt.inspect(self.name(jid)).status, "missing")
        log = (self.project.root / "jobs" / jid / "attempt-1.log").read_text()
        self.assertEqual(log, f"trained {self.name(jid)}\n")
        self.assertEqual(self.backend.logs(jid), log)
        self.assertEqual(self.backend.logs(jid, tail=0), "")

    def test_outcome_classification(self) -> None:
        cases = [("ok", "succeeded", "ok", 0), (2, "failed", "failed", 2), ("preempt", "failed", "preempted", 137),
                 ("oom", "failed", "oom", 137), ("vanish", "failed", "lost", None)]
        for i, (given, state, outcome, rc) in enumerate(cases):
            with self.subTest(given=given):
                rt = FakeRuntime({IMAGE: _train}, outcomes=[given])
                b = self.make_backend(rt)
                jid = b.submit(self.job(f"c{i}", max_attempts=1))
                s = b.wait(jid)
                self.assertEqual((s.state, s.outcome, s.rc), (state, outcome, rc))
        rt = FakeRuntime({IMAGE: _train}, outcomes=["hang"])
        b = self.make_backend(rt)
        jid = b.submit(self.job("cancel"))
        self.assertEqual(b.status(jid).state, "running")
        s = b.cancel(jid)
        self.assertEqual((s.state, s.outcome, s.rc), ("cancelled", "cancelled", 143))
        self.assertEqual(self.events(jid), ["start", "cancel_requested", "end"])
        self.assertEqual(b.cancel(jid), s)                                     # a terminal job: nothing more

    def test_stop_reason_classifies(self) -> None:
        rt = FakeRuntime({IMAGE: _train}, outcomes=["hang"])
        b = self.make_backend(rt)
        jid = b.submit(self.job())
        append_jsonl(self.project.root, f"jobs/{jid}/attempts.jsonl",
                     {"event": "stop", "attempt": 1, "reason": "timed_out", "utc": "2026-11-22T06:00:00Z"})
        rt.stop(self.name(jid))
        s = b.status(jid)
        self.assertEqual((s.state, s.outcome), ("timed_out", "timed_out"))
        self.assertEqual(len(rt.runs), 1)

    def test_retry_uses_fresh_attempt_dir(self) -> None:
        """Each attempt writes into its own directory; artifacts read the succeeded one."""
        rt = FakeRuntime({IMAGE: _train}, outcomes=["preempt", "ok"])
        b = self.make_backend(rt)
        job = self.job()
        jid = b.submit(job)
        s = b.wait(jid)
        self.assertEqual((s.state, s.attempt), ("succeeded", 2))
        self.assertEqual([r.mounts[-1].host for r in rt.runs],
                         [os.path.join(job.output_dir, "a1"), os.path.join(job.output_dir, "a2")])
        self.assertEqual(b.artifacts(jid)["output_dir"], os.path.join(job.output_dir, "a2"))
        self.assertEqual(self.events(jid), ["start", "end", "start", "end"])

    def test_retry_needs_the_job(self) -> None:
        """Environment values are never recorded: a process that did not submit the job does not start it."""
        rt = FakeRuntime({IMAGE: _train}, outcomes=["preempt"])
        jid = self.make_backend(rt).submit(self.job())
        with self.assertRaises(Refused) as cm:
            self.make_backend(rt).status(jid)
        self.assertIn("submit the job again", cm.exception.problems[0].fix)
        self.assertEqual(self.events(jid), ["start", "end"])
        again = self.make_backend(rt)
        self.assertEqual(again.submit(self.job()), jid)
        self.assertEqual(again.wait(jid).state, "succeeded")

    def test_daemon_error_leaves_state(self) -> None:
        rt = FakeRuntime({IMAGE: _train}, outcomes=["unreachable"])
        b = self.make_backend(rt)
        jid = b.submit(self.job())
        for op in (lambda: b.status(jid), lambda: b.logs(jid), lambda: b.cancel(jid)):
            with self.assertRaises(RuntimeFailure):
                op()
        self.assertEqual(self.events(jid), ["start", "cancel_requested"])
        self.assertEqual(b.jobs()[0].state, "running")
        rt.reachable(self.name(jid))
        rt.finish(self.name(jid), 0)
        self.assertEqual(b.status(jid).state, "succeeded")                    # rc 0 wins over the cancel request

    def test_failed_start_removes_leftover(self) -> None:
        class CreatedThenFails(FakeRuntime):
            fail = True

            def run(self, spec):
                if not self.fail:
                    return super().run(spec)
                self.fail = False
                super().run(spec)
                with self._state() as st:
                    self._find(st, spec.name).update(status="created", exit_code=None, finished_utc=None)
                raise RuntimeFailure("run", 125)
        rt = CreatedThenFails({IMAGE: _train}, outcomes=["hang"])
        b = self.make_backend(rt)
        with self.assertRaises(RuntimeFailure):
            b.submit(self.job())
        self.assertEqual(rt.inspect(self.name("0001-train")).status, "missing")
        self.assertEqual(b.jobs()[0].state, "queued")
        self.assertEqual(b.wait("0001-train").state, "succeeded")              # the next poll starts it

    def _gap(self, action) -> LocalBackend:
        """A backend that runs `action` after the job's record is written and before it takes the job's lock to
        start it (the gap between the project lock and the job lock)."""
        outer = self

        class Gap(LocalBackend):
            pending = True

            @contextlib.contextmanager
            def _lock(self, rel):
                if self.pending and rel.endswith("/lock") and rel != "jobs/lock":
                    self.pending = False
                    action()
                with LocalBackend._lock(self, rel):
                    yield
        return Gap(outer.project, outer.rt_gap, host=outer.host, sleep=lambda s: None, clock=lambda: CLOCK)

    def test_start_race_second_driver(self) -> None:
        """Another driver reattaches and starts attempt 1 in the gap; no second container."""
        self.rt_gap = FakeRuntime({IMAGE: _train}, outcomes=["hang", "hang"])
        job = self.job()
        other = self.make_backend(self.rt_gap)

        def race():
            self.assertEqual(other.submit(job), "0001-train")
            self.assertEqual(other.status("0001-train").state, "running")
        jid = self._gap(race).submit(job)
        self.assertEqual(len(self.rt_gap.runs), 1)
        self.assertEqual(self.events(jid), ["start"])

    def test_start_race_cancel(self) -> None:
        """A cancel in the gap; the job never starts."""
        self.rt_gap = FakeRuntime({IMAGE: _train})
        other = self.make_backend(self.rt_gap)
        jid = self._gap(lambda: other.cancel("0001-train")).submit(self.job())
        self.assertEqual(self.rt_gap.runs, [])
        self.assertEqual(self.events(jid), ["cancel_requested"])
        self.assertEqual(other.status(jid).state, "cancelled")

    def test_changed_inputs_new_key_and_output(self) -> None:
        """A changed script, small input or large input is another job with a fresh output
        directory; unchanged inputs (a small file rewritten with the same bytes included) reattach."""
        (self.inputs / "table.json").write_text('{"a": 1}')
        big = self.inputs / "big.bin"
        big.write_bytes(b"\0" * (be.SMALL_FILE_BYTES + 1))
        first = self.job()
        jid = self.backend.submit(first)
        self.assertEqual(self.backend.wait(jid).state, "succeeded")
        (self.inputs / "table.json").write_text('{"a": 1}')                    # same bytes, new mtime
        self.assertEqual(self.job(), first)
        self.assertEqual(self.backend.submit(self.job()), jid)
        st = big.stat()
        changes = {"script": lambda: (self.inputs / SCRIPT).write_text("print('train v2')\n"),
                   "small input": lambda: (self.inputs / "table.json").write_text('{"a": 2}'),
                   "large input": lambda: os.utime(big, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))}
        seen_keys, seen_dirs = {first.key()}, {first.output_dir}
        for what, change in changes.items():
            with self.subTest(what):
                change()
                job = self.job()
                self.assertNotIn(job.key(), seen_keys)
                self.assertNotIn(job.output_dir, seen_dirs)
                seen_keys.add(job.key())
                seen_dirs.add(job.output_dir)
                self.assertNotEqual(self.backend.submit(job), jid)
        self.assertEqual(len(self.rt.runs), 4)
        self.assertEqual(sorted(Path(d).name for d in seen_dirs), ["train", "train.2", "train.3", "train.4"])

    def test_large_script_hashed_by_content(self) -> None:
        """The script argv names is hashed whole, even past SMALL_FILE_BYTES with its size and mtime restored."""
        script = self.inputs / SCRIPT
        script.write_text("#" * be.SMALL_FILE_BYTES + "\nprint(1)\n")
        st = script.stat()
        key = self.job().key()
        script.write_text("#" * be.SMALL_FILE_BYTES + "\nprint(2)\n")
        os.utime(script, ns=(st.st_atime_ns, st.st_mtime_ns))
        self.assertNotEqual(self.job().key(), key)

    def test_fingerprint_bounded_and_no_links_followed(self) -> None:
        outside = self.tmp / "outside"
        outside.mkdir()
        (outside / "f").write_text("1")
        os.symlink(outside, self.inputs / "link")
        key = self.job().key()
        (outside / "f").write_text("2")                                         # behind a link: not read
        self.assertEqual(self.job().key(), key)
        with mock.patch.object(be, "MAX_INPUT_ENTRIES", 2), self.assertRaises(Refused) as cm:
            self.job().key()
        self.assertEqual((cm.exception.problems[0].rule, cm.exception.problems[0].field), ("AN-22", "job.mounts"))

    def test_reattach_refuses_tampered_record(self) -> None:
        rt = FakeRuntime({IMAGE: _train}, outcomes=["hang"])
        job = self.job()
        jid = self.make_backend(rt).submit(job)
        path = self.project.root / "jobs" / jid / "job.json"
        good = path.read_bytes()
        rec = json.loads(good)
        rec["job"]["time_limit_s"] = 1
        path.write_text(json.dumps(rec))
        with self.assertRaises(Refused) as cm:
            self.make_backend(rt).submit(job)
        self.assertEqual([p.field for p in cm.exception.problems], [f"jobs/{jid}/job.json"])
        path.write_bytes(good)
        cases = {"host_config_sha256": dataclasses.replace(self.host, extra_daemon_configs=(str(self.tmp / "d.json"),)),
                 "rootless": dataclasses.replace(self.host, rootless=False)}
        for k, host in cases.items():
            with self.subTest(k), self.assertRaises(Refused) as cm:
                LocalBackend(self.project, rt, host=host).submit(job)
            self.assertIn(f"jobs/{jid}/job.json.{k}", [p.field for p in cm.exception.problems])
        self.assertEqual(self.make_backend(rt).submit(job), jid)
        self.assertEqual(len(rt.runs), 1)

    def test_failed_start_adopts_container_that_ran(self) -> None:
        class RanThenFails(FakeRuntime):
            def run(self, spec):
                super().run(spec)
                raise RuntimeFailure("run", -1)                                # the CLI timed out after the start
        rt = RanThenFails({IMAGE: _train})
        b = self.make_backend(rt)
        jid = b.submit(self.job())
        self.assertEqual(b.wait(jid).state, "succeeded")
        self.assertEqual((len(rt.runs), self.events(jid)), (1, ["start", "end"]))

    def test_jobs_skips_directory_without_record(self) -> None:
        jid = self.backend.submit(self.job())
        (self.project.root / "jobs" / "0002-cut").mkdir()
        self.assertEqual([s.job_id for s in self.backend.jobs()], [jid])
        self.assertEqual(self.backend.submit(self.job("next")), "0003-next")

    def test_jobs_reads_records_only(self) -> None:
        jid = self.backend.submit(self.job())

        class NoRuntime:
            def __getattr__(self, name):
                raise AssertionError(f"runtime.{name} called")
        b = LocalBackend(self.project, NoRuntime(), host=self.host)
        self.assertEqual([(s.job_id, s.state) for s in b.jobs()], [(jid, "running")])


class TestOutputs(_Base):
    def succeeded(self) -> tuple[str, Path]:
        jid = self.backend.submit(self.job())
        self.assertEqual(self.backend.wait(jid).state, "succeeded")
        return jid, Path(self.job().output_dir) / "a1"

    def test_artifacts_manifest(self) -> None:
        jid, a1 = self.succeeded()
        got = self.backend.artifacts(jid)
        sha = lambda b: "sha256:" + hashlib.sha256(b).hexdigest()
        self.assertEqual(got, {"job_id": jid, "output_dir": str(a1), "files": [
            {"path": "model/config.json", "size": 2, "sha256": sha(b"{}")},
            {"path": "model/model.safetensors", "size": 7, "sha256": sha(b"weights")}]})
        os.symlink("/etc/passwd", a1 / "model" / "link")
        os.mkfifo(a1 / "fifo")
        with self.assertRaises(Refused) as cm:
            self.backend.artifacts(jid)
        self.assertEqual(sorted(p.field for p in cm.exception.problems), ["artifacts.fifo", "artifacts.model/link"])

    def test_artifacts_refuses_unfinished_job(self) -> None:
        rt = FakeRuntime({IMAGE: _train}, outcomes=["hang"])
        b = self.make_backend(rt)
        jid = b.submit(self.job())
        for op in (lambda: b.artifacts(jid), lambda: b.hand_off(jid, ["x"])):
            with self.assertRaises(Refused) as cm:
                op()
            self.assertEqual(cm.exception.problems[0].rule, "AN-22")

    def test_hand_off_refusals(self) -> None:
        """The keeper user must be able to read each handed-off file."""
        jid, a1 = self.succeeded()
        model = a1 / "model"
        got = self.backend.hand_off(jid, [str(model / "model.safetensors"), "model/config.json"])
        self.assertEqual(got, [{"path": str(model / "model.safetensors"),
                                "sha256": "sha256:" + hashlib.sha256(b"weights").hexdigest()},
                               {"path": str(model / "config.json"),
                                "sha256": "sha256:" + hashlib.sha256(b"{}").hexdigest()}])
        os.symlink(model / "model.safetensors", model / "link.safetensors")
        os.mkfifo(model / "fifo")
        (model / "private.bin").write_bytes(b"p")
        (model / "private.bin").chmod(0o600)
        (a1 / "closed").mkdir()
        (a1 / "closed" / "w.bin").write_bytes(b"w")
        (a1 / "closed" / "w.bin").chmod(0o644)
        (a1 / "closed").chmod(0o700)
        os.symlink(model, a1 / "linkdir")
        bad = {str(model / "link.safetensors"): "symbolic link", str(model / "fifo"): "not a regular file",
               str(model / "private.bin"): "no o+r", str(a1 / "closed" / "w.bin"): "no o+x",
               str(self.inputs / SCRIPT): "not inside", "../a1/../../inputs/x": "not inside",
               str(a1 / "linkdir" / "config.json"): "symbolic link", str(model): "not a regular file"}
        for path, words in bad.items():
            with self.subTest(path=path), self.assertRaises(Refused) as cm:
                self.backend.hand_off(jid, [str(model / "config.json"), path])
            self.assertEqual(len(cm.exception.problems), 1)
            p = cm.exception.problems[0]
            self.assertEqual((p.rule, p.field), ("AN-22", "hand_off.paths[1]"))
            self.assertIn(words, p.message)
            self.assertIn(path, p.message)
        (a1 / "closed").chmod(0o755)
        self.assertEqual(len(self.backend.hand_off(jid, [str(a1 / "closed" / "w.bin")])), 1)
        self.out_root.chmod(0o700)
        with self.assertRaises(Refused) as cm:
            self.backend.hand_off(jid, [str(model / "config.json")])
        self.assertIn(str(self.out_root), cm.exception.problems[0].message)
        self.out_root.chmod(0o755)

    def test_new_output_dir(self) -> None:
        d = self.backend.new_output_dir("x")
        self.assertEqual(d, self.out_root / f"gatedtrain-{self.backend.project_hex()[:12]}" / "x")
        self.assertEqual(oct(d.stat().st_mode & 0o777), oct(0o755))
        self.assertEqual(oct(d.parent.stat().st_mode & 0o777), oct(0o755))
        self.assertEqual(self.make_backend().new_output_dir("x"), d)           # the same project: returned again
        k1, k2 = "sha256:" + "1" * 64, "sha256:" + "2" * 64
        self.assertEqual(self.backend.new_output_dir("x", key=k1), d.parent / "x.2")   # "x" is taken (no key)
        self.assertEqual(self.backend.new_output_dir("x", key=k2), d.parent / "x.3")
        self.assertEqual(self.make_backend().new_output_dir("x", key=k1), d.parent / "x.2")
        with self.assertRaises(Refused):
            self.backend.new_output_dir("x.2")                                 # a job's directory, not a keyless one
        with self.assertRaises(Refused):
            self.backend.new_output_dir("y", key="sha256:short")
        (d.parent / "foreign").mkdir()
        os.symlink(d, d.parent / "link")
        for name in ("foreign", "link", "../x", "", ".", "a/b"):
            with self.subTest(name=name), self.assertRaises(Refused) as cm:
                self.backend.new_output_dir(name)
            self.assertEqual(cm.exception.problems[0].rule, "AN-22")
        other = create_project(self.tmp / "proj2", goal="g2", data=self.tmp / "data.txt", pipeline="p",
                               budget={"gpu_hours": 1.0, "llm_usd": 0})
        self.assertNotEqual(LocalBackend(other, self.rt, host=self.host).new_output_dir("x"), d)


class TestQuiet(unittest.TestCase):
    def test_constructing_touches_nothing(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "d").write_text("d")
        project = create_project(tmp / "p", goal="g", data=tmp / "d", pipeline="p",
                                 budget={"gpu_hours": 1.0, "llm_usd": 0})
        env = {k: v for k, v in os.environ.items() if not k.startswith(launch.ENV_PREFIX)}
        env[launch.ENV_HOST_CONFIG] = str(tmp / "absent.toml")
        boom = mock.Mock(side_effect=AssertionError("touched"))
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("subprocess.run", boom), \
                mock.patch.object(launch, "host_config", boom), mock.patch("subprocess.Popen", boom):
            b = LocalBackend(project)
            self.assertEqual(b.jobs(), [])
        boom.assert_not_called()
        self.assertFalse((tmp / "p" / "jobs").exists())

    def test_attempt_log_helpers_unchanged(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "d").write_text("d")
        project = create_project(tmp / "p", goal="g", data=tmp / "d", pipeline="p",
                                 budget={"gpu_hours": 1.0, "llm_usd": 0})
        project.append_attempt({"stage": "s", "event": "x", "schema": "spoofed"})
        with open(project.root / "attempts.jsonl", "ab") as f:
            f.write(b'{"cut": ')                                               # a write cut short by a crash
        project.append_attempt({"stage": "t"})
        self.assertEqual([(r["stage"], r["schema"]) for r in project.read_attempts()],
                         [("s", "gatedtrain.ainative.attempt"), ("t", "gatedtrain.ainative.attempt")])
        self.assertEqual(read_jsonl(project.root, "attempts.jsonl"), project.read_attempts())
        self.assertEqual(read_jsonl(project.root, "jobs/0001-x/attempts.jsonl"), [])
        (tmp / "elsewhere").mkdir()
        (project.root / "jobs").mkdir()
        os.symlink(tmp / "elsewhere", project.root / "jobs" / "0001-x")
        for op in (lambda: append_jsonl(project.root, "jobs/0001-x/attempts.jsonl", {"a": 1}),
                   lambda: read_jsonl(project.root, "jobs/0001-x/attempts.jsonl")):
            with self.assertRaises(Refused):
                op()
        self.assertEqual(list((tmp / "elsewhere").iterdir()), [])


class TestReportCompute(_Base):
    """The driver sends `compute` with the keeper's report only when the backend recorded a job, so a keeper that
    does not know `compute` still reports a project without jobs; unreadable job records never block the report."""

    def sent(self) -> tuple[dict, str]:
        from gatedtrain.ainative.keeper_trust import KeeperTrust
        calls = []
        fake = mock.Mock(project=self.project)
        fake._call = lambda op, args=None: calls.append((op, args)) or {"report": {}}
        fake._keep = lambda env, kind: {"kind": kind}
        from gatedtrain import output
        with output.notice_scope():
            self.assertEqual(KeeperTrust.report(fake), {"kind": "report"})
            self.assertEqual([op for op, _ in calls], ["report"])
            return calls[0][1] or {}, "\n".join(output.drain_notices())     # warnings go to the command's warnings list

    def test_no_jobs_no_compute(self) -> None:
        self.assertEqual(self.sent(), ({}, ""))

    def test_jobs_send_compute(self) -> None:
        jid = self.backend.submit(self.job())
        self.backend.wait(jid)
        args, err = self.sent()
        self.assertEqual(args["compute"]["jobs"], 1)
        self.assertEqual(err, "")

    def test_damaged_job_record_warns(self) -> None:
        jid = self.backend.submit(self.job())
        self.backend.wait(jid)
        (self.project.root / "jobs" / jid / "job.json").write_text("{not json")
        args, err = self.sent()
        self.assertNotIn("compute", args)
        self.assertTrue(err.startswith("warning: "), err)
        self.assertIn("job totals", err)


class TestJobProblems(_Base):
    def test_fields_and_messages(self) -> None:
        probs = job_problems(self.job(network="host"), self.host)
        self.assertEqual([(p.rule, p.field) for p in probs], [("AN-22", "job.network")])
        self.assertIn("EXT-81", probs[0].message)
        self.assertEqual(job_problems(self.job(), self.host), [])

    def test_job_key_and_doc(self) -> None:
        a, b = self.job(), self.job()
        self.assertEqual(a.key(), b.key())
        self.assertNotEqual(a.key(), self.job(training_seed=1).key())
        self.assertNotEqual(a.key(), self.job(env={"HF_HUB_OFFLINE": "1", "HOME": "/tmp/other"}).key())
        self.assertEqual(a.doc()["profile"], "full")


if __name__ == "__main__":
    unittest.main()
