"""AN-24: stage access to the backend (design revision 25, 05 §18.10; tests/README.md section 14).

A pack stage reaches the backend through `ctx.backend` (a `StageBackend`); trust stages get none. Only a stage that
declares `needs: gpu` submits GPU jobs, `plan` submits nothing and creates no output directory, and a stage that
submitted a job and then refused is `failed`. The backend is built when first used, so the step 1 synthetic pipeline
needs no host layout. `gatedtrain project job list|logs|cancel` under `--json`.

No test starts Docker or reads a GPU: the backend runs on `FakeRuntime` with a fake GPU probe, the stand-in adapter
and the host layout of `test_an_backend`; the CLI subprocesses get the same fake runtime through the
`GATEDTRAIN_TEST_BACKEND_RUNTIME` switch of `tests/fixtures/ainative/run_cli.py`.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import yaml

from _support import FIXTURES, cli_env

from gatedtrain import cli
from gatedtrain.ainative import machine
from gatedtrain.ainative.backend import LocalBackend, StageBackend
from gatedtrain.ainative.machine import plan_stage, run_stage
from gatedtrain.ainative.project import create_project
from gatedtrain.ainative.stage import TRUST_STAGES, Proposal
from gatedtrain.errors import Problem, Refused
from gatedtrain.ext import launch
from gatedtrain.testing.fake_runtime import FakeRuntime

from test_an_backend import CLOCK, IMAGE, SCRIPT, _Base, _train, setUpModule, tearDownModule  # noqa: F401

S = "gatedtrain.ainative.stub"
WRAPPER = FIXTURES / "ainative" / "run_cli.py"
SEEN: dict[str, object] = {}         # what the stages saw: ctx.backend per stage, per phase


class Probe:
    """A GPU that is always free."""

    def used_mib(self) -> int:
        return 0


def make_trust_stub(stage_name: str, first: bool):
    class Stub:
        name = stage_name
        version = "1"
        inputs = {} if first else {"in": S}
        outputs = {"out": S}
        needs = frozenset()
        params = frozenset()
        trains = False

        def plan(self, ctx):
            SEEN[f"plan:{self.name}"] = ctx.backend
            return Proposal(self.name, "stub", {}, {})

        def run(self, ctx, decision):
            SEEN[f"run:{self.name}"] = ctx.backend
            return {"out": ctx.project.write_artifact(ctx.stage, "out", {"stage": self.name}, S)}

        def check(self, ctx, outputs):
            return []

    return Stub


def make_worker(needs: frozenset, action, plan_action=None):
    """A pack stage that calls `action(ctx)` in `run` (and `plan_action(ctx)` in `plan`)."""

    class Worker:
        name = "worker"
        version = "1"
        inputs: dict = {}
        outputs = {"out": S}
        params = frozenset()
        trains = False

        def plan(self, ctx):
            SEEN["plan:worker"] = ctx.backend
            if plan_action is not None:
                plan_action(ctx)
            return Proposal(self.name, "worker", {}, {})

        def run(self, ctx, decision):
            SEEN["run:worker"] = ctx.backend
            result = action(ctx)
            return {"out": ctx.project.write_artifact(ctx.stage, "out", {"result": result}, S)}

        def check(self, ctx, outputs):
            return []

    Worker.needs = needs
    return Worker


TRUST = {name: make_trust_stub(name, name == "intake_seal") for name in TRUST_STAGES}
PIPELINE = {"name": "t", "stages": [
    {"name": "intake_seal", "stage": "intake_seal", "params": {}, "inputs": {}},
    {"name": "preregister", "stage": "preregister", "params": {}, "inputs": {"in": "intake_seal.out"}},
    {"name": "worker", "stage": "worker", "params": {}, "inputs": {}},          # `needs: gpu` trains: after preregister
    {"name": "evaluate_sealed", "stage": "evaluate_sealed", "params": {}, "inputs": {"in": "preregister.out"}},
    {"name": "gate", "stage": "gate", "params": {}, "inputs": {"in": "evaluate_sealed.out"}},
    {"name": "report", "stage": "report", "params": {}, "inputs": {"in": "gate.out"}},
]}


class _StageBase(_Base):
    """`test_an_backend._Base` with a project that has a pipeline: intake_seal, preregister, the worker, then the
    other trust stubs."""

    def setUp(self) -> None:
        SEEN.clear()
        super().setUp()
        (self.tmp / "pipeline.yaml").write_text(yaml.safe_dump(PIPELINE))
        data = self.tmp / "d.jsonl"
        data.write_text('{"q": 1}\n')
        self.project = create_project(self.tmp / "sproj", goal="g", data=data,
                                      pipeline=str(self.tmp / "pipeline.yaml"),
                                      budget={"gpu_hours": 1.0, "llm_usd": 0})
        self.backend = self.make_backend()
        self.inputs = self.backend.new_output_dir("inputs")
        (self.inputs / SCRIPT).write_text("print('train')\n")

    def make_backend(self, rt: FakeRuntime | None = None) -> LocalBackend:
        return LocalBackend(self.project, rt or self.rt, host=self.host, probe=Probe(), sleep=lambda s: None,
                            clock=lambda: CLOCK)

    def registry(self, worker) -> dict:
        return {**TRUST, "worker": worker}

    def run_(self, ref: str, worker, decision: dict | None = None):
        return run_stage(self.project, ref, decision or {}, registry=self.registry(worker), backend=self.backend)

    def ready(self, worker) -> None:
        """Run the stages before the worker."""
        self.run_("intake_seal", worker)
        self.run_("preregister", worker)

    def last_event(self, stage: str) -> dict:
        return [r for r in self.project.read_attempts() if r["stage"] == stage][-1]

    def job_ids(self) -> list[str]:
        return self.backend._job_ids()


class TestStageAccess(_StageBase):
    def test_gpu_job_from_a_stage_without_gpu_need_is_refused(self) -> None:
        job = self.job(gpu=True, gpu_hours_est=0.1)
        worker = make_worker(frozenset(), lambda ctx: ctx.backend.submit(job))
        self.ready(worker)
        with self.assertRaises(Refused) as cm:
            self.run_("worker", worker)
        self.assertEqual([p.rule for p in cm.exception.problems], ["AN-24"])
        self.assertIn("needs: gpu", cm.exception.problems[0].message)
        row = self.last_event("worker")
        self.assertEqual((row["event"], row["rules"]), ("refused", ["AN-24"]))
        self.assertEqual(self.job_ids(), [])
        self.assertEqual(self.rt.runs, [])

    def test_gpu_stage_submits_a_gpu_job(self) -> None:
        job = self.job(gpu=True, gpu_hours_est=0.1)

        def action(ctx):
            jid = ctx.backend.submit(job)
            return ctx.backend.wait(jid, poll_s=0).state

        worker = make_worker(frozenset({"gpu"}), action)
        self.ready(worker)
        result = self.run_("worker", worker)
        self.assertEqual(self.project.read_artifact(result.outputs["out"]), {"result": "succeeded"})
        self.assertEqual(self.last_event("worker")["event"], "done")
        [jid] = self.job_ids()
        rec = json.loads((self.project.root / "jobs" / jid / "job.json").read_text())
        self.assertEqual((rec["stage"], rec["job"]["gpu"]), ("worker", True))
        self.assertIsInstance(SEEN["run:worker"], StageBackend)

    def test_cpu_job_from_any_stage(self) -> None:
        job = self.job("cpu")
        worker = make_worker(frozenset(), lambda ctx: ctx.backend.wait(ctx.backend.submit(job), poll_s=0).state)
        self.ready(worker)
        result = self.run_("worker", worker)
        self.assertEqual(self.project.read_artifact(result.outputs["out"]), {"result": "succeeded"})

    def test_plan_cannot_submit_or_create_an_output_dir(self) -> None:
        job = self.job("cpu")
        calls = []

        def plan_action(ctx):
            for name, op in (("submit", lambda: ctx.backend.submit(job)),
                             ("new_output_dir", lambda: ctx.backend.new_output_dir("fresh")),
                             ("cancel", lambda: ctx.backend.cancel("0001-cpu"))):
                with self.assertRaises(Refused) as cm:
                    op()
                self.assertEqual([p.rule for p in cm.exception.problems], ["AN-24"], name)
                calls.append(name)
            ctx.backend.jobs()                                  # reading the records is no side effect

        worker = make_worker(frozenset({"gpu"}), lambda ctx: None, plan_action)
        self.ready(worker)
        plan_stage(self.project, "worker", registry=self.registry(worker), backend=self.backend)
        self.assertEqual(calls, ["submit", "new_output_dir", "cancel"])
        self.assertEqual(self.job_ids(), [])
        out_base = Path(self.inputs).parent
        self.assertFalse((out_base / "fresh").exists())

    def test_a_stage_that_submitted_then_refused_is_failed(self) -> None:
        job = self.job("cpu")

        def action(ctx):
            ctx.backend.submit(job)
            raise Refused([Problem("X-1", "f", "refused after a submit", "fix")])

        worker = make_worker(frozenset(), action)
        self.ready(worker)
        with self.assertRaises(Refused):
            self.run_("worker", worker)
        row = self.last_event("worker")
        self.assertEqual((row["event"], row["rules"]), ("failed", ["X-1"]))
        self.assertEqual(len(self.job_ids()), 1)

    def test_a_stage_that_only_created_an_output_dir_then_refused_is_failed(self) -> None:
        def action(ctx):
            ctx.backend.new_output_dir("scratch")
            raise Refused([Problem("X-1", "f", "refused after an output directory", "fix")])

        worker = make_worker(frozenset(), action)
        self.ready(worker)
        with self.assertRaises(Refused):
            self.run_("worker", worker)
        self.assertEqual(self.last_event("worker")["event"], "failed")

    def test_trust_stages_get_no_backend(self) -> None:
        worker = make_worker(frozenset(), lambda ctx: None)
        plan_stage(self.project, "intake_seal", registry=self.registry(worker), backend=self.backend)
        for ref in ("intake_seal", "preregister", "worker", "evaluate_sealed", "gate", "report"):
            self.run_(ref, worker)
        for name in TRUST_STAGES:
            self.assertIn(f"run:{name}", SEEN)
            self.assertIsNone(SEEN[f"run:{name}"], name)
        self.assertIsNone(SEEN["plan:intake_seal"])
        self.assertIsInstance(SEEN["run:worker"], StageBackend)

    def test_stage_backend_passes_the_stage_name(self) -> None:
        sb = StageBackend(self.backend, stage="worker", allow_gpu=False)
        jid = sb.submit(self.job("cpu"))
        rec = json.loads((self.project.root / "jobs" / jid / "job.json").read_text())
        self.assertEqual(rec["stage"], "worker")
        self.assertEqual(sb.wait(jid, poll_s=0).state, "succeeded")
        self.assertEqual([s.job_id for s in sb.jobs()], [jid])
        self.assertIn("trained", sb.logs(jid))

    def test_the_default_backend_is_compute_backend(self) -> None:
        """Without `backend=`, run_stage builds one with `compute_backend` (the production probe and runtime)."""
        built = []

        def fake(project, **kw):
            built.append(project.root)
            return self.backend

        worker = make_worker(frozenset(), lambda ctx: None)
        with mock.patch.object(machine, "compute_backend", fake):
            for ref in ("intake_seal", "preregister"):
                run_stage(self.project, ref, {}, registry=self.registry(worker))
            self.assertEqual(built, [])                         # a trust stage builds none
            run_stage(self.project, "worker", {}, registry=self.registry(worker))
        self.assertEqual(built, [self.project.root])

    def test_compute_backend_uses_the_production_probe_and_context(self) -> None:
        from gatedtrain.ainative.runtime import DockerCLIRuntime, NvidiaSmiProbe

        b = machine.compute_backend(self.project)
        self.assertIsInstance(b.probe, NvidiaSmiProbe)
        self.assertIsInstance(b.runtime, DockerCLIRuntime)
        self.assertEqual(b.runtime.context, launch.DOCKER_CONTEXT)


class TestLazyBackend(_Base):
    """The step 1 synthetic pipeline runs unchanged with no host layout: nothing reads the host file, runs Docker or
    nvidia-smi."""

    def test_synthetic_pipeline_needs_no_host_layout(self) -> None:
        from test_an_synthetic_pack import DECISIONS, fixture_rows, inject_pack, jsonl

        from gatedtrain.ainative.runtime import DockerCLIRuntime, NvidiaSmiProbe
        from gatedtrain.ainative.stage import stage_registry
        from gatedtrain.testing import synthetic_pack as pack

        inject_pack(self)
        env = {k: v for k, v in os.environ.items() if not k.startswith(launch.ENV_PREFIX)}
        env[launch.ENV_HOST_CONFIG] = str(self.tmp / "no-such-host.toml")
        boom = AssertionError("touched the host or a runtime")
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(launch, "host_config", side_effect=boom), \
                mock.patch.object(DockerCLIRuntime, "_cli", side_effect=boom), \
                mock.patch.object(NvidiaSmiProbe, "used_mib", side_effect=boom):
            data = self.tmp / "syn.jsonl"
            data.write_bytes(jsonl(fixture_rows()))
            raw = yaml.safe_load(pack.SCENARIO.pipeline.read_text())
            (self.tmp / "syn.yaml").write_text(yaml.safe_dump(raw))
            proj = create_project(self.tmp / "syn", goal="lookup", data=data, pipeline=str(self.tmp / "syn.yaml"),
                                  budget={"gpu_hours": 1, "llm_usd": 1})
            out = {}
            for s in raw["stages"]:
                out[s["name"]] = run_stage(proj, s["name"], dict(DECISIONS.get(s["name"], {})),
                                           registry=stage_registry()).outputs
        self.assertEqual(proj.read_artifact(out["gate"]["gate_result"])["status"], "PASS")
        self.assertFalse((proj.root / "jobs").exists())


class TestJobCli(_Base):
    """`gatedtrain project job list|logs|cancel` (AN-1 envelope, stderr empty), the jobs submitted in this process
    on a `FakeRuntime` whose state directory the CLI subprocesses share."""

    def setUp(self) -> None:
        super().setUp()
        self.state = Path(tempfile.mkdtemp(dir=self.tmp, prefix="rt-"))
        self.rt = FakeRuntime({IMAGE: _train}, state_dir=self.state, outcomes=["ok", "hang"])
        self.backend = self.make_backend(self.rt)
        self.done = self.backend.submit(self.job("done"), stage="worker")
        self.assertEqual(self.backend.wait(self.done).state, "succeeded")
        self.running = self.backend.submit(self.job("running"), stage="worker")
        self.assertEqual(self.backend.status(self.running).state, "running")

    def cli(self, *args: str, expect: int = 0) -> dict:
        env = {**cli_env(), "GATEDTRAIN_TEST_BACKEND_RUNTIME": f"fake:{self.state}"}
        proc = subprocess.run([sys.executable, str(WRAPPER), "--json", *args], cwd=self.tmp, env=env,
                              capture_output=True, text=True, timeout=120)
        where = f"{args!r}\nstdout: {proc.stdout[-3000:]}\nstderr: {proc.stderr[-3000:]}"
        self.assertEqual(proc.stderr, "", where)
        self.assertEqual(proc.stdout.count("\n"), 1, where)
        self.assertEqual(proc.returncode, expect, where)
        doc = json.loads(proc.stdout)
        self.assertEqual((doc["schema"], doc["exit"]), ("gatedtrain.cli_result", expect), where)
        return doc

    def test_list_logs_cancel(self) -> None:
        root = str(self.project.root)
        doc = self.cli("project", "job", "list", root)
        self.assertEqual(doc["command"], "project job list")
        jobs = {j["job_id"]: j for j in doc["data"]["jobs"]}
        self.assertEqual({k: v["state"] for k, v in jobs.items()}, {self.done: "succeeded", self.running: "running"})
        self.assertEqual(set(jobs[self.done]), {"job_id", "state", "attempt", "rc", "outcome", "gpu_seconds",
                                                "seconds", "reason"})

        doc = self.cli("project", "job", "logs", root, self.done)
        self.assertEqual(doc["command"], "project job logs")
        self.assertEqual(doc["data"]["job_id"], self.done)
        self.assertIn("trained", doc["data"]["log"])
        doc = self.cli("project", "job", "logs", root, self.done, "--tail", "0")
        self.assertEqual(doc["data"]["log"], "")

        doc = self.cli("project", "job", "cancel", root, self.running)
        self.assertEqual(doc["command"], "project job cancel")
        self.assertEqual(doc["data"]["job"]["job_id"], self.running)
        self.assertEqual(doc["data"]["job"]["state"], "cancelled")
        self.assertEqual(self.backend.status(self.running).state, "cancelled")
        doc = self.cli("project", "job", "list", root)
        self.assertEqual({j["job_id"]: j["state"] for j in doc["data"]["jobs"]},
                         {self.done: "succeeded", self.running: "cancelled"})

    def test_unknown_job_is_refused(self) -> None:
        for sub in ("logs", "cancel"):
            doc = self.cli("project", "job", sub, str(self.project.root), "9999-none", expect=3)
            self.assertEqual(doc["problems"][0]["rule"], "AN-22", sub)

    def test_list_reads_the_records_only(self) -> None:
        class NoRuntime:
            def __getattr__(self, name):
                raise AssertionError(f"`project job list` touched the runtime ({name})")

        real = machine.compute_backend
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(machine, "compute_backend", lambda project, **kw: real(project, runtime=NoRuntime())), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(["--json", "project", "job", "list", str(self.project.root)])
        self.assertEqual((code, err.getvalue()), (0, ""))
        doc = json.loads(out.getvalue())
        self.assertEqual({j["job_id"]: j["state"] for j in doc["data"]["jobs"]},
                         {self.done: "succeeded", self.running: "running"})

    def test_text_mode(self) -> None:
        out = io.StringIO()
        with mock.patch.object(machine, "compute_backend", lambda project, **kw: self.backend), \
                contextlib.redirect_stdout(out):
            code = cli.main(["project", "job", "list", str(self.project.root)])
        self.assertEqual(code, 0)
        self.assertIn(f"{self.done} succeeded", out.getvalue())
        self.assertIn(f"{self.running} running", out.getvalue())
