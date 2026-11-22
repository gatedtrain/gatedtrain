"""AN-20 to AN-25: step 3a end to end on the CPU with fake runtimes (design revision 25, 05 §18.10; tests/README.md
section 14).

The batch form of the synthetic pipeline (`synthetic_pack_batch.yaml`) runs through the command line only: every call
is a subprocess of `tests/fixtures/ainative/run_cli.py --json ...`, every decision comes from a file and stderr is empty
on every call (AN-1). Its `choose_task_and_base` stage writes the base model's files to a backend output directory, and
its `train` stage fits the table in process, then runs `synthetic_train_job.py` as a CPU job of the local backend and
hands the job's `table.json` to the keeper as the candidate's model file.

Nothing touches Docker, a GPU or the host's Docker groups:

- the backend's containers run on `FakeRuntime` (`GATEDTRAIN_TEST_BACKEND_RUNTIME=fake:<state dir>`), whose
  containers run the job's script on the host (`subprocess_handler`); the host layout is a temporary directory with a
  host file `rootless = true`;
- the keeper is a same-user subprocess (`tests/_keeper.py`) whose inference containers run on its own `FakeRuntime`
  (`GATEDTRAIN_TEST_KEEPER_RUNTIME`);
- the switches also make the Docker CLI runtime and the `nvidia-smi` probe raise, and every subprocess finds a `docker`
  and an `nvidia-smi` on its `PATH` that only leave a mark: a test fails when a mark exists.

A train job that is preempted is retried in the same job; a driver killed while its job runs is replaced by a new
`project run train`, which reattaches to the running container (the test ends the container with
`FakeRuntime.finish`). No sealed input appears in any output, project file, job log or job record.
`ext-launch` on the stand-in adapter with the rootless host file prints the rootless user and the CDI GPU option.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import yaml

from _ext import container_user, gpu_step_script
from _keeper import KeeperProcess, skip_unless_linux
from _support import FIXTURES, cli_env

from gatedtrain.ainative import machine
from gatedtrain.ainative.project import Project, create_project, read_jsonl
from gatedtrain.ainative.runtime import DockerCLIRuntime, NvidiaSmiProbe
from gatedtrain.ext import launch
from gatedtrain.testing import standin, synthetic_pack as pack
from gatedtrain.testing.fake_runtime import FakeRuntime

WRAPPER = FIXTURES / "ainative" / "run_cli.py"
TOY = FIXTURES / "ainative" / "toy.jsonl"
EXIT_OK, EXIT_NOT_PASS, EXIT_REFUSED = 0, 1, 3
MARGIN = 0.05
KEEPER_IMAGE = "sha256:" + "1" * 64            # the keeper's inference image (held by its fake daemon)
TRAIN_IMAGE = "sha256:" + "ab" * 32            # the train job's image (the pipeline's `image` param in a copy)
BEFORE_CONFIRM = ("intake_seal", "profile_data", "choose_task_and_base")
TO_TRAIN = ("preregister", "build_train_set", "write_recipe", "smoke")
WAIT_S = 120.0
TRIPWIRE = "#!/bin/sh\necho \"$0 $*\" >> '{mark}'\nexit 1\n"


def _wait_for(what: str, cond, timeout_s: float = WAIT_S) -> None:
    deadline = time.monotonic() + timeout_s
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.05)


def load_run_cli():
    spec = importlib.util.spec_from_file_location("gatedtrain_test_run_cli", WRAPPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _E2EBase(unittest.TestCase):
    """A host layout (host work, output root, host file `rootless = true`), a fake backend daemon, tripwire `docker`
    and `nvidia-smi` executables, and the command line."""

    KEEPER = False

    def setUp(self) -> None:
        container_user()                    # skip (fail under CI) where the owner's primary group owns a Docker socket
        self.tmp = Path(os.path.realpath(tempfile.mkdtemp(prefix="gt-e2e3a-")))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        os.chmod(self.tmp, 0o755)
        self.work = self.tmp / "work"
        self.out_root = self.tmp / "out"
        for d in (self.work, self.out_root):
            d.mkdir()
            d.chmod(0o755)
        self.host_file = self.tmp / "host.toml"
        self.host_file.write_text("rootless = true\n")
        self.fake = self.tmp / "fake-backend"
        self.fake.mkdir()
        self.mark = self.tmp / "tripwire.log"
        tripwire = self.tmp / "bin"
        tripwire.mkdir()
        for name in ("docker", "nvidia-smi"):
            (tripwire / name).write_text(TRIPWIRE.format(mark=self.mark))
            (tripwire / name).chmod(0o755)
        env = {k: v for k, v in cli_env().items() if not k.startswith(launch.ENV_PREFIX)}
        env.update({launch.ENV_HOST_WORK: str(self.work), launch.ENV_OUTPUT_ROOT: str(self.out_root),
                    launch.ENV_HOST_CONFIG: str(self.host_file),
                    "GATEDTRAIN_TEST_BACKEND_RUNTIME": f"fake:{self.fake}",
                    "PATH": f"{tripwire}{os.pathsep}{os.environ.get('PATH', os.defpath)}"})
        self.env = env
        self.outputs: list[str] = []
        self._n = 0
        self.data = self.tmp / "data.jsonl"
        self.data.write_bytes(TOY.read_bytes())
        raw = yaml.safe_load(pack.BATCH_PIPELINE.read_text())
        next(s for s in raw["stages"] if s["name"] == "train")["params"]["image"] = TRAIN_IMAGE
        self.pipeline = self.tmp / "pipeline.yaml"
        self.pipeline.write_text(yaml.safe_dump(raw, sort_keys=False))
        self.kp = None
        if self.KEEPER:
            self.keeper_fake = self.tmp / "fake-keeper"
            self.keeper_fake.mkdir()
            sock_dir = self.tmp / "dockerd"
            sock_dir.mkdir(mode=0o700)
            os.chmod(sock_dir, 0o700)
            (sock_dir / "docker.sock").write_bytes(b"")
            self.kp = KeeperProcess(env={"GATEDTRAIN_TEST_KEEPER_RUNTIME": f"fake:{KEEPER_IMAGE}:{self.keeper_fake}",
                                         "PATH": env["PATH"]},
                                    args=["--inference-image", KEEPER_IMAGE,
                                          "--docker-host", f"unix://{sock_dir / 'docker.sock'}"])
            self.kp.__enter__()
            self.addCleanup(self.kp.__exit__, None, None, None)
        self.addCleanup(self.assert_no_tripwire)

    def assert_no_tripwire(self) -> None:
        self.assertFalse(self.mark.exists(), self.mark.read_text() if self.mark.exists() else "")

    # the command line -----------------------------------------------------------------------------------------------
    def argv(self, *args: str) -> list[str]:
        return [sys.executable, str(WRAPPER), "--json", *args]

    def cli(self, *args: str, expect: int | None = EXIT_OK) -> dict:
        proc = subprocess.run(self.argv(*args), cwd=self.tmp, env=self.env, capture_output=True, text=True,
                              timeout=300)
        self.outputs += [proc.stdout, proc.stderr]
        where = f"{args!r}\nstdout: {proc.stdout[-3000:]}\nstderr: {proc.stderr[-3000:]}"
        self.assertEqual(proc.stderr, "", where)                                  # AN-1
        self.assertEqual(proc.stdout.count("\n"), 1, where)
        doc = json.loads(proc.stdout)
        self.assertEqual((doc["schema"], doc["exit"]), ("gatedtrain.cli_result", proc.returncode), where)
        if expect is not None:
            self.assertEqual(proc.returncode, expect, where)
        return doc

    def popen(self, *args: str) -> subprocess.Popen:
        return subprocess.Popen(self.argv(*args), cwd=self.tmp, env=self.env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)

    def finish(self, proc: subprocess.Popen, expect: int = EXIT_OK) -> dict:
        out, err = proc.communicate(timeout=WAIT_S)
        self.outputs += [out, err]
        where = f"stdout: {out[-3000:]}\nstderr: {err[-3000:]}"
        self.assertEqual((proc.returncode, err), (expect, ""), where)
        return json.loads(out)

    def decision(self, doc: dict) -> str:
        self._n += 1
        path = self.tmp / f"decision-{self._n}.json"
        path.write_text(json.dumps(doc))
        return str(path)

    def stage(self, root: Path, ref: str, decision: dict | None = None, *, expect: int | None = EXIT_OK) -> dict:
        extra = ["--decision", self.decision(decision)] if decision else []
        return self.cli("project", "run", str(root), ref, *extra, expect=expect)

    def create(self, name: str = "proj") -> Path:
        root = self.tmp / name
        extra = ["--keeper", str(self.kp.socket)] if self.kp is not None else []
        self.cli("project", "create", str(root), "--goal", "lookup", "--data", str(self.data),
                 "--pipeline", str(self.pipeline), *extra)
        return root

    def to_train(self, root: Path) -> None:
        """Every stage before `train`."""
        for ref in BEFORE_CONFIRM:
            self.stage(root, ref, {"accept_inconclusive": True} if ref == "intake_seal" else None)
        if self.kp is not None:
            self.cli("keeper", "confirm", str(root), "--decision",
                     self.decision({"metric_confirmed": True, "scorer": "exact_match", "margin": MARGIN}), "--yes")
        for ref in TO_TRAIN:
            self.stage(root, ref, {"metric_confirmed": True, "margin": MARGIN} if ref == "preregister" else None)

    # records --------------------------------------------------------------------------------------------------------
    @staticmethod
    def read(root: Path, rel: str) -> dict:
        return json.loads((root / rel).read_text())

    def jobs(self, root: Path) -> list[dict]:
        return self.cli("project", "job", "list", str(root))["data"]["jobs"]

    @staticmethod
    def job_rows(root: Path, job_id: str) -> list[dict]:
        return read_jsonl(root, f"jobs/{job_id}/attempts.jsonl")

    @staticmethod
    def stage_rows(root: Path, ref: str) -> list[dict]:
        return [r for r in Project.load(root).read_attempts() if r.get("stage") == ref]

    def gpu_log_rows(self) -> list[str]:
        text = (self.work / "gpu-log.md").read_text()
        return [line for line in text.splitlines() if line.startswith("| ") and not line.startswith("| Step ")]

    def model_matches_job(self, root: Path, job_id: str) -> None:
        """The candidate reference lists the job's `table.json` (the succeeded attempt's), with its hash."""
        model = self.read(root, "stages/train/model.json")
        self.assertEqual((model["generator"], model["label"]), (pack.BATCH_GENERATOR, "candidate"))
        [entry] = model["files"]
        start = [r for r in self.job_rows(root, job_id) if r["event"] == "start"][-1]
        self.assertEqual(entry["path"], os.path.join(os.path.realpath(start["output"]), "table.json"))
        data = Path(entry["path"]).read_bytes()
        mounts = {m["container"]: m["host"] for m in self.read(root, f"jobs/{job_id}/job.json")["job"]["mounts"]}
        self.assertEqual(data, Path(mounts["/inputs/table.json"]).read_bytes())      # the table fitted in process
        self.assertEqual(entry["sha256"], "sha256:" + hashlib.sha256(data).hexdigest())


@skip_unless_linux
class BatchPipelineWithKeeperTest(_E2EBase):
    KEEPER = True

    def assert_no_sealed_input(self, root: Path) -> None:
        """No input outside the train part in any output, project file (job logs and `jobs/` records included), the
        backend's output root, the fake daemons' state or the keeper's log."""
        every = {json.loads(line)["input"] for line in TOY.read_text().splitlines() if line}
        train = {r["input"] for r in self.read(root, "stages/intake_seal/train_part.json")["rows"]}
        sealed = every - train
        self.assertGreater(len(sealed), 100)
        pattern = re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(s) for s in sorted(sealed, key=len, reverse=True))
                             + r")(?![\w-])")
        seen = {f"output {i}": t for i, t in enumerate(self.outputs)}
        seen["keeper log"] = self.kp.output()
        files = [p for d in (root, self.out_root, self.fake, self.keeper_fake, self.work) for p in d.rglob("*")]
        self.assertTrue(any(p.parent.name.startswith("0001-") and p.name == "attempt-1.log" for p in files))
        for path in files:
            if path.is_file():
                seen[str(path.relative_to(self.tmp))] = path.read_text(encoding="utf-8", errors="replace")
        for where, text in seen.items():
            hit = pattern.search(text)
            self.assertIsNone(hit, f"{where} holds the sealed input {hit and hit.group(0)!r}")

    def test_batch_pipeline_end_to_end(self) -> None:
        root = self.create()
        self.to_train(root)
        self.stage(root, "train")
        self.stage(root, "evaluate_sealed")
        self.stage(root, "gate")
        self.assertEqual(self.read(root, "stages/gate/gate_result.json")["status"], "PASS")
        self.stage(root, "report")
        status = self.cli("project", "status", str(root))
        self.assertEqual({k: v["status"] for k, v in status["data"]["stages"].items()},
                         {k: "done" for k in status["data"]["stages"]})
        self.assertEqual(status["data"]["gate"], "PASS")

        base = self.read(root, "stages/choose_task_and_base/base.json")
        self.assertEqual((base["generator"], base["label"]), (pack.BATCH_GENERATOR, "base"))
        self.assertEqual(json.loads(Path(base["files"][0]["path"]).read_text()), {})

        report = self.read(root, "stages/report/report.json")
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["compute"]["keeper"]["containers"], 2)
        self.assertEqual(report["compute"]["backend"]["jobs"], 1)
        self.assertEqual(report["compute"]["backend"]["gpu_seconds"], 0)
        self.assertIn("- backend: 1 jobs", (root / "stages/report/report.md").read_text())
        self.assertEqual(len(FakeRuntime(state_dir=self.keeper_fake).runs), 2)

        doc = self.cli("verify", str(root), "--keeper", str(self.kp.socket), expect=EXIT_NOT_PASS)
        self.assertIs(doc["data"]["same_user"], True)
        self.assertEqual(doc["problems"], [])

        [job] = self.jobs(root)
        self.assertEqual((job["state"], job["attempt"], job["gpu_seconds"]), ("succeeded", 1, 0))
        rec = self.read(root, f"jobs/{job['job_id']}/job.json")
        self.assertEqual((rec["stage"], rec["rootless"], rec["job"]["adapter"], rec["job"]["gpu"]),
                         ("train", True, standin.NAME, False))
        self.assertEqual(rec["job"]["runtime"], {"kind": "container", "image": TRAIN_IMAGE})
        [spec] = FakeRuntime(state_dir=self.fake).runs
        self.assertEqual((spec.user, spec.network, spec.gpu_options), ("0:0", "none", ()))
        self.assertEqual(spec.argv, ("/usr/local/bin/python3", "/inputs/synthetic_train_job.py", "--table",
                                     "/inputs/table.json", "--out", "/out"))
        self.model_matches_job(root, job["job_id"])

        [row] = self.gpu_log_rows()
        self.assertTrue(row.startswith(f"| {job['job_id']} (train), 1 attempt(s) |"), row)
        self.assertIn("| 0.00 | 0.00 | SUCCEEDED rc=0 |", row)
        self.assert_no_sealed_input(root)


@skip_unless_linux
class TrainJobTest(_E2EBase):
    """The `train` stage's job on the fake backend daemon, the project in process trust (no keeper)."""

    def test_preempted_then_ok(self) -> None:
        root = self.create()
        self.to_train(root)
        FakeRuntime(state_dir=self.fake, outcomes=["preempt", "ok"])
        self.stage(root, "train")
        [job] = self.jobs(root)
        self.assertEqual((job["state"], job["attempt"]), ("succeeded", 2))
        ends = [r for r in self.job_rows(root, job["job_id"]) if r["event"] == "end"]
        self.assertEqual([(r["attempt"], r["outcome"]) for r in ends], [(1, "preempted"), (2, "ok")])
        self.assertEqual([r["event"] for r in self.stage_rows(root, "train")], ["start", "done"])
        self.assertEqual(len(FakeRuntime(state_dir=self.fake).runs), 2)
        self.model_matches_job(root, job["job_id"])
        [row] = self.gpu_log_rows()
        self.assertIn(", 2 attempt(s) |", row)

    def test_killed_driver_reattaches(self) -> None:
        """The driver dies while the job runs; a new `project run train` reattaches to the running
        container (one run), records its end once and hands its output off."""
        root = self.create()
        self.to_train(root)
        rt = FakeRuntime(state_dir=self.fake, outcomes=["hang"])
        first = self.popen("project", "run", str(root), "train")
        self.addCleanup(lambda: first.poll() is None and first.kill())
        jobs_dir = root / "jobs"

        def started() -> bool:
            ids = [p.name for p in jobs_dir.iterdir() if p.name.startswith("0001-")] if jobs_dir.is_dir() else []
            return bool(ids) and any(r["event"] == "start" for r in self.job_rows(root, ids[0]))
        _wait_for("the train job's first attempt", started)
        first.kill()
        first.communicate(timeout=WAIT_S)
        [job] = self.jobs(root)
        self.assertEqual(job["state"], "running")
        self.assertEqual([r["event"] for r in self.stage_rows(root, "train")], ["start"])

        second = self.popen("project", "run", str(root), "train")
        self.addCleanup(lambda: second.poll() is None and second.kill())
        _wait_for("the second driver's stage start",
                  lambda: [r["event"] for r in self.stage_rows(root, "train")] == ["start", "start"])
        time.sleep(0.5)                                          # let it reach its job's polls
        self.assertIsNone(second.poll())
        start = [r for r in self.job_rows(root, job["job_id"]) if r["event"] == "start"][-1]
        table = (self.out_root / f"gatedtrain-{start['name'].split('-')[1]}" / "train-inputs" / "table.json")
        out = Path(start["output"]) / "table.json"               # the container's output, written by the test
        out.write_bytes(table.read_bytes())
        out.chmod(0o644)
        rt.finish(start["name"], 0)
        doc = self.finish(second)
        self.assertEqual(doc["data"]["stage"], "train")

        self.assertEqual(len(FakeRuntime(state_dir=self.fake).runs), 1)
        [job] = self.jobs(root)
        self.assertEqual((job["state"], job["attempt"]), ("succeeded", 1))
        self.assertEqual([r["event"] for r in self.job_rows(root, job["job_id"])].count("end"), 1)
        self.assertEqual([r["event"] for r in self.stage_rows(root, "train")], ["start", "start", "done"])
        self.model_matches_job(root, job["job_id"])
        self.assertEqual(len(self.gpu_log_rows()), 1)


@skip_unless_linux
class ExtLaunchStandinTest(_E2EBase):
    """`ext-launch` prints, never runs: the stand-in adapter under the rootless host file."""

    def request(self, *, gpu: bool) -> Path:
        inputs = self.out_root / "inputs"
        inputs.mkdir(exist_ok=True)
        (inputs / "synthetic_train_job.py").write_text("print('train')\n")
        (inputs / "table.json").write_text("{}")
        out = self.out_root / ("gpu-out" if gpu else "cpu-out")
        out.mkdir()
        req = {"area": 0, "label": "standin-gpu" if gpu else "standin-cpu", "host_work": str(self.work),
               "runtime": {"kind": "container", "image": TRAIN_IMAGE},
               "framework": {"name": standin.FRAMEWORK_PACKAGE, "version": "0.1", "commit": None},
               "training_seed": 0, "gpus": gpu, "network": "none", "network_reason": None, "user": "0:0",
               "env": {"HF_HUB_OFFLINE": "1", "HOME": "/tmp/home"},
               "mounts": [*({"host": str(inputs / n), "container": f"/inputs/{n}", "mode": "ro"}
                            for n in ("synthetic_train_job.py", "table.json")),
                          {"host": str(out), "container": "/out", "mode": "rw"}],
               "tmpfs": ["/tmp"], "output_dir": str(out), "gpu_est_h": 0.25 if gpu else 0.0,
               "argv": ["/usr/local/bin/python3", "/inputs/synthetic_train_job.py", "--table", "/inputs/table.json",
                        "--out", "/out"]}
        path = self.tmp / f"request-{'gpu' if gpu else 'cpu'}.yaml"
        path.write_text(yaml.safe_dump(req, sort_keys=False))
        return path

    def launch(self, *, gpu: bool) -> str:
        log = self.work / "gpu-log.md"
        if not log.exists():
            gpu_step_script(self.work)
            log.write_text("| Step | Started (UTC) | Ended (UTC) | GPU-hours | Running total | Outcome |\n"
                           "|---|---|---|---|---|---|\n| earlier | - | - | 0.00 | 0.00 | SUCCEEDED rc=0 |\n")
        proc = subprocess.run([sys.executable, str(WRAPPER), "ext-launch", "--adapter", standin.NAME,
                               "--profile", "smoke", "--request", str(self.request(gpu=gpu)), "--gpu-log", str(log),
                               "--out-record", str(self.tmp / f"launch-{gpu}.json")],
                              cwd=self.tmp, env=self.env, capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual([line for line in proc.stderr.splitlines() if not line.startswith("warning: ")], [],
                         proc.stderr)                       # the host layout comes from the environment (EXT-85)
        return proc.stdout

    def test_rootless_user_and_cdi(self) -> None:
        cpu = self.launch(gpu=False)
        self.assertIn(" --user 0:0 ", cpu)
        self.assertNotIn("--device", cpu)
        gpu = self.launch(gpu=True)
        self.assertIn(" --user 0:0 ", gpu)
        self.assertIn(" --device nvidia.com/gpu=all ", gpu)
        self.assertNotIn("--gpus", gpu)
        self.assertNotIn("--runtime=nvidia", gpu)
        for flag in (False, True):
            self.assertIs(json.loads((self.tmp / f"launch-{flag}.json").read_text())["rootless"], True)


class DocsTest(unittest.TestCase):
    """README.md quotes the step 3a texts the code states, and the scenario pack contract of `gatedtrain.ainative.batch`;
    docs/gpu-step.md describes the backend's own GPU log rows."""

    def test_readme_and_gpu_step(self) -> None:
        from gatedtrain.ainative.backend import GPU_LOG_HEADER
        from gatedtrain.ainative.keeper import texts

        repo = Path(__file__).resolve().parents[1]
        readme = " ".join((repo / "README.md").read_text(encoding="utf-8").split())
        for text in (texts.KEEPER_GUARANTEE, texts.KEEPER_ROOT_DOCKER_GUARANTEE, texts.COMPUTE_BACKEND_SOURCE):
            self.assertIn(f'"{text}"', readme)
        for value in ('json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)',
                      "a str, an int within ±(2**53-1), a bool or null", "still exposes an empty `generators` mapping",
                      "--user 0:0 --cap-drop ALL --security-opt no-new-privileges --read-only --network none "
                      "--log-driver none", "sudo bash gatedtrain-host-setup.sh", "rootless = true",
                      "`project run <stage>` reattaches"):
            self.assertIn(value, readme)
        gpu_step = (repo / "docs" / "gpu-step.md").read_text(encoding="utf-8")
        self.assertIn("### Rows the local backend writes", gpu_step)
        self.assertIn(GPU_LOG_HEADER.splitlines()[0], gpu_step)


class StandinPluginTest(unittest.TestCase):
    """The stand-in distribution (built and installed only for the hand check on a real host) names objects that
    exist: every stage entry point is a stage of that name, the scenario is the pack's, the adapters are plugins."""

    def test_entry_points_resolve(self) -> None:
        import tomllib

        doc = tomllib.loads((FIXTURES / "ainative" / "standin-plugin" / "pyproject.toml").read_text())
        eps = doc["project"]["entry-points"]

        def load(value: str):
            module, attr = value.split(":")
            return getattr(importlib.import_module(module), attr)

        stages = {name: load(v) for name, v in eps["gatedtrain.stages"].items()}
        self.assertEqual({name: cls.name for name, cls in stages.items()}, {name: name for name in stages})
        self.assertLessEqual({"choose_task_and_base_files", "train_job"}, set(stages))
        self.assertIs(load(eps["gatedtrain.scenarios"]["synthetic"]), pack.SCENARIO)
        self.assertIs(load(eps["gatedtrain.adapters"][standin.NAME]), standin.PLUGIN)
        self.assertEqual(set(load(eps["gatedtrain.adapters"]["synthetic"])["adapters"]), {"synthetic"})
        self.assertEqual(doc["tool"]["setuptools"]["packages"], [])


class RunCliSwitchTest(unittest.TestCase):
    """The test switches of `run_cli.py` keep a CLI subprocess away from the real Docker CLI and `nvidia-smi`, even if
    a code path builds the production runtime or probe."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="gt-switch-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.mod = load_run_cli()
        from gatedtrain.ainative.keeper import inference, server

        for target, attr in ((DockerCLIRuntime, "_cli"), (NvidiaSmiProbe, "used_mib"),
                             (machine, "compute_backend"), (server, "serve"), (inference, "NvidiaSmiProbe")):
            patcher = mock.patch.object(target, attr, getattr(target, attr))      # restored after the test
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(subprocess, "run", side_effect=AssertionError("a process was started"))
        self.run_mock = patcher.start()
        self.addCleanup(patcher.stop)

    def assert_no_real_tool(self) -> None:
        with self.assertRaises(self.mod.RealToolUsed):
            DockerCLIRuntime(context=launch.DOCKER_CONTEXT).inspect("gatedtrain-x")
        with self.assertRaises(self.mod.RealToolUsed):
            DockerCLIRuntime(host="unix:///run/user/1/docker.sock").daemon()
        with self.assertRaises(self.mod.RealToolUsed):
            NvidiaSmiProbe().used_mib()
        self.run_mock.assert_not_called()

    def test_backend_switch(self) -> None:
        self.mod._fake_backend_runtime(f"fake:{self.tmp}")
        self.assert_no_real_tool()
        data = self.tmp / "d.txt"
        data.write_text("x")
        project = create_project(self.tmp / "p", goal="g", data=data, pipeline="p",
                                 budget={"gpu_hours": 1.0, "llm_usd": 0})
        b = machine.compute_backend(project)
        self.assertIsInstance(b.runtime, FakeRuntime)
        self.assertNotIsInstance(b.probe, NvidiaSmiProbe)
        self.assertEqual(b.probe.used_mib(), 0)

    def test_keeper_switch(self) -> None:
        from gatedtrain.ainative.keeper import inference

        self.mod._fake_keeper_runtime(f"fake:{KEEPER_IMAGE}")
        self.assert_no_real_tool()
        self.assertEqual(inference.NvidiaSmiProbe("nvidia-smi").used_mib(), 0)


if __name__ == "__main__":
    unittest.main()
