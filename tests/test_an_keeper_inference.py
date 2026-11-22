"""AN-25, AN-8, AN-9, AN-10, AN-13: the keeper's inference container (design 05 §18.9, §18.10).

A keeper started with an inference image runs a scenario pack's batch generator in a container of its own rootless
Docker daemon: one container per arm, no network, a read-only root, the predictions as its only output. The tests use
`gatedtrain.testing.fake_runtime.FakeRuntime`, never Docker or a GPU:

- `KeeperInference` on its own (start checks, the container's options, the failure mapping, cleanup);
- `KeeperState` in process with a `KeeperInference` on a fake runtime (crashes after the grant, refusals before it);
- the keeper as a subprocess (`tests/_keeper.py`) with `GATEDTRAIN_TEST_KEEPER_RUNTIME=fake:<image id>:<state dir>`
  (`tests/fixtures/ainative/run_cli.py`): the fake runtime's containers run the synthetic pack's batch script on the
  host through `subprocess_handler`.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from _keeper import KeeperProcess, KeeperStartError, skip_unless_linux, write_host_tree

from test_an_keeper_attempts import TOY, arm
from test_an_keeper_intake import confirm_args, intake_args, project_id
from gatedtrain.ainative import batch, runtime as runtime_mod, scorers, signing, trust_stages
from gatedtrain.ainative.alpha_investing import PLANNING_ALPHA
from gatedtrain.ainative.keeper import envcheck, protocol, rootful, server
from gatedtrain.ainative.keeper.protocol import RefusedAfterSpend
from gatedtrain.ainative.keeper.state import KeeperState
from gatedtrain.ainative.runtime import DaemonInfo, DockerCLIRuntime, RuntimeFailure
from gatedtrain.canonical import sha256_bytes
from gatedtrain.errors import Refused
from gatedtrain.ext import launch
from gatedtrain.testing import synthetic_pack as pack
from gatedtrain.testing.fake_runtime import FakeRuntime, subprocess_handler

try:
    from gatedtrain.ainative.keeper import inference
    from gatedtrain.ainative.keeper.inference import KEEPER_ENV, InferenceConfig, KeeperInference
except ImportError:                       # RED: the module does not exist yet; every test below fails
    inference = None
    KEEPER_ENV = InferenceConfig = KeeperInference = None

IMAGE = "sha256:" + "1" * 64
OTHER_IMAGE = "sha256:" + "2" * 64
SECRET = "zz-sealed-input-text"
DEP = {"sampler": dict(pack.SAMPLER), "eval_seeds": list(pack.EVAL_SEEDS)}
SOURCE = "the driver's job records (cooperative, not checked by the keeper)"
_ORIGINAL_EPS = trust_stages._scenario_entry_points


def _tmpdir(case: unittest.TestCase, prefix: str = "gk-inf-") -> Path:
    d = Path(tempfile.mkdtemp(prefix=prefix))
    case.addCleanup(shutil.rmtree, d, True)
    return d


def docker_socket(root: Path) -> Path:
    """A stand-in for the keeper's rootless socket: a file in a 0700 directory owned by this uid."""
    d = root / "dockerd"
    d.mkdir(mode=0o700)
    os.chmod(d, 0o700)
    sock = d / "docker.sock"
    sock.write_bytes(b"")
    return sock


class Clock:
    """A fake monotonic clock that `sleep` advances."""

    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


class Probe:
    def __init__(self, mib: int | None) -> None:
        self.mib, self.calls = mib, 0

    def used_mib(self) -> int | None:
        self.calls += 1
        return self.mib


def write_table(where: Path, table: dict) -> dict:
    where.mkdir(parents=True, exist_ok=True)
    os.chmod(where, 0o755)
    data = json.dumps(table, sort_keys=True).encode("utf-8")
    (where / "table.json").write_bytes(data)
    return {"path": str(where / "table.json"), "sha256": sha256_bytes(data)}


def batch_model(files: list[dict], label: str, variant: str | None = None) -> dict:
    rec = {"schema": trust_stages.SCHEMA_MODEL_REF, "schema_version": 1, "generator": pack.BATCH_GENERATOR,
           "label": label, "files": files}
    if variant is not None:
        rec["variant"] = variant
    return rec


def batch_registration(sealed_id: str, base_sha: str) -> dict:
    spec = scorers.resolve("exact_match")
    v = {"scorer": "exact_match", "direction": spec.direction, "margin": 0.05, "alpha": PLANNING_ALPHA}
    return trust_stages._registration(v, sealed_id, DEP, base_sha, ("synthetic", pack.BATCH_GENERATOR),
                                      scorers.evaluator_entry(spec, {}))


class _ScenarioEP:
    name = "synthetic"
    value = "gatedtrain.testing.synthetic_pack:SCENARIO"

    def __init__(self, gen: batch.BatchGenerator) -> None:
        self.gen = gen

    def load(self):
        return pack.Scenario(pack.SCENARIO.name, dict(pack.SCENARIO.generators), pack.SCENARIO.pipeline,
                             pack.SCENARIO.scorers, batch_generators={pack.BATCH_GENERATOR: self.gen})


def inject_batch(case: unittest.TestCase, gen: batch.BatchGenerator = pack.BATCH) -> None:
    trust_stages._scenario_entry_points = lambda: [_ScenarioEP(gen)]
    trust_stages._reset_scenario_snapshot()
    case.addCleanup(trust_stages._reset_scenario_snapshot)
    case.addCleanup(setattr, trust_stages, "_scenario_entry_points", _ORIGINAL_EPS)


def duplicate_handler(spec, host_path) -> int:
    """The pack's script; for a candidate, its last line is then replaced by a copy of its first (the count is right,
    one item and seed repeats)."""
    rc = subprocess_handler(spec, host_path)
    if host_path("/in/model").name != "baseline":
        out = host_path("/out") / "predictions.jsonl"
        lines = out.read_bytes().split(b"\n")[:-1]
        out.write_bytes(b"\n".join([*lines[:-1], lines[0]]) + b"\n")
    return rc


# ---------------------------------------------------------------------------------------------------------------------
# KeeperInference on its own
# ---------------------------------------------------------------------------------------------------------------------

@skip_unless_linux
class InferenceUnitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = _tmpdir(self)
        self.state = self.tmp / "state"
        self.state.mkdir(mode=0o700)
        self.sock = docker_socket(self.tmp)
        self.clock = Clock()
        self.rt = FakeRuntime({IMAGE: subprocess_handler}, image_ids={IMAGE: IMAGE}, socket=str(self.sock))
        self.model_dir = self.state / "models" / "abc" / "candidate"
        self.model_dir.mkdir(parents=True, mode=0o700)
        self.entry = write_table(self.model_dir, {"a": "x"})
        self.model = batch_model([self.entry], "candidate")

    def config(self, **kw) -> InferenceConfig:
        return InferenceConfig(image_id=kw.pop("image_id", IMAGE), docker_host=f"unix://{self.sock}", **kw)

    def ki(self, cfg=None, *, rt=None, probe=None) -> KeeperInference:
        return KeeperInference(cfg or self.config(), rt or self.rt, self.state, probe=probe or Probe(0),
                               sleep=self.clock.sleep, clock=self.clock)

    def items(self) -> list[dict]:
        return [{"input": f"a-{SECRET}-{i}"} for i in range(3)]

    def run_ok(self, ki=None, gen=pack.BATCH) -> list[dict]:
        ki = ki or self.ki()
        return ki.run_batch(gen, self.model, self.items(), dict(pack.SAMPLER), list(pack.EVAL_SEEDS))

    def leftovers(self) -> list[str]:
        run = self.state / "run"
        return sorted(p.name for p in run.iterdir()) if run.exists() else []

    # -- interface ----------------------------------------------------------------------------------------------------

    def test_keeper_env(self):
        self.assertEqual(KEEPER_ENV, (("DO_NOT_TRACK", "1"), ("HF_HUB_OFFLINE", "1"), ("HOME", "/tmp"),
                                      ("TRANSFORMERS_OFFLINE", "1"), ("VLLM_NO_USAGE_STATS", "1")))

    def test_config(self):
        cfg = self.config()
        self.assertEqual((cfg.eval_trees, cfg.image_ref, cfg.docker), ((), None, Path("/usr/bin/docker")))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cfg.image_id = OTHER_IMAGE
        for bad in ("sha256:" + "1" * 63, "busybox", "sha256:" + "A" * 64):
            with self.subTest(bad), self.assertRaises(ValueError):
                InferenceConfig(image_id=bad, docker_host="unix:///x")
        with self.assertRaises(ValueError):
            InferenceConfig(image_id=IMAGE, docker_host="unix:///x", eval_trees=(Path("relative"),))

    def test_client_waits_long_enough(self):               # pre-flight finding 11: 900 + 2 x 1200 + 600
        self.assertGreaterEqual(protocol.LONG_OP_TIMEOUT_S, 3900)
        self.assertEqual(KeeperClientTimeout.evaluate(), protocol.LONG_OP_TIMEOUT_S)

    # -- the container ------------------------------------------------------------------------------------------------

    def test_container_options(self):
        out = self.run_ok()
        self.assertEqual(len(out), 3 * len(pack.EVAL_SEEDS))
        self.assertEqual([p["prediction"] for p in out[:3]], ["x", "x", "x"])
        (spec,) = self.rt.runs
        self.assertRegex(spec.name, r"^gatedtrain-keeper-[0-9a-f]+$")
        self.assertEqual(spec.image, IMAGE)
        self.assertEqual(spec.argv, batch.container_argv(pack.BATCH))
        self.assertEqual(spec.env, KEEPER_ENV)
        self.assertEqual((spec.user, spec.network, spec.read_only_root, spec.log_driver, spec.shm_size),
                         ("0:0", "none", True, "none", "8g"))
        self.assertEqual(spec.tmpfs, (("/tmp", launch.TMPFS_OPTIONS),))
        self.assertEqual(spec.gpu_options, ())
        self.assertEqual(spec.extra_options, ())
        self.assertEqual(spec.labels, (inference.KEEPER_LABEL,))
        mounts = {m.container: m for m in spec.mounts}
        self.assertEqual(set(mounts), {"/in/items.jsonl", "/in/sampler.json", "/in/model",
                                       f"/gen/{pack.BATCH.script.name}", "/out"})
        for path, m in mounts.items():
            self.assertEqual(m.read_only, path != "/out", path)
        self.assertEqual(mounts["/in/model"].host, str(self.model_dir))
        self.assertEqual(mounts[f"/gen/{pack.BATCH.script.name}"].host, str(pack.BATCH.script))
        run_dir = Path(mounts["/out"].host).parent
        self.assertEqual(run_dir.parent, self.state / "run")
        self.assertEqual(self.leftovers(), [])                   # the run dir is removed
        self.assertEqual(self.rt.containers("=".join(inference.KEEPER_LABEL)), ())   # and the container
        argv = runtime_mod.docker_run_argv(dataclasses.replace(spec, mounts=()), host="unix:///x")
        for flag in ("--read-only", "--network", "--log-driver", "--cap-drop", "--pull"):
            self.assertIn(flag, argv)

    def test_gpu_generator_gets_cdi(self):
        gen = dataclasses.replace(pack.BATCH, gpu=True)
        self.run_ok(gen=gen)
        self.assertEqual(self.rt.runs[0].gpu_options, launch.CDI_GPU_OPTIONS)

    def test_eval_trees_mounted_read_only_when_asked(self):
        trees = (self.tmp / "opt" / "keeper",)
        trees[0].mkdir(parents=True)
        ki = self.ki(self.config(eval_trees=trees))
        self.run_ok(ki)
        self.assertNotIn(str(trees[0]), {m.container for m in self.rt.runs[0].mounts})
        self.run_ok(ki, gen=dataclasses.replace(pack.BATCH, eval_trees=True))
        m = {m.container: m for m in self.rt.runs[1].mounts}[str(trees[0])]
        self.assertEqual((m.host, m.read_only), (str(trees[0]), True))

    def test_items_never_in_args_or_env(self):
        self.run_ok()
        (spec,) = self.rt.runs
        argv = runtime_mod.docker_run_argv(dataclasses.replace(spec, mounts=()), host="unix:///x")
        blob = json.dumps([argv, spec.env, spec.name, [dataclasses.asdict(m) for m in spec.mounts]])
        self.assertNotIn(SECRET, blob)
        self.assertNotIn("a-", " ".join(spec.argv))

    def test_last_run(self):
        ki = self.ki()
        self.run_ok(ki)
        self.assertEqual(ki.last_run, {"arm": None, "image_id": IMAGE, "seconds": 60, "gpu_seconds": 0, "rc": 0})
        self.run_ok(ki, gen=dataclasses.replace(pack.BATCH, gpu=True))
        self.assertEqual((ki.last_run["seconds"], ki.last_run["gpu_seconds"]), (60, 60))

    # -- crashes (AN-9) ----------------------------------------------------------------------------------------------

    def crash(self, ki=None, gen=pack.BATCH) -> Refused:
        with self.assertRaises(Refused) as cm:
            self.run_ok(ki, gen=gen)
        p = cm.exception.problems[0]
        self.assertEqual(p.rule, "AN-9")
        for text in (p.message, p.fix):
            self.assertNotIn(SECRET, text)
            self.assertNotIn(str(self.state), text)
            self.assertNotIn(str(self.tmp), text)
        self.assertEqual(self.leftovers(), [])
        self.assertEqual(self.rt.containers("=".join(inference.KEEPER_LABEL)), ())
        return cm.exception

    def test_nonzero_rc_crash(self):
        self.rt = FakeRuntime({IMAGE: subprocess_handler}, image_ids={IMAGE: IMAGE}, socket=str(self.sock),
                              outcomes=[3])
        ki = self.ki()
        exc = self.crash(ki)
        self.assertIn("the inference container exited with rc 3", exc.problems[0].message)
        self.assertEqual(ki.last_run["rc"], 3)

    def test_timeout_crash(self):
        self.rt = FakeRuntime({IMAGE: subprocess_handler}, image_ids={IMAGE: IMAGE}, socket=str(self.sock),
                              outcomes=["hang"])
        exc = self.crash(gen=dataclasses.replace(pack.BATCH, time_limit_s=5))
        self.assertIn("ran past its time limit", exc.problems[0].message)
        self.assertGreaterEqual(self.clock.t, 5)
        self.assertLess(self.clock.t, 10)

    def test_vanished_container_crash(self):
        self.rt = FakeRuntime({IMAGE: subprocess_handler}, image_ids={IMAGE: IMAGE}, socket=str(self.sock),
                              outcomes=["vanish"])
        self.crash()

    def test_bad_predictions_crash(self):
        self.rt = FakeRuntime({IMAGE: duplicate_handler}, image_ids={IMAGE: IMAGE}, socket=str(self.sock))
        exc = self.crash()
        self.assertIn("repeats an item and seed", exc.problems[0].message)

    def test_runtime_failure_is_a_crash_without_paths(self):
        class Failing(FakeRuntime):
            def run(self, spec):
                raise RuntimeFailure("run", 125)
        self.rt = Failing({IMAGE: subprocess_handler}, image_ids={IMAGE: IMAGE}, socket=str(self.sock))
        exc = self.crash()
        self.assertIn("exit code 125", exc.problems[0].message)

    def test_os_error_is_a_crash_naming_the_class_only(self):
        ki = self.ki()
        with mock.patch.object(batch, "write_items", side_effect=OSError(f"disk full at {self.state}")), \
                mock.patch.object(inference, "write_items", side_effect=OSError(f"disk full at {self.state}")):
            exc = self.crash(ki)
        self.assertIn("OSError", exc.problems[0].message)
        self.assertEqual(self.rt.runs, [])

    def test_model_without_files_is_a_crash(self):
        self.model = batch_model([], "candidate")
        self.crash()

    def test_daemon_rechecked_before_each_start(self):         # Task 3 carry
        ki = self.ki()
        self.run_ok(ki)
        self.rt._daemon = DaemonInfo(False, os.getuid(), str(self.sock))
        with self.assertRaises(Refused) as cm:
            self.run_ok(ki)
        self.assertEqual(cm.exception.problems[0].rule, "AN-13")
        self.assertEqual(len(self.rt.runs), 1)
        self.assertEqual(self.leftovers(), [])

    # -- start (AN-13) -----------------------------------------------------------------------------------------------

    def rules(self, problems) -> list[str]:
        return [p.rule for p in problems]

    def test_clean_start(self):
        self.assertEqual(self.ki().start_problems(), [])

    def test_start_image_changed(self):
        self.assertEqual(self.rules(self.ki(self.config(image_id=OTHER_IMAGE)).start_problems()), ["AN-13"])
        cfg = self.config(image_ref="registry.example/cuda@sha256:" + "3" * 64)
        self.assertEqual(self.rules(self.ki(cfg).start_problems()), ["AN-13"])     # the daemon holds no such ref
        rt = FakeRuntime(image_ids={cfg.image_ref: IMAGE}, socket=str(self.sock))
        self.assertEqual(self.ki(cfg, rt=rt).start_problems(), [])

    def test_start_daemon_not_rootless(self):                   # pre-flight finding 3
        rt = FakeRuntime(image_ids={IMAGE: IMAGE}, socket=str(self.sock), rootless=False)
        problems = self.ki(rt=rt).start_problems()
        self.assertEqual(self.rules(problems), ["AN-13"])
        self.assertIn("not rootless", problems[0].message)

    def test_start_socket_owned_by_another_uid(self):
        rt = FakeRuntime(image_ids={IMAGE: IMAGE}, socket=str(self.sock), socket_uid=os.getuid() + 1)
        problems = self.ki(rt=rt).start_problems()
        self.assertEqual(self.rules(problems), ["AN-13"])
        self.assertIn("owned by uid", problems[0].message)

    def test_start_socket_directory_not_0700(self):
        os.chmod(self.sock.parent, 0o755)
        problems = self.ki().start_problems()
        self.assertEqual(self.rules(problems), ["AN-13"])
        self.assertIn("0700", problems[0].message)

    def test_start_not_a_unix_socket(self):
        rt = FakeRuntime(image_ids={IMAGE: IMAGE}, socket=None)
        self.assertEqual(self.rules(self.ki(rt=rt).start_problems()), ["AN-13"])

    def test_start_waits_for_the_socket(self):                  # pre-flight finding 20
        self.sock.unlink()
        problems = self.ki().start_problems()
        self.assertIn("AN-13", self.rules(problems))
        self.assertGreaterEqual(self.clock.t, 60)
        self.assertLess(self.clock.t, 62)
        # a socket that appears while the keeper waits
        self.clock.t, ki = 0.0, self.ki()

        def appear(s: float) -> None:
            Clock.sleep(self.clock, s)
            if self.clock.t >= 3:
                self.sock.write_bytes(b"")
        ki.sleep = appear
        self.assertEqual(ki.start_problems(), [])
        self.assertLess(self.clock.t, 5)

    def test_start_daemon_unreachable(self):
        class Down(FakeRuntime):
            def daemon(self):
                raise RuntimeFailure("daemon", 1)
        rt = Down(image_ids={IMAGE: IMAGE}, socket=str(self.sock))
        self.assertEqual(self.rules(self.ki(rt=rt).start_problems()), ["AN-13"])

    def test_start_eval_tree_writable(self):
        tree = self.tmp / "keeper-tree"
        (tree / "lib").mkdir(parents=True)
        f = tree / "lib" / "mod.py"
        f.write_text("x = 1\n")
        ki = self.ki(self.config(eval_trees=(tree,)))
        self.assertEqual(ki.start_problems(), [])
        os.chmod(f, 0o666)
        problems = ki.start_problems()
        self.assertEqual(self.rules(problems), ["AN-13"])
        self.assertIn("writable by every user", problems[0].message)

    def test_start_missing_eval_tree(self):
        ki = self.ki(self.config(eval_trees=(self.tmp / "nowhere",)))
        self.assertEqual(self.rules(ki.start_problems()), ["AN-13"])

    def test_start_docker_cli_checked(self):
        cli = self.tmp / "docker"
        cli.write_bytes(b"#!/bin/sh\n")
        os.chmod(cli, 0o777)
        rt = DockerCLIRuntime(host=f"unix://{self.sock}", docker=str(cli))
        with mock.patch.object(rt, "daemon", return_value=DaemonInfo(True, os.getuid(), str(self.sock))), \
                mock.patch.object(rt, "image_id", return_value=IMAGE):
            problems = self.ki(self.config(docker=cli), rt=rt).start_problems()
            self.assertEqual(self.rules(problems), ["AN-13"])
            self.assertIn("writable by every user", problems[0].message)
            os.chmod(cli, 0o755)
            self.assertEqual(self.ki(self.config(docker=cli), rt=rt).start_problems(), [])
            missing = self.config(docker=self.tmp / "no-docker")
            self.assertEqual(self.rules(self.ki(missing, rt=rt).start_problems()), ["AN-13"])

    def test_remove_leftovers(self):
        left = self.state / "run" / "deadbeef"
        left.mkdir(parents=True)
        (left / "items.jsonl").write_bytes(b"{}\n")
        rt = FakeRuntime({IMAGE: subprocess_handler}, image_ids={IMAGE: IMAGE}, socket=str(self.sock),
                         outcomes=["hang"])
        spec = runtime_mod.RunSpec(name="gatedtrain-keeper-old", image=IMAGE, argv=("x",), env=(), mounts=(),
                                   user="0:0", labels=(inference.KEEPER_LABEL,))
        rt.run(spec)
        self.ki(rt=rt).remove_leftovers()
        self.assertEqual(self.leftovers(), [])
        self.assertEqual(rt.inspect("gatedtrain-keeper-old").status, "missing")

    # -- ready (before the grant) ------------------------------------------------------------------------------------

    def refused_ready(self, ki, gen, rule) -> Refused:
        with self.assertRaises(Refused) as cm:
            ki.ready(gen)
        self.assertNotIsInstance(cm.exception, RefusedAfterSpend)
        self.assertEqual(self.rules(cm.exception.problems), [rule])
        return cm.exception

    def test_ready(self):
        self.assertIsNone(self.ki().ready(pack.BATCH))

    def test_ready_image_changed(self):
        ki = self.ki()
        ki.ready(pack.BATCH)
        self.rt.image_ids[IMAGE] = OTHER_IMAGE
        self.refused_ready(ki, pack.BATCH, "AN-13")

    def test_ready_daemon_changed(self):
        ki = self.ki()
        self.rt._daemon = DaemonInfo(False, os.getuid(), str(self.sock))
        self.refused_ready(ki, pack.BATCH, "AN-13")

    def test_ready_eval_trees_needed(self):
        exc = self.refused_ready(self.ki(), dataclasses.replace(pack.BATCH, eval_trees=True), "AN-25")
        self.assertIn("eval tree", exc.problems[0].message)

    def test_ready_script_not_in_the_environment(self):
        script = self.tmp / "gen.py"
        script.write_text("print(1)\n")
        self.refused_ready(self.ki(), dataclasses.replace(pack.BATCH, script=script), "AN-13")

    def test_ready_time_bound(self):                           # pre-flight finding 11
        ki = self.ki()
        ki.ready(dataclasses.replace(pack.BATCH, time_limit_s=1200))     # 900 + 2400 + 600 = 3900
        exc = self.refused_ready(ki, dataclasses.replace(pack.BATCH, time_limit_s=1201), "AN-25")
        self.assertIn("time limit", exc.problems[0].message)
        with mock.patch.object(protocol, "LONG_OP_TIMEOUT_S", 7200.0):
            ki.ready(dataclasses.replace(pack.BATCH, time_limit_s=2800))

    def test_ready_gpu_busy(self):                             # Review Focus 5
        probe = Probe(5000)
        exc = self.refused_ready(self.ki(probe=probe), dataclasses.replace(pack.BATCH, gpu=True), "AN-25")
        self.assertIn("the GPU is busy", exc.problems[0].message)
        self.assertGreaterEqual(self.clock.t, runtime_mod.GPU_WAIT_S - runtime_mod.GPU_POLL_S)
        self.assertGreater(probe.calls, 1)

    def test_ready_gpu_tool_missing(self):                      # pre-flight finding 25
        self.refused_ready(self.ki(probe=Probe(None)), dataclasses.replace(pack.BATCH, gpu=True), "AN-25")

    def test_ready_cpu_generator_does_not_probe(self):
        probe = Probe(None)
        self.ki(probe=probe).ready(pack.BATCH)
        self.assertEqual(probe.calls, 0)

    # -- environment record (AN-13) ---------------------------------------------------------------------------------

    def test_environment(self):
        tree = self.tmp / "tree"
        (tree / "bin").mkdir(parents=True)
        (tree / "bin" / "python").write_bytes(b"interp")
        (tree / "lib.txt").write_bytes(b"lib")
        env = self.ki(self.config(eval_trees=(tree,))).environment()
        self.assertEqual(set(env), {"image_id", "docker_host", "eval_trees", "docker_cli_sha256"})
        self.assertEqual((env["image_id"], env["docker_host"], env["docker_cli_sha256"]),
                         (IMAGE, f"unix://{self.sock}", None))
        self.assertEqual(env["eval_trees"], [envcheck.eval_tree_record(tree)])
        self.assertEqual(env["eval_trees"][0]["path"], str(tree))
        before = env["eval_trees"][0]["files_sha256"]
        (tree / "lib.txt").write_bytes(b"changed")
        self.assertNotEqual(envcheck.eval_tree_record(tree)["files_sha256"], before)


class KeeperClientTimeout:
    @staticmethod
    def evaluate() -> float:
        return protocol.KeeperClient(Path("/nonexistent")).timeout_for("evaluate")


@skip_unless_linux
class EvalTreeTest(unittest.TestCase):
    """`envcheck.eval_tree_problems`: `path_problems` over every node plus links that stay inside the eval trees."""

    def setUp(self):
        self.tmp = _tmpdir(self)
        self.a = self.tmp / "opt" / "python"
        self.b = self.tmp / "opt" / "eval-venv"
        (self.a / "bin").mkdir(parents=True)
        (self.b / "bin").mkdir(parents=True)
        (self.a / "bin" / "python3.12").write_bytes(b"interp")

    def problems(self, trees=None):
        return envcheck.eval_tree_problems(trees or (self.a, self.b), os.getuid())

    def test_clean(self):
        (self.b / "bin" / "python").symlink_to(self.a / "bin" / "python3.12")   # a venv's interpreter link
        self.assertEqual(self.problems(), [])

    def test_link_into_another_configured_tree_only(self):
        (self.b / "bin" / "python").symlink_to(self.a / "bin" / "python3.12")
        self.assertEqual([p.rule for p in self.problems((self.b,))], ["AN-13"])

    def test_link_leaving_the_trees(self):
        outside = self.tmp / "outside.py"
        outside.write_text("")
        (self.b / "bin" / "evil").symlink_to(outside)
        problems = self.problems()
        self.assertEqual([p.rule for p in problems], ["AN-13"])
        self.assertIn("outside the eval trees", problems[0].message)

    def test_dangling_link(self):
        (self.b / "bin" / "gone").symlink_to(self.tmp / "missing")
        self.assertEqual([p.rule for p in self.problems()], ["AN-13"])

    def test_writable_directory(self):
        os.chmod(self.b / "bin", 0o777)
        self.assertTrue(any("writable by every user" in p.message for p in self.problems()))

    def test_group_writable_file(self):
        f = self.a / "bin" / "python3.12"
        os.chmod(f, 0o664)
        ok = envcheck.path_problems([f], os.getuid()) == []          # a user-private group may write
        self.assertEqual(self.problems() == [], ok)

    def test_not_a_directory(self):
        f = self.tmp / "file"
        f.write_text("")
        self.assertEqual([p.rule for p in self.problems((f,))], ["AN-13"])


# ---------------------------------------------------------------------------------------------------------------------
# KeeperState with an inference container, in process
# ---------------------------------------------------------------------------------------------------------------------

@skip_unless_linux
class StateInferenceTest(unittest.TestCase):
    gen = pack.BATCH

    def setUp(self):
        self.tmp = _tmpdir(self)
        self.dir = self.tmp / "state"
        self.dir.mkdir(mode=0o700)
        self.out = self.tmp / "out"
        self.out.mkdir(mode=0o755)
        self.sock = docker_socket(self.tmp)
        self.clock = Clock()
        self.probe = Probe(0)
        inject_batch(self, self.gen)
        self.rt = self.runtime()
        self.base = batch_model([write_table(self.out / "base", {})], "base")
        self.state = self.open()
        self.pid = project_id("p1")
        self.state.open_project(self.pid)
        r = self.state.intake(self.pid, TOY, sha256_bytes(TOY))
        self.train, self.sealed = r["train_rows"], r["sealed_id"]
        args = confirm_args()
        sentence = args.pop("sentence")
        self.state.confirm_metric(self.pid, args, sentence, os.getuid())
        self.reg = self.state.register(self.pid, batch_registration(r["sealed_id"], arm(self.base)["sha256"]))["hash"]

    def runtime(self, handler=subprocess_handler, outcomes=()) -> FakeRuntime:
        return FakeRuntime({IMAGE: handler}, image_ids={IMAGE: IMAGE}, socket=str(self.sock), outcomes=outcomes)

    def open(self, inference_on: bool = True) -> KeeperState:
        ki = None
        if inference_on:
            ki = KeeperInference(InferenceConfig(image_id=IMAGE, docker_host=f"unix://{self.sock}"), self.rt,
                                 self.dir, probe=self.probe, sleep=self.clock.sleep, clock=self.clock)
            self.ki = ki
        st = KeeperState(self.dir, owner_uid=os.getuid(), inference=ki)
        self.addCleanup(st.close)
        return st

    def candidate(self, variant: str = "c", table: dict | None = None) -> dict:
        table = pack.fit(self.train, 0.0) if table is None else table
        return batch_model([write_table(self.out / variant, table)], "candidate", variant)

    def evaluate(self, cand: dict) -> dict:
        return self.state.evaluate(self.pid, self.reg, arm(self.base), arm(cand))

    def left(self, sub: str) -> list[Path]:
        d = self.dir / sub
        return sorted(d.rglob("*")) if d.exists() else []

    def attempt_file(self, j: int = 1) -> dict:
        return json.loads((self.dir / "sets" / self.sealed.removeprefix("sha256:") / "attempts" /
                           f"{j:06d}.json").read_text())

    def test_pass_and_inference_records(self):
        first = self.evaluate(self.candidate("c0", {}))          # answers nothing, as the base: not a PASS
        self.assertEqual(len(self.rt.runs), 2)                    # one container per arm
        models = [{m.container: m.host for m in s.mounts}["/in/model"] for s in self.rt.runs]
        self.assertEqual([Path(m).name for m in models], ["baseline", "candidate"])
        for m in models:                                          # the keeper's copies, never the originals
            self.assertTrue(m.startswith(str(self.dir / "models") + os.sep), m)
        self.assertNotEqual(self.state.gate(self.pid, first["attempt_id"])["summary"]["record"]["status"], "PASS")
        rows = self.attempt_file()["gate_record"]["record"]["inference"]
        self.assertEqual([r["arm"] for r in rows], ["baseline", "candidate"])
        gpu = 60 if self.gen.gpu else 0
        self.assertEqual(rows[0], {"arm": "baseline", "image_id": IMAGE, "seconds": 60, "gpu_seconds": gpu, "rc": 0})
        self.assertEqual(self.left("models"), [])
        self.assertEqual(self.left("run"), [])
        # the baseline is scored once per set and key: the next attempt runs the candidate only
        att = self.evaluate(self.candidate())
        self.assertEqual(len(self.rt.runs), 3)
        self.assertEqual([r["arm"] for r in self.attempt_file(2)["inference"]], ["candidate"])
        self.assertEqual(self.state.gate(self.pid, att["attempt_id"])["summary"]["record"]["status"], "PASS")
        self.assertEqual([r["arm"] for r in self.attempt_file(2)["gate_record"]["record"]["inference"]],
                         ["candidate"])

    def assert_crash_after_grant(self, cand: dict, text: str) -> None:   # Review Focus 3
        before = self.state.alpha(self.pid)
        with self.assertRaises(RefusedAfterSpend) as cm:
            self.evaluate(cand)
        p = cm.exception.problems[0]
        self.assertEqual(p.rule, "AN-9")
        self.assertIn(text, p.message)
        self.assertNotIn(str(self.dir), p.message + p.fix)
        result = cm.exception.result
        self.assertNotEqual(result["summary"]["record"]["status"], "PASS")
        self.assertEqual(result["settle"]["record"]["passed"], False)
        after = self.state.alpha(self.pid)
        self.assertLess(after["wealth"], before["wealth"])        # alpha spent
        self.assertEqual([e["record"]["entry"] for e in after["entries"]], ["grant", "settle"])
        self.assertEqual(self.left("models"), [])
        self.assertEqual(self.left("run"), [])
        self.assertEqual(self.rt.containers("=".join(inference.KEEPER_LABEL)), ())
        rows = self.attempt_file()["gate_record"]["record"]["inference"]
        self.assertEqual(rows[-1]["arm"], "candidate")

    def test_bad_predictions_crash(self):
        self.rt = self.runtime(duplicate_handler)
        self.state.close()
        self.state = self.open()
        self.assert_crash_after_grant(self.candidate(), "repeats an item and seed")

    def test_nonzero_rc_crash(self):
        self.rt = self.runtime(outcomes=["ok", 7])
        self.state.close()
        self.state = self.open()
        self.assert_crash_after_grant(self.candidate(), "rc 7")

    def test_timeout_crash(self):
        self.rt = self.runtime(outcomes=["ok", "hang"])
        self.state.close()
        self.state = self.open()
        self.assert_crash_after_grant(self.candidate(), "ran past its time limit")
        self.assertGreaterEqual(self.clock.t, self.gen.time_limit_s)

    def assert_refused_before_grant(self, rule: str, text: str) -> None:
        before, records = self.state.alpha(self.pid), self.state.records(self.pid)
        with self.assertRaises(Refused) as cm:
            self.evaluate(self.candidate())
        self.assertNotIsInstance(cm.exception, RefusedAfterSpend)
        p = cm.exception.problems[0]
        self.assertEqual(p.rule, rule)
        self.assertIn(text, p.message)
        self.assertEqual(self.state.alpha(self.pid), before)
        self.assertEqual(self.state.records(self.pid), records)
        self.assertEqual(self.left("models"), [])
        self.assertEqual(self.left("run"), [])

    def test_image_id_changed_refused_before_grant(self):
        self.rt.image_ids[IMAGE] = OTHER_IMAGE
        self.assert_refused_before_grant("AN-13", "image")
        self.assertEqual(self.rt.runs, [])

    def test_daemon_changed_refused_before_grant(self):
        self.rt._daemon = DaemonInfo(False, os.getuid(), str(self.sock))
        self.assert_refused_before_grant("AN-13", "rootless")

    def test_no_inference_refuses_batch(self):                 # AN-8
        self.state.close()
        self.state = self.open(inference_on=False)
        self.assert_refused_before_grant("AN-8", "this keeper runs no inference container")

    def test_model_without_files_refused_before_grant(self):
        before = self.state.alpha(self.pid)
        with self.assertRaises(Refused) as cm:
            self.evaluate(batch_model([], "candidate", "nofiles"))
        self.assertNotIsInstance(cm.exception, RefusedAfterSpend)
        self.assertEqual(cm.exception.problems[0].field, "candidate.record.files")
        self.assertEqual(self.state.alpha(self.pid), before)

    def test_report_compute(self):
        att = self.evaluate(self.candidate())
        self.state.gate(self.pid, att["attempt_id"])
        for bad in ({"jobs": 1}, {"jobs": -1, "seconds": 0, "gpu_seconds": 0},
                    {"jobs": True, "seconds": 0, "gpu_seconds": 0}, {"jobs": 1.5, "seconds": 0, "gpu_seconds": 0},
                    {"jobs": 1, "seconds": 0, "gpu_seconds": 0, "extra": 1}, []):
            with self.subTest(bad=bad), self.assertRaises(Refused) as cm:
                self.state.report(self.pid, bad)
            self.assertEqual(cm.exception.problems[0].rule, "AN-10")
        env = self.state.report(self.pid, {"jobs": 1, "seconds": 300, "gpu_seconds": 240})["report"]
        report = env["record"]
        self.assertEqual(report["compute"], {
            "keeper": {"containers": 2, "seconds": 120, "gpu_seconds": 0},
            "backend": {"jobs": 1, "seconds": 300, "gpu_seconds": 240, "source": SOURCE}})
        md = trust_stages.render_markdown(report)
        self.assertIn("- keeper: 2 inference containers, 120 s, 0 GPU s (the keeper's signed records)", md)
        self.assertIn(f"- backend: 1 jobs, 300 s, 240 GPU s ({SOURCE})", md)


@skip_unless_linux
class StateGpuTest(StateInferenceTest):
    """The same flows with a GPU batch generator: the GPU wait comes before the grant."""

    gen = dataclasses.replace(pack.BATCH, gpu=True)

    def test_gpu_busy_before_grant(self):                       # Review Focus 5
        self.probe.mib = 50_000
        self.assert_refused_before_grant("AN-25", "the GPU is busy")
        self.assertEqual(self.rt.runs, [])

    def test_gpu_seconds_recorded(self):
        att = self.evaluate(self.candidate())
        self.state.gate(self.pid, att["attempt_id"])
        rows = self.attempt_file()["gate_record"]["record"]["inference"]
        self.assertEqual([r["gpu_seconds"] for r in rows], [60, 60])
        self.assertEqual(self.rt.runs[0].gpu_options, launch.CDI_GPU_OPTIONS)

    def test_report_compute(self):
        att = self.evaluate(self.candidate())
        self.state.gate(self.pid, att["attempt_id"])
        report = self.state.report(self.pid)["report"]["record"]
        self.assertEqual(report["compute"], {"keeper": {"containers": 2, "seconds": 120, "gpu_seconds": 120},
                                             "backend": None})
        md = trust_stages.render_markdown(report)
        self.assertIn("- keeper: 2 inference containers, 120 s, 120 GPU s", md)
        self.assertNotIn("- backend:", md)


@skip_unless_linux
class PerItemUnchangedTest(unittest.TestCase):
    """A per-item generator never touches the inference container, and its gate record lists no containers."""

    def test_per_item_report_compute(self):
        from test_an_keeper_attempts import BASE, model, registration
        tmp = _tmpdir(self)
        d = tmp / "state"
        d.mkdir(mode=0o700)
        inject_batch(self)
        st = KeeperState(d, owner_uid=os.getuid())
        self.addCleanup(st.close)
        pid = project_id("p")
        st.open_project(pid)
        r = st.intake(pid, TOY, sha256_bytes(TOY))
        args = confirm_args()
        sentence = args.pop("sentence")
        st.confirm_metric(pid, args, sentence, os.getuid())
        reg = st.register(pid, registration(r["sealed_id"]))["hash"]
        att = st.evaluate(pid, reg, arm(BASE), arm(model(pack.fit(r["train_rows"], 0.0), "v")))
        st.gate(pid, att["attempt_id"])
        report = st.report(pid)["report"]["record"]
        self.assertEqual(report["compute"], {"keeper": {"containers": 0, "seconds": 0, "gpu_seconds": 0},
                                             "backend": None})


@skip_unless_linux
class EnvironmentRecordTest(unittest.TestCase):
    """`serve` in process (a thread) with an inference image: the signed environment record holds `inference`."""

    def test_environment_record_has_inference(self):
        tmp = _tmpdir(self)
        state, sock = tmp / "state", tmp / "keeper.sock"
        state.mkdir(mode=0o700)
        dsock = docker_socket(tmp)
        tree = tmp / "tree"
        tree.mkdir()
        (tree / "lib.py").write_bytes(b"x = 1\n")
        cfg = InferenceConfig(image_id=IMAGE, docker_host=f"unix://{dsock}", eval_trees=(tree,))
        rt = FakeRuntime(image_ids={IMAGE: IMAGE}, socket=str(dsock))
        view = rootful.HostView(root=write_host_tree(tmp / "host"), root_uid=os.getuid())
        stop, errors = threading.Event(), []

        def run():
            try:
                server.serve(state, sock, os.getuid(), stop=stop, host_view=view, inference=cfg, runtime=rt)
            except BaseException as exc:
                errors.append(exc)
        with mock.patch.object(envcheck, "environment_record", return_value={"stub": 1}):
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            try:
                deadline = time.monotonic() + 10
                while True:
                    self.assertFalse(errors, errors)
                    self.assertLess(time.monotonic(), deadline, "the in-process keeper did not start")
                    if sock.exists():
                        try:
                            hello = protocol.KeeperClient(sock, timeout=2).call("hello", None, {})
                            break
                        except Refused:
                            pass
                    time.sleep(0.01)
            finally:
                stop.set()
                thread.join(10)
        expected = {"stub": 1, "inference": {"image_id": IMAGE, "docker_host": f"unix://{dsock}",
                                             "eval_trees": [envcheck.eval_tree_record(tree)],
                                             "docker_cli_sha256": None}}
        self.assertEqual(hello["environment_sha256"], envcheck.environment_sha256(expected))


# ---------------------------------------------------------------------------------------------------------------------
# The keeper as a subprocess
# ---------------------------------------------------------------------------------------------------------------------

@skip_unless_linux
class KeeperProcessInferenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = _tmpdir(self, "gk-infp-")
        self.sock = docker_socket(self.tmp)
        self.fake = self.tmp / "fake"
        self.fake.mkdir()
        self.out = self.tmp / "out"
        self.out.mkdir(mode=0o755)
        os.chmod(self.tmp, 0o755)

    def keeper(self, *, image: str = IMAGE, fake_image: str = IMAGE, extra: list[str] | None = None,
               inference_on: bool = True) -> KeeperProcess:
        args = ["--inference-image", image, "--docker-host", f"unix://{self.sock}", *(extra or [])] \
            if inference_on else []
        env = {"GATEDTRAIN_TEST_KEEPER_RUNTIME": f"fake:{fake_image}:{self.fake}"}
        kp = KeeperProcess(env=env, args=args)
        kp.__enter__()
        self.addCleanup(kp.__exit__, None, None, None)
        return kp

    def start_refused(self, **kw) -> KeeperStartError:
        with self.assertRaises(KeeperStartError) as cm:
            self.keeper(**kw)
        self.assertEqual(cm.exception.returncode, 3)
        return cm.exception

    def project(self, kp: KeeperProcess) -> tuple[str, dict, list]:
        c = kp.client
        pid = project_id("p1")
        c.call("open_project", pid, {})
        r = c.call("intake", pid, intake_args(TOY))
        c.call("confirm_metric", pid, confirm_args())
        base = batch_model([write_table(self.out / "base", {})], "base")
        reg = c.call("register", pid, {"registration": batch_registration(r["sealed_id"], arm(base)["sha256"])})
        return pid, {"registration_hash": reg["hash"], "baseline": arm(base)}, r["train_rows"]

    def test_batch_evaluation_through_container(self):
        kp = self.keeper()
        pid, args, train = self.project(kp)
        cand = batch_model([write_table(self.out / "cand", pack.fit(train, 0.0))], "candidate", "c")
        att = kp.client.call("evaluate", pid, {**args, "candidate": arm(cand)})
        gate = kp.client.call("gate", pid, {"attempt_id": att["attempt_id"]})
        self.assertEqual(gate["summary"]["record"]["status"], "PASS")
        runs = FakeRuntime(state_dir=self.fake).runs
        self.assertEqual(len(runs), 2)
        for spec in runs:
            self.assertEqual((spec.image, spec.user, spec.network, spec.read_only_root, spec.log_driver, spec.env),
                             (IMAGE, "0:0", "none", True, "none", KEEPER_ENV))
            self.assertEqual(spec.argv, batch.container_argv(pack.BATCH))
            self.assertEqual({m.container for m in spec.mounts if not m.read_only}, {"/out"})
        model = {m.container: m.host for m in runs[1].mounts}["/in/model"]
        self.assertTrue(model.startswith(str(kp.state / "models") + os.sep), model)
        self.assertEqual(Path(model).name, "candidate")
        self.assertNotIn(str(self.out), json.dumps([dataclasses.asdict(s) for s in runs]))
        self.assertFalse((kp.state / "models").exists() and any((kp.state / "models").iterdir()))
        self.assertFalse((kp.state / "run").exists() and any((kp.state / "run").iterdir()))
        # report: compute block (AN-10)
        env = kp.client.call("report", pid, {"compute": {"jobs": 1, "seconds": 30, "gpu_seconds": 0}})["report"]
        pub = signing.load_public_pem(kp.client.call("hello", None, {})["public_pem"].encode("ascii"))
        report = signing.verify(env, pub, kind="report")
        self.assertEqual(report["compute"]["keeper"], {"containers": 2, "seconds": 120, "gpu_seconds": 0})
        self.assertEqual(report["compute"]["backend"]["source"], SOURCE)
        md = trust_stages.render_markdown(report)
        self.assertIn("- keeper: 2 inference containers", md)
        self.assertIn(SOURCE, md)

    def test_no_inference_refuses_batch(self):                 # AN-8: before the grant
        kp = self.keeper(inference_on=False)
        pid, args, train = self.project(kp)
        cand = batch_model([write_table(self.out / "cand", pack.fit(train, 0.0))], "candidate", "c")
        with self.assertRaises(Refused) as cm:
            kp.client.call("evaluate", pid, {**args, "candidate": arm(cand)})
        self.assertNotIsInstance(cm.exception, RefusedAfterSpend)
        p = cm.exception.problems[0]
        self.assertEqual((p.rule, p.message), ("AN-8", "this keeper runs no inference container"))
        self.assertEqual(kp.client.call("alpha", pid, {})["entries"], [])
        self.assertEqual(FakeRuntime(state_dir=self.fake).runs, [])

    def test_image_id_changed_refused(self):                   # at start
        err = self.start_refused(fake_image=OTHER_IMAGE)
        self.assertIn("AN-13", err.output)

    def test_eval_tree_writable_refused(self):
        tree = self.tmp / "tree"
        tree.mkdir()
        (tree / "lib.py").write_text("")
        os.chmod(tree / "lib.py", 0o666)
        err = self.start_refused(extra=["--eval-tree", str(tree)])
        self.assertIn("AN-13", err.output)
        self.assertIn("writable by every user", err.output)

    def test_bad_inference_image_refused(self):
        err = self.start_refused(image="busybox")
        self.assertIn("inference", err.output)


if __name__ == "__main__":
    unittest.main()
