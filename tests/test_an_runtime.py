"""The container runtime interface, the Docker CLI runtime, the GPU probe and the fake runtime (AN-22, AN-23; design
revision 25, 05 §18.10; tests/README.md section 14).

No test starts Docker or nvidia-smi: `DockerCLIRuntime` and `NvidiaSmiProbe` run Python scripts written to a temporary
directory that record their argument lists and answer from a JSON table baked into the script (the runtime passes on
`PATH`, `HOME` and `XDG_RUNTIME_DIR` only, so the script cannot read its table path from the environment).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gatedtrain.ainative import runtime as rt
from gatedtrain.ainative.runtime import (ContainerState, DaemonInfo, DockerCLIRuntime, Mount, NvidiaSmiProbe, RunSpec,
                                         RuntimeFailure, docker_run_argv, wait_for_gpu)
from gatedtrain.ext import launch
from gatedtrain.testing.fake_runtime import FakeRuntime, subprocess_handler

IMAGE = "sha256:" + "ab" * 32
IMAGE_REF = "nvcr.io/nvidia/cuda@sha256:" + "cd" * 32
CID = "f" * 64


class _Tmp(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def mkdir(self, name: str) -> str:
        p = self.tmp / name
        p.mkdir(parents=True, exist_ok=True)
        return os.path.realpath(p)


# ---------------------------------------------------------------------------
# RunSpec and docker_run_argv
# ---------------------------------------------------------------------------

class TestDockerRunArgv(_Tmp):
    def keeper_spec(self) -> tuple[RunSpec, dict[str, str]]:
        d = {k: self.mkdir(k) for k in ("in", "model", "gen", "out")}
        spec = RunSpec(name="gatedtrain-keeper-0123456789ab-arm1", image=IMAGE, argv=("/opt/venv/bin/python",
                       "/gen/batch.py"), env=(("HF_HUB_OFFLINE", "1"), ("CUDA_MODULE_LOADING", "LAZY")),
                       mounts=(Mount(d["in"], "/in/items"), Mount(d["model"], "/in/model"), Mount(d["gen"], "/gen"),
                               Mount(d["out"], "/out", read_only=False)),
                       user=launch.ROOTLESS_USER, gpu_options=launch.CDI_GPU_OPTIONS, read_only_root=True,
                       tmpfs=(("/tmp", "rw,nosuid,nodev,size=1g"),), shm_size=None, log_driver="none",
                       labels=(("gatedtrain.keeper", "1"),))
        return spec, d

    def test_keeper_golden(self) -> None:
        spec, d = self.keeper_spec()
        got = docker_run_argv(spec, host="unix:///run/user/1001/docker.sock")
        want = ["docker", "-H", "unix:///run/user/1001/docker.sock", "run", "-d", "--pull", "never",
                "--name", "gatedtrain-keeper-0123456789ab-arm1", "--log-driver", "none", "--network", "none",
                "--user", "0:0", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only",
                "--device", "nvidia.com/gpu=all", "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g",
                "-e", "CUDA_MODULE_LOADING=LAZY", "-e", "HF_HUB_OFFLINE=1",
                "-v", f"{d['in']}:/in/items:ro", "-v", f"{d['model']}:/in/model:ro", "-v", f"{d['gen']}:/gen:ro",
                "-v", f"{d['out']}:/out:rw", "--label", "gatedtrain.keeper=1", IMAGE, "/opt/venv/bin/python",
                "/gen/batch.py"]
        self.assertEqual(got, want)
        self.assertNotIn("--rm", got)
        self.assertFalse(any(t.startswith("seccomp") for t in got))

    def test_backend_golden(self) -> None:
        data, out = self.mkdir("data"), self.mkdir("out/a1")
        spec = RunSpec(name="gatedtrain-0123456789ab-3-sft-a1", image=IMAGE_REF, argv=("python", "/work/train.py",
                       "--out", "/out"), env=(("WANDB_MODE", "offline"),), mounts=(Mount(data, "/data"),
                       Mount(out, "/out", read_only=False)), user="1000:1000",
                       gpu_options=launch.ROOTFUL_GPU_OPTIONS, extra_options=("--ulimit", "memlock=-1"),
                       tmpfs=(("/tmp", launch.TMPFS_OPTIONS),),
                       labels=(("gatedtrain.project", "0123456789ab"), ("gatedtrain.job", "3")))
        got = docker_run_argv(spec, context="gatedtrain", docker="/usr/bin/docker")
        want = ["/usr/bin/docker", "--context", "gatedtrain", "run", "-d", "--pull", "never",
                "--name", "gatedtrain-0123456789ab-3-sft-a1", "--network", "none", "--user", "1000:1000",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--shm-size=8g",
                "--runtime=nvidia", "--gpus", "all", "--ulimit", "memlock=-1",
                "--tmpfs", f"/tmp:{launch.TMPFS_OPTIONS}", "-e", "WANDB_MODE=offline",
                "-v", f"{data}:/data:ro", "-v", f"{out}:/out:rw",
                "--label", "gatedtrain.job=3", "--label", "gatedtrain.project=0123456789ab",
                IMAGE_REF, "python", "/work/train.py", "--out", "/out"]
        self.assertEqual(got, want)

    def test_link_source_is_its_target(self) -> None:
        target = self.mkdir("real")
        link = self.tmp / "link"
        link.symlink_to(target)
        spec = RunSpec(name="n", image=IMAGE, argv=(), env=(), mounts=(Mount(str(link), "/in"),), user="0:0")
        got = docker_run_argv(spec, context="gatedtrain")
        self.assertIn(f"{target}:/in:ro", got)
        self.assertNotIn(str(link), " ".join(got))

    def test_unresolved_or_bad_source_raises(self) -> None:
        bad = self.mkdir("a:b")
        for host in (str(self.tmp / "missing"), bad, "-x"):
            spec = RunSpec(name="n", image=IMAGE, argv=(), env=(), mounts=(Mount(host, "/in"),), user="0:0")
            with self.subTest(host=host), self.assertRaises(ValueError):
                docker_run_argv(spec, context="gatedtrain")

    def test_option_like_value_raises(self) -> None:
        base = dict(name="n", image=IMAGE, argv=("-x",), env=(), mounts=(), user="0:0")
        docker_run_argv(RunSpec(**base), context="gatedtrain")              # after the image: an argument
        for change in ({"user": "-x"}, {"network": "-x"}, {"log_driver": "-x"}, {"shm_size": "-x"},
                       {"tmpfs": (("-x", "rw"),)}, {"labels": (("-x", "1"),)}, {"env": (("-X", "1"),)},
                       {"mounts": (Mount(self.mkdir("m"), "-x"),)}, {"gpu_options": ("--privileged",)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                docker_run_argv(RunSpec(**{**base, **change}), context="gatedtrain")
        for kw in ({"context": "-x"}, {"host": "-x"}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                docker_run_argv(RunSpec(**base), **kw)

    def test_context_xor_host(self) -> None:
        spec = RunSpec(name="n", image=IMAGE, argv=(), env=(), mounts=(), user="0:0")
        with self.assertRaises(ValueError):
            docker_run_argv(spec, context="gatedtrain", host="unix:///x.sock")
        with self.assertRaises(ValueError):
            docker_run_argv(spec)

    def test_runspec_name_and_image(self) -> None:
        RunSpec(name="a" * 128, image=IMAGE, argv=(), env=(), mounts=(), user="0:0")
        for kw in ({"name": "a" * 129}, {"name": "-a"}, {"name": "a/b"}, {"name": ""},
                   {"image": "ubuntu:latest"}, {"image": "-" + IMAGE}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                RunSpec(**{"name": "n", "image": IMAGE, "argv": (), "env": (), "mounts": (), "user": "0:0", **kw})

    def test_container_state_values(self) -> None:
        ContainerState("exited", 0, False, "2026-11-22T00:00:00Z", "2026-11-22T00:01:00Z")
        with self.assertRaises(ValueError):
            ContainerState("paused", None, False, None, None)
        with self.assertRaises(ValueError):
            ContainerState("exited", 0, False, "2026-11-22 00:00:00", None)


# ---------------------------------------------------------------------------
# DockerCLIRuntime against a recording script (no Docker)
# ---------------------------------------------------------------------------

_FAKE_CLI = """#!{python}
import json, os, sys
log, table = {log!r}, {table!r}
with open(log, "a") as f:
    f.write(json.dumps({{"argv": sys.argv[1:], "env": sorted(os.environ)}}) + "\\n")
answers = json.load(open(table))
args = sys.argv[1:]
for key, ans in answers.items():
    if key in args:
        sys.stdout.write(ans.get("out", ""))
        sys.stderr.write(ans.get("err", ""))
        sys.exit(ans.get("rc", 0))
sys.exit(0)
"""


class TestDockerCLIRuntime(_Tmp):
    def setUp(self) -> None:
        super().setUp()
        self.log, self.table = self.tmp / "calls.jsonl", self.tmp / "table.json"
        self.answer({})
        self.docker = self.tmp / "docker"
        self.docker.write_text(_FAKE_CLI.format(python=sys.executable, log=str(self.log), table=str(self.table)))
        self.docker.chmod(0o755)
        self.rt = DockerCLIRuntime(context="gatedtrain", docker=str(self.docker), timeout_s=30)

    def answer(self, table: dict) -> None:
        self.table.write_text(json.dumps(table))

    def calls(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_run_records_argv_without_a_shell(self) -> None:
        self.answer({"run": {"out": CID + "\n"}})
        spec = RunSpec(name="job-a1", image=IMAGE, argv=("python", "x; rm -rf /"), env=(("K", "$(id)"),), mounts=(),
                       user="0:0")
        with mock.patch.object(rt.subprocess, "run", wraps=subprocess.run) as run, \
                mock.patch.dict(os.environ, {"DOCKER_HOST": "tcp://1.2.3.4:2375", "SECRET_TOKEN": "s"}):
            cid = self.rt.run(spec)
        self.assertEqual(cid, CID)
        self.assertIsInstance(run.call_args.args[0], list)
        self.assertFalse(run.call_args.kwargs.get("shell", False))
        (call,) = self.calls()
        self.assertEqual(call["argv"], docker_run_argv(spec, context="gatedtrain", docker=str(self.docker))[1:])
        self.assertLessEqual(set(call["env"]) - {"LC_CTYPE"}, {"PATH", "HOME", "XDG_RUNTIME_DIR"})

    def test_failure_has_class_and_rc_only(self) -> None:
        self.answer({"run": {"rc": 125, "err": "sealed-item-text"}, "stop": {"rc": 1, "err": "sealed-item-text"}})
        spec = RunSpec(name="job-a1", image=IMAGE, argv=(), env=(), mounts=(), user="0:0")
        with self.assertRaises(RuntimeFailure) as cm:
            self.rt.run(spec)
        self.assertEqual((cm.exception.op, cm.exception.rc), ("run", 125))
        self.assertNotIn("sealed-item-text", str(cm.exception))
        with self.assertRaises(RuntimeFailure) as cm:
            self.rt.stop("job-a1", timeout_s=5)
        self.assertEqual((cm.exception.op, cm.exception.rc), ("stop", 1))
        self.assertEqual(self.calls()[-1]["argv"], ["--context", "gatedtrain", "stop", "-t", "5", "job-a1"])

    def test_missing_docker_binary(self) -> None:
        r = DockerCLIRuntime(context="gatedtrain", docker=str(self.tmp / "nope"))
        with self.assertRaises(RuntimeFailure) as cm:
            r.remove("x")
        self.assertEqual(cm.exception.rc, 127)

    def test_inspect_missing_only_from_an_empty_listing(self) -> None:
        self.answer({"ps": {"out": ""}})
        self.assertEqual(self.rt.inspect("job.a1").status, "missing")
        self.assertEqual(self.calls()[-1]["argv"], ["--context", "gatedtrain", "ps", "-a", "--no-trunc", "--filter",
                                                    "name=^job\\.a1$", "--format", "{{.ID}}"])
        self.answer({"ps": {"rc": 1, "err": "Cannot connect to the Docker daemon"}})
        with self.assertRaises(RuntimeFailure) as cm:
            self.rt.inspect("job-a1")
        self.assertEqual(cm.exception.op, "inspect")
        self.answer({"ps": {"out": CID + "\n"}, "inspect": {"rc": 1}})
        with self.assertRaises(RuntimeFailure):
            self.rt.inspect("job-a1")
        self.answer({"ps": {"out": ""}})
        self.rt.inspect(CID)
        self.assertIn(f"id={CID}", self.calls()[-1]["argv"])

    def test_inspect_state(self) -> None:
        state = {"Status": "exited", "ExitCode": 137, "OOMKilled": True,
                 "StartedAt": "2026-11-22T01:02:03.123456789Z", "FinishedAt": "2026-11-22T01:12:03.5Z"}
        self.answer({"ps": {"out": CID + "\n"}, "inspect": {"out": json.dumps(state) + "\n"}})
        self.assertEqual(self.rt.inspect("job-a1"), ContainerState("exited", 137, True, "2026-11-22T01:02:03Z",
                                                                   "2026-11-22T01:12:03Z"))
        state = {"Status": "running", "ExitCode": 0, "OOMKilled": False, "StartedAt": "2026-11-22T01:02:03Z",
                 "FinishedAt": "0001-01-01T00:00:00Z"}
        self.answer({"ps": {"out": CID + "\n"}, "inspect": {"out": json.dumps(state) + "\n"}})
        self.assertEqual(self.rt.inspect("job-a1"), ContainerState("running", None, False, "2026-11-22T01:02:03Z",
                                                                   None))

    def test_logs_stop_remove_image_id(self) -> None:
        self.answer({"logs": {"out": "line1\nline2\n"}, "image": {"out": IMAGE + "\n"}})
        self.assertEqual(self.rt.logs("job-a1", tail=10), b"line1\nline2\n")
        self.assertEqual(self.calls()[-1]["argv"], ["--context", "gatedtrain", "logs", "--tail", "10", "job-a1"])
        self.rt.stop("job-a1")
        self.assertEqual(self.calls()[-1]["argv"], ["--context", "gatedtrain", "stop", "-t", "30", "job-a1"])
        self.rt.remove("job-a1")
        self.assertEqual(self.calls()[-1]["argv"], ["--context", "gatedtrain", "rm", "job-a1"])
        self.assertEqual(self.rt.image_id(IMAGE_REF), IMAGE)
        self.answer({"image": {"rc": 1}, "version": {"out": "27.3.1\n"}})
        self.assertIsNone(self.rt.image_id(IMAGE_REF))
        self.answer({"image": {"rc": 1}, "version": {"rc": 1}})
        with self.assertRaises(RuntimeFailure):
            self.rt.image_id(IMAGE_REF)
        with self.assertRaises(ValueError):
            self.rt.image_id("-x")

    def test_containers_by_label(self) -> None:
        self.answer({"ps": {"out": "a\nb\n"}})
        self.assertEqual(self.rt.containers("gatedtrain.keeper=1"), ("a", "b"))
        self.assertEqual(self.calls()[-1]["argv"], ["--context", "gatedtrain", "ps", "-a", "--filter",
                                                    "label=gatedtrain.keeper=1", "--format", "{{.Names}}"])

    def test_daemon(self) -> None:
        sock = self.tmp / "docker.sock"
        sock.write_text("")
        r = DockerCLIRuntime(host=f"unix://{sock}", docker=str(self.docker))
        self.answer({"info": {"out": json.dumps(["name=seccomp,profile=builtin", "name=rootless", "name=cgroupns"])
                              + "\n"}})
        self.assertEqual(r.daemon(), DaemonInfo(True, os.getuid(), str(sock)))
        self.answer({"info": {"out": json.dumps(["name=seccomp,profile=builtin"]) + "\n"}})
        self.assertFalse(r.daemon().rootless)
        self.answer({"info": {"out": "[]\n"}, "context": {"out": f"unix://{sock}\n"}})
        info = self.rt.daemon()
        self.assertEqual(info, DaemonInfo(False, os.getuid(), str(sock)))
        self.assertEqual(self.calls()[-1]["argv"], ["context", "inspect", "--format", "{{.Endpoints.docker.Host}}",
                                                    "gatedtrain"])
        self.answer({"info": {"out": "[]\n"}, "context": {"out": "tcp://10.0.0.1:2376\n"}})
        self.assertEqual(self.rt.daemon(), DaemonInfo(False, None, None))
        self.answer({"info": {"rc": 1}})
        with self.assertRaises(RuntimeFailure):
            self.rt.daemon()

    def test_constructor_needs_one_endpoint(self) -> None:
        with self.assertRaises(ValueError):
            DockerCLIRuntime()
        with self.assertRaises(ValueError):
            DockerCLIRuntime(context="a", host="unix:///b")


# ---------------------------------------------------------------------------
# GPU probe and wait
# ---------------------------------------------------------------------------

class _Probe:
    def __init__(self, values: list) -> None:
        self.values, self.calls = list(values), 0

    def used_mib(self):
        self.calls += 1
        return self.values.pop(0) if len(self.values) > 1 else self.values[0]


class _Clock:
    def __init__(self) -> None:
        self.t, self.sleeps = 0.0, []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


class TestGpu(_Tmp):
    def test_values(self) -> None:
        self.assertEqual((rt.GPU_BUSY_MIB, rt.GPU_WAIT_S), (1024, 900))

    def test_waits_past_two_busy_polls(self) -> None:
        probe, clock = _Probe([4000, 2000, 1024]), _Clock()
        self.assertTrue(wait_for_gpu(probe, sleep=clock.sleep, clock=clock))
        self.assertEqual((probe.calls, clock.sleeps), (3, [10.0, 10.0]))

    def test_false_at_the_bound(self) -> None:
        probe, clock = _Probe([4000]), _Clock()
        self.assertFalse(wait_for_gpu(probe, sleep=clock.sleep, clock=clock))
        self.assertLessEqual(clock.t, rt.GPU_WAIT_S)
        self.assertEqual(probe.calls, 91)

    def test_no_tool_refuses_without_waiting(self) -> None:
        probe, clock = _Probe([None]), _Clock()
        self.assertFalse(wait_for_gpu(probe, sleep=clock.sleep, clock=clock))
        self.assertEqual(clock.sleeps, [])

    def smi(self, out: str, rc: int = 0) -> NvidiaSmiProbe:
        log = self.tmp / "smi.log"
        script = self.tmp / "nvidia-smi"
        script.write_text(f"#!{sys.executable}\nimport sys\nopen({str(log)!r}, 'w').write(' '.join(sys.argv[1:]))\n"
                          f"sys.stdout.write({out!r})\nsys.exit({rc})\n")
        script.chmod(0o755)
        self.smi_log = log
        return NvidiaSmiProbe(str(script))

    def test_nvidia_smi_sum(self) -> None:
        self.assertEqual(self.smi("100\n 250 \n").used_mib(), 350)
        self.assertEqual(self.smi_log.read_text(),
                         "--query-compute-apps=used_memory --format=csv,noheader,nounits")
        self.assertEqual(self.smi("").used_mib(), 0)

    def test_non_numeric_counts_as_busy(self) -> None:
        for out in ("[N/A]\n", "10\n[Insufficient Permissions]\n"):
            with self.subTest(out=out):
                used = self.smi(out).used_mib()
                self.assertGreater(used, rt.GPU_BUSY_MIB)
                self.assertFalse(wait_for_gpu(self.smi(out), timeout_s=0, sleep=lambda s: None))

    def test_no_tool_or_failure_is_none(self) -> None:
        self.assertIsNone(NvidiaSmiProbe(str(self.tmp / "absent")).used_mib())
        self.assertIsNone(self.smi("", rc=9).used_mib())


# ---------------------------------------------------------------------------
# FakeRuntime
# ---------------------------------------------------------------------------

def _spec(name: str = "job-a1", image: str = IMAGE, **kw) -> RunSpec:
    return RunSpec(**{"name": name, "image": image, "argv": ("python", "-c", "pass"), "env": (), "mounts": (),
                      "user": "0:0", **kw})


def _echo(spec: RunSpec, host_path) -> int:
    print("hello from", spec.name)
    return 3


class TestFakeRuntime(_Tmp):
    def test_handler_ok_and_times(self) -> None:
        fake = FakeRuntime({IMAGE: _echo}, duration_s=90)
        self.assertEqual(len(fake.run(_spec("a"))), 64)
        fake.run(_spec("b"))
        self.assertEqual(fake.inspect("a"), ContainerState("exited", 3, False, "2026-11-22T00:00:00Z",
                                                           "2026-11-22T00:01:30Z"))
        self.assertEqual(fake.inspect("b").started_utc, "2026-11-22T00:01:30Z")
        self.assertEqual(fake.logs("a"), b"hello from a\n")
        self.assertEqual([s.name for s in fake.runs], ["a", "b"])

    def test_outcomes(self) -> None:
        fake = FakeRuntime({IMAGE: _echo}, outcomes=["preempt", "oom", 7, "vanish", "hang", "ok"])
        for n in "abcdef":
            fake.run(_spec(n))
        self.assertEqual((fake.inspect("a").exit_code, fake.inspect("a").oom_killed), (137, False))
        self.assertEqual((fake.inspect("b").exit_code, fake.inspect("b").oom_killed), (137, True))
        self.assertEqual(fake.inspect("c").exit_code, 7)
        self.assertEqual(fake.logs("c"), b"")
        self.assertEqual(fake.inspect("d").status, "missing")
        hung = fake.inspect("e")
        self.assertEqual((hung.status, hung.exit_code, hung.finished_utc), ("running", None, None))
        with self.assertRaises(RuntimeFailure):
            fake.remove("e")                                # like Docker: a running container is not removed
        fake.stop("e")
        self.assertEqual((fake.inspect("e").status, fake.inspect("e").exit_code), ("exited", 143))
        fake.stop("e")                                      # stopping a stopped container is a no-op
        self.assertEqual(fake.inspect("f").exit_code, 3)
        fake.remove("e")
        self.assertEqual(fake.inspect("e").status, "missing")
        with self.assertRaises(RuntimeFailure):
            fake.stop("e")

    def test_docker_like_refusals(self) -> None:
        fake = FakeRuntime(image_ids={IMAGE_REF: IMAGE})
        self.assertEqual(fake.image_id(IMAGE_REF), IMAGE)
        self.assertIsNone(fake.image_id("other@sha256:" + "00" * 32))
        with self.assertRaises(RuntimeFailure):
            fake.run(_spec(image="other@sha256:" + "00" * 32))            # --pull never: an unknown image
        fake.run(_spec("x", image=IMAGE_REF))
        with self.assertRaises(RuntimeFailure):
            fake.run(_spec("x", image=IMAGE_REF))                         # the name is taken
        with self.assertRaises(ValueError):
            fake.run(_spec("y", image=IMAGE_REF, user="-x"))               # the same checks as docker_run_argv
        self.assertEqual(len(fake.runs), 1)
        self.assertEqual(FakeRuntime().image_id(IMAGE), IMAGE)

    def test_labels_and_daemon(self) -> None:
        fake = FakeRuntime(rootless=False, socket_uid=4242)
        fake.run(_spec("a", labels=(("gatedtrain.keeper", "1"),)))
        fake.run(_spec("b"))
        self.assertEqual(fake.containers("gatedtrain.keeper=1"), ("a",))
        self.assertEqual(fake.daemon(), DaemonInfo(False, 4242, None))
        self.assertEqual(FakeRuntime().daemon().socket_uid, os.getuid())
        self.assertTrue(FakeRuntime().daemon().rootless)

    def test_state_dir_shared_and_finish(self) -> None:
        state = self.tmp / "state"
        one = FakeRuntime({IMAGE: _echo}, outcomes=["hang"], state_dir=state)
        one.run(_spec("a"))
        two = FakeRuntime({IMAGE: _echo}, state_dir=state)
        self.assertEqual(two.inspect("a").status, "running")
        two.finish("a", 0)
        self.assertEqual(one.inspect("a").exit_code, 0)
        two.run(_spec("b"))
        self.assertEqual([s.name for s in one.runs], ["a", "b"])
        self.assertEqual(one.runs[0], _spec("a"))
        self.assertEqual(one.logs("b"), b"hello from b\n")
        self.assertEqual(one.inspect("b").started_utc, "2026-11-22T00:01:00Z")


class TestSubprocessHandler(_Tmp):
    def test_writes_into_the_host_output(self) -> None:
        gen, out = self.mkdir("gen"), self.mkdir("out")
        Path(gen, "write.py").write_text("import sys, os\nopen(sys.argv[1], 'w').write(os.environ['V'] + "
                                         "str(os.path.isdir(sys.argv[2])))\nprint('done')\n")
        spec = RunSpec(name="w", image=IMAGE, argv=("/usr/bin/python3", "/gen/write.py", "/out/x", "/tmp"),
                       env=(("V", "v1"),), mounts=(Mount(gen, "/gen"), Mount(out, "/out", read_only=False)),
                       user="0:0", tmpfs=(("/tmp", "rw"),))
        fake = FakeRuntime({IMAGE: subprocess_handler})
        fake.run(spec)
        self.assertEqual(fake.inspect("w").exit_code, 0)
        self.assertEqual(Path(out, "x").read_text(), "v1True")
        self.assertEqual(fake.logs("w"), b"done\n")

    def test_unmapped_paths_untouched(self) -> None:
        spec = RunSpec(name="w", image=IMAGE, argv=("python", "-c", "import sys; print(sys.argv[1])", "/outside"),
                       env=(), mounts=(), user="0:0")
        fake = FakeRuntime({IMAGE: subprocess_handler})
        fake.run(spec)
        self.assertEqual(fake.logs("w"), b"/outside\n")


if __name__ == "__main__":
    unittest.main()
