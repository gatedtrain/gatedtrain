"""Paired sidecar container (EXT-190, EXT-249; MULTIMODAL-SPEC MM-34a), tested on the framework alone.

The recipes that declare a sidecar row (the CAD recipe) install gatedtrain from PyPI, so a change to
`launch.sidecar_problems` or `launch.sidecar_commands` on main is tested by no recipe CI until it is released. These
tests run the rules in process: the synthetic adapter with a sidecar row given through `HostConfig.sidecar` (the
field `adapter_config` otherwise fills from the registry), the golden request rooted under a test directory
(`test_ext_launch.rooted`), and an executor directory (interpreter and script stubs, a lock file) beside the runtime.
The row is shaped as the CAD recipe's (`executor/run-executor.sh serve`) with the B18 policy of `gatedtrain.ext.executor`.

Accepted: the record's `sidecar` and the three printed lines (start, trainer, stop). Refused, each with its field:
image and lock, the isolation flags, network, user, limits, tmpfs, mounts (another directory, a second read-write
mount, a socket, a credential file, the spool missing or misplaced, the trainer not sharing it), environment, argv,
unknown and missing keys. `--recheck` re-judges the recorded sidecar (EXT-83: the executor's script and interpreter
must keep their SHA-256).
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import shutil
import socket
import tempfile
import unittest
from pathlib import Path

import yaml

from _ext import ENV_HOST_WORK, container_user
from _support import file_sha256
from test_ext_launch import VENV_LOCK, bound_spec, golden_request, host_layout, rooted

from gatedtrain.errors import Refused
from gatedtrain.ext import launch
from gatedtrain.ext.executor import EXECUTOR_ENV, EXECUTOR_LIMITS, EXECUTOR_TMPFS_OPTIONS

EXEC_IMAGE = "sha256:" + "cd" * 32
EXEC_DIR = "/data/gatedtrain-mm/executor"              # container path = host path under the mirror root (identity)
EXEC_PYTHON = f"{EXEC_DIR}/.venv/bin/python"
EXEC_SCRIPT = f"{EXEC_DIR}/serve.py"
SOCKET = "/spool/exec.sock"
ARGV = (EXEC_PYTHON, "-I", EXEC_SCRIPT, "--serve", SOCKET, "--workers", "2")
SIDE_ENV = {"HOME": "/tmp/home", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
IMAGE_PYTHON = "/opt/venv/bin/python"                  # EXT-249: an interpreter inside the executor image


def sidecar_row(lock_sha256: str, **overrides) -> dict:
    """A registry `sidecar` row in the CAD recipe's shape (rows.py `sidecar`)."""
    row = {"image": EXEC_IMAGE, "venv_lock_sha256": lock_sha256, "argv": ARGV, "ro_mounts": (EXEC_DIR,),
           "name_prefix": "gatedtrain-exec-", "spool": "/spool", "socket": SOCKET, "env_names": EXECUTOR_ENV,
           "env_paths": ("HOME",), "tmpfs": ("/tmp",), "tmpfs_options": EXECUTOR_TMPFS_OPTIONS,
           "limits": EXECUTOR_LIMITS}
    row.update(overrides)
    return row


class SidecarCase(unittest.TestCase):
    """The golden request with a sidecar, rooted under `<tmp>/host`; `launch()` runs ext-launch in process."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-sidecar-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = self.tmp / "host"
        req = rooted(golden_request(), self.root)
        shutil.copyfile(VENV_LOCK, self.tmp / "uv.lock")
        req["runtime"]["venv_lock"] = str(self.tmp / "uv.lock")
        self.exec_dir = self.root / EXEC_DIR.lstrip("/")
        (self.exec_dir / ".venv" / "bin").mkdir(parents=True)
        (self.exec_dir / ".venv" / "bin" / "python").write_text("# interpreter stand-in; never executed\n",
                                                                encoding="utf-8")
        (self.exec_dir / "serve.py").write_text("# executor stand-in; never executed\n", encoding="utf-8")
        self.lock = self.exec_dir / "uv.lock"
        self.lock.write_text("# executor lock stand-in\n", encoding="utf-8")
        # <run>/train is the trainer's output; <run>/spool beside it is the one directory both containers share.
        self.output = Path(req["output_dir"])
        self.spool = self.output.parent / "spool"
        self.spool.mkdir()
        req["mounts"].append({"host": str(self.spool), "container": "/spool", "mode": "rw"})
        req["sidecar"] = {
            "runtime": {"kind": "venv_in_container", "image": EXEC_IMAGE, "venv_lock": str(self.lock)},
            "argv": list(ARGV),
            "mounts": [{"host": str(self.exec_dir), "container": EXEC_DIR, "mode": "ro"},
                       {"host": str(self.spool), "container": "/spool", "mode": "rw"}],
            "env": dict(SIDE_ENV), "network": "none", "user": req["user"], "tmpfs": ["/tmp"], "read_only": True,
            "cap_drop": ["ALL"], "security_opt": ["no-new-privileges"], "gpus": False, "limits": dict(EXECUTOR_LIMITS)}
        self.req = req
        self.layout = host_layout(req, self.root)
        self.spec = bound_spec(self.tmp, req, self.root)
        self.row = sidecar_row(file_sha256(self.lock))
        self.n = 0

    def config(self, row: dict | None = None) -> launch.HostConfig:
        cfg = launch.host_config({**self.layout, launch.ENV_HOST_CONFIG: os.devnull})
        trees = (*cfg.interpreter_trees, str(self.exec_dir / ".venv"))       # EXT-113: the executor's interpreter
        return dataclasses.replace(cfg, sidecar=self.row if row is None else row, interpreter_trees=trees)

    def launch(self, req: dict | None = None, *, row: dict | None = None) -> tuple[str, Path]:
        self.n += 1
        path = self.tmp / f"request-{self.n}.yaml"
        path.write_text(yaml.safe_dump(self.req if req is None else req, sort_keys=False), encoding="utf-8")
        out = self.tmp / f"launch-{self.n}.json"
        cmd = launch.ext_launch(adapter="synthetic", profile="smoke", request=path,
                                gpu_log=Path(self.layout[ENV_HOST_WORK]) / "gpu-log.md", out_record=out,
                                spec=self.spec, config=self.config(row))
        return cmd, out

    def refused(self, mutate, field: str, *, row: dict | None = None, text: str | None = None) -> list:
        """ext-launch refuses the mutated request with EXT-LAUNCH at `--request.<field>` and writes no record."""
        req = copy.deepcopy(self.req)
        mutate(req)
        with self.assertRaises(Refused) as cm:
            self.launch(req, row=row)
        hits = [p for p in cm.exception.problems if p.rule == "EXT-LAUNCH" and p.field == f"--request.{field}"]
        self.assertTrue(hits, f"expected --request.{field}; got {[(p.field, p.message) for p in cm.exception.problems]}")
        if text is not None:
            self.assertTrue(any(text in p.message for p in hits), [p.message for p in hits])
        self.assertFalse((self.tmp / f"launch-{self.n}.json").exists(), "a refused launch writes no record")
        return hits


class TestSidecarAccepted(SidecarCase):
    """EXT-190: the accepted shape, the record and the printed commands."""

    def test_record_and_three_lines(self):
        cmd, out = self.launch()
        rec = json.loads(out.read_text(encoding="utf-8"))
        side = rec["sidecar"]
        self.assertEqual(sorted(side), sorted(launch.SIDECAR_RECORD_KEYS))
        self.assertEqual(side["runtime"], {"kind": "venv_in_container", "image": EXEC_IMAGE,
                                           "venv_lock_sha256": file_sha256(self.lock)})
        self.assertEqual(side["env_names"], sorted(SIDE_ENV))                     # names only (MM-35)
        self.assertNotIn("env", side)
        self.assertEqual(side["interpreter_sha256"], file_sha256(self.exec_dir / ".venv" / "bin" / "python"))
        self.assertEqual(side["script_sha256"], file_sha256(self.exec_dir / "serve.py"))
        self.assertEqual((side["network"], side["user"], side["read_only"], side["gpus"]),
                         ("none", container_user(), True, False))
        lines = cmd.split("\n")
        self.assertEqual(len(lines), 3, cmd)
        name = f"gatedtrain-exec-{rec['label']}"
        execd, spool = os.path.realpath(self.exec_dir), os.path.realpath(self.spool)
        self.assertEqual(lines[0], (
            f"docker --context gatedtrain run -d --name {name} --rm --network none --user {container_user()} --cap-drop ALL "
            "--security-opt no-new-privileges --read-only --tmpfs /tmp:rw,nosuid,nodev,noexec,size=4g --cpus 8 "
            "--memory 8g --memory-swap 8g --pids-limit 256 --ulimit nofile=1024:1024 --ulimit core=0 "
            "-e HOME=/tmp/home -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 "
            f"-v {execd}:{EXEC_DIR}:ro -v {spool}:/spool:rw {EXEC_IMAGE} "
            f"{EXEC_PYTHON} -I {EXEC_SCRIPT} --serve /spool/exec.sock --workers 2"))
        self.assertIn(f"/bin/gpu-step.sh {rec['label']} --rm ", lines[1])
        self.assertIn(f" -v {spool}:/spool:rw ", lines[1])                       # the trainer mounts the spool too
        self.assertEqual(lines[2], f"docker --context gatedtrain stop {name}")

    def test_recheck_rejudges_the_recorded_sidecar(self):
        """EXT-69, EXT-83: --recheck passes on the fresh record and refuses once the executor's script changed."""
        _, out = self.launch()
        self.assertTrue(launch.recheck(adapter="synthetic", record_path=out, spec=self.spec,
                                       config=self.config()).startswith("recheck: ok "))
        (self.exec_dir / "serve.py").write_text("# edited after the launch\n", encoding="utf-8")
        with self.assertRaises(Refused) as cm:
            launch.recheck(adapter="synthetic", record_path=out, spec=self.spec, config=self.config())
        self.assertIn("--recheck.sidecar.script_sha256", [p.field for p in cm.exception.problems])

    def test_recheck_refuses_an_edited_record(self):
        """kind "record": env_names and the interpreter hash are judged as recorded."""
        _, out = self.launch()
        rec = json.loads(out.read_text(encoding="utf-8"))
        rec["sidecar"]["env_names"] = ["HOME"]
        rec["sidecar"]["interpreter_sha256"] = "sha256:" + "0" * 64
        rec["sidecar"]["runtime"]["venv_lock_sha256"] = "sha256:" + "1" * 64
        out.write_text(json.dumps(rec), encoding="utf-8")
        with self.assertRaises(Refused) as cm:
            launch.recheck(adapter="synthetic", record_path=out, spec=self.spec, config=self.config())
        fields = {p.field for p in cm.exception.problems}
        for f in ("sidecar.env_names", "sidecar.interpreter_sha256", "sidecar.runtime.venv_lock"):
            self.assertIn(f"--recheck.{f}", fields)

    def test_image_interpreter(self):
        """EXT-249: a row with `image_interpreter` takes runtime {kind: container, image}, records no lock and no
        interpreter hash, and binds only the script; a lock in the request is refused."""
        row = sidecar_row(file_sha256(self.lock), image_interpreter=IMAGE_PYTHON,
                          argv=(IMAGE_PYTHON, "-I", EXEC_SCRIPT, "--serve", SOCKET, "--workers", "2"))
        req = copy.deepcopy(self.req)
        req["sidecar"]["runtime"] = {"kind": "container", "image": EXEC_IMAGE}
        req["sidecar"]["argv"] = list(row["argv"])
        cmd, out = self.launch(req, row=row)
        side = json.loads(out.read_text(encoding="utf-8"))["sidecar"]
        self.assertEqual(side["runtime"], {"kind": "container", "image": EXEC_IMAGE, "venv_lock_sha256": None})
        self.assertIsNone(side["interpreter_sha256"])
        self.assertEqual(side["script_sha256"], file_sha256(self.exec_dir / "serve.py"))
        self.assertIn(f"{EXEC_IMAGE} {IMAGE_PYTHON} -I {EXEC_SCRIPT} ", cmd.split("\n")[0])
        self.refused(lambda r: (r["sidecar"].update(argv=list(row["argv"])),
                                r["sidecar"]["runtime"].__setitem__("venv_lock", str(self.lock))),
                     "sidecar.runtime", row=row)
        self.refused(lambda r: r["sidecar"].update(argv=[IMAGE_PYTHON, "-I", f"{EXEC_DIR}/missing.py",
                                                        "--serve", SOCKET, "--workers", "2"]),
                     "sidecar.argv", row=dict(row, argv=(IMAGE_PYTHON, "-I", f"{EXEC_DIR}/missing.py", "--serve",
                                                         SOCKET, "--workers", "2")), text="EXT-91")


class TestSidecarRefusedIsolation(SidecarCase):
    """EXT-190: each isolation setting is the row's, exactly."""

    def test_other_image(self):
        self.refused(lambda r: r["sidecar"]["runtime"].update(image="sha256:" + "ef" * 32), "sidecar.runtime.image")

    def test_other_lock(self):
        other = self.tmp / "other.lock"
        other.write_text("# another lock\n", encoding="utf-8")
        self.refused(lambda r: r["sidecar"]["runtime"].update(venv_lock=str(other)), "sidecar.runtime.venv_lock",
                     text="MM-32")

    def test_runtime_kind(self):
        self.refused(lambda r: r["sidecar"]["runtime"].update(kind="container"), "sidecar.runtime")

    def test_network(self):
        for net in ("bridge", "host", None):
            with self.subTest(network=net):
                self.refused(lambda r: r["sidecar"].update(network=net), "sidecar.network", text="network none")

    def test_user(self):
        """The executor binds its socket with umask 077: only the trainer's uid may connect. The whole `uid:gid` must
        equal the trainer's (the running user), so a matching uid with another gid is refused too."""
        uid, gid = container_user().split(":")
        other_uid, other_gid = str(int(uid) + 1), str(int(gid) + 1)
        for user in (f"{other_uid}:{other_gid}", f"{uid}:{other_gid}", f"{other_uid}:{gid}"):
            with self.subTest(user=user):
                self.refused(lambda r: r["sidecar"].update(user=user), "sidecar.user",
                             text=f"user {user!r} is not the trainer's {container_user()!r}")

    def test_flags(self):
        for key, value in (("read_only", False), ("cap_drop", []), ("cap_drop", ["NET_RAW"]), ("security_opt", []),
                           ("gpus", True), ("gpus", 0), ("read_only", 1)):
            with self.subTest(key=key, value=value):
                self.refused(lambda r: r["sidecar"].update({key: value}), f"sidecar.{key}",
                             text=f"{key} is {value!r}, not ")

    def test_limits(self):
        for key, value in (("memory", "16g"), ("pids", 4096), ("cpus", 8.0), ("core", "0")):
            with self.subTest(key=key, value=value):
                self.refused(lambda r: r["sidecar"]["limits"].update({key: value}), "sidecar.limits",
                             text="are not the executor's")
        self.refused(lambda r: r["sidecar"]["limits"].pop("nofile"), "sidecar.limits", text="are not the executor's")

    def test_tmpfs(self):
        self.refused(lambda r: r["sidecar"].update(tmpfs=["/tmp", "/var/tmp"]), "sidecar.tmpfs")

    def test_keys(self):
        self.refused(lambda r: r["sidecar"].update(privileged=False), "sidecar.privileged", text="unknown sidecar key")
        self.refused(lambda r: r["sidecar"].pop("limits"), "sidecar.limits", text="missing sidecar key")
        self.refused(lambda r: r.update(sidecar=["not", "a", "mapping"]), "sidecar", text="not a mapping")

    def test_adapter_without_a_sidecar_row(self):
        with self.assertRaises(Refused) as cm:
            self.launch(row={})
        self.assertIn("--request.sidecar", [p.field for p in cm.exception.problems])


class TestSidecarRefusedMounts(SidecarCase):
    """EXT-190 mounts: the row's read-only directories at their own host paths and one read-write spool, the
    directory beside the trainer's output that the trainer mounts too; EXT-40, EXT-51, EXT-66 as for the trainer."""

    def test_another_read_only_directory(self):
        base = next(m["host"] for m in self.req["mounts"] if m["container"] == "/inputs/base-model")
        self.refused(lambda r: r["sidecar"]["mounts"].append({"host": base, "container": "/inputs/base-model",
                                                              "mode": "ro"}),
                     "sidecar.mounts[2].container")

    def test_read_only_mount_not_at_its_host_path(self):
        self.refused(lambda r: r["sidecar"]["mounts"][0].update(container="/executor"),
                     "sidecar.mounts[0].container")

    def test_executor_mount_missing(self):
        self.refused(lambda r: r["sidecar"]["mounts"].pop(0), "sidecar.mounts", text="read-only at its own host path")

    def test_second_read_write_mount(self):
        """Only the spool is read-write: the trainer's output at another path, or a second spool mount."""
        self.refused(lambda r: r["sidecar"]["mounts"].append({"host": str(self.output), "container": "/out",
                                                              "mode": "rw"}),
                     "sidecar.mounts[2].mode")
        self.refused(lambda r: r["sidecar"]["mounts"].append({"host": str(self.spool), "container": "/spool",
                                                              "mode": "rw"}),
                     "sidecar.mounts[2].mode", text="one read-write mount")
        # the spool directory itself, read-write at another container path than the row's spool
        self.refused(lambda r: r["sidecar"]["mounts"][1].update(container="/work"), "sidecar.mounts[1].mode",
                     text="only read-write mount is its spool")

    def test_spool_missing(self):
        """Without the read-write spool the sidecar has no socket the trainer can reach."""
        self.refused(lambda r: r["sidecar"]["mounts"].pop(1), "sidecar.mounts", text="no read-write spool")

    def test_spool_not_beside_the_output(self):
        other = self.output.parent.parent / "elsewhere"
        other.mkdir()

        def mutate(r):
            r["sidecar"]["mounts"][1]["host"] = str(other)
            next(m for m in r["mounts"] if m["container"] == "/spool")["host"] = str(other)
        self.refused(mutate, "sidecar.mounts[1].host", text="beside the trainer's output")

    def test_spool_is_the_output(self):
        def mutate(r):
            r["sidecar"]["mounts"][1]["host"] = str(self.output)
        self.refused(mutate, "sidecar.mounts[1].host", text="the trainer and the sidecar share only the spool")

    def test_spool_not_empty(self):
        """A new launch's spool is empty (fresh, EXT-63)."""
        (self.spool / "stale.sock").write_text("", encoding="utf-8")
        self.refused(lambda r: None, "sidecar.mounts[1].host", text="the spool: spool ")
        self.refused(lambda r: None, "sidecar.mounts[1].host", text="is not empty; a new run writes into a new, empty")

    def test_trainer_does_not_share_the_spool(self):
        self.refused(lambda r: r["mounts"].pop(), "sidecar.mounts", text="0 times")

    def test_trainer_mount_reaches_the_spool(self):
        """A trainer input mount that holds the spool would reach it a second way."""
        self.refused(lambda r: r["mounts"].append({"host": str(self.output.parent), "container": "/inputs/run",
                                                   "mode": "ro"}),
                     "sidecar.mounts[1].host", text="lies inside the spool")

    def test_socket_mount(self):
        """EXT-40: a socket is never mounted, here a Unix socket in the input root."""
        sock_dir = self.root / "data" / "s"
        sock_dir.mkdir()
        path = sock_dir / "d.sock"
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.addCleanup(s.close)
        s.bind(str(path))
        self.refused(lambda r: r["sidecar"]["mounts"].append({"host": str(path), "container": "/var/run/d.sock",
                                                              "mode": "ro"}),
                     "sidecar.mounts[2].host", text="EXT-40")

    def test_kernel_path(self):
        """EXT-51: /proc is never a mount source."""
        self.refused(lambda r: r["sidecar"]["mounts"].append({"host": "/proc/1/root", "container": "/host",
                                                              "mode": "ro"}),
                     "sidecar.mounts[2].host", text="EXT-51")

    def test_credential_file(self):
        """EXT-80, EXT-93: a credential file name inside the executor directory refuses the mount."""
        (self.exec_dir / ".env").write_text("X=1\n", encoding="utf-8")
        self.refused(lambda r: None, "sidecar.mounts[0].host", text="credential file")

    def test_malformed_mount(self):
        self.refused(lambda r: r["sidecar"]["mounts"][0].update(extra=1), "sidecar.mounts[0]")
        self.refused(lambda r: r["sidecar"]["mounts"][0].update(mode="rx"), "sidecar.mounts[0].mode")
        self.refused(lambda r: r["sidecar"].update(mounts={}), "sidecar.mounts", text="must be a list")


class TestSidecarRefusedEnvAndArgv(SidecarCase):
    """EXT-190: the row's environment names exactly (EXT-39, EXT-92, EXT-114) and the row's argv (EXT-91, EXT-113)."""

    def test_credential_name(self):
        self.refused(lambda r: r["sidecar"]["env"].update(HF_TOKEN="x"), "sidecar.env.HF_TOKEN",
                     text="credentials are never passed")

    def test_name_outside_the_row(self):
        self.refused(lambda r: r["sidecar"]["env"].update(LD_PRELOAD="/tmp/x.so"), "sidecar.env.LD_PRELOAD",
                     text="EXT-92")

    def test_bad_name_and_value(self):
        self.refused(lambda r: r["sidecar"]["env"].update({"1X": "1"}), "sidecar.env.1X", text="EXT-39")
        self.refused(lambda r: r["sidecar"]["env"].update(PYTHONUNBUFFERED="1\n"), "sidecar.env.PYTHONUNBUFFERED")

    def test_name_missing(self):
        self.refused(lambda r: r["sidecar"]["env"].pop("HOME"), "sidecar.env.HOME", text="is not set")

    def test_home_outside_tmp(self):
        self.refused(lambda r: r["sidecar"]["env"].update(HOME="/data/home"), "sidecar.env.HOME", text="EXT-114")

    def test_other_argv(self):
        for argv in (list(ARGV[:-1]) + ["64"], list(ARGV) + ["--debug"], ["/bin/sh", "-c", "true"]):
            with self.subTest(argv=argv):
                self.refused(lambda r: r["sidecar"].update(argv=argv), "sidecar.argv")

    def test_interpreter_outside_the_trees(self):
        """EXT-113: the row's interpreter resolves, but not inside the adapter's interpreter trees."""
        cfg_trees = self.config
        self.config = lambda row=None: dataclasses.replace(cfg_trees(row), interpreter_trees=(
            str(self.root / "data" / "gatedtrain-mm" / "synthetic-runtime" / ".venv"),))
        self.refused(lambda r: None, "sidecar.argv", text="EXT-113")

    def test_script_not_in_a_read_only_mount(self):
        (self.exec_dir / "serve.py").unlink()
        self.refused(lambda r: None, "sidecar.argv", text="EXT-91")


class TestSidecarCommands(unittest.TestCase):
    """EXT-190, EXT-69: `sidecar_commands` takes every isolation option from code and refuses values docker would
    read as options or mount sources that do not resolve."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-sidecar-cmd-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.row = sidecar_row("sha256:" + "0" * 64)
        self.side = {"runtime": {"image": EXEC_IMAGE}, "argv": list(ARGV), "user": "1000:1000", "tmpfs": ["/tmp"],
                     "limits": dict(EXECUTOR_LIMITS), "mounts": [{"host": str(self.tmp), "container": "/spool",
                                                                  "mode": "rw"}]}

    def test_shape(self):
        start, stop = launch.sidecar_commands(self.side, {"HOME": "/tmp/a b"}, self.row, "gatedtrain-exec-x")
        self.assertTrue(start.startswith("docker --context gatedtrain run -d --name gatedtrain-exec-x --rm "
                                         "--network none --user 1000:1000 "), start)
        self.assertIn(" -e 'HOME=/tmp/a b' ", start)                            # shell-quoted values
        self.assertIn(f" -v {os.path.realpath(self.tmp)}:/spool:rw {EXEC_IMAGE} ", start)
        self.assertEqual(stop, "docker --context gatedtrain stop gatedtrain-exec-x")
        self.assertEqual(launch.sidecar_name({"label": "l-0123"}, self.row), "gatedtrain-exec-l-0123")

    def test_option_like_values_raise(self):
        with self.assertRaises(ValueError):
            launch.sidecar_commands(self.side, {}, self.row, "--privileged")
        with self.assertRaises(ValueError):
            launch.sidecar_commands(dict(self.side, user="--user=0"), {}, self.row, "n")

    def test_unresolved_mount_raises(self):
        side = dict(self.side, mounts=[{"host": str(self.tmp / "gone"), "container": "/spool", "mode": "rw"}])
        with self.assertRaises(ValueError):
            launch.sidecar_commands(side, {}, self.row, "n")


if __name__ == "__main__":
    unittest.main()
