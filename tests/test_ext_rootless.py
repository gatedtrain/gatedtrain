"""The launch rules on a rootless host (EXT-263; design revision 25, 05 §18.10; tests/README.md section 14).

`rootless = true` in the host file switches the user rule to `0:0` (the owner on the host under a rootless daemon),
checks the output directory and a sidecar's spool against the owner's uid, refuses an owner whose primary group owns a
rootful Docker socket (the owner's own socket under `/run/user/<uid>/` is ignored), prints Docker's native CDI GPU
option, and records `"rootless": true`. Without the key every rule, printed command, configuration hash and label is
as before (`GOLDEN_SHA` values computed on 73e5885, before the change).

No test reads this host's Docker sockets or groups: socket groups are injected (`socket_gids`, `globs`, a patched
`rootful_socket_gids`), the primary group is `HostConfig.primary_gid`, and the owner's socket directory is
`RUN_USER_DIR` patched to a test directory.
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
from unittest import mock

import yaml

from _ext import ENV_HOST_WORK
from _support import file_sha256
from test_ext_launch import GOLDEN_REQUEST, VENV_LOCK, bound_spec, host_layout, rooted
from test_ext_review2 import unit_config
from test_ext_sidecar import ARGV, EXEC_DIR, EXEC_IMAGE, SIDE_ENV, sidecar_row

from gatedtrain.errors import Refused
from gatedtrain.ext import launch
from gatedtrain.ext.executor import EXECUTOR_LIMITS
from gatedtrain.ext.launch import (CDI_GPU_OPTIONS, ROOTFUL_GPU_OPTIONS, ROOTLESS_USER, HostConfig, build_command,
                                   container_user, gpu_options, host_config, host_config_sha256, host_file,
                                   isolation_problems, mount_source, option_value, owner_uid, rootful_socket_gids)

# host_config_sha256 of these layouts, computed on 73e5885 (before EXT-263); they must never change.
GOLDEN_CFGS = (
    (HostConfig(host_work="/w", output_root="/o", input_roots=("/a", "/b")),
     "sha256:2b1a77b558a83a3e52e9cd54fda972205480807d5de89bb7da6e8cebcb2cf87d"),
    (HostConfig(host_work="/w", output_root="/o", input_roots=("/a",), mirror_root="/m", interpreter="/i/bin/python",
                interpreter_trees=("/t",), extra_daemon_configs=("/c.json",), extra_daemon_sockets=("/s.sock",)),
     "sha256:787a649ab4c27c47f54564ef72360a449c4be30de2bd9b6ea339e30bf4880491"),
)
PRIMARY = 4242                      # the owner's primary group in these tests (HostConfig.primary_gid)
IMAGE = "sha256:" + "ab" * 32


def rootless_request(root: Path) -> dict:
    """The golden request rooted under `root` with the user `0:0` and a plain container runtime."""
    req = yaml.safe_load(GOLDEN_REQUEST.read_text(encoding="utf-8").replace("<user>", ROOTLESS_USER))
    req = rooted(req, root)
    req["runtime"] = {"kind": "container", "image": IMAGE}
    return req


def as_record(req: dict, **extra) -> dict:
    """A request turned into the record shape isolation_problems judges (`kind="record"`)."""
    doc = {k: v for k, v in req.items() if k != "env"}
    doc.update(env_names=sorted(req["env"]), allow_network=False, **extra)
    return doc


def fields(out: list[tuple[str, str]]) -> list[str]:
    return [f for f, _ in out]


def messages(out: list[tuple[str, str]], field: str) -> list[str]:
    return [m for f, m in out if f == field]


class TmpCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-rootless-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def cfg(self, req: dict, *, rootless: bool = True, **kw) -> HostConfig:
        return dataclasses.replace(unit_config(req), rootless=rootless, primary_gid=lambda uid: PRIMARY, **kw)

    def problems(self, req: dict, *, kind: str = "request", gids=frozenset(), **kw) -> list[tuple[str, str]]:
        return isolation_problems(req, kind=kind, socket_gids=gids, config=self.cfg(req, **kw))


class TestHostFile(TmpCase):
    """EXT-263: `rootless` is a TOML boolean of the host file, read from no environment variable."""

    def env(self, text: str) -> dict:
        path = self.root / "host.toml"
        path.write_text(text, encoding="utf-8")
        return {launch.ENV_HOST_CONFIG: str(path), launch.ENV_HOST_WORK: "/w", launch.ENV_OUTPUT_ROOT: "/o"}

    def test_host_file_bool_only(self):
        for bad in ('"yes"', '"true"', "1", "0", "[true]"):
            with self.subTest(value=bad):
                with self.assertRaises(Refused) as cm:
                    host_file(self.env(f"rootless = {bad}\n"))
                (p,) = cm.exception.problems
                self.assertEqual((p.rule, p.field.rsplit(".", 1)[-1]), ("EXT-HOST", "rootless"))
                self.assertIn("is not true or false", p.message)
        self.assertIs(host_config(self.env("rootless = true\n")).rootless, True)
        self.assertIs(host_config(self.env("rootless = false\n")).rootless, False)
        self.assertIs(host_config(self.env("")).rootless, False)
        self.assertEqual(host_file(self.env('rootless = true\nhost_work = "/w"\n')), {"rootless": True, "host_work": "/w"})

    def test_no_environment_override(self):
        env = self.env("")
        env["GATEDTRAIN_EXT_ROOTLESS"] = "1"
        self.assertIs(host_config(env).rootless, False)

    def test_defaults(self):
        cfg = HostConfig()
        self.assertIs(cfg.rootless, False)
        self.assertIsNone(cfg.owner_uid)
        self.assertEqual(owner_uid(cfg), os.getuid())
        self.assertEqual(owner_uid(HostConfig(owner_uid=1234)), 1234)


class TestHash(unittest.TestCase):

    def test_hash_unchanged_without_key(self):
        for cfg, golden in GOLDEN_CFGS:
            self.assertEqual(host_config_sha256(cfg), golden)
            self.assertEqual(host_config_sha256(dataclasses.replace(cfg, rootless=False)), golden)
            self.assertEqual(host_config_sha256(dataclasses.replace(cfg, owner_uid=1234)), golden)

    def test_hash_changes_with_key(self):
        for cfg, golden in GOLDEN_CFGS:
            self.assertNotEqual(host_config_sha256(dataclasses.replace(cfg, rootless=True)), golden)


class TestUserRule(TmpCase):

    def test_user_must_be_root_in_userns(self):
        req = rootless_request(self.root)
        msgs = messages(self.problems(dict(req, user="1000:1000")), "user")
        self.assertTrue(msgs and all("0:0" in m and "owner on the host" in m and "EXT-263" in m for m in msgs), msgs)
        self.assertIn("user", fields(self.problems(dict(req, user="0:1"))))
        self.assertIn("user", fields(self.problems(dict(req, user="root"))))
        self.assertNotIn("user", fields(self.problems(req)))
        self.assertEqual(self.problems(req), [])

    def test_rootful_unchanged(self):
        """Without the key `0:0` stays refused, and the owner's own uid:gid is accepted."""
        req = rootless_request(self.root)
        self.assertIn("user", fields(self.problems(req, rootless=False)))
        own = dict(req, user=f"{os.getuid()}:{PRIMARY}")
        self.assertNotIn("user", fields(self.problems(own, rootless=False)))
        self.assertIn("user", fields(self.problems(own)))

    def test_container_user(self):
        req = rootless_request(self.root)
        self.assertEqual(container_user(self.cfg(req)), "0:0")
        self.assertEqual(container_user(self.cfg(req, rootless=False)), f"{os.getuid()}:{PRIMARY}")
        self.assertEqual(container_user(self.cfg(req, rootless=False, owner_uid=1234)), f"1234:{PRIMARY}")

    def test_primary_gid_owning_rootful_socket(self):
        req = rootless_request(self.root)
        msgs = messages(self.problems(req, gids={PRIMARY}), "user")
        self.assertTrue(msgs and all("rootful Docker socket" in m and "gid 0" in m for m in msgs), msgs)
        self.assertNotIn("user", fields(self.problems(req, gids={PRIMARY + 1})))
        unknown = isolation_problems(req, kind="request", socket_gids=frozenset(),
                                     config=dataclasses.replace(self.cfg(req), primary_gid=lambda uid: None))
        self.assertIn("user", fields(unknown))

    def test_rootful_socket_gids_used_by_default(self):
        """Without injected socket groups the rule asks `rootful_socket_gids` (patched: no host state is read)."""
        req = rootless_request(self.root)
        with mock.patch.object(launch, "rootful_socket_gids", return_value={PRIMARY}) as m:
            got = isolation_problems(req, kind="request", config=self.cfg(req))
        self.assertTrue(m.called)
        self.assertIn("user", fields(got))


class TestOutputOwner(TmpCase):

    def test_output_owned_by_owner(self):
        req = rootless_request(self.root)
        self.assertNotIn("output_dir", fields(self.problems(req)))
        msgs = messages(self.problems(req, owner_uid=os.getuid() + 1), "output_dir")
        self.assertTrue(any(f"uid {os.getuid()}" in m and "owner" in m and "EXT-263" in m for m in msgs), msgs)


class TestRootfulSocketGids(TmpCase):

    def test_own_rootless_socket_ignored(self):
        uid = os.getuid()
        run_user = self.root / "run" / "user"
        own_dir = run_user / str(uid)
        own_dir.mkdir(parents=True)
        own = own_dir / "docker.sock"
        other = self.root / "docker-other.sock"
        for p in (own, other):
            s = socket.socket(socket.AF_UNIX)
            s.bind(str(p))
            s.close()
        gid = os.stat(other).st_gid
        cfg = HostConfig()
        kw = dict(run_user_dir=str(run_user))
        self.assertEqual(rootful_socket_gids(cfg, globs=(str(own),), **kw), set())
        self.assertEqual(rootful_socket_gids(cfg, globs=(str(own), str(other)), **kw), {gid})
        self.assertEqual(rootful_socket_gids(cfg, globs=(str(run_user / "*" / "docker.sock"),), **kw), set())
        # The extra sockets of the host file count; the default globs are replaced, not the extra list.
        self.assertEqual(rootful_socket_gids(HostConfig(extra_daemon_sockets=(str(other),)), globs=(str(own),), **kw),
                         {gid})
        # Another user's directory, or a link in the owner's directory to a socket elsewhere, is not ignored.
        self.assertEqual(rootful_socket_gids(HostConfig(owner_uid=uid + 1), globs=(str(own),), **kw), {gid})
        link = own_dir / "rootful.sock"
        link.symlink_to(other)
        self.assertEqual(rootful_socket_gids(cfg, globs=(str(link),), **kw), {gid})
        # docker_socket_gids itself is unchanged: it ignores nothing.
        self.assertEqual(launch.docker_socket_gids((str(own),)), {gid})


class TestRecordMode(TmpCase):

    def test_record_mode_mismatch(self):
        req = rootless_request(self.root)
        rootful_req = dict(req, user=f"{os.getuid()}:{PRIMARY}")
        cases = (
            (as_record(req, rootless=True), True, False),
            (as_record(req), True, True),
            (as_record(rootful_req), False, False),
            (as_record(rootful_req, rootless=True), False, True),
            (as_record(rootful_req, rootless=False), False, True),      # a rootful record omits the key
            (as_record(req, rootless="true"), True, True),
        )
        for doc, rootless, refused in cases:
            with self.subTest(record=doc.get("rootless", "absent"), host_rootless=rootless):
                msgs = messages(self.problems(doc, kind="record", rootless=rootless), "rootless")
                self.assertEqual(bool(msgs), refused, msgs)
                if refused and isinstance(doc.get("rootless", False), bool) and doc.get("rootless") is not False:
                    self.assertTrue(any("rootless=" in m for m in msgs), msgs)

    def test_record_keys(self):
        self.assertEqual(launch.OPTIONAL_RECORD_KEYS, ("sidecar", "rootless"))
        self.assertEqual(launch.HOST_FILE_BOOL_KEYS, ("rootless",))


class TestCommand(TmpCase):

    def record(self, **kw) -> dict:
        src = self.root / "inputs"
        src.mkdir(exist_ok=True)
        rec = {"adapter": "synthetic", "argv": ["/inputs/bin/python", "/inputs/train.py"], "gpu_used_h": 0,
               "host_work": "/w", "label": "run-0123456789ab", "gpus": True, "network": "none", "user": "0:0",
               "mounts": [{"host": str(src), "container": "/inputs", "mode": "ro"}], "tmpfs": ["/tmp"],
               "runtime": {"kind": "container", "image": IMAGE}}
        rec.update(kw)
        return rec

    def test_gpu_options(self):
        self.assertEqual(ROOTLESS_USER, "0:0")
        self.assertEqual(CDI_GPU_OPTIONS, ("--device", "nvidia.com/gpu=all"))
        self.assertEqual(ROOTFUL_GPU_OPTIONS, ("--runtime=nvidia", "--gpus", "all"))
        self.assertEqual(gpu_options(True), CDI_GPU_OPTIONS)
        self.assertEqual(gpu_options(False), ROOTFUL_GPU_OPTIONS)

    def test_build_command_cdi(self):
        cmd = build_command(self.record(rootless=True), {}, est_h=1.0)
        self.assertIn(" --rm --device nvidia.com/gpu=all --network none ", cmd)
        self.assertNotIn("--gpus", cmd)
        self.assertNotIn("--runtime=nvidia", cmd)
        self.assertIn("--user 0:0", cmd)
        rootful = build_command(self.record(user="1000:1000"), {}, est_h=1.0)
        self.assertIn(" --rm --runtime=nvidia --gpus all --network none ", rootful)
        self.assertNotIn("--device", rootful)
        for rootless in (True, False):
            cpu = build_command(self.record(gpus=False, **({"rootless": True} if rootless else {})), {}, est_h=0.0)
            self.assertNotIn("--device", cpu)
            self.assertNotIn("--gpus", cpu)
        self.assertEqual(cmd.replace("--device nvidia.com/gpu=all", "--runtime=nvidia --gpus all"),
                         build_command(self.record(), {}, est_h=1.0))

    def test_option_value_and_mount_source(self):
        """Finding 17: the value and mount-source guards of the printed commands, in one place."""
        self.assertEqual(option_value("none"), "none")
        for bad in ("", "-v", "--privileged"):
            with self.assertRaises(ValueError):
                option_value(bad)
        d = self.root / "d"
        d.mkdir()
        link = self.root / "link"
        link.symlink_to(d)
        self.assertEqual(mount_source(str(link)), os.path.realpath(d))
        with self.assertRaises(ValueError):
            mount_source(str(self.root / "missing"))
        with self.assertRaises(ValueError):
            mount_source("-x")
        odd = self.root / "a,b"
        odd.mkdir()
        with self.assertRaises(ValueError):
            mount_source(str(odd))


class RootlessLaunchCase(unittest.TestCase):
    """ext-launch in process on the golden request (user `0:0`) with a host file `rootless = true`."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-rootless-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = self.tmp / "host"
        req = yaml.safe_load(GOLDEN_REQUEST.read_text(encoding="utf-8").replace("<user>", ROOTLESS_USER))
        req = rooted(req, self.root)
        shutil.copyfile(VENV_LOCK, self.tmp / "uv.lock")
        req["runtime"]["venv_lock"] = str(self.tmp / "uv.lock")
        self.req = req
        self.layout = host_layout(req, self.root)
        self.spec = bound_spec(self.tmp, req, self.root)
        self.n = 0
        # The owner's rootless socket directory (`--recheck` stats `<RUN_USER_DIR>/<owner uid>/docker.sock`).
        self.run_user = self.tmp / "run" / "user"
        patch = mock.patch.object(launch, "RUN_USER_DIR", str(self.run_user))
        patch.start()
        self.addCleanup(patch.stop)
        # No host socket is read: the rootful sockets' groups are none here.
        patch = mock.patch.object(launch, "rootful_socket_gids", return_value=set())
        patch.start()
        self.addCleanup(patch.stop)

    def host_toml(self, text: str) -> str:
        path = self.tmp / f"host-{self.n}.toml"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def config(self, *, rootless: bool = True, **kw) -> HostConfig:
        cfg = launch.host_config({**self.layout,
                                  launch.ENV_HOST_CONFIG: self.host_toml("rootless = true\n" if rootless else "")})
        self.assertIs(cfg.rootless, rootless)
        return dataclasses.replace(cfg, primary_gid=lambda uid: PRIMARY, **kw)

    def launch(self, req: dict | None = None, **kw) -> tuple[str, Path]:
        self.n += 1
        path = self.tmp / f"request-{self.n}.yaml"
        path.write_text(yaml.safe_dump(self.req if req is None else req, sort_keys=False), encoding="utf-8")
        out = self.tmp / f"launch-{self.n}.json"
        cmd = launch.ext_launch(adapter="synthetic", profile="smoke", request=path,
                                gpu_log=Path(self.layout[ENV_HOST_WORK]) / "gpu-log.md", out_record=out,
                                spec=self.spec, config=self.config(**kw))
        return cmd, out

    def daemon_socket(self, uid: int | None = None) -> Path:
        d = self.run_user / str(os.getuid() if uid is None else uid)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "docker.sock"
        s = socket.socket(socket.AF_UNIX)
        s.bind(str(p))
        s.close()
        return p


class TestExtLaunchRootless(RootlessLaunchCase):

    def test_ext_launch_rootless_record(self):
        cmd, out = self.launch()
        rec = json.loads(out.read_text(encoding="utf-8"))
        self.assertIs(rec["rootless"], True)
        self.assertEqual(rec["user"], "0:0")
        self.assertEqual(rec["host_config_sha256"], host_config_sha256(launch.adapter_config(self.config(),
                                                                                             "synthetic")))
        self.assertNotEqual(rec["host_config_sha256"],
                            host_config_sha256(launch.adapter_config(self.config(rootless=False), "synthetic")))
        self.assertIsNone(launch.label_problem(rec))
        base = launch.HASHED_LABEL_RE.fullmatch(rec["label"]).group("base")
        rootful = {k: v for k, v in rec.items() if k != "rootless"}
        self.assertNotEqual(launch.launch_label(dict(rootful, label=base)), rec["label"])  # the key is in the label
        self.assertIn(" --device nvidia.com/gpu=all ", cmd)
        self.assertIn(" --user 0:0 ", cmd)
        self.assertNotIn("--gpus", cmd)
        self.assertNotIn("--runtime=nvidia", cmd)

    def test_ext_launch_refuses_rootful_user(self):
        req = dict(self.req, user=f"{os.getuid()}:{PRIMARY}")
        with self.assertRaises(Refused) as cm:
            self.launch(req)
        self.assertIn("--request.user", [p.field for p in cm.exception.problems])

    def test_collect_refuses_other_mode(self):
        """ext-collect's record check: a rootless record on a rootful host is refused, and the reverse."""
        _, out = self.launch()
        rec = json.loads(out.read_text(encoding="utf-8"))
        got = launch.record_problems(rec, adapter="synthetic", stage_out=None, config=self.config(rootless=False))
        self.assertIn("--launch-record.rootless", [p.field for p in got])
        got = launch.record_problems(rec, adapter="synthetic", stage_out=None, config=self.config())
        self.assertEqual([p.field for p in got], [])

    def test_recheck_daemon_socket(self):
        """`--recheck` checks, by stat only, that the owner's rootless socket exists, is a socket and is the owner's."""
        _, out = self.launch()
        with self.assertRaises(Refused) as cm:                       # no socket
            launch.recheck(adapter="synthetic", record_path=out, spec=self.spec, config=self.config())
        self.assertEqual([p.field for p in cm.exception.problems], ["--recheck.daemon"])
        self.assertIn("EXT-263", cm.exception.problems[0].message)
        d = self.run_user / str(os.getuid())
        d.mkdir(parents=True)
        (d / "docker.sock").write_text("", encoding="utf-8")         # not a socket
        with self.assertRaises(Refused) as cm:
            launch.recheck(adapter="synthetic", record_path=out, spec=self.spec, config=self.config())
        self.assertEqual([p.field for p in cm.exception.problems], ["--recheck.daemon"])
        (d / "docker.sock").unlink()
        self.daemon_socket()
        self.assertTrue(launch.recheck(adapter="synthetic", record_path=out, spec=self.spec,
                                       config=self.config()).startswith("recheck: ok "))
        # Owned by another uid than the owner's: refused (the stat is injected; nothing is chowned).
        real_stat = os.stat

        def stat(p, *a, **kw):
            st = real_stat(p, *a, **kw)
            if str(p) == str(d / "docker.sock"):
                return os.stat_result((st.st_mode, st.st_ino, st.st_dev, st.st_nlink, st.st_uid + 1, *tuple(st)[5:]))
            return st
        with mock.patch.object(launch.os, "lstat", side_effect=stat), \
                mock.patch.object(launch.os, "stat", side_effect=stat):
            with self.assertRaises(Refused) as cm:
                launch.recheck(adapter="synthetic", record_path=out, spec=self.spec, config=self.config())
        self.assertIn("--recheck.daemon", [p.field for p in cm.exception.problems])

    def test_recheck_rootful_reads_no_socket(self):
        """A rootful record's --recheck has no daemon check (as before)."""
        self.assertIsNone(launch.rootless_daemon_problem(self.config(rootless=False)))


class TestSidecarRootless(RootlessLaunchCase):
    """Finding 13: under rootless the sidecar's spool is owned by the owner (the sidecar runs as 0:0 too)."""

    def setUp(self):
        super().setUp()
        self.exec_dir = self.root / EXEC_DIR.lstrip("/")
        (self.exec_dir / ".venv" / "bin").mkdir(parents=True)
        (self.exec_dir / ".venv" / "bin" / "python").write_text("# interpreter stand-in\n", encoding="utf-8")
        (self.exec_dir / "serve.py").write_text("# executor stand-in\n", encoding="utf-8")
        lock = self.exec_dir / "uv.lock"
        lock.write_text("# executor lock stand-in\n", encoding="utf-8")
        output = Path(self.req["output_dir"])
        self.spool = output.parent / "spool"
        self.spool.mkdir()
        self.req["mounts"].append({"host": str(self.spool), "container": "/spool", "mode": "rw"})
        self.req["sidecar"] = {
            "runtime": {"kind": "venv_in_container", "image": EXEC_IMAGE, "venv_lock": str(lock)},
            "argv": list(ARGV),
            "mounts": [{"host": str(self.exec_dir), "container": EXEC_DIR, "mode": "ro"},
                       {"host": str(self.spool), "container": "/spool", "mode": "rw"}],
            "env": dict(SIDE_ENV), "network": "none", "user": ROOTLESS_USER, "tmpfs": ["/tmp"], "read_only": True,
            "cap_drop": ["ALL"], "security_opt": ["no-new-privileges"], "gpus": False, "limits": dict(EXECUTOR_LIMITS)}
        self.row = sidecar_row(file_sha256(lock))

    def config(self, **kw) -> HostConfig:
        cfg = super().config(**kw)
        return dataclasses.replace(cfg, sidecar=self.row,
                                   interpreter_trees=(*cfg.interpreter_trees, str(self.exec_dir / ".venv")))

    def test_spool_owned_by_owner(self):
        cmd, out = self.launch()
        rec = json.loads(out.read_text(encoding="utf-8"))
        self.assertIs(rec["rootless"], True)
        start, trainer, stop = cmd.split("\n")
        self.assertIn(" --user 0:0 ", start)
        self.assertNotIn("--device", start)
        self.assertIn(" --device nvidia.com/gpu=all ", trainer)
        with self.assertRaises(Refused) as cm:
            self.launch(owner_uid=os.getuid() + 1)
        hits = [p for p in cm.exception.problems if p.field == "--request.sidecar.mounts[1].host"]
        self.assertTrue(hits and any("spool" in p.message and "owner" in p.message for p in hits),
                        [(p.field, p.message) for p in cm.exception.problems])


if __name__ == "__main__":
    unittest.main()
